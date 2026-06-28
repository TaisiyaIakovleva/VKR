from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

from skin_dss.data.skin_extra_dataset import (
    SkinExtraDataset,
    get_eval_transforms,
    get_train_transforms,
)

try:
    from skin_dss.data.hf_sd198_dataset import HfSd198Dataset
except Exception:
    HfSd198Dataset = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_weighted_sampler(dataset: SkinExtraDataset):
    labels = dataset.data["label"].tolist()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds, np.array(all_probs)


def main(use_hf: bool = False, hf_repo_id: str | None = None, hf_cache_dir: str | None = None, hf_token: str | None = None, hf_manifest: str | None = None):
    csv_path = Path("data/processed/skin_extra_splits.csv")
    label_map_path = Path("data/processed/skin_extra_label_map.csv")

    model_dir = Path("models/skin_extra")
    experiments_dir = Path("experiments/skin_extra")
    model_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    if use_hf:
        if HfSd198Dataset is None:
            raise RuntimeError("HfSd198Dataset not available. Install 'datasets' and ensure hf_sd198_dataset.py is present.")
        try:
            hf_tmp = HfSd198Dataset(repo_id=hf_repo_id or "resyhgerwshshgdfghsdfgh/SD-198", split="train", hf_token=hf_token, local_manifest=Path(hf_manifest) if hf_manifest else None)
            class_names = hf_tmp.label_names if getattr(hf_tmp, "label_names", None) else []
            del hf_tmp
        except Exception:
            class_names = []
    else:
        label_df = pd.read_csv(label_map_path)
        class_names = label_df.sort_values("label")["class_name"].tolist()

    batch_size = 16
    num_epochs = 5
    num_classes = len(class_names)
    lr = 1e-4
    image_size = 224

    device = get_device()
    print(f"устройство: {device}")
    print(f"классов: {num_classes}")

    if use_hf:
        repo = hf_repo_id or "resyhgerwshshgdfghsdfgh/SD-198"

        def _make_hf(split_name: str):
            candidates = (split_name, "validation") if split_name == "val" else (split_name,)
            last_exc = None
            for c in candidates:
                try:
                    return HfSd198Dataset(repo_id=repo, split=c, transform=(get_train_transforms(image_size) if c == "train" else get_eval_transforms(image_size)), local_cache_dir=Path(hf_cache_dir) if hf_cache_dir else None, hf_token=hf_token, local_manifest=Path(hf_manifest) if hf_manifest else None)
                except Exception as e:
                    last_exc = e
                    continue
            raise RuntimeError(f"Could not load split '{split_name}' from HF repo {repo}") from last_exc

        train_dataset = _make_hf("train")
        val_dataset = _make_hf("val")
        test_dataset = _make_hf("test")

        try:
            labels = [train_dataset[i][1] for i in range(len(train_dataset))]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[label] for label in labels]
            train_sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True,
            )
        except Exception:
            train_sampler = None
    else:
        train_dataset = SkinExtraDataset(csv_path, "train", get_train_transforms(image_size))
        val_dataset = SkinExtraDataset(csv_path, "val", get_eval_transforms(image_size))
        test_dataset = SkinExtraDataset(csv_path, "test", get_eval_transforms(image_size))

        train_sampler = make_weighted_sampler(train_dataset)

    if train_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    best_model_path = model_dir / "best_skin_extra_efficientnet_b0.pth"
    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device, split_name="Val")

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Train macro-F1: {train_f1:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f} | Val macro-F1:   {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print("лучшая модель сохранена")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        })

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    pd.DataFrame(history).to_csv(experiments_dir / "history_skin_extra.csv", index=False)

    test_loss, test_acc, test_f1, y_true, y_pred, y_probs = evaluate(
        model, test_loader, criterion, device, split_name="Test"
    )

    print(f"\ntest: loss={test_loss:.4f}, acc={test_acc:.4f}, f1={test_f1:.4f}")
    try:
        topk = np.argsort(y_probs, axis=1)[:, -3:]
        top3_hit = [int(y_true[i] in topk[i]) for i in range(len(y_true))]
        top3_acc = float(np.mean(top3_hit))
    except Exception:
        top3_acc = None
    if top3_acc is not None:
        print(f"Test top-3 acc: {top3_acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (experiments_dir / "classification_report_skin_extra.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Skin Extra")
    fig = plt.gcf()
    lines = []
    lines.append(f"Accuracy: {test_acc:.4f}")
    if top3_acc is not None:
        lines.append(f"Top-3 acc: {top3_acc:.4f}")
    lines.append(f"Macro-F1: {test_f1:.4f}")
    fig.text(0.90, 0.5, '\n'.join(lines), transform=fig.transFigure,
             fontsize=10, va='center', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout(rect=(0, 0, 0.88, 1))
    plt.savefig(experiments_dir / "confusion_matrix_skin_extra.png", bbox_inches="tight")
    plt.close()

    print("сохранено:", experiments_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Skin-Extra (or HF SD-198) EfficientNet-B0")
    parser.add_argument("--use-hf", action="store_true", help="Use HuggingFace SD-198 dataset instead of local CSV-based skin_extra")
    parser.add_argument("--hf-repo", type=str, default="resyhgerwshshgdfghsdfgh/SD-198", help="HF repo id for SD-198 (when --use-hf is set)")
    parser.add_argument("--hf-cache", type=str, default=None, help="Optional cache dir for HF datasets")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--hf-manifest", type=str, default=None, help="Local manifest path (CSV/JSONL) to use instead of querying the Hub")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    main(use_hf=args.use_hf, hf_repo_id=args.hf_repo, hf_cache_dir=args.hf_cache, hf_token=hf_token, hf_manifest=args.hf_manifest)