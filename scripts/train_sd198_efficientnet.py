from pathlib import Path
import argparse

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

from skin_dss.data.sd198_dataset import (
    SD198Dataset,
    get_sd198_eval_transforms,
    get_sd198_train_transforms,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_weighted_sampler(dataset: SD198Dataset):
    labels = dataset.data["label"].tolist()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True
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


def main(csv_path: str = "data/processed/sd198/sd198_splits.csv", batch_size: int = 32, num_epochs: int = 30, image_size: int = 224, lr: float = 1e-4):
    csv_path = Path(csv_path)
    model_dir = Path("models/sd198")
    experiments_dir = Path("experiments/sd198")
    model_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    mapping = df[["label", "diagnosis"]].drop_duplicates().sort_values("label")
    class_names = mapping["diagnosis"].tolist()

    num_classes = len(class_names)

    device = get_device()
    print(f"устройство: {device}")
    print(f"классов: {num_classes}")

    train_dataset = SD198Dataset(csv_path, "train", get_sd198_train_transforms(image_size))
    val_dataset = SD198Dataset(csv_path, "val", get_sd198_eval_transforms(image_size))
    test_dataset = SD198Dataset(csv_path, "test", get_sd198_eval_transforms(image_size))

    train_sampler = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
    )

    best_val_f1 = 0.0
    best_model_path = model_dir / "sd198_efficientnet_b0.pth"
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

        scheduler.step()

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    pd.DataFrame(history).to_csv(experiments_dir / "history_sd198.csv", index=False)

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

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    (experiments_dir / "classification_report_sd198.txt").write_text(report, encoding="utf-8")

    cm_counts = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    cm_counts_df = pd.DataFrame(cm_counts, index=class_names, columns=class_names)
    cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cm_counts_df.to_csv(experiments_dir / "confusion_matrix_counts_sd198.csv")
    cm_norm_df.to_csv(experiments_dir / "confusion_matrix_normalized_sd198.csv")

    plt.figure(figsize=(22, 18))
    sns.heatmap(cm_norm_df, annot=False, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (normalized) - SD-198 EfficientNet-B0")
    plt.tight_layout()
    plt.savefig(experiments_dir / "confusion_matrix_sd198_norm_full.png", bbox_inches="tight")
    plt.close()

    support_counts = pd.Series(y_true).value_counts().sort_values(ascending=False)
    top_n = 40
    top_classes_idx = support_counts.index[:top_n].tolist()
    top_class_names = [class_names[i] for i in top_classes_idx]
    cm_top_norm = cm_norm_df.loc[top_class_names, top_class_names]

    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_top_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=top_class_names, yticklabels=top_class_names)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (normalized) - top {top_n} classes by support")
    plt.tight_layout()
    plt.savefig(experiments_dir / f"confusion_matrix_sd198_top{top_n}.png", bbox_inches="tight")
    plt.close()

    cm_offdiag = cm_counts.copy()
    np.fill_diagonal(cm_offdiag, 0)
    pairs = []
    rows, cols = np.where(cm_offdiag > 0)
    for r, c in zip(rows, cols):
        pairs.append((int(cm_offdiag[r, c]), class_names[r], class_names[c]))
    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
    top_confusions = pairs_sorted[:100]
    conf_df = pd.DataFrame(top_confusions, columns=["count", "true", "predicted"])
    conf_df.to_csv(experiments_dir / "top_confusions_sd198.csv", index=False)

    def get_error_group(name: str):
        d = name.lower()

        if "nevus" in d or "melanoma" in d or "lentigo" in d:
            return "pigmented_lesions"
        if "eczema" in d or "dermatitis" in d:
            return "eczema_dermatitis"
        if "psoriasis" in d:
            return "psoriasis"
        if "tinea" in d or "candidiasis" in d:
            return "fungal"
        if "onych" in d or "nail" in d or "leukonychia" in d:
            return "nail"
        if "keratosis" in d or "horn" in d:
            return "keratosis"
        if "carcinoma" in d or "bowen" in d:
            return "tumor"
        if "angioma" in d or "vascular" in d:
            return "vascular"
        return "other"


    conf_df["true_group"] = conf_df["true"].apply(get_error_group)
    conf_df["predicted_group"] = conf_df["predicted"].apply(get_error_group)
    conf_df["same_group_error"] = conf_df["true_group"] == conf_df["predicted_group"]

    conf_df.to_csv(experiments_dir / "top_confusions_sd198_with_groups.csv", index=False)

    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_class_df = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=class_names)
    per_class_df.to_csv(experiments_dir / "per_class_metrics_sd198.csv")

    worst_recall_df = per_class_df[per_class_df["support"] >= 5].sort_values("recall").head(30)
    worst_recall_df.to_csv(experiments_dir / "worst_recall_classes_sd198.csv")

    fig = plt.figure(figsize=(4, 3))
    lines = []
    lines.append(f"Accuracy: {test_acc:.4f}")
    if top3_acc is not None:
        lines.append(f"Top-3 acc: {top3_acc:.4f}")
    lines.append(f"Macro-F1: {test_f1:.4f}")
    fig.text(0.01, 0.5, '\n'.join(lines), fontsize=10, va='center')
    fig.savefig(experiments_dir / "metrics_summary_sd198.png", bbox_inches="tight")
    plt.close()

    print("сохранено:", experiments_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on SD-198")
    parser.add_argument("--csv", type=str, default="data/processed/sd198/sd198_splits.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    main(csv_path=args.csv, batch_size=args.batch_size, num_epochs=args.epochs, image_size=args.image_size, lr=args.lr)
