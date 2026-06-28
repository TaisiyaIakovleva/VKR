from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

from skin_dss.data.ham10000_dataset import (
    HAM10000Dataset,
    get_eval_transforms,
    get_train_transforms,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_weighted_sampler(dataset: HAM10000Dataset):
    labels = dataset.data["label"].tolist()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
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


def main():
    csv_path = Path("data/processed/ham10000_splits.csv")
    model_dir = Path("models/efficientnet_b0")   
    model_dir.mkdir(exist_ok=True)

    batch_size = 16
    num_epochs = 5
    num_classes = 7
    lr = 1e-4

    device = get_device()
    print(f"устройство: {device}")

    train_dataset = HAM10000Dataset(
        csv_path=csv_path,
        split="train",
        transform=get_train_transforms(image_size=224),
    )
    val_dataset = HAM10000Dataset(
        csv_path=csv_path,
        split="val",
        transform=get_eval_transforms(image_size=224),
    )
    test_dataset = HAM10000Dataset(
        csv_path=csv_path,
        split="test",
        transform=get_eval_transforms(image_size=224),
    )

    train_sampler = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device, split_name="Val"
        )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | Train macro-F1: {train_f1:.4f}"
        )
        print(
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | Val macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_dir / "best_efficientnet_b0_ham10000.pth")
            print("лучшая модель сохранена")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

    model.load_state_dict(torch.load(model_dir / "best_efficientnet_b0_ham10000.pth", map_location=device))

    history_df = pd.DataFrame(history)

    history_path = Path("experiments/efficientnet_b0/history_efficientnet_b0_ham10000.csv")
    history_path.parent.mkdir(exist_ok=True)

    history_df.to_csv(history_path, index=False)

    print("история:", history_path)

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

    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    report_path = Path("experiments/efficientnet_b0/classification_report_efficientnet_b0_ham10000.txt")
    report_path.write_text(report)
    print("отчёт:", report_path)

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
    plt.title("Confusion Matrix - EfficientNet-B0 HAM10000")
    fig = plt.gcf()
    lines = []
    lines.append(f"Accuracy: {test_acc:.4f}")
    if top3_acc is not None:
        lines.append(f"Top-3 acc: {top3_acc:.4f}")
    lines.append(f"Macro-F1: {test_f1:.4f}")
    fig.text(0.90, 0.5, '\n'.join(lines), transform=fig.transFigure,
             fontsize=10, va='center', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    cm_path = Path("experiments/efficientnet_b0/confusion_matrix_efficientnet_b0_ham10000.png")
    plt.tight_layout(rect=(0, 0, 0.88, 1))
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print("матрица ошибок:", cm_path)

if __name__ == "__main__":
    main()