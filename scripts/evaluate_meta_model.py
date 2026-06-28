from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib

from skin_dss.data.sd198_dataset import (
    SD198Dataset,
    get_sd198_eval_transforms,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def get_image_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="предсказания", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def main(
    image_model_path: str = "models/sd198/sd198_efficientnet_b0.pth",
    symptoms_model_path: str = "models/symptoms/random_forest_symptoms.joblib",
    meta_model_dir: str = "models/meta_model",
    csv_path: str = "data/processed/sd198/sd198_splits.csv",
    batch_size: int = 32,
    image_size: int = 224,
):
    device = get_device()
    print(f"устройство: {device}")

    exp_dir = Path("experiments/meta_model")
    exp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    mapping = df[["label", "diagnosis"]].drop_duplicates().sort_values("label")
    class_names = mapping["diagnosis"].tolist()
    num_classes = len(class_names)

    print(f"классов: {num_classes}")

    image_model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'efficientnet_b0',
        pretrained=False,
        verbose=False
    )
    image_model.classifier[1] = nn.Linear(image_model.classifier[1].in_features, num_classes)
    image_model.load_state_dict(torch.load(image_model_path, map_location=device))
    image_model = image_model.to(device)

    symptoms_model = joblib.load(symptoms_model_path)

    lr_model = joblib.load(Path(meta_model_dir) / "meta_logistic_regression.joblib")
    scaler = joblib.load(Path(meta_model_dir) / "meta_scaler.joblib")

    test_dataset = SD198Dataset(csv_path, "test", get_sd198_eval_transforms(image_size))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    image_probs, image_labels = get_image_predictions(image_model, test_loader, device)
    image_preds = np.argmax(image_probs, axis=1)

    symptoms_df = pd.read_csv("data/symptoms/symptoms_dataset.csv")
    symptoms_df = symptoms_df.copy()
    symptoms_df["label"] = symptoms_df["diagnosis"].map(
        dict(zip(mapping["diagnosis"], mapping["label"]))
    )
    symptoms_df = symptoms_df.dropna(subset=["label"])
    symptoms_df["label"] = symptoms_df["label"].astype(int)

    if len(symptoms_df) > 0:
        symptoms_features = pd.read_csv("models/symptoms/feature_names.csv")["feature_name"].tolist()
        X_symptoms = symptoms_df[symptoms_features].values[:len(image_labels)]
        symptoms_probs = symptoms_model.predict_proba(X_symptoms)
        symptoms_labels = symptoms_df["label"].values[:len(image_labels)]
    else:
        print("нет данных симптомов, используем нули")
        symptoms_probs = np.zeros_like(image_probs)
        symptoms_labels = image_labels

    symptoms_preds = np.argmax(symptoms_probs, axis=1)

    baseline_probs = 0.75 * image_probs + 0.25 * symptoms_probs
    baseline_preds = np.argmax(baseline_probs, axis=1)

    def create_meta_features(img_probs, symp_probs):
        num_samples = img_probs.shape[0]
        features_list = []
        for i in range(num_samples):
            img = img_probs[i]
            symp = symp_probs[i]
            f = np.concatenate([img, symp])
            epsilon = 1e-10
            img_entropy = -np.sum(img * np.log(img + epsilon))
            symp_entropy = -np.sum(symp * np.log(symp + epsilon))
            f = np.concatenate([f, [img_entropy, symp_entropy]])
            f = np.concatenate([f, [np.max(img), np.max(symp)]])
            f = np.concatenate([f, np.abs(img - symp)])
            features_list.append(f)
        return np.array(features_list)

    meta_features = create_meta_features(image_probs, symptoms_probs)
    meta_features_scaled = scaler.transform(meta_features)

    lr_probs = lr_model.predict_proba(meta_features_scaled)
    lr_preds = np.argmax(lr_probs, axis=1)

    results = []

    for name, preds, probs in [
        ("Image Model", image_preds, image_probs),
        ("Symptoms Model", symptoms_preds, symptoms_probs),
        ("Baseline (0.75*img + 0.25*symp)", baseline_preds, baseline_probs),
        ("Meta-Model (LR)", lr_preds, lr_probs),
    ]:
        acc = accuracy_score(image_labels, preds)
        f1 = f1_score(image_labels, preds, average="macro")

        try:
            if num_classes == 2:
                auc = roc_auc_score(image_labels, probs[:, 1])
            else:
                auc = roc_auc_score(image_labels, probs, multi_class="ovr", average="macro")
        except:
            auc = None

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro-F1": f1,
            "ROC-AUC": auc,
        })

        print(f"\n{name}: Acc={acc:.4f}, F1={f1:.4f}" + (f", AUC={auc:.4f}" if auc else ""))

    results_df = pd.DataFrame(results)
    results_df.to_csv(exp_dir / "eval_comparison.csv", index=False)

    for name, preds in [
        ("image", image_preds),
        ("symptoms", symptoms_preds),
        ("baseline", baseline_preds),
        ("meta_lr", lr_preds),
    ]:
        report = classification_report(
            image_labels, preds,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        (exp_dir / f"classification_report_{name}.txt").write_text(report, encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    models_for_plot = [
        ("Image Model", image_preds),
        ("Symptoms Model", symptoms_preds),
        ("Baseline", baseline_preds),
        ("Meta-LR", lr_preds),
    ]

    for idx, (name, preds) in enumerate(models_for_plot):
        cm = confusion_matrix(image_labels, preds)
        sns.heatmap(
            cm, ax=axes[idx], cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names
        )
        axes[idx].set_title(f"{name}\n(Acc: {accuracy_score(image_labels, preds):.3f})")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(exp_dir / "confusion_matrices_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [r["Model"] for r in results]
    accs = [r["Accuracy"] for r in results]
    f1s = [r["Macro-F1"] for r in results]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", alpha=0.8)
    bars2 = ax.bar(x + width/2, f1s, width, label="Macro-F1", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Сравнение моделей")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(exp_dir / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    probs_list = [image_probs, symptoms_probs, baseline_probs, lr_probs]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for idx, (ax, (name, _), probs) in enumerate(zip(axes, models_for_plot, probs_list)):
        max_probs = np.max(probs, axis=1)
        ax.hist(max_probs, bins=30, alpha=0.7, edgecolor="black")
        ax.set_title(f"{name}")
        ax.set_xlabel("Max Probability")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / "probability_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nвсё сохранено в:", exp_dir)


if __name__ == "__main__":
    main()
