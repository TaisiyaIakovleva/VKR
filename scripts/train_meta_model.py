from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from torchvision import models
from tqdm import tqdm

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

    for images, labels in tqdm(loader, desc="image model", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def get_symptoms_predictions(model, symptoms_df, feature_names, class_names):
    cols = []
    for f in feature_names:
        if f in symptoms_df.columns:
            cols.append(f)
        else:
            symptoms_df[f] = 0
            cols.append(f)

    X = symptoms_df[cols].values
    probs = model.predict_proba(X)
    labels = symptoms_df["label"].values

    return probs, labels


def create_meta_features(image_probs, symptoms_probs):
    features = np.concatenate([image_probs, symptoms_probs], axis=1)

    epsilon = 1e-10
    image_entropy = -np.sum(image_probs * np.log(image_probs + epsilon), axis=1, keepdims=True)
    symptoms_entropy = -np.sum(symptoms_probs * np.log(symptoms_probs + epsilon), axis=1, keepdims=True)
    features = np.concatenate([features, image_entropy, symptoms_entropy], axis=1)

    image_max_prob = np.max(image_probs, axis=1, keepdims=True)
    symptoms_max_prob = np.max(symptoms_probs, axis=1, keepdims=True)
    features = np.concatenate([features, image_max_prob, symptoms_max_prob], axis=1)

    prob_diff = np.abs(image_probs - symptoms_probs)
    features = np.concatenate([features, prob_diff], axis=1)

    return features


def main(
    image_model_path: str = "models/sd198/sd198_efficientnet_b0.pth",
    symptoms_model_path: str = "models/symptoms/random_forest_symptoms_sd198.joblib",
    symptoms_features_path: str = "models/symptoms/feature_names.csv",
    csv_path: str = "data/processed/sd198/sd198_splits.csv",
    batch_size: int = 32,
    image_size: int = 224,
):
    model_dir = Path("models/meta_model")
    exp_dir = Path("experiments/meta_model")

    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"устройство: {device}")

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    mapping = df[["label", "diagnosis"]].drop_duplicates().sort_values("label")
    class_names = mapping["diagnosis"].tolist()
    num_classes = len(class_names)

    image_model = models.efficientnet_b0(weights=None)
    image_model.classifier[1] = nn.Linear(
        image_model.classifier[1].in_features,
        num_classes
    )

    checkpoint_path = "models/sd198/sd198_efficientnet_b0.pth"
    image_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    image_model = image_model.to(device)
    image_model.eval()

    image_model.classifier[1] = nn.Linear(image_model.classifier[1].in_features, num_classes)
    image_model.load_state_dict(torch.load(image_model_path, map_location=device))
    image_model = image_model.to(device)
    print(f"модель: {image_model_path}")

    symptoms_model = joblib.load(symptoms_model_path)
    model_feature_names = getattr(symptoms_model, "feature_names_in_", None)
    if model_feature_names is not None:
        symptoms_features = list(model_feature_names)
    else:
        symptoms_features = pd.read_csv(symptoms_features_path)["feature_name"].tolist()
    print(f"симптомы: {symptoms_model_path} ({len(symptoms_features)} признаков)")

    val_dataset = SD198Dataset(csv_path, "val", get_sd198_eval_transforms(image_size))
    test_dataset = SD198Dataset(csv_path, "test", get_sd198_eval_transforms(image_size))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    symptoms_csv = Path("data/symptoms/symptoms_dataset.csv")
    symptoms_df = pd.read_csv(symptoms_csv)
    symptoms_df = symptoms_df.copy()
    symptoms_df["label"] = symptoms_df["diagnosis"].map(
        dict(zip(mapping["diagnosis"], mapping["label"]))
    )
    symptoms_df = symptoms_df.dropna(subset=["label"])
    symptoms_df["label"] = symptoms_df["label"].astype(int)

    print(f"val: {len(val_dataset)}, test: {len(test_dataset)}, symptoms: {len(symptoms_df)}")

    val_image_probs, val_image_labels = get_image_predictions(image_model, val_loader, device)
    test_image_probs, test_image_labels = get_image_predictions(image_model, test_loader, device)

    if len(symptoms_df) > 0:
        val_symptoms_probs, val_symptoms_labels = get_symptoms_predictions(
            symptoms_model,
            symptoms_df.iloc[:len(val_image_probs)],
            symptoms_features,
            class_names
        )
    else:
        print("нет симптомов, используем нули")
        val_symptoms_probs = np.zeros_like(val_image_probs)
        val_symptoms_labels = val_image_labels

    meta_features_val = create_meta_features(val_image_probs, val_symptoms_probs)
    meta_labels_val = val_image_labels

    val_baseline_probs = 0.75 * val_image_probs + 0.25 * val_symptoms_probs
    val_baseline_pred = np.argmax(val_baseline_probs, axis=1)
    val_baseline_acc = accuracy_score(meta_labels_val, val_baseline_pred)
    val_baseline_f1 = f1_score(meta_labels_val, val_baseline_pred, average="macro")
    print(f"baseline val: Acc={val_baseline_acc:.4f}, F1={val_baseline_f1:.4f}")

    scaler = StandardScaler()
    meta_features_val_scaled = scaler.fit_transform(meta_features_val)

    print("обучаем LogisticRegression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    )
    lr_model.fit(meta_features_val_scaled, meta_labels_val)

    val_lr_pred = lr_model.predict(meta_features_val_scaled)
    val_lr_acc = accuracy_score(meta_labels_val, val_lr_pred)
    val_lr_f1 = f1_score(meta_labels_val, val_lr_pred, average="macro")
    print(f"LogisticRegression val: Acc={val_lr_acc:.4f}, F1={val_lr_f1:.4f}")

    if len(symptoms_df) > 0:
        test_symptoms_probs, test_symptoms_labels = get_symptoms_predictions(
            symptoms_model,
            symptoms_df.iloc[len(val_image_probs):len(val_image_probs)+len(test_image_probs)],
            symptoms_features,
            class_names
        )
    else:
        test_symptoms_probs = np.zeros_like(test_image_probs)
        test_symptoms_labels = test_image_labels

    meta_features_test = create_meta_features(test_image_probs, test_symptoms_probs)
    meta_features_test_scaled = scaler.transform(meta_features_test)
    meta_labels_test = test_image_labels

    test_lr_pred = lr_model.predict(meta_features_test_scaled)
    test_lr_acc = accuracy_score(meta_labels_test, test_lr_pred)
    test_lr_f1 = f1_score(meta_labels_test, test_lr_pred, average="macro")

    print(f"Test LR: Acc={test_lr_acc:.4f}, F1={test_lr_f1:.4f}")

    baseline_test_probs = 0.75 * test_image_probs + 0.25 * test_symptoms_probs
    baseline_test_pred = np.argmax(baseline_test_probs, axis=1)
    baseline_test_acc = accuracy_score(meta_labels_test, baseline_test_pred)
    baseline_test_f1 = f1_score(meta_labels_test, baseline_test_pred, average="macro")
    print(f"baseline test (0.75*img + 0.25*sym): acc={baseline_test_acc:.4f}, f1={baseline_test_f1:.4f}")

    joblib.dump(lr_model, model_dir / "meta_logistic_regression.joblib")
    joblib.dump(scaler, model_dir / "meta_scaler.joblib")

    with open(model_dir / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    meta_info = {
        "num_classes": num_classes,
        "class_names": class_names,
        "num_meta_features": meta_features_val.shape[1],
        "image_model_path": str(image_model_path),
        "symptoms_model_path": str(symptoms_model_path),
    }
    joblib.dump(meta_info, model_dir / "meta_info.joblib")

    print(f"сохранено в {model_dir}/")

    results = {
        "model": ["LogisticRegression", "Baseline (0.75*image + 0.25*symptoms)"],
        "val_accuracy": [val_lr_acc, val_baseline_acc],
        "val_f1": [val_lr_f1, val_baseline_f1],
        "test_accuracy": [test_lr_acc, baseline_test_acc],
        "test_f1": [test_lr_f1, baseline_test_f1],
    }

    results_df = pd.DataFrame(results)
    results_df.to_csv(exp_dir / "meta_model_results.csv", index=False)
    print(f"результаты: {exp_dir / 'meta_model_results.csv'}")

    print("\n" + results_df.to_string(index=False))

    best_pred = test_lr_pred
    best_name = "LogisticRegression"

    report = classification_report(
        meta_labels_test, best_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    (exp_dir / f"classification_report_meta_model_{best_name}.txt").write_text(
        report, encoding="utf-8"
    )
    print(f"\n{best_name} report:")
    print(report)


if __name__ == "__main__":
    main()
