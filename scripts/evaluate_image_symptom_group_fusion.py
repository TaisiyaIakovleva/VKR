from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score
from torchvision import models
from tqdm import tqdm

from skin_dss.data.sd198_dataset import SD198Dataset, get_sd198_eval_transforms


CSV_PATH = Path("data/processed/sd198/sd198_splits.csv")
IMAGE_MODEL_PATH = Path("models/sd198/sd198_efficientnet_b0.pth")

SYMPTOMS_DATA_PATH = Path("data/symptoms/symptoms_dataset.csv")
SYMPTOMS_GROUP_MODEL_DIR = Path("models/symptoms_group_model")

OUT_DIR = Path("experiments/group_fusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_diagnosis_group(diagnosis: str) -> str:
    d = diagnosis.lower()

    if "melanoma" in d or "carcinoma" in d or "bowen" in d or "keratoacanthoma" in d:
        return "tumor"
    if "nevus" in d or "lentigo" in d or "macule" in d or "melasma" in d or "vitiligo" in d:
        return "pigment_nevus"
    if "keratosis" in d or "horn" in d or "callus" in d or "hyperkeratosis" in d:
        return "keratosis"
    if "psoriasis" in d:
        return "psoriasis"
    if "dermatitis" in d or "eczema" in d or "erythema" in d or "urticaria" in d or "xerosis" in d:
        return "inflammatory"
    if "tinea" in d or "candid" in d or "pityrosporum" in d or "onychomycosis" in d:
        return "fungal"
    if (
        "onych" in d or "nail" in d or "leukonychia" in d or "beau" in d
        or "koilonychia" in d or "terry" in d or "subungual" in d or "paronychia" in d
    ):
        return "nail"
    if "alopecia" in d or "hair" in d or "tricho" in d or "hypertrichosis" in d:
        return "hair"
    if "stomatitis" in d or "cheilitis" in d or "aphthous" in d or "tongue" in d or "mouth" in d:
        return "mouth"
    if "follic" in d or "acne" in d or "hidradenitis" in d or "comedonicus" in d:
        return "follicular_acne"
    if "herpes" in d or "molluscum" in d or "varicella" in d or "verruca" in d:
        return "viral"
    if (
        "angioma" in d or "hemangioma" in d or "vascular" in d
        or "telangiectasia" in d or "purpura" in d or "livedo" in d
        or "vasculitis" in d or "schamberg" in d
    ):
        return "vascular"
    if "ulcer" in d or "wound" in d or "pyoderma" in d or "impetigo" in d or "cellulitis" in d:
        return "infection_ulcer"
    if "cyst" in d or "fibroma" in d or "lipoma" in d or "skin_tag" in d or "syringoma" in d:
        return "benign_growth"

    return "other"


@torch.no_grad()
def get_image_probs(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="предсказания"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)


def load_image_model(num_classes: int, device):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def build_group_matrix(class_names):
    groups = sorted(set(get_diagnosis_group(c) for c in class_names))
    group_to_idx = {g: i for i, g in enumerate(groups)}

    matrix = np.zeros((len(class_names), len(groups)), dtype=np.float32)

    for class_idx, diagnosis in enumerate(class_names):
        group = get_diagnosis_group(diagnosis)
        matrix[class_idx, group_to_idx[group]] = 1.0

    return matrix, groups, group_to_idx


def get_synthetic_symptom_group_probs(num_samples, true_labels, class_names):
    symptoms_df = pd.read_csv(SYMPTOMS_DATA_PATH)

    model = joblib.load(SYMPTOMS_GROUP_MODEL_DIR / "symptoms_group_model.joblib")
    label_encoder = joblib.load(SYMPTOMS_GROUP_MODEL_DIR / "group_label_encoder.joblib")
    feature_names = joblib.load(SYMPTOMS_GROUP_MODEL_DIR / "feature_names.joblib")

    rows = []

    for label in true_labels:
        diagnosis = class_names[int(label)]
        candidates = symptoms_df[symptoms_df["diagnosis"] == diagnosis]

        if len(candidates) == 0:
            rows.append(pd.Series({f: 0 for f in feature_names}))
        else:
            rows.append(candidates.sample(1, random_state=None).iloc[0])

    X = pd.DataFrame(rows)

    for f in feature_names:
        if f not in X.columns:
            X[f] = 0

    X = X[feature_names]

    group_probs = model.predict_proba(X)
    group_classes = list(label_encoder.classes_)

    return group_probs, group_classes


def fuse_image_with_symptom_groups(image_probs, symptom_group_probs, class_to_group_matrix, group_classes, all_groups, alpha):
    group_index_from_symptom_model = {g: i for i, g in enumerate(group_classes)}
    group_index_all = {g: i for i, g in enumerate(all_groups)}

    aligned_group_probs = np.zeros((symptom_group_probs.shape[0], len(all_groups)), dtype=np.float32)

    for group_name, symptom_idx in group_index_from_symptom_model.items():
        if group_name in group_index_all:
            aligned_group_probs[:, group_index_all[group_name]] = symptom_group_probs[:, symptom_idx]

    class_group_scores = aligned_group_probs @ class_to_group_matrix.T
    fused = image_probs * (1.0 + alpha * class_group_scores)
    fused = fused / fused.sum(axis=1, keepdims=True)

    return fused


def evaluate_probs(name, probs, y_true, class_names):
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    try:
        top3 = top_k_accuracy_score(
            y_true,
            probs,
            k=3,
            labels=np.arange(len(class_names)),
        )
    except Exception:
        top3 = None

    result = {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_accuracy": top3,
    }

    return result, y_pred


def main():
    device = get_device()
    print(f"устройство: {device}")

    df = pd.read_csv(CSV_PATH)
    mapping = df[["label", "diagnosis"]].drop_duplicates().sort_values("label")
    class_names = mapping["diagnosis"].tolist()
    num_classes = len(class_names)

    test_dataset = SD198Dataset(CSV_PATH, "test", get_sd198_eval_transforms(224))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    image_model = load_image_model(num_classes, device)

    image_probs, y_true = get_image_probs(image_model, test_loader, device)
    class_to_group_matrix, all_groups, group_to_idx = build_group_matrix(class_names)

    symptom_group_probs, symptom_group_classes = get_synthetic_symptom_group_probs(
        num_samples=len(y_true),
        true_labels=y_true,
        class_names=class_names,
    )
    results = []

    image_result, image_pred = evaluate_probs(
        "image_only",
        image_probs,
        y_true,
        class_names,
    )
    results.append(image_result)

    best_probs = image_probs
    best_name = "image_only"
    best_acc = image_result["accuracy"]

    for alpha in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
        fused_probs = fuse_image_with_symptom_groups(
            image_probs=image_probs,
            symptom_group_probs=symptom_group_probs,
            class_to_group_matrix=class_to_group_matrix,
            group_classes=symptom_group_classes,
            all_groups=all_groups,
            alpha=alpha,
        )

        result, pred = evaluate_probs(
            f"image_plus_symptom_groups_alpha_{alpha}",
            fused_probs,
            y_true,
            class_names,
        )
        results.append(result)

        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            best_name = result["model"]
            best_probs = fused_probs

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(OUT_DIR / "image_symptom_group_fusion_results.csv", index=False)

    best_pred = np.argmax(best_probs, axis=1)

    report = classification_report(
        y_true,
        best_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (OUT_DIR / "classification_report_best_group_fusion.txt").write_text(report, encoding="utf-8")

    print(results_df.to_string(index=False))
    print(f"\nлучший: {best_name}, acc={best_acc:.4f}")
    print(f"сохранено в: {OUT_DIR}")


if __name__ == "__main__":
    main()

