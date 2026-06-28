from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_PATH = Path("/Users/Taisia1/Desktop/VKR/data/symptoms/symptoms_dataset.csv")
SAVE_DIR = Path("/Users/Taisia1/Desktop/VKR/models/symptoms_group_model")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


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
        "onych" in d
        or "nail" in d
        or "leukonychia" in d
        or "beau" in d
        or "koilonychia" in d
        or "terry" in d
        or "subungual" in d
        or "paronychia" in d
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
        "angioma" in d
        or "hemangioma" in d
        or "vascular" in d
        or "telangiectasia" in d
        or "purpura" in d
        or "livedo" in d
        or "vasculitis" in d
        or "schamberg" in d
    ):
        return "vascular"

    if "ulcer" in d or "wound" in d or "pyoderma" in d or "impetigo" in d or "cellulitis" in d:
        return "infection_ulcer"

    if "cyst" in d or "fibroma" in d or "lipoma" in d or "skin_tag" in d or "syringoma" in d:
        return "benign_growth"

    return "other"


def main():
    df = pd.read_csv(DATA_PATH)

    if "diagnosis" not in df.columns:
        raise ValueError("В датасете нет столбца diagnosis")

    df["diagnosis_group"] = df["diagnosis"].apply(get_diagnosis_group)

    print("датасет:", DATA_PATH)
    print("примеров:", len(df))
    print("диагнозов:", df["diagnosis"].nunique())
    print("групп:", df["diagnosis_group"].nunique())
    print(df["diagnosis_group"].value_counts())

    X = df.drop(columns=["diagnosis", "diagnosis_group"])
    y = df["diagnosis_group"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    best_name = None
    best_model = None
    best_acc = -1

    for name, model in models.items():
        print(f"\n{name}...")
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        weighted_f1 = f1_score(y_test, pred, average="weighted")

        print(f"{name}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, Weighted-F1={weighted_f1:.4f}")
        print(
            classification_report(
                y_test,
                pred,
                target_names=label_encoder.classes_,
                digits=4,
                zero_division=0,
            )
        )

        results.append(
            {
                "model": name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)

    print("\nрезультаты:")
    print(results_df)

    joblib.dump(best_model, SAVE_DIR / "symptoms_group_model.joblib")
    joblib.dump(label_encoder, SAVE_DIR / "group_label_encoder.joblib")
    joblib.dump(list(X.columns), SAVE_DIR / "feature_names.joblib")
    results_df.to_csv(SAVE_DIR / "symptoms_group_results.csv", index=False)

    print(f"\nлучшая модель: {best_name}, acc={round(best_acc, 4)}")
    print("сохранено в:", SAVE_DIR)


if __name__ == "__main__":
    main()
