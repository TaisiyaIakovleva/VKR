import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score


def load_feature_names(path: Path):
    df = pd.read_csv(path)
    return df["feature_name"].tolist()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "symptoms" / "symptoms_dataset.csv"
    model_path = repo_root / "models" / "symptoms" / "random_forest_symptoms.joblib"
    feat_path = repo_root / "models" / "symptoms" / "feature_names.csv"
    out_dir = repo_root / "experiments" / "symptoms"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    feature_names = load_feature_names(feat_path)

    model = joblib.load(model_path)

    y_true = []
    y_pred = []
    top3_hits = 0
    total = 0

    classes = list(model.classes_)

    preds = []

    for _, row in df.iterrows():
        total += 1
        X = row[feature_names].to_frame().T
        probs = model.predict_proba(X)[0]
        cls_probs = list(zip(classes, probs))
        cls_probs.sort(key=lambda x: x[1], reverse=True)
        top1 = cls_probs[0][0]
        topk = [c for c, p in cls_probs[:3]]

        true = row["diagnosis"]
        y_true.append(true)
        y_pred.append(top1)

        if true in topk:
            top3_hits += 1

        preds.append({"true": true, "pred_top1": top1, "pred_top3": topk, "probs": {c: float(p) for c, p in cls_probs}})

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    top3 = top3_hits / total

    metrics = {"accuracy": acc, "macro_f1": macro_f1, "top3_accuracy": top3, "n_samples": total}

    (out_dir / "symptoms_eval.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(preds).to_csv(out_dir / "symptoms_predictions.csv", index=False)

    print("метрики:", metrics)


if __name__ == "__main__":
    main()
