from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


def main():
    data_path = Path("/Users/Taisia1/Desktop/VKR/symptoms_dataset_sd198.csv")
    model_dir = Path("models/symptoms")
    exp_dir = Path("experiments/symptoms")
    output_name = "random_forest_symptoms_sd198.joblib"

    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"датасет: {data_path}, примеров={len(df)}, классов={df['diagnosis'].nunique()}")

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    test_size = max(y.nunique(), int(round(len(df) * 0.2)))
    test_size_ratio = test_size / len(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_ratio,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        class_weight=None,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=4)

    print(f"acc={round(acc, 4)}, f1={round(macro_f1, 4)}")
    print(report)

    model_path = model_dir / output_name
    joblib.dump(model, model_path)

    feature_df = pd.DataFrame({"feature_name": X.columns.tolist()})
    feature_df.to_csv(model_dir / "feature_names.csv", index=False)

    (exp_dir / f"classification_report_{model_path.stem}.txt").write_text(report, encoding="utf-8")

    print("модель:", model_path)
    print("признаки:", model_dir / "feature_names.csv")


if __name__ == "__main__":
    main()