from pathlib import Path

import joblib
import numpy as np


class SymptomsGroupPredictor:
    def __init__(self, model_dir: str = "models/symptoms_group_model"):
        model_dir = Path(model_dir)
        # загружаем обученную модель, энкодер меток и список признаков
        self.model = joblib.load(model_dir / "symptoms_group_model.joblib")
        self.label_encoder = joblib.load(model_dir / "group_label_encoder.joblib")
        self.feature_names = joblib.load(model_dir / "feature_names.joblib")

    def predict(self, symptoms_vector, top_n: int = 3):
        # если пришёл dict -> преобразуем в список признаков
        if isinstance(symptoms_vector, dict):
            x = [[symptoms_vector.get(f, 0) for f in self.feature_names]]
        else:
            x = [symptoms_vector]

        probs = self.model.predict_proba(x)[0]

        # берём топ-N групп по вероятности
        top_idx = np.argsort(probs)[::-1][:top_n]

        results = []

        for idx in top_idx:
            group_code = self.label_encoder.inverse_transform([idx])[0]

            results.append(
                {
                    "diagnosis_code": group_code,
                    "probability": float(probs[idx]),
                }
            )

        return results