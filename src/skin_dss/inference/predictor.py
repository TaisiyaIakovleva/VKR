from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from skin_dss.data.ham10000_dataset import get_eval_transforms
from skin_dss.utils.skin_extra_labels import load_skin_extra_labels
from skin_dss.data.sd198_dataset import SD198Dataset, get_sd198_eval_transforms

HAM10000_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _BasePredictor:
    def _load_model(self, model_path, num_classes):
        # берём EfficientNet-B0 с предобученными весами и меняем последний слой под наши классы
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        # загружаем сохранённые веса модели
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image: Image.Image, top_n: Optional[int] = 3):
        # приводим к RGB и применяем трансформации
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            # переводим logits в вероятности
            probs = torch.softmax(logits, dim=1)
            # берём топ-N классов
            k = probs.size(1) if top_n is None or top_n > probs.size(1) else top_n
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                "diagnosis_code": self.class_names[idx.item()],
                "probability": round(float(prob.item()), 4),
            })
        return results


class SkinDiseasePredictor(_BasePredictor):
    def __init__(self, model_path: str | Path):
        self.device = get_device()
        self.class_names = HAM10000_CLASS_NAMES
        self.transform = get_eval_transforms(image_size=224)
        self.model = self._load_model(model_path, len(self.class_names))


class SkinExtraPredictor(_BasePredictor):
    def __init__(self, model_path: str | Path):
        self.device = get_device()
        self.class_names = load_skin_extra_labels()
        self.transform = get_eval_transforms(image_size=224)
        self.model = self._load_model(model_path, len(self.class_names))


class SD198Predictor(_BasePredictor):
    def __init__(self, model_path: str | Path, csv_path: str | Path = "data/processed/sd198/sd198_splits.csv"):
        self.device = get_device()
        ds = SD198Dataset(csv_path, split="train", transform=None)
        self.class_names = ds.class_names
        self.transform = get_sd198_eval_transforms(image_size=224)
        self.model = self._load_model(model_path, len(self.class_names))
