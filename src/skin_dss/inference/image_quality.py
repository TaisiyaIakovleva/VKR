from typing import Dict, List

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def check_image_quality(image: Image.Image) -> Dict:
    img = pil_to_cv2(image)

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    issues: List[str] = []

    # слишком маленькое
    if width < 224 or height < 224:
        issues.append("Слишком маленькое изображение.")

    # размытость
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20:
        issues.append("Изображение размытое.")

    # яркость
    brightness = float(np.mean(gray))
    if brightness < 40:
        issues.append("Изображение слишком тёмное.")
    elif brightness > 220:
        issues.append("Изображение слишком светлое.")

    # проверка на скриншот/интерфейс
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = float(np.mean(edges > 0))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=80,
        maxLineGap=5,
    )
    line_count = 0 if lines is None else len(lines)

    if edge_ratio > 0.18 and line_count > 20:
        issues.append("Изображение похоже на скриншот, интерфейс или текст, а не на фото кожи.")

    # базовая проверка что на фото есть кожа
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 20, 50], dtype=np.uint8)
    upper1 = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([160, 20, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    skin_mask = cv2.bitwise_or(mask1, mask2)
    skin_ratio = float(np.mean(skin_mask > 0))

    if skin_ratio < 0.05:
        issues.append("Изображение может не содержать достаточного участка кожи.")

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "issues": issues,
        "metrics": {
            "width": width,
            "height": height,
            "blur_score": round(laplacian_var, 2),
            "brightness": round(brightness, 2),
            "edge_ratio": round(edge_ratio, 4),
            "line_count": line_count,
            "skin_ratio": round(skin_ratio, 4),
        },
    }
