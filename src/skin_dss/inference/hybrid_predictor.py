from typing import Dict, List
from skin_dss.utils.diagnosis_names import normalize_diagnosis_code


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
        "onych" in d or "nail" in d or "leukonychia" in d
        or "beau" in d or "koilonychia" in d or "terry" in d
        or "subungual" in d or "paronychia" in d
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


def _to_score_map(results: List[dict]) -> Dict[str, float]:
    # собираем вероятности в словарь: код -> вероятность
    score_map = {}

    for item in results:
        code = normalize_diagnosis_code(item["diagnosis_code"])
        score_map[code] = score_map.get(code, 0.0) + float(item["probability"])

    return score_map


def hybrid_fusion(
    image_results: List[dict],
    symptom_results: List[dict],
    selected_model: str,
    top_n: int = 3,
) -> List[dict]:

    image_scores = _to_score_map(image_results)
    symptom_group_scores = _to_score_map(symptom_results)

    # берём диагнозы из изображения и усиливаем их оценкой группы симптомов
    fused = {}

    for diagnosis_code, image_prob in image_scores.items():
        group = get_diagnosis_group(diagnosis_code)

        group_score = symptom_group_scores.get(group, 0.0)

        # если симптомы совпадают с группой диагноза — увеличиваем вероятность
        boost = 1.0 + group_score

        fused[diagnosis_code] = image_prob * boost

    total = sum(fused.values())

    # нормализуем чтобы сумма вероятностей была 1
    if total > 0:
        for key in fused:
            fused[key] /= total

    results = [
        {
            "diagnosis_code": key,
            "probability": value,
        }
        for key, value in fused.items()
    ]

    results.sort(key=lambda x: x["probability"], reverse=True)

    return results[:top_n]