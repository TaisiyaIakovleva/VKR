from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from skin_dss.inference.hybrid_predictor import hybrid_fusion
from skin_dss.inference.predictor import SkinDiseasePredictor, SkinExtraPredictor, SD198Predictor
from skin_dss.inference.router import choose_model
from skin_dss.inference.image_quality import check_image_quality
from skin_dss.utils.diagnosis_names import get_ru_label
from skin_dss.utils.recommendations import make_recommendation
from skin_dss.data.symptoms_vectorizer import vectorize_symptoms
from skin_dss.inference.symptoms_group_predictor import SymptomsGroupPredictor

app = FastAPI()
templates = Jinja2Templates(directory="api/templates")

HAM10000_MODEL_PATH = Path("models/efficientnet_b0/best_efficientnet_b0_ham10000.pth")
SKIN_EXTRA_MODEL_PATH = Path("models/skin_extra/best_skin_extra_efficientnet_b0.pth")
SD198_MODEL_PATH = Path("models/sd198/sd198_efficientnet_b0.pth")

# загружаем модели при запуске приложения
ham_predictor = SkinDiseasePredictor(HAM10000_MODEL_PATH)
skin_extra_predictor = SkinExtraPredictor(SKIN_EXTRA_MODEL_PATH)
sd198_predictor = None
if SD198_MODEL_PATH.exists():
    try:
        sd198_predictor = SD198Predictor(SD198_MODEL_PATH)
    except Exception:
        sd198_predictor = None
symptoms_predictor = SymptomsGroupPredictor()

GROUP_LABELS_RU = {
    "tumor": "Опухолевые образования",
    "pigment_nevus": "Пигментные и невусные образования",
    "keratosis": "Кератозы и гиперкератозы",
    "psoriasis": "Псориаз",
    "inflammatory": "Воспалительные заболевания",
    "fungal": "Грибковые заболевания",
    "nail": "Заболевания ногтей",
    "hair": "Заболевания волос и кожи головы",
    "mouth": "Поражения слизистой рта",
    "follicular_acne": "Акне и фолликулярные заболевания",
    "viral": "Вирусные заболевания",
    "vascular": "Сосудистые поражения",
    "infection_ulcer": "Инфекции и язвенные поражения",
    "benign_growth": "Доброкачественные образования",
    "other": "Другое",
}


def prettify_results(results: list[dict]) -> list[dict]:
    pretty = []
    for item in results:
        code = item["diagnosis_code"]
        pretty.append(
            {
                "code": code,
                "label": GROUP_LABELS_RU.get(code, get_ru_label(code)),
                "probability": item["probability"],
            }
        )
    return pretty

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    itching: Optional[str] = Form(None),
    pain: Optional[str] = Form(None),
    bleeding: Optional[str] = Form(None),
    peeling: Optional[str] = Form(None),
    oozing: Optional[str] = Form(None),
    pus: Optional[str] = Form(None),
    clear_border: Optional[str] = Form(None),
    affected_area: Optional[float] = Form(None),
    diameter_mm: Optional[float] = Form(None),
    duration: Optional[str] = Form(None),
    localization: List[str] = Form([]),
    evolution: List[str] = Form([]),
    show_all_image_probs: Optional[str] = Form(None),
):
    # проверяем что передано — фото, симптомы или оба
    has_image = file is not None and file.filename is not None and file.filename != ""

    has_symptoms = any([
        itching is not None,
        pain is not None,
        bleeding is not None,
        peeling is not None,
        oozing is not None,
        pus is not None,
        clear_border is not None,
        affected_area is not None,
        diameter_mm is not None,
        duration is not None and duration != "",
        len(localization) > 0,
        len(evolution) > 0,
    ])

    if not has_image and not has_symptoms:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error_message": "Загрузите изображение, заполните симптомы или используйте оба варианта.",
            },
        )

    image = None
    if has_image:
        # открываем изображение и проверяем качество
        image = Image.open(file.file).convert("RGB")

        quality_check = check_image_quality(image)

        if not quality_check["is_valid"]:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "request": request,
                    "error_message": "Изображение не прошло предварительную проверку качества.",
                    "quality_issues": quality_check["issues"],
                },
            )

    # собираем симптомы в словарь
    symptoms = {
        "itching": 1 if itching else 0,
        "pain": 1 if pain else 0,
        "bleeding": 1 if bleeding else 0,
        "peeling": 1 if peeling else 0,
        "oozing": 1 if oozing else 0,
        "pus": 1 if pus else 0,
        "clear_border": 1 if clear_border else 0,
        "affected_area": float(affected_area) if affected_area is not None else 0.0,
        "diameter_mm": float(diameter_mm) if diameter_mm is not None else 0.0,
        "duration": duration if duration is not None else "",
        "localization": localization if localization else [],
        "evolution": evolution if evolution else [],
    }

    image_top_n = None if show_all_image_probs is not None else 3

    image_results = []
    symptom_results = []
    hybrid_results = []
    selected_model = None
    image_model = None
    analysis_mode = None

    if has_image and has_symptoms:
        # гибридный режим (фото + симптомы)
        selected_model = choose_model(symptoms)

        if sd198_predictor is not None:
            image_model = "sd198"
            image_results = sd198_predictor.predict(image=image, top_n=image_top_n)
        else:
            image_model = selected_model
            if selected_model == "skin_extra":
                image_results = skin_extra_predictor.predict(image=image, top_n=image_top_n)
            else:
                image_results = ham_predictor.predict(image=image, top_n=image_top_n)

        symptoms_vector = vectorize_symptoms(symptoms)
        symptom_results = symptoms_predictor.predict(symptoms_vector, top_n=3)

        # объединяем результаты изображения и симптомов
        hybrid_results = hybrid_fusion(
            image_results=image_results,
            symptom_results=symptom_results,
            selected_model=selected_model,
            top_n=3,
        )

        final_results = hybrid_results
        analysis_mode = "hybrid"

    elif has_image:
        # только изображение
        selected_model = "image_only"
        if sd198_predictor is not None:
            image_model = "sd198"
            image_results = sd198_predictor.predict(image=image, top_n=image_top_n)
        else:
            image_model = "skin_extra"
            image_results = skin_extra_predictor.predict(image=image, top_n=image_top_n)
        final_results = image_results
        analysis_mode = "image_only"

    else:
        # только симптомы
        symptoms_vector = vectorize_symptoms(symptoms)
        symptom_results = symptoms_predictor.predict(symptoms_vector, top_n=3)
        final_results = symptom_results
        analysis_mode = "symptoms_only"

    recommendation = make_recommendation(
        top_diagnosis_code=final_results[0]["diagnosis_code"],
        symptoms=symptoms,
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "filename": file.filename if has_image else None,
            "selected_model": selected_model,
            "image_model": image_model,
            "symptoms": symptoms,
            "image_results": prettify_results(image_results) if image_results else [],
            "symptom_results": prettify_results(symptom_results) if symptom_results else [],
            "hybrid_results": prettify_results(hybrid_results) if hybrid_results else [],
            "recommendation": recommendation,
            "analysis_mode": analysis_mode,
            "has_image": has_image,
            "has_symptoms": has_symptoms,
            "show_all_image_probs": show_all_image_probs is not None,
        },
    )