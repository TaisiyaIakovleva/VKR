HAM10000_LABELS_RU = {
    "akiec": "Актинический кератоз",
    "bcc": "Базальноклеточный рак",
    "bkl": "Доброкачественный кератоз",
    "df": "Дерматофиброма",
    "mel": "Меланома",
    "nv": "Меланоцитарный невус",
    "vasc": "Сосудистое образование",
}

SKIN_EXTRA_LABELS_RU = {
    "Actinic keratosis": "Актинический кератоз",
    "Atopic Dermatitis": "Атопический дерматит",
    "Benign keratosis": "Доброкачественный кератоз",
    "Dermatofibroma": "Дерматофиброма",
    "Melanocytic nevus": "Меланоцитарный невус",
    "Melanoma": "Меланома",
    "Squamous cell carcinoma": "Плоскоклеточный рак кожи",
    "Tinea Ringworm Candidiasis": "Грибковое поражение кожи",
    "Vascular lesion": "Сосудистое образование",
    "Basal cell carcinoma": "Базальноклеточный рак",
    "Lymphocytic Infiltrate of Jessner": "Лимфоцитарная инфильтрация Джесснера",
    "Infantile Atopic Dermatitis": "Детский атопический дерматит",
}

CANONICAL_CODES = {
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "bkl": "Benign keratosis",
    "akiec": "Actinic keratosis",
    "vasc": "Vascular lesion",
    "df": "Dermatofibroma",
    "bcc": "Basal cell carcinoma",
    "Melanoma": "Melanoma",
    "Melanocytic nevus": "Melanocytic nevus",
    "Benign keratosis": "Benign keratosis",
    "Actinic keratosis": "Actinic keratosis",
    "Vascular lesion": "Vascular lesion",
    "Dermatofibroma": "Dermatofibroma",
    "Basal cell carcinoma": "Basal cell carcinoma",
    "Atopic Dermatitis": "Atopic Dermatitis",
    "Squamous cell carcinoma": "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis": "Tinea Ringworm Candidiasis",
}

from pathlib import Path
import csv

# Попытка загрузить готовый CSV с переводами SD-198 (label -> ru_label)
SD198_LABELS_RU = {}
_sd198_csv_path = Path("data/processed/sd198_labels_translated.csv")
if _sd198_csv_path.exists():
    try:
        with _sd198_csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                lbl = row.get("label") or row.get("Label")
                ru = row.get("ru_label") or row.get("ru") or row.get("ru_label")
                if lbl and ru:
                    SD198_LABELS_RU[lbl] = ru
    except Exception:
        # не критично — продолжим использовать словари и эвристику
        SD198_LABELS_RU = {}


def normalize_diagnosis_code(code: str) -> str:
    return CANONICAL_CODES.get(code, code)


def get_ru_label(code: str) -> str:
    normalized_code = normalize_diagnosis_code(code)
    if normalized_code in SD198_LABELS_RU:
        return SD198_LABELS_RU[normalized_code]
    if normalized_code in SKIN_EXTRA_LABELS_RU:
        return SKIN_EXTRA_LABELS_RU[normalized_code]
    if normalized_code in HAM10000_LABELS_RU:
        return HAM10000_LABELS_RU[normalized_code]

    # эвристика для остальных кодов SD-198
    return _heuristic_translate(normalized_code)


_TOKEN_TRANSLATIONS = {
    "acne": "акне",
    "keloid": "келоид",
    "vulgaris": "вульгарис",
    "actinic": "актинический",
    "solar": "солнечный",
    "damage": "повреждение",
    "cheilitis": "хейлит",
    "keratosis": "кератоз",
    "seborrheic": "себорейный",
    "cutis": "кожа",
    "rhomboidalis": "ромбовидный",
    "pigmentation": "пигментация",
    "elastosis": "эластоз",
    "purpura": "пурпура",
    "telangiectasia": "телеангиэктазия",
    "eczema": "экзема",
    "dermatitis": "дерматит",
    "alopecia": "алопеция",
    "areata": "areata",
    "androgenetic": "андрогенетическая",
    "angioma": "ангиома",
    "apocrine": "апокринный",
    "hydrocystoma": "гидроцистома",
    "arsenical": "мышьяковый",
    "balanitis": "баланит",
    "xerotica": "ксеротическая",
    "obliterans": "облитерирующий",
    "basal": "базальноклеточный",
    "cell": "клеточный",
    "carcinoma": "карцинома",
    "beau's": "линии Бё",
    "nevus": "невус",
    "melanoma": "меланома",
    "dermatofibroma": "дерматофиброма",
    "candidiasis": "кандидоз",
    "cellulitis": "целлюлит",
    "chalazion": "халазион",
    "fibroma": "фиброма",
    "psoriasis": "псориаз",
    "lichen": "лишай",
    "planus": "плоский",
    "kerion": "керион",
    "melasma": "мелазма",
    "lipoma": "липома",
    "pustular": "пустулёзный",
    "follicular": "фолликулярный",
    "granuloma": "гранулёма",
    "ulcer": "язва",
    "impetigo": "импетиго",
    "onychomycosis": "онихомикоз",
    "onycholysis": "ониколизис",
    "onychorrhexis": "онихорексис",
    "steatocystoma": "стеатокистома",
    "poroma": "порома",
    "drug": "лекарственная",
    "eruption": "сыпь",
    "malignant": "злокачественная",
    "congenital": "врождённый",
    "seborrheic_keratosis": "себорейный кератоз",
    "lymphocytic": "лимфоцитарная",
    "infiltrate": "инфильтрация",
    "jessner": "Джесснера",
    "infantile": "детский",
    "atopic": "атопический",
    "of": "",
}


def _heuristic_translate(code: str) -> str:
    main = code
    paren = None
    if "(" in code and ")" in code:
        left = code.find("(")
        right = code.rfind(")")
        main = code[:left]
        paren = code[left + 1:right]

    def translate_part(part: str) -> str:
        tokens = [t for t in part.replace("_", " ").split() if t]
        out_tokens = []
        for t in tokens:
            key = t.strip().lower().strip("'\".,")
            if key in _TOKEN_TRANSLATIONS:
                out_tokens.append(_TOKEN_TRANSLATIONS[key])
            else:
                out_tokens.append(t.replace("_", " ").lower())
        phrase = " ".join(out_tokens).strip()
        if not phrase:
            return part
        return phrase[0].upper() + phrase[1:]

    translated = translate_part(main)
    if paren:
        translated_paren = translate_part(paren)
        return f"{translated} ({translated_paren})"
    return translated