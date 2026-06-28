def choose_model(symptoms: dict) -> str:
    # читаем симптомы
    itching = int(symptoms.get("itching", 0))
    peeling = int(symptoms.get("peeling", 0))
    oozing = int(symptoms.get("oozing", 0))
    pus = int(symptoms.get("pus", 0))
    affected_area = float(symptoms.get("affected_area", 0))
    localization = symptoms.get("localization", [])
    evolution = symptoms.get("evolution", [])

    # считаем балл воспалительных признаков
    inflammatory_score = 0

    if itching == 1:
        inflammatory_score += 1
    if peeling == 1:
        inflammatory_score += 1
    if oozing == 1:
        inflammatory_score += 1
    if pus == 1:
        inflammatory_score += 1
    if affected_area >= 10:
        inflammatory_score += 1

    if any(loc in {"face", "trunk", "arms", "legs", "folds", "feet", "hair", "nails"} for loc in localization):
        inflammatory_score += 1

    if len(evolution) == 0:
        inflammatory_score += 1

    # если много воспалительных симптомов — выбираем модель с широким набором болезней
    if inflammatory_score >= 3:
        return "skin_extra"

    return "ham10000"