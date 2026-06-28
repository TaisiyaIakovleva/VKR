from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "error_analysis"


HAM_LABELS = {
    "0": "Actinic keratosis",
    "1": "Basal cell carcinoma",
    "2": "Benign keratosis",
    "3": "Dermatofibroma",
    "4": "Melanoma",
    "5": "Melanocytic nevus",
    "6": "Vascular lesion",
}


CATEGORY_BY_DISEASE = {
    "Actinic keratosis": "precancer_keratosis",
    "Basal cell carcinoma": "tumor",
    "Benign keratosis": "benign_keratosis",
    "Dermatofibroma": "benign_tumor",
    "Melanoma": "pigmented_malignant",
    "Melanocytic nevus": "pigmented_benign",
    "Vascular lesion": "vascular",
    "Atopic Dermatitis": "eczema_dermatitis",
    "Tinea Ringworm Candidiasis": "fungal",
    "Squamous cell carcinoma": "tumor",
}


OVERLAP_GROUPS = [
    {
        "disease": "Actinic keratosis",
        "ham_label": "0",
        "skin_label": "0",
        "sd198_labels": ["Actinic_solar_Damage(Actinic_Keratosis)"],
    },
    {
        "disease": "Basal cell carcinoma",
        "ham_label": "1",
        "skin_label": None,
        "sd198_labels": ["Basal_Cell_Carcinoma"],
    },
    {
        "disease": "Benign keratosis",
        "ham_label": "2",
        "skin_label": "2",
        "sd198_labels": ["Benign_Keratosis", "Seborrheic_Keratosis"],
    },
    {
        "disease": "Dermatofibroma",
        "ham_label": "3",
        "skin_label": "3",
        "sd198_labels": ["Dermatofibroma"],
    },
    {
        "disease": "Melanoma",
        "ham_label": "4",
        "skin_label": "5",
        "sd198_labels": ["Malignant_Melanoma", "Lentigo_Maligna_Melanoma"],
    },
    {
        "disease": "Melanocytic nevus",
        "ham_label": "5",
        "skin_label": "4",
        "sd198_labels": [
            "Becker's_Nevus",
            "Blue_Nevus",
            "Compound_Nevus",
            "Congenital_Nevus",
            "Dysplastic_Nevus",
            "Halo_Nevus",
            "Junction_Nevus",
            "Nevus_Incipiens",
            "Nevus_Spilus",
        ],
    },
    {
        "disease": "Vascular lesion",
        "ham_label": "6",
        "skin_label": "8",
        "sd198_labels": [
            "Angioma",
            "Lymphangioma_Circumscriptum",
            "Pyogenic_Granuloma",
            "Strawberry_Hemangioma",
        ],
    },
    {
        "disease": "Atopic Dermatitis",
        "ham_label": None,
        "skin_label": "1",
        "sd198_labels": ["Infantile_Atopic_Dermatitis"],
    },
    {
        "disease": "Tinea Ringworm Candidiasis",
        "ham_label": None,
        "skin_label": "7",
        "sd198_labels": [
            "Candidiasis",
            "Tinea_Corporis",
            "Tinea_Cruris",
            "Tinea_Faciale",
            "Tinea_Manus",
            "Tinea_Pedis",
            "Tinea_Versicolor",
        ],
    },
    {
        "disease": "Squamous cell carcinoma",
        "ham_label": None,
        "skin_label": "6",
        "sd198_labels": [],
        "note": "No exact SD-198 label found; Bowen's disease/keratoacanthoma are related but not identical.",
    },
]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def metric_from_report(report: dict, label: str, metric: str) -> float | None:
    row = report.get(label)
    if not row:
        return None
    key = "f1-score" if metric == "f1" else metric
    value = row.get(key)
    return float(value) if value is not None else None


def support_from_report(report: dict, label: str) -> int:
    row = report.get(label)
    if not row:
        return 0
    return int(float(row.get("support", 0)))


def read_skin_label_names(path: Path) -> dict[str, str]:
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["label"]] = row["class_name"]
    return out


def read_sd_metrics(path: Path) -> dict[str, dict[str, float]]:
    metrics = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        class_col = reader.fieldnames[0] if reader.fieldnames else ""
        for row in reader:
            label = row[class_col]
            metrics[label] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "support": int(float(row["support"])),
            }
    return metrics


def read_sd_confusion(path: Path) -> tuple[list[str], dict[str, dict[str, int]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        labels = [name for name in reader.fieldnames[1:]]
        matrix = {}
        row_label_col = reader.fieldnames[0]
        for row in reader:
            true_label = row[row_label_col]
            matrix[true_label] = {pred: int(row[pred]) for pred in labels}
    return labels, matrix


def aggregate_sd_group(
    labels: list[str],
    matrix: dict[str, dict[str, int]],
    group_labels: list[str],
) -> dict[str, float | int | None]:
    present = [label for label in group_labels if label in matrix]
    if not present:
        return {
            "support": 0,
            "tp": 0,
            "recall": None,
            "precision": None,
            "f1": None,
            "outside_errors": 0,
            "inside_errors": 0,
        }

    group = set(present)
    support = sum(sum(matrix[t].values()) for t in present)
    tp = sum(matrix[t].get(p, 0) for t in present for p in present)
    predicted_as_group = sum(matrix[t].get(p, 0) for t in labels for p in present)
    outside_errors = sum(matrix[t].get(p, 0) for t in present for p in labels if p not in group)
    inside_errors = sum(matrix[t].get(p, 0) for t in present for p in present if p != t)

    recall = tp / support if support else None
    precision = tp / predicted_as_group if predicted_as_group else None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "support": support,
        "tp": tp,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "outside_errors": outside_errors,
        "inside_errors": inside_errors,
    }


def group_for_sd_label(label: str) -> str:
    lower = label.lower()
    if "nevus" in lower or "melanoma" in lower or "lentigo" in lower:
        return "pigmented_lesions"
    if "carcinoma" in lower or "bowen" in lower or "keratoacanthoma" in lower:
        return "tumor"
    if "keratosis" in lower or "actinic" in lower or "porokeratosis" in lower:
        return "keratosis"
    if "eczema" in lower or "dermatitis" in lower:
        return "eczema_dermatitis"
    if "tinea" in lower or "candidiasis" in lower or "onychomycosis" in lower:
        return "fungal"
    if "angioma" in lower or "hemangioma" in lower or "vascular" in lower or "pyogenic" in lower:
        return "vascular"
    if "psoriasis" in lower:
        return "psoriasis"
    if "nail" in lower or "onycho" in lower or "koilonychia" in lower:
        return "nail"
    return "other"


def top_group_confusions(
    labels: list[str],
    matrix: dict[str, dict[str, int]],
    limit: int = 30,
) -> list[dict[str, str | int]]:
    rows = []
    for true_label in labels:
        true_support = sum(matrix[true_label].values())
        for pred_label, count in matrix[true_label].items():
            if true_label == pred_label or count <= 0:
                continue
            true_group = group_for_sd_label(true_label)
            pred_group = group_for_sd_label(pred_label)
            rows.append(
                {
                    "count": count,
                    "true_support": true_support,
                    "share_of_true_class": f"{count / true_support:.3f}" if true_support else "0.000",
                    "true": true_label,
                    "predicted": pred_label,
                    "true_group": true_group,
                    "predicted_group": pred_group,
                    "same_group_error": str(true_group == pred_group),
                }
            )
    rows.sort(key=lambda row: (-int(row["count"]), row["true"], row["predicted"]))
    return rows[:limit]


def category_confusion_summary(
    labels: list[str],
    matrix: dict[str, dict[str, int]],
) -> list[dict[str, str | int | float]]:
    counts = Counter()
    support = Counter()
    correct = Counter()
    for true_label in labels:
        true_group = group_for_sd_label(true_label)
        for pred_label, count in matrix[true_label].items():
            pred_group = group_for_sd_label(pred_label)
            counts[(true_group, pred_group)] += count
            support[true_group] += count
            if true_group == pred_group:
                correct[true_group] += count

    rows = []
    for (true_group, pred_group), count in counts.items():
        if count == 0:
            continue
        rows.append(
            {
                "true_group": true_group,
                "predicted_group": pred_group,
                "count": count,
                "share_of_true_group": count / support[true_group] if support[true_group] else 0.0,
                "same_group": str(true_group == pred_group),
            }
        )
    rows.sort(key=lambda row: (-int(row["count"]), row["true_group"], row["predicted_group"]))
    return rows


def stable_sd_classes(metrics: dict[str, dict[str, float]], min_support: int = 5) -> list[dict[str, str]]:
    rows = []
    for label, values in metrics.items():
        if values["support"] >= min_support and values["recall"] >= 0.9 and values["precision"] >= 0.8:
            rows.append(
                {
                    "label": label,
                    "category": group_for_sd_label(label),
                    "precision": fmt(values["precision"]),
                    "recall": fmt(values["recall"]),
                    "f1": fmt(values["f1"]),
                    "support": str(int(values["support"])),
                }
            )
    rows.sort(key=lambda row: (-float(row["recall"]), -float(row["precision"]), row["label"]))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ham_eval = read_json(ROOT / "experiments" / "eval_efficientnet_b0.json")
    skin_eval = read_json(ROOT / "experiments" / "eval_skin_extra.json")
    skin_names = read_skin_label_names(ROOT / "data" / "processed" / "skin_extra_label_map.csv")
    sd_metrics = read_sd_metrics(ROOT / "experiments" / "sd198" / "per_class_metrics_sd198.csv")
    sd_labels, sd_matrix = read_sd_confusion(
        ROOT / "experiments" / "sd198" / "confusion_matrix_counts_sd198.csv"
    )

    comparison_rows = []
    for group in OVERLAP_GROUPS:
        disease = group["disease"]
        ham_label = group.get("ham_label")
        skin_label = group.get("skin_label")
        sd_labels_for_group = group.get("sd198_labels", [])
        sd_group_metrics = aggregate_sd_group(sd_labels, sd_matrix, sd_labels_for_group)

        comparison_rows.append(
            {
                "disease_group": disease,
                "category": CATEGORY_BY_DISEASE.get(disease, "other"),
                "ham10000_recall": fmt(metric_from_report(ham_eval["classification_report"], ham_label, "recall"))
                if ham_label is not None
                else "",
                "ham10000_f1": fmt(metric_from_report(ham_eval["classification_report"], ham_label, "f1"))
                if ham_label is not None
                else "",
                "ham10000_support": str(support_from_report(ham_eval["classification_report"], ham_label))
                if ham_label is not None
                else "",
                "skin_extra_recall": fmt(
                    metric_from_report(skin_eval["classification_report"], skin_label, "recall")
                )
                if skin_label is not None
                else "",
                "skin_extra_f1": fmt(metric_from_report(skin_eval["classification_report"], skin_label, "f1"))
                if skin_label is not None
                else "",
                "skin_extra_support": str(support_from_report(skin_eval["classification_report"], skin_label))
                if skin_label is not None
                else "",
                "skin_extra_class": skin_names.get(skin_label, "") if skin_label is not None else "",
                "sd198_group_recall": fmt(sd_group_metrics["recall"]),
                "sd198_group_f1": fmt(sd_group_metrics["f1"]),
                "sd198_support": str(sd_group_metrics["support"]),
                "sd198_inside_group_errors": str(sd_group_metrics["inside_errors"]),
                "sd198_outside_group_errors": str(sd_group_metrics["outside_errors"]),
                "sd198_labels": "; ".join(sd_labels_for_group),
                "note": group.get("note", ""),
            }
        )

    write_csv(
        OUT_DIR / "overlap_quality_comparison.csv",
        comparison_rows,
        [
            "disease_group",
            "category",
            "ham10000_recall",
            "ham10000_f1",
            "ham10000_support",
            "skin_extra_recall",
            "skin_extra_f1",
            "skin_extra_support",
            "skin_extra_class",
            "sd198_group_recall",
            "sd198_group_f1",
            "sd198_support",
            "sd198_inside_group_errors",
            "sd198_outside_group_errors",
            "sd198_labels",
            "note",
        ],
    )

    top_confusions = top_group_confusions(sd_labels, sd_matrix, limit=40)
    write_csv(
        OUT_DIR / "sd198_top_confusions_grouped.csv",
        top_confusions,
        [
            "count",
            "true_support",
            "share_of_true_class",
            "true",
            "predicted",
            "true_group",
            "predicted_group",
            "same_group_error",
        ],
    )

    category_rows = category_confusion_summary(sd_labels, sd_matrix)
    write_csv(
        OUT_DIR / "sd198_category_confusion_summary.csv",
        [
            {
                **row,
                "share_of_true_group": fmt(float(row["share_of_true_group"])),
            }
            for row in category_rows
        ],
        ["true_group", "predicted_group", "count", "share_of_true_group", "same_group"],
    )

    stable_rows = stable_sd_classes(sd_metrics)
    write_csv(
        OUT_DIR / "sd198_stable_high_quality_classes.csv",
        stable_rows,
        ["label", "category", "precision", "recall", "f1", "support"],
    )

    cross_group = [row for row in top_confusions if row["same_group_error"] == "False"]
    overlap_sorted = sorted(
        comparison_rows,
        key=lambda row: (
            float(row["sd198_group_recall"]) if row["sd198_group_recall"] else 2.0,
            row["disease_group"],
        ),
    )

    md = [
        "# Анализ ошибок между датасетами",
        "",
        "## Что сравнивать",
        "",
        "Для защиты удобно показывать не все 198 классов SD-198, а группы заболеваний, которые пересекаются со старыми датасетами. "
        "В таблице `overlap_quality_comparison.csv` сравниваются recall/F1 на HAM10000, Skin Extra и агрегированное качество на соответствующих классах SD-198.",
        "",
        "Такой формат уменьшает объем таблицы: устойчивые классы можно описать обобщенно, а подробно разобрать только группы, где качество снизилось или модель часто путает диагнозы.",
        "",
        "## Пересекающиеся заболевания",
        "",
    ]
    for row in overlap_sorted:
        md.append(
            f"- {row['disease_group']}: HAM10000 recall={row['ham10000_recall'] or 'нет'}, "
            f"Skin Extra recall={row['skin_extra_recall'] or 'нет'}, "
            f"SD-198 grouped recall={row['sd198_group_recall'] or 'нет'}, "
            f"support в SD-198={row['sd198_support']}."
        )

    md.extend(
        [
            "",
            "## Самые заметные ошибки SD-198",
            "",
        ]
    )
    for row in top_confusions[:12]:
        same = "внутри категории" if row["same_group_error"] == "True" else "между категориями"
        md.append(
            f"- {row['true']} -> {row['predicted']}: {row['count']} из {row['true_support']} "
            f"({float(row['share_of_true_class']):.1%}), {same} "
            f"({row['true_group']} -> {row['predicted_group']})."
        )

    md.extend(
        [
            "",
            "## Ошибки между разными категориями",
            "",
        ]
    )
    for row in cross_group[:10]:
        md.append(
            f"- {row['true']} -> {row['predicted']}: {row['count']} из {row['true_support']} "
            f"({float(row['share_of_true_class']):.1%}) "
            f"({row['true_group']} -> {row['predicted_group']})."
        )

    md.extend(
        [
            "",
            "## Устойчиво распознаваемые классы",
            "",
            f"Классов с precision >= 0.8, recall >= 0.9 и support >= 5: {len(stable_rows)}.",
        ]
    )
    (OUT_DIR / "analysis_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"анализ записан в {OUT_DIR}")


if __name__ == "__main__":
    main()
