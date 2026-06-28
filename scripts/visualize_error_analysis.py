from __future__ import annotations

import csv
import html
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "error_analysis"
FIG_DIR = ANALYSIS_DIR / "figures"

COLORS = {
    "ham": "#2F6F73",
    "skin": "#D08C60",
    "sd": "#6C5B7B",
    "same": "#4C78A8",
    "cross": "#D65F5F",
    "grid": "#D9DEE5",
    "text": "#1F2933",
    "muted": "#667085",
    "bg": "#FFFFFF",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fnum(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def svg_text(x: float, y: float, text: str, size: int = 13, weight: int = 400, color: str = COLORS["text"], anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(text)}</text>'
    )


def wrap_label(text: str, max_chars: int = 24) -> list[str]:
    words = text.replace("_", " ").split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines[:3]


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        *body,
        "</svg>",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def plot_overlap_recall(rows: list[dict[str, str]]) -> Path:
    rows = [row for row in rows if row["disease_group"] != "Squamous cell carcinoma"]
    width = 1180
    row_h = 58
    top = 95
    left = 245
    chart_w = 760
    height = top + row_h * len(rows) + 85
    body: list[str] = []

    body.append(svg_text(32, 38, "Сравнение recall на пересекающихся заболеваниях", 22, 700))
    body.append(svg_text(32, 64, "HAM10000 и Skin Extra сравниваются с агрегированными группами SD-198", 13, 400, COLORS["muted"]))

    legend = [("HAM10000", COLORS["ham"]), ("Skin Extra", COLORS["skin"]), ("SD-198", COLORS["sd"])]
    lx = left
    for name, color in legend:
        body.append(f'<rect x="{lx}" y="52" width="14" height="14" rx="2" fill="{color}"/>')
        body.append(svg_text(lx + 20, 64, name, 12, 600, COLORS["muted"]))
        lx += 118

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * chart_w
        body.append(f'<line x1="{x:.1f}" y1="{top - 20}" x2="{x:.1f}" y2="{height - 54}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        body.append(svg_text(x, height - 30, f"{tick:.2f}", 11, 400, COLORS["muted"], "middle"))

    bar_h = 11
    for i, row in enumerate(rows):
        y = top + i * row_h
        label_lines = wrap_label(row["disease_group"], 25)
        for j, line in enumerate(label_lines):
            body.append(svg_text(32, y + 8 + j * 15, line, 13, 600 if j == 0 else 400))

        values = [
            ("ham10000_recall", COLORS["ham"], y - 4),
            ("skin_extra_recall", COLORS["skin"], y + 12),
            ("sd198_group_recall", COLORS["sd"], y + 28),
        ]
        for key, color, by in values:
            value = fnum(row[key])
            if value is None:
                body.append(svg_text(left + 4, by + 10, "нет данных", 11, 400, COLORS["muted"]))
                continue
            bw = max(2, value * chart_w)
            body.append(f'<rect x="{left}" y="{by}" width="{bw:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
            body.append(svg_text(left + bw + 7, by + 10, f"{value:.3f}", 11, 600, COLORS["text"]))

    path = FIG_DIR / "overlap_recall_comparison.svg"
    write_svg(path, width, height, body)
    return path


def blend_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    start = (238, 243, 248)
    end = (67, 97, 145)
    rgb = tuple(round(start[i] + (end[i] - start[i]) * value) for i in range(3))
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def plot_category_heatmap(rows: list[dict[str, str]]) -> Path:
    categories = sorted({row["true_group"] for row in rows} | {row["predicted_group"] for row in rows})
    matrix = defaultdict(int)
    max_count = 0
    for row in rows:
        count = int(row["count"])
        matrix[(row["true_group"], row["predicted_group"])] += count
        max_count = max(max_count, count)

    cell = 64
    left = 185
    top = 130
    width = left + cell * len(categories) + 45
    height = top + cell * len(categories) + 105
    body: list[str] = []

    body.append(svg_text(32, 38, "Матрица ошибок SD-198 по категориям", 22, 700))
    body.append(svg_text(32, 64, "Строка - истинная категория, столбец - предсказанная категория", 13, 400, COLORS["muted"]))

    for i, category in enumerate(categories):
        x = left + i * cell + cell / 2
        for j, line in enumerate(wrap_label(category, 10)):
            body.append(svg_text(x, 92 + j * 13, line, 10, 600, COLORS["muted"], "middle"))

    for r, true_group in enumerate(categories):
        y = top + r * cell
        for j, line in enumerate(wrap_label(true_group, 20)):
            body.append(svg_text(32, y + 26 + j * 13, line, 11, 600 if j == 0 else 400, COLORS["text"]))
        for c, pred_group in enumerate(categories):
            x = left + c * cell
            count = matrix[(true_group, pred_group)]
            color = blend_color(math.sqrt(count / max_count)) if count else "#F7F9FC"
            stroke = "#7A8798" if true_group == pred_group else COLORS["grid"]
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="{stroke}" stroke-width="1"/>')
            if count:
                text_color = "#FFFFFF" if count / max_count > 0.35 else COLORS["text"]
                body.append(svg_text(x + cell / 2, y + 38, str(count), 14, 700, text_color, "middle"))

    body.append(svg_text(32, height - 35, "Диагональ показывает ошибки/попадания внутри одной укрупненной категории.", 12, 400, COLORS["muted"]))
    path = FIG_DIR / "sd198_category_confusion_heatmap.svg"
    write_svg(path, width, height, body)
    return path


def plot_top_confusions(rows: list[dict[str, str]], limit: int = 20) -> Path:
    rows = rows[:limit]
    width = 1160
    left = 420
    top = 92
    row_h = 30
    chart_w = 590
    height = top + row_h * len(rows) + 70
    max_count = max(int(row["count"]) for row in rows) if rows else 1
    body: list[str] = []

    body.append(svg_text(32, 38, "Топ ошибок SD-198", 22, 700))
    body.append(svg_text(32, 64, "Синие - внутри категории, красные - между разными категориями", 13, 400, COLORS["muted"]))

    for i, row in enumerate(rows):
        y = top + i * row_h
        label = f"{row['true']} -> {row['predicted']}"
        body.append(svg_text(32, y + 14, label, 11, 500))
        count = int(row["count"])
        color = COLORS["same"] if row["same_group_error"] == "True" else COLORS["cross"]
        bw = count / max_count * chart_w
        body.append(f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="17" rx="4" fill="{color}"/>')
        body.append(svg_text(left + bw + 8, y + 14, str(count), 12, 700))
        body.append(svg_text(left + chart_w + 54, y + 14, f"{row['true_group']} -> {row['predicted_group']}", 10, 400, COLORS["muted"]))

    path = FIG_DIR / "sd198_top_confusions_bar.svg"
    write_svg(path, width, height, body)
    return path


def plot_error_type_by_category(rows: list[dict[str, str]]) -> Path:
    totals = defaultdict(lambda: Counter({"same": 0, "cross": 0}))
    for row in rows:
        key = "same" if row["same_group"] == "True" else "cross"
        totals[row["true_group"]][key] += int(row["count"])
    categories = sorted(totals, key=lambda cat: -(totals[cat]["same"] + totals[cat]["cross"]))

    width = 1000
    left = 230
    top = 88
    row_h = 40
    chart_w = 610
    height = top + row_h * len(categories) + 70
    max_total = max(totals[cat]["same"] + totals[cat]["cross"] for cat in categories)
    body: list[str] = []

    body.append(svg_text(32, 38, "Ошибки внутри категории и между категориями", 22, 700))
    body.append(svg_text(32, 64, "По истинной категории SD-198", 13, 400, COLORS["muted"]))
    body.append(f'<rect x="{left}" y="50" width="14" height="14" rx="2" fill="{COLORS["same"]}"/>')
    body.append(svg_text(left + 20, 62, "внутри категории", 12, 600, COLORS["muted"]))
    body.append(f'<rect x="{left + 150}" y="50" width="14" height="14" rx="2" fill="{COLORS["cross"]}"/>')
    body.append(svg_text(left + 170, 62, "между категориями", 12, 600, COLORS["muted"]))

    for i, cat in enumerate(categories):
        y = top + i * row_h
        same = totals[cat]["same"]
        cross = totals[cat]["cross"]
        total = same + cross
        scale = chart_w / max_total
        same_w = same * scale
        cross_w = cross * scale
        body.append(svg_text(32, y + 17, cat, 12, 600))
        body.append(f'<rect x="{left}" y="{y}" width="{same_w:.1f}" height="20" rx="4" fill="{COLORS["same"]}"/>')
        body.append(f'<rect x="{left + same_w:.1f}" y="{y}" width="{cross_w:.1f}" height="20" rx="4" fill="{COLORS["cross"]}"/>')
        body.append(svg_text(left + same_w + cross_w + 8, y + 16, f"{total} ({cross} между)", 11, 600))

    path = FIG_DIR / "sd198_error_type_by_category.svg"
    write_svg(path, width, height, body)
    return path


def update_markdown(figures: list[Path]) -> None:
    md_path = ANALYSIS_DIR / "analysis_summary.md"
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    marker = "## Визуализации"
    block = [
        marker,
        "",
        "Эти SVG-файлы можно открыть в браузере или вставить в презентацию:",
        "",
    ]
    for path in figures:
        rel = path.relative_to(ANALYSIS_DIR)
        title = path.stem.replace("_", " ")
        block.append(f"- [{title}]({rel})")
    block.append("")

    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n" + "\n".join(block)
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block)
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    overlap = read_csv(ANALYSIS_DIR / "overlap_quality_comparison.csv")
    categories = read_csv(ANALYSIS_DIR / "sd198_category_confusion_summary.csv")
    top_confusions = read_csv(ANALYSIS_DIR / "sd198_top_confusions_grouped.csv")

    figures = [
        plot_overlap_recall(overlap),
        plot_category_heatmap(categories),
        plot_top_confusions(top_confusions),
        plot_error_type_by_category(categories),
    ]
    update_markdown(figures)

    print("графики:")
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
