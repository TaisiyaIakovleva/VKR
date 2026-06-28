#!/usr/bin/env python3
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "experiments" / "error_analysis"
FIG_DIR = ANALYSIS_DIR / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(FIG_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = {
    "HAM10000": "#2F6F73",
    "Skin Extra": "#D08C60",
    "SD-198": "#6C5B7B",
    "Inside category": "#4C78A8",
    "Between categories": "#D65F5F",
}


def save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def setup() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    FIG_DIR.mkdir(exist_ok=True)


def overlap_recall() -> Path:
    df = pd.read_csv(ANALYSIS_DIR / "overlap_quality_comparison.csv")
    df = df[df["disease_group"] != "Squamous cell carcinoma"].copy()
    plot_df = df.melt(
        id_vars=["disease_group"],
        value_vars=["ham10000_recall", "skin_extra_recall", "sd198_group_recall"],
        var_name="dataset",
        value_name="recall",
    ).dropna()
    plot_df["dataset"] = plot_df["dataset"].map(
        {
            "ham10000_recall": "HAM10000",
            "skin_extra_recall": "Skin Extra",
            "sd198_group_recall": "SD-198",
        }
    )
    order = (
        plot_df[plot_df["dataset"] == "SD-198"]
        .sort_values("recall")["disease_group"]
        .tolist()
    )

    plt.figure(figsize=(11, 6.2))
    ax = sns.barplot(
        data=plot_df,
        y="disease_group",
        x="recall",
        hue="dataset",
        order=order,
        palette=PALETTE,
    )
    ax.set_title("Сравнение recall на пересекающихся заболеваниях")
    ax.set_xlabel("Recall")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.05)
    ax.legend(title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    path = FIG_DIR / "overlap_recall_comparison_seaborn.png"
    save(path)
    return path


def category_heatmap() -> Path:
    df = pd.read_csv(ANALYSIS_DIR / "sd198_category_confusion_summary.csv")
    matrix = df.pivot_table(
        index="true_group",
        columns="predicted_group",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    order = matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
    matrix = matrix.reindex(index=order, columns=order, fill_value=0)

    plt.figure(figsize=(10.5, 8.5))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="#E4E7EC",
        cbar_kws={"label": "Количество случаев"},
    )
    ax.set_title("Матрица ошибок SD-198 по категориям")
    ax.set_xlabel("Предсказанная категория")
    ax.set_ylabel("Истинная категория")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path = FIG_DIR / "sd198_category_confusion_heatmap_seaborn.png"
    save(path)
    return path


def category_heatmap_shares() -> Path:
    df = pd.read_csv(ANALYSIS_DIR / "sd198_category_confusion_summary.csv")
    matrix = df.pivot_table(
        index="true_group",
        columns="predicted_group",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    order = matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
    matrix = matrix.reindex(index=order, columns=order, fill_value=0)
    shares = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)

    annot = shares.map(lambda value: f"{value:.0%}" if value >= 0.005 else "")

    plt.figure(figsize=(10.5, 8.5))
    ax = sns.heatmap(
        shares,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="#E4E7EC",
        cbar_kws={"label": "Доля от истинной категории"},
    )
    ax.set_title("Нормализованная матрица ошибок SD-198 по категориям")
    ax.set_xlabel("Предсказанная категория")
    ax.set_ylabel("Истинная категория")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path = FIG_DIR / "sd198_category_confusion_share_heatmap_seaborn.png"
    save(path)
    return path


def top_confusions() -> Path:
    df = pd.read_csv(ANALYSIS_DIR / "sd198_top_confusions_grouped.csv").head(20).copy()
    if "true_support" in df.columns:
        df["pair"] = (
            df["true"]
            + " -> "
            + df["predicted"]
            + "  ("
            + df["count"].astype(str)
            + "/"
            + df["true_support"].astype(str)
            + ")"
        )
    else:
        df["pair"] = df["true"] + " -> " + df["predicted"]
    df["error_type"] = df["same_group_error"].map(
        {
            True: "Inside category",
            False: "Between categories",
            "True": "Inside category",
            "False": "Between categories",
        }
    )

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df,
        y="pair",
        x="count",
        hue="error_type",
        dodge=False,
        palette=PALETTE,
    )
    ax.set_title("Топ ошибок SD-198")
    ax.set_xlabel("Количество ошибок")
    ax.set_ylabel("")
    ax.legend(title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)

    path = FIG_DIR / "sd198_top_confusions_bar_seaborn.png"
    save(path)
    return path


def error_type_by_category() -> Path:
    df = pd.read_csv(ANALYSIS_DIR / "sd198_category_confusion_summary.csv")
    df["error_type"] = df["same_group"].map(
        {
            True: "Inside category",
            False: "Between categories",
            "True": "Inside category",
            "False": "Between categories",
        }
    )
    grouped = df.groupby(["true_group", "error_type"], as_index=False)["count"].sum()
    totals = grouped.groupby("true_group")["count"].sum().sort_values(ascending=False)
    grouped["true_group"] = pd.Categorical(
        grouped["true_group"], categories=totals.index, ordered=True
    )
    grouped = grouped.sort_values("true_group")

    plt.figure(figsize=(10.5, 6.2))
    ax = sns.barplot(
        data=grouped,
        y="true_group",
        x="count",
        hue="error_type",
        palette=PALETTE,
    )
    ax.set_title("Ошибки внутри категории и между категориями")
    ax.set_xlabel("Количество случаев")
    ax.set_ylabel("Истинная категория")
    ax.legend(title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)

    path = FIG_DIR / "sd198_error_type_by_category_seaborn.png"
    save(path)
    return path


def main() -> None:
    setup()
    paths = [
        overlap_recall(),
        category_heatmap(),
        category_heatmap_shares(),
        top_confusions(),
        error_type_by_category(),
    ]
    print("графики:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
