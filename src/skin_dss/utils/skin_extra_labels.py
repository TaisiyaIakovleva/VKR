from pathlib import Path
import pandas as pd


def load_skin_extra_labels(
    label_map_path: str | Path = "data/processed/skin_extra_label_map.csv",
):
    df = pd.read_csv(label_map_path)
    df = df.sort_values("label")
    return df["class_name"].tolist()