from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw/skin_extra")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_rows(split_dir: Path, split_name: str, class_to_label: dict):
    rows = []

    for cls_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() in ALLOWED_EXTENSIONS:
                rows.append(
                    {
                        "image_path": str(img_path),
                        "class_name": cls_dir.name,
                        "label": class_to_label[cls_dir.name],
                        "source_split": split_name,
                    }
                )

    return rows


train_class_dirs = sorted([p for p in TRAIN_DIR.iterdir() if p.is_dir()])

class_to_label = {cls_dir.name: idx for idx, cls_dir in enumerate(train_class_dirs)}

train_rows = collect_rows(TRAIN_DIR, "train", class_to_label)
val_rows_raw = collect_rows(VAL_DIR, "val", class_to_label)

train_df = pd.DataFrame(train_rows)
val_raw_df = pd.DataFrame(val_rows_raw)

if train_df.empty:
    raise ValueError("No images found in train folder.")

if val_raw_df.empty:
    raise ValueError("No images found in val folder.")

val_df, test_df = train_test_split(
    val_raw_df,
    test_size=0.5,
    stratify=val_raw_df["label"],
    random_state=42,
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

splits_path = OUTPUT_DIR / "skin_extra_splits.csv"
label_map_path = OUTPUT_DIR / "skin_extra_label_map.csv"

final_df.to_csv(splits_path, index=False)

labels_df = pd.DataFrame(
    [{"class_name": k, "label": v} for k, v in class_to_label.items()]
).sort_values("label")
labels_df.to_csv(label_map_path, index=False)

print("сохранено:", splits_path)
print("сохранено:", label_map_path)
print(final_df["split"].value_counts())
print(labels_df)
print("изображений:", len(final_df))