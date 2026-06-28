from pathlib import Path
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download

DATASET_NAME = "resyhgerwshshgdfghsdfgh/SD-198"

ROOT_DIR = Path("data")
RAW_DIR = ROOT_DIR / "raw" / "sd198"
EXTRACT_DIR = RAW_DIR / "extracted"
OUT_DIR = ROOT_DIR / "processed" / "sd198"
OUT_CSV = OUT_DIR / "sd198_splits.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("скачиваем sd-198.zip")

zip_path = hf_hub_download(
    repo_id=DATASET_NAME,
    repo_type="dataset",
    filename="sd-198.zip",
    local_dir=RAW_DIR,
)

zip_path = Path(zip_path)

print("zip скачан:", zip_path)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)


image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
rows = []

for img_path in EXTRACT_DIR.rglob("*"):
    if img_path.suffix.lower() not in image_extensions:
        continue

    diagnosis = img_path.parent.name

    rows.append({
        "image_path": str(img_path),
        "diagnosis": diagnosis,
    })

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("Images not found after extracting ZIP.")

label_names = sorted(df["diagnosis"].unique())
label_to_id = {name: i for i, name in enumerate(label_names)}
df["label"] = df["diagnosis"].map(label_to_id)

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42,
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
final_df = final_df[["image_path", "label", "diagnosis", "split"]]

final_df.to_csv(OUT_CSV, index=False)

print(f"изображений: {len(final_df)}, классов: {final_df['diagnosis'].nunique()}")
print(final_df["split"].value_counts())
print("сохранено:", OUT_CSV)