from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# пути к данным
DATA_DIR = Path("data/raw/HAM10000")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata_path = DATA_DIR / "HAM10000_metadata.csv"

df = pd.read_csv(metadata_path)

# создаю колонку с путями к изображениям
image_paths = []

for image_id in df["image_id"]:
    
    path1 = DATA_DIR / "HAM10000_images_part_1" / f"{image_id}.jpg"
    path2 = DATA_DIR / "HAM10000_images_part_2" / f"{image_id}.jpg"

    if path1.exists():
        image_paths.append(str(path1))
    else:
        image_paths.append(str(path2))

df["image_path"] = image_paths

# кодирую классы в числа
label_map = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

df["label"] = df["dx"].map(label_map)

# делаю stratified split
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df])

# сохраняю
output_file = OUTPUT_DIR / "ham10000_splits.csv"
final_df.to_csv(output_file, index=False)

print("сохранено:", output_file)
print(final_df["split"].value_counts())
print(final_df["dx"].value_counts())