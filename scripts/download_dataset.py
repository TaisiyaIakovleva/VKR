import kagglehub
from pathlib import Path
import shutil

# скачиваем датасет
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("скачано:", path)

source_dir = Path(path)
target_dir = Path("data/raw/HAM10000")

target_dir.mkdir(parents=True, exist_ok=True)

for item in source_dir.iterdir():
    
    dest = target_dir / item.name
    
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)

print("скопировано в:", target_dir)