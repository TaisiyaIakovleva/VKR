from pathlib import Path

DATA_DIR = Path("data/raw/skin_extra")

if not DATA_DIR.exists():
    print("папка не найдена:", DATA_DIR)
    raise SystemExit

print("содержимое:")
for p in sorted(DATA_DIR.iterdir()):
    print("-", p.name)