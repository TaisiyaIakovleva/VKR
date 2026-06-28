from pathlib import Path
import pandas as pd

csv_path = Path("experiments/sd198/per_class_metrics_sd198.csv")
out_path = Path("data/processed/sd198_labels.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    raise SystemExit(f"Source file not found: {csv_path}")

df = pd.read_csv(csv_path, index_col=0)
labels = df.index.tolist()

pd.DataFrame({"label": labels}).to_csv(out_path, index=False)
print(f"записано {len(labels)} меток в {out_path}")
