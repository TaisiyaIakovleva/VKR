from pathlib import Path
import pandas as pd

from skin_dss.utils.diagnosis_names import get_ru_label

SRC = Path("experiments/sd198/per_class_metrics_sd198.csv")
OUT_FULL = Path("data/processed/sd198_labels_translated.csv")
OUT_TOP30 = Path("experiments/sd198/top30_labels_translated.csv")
OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
OUT_TOP30.parent.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    raise SystemExit(f"Source file not found: {SRC}")

df = pd.read_csv(SRC, index_col=0)
labels = df.index.tolist()
supports = df['support'].tolist() if 'support' in df.columns else [None]*len(labels)

rows = []
for lbl, sup in zip(labels, supports):
    ru = get_ru_label(lbl)
    rows.append({"label": lbl, "support": int(sup) if pd.notna(sup) else None, "ru_label": ru})

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_FULL, index=False)

if 'support' in df.columns:
    top30 = out_df.sort_values('support', ascending=False).head(30)
else:
    top30 = out_df.head(30)

top30.to_csv(OUT_TOP30, index=False)

print(f"все переводы: {OUT_FULL}")
print(f"топ-30: {OUT_TOP30}")
print(top30.head(10).to_string(index=False))
