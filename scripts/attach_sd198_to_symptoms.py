from pathlib import Path
import csv


SYM_PATH = Path("data/symptoms/symptoms_dataset.csv")
LABELS_CSV = Path("data/processed/sd198_labels.csv")
PER_CLASS = Path("experiments/sd198/per_class_metrics_sd198.csv")
OUT_PATH = Path("data/symptoms/symptoms_dataset_with_sd198.csv")

if not SYM_PATH.exists():
    raise SystemExit(f"Symptoms CSV не найден: {SYM_PATH}")

labels = []
if LABELS_CSV.exists():
    with LABELS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'label' in row and row['label']:
                labels.append(row['label'])
            else:
                vals = [v for v in row.values()]
                if vals:
                    labels.append(vals[0])
elif PER_CLASS.exists():
    with PER_CLASS.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) > 0:
                labels.append(row[0])
else:
    raise SystemExit("Не найдены метки SD-198: нужен data/processed/sd198_labels.csv или experiments/sd198/per_class_metrics_sd198.csv")

labels = [l.strip() for l in labels if l and l.strip()]
print(f"Загружено {len(labels)} меток")

with SYM_PATH.open("r", encoding="utf-8", newline='') as inf, OUT_PATH.open("w", encoding="utf-8", newline='') as outf:
    reader = csv.DictReader(inf)
    fieldnames = reader.fieldnames[:] if reader.fieldnames else []
    if 'sd198_label' not in fieldnames:
        fieldnames.append('sd198_label')
    if 'sd198_candidates' not in fieldnames:
        fieldnames.append('sd198_candidates')

    writer = csv.DictWriter(outf, fieldnames=fieldnames)
    writer.writeheader()

    candidates_str = ";".join(labels)

    for r in reader:
        r['sd198_label'] = ""
        r['sd198_candidates'] = candidates_str
        writer.writerow(r)

print(f"Записано в: {OUT_PATH}")
