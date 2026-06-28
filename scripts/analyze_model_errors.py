import json
from pathlib import Path

import pandas as pd


def analyze_sd198(perf_csv: Path, out_dir: Path, min_support: int = 5, recall_thresh: float = 0.5):
    df = pd.read_csv(perf_csv, index_col=0)
    df = df.sort_values(by="recall")
    df['support'] = df['support'].astype(int)
    candidates = df[(df['support'] >= min_support) & (df['recall'] < recall_thresh)]
    out = {
        'n_classes': len(df),
        'n_candidates': len(candidates),
        'candidates': candidates[['precision','recall','f1','support']].to_dict(orient='index')
    }
    (out_dir / 'sd198_error_cases.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    return out


def analyze_symptoms(preds_csv: Path, out_dir: Path, min_support: int = 5, recall_thresh: float = 0.5):
    df = pd.read_csv(preds_csv)
    y_true = df['true']
    y_pred = df['pred_top1']
    per = y_true.value_counts().to_dict()
    classes = sorted(set(y_true.unique()) | set(y_pred.unique()))
    rows = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        support = int((y_true == c).sum())
        recall = tp / support if support > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append((c, precision, recall, f1, support, tp, fp, fn))

    dfp = pd.DataFrame(rows, columns=['class','precision','recall','f1','support','tp','fp','fn'])
    dfp = dfp.sort_values('recall')
    candidates = dfp[(dfp['support'] >= min_support) & (dfp['recall'] < recall_thresh)]
    out = {
        'n_classes': len(dfp),
        'n_candidates': len(candidates),
        'candidates': candidates.set_index('class')[['precision','recall','f1','support','tp','fp','fn']].to_dict(orient='index')
    }
    (out_dir / 'symptoms_error_cases.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    return out


def main():
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / 'experiments' / 'model_errors'
    out_dir.mkdir(parents=True, exist_ok=True)

    sd198_perf = repo / 'experiments' / 'sd198' / 'per_class_metrics_sd198.csv'
    sd198_out = None
    if sd198_perf.exists():
        sd198_out = analyze_sd198(sd198_perf, out_dir)
        print('sd198 кандидатов:', len(sd198_out['candidates']))
    else:
        print('нет файла метрик sd198:', sd198_perf)

    symptoms_preds = repo / 'experiments' / 'symptoms' / 'symptoms_predictions.csv'
    sym_out = None
    if symptoms_preds.exists():
        sym_out = analyze_symptoms(symptoms_preds, out_dir)
        print('симптомы кандидатов:', len(sym_out['candidates']))
    else:
        print('нет предсказаний симптомов:', symptoms_preds)

    summary = {'sd198': sd198_out, 'symptoms': sym_out}
    (out_dir / 'error_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('анализ записан в', out_dir)


if __name__ == '__main__':
    main()
