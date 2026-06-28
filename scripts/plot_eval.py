from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


HAM10000_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_report(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_confusion_matrix(cm, class_names, out_path: Path, metrics: dict | None = None):
    cm = np.array(cm)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if metrics is not None:
        lines = []
        if 'accuracy' in metrics:
            lines.append(f"Accuracy: {metrics['accuracy']:.3f}")
        if 'top3_accuracy' in metrics and metrics['top3_accuracy'] is not None:
            lines.append(f"Top-3 acc: {metrics['top3_accuracy']:.3f}")
        cr = metrics.get('classification_report')
        if cr and isinstance(cr, dict):
            macro = cr.get('macro avg', cr.get('macro_avg'))
            if macro and 'f1-score' in macro:
                lines.append(f"Macro-F1: {float(macro['f1-score']):.3f}")
        if 'roc_auc_macro' in metrics and metrics['roc_auc_macro'] is not None:
            lines.append(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.3f}")

        if lines:
            txt = '\n'.join(lines)
            fig = plt.gcf()
            fig.text(0.90, 0.5, txt, transform=fig.transFigure,
                     fontsize=10, va='center', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 0.88, 1))
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(report, class_names, out_path: Path, metrics: dict | None = None):
    m = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    labels = []
    for i, name in enumerate(class_names):
        key = str(i)
        if key in report:
            entry = report[key]
            m['precision'].append(entry.get('precision', 0.0))
            m['recall'].append(entry.get('recall', 0.0))
            m['f1-score'].append(entry.get('f1-score', 0.0))
            m['support'].append(int(entry.get('support', 0)))
            labels.append(name)
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, m['precision'], width=width, label='Precision')
    plt.bar(x, m['recall'], width=width, label='Recall')
    plt.bar(x + width, m['f1-score'], width=width, label='F1')
    plt.xticks(x, labels, rotation=45)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Per-class metrics')

    try:
        macro_f1 = None
        if isinstance(metrics, dict):
            cr = metrics.get('classification_report')
            if cr:
                macro = cr.get('macro avg', cr.get('macro_avg'))
                if macro and 'f1-score' in macro:
                    macro_f1 = float(macro['f1-score'])

        if macro_f1 is None:
            macro_f1 = report.get('macro avg', {}).get('f1-score')

        texts = []
        if macro_f1 is not None:
            texts.append(f"Macro-F1: {float(macro_f1):.3f}")
        if isinstance(metrics, dict) and metrics.get('top3_accuracy') is not None:
            texts.append(f"Top-3 acc: {metrics['top3_accuracy']:.3f}")

        if texts:
            fig = plt.gcf()
            fig.text(0.90, 0.98, '\n'.join(texts), transform=fig.transFigure,
                     fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.tight_layout(rect=(0, 0, 0.88, 1))
    except Exception:
        pass
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--class-names', default=None, help='Comma-separated class names in order')
    args = parser.parse_args()

    report_path = Path(args.input)
    if not report_path.exists():
        raise SystemExit(f'Report not found: {report_path}')

    data = load_report(report_path)

    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(',')]
    else:
        if 'classification_report' in data and len(data['classification_report']) >= len(HAM10000_CLASS_NAMES):
            class_names = HAM10000_CLASS_NAMES
        else:
            cr = data.get('classification_report', {})
            class_names = [k for k in cr.keys() if k.isdigit()]

    outdir = Path(args.outdir) if args.outdir else report_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    cm = data.get('confusion_matrix')
    if cm is not None:
        cm_out = outdir / 'confusion_matrix.png'
        plot_confusion_matrix(cm, class_names, cm_out, metrics=data)
        print('сохранено:', cm_out)
    else:
        print('confusion_matrix не найдена в отчёте')

    cr = data.get('classification_report')
    if cr is not None:
        metrics_out = outdir / 'per_class_metrics.png'
        plot_per_class_metrics(cr, class_names, metrics_out, metrics=data)
        print('сохранено:', metrics_out)
    else:
        print('classification_report не найден в отчёте')


if __name__ == '__main__':
    main()
