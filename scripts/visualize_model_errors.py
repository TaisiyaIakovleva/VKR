from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sd198(perf_csv: Path, out_dir: Path, min_support: int = 5, top_n: int = 30):
    df = pd.read_csv(perf_csv, index_col=0)
    df['support'] = df['support'].astype(int)
    df = df[df['support'] >= min_support]
    df = df.sort_values('recall')

    out_dir.mkdir(parents=True, exist_ok=True)

    worst = df.head(top_n)
    plt.figure(figsize=(10, max(4, 0.25 * len(worst))))
    sns.barplot(x='recall', y=worst.index, data=worst.reset_index(), palette='magma')
    plt.xlabel('Recall')
    plt.title('SD-198: Lowest recall classes (support >= {})'.format(min_support))
    plt.xlim(0, 1)
    plt.tight_layout()
    bar_path = out_dir / 'sd198_low_recall_bar.png'
    plt.savefig(bar_path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='support', y='recall', data=df.reset_index())
    for i, row in df.reset_index().iterrows():
        if row['recall'] < 0.6:
            plt.text(row['support'], row['recall'], row['index'], fontsize=8)
    plt.xlabel('Support (num samples)')
    plt.ylabel('Recall')
    plt.title('SD-198: Recall vs Support')
    plt.ylim(0, 1)
    plt.tight_layout()
    scatter_path = out_dir / 'sd198_recall_vs_support.png'
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    return bar_path, scatter_path


def main():
    repo = Path(__file__).resolve().parents[1]
    perf_csv = repo / 'experiments' / 'sd198' / 'per_class_metrics_sd198.csv'
    out_dir = repo / 'experiments' / 'model_errors'
    if not perf_csv.exists():
        print('нет файла метрик:', perf_csv)
        return
    bar, scatter = plot_sd198(perf_csv, out_dir)
    print('сохранено:', bar, scatter)


if __name__ == '__main__':
    main()
