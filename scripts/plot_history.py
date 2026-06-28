from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    history_path = Path("experiments/history.csv")
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(history_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_f1"], marker="o", label="Train Macro-F1")
    plt.plot(df["epoch"], df["val_f1"], marker="o", label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training and Validation Macro-F1")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "f1_curve.png", bbox_inches="tight")
    plt.close()

    print("сохранено:", output_dir / "loss_curve.png")
    print("сохранено:", output_dir / "f1_curve.png")


if __name__ == "__main__":
    main()