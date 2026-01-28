#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["test_accuracy", "test_f1", "test_precision", "test_recall"]


def normalize_variant_name(variant: str) -> str:
    """Prettify train_variant names for plotting."""
    mapping = {
        "only": "only",
        "codellama": "CodeLLaMA-34b",
        "gpt-4o": "GPT-4o",
        "vul_codellama": "Vul_CodeLLaMA-34b",
        "vul_gpt-4o": "Vul_GPT-4o",
    }
    return mapping.get(variant, variant)


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # enforce expected columns
    required_cols = {
        "dataset",
        "train_variant",
        "test_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Fehlende Spalten in {csv_path}: {sorted(missing)}")
    return df


def plot_metrics_for_dataset(df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    """Create a 2x2 subplot figure of accuracy, F1, precision, recall for one dataset."""
    df_ds = df[df["dataset"] == dataset].copy()
    if df_ds.empty:
        return

    # Sort variants in a meaningful order
    order: List[str] = ["only", "codellama", "gpt-4o", "vul_codellama", "vul_gpt-4o"]
    df_ds["_order"] = df_ds["train_variant"].apply(lambda v: order.index(v) if v in order else len(order))
    df_ds = df_ds.sort_values("_order")

    x_labels = [normalize_variant_name(v) for v in df_ds["train_variant"].tolist()]
    x = range(len(df_ds))

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        values = df_ds[metric].astype(float).tolist()
        ax.bar(x, values, color="steelblue", edgecolor="black")
        ax.set_title(metric.replace("test_", "").capitalize(), fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for xi, v in zip(x, values):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    for ax in axes[2:]:
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=30, ha="right")

    fig.suptitle(f"Evaluation metrics for {dataset}", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"metrics_{dataset}"
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot evaluation metrics (accuracy, F1, precision, recall) per dataset "
            "from a summary CSV wie summerized_results.csv."
        )
    )
    parser.add_argument(
        "summary_csv",
        help=(
            "Pfad zur CSV (z.B. summerized_results.csv) mit Spalten: "
            "dataset, train_variant, test_accuracy, test_f1, test_precision, test_recall."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="plots",
        help=(
            "Ausgabeverzeichnis fuer die Plots (PNG und PDF). "
            "Standard: 'plots' im gleichen Verzeichnis wie die CSV."
        ),
    )

    args = parser.parse_args()
    csv_path = Path(args.summary_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV-Datei existiert nicht: {csv_path}")

    df = load_results(csv_path)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = csv_path.parent / out_dir

    for dataset in sorted(df["dataset"].unique()):
        plot_metrics_for_dataset(df, dataset, out_dir)

    print(f"Plots gespeichert unter: {out_dir}")


if __name__ == "__main__":
    main()

"""
 python plot_metrics.py best_testing_logs/summerized_results.csv -o plots
"""