import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Beschriftungen konfigurierbar machen
BAR_LABEL_NEW = "new TPs"        # Legendenname für neue True Positives
BAR_LABEL_LOST = "lost TPs"      # Legendenname für verlorene True Positives

# Optionale Umbenennung der Varianten (x-Achse).
# Key = Wert in der CSV-Spalte "compare_variant", Value = gewünschter Name im Plot
VARIANT_LABELS = {
    "codellama": "CodeLlama-34B",
    "gpt-4o": "GPT-4o",
    "vul_codellama": "Vul-CodeLlama-34B",
    "vul_gpt-4o": "Vul-GPT-4o",
}

def plot_tp_deltas(csv_path: str):
    # CSV einlesen
    df = pd.read_csv(csv_path)

    datasets = df["dataset"].unique()
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset].copy()

        x = np.arange(len(sub))
        width = 0.35

        # Säulen (mit konfigurierbaren Legenden-Namen)
        bars_new = ax.bar(x - width/2, sub["new_tp_count"],  width=width, label=BAR_LABEL_NEW,  color="#4caf50")
        bars_lost = ax.bar(x + width/2, sub["lost_tp_count"], width=width, label=BAR_LABEL_LOST, color="#f44336")

        # Werte auf die Säulen schreiben
        for bars in (bars_new, bars_lost):
            for rect in bars:
                height = rect.get_height()
                ax.annotate(
                    f"{int(height)}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # X-Achsen-Beschriftung ggf. umbenennen
        raw_variants = sub["compare_variant"].tolist()
        xticklabels = [VARIANT_LABELS.get(v, v) for v in raw_variants]
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        ax.set_title(f"Dataset: {dataset}")
        ax.set_ylabel("Number of TPs")
        ax.legend()

    plt.tight_layout()
    # Figure speichern
    plt.savefig("logs/tp_deltas.png", bbox_inches="tight", dpi=200)
    plt.show()

if __name__ == "__main__":
    # Pfad ggf. anpassen
    plot_tp_deltas("logs/test_tp_deltas.csv")