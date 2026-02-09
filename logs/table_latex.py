import csv
import ast
from pathlib import Path

# Pfad zur CSV-Datei anpassen
csv_path = Path("test_summary_results.csv")

rows = []
with csv_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # String wie "[1, 2, 3]" -> echte Liste
        indices_str = row["true_positive_indices"].strip()
        if indices_str:
            try:
                indices_list = ast.literal_eval(indices_str)
                num_tp = len(indices_list)
            except (SyntaxError, ValueError):
                num_tp = 0
        else:
            num_tp = 0

        # Für LaTeX Unterstriche escapen
        def esc(s: str) -> str:
            return s.replace("_", r"\_")

        rows.append({
            "log_file": esc(row["log_file"]),
            "dataset": esc(row["dataset"]),
            "train_variant": esc(row["train_variant"]),
            "num_tp": num_tp,
        })

# LaTeX-Tabelle ausgeben
print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\begin{tabular}{l l r}")
print(r"\hline")
print(r"dataset & train\_variant & \#TP \\")
print(r"\hline")

for r in rows:
    print(f"{r['dataset']} & {r['train_variant']} & {r['num_tp']} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Anzahl wahr-positiver Indizes pro Testfall}")
print(r"\label{tab:true_positive_counts}")
print(r"\end{table}")