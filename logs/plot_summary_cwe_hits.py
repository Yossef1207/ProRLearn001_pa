import argparse
import ast
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt


def parse_cwe_list(field: str) -> list[str]:
    """Parse a stringified Python list of CWE IDs into a list of strings."""
    if not field:
        return []
    field = field.strip()
    if not field:
        return []
    try:
        value = ast.literal_eval(field)
    except Exception:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(x) for x in value]
    return []


def aggregate_summary_cwes(
    csv_path: Path,
    dataset: Optional[str] = None,
    train_variant: Optional[str] = None,
) -> Counter:
    """Aggregate CWE counts from test_summary_results_cwe.csv.

    Each row contributes its true_positive_cwes list.
    Optional filters: dataset (primevul/reposvul) and train_variant.
    """
    counter: Counter = Counter()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if dataset is not None and row.get("dataset") != dataset:
                continue
            if train_variant is not None and row.get("train_variant") != train_variant:
                continue
            cwes = parse_cwe_list(row.get("true_positive_cwes", ""))
            counter.update(cwes)

    return counter


def plot_cwe_hits(
    counter: Counter,
    title: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot a bar chart of CWE hit counts."""
    if not counter:
        print("Keine CWEs zum Plotten gefunden.")
        return

    most_common = counter.most_common(top_n)
    labels = [cwe for cwe, _ in most_common]
    values = [count for _, count in most_common]

    x = range(len(labels))

    plt.figure(figsize=(max(8, 0.4 * len(labels)), 5))
    plt.bar(x, values, color="#377eb8")
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Anzahl Treffer (True Positives)")
    plt.title(title)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot gespeichert unter: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CWE-Treffer aus test_summary_results_cwe.csv",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Pfad zu test_summary_results_cwe.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optionales Dataset-Filter, z.B. primevul oder reposvul",
    )
    parser.add_argument(
        "--train-variant",
        type=str,
        default=None,
        help="Optionales train_variant-Filter, z.B. only, codellama, gpt-4o",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Anzahl CWEs mit höchsten Trefferzahlen (Default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("summary_cwe_hits.png"),
        help="Pfad für die Ausgabegrafik (PNG)",
    )

    args = parser.parse_args()

    counter = aggregate_summary_cwes(
        args.csv_path,
        dataset=args.dataset,
        train_variant=args.train_variant,
    )

    title_parts = ["CWE True-Positive Hits"]
    if args.dataset:
        title_parts.append(f"dataset={args.dataset}")
    if args.train_variant:
        title_parts.append(f"train_variant={args.train_variant}")
    title = ", ".join(title_parts)

    plot_cwe_hits(counter, title, args.output, top_n=args.top_n)


if __name__ == "__main__":
    main()



"""
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset 
primevul --train-variant codellama --top-n 20 --output primevul_codellama_summary_cwe_hits.png
Plot gespeichert unter: primevul_codellama_summary_cwe_hits.png



python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset primevul --train-variant codellama --top-n 20 --output cwe/primevul_codellama_summary_cwe_hits.png
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset primevul --train-variant vul_codellama --top-n 20 --output cwe/primevul_vul_codellama_summary_cwe_hits.png
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset primevul --train-variant gpt-4o --top-n 20 --output cwe/primevul_gpt-4o_summary_cwe_hits.png
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset primevul --train-variant vul_gpt-4o --top-n 20 --output cwe/primevul_vul_gpt-4o_summary_cwe_hits.png
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset primevul --train-variant only --top-n 20 --output cwe/primevul_only_summary_cwe_hits.png

python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset reposvul --train-variant codellama --top-n 20 --output cwe/reposvul_codellama_summary_cwe_hits.png
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset reposvul --train-variant vul_codellama --top-n 20 --output cwe/reposvul_vul_codellama_summary_cwe_hits.png
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset reposvul --train-variant gpt-4o --top-n 20 --output cwe/reposvul_gpt-4o_summary_cwe_hits.png
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset reposvul --train-variant vul_gpt-4o --top-n 20 --output cwe/reposvul_vul_gpt-4o_summary_cwe_hits.png
python plot_summary_cwe_hits.py test_summary_results_cwe.csv --dataset reposvul --train-variant only --top-n 20 --output cwe/reposvul_only_summary_cwe_hits.png




Beispiele:

Primevul, alle train_variants zusammen:

cd /work/cps/czt0517/LineVul_pa/linevul
python plot_summary_cwe_hits.py best_testing_logs/test_summary_results_cwe.csv --dataset primevul --top-n 20 --output best_testing_logs/primevul_summary_cwe_hits.png
Reposvul, alle train_variants zusammen:

python plot_summary_cwe_hits.py best_testing_logs/test_summary_results_cwe.csv --dataset reposvul --top-n 20 --output best_testing_logs/reposvul_summary_cwe_hits.png
Optional pro train_variant (z.B. nur primevul + codellama):

python plot_summary_cwe_hits.py best_testing_logs/test_summary_results_cwe.csv --dataset primevul --train-variant codellama --top-n 20 --output best_testing_logs/primevul_codellama_summary_cwe_hits.png
"""