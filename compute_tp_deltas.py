#!/usr/bin/env python3
import argparse
import ast
import csv
from pathlib import Path
from typing import Dict, List, Set


def parse_indices(indices_str: str) -> Set[int]:
    """Parse a string like "[35, 76, 77]" or "35,76,77" into a set of ints."""
    indices_str = indices_str.strip()
    if not indices_str:
        return set()

    # Try Python literal first (e.g. "[35, 76, 77]")
    try:
        value = ast.literal_eval(indices_str)
        if isinstance(value, (list, tuple, set)):
            return {int(x) for x in value}
    except (SyntaxError, ValueError):
        pass

    # Fallback: comma-separated list
    parts = [p.strip() for p in indices_str.split(",") if p.strip()]
    return {int(p) for p in parts}


def load_summary(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_by_dataset(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        dataset = row.get("dataset", "").strip()
        if not dataset:
            continue
        grouped.setdefault(dataset, []).append(row)
    return grouped


def compute_deltas_for_dataset(dataset: str, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Compute TP set deltas per dataset, using the 'only' variant as baseline."""
    # Find baseline: train_variant == 'only'
    baseline_rows = [r for r in rows if r.get("train_variant", "").strip() == "only"]
    if not baseline_rows:
        return []
    baseline = baseline_rows[0]
    baseline_variant = baseline.get("train_variant", "only")
    baseline_indices = parse_indices(baseline.get("true_positive_indices", "[]"))

    deltas: List[Dict[str, str]] = []

    for row in rows:
        compare_variant = row.get("train_variant", "").strip()
        if compare_variant == "only":
            continue

        compare_indices = parse_indices(row.get("true_positive_indices", "[]"))

        inter = baseline_indices & compare_indices
        new = compare_indices - baseline_indices
        lost = baseline_indices - compare_indices

        delta_row: Dict[str, str] = {
            "dataset": dataset,
            "baseline_variant": baseline_variant,
            "compare_variant": compare_variant,
            "baseline_tp_count": str(len(baseline_indices)),
            "compare_tp_count": str(len(compare_indices)),
            "intersection_tp_count": str(len(inter)),
            "new_tp_count": str(len(new)),
            "lost_tp_count": str(len(lost)),
            "new_tp_indices": str(sorted(new)),
            "lost_tp_indices": str(sorted(lost)),
            "intersection_tp_indices": str(sorted(inter)),
        }
        deltas.append(delta_row)

    return deltas


def write_deltas_csv(deltas: List[Dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "baseline_variant",
        "compare_variant",
        "baseline_tp_count",
        "compare_tp_count",
        "intersection_tp_count",
        "new_tp_count",
        "lost_tp_count",
        "new_tp_indices",
        "lost_tp_indices",
        "intersection_tp_indices",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in deltas:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TP set deltas relative to the 'only' baseline per dataset "
            "(reposvul/primevul) from a summary CSV produced by collect_test_results.py."
        )
    )
    parser.add_argument(
        "summary_csv",
        help=(
            "Pfad zur Zusammenfassungs-CSV (z.B. my_results.csv oder "
            "test_summary_results.csv), die u.a. die Spalten 'dataset', "
            "'train_variant' und 'true_positive_indices' enthaelt."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test_tp_deltas.csv",
        help=(
            "Pfad zur Ausgabedatei (CSV) fuer die Deltas. Standard: test_tp_deltas.csv "
            "im gleichen Verzeichnis wie die Eingabedatei."
        ),
    )

    args = parser.parse_args()
    summary_path = Path(args.summary_csv).expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Summary-CSV existiert nicht: {summary_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = summary_path.parent / output_path

    rows = load_summary(summary_path)
    grouped = group_by_dataset(rows)

    all_deltas: List[Dict[str, str]] = []
    for dataset, ds_rows in grouped.items():
        ds_deltas = compute_deltas_for_dataset(dataset, ds_rows)
        all_deltas.extend(ds_deltas)

    if not all_deltas:
        raise SystemExit("Keine Deltas berechnet (evtl. keine 'only'-Baseline gefunden).")

    write_deltas_csv(all_deltas, output_path)
    print(f"Geschriebene TP-Delta-Tabelle: {output_path}")


if __name__ == "__main__":
    main()


"""
 python compute_tp_deltas.py best_testing_logs/summerized_results.csv -o test_tp_deltas.csv
 # Standard: liest my_results.csv im aktuellen Verzeichnis und schreibt test_tp_deltas.csv dort hinein
"""
