import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Allow large fields
csv.field_size_limit(10**9)


def load_index_to_cwe(test_csv: Path) -> Dict[int, List[str]]:
    """Build mapping index -> list of CWE IDs from a *test.csv file.

    Expects columns 'index' and 'cwe_id'. 'cwe_id' may contain JSON-like
    lists (e.g. ["CWE-787"]) or a single string.
    """
    mapping: Dict[int, List[str]] = {}

    with test_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_raw = (row.get("index") or "").strip()
            if not idx_raw:
                continue
            try:
                idx = int(idx_raw)
            except ValueError:
                continue

            cwe_raw = (row.get("cwe_id") or "").strip()
            if not cwe_raw:
                cwes: List[str] = []
            else:
                # cwe_id can be like ["CWE-787"] or a plain string
                if cwe_raw.startswith("[") and cwe_raw.endswith("]"):
                    try:
                        parsed = ast.literal_eval(cwe_raw)
                        if isinstance(parsed, (list, tuple)):
                            cwes = [str(x) for x in parsed]
                        else:
                            cwes = [str(parsed)]
                    except Exception:
                        cwes = [cwe_raw]
                else:
                    cwes = [cwe_raw]

            mapping[idx] = cwes

    return mapping


def indices_to_cwes(index_list_str: str, idx2cwe: Dict[int, List[str]]) -> List[str]:
    """Convert a stringified index list into a flat list of CWE IDs."""
    index_list_str = (index_list_str or "").strip()
    if not index_list_str:
        return []

    try:
        indices = ast.literal_eval(index_list_str)
    except Exception:
        return []

    if not isinstance(indices, (list, tuple)):
        return []

    result: List[str] = []
    for idx in indices:
        try:
            i = int(idx)
        except Exception:
            continue
        result.extend(idx2cwe.get(i, []))
    return result


def convert_summary_tp_to_cwe(
    summary_csv_path: str,
    primevul_test_csv: str,
    reposvul_test_csv: str,
    output_csv_path: str,
) -> None:
    """Create a table with dataset, train_variant, true_positive_cwes.

    - summary_csv_path: best_testing_logs/test_summary_results.csv
    - primevul_test_csv: primevul_dataset/test.csv
    - reposvul_test_csv: reposvul_dataset/test.csv
    """
    summary_csv = Path(summary_csv_path)
    primevul_csv = Path(primevul_test_csv)
    reposvul_csv = Path(reposvul_test_csv)
    output_csv = Path(output_csv_path)

    if not summary_csv.is_file():
        raise FileNotFoundError(f"summary CSV not found: {summary_csv}")
    if not primevul_csv.is_file():
        raise FileNotFoundError(f"primevul test CSV not found: {primevul_csv}")
    if not reposvul_csv.is_file():
        raise FileNotFoundError(f"reposvul test CSV not found: {reposvul_csv}")

    idx2cwe_primevul = load_index_to_cwe(primevul_csv)
    idx2cwe_reposvul = load_index_to_cwe(reposvul_csv)

    with summary_csv.open(newline="", encoding="utf-8") as fin, \
         output_csv.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = ["dataset", "train_variant", "true_positive_cwes"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            dataset = row.get("dataset")
            train_variant = row.get("train_variant")
            indices_str = row.get("true_positive_indices", "")

            if dataset == "primevul":
                mapping = idx2cwe_primevul
            elif dataset == "reposvul":
                mapping = idx2cwe_reposvul
            else:
                # Unknown dataset: skip row
                continue

            cwes = indices_to_cwes(indices_str, mapping)
            writer.writerow(
                {
                    "dataset": dataset,
                    "train_variant": train_variant,
                    "true_positive_cwes": repr(cwes),
                }
            )


if __name__ == "__main__":
    # Example for your paths:
    #   python summary_tp_indices_to_cwe.py \
    #     best_testing_logs/test_summary_results.csv \
    #     /work/cps/czt0517/LineVul_pa/data/primevul_dataset/test.csv \
    #     /work/cps/czt0517/LineVul_pa/data/reposvul_dataset/test.csv \
    #     best_testing_logs/test_summary_results_cwe.csv

    import sys

    if len(sys.argv) != 5:
        print("Benutzung:")
        print(
            "  python summary_tp_indices_to_cwe.py "
            "<summary_csv> <primevul_test_csv> <reposvul_test_csv> <output_csv>"
        )
        raise SystemExit(1)

    summary_csv_path = sys.argv[1]
    primevul_test_csv = sys.argv[2]
    reposvul_test_csv = sys.argv[3]
    output_csv_path = sys.argv[4]

    convert_summary_tp_to_cwe(
        summary_csv_path,
        primevul_test_csv,
        reposvul_test_csv,
        output_csv_path,
    )



"""
python summary_tp_indices_to_cwe.py test_summary_results.csv /work/cps/czt0517/LineVul_pa/data/primevul_dataset/test.csv /work/cps/czt0517/LineVul_pa/data/reposvul_dataset/test.csv test_summary_results_cwe.csv

"""