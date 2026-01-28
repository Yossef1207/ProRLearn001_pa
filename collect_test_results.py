#!/usr/bin/env python3
"""Sammelt Testergebnisse aus ProRLearn-Logs und schreibt eine CSV-Zusammenfassung.

Erwartet Log-Dateien wie primevul_dataset.log im angegebenen Log-Verzeichnis.
Es wird der letzte "test"-Block im Log ausgewertet.
"""
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

METRIC_KEYS = ["acc", "recall", "precision", "f1"]
INDEX_LABELS = [
    "True Positive indices (CSV index column):",
    "False Positive indices (CSV index column):",
    "True Negative indices (CSV index column):",
    "False Negative indices (CSV index column):",
]


def parse_filename(path: Path) -> Dict[str, str]:
    """Leite Datensatznamen und Train-Variante aus dem Dateinamen ab.
    
    Erwartet Format: primevul_dataset.log, primevul_with_codellama.log, etc.
    """
    name = path.name
    basename = name[:-4] if name.lower().endswith(".log") else name

    # Falls die Logs mit "test_" beginnen (z.B. test_primevul_dataset.log),
    # diesen Präfix für die weitere Auswertung entfernen
    if basename.startswith("test_"):
        basename = basename[len("test_"):]
    
    # Trenne Dataset und Variante
    # z.B. "primevul_dataset" -> dataset="primevul", train_variant="dataset"
    #      "primevul_with_codellama" -> dataset="primevul", train_variant="with_codellama"
    parts = basename.split("_", 1)
    dataset = parts[0]  # "primevul" oder "reposvul"
    
    if len(parts) > 1:
        rest = parts[1]  # z.B. "dataset", "with_codellama"
        # Wenn "with_" in der Variante ist, entferne es für die Spalte
        if rest.startswith("with_"):
            train_variant = rest[5:]  # "codellama", "gpt-4o", etc.
        else:
            train_variant = rest  # "dataset"
    else:
        train_variant = "unknown"
    
    return {"log_file": name, "dataset": dataset, "train_variant": train_variant}


def find_last_test_block(lines: List[str]) -> Optional[str]:
    """Suche den letzten Test-Block im Log und gib ihn als String zurück."""
    marker = "------------test------------"
    for idx in range(len(lines) - 1, -1, -1):
        if marker in lines[idx]:
            # Nimm den Block ab Marker bis zum nächsten leeren Abschnitt oder Datei-Ende
            return "".join(lines[idx:])
    return None


def parse_confusion(block: str) -> Dict[str, str]:
    """Extrahiere optionale Konfusionsmatrix-Zeilen."""
    # Zeilen sehen typischerweise so aus:
    # [[18602    48]
    #  [  345    17]]
    pairs = re.findall(r"\[\s*\[?\s*([0-9]+)\s+([0-9]+)\s*\]?", block)
    result = {
        "confusion_row0": "",
        "confusion_row1": "",
        "tn": "",
        "fp": "",
        "fn": "",
        "tp": "",
    }
    if len(pairs) >= 1:
        tn_val, fp_val = pairs[0]
        result["confusion_row0"] = f"[{tn_val} {fp_val}]"
        result["tn"] = tn_val
        result["fp"] = fp_val
    if len(pairs) >= 2:
        fn_val, tp_val = pairs[1]
        result["confusion_row1"] = f"[{fn_val} {tp_val}]"
        result["fn"] = fn_val
        result["tp"] = tp_val
    return result


def parse_log_file(path: Path) -> Optional[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    if not lines:
        return None

    block = find_last_test_block(lines)
    # Fallback: Falls kein spezieller Test-Marker gefunden wird,
    # nutze den gesamten Dateiinhalt als Suchraum.
    if not block:
        block = "".join(lines)

    result: Dict[str, str] = {}

    # Metrics - suche nach der letzten acc: Zeile oder "result acc:"
    # Bevorzugt "result acc:" falls vorhanden (finale Ergebnisse)
    m = re.search(
        r"result acc:\s*([0-9.]+)\s+recall:\s*([0-9.]+)\s+precision:\s*([0-9.]+)\s+f1:\s*([0-9.]+)",
        block,
    )
    if not m:
        # Fallback: nimm die erste acc: Zeile nach dem test-Marker
        m = re.search(
            r"acc:\s*([0-9.]+)\s+recall:\s*([0-9.]+)\s+precision:\s*([0-9.]+)\s+f1:\s*([0-9.]+)",
            block,
        )
    
    if m:
        # Umbennung zu LineVul-Format
        result["test_accuracy"] = m.group(1)
        result["test_recall"] = m.group(2)
        result["test_precision"] = m.group(3)
        result["test_f1"] = m.group(4)
    else:
        return None

    # Confusion matrix rows (optional)
    result.update(parse_confusion(block))

    # TP/FP/TN/FN indices (optional)
    label_map = {
        "True Positive indices (CSV index column):": "true_positive_indices",
        "False Positive indices (CSV index column):": "false_positive_indices",
        "True Negative indices (CSV index column):": "true_negative_indices",
        "False Negative indices (CSV index column):": "false_negative_indices",
    }
    for label, key in label_map.items():
        m = re.search(rf"{re.escape(label)}\s*(\[[^\]]*\])", block)
        result[key] = m.group(1) if m else "[]"

    return result


def collect_results(log_dir: Path) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    for path in sorted(log_dir.glob("*.log")):
        parsed = parse_log_file(path)
        if not parsed:
            continue
        meta = parse_filename(path)
        row: Dict[str, str] = {}
        row.update(meta)
        row["test_accuracy"] = parsed.get("test_accuracy", "")
        row["test_f1"] = parsed.get("test_f1", "")
        row["test_precision"] = parsed.get("test_precision", "")
        row["test_recall"] = parsed.get("test_recall", "")
        row["test_threshold"] = ""  # Optional: kann aus Log extrahiert werden
        row["true_positive_indices"] = parsed.get("true_positive_indices", "[]")
        row["false_positive_indices"] = parsed.get("false_positive_indices", "[]")
        row["true_negative_indices"] = parsed.get("true_negative_indices", "[]")
        row["false_negative_indices"] = parsed.get("false_negative_indices", "[]")
        results[path.name] = row
    return results


def write_csv(results: Dict[str, Dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "log_file",
        "dataset",
        "train_variant",
        "test_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_threshold",
        "true_positive_indices",
        "false_positive_indices",
        "true_negative_indices",
        "false_negative_indices",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in sorted(results.items()):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sammelt die letzten Test-Ergebnisse aus ProRLearn-Logs und schreibt eine CSV."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="./logs",
        help="Verzeichnis mit *.log Dateien (Standard: ./logs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test_summary_results.csv",
        help="Ausgabedatei (CSV). Relativ zum Log-Verzeichnis, falls nicht absolut.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Log-Verzeichnis existiert nicht: {log_dir}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = log_dir / output_path

    results = collect_results(log_dir)
    if not results:
        raise SystemExit(f"Keine auswertbaren Logs in {log_dir} gefunden.")

    write_csv(results, output_path)
    print(f"Geschriebene Zusammenfassung: {output_path}")


if __name__ == "__main__":
    main()
