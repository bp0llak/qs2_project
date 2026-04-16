"""
Shared evaluation helpers used by both the classical and quantum pipelines.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Evaluation
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, pipeline_name: str = "") -> Dict:
    """Return a dict of classification metrics.

    Parameters:

    y_true        : ground-truth labels
    y_pred        : predicted labels
    pipeline_name : label for reporting

    Returns:
    
    metrics : dict with accuracy, precision, recall, f1, confusion_matrix
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()

    return {
        "pipeline":         pipeline_name,
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm,
    }


# Reporting

def compare_results(results: List[Dict]) -> None:
    """Print a formatted comparison table for a list of pipeline result dicts."""
    col_w = 22
    metrics = ["accuracy", "precision", "recall", "f1"]

    header = f"{'Pipeline':<{col_w}}" + "".join(f"{m:>12}" for m in metrics)
    sep    = "-" * len(header)

    print(f"\n{sep}")
    print("  PIPELINE COMPARISON")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        row = f"{r.get('pipeline','?'):<{col_w}}"
        for m in metrics:
            val = r.get(m, float("nan"))
            row += f"{val:>12.4f}"
        print(row)

    print(sep)

    # Highlight winner on accuracy
    best = max(results, key=lambda r: r.get("accuracy", 0))
    print(f"\n  Best accuracy → {best['pipeline']}  ({best['accuracy']:.4f})\n")


# Saving helpers

def save_results_csv(results: List[Dict], filename: str = "results_summary.csv") -> Path:
    """Write a results list to a CSV file in the results directory."""
    path = RESULTS_DIR / filename
    metrics = ["pipeline", "accuracy", "precision", "recall", "f1"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in metrics})

    print(f"  Results saved → {path}")
    return path


def save_results_json(results: List[Dict], filename: str = "results_summary.json") -> Path:
    """Write full results (including confusion matrices) to JSON."""
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {path}")
    return path
