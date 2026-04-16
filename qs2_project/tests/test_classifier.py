import numpy as np
import json
import csv
from pathlib import Path

from qs2_project.classifier import evaluate, save_results_csv, save_results_json


# Testing evaluate()

def test_evaluate_basic_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = evaluate(y_true, y_pred, pipeline_name="test")

    assert result["pipeline"] == "test"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0


def test_evaluate_perfect_classifier():
    y = np.array([0, 1, 0, 1])
    result = evaluate(y, y, pipeline_name="perfect")

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0

    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_evaluate_rounding_behavior():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1])

    result = evaluate(y_true, y_pred)

    assert isinstance(result["accuracy"], float)
    assert len(str(result["accuracy"]).split(".")[-1]) <= 4


# Testing saves

def test_save_results_csv(tmp_path):
    results = [
        {
            "pipeline": "fft",
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
        }
    ]

    from qs2_project import classifier
    classifier.RESULTS_DIR = tmp_path

    path = save_results_csv(results, filename="test.csv")

    assert path.exists()

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert rows[0]["pipeline"] == "fft"
    assert float(rows[0]["accuracy"]) == 0.9


def test_save_results_json(tmp_path):
    results = [
        {
            "pipeline": "qft",
            "accuracy": 0.95,
            "confusion_matrix": [[5, 0], [1, 4]],
        }
    ]

    from qs2_project import classifier
    classifier.RESULTS_DIR = tmp_path

    path = save_results_json(results, filename="test.json")

    assert path.exists()

    with open(path) as f:
        loaded = json.load(f)

    assert loaded[0]["pipeline"] == "qft"
    assert loaded[0]["confusion_matrix"] == [[5, 0], [1, 4]]