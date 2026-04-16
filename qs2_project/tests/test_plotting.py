"""
Unit tests for visualization and plotting

Run from the project root:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import matplotlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from qs2_project import viz

def test_plot_sample_signals(tmp_path):
    X = np.random.rand(10, 50)
    y = np.array([0]*5 + [1]*5)
    class_freqs = {0: 5, 1: 10}

    viz.RESULTS_DIR = tmp_path  # redirect output

    viz.plot_sample_signals(X, y, class_freqs, show=False)

    assert (tmp_path / "sample_signals.png").exists()


def test_plot_fft_spectrum(tmp_path):
    X = np.random.rand(10, 64)
    y = np.array([0]*5 + [1]*5)
    class_freqs = {0: 5, 1: 10}

    viz.RESULTS_DIR = tmp_path

    viz.plot_fft_spectrum(X, y, class_freqs, show=False)

    assert (tmp_path / "fft_spectra.png").exists()


def test_plot_qft_probabilities(tmp_path):
    prob_vectors = np.random.rand(10, 8)
    prob_vectors /= prob_vectors.sum(axis=1, keepdims=True)
    y = np.array([0]*5 + [1]*5)

    viz.RESULTS_DIR = tmp_path

    viz.plot_qft_probabilities(prob_vectors, y, show=False)

    assert (tmp_path / "qft_probabilities.png").exists()


def test_plot_accuracy_bar(tmp_path):
    results = [
        {"pipeline": "fft", "accuracy": 0.8, "precision": 0.75},
        {"pipeline": "quantum", "accuracy": 0.7, "precision": 0.65},
    ]

    viz.RESULTS_DIR = tmp_path

    viz.plot_accuracy_bar(results, show=False)

    assert (tmp_path / "pipeline_comparison.png").exists()


def test_plot_noise_sweep(tmp_path):
    noise_levels = [0.1, 0.2]
    results_by_noise = {
        0.1: [{"pipeline": "fft", "accuracy": 0.8}],
        0.2: [{"pipeline": "fft", "accuracy": 0.7}],
    }

    viz.RESULTS_DIR = tmp_path

    viz.plot_noise_sweep(noise_levels, results_by_noise, show=False)

    assert (tmp_path / "noise_sweep.png").exists()


def test_plot_transform_comparison(tmp_path):
    signal = np.random.rand(8)
    probs = np.random.rand(8)
    probs /= probs.sum()

    viz.RESULTS_DIR = tmp_path

    viz.plot_transform_comparison(signal, probs, show=False)

    assert (tmp_path / "transform_comparison.png").exists()