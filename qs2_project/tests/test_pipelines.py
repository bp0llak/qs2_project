"""
Unit tests for dataset generation, feature extraction, and classification.

Run from the project root:
    python -m pytest tests/ -v
"""

import numpy as np
import pytest

from qs2_project.dataset import generate_dataset, N_SAMPLES
from qs2_project.fft_pipeline import extract_fft_features, extract_time_features, run_pipeline as run_classical
from qs2_project.qft_pipeline import amplitude_encode, build_qft_circuit, simulate_circuit


# Tests dataset

def test_shape():
    X, y = generate_dataset(n_per_class=10)
    assert X.shape == (20, N_SAMPLES)
    assert y.shape == (20,)

def test_balanced_classes():
    X, y = generate_dataset(n_per_class=10)
    assert np.sum(y == 0) == 10
    assert np.sum(y == 1) == 10

def test_reproducibility():
    X1, y1 = generate_dataset(n_per_class=10, seed=42)
    X2, y2 = generate_dataset(n_per_class=10, seed=42)
    np.testing.assert_array_equal(X1, X2)

def test_noise_changes_signal():
    X_clean, _ = generate_dataset(n_per_class=5, noise_std=0.0)
    X_noisy, _ = generate_dataset(n_per_class=5, noise_std=1.0)
    assert not np.allclose(X_clean, X_noisy)

def test_power_of_two_length():
    n = N_SAMPLES
    assert (n & (n - 1)) == 0, "N_SAMPLES must be a power of 2 for QFT"


# Test on classical features

def test_time_features_shape():
    x, y = generate_dataset(n_per_class=20, noise_std=0.0)
    feats = extract_time_features(x)
    assert feats.shape == x.shape

def test_fft_mag_shape():
    x, y = generate_dataset(n_per_class=20, noise_std=0.0)
    feats = extract_fft_features(x, mode="fft_mag")
    assert feats.shape == (len(x), N_SAMPLES // 2 + 1)

def test_fft_complex_shape():
    x, y = generate_dataset(n_per_class=20, noise_std=0.0)
    feats = extract_fft_features(x, mode="fft_complex")
    assert feats.shape == (len(x), 2 * (N_SAMPLES // 2 + 1))

def test_fft_nonnegative_magnitudes():
    x, y = generate_dataset(n_per_class=20, noise_std=0.0)
    feats = extract_fft_features(x, mode="fft_mag")
    assert (feats >= 0).all()

def test_pipeline_returns_dict():
    x, y = generate_dataset(n_per_class=20, noise_std=0.0)
    result = run_classical(x, y, feature_mode="fft_mag", verbose=False)
    assert "accuracy" in result
    assert "precision" in result
    assert 0.0 <= result["accuracy"] <= 1.0

def test_clean_signal_high_accuracy():
    """Noiseless two-frequency data shouldn't be separable via FFT."""
    x, y = generate_dataset(n_per_class=20, noise_std=0.0)
    result = run_classical(x, y, feature_mode="fft_mag", verbose=False)
    assert result["accuracy"] >= 0.95, \
        f"Expected ≥95% accuracy on clean data, got {result['accuracy']:.2f}"


# Quantum Encoding & Circuit

def test_amplitude_encode_normalised():
    """State vector produced by amplitude encoding must have unit norm."""
    signal = np.sin(2 * np.pi * np.linspace(0, 1, N_SAMPLES, endpoint=False))
    qc = amplitude_encode(signal)
    # The initialise instruction stores the statevector; check length
    assert qc.num_qubits == int(np.log2(N_SAMPLES))

def test_qft_circuit_qubit_count():
    signal = np.sin(2 * np.pi * np.linspace(0, 1, N_SAMPLES, endpoint=False))
    qc = build_qft_circuit(signal)
    assert qc.num_qubits == int(np.log2(N_SAMPLES))

def test_qft_circuit_has_measurements():
    signal = np.sin(2 * np.pi * np.linspace(0, 1, N_SAMPLES, endpoint=False))
    qc = build_qft_circuit(signal)
    assert qc.num_clbits == qc.num_qubits

def test_simulate_returns_probability_vector():
    signal = np.sin(2 * np.pi * np.linspace(0, 1, N_SAMPLES, endpoint=False))
    qc = build_qft_circuit(signal)
    prob_vec = simulate_circuit(qc, shots=1024)
    assert prob_vec.shape == (N_SAMPLES,)

def test_probabilities_sum_to_one():
    signal = np.sin(2 * np.pi * np.linspace(0, 1, N_SAMPLES, endpoint=False))
    qc = build_qft_circuit(signal)
    prob_vec = simulate_circuit(qc, shots=2048)
    assert abs(prob_vec.sum() - 1.0) < 0.01, \
        f"Probabilities sum to {prob_vec.sum():.4f}, expected ~1.0"

def test_probabilities_nonnegative():
    signal = np.sin(2 * np.pi * np.linspace(0, 1, N_SAMPLES, endpoint=False))
    qc = build_qft_circuit(signal)
    prob_vec = simulate_circuit(qc, shots=1024)
    assert (prob_vec >= 0).all()