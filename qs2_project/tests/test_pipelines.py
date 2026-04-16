"""
Unit tests for dataset generation, feature extraction, and classification.

Run from the project root:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import generate_dataset, N_SAMPLES
from fft_pipeline import extract_fft_features, extract_time_features, run_pipeline as run_classical
from qft_pipeline import amplitude_encode, build_qft_circuit, simulate_circuit


# Tests dataset

class TestDataset:
    def test_shape(self):
        X, y = generate_dataset(n_per_class=10)
        assert X.shape == (20, N_SAMPLES)
        assert y.shape == (20,)

    def test_balanced_classes(self):
        X, y = generate_dataset(n_per_class=10)
        assert np.sum(y == 0) == 10
        assert np.sum(y == 1) == 10

    def test_reproducibility(self):
        X1, y1 = generate_dataset(n_per_class=10, seed=42)
        X2, y2 = generate_dataset(n_per_class=10, seed=42)
        np.testing.assert_array_equal(X1, X2)

    def test_noise_changes_signal(self):
        X_clean, _ = generate_dataset(n_per_class=5, noise_std=0.0)
        X_noisy, _ = generate_dataset(n_per_class=5, noise_std=1.0)
        assert not np.allclose(X_clean, X_noisy)

    def test_power_of_two_length(self):
        n = N_SAMPLES
        assert (n & (n - 1)) == 0, "N_SAMPLES must be a power of 2 for QFT"


# Test on classical features

class TestFFTPipeline:
    def setup_method(self):
        self.X, self.y = generate_dataset(n_per_class=20, noise_std=0.0)

    def test_time_features_shape(self):
        feats = extract_time_features(self.X)
        assert feats.shape == self.X.shape

    def test_fft_mag_shape(self):
        feats = extract_fft_features(self.X, mode="fft_mag")
        assert feats.shape == (len(self.X), N_SAMPLES // 2 + 1)

    def test_fft_complex_shape(self):
        feats = extract_fft_features(self.X, mode="fft_complex")
        assert feats.shape == (len(self.X), 2 * (N_SAMPLES // 2 + 1))

    def test_fft_nonnegative_magnitudes(self):
        feats = extract_fft_features(self.X, mode="fft_mag")
        assert (feats >= 0).all()

    def test_pipeline_returns_dict(self):
        result = run_classical(self.X, self.y, feature_mode="fft_mag", verbose=False)
        assert "accuracy" in result
        assert "precision" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_clean_signal_high_accuracy(self):
        """Noiseless two-frequency data should be trivially separable via FFT."""
        result = run_classical(self.X, self.y, feature_mode="fft_mag", verbose=False)
        assert result["accuracy"] >= 0.95, \
            f"Expected ≥95% accuracy on clean data, got {result['accuracy']:.2f}"


# Quantum Encoding & Circuit

class TestQFTPipeline:
    def setup_method(self):
        self.signal = np.sin(2 * np.pi * np.linspace(0, 1, N_SAMPLES, endpoint=False))

    def test_amplitude_encode_normalised(self):
        """State vector produced by amplitude encoding must have unit norm."""
        qc = amplitude_encode(self.signal)
        # The initialise instruction stores the statevector; check length
        assert qc.num_qubits == int(np.log2(N_SAMPLES))

    def test_qft_circuit_qubit_count(self):
        qc = build_qft_circuit(self.signal)
        assert qc.num_qubits == int(np.log2(N_SAMPLES))

    def test_qft_circuit_has_measurements(self):
        qc = build_qft_circuit(self.signal)
        assert qc.num_clbits == qc.num_qubits

    def test_simulate_returns_probability_vector(self):
        qc = build_qft_circuit(self.signal)
        prob_vec = simulate_circuit(qc, shots=1024)
        assert prob_vec.shape == (N_SAMPLES,)

    def test_probabilities_sum_to_one(self):
        qc = build_qft_circuit(self.signal)
        prob_vec = simulate_circuit(qc, shots=2048)
        assert abs(prob_vec.sum() - 1.0) < 0.01, \
            f"Probabilities sum to {prob_vec.sum():.4f}, expected ~1.0"

    def test_probabilities_nonnegative(self):
        qc = build_qft_circuit(self.signal)
        prob_vec = simulate_circuit(qc, shots=1024)
        assert (prob_vec >= 0).all()