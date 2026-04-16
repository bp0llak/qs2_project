"""
Generates synthetic time-series signals for QFT vs. FFT classification

Each signal is a sampled sinusoid of one of two frequencies.

Signal length is kept at N=8 samples so the QFT circuit size stays small and simulation is fast
"""

import numpy as np
from typing import Tuple


#Constants
N_SAMPLES: int = 8
SAMPLE_RATE: float = 1.0
CLASS_FREQUENCIES = {
    0: 1,
    1: 3,
}





def _make_signal(freq: float, n_samples: int = N_SAMPLES, noise_std: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """
    Return a single sinusoidal signal of length 'N_SAMPLES'

    Parameters:
    freq        : number of full cycles in the window
    n_samples   : number of time-domain samples
    noise_std   : standard deviation of additive Gaussian noise (0 = noiseless)
    rng         : numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 1, n_samples, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)

    if noise_std > 0.0:
        signal += rng.normal(0, noise_std, size=n_samples)
    
    return signal.astype(np.float64)


def generate_dataset(n_per_class: int = 50, noise_std: float = 0.0, n_samples: int = N_SAMPLES, seed: int = 67) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced two-class time-series dataset

    Parameters:
    n_per_class : number of examples per class
    noise_std   : std of additive Gaussian noise
    n_samples   : length of each signal
    seed        : random seed

    Returns:
    x   : ndarray of shape (2*n_per_class, n_samples), raw time-domain signals
    y   : ndarray of shape (2*n_per_class), integer class labels
    """
    assert (n_samples & (n_samples - 1)) == 0   # ensures power of 2

    rng = np.random.default_rng(seed)
    signals, labels = [], []

    for label, freq in CLASS_FREQUENCIES.items():
        for _ in range(n_per_class):
            signals.append(_make_signal(freq, n_samples, noise_std, rng))
            labels.append(label)
    
    x = np.array(signals)
    y = np.array(labels, dtype=int)

    idx = rng.permutation(len(y))
    return x[idx], y[idx]

def noise_sweep_datasets(noise_levels: list, n_per_class: int = 50, n_samples: int = N_SAMPLES, seed: int = 67) -> dict:
    """
    Return a dict mapping noise_std -> (x, y) for a sweep of noise levels
    """
    return {std: generate_dataset(n_per_class, std, n_samples, seed) for std in noise_levels}



# Quick visual check

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y = generate_dataset(n_per_class=3, noise_std=0.1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

    for cls, ax in enumerate(axes):
        samples = X[y == cls]
        t = np.linspace(0, 1, N_SAMPLES, endpoint=False)
        for sig in samples:
            ax.plot(t, sig, alpha=0.7)
        ax.set_title(f"Class {cls}  (freq={CLASS_FREQUENCIES[cls]} Hz)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig("../results/sample_signals.png", dpi=150)
    plt.show()
    print("Dataset generated — shape:", X.shape, "Labels:", np.unique(y))