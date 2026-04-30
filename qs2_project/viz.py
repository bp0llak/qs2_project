"""
Plotting utilities for the QFT vs FFT classification experiment.

All functions save figures to ../results/ and optionally display them.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Consistent color palette across all plots
PALETTE = {
    "time":    "#7f8c8d",
    "fft_mag": "#2980b9",
    "fft_complex": "#1da632",
    "quantum": "#8e44ad",
}


# Signal inspection

def plot_sample_signals(
    X: np.ndarray,
    y: np.ndarray,
    class_freqs: dict,
    n_examples: int = 3,
    save: bool = True,
    show: bool = True,
):
    """Plot a few raw time-domain signals per class."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    t = np.linspace(0, 1, X.shape[1], endpoint=False)

    for cls, ax in enumerate(axes):
        samples = X[y == cls][:n_examples]
        for i, sig in enumerate(samples):
            ax.plot(t, sig, alpha=0.8, lw=1.8, label=f"Sample {i+1}")
        ax.set_title(f"Class {cls}  —  freq = {class_freqs[cls]} Hz", fontsize=12)
        ax.set_xlabel("Time (normalised)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sample Time-Domain Signals", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "sample_signals.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_fft_spectrum(
    X: np.ndarray,
    y: np.ndarray,
    class_freqs: dict,
    save: bool = True,
    show: bool = True,
):
    """Plot mean FFT magnitude spectrum for each class."""
    from numpy.fft import fft

    n = X.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / n)   # cycles per window

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)

    for cls, ax in enumerate(axes):
        spectra = np.abs(fft(X[y == cls], axis=1)[:, : n // 2 + 1])
        mean_spec = spectra.mean(axis=0)
        std_spec = spectra.std(axis=0)

        ax.bar(freqs, mean_spec, color=PALETTE["fft_mag"], alpha=0.8, width=0.3,
               label="Mean |FFT|")
        ax.fill_between(freqs, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.3, color=PALETTE["fft_mag"], label="±1 std")
        ax.axvline(class_freqs[cls], color="red", ls="--", lw=1.5,
                   label=f"True freq = {class_freqs[cls]}")
        ax.set_title(f"Class {cls} — FFT spectrum", fontsize=12)
        ax.set_xlabel("Frequency (cycles/window)")
        ax.set_ylabel("|FFT|")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("FFT Magnitude Spectrum per Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "fft_spectra.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_qft_probabilities(
    prob_vectors: np.ndarray,
    y: np.ndarray,
    save: bool = True,
    show: bool = True,
):
    """Plot mean QFT measurement probability distribution per class."""
    n_basis = prob_vectors.shape[1]
    basis_states = np.arange(n_basis)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)

    for cls, ax in enumerate(axes):
        vecs = prob_vectors[y == cls]
        mean_p = vecs.mean(axis=0)
        std_p = vecs.std(axis=0)

        ax.bar(basis_states, mean_p, color=PALETTE["quantum"], alpha=0.8,
               label="Mean prob")
        ax.fill_between(basis_states, mean_p - std_p, mean_p + std_p,
                        alpha=0.3, color=PALETTE["quantum"], label="±1 std")
        ax.set_title(f"Class {cls} — QFT probabilities", fontsize=12)
        ax.set_xlabel("Basis state index")
        ax.set_ylabel("Measurement probability")
        ax.set_xticks(basis_states)
        ax.set_xticklabels([f"|{i:0{int(np.log2(n_basis))}b}⟩" for i in basis_states],
                           fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("QFT Measurement Probability Distributions per Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "qft_probabilities.png", dpi=150,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# Performance comparison

def plot_accuracy_bar(
    results: List[Dict],
    save: bool = True,
    show: bool = True,
):
    """Bar chart comparing accuracy and precision across pipelines."""
    labels = [r["pipeline"].replace("_", "\n") for r in results]
    accuracies = [r["accuracy"] for r in results]
    precisions = [r["precision"] for r in results]

    colors = []
    for r in results:
        if "quantum" in r["pipeline"]:
            colors.append(PALETTE["quantum"])
        elif "fft_complex" in r["pipeline"]:
            colors.append(PALETTE["fft_complex"])
        elif "fft" in r["pipeline"]:
            colors.append(PALETTE["fft_mag"])
        else:
            colors.append(PALETTE["time"])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2), 4.5))
    bars_acc = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
                      color=colors, alpha=0.9, edgecolor="white")
    bars_prec = ax.bar(x + width / 2, precisions, width, label="Precision",
                       color=colors, alpha=0.55, edgecolor="white", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Pipeline Comparison — Accuracy & Precision", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in list(bars_acc) + list(bars_prec):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "pipeline_comparison.png", dpi=150,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_noise_sweep(
    noise_levels: List[float],
    results_by_noise: Dict[float, List[Dict]],
    save: bool = True,
    show: bool = True,
):
    """Line chart of accuracy vs noise level for each pipeline."""
    pipeline_names = [r["pipeline"] for r in list(results_by_noise.values())[0]]

    fig, ax = plt.subplots(figsize=(9, 5))

    for pipeline in pipeline_names:
        accs = [
            next(r["accuracy"] for r in results_by_noise[n] if r["pipeline"] == pipeline)
            for n in noise_levels
        ]
        color = (PALETTE["quantum"] if "quantum" in pipeline
                  else PALETTE["fft_complex"] if "fft_complex" in pipeline
                  else PALETTE["fft_mag"] if "fft" in pipeline
                  else PALETTE["time"])
        ax.plot(noise_levels, accs, marker="o", lw=2, color=color,
                label=pipeline.replace("_", " "))

    ax.set_xlabel("Noise std", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Accuracy vs. Signal Noise Level", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "noise_sweep.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_transform_comparison(
    signal: np.ndarray,
    qft_prob_vector: np.ndarray,
    label: str = "Example signal",
    save: bool = True,
    show: bool = True,
):
    """
    Side-by-side comparison of FFT complex output vs QFT probability distribution
    for a single signal, illustrating the loss of phase information after measurement.

    Parameters:

    signal          : 1-D time-domain signal (length N)
    qft_prob_vector : probability distribution from QFT measurement (length N)
    label           : title label identifying the signal's class
    """
    from numpy.fft import fft

    n = len(signal)
    half = n // 2 + 1
    freqs = np.fft.rfftfreq(n, d=1.0 / n)
    basis_states = np.arange(n)
    n_qubits = int(np.log2(n))

    spectrum = fft(signal)[:half]
    mag  = np.abs(spectrum)
    phase = np.angle(spectrum)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Panel 1: FFT magnitude ---
    axes[0].bar(freqs, mag, width=0.25, color=PALETTE["fft_mag"], alpha=0.9)
    axes[0].set_title("FFT — Magnitude |X[k]|", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Frequency bin")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: FFT phase (present in FFT, absent after QFT measurement) ---
    axes[1].bar(freqs, phase, width=0.25, color="#e67e22", alpha=0.9)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("FFT — Phase ∠X[k]  ✓ accessible", fontsize=11,
                       fontweight="bold", color="#e67e22")
    axes[1].set_xlabel("Frequency bin")
    axes[1].set_ylabel("Phase (radians)")
    axes[1].set_ylim(-np.pi - 0.3, np.pi + 0.3)
    axes[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes[1].set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: QFT measurement probabilities (no phase) ---
    axes[2].bar(basis_states, qft_prob_vector, color=PALETTE["quantum"], alpha=0.9)
    axes[2].set_title("QFT — Measurement probabilities\n✗ phase lost on measurement",
                       fontsize=11, fontweight="bold", color=PALETTE["quantum"])
    axes[2].set_xlabel("Basis state |k⟩")
    axes[2].set_ylabel("Probability")
    axes[2].set_xticks(basis_states)
    axes[2].set_xticklabels(
        [f"|{i:0{n_qubits}b}⟩" for i in basis_states], fontsize=7
    )
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"FFT vs QFT Output Comparison — {label}\n"
        "FFT gives direct access to complex amplitudes (magnitude + phase); "
        "QFT measurement yields only probabilities",
        fontsize=11, style="italic"
    )
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "transform_comparison.png", dpi=150,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)