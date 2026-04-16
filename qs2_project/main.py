"""
main.py
-------
Orchestrates the full QFT vs FFT experiment from the project proposal:

  Pipeline A — Raw time-domain features (control baseline)
  Pipeline B — Classical FFT magnitude features
  Pipeline C — Quantum QFT probability distribution features

Runs each pipeline on the same dataset, compares results, then sweeps
over noise levels to evaluate robustness.

Usage:
    python main.py                  # full run (all pipelines + noise sweep)
    python main.py --no-quantum     # skip quantum pipeline (much faster)
    python main.py --shots 1024     # reduce shot count for speed
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from dataset import generate_dataset, noise_sweep_datasets, CLASS_FREQUENCIES
from fft_pipeline import run_pipeline as run_classical
from qft_pipeline import run_pipeline as run_quantum, extract_qft_features
from classifier import evaluate, compare_results, save_results_csv, save_results_json
from qs2_project.viz import (
    plot_sample_signals,
    plot_fft_spectrum,
    plot_qft_probabilities,
    plot_accuracy_bar,
    plot_noise_sweep,
)


# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="QFT vs FFT Classification Experiment")
    p.add_argument("--no-quantum",  action="store_true",
                   help="Skip the quantum pipeline (useful for quick testing)")
    p.add_argument("--shots",       type=int, default=4096,
                   help="Measurement shots per QFT circuit (default: 4096)")
    p.add_argument("--n-per-class", type=int, default=50,
                   help="Training examples per class (default: 50)")
    p.add_argument("--noise",       type=float, default=0.1,
                   help="Signal noise std for main experiment (default: 0.1)")
    p.add_argument("--no-sweep",    action="store_true",
                   help="Skip the noise sweep")
    p.add_argument("--no-show",     action="store_true",
                   help="Save plots but do not display them interactively")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main experiment                                                              #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    show_plots = not args.no_show

    print("\n" + "="*60)
    print("  QFT vs FFT — Time-Series Classification Experiment")
    print("="*60)
    print(f"  n_per_class : {args.n_per_class}")
    print(f"  noise_std   : {args.noise}")
    print(f"  shots       : {args.shots}")
    print(f"  quantum     : {not args.no_quantum}")
    print("="*60)

    # ------------------------------------------------------------------ #
    # 1. Generate dataset                                                  #
    # ------------------------------------------------------------------ #
    print("\n[1/5] Generating dataset …")
    X, y = generate_dataset(n_per_class=args.n_per_class, noise_std=args.noise)
    print(f"      X: {X.shape}   y: {y.shape}   classes: {np.unique(y)}")

    plot_sample_signals(X, y, CLASS_FREQUENCIES, save=True, show=show_plots)
    plot_fft_spectrum(X, y, CLASS_FREQUENCIES, save=True, show=show_plots)

    # ------------------------------------------------------------------ #
    # 2. Classical pipelines                                               #
    # ------------------------------------------------------------------ #
    print("\n[2/5] Running classical pipelines …")
    res_time = run_classical(X, y, feature_mode="time",    verbose=True)
    res_fft  = run_classical(X, y, feature_mode="fft_mag", verbose=True)

    # Attach full metrics
    res_time.update(evaluate(res_time["y_test"], res_time["y_pred"], "classical_time"))
    res_fft.update(evaluate(res_fft["y_test"],  res_fft["y_pred"],  "classical_fft_mag"))

    all_results = [res_time, res_fft]

    # ------------------------------------------------------------------ #
    # 3. Quantum pipeline                                                  #
    # ------------------------------------------------------------------ #
    if not args.no_quantum:
        print("\n[3/5] Running quantum pipeline (this may take a few minutes) …")
        res_qft = run_quantum(X, y, shots=args.shots, verbose=True)
        res_qft.update(evaluate(res_qft["y_test"], res_qft["y_pred"], "quantum_qft"))
        all_results.append(res_qft)

        # Visualise QFT probability distributions
        print("      Plotting QFT probability distributions …")
        qft_feats = extract_qft_features(X, shots=args.shots, verbose=False)
        plot_qft_probabilities(qft_feats, y, save=True, show=show_plots)
    else:
        print("\n[3/5] Quantum pipeline skipped (--no-quantum).")

    # ------------------------------------------------------------------ #
    # 4. Compare & save results                                            #
    # ------------------------------------------------------------------ #
    print("\n[4/5] Comparing results …")
    compare_results(all_results)
    save_results_csv(all_results)
    save_results_json(all_results)

    plot_accuracy_bar(all_results, save=True, show=show_plots)

    # ------------------------------------------------------------------ #
    # 5. Noise sweep                                                       #
    # ------------------------------------------------------------------ #
    if not args.no_sweep:
        print("\n[5/5] Running noise sweep …")
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
        results_by_noise = {}

        for std in noise_levels:
            print(f"  noise_std = {std:.2f}")
            Xn, yn = generate_dataset(n_per_class=args.n_per_class, noise_std=std)

            sweep_res = []
            rt = run_classical(Xn, yn, feature_mode="time",    verbose=False)
            rt.update(evaluate(rt["y_test"], rt["y_pred"], "classical_time"))
            sweep_res.append(rt)

            rf = run_classical(Xn, yn, feature_mode="fft_mag", verbose=False)
            rf.update(evaluate(rf["y_test"], rf["y_pred"], "classical_fft_mag"))
            sweep_res.append(rf)

            if not args.no_quantum:
                rq = run_quantum(Xn, yn, shots=args.shots, verbose=False)
                rq.update(evaluate(rq["y_test"], rq["y_pred"], "quantum_qft"))
                sweep_res.append(rq)

            results_by_noise[std] = sweep_res

        plot_noise_sweep(noise_levels, results_by_noise, save=True, show=show_plots)
        save_results_json(
            [{"noise_std": k, "results": v} for k, v in results_by_noise.items()],
            filename="noise_sweep_results.json",
        )
    else:
        print("\n[5/5] Noise sweep skipped (--no-sweep).")

    print("\n  All done!  Results and plots saved to ./results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
