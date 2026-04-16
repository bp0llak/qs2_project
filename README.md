QS2_Project
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/bp0llak/qs2_project/workflows/CI/badge.svg)](https://github.com/bp0llak/qs2_project/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/bp0llak/QS2_Project/branch/main/graph/badge.svg)](https://codecov.io/gh/bp0llak/QS2_Project/branch/main)


A Python experiment comparing the Quantum Fourier Transform (QFT) and the classical
Fast Fourier Transform (FFT) as feature extraction methods in a hybrid
quantum-classical machine learning pipeline for time-series classification.

### Copyright

Copyright (c) 2026, Bryce Pollak


## Project Structure

QS2_Project/
├── qs2_project/
│   ├── dataset.py        # Synthetic signal generator (two-frequency classes)
│   ├── fft_pipeline.py   # Classical FFT feature extraction + Logistic Regression
│   ├── qft_pipeline.py   # Amplitude encoding + QFT circuit + Logistic Regression
│   ├── classifier.py     # Shared evaluation helpers (metrics, CSV/JSON export)
│   └── visualize.py      # All Matplotlib plotting functions
├── notebooks/
│   └── exploration.ipynb # Interactive walkthrough of the full experiment
├── results/              # Auto-created; plots and result CSVs saved here
├── main.py               # CLI entry point — runs all pipelines end-to-end


## Pipelines

| Pipeline | Features | Notes |
|---|---|---|
| **Baseline** | Raw time-domain samples | Control — no transform |
| **Classical FFT** | FFT magnitude spectrum |
| **Quantum QFT** | Measurement probability distribution | Amplitude-encoded + QFT circuit |


## Running the Experiment

### Installing dependencies
```bash
pip install -r requirements.txt
```

### Full run (all three pipelines + noise sweep)
```bash
python main.py
```

### Skip the quantum pipeline for a quick test
```bash
python main.py --no-quantum
```

### Tune parameters
```bash
python main.py --shots 2048 --n-per-class 30 --noise 0.2
```

### Interactive notebook
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Key Parameters

| Flag | Default | Description |
|---|---|---|
| `--shots` | 4096 | Measurement shots per QFT circuit |
| `--n-per-class` | 50 | Training samples per class |
| `--noise` | 0.1 | Gaussian noise std added to signals |
| `--no-sweep` | off | Skip the noise robustness sweep |
| `--no-quantum` | off | Skip the quantum pipeline |
| `--no-show` | off | Save plots but don't display them |

## Dataset

Signals are 8-sample (2³) sinusoids of two frequencies:
- Class 0: 1 cycle per window
- Class 1: 3 cycles per window

The 8-sample length maps directly to a 3-qubit QFT circuit.
Gaussian noise at configurable SNR is added to test robustness.

## Dependencies

- `qiskit >= 1.0` + `qiskit-aer` — quantum circuit simulation
- `numpy` — FFT and signal generation
- `scikit-learn` — Logistic Regression, metrics
- `matplotlib` — visualisation
- `jupyter` — interactive notebook
