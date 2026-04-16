"""
Classical pipeline: extract FFT magnitude spectrum features from raw
time-domain signals, then train a Logistic Regression classifier.

Three feature modes are supported so you can compare them side-by-side:
  - 'time'      : raw time-domain samples (control / baseline)
  - 'fft_mag'   : FFT magnitude spectrum (standard ML practice)
  - 'fft_complex': real + imaginary parts concatenated
"""

import numpy as np
from numpy.fft import fft
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report
from typing import Dict


# --------------------------------------------------------------------------- #
# Feature extraction                                                           #
# --------------------------------------------------------------------------- #

def extract_time_features(x: np.ndarray) -> np.ndarray:
    """Return raw time-domain samples as features (baseline control)."""
    return x.copy()


def extract_fft_features(x: np.ndarray, mode: str = "fft_mag") -> np.ndarray:
    """Compute FFT-based features for every signal in X.

    Parameters:

    X    : (n_samples, signal_len) time-domain signals
    mode : 'fft_mag'     — magnitude of the one-sided spectrum
           'fft_complex' — real and imaginary parts concatenated

    Returns:

    Features array of shape (n_samples, n_features)
    """
    spectra = fft(x, axis=1)
    n = x.shape[1]
    half = n // 2 + 1           # one-sided spectrum length

    if mode == "fft_mag":
        # Magnitude of the one-sided spectrum
        mag = np.abs(spectra[:, :half])
        return mag

    elif mode == "fft_complex":
        # Concatenate real and imaginary parts
        real_part = np.real(spectra[:, :half])
        imag_part = np.imag(spectra[:, :half])
        return np.hstack([real_part, imag_part])

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'fft_mag' or 'fft_complex'.")


# Pipeline
def run_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    feature_mode: str = "fft_mag",
    test_size: float = 0.25,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    End-to-end classical pipeline: feature extraction → scale → classify.

    Parameters:

    x            : raw time-domain signals  (n_samples, signal_len)
    y            : integer class labels
    feature_mode : 'time', 'fft_mag', or 'fft_complex'
    test_size    : fraction of data held out for evaluation
    seed         : random seed
    verbose      : print a classification report if True

    Returns:

    Dictionary with keys: accuracy, precision, features_train, features_test,
                          y_train, y_test, y_pred, model, scaler
    """
    # 1. Extract features
    if feature_mode == "time":
        features = extract_time_features(x)
    else:
        features = extract_fft_features(x, mode=feature_mode)

    # 2. Train / test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=seed, stratify=y
    )

    # 3. Standardise
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    # 4. Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x_train_sc, y_train)

    # 5. Evaluate
    y_pred = model.predict(x_test_sc)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)

    if verbose:
        tag = feature_mode.upper()
        print(f"\n{'='*50}")
        print(f"  Classical Pipeline  [{tag}]")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")

    return {
        "pipeline": f"classical_{feature_mode}",
        "accuracy": acc,
        "precision": prec,
        "features_train": x_train_sc,
        "features_test": x_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "model": model,
        "scaler": scaler,
    }


# Smoke-test
if __name__ == "__main__":
    from dataset import generate_dataset

    x, y = generate_dataset(n_per_class=50, noise_std=0.2)

    for mode in ("time", "fft_mag"):
        run_pipeline(x, y, feature_mode=mode)
