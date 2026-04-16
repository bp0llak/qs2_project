"""
Quantum pipeline:
  1. Amplitude-encode each time-domain signal into an n-qubit state |ψ⟩
  2. Apply the Quantum Fourier Transform circuit
  3. Simulate and measure → probability distribution over 2^n basis states
  4. Use the probability vector as a feature vector for Logistic Regression

"""

import numpy as np
from typing import Dict

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report


# --------------------------------------------------------------------------- #
# Amplitude encoding                                                           #
# --------------------------------------------------------------------------- #

def _normalize(signal: np.ndarray) -> np.ndarray:
    """
    L2-normalise a signal so it can serve as a valid quantum state vector.
    """
    norm = np.linalg.norm(signal)
    if norm < 1e-12:
        raise ValueError("Signal has near-zero norm; cannot amplitude-encode.")
    return signal / norm


def amplitude_encode(signal: np.ndarray) -> QuantumCircuit:
    """
    Build a Qiskit circuit that amplitude-encodes *signal* into |ψ⟩.

    The signal length must be 2^n.  Uses Qiskit's initialize instruction
    which compiles to the required state-preparation gates automatically.

    Parameters:
    
    signal : 1-D real array of length 2^n

    Returns:
    
    QuantumCircuit of n qubits with the state prepared (no measurement)
    """
    n_qubits = int(np.log2(len(signal)))
    assert 2 ** n_qubits == len(signal), "Signal length must be a power of 2."

    state_vector = _normalize(signal.astype(complex))

    qc = QuantumCircuit(n_qubits)
    qc.initialize(state_vector, qubits=list(range(n_qubits)))
    return qc


# QFT circuit
def build_qft_circuit(signal: np.ndarray, inverse: bool = False) -> QuantumCircuit:
    """
    Return a full circuit: amplitude encoding → QFT → measurement.

    Parameters:
    
    signal  : time-domain signal (length 2^n)
    inverse : use IQFT instead of QFT (experimental)

    Returns:
    
    QuantumCircuit ready for simulation
    """
    n_qubits = int(np.log2(len(signal)))

    # State preparation
    encode_qc = amplitude_encode(signal)

    # QFT from Qiskit's circuit library
    qft_gate = QFT(num_qubits=n_qubits, inverse=inverse, do_swaps=True,
                   name="IQFT" if inverse else "QFT")

    # Combine and add measurements
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.compose(encode_qc, inplace=True)
    qc.append(qft_gate, qargs=list(range(n_qubits)))
    qc.measure(range(n_qubits), range(n_qubits))

    return qc



# Simulation & feature extraction
def simulate_circuit(
    qc: QuantumCircuit,
    shots: int = 4096,
    noise_model=None,
) -> np.ndarray:
    """
    Run the circuit on AerSimulator and return a probability vector.

    Parameters:

    qc          : QuantumCircuit with measurements
    shots       : number of measurement shots
    noise_model : optional Qiskit noise model for realistic simulation

    Returns:
    
    prob_vector : 1-D array of length 2^n_qubits (normalised counts)
    """
    backend = AerSimulator(noise_model=noise_model)
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=shots)
    counts = job.result().get_counts()

    n_qubits = qc.num_qubits
    dim = 2 ** n_qubits
    prob_vector = np.zeros(dim)

    for bitstring, count in counts.items():
        # Qiskit returns big-endian bitstrings; convert to index
        idx = int(bitstring, 2)
        prob_vector[idx] = count / shots

    return prob_vector


def extract_qft_features(
    x: np.ndarray,
    shots: int = 4096,
    noise_model=None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply amplitude encoding + QFT to every signal and collect features.

    Parameters:
    
    x           : (n_signals, signal_len) time-domain signals
    shots       : measurement shots per signal
    noise_model : optional Qiskit noise model
    verbose     : print progress

    Returns:
    
    features : (n_signals, signal_len) probability distributions
    """
    features = []

    for i, signal in enumerate(x):
        if verbose and i % 10 == 0:
            print(f"  Encoding signal {i+1}/{len(x)} …")

        qc = build_qft_circuit(signal)
        prob_vec = simulate_circuit(qc, shots=shots, noise_model=noise_model)
        features.append(prob_vec)

    return np.array(features)


# Pipeline

def run_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    shots: int = 4096,
    test_size: float = 0.25,
    seed: int = 42,
    noise_model=None,
    verbose: bool = True,
) -> Dict:
    """
    End-to-end quantum pipeline: QFT encoding → scale → classify.

    Parameters:
    
    x           : raw time-domain signals  (n_samples, signal_len)
    y           : integer class labels
    shots       : measurement shots per signal circuit
    test_size   : fraction held out for evaluation
    seed        : random seed
    noise_model : optional Qiskit noise model
    verbose     : print progress and report

    Returns:
    
    Dictionary with keys: accuracy, precision, features_train, features_test,
                          y_train, y_test, y_pred, model, scaler
    """
    if verbose:
        print("\n  Extracting QFT features …")

    # 1. QFT feature extraction
    features = extract_qft_features(x, shots=shots, noise_model=noise_model,
                                    verbose=verbose)

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
        print(f"\n{'='*50}")
        print(f"  Quantum Pipeline  [QFT | shots={shots}]")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Class 0", "Class 1"]))
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")

    return {
        "pipeline": "quantum_qft",
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


#Smoke-test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset import generate_dataset

    x, y = generate_dataset(n_per_class=10, noise_std=0.0)
    run_pipeline(x, y, shots=2048, verbose=True)
