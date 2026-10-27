from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantum_simulator import H, QuantumState


bell = QuantumState.zero(2).apply(H, 0).apply_cnot(0, 1)
print("Bell amplitudes:", bell.amplitudes)
print("Bell probabilities:", bell.probabilities)
print("One measurement:", bell.measure(rng=7))
print("H-Z-H on |0>:", QuantumState.zero().apply(H, 0).apply(
    np.array([[1, 0], [0, -1]], dtype=complex), 0
).apply(H, 0).amplitudes)
