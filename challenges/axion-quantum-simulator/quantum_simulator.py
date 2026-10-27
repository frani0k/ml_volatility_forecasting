"""State-vector simulator for circuits with any memory-feasible qubit count."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import SupportsFloat

import numpy as np


_SQRT2 = math.sqrt(2.0)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / _SQRT2

# Basis order |00>, |01>, |10>, |11>; q0 is the leftmost (most significant) bit.
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)


def Ry(theta: SupportsFloat) -> np.ndarray:
    """Return the single-qubit Y-axis rotation matrix for a finite angle in radians."""
    try:
        angle = float(theta)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta must be a finite real number") from exc
    if not math.isfinite(angle):
        raise ValueError("theta must be a finite real number")
    cosine = math.cos(angle / 2)
    sine = math.sin(angle / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


@dataclass
class QuantumState:
    """Normalized state vector; basis index bits are ordered q0, q1, ... (MSB first)."""

    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        try:
            vector = np.asarray(self.amplitudes, dtype=np.complex128)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("amplitudes must be convertible to finite complex numbers") from exc
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("amplitudes must be a non-empty one-dimensional vector")
        if vector.size & (vector.size - 1):
            raise ValueError("amplitude count must be a power of two")
        if not np.all(np.isfinite(vector)):
            raise ValueError("amplitudes must be finite")
        norm_squared = float(np.vdot(vector, vector).real)
        if not math.isclose(norm_squared, 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("state amplitudes must have unit norm")
        self.amplitudes = (vector / math.sqrt(norm_squared)).copy()

    @classmethod
    def zero(cls, num_qubits: int = 1) -> "QuantumState":
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer")
        vector = np.zeros(1 << num_qubits, dtype=np.complex128)
        vector[0] = 1.0
        return cls(vector)

    @property
    def num_qubits(self) -> int:
        return self.amplitudes.size.bit_length() - 1

    @property
    def probabilities(self) -> np.ndarray:
        probabilities = np.abs(self.amplitudes) ** 2
        total = float(probabilities.sum())
        if not np.all(np.isfinite(probabilities)) or not math.isclose(
            total, 1.0, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("state probabilities must be finite and sum to one")
        return probabilities

    def apply(self, gate: np.ndarray, qubit: int) -> "QuantumState":
        """Apply a 2x2 gate to one qubit without constructing a full-system matrix."""
        matrix = np.asarray(gate, dtype=np.complex128)
        if matrix.shape != (2, 2):
            raise ValueError("single-qubit gate must have shape (2, 2)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("gate entries must be finite")
        if isinstance(qubit, bool) or not isinstance(qubit, int) or not 0 <= qubit < self.num_qubits:
            raise ValueError("qubit index is out of range")
        tensor = self.amplitudes.reshape((2,) * self.num_qubits)
        moved = np.moveaxis(tensor, qubit, 0)
        transformed = np.tensordot(matrix, moved, axes=(1, 0))
        result = np.moveaxis(transformed, 0, qubit).reshape(-1)
        return QuantumState(result)

    def apply_cnot(self, control: int = 0, target: int = 1) -> "QuantumState":
        """Apply CNOT at any two distinct positions; q0 is the leftmost bit."""
        for index in (control, target):
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.num_qubits:
                raise ValueError("qubit index is out of range")
        if control == target:
            raise ValueError("control and target must be different qubits")
        tensor = self.amplitudes.reshape((2,) * self.num_qubits)
        result = tensor.copy()
        control_target_zero = [slice(None)] * self.num_qubits
        control_target_one = [slice(None)] * self.num_qubits
        control_target_zero[control] = control_target_one[control] = 1
        control_target_zero[target] = 0
        control_target_one[target] = 1
        zero_index = tuple(control_target_zero)
        one_index = tuple(control_target_one)
        result[zero_index] = tensor[one_index]
        result[one_index] = tensor[zero_index]
        return QuantumState(result.reshape(-1))

    def measure(self, rng: np.random.Generator | int | None = None) -> str:
        """Sample all qubits, returning a q0-first bitstring."""
        generator = _as_generator(rng)
        index = int(generator.choice(self.amplitudes.size, p=self.probabilities))
        return format(index, f"0{self.num_qubits}b")

    def measure_qubit(
        self, qubit: int, rng: np.random.Generator | int | None = None
    ) -> int:
        """Measure one qubit and collapse the state in place; return 0 or 1."""
        if isinstance(qubit, bool) or not isinstance(qubit, int) or not 0 <= qubit < self.num_qubits:
            raise ValueError("qubit index is out of range")
        generator = _as_generator(rng)
        axis = self.num_qubits - 1 - qubit
        bits = ((np.arange(self.amplitudes.size) >> axis) & 1).astype(bool)
        probabilities = self.probabilities
        probability_one = float(probabilities[bits].sum())
        outcome = int(generator.choice(2, p=[1.0 - probability_one, probability_one]))
        keep = bits if outcome else ~bits
        self.amplitudes[~keep] = 0
        norm = float(np.linalg.norm(self.amplitudes))
        if norm == 0:
            raise RuntimeError("measurement selected an impossible outcome")
        self.amplitudes /= norm
        return outcome


def _as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None or (isinstance(rng, int) and not isinstance(rng, bool)):
        return np.random.default_rng(rng)
    raise ValueError("rng must be None, an integer seed, or numpy.random.Generator")
