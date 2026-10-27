import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantum_simulator import CNOT, H, X, Z, QuantumState, Ry


class QuantumSimulatorTests(unittest.TestCase):
    def test_state_sizes_and_single_qubit_gates_on_every_position(self):
        for num_qubits in range(1, 5):
            zero = QuantumState.zero(num_qubits)
            self.assertEqual(zero.num_qubits, num_qubits)
            for qubit in range(num_qubits):
                mask = 1 << (num_qubits - 1 - qubit)
                expected_basis = np.zeros(1 << num_qubits, dtype=complex)
                expected_basis[mask] = 1
                flipped = zero.apply(X, qubit)
                np.testing.assert_allclose(flipped.amplitudes, expected_basis)
                self.assertEqual(flipped.measure(rng=0), format(mask, f"0{num_qubits}b"))

                phase_flipped = flipped.apply(Z, qubit)
                np.testing.assert_allclose(phase_flipped.amplitudes, -expected_basis)
                rotated = zero.apply(Ry(0), qubit)
                np.testing.assert_allclose(rotated.amplitudes, zero.amplitudes)

                plus = zero.apply(H, qubit)
                expected_probabilities = np.zeros(1 << num_qubits)
                expected_probabilities[[0, mask]] = 0.5
                np.testing.assert_allclose(plus.probabilities, expected_probabilities)

    def test_x_and_z_gates(self):
        zero = QuantumState.zero()
        one = np.array([0, 1], dtype=complex)
        np.testing.assert_allclose(zero.apply(X, 0).amplitudes, one)
        np.testing.assert_allclose(zero.apply(Z, 0).amplitudes, [1, 0])
        np.testing.assert_allclose(QuantumState(one).apply(Z, 0).amplitudes, [0, -1])

    def test_h_probabilities_and_h_squared(self):
        plus = QuantumState.zero().apply(H, 0)
        np.testing.assert_allclose(plus.probabilities, [0.5, 0.5])
        np.testing.assert_allclose(plus.apply(H, 0).amplitudes, [1, 0])

    def test_ry_zero_is_identity(self):
        state = QuantumState(np.array([1 / np.sqrt(2), 1j / np.sqrt(2)]))
        np.testing.assert_allclose(state.apply(Ry(0), 0).amplitudes, state.amplitudes)

    def test_cnot_basis_mapping_and_bit_order(self):
        np.testing.assert_array_equal(
            CNOT,
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=complex,
            ),
        )
        expected = ["00", "01", "11", "10"]
        for index, output in enumerate(expected):
            amplitudes = np.zeros(4, dtype=complex)
            amplitudes[index] = 1
            result = QuantumState(amplitudes).apply_cnot()
            self.assertEqual(result.measure(rng=0), output)
        three_qubits = QuantumState.zero(3).apply(X, 1)
        self.assertEqual(three_qubits.measure(rng=0), "010")
        reversed_positions = QuantumState.zero(3).apply(X, 2).apply_cnot(2, 0)
        self.assertEqual(reversed_positions.measure(rng=0), "101")

    def test_cnot_supports_every_distinct_control_and_target_position(self):
        for num_qubits in range(2, 5):
            for control in range(num_qubits):
                for target in range(num_qubits):
                    if control == target:
                        continue
                    control_mask = 1 << (num_qubits - 1 - control)
                    target_mask = 1 << (num_qubits - 1 - target)
                    for basis_index in range(1 << num_qubits):
                        amplitudes = np.zeros(1 << num_qubits, dtype=complex)
                        amplitudes[basis_index] = 1
                        expected_index = basis_index
                        if basis_index & control_mask:
                            expected_index ^= target_mask
                        expected = np.zeros_like(amplitudes)
                        expected[expected_index] = 1
                        result = QuantumState(amplitudes).apply_cnot(control, target)
                        np.testing.assert_allclose(result.amplitudes, expected)

    def test_full_measurement_returns_q0_first_n_bit_string(self):
        for num_qubits in range(1, 5):
            for basis_index in range(1 << num_qubits):
                amplitudes = np.zeros(1 << num_qubits, dtype=complex)
                amplitudes[basis_index] = 1
                state = QuantumState(amplitudes)
                self.assertEqual(
                    state.measure(rng=0), format(basis_index, f"0{num_qubits}b")
                )

    def test_bell_state_amplitudes_probabilities_and_measurements(self):
        bell = QuantumState.zero(2).apply(H, 0).apply_cnot(0, 1)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(bell.amplitudes, expected)
        np.testing.assert_allclose(bell.probabilities, [0.5, 0, 0, 0.5])
        outcomes = {bell.measure(rng=1234 + seed) for seed in range(200)}
        self.assertEqual(outcomes, {"00", "11"})

    def test_three_qubit_ghz_state(self):
        ghz = QuantumState.zero(3).apply(H, 0).apply_cnot(0, 1).apply_cnot(0, 2)
        expected = np.zeros(8, dtype=complex)
        expected[[0, 7]] = 1 / np.sqrt(2)
        np.testing.assert_allclose(ghz.amplitudes, expected)
        np.testing.assert_allclose(ghz.probabilities, [0.5, 0, 0, 0, 0, 0, 0, 0.5])
        outcomes = {ghz.measure(rng=700 + seed) for seed in range(100)}
        self.assertEqual(outcomes, {"000", "111"})

    def test_measurement_is_seeded_and_single_qubit_measurement_collapses(self):
        plus = QuantumState.zero().apply(H, 0)
        self.assertEqual(plus.measure(rng=42), plus.measure(rng=42))
        state = QuantumState.zero(2).apply(H, 0).apply_cnot()
        measured = state.measure_qubit(0, rng=3)
        expected_bitstring = str(measured) * 2
        self.assertEqual(state.measure(rng=1), expected_bitstring)

    def test_measurement_handles_rounding_within_accepted_norm_tolerance(self):
        almost_one = QuantumState(np.array([0, np.sqrt(1 + 5e-11)], dtype=complex))
        np.testing.assert_allclose(almost_one.probabilities, [0, 1])
        self.assertEqual(almost_one.measure_qubit(0, rng=0), 1)

    def test_invalid_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "unit norm"):
            QuantumState(np.array([1, 1], dtype=complex))
        with self.assertRaisesRegex(ValueError, "power of two"):
            QuantumState(np.array([1, 0, 0], dtype=complex))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            QuantumState(np.array([[1, 0], [0, 0]], dtype=complex))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            QuantumState(np.array([], dtype=complex))
        with self.assertRaisesRegex(ValueError, "finite"):
            QuantumState(np.array([np.nan, 0], dtype=complex))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            QuantumState.zero(0)
        with self.assertRaisesRegex(ValueError, "qubit index"):
            QuantumState.zero().apply(X, 1)
        with self.assertRaisesRegex(ValueError, "qubit index"):
            QuantumState.zero().apply(X, -1)
        with self.assertRaisesRegex(ValueError, "different"):
            QuantumState.zero(2).apply_cnot(0, 0)
        with self.assertRaisesRegex(ValueError, "qubit index"):
            QuantumState.zero(2).apply_cnot(2, 0)
        with self.assertRaisesRegex(ValueError, "qubit index"):
            QuantumState.zero(2).measure_qubit(2)
        with self.assertRaisesRegex(ValueError, "shape"):
            QuantumState.zero().apply(np.eye(3), 0)
        with self.assertRaisesRegex(ValueError, "theta"):
            Ry(float("nan"))
        with self.assertRaisesRegex(ValueError, "rng"):
            QuantumState.zero().measure(rng="seed")


if __name__ == "__main__":
    unittest.main()
