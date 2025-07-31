"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using the statevector simulator of qiskit.

Date created: 2025-04-23
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS, load
from qiskit.quantum_info import Statevector


class SVSolver:
    """
    Compute the amplitude of a bitstring for a quantum circuit using the statevector simulator of qiskit (https://github.com/Qiskit/qiskit).
    """

    def __init__(self):
        pass

    def compute_amplitude(
        self, circuit_file_path: str, final_state: np.ndarray
    ) -> tuple[complex, float]:
        """
        Compute the amplitude of a bitstring for a quantum circuit using the statevector simulator of qiskit.

        Parameters
        ----------
        circuit_file_path: str
            The path to the QASM file containing the quantum circuit.
        final_state: np.ndarray
            The final state of the circuit, represented as a bitstring.

        Returns
        -------
        amplitude: complex
            The amplitude of the final state.
        """

        # Load the circuit from the QASM file
        circuit = load(
            circuit_file_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS
        )

        start_time = time.time()
        psi = Statevector.from_instruction(circuit)
        end_time = time.time()

        # Convert z_array to bitstring, then reverse for little-endian
        z_str = "".join(str(b) for b in final_state[::-1])
        z_index = int(z_str, 2)

        elapsed_time = end_time - start_time

        return psi[z_index], elapsed_time

    def plot_circuit(self, circuit_file_path: str) -> None:
        """
        Plot the quantum circuit.

        Parameters
        ----------
        circuit_file_path: str
            The path to the QASM file containing the quantum circuit.
        """

        # Load the circuit from the QASM file
        circuit = load(
            circuit_file_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS
        )

        # Plot the circuit
        circuit.draw(output="mpl")
        plt.show()
