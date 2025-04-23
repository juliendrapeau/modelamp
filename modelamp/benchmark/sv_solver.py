"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using the statevector simulator of qiskit.

Date created: 2025-04-23
"""

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class SVSolver:
    """
    Compute the amplitude of a bitstring for a quantum circuit using the statevector simulator of qiski (https://github.com/Qiskit/qiskit).
    """

    def __init__(self):
        pass

    def compute_amplitude(self, circuit: QuantumCircuit, final_state: np.ndarray):
        """
        Compute the amplitude of a bitstring for a quantum circuit using the statevector simulator of qiskit.

        Parameters
        ----------
        circuit: QuantumCircuit
            The quantum circuit to simulate.
        final_state: np.ndarray
            The final state of the circuit, represented as a bitstring.

        Returns
        -------
        amplitude: complex
            The amplitude of the final state.
        """

        start_time = time.time()
        psi = Statevector.from_instruction(circuit)
        end_time = time.time()
        
        # Convert z_array to bitstring, then reverse for little-endian
        z_str = "".join(str(b) for b in final_state[::-1])
        z_index = int(z_str, 2)

        elapsed_time = end_time - start_time

        return psi[z_index], elapsed_time
