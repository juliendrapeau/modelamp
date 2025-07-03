"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using tensor network contraction with quimb.

Date created: 2025-04-23
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from quimb.tensor.circuit import Circuit


class TNSolver:
    """
    Compute the amplitude of a bitstring for a quantum circuit using tensor network contraction with quimb (https://github.com/jcmgray/quimb).

    Attributes
    ----------
    contract_kwargs: dict
        Additional arguments for the tensor network contraction.
    """

    def __init__(self, contract_kwargs: dict = {}):

        self.contract_kwargs = contract_kwargs

    def compute_amplitude(
        self, circuit_file_path: str, final_state: np.ndarray
    ) -> tuple[(complex | dict[str, complex]), float]:
        """
        Compute the amplitude of a bitstring for a quantum circuit using tensor network contraction with quimb

        Parameters
        ----------
        circuit_file_path: str
            Path to the circuit file in QASM format.
        final_state: list
            The final state of the circuit as a list of bits.
        contract_kwargs: dict
            Additional arguments for the tensor network contraction.

        Returns
        -------
        amplitude: complex
            The computed amplitude.
        """

        # Convert the circuit to a tensor network and contract it
        circuit = Circuit.from_openqasm2_file(circuit_file_path)

        # Change ordering?
        # final_state = "".join(str(b) for b in final_state[::-1])

        # Compute the amplitude using the tensor network contraction
        start_time = time.time()
        amplitude = circuit.amplitude(b=final_state, **self.contract_kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time

        return amplitude, elapsed_time

    def plot_circuit(self, circuit_file_path: str) -> None:
        """
        Plot the circuit.

        Parameters
        ----------
        circuit_file_path: str
            Path to the circuit file in QASM format.
        """

        # Load the circuit from the QASM file
        circuit = Circuit.from_openqasm2_file(circuit_file_path)

        # Plot the circuit
        circuit.draw()
        plt.show()
