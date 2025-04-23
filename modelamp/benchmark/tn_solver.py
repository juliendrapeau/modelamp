"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using tensor networks contraction with quimb.

Date created: 2025-04-23
"""

from quimb.tensor.circuit import Circuit
import numpy as np
import time

class TNSolver:
    """
    Compute the amplitude of a bitstring for a quantum circuit using tensor network contraction with quimb (https://github.com/jcmgray/quimb).

    Attributes
    ----------
    contract_kwargs: dict
        Additional arguments for the tensor network contraction.
    """

    def __init__(self, contract_kwargs={}):

        self.contract_kwargs = contract_kwargs

    def compute_amplitude(self, circuit_file_path: str, final_state: np.ndarray):
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

        start_time = time.time()
        # Compute the amplitude using the tensor network contraction
        amplitude = circuit.amplitude(b=final_state, **self.contract_kwargs)
        end_time = time.time()
        
        elapsed_time = end_time - start_time

        return amplitude, elapsed_time
