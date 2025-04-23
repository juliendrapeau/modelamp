"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) with the probabilistic exact model counter Ganak.

Date created: 2025-04-11
"""

import numpy as np
from qiskit import QuantumCircuit
from modelamp.cwmc.call_ganak import GanakSolver
from modelamp.cwmc.convert_circuits import (
    circuit_to_cnf,
    save_and_return_dimacs_with_weights,
)
import os


class CWMCSolver:
    """
    Compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) with the probabilistic exact model counter Ganak.

    Attributes
    ----------
    output_dir: str
        Directory to save the CNF file and results.
    ganak_path: str
        Path to the Ganak executable.
    ganak_kwargs: dict
        Additional arguments for Ganak.
    solver: GanakSolver
    """

    def __init__(self, output_dir: str, ganak_path="./ganak", ganak_kwargs={"mode": 2}):

        self.output_dir = output_dir
        self.ganak_path = ganak_path
        self.ganak_kwargs = ganak_kwargs
        self.solver = GanakSolver(ganak_path=ganak_path, ganak_kwargs=ganak_kwargs)

    def compute_amplitude(
        self,
        circuit: QuantumCircuit,
        initial_state: np.ndarray,
        final_state: np.ndarray,
        verbose=False,
    ):
        """
        Compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) with the probabilistic exact model counter Ganak.

        Parameters
        ----------
        circuit: QuantumCircuit
            The quantum circuit to compute the amplitude for.
        initial_state: np.ndarray
            The initial state of the quantum circuit.
        final_state: np.ndarray
            The final state of the quantum circuit.

        Returns
        -------
        model_count: int
            The number of models (amplitude) for the given bitstring.
        time: float
            The time taken to compute the amplitude.
        """

        # Convert the circuit to CNF
        formula, weights = circuit_to_cnf(
            circuit=circuit, initial_state=initial_state, final_state=final_state
        )

        cnf_file_path = os.path.join(self.output_dir, "circuit.cnf")

        # Save the CNF to a file
        save_and_return_dimacs_with_weights(
            clauses=formula, weights=weights, file_path=cnf_file_path
        )

        # Solve the CNF formula using Ganak
        model_count, time = self.solver.solve(
            cnf_file_path, verbose=verbose, output_dir=self.output_dir
        )

        return model_count, time
