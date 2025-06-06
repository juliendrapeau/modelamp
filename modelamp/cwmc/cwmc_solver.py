"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) with the probabilistic exact model counter Ganak.

Date created: 2025-04-11
"""

import os
import tempfile

import numpy as np
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS, load

from modelamp.cwmc.call_ganak import GanakSolver
from modelamp.cwmc.convert_circuits import CircuitToCNFConverter


class CWMCSolver:
    """
    Compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) with the probabilistic exact model counter Ganak.

    Attributes
    ----------
    output_dir: str
        Directory to store the output files. If None, a temporary directory will be created.
    ganak_path: str
        Path to the Ganak executable.
    ganak_kwargs: dict
        Additional arguments for Ganak.
    solver: GanakSolver
    """

    def __init__(
        self,
        output_dir: str | None = None,
        ganak_path: str = "./ganak",
        ganak_kwargs: dict = {"mode": 2},
    ):

        self.output_dir = output_dir
        self.ganak_path = ganak_path
        self.ganak_kwargs = ganak_kwargs
        self.solver = GanakSolver(ganak_path=ganak_path, ganak_kwargs=ganak_kwargs)

    def compute_amplitude(
        self,
        circuit_file_path: str,
        final_state: np.ndarray,
        initial_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> tuple[int, float, int, int]:
        """
        Compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) with the probabilistic exact model counter Ganak.

        Parameters
        ----------
        circuit_file_path: str
            The path to the QASM file containing the quantum circuit.
        final_state: np.ndarray
            The final state of the quantum circuit.
        initial_state: np.ndarray
            The initial state of the quantum circuit. If None, it defaults to |0> for all qubits.
        verbose: bool
            If True, print additional information during the computation.

        Returns
        -------
        model_count: int
            The number of models (amplitude) for the given bitstring.
        time: float
            The time taken to compute the amplitude.
        num_vars: int
            The number of variables in the CNF formula.
        num_clauses: int
            The number of clauses in the CNF formula.
        """

        # Load the circuit from the QASM file
        circuit = load(
            circuit_file_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS
        )

        if self.output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            output_dir = self.output_dir

        cnf_file_path = os.path.join(output_dir, "circuit.cnf")

        if initial_state is None:
            initial_state = np.zeros(circuit.num_qubits, dtype=int)

        converter = CircuitToCNFConverter()
        converter.convert(
            circuit=circuit, initial_state=initial_state, final_state=final_state
        )
        converter.export_dimacs(filename=cnf_file_path)
        num_vars = converter.num_vars
        num_clauses = converter.num_clauses

        # Solve the CNF formula using Ganak
        model_count, time = self.solver.solve(
            cnf_file_path, verbose=verbose, output_dir=self.output_dir
        )

        return model_count, time, num_vars, num_clauses
