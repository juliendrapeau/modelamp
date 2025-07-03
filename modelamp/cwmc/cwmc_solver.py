"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) with the probabilistic exact model counter Ganak.

Date created: 2025-04-11
"""

import os
import tempfile
import time
from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np
from qiskit import QuantumCircuit
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
    encoding_method: str
        Method to encode the quantum circuit into CNF. Options include "valid-paths" and "all-paths".
    ganak_path: str
        Path to the Ganak executable.
    ganak_kwargs: dict
        Additional arguments for Ganak.
    solver: GanakSolver
    """

    def __init__(
        self,
        output_dir: str | None = None,
        encoding_method: str = "valid-paths",
        ganak_path: str = "./ganak",
        ganak_kwargs: dict = {"mode": 2},
    ):

        self.output_dir = output_dir
        self.encoding_method = encoding_method
        self.ganak_path = ganak_path
        self.ganak_kwargs = ganak_kwargs
        self.solver = GanakSolver(ganak_path=ganak_path, ganak_kwargs=ganak_kwargs)

    def compute_amplitude(
        self,
        circuit_file_path: str,
        final_state: np.ndarray,
        initial_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> tuple[complex, float, int, int]:
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
        model_count: complex
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

        converter = CircuitToCNFConverter(encoding_method=self.encoding_method)
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

    def recursive_compute_amplitude(
        self,
        circuit_file_path: str,
        final_state: np.ndarray,
        initial_state: np.ndarray | None = None,
        num_cuts: int = 1,
        verbose: bool = False,
    ):
        """
        INCOMPLETE. Compute the amplitude recursively for a quantum circuit.
        """

        # Load the circuit from the QASM file
        circuit = load(
            circuit_file_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS
        )

        if initial_state is None:
            initial_state = np.zeros(circuit.num_qubits, dtype=int)

        circuits = split_circuit(circuit, num_cuts)

        bits = product([0, 1], repeat=circuit.num_qubits)
        indices = product(bits, repeat=num_cuts)

        worker_fn = partial(
            path_amplitudes,
            circuits=circuits,
            initial_state=initial_state,
            final_state=final_state,
        )

        start_time = time.time()
        with Pool() as pool:
            results = pool.map(worker_fn, indices)
        end_time = time.time()

        total_time = end_time - start_time
        total_model_count = sum(results)

        return total_model_count, total_time, None, None


def compute_amplitude_for_cut(circuit, initial_state, final_state):
    """
    INCOMPLETE. Compute the amplitude for a single cut of the circuit.
    """

    output_dir = tempfile.mkdtemp()

    converter = CircuitToCNFConverter()
    converter.convert(
        circuit=circuit, initial_state=initial_state, final_state=final_state
    )
    cnf_file_path = os.path.join(output_dir, "circuit.cnf")
    converter.export_dimacs(filename=cnf_file_path)

    # Solve the CNF formula using Ganak
    solver = GanakSolver()
    model_count, time = solver.solve(cnf_file_path, output_dir=output_dir)

    return model_count


def path_amplitudes(index, circuits, initial_state, final_state):
    """
    INCOMPLETE. Compute the amplitude for a path through the circuit given by the index.
    """

    amp = complex(1, 0)
    initial = initial_state

    for i in range(len(circuits) - 1):
        final = np.array(index[i])

        model_count = compute_amplitude_for_cut(circuits[i], initial, final)
        amp *= model_count
        initial = final

    model_count = compute_amplitude_for_cut(circuits[-1], initial, final_state)
    amp *= model_count

    return amp


def split_circuit(circuit, num_cuts):
    """
    INCOMPLETE. Split a quantum circuit into multiple smaller circuits for recursive computation.
    """

    n = len(circuit.qubits)
    num_gates = len(circuit.data)

    circuit_list = []
    for i in range(num_cuts + 1):
        qc = QuantumCircuit(n)
        start = i * (num_gates // num_cuts)
        end = (i + 1) * (num_gates // num_cuts) if i < num_cuts - 1 else num_gates
        for j in range(start, end):
            qc.append(circuit.data[j][0], circuit.data[j][1])
        circuit_list.append(qc)

    return circuit_list
