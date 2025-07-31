import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import tempfile

import numpy as np
from qiskit import QuantumCircuit

from modelamp.cwmc.call_ganak import GanakSolver
from modelamp.cwmc.convert_circuits import CircuitToCNFConverter


def test_simple_circuit():
    """INCOMPLETE."""

    num_qubits = 2
    # Create a simple quantum circuit
    qc = QuantumCircuit(num_qubits)
    # qc.h(0)
    # qc.cx(0, 1)
    # qc.x(1)

    qc.u(0.1, 0.2, 0.3, 0)  # Apply a U3 gate to qubit 0
    qc.u(0.4, 0.5, 0.6, 1)  # Apply a U3 gate to qubit 1

    initial_state = np.array([1, 0])
    final_state = np.array([1, 0])  # |10> state

    # Convert the circuit to CNF
    converter = CircuitToCNFConverter()
    clauses, weights = converter.convert(
        circuit=qc, initial_state=initial_state, final_state=final_state
    )
    print(converter.var_mgr.io_map)
    print("Clauses:", clauses)
    print("Weights:", weights)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "test_circuit.cnf")
        print("Exporting CNF to:", filename)

        converter.export_dimacs(filename=filename)

        # Solve the CNF using CWMC_Solver
        ganak_solver = GanakSolver(ganak_path="./ganak", ganak_kwargs={"mode": 2})
        count, time = ganak_solver.solve(cnf_file_path=filename)
        print("Model count:", count)
        print("Probability:", np.abs(count) ** 2)
        print("Time taken:", time)


"""
Add a test to verify if the number of variables and clauses in the CNF file matches the expected values.
"""

if __name__ == "__main__":
    test_simple_circuit()
