"""
Purpose: Generate random quantum circuits for later simulation. These will be "brickwork" circuits. Here is an example for 5 qubits:

        q0 ───■───────■───────
              │      │
        q1 ───■───■───■───■───
                  │      │
        q2 ───■───■───■───■───
              │      │
        q3 ───■───■───■───■───
                  │      │
        q4 ───────■───────■───

Date created: 2025-04-11
"""

from qiskit.quantum_info import random_unitary
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
import numpy as np


def generate_brickwork_circuit(
    num_qubits, num_layers, file_path=None, rng=np.random.default_rng()
):
    """
    Generate a random brickwork circuit with the specified number of qubits and layers.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the circuit.
    num_layers: int
        The number of layers in the circuit.
    file_path: str, optional
        If provided, the circuit will be saved to this file in QASM format.
    rng: np.random.Generator, optional
        A random number generator for reproducibility. Default is a new default_rng instance.

    Returns
    -------
    circuit: QuantumCircuit
        The generated quantum circuit.
    """

    circuit = QuantumCircuit(num_qubits)

    for _ in range(num_layers):

        # Apply two-qubit gates starting from qubit 0
        for qubit in range(0, num_qubits - 1, 2):
            unitary = random_unitary(dims=4, seed=rng.integers(0, 2**32 - 1))
            circuit.append(unitary, [qubit, qubit + 1])

        # Apply two-qubit gates starting from qubit 1
        for qubit in range(1, num_qubits - 1, 2):
            unitary = random_unitary(dims=4, seed=rng.integers(0, 2**32 - 1))
            circuit.append(unitary, [qubit, qubit + 1])

    if file_path is not None:
        dump(circuit, file_path)

    return circuit
