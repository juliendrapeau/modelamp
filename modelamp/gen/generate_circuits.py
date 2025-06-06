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

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump
from qiskit.quantum_info import random_unitary


def generate_brickwork_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng()
) -> QuantumCircuit:
    """
    Generate a random brickwork circuit with the specified number of qubits and layers.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the circuit.
    num_layers: int
        The number of layers in the circuit.
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

    return circuit


def generate_random_u3_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng()
) -> QuantumCircuit:
    """
    Generate a random quantum circuit with one-qubit gates applied to each qubit in each layer.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the circuit.
    num_layers: int
        The number of layers in the circuit.
    rng: np.random.Generator, optional
        A random number generator for reproducibility. Default is a new default_rng instance.

    Returns
    -------
    circuit: QuantumCircuit
        The generated quantum circuit.
    """

    circuit = QuantumCircuit(num_qubits)

    for _ in range(num_layers):

        for qubit in range(num_qubits):
            unitary = random_unitary(dims=2, seed=rng.integers(0, 2**32 - 1))
            circuit.append(unitary, [qubit])

    return circuit


def transpile_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Transpile the quantum circuit by transforming it to a basis of basic gates (cx and u3).

    Parameters
    ----------
    circuit: QuantumCircuit
        The quantum circuit to transpile.
    file_path: str, optional
        If provided, the transpiled circuit will be saved to this file in QASM format.

    Returns
    -------
    transpiled_circuit: QuantumCircuit
        The transpiled quantum circuit.
    """

    basic_gates = ["cx", "u3"]
    transpiled_circuit = transpile(
        circuit, basis_gates=basic_gates, optimization_level=3
    )

    return transpiled_circuit


def save_circuit_to_file(circuit: QuantumCircuit, file_path: str) -> None:
    """
    Save the quantum circuit to a file in QASM format.

    Parameters
    ----------
    circuit: QuantumCircuit
        The quantum circuit to save.
    file_path: str
        The path to the file where the circuit will be saved.
    """

    dump(circuit, file_path)
