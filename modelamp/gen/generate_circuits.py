"""
Purpose: Generate random quantum circuits for later simulation.
Date created: 2025-04-11
"""

from os import PathLike

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump
from qiskit.quantum_info import random_unitary

def generate_circuit(
    method: str,
    num_qubits: int,
    num_layers: int,
    rng: np.random.Generator,
    transpile_to: list | None = None,
    **kwargs,
) -> QuantumCircuit:
    """
    Generate a random quantum circuit based on the specified method.

    Parameters
    ----------
    method: str
        The method to use for generating the circuit. Options include:
        - "brickwork": Generates a brickwork circuit.
        - "uncomputed-brickwork": Generates a brickwork circuit and uncomputes all gates by appending their Hermitian conjugates.
        - "random-u3": Generates a circuit with random one-qubit gates (U3) applied to each qubit in each layer.
        - "random-unitaries": Generates a circuit with random unitary gates applied to pairs of qubits in each layer.
        - "k-brickwork": Generates a k-qubit brickwork circuit with the specified number of qubits and layers.
    num_qubits: int
        The number of qubits in the circuit.
    num_layers: int
        The number of layers in the circuit.
    rng: np.random.Generator
        A random number generator for reproducibility. Default is a new default_rng instance.
    transpile_to: list, optional
        A list of basic gates to transpile the circuit to. If None, no transpilation is performed.
    kwargs: dict, optional
        Additional keyword arguments for specific circuit generation methods.

    Returns
    -------
    circuit: QuantumCircuit
        The generated quantum circuit.
    """

    generate_circuit_dict = {
        "brickwork": generate_brickwork_circuit,
        "semi-brickwork": generate_semi_brickwork_circuit,
        "uncomputed-brickwork": generate_uncomputed_brickwork_circuit,
        "random-u3": generate_random_u3_circuit,
        "cnot-brickwork": generate_cnot_brickwork_circuit,
        "swap-brickwork": generate_swap_brickwork_circuit,
        "random-unitaries": generate_random_unitaries_circuit,
        "k-brickwork": generate_k_brickwork_circuit,
    }

    if method not in generate_circuit_dict:
        raise ValueError(f"Unknown circuit generation method: {method}")

    circuit = generate_circuit_dict[method](num_qubits, num_layers, rng, **kwargs)

    if transpile_to is not None and isinstance(transpile_to, list):
        transpiled_circuit = transpile_circuit(circuit, basic_gates=transpile_to)
    else:
        transpiled_circuit = circuit

    return transpiled_circuit


def generate_brickwork_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng(), **kwargs
) -> QuantumCircuit:
    """
    Generate a random brickwork circuit. Here is an example for 5 qubits:


        q0 ───■───────■───────
              │       │
        q1 ───■───■───■───■───
                  │       │
        q2 ───■───■───■───■───
              │       │
        q3 ───■───■───■───■───
                  │       │
        q4 ───────■───────■───


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

def generate_semi_brickwork_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng(), r=0.5, **kwargs) -> QuantumCircuit:
    """
    Generate a semi-brickwork circuit with a specified fraction of two-qubit gates replaced by two single-qubit gates.
    """

    circuit = QuantumCircuit(num_qubits)
    
    list_unitaries = []
    for _ in range(num_layers):

        # Apply two-qubit gates starting from qubit 0
        for qubit in range(0, num_qubits - 1, 2):
            unitary = random_unitary(dims=4, seed=rng.integers(0, 2**32 - 1))
            list_unitaries.append((unitary, [qubit, qubit + 1]))

        # Apply two-qubit gates starting from qubit 1
        for qubit in range(1, num_qubits - 1, 2):
            unitary = random_unitary(dims=4, seed=rng.integers(0, 2**32 - 1))
            list_unitaries.append((unitary, [qubit, qubit + 1]))
    
    # Randomly select unique indices to replace
    replace_indices = sorted(rng.choice(len(list_unitaries), size=int(r*len(list_unitaries)), replace=False), reverse=True)
    for idx in replace_indices:
        u3_1 = random_unitary(dims=2, seed=rng.integers(0, 2**32 - 1))
        u3_2 = random_unitary(dims=2, seed=rng.integers(0, 2**32 - 1))
        unitary, qubits = list_unitaries[idx]
        # Replace the two-qubit unitary with two single-qubit unitaries on the same qubits
        list_unitaries[idx:idx+1] = [(u3_1, [qubits[0]]), (u3_2, [qubits[1]])]

    # Append all gates to the circuit
    for unitary, qubits in list_unitaries:
        circuit.append(unitary, qubits)
    
    return circuit

def generate_uncomputed_brickwork_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng(), **kwargs
) -> QuantumCircuit:
    """
    Generate a random brickwork circuit and uncompute all gates by appending their Hermitian conjugates.

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
    list_unitaries = []
    for _ in range(num_layers):

        # Apply two-qubit gates starting from qubit 0
        for qubit in range(0, num_qubits - 1, 2):
            unitary = random_unitary(dims=4, seed=rng.integers(0, 2**32 - 1))
            list_unitaries.append((unitary, [qubit, qubit + 1]))
            circuit.append(unitary, [qubit, qubit + 1])

        # Apply two-qubit gates starting from qubit 1
        for qubit in range(1, num_qubits - 1, 2):
            unitary = random_unitary(dims=4, seed=rng.integers(0, 2**32 - 1))
            list_unitaries.append((unitary, [qubit, qubit + 1]))
            circuit.append(unitary, [qubit, qubit + 1])

    # Append the Hermitian conjugates (transpose circuit)
    for unitary, qubits in reversed(list_unitaries):
        circuit.append(unitary.adjoint(), qubits)

    return circuit

def generate_cnot_brickwork_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng(), **kwargs
) -> QuantumCircuit:
    """
    Generate a random CNOT brickwork circuit.

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
            circuit.cx(qubit, qubit + 1)

        # Apply two-qubit gates starting from qubit 1
        for qubit in range(1, num_qubits - 1, 2):
            circuit.cx(qubit+1, qubit)

    return circuit

def generate_swap_brickwork_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng(), **kwargs
) -> QuantumCircuit:
    """
    Generate a random SWAP brickwork circuit.
    
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
            circuit.swap(qubit, qubit + 1)

        # Apply two-qubit gates starting from qubit 1
        for qubit in range(1, num_qubits - 1, 2):
            circuit.swap(qubit + 1, qubit)

    return circuit

def generate_random_u3_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng(), **kwargs
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

def generate_random_unitaries_circuit(
    num_qubits: int, num_layers: int, rng=np.random.default_rng(), k: int = 2, **kwargs
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
    k: int, optional
        The number of qubits to be entangled in each layer. Default is 2.

    Returns
    -------
    circuit: QuantumCircuit
        The generated quantum circuit.
    """

    circuit = QuantumCircuit(num_qubits)

    for _ in range(num_layers):

        for qubit in range(0, num_qubits - 1, k):
            unitary = random_unitary(dims=2**k, seed=rng.integers(0, 2**32 - 1))
            circuit.append(unitary, [qubit, qubit + 1])

    return circuit

def generate_k_brickwork_circuit(
    num_qubits: int, num_layers: int, k: int, rng=np.random.default_rng()
) -> QuantumCircuit:
    """
    INCOMPLETE. Generate a random k-qubit brickwork circuit with the specified number of qubits and layers.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the circuit.
    num_layers: int
        The number of layers in the circuit.
    k: int
        The number of qubits to be entangled in each layer.
    rng: np.random.Generator, optional
        A random number generator for reproducibility. Default is a new default_rng instance.

    Returns
    -------
    circuit: QuantumCircuit
        The generated quantum circuit.
    """

    circuit = QuantumCircuit(num_qubits)

    list_unitaries_1 = []
    list_unitaries_2 = []
    for _ in range(num_layers):

        # Apply two-qubit gates starting from qubit 0
        for qubit in range(0, num_qubits - 1, 2):
            unitary = random_unitary(dims=4, seed=rng.integers(0, 2**32 - 1))
            list_unitaries_1.append((unitary, [qubit, qubit + 1]))

        # Apply two-qubit gates starting from qubit 1
        for qubit in range(1, num_qubits - 1, 2):
            unitary = random_unitary(dims=4, seed=rng.integers(0, 2**32 - 1))
            list_unitaries_2.append((unitary, [qubit, qubit + 1]))

        for qubit in range(0, num_qubits, 2):

            unitary = list_unitaries_1[qubit][0].expand() @ list_unitaries_2[qubit][0]
            circuit.append(unitary, [qubit, qubit + 1, qubit + 2])

    return circuit

def transpile_circuit(
    circuit: QuantumCircuit, basic_gates: list = ["cx", "u3"]
) -> QuantumCircuit:
    """
    Transpile the quantum circuit by transforming it to a basis of basic gates (cx and u3).

    Parameters
    ----------
    circuit: QuantumCircuit
        The quantum circuit to transpile.

    Returns
    -------
    transpiled_circuit: QuantumCircuit
        The transpiled quantum circuit.
    """

    transpiled_circuit = transpile(
        circuit, basis_gates=basic_gates, optimization_level=3
    )

    return transpiled_circuit

def save_circuit_to_file(circuit: QuantumCircuit, file_path: PathLike[str]) -> None:
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
