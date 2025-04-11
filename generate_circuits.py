"""
Purpose: Generate random quantum circuits for later simulation. These will be "brickwork"
         circuits. Here is an example for 5 qubits:

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
import os

def generate_brickwork_circuit(num_qubits, num_layers):
    """
    Purpose: Generate a brickwork random quantum circuit.

    Inputs:
        num_qubits (int): The number of qubits in the circuit.
        num_layers (int): The number of layers in the circuit.

    Output:
        A QuantumCircuit object representing the generated circuit.

    TODO: Add a seed for reproducibility.
    """

    circuit = QuantumCircuit(num_qubits)

    for _ in range(num_layers):
        # Apply two-qubit gates starting from qubit 0
        for qubit in range(0, num_qubits - 1, 2):
            unitary = random_unitary(dims = 4)
            circuit.append(unitary, [qubit, qubit + 1])
        # Apply two-qubit gates starting from qubit 1
        for qubit in range(1, num_qubits - 1, 2):
            unitary = random_unitary(dims = 4)
            circuit.append(unitary, [qubit, qubit + 1])
    return circuit

if __name__ == "__main__":
    num_circuits = 20
    num_qubits = 10
    num_layers = 5
    save_path = "Data/circuits/n{}/".format(num_qubits)

    # Create data folder
    try:
        os.makedirs(save_path)
    except:
        pass


    for sample in range(num_circuits):
        print("Sample: ", sample)
        circuit = generate_brickwork_circuit(num_qubits=num_qubits, num_layers=num_layers)
        qasm = dump(circuit, save_path + "circuit{}.qasm".format(sample))