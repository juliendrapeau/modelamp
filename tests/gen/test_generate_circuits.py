import numpy as np
from pytest import fixture, mark, raises
from modelamp.gen.generate_circuits import generate_brickwork_circuit


@mark.parametrize(
    "num_qubits, num_layers",
    np.random.randint(5, 16, size=[5, 2]),
)
def test_equivalence_circuit_qasm(num_qubits, num_layers):
    """
    Test the equivalence between a circuit and its QASM file. The QASM file is generated from the circuit and then loaded back into a QuantumCircuit object. The test checks if the statevector of the original circuit matches the statevector of the loaded QASM circuit. Since the QASM file does not store the global phase, we set it manually to match the original circuit.
    """
    
    import tempfile

    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    
    with tempfile.NamedTemporaryFile() as temp_file:

        # Generate a random brickwork circuit
        circuit = generate_brickwork_circuit(num_qubits, num_layers, file_path=temp_file.name)
            
        # Load the QASM file
        qasm_circuit = qasm2.load(temp_file.name, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

    # Set the global phase of the QASM circuit to match the original circuit
    qasm_circuit.global_phase = circuit.decompose().global_phase
    
    statevector = Statevector.from_instruction(circuit)
    qasm_statevector = Statevector.from_instruction(qasm_circuit)

    # Compare the two circuits
    assert np.allclose(np.abs(statevector)**2, np.abs(qasm_statevector)**2), "The probabilities of the circuit does not match its equivalent in the QASM file."
    assert np.allclose(statevector, qasm_statevector), "The statevector of the circuit does not match its equivalent in the QASM file."
