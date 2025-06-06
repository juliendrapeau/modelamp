import tempfile

import numpy as np
from pytest import fixture, mark, raises

from modelamp.gen.generate_circuits import (
    generate_brickwork_circuit,
    save_circuit_to_file,
    transpile_circuit,
)


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
        circuit = generate_brickwork_circuit(num_qubits, num_layers)
        # Save the circuit to a QASM file
        save_circuit_to_file(circuit, temp_file.name)

        # Load the QASM file
        qasm_circuit = qasm2.load(
            temp_file.name, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )

    # Set the global phase of the QASM circuit to match the original circuit
    qasm_circuit.global_phase = circuit.decompose().global_phase

    statevector = Statevector.from_instruction(circuit)
    qasm_statevector = Statevector.from_instruction(qasm_circuit)

    # Compare the two circuits
    assert np.allclose(
        np.abs(statevector) ** 2, np.abs(qasm_statevector) ** 2
    ), "The probabilities of the circuit does not match its equivalent in the QASM file."
    assert np.allclose(
        statevector, qasm_statevector
    ), "The statevector of the circuit does not match its equivalent in the QASM file."


@mark.parametrize("num_qubits, num_layers", np.random.randint(4, 8, size=[5, 2]))
def test_transpilation(num_qubits, num_layers):
    """
    Test the solver with a brickwork circuit.
    """

    from modelamp.benchmark.sv_solver import SVSolver

    sv_solver = SVSolver()

    initial_state = np.zeros(num_qubits, dtype=int)
    final_state = np.random.choice(a=[0, 1], size=num_qubits)

    circuit = generate_brickwork_circuit(num_qubits=num_qubits, num_layers=num_layers)
    with tempfile.NamedTemporaryFile(delete=False) as temp1_file:
        save_circuit_to_file(circuit, temp1_file.name)

    with tempfile.NamedTemporaryFile(delete=False) as temp2_file:
        transpiled_circuit = transpile_circuit(circuit)
        save_circuit_to_file(transpiled_circuit, temp2_file.name)

    assert (
        transpiled_circuit.data != circuit.data
    ), "The transpiled circuit does not match the original circuit."

    amplitude_circuit = sv_solver.compute_amplitude(temp1_file.name, final_state)[0]
    amplitude_transpiled = sv_solver.compute_amplitude(temp2_file.name, final_state)[0]

    assert np.isclose(
        np.abs(amplitude_circuit) ** 2, np.abs(amplitude_transpiled) ** 2, atol=1e-4
    ), "The probabilities of the original and transpiled circuits do not match."
