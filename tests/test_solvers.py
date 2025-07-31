"""
Compare with known circuits.
"""

import tempfile

import numpy as np
import qiskit.qasm2 as qasm2
from pytest import fixture, mark, raises

from modelamp.benchmark.sv_solver import SVSolver
from modelamp.benchmark.tn_solver import TNSolver
from modelamp.cwmc.cwmc_solver import CWMCSolver


@mark.parametrize(
    "num_qubits, depth",
    np.random.randint(4, 8, size=[5, 2]),
)
def test_solvers_random_circuits(num_qubits, depth):
    """
    INCORRECT. Generate a random circuit with the specified number of qubits and layers.
    """
    from qiskit.circuit.random import random_circuit

    sv_solver = SVSolver()
    tn_solver = TNSolver()
    cwmc_solver = CWMCSolver()

    initial_state = np.zeros(num_qubits, dtype=int)
    final_state = np.random.randint(0, 2, size=num_qubits)

    circuit = random_circuit(num_qubits, depth)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        qasm2.dump(circuit, temp_file.name)

    amplitude_sv = sv_solver.compute_amplitude(temp_file.name, final_state)[0]
    amplitude_tn = tn_solver.compute_amplitude(temp_file.name, final_state)[0]
    amplitude_cwmc = cwmc_solver.compute_amplitude(
        temp_file.name, final_state, initial_state=initial_state
    )[0]

    assert np.isclose(
        amplitude_sv, amplitude_tn
    ), "The amplitudes from SVSolver and TNSolver do not match."
    assert np.isclose(
        amplitude_sv, amplitude_cwmc
    ), "The amplitudes from SVSolver and CWMCSolver do not match."
    assert np.isclose(amplitude_tn, amplitude_cwmc), "The amplitudes from TNSolver and CWMCSolver do not match."  # type: ignore


@mark.parametrize("num_qubits, num_layers", np.random.randint(4, 8, size=[5, 2]))
def test_solvers_brickwork_circuit(num_qubits, num_layers):
    """
    Test the solver with random brickwork circuits.
    """
    from modelamp.gen.generate_circuits import (
        generate_brickwork_circuit,
        save_circuit_to_file,
    )

    sv_solver = SVSolver()
    tn_solver = TNSolver()
    cwmc_solver_all_paths = CWMCSolver(encoding_method="all-paths")
    cwmc_solver_valid_paths = CWMCSolver(encoding_method="valid-paths")

    initial_state = np.zeros(num_qubits, dtype=int)
    final_state = np.random.choice(a=[0, 1], size=num_qubits)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        circuit = generate_brickwork_circuit(
            num_qubits=num_qubits, num_layers=num_layers
        )
        save_circuit_to_file(circuit, temp_file.name)

    amplitude_sv = sv_solver.compute_amplitude(temp_file.name, final_state)[0]
    amplitude_tn = tn_solver.compute_amplitude(temp_file.name, final_state)[0]
    amplitude_cwmc_all_paths = cwmc_solver_all_paths.compute_amplitude(
        temp_file.name, final_state, initial_state=initial_state
    )[0]
    amplitude_cwmc_valid_paths = cwmc_solver_valid_paths.compute_amplitude(
        temp_file.name, final_state, initial_state=initial_state)[0]

    assert np.isclose(
        amplitude_cwmc_valid_paths, amplitude_cwmc_all_paths, atol=1e-4), "The amplitudes from CWMCSolver with valid paths and all paths do not match."
    assert np.isclose(
        amplitude_sv, amplitude_cwmc_valid_paths, atol=1e-4
    ), "The amplitudes from SVSolver and CWMCSolver do not match."
    assert np.isclose(
        amplitude_tn, amplitude_cwmc_valid_paths, atol=1e-4
    ), "The amplitudes from TNSolver and CWMCSolver do not match."
    assert np.isclose(
        amplitude_sv, amplitude_tn, atol=1e-4
    ), "The amplitudes from SVSolver and TNSolver do not match."


@mark.parametrize("num_qubits, num_layers", np.random.randint(4, 8, size=[5, 2]))
def test_solvers_transpiled_brickwork_circuit(num_qubits, num_layers):
    """
    Test the solver with random transpiled brickwork circuits.
    """
    from modelamp.gen.generate_circuits import (
        generate_brickwork_circuit,
        save_circuit_to_file,
        transpile_circuit,
    )

    sv_solver = SVSolver()
    tn_solver = TNSolver()
    cwmc_solver = CWMCSolver()

    initial_state = np.zeros(num_qubits, dtype=int)
    final_state = np.random.choice(a=[0, 1], size=num_qubits)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        circuit = generate_brickwork_circuit(
            num_qubits=num_qubits, num_layers=num_layers
        )
        transpiled_circuit = transpile_circuit(circuit)
        save_circuit_to_file(transpiled_circuit, temp_file.name)

    assert (
        transpiled_circuit.data != circuit.data
    ), "The transpiled circuit does not match the original circuit."

    amplitude_sv = sv_solver.compute_amplitude(temp_file.name, final_state)[0]
    amplitude_tn = tn_solver.compute_amplitude(temp_file.name, final_state)[0]
    amplitude_cwmc = cwmc_solver.compute_amplitude(
        temp_file.name, final_state, initial_state=initial_state
    )[0]

    assert np.isclose(
        amplitude_sv, amplitude_cwmc, atol=1e-4
    ), "The amplitudes from SVSolver and CWMCSolver do not match."
    assert np.isclose(
        amplitude_tn, amplitude_cwmc, atol=1e-4
    ), "The amplitudes from TNSolver and CWMCSolver do not match."
    assert np.isclose(
        amplitude_sv, amplitude_tn, atol=1e-4
    ), "The amplitudes from SVSolver and TNSolver do not match."
