"""
Compare with known circuits
"""

import numpy as np
from pytest import fixture, mark, raises
from modelamp.benchmark.sv_solver import SVSolver
from modelamp.benchmark.tn_solver import TNSolver
from modelamp.cwmc.cwmc_solver import CWMCSolver
import qiskit.qasm2 as qasm2
import tempfile


@mark.parametrize(
    "num_qubits, depth",
    np.random.randint(5, 16, size=[5, 2]),
)
def test_solvers_random_circuits(num_qubits, depth):
    """
    Generate a random circuit with the specified number of qubits and layers.
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
    amplitude_cwmc = cwmc_solver.compute_amplitude(temp_file.name, final_state, initial_state=initial_state)[0]

    assert np.isclose(amplitude_sv, amplitude_tn), "The amplitudes from SVSolver and TNSolver do not match."
    assert np.isclose(amplitude_sv, amplitude_cwmc), "The amplitudes from SVSolver and CWMCSolver do not match."
    assert np.isclose(amplitude_tn, amplitude_cwmc), "The amplitudes from TNSolver and CWMCSolver do not match." # type: ignore


@mark.parametrize("num_qubits, num_layers", np.random.randint(4, 8, size=[5, 2]))
def test_solvers_brickwork_circuit(num_qubits, num_layers):
    """
    Test the solver with a brickwork circuit.
    """
    from modelamp.gen.generate_circuits import generate_brickwork_circuit

    sv_solver = SVSolver()
    tn_solver = TNSolver()
    cwmc_solver = CWMCSolver()

    initial_state = np.zeros(num_qubits, dtype=int)
    final_state = np.random.choice(a=[0, 1], size=num_qubits)
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        generate_brickwork_circuit(
            num_qubits=num_qubits, num_layers=num_layers, file_path=temp_file.name
        )
        
    amplitude_sv = sv_solver.compute_amplitude(temp_file.name, final_state)[0]
    amplitude_tn = tn_solver.compute_amplitude(temp_file.name, final_state)[0]
    amplitude_cwmc = cwmc_solver.compute_amplitude(temp_file.name, final_state, initial_state=initial_state)[0]
        
    assert np.isclose(amplitude_sv, amplitude_cwmc, atol=1e-4), "The amplitudes from SVSolver and CWMCSolver do not match."
    assert np.isclose(amplitude_tn, amplitude_cwmc, atol=1e-4), "The amplitudes from TNSolver and CWMCSolver do not match."
    assert np.isclose(amplitude_sv, amplitude_tn, atol=1e-4), "The amplitudes from SVSolver and TNSolver do not match."