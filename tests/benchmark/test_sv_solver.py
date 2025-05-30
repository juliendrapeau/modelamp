"""
Compare with known circuits
"""

import numpy as np
from pytest import fixture, mark, raises
from modelamp.benchmark.sv_solver import SVSolver
import tempfile

def test_bell_state():
    """
    Test the SVSolver with a Bell state circuit.
    """
    from qiskit import QuantumCircuit
    from qiskit.qasm2 import dump
    
    # Create a Bell state circuit
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Save the circuit to a temporary QASM file
        dump(circuit, temp_file.name) 
           
    # Compute the expected statevector for the Bell state |00> + |11>
    expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # |00> + |11>

    # Use SVSolver to compute the amplitude
    sv_solver = SVSolver()
    
    found_state = []
    for amplitude in [[0,0], [0,1], [1,0], [1,1]]:
        found_state.append(sv_solver.compute_amplitude(temp_file.name, final_state=np.array(amplitude))[0])

    # Check if the computed amplitude matches the expected state
    assert np.allclose(found_state, expected_state), "The amplitude for the Bell state does not match."