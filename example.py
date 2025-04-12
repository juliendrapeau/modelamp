"""
Purpose: Define a circuit unitary U and calculate <z|U|x> using the ganak model counter.

Date created: 2025-04-11
"""

import numpy as np
from qiskit import QuantumCircuit
from convert_circuits import circuit_to_cnf, save_and_return_dimacs_with_weights
from call_ganak import write_cnf_to_tempfile, run_ganak_on_cnf_file, parse_ganak_complex_output
from qiskit.quantum_info import Statevector
from generate_circuits import generate_brickwork_circuit

def compute_amplitude_z_array(circuit: QuantumCircuit, z_array: np.ndarray):
    """
        Purpose: Compute ⟨z|ψ⟩ where z is given as a NumPy array of bits.

        Inputs:
            - circuit: A Qiskit QuantumCircuit.
            - z_array: NumPy array of bits (e.g., np.array([0,1,0])).
        Returns:
            A complex amplitude ⟨z|ψ⟩.
    """
    psi = Statevector.from_instruction(circuit)

    # Convert z_array to bitstring, then reverse for little-endian
    z_str = ''.join(str(b) for b in z_array[::-1])
    z_index = int(z_str, 2)
    return psi[z_index]


if __name__ == "__main__":
    num_qubits = 5
    num_layers = 3
    #circuit = QuantumCircuit(num_qubits)
    #circuit.h(1)
    #circuit.y(0)
    
    circuit = generate_brickwork_circuit(num_qubits=num_qubits, num_layers=num_layers)

    initial_state = np.zeros(num_qubits, dtype=int) # |0>
    final_state = np.random.choice(a = [0,1], size = num_qubits)   # |z>
    formula, weights = circuit_to_cnf(circuit=circuit, initial_state=initial_state, final_state=final_state)
    cnf = save_and_return_dimacs_with_weights(clauses=formula, weights=weights)
    #print("CNF: ")
    #print(cnf)
    print("Number of constraints: ", len(formula))
    
    cnf_path = write_cnf_to_tempfile(cnf)
    #print(cnf_path)
    stdout, stderr = run_ganak_on_cnf_file(cnf_path)
    #print(stdout)
    model_count, time = parse_ganak_complex_output(stdout)
    print("Time       : ", time)
    print("Model count: ", model_count)

    if num_qubits <= 10:
        # Compare with statevector
        amplitude = compute_amplitude_z_array(circuit=circuit, z_array=final_state)
        print("Amplitude  : ", amplitude)
        if num_qubits <= 5:
            print("Psi: ")
            print(Statevector.from_instruction(circuit))