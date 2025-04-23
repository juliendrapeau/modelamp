"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using the statevector simulator.

Date created: 2025-04-23
"""

import sys
import os
import numpy as np
from qiskit.qasm2 import load
from qiskit import qasm2
from modelamp.benchmark.sv_solver import SVSolver
import json

def compute_amplitude_sv(params):

    num_qubits, num_layers, instance, input_path = params

    output_dir = "data/sv/" + f"q{num_qubits}-l{num_layers}-i{instance}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the circuit from the QASM file
    circuit = load(input_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

    # Generate a random initial and final state
    final_state = np.random.choice(a=[0, 1], size=num_qubits)  # |z>

    # Compute the amplitude using Complex Weighted Model Counting (CWMC)
    sv_solver = SVSolver()
    model_count, time = sv_solver.compute_amplitude(
        circuit=circuit,
        final_state=final_state,
    )
    
    # Save the results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "instance": instance,
                "model_count": [model_count.real, model_count.imag],
                "time": time,
            },
            f,
        )

    return model_count, time


if __name__ == "__main__":

    num_qubits = int(sys.argv[1])
    num_layers = int(sys.argv[2])
    instance = int(sys.argv[3])
    input_path = sys.argv[4]

    compute_amplitude_sv((num_qubits, num_layers, instance, input_path))
