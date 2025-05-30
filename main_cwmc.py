"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using complex weighted model counting (CWMC).

Date created: 2025-04-23
"""

import json
import os
import sys

import numpy as np
from qiskit import qasm2
from qiskit.qasm2 import load

from modelamp.cwmc.cwmc_solver import CWMCSolver


def compute_amplitude_cwmc(params):

    num_qubits, num_layers, instance, input_path = params

    ganak_path = "./ganak"
    ganak_kwargs = {
        "mode": 2,
        "delta": 0.05,
    }

    output_dir = "data/cwmc/" + f"q{num_qubits}-l{num_layers}-i{instance}"
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "results.json")
    if os.path.exists(output_file_path):
        print(f"Results already exist for {output_file_path}. Skipping computation.")
        return

    # Load the circuit from the QASM file
    circuit = load(input_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

    # Generate a random initial and final state
    initial_state = np.zeros(num_qubits, dtype=int)  # |0>
    final_state = np.random.choice(a=[0, 1], size=num_qubits)  # |z>

    # Compute the amplitude using Complex Weighted Model Counting (CWMC)
    cwmc_solver = CWMCSolver(
        output_dir=output_dir, ganak_path=ganak_path, ganak_kwargs=ganak_kwargs
    )
    model_count, time = cwmc_solver.compute_amplitude(
        circuit_file_path=input_path,
        initial_state=initial_state,
        final_state=final_state,
        verbose=False,
    )

    # Save the results
    with open(output_file_path, "w") as f:
        json.dump(
            {
                "solver": "cwmc",
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "instance": instance,
                "model_count": [model_count.real, model_count.imag],
                "time": time,
            },
            f,
            indent=4,
        )

    return model_count, time


if __name__ == "__main__":

    num_qubits = int(sys.argv[1])
    num_layers = int(sys.argv[2])
    instance = int(sys.argv[3])
    input_path = sys.argv[4]

    compute_amplitude_cwmc((num_qubits, num_layers, instance, input_path))
