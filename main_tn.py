"""
Purpose: Compute the amplitude of a bitstring for a quantum circuit using the statevector simulator.

Date created: 2025-04-23
"""

import json
import os
import sys

import numpy as np

from modelamp.benchmark.tn_solver import TNSolver


def compute_amplitude_tn(params):

    circuit_type, num_qubits, num_layers, instance, input_path = params[:5]

    output_dir = (
        "data/tn/" + f"{circuit_type}/" + f"q{num_qubits}-l{num_layers}-i{instance}"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "results.json")
    if os.path.exists(output_file_path):
        print(f"Results already exist for {output_file_path}. Skipping computation.")
        return

    # Generate a random final state
    rng = np.random.default_rng(seed=42)
    final_state = rng.integers(0, 2, size=num_qubits)  # |z>

    # Compute the amplitude using Complex Weighted Model Counting (CWMC)
    sv_solver = TNSolver()
    model_count, time = sv_solver.compute_amplitude(
        circuit_file_path=input_path,
        final_state=final_state,
    )

    # Save the results
    with open(output_file_path, "w") as f:
        json.dump(
            {
                "solver": "tn",
                "circuit_type": circuit_type,
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "instance": instance,
                "final_state": final_state.tolist(),
                "model_count": [model_count.real, model_count.imag],
                "time": time,
            },
            f,
            indent=4,
        )

    return model_count, time


if __name__ == "__main__":

    circuit_type = str(sys.argv[1])
    num_qubits = int(sys.argv[2])
    num_layers = int(sys.argv[3])
    instance = int(sys.argv[4])
    input_path = str(sys.argv[5])

    compute_amplitude_tn((circuit_type, num_qubits, num_layers, instance, input_path))
