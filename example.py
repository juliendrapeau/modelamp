"""
Purpose: Example script to compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) and compare it with the statevector simulator and tensor network methods.

Date created: 2025-04-11
"""

import json
import os
import tempfile

import numpy as np
from qiskit.quantum_info import Statevector

from modelamp.benchmark.sv_solver import SVSolver
from modelamp.benchmark.tn_solver import TNSolver
from modelamp.cwmc import cwmc_solver
from modelamp.cwmc.cwmc_solver import CWMCSolver
from modelamp.generate_circuits import generate_brickwork_circuit


if __name__ == "__main__":

    # PARAMETERS

    num_qubits = 10
    num_layers = 5
    instance = 0
    rng = np.random.default_rng(seed=instance)
    
    verbose = False
    data_dir = "data/example/"  # None for temporary directory


    # DIRECTORY SETUP

    dir_prefix = os.path.join(f"q{num_qubits}" + f"-l{num_layers}" + f"-i{instance}")

    # Create the output directory if it doesn't exist
    # If data_dir is None, create a temporary directory
    if data_dir is None:
        data_dir = tempfile.TemporaryDirectory().name
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, dir_prefix), exist_ok=True)
    else:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, dir_prefix), exist_ok=True)

    output_dir = os.path.join(data_dir, dir_prefix)
    qasm_path = os.path.join(output_dir, "circuit.qasm")


    # CIRCUIT GENERATION

    # Define initial and final states
    initial_state = np.zeros(num_qubits, dtype=int)  # |0>
    final_state = np.random.choice(a=[0, 1], size=num_qubits)  # |z>

    # Generate a random quantum circuit
    circuit = generate_brickwork_circuit(
        num_qubits=num_qubits, num_layers=num_layers, file_path=qasm_path, rng=rng
    )


    # COMPUTE AMPLITUDE WITH CWMC

    # Compute the amplitude using Complex Weighted Model Counting (CWMC)
    cwmc_solver = CWMCSolver(output_dir=output_dir)
    model_count, time = cwmc_solver.compute_amplitude(
        circuit=circuit,
        initial_state=initial_state,
        final_state=final_state,
        verbose=verbose,
    )

    print("Amplitude with CWMC: ", model_count)
    print("Time: ", time)
    print()
    
    
    # COMPUTE AMPLITUDE WITH STATEVECTOR

    # Avoid using the statevector simulator for large circuits
    if num_qubits <= 10:

        sv_solver = SVSolver()
        amplitude, time = sv_solver.compute_amplitude(
            circuit=circuit,
            final_state=final_state,
        )

        print("Amplitude with SV: ", amplitude)
        print("Time: ", time)
        print()


    # COMPUTE AMPLITUDE WITH TENSOR NETWORKS

    tn_solver = TNSolver()
    amplitude, time = tn_solver.compute_amplitude(
        circuit_file_path=qasm_path,
        final_state=final_state,
    )
    
    print("Amplitude with TN: ", amplitude)
    print("Time: ", time)
    print()
    
    # Print the statevector if the number of qubits is small
    if num_qubits <= 5:
        print("Statevector: ")
        print(Statevector.from_instruction(circuit))
            
    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "instance": instance,
                "initial_state": initial_state.tolist(),
                "final_state": final_state.tolist(),
                "amplitude": [model_count.real, model_count.imag],
                "time": time,
            },
            f,
        )
