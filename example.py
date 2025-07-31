"""
Purpose: Example script to compute the amplitude of a bitstring for a quantum circuit using Complex Weighted Model Counting (CWMC) and compare it with the statevector simulator and tensor network methods.

Date created: 2025-04-11
"""

import json
import os

# Uncomment to compare the performance of different solvers without parallel execution
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import tempfile

import numpy as np
from qiskit.quantum_info import Statevector

from modelamp.benchmark.sv_solver import SVSolver
from modelamp.benchmark.tn_solver import TNSolver
from modelamp.cwmc import cwmc_solver
from modelamp.cwmc.cwmc_solver import CWMCSolver
from modelamp.gen.generate_circuits import generate_circuit, save_circuit_to_file

if __name__ == "__main__":

    # PARAMETERS

    num_qubits = 4
    num_layers = 4
    instance = 1
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
    final_state = rng.choice(a=[0, 1], size=num_qubits)  # |z>

    # Generate a random quantum circuit
    circuit = generate_circuit(
        method="brickwork", num_qubits=num_qubits, num_layers=num_layers, rng=rng
    )
    # Save the circuit to a QASM file
    save_circuit_to_file(circuit, qasm_path) # type: ignore

    # COMPUTE AMPLITUDE WITH CWMC

    # Compute the amplitude using Complex Weighted Model Counting (CWMC)
    
    cwmc_solver = CWMCSolver(output_dir=output_dir)
    amplitude_cwmc, time_cwmc = cwmc_solver.recursive_compute_amplitude(
        circuit_file_path=qasm_path,
        initial_state=initial_state,
        final_state=final_state,
        verbose=verbose,
    )[:2]


    print("Amplitude with CWMC: ", amplitude_cwmc)
    print("Time with CWMC: ", time_cwmc)
    print()

    # COMPUTE AMPLITUDE WITH STATEVECTOR

    # Avoid using the statevector simulator for large circuits
    amplitude_sv = None
    time_sv = None
    if num_qubits <= 31:

        sv_solver = SVSolver()  
        amplitude_sv, time_sv = sv_solver.compute_amplitude(
            circuit_file_path=qasm_path,
            final_state=final_state,
        )

        print("Amplitude with SV: ", amplitude_sv)
        print("Time with SV: ", time_sv)
        print()

    # COMPUTE AMPLITUDE WITH TENSOR NETWORKS

    tn_solver = TNSolver()
    amplitude_tn, time_tn = tn_solver.compute_amplitude(
        circuit_file_path=qasm_path,
        final_state=final_state,
    )

    print("Amplitude with TN: ", amplitude_tn)
    print("Time with TN: ", time_tn)
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
                "amplitude_cwmc": [amplitude_cwmc.real, amplitude_cwmc.imag],
                "amplitude_sv": [amplitude_sv.real, amplitude_sv.imag] if amplitude_sv is not None else None,
                "amplitude_tn": [amplitude_tn.real, amplitude_tn.imag],
                "time_cwmc": time_cwmc,
                "time_sv": time_sv,
                "time_tn": time_tn,
                "circuit_file_path": qasm_path,
            },
            f,
        )
