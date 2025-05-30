"""
Purpose: Launch multiple processes to compute the amplitude of a bitstring for a quantum circuit using complex weighted model counting (CWMC).

Date created: 2025-04-23
"""

import os
import tqdm
from multiprocessing import Pool
import itertools
from main_cwmc import compute_amplitude_cwmc
from main_sv import compute_amplitude_sv
from main_tn import compute_amplitude_tn

if __name__ == "__main__":

    simulator = "cwmc"  # Choose from 'cwmc', 'sv', or 'tn'
    input_dir = os.path.join("data/circuits/")

    parameters_space = {
        "num_qubits": range(4, 31, 2),
        "num_layers": range(10, 11, 1),
        "num_instances": range(1, 11),
    }

    parameters_list = []
    for num_qubits, num_layers, instance in itertools.product(
        parameters_space["num_qubits"],
        parameters_space["num_layers"],
        parameters_space["num_instances"],  
    ):
        dir_prefix = input_dir + f"q{num_qubits}-l{num_layers}-i{instance}"

        parameters_list.append(
            (num_qubits, num_layers, instance, str(dir_prefix) + "/circuit.qasm")
        )

    if simulator == "cwmc":
        compute_amplitude = compute_amplitude_cwmc
    elif simulator == "sv":
        compute_amplitude = compute_amplitude_sv
    elif simulator == "tn":
        compute_amplitude = compute_amplitude_tn
    else:
        raise ValueError("Invalid simulator type. Choose 'cwmc' or 'sv'.")

    with Pool(processes=4) as pool:
        list(
            tqdm.tqdm(
                pool.imap(compute_amplitude, parameters_list),
                total=len(parameters_list),
            )
        )
