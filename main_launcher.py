"""
Purpose: Launch multiple processes to compute the amplitude of a bitstring for a quantum circuit using complex weighted model counting (CWMC).

Date created: 2025-04-23
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import itertools
from multiprocessing import Pool

import tqdm

from main_cwmc import compute_amplitude_cwmc
from main_sv import compute_amplitude_sv
from main_tn import compute_amplitude_tn

if __name__ == "__main__":

    simulator = "cwmc"  # Choose from 'cwmc', 'sv', or 'tn'
    circuit_type = "brickwork"  # Options: "brickwork", "random_u3"
    transpiled = False  # Set to True if you used transpiled circuits
    encoding_method = "valid-paths"  # Options: "all-paths", "valid-path"

    if transpiled:
        input_dir = os.path.join(f"instances/{circuit_type}-transpiled/")
    else:
        input_dir = os.path.join(f"instances/{circuit_type}/")

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
            (
                circuit_type,
                num_qubits,
                num_layers,
                instance,
                str(dir_prefix) + "/circuit.qasm",
                encoding_method,  # Only used for CWMC
            )
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
