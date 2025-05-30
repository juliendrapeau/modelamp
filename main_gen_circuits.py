"""
Purpose: Generate random quantum circuits using the Brickwork circuit generator.

Date created: 2025-04-23
"""

import itertools
import json
import os
from multiprocessing import Pool

import numpy as np
import tqdm

from modelamp.gen.generate_circuits import generate_brickwork_circuit


def generate_circuits(params):

    num_qubits, num_layers, instance, dir_prefix = params

    file_path = os.path.join(dir_prefix, "circuit.qasm")

    rng = np.random.default_rng(seed=instance)

    generate_brickwork_circuit(
        num_qubits=num_qubits, num_layers=num_layers, file_path=file_path, rng=rng
    )

    with open(os.path.join(dir_prefix, "parameters.json"), "w") as f:
        json.dump(
            {
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "instance": instance,
            },
            f,
        )


if __name__ == "__main__":

    output_dir = "data/circuits/"

    parameters_space = {
        "num_qubits": range(4, 31, 2),
        "num_layers": range(10, 11, 1),
        "num_instances": range(1, 11),
    }

    output_dir_path = os.path.join(output_dir)
    os.makedirs(output_dir_path, exist_ok=True)

    parameters_list = []
    for num_qubits, num_layers, instance in itertools.product(
        parameters_space["num_qubits"],
        parameters_space["num_layers"],
        parameters_space["num_instances"],
    ):
        dir_prefix = output_dir_path + f"q{num_qubits}-l{num_layers}-i{instance}"
        os.makedirs(dir_prefix, exist_ok=True)

        parameters_list.append((num_qubits, num_layers, instance, str(dir_prefix)))

    with Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap(generate_circuits, parameters_list),
                total=len(parameters_list),
            )
        )
