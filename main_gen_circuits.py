"""
Purpose: Generate random quantum circuits.

Date created: 2025-04-23
"""

import itertools
import json
import os
from multiprocessing import Pool

import numpy as np
import tqdm
from qiskit import transpile

from modelamp.gen.generate_circuits import generate_circuit, save_circuit_to_file


def generate_circuits(params):
    """
    Generate a quantum circuit based on the specified parameters and save it to a file.

    Parameters
    ----------
    params: tuple
        A tuple containing the circuit type, number of qubits, number of layers, instance number, and directory prefix.
        Example: ("brickwork", 4, 10, 1, "instances/brickwork/q4-l10-i1")
    """

    circuit_type, num_qubits, num_layers, instance, transpile_to, dir_prefix = params

    file_path = os.path.join(dir_prefix, "circuit.qasm")

    rng = np.random.default_rng(seed=instance)

    if circuit_type == "k-qubit-brickwork":
        k = 3
    else:
        k = None

    circuit = generate_circuit(
        method=circuit_type,
        num_qubits=num_qubits,
        num_layers=num_layers,
        rng=rng,
        transpile_to=transpile_to,
        k=k,
    )

    save_circuit_to_file(circuit, file_path)  # type: ignore

    with open(os.path.join(dir_prefix, "parameters.json"), "w") as f:
        json.dump(
            {
                "circuit_type": circuit_type,
                "transpile_to": transpile_to,
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "instance": instance,
                "seed": instance,
            },
            f,
        )


if __name__ == "__main__":

    circuit_type = "brickwork"  # Options: "brickwork", "random-u3"
    transpile_to = ["cx", "u3"]  # Options: None, ["cx", "u3"]

    if transpile_to is None:
        output_dir = f"instances/{circuit_type}/"
    else:
        output_dir = f"instances/{circuit_type}-transpiled/"

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

        parameters_list.append(
            (
                circuit_type,
                num_qubits,
                num_layers,
                instance,
                transpile_to,
                str(dir_prefix),
            )
        )

    with Pool(8) as pool:
        list(
            tqdm.tqdm(
                pool.imap(generate_circuits, parameters_list),
                total=len(parameters_list),
            )
        )
