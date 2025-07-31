# modelamp

In this repository, we encode a quantum circuit as a weighted model counting instance. Specifically, if $C$ represents a quantum circuit on $n$-qubits and $z \in \left\\{0,1\right\\}^n$, then we use the model counter [Ganak](https://github.com/meelgroup/ganak) to compute $\braket{z|C|0}$.

## Installation

To run the code, you need to have Ganak in this directory. We use version 2.4.3, which you can find on [GitHub](https://github.com/meelgroup/ganak/releases/tag/release%2F2.4.3). Make sure you name the executable `ganak`.

Otherwise, install the packages in `requirements.txt`.

## Example

In `example.py`, one can specify a circuit and get the desired amplitude with a model counting, tensor network or statevector approach. For small numbers of qubits, one can check against the actual state vector using Qiskit.