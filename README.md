# modelamp

In this repository, we have code to cast a quantum circuit as a weighted model counting instance. Specifically, if $C$ represents a quantum circuit on $n$-qubits and $z \in \left\\{0,1\right\\}^n$, then the weighted model counter Ganak will compute $\braket{z|C|0}$.

## Background

The big idea behind this project is that we can represent quantum circuits and states using weighted model counting instances. By convention, I will use $x$ to present the Boolean variables of the instance. To convert a circuit into a weighted model counting instance, we must encode the initial state, the final state, and the circuit as conjunctive normal form (CNF) formulas.

First, we encode the initial state as a formula. For simplicity, the input state will be $\ket{0}^{\otimes n}$. We will encode this in the first $n$ variables of $x$, so $x_1, x_2, \ldots, x_n$. The corresponding formula is then $F_{\ket{0}} \equiv \bigwedge_{i = 1}^n - x_i$. Note that a subtraction sign is a logical negation, and it's there to mean that the solution to the formula is when all variables are set to False (0).

Second, we encode each gate as a formula. Let $U$ be a $2^k \times 2^k$ unitary. We can reshape the matrix into one of size $(2,2,2\ldots,2)$, so $U_{t}$ with $t \in \left\\{0,1\right\\}^{2k}$ labels the indices of $U$. Each unitary gate will have $2k$ "external" variables associated to it (for the input/output of the gate), and $2^{2k}$ "internal" variables associated to the elements of $U$. Let $x_{i}, x_{i+1}, \ldots, x_{i+k-1}$ be the external input variables, let $u_{i+k}, x_{i+k+1}, \ldots, u_{i+k+2^{2k}-1}$ be internal variables associated to $U$, and let $x_{i+k+2^{2k}}, \ldots, x_{i+2k+2^{2k}-1}$ be the external output variables. For compact notation, define $X \equiv \left(x_{i}, x_{i+1}, \ldots, x_{i+k-1}, x_{i+k+2^{2k}}, \ldots, x_{i+2k+2^{2k}-1} \right)$ be the external variables. The corresponding formula for element $t \in \left\\{0,1\right\\}^{2k}$ of $U$ is then:

$$F_{t} = -u_{t} \bigvee_{s = 1}^{2k} (-1)^{t_s} x_s.$$

Note that the decomposition of $U$ into components $U_t$ requires that the first $k$ components of $t$ specify the input of $U$, while the last $k$ components of $t$ specify the output of $U$.

The formula for the entire gate is then $F_U = \bigwedge_{t \in \left\\{0,1\right\\}^{2k}} F_t$.

Finally, we set an explicit weight $W$ for the internal variables, based on the corresponding entry of $U$. In particular, we set:
$$W[-u_t] \equiv U_{t}, \,\,\, W[-u_t] \equiv 1 - U_{t}.$$

As an example, consider the single qubit ($k = 1$) gate $U$, with $U_{00} = a$, $U_{01} = b$, $U_{10} = c$, and $U_{11} = d$. The corresponding formula is

$$F_U = \bigwedge_{t \in \left\\{0,1\right\\}^2} F_{t},$$
where
$$F_{t} = -u_{t} \lor (-1)^{t_1} x_1 \lor (-1)^{t_2} x_2.$$

The weights will then be:
$$W[-u_{00}] = a, \,\,\, W[u_{00}] = 1 - a.$$
$$W[-u_{01}] = b, \,\,\, W[u_{01}] = 1 - b.$$
$$W[-u_{10}] = c, \,\,\, W[u_{10}] = 1 - c.$$
$$W[-u_{11}] = d, \,\,\, W[u_{11}] = 1 - d.$$

Finally, we must "connect" gates together to create a circuit. To do so, we must match external variables between sequential gates in the circuit. For example, if we apply a Hadamard gate $H$ and then a $Y$ rotation to a single qubit, then we might use the external variables $x_1$ and $x_2$ for $F_H$, and the external variables $x_3$ and $x_4$ for $F_Y$. However, we want $x_2 = x_3$ if these gates are sequential, which means setting the constraint $x_2 \iff x_3$, which is equivalent to the formula $F = (x_2 \lor -x_3) \land (-x_2 \lor x_3)$. Alternatively, we can just reuse the same variable, which is the approach we take in the code.

## Installation

To run the code, you need to have Ganak in this directory. We use version 2.4.3, which you can find on [GitHub](https://github.com/meelgroup/ganak/releases/tag/release%2F2.4.3). Make sure you name the executable `ganak`.

Otherwise, install the packages in `requirements.txt`.

## Example

In `example.py`, one can specify a circuit and get the desired amplitude. For small numbers of qubits, one can check against the actual state vector using Qiskit.

## Files

`generate_circuits.py`: Creates a new directory `/Data/` and saves brickwork random quantum circuits as QASM files.

`convert_circuits.py`: Takes a quantum circuit in QASM format, loads it as a Qiskit circuit, and then defines the corresponding CNF formula in [DIMACS format](https://jix.github.io/varisat/manual/0.2.0/formats/dimacs.html) as a string. There are also helper functions for writing the CNF formula to `.cnf` files.

`call_ganak.py`: Calls Ganak through Python using `subprocess`. Also, there's a function which extracts the result from Ganak.


## To Do

Here's a list of tasks left to do:
- I (Jeremy) used ChatGPT to write a bunch of helper functions that I haven't thoroughly tested: `generate_brickwork_circuit.py`, `get_unique_block`, `computational_basis_state_to_formula`, `save_and_return_dimacs_with_weights`, `compute_amplitude_z_array`, and all of `call_ganak.py`. We need to test them.
- We need to also test the functions I wrote.
- For `circuit_to_cnf`, I reshape the unitary matrix and then do `reshaped_unitary[bits[::-1]]`. Can you (Julien) understand the reshaping process of the unitary and why we need to reverse the order of the bits? I think it has something to do with Qiskit's [little-endian ordering](https://docs.quantum.ibm.com/guides/bit-ordering).
- `circuit_to_cnf` currently assumes that there is only one QuantumRegister object in Qiskit that labels the qubits (I call for their indices). Does the function work for circuits with multiple QuantumRegisters? Is there an easy generalization, or should we indicate in our code that we *must* have one register?
- Think about how to save data. Currently, I use `Data/circuits/n()/` to store circuits, and then in `convert_circuits.py` we can save the CNF files. Alternatively, `example.py` shows how we can use temporary CNF files without saving them (since we have the underlying circuit). What is best? And which data should we save? I'm thinking: circuit, amplitude, execution time.
- How do we calculate execution time? In `call_ganak.py`, I take the output from Ganak, but it's unclear to me if this is the right one to use. Uncomment line 52 of `example.py` to see the output from Ganak.
- Does `write_cnf_to_tempfile` have any memory issues? I think it deletes the file after, but I haven't thoroughly tested it.
- Write a script to batch launch many circuits, storing the output from Ganak into files.
- Write a script that takes the results and plots them according to time and number of qubits.
- Test out the code against Google's Sycamore circuits, which you can find in `.tar.gz` files [here](https://datadryad.org/dataset/doi:10.5061/dryad.k6t1rj8).
- In general, look for any code improvements, speedups, and documentation improvements.
