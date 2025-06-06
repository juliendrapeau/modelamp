from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit


class VariableManager:
    """
    Manages CNF variable allocation and mapping for qubits during conversion from quantum circuits to CNF. Unique CNF variables are allocated for each qubit, and a mapping is maintained to track which output CNF variable corresponds to which qubit index.

    Attributes
    ----------
    var_counter: int
        Counter for allocating unique CNF variables, starting from 1 as per DIMACS CNF standard.
    io_map: Dict[int, int]
        Maps qubit indices to their corresponding output CNF variable indices. This allows for easy retrieval of output CNF variables associated with specific qubits.
    """

    def __init__(self):
        self.var_counter = 1  # DIMACS CNF variable index starts at 1
        self.io_map: Dict[int, int] = (
            {}
        )  # Maps qubit index -> current output CNF variable

    def reset(self):
        """
        Reset the variable manager to its initial state, clearing the variable counter and the I/O map.
        """
        self.var_counter = 1
        self.io_map.clear()

    def allocate_unique_vars(self, n: int) -> List[int]:
        """
        Allocate a list of n unique CNF variables, starting from the current variable counter.

        Parameters
        ----------
        n: int
            The number of unique CNF variables to allocate.

        Returns
        -------
        vars: List[int]
            A list of n unique CNF variable indices, starting from the current variable counter.
        """
        vars = list(range(self.var_counter, self.var_counter + n))
        self.var_counter += n
        return vars

    def update_io_map(self, qubit_indices: List[int], variables: List[int]):
        """
        Update the I/O map with the given qubit indices and their corresponding CNF variables.

        Parameters
        ----------
        qubit_indices: List[int]
            The indices of the qubits to map.
        variables: List[int]
            The CNF variables to associate with the qubit indices.
        """
        for q, v in zip(qubit_indices, variables):
            self.io_map[q] = v

    def get_output_vars_from_map(self, qubit_indices: List[int]) -> List[int]:
        """
        Retrieve the CNF variables associated with the given qubits.

        Parameters
        ----------
        qubit_indices: List[int]
            The indices of the qubits for which to retrieve the CNF variables.
        Returns
        -------
        List[int]
            A list of CNF variable indices corresponding to the provided qubit indices.
        """
        try:
            return [self.io_map[q] for q in qubit_indices]
        except KeyError as e:
            raise ValueError(f"Qubit index {e.args[0]} not found in mapping.") from None


class CircuitToCNFConverter:
    """
    Converts a Qiskit quantum circuit into CNF clauses with complex weights for PWMCC solvers. This class handles the conversion of quantum gates into CNF format, including general unitary gates and CNOT gates, and encodes initial and final states as CNF clauses.

    Attributes
    ----------
    var_mgr: VariableManager
    clauses: List[List[int]]
        List of CNF clauses, where each clause is a list of literals (CNF variables).
    weights: Dict[int, complex]
        Dictionary mapping CNF variable literals to their complex weights, representing the amplitude contributions of each clause.
    """

    def __init__(self):
        self.var_mgr = VariableManager()
        self.clauses: List[List[int]] = []
        self.weights: Dict[int, complex] = {}

    @property
    def num_vars(self) -> int:
        """
        Get the total number of CNF variables allocated.
        """
        return self.var_mgr.var_counter - 1

    @property
    def num_clauses(self) -> int:
        """
        Get the total number of CNF clauses generated.
        """
        return len(self.clauses)

    def _reset(self):
        """
        Internal reset for processing a new circuit.
        """
        self.var_mgr.reset()
        self.clauses.clear()
        self.weights.clear()

    def _add_basis_state(self, state: np.ndarray, variables: List[int]):
        """
        Encode a basis state |x⟩ or ⟨y| into unit CNF clauses: [±v1], [±v2], ...
        Each qubit is set to True (v) or False (-v) based on the bit value.
        """

        if not ((state == 0) | (state == 1)).all():
            raise ValueError("State vector must consist of 0s and 1s only.")

        self.clauses.extend([[v if bit else -v] for bit, v in zip(state, variables)])

    def _add_general_unitary(self, unitary: np.ndarray, qubit_indices: List[int]):
        """
        Encode a general unitary matrix and convert it into CNF clauses. For a k-qubit gate, this method will create 2^(2k) clauses, each representing a possible input-output mapping of the gate. The method constructs clauses in the form:
            [-internal_var, ±x1, ±x2, ..., ±x_{2k}]
        where:
        - `internal_var` is a unique variable representing the input-output mapping.
        - `±xi` are the input/output variables for the qubits, depending on the bit value (0 or 1).

        Parameters:
        ----------
        unitary: np.ndarray
            The unitary matrix to encode, expected to be of shape (2^k, 2^k) for a k-qubit gate.
        qubit_indices: List[int]
            The indices of the qubits that this unitary acts on. The length of this list determines the number of qubits (k) and thus the shape of the unitary matrix.
        """

        k = len(qubit_indices)  # Number of qubits this gate acts on

        if unitary.shape != (2**k, 2**k):
            raise ValueError(
                f"Expected shape {(2**k, 2**k)} for a {k}-qubit gate, got {unitary.shape}."
            )

        # Get input variables for the unitary gate
        input_vars = self.var_mgr.get_output_vars_from_map(qubit_indices)

        # Allocate unique variables for output and internal states
        output_vars = self.var_mgr.allocate_unique_vars(k)
        internal_vars = self.var_mgr.allocate_unique_vars(2 ** (2 * k))

        # Update variable manager with new output variables
        self.var_mgr.update_io_map(qubit_indices, output_vars)

        io_vars = input_vars + output_vars

        # Reshape unitary into a 2k-dimensional tensor
        tensor = unitary.reshape([2] * (2 * k))

        # Iterate over all possible input combinations (2^k for inputs, 2^k for outputs)
        for bits, internal in zip(product([0, 1], repeat=2 * k), internal_vars):
            literals = [-v if b else v for b, v in zip(bits, io_vars)]

            self.clauses.append([-internal] + literals)
            # Store weight (note reversed bit order for qiskit indexing)
            self.weights[-internal] = tensor[tuple(bits[::-1])]

    def _add_cnot_gate(self, qubit_indices: List[int]):
        """
        Encode a CNOT gate as CNF clauses.

        Parameters:
        ----------
        qubit_indices: List[int]
            The indices of the control and target qubits for the CNOT gate.
        """

        if len(qubit_indices) != 2:
            raise ValueError(
                "CNOT gate requires exactly two qubit indices: [control, target]."
            )

        # Allocate variables for control and target qubits
        input_vars = self.var_mgr.get_output_vars_from_map(qubit_indices)

        output_vars = self.var_mgr.allocate_unique_vars(2)
        internal_var = self.var_mgr.allocate_unique_vars(1)

        # Update variable manager with new output variables
        self.var_mgr.update_io_map(qubit_indices, output_vars)

        io_vars = input_vars + output_vars

        # Set the tensor values for CNOT operation
        bits_list = list(product([0, 1], repeat=4))
        bits_list.remove((0, 0, 0, 0))
        bits_list.remove((0, 1, 0, 1))
        bits_list.remove((1, 0, 1, 1))
        bits_list.remove((1, 1, 1, 0))

        self.weights[-internal_var[0]] = complex(0)

        # Iterate over all possible input combinations
        for bits in bits_list:
            literals = [-v if b else v for b, v in zip(bits, io_vars)]

            self.clauses.append([-internal_var[0]] + literals)

    def _dispatch_gate(self, instr):
        """
        Dispatch a gate instruction to the appropriate conversion method, which encodes the gate as CNF clauses depending on its type.

        Parameters:
        ----------
        instr: Instruction
            The gate instruction to process, which contains the operation and qubits.
        """

        name = instr.operation.name.lower()
        qubit_indices = [q._index for q in instr.qubits]
        matrix = instr.matrix

        if name == "cx":
            self._add_cnot_gate(qubit_indices)
        else:
            self._add_general_unitary(matrix, qubit_indices)

    def convert(
        self,
        circuit: QuantumCircuit,
        initial_state: np.ndarray,
        final_state: np.ndarray,
    ) -> Tuple[List[List[int]], Dict[int, complex]]:
        """
        Convert a Qiskit quantum circuit into CNF clauses with complex weights.
        This method processes the circuit's gates and encodes the initial and final states into CNF format, suitable for solving with PWMCC solvers.

        Parameters:
        ----------
        circuit: QuantumCircuit
            The quantum circuit to convert.
        initial_state: np.ndarray
            The initial state of the circuit as a binary vector (0s and 1s).
        final_state: np.ndarray
            The final state of the circuit as a binary vector (0s and 1s).

        Returns:
        -------
        Tuple[List[List[int]], Dict[int, complex]]
            A tuple containing:
            - clauses: List of CNF clauses, where each clause is a list of literals.
            - weigths: Dictionary mapping CNF variable literals to their complex weights.
        """

        self._reset()

        n = circuit.num_qubits
        if len(initial_state) != n or len(final_state) != n:
            raise ValueError("Initial and final states must match circuit width.")

        # Assign variables for initial state
        input_vars = self.var_mgr.allocate_unique_vars(n)
        # Update the variable manager with input qubit indices
        self.var_mgr.update_io_map(list(range(n)), input_vars)

        # Add clauses for the initial state
        self._add_basis_state(initial_state, input_vars)

        # Convert each gate
        for instr in circuit.data:
            self._dispatch_gate(instr)

        # Get output variables for final state
        output_vars = self.var_mgr.get_output_vars_from_map(list(range(n)))

        # Add clauses for the final state
        self._add_basis_state(final_state, output_vars)

        return self.clauses, self.weights

    def export_dimacs(self, filename: Optional[str] = None) -> str:
        """
        Export the CNF clauses and weights to a DIMACS format string or save to a file.

        Parameters
        ----------
        filename: Optional[str]
            If provided, the CNF will be saved to this file. If None, the CNF string is returned.
        Returns
        -------
        str
            A string representing the CNF in DIMACS format, or None if saved to a file.
        """

        if not self.clauses:
            raise ValueError("No clauses generated. Run convert() first.")

        num_vars = self.num_vars
        num_clauses = self.num_clauses

        lines = [f"c t pwmc", f"p cnf {num_vars} {num_clauses}"]

        for lit, w in self.weights.items():
            if not isinstance(w, complex):
                raise TypeError(
                    f"Expected complex weight for literal {lit}, got {type(w)}"
                )
            r, i = w.real, w.imag
            lines.append(f"c p weight {lit} {r} {i} 0")
            lines.append(f"c p weight {-lit} {1 - r} {-i} 0")

        for clause in self.clauses:
            lines.append(" ".join(map(str, clause)) + " 0")

        output = "\n".join(lines)

        if filename:
            with open(filename, "w") as f:
                f.write(output + "\n")

        return output
