"""
Purpose: Given a QASM file of a circuit, convert it into a weighted model counting CNF instance.

Date created: 2025-04-11
"""

import numpy as np
from itertools import product


def get_unique_block(n: int) -> list:
    """
    Purpose: Generate n unique, sequential integers, starting
             from global_var_counter.

    Input:
        - n (integer): The number of unique integers.
    Output:
        - Returns the next n unique integers using a global counter.
    """
    global global_var_counter
    block = list(range(global_var_counter, global_var_counter + n))
    global_var_counter += n
    return block


def computational_basis_state_to_formula(state, variables):
    """
    Purpose: Given a basis state, compute the corresponding CNF formula.
    Input:
        - state (array): A binary vector.
        - variables (array): The variables for the formula.
    Output:
        - formula (list of lists): The CNF formula encoding the state.
    """
    z = np.array(state)
    assert np.all((z == 0) | (z == 1))
    return [[int(i)] for i in (-1) ** (z + 1) * variables]


def circuit_to_cnf(circuit, initial_state, final_state):
    """
    Purpose: Convert a Qiskit ciruit into a CNF string.

    Input:
        - circuit (Qiskit): A circuit description.
        - initial_state (array): The computational basis starting state.
        - final_state (array): The computational basis final state.
    Output:
        - cnf_constraints (list of lists of integers): All the CNF
        constraints for the circuit.
        - weights (list of lists of integers): The weights for literals
        in the CNF. Note that literals which don't appear have weight 1.
        Also, the negation of a literal will have weight 1 - w.
    """
    global global_var_counter
    global_var_counter = 1

    n = circuit.num_qubits

    external_map = (
        {}
    )  # Maps qubits 1 to n to the external variable that occurs after applying any gate
    weights = {}  # Weights for variables. If variable isn't present, it's 1 by default.
    block = get_unique_block(n)
    for i, index in enumerate(block):
        external_map[i + 1] = index

    starting_state_constraints = computational_basis_state_to_formula(
        state=initial_state, variables=np.array(block)
    )

    cnf_constraints = []
    for gate_instruction in circuit.data:
        unitary = gate_instruction.matrix
        qubits = gate_instruction.qubits
        indices = [q._index for q in qubits]
        k = len(qubits)
        external_variables = [external_map[i + 1] for i in indices]

        # Get external output variables
        external_output_variables = get_unique_block(k)
        external_variables.extend(external_output_variables)

        # Update external map
        for e, i in enumerate(indices):
            external_map[i + 1] = external_output_variables[e]

        reshaped_unitary = unitary.reshape((2,) * (2 * k))
        internal_variables = get_unique_block(2 ** (2 * k))
        # print("internal_variables: ", internal_variables)
        for bits, internal_variable in zip(
            product([0, 1], repeat=2 * k), internal_variables
        ):
            literals = (-1) ** np.array(bits) * np.array(external_variables)
            clause = [-internal_variable]
            weights[-internal_variable] = reshaped_unitary[
                bits[::-1]
            ]  # Note: This reordering of bits is important! Something to do with ordering in Qiskit
            clause.extend([int(i) for i in literals])
            cnf_constraints.append(clause)

    final_external_variables = []
    for i in range(1, n + 1):
        final_external_variables.append(external_map[i])

    final_external_variables = np.array(final_external_variables)
    # print("Final external: ", final_external_variables)
    final_state_constraints = computational_basis_state_to_formula(
        state=final_state, variables=np.array(final_external_variables)
    )

    # Add initial and final state constraints
    cnf_constraints.extend(starting_state_constraints)
    cnf_constraints.extend(final_state_constraints)

    return cnf_constraints, weights


def save_and_return_dimacs_with_weights(clauses, weights, file_path=None):
    """
    Purpose: Generates a DIMACS CNF string with complex weights and optionally
            saves it to a file.

    Inputs:
        - clauses (list of lists of integers): Each inner list is a clause.
        - weights (dictionary): Maps integer literals to complex weights.
        - file_path (string): Optional path to output file.
    Output:
        dimacs_str (string): A string representing the DIMACS CNF content.
    """
    num_vars = max(abs(lit) for clause in clauses for lit in clause)
    num_clauses = len(clauses)

    lines = [f"c t pwmc", f"p cnf {num_vars} {num_clauses}"]

    for lit, w in weights.items():
        if not isinstance(w, complex):
            raise ValueError(f"Weight for literal {lit} must be complex.")
        r, i = w.real, w.imag
        lines.append(f"c p weight {lit} {r} {i} 0")
        lines.append(f"c p weight {-lit} {1 - r} {-i} 0")

    for clause in clauses:
        clause_line = " ".join(map(str, clause)) + " 0"
        lines.append(clause_line)

    dimacs_str = "\n".join(lines)

    if file_path:
        with open(file_path, "w") as f:
            f.write(dimacs_str + "\n")

    return dimacs_str
