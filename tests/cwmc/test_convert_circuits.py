"""
Verify if the number of variables and clauses in the CNF file mathches the expected values.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np


def test_simple_circuit():
    from modelamp.cwmc.convert_circuits import CircuitToCNFConverter
    from qiskit import QuantumCircuit
    from modelamp.cwmc.convert_circuits_good import CircuitToCNFConverterGood
    num_qubits = 2
    # Create a simple quantum circuit
    qc = QuantumCircuit(num_qubits)
    # qc.cx(0,1)
    # qc.x(1)
    qc.x(0)
    # qc.x(0)
    
    initial_state = np.ones(qc.num_qubits, dtype=int)
    final_state = np.zeros(qc.num_qubits, dtype=int)
    
    # Convert the circuit to CNF
    converter = CircuitToCNFConverter()
    clauses, weights = converter.convert(circuit=qc, initial_state=initial_state, final_state=final_state)
    print("Clauses:", clauses)
    print("Weights:", weights)
    
    converter_good = CircuitToCNFConverterGood()
    clauses_good, weights_good = converter_good.convert(circuit=qc, initial_state=initial_state, final_state=final_state)
    print("Clauses Good:", clauses_good)
    print("Weights Good:", weights_good)

    # # Check the number of clauses and variables
    # assert len(clauses) == 1, "Expected 1 clause for a simple circuit"
    # assert len(weight) == 2, "Expected 2 variables for a simple circuit (1 qubit)"
    # assert weight[-1] == 1, "Expected the weight of the single clause to be 1"
    # assert weight[1] == 1, "Expected the weight of the single clause to be 1"
    # assert weight[0] == 0, "Expected the weight of the initial state variable to be 0"
    # assert weight[2] == 0, "Expected the weight of the final state variable to be 0"
    

if __name__ == "__main__":
    test_simple_circuit()