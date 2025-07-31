import tempfile

import numpy as np
from pytest import mark

from modelamp.cwmc.call_ganak import GanakSolver


@mark.parametrize(
    "num_var, num_clauses",
    np.random.randint(10, 20, size=[10, 2]),
)
def test_compare_ganak_pysat(num_var, num_clauses):
    """
    Compare the model count of a random CNF formula using Ganak and PySAT.
    This test generates a random CNF formula, saves it to a temporary file, and then uses both Ganak and PySAT to count the models. The counts should be equal.
    """

    from pysat.formula import CNF
    from pysat.solvers import Glucose4

    rng = np.random.default_rng()

    # Generate a random CNF formula
    cnf = CNF()
    for _ in range(num_clauses):
        clause = [
            int(rng.integers(1, num_var) * (-1 if rng.random() < 0.5 else 1))
            for _ in range(rng.integers(1, 3))
        ]
        cnf.append(clause)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        cnf.to_file(temp_file.name)

    ganak_solver = GanakSolver()
    pysat_solver = Glucose4()

    # Solve with PySAT
    count_pysat = 0
    pysat_solver.append_formula(cnf)
    for sol in pysat_solver.enum_models():
        count_pysat += 1
    pysat_solver.delete()

    # Solve with Ganak
    count_ganak = ganak_solver.solve(temp_file.name)[0]

    assert np.isclose(count_ganak.real, count_pysat)
