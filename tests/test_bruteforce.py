import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.stats.two_sample import difference_in_means


def test_bruteforce_exact_known_value():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_obs = np.array([0, 0, 1, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )
    result = BruteForceExactSolver(problem, max_permutations=100).compute()
    assert result.n_permutations == 6
    assert result.tail_hits == 1
    assert np.isclose(result.p_value, 1.0 / 6.0)


def test_bruteforce_guardrail_raises():
    x = np.linspace(0.0, 1.0, 12)
    y_obs = np.array([1] * 6 + [0] * 6, dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )
    solver = BruteForceExactSolver(problem, max_permutations=100)
    try:
        solver.compute()
        raised = False
    except ValueError:
        raised = True
    assert raised
