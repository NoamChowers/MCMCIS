import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.methods.random_sampling import run_random_sampling
from perm_pval.stats.two_sample import difference_in_means


def test_random_sampling_matches_exact_small_n():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_obs = np.array([0, 0, 1, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )

    exact = BruteForceExactSolver(problem, max_permutations=100).compute()
    rs = run_random_sampling(problem, n_samples=30_000, seed=11)
    assert abs(rs.estimate - exact.p_value) < 0.02


def test_random_sampling_seed_reproducibility():
    x = np.array([-1.0, -0.2, 0.3, 0.9, 1.2, 1.5], dtype=float)
    y_obs = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )
    r1 = run_random_sampling(problem, n_samples=5_000, seed=123)
    r2 = run_random_sampling(problem, n_samples=5_000, seed=123)
    assert r1.tail_hits == r2.tail_hits
    assert r1.estimate == r2.estimate
