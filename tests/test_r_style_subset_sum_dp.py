import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.exact.linear_statistic_dp import LinearStatisticDPSolver
from perm_pval.exact.r_style_subset_sum_dp import perm_exact_pval_diff_r_style
from perm_pval.stats.two_sample import difference_in_means


def _treated_sum_stat(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(np.asarray(x, dtype=float), np.asarray(y, dtype=float)))


def test_r_style_subset_sum_dp_matches_bruteforce_sum_stat():
    x = np.array([0, 1, 2, 3], dtype=int)
    y = np.array([1, 1, 2, 4], dtype=int)
    xy = np.concatenate([x, y]).astype(float)
    y_obs = np.array([1] * x.size + [0] * y.size, dtype=np.int8)

    problem = PermutationTestProblem(
        X=xy,
        y_obs=y_obs,
        statistic=_treated_sum_stat,
        tail="left",
    )

    brute = BruteForceExactSolver(problem, max_permutations=100_000).compute()
    p_r, tail_r, nperm_r = perm_exact_pval_diff_r_style(x, y)

    assert nperm_r == brute.n_permutations
    assert tail_r == brute.tail_hits
    assert np.isclose(p_r, brute.p_value, atol=1e-15)


def test_r_style_subset_sum_dp_matches_linear_stat_difference_in_means():
    x = np.array([0, 1, 2, 3, 4], dtype=int)
    y = np.array([2, 3, 4, 5, 6], dtype=int)
    xy = np.concatenate([x, y]).astype(float)
    y_obs = np.array([1] * x.size + [0] * y.size, dtype=np.int8)

    problem = PermutationTestProblem(
        X=xy,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="left",
    )
    dp_linear = LinearStatisticDPSolver.from_difference_in_means(
        problem,
        score_scale=1,
    ).compute()

    p_r, tail_r, nperm_r = perm_exact_pval_diff_r_style(x, y)
    assert nperm_r == dp_linear.n_permutations
    assert tail_r == dp_linear.tail_hits
    assert np.isclose(p_r, dp_linear.p_value, atol=1e-15)
