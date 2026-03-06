import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.exact.linear_statistic_dp import LinearStatisticDPSolver
from perm_pval.exact.rank_sum_dp import RankSumDPSolver
from perm_pval.stats.ranks import mann_whitney_u
from perm_pval.stats.two_sample import difference_in_means


def test_rank_sum_dp_matches_bruteforce_u():
    x = np.array([1.2, 0.4, 3.1, 2.2, 5.4, 4.8], dtype=float)
    y_obs = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=mann_whitney_u,
        tail="right",
    )

    brute = BruteForceExactSolver(problem, max_permutations=1_000).compute()
    dp = RankSumDPSolver(problem, statistic_type="u").compute()
    assert dp.tail_hits == brute.tail_hits
    assert dp.n_permutations == brute.n_permutations
    assert np.isclose(dp.p_value, brute.p_value, atol=1e-15)


def test_rank_sum_dp_rejects_ties():
    x = np.array([1.0, 1.0, 2.0, 3.0], dtype=float)
    y_obs = np.array([0, 1, 0, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=mann_whitney_u,
        tail="right",
    )
    raised = False
    try:
        RankSumDPSolver(problem).compute()
    except ValueError:
        raised = True
    assert raised


def test_linear_statistic_dp_matches_bruteforce_difference_in_means():
    x = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0], dtype=float)
    y_obs = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )

    brute = BruteForceExactSolver(problem, max_permutations=10_000).compute()
    dp = LinearStatisticDPSolver.from_difference_in_means(problem, score_scale=1).compute()
    assert dp.tail_hits == brute.tail_hits
    assert dp.n_permutations == brute.n_permutations
    assert np.isclose(dp.p_value, brute.p_value, atol=1e-15)


def test_linear_statistic_dp_with_scaled_scores():
    x = np.array([0.1, 0.2, 0.4, 1.1, 1.5, 1.7], dtype=float)
    y_obs = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )

    brute = BruteForceExactSolver(problem, max_permutations=2_000).compute()
    dp = LinearStatisticDPSolver.from_difference_in_means(problem, score_scale=10).compute()
    assert dp.tail_hits == brute.tail_hits
    assert np.isclose(dp.p_value, brute.p_value, atol=1e-15)
