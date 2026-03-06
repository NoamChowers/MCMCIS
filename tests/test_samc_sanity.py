import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver, iter_fixed_group_labelings
from perm_pval.methods.samc import run_samc
from perm_pval.stats.two_sample import difference_in_means


def test_samc_visitation_is_approximately_flat():
    x = np.arange(10, dtype=float)
    y_obs = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )

    all_stats = []
    for y in iter_fixed_group_labelings(problem.n, problem.n_treated):
        all_stats.append(problem.compute_stat(y))
    all_stats = np.sort(np.asarray(all_stats, dtype=float))

    samc = run_samc(
        problem,
        n_steps=12_000,
        burn_in=2_000,
        n_bins=6,
        lambda_min=float(np.min(all_stats)),
        seed=42,
        t0=500.0,
        init="random",
        trace_every=20,
    )

    target = np.full(samc.visit_counts.size, 1.0 / samc.visit_counts.size)
    assert np.max(np.abs(samc.visitation_frequency - target)) < 0.20
    assert samc.pvalue_estimator == "samc_paper_eq_3_2"
    assert samc.tail_bin_index == samc.visit_counts.size - 1
    assert samc.convergence_reached is True

    exact = BruteForceExactSolver(problem, max_permutations=1_000).compute()
    assert abs(samc.estimate - exact.p_value) < 0.08
