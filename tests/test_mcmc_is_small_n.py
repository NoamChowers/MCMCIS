import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.methods.mcmc_is import (
    hard_step_beta_for_target_tail_mass,
    hard_step_r_for_target_tail_mass,
    run_hard_step_mcmc_is,
    run_mcmc_is,
)
from perm_pval.stats.two_sample import difference_in_means


def _build_problem() -> PermutationTestProblem:
    x = np.arange(10, dtype=float)
    y_obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    return PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )


def test_mcmc_is_matches_exact_small_n():
    problem = _build_problem()
    exact = BruteForceExactSolver(problem, max_permutations=1_000).compute()
    result = run_mcmc_is(
        problem,
        beta=2.0,
        n_steps=30_000,
        burn_in=5_000,
        thin=5,
        n_chains=2,
        seed=321,
        init="random",
    )
    assert abs(result.estimate - exact.p_value) < 0.01
    assert result.ess > 5.0
    assert result.snis_mcse_obm is not None
    assert np.isfinite(result.snis_mcse_obm)
    assert result.snis_mcse_obm >= 0.0


def test_swap_proposal_preserves_group_sizes():
    problem = _build_problem()
    rng = np.random.default_rng(9)
    y = problem.sample_uniform_labels(rng)
    for _ in range(500):
        move = problem.propose_local_move(y, rng)
        y = problem.apply_move(y, move, in_place=False)
        assert np.all((y == 0) | (y == 1))
        assert int(np.sum(y)) == problem.n_treated


def test_mcmc_is_seed_reproducibility():
    problem = _build_problem()
    r1 = run_mcmc_is(
        problem,
        beta=1.5,
        n_steps=12_000,
        burn_in=2_000,
        thin=4,
        n_chains=2,
        seed=777,
        init="random",
    )
    r2 = run_mcmc_is(
        problem,
        beta=1.5,
        n_steps=12_000,
        burn_in=2_000,
        thin=4,
        n_chains=2,
        seed=777,
        init="random",
    )
    assert r1.estimate == r2.estimate
    assert r1.snis_mcse_obm == r2.snis_mcse_obm
    assert np.array_equal(r1.t_samples, r2.t_samples)
    assert np.array_equal(r1.log_weights, r2.log_weights)


def test_mcmc_is_variance_toggle():
    problem = _build_problem()
    r = run_mcmc_is(
        problem,
        beta=1.5,
        n_steps=8_000,
        burn_in=1_000,
        thin=4,
        n_chains=2,
        seed=888,
        init="random",
        estimate_variance=False,
    )
    assert r.snis_variance_obm is None
    assert r.snis_mcse_obm is None
    assert r.obm_chain_batch_sizes == []
    assert r.obm_chain_long_run_variances == []


def test_mcmc_is_requires_right_tail_for_current_tilt():
    x = np.arange(8, dtype=float)
    y_obs = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="left",
    )
    raised = False
    try:
        run_mcmc_is(problem, beta=1.0, n_steps=2_000, burn_in=200, thin=2, seed=1)
    except NotImplementedError:
        raised = True
    assert raised


def test_mcmc_is_step_tilt_mode_runs_and_is_reproducible():
    problem = _build_problem()
    r1 = run_mcmc_is(
        problem,
        beta=1.2,
        n_steps=10_000,
        burn_in=2_000,
        thin=4,
        n_chains=2,
        seed=4242,
        init="random",
        tilt_mode="step",
    )
    r2 = run_mcmc_is(
        problem,
        beta=1.2,
        n_steps=10_000,
        burn_in=2_000,
        thin=4,
        n_chains=2,
        seed=4242,
        init="random",
        tilt_mode="step",
    )
    assert r1.tilt_mode == "step"
    assert 0.0 <= r1.estimate <= 1.0
    assert r1.estimate == r2.estimate
    assert np.array_equal(r1.log_weights, r2.log_weights)


def test_hard_step_formula_calibrates_target_tail_mass():
    p0 = 0.01
    q = 0.20
    r = hard_step_r_for_target_tail_mass(p0, q)
    achieved = r * p0 / (1.0 - p0 + r * p0)

    assert np.isclose(achieved, q, rtol=1e-15, atol=0.0)
    assert np.isclose(hard_step_beta_for_target_tail_mass(p0, q), np.log(r), rtol=1e-15, atol=0.0)


def test_run_hard_step_mcmc_is_wraps_step_tilt():
    problem = _build_problem()
    p0 = 0.05
    q = 0.30
    beta = hard_step_beta_for_target_tail_mass(p0, q)
    hard = run_hard_step_mcmc_is(
        problem,
        p0=p0,
        q=q,
        n_steps=6_000,
        burn_in=1_000,
        thin=5,
        n_chains=2,
        seed=5151,
        proposal_size=1,
    )
    direct = run_mcmc_is(
        problem,
        beta=beta,
        sigma_t=1.0,
        n_steps=6_000,
        burn_in=1_000,
        thin=5,
        n_chains=2,
        seed=5151,
        init="random",
        tilt_mode="step",
        proposal_size=1,
    )

    assert hard.tilt_mode == "step"
    assert np.isclose(hard.beta, beta)
    assert hard.estimate == direct.estimate
    assert np.array_equal(hard.log_weights, direct.log_weights)
