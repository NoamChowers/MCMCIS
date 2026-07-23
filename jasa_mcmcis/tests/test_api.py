from __future__ import annotations

import numpy as np

from jasa_mcmcis import (
    ABOVE_THRESHOLD_SCENARIO_KEYS,
    ARTICLE_SCENARIO_KEYS,
    CROSS_METHOD_SCENARIO_KEYS,
    GWAS_ABOVE_THRESHOLD_SCENARIO_KEYS,
    GWAS_NEAR_THRESHOLD_SCENARIO_KEYS,
    HEP_ABOVE_THRESHOLD_SCENARIO_KEYS,
    HEP_NEAR_THRESHOLD_SCENARIO_KEYS,
    LinearStatisticDPSolver,
    NEAR_THRESHOLD_SCENARIO_KEYS,
    PermutationTestProblem,
    available_scenarios,
    difference_in_means,
    load_scenario,
    load_scenarios,
    run_mcmc_is,
    run_samc,
)


def test_bundled_scenarios_match_the_cross_method_inventory() -> None:
    assert available_scenarios() == CROSS_METHOD_SCENARIO_KEYS
    assert len(ARTICLE_SCENARIO_KEYS) == 6
    assert len(GWAS_NEAR_THRESHOLD_SCENARIO_KEYS) == 50
    assert len(HEP_NEAR_THRESHOLD_SCENARIO_KEYS) == 50
    assert len(GWAS_ABOVE_THRESHOLD_SCENARIO_KEYS) == 50
    assert len(HEP_ABOVE_THRESHOLD_SCENARIO_KEYS) == 50
    assert len(NEAR_THRESHOLD_SCENARIO_KEYS) == 100
    assert len(ABOVE_THRESHOLD_SCENARIO_KEYS) == 100
    scenarios = load_scenarios()
    assert [scenario.key for scenario in scenarios] == list(CROSS_METHOD_SCENARIO_KEYS)
    for scenario in scenarios:
        assert np.isclose(scenario.problem.t_obs, scenario.extra.get("observed_score", scenario.problem.t_obs)) or scenario.key.startswith("poisson_")
        assert scenario.exact_p_value > 0.0
        assert scenario.tail_hits > 0
        assert scenario.problem.tail == "right"


def test_linear_dp_matches_bundled_metadata_for_gwas_scenario() -> None:
    scenario = load_scenario("gwas_additive_score_sig_n100")
    exact = LinearStatisticDPSolver.from_scenario(scenario).compute()
    assert exact.tail_hits == scenario.tail_hits
    assert exact.n_permutations == scenario.n_permutations
    assert np.isclose(exact.p_value, scenario.exact_p_value, rtol=1e-15, atol=0.0)


def test_near_threshold_inventory_metadata_is_in_band() -> None:
    scenarios = load_scenarios(NEAR_THRESHOLD_SCENARIO_KEYS)
    assert len(scenarios) == 100

    for scenario in scenarios:
        p0 = float(scenario.extra["known_significance_threshold"])
        ratio = float(scenario.extra["p_over_p0"])
        assert np.isclose(ratio, scenario.exact_p_value / p0, rtol=0.0, atol=0.0)
        assert 0.75 <= ratio <= 0.99
        assert "near_threshold_variety" in scenario.portfolio["groups"]


def test_above_threshold_inventory_metadata_is_in_band() -> None:
    scenarios = load_scenarios(ABOVE_THRESHOLD_SCENARIO_KEYS)
    assert len(scenarios) == 100

    for scenario in scenarios:
        p0 = float(scenario.extra["known_significance_threshold"])
        ratio = float(scenario.extra["p_over_p0"])
        assert np.isclose(ratio, scenario.exact_p_value / p0, rtol=0.0, atol=0.0)
        assert 1.01 <= ratio <= 1.25
        assert "above_threshold_variety" in scenario.portfolio["groups"]


def _toy_problem() -> PermutationTestProblem:
    x = np.arange(10, dtype=float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    return PermutationTestProblem(x=x, y_obs=y, statistic=difference_in_means, tail="right")


def test_mcmcis_runs_on_toy_problem() -> None:
    problem = _toy_problem()
    result = run_mcmc_is(
        problem,
        beta=2.0,
        sigma_t=1.0,
        n_steps=4_000,
        burn_in=500,
        thin=2,
        n_chains=2,
        seed=7,
        proposal_size=1,
    )
    assert 0.0 <= result.estimate <= 1.0
    assert result.ess > 0.0
    assert result.mcse_obm is not None
    assert result.n_weighted_samples == 3500


def test_samc_runs_on_toy_problem() -> None:
    problem = _toy_problem()
    result = run_samc(
        problem,
        n_steps=4_000,
        burn_in=500,
        n_bins=6,
        lambda_min=-5.0,
        seed=8,
        proposal_size=1,
        trace_every=100,
    )
    assert 0.0 <= result.estimate <= 1.0
    assert result.tail_bin_index == result.visit_counts.size - 1
    assert result.theta_trace.shape[1] == result.visit_counts.size
