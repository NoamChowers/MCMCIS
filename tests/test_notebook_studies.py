import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.experiments.notebook_studies import (
    BetaSweepStudyConfig,
    CrossMethodStudyConfig,
    LoadedScenario,
    MCMCWorkflowConfig,
    SAMCWorkflowConfig,
    build_beta_workflow,
    run_beta_checkpoint_study,
    run_cross_method_study,
)
from perm_pval.stats.two_sample import difference_in_means


def _small_right_tail_scenario() -> LoadedScenario:
    x = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 7.0], dtype=float)
    y_obs = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )
    exact = BruteForceExactSolver(problem, max_permutations=1000).compute()
    return LoadedScenario(
        key="toy",
        description="Small right-tail toy problem",
        problem=problem,
        exact_p=float(exact.p_value),
        exact_tail_hits=int(exact.tail_hits),
        exact_n_perm=int(exact.n_permutations),
        exact_method="BruteForceExactSolver",
        notes="",
        extra={},
    )


def _small_mcmc_cfg() -> MCMCWorkflowConfig:
    return MCMCWorkflowConfig(
        pilot_samples=40,
        tune_steps=30,
        tune_burn_in_fraction=0.2,
        tune_thin=1,
        tune_max_bracket=4,
        tune_max_bisect=4,
        local_scan_total_steps=20,
        local_scan_chains=1,
        local_scan_thin=1,
        chains=1,
        thin=1,
    )


def _small_samc_cfg() -> SAMCWorkflowConfig:
    return SAMCWorkflowConfig(
        n_bins=5,
        trace_every=5,
        lambda_min_pilot=20,
    )


def test_run_cross_method_study_emits_rows_for_all_methods_and_checkpoints():
    scenario = _small_right_tail_scenario()
    cross_cfg = CrossMethodStudyConfig(
        estimation_points=(500, 800),
        repeats=1,
        iid_density_samples=20,
        base_seed=123,
    )
    study = run_cross_method_study(
        scenario,
        cross_cfg,
        _small_mcmc_cfg(),
        _small_samc_cfg(),
    )

    assert len(study["records"]) == 2 * 3
    assert sorted({row["checkpoint"] for row in study["records"]}) == [500, 800]
    assert sorted({row["method"] for row in study["records"]}) == ["iid", "mcmc_is", "samc"]
    assert int(study["mcmc_beta_selection_budget"]) > 0
    mcmc_rows = [row for row in study["records"] if row["method"] == "mcmc_is"]
    assert all(int(row["mcmc_chain_budget"]) < int(row["checkpoint"]) for row in mcmc_rows)
    assert all(int(row["beta_selection_budget"]) == int(study["mcmc_beta_selection_budget"]) for row in mcmc_rows)


def test_run_beta_checkpoint_study_emits_rows_for_each_checkpoint():
    scenario = _small_right_tail_scenario()
    mcmc_cfg = _small_mcmc_cfg()
    beta_workflow = build_beta_workflow(
        scenario.problem,
        scenario.exact_p,
        mcmc_cfg,
        seed=321,
    )
    beta_cfg = BetaSweepStudyConfig(
        estimation_points=(20, 40),
        repeats=1,
        beta_multipliers=(1.0,),
        chains=1,
        thin=1,
        base_seed=999,
    )
    study = run_beta_checkpoint_study(
        scenario.problem,
        scenario.exact_p,
        beta_center=float(beta_workflow["beta_used"]),
        sigma_t=float(beta_workflow["sigma_t"]),
        beta_cfg=beta_cfg,
    )

    assert len(study["records"]) == 2
    assert sorted({row["checkpoint"] for row in study["records"]}) == [20, 40]
