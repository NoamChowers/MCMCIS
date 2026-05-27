import numpy as np
import matplotlib.image as mpimg
import pytest

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.experiments.exact_scenarios import (
    _make_gwas_additive_score_scenario,
    _make_zero_inflated_poisson_diffmeans_scenario,
)
from perm_pval.experiments.notebook_studies import (
    BetaSweepStudyConfig,
    CrossMethodStudyConfig,
    LoadedScenario,
    MCMCWorkflowConfig,
    MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES,
    SAMCWorkflowConfig,
    build_beta_workflow,
    build_mcmc_objective_grid_candidates,
    load_beta_sweep_saved_output,
    load_cross_method_saved_output,
    regenerate_beta_sweep_plots_from_saved,
    regenerate_cross_method_plots_from_saved,
    run_beta_checkpoint_study,
    run_cross_method_study,
    run_mcmc_objective_grid_study,
    save_beta_sweep_outputs,
    save_cross_method_outputs,
    score_mcmc_objective_grid_repeat_row,
    select_mcmc_objective_grid_winners,
    summarize_mcmc_objective_grid_configs,
    load_selected_scenarios,
)
from perm_pval.methods.beta_tuning import estimate_scale_T, iid_pilot_statistics
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
        local_scan_q_multipliers=(0.01, 0.1),
        local_scan_swap_counts=(1, 2),
        local_scan_screen_total_steps=10,
        local_scan_screen_chains=1,
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


def _assert_same_png_pixels(path_a, path_b) -> None:
    img_a = mpimg.imread(path_a)
    img_b = mpimg.imread(path_b)
    assert img_a.shape == img_b.shape
    assert np.array_equal(img_a, img_b)


def _strip_runtime_fields(rows):
    out = []
    for row in rows:
        clean = dict(row)
        clean.pop("wall_time_sec", None)
        out.append(clean)
    return out


def _strip_runtime_summary_fields(rows):
    out = []
    for row in rows:
        clean = dict(row)
        clean.pop("mean_wall_time_sec", None)
        out.append(clean)
    return out


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
    if study["beta_workflow"].get("production_init_states") is not None:
        assert all(int(row["state_reused_init"]) == 1 for row in mcmc_rows)
        assert all(int(row["burn_in"]) == 0 for row in mcmc_rows)
    samc_rows = [row for row in study["records"] if row["method"] == "samc"]
    assert all("samc_estimate_no_empty_bin_correction" in row for row in samc_rows)
    assert all("samc_empty_bin_correction_delta" in row for row in samc_rows)
    assert all("samc_empty_bin_correction_ratio" in row for row in samc_rows)


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


def test_saved_cross_method_outputs_can_be_reloaded_and_replotted(tmp_path):
    scenario = _small_right_tail_scenario()
    cross_cfg = CrossMethodStudyConfig(
        estimation_points=(500, 800),
        repeats=1,
        iid_density_samples=20,
        base_seed=123,
    )
    mcmc_cfg = _small_mcmc_cfg()
    samc_cfg = _small_samc_cfg()
    study = run_cross_method_study(scenario, cross_cfg, mcmc_cfg, samc_cfg)

    out_dir = tmp_path / "cross"
    save_cross_method_outputs(
        scenario,
        study,
        output_dir=out_dir,
        cross_cfg=cross_cfg,
        mcmc_cfg=mcmc_cfg,
        samc_cfg=samc_cfg,
    )

    loaded = load_cross_method_saved_output(out_dir)
    assert loaded["metadata"]["scenario"] == scenario.key
    regen_dir = tmp_path / "cross_regen"
    regen = regenerate_cross_method_plots_from_saved(out_dir, save_dir=regen_dir)
    assert all(path.exists() for path in regen.values())
    _assert_same_png_pixels(out_dir / "cross_method_max_budget.png", regen["cross_method_max_budget"])
    _assert_same_png_pixels(out_dir / "cross_method_convergence.png", regen["cross_method_convergence"])
    _assert_same_png_pixels(out_dir / "cross_method_diagnostics.png", regen["cross_method_diagnostics"])


def test_saved_beta_outputs_can_be_reloaded_and_replotted(tmp_path):
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

    out_dir = tmp_path / "beta"
    save_beta_sweep_outputs(
        study,
        output_dir=out_dir,
        scenario_name=scenario.description,
        exact_p=scenario.exact_p,
        beta_cfg=beta_cfg,
        beta_workflow=beta_workflow,
    )

    loaded = load_beta_sweep_saved_output(out_dir)
    assert loaded["metadata"]["scenario_display"] == scenario.description
    regen_dir = tmp_path / "beta_regen"
    regen = regenerate_beta_sweep_plots_from_saved(out_dir, save_dir=regen_dir)
    assert all(path.exists() for path in regen.values())
    _assert_same_png_pixels(out_dir / "beta_max_budget.png", regen["beta_max_budget"])
    _assert_same_png_pixels(out_dir / "beta_convergence.png", regen["beta_convergence"])


def test_cross_method_parallel_matches_serial():
    scenario = _small_right_tail_scenario()
    serial_cfg = CrossMethodStudyConfig(
        estimation_points=(500, 800),
        repeats=2,
        iid_density_samples=20,
        base_seed=123,
        n_jobs=1,
    )
    parallel_cfg = CrossMethodStudyConfig(
        estimation_points=(500, 800),
        repeats=2,
        iid_density_samples=20,
        base_seed=123,
        n_jobs=2,
    )
    mcmc_cfg = _small_mcmc_cfg()
    samc_cfg = _small_samc_cfg()

    serial = run_cross_method_study(scenario, serial_cfg, mcmc_cfg, samc_cfg)
    parallel = run_cross_method_study(scenario, parallel_cfg, mcmc_cfg, samc_cfg)

    assert _strip_runtime_fields(serial["records"]) == _strip_runtime_fields(parallel["records"])
    assert _strip_runtime_summary_fields(serial["summary"]) == _strip_runtime_summary_fields(parallel["summary"])


def test_beta_study_parallel_matches_serial():
    scenario = _small_right_tail_scenario()
    mcmc_cfg = _small_mcmc_cfg()
    beta_workflow = build_beta_workflow(
        scenario.problem,
        scenario.exact_p,
        mcmc_cfg,
        seed=321,
    )
    serial_cfg = BetaSweepStudyConfig(
        estimation_points=(20, 40),
        repeats=2,
        beta_multipliers=(1.0, 1.2),
        chains=1,
        thin=1,
        base_seed=999,
        n_jobs=1,
    )
    parallel_cfg = BetaSweepStudyConfig(
        estimation_points=(20, 40),
        repeats=2,
        beta_multipliers=(1.0, 1.2),
        chains=1,
        thin=1,
        base_seed=999,
        n_jobs=2,
    )

    serial = run_beta_checkpoint_study(
        scenario.problem,
        scenario.exact_p,
        beta_center=float(beta_workflow["beta_used"]),
        sigma_t=float(beta_workflow["sigma_t"]),
        beta_cfg=serial_cfg,
    )
    parallel = run_beta_checkpoint_study(
        scenario.problem,
        scenario.exact_p,
        beta_center=float(beta_workflow["beta_used"]),
        sigma_t=float(beta_workflow["sigma_t"]),
        beta_cfg=parallel_cfg,
    )

    assert _strip_runtime_fields(serial["records"]) == _strip_runtime_fields(parallel["records"])
    assert _strip_runtime_summary_fields(serial["summary"]) == _strip_runtime_summary_fields(parallel["summary"])


def test_build_mcmc_objective_grid_candidates_retains_invalid_q_map_rows():
    scenario = _small_right_tail_scenario()
    pilot_t = iid_pilot_statistics(scenario.problem, n_samples=40, seed=123)
    sigma_t = estimate_scale_T(pilot_t, method="sd")

    candidates = build_mcmc_objective_grid_candidates(
        scenario.problem,
        pilot_t=pilot_t,
        sigma_t=sigma_t,
        p0_for_qtarget=scenario.exact_p,
        q_target=0.1,
        q_multipliers=(1e-5, 1.0),
        n_swap_pairs_values=(1, 2),
        beta_max=1e6,
        q_floor=1e-12,
    )

    assert [c["q_multiplier"] for c in candidates[:2]] == [1e-5, 1e-5]
    assert [c["n_swap_pairs"] for c in candidates[:2]] == [1, 2]
    assert len(candidates) == 4
    assert any(c["status"] == "invalid_q_map" for c in candidates)
    assert any(c["status"] == "ok" and np.isfinite(c["beta"]) and c["beta"] > 0.0 for c in candidates)


def test_score_mcmc_objective_grid_repeat_row_uses_exact_penalties():
    row = score_mcmc_objective_grid_repeat_row(
        {
            "variance_estimate": 3.0,
            "q_tilt_tail_share": 0.001,
            "q_trial": 0.01,
            "n_weighted_samples": 100,
            "weight_cv": 9.0,
            "abs_log10_error": 0.3,
        }
    )

    p_q = 1.0 + abs(np.log((0.001 + 0.01) / (0.01 + 0.01)))
    p_deg = 1.0 + 0.25 * np.log1p(9.0)

    assert row["P_q"] == pytest.approx(p_q)
    assert row["P_deg"] == pytest.approx(p_deg)
    assert row["objective_varhat"] == pytest.approx(3.0)
    assert row["objective_varhat_qmatch_soft"] == pytest.approx(3.0 * p_q)
    assert row["objective_varhat_degeneracy_soft"] == pytest.approx(3.0 * p_deg)
    assert row["objective_varhat_qmatch_degeneracy_soft"] == pytest.approx(3.0 * p_q * p_deg)


def test_objective_grid_aggregation_and_winner_tie_breaks_are_deterministic():
    rows = []
    for repeat_idx in range(2):
        base = {
            "checkpoint": 10,
            "exact_p": 1.0,
            "tail_hits": 30,
            "acceptance_rate": 0.2,
            "q_tilt_tail_share": 0.01,
            "q_trial": 0.01,
            "q_trial_raw": 0.01,
            "q_target": 0.1,
            "q_floor": 1e-12,
            "q_floor_applied": 0,
            "n_weighted_samples": 100,
            "sigma_t": 1.0,
            "proposal_size": 1,
            "status": "ok",
            "invalid_reason": None,
            "wall_time_sec": 1.0,
            "eval_excl_tuning": 10.0,
            "eval_incl_tuning": 10.0,
            "ess": 10.0,
            "weight_cv": 2.0,
            "zero_hits": 0,
            "steps_per_chain": 10,
            "burn_in": 2,
            "chains": 1,
            "thin": 1,
            "trial_budget": 10,
            "seed": 100 + repeat_idx,
            "trial_repeat": repeat_idx,
        }
        rows.append(
            score_mcmc_objective_grid_repeat_row(
                {
                    **base,
                    "config_id": "q00_s1",
                    "label": "q00_s1",
                    "q_index": 0,
                    "q_multiplier": 0.1,
                    "n_swap_pairs": 1,
                    "beta": 1.0,
                    "estimate": 1.3,
                    "squared_error": 0.09,
                    "abs_log10_error": 0.1,
                    "variance_estimate": 1.0,
                    "weight_cv": 2.0,
                }
            )
        )
        rows.append(
            score_mcmc_objective_grid_repeat_row(
                {
                    **base,
                    "config_id": "q01_s2",
                    "label": "q01_s2",
                    "q_index": 1,
                    "q_multiplier": 0.2,
                    "n_swap_pairs": 2,
                    "proposal_size": 2,
                    "beta": 2.0,
                    "estimate": 1.1,
                    "squared_error": 0.01,
                    "abs_log10_error": 0.05,
                    "variance_estimate": 1.0,
                    "weight_cv": 2.0,
                }
            )
        )

    config_summary = summarize_mcmc_objective_grid_configs(rows)
    winners = select_mcmc_objective_grid_winners(
        config_summary,
        q_multipliers=(0.1, 0.2),
        n_swap_pairs_values=(1, 2, 3, 4),
    )

    oracle_row = next(row for row in winners["objective_winners"] if row["objective_name"] == "oracle_rmse")
    varhat_row = next(row for row in winners["objective_winners"] if row["objective_name"] == "varhat")

    assert oracle_row["config_id"] == "q01_s2"
    assert varhat_row["config_id"] == "q00_s1"
    assert varhat_row["oracle_exact_match"] == 0
    assert varhat_row["oracle_q_index_distance"] == 1
    assert varhat_row["oracle_swap_distance"] == 1
    assert varhat_row["oracle_fuzzy_similarity"] == pytest.approx(0.5)


def test_objective_grid_smoke_runs_end_to_end():
    scenario = _small_right_tail_scenario()
    mcmc_cfg = _small_mcmc_cfg()

    study = run_mcmc_objective_grid_study(
        scenario.problem,
        scenario.exact_p,
        mcmc_cfg=mcmc_cfg,
        q_multipliers=(1e-5, 1.0),
        n_swap_pairs_values=(1, 2),
        trial_repeats=2,
        trial_budget=40,
        base_seed=123,
        n_jobs=1,
    )

    assert len(study["repeat_records"]) == 8
    assert len(study["config_summary"]) == 4
    assert len(study["objective_seed_noise"]) == len(MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES)
    assert any(row["status"] == "invalid_q_map" for row in study["repeat_records"])
    assert any(row["status"] == "ok" for row in study["repeat_records"])
    assert study["oracle_winner"]["objective_name"] == "oracle_rmse"
    assert {row["objective_name"] for row in study["objective_winners"]} == set(("oracle_rmse", "oracle_abs_log10", *MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES))


def test_load_selected_scenarios_supports_portfolio_group(tmp_path):
    from perm_pval.experiments.exact_scenarios import save_exact_scenarios

    scenarios = [_make_gwas_additive_score_scenario(), _make_zero_inflated_poisson_diffmeans_scenario()]
    save_exact_scenarios(scenarios, tmp_path)
    loaded = load_selected_scenarios(
        catalog_path=tmp_path / "catalog.json",
        portfolio_group="core_claim",
    )

    assert [scenario.key for scenario in loaded] == ["gwas_additive_score_n40", "zip_diffmeans_righttail_n40"]
