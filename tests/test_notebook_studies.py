import numpy as np
import matplotlib.image as mpimg
import pytest

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.experiments.notebook_studies import (
    BetaSweepStudyConfig,
    CrossMethodStudyConfig,
    LoadedScenario,
    MCMCWorkflowConfig,
    SAMCWorkflowConfig,
    build_beta_workflow,
    deduplicate_selected_trial_configs,
    load_beta_sweep_saved_output,
    load_cross_method_saved_output,
    map_log10_q_multiplier_to_mcmc_candidate,
    regenerate_beta_sweep_plots_from_saved,
    regenerate_cross_method_plots_from_saved,
    run_beta_checkpoint_study,
    run_cross_method_study,
    run_mcmc_optuna_trial_table,
    run_named_mcmc_checkpoint_study,
    save_beta_sweep_outputs,
    save_cross_method_outputs,
    select_best_mcmc_trial_by_objective,
    summarize_mcmc_trial_repeats,
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
        tune_steps=30,
        tune_burn_in_fraction=0.2,
        tune_thin=1,
        tune_max_bracket=4,
        tune_max_bisect=4,
        local_scan_screen_total_steps=10,
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


def test_map_log10_q_multiplier_to_mcmc_candidate_clips_high_q_and_returns_finite_beta():
    scenario = _small_right_tail_scenario()
    pilot_t = iid_pilot_statistics(scenario.problem, n_samples=40, seed=123)
    sigma_t = estimate_scale_T(pilot_t, method="sd")

    candidate = map_log10_q_multiplier_to_mcmc_candidate(
        scenario.problem,
        pilot_t=pilot_t,
        sigma_t=sigma_t,
        p0_for_qtarget=scenario.exact_p,
        q_target=0.1,
        log10_q_multiplier=3.0,
        beta_max=1e6,
        q_clip_min=1e-6,
        q_clip_max=0.95,
    )

    assert candidate["q_clipped"] == 1
    assert candidate["q_trial"] == pytest.approx(0.95)
    assert np.isfinite(candidate["beta_trial"])
    assert candidate["beta_trial"] > 0.0


def test_map_log10_q_multiplier_to_mcmc_candidate_raises_when_q_map_has_no_positive_beta():
    scenario = _small_right_tail_scenario()
    pilot_t = iid_pilot_statistics(scenario.problem, n_samples=40, seed=123)
    sigma_t = estimate_scale_T(pilot_t, method="sd")

    with pytest.raises(ValueError, match="non-finite or non-positive beta"):
        map_log10_q_multiplier_to_mcmc_candidate(
            scenario.problem,
            pilot_t=pilot_t,
            sigma_t=sigma_t,
            p0_for_qtarget=scenario.exact_p,
            q_target=1e-4,
            log10_q_multiplier=-4.0,
            beta_max=1e6,
            q_clip_min=1e-6,
            q_clip_max=0.95,
        )


def test_trial_objective_summary_and_deduplication_handle_invalid_rows():
    invalid_rows = [
        {
            "trial_number": 0,
            "estimate": 0.0,
            "squared_error": 1.0,
            "variance_estimate": np.nan,
            "exact_p": 1.0,
            "wall_time_sec": 1.0,
            "eval_excl_tuning": 10.0,
            "selection_objective_p0": np.nan,
            "tail_hits": 0,
            "beta": 1.0,
            "sigma_t": 1.0,
            "proposal_size": 1,
            "n_swap_pairs": 1,
            "log10_q_multiplier": -1.0,
            "q_multiplier": 0.1,
            "q_target": 1e-3,
            "q_trial": 1e-4,
            "q_trial_raw": 1e-4,
            "q_clipped": 0,
        },
        {
            "trial_number": 0,
            "estimate": 0.0,
            "squared_error": 1.0,
            "variance_estimate": np.nan,
            "exact_p": 1.0,
            "wall_time_sec": 1.0,
            "eval_excl_tuning": 10.0,
            "selection_objective_p0": np.nan,
            "tail_hits": 0,
            "beta": 1.0,
            "sigma_t": 1.0,
            "proposal_size": 1,
            "n_swap_pairs": 1,
            "log10_q_multiplier": -1.0,
            "q_multiplier": 0.1,
            "q_target": 1e-3,
            "q_trial": 1e-4,
            "q_trial_raw": 1e-4,
            "q_clipped": 0,
        },
    ]
    valid_rows = [
        {
            "trial_number": 1,
            "estimate": 0.9,
            "squared_error": 0.01,
            "variance_estimate": 0.02,
            "exact_p": 1.0,
            "wall_time_sec": 1.0,
            "eval_excl_tuning": 10.0,
            "selection_objective_p0": 0.03,
            "tail_hits": 4,
            "tail_share_raw": 0.2,
            "acceptance_rate": 0.4,
            "weight_cv": 2.0,
            "ess": 10.0,
            "beta": 0.5,
            "sigma_t": 1.0,
            "proposal_size": 1,
            "n_swap_pairs": 1,
            "log10_q_multiplier": -0.5,
            "q_multiplier": 10 ** -0.5,
            "q_target": 1e-3,
            "q_trial": 3e-4,
            "q_trial_raw": 3e-4,
            "q_clipped": 0,
            "abs_log10_error": abs(np.log10(0.9) - np.log10(1.0)),
        },
        {
            "trial_number": 1,
            "estimate": 1.1,
            "squared_error": 0.01,
            "variance_estimate": 0.01,
            "exact_p": 1.0,
            "wall_time_sec": 1.0,
            "eval_excl_tuning": 10.0,
            "selection_objective_p0": 0.02,
            "tail_hits": 3,
            "tail_share_raw": 0.15,
            "acceptance_rate": 0.45,
            "weight_cv": 1.5,
            "ess": 12.0,
            "beta": 0.5,
            "sigma_t": 1.0,
            "proposal_size": 1,
            "n_swap_pairs": 1,
            "log10_q_multiplier": -0.5,
            "q_multiplier": 10 ** -0.5,
            "q_target": 1e-3,
            "q_trial": 3e-4,
            "q_trial_raw": 3e-4,
            "q_clipped": 0,
            "abs_log10_error": abs(np.log10(1.1) - np.log10(1.0)),
        },
    ]

    invalid_summary = summarize_mcmc_trial_repeats(invalid_rows)
    valid_summary = summarize_mcmc_trial_repeats(valid_rows)

    assert np.isinf(invalid_summary["objective_oracle_abs_log10"])
    assert np.isinf(invalid_summary["objective_diag_selection_objective_p0"])
    assert np.isinf(invalid_summary["objective_diag_variance_estimate"])
    assert np.isinf(invalid_summary["objective_diag_repeat_stability"])

    selected = select_best_mcmc_trial_by_objective([invalid_summary, valid_summary])
    dedup = deduplicate_selected_trial_configs(selected)

    assert len(dedup["configs"]) == 1
    assert set(dedup["objective_to_config"]) == {
        "oracle_rmse",
        "oracle_abs_log10",
        "diag_selection_objective_p0",
        "diag_variance_estimate",
        "diag_repeat_stability",
    }


def test_optuna_offline_smoke_runs_fixed_config_followup():
    pytest.importorskip("optuna")
    scenario = _small_right_tail_scenario()
    mcmc_cfg = _small_mcmc_cfg()

    trial_table = run_mcmc_optuna_trial_table(
        scenario.problem,
        scenario.exact_p,
        mcmc_cfg=mcmc_cfg,
        trials_per_scenario=2,
        trial_repeats=1,
        trial_budget=40,
        base_seed=123,
        q_log10_bounds=(-1.0, 1.0),
        n_jobs=1,
    )

    assert len(trial_table["trial_summary"]) == 2
    assert len(trial_table["selected_configs"]) >= 1

    final_study = run_named_mcmc_checkpoint_study(
        scenario.problem,
        scenario.exact_p,
        config_specs=[trial_table["selected_configs"][0]],
        sigma_t=float(trial_table["trial_context"]["sigma_t"]),
        estimation_points=(20, 40),
        repeats=1,
        base_seed=999,
        template_cfg=mcmc_cfg,
        n_jobs=1,
    )

    assert sorted({row["checkpoint"] for row in final_study["records"]}) == [20, 40]
    assert sorted({row["label"] for row in final_study["records"]}) == [trial_table["selected_configs"][0]["label"]]
