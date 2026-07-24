from __future__ import annotations

import json
import textwrap
from pathlib import Path


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _common_setup_code() -> str:
    return """
    from __future__ import annotations

    from dataclasses import replace
    import json
    import os
    import shutil
    import sys
    from pathlib import Path

    import pandas as pd
    from IPython.display import Image, display


    def find_project_root(start: Path | None = None) -> Path:
        current = (start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / "perm_pval").exists() and (candidate / "results").exists():
                return candidate
        raise RuntimeError("Could not locate project root containing perm_pval/ and results/.")


    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".matplotlib"))

    from perm_pval.experiments.notebook_studies import (
        BetaSweepStudyConfig,
        CrossMethodStudyConfig,
        DEFAULT_MCMC_OBJECTIVE_GRID_Q_MULTIPLIERS,
        DEFAULT_MCMC_OBJECTIVE_GRID_SWAP_COUNTS,
        MCMCWorkflowConfig,
        MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES,
        SAMCWorkflowConfig,
        _mcmc_eval_count,
        _steps_per_chain,
        build_beta_initialization,
        build_beta_workflow,
        create_timestamped_run_dir,
        load_cross_method_saved_output,
        load_beta_sweep_saved_output,
        load_mcmc_objective_grid_saved_output,
        load_selected_scenarios,
        known_threshold_p0_for_scenario,
        plot_named_method_convergence,
        plot_named_method_max_budget,
        bootstrap_threshold_grid_best_practical_diagnostics,
        plot_threshold_grid_estimate_vs_threshold_ratio,
        plot_threshold_grid_best_practical_scenario_scatter,
        plot_threshold_grid_best_practical_rrmse,
        plot_threshold_grid_gamma_swap_rrmse,
        plot_threshold_grid_tilt_family_rrmse,
        read_json,
        run_named_mcmc_checkpoint_study,
        run_mcmc_objective_grid_study,
        run_threshold_grid_method_block,
        save_mcmc_objective_grid_outputs,
        regenerate_beta_sweep_plots_from_saved,
        regenerate_cross_method_plots_from_saved,
        run_beta_checkpoint_study,
        run_cross_method_study,
        save_beta_sweep_outputs,
        save_cross_method_outputs,
        summarize_threshold_grid_gamma_swap_rrmse,
        summarize_records,
        tune_samc_setup,
        write_json,
        write_jsonl,
        _effective_n_jobs,
        _iid_replicate_worker,
        _samc_replicate_worker,
        _try_make_process_pool,
    )

    pd.set_option("display.max_columns", 100)
    project_root
    """


def _build_cross_method_notebook_legacy() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: Cross-Method Tiny-p Study

            Objective:
            - Compare `iid`, `mcmc_is`, `hard_step`, and `samc` on fixed exact scenarios.
            - Track estimates and diagnostics at intermediate estimation points, not just at the final budget.
            """
        ),
        code_cell(_common_setup_code()),
        markdown_cell(
            """
            ## Configuration

            `ESTIMATION_POINTS` controls the intermediate checkpoints.  
            The largest checkpoint is used for the main boxplots; all checkpoints are used for convergence diagnostics.  
            In the cross-method notebook these are total budgets. For MCMC-IS, a fixed beta-selection budget is deducted first, and the production chain uses the remaining budget.
            The hard-step method uses the same local proposal kernel with `q = p0 ** d_alpha`, where `p0` is the scenario's known threshold when present.
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            OUTPUT_ROOT = project_root / "results" / "cross_method_notebook"

            SCENARIO_GROUP = "core_claim"
            SCENARIO_KEYS_OVERRIDE = [
                "gwas_additive_score_ultra_n100",
                "poisson_diffmeans_hep_ultra_n200",
                "gwas_additive_score_sig_n100",
                "poisson_diffmeans_hep_sig_n200",
                "gwas_additive_score_above_n100",
                "poisson_diffmeans_hep_above_n200",
            ]

            ESTIMATION_POINTS = (500_000, 1_000_000, 2_500_000, 5_000_000) if not FAST_MODE else (200_000,)
            N_REPEATS = 5 if not FAST_MODE else 2
            N_JOBS = min(6, N_REPEATS, os.cpu_count() or 1)
            MIN_TAIL_STATES = 2
            BASE_SEED = 12_345
            MCMC_LOCAL_SCAN_STRATEGY = "adaptive_q"
            MCMC_LOCAL_SCAN_OBJECTIVE = "varhat_qmatch_soft"
            MCMC_PRODUCTION_ESTIMATOR_VARIANT = "selected_scan_plus_production"
            MCMC_SCAN_REFINE_TO_SCREEN_RATIO = 1.0
            MCMC_SCAN_FINAL_TO_SCREEN_RATIO = 2.0
            MCMC_SCAN_FINALIST_COUNT = 4
            DEFAULT_PROPOSAL_SIZE_BY_SAMPLE_BAND = {
                "small": 1,
                "medium": 1,
                "large": 2,
            }
            MCMC_LOCAL_SCAN_Q_MULTIPLIERS = (0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.33, 0.5, 1.0)
            MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS = (0.001, 0.01, 0.10, 0.5)

            cross_cfg = CrossMethodStudyConfig(
                estimation_points=ESTIMATION_POINTS,
                repeats=N_REPEATS,
                base_seed=BASE_SEED,
                iid_density_samples=150_000 if not FAST_MODE else 10_000,
                min_tail_states=MIN_TAIL_STATES,
                n_jobs=N_JOBS,
            )
            base_mcmc_cfg = MCMCWorkflowConfig(
                pilot_samples=25_000 if not FAST_MODE else 1_000,
                tune_steps=3_000 if not FAST_MODE else 1_000,
                local_scan_screen_total_steps=14_000 if not FAST_MODE else 1_000,
                local_scan_refine_total_steps=14_000 if not FAST_MODE else 1_000,
                local_scan_total_steps=32_000 if not FAST_MODE else 6_000,
                chains=2,
                thin=1,
                estimate_variance=True,
                production_estimator_variant=MCMC_PRODUCTION_ESTIMATOR_VARIANT,
                proposal_size=1,
                local_scan_strategy=MCMC_LOCAL_SCAN_STRATEGY,
                local_scan_objective=MCMC_LOCAL_SCAN_OBJECTIVE,
                local_scan_q_multipliers=MCMC_LOCAL_SCAN_Q_MULTIPLIERS,
                local_scan_coarse_q_multipliers=MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS,
                local_scan_refine_top_k=2,
                local_scan_refine_radius=1,
                local_scan_refine_max_q_points=6,
                local_scan_finalist_count=MCMC_SCAN_FINALIST_COUNT,
            )
            samc_cfg = SAMCWorkflowConfig(
                n_bins=100,
                t0=1_000.0,
                trace_every=200 if not FAST_MODE else 50,
                lambda_min_pilot=10_000 if not FAST_MODE else 500,
                proposal_size=0.1,
            )

            def proposal_size_for_scenario(scenario):
                setting_key = str(scenario.extra.get("application_setting_key", ""))
                if setting_key == "gwas_threshold_suite":
                    return 2
                if setting_key == "hep_threshold_suite":
                    return 6
                band = str(scenario.portfolio.get("sample_size_band", "medium"))
                return int(DEFAULT_PROPOSAL_SIZE_BY_SAMPLE_BAND.get(band, 1))


            def mcmc_proposal_size_for_scenario(scenario):
                return int(proposal_size_for_scenario(scenario))


            def samc_cfg_for_scenario(scenario):
                return replace(
                    samc_cfg,
                    proposal_size=int(proposal_size_for_scenario(scenario)),
                )


            def reference_p0_for_scenario(scenario) -> float:
                threshold = scenario.extra.get("known_significance_threshold")
                if threshold is not None:
                    threshold_f = float(threshold)
                    if threshold_f == threshold_f and threshold_f > 0.0:
                        return threshold_f
                return float(scenario.exact_p)


            def target_scan_budget_from_p0(p0: float) -> int:
                return 1_000_000


            def stage_eval_total(total_steps: int, *, n_candidates: int, n_chains: int) -> int:
                steps_per_chain = _steps_per_chain(int(total_steps), int(n_chains))
                return int(n_candidates) * _mcmc_eval_count(steps_per_chain, int(n_chains))


            def adaptive_candidate_counts(cfg: MCMCWorkflowConfig) -> tuple[int, int]:
                coarse_count = len(tuple(cfg.local_scan_coarse_q_multipliers)) or len(tuple(cfg.local_scan_q_multipliers))
                refine_count = int(cfg.local_scan_refine_max_q_points) if str(cfg.local_scan_strategy) == "adaptive_q" else 0
                return int(coarse_count), int(refine_count)


            def split_scan_budget_for_scenario(scenario, cfg: MCMCWorkflowConfig) -> dict:
                target_beta_selection_budget = int(target_scan_budget_from_p0(reference_p0_for_scenario(scenario)))
                scan_budget_ex_pilot = max(int(target_beta_selection_budget) - int(cfg.pilot_samples), 1)
                coarse_count, refine_count = adaptive_candidate_counts(cfg)
                finalist_count = int(cfg.local_scan_finalist_count)
                refine_ratio = float(MCMC_SCAN_REFINE_TO_SCREEN_RATIO) if str(cfg.local_scan_strategy) == "adaptive_q" else 0.0
                final_ratio = float(MCMC_SCAN_FINAL_TO_SCREEN_RATIO)
                budget_units = float(coarse_count) + float(refine_count) * refine_ratio + float(finalist_count) * final_ratio
                screen_total_steps = max(int(scan_budget_ex_pilot / max(budget_units, 1.0)), 1)
                refine_total_steps = (
                    int(max(1, round(refine_ratio * screen_total_steps)))
                    if str(cfg.local_scan_strategy) == "adaptive_q"
                    else None
                )
                final_total_steps = int(max(1, round(final_ratio * screen_total_steps)))
                expected_screen_eval_total = stage_eval_total(
                    screen_total_steps,
                    n_candidates=coarse_count,
                    n_chains=int(cfg.local_scan_screen_chains),
                )
                expected_refine_eval_total = (
                    stage_eval_total(
                        int(refine_total_steps),
                        n_candidates=refine_count,
                        n_chains=int(
                            cfg.local_scan_refine_chains
                            if cfg.local_scan_refine_chains is not None
                            else cfg.local_scan_screen_chains
                        ),
                    )
                    if refine_total_steps is not None and refine_count > 0
                    else 0
                )
                expected_final_eval_total = stage_eval_total(
                    final_total_steps,
                    n_candidates=finalist_count,
                    n_chains=int(cfg.local_scan_chains),
                )
                return {
                    "target_beta_selection_budget": int(target_beta_selection_budget),
                    "screen_total_steps": int(screen_total_steps),
                    "refine_total_steps": int(refine_total_steps) if refine_total_steps is not None else None,
                    "final_total_steps": int(final_total_steps),
                    "expected_beta_selection_budget_upper": int(
                        int(cfg.pilot_samples)
                        + int(expected_screen_eval_total)
                        + int(expected_refine_eval_total)
                        + int(expected_final_eval_total)
                    ),
                }


            def mcmc_cfg_for_scenario(scenario):
                proposal_size = int(mcmc_proposal_size_for_scenario(scenario))
                scan_budget = split_scan_budget_for_scenario(scenario, base_mcmc_cfg)
                reference_p0 = float(reference_p0_for_scenario(scenario))
                return replace(
                    base_mcmc_cfg,
                    use_true_p0_for_q_target=False,
                    p0_guess=reference_p0,
                    proposal_size=proposal_size,
                    local_scan_swap_counts=(proposal_size,),
                    local_scan_screen_total_steps=int(scan_budget["screen_total_steps"]),
                    local_scan_refine_total_steps=(
                        int(scan_budget["refine_total_steps"])
                        if scan_budget["refine_total_steps"] is not None
                        else None
                    ),
                    local_scan_total_steps=int(scan_budget["final_total_steps"]),
                )

            print(json.dumps({
                "FAST_MODE": FAST_MODE,
                "SCENARIO_GROUP": SCENARIO_GROUP,
                "SCENARIO_KEYS_OVERRIDE": SCENARIO_KEYS_OVERRIDE,
                "ESTIMATION_POINTS": ESTIMATION_POINTS,
                "N_REPEATS": N_REPEATS,
                "N_JOBS": N_JOBS,
                "SAVE_OUTPUTS": SAVE_OUTPUTS,
                "MCMC_LOCAL_SCAN_STRATEGY": MCMC_LOCAL_SCAN_STRATEGY,
                "MCMC_LOCAL_SCAN_OBJECTIVE": MCMC_LOCAL_SCAN_OBJECTIVE,
                "MCMC_PRODUCTION_ESTIMATOR_VARIANT": MCMC_PRODUCTION_ESTIMATOR_VARIANT,
                "MCMC_SCAN_FINALIST_COUNT": MCMC_SCAN_FINALIST_COUNT,
                "MCMC_LOCAL_SCAN_Q_MULTIPLIERS": MCMC_LOCAL_SCAN_Q_MULTIPLIERS,
                "MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS": MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS,
                    "DEFAULT_PROPOSAL_SIZE_BY_SAMPLE_BAND": DEFAULT_PROPOSAL_SIZE_BY_SAMPLE_BAND,
                    "APPLICATION_PROPOSAL_SIZE_OVERRIDES": {
                        "gwas_threshold_suite": 2,
                        "hep_threshold_suite": 6,
                    },
                    "REFERENCE_P0_MODE": "known_significance_threshold_else_exact_p",
                }, indent=2))
            """
        ),
        markdown_cell("## Load Scenarios"),
        code_cell(
            """
            scenarios = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=SCENARIO_KEYS_OVERRIDE,
                portfolio_group=None if SCENARIO_KEYS_OVERRIDE is not None else SCENARIO_GROUP,
                min_tail_states=MIN_TAIL_STATES,
            )

            run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "cross_method") if SAVE_OUTPUTS else None

            pd.DataFrame([
                {
                    "scenario": s.key,
                    "exact_p": s.exact_p,
                    "tail_hits": s.exact_tail_hits,
                    "n_perm": s.exact_n_perm,
                    "exact_method": s.exact_method,
                    "family": s.portfolio.get("family"),
                    "rarity_band": s.portfolio.get("rarity_band"),
                    "difficulty": s.portfolio.get("expected_difficulty"),
                    "groups": ",".join(s.portfolio.get("groups", [])),
                }
                for s in scenarios
            ])
            """
        ),
        markdown_cell(
            """
            ## Run Cross-Method Study

            For each scenario:
            - build one MCMC-IS beta workflow,
            - evaluate all methods at every checkpoint in `ESTIMATION_POINTS`,
            - save max-budget and convergence plots.
            """
        ),
        code_cell(
            """
            cross_results = {}

            for scenario in scenarios:
                scenario_mcmc_cfg = mcmc_cfg_for_scenario(scenario)
                scenario_samc_cfg = samc_cfg_for_scenario(scenario)
                print(f"Running {scenario.key} | exact p={scenario.exact_p:.3e}")
                print(json.dumps({
                    "scenario": scenario.key,
                    "application_setting_key": scenario.extra.get("application_setting_key"),
                    "known_significance_threshold": scenario.extra.get("known_significance_threshold"),
                    "reference_p0_for_qtarget": reference_p0_for_scenario(scenario),
                    "sample_size_band": scenario.portfolio.get("sample_size_band"),
                    "mcmc_proposal_size": scenario_mcmc_cfg.proposal_size,
                    "samc_proposal_size": scenario_samc_cfg.proposal_size,
                    "mcmc_local_scan_strategy": scenario_mcmc_cfg.local_scan_strategy,
                    "mcmc_local_scan_swap_counts": scenario_mcmc_cfg.local_scan_swap_counts,
                    "mcmc_local_scan_objective": scenario_mcmc_cfg.local_scan_objective,
                    "mcmc_production_estimator_variant": scenario_mcmc_cfg.production_estimator_variant,
                    "hard_step_enabled": scenario_mcmc_cfg.hard_step_enabled,
                    "hard_step_target_tail_mass": scenario_mcmc_cfg.hard_step_target_tail_mass,
                    "target_beta_selection_budget": target_scan_budget_from_p0(reference_p0_for_scenario(scenario)),
                    "local_scan_screen_total_steps": scenario_mcmc_cfg.local_scan_screen_total_steps,
                    "local_scan_refine_total_steps": scenario_mcmc_cfg.local_scan_refine_total_steps,
                    "local_scan_final_total_steps": scenario_mcmc_cfg.local_scan_total_steps,
                }, indent=2))
                study = run_cross_method_study(
                    scenario,
                    cross_cfg=cross_cfg,
                    mcmc_cfg=scenario_mcmc_cfg,
                    samc_cfg=scenario_samc_cfg,
                )
                cross_results[scenario.key] = study

                if SAVE_OUTPUTS and run_dir is not None:
                    save_cross_method_outputs(
                        scenario,
                        study,
                        output_dir=run_dir / scenario.key,
                        cross_cfg=cross_cfg,
                        mcmc_cfg=scenario_mcmc_cfg,
                        samc_cfg=scenario_samc_cfg,
                    )

                print(json.dumps({
                    "scenario": scenario.key,
                    "mcmc_beta_selection_budget": study["mcmc_beta_selection_budget"],
                    "mcmc_reported_checkpoints": study.get("mcmc_reported_checkpoints", []),
                    "beta_used": study["beta_workflow"]["beta_used"],
                    "hard_step_workflow": study.get("hard_step_workflow", {}),
                }, indent=2))
                summary_df = pd.DataFrame(study["summary"]).sort_values(["checkpoint", "method"])
                display(summary_df[[
                    "checkpoint",
                    "method",
                    "mean_estimate",
                    "rmse",
                    "mean_variance_estimate",
                    "mean_eval_incl_tuning",
                    "mean_q_tilt_tail_share",
                    "mean_ess",
                    "mean_zero_rate",
                    "mean_samc_max_rel_freq_error",
                ]])

            family_rows = []
            for scenario in scenarios:
                meta = scenario.portfolio
                for row in cross_results[scenario.key]["summary"]:
                    family_rows.append({
                        "scenario": scenario.key,
                        "family": meta.get("family"),
                        "rarity_band": meta.get("rarity_band"),
                        "difficulty": meta.get("expected_difficulty"),
                        **row,
                    })

            family_df = pd.DataFrame(family_rows)
            display(
                family_df.groupby(["family", "rarity_band", "method", "checkpoint"], as_index=False)
                .agg(
                    mean_rmse=("rmse", "mean"),
                    mean_estimate=("mean_estimate", "mean"),
                    mean_q_tilt_tail_share=("mean_q_tilt_tail_share", "mean"),
                    mean_ess=("mean_ess", "mean"),
                )
                .sort_values(["family", "rarity_band", "checkpoint", "method"])
            )
            """
        ),
        markdown_cell("## Review Saved Figures"),
        code_cell(
            """
            if SAVE_OUTPUTS and run_dir is not None:
                print(f"Saved outputs under: {run_dir}")
                for scenario in scenarios:
                    scenario_dir = run_dir / scenario.key
                    print(f"\\n{scenario.key}")
                    display(Image(filename=str(scenario_dir / "cross_method_max_budget.png")))
                    display(Image(filename=str(scenario_dir / "cross_method_convergence.png")))
                    display(Image(filename=str(scenario_dir / "cross_method_diagnostics.png")))
                    display(Image(filename=str(scenario_dir / "iid_density.png")))
            else:
                print("SAVE_OUTPUTS=False, so no saved figures to display.")
            """
        ),
        markdown_cell("## Reload Saved Results Without Rerunning"),
        code_cell(
            """
            # RELOAD_SCENARIO_DIR = None
            # # Example:
            # # RELOAD_SCENARIO_DIR = project_root / "results" / "cross_method_notebook" / "20260306_120000_cross_method" / "gwas_additive_score_sig_n100"

            # if RELOAD_SCENARIO_DIR is not None:
            #     saved = load_cross_method_saved_output(RELOAD_SCENARIO_DIR)
            #     print(json.dumps({
            #         "scenario": saved["metadata"]["scenario"],
            #         "exact_p": saved["metadata"]["exact_p"],
            #         "mcmc_beta_selection_budget": saved["metadata"]["beta_workflow"]["beta_selection_eval_total"],
            #     }, indent=2))
            #     regen = regenerate_cross_method_plots_from_saved(RELOAD_SCENARIO_DIR)
            #     for name, path in regen.items():
            #         print(name, path)
            #         display(Image(filename=str(path)))
            # else:
            #     print("Set RELOAD_SCENARIO_DIR to a saved scenario directory to regenerate plots from disk only.")
            """
        ),
    ]
    return notebook(cells)


def build_cross_method_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: Cross-Method Oracle vs Baselines

            Objective:
            - Use the same six threshold-suite scenarios as the oracle-beta notebook.
            - Compare `oracle_mcmcis`, `simple_mcmcis`, and `samc`.
            - Reuse the saved simple/oracle MCMC-IS settings from the completed beta-oracle run.
            - Track every method at dense checkpoints from `0.25M` through `5M` budget per run.
            """
        ),
        code_cell(_common_setup_code()),
        markdown_cell(
            """
            ## Configuration

            This notebook reuses the saved **oracle** MCMC-IS settings from the latest oracle-beta-search run.  
            The **naive/simple** MCMC-IS setting is rebuilt once per scenario from the current `gamma = 1/3` rule using a shared pilot that is **not** charged to the production `5M` chain budget.  
            Each method is evaluated with `20` independent one-chain runs, so the full `5M` is the actual chain budget for every reported run.
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            BETA_RESULTS_ROOT = project_root / "results" / "mcmcis_beta_notebook"


            def latest_oracle_beta_run_dir(root: Path) -> Path:
                candidates = sorted(
                    [
                        path for path in Path(root).iterdir()
                        if path.is_dir() and path.name.endswith("_oracle_beta_search")
                    ]
                )
                if not candidates:
                    raise FileNotFoundError(
                        "No oracle-beta-search run found under results/mcmcis_beta_notebook. "
                        "Run mcmcis_oracle_beta_search.ipynb first, or set BETA_RUN_DIR manually."
                    )
                return candidates[-1]


            BETA_RUN_DIR = latest_oracle_beta_run_dir(BETA_RESULTS_ROOT)
            OUTPUT_ROOT = project_root / "results" / "cross_method_notebook"

            SCENARIO_KEYS_OVERRIDE = [
                "gwas_additive_score_ultra_n100",
                "poisson_diffmeans_hep_ultra_n200",
                "gwas_additive_score_sig_n100",
                "poisson_diffmeans_hep_sig_n200",
                "gwas_additive_score_above_n100",
                "poisson_diffmeans_hep_above_n200",
            ]

            MIN_TAIL_STATES = 2
            BASE_SEED = 12_345
            N_METHOD_RUNS = 20 if not FAST_MODE else 3
            N_JOBS = min(6, os.cpu_count() or 1)
            CHAIN_BUDGET = 5_000_000 if not FAST_MODE else 1_000_000
            CHECKPOINT_STEP = 250_000 if not FAST_MODE else 250_000
            ESTIMATION_POINTS = tuple(range(CHECKPOINT_STEP, CHAIN_BUDGET + CHECKPOINT_STEP, CHECKPOINT_STEP))

            METHOD_ORDER = ["samc", "simple_mcmcis", "oracle_mcmcis"]
            METHOD_LABELS = {
                "samc": "SAMC",
                "simple_mcmcis": "Simple-init MCMC-IS",
                "oracle_mcmcis": "Oracle MCMC-IS",
            }
            METHOD_COLORS = {
                "samc": "#4c8c77",
                "simple_mcmcis": "#c48a3a",
                "oracle_mcmcis": "#b04a5a",
            }

            cross_cfg = CrossMethodStudyConfig(
                estimation_points=ESTIMATION_POINTS,
                repeats=N_METHOD_RUNS,
                base_seed=BASE_SEED,
                iid_density_samples=150_000 if not FAST_MODE else 20_000,
                min_tail_states=MIN_TAIL_STATES,
                n_jobs=N_JOBS,
            )
            simple_init_cfg = MCMCWorkflowConfig(
                use_true_p0_for_q_target=False,
                d_alpha=MCMCWorkflowConfig().d_alpha,
                pilot_samples=200_000 if not FAST_MODE else 1_000,
                scale_method="sd",
                beta_max_init=1e6,
                tune_steps=0,
                chains=1,
                burn_in_fraction=0.20,
                thin=1,
                estimate_variance=True,
                obm_batch_size=None,
                chain_n_jobs=1,
                tilt_mode="smooth_hinge",
                proposal_size=1,
                local_scan_enabled=False,
            )
            mcmc_template_cfg = MCMCWorkflowConfig(
                pilot_samples=0,
                tune_steps=0,
                chains=1,
                burn_in_fraction=0.20,
                thin=1,
                estimate_variance=True,
                obm_batch_size=None,
                chain_n_jobs=1,
                tilt_mode="smooth_hinge",
                proposal_size=1,
                local_scan_enabled=False,
            )
            samc_base_cfg = SAMCWorkflowConfig(
                n_bins=100,
                t0=1_000.0,
                trace_every=200 if not FAST_MODE else 50,
                lambda_min_pilot=10_000 if not FAST_MODE else 1_000,
                proposal_size=0.1,
            )

            NOTEBOOK_CONFIG = {
                "FAST_MODE": FAST_MODE,
                "SAVE_OUTPUTS": SAVE_OUTPUTS,
                "BETA_RUN_DIR": str(BETA_RUN_DIR),
                "SCENARIO_KEYS_OVERRIDE": SCENARIO_KEYS_OVERRIDE,
                "MIN_TAIL_STATES": MIN_TAIL_STATES,
                "BASE_SEED": BASE_SEED,
                "N_METHOD_RUNS": N_METHOD_RUNS,
                "N_JOBS": N_JOBS,
                "CHAIN_BUDGET": CHAIN_BUDGET,
                "CHECKPOINT_STEP": CHECKPOINT_STEP,
                "N_CHECKPOINTS": len(ESTIMATION_POINTS),
                "METHOD_ORDER": METHOD_ORDER,
                "USES_SAVED_ORACLE_ONLY": True,
                "SIMPLE_INIT_GAMMA": float(simple_init_cfg.d_alpha),
                "SIMPLE_INIT_PILOT_SAMPLES": int(simple_init_cfg.pilot_samples),
                "COUNTS_INCLUDE_PILOT_BUDGET": False,
                "PLOT_X_SCALE": "linear",
                "CONVERGENCE_FIGURES": ["mean_estimate", "median_estimate"],
            }

            print(json.dumps(NOTEBOOK_CONFIG, indent=2))
            """
        ),
        markdown_cell("## Notebook Helpers"),
        code_cell(
            """
            def proposal_size_for_scenario(scenario) -> int:
                setting_key = str(scenario.extra.get("application_setting_key", ""))
                if setting_key == "gwas_threshold_suite":
                    return 2
                if setting_key == "hep_threshold_suite":
                    return 6
                band = str(scenario.portfolio.get("sample_size_band", "medium"))
                if band == "small":
                    return 1
                if band == "large":
                    return 2
                return 1


            def samc_cfg_for_scenario(scenario):
                return replace(
                    samc_base_cfg,
                    proposal_size=int(proposal_size_for_scenario(scenario)),
                )


            def load_beta_reference(beta_run_dir: Path, scenario_key: str) -> dict:
                scenario_dir = Path(beta_run_dir) / scenario_key
                best_config = read_json(scenario_dir / "best_config.json")
                oracle_summary = read_json(scenario_dir / "oracle_best_trial.json")
                metadata = read_json(scenario_dir / "metadata.json")
                return {
                    "scenario": scenario_key,
                    "beta_run_dir": str(scenario_dir),
                    "oracle_sigma_t": float(best_config["sigma_t"]),
                    "reference_p0": float(best_config["reference_p0"]),
                    "simple_reference_p0": float(best_config.get("simple_reference_p0", best_config["reference_p0"])),
                    "canonical_threshold_p0": best_config.get("canonical_threshold_p0"),
                    "oracle_beta": float(best_config["best_beta"]),
                    "oracle_proposal_size": int(best_config["best_proposal_size"]),
                    "oracle_trial_number": int(best_config["best_trial_number"]),
                    "oracle_objective_value": float(best_config["best_objective_value"]),
                    "selection_source": str(best_config.get("selection_source", "unknown")),
                    "application_setting_key": metadata.get("application_setting_key"),
                    "known_significance_threshold": metadata.get("known_significance_threshold"),
                    "oracle_summary": oracle_summary,
                }


            def build_simple_reference(scenario, beta_reference: dict, *, seed: int) -> dict:
                p0_reference = float(beta_reference["simple_reference_p0"])
                init_payload = build_beta_initialization(
                    scenario.problem,
                    scenario.exact_p,
                    simple_init_cfg,
                    seed=int(seed),
                    p0_reference=p0_reference,
                )
                proposal_size = int(proposal_size_for_scenario(scenario))
                return {
                    "scenario": scenario.key,
                    "simple_gamma": float(simple_init_cfg.d_alpha),
                    "simple_reference_p0": p0_reference,
                    "simple_q_target": float(init_payload["q_target"]),
                    "simple_beta": float(init_payload["beta0_laplace"]),
                    "simple_sigma_t": float(init_payload["sigma_t"]),
                    "simple_proposal_size": proposal_size,
                    "pilot_eval_total": int(init_payload["pilot_eval_total"]),
                    "pilot_wall_time_sec": float(init_payload["pilot_wall_time_sec"]),
                }


            def build_mcmc_config_specs(simple_reference: dict, beta_reference: dict) -> list[dict]:
                return [
                    {
                        "label": "simple_mcmcis",
                        "config_id": "simple_mcmcis",
                        "beta": float(simple_reference["simple_beta"]),
                        "proposal_size": int(simple_reference["simple_proposal_size"]),
                        "source": "current_simple_rule",
                    },
                    {
                        "label": "oracle_mcmcis",
                        "config_id": f"oracle_trial_{int(beta_reference['oracle_trial_number']):02d}",
                        "beta": float(beta_reference["oracle_beta"]),
                        "proposal_size": int(beta_reference["oracle_proposal_size"]),
                        "source": "oracle_beta_search_oracle",
                    },
                ]


            def run_parallel_worker_jobs(worker_fn, jobs: list[dict], *, n_jobs: int) -> list[dict]:
                n_workers = _effective_n_jobs(int(n_jobs), len(jobs))
                executor = _try_make_process_pool(n_workers) if n_workers > 1 else None
                rows: list[dict] = []
                if executor is None:
                    for job in jobs:
                        rows.extend(worker_fn(**job))
                    return rows

                with executor:
                    futures = [executor.submit(worker_fn, **job) for job in jobs]
                    for future in futures:
                        rows.extend(future.result())
                return rows


            def run_samc_baseline(scenario, *, base_seed: int, samc_cfg) -> tuple[list[dict], dict]:
                checkpoints = tuple(int(v) for v in cross_cfg.estimation_points)
                samc_setup = tune_samc_setup(
                    scenario.problem,
                    samc_cfg,
                    seed=int(base_seed + 50_000),
                )
                samc_jobs = [
                    {
                        "scenario_key": scenario.key,
                        "scenario_display": scenario.description,
                        "problem": scenario.problem,
                        "exact_p": scenario.exact_p,
                        "checkpoints": checkpoints,
                        "samc_setup": samc_setup,
                        "samc_cfg": samc_cfg,
                        "rep": int(rep),
                        "rep_seed": int(base_seed + 100_000 + 1_000 * rep),
                    }
                    for rep in range(int(cross_cfg.repeats))
                ]
                samc_rows = run_parallel_worker_jobs(_samc_replicate_worker, samc_jobs, n_jobs=int(cross_cfg.n_jobs))
                for row in samc_rows:
                    row["label"] = "samc"

                return samc_rows, samc_setup


            def save_oracle_cross_method_outputs(
                scenario,
                study: dict,
                *,
                output_dir: Path,
                beta_reference: dict,
                samc_cfg,
            ) -> None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                known_threshold = beta_reference.get("known_significance_threshold")

                plot_named_method_max_budget(
                    study["records"],
                    scenario_name=study["scenario_display"],
                    scenario_key=study["scenario"],
                    exact_p=float(study["exact_p"]),
                    max_budget=max(int(v) for v in study["estimation_points"]),
                    method_order=METHOD_ORDER,
                    method_labels=METHOD_LABELS,
                    method_colors=METHOD_COLORS,
                    n_control=int(scenario.problem.n_control),
                    n_treated=int(scenario.problem.n_treated),
                    known_significance_threshold=(
                        float(known_threshold)
                        if known_threshold is not None
                        else None
                    ),
                    save_path=output_dir / "cross_method_max_budget.png",
                )
                plot_named_method_convergence(
                    study["summary"],
                    scenario_name=study["scenario_display"],
                    scenario_key=study["scenario"],
                    exact_p=float(study["exact_p"]),
                    method_order=METHOD_ORDER,
                    method_labels=METHOD_LABELS,
                    method_colors=METHOD_COLORS,
                    n_control=int(scenario.problem.n_control),
                    n_treated=int(scenario.problem.n_treated),
                    known_significance_threshold=(
                        float(known_threshold)
                        if known_threshold is not None
                        else None
                    ),
                    x_label="Budget per run",
                    save_path=output_dir / "cross_method_convergence.png",
                )
                plot_named_method_convergence(
                    study["summary"],
                    scenario_name=study["scenario_display"],
                    scenario_key=study["scenario"],
                    exact_p=float(study["exact_p"]),
                    method_order=METHOD_ORDER,
                    method_labels=METHOD_LABELS,
                    method_colors=METHOD_COLORS,
                    n_control=int(scenario.problem.n_control),
                    n_treated=int(scenario.problem.n_treated),
                    known_significance_threshold=(
                        float(known_threshold)
                        if known_threshold is not None
                        else None
                    ),
                    x_label="Budget per run",
                    estimate_field="median_estimate",
                    estimate_title="Median estimate",
                    estimate_ylabel=r"median $\\hat{p}$",
                    save_path=output_dir / "cross_method_convergence_median.png",
                )
                write_jsonl(output_dir / "run_records.jsonl", study["records"])
                write_json(output_dir / "summary.json", study["summary"])
                write_json(
                    output_dir / "metadata.json",
                    {
                        "scenario": study["scenario"],
                        "scenario_display": study["scenario_display"],
                        "scenario_portfolio": study["scenario_portfolio"],
                        "exact_p": study["exact_p"],
                        "exact_method": study["exact_method"],
                        "exact_tail_hits": study["exact_tail_hits"],
                        "exact_n_perm": study["exact_n_perm"],
                        "n_treated": int(scenario.problem.n_treated),
                        "n_control": int(scenario.problem.n_control),
                        "n_total": int(scenario.problem.n),
                        "estimation_points": study["estimation_points"],
                        "method_order": METHOD_ORDER,
                        "method_labels": METHOD_LABELS,
                        "method_colors": METHOD_COLORS,
                        "x_label": "Budget per run",
                        "x_scale": "linear",
                        "convergence_plot_fields": {
                            "mean": "mean_estimate",
                            "median": "median_estimate",
                        },
                        "cross_config": cross_cfg,
                        "simple_init_config": simple_init_cfg,
                        "mcmc_template_config": mcmc_template_cfg,
                        "samc_config": samc_cfg,
                        "simple_reference": study["simple_reference"],
                        "beta_reference": beta_reference,
                        "samc_setup": study["samc_setup"],
                        "notebook_config": NOTEBOOK_CONFIG,
                    },
                )


            def regenerate_oracle_cross_method_plots(
                scenario_dir: Path,
                *,
                save_dir: Path | None = None,
            ) -> dict[str, Path]:
                saved = load_cross_method_saved_output(scenario_dir)
                metadata = saved["metadata"]
                beta_reference = dict(metadata.get("beta_reference", {}))
                save_dir = Path(save_dir) if save_dir is not None else Path(scenario_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                max_budget = max(int(v) for v in metadata["estimation_points"])
                known_threshold = beta_reference.get("known_significance_threshold")
                out = {
                    "cross_method_max_budget": save_dir / "cross_method_max_budget.png",
                    "cross_method_convergence": save_dir / "cross_method_convergence.png",
                    "cross_method_convergence_median": save_dir / "cross_method_convergence_median.png",
                }
                plot_named_method_max_budget(
                    saved["records"],
                    scenario_name=str(metadata["scenario_display"]),
                    scenario_key=str(metadata["scenario"]),
                    exact_p=float(metadata["exact_p"]),
                    max_budget=max_budget,
                    method_order=list(metadata.get("method_order", METHOD_ORDER)),
                    method_labels=dict(METHOD_LABELS),
                    method_colors=dict(METHOD_COLORS),
                    n_control=int(metadata["n_control"]),
                    n_treated=int(metadata["n_treated"]),
                    known_significance_threshold=(
                        float(known_threshold)
                        if known_threshold is not None
                        else None
                    ),
                    save_path=out["cross_method_max_budget"],
                )
                plot_named_method_convergence(
                    saved["summary"],
                    scenario_name=str(metadata["scenario_display"]),
                    scenario_key=str(metadata["scenario"]),
                    exact_p=float(metadata["exact_p"]),
                    method_order=list(metadata.get("method_order", METHOD_ORDER)),
                    method_labels=dict(METHOD_LABELS),
                    method_colors=dict(METHOD_COLORS),
                    n_control=int(metadata["n_control"]),
                    n_treated=int(metadata["n_treated"]),
                    known_significance_threshold=(
                        float(known_threshold)
                        if known_threshold is not None
                        else None
                    ),
                    x_label=str(metadata.get("x_label", "Budget per run")),
                    save_path=out["cross_method_convergence"],
                )
                plot_named_method_convergence(
                    saved["summary"],
                    scenario_name=str(metadata["scenario_display"]),
                    scenario_key=str(metadata["scenario"]),
                    exact_p=float(metadata["exact_p"]),
                    method_order=list(metadata.get("method_order", METHOD_ORDER)),
                    method_labels=dict(METHOD_LABELS),
                    method_colors=dict(METHOD_COLORS),
                    n_control=int(metadata["n_control"]),
                    n_treated=int(metadata["n_treated"]),
                    known_significance_threshold=(
                        float(known_threshold)
                        if known_threshold is not None
                        else None
                    ),
                    x_label=str(metadata.get("x_label", "Budget per run")),
                    estimate_field="median_estimate",
                    estimate_title="Median estimate",
                    estimate_ylabel=r"median $\\hat{p}$",
                    save_path=out["cross_method_convergence_median"],
                )
                return out
            """
        ),
        markdown_cell("## Load Scenarios and Oracle References"),
        code_cell(
            """
            scenarios = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=SCENARIO_KEYS_OVERRIDE,
                portfolio_group=None,
                min_tail_states=MIN_TAIL_STATES,
            )

            beta_references = {
                scenario.key: load_beta_reference(BETA_RUN_DIR, scenario.key)
                for scenario in scenarios
            }
            simple_references = {
                scenario.key: build_simple_reference(
                    scenario,
                    beta_references[scenario.key],
                    seed=int(BASE_SEED + 50_000 * (scenario_idx + 1)),
                )
                for scenario_idx, scenario in enumerate(scenarios)
            }
            run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "cross_method_oracle_compare") if SAVE_OUTPUTS else None

            pd.DataFrame(
                [
                    {
                        "scenario": scenario.key,
                        "exact_p": scenario.exact_p,
                        "tail_hits": scenario.exact_tail_hits,
                        "n_perm": scenario.exact_n_perm,
                        "family": scenario.portfolio.get("family"),
                        "rarity_band": scenario.portfolio.get("rarity_band"),
                        "simple_beta": simple_references[scenario.key]["simple_beta"],
                        "simple_prop": simple_references[scenario.key]["simple_proposal_size"],
                        "simple_gamma": simple_references[scenario.key]["simple_gamma"],
                        "oracle_beta": beta_references[scenario.key]["oracle_beta"],
                        "oracle_prop": beta_references[scenario.key]["oracle_proposal_size"],
                        "reference_p0": beta_references[scenario.key]["reference_p0"],
                    }
                    for scenario in scenarios
                ]
            )
            """
        ),
        markdown_cell(
            """
            ## Run Cross-Method Study

            For each scenario:
            - load the saved **oracle** MCMC-IS setting from `BETA_RUN_DIR`,
            - rebuild the **simple-init** MCMC-IS setting from the current `gamma = 1/3` rule without charging that pilot to the production budget,
            - run `20` independent one-chain replicates for each method,
            - save the article-facing max-budget plot plus both mean- and median-based convergence plots.
            """
        ),
        code_cell(
            """
            cross_results = {}

            for scenario_idx, scenario in enumerate(scenarios):
                beta_reference = beta_references[scenario.key]
                simple_reference = simple_references[scenario.key]
                scenario_seed = int(BASE_SEED + 1_000_000 * (scenario_idx + 1))
                scenario_samc_cfg = samc_cfg_for_scenario(scenario)
                simple_specs = build_mcmc_config_specs(simple_reference, beta_reference)[:1]
                oracle_specs = build_mcmc_config_specs(simple_reference, beta_reference)[1:]

                print(f"Running {scenario.key} | exact p={scenario.exact_p:.3e}")
                print(json.dumps({
                    "scenario": scenario.key,
                    "n_method_runs": int(cross_cfg.repeats),
                    "n_jobs": int(cross_cfg.n_jobs),
                    "chain_budget": CHAIN_BUDGET,
                    "first_checkpoints": [int(v) for v in ESTIMATION_POINTS[:4]],
                    "last_checkpoint": int(ESTIMATION_POINTS[-1]),
                    "simple_beta": simple_reference["simple_beta"],
                    "simple_proposal_size": simple_reference["simple_proposal_size"],
                    "simple_gamma": simple_reference["simple_gamma"],
                    "simple_sigma_t": simple_reference["simple_sigma_t"],
                    "simple_pilot_eval_total": simple_reference["pilot_eval_total"],
                    "oracle_beta": beta_reference["oracle_beta"],
                    "oracle_proposal_size": beta_reference["oracle_proposal_size"],
                    "oracle_sigma_t": beta_reference["oracle_sigma_t"],
                    "samc_proposal_size": scenario_samc_cfg.proposal_size,
                }, indent=2))

                simple_study = run_named_mcmc_checkpoint_study(
                    scenario.problem,
                    scenario.exact_p,
                    config_specs=simple_specs,
                    sigma_t=float(simple_reference["simple_sigma_t"]),
                    estimation_points=tuple(int(v) for v in cross_cfg.estimation_points),
                    repeats=int(cross_cfg.repeats),
                    base_seed=scenario_seed,
                    template_cfg=mcmc_template_cfg,
                    n_jobs=int(cross_cfg.n_jobs),
                )
                oracle_study = run_named_mcmc_checkpoint_study(
                    scenario.problem,
                    scenario.exact_p,
                    config_specs=oracle_specs,
                    sigma_t=float(beta_reference["oracle_sigma_t"]),
                    estimation_points=tuple(int(v) for v in cross_cfg.estimation_points),
                    repeats=int(cross_cfg.repeats),
                    base_seed=scenario_seed + 200_000,
                    template_cfg=mcmc_template_cfg,
                    n_jobs=int(cross_cfg.n_jobs),
                )
                samc_rows, samc_setup = run_samc_baseline(
                    scenario,
                    base_seed=scenario_seed + 400_000,
                    samc_cfg=scenario_samc_cfg,
                )

                records = list(simple_study["records"]) + list(oracle_study["records"]) + list(samc_rows)
                records = sorted(
                    records,
                    key=lambda row: (
                        str(row.get("label", row.get("method", ""))),
                        int(row["replicate"]),
                        int(row["checkpoint"]),
                    ),
                )
                summary = summarize_records(records, group_fields=("checkpoint", "label"))

                study = {
                    "scenario": scenario.key,
                    "scenario_display": scenario.description,
                    "scenario_portfolio": dict(scenario.portfolio),
                    "exact_p": float(scenario.exact_p),
                    "exact_method": scenario.exact_method,
                    "exact_tail_hits": int(scenario.exact_tail_hits),
                    "exact_n_perm": int(scenario.exact_n_perm),
                    "estimation_points": [int(v) for v in cross_cfg.estimation_points],
                    "records": records,
                    "summary": summary,
                    "beta_reference": beta_reference,
                    "simple_reference": simple_reference,
                    "samc_setup": {
                        "lambda_min": float(samc_setup["lambda_min"]),
                        "bin_edges": samc_setup["bin_edges"],
                    },
                }
                cross_results[scenario.key] = study

                if SAVE_OUTPUTS and run_dir is not None:
                    save_oracle_cross_method_outputs(
                        scenario,
                        study,
                        output_dir=run_dir / scenario.key,
                        beta_reference=beta_reference,
                        samc_cfg=scenario_samc_cfg,
                    )

                summary_df = pd.DataFrame(summary).sort_values(["checkpoint", "label"])
                display(
                    summary_df[
                        [
                            "checkpoint",
                            "label",
                            "mean_estimate",
                            "median_estimate",
                            "rmse",
                            "mean_abs_log10_error",
                            "mean_eval_excl_tuning",
                            "mean_q_tilt_tail_share",
                            "mean_ess",
                            "mean_acceptance_rate",
                            "mean_zero_rate",
                            "mean_samc_max_rel_freq_error",
                        ]
                    ]
                )

            family_rows = []
            for scenario in scenarios:
                meta = scenario.portfolio
                for row in cross_results[scenario.key]["summary"]:
                    family_rows.append(
                        {
                            "scenario": scenario.key,
                            "family": meta.get("family"),
                            "rarity_band": meta.get("rarity_band"),
                            "difficulty": meta.get("expected_difficulty"),
                            **row,
                        }
                    )

            family_df = pd.DataFrame(family_rows)
            display(
                family_df.groupby(["family", "rarity_band", "label", "checkpoint"], as_index=False)
                .agg(
                    mean_rmse=("rmse", "mean"),
                    mean_estimate=("mean_estimate", "mean"),
                    mean_abs_log10_error=("mean_abs_log10_error", "mean"),
                    mean_q_tilt_tail_share=("mean_q_tilt_tail_share", "mean"),
                    mean_ess=("mean_ess", "mean"),
                    mean_acceptance_rate=("mean_acceptance_rate", "mean"),
                )
                .sort_values(["family", "rarity_band", "checkpoint", "label"])
            )
            """
        ),
        markdown_cell("## Review Saved Figures"),
        code_cell(
            """
            if SAVE_OUTPUTS and run_dir is not None:
                print(f"Saved outputs under: {run_dir}")
                for scenario in scenarios:
                    scenario_dir = run_dir / scenario.key
                    print(f"\\n{scenario.key}")
                    display(Image(filename=str(scenario_dir / "cross_method_max_budget.png")))
                    display(Image(filename=str(scenario_dir / "cross_method_convergence.png")))
                    display(Image(filename=str(scenario_dir / "cross_method_convergence_median.png")))
            else:
                print("SAVE_OUTPUTS=False, so no saved figures to display.")
            """
        ),
        markdown_cell("## Reload Saved Results Without Rerunning"),
        code_cell(
            """
            # RELOAD_SCENARIO_DIR = None
            # # Example:
            # # RELOAD_SCENARIO_DIR = project_root / "results" / "cross_method_notebook" / "20260429_120000_cross_method_oracle_compare" / "gwas_additive_score_sig_n100"

            # if RELOAD_SCENARIO_DIR is not None:
            #     saved = load_cross_method_saved_output(RELOAD_SCENARIO_DIR)
            #     print(json.dumps({
            #         "scenario": saved["metadata"]["scenario"],
            #         "exact_p": saved["metadata"]["exact_p"],
            #         "methods": saved["metadata"]["method_order"],
            #     }, indent=2))
            #     regen = regenerate_oracle_cross_method_plots(RELOAD_SCENARIO_DIR)
            #     for name, path in regen.items():
            #         print(name, path)
            #         display(Image(filename=str(path)))
            # else:
            #     print("Set RELOAD_SCENARIO_DIR to a saved scenario directory to regenerate the three article-facing plots.")
            """
        ),
    ]
    return notebook(cells)


def build_oracle_beta_search_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: Oracle Beta Search

            Objective:
            - Use the same six application-style scenarios as the cross-method notebook.
            - Initialize beta from a pilot-only rule, without the discrete scan.
            - Run Optuna HPO over the q-target power `gamma` and swap size, deriving beta from the shared IID pilot for each trial.
            - Evaluate each trial as multiple independent one-chain MCMC-IS runs at the HPO checkpoint.
            - Minimize the per-chain RMSE relative to the exact permutation p-value.
            - Compare the best oracle trial directly against the simple pilot initialization under that same per-chain objective.
            """
        ),
        code_cell(_common_setup_code()),
        code_cell(
            """
            import json
            import numpy as np
            import optuna
            from optuna.samplers import TPESampler
            from perm_pval.methods.beta_tuning import init_beta_from_iid_pilot
            """
        ),
        markdown_cell(
            """
            ## Configuration

            This notebook evaluates one final HPO budget per scenario, while saving intermediate checkpoints every `250k`.
            Each Optuna trial consists of multiple independent one-chain MCMC-IS runs at that checkpoint.
            The objective is the RMSE across those one-chain estimates.
            `HPO_CHAIN_BUDGET` is the budget of each one-chain estimator.
            There is no separate recheck stage and no separate final rerun of the winning configuration.
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            OUTPUT_ROOT = project_root / "results" / "mcmcis_beta_notebook"

            SCENARIO_GROUP = "threshold_suite"
            SCENARIO_KEYS_OVERRIDE = [
                "gwas_additive_score_ultra_n100",
                "poisson_diffmeans_hep_ultra_n200",
                "gwas_additive_score_sig_n100",
                "poisson_diffmeans_hep_sig_n200",
                "gwas_additive_score_above_n100",
                "poisson_diffmeans_hep_above_n200",
            ]
            HPO_CHAIN_COUNT = 18
            HPO_N_JOBS = 6
            HPO_CHAIN_BUDGET = 5_000_000 if not FAST_MODE else 200_000
            HPO_CHECKPOINT_STEP = 250_000 if not FAST_MODE else 50_000
            HPO_ESTIMATION_POINTS = tuple(range(HPO_CHECKPOINT_STEP, HPO_CHAIN_BUDGET + HPO_CHECKPOINT_STEP, HPO_CHECKPOINT_STEP))
            HPO_OBJECTIVE_CHECKPOINT = HPO_CHAIN_BUDGET
            HPO_N_TRIALS = 10 if not FAST_MODE else 5
            HPO_OBJECTIVE_METRIC = "rmse"
            HPO_GAMMA_LADDER = (
                0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
                0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50,
                0.60, 0.75, 0.90, 1.00,
            )
            CANONICAL_GAMMA_SCAN = tuple(round(float(v), 4) for v in np.linspace(0.001, 1.0, 1000))
            HPO_SWAP_CHOICES_DEFAULT = (1, 2, 3)
            HPO_SWAP_CHOICES_BY_SETTING = {
                "gwas_threshold_suite": (1, 2),
                "hep_threshold_suite": (1, 2, 4, 6, 8),
            }
            DEFAULT_BASELINE_SWAP_BY_SETTING = {
                "gwas_threshold_suite": 2,
                "hep_threshold_suite": 6,
            }
            MIN_TAIL_STATES = 2
            BASE_SEED = 54_321

            init_cfg = MCMCWorkflowConfig(
                use_true_p0_for_q_target=False,
                p0_guess=1e-8,
                pilot_samples=200_000 if not FAST_MODE else 1_000,
                scale_method="sd",
                beta_max_init=1e6,
                chains=2,
                thin=1,
                estimate_variance=False,
                proposal_size=1,
            )

            hpo_eval_cfg = BetaSweepStudyConfig(
                estimation_points=HPO_ESTIMATION_POINTS,
                repeats=HPO_CHAIN_COUNT,
                beta_multipliers=(1.0,),
                chains=1,
                thin=1,
                estimate_variance=False,
                chain_n_jobs=1,
                proposal_size=1,
                base_seed=BASE_SEED,
                n_jobs=HPO_N_JOBS,
            )

            def search_reference_p0_for_scenario(scenario) -> float:
                return float(scenario.exact_p)


            def canonical_threshold_for_scenario(scenario) -> float | None:
                threshold = scenario.extra.get("known_significance_threshold")
                if threshold is not None:
                    threshold_f = float(threshold)
                    if threshold_f == threshold_f and threshold_f > 0.0:
                        return threshold_f
                return None


            def beta_from_gamma_reference(
                *,
                gamma: float,
                p0_reference: float,
                pilot_t: np.ndarray,
                sigma_t: float,
                problem,
                beta_max: float,
            ) -> dict[str, float]:
                gamma_f = float(gamma)
                q_target = float(p0_reference ** gamma_f)
                beta = float(
                    init_beta_from_iid_pilot(
                        pilot_T=pilot_t,
                        T_obs=problem.t_obs,
                        sigma_T=sigma_t,
                        p0=float(p0_reference),
                        q_target=q_target,
                        beta_max=float(beta_max),
                    )
                )
                return {
                    "gamma": gamma_f,
                    "q_target": q_target,
                    "beta": beta,
                }


            def implied_gamma_for_beta_under_reference(
                *,
                target_beta: float,
                p0_reference: float | None,
                pilot_t: np.ndarray,
                sigma_t: float,
                problem,
                beta_max: float,
                gamma_grid: tuple[float, ...],
            ) -> dict[str, float | int | None]:
                if p0_reference is None or not np.isfinite(float(p0_reference)) or float(p0_reference) <= 0.0:
                    return {
                        "gamma": None,
                        "q_target": None,
                        "beta": None,
                        "beta_abs_error": None,
                        "beta_log_error": None,
                        "grid_size": int(len(gamma_grid)),
                    }
                rows = [
                    beta_from_gamma_reference(
                        gamma=float(gamma),
                        p0_reference=float(p0_reference),
                        pilot_t=pilot_t,
                        sigma_t=float(sigma_t),
                        problem=problem,
                        beta_max=float(beta_max),
                    )
                    for gamma in gamma_grid
                ]
                target_beta_f = float(target_beta)
                eps = 1e-12
                best = min(
                    rows,
                    key=lambda row: abs(np.log(max(float(row["beta"]), eps)) - np.log(max(target_beta_f, eps))),
                )
                best_beta = float(best["beta"])
                return {
                    "gamma": float(best["gamma"]),
                    "q_target": float(best["q_target"]),
                    "beta": best_beta,
                    "beta_abs_error": float(abs(best_beta - target_beta_f)),
                    "beta_log_error": float(abs(np.log(max(best_beta, eps)) - np.log(max(target_beta_f, eps)))),
                    "grid_size": int(len(gamma_grid)),
                }


            def swap_choices_for_scenario(scenario) -> tuple[int, ...]:
                setting_key = str(scenario.extra.get("application_setting_key", ""))
                if setting_key in HPO_SWAP_CHOICES_BY_SETTING:
                    return tuple(int(v) for v in HPO_SWAP_CHOICES_BY_SETTING[setting_key])
                return tuple(int(v) for v in HPO_SWAP_CHOICES_DEFAULT)


            def baseline_swap_for_scenario(scenario) -> int:
                setting_key = str(scenario.extra.get("application_setting_key", ""))
                if setting_key in DEFAULT_BASELINE_SWAP_BY_SETTING:
                    return int(DEFAULT_BASELINE_SWAP_BY_SETTING[setting_key])
                return int(swap_choices_for_scenario(scenario)[0])


            def trial_objective_value(summary_row: dict, metric: str) -> float:
                metric_val = float(summary_row[metric])
                if not np.isfinite(metric_val):
                    return float("inf")
                return metric_val


            print(json.dumps({
                "FAST_MODE": FAST_MODE,
                "SCENARIO_GROUP": SCENARIO_GROUP,
                "SCENARIO_KEYS_OVERRIDE": SCENARIO_KEYS_OVERRIDE,
                "HPO_CHAIN_BUDGET": HPO_CHAIN_BUDGET,
                "HPO_CHECKPOINT_STEP": HPO_CHECKPOINT_STEP,
                "HPO_OBJECTIVE_CHECKPOINT": HPO_OBJECTIVE_CHECKPOINT,
                "HPO_N_CHECKPOINTS": len(HPO_ESTIMATION_POINTS),
                "HPO_N_TRIALS": HPO_N_TRIALS,
                "HPO_OBJECTIVE_METRIC": HPO_OBJECTIVE_METRIC,
                "HPO_GAMMA_LADDER": list(HPO_GAMMA_LADDER),
                "HPO_SWAP_CHOICES_BY_SETTING": HPO_SWAP_CHOICES_BY_SETTING,
                "HPO_ONE_CHAIN_RUNS_PER_TRIAL": int(hpo_eval_cfg.repeats),
                "HPO_CHAIN_BUDGET_PER_RUN": int(HPO_CHAIN_BUDGET),
                "HPO_TOP_LEVEL_N_JOBS": int(hpo_eval_cfg.n_jobs),
                "HPO_CHAIN_N_JOBS": int(hpo_eval_cfg.chain_n_jobs),
                "SAVE_OUTPUTS": SAVE_OUTPUTS,
                "REFERENCE_P0_MODE": "oracle_exact_p__simple_known_threshold_else_exact_p",
                "CANONICAL_GAMMA_SCAN_POINTS": int(len(CANONICAL_GAMMA_SCAN)),
            }, indent=2))
            """
        ),
        markdown_cell("## Load Scenarios"),
        code_cell(
            """
            scenarios = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=SCENARIO_KEYS_OVERRIDE,
                portfolio_group=None if SCENARIO_KEYS_OVERRIDE is not None else SCENARIO_GROUP,
                min_tail_states=MIN_TAIL_STATES,
            )

            run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "oracle_beta_search") if SAVE_OUTPUTS else None

            pd.DataFrame([
                {
                    "scenario": s.key,
                    "exact_p": s.exact_p,
                    "tail_hits": s.exact_tail_hits,
                    "n_perm": s.exact_n_perm,
                    "exact_method": s.exact_method,
                    "family": s.portfolio.get("family"),
                    "rarity_band": s.portfolio.get("rarity_band"),
                    "difficulty": s.portfolio.get("expected_difficulty"),
                }
                for s in scenarios
            ])
            """
        ),
        markdown_cell("## Run Oracle Beta HPO"),
        code_cell(
            """
            oracle_results = {}

            for scenario_idx, scenario in enumerate(scenarios):
                scenario_seed = BASE_SEED + 50_000 * scenario_idx
                search_reference_p0 = search_reference_p0_for_scenario(scenario)
                canonical_threshold_p0 = canonical_threshold_for_scenario(scenario)
                simple_reference_p0 = (
                    float(canonical_threshold_p0)
                    if canonical_threshold_p0 is not None
                    else float(search_reference_p0)
                )
                init_payload = build_beta_initialization(
                    scenario.problem,
                    scenario.exact_p,
                    init_cfg,
                    seed=scenario_seed,
                    p0_reference=simple_reference_p0,
                )
                beta_init = float(init_payload["beta0_laplace"])
                sigma_t = float(init_payload["sigma_t"])
                pilot_t_shared = np.asarray(init_payload["pilot_t"], dtype=float)
                simple_gamma = float(init_cfg.d_alpha)
                swap_choices = swap_choices_for_scenario(scenario)
                baseline_swap = baseline_swap_for_scenario(scenario)
                trial_rows = []
                trial_checkpoint_rows = []
                trial_record_rows = []
                scenario_dir = (run_dir / scenario.key) if (SAVE_OUTPUTS and run_dir is not None) else None
                if scenario_dir is not None:
                    scenario_dir.mkdir(parents=True, exist_ok=True)

                print(json.dumps({
                    "scenario": scenario.key,
                    "exact_p": scenario.exact_p,
                    "application_setting_key": scenario.extra.get("application_setting_key"),
                    "known_significance_threshold": scenario.extra.get("known_significance_threshold"),
                    "search_reference_p0": search_reference_p0,
                    "simple_reference_p0": simple_reference_p0,
                    "canonical_threshold_p0": canonical_threshold_p0,
                    "beta0_formula": init_payload["beta0_formula"],
                    "beta0_laplace": beta_init,
                    "simple_gamma": simple_gamma,
                    "q_target": init_payload["q_target"],
                    "sigma_t": sigma_t,
                    "pilot_samples": int(init_cfg.pilot_samples),
                    "swap_choices": swap_choices,
                    "baseline_swap": baseline_swap,
                }, indent=2))

                init_eval = run_named_mcmc_checkpoint_study(
                    scenario.problem,
                    scenario.exact_p,
                    config_specs=[
                        {
                            "label": "simple_init",
                            "config_id": "simple_init",
                            "beta": float(beta_init),
                            "proposal_size": int(baseline_swap),
                            "source": "pilot_init",
                        }
                    ],
                    sigma_t=sigma_t,
                    estimation_points=HPO_ESTIMATION_POINTS,
                    repeats=int(hpo_eval_cfg.repeats),
                    base_seed=BASE_SEED + 250_000 + 25_000 * scenario_idx,
                    template_cfg=hpo_eval_cfg,
                    n_jobs=int(hpo_eval_cfg.n_jobs),
                )
                init_summary_rows = [dict(row) for row in init_eval["summary"]]
                init_summary = next(row for row in init_summary_rows if int(row["checkpoint"]) == int(HPO_OBJECTIVE_CHECKPOINT))
                init_summary["label"] = "simple_init"
                init_summary["beta"] = float(beta_init)
                init_summary["gamma"] = float(simple_gamma)
                init_summary["q_target"] = float(simple_reference_p0 ** simple_gamma)
                init_summary["reference_p0"] = float(simple_reference_p0)
                init_summary["proposal_size"] = int(baseline_swap)
                init_summary["source"] = "pilot_init"
                init_summary_json = json.loads(pd.DataFrame([init_summary]).to_json(orient="records"))[0]

                def objective(trial):
                    gamma = float(trial.suggest_categorical("gamma", list(HPO_GAMMA_LADDER)))
                    proposal_size = int(trial.suggest_categorical("proposal_size", list(swap_choices)))
                    beta_payload = beta_from_gamma_reference(
                        gamma=gamma,
                        p0_reference=float(search_reference_p0),
                        pilot_t=pilot_t_shared,
                        sigma_t=float(sigma_t),
                        problem=scenario.problem,
                        beta_max=float(init_cfg.beta_max_init),
                    )
                    q_target = float(beta_payload["q_target"])
                    beta = float(beta_payload["beta"])
                    trial_eval = run_named_mcmc_checkpoint_study(
                        scenario.problem,
                        scenario.exact_p,
                        config_specs=[
                            {
                                "label": "oracle_trial",
                                "config_id": f"trial_{trial.number}",
                                "beta": beta,
                                "proposal_size": proposal_size,
                                "source": "oracle_hpo",
                            }
                        ],
                        sigma_t=sigma_t,
                        estimation_points=HPO_ESTIMATION_POINTS,
                        repeats=int(hpo_eval_cfg.repeats),
                        base_seed=BASE_SEED + 1_000_000 * scenario_idx + 10_000 * trial.number,
                        template_cfg=hpo_eval_cfg,
                        n_jobs=int(hpo_eval_cfg.n_jobs),
                    )
                    summary_rows = [dict(row) for row in trial_eval["summary"]]
                    summary_row = next(row for row in summary_rows if int(row["checkpoint"]) == int(HPO_OBJECTIVE_CHECKPOINT))
                    objective_value = trial_objective_value(summary_row, HPO_OBJECTIVE_METRIC)
                    for checkpoint_row in summary_rows:
                        trial_checkpoint_rows.append(
                            {
                                "trial_number": int(trial.number),
                                "scenario": scenario.key,
                                "gamma": gamma,
                                "q_target": q_target,
                                "beta": beta,
                                "proposal_size": proposal_size,
                                **checkpoint_row,
                            }
                        )
                    for record_row in trial_eval["records"]:
                        trial_record_rows.append(
                            {
                                "trial_number": int(trial.number),
                                "scenario": scenario.key,
                                "gamma": gamma,
                                "q_target": q_target,
                                "beta": beta,
                                "proposal_size": proposal_size,
                                **dict(record_row),
                            }
                        )
                    trial_payload = {
                        "trial_number": int(trial.number),
                        "scenario": scenario.key,
                        "gamma": gamma,
                        "q_target": q_target,
                        "beta": beta,
                        "proposal_size": proposal_size,
                        "objective_metric": HPO_OBJECTIVE_METRIC,
                        "objective_checkpoint": int(HPO_OBJECTIVE_CHECKPOINT),
                        "objective_value": float(objective_value),
                        "reference_p0": float(search_reference_p0),
                        **summary_row,
                    }
                    trial_rows.append(trial_payload)
                    if scenario_dir is not None:
                        pd.DataFrame(trial_rows).to_json(
                            scenario_dir / "hpo_trials.jsonl",
                            orient="records",
                            lines=True,
                        )
                        pd.DataFrame(trial_checkpoint_rows).to_json(
                            scenario_dir / "hpo_trial_checkpoint_summaries.jsonl",
                            orient="records",
                            lines=True,
                        )
                        pd.DataFrame(trial_record_rows).to_json(
                            scenario_dir / "hpo_trial_records.jsonl",
                            orient="records",
                            lines=True,
                        )
                        (scenario_dir / "hpo_status.json").write_text(
                            json.dumps(
                                {
                                    "scenario": scenario.key,
                                    "completed_trials": len(trial_rows),
                                    "target_trials": int(HPO_N_TRIALS),
                                    "latest_trial_number": int(trial.number),
                                    "objective": "per_chain_rmse",
                                    "objective_checkpoint": int(HPO_OBJECTIVE_CHECKPOINT),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    for key, value in trial_payload.items():
                        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                            trial.set_user_attr(str(key), value)
                    return float(objective_value)

                def persist_hpo_state(study, trial):
                    if scenario_dir is None or len(study.trials) == 0:
                        return
                    best_trial = study.best_trial
                    best_gamma = float(best_trial.params["gamma"])
                    best_q_target = float(search_reference_p0 ** best_gamma)
                    best_beta = float(best_trial.user_attrs["beta"])
                    implied_canonical = implied_gamma_for_beta_under_reference(
                        target_beta=best_beta,
                        p0_reference=canonical_threshold_p0,
                        pilot_t=pilot_t_shared,
                        sigma_t=float(sigma_t),
                        problem=scenario.problem,
                        beta_max=float(init_cfg.beta_max_init),
                        gamma_grid=CANONICAL_GAMMA_SCAN,
                    )
                    best_payload = {
                        "scenario": scenario.key,
                        "reference_p0": float(search_reference_p0),
                        "simple_reference_p0": float(simple_reference_p0),
                        "canonical_threshold_p0": (
                            float(canonical_threshold_p0)
                            if canonical_threshold_p0 is not None
                            else None
                        ),
                        "beta_init": float(beta_init),
                        "beta0_formula": float(init_payload["beta0_formula"]),
                        "beta0_laplace": float(init_payload["beta0_laplace"]),
                        "sigma_t": float(sigma_t),
                        "best_gamma": best_gamma,
                        "best_q_target": best_q_target,
                        "best_beta": best_beta,
                        "implied_gamma_canonical": implied_canonical["gamma"],
                        "implied_q_target_canonical": implied_canonical["q_target"],
                        "implied_beta_canonical": implied_canonical["beta"],
                        "implied_beta_abs_error_canonical": implied_canonical["beta_abs_error"],
                        "implied_beta_log_error_canonical": implied_canonical["beta_log_error"],
                        "best_proposal_size": int(best_trial.params["proposal_size"]),
                        "best_objective_value": float(study.best_value),
                        "best_trial_number": int(best_trial.number),
                        "completed_trials": int(len(study.trials)),
                    }
                    (scenario_dir / "best_config.partial.json").write_text(
                        json.dumps(best_payload, indent=2),
                        encoding="utf-8",
                    )
                    pd.DataFrame(
                        [
                            {
                                "trial_number": int(t.number),
                                "state": str(t.state),
                                "value": float(t.value) if t.value is not None else None,
                                **{str(k): v for k, v in t.params.items()},
                            }
                            for t in study.trials
                        ]
                    ).to_json(
                        scenario_dir / "optuna_trials.json",
                        orient="records",
                        indent=2,
                    )

                study = optuna.create_study(
                    direction="minimize",
                    sampler=TPESampler(seed=BASE_SEED + 10_000 * scenario_idx + 7),
                    study_name=f"{scenario.key}_oracle_beta",
                )
                study.optimize(
                    objective,
                    n_trials=HPO_N_TRIALS,
                    show_progress_bar=False,
                        callbacks=[persist_hpo_state],
                    )

                best_trial = study.best_trial
                best_gamma = float(best_trial.params["gamma"])
                best_q_target = float(search_reference_p0 ** best_gamma)
                best_beta = float(best_trial.user_attrs["beta"])
                best_proposal_size = int(best_trial.params["proposal_size"])
                best_source = "oracle_hpo"
                best_trial_number = int(best_trial.number)
                implied_canonical = implied_gamma_for_beta_under_reference(
                    target_beta=best_beta,
                    p0_reference=canonical_threshold_p0,
                    pilot_t=pilot_t_shared,
                    sigma_t=float(sigma_t),
                    problem=scenario.problem,
                    beta_max=float(init_cfg.beta_max_init),
                    gamma_grid=CANONICAL_GAMMA_SCAN,
                )
                best_trial_payload = dict(best_trial.user_attrs)
                best_trial_payload.update(
                    {
                        "trial_number": int(best_trial.number),
                        "gamma": best_gamma,
                        "q_target": best_q_target,
                        "beta": float(best_beta),
                        "proposal_size": int(best_proposal_size),
                        "objective_metric": HPO_OBJECTIVE_METRIC,
                        "objective_checkpoint": int(HPO_OBJECTIVE_CHECKPOINT),
                        "objective_value": float(study.best_value),
                        "reference_p0": float(search_reference_p0),
                        "canonical_threshold_p0": (
                            float(canonical_threshold_p0)
                            if canonical_threshold_p0 is not None
                            else None
                        ),
                        "implied_gamma_canonical": implied_canonical["gamma"],
                        "implied_q_target_canonical": implied_canonical["q_target"],
                        "implied_beta_canonical": implied_canonical["beta"],
                        "label": "oracle",
                    }
                )
                best_trial_payload_json = json.loads(pd.DataFrame([best_trial_payload]).to_json(orient="records"))[0]

                best_config = {
                    "scenario": scenario.key,
                    "reference_p0": float(search_reference_p0),
                    "simple_reference_p0": float(simple_reference_p0),
                    "canonical_threshold_p0": (
                        float(canonical_threshold_p0)
                        if canonical_threshold_p0 is not None
                        else None
                    ),
                    "beta_init": float(beta_init),
                    "simple_gamma": float(simple_gamma),
                    "beta0_formula": float(init_payload["beta0_formula"]),
                    "beta0_laplace": float(init_payload["beta0_laplace"]),
                    "sigma_t": float(sigma_t),
                    "best_gamma": best_gamma,
                    "best_q_target": best_q_target,
                    "best_beta": best_beta,
                    "implied_gamma_canonical": implied_canonical["gamma"],
                    "implied_q_target_canonical": implied_canonical["q_target"],
                    "implied_beta_canonical": implied_canonical["beta"],
                    "implied_beta_abs_error_canonical": implied_canonical["beta_abs_error"],
                    "implied_beta_log_error_canonical": implied_canonical["beta_log_error"],
                    "best_proposal_size": int(best_proposal_size),
                    "best_objective_value": float(study.best_value),
                    "best_trial_number": int(best_trial_number),
                    "n_trials": int(len(study.trials)),
                    "selection_source": best_source,
                }

                oracle_results[scenario.key] = {
                    "best_config": best_config,
                    "trial_rows": list(trial_rows),
                    "trial_checkpoint_rows": list(trial_checkpoint_rows),
                    "trial_record_rows": list(trial_record_rows),
                    "init_summary": dict(init_summary_json),
                    "best_trial_summary": dict(best_trial_payload_json),
                    "init_payload": {
                        key: value
                        for key, value in init_payload.items()
                        if key != "pilot_t"
                    },
                }

                if SAVE_OUTPUTS and run_dir is not None:
                    pd.DataFrame(trial_rows).to_json(
                        scenario_dir / "hpo_trials.jsonl",
                        orient="records",
                        lines=True,
                    )
                    pd.DataFrame(trial_checkpoint_rows).to_json(
                        scenario_dir / "hpo_trial_checkpoint_summaries.jsonl",
                        orient="records",
                        lines=True,
                    )
                    pd.DataFrame(trial_record_rows).to_json(
                        scenario_dir / "hpo_trial_records.jsonl",
                        orient="records",
                        lines=True,
                    )
                    pd.DataFrame(init_eval["records"]).to_json(
                        scenario_dir / "simple_init_records.jsonl",
                        orient="records",
                        lines=True,
                    )
                    pd.DataFrame(init_summary_rows).to_json(
                        scenario_dir / "simple_init_checkpoint_summaries.json",
                        orient="records",
                        indent=2,
                    )
                    (scenario_dir / "simple_init_summary.json").write_text(
                        json.dumps(init_summary_json, indent=2),
                        encoding="utf-8",
                    )
                    (scenario_dir / "oracle_best_trial.json").write_text(
                        json.dumps(best_trial_payload_json, indent=2),
                        encoding="utf-8",
                    )
                    (scenario_dir / "best_config.json").write_text(
                        json.dumps(best_config, indent=2),
                        encoding="utf-8",
                    )
                    (scenario_dir / "metadata.json").write_text(
                        json.dumps(
                            {
                                "scenario": scenario.key,
                                "scenario_display": scenario.description,
                                "exact_p": float(scenario.exact_p),
                                "known_significance_threshold": scenario.extra.get("known_significance_threshold"),
                                "application_setting_key": scenario.extra.get("application_setting_key"),
                                "reference_p0": float(search_reference_p0),
                                "reference_p0_mode": "exact_p",
                                "simple_reference_p0": float(simple_reference_p0),
                                "simple_reference_p0_mode": (
                                    "known_significance_threshold"
                                    if canonical_threshold_p0 is not None
                                    else "exact_p"
                                ),
                                "canonical_threshold_p0": (
                                    float(canonical_threshold_p0)
                                    if canonical_threshold_p0 is not None
                                    else None
                                ),
                                "init_payload": oracle_results[scenario.key]["init_payload"],
                                "hpo_settings": {
                                    "n_trials": int(HPO_N_TRIALS),
                                    "objective_definition": "rmse_across_one_chain_estimates",
                                    "objective_metric": str(HPO_OBJECTIVE_METRIC),
                                    "gamma_ladder": [float(v) for v in HPO_GAMMA_LADDER],
                                    "canonical_gamma_scan": {
                                        "min": float(CANONICAL_GAMMA_SCAN[0]),
                                        "max": float(CANONICAL_GAMMA_SCAN[-1]),
                                        "n_points": int(len(CANONICAL_GAMMA_SCAN)),
                                    },
                                    "swap_choices": [int(v) for v in swap_choices],
                                    "one_chain_runs_per_trial": int(hpo_eval_cfg.repeats),
                                    "chain_budget": int(HPO_CHAIN_BUDGET),
                                    "checkpoint_step": int(HPO_CHECKPOINT_STEP),
                                    "estimation_points": [int(v) for v in HPO_ESTIMATION_POINTS],
                                    "objective_checkpoint": int(HPO_OBJECTIVE_CHECKPOINT),
                                    "chain_n_jobs": int(hpo_eval_cfg.chain_n_jobs),
                                    "top_level_n_jobs": int(hpo_eval_cfg.n_jobs),
                                    "default_gamma": float(simple_gamma),
                                },
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                best_row = pd.DataFrame([best_config])
                comparison_df = pd.DataFrame([
                    dict(init_summary_json),
                    dict(best_trial_payload_json),
                ]).sort_values("label")
                display(best_row)
                display(comparison_df[[
                    "checkpoint",
                    "label",
                    "mean_estimate",
                    "rmse",
                    "mean_abs_log10_error",
                    "mean_variance_estimate",
                    "mean_q_tilt_tail_share",
                    "mean_ess",
                    "mean_acceptance_rate",
                    "mean_weight_cv",
                ]])
            """
        ),
        markdown_cell("## Review Saved Outputs"),
        code_cell(
            """
            if SAVE_OUTPUTS and run_dir is not None:
                print(f"Saved outputs under: {run_dir}")
                display(pd.DataFrame([
                    {
                        "scenario": key,
                        "simple_gamma": payload["best_config"]["simple_gamma"],
                        "simple_beta": payload["best_config"]["beta_init"],
                        "best_gamma": payload["best_config"]["best_gamma"],
                        "best_beta": payload["best_config"]["best_beta"],
                        "best_proposal_size": payload["best_config"]["best_proposal_size"],
                        "simple_rmse": payload["init_summary"]["rmse"],
                        "oracle_rmse": payload["best_trial_summary"]["rmse"],
                        "best_objective_value": payload["best_config"]["best_objective_value"],
                    }
                    for key, payload in oracle_results.items()
                ]))
            else:
                print("SAVE_OUTPUTS=False, so no saved outputs to display.")
            """
        ),
        markdown_cell("## Reload Saved Results Without Rerunning"),
        code_cell(
            """
            # RELOAD_BETA_DIR = None
            # # Example:
            # # RELOAD_BETA_DIR = project_root / "results" / "mcmcis_beta_notebook" / "20260411_120000_oracle_beta_search" / "gwas_additive_score_sig_n100"

            # if RELOAD_BETA_DIR is not None:
            #     print(json.loads((RELOAD_BETA_DIR / "best_config.json").read_text()))
            #     display(pd.DataFrame([
            #         json.loads((RELOAD_BETA_DIR / "simple_init_summary.json").read_text()),
            #         json.loads((RELOAD_BETA_DIR / "oracle_best_trial.json").read_text()),
            #     ]))
            #     display(pd.read_json(RELOAD_BETA_DIR / "hpo_trials.jsonl", lines=True).sort_values("objective_value").head())
            # else:
            #     print("Set RELOAD_BETA_DIR to a saved oracle-beta directory to inspect saved results from disk only.")
            """
        ),
    ]
    return notebook(cells)


def build_mcmc_objective_grid_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: Offline MCMC-IS Objective Grid Study

            Objective:
            - Evaluate a deterministic, production-like discrete hyperparameter grid for MCMC-IS.
            - Compare realistic production objectives against oracle RMSE on the same scenarios used in the production notebooks.
            - Quantify objective noise across repeat seeds without running the old long final-budget follow-up.
            """
        ),
        code_cell(_common_setup_code()),
        code_cell(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            """
        ),
        markdown_cell(
            """
            ## Configuration

            This notebook is intentionally heavy by default.  
            It is an offline objective study, not a long-run estimator comparison.
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            OUTPUT_ROOT = project_root / "results" / "mcmcis_objective_grid"

            SCENARIO_GROUP = "exploratory_exact"
            SCENARIO_KEYS_OVERRIDE = None

            Q_MULTIPLIERS = DEFAULT_MCMC_OBJECTIVE_GRID_Q_MULTIPLIERS
            N_SWAP_PAIRS = DEFAULT_MCMC_OBJECTIVE_GRID_SWAP_COUNTS
            TRIAL_REPEATS = 5 if not FAST_MODE else 2
            TRIAL_BUDGET = 200_000 if not FAST_MODE else 20_000
            EXTRA_TRIAL_BUDGETS = tuple()
            Q_FLOOR = 1e-12
            N_JOBS = min(os.cpu_count() or 1, len(Q_MULTIPLIERS) * len(N_SWAP_PAIRS) * TRIAL_REPEATS)
            MIN_TAIL_STATES = 2
            BASE_SEED = 31_415

            mcmc_cfg = MCMCWorkflowConfig(
                pilot_samples=20_000 if not FAST_MODE else 1_000,
                scale_method="sd",
                beta_max_init=1e6,
                chains=2,
                burn_in_fraction=0.20,
                thin=1,
                estimate_variance=True,
                obm_batch_size=None,
                tilt_mode="smooth_hinge",
                proposal_size=0.075,
            )

            NOTEBOOK_CONFIG = {
                "FAST_MODE": FAST_MODE,
                "SCENARIO_GROUP": SCENARIO_GROUP,
                "SCENARIO_KEYS_OVERRIDE": SCENARIO_KEYS_OVERRIDE,
                "Q_MULTIPLIERS": Q_MULTIPLIERS,
                "N_SWAP_PAIRS": N_SWAP_PAIRS,
                "TRIAL_REPEATS": TRIAL_REPEATS,
                "TRIAL_BUDGET": TRIAL_BUDGET,
                "EXTRA_TRIAL_BUDGETS": EXTRA_TRIAL_BUDGETS,
                "Q_FLOOR": Q_FLOOR,
                "N_JOBS": N_JOBS,
                "BASE_SEED": BASE_SEED,
            }

            print(json.dumps(NOTEBOOK_CONFIG, indent=2))
            """
        ),
        markdown_cell("## Notebook Helpers"),
        code_cell(
            """
            REALISTIC_OBJECTIVES = list(MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES)


            def plot_oracle_rmse_heatmap(
                config_summary: list[dict],
                q_multipliers: tuple[float, ...],
                n_swap_pairs_values: tuple[int, ...],
                scenario_name: str,
                *,
                save_path: Path | None = None,
            ) -> None:
                df = pd.DataFrame(config_summary)
                heat = np.full((len(n_swap_pairs_values), len(q_multipliers)), np.nan, dtype=float)
                for _, row in df.iterrows():
                    swap_idx = list(n_swap_pairs_values).index(int(row["n_swap_pairs"]))
                    q_idx = int(row["q_index"])
                    val = float(row["mean_oracle_rmse"])
                    heat[swap_idx, q_idx] = np.log10(val) if np.isfinite(val) and val > 0.0 else np.nan

                fig, ax = plt.subplots(figsize=(13, 4.5))
                im = ax.imshow(heat, aspect="auto", cmap="viridis")
                ax.set_title(f"Oracle RMSE grid: {scenario_name}")
                ax.set_xlabel("q_multiplier")
                ax.set_ylabel("n_swap_pairs")
                ax.set_xticks(np.arange(len(q_multipliers)))
                ax.set_xticklabels([f"{v:g}" for v in q_multipliers], rotation=45, ha="right")
                ax.set_yticks(np.arange(len(n_swap_pairs_values)))
                ax.set_yticklabels([str(v) for v in n_swap_pairs_values])
                fig.colorbar(im, ax=ax, label="log10(mean oracle RMSE)")
                plt.tight_layout()
                if save_path is not None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path, dpi=170, bbox_inches="tight")
                plt.close(fig)


            def plot_objective_seed_noise_heatmap(
                seed_noise_rows: list[dict],
                scenario_name: str,
                *,
                save_path: Path | None = None,
            ) -> None:
                df = pd.DataFrame(seed_noise_rows)
                metrics = ["exact_match_rate", "mean_fuzzy_similarity"]
                heat = np.asarray([[float(df.loc[df["objective_name"] == obj, metric].iloc[0]) for metric in metrics] for obj in REALISTIC_OBJECTIVES], dtype=float)
                fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(REALISTIC_OBJECTIVES))))
                im = ax.imshow(heat, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
                ax.set_title(f"Objective seed-noise summary: {scenario_name}")
                ax.set_xticks(np.arange(len(metrics)))
                ax.set_xticklabels(metrics, rotation=20, ha="right")
                ax.set_yticks(np.arange(len(REALISTIC_OBJECTIVES)))
                ax.set_yticklabels(REALISTIC_OBJECTIVES)
                fig.colorbar(im, ax=ax, label="score")
                plt.tight_layout()
                if save_path is not None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path, dpi=170, bbox_inches="tight")
                plt.close(fig)


            def build_cross_scenario_leaderboard(objective_rows: list[dict]) -> pd.DataFrame:
                df = pd.DataFrame([row for row in objective_rows if row["objective_kind"] == "realistic"])
                leaderboard = (
                    df.groupby("objective_name", as_index=False)
                    .agg(
                        exact_match_count=("oracle_exact_match", "sum"),
                        mean_fuzzy_similarity=("oracle_fuzzy_similarity", "mean"),
                        mean_q_index_distance=("oracle_q_index_distance", "mean"),
                        mean_swap_distance=("oracle_swap_distance", "mean"),
                    )
                    .sort_values(
                        ["exact_match_count", "mean_fuzzy_similarity", "mean_q_index_distance", "mean_swap_distance"],
                        ascending=[False, False, True, True],
                    )
                    .reset_index(drop=True)
                )
                return leaderboard


            def plot_cross_scenario_fuzzy_similarity(
                objective_rows: list[dict],
                *,
                save_path: Path | None = None,
            ) -> None:
                df = pd.DataFrame([row for row in objective_rows if row["objective_kind"] == "realistic"])
                scenarios = list(dict.fromkeys(df["scenario_key"]))
                heat = np.full((len(REALISTIC_OBJECTIVES), len(scenarios)), np.nan, dtype=float)
                for _, row in df.iterrows():
                    obj_idx = REALISTIC_OBJECTIVES.index(str(row["objective_name"]))
                    scn_idx = scenarios.index(str(row["scenario_key"]))
                    heat[obj_idx, scn_idx] = float(row["oracle_fuzzy_similarity"])

                fig, ax = plt.subplots(figsize=(10, max(4.5, 0.35 * len(REALISTIC_OBJECTIVES))))
                im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
                ax.set_title("Cross-scenario fuzzy similarity to oracle RMSE")
                ax.set_xticks(np.arange(len(scenarios)))
                ax.set_xticklabels(scenarios, rotation=25, ha="right")
                ax.set_yticks(np.arange(len(REALISTIC_OBJECTIVES)))
                ax.set_yticklabels(REALISTIC_OBJECTIVES)
                fig.colorbar(im, ax=ax, label="fuzzy similarity")
                plt.tight_layout()
                if save_path is not None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path, dpi=170, bbox_inches="tight")
                plt.close(fig)
            """
        ),
        markdown_cell("## Load Scenarios"),
        code_cell(
            """
            scenarios = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=SCENARIO_KEYS_OVERRIDE,
                portfolio_group=None if SCENARIO_KEYS_OVERRIDE is not None else SCENARIO_GROUP,
                min_tail_states=MIN_TAIL_STATES,
            )

            run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "objective_grid") if SAVE_OUTPUTS else None

            pd.DataFrame([
                {
                    "scenario": s.key,
                    "exact_p": s.exact_p,
                    "tail_hits": s.exact_tail_hits,
                    "n_perm": s.exact_n_perm,
                    "exact_method": s.exact_method,
                    "family": s.portfolio.get("family"),
                    "rarity_band": s.portfolio.get("rarity_band"),
                    "difficulty": s.portfolio.get("expected_difficulty"),
                    "groups": ",".join(s.portfolio.get("groups", [])),
                }
                for s in scenarios
            ])
            """
        ),
        markdown_cell("## Run Offline Objective Grid Study"),
        code_cell(
            """
            grid_results = {}
            cross_scenario_objective_rows = []

            for scenario_idx, scenario in enumerate(scenarios):
                print(f"Running offline objective grid for {scenario.key} | exact p={scenario.exact_p:.3e}")
                study_seed = BASE_SEED + 10_000 * (scenario_idx + 1)
                grid_study = run_mcmc_objective_grid_study(
                    scenario.problem,
                    scenario.exact_p,
                    mcmc_cfg=mcmc_cfg,
                    trial_repeats=TRIAL_REPEATS,
                    trial_budget=TRIAL_BUDGET,
                    base_seed=study_seed,
                    q_multipliers=Q_MULTIPLIERS,
                    n_swap_pairs_values=N_SWAP_PAIRS,
                    q_floor=Q_FLOOR,
                    n_jobs=N_JOBS,
                )

                objective_winners_df = pd.DataFrame(grid_study["objective_winners"]).sort_values(["objective_kind", "objective_name"])
                objective_winners_df["scenario_key"] = scenario.key
                objective_winners_df["family"] = scenario.portfolio.get("family")
                objective_winners_df["rarity_band"] = scenario.portfolio.get("rarity_band")
                cross_scenario_objective_rows.extend(objective_winners_df.to_dict(orient="records"))

                scenario_dir = (run_dir / scenario.key) if (SAVE_OUTPUTS and run_dir is not None) else None
                if scenario_dir is not None:
                    plot_oracle_rmse_heatmap(
                        grid_study["config_summary"],
                        Q_MULTIPLIERS,
                        N_SWAP_PAIRS,
                        scenario.description,
                        save_path=scenario_dir / "oracle_rmse_heatmap.png",
                    )
                    plot_objective_seed_noise_heatmap(
                        grid_study["objective_seed_noise"],
                        scenario.description,
                        save_path=scenario_dir / "objective_seed_noise_heatmap.png",
                    )
                    save_mcmc_objective_grid_outputs(
                        grid_study,
                        output_dir=scenario_dir,
                        scenario_name=scenario.description,
                        exact_p=scenario.exact_p,
                        notebook_config=NOTEBOOK_CONFIG,
                    )

                grid_results[scenario.key] = {
                    "grid_study": grid_study,
                }

                print(json.dumps({
                    "scenario": scenario.key,
                    "sigma_t": grid_study["study_context"]["sigma_t"],
                    "q_target": grid_study["study_context"]["q_target"],
                    "oracle_winner": grid_study["oracle_winner"],
                }, indent=2))

                display(pd.DataFrame(grid_study["config_summary"]).sort_values("mean_oracle_rmse").head(12)[[
                    "config_id",
                    "beta",
                    "n_swap_pairs",
                    "q_multiplier",
                    "q_trial",
                    "mean_oracle_rmse",
                    "mean_oracle_abs_log10",
                    "mean_objective_varhat",
                    "mean_objective_varhat_qmatch_soft",
                    "mean_objective_varhat_degeneracy_soft",
                    "mean_objective_varhat_qmatch_degeneracy_soft",
                    "mean_tail_hits",
                    "mean_acceptance_rate",
                    "mean_ess",
                    "mean_weight_cv",
                ]])
                display(objective_winners_df[[
                    "objective_name",
                    "config_id",
                    "q_multiplier",
                    "n_swap_pairs",
                    "beta",
                    "selected_objective_value",
                    "oracle_exact_match",
                    "oracle_fuzzy_similarity",
                    "oracle_q_index_distance",
                    "oracle_swap_distance",
                ]])
                display(pd.DataFrame(grid_study["objective_seed_noise"]).sort_values("objective_name"))
            """
        ),
        markdown_cell("## Cross-Scenario Objective Summary"),
        code_cell(
            """
            cross_scenario_df = pd.DataFrame(cross_scenario_objective_rows)
            realistic_cross_scenario_df = cross_scenario_df[cross_scenario_df["objective_kind"] == "realistic"].copy()
            leaderboard_df = build_cross_scenario_leaderboard(cross_scenario_objective_rows)
            display(leaderboard_df)
            display(
                realistic_cross_scenario_df.groupby(["family", "rarity_band", "objective_name"], as_index=False)
                .agg(
                    mean_fuzzy_similarity=("oracle_fuzzy_similarity", "mean"),
                    exact_match_count=("oracle_exact_match", "sum"),
                )
                .sort_values(["family", "rarity_band", "objective_name"])
            )
            display(realistic_cross_scenario_df.sort_values(["objective_name", "scenario_key"])[[
                "scenario_key",
                "family",
                "rarity_band",
                "objective_name",
                "config_id",
                "q_multiplier",
                "n_swap_pairs",
                "beta",
                "oracle_exact_match",
                "oracle_fuzzy_similarity",
                "oracle_q_index_distance",
                "oracle_swap_distance",
            ]])

            if SAVE_OUTPUTS and run_dir is not None:
                plot_cross_scenario_fuzzy_similarity(
                    cross_scenario_objective_rows,
                    save_path=run_dir / "cross_scenario_fuzzy_similarity_heatmap.png",
                )
                display(Image(filename=str(run_dir / "cross_scenario_fuzzy_similarity_heatmap.png")))
            else:
                plot_cross_scenario_fuzzy_similarity(cross_scenario_objective_rows)
                print("SAVE_OUTPUTS=False, so the cross-scenario heatmap was not saved.")
            """
        ),
        markdown_cell("## Review Saved Figures"),
        code_cell(
            """
            if SAVE_OUTPUTS and run_dir is not None:
                print(f"Saved outputs under: {run_dir}")
                for scenario in scenarios:
                    scenario_dir = run_dir / scenario.key
                    print(f"\\n{scenario.key}")
                    display(Image(filename=str(scenario_dir / "oracle_rmse_heatmap.png")))
                    display(Image(filename=str(scenario_dir / "objective_seed_noise_heatmap.png")))
            else:
                print("SAVE_OUTPUTS=False, so no saved figures to display.")
            """
        ),
        markdown_cell("## Reload Saved Results Without Rerunning"),
        code_cell(
            """
            # RELOAD_GRID_DIR = None
            # # Example:
            # # RELOAD_GRID_DIR = project_root / "results" / "mcmcis_objective_grid" / "20260312_120000_objective_grid" / "linear_stat_dp_n40"

            # if RELOAD_GRID_DIR is not None:
            #     saved = load_mcmc_objective_grid_saved_output(RELOAD_GRID_DIR)
            #     print(json.dumps({
            #         "scenario_display": saved["config_summary_payload"]["scenario_display"],
            #         "exact_p": saved["config_summary_payload"]["exact_p"],
            #     }, indent=2))
            #     display(pd.DataFrame(saved["config_summary_payload"]["config_summary"]).head())
            #     display(pd.DataFrame(saved["objective_winners_payload"]["objective_winners"]).head())
            #     display(pd.DataFrame(saved["objective_seed_noise_payload"]["objective_seed_noise"]).head())
            # else:
            #     print("Set RELOAD_GRID_DIR to a saved objective-grid scenario directory to inspect saved results.")
            """
        ),
        markdown_cell("## Optional Lower-Budget Selector Stress Test"),
        code_cell(
            """
            # Set EXTRA_TRIAL_BUDGETS above to something like (100_000, 50_000) to rerun the
            # same objective grid at smaller scan budgets and compare whether the realistic
            # objective ranking remains stable.
            if EXTRA_TRIAL_BUDGETS:
                budget_leaderboards = {}
                for extra_budget in EXTRA_TRIAL_BUDGETS:
                    rows = []
                    for scenario_idx, scenario in enumerate(scenarios):
                        study_seed = BASE_SEED + 200_000 + 10_000 * (scenario_idx + 1) + int(extra_budget)
                        rerun = run_mcmc_objective_grid_study(
                            scenario.problem,
                            scenario.exact_p,
                            mcmc_cfg=mcmc_cfg,
                            trial_repeats=TRIAL_REPEATS,
                            trial_budget=int(extra_budget),
                            base_seed=study_seed,
                            q_multipliers=Q_MULTIPLIERS,
                            n_swap_pairs_values=N_SWAP_PAIRS,
                            q_floor=Q_FLOOR,
                            n_jobs=N_JOBS,
                        )
                        df = pd.DataFrame(rerun["objective_winners"])
                        df["scenario_key"] = scenario.key
                        rows.extend(df.to_dict(orient="records"))
                    budget_leaderboards[int(extra_budget)] = build_cross_scenario_leaderboard(rows)

                for budget, leaderboard in budget_leaderboards.items():
                    print(f"\\nTrial budget = {budget:,}")
                    display(leaderboard)
            else:
                print("Set EXTRA_TRIAL_BUDGETS to rerun the objective grid at smaller scan budgets.")
            """
        ),
    ]
    return notebook(cells)


def build_mcmc_scan_budget_policy_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: MCMC-IS Scan-Budget Policy Calibration

            Objective:
            - Learn a simple rule for allocating MCMC-IS beta-scan budget as a function of an oracle/guessed \\(p_0\\).
            - Compare policies at a fixed total budget, so each method pays for beta selection before production.
            - Avoid duplicate production work by grouping identical selected betas and evaluating all required production checkpoints from one cumulative run.

            The notebook is an offline calibration tool: it uses exact \\(p_0\\) by default so we can learn a practical policy, but the same policy form can later be driven by a guessed \\(p_0\\).
            """
        ),
        code_cell(_common_setup_code()),
        code_cell(
            """
            import concurrent.futures as cf
            import math
            import time

            import matplotlib.pyplot as plt
            import numpy as np

            from perm_pval.core.proposals import resolve_n_swap_pairs
            from perm_pval.diagnostics.is_weights import effective_sample_size, summarize_weights
            from perm_pval.experiments.notebook_studies import (
                _annotate_error_fields,
                _burn_in,
                _effective_n_jobs,
                _kept_samples_per_chain,
                _mcmc_eval_count,
                _mcmc_prefix_row,
                _run_single_chain_full_trace,
                _steps_per_chain,
                _try_make_process_pool,
                local_beta_scan,
                run_scan_budget_repeat_job,
            )
            from perm_pval.methods.beta_tuning import estimate_scale_T, iid_pilot_statistics
            """
        ),
        markdown_cell(
            """
            ## Configuration

            `TOTAL_BUDGET` is the total iteration budget for MCMC-IS: pilot + scan + production.
            The p0-dependent policies below choose the scan budget, and the production budget is the leftover.

            The `tail_hit_cap` policies use
            \\[
            q_{\\mathrm{ref}} = p_0^{1/3} m_{\\mathrm{ref}},
            \\]
            then choose a per-candidate screen length that would deliver a target number of expected tilted-tail hits at \\(q_{\\mathrm{ref}}\\), capped by a fixed fraction of the total budget.
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            OUTPUT_ROOT = project_root / "results" / "mcmcis_scan_budget_policy"

            SCENARIO_KEYS_OVERRIDE = [
                "bruteforce_welch_nonextreme_n22",
                "gwas_additive_score_sig_n100",
                "hypergeom_1e7",
                "linear_stat_dp_n40",
                "rank_sum_dp_n40",
            ]
            MIN_TAIL_STATES = 2

            TOTAL_BUDGET = 2_500_000 if not FAST_MODE else 150_000
            N_REPEATS = 5 if not FAST_MODE else 1
            N_JOBS = min(6, os.cpu_count() or 1) if not FAST_MODE else 1
            BASE_SEED = 71_071

            Q_MULTIPLIERS = (0.001, 0.005, 0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.33, 1.0)
            MCMC_LOCAL_SCAN_STRATEGY = "fixed_grid"
            MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS = ()
            MCMC_LOCAL_SCAN_OBJECTIVE = "varhat_qmatch_soft"
            PRODUCTION_ESTIMATE_VARIANCE = False
            MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND = {
                "small": 1,
                "medium": 1,
                "large": 2,
            }

            # These are deliberately simple policy classes.  The one-standard-error
            # style choice should happen after looking at the resulting leaderboard.
            BUDGET_RULE_SPECS = [
                {
                    "name": "fixed_current",
                    "kind": "fixed",
                    "screen_total_steps": 14_000,
                    "final_total_steps": 32_000,
                    "finalist_count": 3,
                },
                {
                    "name": "hits10_cap05",
                    "kind": "tail_hit_cap",
                    "min_tail_hits": 10,
                    "reference_q_multiplier": 0.05,
                    "max_scan_fraction": 0.05,
                    "finalist_count": 5,
                    "final_to_screen_ratio": 2.0,
                },
                {
                    "name": "hits25_cap10",
                    "kind": "tail_hit_cap",
                    "min_tail_hits": 25,
                    "reference_q_multiplier": 0.05,
                    "max_scan_fraction": 0.10,
                    "finalist_count": 5,
                    "final_to_screen_ratio": 2.0,
                },
                {
                    "name": "hits50_cap15",
                    "kind": "tail_hit_cap",
                    "min_tail_hits": 50,
                    "reference_q_multiplier": 0.05,
                    "max_scan_fraction": 0.15,
                    "finalist_count": 5,
                    "final_to_screen_ratio": 2.0,
                },
            ]

            base_mcmc_cfg = MCMCWorkflowConfig(
                pilot_samples=25_000 if not FAST_MODE else 1_000,
                chains=2,
                thin=1,
                estimate_variance=True,
                proposal_size=1,
                local_scan_strategy=MCMC_LOCAL_SCAN_STRATEGY,
                local_scan_objective=MCMC_LOCAL_SCAN_OBJECTIVE,
                local_scan_q_multipliers=Q_MULTIPLIERS,
                local_scan_coarse_q_multipliers=MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS,
                local_scan_screen_chains=1,
                local_scan_chains=2,
                local_scan_burn_in_fraction=0.20,
                local_scan_thin=1,
            )

            print(json.dumps({
                "FAST_MODE": FAST_MODE,
                "SCENARIO_KEYS_OVERRIDE": SCENARIO_KEYS_OVERRIDE,
                "TOTAL_BUDGET": TOTAL_BUDGET,
                "N_REPEATS": N_REPEATS,
                "N_JOBS": N_JOBS,
                "Q_MULTIPLIERS": Q_MULTIPLIERS,
                "MCMC_LOCAL_SCAN_STRATEGY": MCMC_LOCAL_SCAN_STRATEGY,
                "MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS": MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS,
                "MCMC_LOCAL_SCAN_OBJECTIVE": MCMC_LOCAL_SCAN_OBJECTIVE,
                "PRODUCTION_ESTIMATE_VARIANCE": PRODUCTION_ESTIMATE_VARIANCE,
                "MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND": MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND,
                "BUDGET_RULE_SPECS": BUDGET_RULE_SPECS,
            }, indent=2))
            """
        ),
        markdown_cell("## Helpers"),
        code_cell(
            """
            def mcmc_proposal_size_for_scenario(scenario) -> int:
                band = str(scenario.portfolio.get("sample_size_band", "medium"))
                return int(MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND.get(band, 1))


            def q_target_from_p0(p0: float, cfg: MCMCWorkflowConfig) -> float:
                return float(float(p0) ** float(cfg.d_alpha))


            def kept_samples_total(total_steps: int, *, n_chains: int, burn_in_fraction: float, thin: int) -> int:
                steps_per_chain = _steps_per_chain(int(total_steps), int(n_chains))
                burn_in = _burn_in(steps_per_chain, float(burn_in_fraction))
                return int(n_chains) * _kept_samples_per_chain(steps_per_chain, burn_in, int(thin))


            def total_steps_for_expected_tail_hits(
                *,
                q_ref: float,
                min_tail_hits: int,
                n_chains: int,
                burn_in_fraction: float,
                thin: int,
            ) -> int:
                if not np.isfinite(q_ref) or q_ref <= 0.0:
                    return 1
                target_kept = max(int(math.ceil(float(min_tail_hits) / float(q_ref))), 1)
                lo = 1
                hi = max(10, int(math.ceil(target_kept * int(thin) / max(int(n_chains), 1) / max(1.0 - float(burn_in_fraction), 1e-6))))
                while kept_samples_total(hi, n_chains=n_chains, burn_in_fraction=burn_in_fraction, thin=thin) < target_kept:
                    hi *= 2
                while lo < hi:
                    mid = (lo + hi) // 2
                    if kept_samples_total(mid, n_chains=n_chains, burn_in_fraction=burn_in_fraction, thin=thin) >= target_kept:
                        hi = mid
                    else:
                        lo = mid + 1
                return int(lo)


            def stage_eval_total(total_steps: int, *, n_candidates: int, n_chains: int) -> int:
                steps_per_chain = _steps_per_chain(int(total_steps), int(n_chains))
                return int(n_candidates) * _mcmc_eval_count(steps_per_chain, int(n_chains))


            def candidate_count_for_scan_strategy(cfg: MCMCWorkflowConfig, proposal_size_count: int) -> tuple[int, int]:
                if str(cfg.local_scan_strategy) == "adaptive_q":
                    coarse_count = len(tuple(cfg.local_scan_coarse_q_multipliers)) or len(tuple(cfg.local_scan_q_multipliers))
                    refine_count = int(cfg.local_scan_refine_max_q_points)
                    return int(coarse_count * proposal_size_count), int(refine_count * proposal_size_count)
                full_count = len(tuple(cfg.local_scan_q_multipliers))
                return int(full_count * proposal_size_count), 0


            def resolve_budget_rule(rule: dict, scenario, cfg: MCMCWorkflowConfig) -> dict:
                proposal_size = int(mcmc_proposal_size_for_scenario(scenario))
                proposal_size_count = len((proposal_size,))
                n_candidates, n_refine_candidates = candidate_count_for_scan_strategy(cfg, proposal_size_count)
                finalist_count = min(int(rule.get("finalist_count", cfg.local_scan_finalist_count)), max(n_candidates, 1))
                q_target = q_target_from_p0(scenario.exact_p, cfg)
                refine_to_screen_ratio = float(rule.get("refine_to_screen_ratio", 1.0))

                if rule["kind"] == "fixed":
                    screen_total_steps = int(rule["screen_total_steps"])
                    refine_total_steps = (
                        int(rule["refine_total_steps"])
                        if rule.get("refine_total_steps") is not None
                        else (screen_total_steps if str(cfg.local_scan_strategy) == "adaptive_q" else None)
                    )
                    final_total_steps = int(rule["final_total_steps"])
                    q_ref = np.nan
                    tail_hit_steps = np.nan
                    screen_cap_steps = np.nan
                elif rule["kind"] == "tail_hit_cap":
                    reference_multiplier = float(rule.get("reference_q_multiplier", 0.05))
                    q_ref = max(float(q_target) * reference_multiplier, 1e-12)
                    tail_hit_steps = total_steps_for_expected_tail_hits(
                        q_ref=q_ref,
                        min_tail_hits=int(rule["min_tail_hits"]),
                        n_chains=int(cfg.local_scan_screen_chains),
                        burn_in_fraction=float(cfg.local_scan_burn_in_fraction),
                        thin=int(cfg.local_scan_thin),
                    )
                    total_scan_cap_ex_pilot = max(
                        int(float(rule["max_scan_fraction"]) * int(TOTAL_BUDGET)) - int(cfg.pilot_samples),
                        1,
                    )
                    final_to_screen_ratio = float(rule.get("final_to_screen_ratio", 2.0))
                    budget_units = float(n_candidates) + float(n_refine_candidates) * refine_to_screen_ratio + float(finalist_count) * final_to_screen_ratio
                    screen_cap_steps = max(int(total_scan_cap_ex_pilot / max(budget_units, 1.0)), 1)
                    screen_total_steps = int(min(tail_hit_steps, screen_cap_steps))
                    refine_total_steps = (
                        int(max(1, round(refine_to_screen_ratio * screen_total_steps)))
                        if str(cfg.local_scan_strategy) == "adaptive_q"
                        else None
                    )
                    final_total_steps = int(max(1, round(final_to_screen_ratio * screen_total_steps)))
                else:
                    raise ValueError(f"Unknown budget rule kind: {rule['kind']}")

                screen_eval_total = stage_eval_total(
                    screen_total_steps,
                    n_candidates=n_candidates,
                    n_chains=int(cfg.local_scan_screen_chains),
                )
                refine_eval_total_upper = (
                    stage_eval_total(
                        int(refine_total_steps),
                        n_candidates=n_refine_candidates,
                        n_chains=int(
                            cfg.local_scan_refine_chains
                            if cfg.local_scan_refine_chains is not None
                            else cfg.local_scan_screen_chains
                        ),
                    )
                    if refine_total_steps is not None and n_refine_candidates > 0
                    else 0
                )
                final_eval_total_upper = stage_eval_total(
                    final_total_steps,
                    n_candidates=finalist_count,
                    n_chains=int(cfg.local_scan_chains),
                )
                expected_beta_selection_budget = int(cfg.pilot_samples) + int(screen_eval_total) + int(refine_eval_total_upper) + int(final_eval_total_upper)
                return {
                    "rule_name": str(rule["name"]),
                    "rule_kind": str(rule["kind"]),
                    "proposal_size": proposal_size,
                    "n_candidates": int(n_candidates),
                    "n_refine_candidates": int(n_refine_candidates),
                    "finalist_count": int(finalist_count),
                    "q_target": float(q_target),
                    "q_ref": float(q_ref) if np.isfinite(q_ref) else np.nan,
                    "tail_hit_steps": float(tail_hit_steps) if np.isfinite(tail_hit_steps) else np.nan,
                    "screen_cap_steps": float(screen_cap_steps) if np.isfinite(screen_cap_steps) else np.nan,
                    "refine_to_screen_ratio": float(refine_to_screen_ratio),
                    "screen_total_steps": int(screen_total_steps),
                    "refine_total_steps": (int(refine_total_steps) if refine_total_steps is not None else None),
                    "final_total_steps": int(final_total_steps),
                    "expected_screen_eval_total": int(screen_eval_total),
                    "expected_refine_eval_total_upper": int(refine_eval_total_upper),
                    "expected_final_eval_total_upper": int(final_eval_total_upper),
                    "expected_beta_selection_budget_upper": int(expected_beta_selection_budget),
                }


            def mcmc_cfg_for_budget_rule(rule_budget: dict, cfg: MCMCWorkflowConfig) -> MCMCWorkflowConfig:
                return replace(
                    cfg,
                    proposal_size=int(rule_budget["proposal_size"]),
                    local_scan_swap_counts=(int(rule_budget["proposal_size"]),),
                    local_scan_finalist_count=int(rule_budget["finalist_count"]),
                    local_scan_screen_total_steps=int(rule_budget["screen_total_steps"]),
                    local_scan_refine_total_steps=(
                        int(rule_budget["refine_total_steps"])
                        if rule_budget.get("refine_total_steps") is not None
                        else cfg.local_scan_refine_total_steps
                    ),
                    local_scan_total_steps=int(rule_budget["final_total_steps"]),
                )


            def selected_scan_row(scan: dict) -> dict:
                rows = list(scan.get("rows", []))
                if not rows:
                    return {}
                beta = float(scan.get("selected_beta", np.nan))
                prop = int(scan.get("selected_proposal_size", -1))
                matches = [
                    row
                    for row in rows
                    if int(row.get("n_swap_pairs", -999)) == prop
                    and np.isfinite(row.get("beta", np.nan))
                    and abs(float(row.get("beta")) - beta) <= 1e-10
                ]
                return dict(matches[0] if matches else rows[0])


            def pack_scan_sample_batches(sample_batches: list[dict]) -> dict:
                log_weight_blocks = []
                tail_blocks = []
                for batch in sample_batches:
                    logw = np.asarray(batch.get("log_weights", []), dtype=float)
                    tail = np.asarray(batch.get("tail_indicators", []), dtype=np.int8)
                    if logw.size != tail.size:
                        raise ValueError("scan sample batch log_weights and tail_indicators must have matching sizes")
                    if logw.size:
                        log_weight_blocks.append(logw)
                        tail_blocks.append(tail)
                if log_weight_blocks:
                    log_weights = np.concatenate(log_weight_blocks)
                    tail_indicators = np.concatenate(tail_blocks)
                else:
                    log_weights = np.asarray([], dtype=float)
                    tail_indicators = np.asarray([], dtype=np.int8)
                return {
                    "log_weights": log_weights,
                    "tail_indicators": tail_indicators,
                    "n_batches": int(len(sample_batches)),
                    "n_weighted_samples": int(tail_indicators.size),
                }


            def run_production_traces(
                problem,
                *,
                checkpoints: tuple[int, ...],
                beta: float,
                sigma_t: float,
                cfg,
                seed: int,
                init_states: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
            ) -> tuple[list[dict], dict[int, int]]:
                checkpoints = tuple(sorted({int(v) for v in checkpoints}))
                max_total_steps = int(checkpoints[-1])
                max_steps_per_chain = _steps_per_chain(max_total_steps, int(cfg.chains))
                steps_per_checkpoint = {int(cp): _steps_per_chain(int(cp), int(cfg.chains)) for cp in checkpoints}
                unique_step_checkpoints = tuple(sorted(set(steps_per_checkpoint.values())))
                n_swap_pairs = resolve_n_swap_pairs(
                    problem.n_treated,
                    problem.n_control,
                    proposal_size=cfg.proposal_size,
                )
                seed_seq = np.random.SeedSequence(seed)
                spawned = seed_seq.spawn(int(cfg.chains))
                if init_states is None:
                    normalized_init_states = [None] * int(cfg.chains)
                else:
                    if len(init_states) != int(cfg.chains):
                        raise ValueError("init_states length must match cfg.chains")
                    normalized_init_states = [np.asarray(state, dtype=np.int8).copy() for state in init_states]
                traces = []
                for chain_idx, ss in enumerate(spawned):
                    rng = np.random.default_rng(ss)
                    traces.append(
                        _run_single_chain_full_trace(
                            problem,
                            rng,
                            beta=float(beta),
                            sigma_t=float(sigma_t),
                            n_steps=max_steps_per_chain,
                            init="random" if normalized_init_states[chain_idx] is None else "observed",
                            init_state=normalized_init_states[chain_idx],
                            tilt_mode=str(cfg.tilt_mode),
                            n_swap_pairs=n_swap_pairs,
                            checkpoint_steps=unique_step_checkpoints,
                        )
                    )
                return traces, steps_per_checkpoint


            def production_prefix_payload(
                *,
                traces: list[dict],
                steps_per_chain: int,
                burn_in: int,
                thin: int,
                beta: float,
            ) -> dict:
                q_chunks = []
                tail_chunks = []
                for trace in traces:
                    q_chunks.append(np.asarray(trace["q_trace"][burn_in:steps_per_chain:thin], dtype=float))
                    tail_chunks.append(np.asarray(trace["tail_trace"][burn_in:steps_per_chain:thin], dtype=np.int8))
                q_samples = np.concatenate(q_chunks) if q_chunks else np.asarray([], dtype=float)
                tail_indicators = np.concatenate(tail_chunks) if tail_chunks else np.asarray([], dtype=np.int8)
                return {
                    "log_weights": np.asarray(float(beta) * q_samples, dtype=float),
                    "tail_indicators": np.asarray(tail_indicators, dtype=np.int8),
                }


            def pooled_scan_plus_production_row(
                *,
                exact_p: float,
                base_row: dict,
                scan_sample_pack: dict,
                production_payload: dict,
            ) -> dict:
                scan_logw = np.asarray(scan_sample_pack.get("log_weights", []), dtype=float)
                scan_tail = np.asarray(scan_sample_pack.get("tail_indicators", []), dtype=np.int8)
                prod_logw = np.asarray(production_payload.get("log_weights", []), dtype=float)
                prod_tail = np.asarray(production_payload.get("tail_indicators", []), dtype=np.int8)

                log_weight_blocks = [arr for arr in (scan_logw, prod_logw) if arr.size]
                tail_blocks = [arr for arr in (scan_tail, prod_tail) if arr.size]
                if not log_weight_blocks or not tail_blocks:
                    raise ValueError("pooled estimator requires at least one non-empty sample block")

                all_logw = np.concatenate(log_weight_blocks)
                all_tail = np.concatenate(tail_blocks)
                shift = float(np.max(all_logw))
                weights = np.exp(all_logw - shift)
                weight_sum = float(np.sum(weights))
                estimate = float(np.dot(weights, all_tail) / weight_sum)
                weight_summary = summarize_weights(weights)

                row = dict(base_row)
                row["estimate"] = float(estimate)
                row["variance_estimate"] = np.nan
                row["snis_mcse_obm"] = np.nan
                row["tail_hits"] = int(np.sum(all_tail))
                row["tail_share_raw"] = np.nan
                row["ess"] = float(effective_sample_size(weights))
                row["weight_cv"] = float(weight_summary.cv)
                row["n_weighted_samples"] = int(weights.size)
                row["estimator_variant"] = "scan_plus_production"
                row["scan_n_weighted_samples"] = int(scan_tail.size)
                row["production_n_weighted_samples"] = int(prod_tail.size)
                row["pooled_scan_batch_count"] = int(scan_sample_pack.get("n_batches", 0))
                return _annotate_error_fields(row, float(exact_p))
            """
        ),
        markdown_cell("## Load Scenarios"),
        code_cell(
            """
            scenarios = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=SCENARIO_KEYS_OVERRIDE,
                portfolio_group=None,
                min_tail_states=MIN_TAIL_STATES,
            )
            scenario_by_key = {scenario.key: scenario for scenario in scenarios}
            run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "scan_budget_policy") if SAVE_OUTPUTS else None

            pd.DataFrame([
                {
                    "scenario": s.key,
                    "exact_p": s.exact_p,
                    "family": s.portfolio.get("family"),
                    "statistic_family": s.portfolio.get("statistic_family"),
                    "sample_size_band": s.portfolio.get("sample_size_band"),
                    "proposal_size": mcmc_proposal_size_for_scenario(s),
                }
                for s in scenarios
            ])
            """
        ),
        markdown_cell(
            """
            ## Run Policy Scans

            Each `(scenario, repeat)` job uses one common IID pilot and reuses it across all budget policies. Jobs are independent, so the notebook can parallelize at that outer level while still charging the pilot cost to every policy.
            """
        ),
        code_cell(
            """
            scan_records: list[dict] = []
            production_records: list[dict] = []
            repeat_jobs: list[dict] = []

            for scenario_idx, scenario in enumerate(scenarios):
                rule_budgets = [resolve_budget_rule(rule, scenario, base_mcmc_cfg) for rule in BUDGET_RULE_SPECS]
                print(
                    f"Prepared scenario {scenario.key} | exact p={scenario.exact_p:.3e} "
                    f"| repeats={N_REPEATS} | rules={len(rule_budgets)}"
                )
                for repeat_idx in range(N_REPEATS):
                    pilot_seed = BASE_SEED + 100_000 * scenario_idx + 10_000 * repeat_idx
                    repeat_jobs.append(
                        {
                            "scenario_key": str(scenario.key),
                            "scenario_display": str(scenario.description),
                            "family": scenario.portfolio.get("family"),
                            "statistic_family": scenario.portfolio.get("statistic_family"),
                            "sample_size_band": scenario.portfolio.get("sample_size_band"),
                            "problem": scenario.problem,
                            "exact_p": float(scenario.exact_p),
                            "repeat_idx": int(repeat_idx),
                            "pilot_seed": int(pilot_seed),
                            "total_budget": int(TOTAL_BUDGET),
                            "base_mcmc_cfg": base_mcmc_cfg,
                            "rule_budgets": [dict(rule_budget) for rule_budget in rule_budgets],
                            "production_estimate_variance": bool(PRODUCTION_ESTIMATE_VARIANCE),
                        }
                    )

            t0 = time.perf_counter()
            n_workers = _effective_n_jobs(N_JOBS, len(repeat_jobs))
            executor = _try_make_process_pool(n_workers) if n_workers > 1 else None
            print(f"Running {len(repeat_jobs)} scenario-repeat jobs with n_workers={n_workers}")

            def _consume_job_result(result: dict) -> None:
                scan_records.extend(result["scan_records"])
                production_records.extend(result["production_records"])


            if executor is None:
                for job_idx, job in enumerate(repeat_jobs, start=1):
                    _consume_job_result(run_scan_budget_repeat_job(**job))
                    if job_idx % max(1, len(repeat_jobs) // 6) == 0 or job_idx == len(repeat_jobs):
                        print(f"Completed {job_idx}/{len(repeat_jobs)} jobs")
            else:
                with executor:
                    futures = [executor.submit(run_scan_budget_repeat_job, **job) for job in repeat_jobs]
                    for job_idx, future in enumerate(cf.as_completed(futures), start=1):
                        _consume_job_result(future.result())
                        if job_idx % max(1, len(repeat_jobs) // 6) == 0 or job_idx == len(repeat_jobs):
                            print(f"Completed {job_idx}/{len(repeat_jobs)} jobs")

            scan_df = pd.DataFrame(scan_records)
            production_df = pd.DataFrame(production_records)
            print(
                f"Completed {len(scan_df)} scans and {len(production_df)} production rows "
                f"in {time.perf_counter() - t0:.1f}s"
            )
            display(scan_df.sort_values(["scenario", "repeat", "rule_name"])[[
                "scenario", "repeat", "rule_name", "beta_selection_budget", "production_budget",
                "screen_total_steps", "refine_total_steps", "final_total_steps",
                "scan_n_weighted_samples", "selected_scan_n_weighted_samples", "selected_q_multiplier",
                "selected_beta", "selected_q_tilt_tail_share", "selected_ess", "selected_weight_cv",
            ]])
            """
        ),
        markdown_cell(
            """
            ## Run Deduplicated Production

            Production is already executed inside each parallel `(scenario, repeat)` job. Within a job, if multiple budget policies select the same beta/proposal, we run one cumulative production chain up to the largest required leftover budget and read off the shorter checkpoints for the other policies.

            For each rule we record two estimators:
            1. `production_only`, which uses only the post-scan production chain.
            2. `selected_scan_plus_production`, which pools only the selected beta/proposal's scan batches with production.
            3. `all_scan_plus_production`, which pools all stored scan sample batches with production.

            We intentionally do not reuse scan final states here. That keeps the production runs groupable and isolates the beta/proposal selection effect from differences in scan-generated initialization states.
            """
        ),
        code_cell(
            """
            print(f"Loaded {len(production_df)} policy production rows from the completed scenario-repeat jobs")
            display(production_df.sort_values(["scenario", "repeat", "rule_name", "estimator_variant"])[[
                "scenario", "repeat", "rule_name", "estimator_variant", "estimate", "exact_p", "root_squared_error",
                "abs_log10_error", "selected_q_multiplier", "selected_beta", "beta_selection_budget",
                "production_budget", "available_scan_n_weighted_samples", "scan_n_weighted_samples",
                "production_n_weighted_samples", "ess", "weight_cv", "acceptance_rate",
            ]])
            """
        ),
        markdown_cell("## Summarize Budget Policies"),
        code_cell(
            """
            def summarize_policy_records(df: pd.DataFrame) -> pd.DataFrame:
                rows = []
                for (scenario, rule_name, estimator_variant), sub in df.groupby(["scenario", "rule_name", "estimator_variant"], sort=True):
                    exact_p = float(sub["exact_p"].iloc[0])
                    estimates = sub["estimate"].astype(float).to_numpy()
                    rows.append({
                        "scenario": scenario,
                        "rule_name": rule_name,
                        "estimator_variant": estimator_variant,
                        "n_runs": int(len(sub)),
                        "exact_p": exact_p,
                        "mean_estimate": float(np.mean(estimates)),
                        "rmse": float(np.sqrt(np.mean((estimates - exact_p) ** 2))),
                        "mean_abs_log10_error": float(np.nanmean(sub["abs_log10_error"].astype(float))),
                        "mean_selected_q_multiplier": float(np.nanmean(sub["selected_q_multiplier"].astype(float))),
                        "median_selected_q_multiplier": float(np.nanmedian(sub["selected_q_multiplier"].astype(float))),
                        "mean_beta_selection_budget": float(np.mean(sub["beta_selection_budget"].astype(float))),
                        "mean_production_budget": float(np.mean(sub["production_budget"].astype(float))),
                        "mean_q_tilt_tail_share": float(np.nanmean(sub["tail_share_raw"].astype(float))),
                        "mean_ess": float(np.nanmean(sub["ess"].astype(float))),
                        "mean_weight_cv": float(np.nanmean(sub["weight_cv"].astype(float))),
                        "mean_acceptance_rate": float(np.nanmean(sub["acceptance_rate"].astype(float))),
                        "mean_available_scan_n_weighted_samples": float(np.nanmean(sub["available_scan_n_weighted_samples"].astype(float))),
                        "mean_available_selected_scan_n_weighted_samples": float(np.nanmean(sub.get("available_selected_scan_n_weighted_samples", pd.Series(dtype=float)).astype(float))) if "available_selected_scan_n_weighted_samples" in sub else np.nan,
                        "mean_scan_n_weighted_samples": float(np.nanmean(sub["scan_n_weighted_samples"].astype(float))),
                        "mean_production_n_weighted_samples": float(np.nanmean(sub["production_n_weighted_samples"].astype(float))),
                    })
                return pd.DataFrame(rows).sort_values(["scenario", "estimator_variant", "rmse", "mean_abs_log10_error"]).reset_index(drop=True)


            policy_summary = summarize_policy_records(production_df)
            policy_summary["rmse_rank"] = policy_summary.groupby(["scenario", "estimator_variant"])["rmse"].rank(method="min")
            policy_summary["abs_log10_error_rank"] = policy_summary.groupby(["scenario", "estimator_variant"])["mean_abs_log10_error"].rank(method="min")
            display(policy_summary)

            leaderboard = (
                policy_summary
                .groupby(["estimator_variant", "rule_name"], as_index=False)
                .agg(
                    mean_rmse_rank=("rmse_rank", "mean"),
                    mean_abs_log10_error_rank=("abs_log10_error_rank", "mean"),
                    mean_rmse=("rmse", "mean"),
                    mean_abs_log10_error=("mean_abs_log10_error", "mean"),
                    mean_beta_selection_budget=("mean_beta_selection_budget", "mean"),
                    mean_production_budget=("mean_production_budget", "mean"),
                    mean_available_scan_n_weighted_samples=("mean_available_scan_n_weighted_samples", "mean"),
                    mean_available_selected_scan_n_weighted_samples=("mean_available_selected_scan_n_weighted_samples", "mean"),
                    mean_scan_n_weighted_samples=("mean_scan_n_weighted_samples", "mean"),
                    mean_production_n_weighted_samples=("mean_production_n_weighted_samples", "mean"),
                    mean_ess=("mean_ess", "mean"),
                    mean_weight_cv=("mean_weight_cv", "mean"),
                )
                .sort_values(["estimator_variant", "mean_rmse_rank", "mean_abs_log10_error_rank", "mean_rmse"])
                .reset_index(drop=True)
            )
            display(leaderboard)

            comparison_rows = []
            for (scenario, rule_name), sub in policy_summary.groupby(["scenario", "rule_name"], sort=True):
                if "production_only" not in set(sub["estimator_variant"]):
                    continue
                base = sub[sub["estimator_variant"] == "production_only"].iloc[0]
                for _, alt in sub[sub["estimator_variant"] != "production_only"].iterrows():
                    comparison_rows.append({
                        "scenario": scenario,
                        "rule_name": rule_name,
                        "exact_p": float(base["exact_p"]),
                        "comparison_variant": str(alt["estimator_variant"]),
                        "production_only_rmse": float(base["rmse"]),
                        "comparison_rmse": float(alt["rmse"]),
                        "rmse_ratio_vs_production_only": (
                            float(alt["rmse"] / base["rmse"])
                            if np.isfinite(base["rmse"]) and float(base["rmse"]) > 0.0
                            else np.nan
                        ),
                        "production_only_abs_log10_error": float(base["mean_abs_log10_error"]),
                        "comparison_abs_log10_error": float(alt["mean_abs_log10_error"]),
                        "abs_log10_error_delta": float(alt["mean_abs_log10_error"] - base["mean_abs_log10_error"]),
                        "scan_n_weighted_samples_added": float(alt["mean_scan_n_weighted_samples"]),
                    })
            comparison_df = pd.DataFrame(comparison_rows).sort_values(["scenario", "comparison_variant", "rmse_ratio_vs_production_only"])
            display(comparison_df)
            """
        ),
        markdown_cell("## Save Outputs"),
        code_cell(
            """
            if SAVE_OUTPUTS and run_dir is not None:
                run_dir.mkdir(parents=True, exist_ok=True)
                scan_df.to_json(run_dir / "scan_records.jsonl", orient="records", lines=True)
                production_df.to_json(run_dir / "production_records.jsonl", orient="records", lines=True)
                policy_summary.to_json(run_dir / "policy_summary.json", orient="records", indent=2)
                leaderboard.to_json(run_dir / "policy_leaderboard.json", orient="records", indent=2)
                comparison_df.to_json(run_dir / "estimator_comparison.json", orient="records", indent=2)
                (run_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "TOTAL_BUDGET": TOTAL_BUDGET,
                            "N_REPEATS": N_REPEATS,
                            "N_JOBS": N_JOBS,
                            "SCENARIO_KEYS_OVERRIDE": SCENARIO_KEYS_OVERRIDE,
                            "Q_MULTIPLIERS": Q_MULTIPLIERS,
                            "BUDGET_RULE_SPECS": BUDGET_RULE_SPECS,
                            "MCMC_LOCAL_SCAN_STRATEGY": MCMC_LOCAL_SCAN_STRATEGY,
                            "MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS": MCMC_LOCAL_SCAN_COARSE_Q_MULTIPLIERS,
                            "MCMC_LOCAL_SCAN_OBJECTIVE": MCMC_LOCAL_SCAN_OBJECTIVE,
                            "PRODUCTION_ESTIMATE_VARIANCE": PRODUCTION_ESTIMATE_VARIANCE,
                            "ESTIMATOR_VARIANTS": [
                                "production_only",
                                "selected_scan_plus_production",
                                "all_scan_plus_production",
                            ],
                            "MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND": MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"Saved scan-budget policy outputs to {run_dir}")
            else:
                print("SAVE_OUTPUTS is False; nothing saved.")
            """
        ),
    ]
    return notebook(cells)


def build_mcmc_scan_budget_grid_notebook() -> dict:
    nb = build_mcmc_scan_budget_policy_notebook()
    cells = nb["cells"]
    cells[0] = markdown_cell(
        """
        # Experiment: Fixed Scan-Budget Grid for MCMC-IS

        Objective:
        - At a fixed total MCMC-IS budget of 10 million iterations, compare several fixed beta-selection budgets.
        - For each scan budget, choose beta by the same local scan objective and then spend the remaining budget on production.
        - Avoid duplicate production work by grouping identical selected beta/proposal pairs and evaluating all required production cutoffs from one cumulative run.

        This is an offline calibration tool. After the run, we can fit a coarse rule of thumb such as scan budget as a function of total budget and/or \\(p_0\\).
        """
    )
    cells[3] = markdown_cell(
        """
        ## Configuration

        `TOTAL_BUDGET` is the full MCMC-IS iteration budget: IID pilot + beta scan + production. `SCAN_BUDGETS` below are target beta-selection budgets, including the IID pilot. The actual charged budget can differ by a few iterations because MCMC chains are split across chains and candidates.

        The scan-budget grid is deliberately simple: it asks whether spending roughly `50k`, `100k`, `500k`, `1m`, or `2m` on beta selection is worthwhile when the full budget is `10m`.
        """
    )
    cells[4] = code_cell(
        """
        FAST_MODE = False
        SAVE_OUTPUTS = True

        CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
        OUTPUT_ROOT = project_root / "results" / "mcmcis_scan_budget_grid"

        SCENARIO_KEYS_OVERRIDE = [
            "gwas_additive_score_sig_n100",
            "linear_stat_dp_n40",
            "rank_sum_dp_n40",
        ]
        MIN_TAIL_STATES = 2

        TOTAL_BUDGET = 10_000_000 if not FAST_MODE else 200_000
        N_REPEATS = 6 if not FAST_MODE else 1
        N_JOBS = min(6, os.cpu_count() or 1) if not FAST_MODE else 1
        BASE_SEED = 83_083

        SCAN_BUDGETS = (
            50_000,
            100_000,
            500_000,
            1_000_000,
            2_000_000,
        ) if not FAST_MODE else (25_000, 50_000)
        SCAN_FINALIST_COUNT = 4
        SCAN_REFINE_TO_SCREEN_RATIO = 1.0
        SCAN_FINAL_TO_SCREEN_RATIO = 2.0

        Q_MULTIPLIERS = (0.001, 0.005, 0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.33, 0.5, 1.0)
        COARSE_Q_MULTIPLIERS = (0.001, 0.01, 0.10, 0.5)
        MCMC_LOCAL_SCAN_STRATEGY = "adaptive_q"
        MCMC_LOCAL_SCAN_OBJECTIVE = "varhat_qmatch_soft"
        PRODUCTION_ESTIMATE_VARIANCE = False
        MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND = {
            "small": 1,
            "medium": 1,
            "large": 2,
        }

        BUDGET_RULE_SPECS = [
            {
                "name": f"scan_{budget // 1_000_000}m" if budget >= 1_000_000 else f"scan_{budget // 1_000}k",
                "kind": "fixed_scan_budget",
                "target_beta_selection_budget": int(budget),
                "finalist_count": SCAN_FINALIST_COUNT,
                "refine_to_screen_ratio": SCAN_REFINE_TO_SCREEN_RATIO,
                "final_to_screen_ratio": SCAN_FINAL_TO_SCREEN_RATIO,
            }
            for budget in SCAN_BUDGETS
        ]

        base_mcmc_cfg = MCMCWorkflowConfig(
            pilot_samples=25_000 if not FAST_MODE else 5_000,
            chains=2,
            thin=1,
            estimate_variance=True,
            proposal_size=1,
            local_scan_strategy=MCMC_LOCAL_SCAN_STRATEGY,
            local_scan_objective=MCMC_LOCAL_SCAN_OBJECTIVE,
            local_scan_q_multipliers=Q_MULTIPLIERS,
            local_scan_coarse_q_multipliers=COARSE_Q_MULTIPLIERS,
            local_scan_screen_chains=1,
            local_scan_refine_top_k=2,
            local_scan_refine_radius=1,
            local_scan_refine_max_q_points=6,
            local_scan_chains=2,
            local_scan_burn_in_fraction=0.20,
            local_scan_thin=1,
        )

        print(json.dumps({
            "FAST_MODE": FAST_MODE,
            "SCENARIO_KEYS_OVERRIDE": SCENARIO_KEYS_OVERRIDE,
            "TOTAL_BUDGET": TOTAL_BUDGET,
            "N_REPEATS": N_REPEATS,
            "N_JOBS": N_JOBS,
            "SCAN_BUDGETS": SCAN_BUDGETS,
            "SCAN_FINALIST_COUNT": SCAN_FINALIST_COUNT,
            "SCAN_REFINE_TO_SCREEN_RATIO": SCAN_REFINE_TO_SCREEN_RATIO,
            "SCAN_FINAL_TO_SCREEN_RATIO": SCAN_FINAL_TO_SCREEN_RATIO,
            "Q_MULTIPLIERS": Q_MULTIPLIERS,
            "COARSE_Q_MULTIPLIERS": COARSE_Q_MULTIPLIERS,
            "MCMC_LOCAL_SCAN_STRATEGY": MCMC_LOCAL_SCAN_STRATEGY,
            "MCMC_LOCAL_SCAN_OBJECTIVE": MCMC_LOCAL_SCAN_OBJECTIVE,
            "PRODUCTION_ESTIMATE_VARIANCE": PRODUCTION_ESTIMATE_VARIANCE,
            "MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND": MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND,
            "BUDGET_RULE_SPECS": BUDGET_RULE_SPECS,
        }, indent=2))
        """
    )
    cells[6] = code_cell(
        """
        def mcmc_proposal_size_for_scenario(scenario) -> int:
            band = str(scenario.portfolio.get("sample_size_band", "medium"))
            return int(MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND.get(band, 1))


        def q_target_from_p0(p0: float, cfg: MCMCWorkflowConfig) -> float:
            return float(float(p0) ** float(cfg.d_alpha))


        def kept_samples_total(total_steps: int, *, n_chains: int, burn_in_fraction: float, thin: int) -> int:
            steps_per_chain = _steps_per_chain(int(total_steps), int(n_chains))
            burn_in = _burn_in(steps_per_chain, float(burn_in_fraction))
            return int(n_chains) * _kept_samples_per_chain(steps_per_chain, burn_in, int(thin))


        def stage_eval_total(total_steps: int, *, n_candidates: int, n_chains: int) -> int:
            steps_per_chain = _steps_per_chain(int(total_steps), int(n_chains))
            return int(n_candidates) * _mcmc_eval_count(steps_per_chain, int(n_chains))


        def split_fixed_scan_budget(
            *,
            target_beta_selection_budget: int,
            cfg: MCMCWorkflowConfig,
            n_candidates: int,
            finalist_count: int,
            final_to_screen_ratio: float,
        ) -> tuple[int, int]:
            scan_budget_ex_pilot = max(int(target_beta_selection_budget) - int(cfg.pilot_samples), 1)
            budget_units = float(n_candidates) + float(finalist_count) * float(final_to_screen_ratio)
            screen_total_steps = max(int(scan_budget_ex_pilot / max(budget_units, 1.0)), 1)
            final_total_steps = max(int(round(float(final_to_screen_ratio) * screen_total_steps)), 1)
            return int(screen_total_steps), int(final_total_steps)


        def resolve_budget_rule(rule: dict, scenario, cfg: MCMCWorkflowConfig) -> dict:
            proposal_size = int(mcmc_proposal_size_for_scenario(scenario))
            n_candidates = len(tuple(cfg.local_scan_q_multipliers))
            finalist_count = min(int(rule.get("finalist_count", cfg.local_scan_finalist_count)), n_candidates)
            q_target = q_target_from_p0(scenario.exact_p, cfg)

            if rule["kind"] != "fixed_scan_budget":
                raise ValueError(f"Unknown budget rule kind for this notebook: {rule['kind']}")

            target_beta_selection_budget = int(rule["target_beta_selection_budget"])
            if target_beta_selection_budget >= int(TOTAL_BUDGET):
                raise ValueError("target_beta_selection_budget must be smaller than TOTAL_BUDGET")
            final_to_screen_ratio = float(rule.get("final_to_screen_ratio", 2.0))
            screen_total_steps, final_total_steps = split_fixed_scan_budget(
                target_beta_selection_budget=target_beta_selection_budget,
                cfg=cfg,
                n_candidates=n_candidates,
                finalist_count=finalist_count,
                final_to_screen_ratio=final_to_screen_ratio,
            )

            screen_eval_total = stage_eval_total(
                screen_total_steps,
                n_candidates=n_candidates,
                n_chains=int(cfg.local_scan_screen_chains),
            )
            final_eval_total_upper = stage_eval_total(
                final_total_steps,
                n_candidates=finalist_count,
                n_chains=int(cfg.local_scan_chains),
            )
            expected_beta_selection_budget = int(cfg.pilot_samples) + int(screen_eval_total) + int(final_eval_total_upper)
            return {
                "rule_name": str(rule["name"]),
                "rule_kind": str(rule["kind"]),
                "proposal_size": proposal_size,
                "n_candidates": int(n_candidates),
                "finalist_count": int(finalist_count),
                "q_target": float(q_target),
                "q_ref": np.nan,
                "tail_hit_steps": np.nan,
                "screen_cap_steps": np.nan,
                "target_beta_selection_budget": int(target_beta_selection_budget),
                "target_scan_budget_ex_pilot": int(max(target_beta_selection_budget - int(cfg.pilot_samples), 1)),
                "final_to_screen_ratio": float(final_to_screen_ratio),
                "screen_total_steps": int(screen_total_steps),
                "final_total_steps": int(final_total_steps),
                "expected_screen_eval_total": int(screen_eval_total),
                "expected_final_eval_total_upper": int(final_eval_total_upper),
                "expected_beta_selection_budget_upper": int(expected_beta_selection_budget),
            }


        def mcmc_cfg_for_budget_rule(rule_budget: dict, cfg: MCMCWorkflowConfig) -> MCMCWorkflowConfig:
            return replace(
                cfg,
                proposal_size=int(rule_budget["proposal_size"]),
                local_scan_swap_counts=(int(rule_budget["proposal_size"]),),
                local_scan_finalist_count=int(rule_budget["finalist_count"]),
                local_scan_screen_total_steps=int(rule_budget["screen_total_steps"]),
                local_scan_total_steps=int(rule_budget["final_total_steps"]),
            )


        def selected_scan_row(scan: dict) -> dict:
            rows = list(scan.get("rows", []))
            if not rows:
                return {}
            beta = float(scan.get("selected_beta", np.nan))
            prop = int(scan.get("selected_proposal_size", -1))
            matches = [
                row
                for row in rows
                if int(row.get("n_swap_pairs", -999)) == prop
                and np.isfinite(row.get("beta", np.nan))
                and abs(float(row.get("beta")) - beta) <= 1e-10
            ]
            return dict(matches[0] if matches else rows[0])
        """
    )
    cells[8] = code_cell(
        """
        scenarios = load_selected_scenarios(
            catalog_path=CATALOG_PATH,
            scenario_keys=SCENARIO_KEYS_OVERRIDE,
            portfolio_group=None,
            min_tail_states=MIN_TAIL_STATES,
        )
        scenario_by_key = {scenario.key: scenario for scenario in scenarios}
        run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "scan_budget_grid") if SAVE_OUTPUTS else None

        pd.DataFrame([
            {
                "scenario": s.key,
                "exact_p": s.exact_p,
                "family": s.portfolio.get("family"),
                "statistic_family": s.portfolio.get("statistic_family"),
                "sample_size_band": s.portfolio.get("sample_size_band"),
                "proposal_size": mcmc_proposal_size_for_scenario(s),
            }
            for s in scenarios
        ])
        """
    )
    cells[9] = markdown_cell(
        """
        ## Run Fixed-Budget Scans

        Each `(scenario, repeat)` job uses one common IID pilot and reuses it across all scan budgets. Jobs are independent, so the notebook can parallelize at that outer level while still charging the pilot cost to every budget setting.
        """
    )
    cells[11] = markdown_cell(
        """
        ## Run Deduplicated Production

        Production is already executed inside each parallel `(scenario, repeat)` job. Within a job, if multiple scan budgets select the same beta/proposal, we run one cumulative production chain up to the largest required leftover budget and read off the shorter checkpoints for the other scan budgets.

        We intentionally do not reuse scan final states here. That keeps the production runs groupable and isolates the beta/proposal selection effect from differences in scan-generated initialization states.
        """
    )
    cells[13] = markdown_cell("## Summarize Fixed Scan Budgets")
    cells[15] = markdown_cell("## Save Outputs")
    cells[16] = code_cell(
        """
        if SAVE_OUTPUTS and run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            scan_df.to_json(run_dir / "scan_records.jsonl", orient="records", lines=True)
            production_df.to_json(run_dir / "production_records.jsonl", orient="records", lines=True)
            policy_summary.to_json(run_dir / "policy_summary.json", orient="records", indent=2)
            leaderboard.to_json(run_dir / "policy_leaderboard.json", orient="records", indent=2)
            comparison_df.to_json(run_dir / "estimator_comparison.json", orient="records", indent=2)
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "TOTAL_BUDGET": TOTAL_BUDGET,
                        "N_REPEATS": N_REPEATS,
                        "N_JOBS": N_JOBS,
                        "SCENARIO_KEYS_OVERRIDE": SCENARIO_KEYS_OVERRIDE,
                        "SCAN_BUDGETS": SCAN_BUDGETS,
                        "SCAN_FINALIST_COUNT": SCAN_FINALIST_COUNT,
                        "SCAN_REFINE_TO_SCREEN_RATIO": SCAN_REFINE_TO_SCREEN_RATIO,
                        "SCAN_FINAL_TO_SCREEN_RATIO": SCAN_FINAL_TO_SCREEN_RATIO,
                        "Q_MULTIPLIERS": Q_MULTIPLIERS,
                        "COARSE_Q_MULTIPLIERS": COARSE_Q_MULTIPLIERS,
                        "BUDGET_RULE_SPECS": BUDGET_RULE_SPECS,
                        "MCMC_LOCAL_SCAN_STRATEGY": MCMC_LOCAL_SCAN_STRATEGY,
                        "MCMC_LOCAL_SCAN_OBJECTIVE": MCMC_LOCAL_SCAN_OBJECTIVE,
                        "PRODUCTION_ESTIMATE_VARIANCE": PRODUCTION_ESTIMATE_VARIANCE,
                        "ESTIMATOR_VARIANTS": [
                            "production_only",
                            "selected_scan_plus_production",
                            "all_scan_plus_production",
                        ],
                        "MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND": MCMC_PROPOSAL_SIZE_BY_SAMPLE_BAND,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved fixed scan-budget grid outputs to {run_dir}")
        else:
            print("SAVE_OUTPUTS is False; nothing saved.")
        """
    )
    return nb


def build_cross_method_threshold_grid_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: Threshold-Variety Cross-Method Grid

            Objective:
            - Compare SAMC, hard-step IS, and MCMC-IS without oracle p-values on the 50 GWAS-like and 50 HEP-like threshold scenarios.
            - Use a 5M maximum reported budget with 250k checkpoints.
            - Sweep swap size over 5% and 10% of the smaller group size.
            - For MCMC-IS and hard-step, sweep `gamma in {0.25, 1/3, 0.4, 0.5}` with `q = p0 ** gamma`, where `p0` is the known application threshold stored in the scenario metadata.

            Each family/swap/method/gamma block writes its own partial and final JSONL files as soon as it finishes.
            """
        ),
        code_cell(_common_setup_code()),
        markdown_cell(
            """
            ## Configuration

            The default `THRESHOLD_BANDS = ("near",)` runs the 50-per-family near-threshold inventory. Set it to `("above",)` for the `[1.01, 1.25]` above-threshold inventory, or to both bands for the full threshold-variety catalog.

            MCMC-IS uses the known threshold p0, not the exact scenario p-value, when setting `q` and initializing beta. Exact p-values are used only for error summaries.
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True
            RESUME_BLOCKS = True
            RUN_DIR_OVERRIDE = None  # Set to a previous run_dir string to resume after a kernel restart.
            SKIP_IF_PRESENT = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            OUTPUT_ROOT = project_root / "results" / "cross_method_threshold_grid"
            LATEST_RESULTS_DIR = OUTPUT_ROOT / "20260723_205321_threshold_grid_cross_method"

            THRESHOLD_BANDS = ("near",)
            FAMILIES = ("gwas", "hep")
            SCENARIOS_PER_FAMILY = 50
            MIN_TAIL_STATES = 2

            MAX_BUDGET = 5_000_000 if not FAST_MODE else 20_000
            CHECKPOINT_STEP = 250_000 if not FAST_MODE else 5_000
            CHECKPOINTS = tuple(range(CHECKPOINT_STEP, MAX_BUDGET + CHECKPOINT_STEP, CHECKPOINT_STEP))
            APPENDIX_FIGURE_BUDGETS = (1_000_000, 2_500_000) if not FAST_MODE else tuple()
            SWAP_FRACTIONS = (0.05, 0.10)
            GAMMAS = (0.25, 1.0 / 3.0, 0.40, 0.50)

            BETA_INIT_PILOT_SAMPLES = 100_000 if not FAST_MODE else 1_000
            N_JOBS = min(6, os.cpu_count() or 1) if not FAST_MODE else 1
            BASE_SEED = 91_337
            BOOTSTRAP_RESAMPLES = 10_000 if not FAST_MODE else 1_000
            BOOTSTRAP_SEED = BASE_SEED + 30_000

            MCMC_CHAINS = 1
            MCMC_ESTIMATE_VARIANCE = False
            MCMC_CFG = MCMCWorkflowConfig(
                use_true_p0_for_q_target=False,
                p0_guess=1e-8,
                pilot_samples=BETA_INIT_PILOT_SAMPLES,
                scale_method="sd",
                beta_max_init=1e6,
                chains=MCMC_CHAINS,
                thin=1,
                estimate_variance=MCMC_ESTIMATE_VARIANCE,
                chain_n_jobs=1,
                production_estimator_variant="production_only",
                tilt_mode="smooth_hinge",
                proposal_size=SWAP_FRACTIONS[0],
                local_scan_enabled=False,
            )
            SAMC_CFG = SAMCWorkflowConfig(
                burn_in_fraction=0.20,
                n_bins=100 if not FAST_MODE else 10,
                t0=1_000.0,
                trace_every=200 if not FAST_MODE else 50,
                convergence_tolerance=20.0,
                lambda_min_pilot=10_000 if not FAST_MODE else 1_000,
                proposal_size=SWAP_FRACTIONS[0],
            )

            print(json.dumps({
                "FAST_MODE": FAST_MODE,
                "THRESHOLD_BANDS": THRESHOLD_BANDS,
                "FAMILIES": FAMILIES,
                "SCENARIOS_PER_FAMILY": SCENARIOS_PER_FAMILY,
                "MAX_BUDGET": MAX_BUDGET,
                "CHECKPOINT_STEP": CHECKPOINT_STEP,
                "N_CHECKPOINTS": len(CHECKPOINTS),
                "APPENDIX_FIGURE_BUDGETS": APPENDIX_FIGURE_BUDGETS,
                "SWAP_FRACTIONS": SWAP_FRACTIONS,
                "GAMMAS": GAMMAS,
                "BETA_INIT_PILOT_SAMPLES": BETA_INIT_PILOT_SAMPLES,
                "BOOTSTRAP_RESAMPLES": BOOTSTRAP_RESAMPLES,
                "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
                "MCMC_CHAINS": MCMC_CHAINS,
                "MCMC_ESTIMATE_VARIANCE": MCMC_ESTIMATE_VARIANCE,
                "SAMC_N_BINS": SAMC_CFG.n_bins,
                "SAMC_LAMBDA_MIN_PILOT": SAMC_CFG.lambda_min_pilot,
                "N_JOBS": N_JOBS,
                "SAVE_OUTPUTS": SAVE_OUTPUTS,
                "RESUME_BLOCKS": RESUME_BLOCKS,
                "SKIP_IF_PRESENT": SKIP_IF_PRESENT,
                "LATEST_RESULTS_DIR": str(LATEST_RESULTS_DIR) if LATEST_RESULTS_DIR is not None else None,
            }, indent=2))
            """
        ),
        markdown_cell("## Load Scenario Inventory"),
        code_cell(
            """
            def scenario_keys_for_family(family: str, threshold_band: str, n: int) -> list[str]:
                if family == "gwas":
                    return [
                        f"gwas_additive_score_{threshold_band}_v{idx:02d}_n120"
                        for idx in range(1, int(n) + 1)
                    ]
                if family == "hep":
                    return [
                        f"poisson_diffmeans_hep_{threshold_band}_v{idx:02d}_n200"
                        for idx in range(1, int(n) + 1)
                    ]
                raise ValueError(f"Unknown family: {family}")


            def family_from_key(key: str) -> str:
                if key.startswith("gwas_additive_score_"):
                    return "gwas"
                if key.startswith("poisson_diffmeans_hep_"):
                    return "hep"
                raise ValueError(f"Could not infer family from key: {key}")


            def threshold_band_from_key(key: str) -> str:
                if "_near_" in key:
                    return "near"
                if "_above_" in key:
                    return "above"
                raise ValueError(f"Could not infer threshold band from key: {key}")


            scenario_keys = [
                key
                for threshold_band in THRESHOLD_BANDS
                for family in FAMILIES
                for key in scenario_keys_for_family(family, threshold_band, SCENARIOS_PER_FAMILY)
            ]
            scenarios = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=scenario_keys,
                portfolio_group=None,
                min_tail_states=MIN_TAIL_STATES,
            )
            scenarios_by_block = {
                (threshold_band, family): [
                    scenario
                    for scenario in scenarios
                    if threshold_band_from_key(scenario.key) == threshold_band
                    and family_from_key(scenario.key) == family
                ]
                for threshold_band in THRESHOLD_BANDS
                for family in FAMILIES
            }

            inventory_rows = []
            for scenario in scenarios:
                p0, p0_source = known_threshold_p0_for_scenario(scenario)
                inventory_rows.append({
                    "threshold_band": threshold_band_from_key(scenario.key),
                    "family": family_from_key(scenario.key),
                    "scenario": scenario.key,
                    "n": scenario.problem.n,
                    "n_treated": scenario.problem.n_treated,
                    "n_control": scenario.problem.n_control,
                    "exact_p": scenario.exact_p,
                    "p0": p0,
                    "p0_source": p0_source,
                    "p_over_p0": scenario.exact_p / p0,
                    "tail_hits": scenario.exact_tail_hits,
                    "n_perm": scenario.exact_n_perm,
                })
            inventory_df = pd.DataFrame(inventory_rows)
            display(
                inventory_df
                .groupby(["threshold_band", "family"])
                .agg(
                    n_scenarios=("scenario", "count"),
                    min_p_over_p0=("p_over_p0", "min"),
                    median_p_over_p0=("p_over_p0", "median"),
                    max_p_over_p0=("p_over_p0", "max"),
                    p0=("p0", "first"),
                )
                .reset_index()
            )
            inventory_df.head()
            """
        ),
        markdown_cell("## Build Method Blocks"),
        code_cell(
            """
            def gamma_label(gamma: float | None) -> str:
                return "none" if gamma is None else f"{float(gamma):.12g}".replace(".", "p")


            def swap_label(swap_fraction: float | int) -> str:
                if isinstance(swap_fraction, float):
                    return f"{int(round(100 * swap_fraction)):02d}pct"
                return f"{int(swap_fraction)}swap"


            def block_slug(block: dict) -> str:
                parts = [
                    f"band_{block['threshold_band']}",
                    f"family_{block['family']}",
                    f"swap_{swap_label(block['swap_fraction'])}",
                    f"method_{block['method']}",
                ]
                if block.get("gamma") is not None:
                    parts.append(f"gamma_{gamma_label(block['gamma'])}")
                return "__".join(parts)


            method_blocks = []
            for threshold_band in THRESHOLD_BANDS:
                for family in FAMILIES:
                    block_scenarios = scenarios_by_block[(threshold_band, family)]
                    if len(block_scenarios) != SCENARIOS_PER_FAMILY:
                        raise RuntimeError(
                            f"Expected {SCENARIOS_PER_FAMILY} scenarios for {(threshold_band, family)}, "
                            f"found {len(block_scenarios)}."
                        )
                    for swap_fraction in SWAP_FRACTIONS:
                        method_blocks.append({
                            "method": "samc",
                            "threshold_band": threshold_band,
                            "family": family,
                            "swap_fraction": float(swap_fraction),
                            "gamma": None,
                            "scenarios": block_scenarios,
                        })
                        for gamma in GAMMAS:
                            method_blocks.append({
                                "method": "mcmc_is",
                                "threshold_band": threshold_band,
                                "family": family,
                                "swap_fraction": float(swap_fraction),
                                "gamma": float(gamma),
                                "scenarios": block_scenarios,
                            })
                            method_blocks.append({
                                "method": "hard_step",
                                "threshold_band": threshold_band,
                                "family": family,
                                "swap_fraction": float(swap_fraction),
                                "gamma": float(gamma),
                                "scenarios": block_scenarios,
                            })

            block_plan = pd.DataFrame([
                {
                    "block_index": idx,
                    "block_slug": block_slug(block),
                    "threshold_band": block["threshold_band"],
                    "family": block["family"],
                    "method": block["method"],
                    "swap_fraction": block["swap_fraction"],
                    "gamma": block["gamma"],
                    "n_scenarios": len(block["scenarios"]),
                }
                for idx, block in enumerate(method_blocks)
            ])
            display(block_plan)
            print(f"{len(block_plan)} method blocks; each block saves independently.")
            """
        ),
        markdown_cell("## Run Blocks"),
        code_cell(
            """
            def resolve_latest_results_dir() -> Path | None:
                if not bool(SKIP_IF_PRESENT):
                    return None
                if LATEST_RESULTS_DIR is None:
                    raise ValueError("SKIP_IF_PRESENT=True requires LATEST_RESULTS_DIR to be set.")
                latest = Path(LATEST_RESULTS_DIR).expanduser()
                if not latest.is_absolute():
                    latest = (project_root / latest).resolve()
                else:
                    latest = latest.resolve()
                if not latest.exists():
                    raise FileNotFoundError(f"LATEST_RESULTS_DIR does not exist: {latest}")
                return latest


            latest_results_dir = resolve_latest_results_dir()


            def _same_optional_float(left, right) -> bool:
                if left is None and right is None:
                    return True
                if left is None or right is None:
                    return False
                return abs(float(left) - float(right)) <= 1e-12


            def prior_block_is_compatible(prior_block_dir: Path, block: dict) -> bool:
                records_path = prior_block_dir / "block_records.jsonl"
                if not records_path.exists():
                    return False
                metadata_path = prior_block_dir / "block_metadata.json"
                if metadata_path.exists():
                    metadata = read_json(metadata_path)
                    if str(metadata.get("method")) != str(block["method"]):
                        return False
                    if str(metadata.get("family")) != str(block["family"]):
                        return False
                    if str(metadata.get("threshold_band")) != str(block["threshold_band"]):
                        return False
                    if not _same_optional_float(metadata.get("swap_fraction"), block["swap_fraction"]):
                        return False
                    if not _same_optional_float(metadata.get("gamma"), block["gamma"]):
                        return False
                    if int(metadata.get("max_budget", -1)) != int(MAX_BUDGET):
                        return False
                    if tuple(int(x) for x in metadata.get("checkpoints", [])) != tuple(int(x) for x in CHECKPOINTS):
                        return False
                records = pd.read_json(records_path, orient="records", lines=True)
                expected_scenarios = {scenario.key for scenario in block["scenarios"]}
                if set(records["scenario"].dropna().astype(str)) != expected_scenarios:
                    return False
                if int(records["checkpoint"].max()) != int(MAX_BUDGET):
                    return False
                return True


            def copy_prior_block_if_present(block: dict, slug: str, block_dir: Path | None) -> Path | None:
                if latest_results_dir is None:
                    return None
                prior_block_dir = latest_results_dir / slug
                if not prior_block_is_compatible(prior_block_dir, block):
                    return None
                if block_dir is not None and block_dir.resolve() != prior_block_dir.resolve():
                    block_dir.mkdir(parents=True, exist_ok=True)
                    for filename in ("block_records.jsonl", "block_summary.json", "block_metadata.json"):
                        source = prior_block_dir / filename
                        if source.exists():
                            shutil.copy2(source, block_dir / filename)
                return prior_block_dir


            if SAVE_OUTPUTS:
                if RUN_DIR_OVERRIDE is not None:
                    run_dir = Path(RUN_DIR_OVERRIDE).expanduser().resolve()
                    run_dir.mkdir(parents=True, exist_ok=True)
                else:
                    run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "threshold_grid_cross_method")
            else:
                run_dir = None

            block_results = []
            for block_index, block in enumerate(method_blocks):
                slug = block_slug(block)
                block_dir = (run_dir / slug) if run_dir is not None else None
                reused_from_prior = copy_prior_block_if_present(block, slug, block_dir)
                print(json.dumps({
                    "block": f"{block_index + 1}/{len(method_blocks)}",
                    "slug": slug,
                    "method": block["method"],
                    "threshold_band": block["threshold_band"],
                    "family": block["family"],
                    "swap_fraction": block["swap_fraction"],
                    "gamma": block["gamma"],
                    "n_scenarios": len(block["scenarios"]),
                    "output_dir": str(block_dir) if block_dir is not None else None,
                    "reused_from_prior": str(reused_from_prior) if reused_from_prior is not None else None,
                }, indent=2))
                result = run_threshold_grid_method_block(
                    block["scenarios"],
                    method=block["method"],
                    family=block["family"],
                    threshold_band=block["threshold_band"],
                    checkpoints=CHECKPOINTS,
                    swap_fraction=block["swap_fraction"],
                    gamma=block["gamma"],
                    mcmc_cfg=MCMC_CFG,
                    samc_cfg=SAMC_CFG,
                    base_seed=BASE_SEED + 1_000_000 * block_index,
                    n_jobs=N_JOBS,
                    output_dir=block_dir,
                    resume=RESUME_BLOCKS,
                    block_label=slug,
                )
                block_results.append({
                    "block_index": block_index,
                    "block_slug": slug,
                    "threshold_band": block["threshold_band"],
                    "family": block["family"],
                    "method": block["method"],
                    "swap_fraction": block["swap_fraction"],
                    "gamma": block["gamma"],
                    "n_records": len(result["records"]),
                    "n_summary_rows": len(result["summary"]),
                    "output_dir": str(result["output_dir"]) if result["output_dir"] is not None else None,
                    "loaded_from": str(result["loaded_from"]) if result.get("loaded_from") is not None else None,
                    "reused_from_prior": str(reused_from_prior) if reused_from_prior is not None else None,
                    "wall_time_sec": result["metadata"].get("wall_time_sec"),
                    "status": result["metadata"].get("status"),
                    "result": result,
                })

            block_manifest = pd.DataFrame([
                {key: value for key, value in item.items() if key != "result"}
                for item in block_results
            ])
            display(block_manifest)
            run_dir
            """
        ),
        markdown_cell("## Aggregate And Save"),
        code_cell(
            """
            all_records = [
                row
                for block_result in block_results
                for row in block_result["result"]["records"]
            ]
            all_summary = summarize_records(
                all_records,
                group_fields=("threshold_band", "family", "swap_fraction", "method", "gamma_label", "checkpoint"),
            )
            records_df = pd.DataFrame(all_records)
            summary_df = pd.DataFrame(all_summary)

            if SAVE_OUTPUTS and run_dir is not None:
                records_df.to_json(run_dir / "all_records.jsonl", orient="records", lines=True)
                summary_df.to_json(run_dir / "summary.json", orient="records", indent=2)
                block_manifest.to_json(run_dir / "block_manifest.json", orient="records", indent=2)
                inventory_df.to_json(run_dir / "scenario_inventory.json", orient="records", indent=2)
                (run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "CATALOG_PATH": str(CATALOG_PATH),
                            "THRESHOLD_BANDS": THRESHOLD_BANDS,
                            "FAMILIES": FAMILIES,
                            "SCENARIOS_PER_FAMILY": SCENARIOS_PER_FAMILY,
                            "MAX_BUDGET": MAX_BUDGET,
                            "CHECKPOINT_STEP": CHECKPOINT_STEP,
                            "CHECKPOINTS": CHECKPOINTS,
                            "SWAP_FRACTIONS": SWAP_FRACTIONS,
                            "GAMMAS": GAMMAS,
                            "BETA_INIT_PILOT_SAMPLES": BETA_INIT_PILOT_SAMPLES,
                            "N_JOBS": N_JOBS,
                            "BASE_SEED": BASE_SEED,
                            "MCMC_CHAINS": MCMC_CHAINS,
                            "MCMC_ESTIMATE_VARIANCE": MCMC_ESTIMATE_VARIANCE,
                            "SAMC_N_BINS": SAMC_CFG.n_bins,
                            "SAMC_LAMBDA_MIN_PILOT": SAMC_CFG.lambda_min_pilot,
                            "RESUME_BLOCKS": RESUME_BLOCKS,
                            "SKIP_IF_PRESENT": SKIP_IF_PRESENT,
                            "LATEST_RESULTS_DIR": str(latest_results_dir) if latest_results_dir is not None else None,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"Saved aggregate outputs to {run_dir}")

            display(
                summary_df
                .query("checkpoint == @MAX_BUDGET")
                .sort_values(["threshold_band", "family", "swap_fraction", "method", "gamma_label"])
                .reset_index(drop=True)
            )
            """
        ),
        markdown_cell("## Figure 1: Tuning Behavior"),
        code_cell(
            """
            figure_dir = (run_dir / "figures") if (SAVE_OUTPUTS and run_dir is not None) else None
            figure_support_dir = (figure_dir / "support") if figure_dir is not None else None
            if figure_support_dir is not None:
                figure_support_dir.mkdir(parents=True, exist_ok=True)

            def figure_budget_suffix(budget: int) -> str:
                if int(budget) == int(MAX_BUDGET):
                    return ""
                label = f"{float(budget) / 1_000_000.0:g}".replace(".", "p")
                return f"_budget_{label}m"

            figure_budgets = tuple(dict.fromkeys((*APPENDIX_FIGURE_BUDGETS, MAX_BUDGET)))
            figure1_summaries = {}
            for figure_budget in figure_budgets:
                suffix = figure_budget_suffix(int(figure_budget))
                fig1_median_path = (figure_support_dir / f"figure1_tilt_gamma_median_rrmse{suffix}.png") if figure_support_dir is not None else None
                fig1_mean_path = (figure_support_dir / f"figure1_tilt_gamma_mean_rrmse{suffix}.png") if figure_support_dir is not None else None
                figure1_median_summary = plot_threshold_grid_tilt_family_rrmse(
                    all_records,
                    max_budget=int(figure_budget),
                    metric="median",
                    families=tuple(FAMILIES),
                    gamma_values=tuple(GAMMAS),
                    swap_fractions=tuple(SWAP_FRACTIONS),
                    save_path=fig1_median_path,
                    table_save_path=(figure_support_dir / f"figure1_tilt_gamma_median_rrmse{suffix}_summary.json") if figure_support_dir is not None else None,
                )
                figure1_mean_summary = plot_threshold_grid_tilt_family_rrmse(
                    all_records,
                    max_budget=int(figure_budget),
                    metric="mean",
                    families=tuple(FAMILIES),
                    gamma_values=tuple(GAMMAS),
                    swap_fractions=tuple(SWAP_FRACTIONS),
                    save_path=fig1_mean_path,
                    table_save_path=(figure_support_dir / f"figure1_tilt_gamma_mean_rrmse{suffix}_summary.json") if figure_support_dir is not None else None,
                )
                figure1_summaries[int(figure_budget)] = {
                    "median": figure1_median_summary,
                    "mean": figure1_mean_summary,
                    "median_path": fig1_median_path,
                    "mean_path": fig1_mean_path,
                }
                if figure_dir is not None:
                    plot_threshold_grid_tilt_family_rrmse(
                        all_records,
                        max_budget=int(figure_budget),
                        metric="median",
                        families=tuple(FAMILIES),
                        gamma_values=tuple(GAMMAS),
                        swap_fractions=tuple(SWAP_FRACTIONS),
                        save_path=figure_dir / f"figure1_tilt_gamma_median_rrmse{suffix}.pdf",
                    )
                    plot_threshold_grid_tilt_family_rrmse(
                        all_records,
                        max_budget=int(figure_budget),
                        metric="mean",
                        families=tuple(FAMILIES),
                        gamma_values=tuple(GAMMAS),
                        swap_fractions=tuple(SWAP_FRACTIONS),
                        save_path=figure_dir / f"figure1_tilt_gamma_mean_rrmse{suffix}.pdf",
                    )
            figure1_median_summary = figure1_summaries[int(MAX_BUDGET)]["median"]
            figure1_mean_summary = figure1_summaries[int(MAX_BUDGET)]["mean"]
            fig1_median_path = figure1_summaries[int(MAX_BUDGET)]["median_path"]
            fig1_mean_path = figure1_summaries[int(MAX_BUDGET)]["mean_path"]
            if fig1_median_path is not None:
                display(Image(filename=str(fig1_median_path)))
                display(Image(filename=str(fig1_mean_path)))
                print(f"Saved Figure 1 median variant to {fig1_median_path}")
                print(f"Saved Figure 1 mean variant to {fig1_mean_path}")
                if APPENDIX_FIGURE_BUDGETS:
                    for appendix_budget in APPENDIX_FIGURE_BUDGETS:
                        paths = figure1_summaries[int(appendix_budget)]
                        print(f"Saved appendix Figure 1 variants to {paths['median_path']} and {paths['mean_path']}")
            display(
                pd.DataFrame(figure1_median_summary)
                .query("method in ['mcmc_is_no_oracle', 'hard_step']")
                .sort_values(["family", "method", "swap_fraction", "gamma"])
                .reset_index(drop=True)
            )
            """
        ),
        markdown_cell("## Figure 2: Practical Takeaway"),
        code_cell(
            """
            figure2_summaries = {}
            for figure_budget in figure_budgets:
                suffix = figure_budget_suffix(int(figure_budget))
                fig2_path = (figure_support_dir / f"figure2_best_smooth_vs_samc_rrmse{suffix}.png") if figure_support_dir is not None else None
                figure2_summary = plot_threshold_grid_best_practical_rrmse(
                    all_records,
                    max_budget=int(figure_budget),
                    families=tuple(FAMILIES),
                    selection_metric="median",
                    save_path=fig2_path,
                    table_save_path=(figure_support_dir / f"figure2_best_smooth_vs_samc_rrmse{suffix}_summary.json") if figure_support_dir is not None else None,
                )
                figure2_summaries[int(figure_budget)] = {
                    "summary": figure2_summary,
                    "path": fig2_path,
                }
                if figure_dir is not None:
                    plot_threshold_grid_best_practical_rrmse(
                        all_records,
                        max_budget=int(figure_budget),
                        families=tuple(FAMILIES),
                        selection_metric="median",
                        save_path=figure_dir / f"figure2_best_smooth_vs_samc_rrmse{suffix}.pdf",
                    )
            figure2_summary = figure2_summaries[int(MAX_BUDGET)]["summary"]
            fig2_path = figure2_summaries[int(MAX_BUDGET)]["path"]
            if fig2_path is not None:
                display(Image(filename=str(fig2_path)))
                print(f"Saved Figure 2 to {fig2_path}")
                if APPENDIX_FIGURE_BUDGETS:
                    for appendix_budget in APPENDIX_FIGURE_BUDGETS:
                        print(f"Saved appendix Figure 2 to {figure2_summaries[int(appendix_budget)]['path']}")
            display(pd.DataFrame(figure2_summary))
            """
        ),
        markdown_cell("## Figure 3: Scenario-Level Comparison"),
        code_cell(
            """
            figure3_summaries = {}
            for figure_budget in figure_budgets:
                suffix = figure_budget_suffix(int(figure_budget))
                fig3_path = (figure_support_dir / f"figure3_best_smooth_vs_samc_scenario_rrmse{suffix}.png") if figure_support_dir is not None else None
                figure3_summary = plot_threshold_grid_best_practical_scenario_scatter(
                    all_records,
                    max_budget=int(figure_budget),
                    families=tuple(FAMILIES),
                    selection_metric="median",
                    save_path=fig3_path,
                    table_save_path=(figure_support_dir / f"figure3_best_smooth_vs_samc_scenario_rrmse{suffix}_summary.json") if figure_support_dir is not None else None,
                )
                figure3_summaries[int(figure_budget)] = {
                    "summary": figure3_summary,
                    "path": fig3_path,
                }
                if figure_dir is not None:
                    plot_threshold_grid_best_practical_scenario_scatter(
                        all_records,
                        max_budget=int(figure_budget),
                        families=tuple(FAMILIES),
                        selection_metric="median",
                        save_path=figure_dir / f"figure3_best_smooth_vs_samc_scenario_rrmse{suffix}.pdf",
                    )
            figure3_summary = figure3_summaries[int(MAX_BUDGET)]["summary"]
            fig3_path = figure3_summaries[int(MAX_BUDGET)]["path"]
            if fig3_path is not None:
                display(Image(filename=str(fig3_path)))
                print(f"Saved Figure 3 to {fig3_path}")
                if APPENDIX_FIGURE_BUDGETS:
                    for appendix_budget in APPENDIX_FIGURE_BUDGETS:
                        print(f"Saved appendix Figure 3 to {figure3_summaries[int(appendix_budget)]['path']}")
            display(pd.DataFrame(figure3_summary).drop(columns=["paired_points"], errors="ignore"))
            """
        ),
        markdown_cell("## Figure 4: Threshold-Scale Estimates"),
        code_cell(
            """
            fig4_path = (figure_support_dir / "figure4_estimate_vs_threshold_ratio.png") if figure_support_dir is not None else None
            figure4_points = plot_threshold_grid_estimate_vs_threshold_ratio(
                all_records,
                max_budget=int(MAX_BUDGET),
                families=tuple(FAMILIES),
                selection_metric="median",
                save_path=fig4_path,
                table_save_path=(figure_support_dir / "figure4_estimate_vs_threshold_ratio_points.json") if figure_support_dir is not None else None,
            )
            if figure_dir is not None:
                plot_threshold_grid_estimate_vs_threshold_ratio(
                    all_records,
                    max_budget=int(MAX_BUDGET),
                    families=tuple(FAMILIES),
                    selection_metric="median",
                    save_path=figure_dir / "figure4_estimate_vs_threshold_ratio.pdf",
                )
            if fig4_path is not None:
                display(Image(filename=str(fig4_path)))
                print(f"Saved Figure 4 to {fig4_path}")
            figure4_summary = (
                pd.DataFrame(figure4_points)
                .groupby(["family", "display_method", "config_label"], dropna=False)
                .agg(
                    n_scenarios=("scenario", "count"),
                    n_estimate_above_threshold=("estimate_above_threshold", "sum"),
                    n_false_negative=("false_negative", "sum"),
                    median_abs_ratio_error=("abs_ratio_error", "median"),
                )
                .reset_index()
            )
            display(figure4_summary)
            """
        ),
        markdown_cell("## Bootstrap Diagnostics"),
        code_cell(
            """
            bootstrap_summaries = {}
            bootstrap_summary_rows = []
            bootstrap_budgets = tuple(CHECKPOINTS)

            def bootstrap_budget_suffix(budget: int) -> str:
                if int(budget) % 1_000_000 == 0:
                    label = f"{int(budget) // 1_000_000}m"
                elif int(budget) % 1_000 == 0:
                    label = f"{int(budget) // 1_000}k"
                else:
                    label = str(int(budget))
                return f"_budget_{label}"

            for figure_budget in bootstrap_budgets:
                suffix = bootstrap_budget_suffix(int(figure_budget))
                bootstrap_path = (
                    figure_support_dir / f"figure3_bootstrap_diagnostics{suffix}.json"
                ) if figure_support_dir is not None else None
                bootstrap_rows = bootstrap_threshold_grid_best_practical_diagnostics(
                    all_records,
                    max_budget=int(figure_budget),
                    families=tuple(FAMILIES),
                    selection_metric="median",
                    n_bootstrap=int(BOOTSTRAP_RESAMPLES),
                    seed=int(BOOTSTRAP_SEED + int(figure_budget)),
                    save_path=bootstrap_path,
                )
                bootstrap_summaries[int(figure_budget)] = {
                    "summary": bootstrap_rows,
                    "path": bootstrap_path,
                }
                compact_rows = [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"bootstrap_rrmse_ratios", "bootstrap_win_rates", "paired_points"}
                    }
                    for row in bootstrap_rows
                ]
                bootstrap_summary_rows.extend(compact_rows)
                if bootstrap_path is not None:
                    print(f"Saved bootstrap diagnostics to {bootstrap_path}")

            bootstrap_summary_df = pd.DataFrame(bootstrap_summary_rows).sort_values(["checkpoint", "family"])
            if figure_dir is not None:
                bootstrap_summary_path = figure_support_dir / "figure3_bootstrap_diagnostics_summary.json"
                bootstrap_summary_csv_path = figure_support_dir / "figure3_bootstrap_diagnostics_summary.csv"
                write_json(bootstrap_summary_path, bootstrap_summary_rows)
                bootstrap_summary_df.to_csv(bootstrap_summary_csv_path, index=False)
                print(f"Saved bootstrap diagnostic summary to {bootstrap_summary_path}")
                print(f"Saved bootstrap diagnostic summary to {bootstrap_summary_csv_path}")
            display(bootstrap_summary_df)
            """
        ),
        markdown_cell("## Bootstrap Diagnostics: Fixed 5M Configurations"),
        code_cell(
            """
            fixed_config_bootstrap_summaries = {}
            fixed_config_bootstrap_summary_rows = []
            fixed_config_selection_budget = int(MAX_BUDGET)
            for figure_budget in bootstrap_budgets:
                suffix = bootstrap_budget_suffix(int(figure_budget))
                bootstrap_path = (
                    figure_support_dir / f"figure3_bootstrap_fixed_5m_config_diagnostics{suffix}.json"
                ) if figure_support_dir is not None else None
                bootstrap_rows = bootstrap_threshold_grid_best_practical_diagnostics(
                    all_records,
                    max_budget=int(figure_budget),
                    config_selection_budget=fixed_config_selection_budget,
                    families=tuple(FAMILIES),
                    selection_metric="median",
                    n_bootstrap=int(BOOTSTRAP_RESAMPLES),
                    seed=int(BOOTSTRAP_SEED + 7_000_000 + int(figure_budget)),
                    save_path=bootstrap_path,
                )
                fixed_config_bootstrap_summaries[int(figure_budget)] = {
                    "summary": bootstrap_rows,
                    "path": bootstrap_path,
                }
                compact_rows = [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"bootstrap_rrmse_ratios", "bootstrap_win_rates", "paired_points"}
                    }
                    for row in bootstrap_rows
                ]
                fixed_config_bootstrap_summary_rows.extend(compact_rows)
                if bootstrap_path is not None:
                    print(f"Saved fixed-config bootstrap diagnostics to {bootstrap_path}")

            fixed_config_bootstrap_summary_df = (
                pd.DataFrame(fixed_config_bootstrap_summary_rows)
                .sort_values(["checkpoint", "family"])
            )
            if figure_dir is not None:
                fixed_summary_path = figure_support_dir / "figure3_bootstrap_fixed_5m_config_diagnostics_summary.json"
                fixed_summary_csv_path = figure_support_dir / "figure3_bootstrap_fixed_5m_config_diagnostics_summary.csv"
                write_json(fixed_summary_path, fixed_config_bootstrap_summary_rows)
                fixed_config_bootstrap_summary_df.to_csv(fixed_summary_csv_path, index=False)
                print(f"Saved fixed-config bootstrap diagnostic summary to {fixed_summary_path}")
                print(f"Saved fixed-config bootstrap diagnostic summary to {fixed_summary_csv_path}")
            display(fixed_config_bootstrap_summary_df)
            """
        ),
        markdown_cell("## Bootstrap Diagnostics: Fixed 1M Configurations"),
        code_cell(
            """
            fixed_1m_config_bootstrap_summaries = {}
            fixed_1m_config_bootstrap_summary_rows = []
            fixed_1m_config_selection_budget = 1_000_000
            for figure_budget in bootstrap_budgets:
                suffix = bootstrap_budget_suffix(int(figure_budget))
                bootstrap_path = (
                    figure_support_dir / f"figure3_bootstrap_fixed_1m_config_diagnostics{suffix}.json"
                ) if figure_support_dir is not None else None
                bootstrap_rows = bootstrap_threshold_grid_best_practical_diagnostics(
                    all_records,
                    max_budget=int(figure_budget),
                    config_selection_budget=fixed_1m_config_selection_budget,
                    families=tuple(FAMILIES),
                    selection_metric="median",
                    n_bootstrap=int(BOOTSTRAP_RESAMPLES),
                    seed=int(BOOTSTRAP_SEED + 1_000_000 + int(figure_budget)),
                    save_path=bootstrap_path,
                )
                fixed_1m_config_bootstrap_summaries[int(figure_budget)] = {
                    "summary": bootstrap_rows,
                    "path": bootstrap_path,
                }
                compact_rows = [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"bootstrap_rrmse_ratios", "bootstrap_win_rates", "paired_points"}
                    }
                    for row in bootstrap_rows
                ]
                fixed_1m_config_bootstrap_summary_rows.extend(compact_rows)
                if bootstrap_path is not None:
                    print(f"Saved fixed-1M-config bootstrap diagnostics to {bootstrap_path}")

            fixed_1m_config_bootstrap_summary_df = (
                pd.DataFrame(fixed_1m_config_bootstrap_summary_rows)
                .sort_values(["checkpoint", "family"])
            )
            if figure_dir is not None:
                fixed_1m_summary_path = figure_support_dir / "figure3_bootstrap_fixed_1m_config_diagnostics_summary.json"
                fixed_1m_summary_csv_path = figure_support_dir / "figure3_bootstrap_fixed_1m_config_diagnostics_summary.csv"
                write_json(fixed_1m_summary_path, fixed_1m_config_bootstrap_summary_rows)
                fixed_1m_config_bootstrap_summary_df.to_csv(fixed_1m_summary_csv_path, index=False)
                print(f"Saved fixed-1M-config bootstrap diagnostic summary to {fixed_1m_summary_path}")
                print(f"Saved fixed-1M-config bootstrap diagnostic summary to {fixed_1m_summary_csv_path}")
            display(fixed_1m_config_bootstrap_summary_df)
            """
        ),
        markdown_cell("## Reload A Saved Run"),
        code_cell(
            """
            def load_threshold_grid_run(saved_run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                saved_run_dir = Path(saved_run_dir)
                records = pd.read_json(saved_run_dir / "all_records.jsonl", orient="records", lines=True)
                summary = pd.read_json(saved_run_dir / "summary.json")
                manifest = pd.read_json(saved_run_dir / "block_manifest.json")
                return records, summary, manifest


            if SAVE_OUTPUTS and run_dir is not None:
                reloaded_records_df, reloaded_summary_df, reloaded_manifest_df = load_threshold_grid_run(run_dir)
                display(reloaded_manifest_df[["block_index", "block_slug", "status", "n_records"]].head())
                print({
                    "records": len(reloaded_records_df),
                    "summary_rows": len(reloaded_summary_df),
                    "run_dir": str(run_dir),
                })
            """
        ),
    ]
    return notebook(cells)


def build_beta_notebook() -> dict:
    return build_oracle_beta_search_notebook()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    notebooks_dir = repo_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        notebooks_dir / "cross_method_simulation.ipynb": build_cross_method_notebook(),
        notebooks_dir / "mcmcis_oracle_beta_search.ipynb": build_oracle_beta_search_notebook(),
        notebooks_dir / "mcmcis_offline_objective_grid.ipynb": build_mcmc_objective_grid_notebook(),
        notebooks_dir / "mcmcis_scan_budget_policy.ipynb": build_mcmc_scan_budget_policy_notebook(),
        notebooks_dir / "mcmcis_scan_budget_grid.ipynb": build_mcmc_scan_budget_grid_notebook(),
        notebooks_dir / "cross_method_threshold_grid.ipynb": build_cross_method_threshold_grid_notebook(),
    }
    for path, data in outputs.items():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
