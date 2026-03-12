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

    import json
    import os
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

    from perm_pval.experiments.notebook_studies import (
        BetaSweepStudyConfig,
        CrossMethodStudyConfig,
        DEFAULT_MCMC_OBJECTIVE_GRID_Q_MULTIPLIERS,
        DEFAULT_MCMC_OBJECTIVE_GRID_SWAP_COUNTS,
        MCMCWorkflowConfig,
        MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES,
        SAMCWorkflowConfig,
        build_beta_workflow,
        create_timestamped_run_dir,
        load_beta_sweep_saved_output,
        load_cross_method_saved_output,
        load_mcmc_objective_grid_saved_output,
        load_selected_scenarios,
        run_mcmc_objective_grid_study,
        save_mcmc_objective_grid_outputs,
        regenerate_beta_sweep_plots_from_saved,
        regenerate_cross_method_plots_from_saved,
        run_beta_checkpoint_study,
        run_cross_method_study,
        save_beta_sweep_outputs,
        save_cross_method_outputs,
        summarize_records,
    )

    pd.set_option("display.max_columns", 100)
    project_root
    """


def build_cross_method_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: Cross-Method Tiny-p Study

            Objective:
            - Compare `iid`, `mcmc_is`, and `samc` on fixed exact scenarios.
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
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            OUTPUT_ROOT = project_root / "results" / "cross_method_notebook"

            SCENARIO_KEYS_TO_RUN = [
                "hypergeom_1e7",
                "gwas_additive_score_n40",
                "linear_stat_dp_n40",
                "bruteforce_welch_nonextreme_n22",
            ]

            ESTIMATION_POINTS = (333_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000) if not FAST_MODE else (50_000, 100_000, 200_000)
            N_REPEATS = 7 if not FAST_MODE else 2
            N_JOBS = min(N_REPEATS, os.cpu_count() or 1)
            MIN_TAIL_STATES = 2
            BASE_SEED = 12_345

            cross_cfg = CrossMethodStudyConfig(
                estimation_points=ESTIMATION_POINTS,
                repeats=N_REPEATS,
                base_seed=BASE_SEED,
                iid_density_samples=120_000 if not FAST_MODE else 10_000,
                min_tail_states=MIN_TAIL_STATES,
                n_jobs=N_JOBS,
            )
            mcmc_cfg = MCMCWorkflowConfig(
                pilot_samples=20_000 if not FAST_MODE else 1_000,
                tune_steps=2_000 if not FAST_MODE else 1_000,
                local_scan_screen_total_steps=12_000 if not FAST_MODE else 1_000,
                local_scan_total_steps=64_000 if not FAST_MODE else 6_000,
                chains=2,
                thin=1,
                estimate_variance=True,
                proposal_size=0.1,
            )
            samc_cfg = SAMCWorkflowConfig(
                n_bins=50,
                t0=1_000.0,
                trace_every=200 if not FAST_MODE else 50,
                lambda_min_pilot=10_000 if not FAST_MODE else 500,
                proposal_size=0.1,
            )

            print(json.dumps({
                "FAST_MODE": FAST_MODE,
                "SCENARIO_KEYS_TO_RUN": SCENARIO_KEYS_TO_RUN,
                "ESTIMATION_POINTS": ESTIMATION_POINTS,
                "N_REPEATS": N_REPEATS,
                "N_JOBS": N_JOBS,
                "SAVE_OUTPUTS": SAVE_OUTPUTS,
            }, indent=2))
            """
        ),
        markdown_cell("## Load Scenarios"),
        code_cell(
            """
            scenarios = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=SCENARIO_KEYS_TO_RUN,
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
                print(f"Running {scenario.key} | exact p={scenario.exact_p:.3e}")
                study = run_cross_method_study(
                    scenario,
                    cross_cfg=cross_cfg,
                    mcmc_cfg=mcmc_cfg,
                    samc_cfg=samc_cfg,
                )
                cross_results[scenario.key] = study

                if SAVE_OUTPUTS and run_dir is not None:
                    save_cross_method_outputs(
                        scenario,
                        study,
                        output_dir=run_dir / scenario.key,
                        cross_cfg=cross_cfg,
                        mcmc_cfg=mcmc_cfg,
                        samc_cfg=samc_cfg,
                    )

                print(json.dumps({
                    "scenario": scenario.key,
                    "mcmc_beta_selection_budget": study["mcmc_beta_selection_budget"],
                    "beta_used": study["beta_workflow"]["beta_used"],
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
            # # RELOAD_SCENARIO_DIR = project_root / "results" / "cross_method_notebook" / "20260306_120000_cross_method" / "gwas_additive_score_n40"

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


def build_beta_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: MCMC-IS Beta Diagnostics

            Objective:
            - Fix one exact scenario.
            - Compare MCMC-IS performance across beta values.
            - Record estimates and diagnostics at intermediate estimation points.
            """
        ),
        code_cell(_common_setup_code()),
        markdown_cell(
            """
            ## Configuration

            `ESTIMATION_POINTS` controls checkpoint budgets.  
            The final checkpoint is used for the main beta boxplots; all checkpoints are used for convergence plots.
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            OUTPUT_ROOT = project_root / "results" / "mcmcis_beta_notebook"

            SCENARIO_KEYS_TO_RUN = [
                "hypergeom_1e7",
                "gwas_additive_score_n40",
                "linear_stat_dp_n40",
                "bruteforce_welch_nonextreme_n22",
            ]
            ESTIMATION_POINTS = (333_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000) if not FAST_MODE else (2_000, 10_000, 20_000)
            BETA_MULTIPLIERS = (0.5, 0.75, 1.00, 1.25, 1.5)
            BETA_REPEATS = 5 if not FAST_MODE else 2
            N_JOBS = min(BETA_REPEATS, os.cpu_count() or 1)
            MIN_TAIL_STATES = 2
            BASE_SEED = 54_321

            mcmc_cfg = MCMCWorkflowConfig(
                pilot_samples=20_000 if not FAST_MODE else 1_000,
                tune_steps=2_000 if not FAST_MODE else 1_000,
                local_scan_screen_total_steps=12_000 if not FAST_MODE else 1_000,
                local_scan_total_steps=64_000 if not FAST_MODE else 6_000,
                chains=2,
                thin=1,
                estimate_variance=True,
                proposal_size=0.1,
            )
            beta_cfg = BetaSweepStudyConfig(
                estimation_points=ESTIMATION_POINTS,
                repeats=BETA_REPEATS,
                beta_multipliers=BETA_MULTIPLIERS,
                chains=2,
                thin=1,
                estimate_variance=True,
                proposal_size=0.075,
                base_seed=BASE_SEED,
                n_jobs=N_JOBS,
            )

            print(json.dumps({
                "FAST_MODE": FAST_MODE,
                "SCENARIO_KEYS_TO_RUN": SCENARIO_KEYS_TO_RUN,
                "ESTIMATION_POINTS": ESTIMATION_POINTS,
                "BETA_MULTIPLIERS": BETA_MULTIPLIERS,
                "BETA_REPEATS": BETA_REPEATS,
                "N_JOBS": N_JOBS,
                "SAVE_OUTPUTS": SAVE_OUTPUTS,
            }, indent=2))
            """
        ),
        markdown_cell("## Load Scenario And Build Beta Workflow"),
        code_cell(
            """
            scenarios = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=SCENARIO_KEYS_TO_RUN,
                min_tail_states=MIN_TAIL_STATES,
            )

            run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "beta_diag") if SAVE_OUTPUTS else None

            pd.DataFrame([
                {
                    "scenario": s.key,
                    "exact_p": s.exact_p,
                    "tail_hits": s.exact_tail_hits,
                    "n_perm": s.exact_n_perm,
                    "exact_method": s.exact_method,
                }
                for s in scenarios
            ])
            """
        ),
        markdown_cell("## Run Beta Sweep Across Checkpoints"),
        code_cell(
            """
            beta_results = {}

            for scenario_idx, scenario in enumerate(scenarios):
                workflow_seed = BASE_SEED + 10_000 + 1_000 * scenario_idx
                beta_workflow = build_beta_workflow(
                    scenario.problem,
                    scenario.exact_p,
                    mcmc_cfg,
                    seed=workflow_seed,
                )

                print(json.dumps({
                    "scenario": scenario.key,
                    "exact_p": scenario.exact_p,
                    "beta0_laplace": beta_workflow["beta0_laplace"],
                    "beta_hat_tuned": beta_workflow["beta_hat_tuned"],
                    "beta_used": beta_workflow["beta_used"],
                    "q_target": beta_workflow["q_target"],
                }, indent=2))

                beta_study = run_beta_checkpoint_study(
                    scenario.problem,
                    scenario.exact_p,
                    beta_center=float(beta_workflow["beta_used"]),
                    sigma_t=float(beta_workflow["sigma_t"]),
                    beta_cfg=beta_cfg,
                )
                beta_results[scenario.key] = {
                    "study": beta_study,
                    "beta_workflow": beta_workflow,
                }

                if SAVE_OUTPUTS and run_dir is not None:
                    save_beta_sweep_outputs(
                        beta_study,
                        output_dir=run_dir / scenario.key,
                        scenario_name=scenario.description,
                        exact_p=scenario.exact_p,
                        beta_cfg=beta_cfg,
                        beta_workflow=beta_workflow,
                    )

                beta_summary_df = pd.DataFrame(beta_study["summary"]).sort_values(["checkpoint", "beta"])
                display(beta_summary_df[[
                    "checkpoint",
                    "beta",
                    "mean_estimate",
                    "rmse",
                    "mean_variance_estimate",
                    "mean_q_tilt_tail_share",
                    "mean_ess",
                    "mean_acceptance_rate",
                    "mean_weight_cv",
                ]])
            """
        ),
        markdown_cell("## Review Saved Figures"),
        code_cell(
            """
            if SAVE_OUTPUTS and run_dir is not None:
                scenario_dir = run_dir / scenario.key
                print(f"Saved outputs under: {scenario_dir}")
                display(Image(filename=str(scenario_dir / "beta_max_budget.png")))
                display(Image(filename=str(scenario_dir / "beta_convergence.png")))
            else:
                print("SAVE_OUTPUTS=False, so no saved figures to display.")
            """
        ),
        markdown_cell("## Reload Saved Results Without Rerunning"),
        code_cell(
            """
            # RELOAD_BETA_DIR = None
            # # Example:
            # # RELOAD_BETA_DIR = project_root / "results" / "mcmcis_beta_notebook" / "20260306_120000_beta_diag" / "gwas_additive_score_n40"

            # if RELOAD_BETA_DIR is not None:
            #     saved = load_beta_sweep_saved_output(RELOAD_BETA_DIR)
            #     print(json.dumps({
            #         "scenario_display": saved["metadata"]["scenario_display"],
            #         "exact_p": saved["metadata"]["exact_p"],
            #         "beta_center": saved["metadata"]["settings"]["beta_center"],
            #     }, indent=2))
            #     regen = regenerate_beta_sweep_plots_from_saved(RELOAD_BETA_DIR)
            #     for name, path in regen.items():
            #         print(name, path)
            #         display(Image(filename=str(path)))
            # else:
            #     print("Set RELOAD_BETA_DIR to a saved beta-study directory to regenerate plots from disk only.")
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

            SCENARIO_KEYS_TO_RUN = [
                "hypergeom_1e7",
                "gwas_additive_score_n40",
                "linear_stat_dp_n40",
                "bruteforce_welch_nonextreme_n22",
            ]

            Q_MULTIPLIERS = DEFAULT_MCMC_OBJECTIVE_GRID_Q_MULTIPLIERS
            N_SWAP_PAIRS = DEFAULT_MCMC_OBJECTIVE_GRID_SWAP_COUNTS
            TRIAL_REPEATS = 5 if not FAST_MODE else 2
            TRIAL_BUDGET = 200_000 if not FAST_MODE else 20_000
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
                proposal_size=0.1,
            )

            NOTEBOOK_CONFIG = {
                "FAST_MODE": FAST_MODE,
                "SCENARIO_KEYS_TO_RUN": SCENARIO_KEYS_TO_RUN,
                "Q_MULTIPLIERS": Q_MULTIPLIERS,
                "N_SWAP_PAIRS": N_SWAP_PAIRS,
                "TRIAL_REPEATS": TRIAL_REPEATS,
                "TRIAL_BUDGET": TRIAL_BUDGET,
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
                scenario_keys=SCENARIO_KEYS_TO_RUN,
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
                    "mean_objective_selobj",
                    "mean_objective_varhat",
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
            display(realistic_cross_scenario_df.sort_values(["objective_name", "scenario_key"])[[
                "scenario_key",
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
    ]
    return notebook(cells)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    notebooks_dir = repo_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        notebooks_dir / "cross_method_simulation.ipynb": build_cross_method_notebook(),
        notebooks_dir / "mcmcis_beta_diagnostics.ipynb": build_beta_notebook(),
        notebooks_dir / "mcmcis_offline_objective_grid.ipynb": build_mcmc_objective_grid_notebook(),
    }
    for path, data in outputs.items():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
