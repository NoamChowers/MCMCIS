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
        MCMCWorkflowConfig,
        SAMCWorkflowConfig,
        build_beta_workflow,
        create_timestamped_run_dir,
        load_beta_sweep_saved_output,
        load_cross_method_saved_output,
        load_mcmc_optuna_offline_saved_output,
        load_selected_scenarios,
        run_mcmc_optuna_trial_table,
        run_named_mcmc_checkpoint_study,
        save_mcmc_optuna_offline_outputs,
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


def build_mcmc_optuna_offline_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: Offline Optuna Tuning For MCMC-IS

            Objective:
            - Tune MCMC-IS hyperparameters offline on the same scenarios used in the production notebooks.
            - Compare several tuning objectives on a shared Optuna-managed trial table.
            - Re-run the chosen fixed settings up to the production checkpoints and compare them with the saved production baseline.
            """
        ),
        code_cell(_common_setup_code()),
        code_cell(
            """
            import matplotlib.pyplot as plt
            import numpy as np

            try:
                import optuna
            except ImportError as exc:
                raise ImportError(
                    "optuna is required for this notebook. Install the project dev dependencies, "
                    "for example with `uv pip install -e '.[dev]'`."
                ) from exc
            """
        ),
        markdown_cell(
            """
            ## Configuration

            This notebook is intentionally heavy by default.  
            It uses exact `p` values only for offline-oracle objective scoring and ignores tuning cost when it reruns the selected fixed settings.
            """
        ),
        code_cell(
            """
            FAST_MODE = False
            SAVE_OUTPUTS = True

            CATALOG_PATH = project_root / "results" / "exact_scenarios" / "v1" / "catalog.json"
            OUTPUT_ROOT = project_root / "results" / "mcmcis_optuna_offline"
            PRODUCTION_BASELINE_ROOT = project_root / "results" / "cross_method_notebook" / "20260307_150041_cross_method"

            SCENARIO_KEYS_TO_RUN = [
                "hypergeom_1e7",
                "gwas_additive_score_n40",
                "linear_stat_dp_n40",
                "bruteforce_welch_nonextreme_n22",
            ]

            Q_LOG10_BOUNDS = (-3.0, 3.0)
            TRIALS_PER_SCENARIO = 64 if not FAST_MODE else 8
            TRIAL_REPEATS = 3 if not FAST_MODE else 1
            TRIAL_BUDGET = 200_000 if not FAST_MODE else 20_000
            FINAL_REPEATS = 7 if not FAST_MODE else 2
            FINAL_ESTIMATION_POINTS = (333_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000) if not FAST_MODE else (20_000, 100_000, 200_000)
            N_JOBS = min(os.cpu_count() or 1, FINAL_REPEATS)
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
                "Q_LOG10_BOUNDS": Q_LOG10_BOUNDS,
                "TRIALS_PER_SCENARIO": TRIALS_PER_SCENARIO,
                "TRIAL_REPEATS": TRIAL_REPEATS,
                "TRIAL_BUDGET": TRIAL_BUDGET,
                "FINAL_REPEATS": FINAL_REPEATS,
                "FINAL_ESTIMATION_POINTS": FINAL_ESTIMATION_POINTS,
                "N_JOBS": N_JOBS,
                "BASE_SEED": BASE_SEED,
                "PRODUCTION_BASELINE_ROOT": str(PRODUCTION_BASELINE_ROOT),
            }

            print(json.dumps(NOTEBOOK_CONFIG, indent=2))
            """
        ),
        markdown_cell("## Notebook Helpers"),
        code_cell(
            """
            OBJECTIVE_COLUMNS = [
                ("oracle_rmse", "objective_oracle_rmse"),
                ("oracle_abs_log10", "objective_oracle_abs_log10"),
                ("diag_selection_objective_p0", "objective_diag_selection_objective_p0"),
                ("diag_variance_estimate", "objective_diag_variance_estimate"),
                ("diag_repeat_stability", "objective_diag_repeat_stability"),
            ]


            def build_comparison_records(final_study: dict, baseline_saved: dict) -> tuple[list[dict], list[dict]]:
                baseline_records = []
                for row in baseline_saved["records"]:
                    if row["method"] not in {"iid", "mcmc_is", "samc"}:
                        continue
                    clean = dict(row)
                    clean["label"] = f"production_{row['method']}"
                    baseline_records.append(clean)
                combined = [dict(row) for row in final_study["records"]] + baseline_records
                combined_summary = summarize_records(combined, group_fields=("checkpoint", "label"))
                return combined, combined_summary


            def plot_trial_objectives(trial_summary: list[dict], scenario_name: str, *, save_path: Path | None = None) -> None:
                df = pd.DataFrame(trial_summary)
                fig, axes = plt.subplots(2, 3, figsize=(16, 9))
                axes = axes.ravel()
                for ax, (title, column) in zip(axes, OBJECTIVE_COLUMNS):
                    values = np.asarray(df[column], dtype=float)
                    mask = np.isfinite(values) & (values > 0.0)
                    ax.set_title(title)
                    ax.set_xlabel("log10(q_multiplier)")
                    ax.set_ylabel("n_swap_pairs")
                    if np.any(mask):
                        colors = np.log10(values[mask])
                        scatter = ax.scatter(
                            df.loc[mask, "log10_q_multiplier"],
                            df.loc[mask, "n_swap_pairs"],
                            c=colors,
                            cmap="viridis",
                            s=55,
                            alpha=0.9,
                        )
                        fig.colorbar(scatter, ax=ax, label=f"log10({title})")
                    else:
                        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
                if len(axes) > len(OBJECTIVE_COLUMNS):
                    axes[-1].axis("off")
                fig.suptitle(f"Optuna trial table diagnostics: {scenario_name}")
                plt.tight_layout()
                if save_path is not None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path, dpi=170, bbox_inches="tight")
                plt.close(fig)


            def plot_final_budget_comparison(
                combined_records: list[dict],
                scenario_name: str,
                *,
                exact_p: float,
                max_budget: int,
                save_path: Path | None = None,
            ) -> None:
                sub = [dict(row) for row in combined_records if int(row["checkpoint"]) == int(max_budget)]
                labels = sorted({str(row["label"]) for row in sub})
                estimate_data = [np.asarray([float(row["estimate"]) for row in sub if str(row["label"]) == label], dtype=float) for label in labels]
                summary = summarize_records(sub, group_fields=("label",))
                rmse_map = {str(row["label"]): float(row["rmse"]) for row in summary}

                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                axes[0].boxplot(estimate_data, tick_labels=labels, showfliers=False)
                axes[0].axhline(float(exact_p), color="black", linestyle="--", linewidth=1.2, label="exact p")
                axes[0].set_title(f"Max-budget estimates: {scenario_name}")
                axes[0].set_ylabel("estimate")
                axes[0].tick_params(axis="x", rotation=35)
                axes[0].legend(loc="best")

                rmse_vals = [rmse_map[label] for label in labels]
                axes[1].bar(labels, rmse_vals, color="#4e79a7")
                if any(val > 0.0 for val in rmse_vals):
                    axes[1].set_yscale("log")
                axes[1].set_title(f"RMSE at B={max_budget:,}")
                axes[1].set_ylabel("rmse")
                axes[1].tick_params(axis="x", rotation=35)

                plt.tight_layout()
                if save_path is not None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path, dpi=170, bbox_inches="tight")
                plt.close(fig)


            def plot_convergence_comparison(
                comparison_summary: list[dict],
                scenario_name: str,
                *,
                exact_p: float,
                save_path: Path | None = None,
            ) -> None:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                labels = sorted({str(row["label"]) for row in comparison_summary})
                for label in labels:
                    sub = sorted(
                        [row for row in comparison_summary if str(row["label"]) == label],
                        key=lambda row: int(row["checkpoint"]),
                    )
                    x = np.asarray([int(row["checkpoint"]) for row in sub], dtype=float)
                    mean_est = np.asarray([float(row["mean_estimate"]) for row in sub], dtype=float)
                    rmse = np.asarray([float(row["rmse"]) for row in sub], dtype=float)
                    axes[0].plot(x, mean_est, marker="o", label=label)
                    axes[1].plot(x, rmse, marker="o", label=label)

                axes[0].axhline(float(exact_p), color="black", linestyle="--", linewidth=1.2, label="exact p")
                axes[0].set_xscale("log")
                axes[0].set_yscale("log")
                axes[0].set_title(f"Mean estimate: {scenario_name}")
                axes[0].set_xlabel("budget")
                axes[0].set_ylabel("mean estimate")
                axes[0].legend(loc="best", fontsize=8)

                axes[1].set_xscale("log")
                if any(float(row["rmse"]) > 0.0 for row in comparison_summary):
                    axes[1].set_yscale("log")
                axes[1].set_title(f"RMSE convergence: {scenario_name}")
                axes[1].set_xlabel("budget")
                axes[1].set_ylabel("rmse")
                axes[1].legend(loc="best", fontsize=8)

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

            run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "optuna_offline") if SAVE_OUTPUTS else None

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
        markdown_cell("## Run Offline Optuna Study"),
        code_cell(
            """
            optuna_results = {}

            for scenario_idx, scenario in enumerate(scenarios):
                print(f"Running offline Optuna study for {scenario.key} | exact p={scenario.exact_p:.3e}")
                baseline_dir = PRODUCTION_BASELINE_ROOT / scenario.key
                if not baseline_dir.exists():
                    raise FileNotFoundError(f"Missing production baseline directory: {baseline_dir}")
                baseline_saved = load_cross_method_saved_output(baseline_dir)

                trial_seed = BASE_SEED + 10_000 * (scenario_idx + 1)
                trial_study = run_mcmc_optuna_trial_table(
                    scenario.problem,
                    scenario.exact_p,
                    mcmc_cfg=mcmc_cfg,
                    trials_per_scenario=TRIALS_PER_SCENARIO,
                    trial_repeats=TRIAL_REPEATS,
                    trial_budget=TRIAL_BUDGET,
                    base_seed=trial_seed,
                    q_log10_bounds=Q_LOG10_BOUNDS,
                    n_jobs=N_JOBS,
                )

                objective_best_df = pd.DataFrame([
                    {
                        "objective": objective_name,
                        "trial_number": row["trial_number"],
                        "config_id": trial_study["objective_to_config"][objective_name],
                        "beta": row["beta"],
                        "n_swap_pairs": row["n_swap_pairs"],
                        "log10_q_multiplier": row["log10_q_multiplier"],
                        "q_trial": row["q_trial"],
                        "selected_objective_value": row["selected_objective_value"],
                    }
                    for objective_name, row in trial_study["objective_best"].items()
                ]).sort_values("objective")

                selected_configs = [dict(cfg, source="optuna") for cfg in trial_study["selected_configs"]]
                production_proposal_size = baseline_saved["metadata"]["mcmc_config"]["proposal_size"]
                production_beta = float(baseline_saved["metadata"]["beta_workflow"]["beta_used"])
                selected_configs.append(
                    {
                        "config_id": "production_fixed",
                        "label": "production_fixed",
                        "beta": production_beta,
                        "proposal_size": production_proposal_size,
                        "selected_by_objectives": ["saved_production_beta"],
                        "source": "production_saved_metadata",
                    }
                )

                final_study = run_named_mcmc_checkpoint_study(
                    scenario.problem,
                    scenario.exact_p,
                    config_specs=selected_configs,
                    sigma_t=float(trial_study["trial_context"]["sigma_t"]),
                    estimation_points=FINAL_ESTIMATION_POINTS,
                    repeats=FINAL_REPEATS,
                    base_seed=BASE_SEED + 500_000 + 10_000 * scenario_idx,
                    template_cfg=mcmc_cfg,
                    n_jobs=N_JOBS,
                )

                combined_records, comparison_summary = build_comparison_records(final_study, baseline_saved)

                scenario_dir = (run_dir / scenario.key) if (SAVE_OUTPUTS and run_dir is not None) else None
                if scenario_dir is not None:
                    plot_trial_objectives(
                        trial_study["trial_summary"],
                        scenario.description,
                        save_path=scenario_dir / "trial_objectives.png",
                    )
                    plot_final_budget_comparison(
                        combined_records,
                        scenario.description,
                        exact_p=scenario.exact_p,
                        max_budget=max(FINAL_ESTIMATION_POINTS),
                        save_path=scenario_dir / "final_budget_compare.png",
                    )
                    plot_convergence_comparison(
                        comparison_summary,
                        scenario.description,
                        exact_p=scenario.exact_p,
                        save_path=scenario_dir / "convergence_compare.png",
                    )
                    save_mcmc_optuna_offline_outputs(
                        {
                            "trial_records": trial_study["trial_records"],
                            "trial_summary": trial_study["trial_summary"],
                            "objective_best": trial_study["objective_best"],
                            "selected_configs": trial_study["selected_configs"],
                            "objective_to_config": trial_study["objective_to_config"],
                            "trial_context": trial_study["trial_context"],
                            "final_records": final_study["records"],
                            "final_summary": final_study["summary"],
                            "final_settings": final_study["settings"],
                        },
                        output_dir=scenario_dir,
                        scenario_name=scenario.description,
                        exact_p=scenario.exact_p,
                        notebook_config=NOTEBOOK_CONFIG,
                        production_baseline_dir=baseline_dir,
                    )

                optuna_results[scenario.key] = {
                    "trial_study": trial_study,
                    "baseline_saved": baseline_saved,
                    "final_study": final_study,
                    "comparison_summary": comparison_summary,
                    "combined_records": combined_records,
                }

                print(json.dumps({
                    "scenario": scenario.key,
                    "sigma_t": trial_study["trial_context"]["sigma_t"],
                    "q_target": trial_study["trial_context"]["q_target"],
                    "selected_configs": trial_study["selected_configs"],
                }, indent=2))

                display(pd.DataFrame(trial_study["trial_summary"]).sort_values("objective_oracle_rmse").head(12)[[
                    "trial_number",
                    "beta",
                    "n_swap_pairs",
                    "log10_q_multiplier",
                    "q_trial",
                    "objective_oracle_rmse",
                    "objective_oracle_abs_log10",
                    "objective_diag_selection_objective_p0",
                    "objective_diag_variance_estimate",
                    "objective_diag_repeat_stability",
                ]])
                display(objective_best_df)
                display(pd.DataFrame(trial_study["selected_configs"]).sort_values("config_id"))
                display(pd.DataFrame(comparison_summary).sort_values(["checkpoint", "label"])[[
                    "checkpoint",
                    "label",
                    "mean_estimate",
                    "rmse",
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
                print(f"Saved outputs under: {run_dir}")
                for scenario in scenarios:
                    scenario_dir = run_dir / scenario.key
                    print(f"\\n{scenario.key}")
                    display(Image(filename=str(scenario_dir / "trial_objectives.png")))
                    display(Image(filename=str(scenario_dir / "final_budget_compare.png")))
                    display(Image(filename=str(scenario_dir / "convergence_compare.png")))
            else:
                print("SAVE_OUTPUTS=False, so no saved figures to display.")
            """
        ),
        markdown_cell("## Reload Saved Results Without Rerunning"),
        code_cell(
            """
            # RELOAD_OPTUNA_DIR = None
            # # Example:
            # # RELOAD_OPTUNA_DIR = project_root / "results" / "mcmcis_optuna_offline" / "20260307_120000_optuna_offline" / "linear_stat_dp_n40"

            # if RELOAD_OPTUNA_DIR is not None:
            #     saved = load_mcmc_optuna_offline_saved_output(RELOAD_OPTUNA_DIR)
            #     print(json.dumps({
            #         "scenario_display": saved["metadata"]["scenario_display"],
            #         "exact_p": saved["metadata"]["exact_p"],
            #         "production_baseline_dir": saved["metadata"]["production_baseline_dir"],
            #     }, indent=2))
            #     display(pd.DataFrame(saved["trial_summary"]).head())
            #     display(pd.DataFrame(saved["final_summary"]).head())
            # else:
            #     print("Set RELOAD_OPTUNA_DIR to a saved Optuna-offline scenario directory to inspect saved results.")
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
        notebooks_dir / "mcmcis_optuna_offline.ipynb": build_mcmc_optuna_offline_notebook(),
    }
    for path, data in outputs.items():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
