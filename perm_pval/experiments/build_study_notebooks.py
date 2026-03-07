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
        load_selected_scenarios,
        regenerate_beta_sweep_plots_from_saved,
        regenerate_cross_method_plots_from_saved,
        run_beta_checkpoint_study,
        run_cross_method_study,
        save_beta_sweep_outputs,
        save_cross_method_outputs,
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
                "bruteforce_welch_nonextreme_n22",
                "hypergeom_1e7",
                "rank_sum_dp_n40",
                "linear_stat_dp_n40",
                "poisson_diffmeans_righttail_tiny_n200",
            ]

            ESTIMATION_POINTS = (333_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000) if not FAST_MODE else (50_000, 100_000, 200_000)
            N_REPEATS = 5 if not FAST_MODE else 2
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
                proposal_fraction=0.075,
            )
            samc_cfg = SAMCWorkflowConfig(
                n_bins=40,
                t0=1_000.0,
                trace_every=200 if not FAST_MODE else 50,
                lambda_min_pilot=10_000 if not FAST_MODE else 500,
                proposal_fraction=0.075,
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
            RELOAD_SCENARIO_DIR = None
            # Example:
            # RELOAD_SCENARIO_DIR = project_root / "results" / "cross_method_notebook" / "20260306_120000_cross_method" / "rank_sum_dp_n40"

            if RELOAD_SCENARIO_DIR is not None:
                saved = load_cross_method_saved_output(RELOAD_SCENARIO_DIR)
                print(json.dumps({
                    "scenario": saved["metadata"]["scenario"],
                    "exact_p": saved["metadata"]["exact_p"],
                    "mcmc_beta_selection_budget": saved["metadata"]["beta_workflow"]["beta_selection_eval_total"],
                }, indent=2))
                regen = regenerate_cross_method_plots_from_saved(RELOAD_SCENARIO_DIR)
                for name, path in regen.items():
                    print(name, path)
                    display(Image(filename=str(path)))
            else:
                print("Set RELOAD_SCENARIO_DIR to a saved scenario directory to regenerate plots from disk only.")
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

            SCENARIO_KEY = "rank_sum_dp_n40"
            ESTIMATION_POINTS = (10_000, 100_000, 1_000_000) if not FAST_MODE else (2_000, 10_000, 20_000)
            BETA_MULTIPLIERS = (0.70, 0.90, 1.00, 1.15, 1.35)
            BETA_REPEATS = 5 if not FAST_MODE else 2
            N_JOBS = min(BETA_REPEATS, os.cpu_count() or 1)
            BASE_SEED = 54_321

            mcmc_cfg = MCMCWorkflowConfig(
                pilot_samples=20_000 if not FAST_MODE else 1_000,
                tune_steps=2_000 if not FAST_MODE else 1_000,
                local_scan_screen_total_steps=12_000 if not FAST_MODE else 1_000,
                local_scan_total_steps=64_000 if not FAST_MODE else 6_000,
                chains=2,
                thin=1,
                estimate_variance=True,
                proposal_fraction=0.075,
            )
            beta_cfg = BetaSweepStudyConfig(
                estimation_points=ESTIMATION_POINTS,
                repeats=BETA_REPEATS,
                beta_multipliers=BETA_MULTIPLIERS,
                chains=2,
                thin=1,
                estimate_variance=True,
                proposal_fraction=0.075,
                base_seed=BASE_SEED,
                n_jobs=N_JOBS,
            )

            print(json.dumps({
                "FAST_MODE": FAST_MODE,
                "SCENARIO_KEY": SCENARIO_KEY,
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
            scenario = load_selected_scenarios(
                catalog_path=CATALOG_PATH,
                scenario_keys=[SCENARIO_KEY],
                min_tail_states=2,
            )[0]

            beta_workflow = build_beta_workflow(
                scenario.problem,
                scenario.exact_p,
                mcmc_cfg,
                seed=BASE_SEED + 10_000,
            )

            run_dir = create_timestamped_run_dir(OUTPUT_ROOT, "beta_diag") if SAVE_OUTPUTS else None

            print(json.dumps({
                "scenario": scenario.key,
                "exact_p": scenario.exact_p,
                "beta0_laplace": beta_workflow["beta0_laplace"],
                "beta_hat_tuned": beta_workflow["beta_hat_tuned"],
                "beta_used": beta_workflow["beta_used"],
                "q_target": beta_workflow["q_target"],
            }, indent=2))
            """
        ),
        markdown_cell("## Run Beta Sweep Across Checkpoints"),
        code_cell(
            """
            beta_study = run_beta_checkpoint_study(
                scenario.problem,
                scenario.exact_p,
                beta_center=float(beta_workflow["beta_used"]),
                sigma_t=float(beta_workflow["sigma_t"]),
                beta_cfg=beta_cfg,
            )

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
            RELOAD_BETA_DIR = None
            # Example:
            # RELOAD_BETA_DIR = project_root / "results" / "mcmcis_beta_notebook" / "20260306_120000_beta_diag" / "rank_sum_dp_n40"

            if RELOAD_BETA_DIR is not None:
                saved = load_beta_sweep_saved_output(RELOAD_BETA_DIR)
                print(json.dumps({
                    "scenario_display": saved["metadata"]["scenario_display"],
                    "exact_p": saved["metadata"]["exact_p"],
                    "beta_center": saved["metadata"]["settings"]["beta_center"],
                }, indent=2))
                regen = regenerate_beta_sweep_plots_from_saved(RELOAD_BETA_DIR)
                for name, path in regen.items():
                    print(name, path)
                    display(Image(filename=str(path)))
            else:
                print("Set RELOAD_BETA_DIR to a saved beta-study directory to regenerate plots from disk only.")
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
    }
    for path, data in outputs.items():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
