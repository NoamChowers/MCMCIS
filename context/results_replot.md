# Saved Results And Replotting

This note records the current contract for saved simulation outputs and how to regenerate comparison figures from disk only, without rerunning simulations and without relying on an existing notebook kernel state.

Scope:
- cross-method notebook outputs
- MCMC-IS beta notebook outputs
- current comparison/diagnostic figures only

## 1) What is saved

Each scenario output directory currently saves:

1. `run_records.jsonl`
- one JSON object per run-level row
- contains replicate-level checkpoint records
- enough for max-budget boxplots and other replicate-level summaries

2. `summary.json`
- aggregated rows by checkpoint and grouping key
- used for convergence and diagnostic line plots

3. `metadata.json`
- scenario metadata
- exact p-value
- configs
- beta workflow
- MCMC-IS beta-selection budget
- other labels needed for titles/annotations

Additionally, PNG figures are saved for convenience, but they are no longer the only way to recover plots.

## 2) What can be regenerated from disk only

Using the saved JSON/JSONL files, the current code can regenerate:

### Cross-method outputs

- `cross_method_max_budget.png`
- `cross_method_convergence.png`
- `cross_method_diagnostics.png`

### Beta-sweep outputs

- `beta_max_budget.png`
- `beta_convergence.png`

These are the figures supported by the current disk-backed replot helpers.

## 3) What is intentionally not reconstructable from disk

The following are not part of the saved-results contract:

- raw IID statistic draws used for the exploratory density plot
- raw MCMC traces
- raw SAMC theta traces
- raw importance weights
- any other full trajectory objects

Reason:
- these are exploratory / optional
- they add size and complexity
- the main comparison figures do not require them

So `iid_density.png` is saved as an image for convenience, but it is not regenerated from saved JSON artifacts.

## 4) Fresh-kernel replot API

Authoritative loader/replot helpers live in:

- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/perm_pval/experiments/notebook_studies.py`

### Cross-method

Load saved data:

- `load_cross_method_saved_output(output_dir)`

Regenerate plots:

- `regenerate_cross_method_plots_from_saved(output_dir, save_dir=None)`

### Beta diagnostics

Load saved data:

- `load_beta_sweep_saved_output(output_dir)`

Regenerate plots:

- `regenerate_beta_sweep_plots_from_saved(output_dir, save_dir=None)`

If `save_dir` is omitted, regenerated figures overwrite the standard PNG names in the original output directory.

## 5) Notebook behavior

Current notebooks:

- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/notebooks/cross_method_simulation.ipynb`
- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/notebooks/mcmcis_beta_diagnostics.ipynb`

now include a final section named:

- `Reload Saved Results Without Rerunning`

Those cells are the intended fresh-kernel workflow.

## 6) Budget semantics to remember

### Cross-method notebook

Checkpoint budgets are total budgets.

For MCMC-IS:

- `total budget = beta-selection budget + production-chain budget`

The saved metadata stores the fixed beta-selection budget under:

- `metadata["beta_workflow"]["beta_selection_eval_total"]`

and also in the study object as:

- `mcmc_beta_selection_budget`

This must be reflected in figure annotations and any future replots.

### Beta notebook

Checkpoint budgets are production budgets only.

The beta notebook intentionally ignores beta-selection cost because it studies estimator behavior conditional on fixed beta values.

## 7) Minimal saved-results contract

If future code changes the saved schema, maintain at least:

### Cross-method

- `metadata.json`
  - `scenario`
  - `scenario_display`
  - `exact_p`
  - `estimation_points`
  - `beta_workflow`
- `summary.json`
  - aggregated checkpoint summaries
- `run_records.jsonl`
  - replicate-level rows for max-budget boxplots

### Beta notebook

- `metadata.json`
  - `scenario_display`
  - `exact_p`
  - `settings`
- `summary.json`
- `run_records.jsonl`

If these remain stable, the disk-only replot path remains available.

## 8) Current implementation note

Plot generation is now robust in headless/fresh runs because:

- the plotting module uses the non-interactive matplotlib backend `Agg`
- log scaling is only applied when there are positive finite values to support it

That matters for tests, CI-like environments, and fresh kernels launched outside a GUI session.
