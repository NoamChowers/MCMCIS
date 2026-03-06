# Checkpoint Semantics

This note defines what a "checkpoint" means in the current simulation notebooks and helpers.

Scope:
- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/notebooks/cross_method_simulation.ipynb`
- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/notebooks/mcmcis_beta_diagnostics.ipynb`
- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/perm_pval/experiments/notebook_studies.py`

## 1) Core rule

Checkpoints are prefixes of one continued run.

They are not independent reruns from scratch.

If:

- `ESTIMATION_POINTS = [10_000, 100_000, 1_000_000]`

then each method runs one trajectory up to the maximum budget and reports the estimate/diagnostics at each prefix.

Interpretation:

- the checkpoint at `100_000` means "what would the estimator have been if we had stopped at budget 100,000?"

## 2) Burn-in rule

Burn-in is dynamic per checkpoint.

For each checkpoint `N`:

- compute that checkpoint's estimator using only the first `N` iterations / evaluations
- recompute burn-in from that checkpoint's budget

So a sample can:

- count for an early checkpoint
- but be excluded for a later checkpoint if the later checkpoint has a larger burn-in threshold

This is intentional.

It matches the stop-at-`N` interpretation.

## 3) Method-specific checkpoint semantics

### IID

No burn-in.

Checkpoint `N` uses:

- the first `N` IID permutation draws
- cumulative tail-hit counts up to `N`

### MCMC-IS

Checkpoint `N` uses:

- the first `N` production-budget iterations
- per-checkpoint burn-in
- per-checkpoint thinning
- SNIS estimate recomputed from the retained prefix
- ESS, variance estimate, and other diagnostics recomputed from the retained prefix

For multi-chain runs:

- each chain is continued to the maximum per-chain budget
- each checkpoint slices the corresponding per-chain prefix

### SAMC

Checkpoint `N` uses:

- the first `N` SAMC updates of one continued adaptive chain
- `theta` as it stands at step `N`
- post-burn-in visit counts from the prefix up to `N`
- paper-style p-value estimator applied at that checkpoint

SAMC adaptation itself is continuous from step 1 through the max budget.

## 4) Cross-method notebook budget semantics

This is the most important special case.

In the cross-method notebook, checkpoints are total budgets.

For `iid` and `samc`, the reported checkpoint is the production budget directly.

For `mcmc_is`, the reported checkpoint is:

- `total budget = beta-selection budget + production-chain budget`

The MCMC-IS workflow first spends a fixed one-time budget on:

- IID pilot
- beta initialization
- beta tuning
- local beta scan

Call this:

- `beta_selection_budget`

Then for a user checkpoint `B`, the production MCMC chain gets:

- `production_chain_budget = B - beta_selection_budget`

Implementation rule:

- all cross-method checkpoints must satisfy `B > beta_selection_budget`

If not, the run should fail fast instead of silently producing nonsense.

## 5) Beta notebook budget semantics

The beta notebook is different.

There, checkpoints are production budgets only.

The beta-selection budget is intentionally ignored because the beta notebook studies:

- estimator quality conditional on a fixed beta value

not:

- end-to-end method cost including beta search

## 6) Saved fields to preserve

At run level, the current code saves enough information to keep these semantics explicit.

Important fields:

### Common

- `checkpoint`
- `eval_excl_tuning`
- `eval_incl_tuning`

### MCMC-IS in cross-method studies

- `mcmc_chain_budget`
- `mcmc_reported_budget`
- `beta_selection_budget`

These fields should not be removed without replacing them with equivalent information.

## 7) Plotting implications

Cross-method figures should make it explicit that:

- MCMC-IS total budget includes a fixed beta-selection budget

Beta-diagnostic figures should not include that note, because those checkpoints are production-only.

## 8) Implementation note

The relevant implementation currently lives in:

- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/perm_pval/experiments/notebook_studies.py`

If future code changes checkpoint behavior, update this note together with:

- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/context/diagnostics.md`
- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/context/results_replot.md`
