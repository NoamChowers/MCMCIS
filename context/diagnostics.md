# Diagnostics Specification for Tiny-p Permutation Experiments

This document defines what to save for each method (`iid`, `mcmc_is`, `samc`) and how to compare estimator quality across methods.

Scope:
- right-tail permutation tests
- tiny p-value regime
- replicated studies across fixed compute budgets

## 1) Output artifacts

Save three artifacts per study:

1. `run_records.jsonl` (or CSV): one row per `(scenario, budget, replicate, method)`.
2. `summary_records.jsonl` (or CSV): aggregated rows per `(scenario, budget, method)`.
3. `metadata.json`: experiment config, seeds, code version, and tuning metadata.

## 2) Common run-level fields (all methods)

Required fields:
- `scenario`: scenario id/name.
- `scenario_display`: human-readable scenario label.
- `method`: one of `iid`, `mcmc_is`, `samc`.
- `replicate`: replicate id.
- `budget`: method budget used for this run.
- `exact_p`: exact p-value for the scenario.
- `estimate`: method estimate `p_hat`.
- `variance_estimate`: method-reported variance estimate for `p_hat` (if unavailable: `NaN`).
- `wall_time_sec`: production wall time only.
- `eval_excl_tuning`: evaluation count excluding tuning.
- `eval_incl_tuning`: evaluation count including tuning.
- `wall_time_incl_tuning_sec`: wall time including tuning.

Derived comparison fields (recommended at run level):
- `bias = estimate - exact_p`
- `squared_error = (estimate - exact_p)^2`
- `rel_error = (estimate - exact_p) / exact_p`
- `abs_log10_error = abs(log10(max(estimate, 1e-300)) - log10(exact_p))`

## 3) Method-specific run-level fields

### 3.1 IID (`run_random_sampling`)

Required:
- `n_samples`
- `tail_hits`
- `tail_share_raw = tail_hits / n_samples`
- `standard_error`
- `ci_low`, `ci_high`, `confidence_level`
- `zero_hits = 1{tail_hits == 0}`

Notes:
- For tiny p, `zero_hits` is a critical diagnostic.

### 3.2 MCMC-IS (`run_mcmc_is`)

Required:
- `beta`
- `sigma_t`
- `n_weighted_samples`
- `tail_hits_weighted_sample`
- `tail_share_raw_sample` (proxy for tilted-tail occupancy `q_hat`)
- `ess`
- `overall_acceptance_rate`
- `acceptance_rates` (per chain)
- `chain_diagnostics` (per chain):
  - `acceptance_rate`, `n_proposals`, `n_accepted`, `mean_stat`, `std_stat`, `iact_stat`
- `weight_summary`:
  - `min_weight`, `median_weight`, `max_weight`, `mean_weight`, `cv`
- `snis_variance_obm` and `snis_mcse_obm` if variance estimation is enabled
- `obm_batch_size_requested`, `obm_chain_batch_sizes`, `obm_chain_long_run_variances`

Recommended:
- `q_tilt_tail_share = tail_share_raw_sample`
- `seed`, `chain_seeds`

Notes:
- Keep tuning-stage outputs (beta initialization/tuning) separate from production outputs.

### 3.3 SAMC (`run_samc`)

Required:
- `n_steps`, `burn_in`, `n_retained_after_burn_in`
- `acceptance_rate`
- `pvalue_estimator`
- `tail_bin_index`
- `pi0_adjustment`
- `empty_bin_indices`
- `max_abs_relative_frequency_error`
- `convergence_reached`
- `localized_swaps_per_proposal`

Recommended:
- `visit_counts`
- `visitation_frequency`
- `target_visitation`
- `relative_frequency_error`
- `theta_final`
- `theta_trace`
- `step_sizes`
- `bin_edges`

Notes:
- Use post-burn-in visit counts for histogram diagnostics and empty-bin handling.

## 4) Tuning-stage metrics (store separately)

### 4.1 MCMC-IS beta workflow

Save:
- `pilot_samples`, `pilot_scale_method`, `sigma_t`
- `beta_formula` (if used)
- `beta0_laplace` (IID pilot initialization)
- `beta_hat_tuned`
- `beta_used` (after optional override)
- `q_target`, `q_hat_beta_hat`
- `bracket_succeeded`
- `history` (or compacted history): per-iteration `beta`, `q_hat`, `accept_rate`, `stage`
- tuning costs: `tuning_evals`, `tuning_time_sec`

### 4.2 SAMC setup/tuning

Save:
- binning parameters (`n_bins`, `lambda_min` or provided `bin_edges`)
- pilot samples used for `lambda_min` (if estimated)
- tuning costs: `tuning_evals`, `tuning_time_sec`

## 5) Summary metrics for estimator comparison

Aggregate by `(scenario, budget, method)` over replicates.

Core estimator comparison:
- `mean_estimate`
- `median_estimate`
- `bias = mean_estimate - exact_p`
- `rel_bias = bias / exact_p`
- `empirical_var = Var(estimate across replicates)`
- `rmse = sqrt(mean(squared_error))`
- `mean_abs_log10_error`

Variance-estimator calibration:
- `mean_variance_estimate`
- `var_calibration_ratio = empirical_var / mean_variance_estimate` (when finite and positive)

Efficiency:
- `mean_eval_excl_tuning`, `mean_eval_incl_tuning`
- `mean_time_excl_tuning_sec`, `mean_time_incl_tuning_sec`
- optional normalized efficiency score:
  - `mse_per_1e6_evals = mean(squared_error) * 1e6 / mean_eval_excl_tuning`
  - `mse_per_sec = mean(squared_error) / mean_time_excl_tuning_sec`

Method-specific summary diagnostics:
- IID: `mean_zero_rate`, `mean_tail_hits`
- MCMC-IS: `mean_q_tilt_tail_share`, `mean_ess`, `mean_acceptance_rate`, `mean_weight_cv`
- SAMC: `mean_samc_max_rel_freq_error`, `samc_convergence_rate`, `mean_empty_bin_count`

## 6) Required plots

Generate these plots per scenario. Save both PNG and PDF where possible.

### 6.1 Cross-method estimator comparison plots

1. RMSE vs compute budget:
- x-axis: `mean_eval_excl_tuning` (log scale)
- y-axis: `rmse` (log scale)
- one curve per method (`iid`, `mcmc_is`, `samc`)
- optional overlay: include-tuning curves using `mean_eval_incl_tuning`

2. RMSE vs wall-clock:
- x-axis: `mean_time_excl_tuning_sec` (log scale)
- y-axis: `rmse` (log scale)
- optional include-tuning overlay via `mean_time_incl_tuning_sec`

3. Estimator distribution at largest budget:
- boxplot of `log10(estimate)` across replicates by method
- horizontal reference line at `log10(exact_p)`

4. Variance-estimate distribution at largest budget:
- boxplot of `log10(variance_estimate)` across replicates by method
- exclude nonpositive or missing variance values from log transform

5. Absolute log-error comparison:
- boxplot or violin of `abs_log10_error` across methods
- optional trend vs budget line plot of `mean_abs_log10_error`

### 6.2 IID-specific diagnostic plots

1. Tail-hit histogram:
- histogram of `tail_hits` over replicates (for each budget)

2. Zero-hit rate vs budget:
- x-axis: `budget`
- y-axis: `mean_zero_rate`
- highlights when IID is underpowered for tiny p

### 6.3 MCMC-IS-specific diagnostic plots

1. Beta sweep diagnostics (if sweep run):
- boxplots vs beta for:
  - `q_tilt_tail_share`
  - `log10(estimate)` with exact reference line
  - `log10(variance_estimate)` (or `log10(snis_variance_obm)`)
  - `ess`
  - `overall_acceptance_rate`

2. Weight degeneracy diagnostics:
- histogram of normalized importance weights (or log-weights)
- track `weight_summary.cv` vs budget/beta

3. Chain-mixing diagnostics:
- trace plot of statistic `T`
- ACF plot for `T`
- per-chain `iact_stat` summary across replicates

4. Beta tuning trace:
- line plot of tuning iteration:
  - x-axis: tuning iteration
  - y-axis left: `beta`
  - y-axis right: `q_hat`
  - reference line at `q_target`

### 6.3.1 MCMC-IS beta comparison protocol (required for beta studies)

Use this protocol when explicitly comparing p-estimators across beta values.

1. Fix one scenario with known exact p-value:
- choose one concrete `PermutationTestProblem`
- compute `exact_p` using an exact solver (DP preferred for tiny p)
- keep the scenario fixed across all beta values

2. Define beta grid:
- center beta at tuned `beta_used` (or an explicit baseline)
- build grid like `beta = beta_center * m`, with multipliers such as
  `[0.25, 0.6, 0.85, 1.0, 1.2, 1.5, 2.0]`
- store `beta_center`, multipliers, and resulting absolute beta values

3. Replicate by seed:
- for each beta, run `R` independent runs with distinct seeds
- recommended `R >= 20` for publication figures (smaller allowed for smoke tests)
- keep `n_steps`, `burn_in`, `thin`, `n_chains`, `sigma_t` fixed across beta

4. Save run-level beta diagnostics:
- `beta`, `seed`
- `estimate`
- `variance_estimate` (OBM when enabled)
- `snis_mcse_obm`
- `q_tilt_tail_share`
- `ess`
- `overall_acceptance_rate`
- `weight_summary.cv`
- `abs_log10_error`

5. Required beta-comparison plots:
- boxplot: `log10(estimate)` by beta with horizontal `log10(exact_p)` line
- boxplot: `log10(variance_estimate)` by beta
- boxplot: `q_tilt_tail_share` by beta
- boxplot: `ess` by beta
- boxplot: `overall_acceptance_rate` by beta

6. Required beta-comparison summary table (one row per beta):
- `beta`
- `n_runs`
- `mean_estimate`, `median_estimate`
- `bias`, `rel_bias`
- `rmse`
- `mean_abs_log10_error`
- `mean_variance_estimate`
- `empirical_var`
- `var_calibration_ratio`
- `mean_q_tilt_tail_share`
- `mean_ess`
- `mean_acceptance_rate`

7. Interpretation targets:
- prefer beta regions with low `rmse` and low `mean_abs_log10_error`
- reject beta values with severe weight degeneracy (`ESS` collapse, high weight CV)
- reject beta values with pathological acceptance (near 0 or near 1)
- use `q_tilt_tail_share` stability to confirm tail is sufficiently sampled under tilt

### 6.4 SAMC-specific diagnostic plots

1. Bin visitation vs target:
- bar plot of `visitation_frequency` with `target_visitation` overlay
- use post-burn-in counts only

2. Relative frequency error by bin:
- bar plot of `relative_frequency_error` (%)
- annotate `max_abs_relative_frequency_error`

3. Theta traces:
- line plot of selected components of `theta_trace` over iterations
- include tail-bin theta and 2-3 interior bins

4. Empty-bin monitoring:
- plot/count of `len(empty_bin_indices)` vs budget
- optional relation to estimator instability

### 6.5 Plot metadata to persist

For each plot file, record:
- `scenario`
- `budget` (or budget range)
- methods included
- replicate count
- exact p-value reference
- creation timestamp

Recommended naming:
- `plots/{scenario}/rmse_vs_eval.png`
- `plots/{scenario}/rmse_vs_time.png`
- `plots/{scenario}/box_estimate_budget_{budget}.png`
- `plots/{scenario}/box_variance_budget_{budget}.png`
- `plots/{scenario}/mcmc_beta_sweep.png`
- `plots/{scenario}/samc_visitation_budget_{budget}.png`

## 7) Recommended sanity thresholds (soft checks)

These are not hard validity rules, but useful flags:

- IID:
  - `mean_zero_rate` near 1 implies IID budget is too small for this p regime.

- MCMC-IS:
  - `overall_acceptance_rate < 0.01` or `> 0.99` indicates poor proposal/beta regime.
  - Very low `ess` relative to `n_weighted_samples` suggests severe weight degeneracy.

- SAMC:
  - Large `max_abs_relative_frequency_error` suggests poor flattening.
  - High empty-bin count indicates too many bins or poor proposal mobility.

## 8) Minimal canonical row schema (run level)

```json
{
  "scenario": "rank_linear_x",
  "method": "mcmc_is",
  "replicate": 3,
  "budget": 200000,
  "exact_p": 8.445624e-08,
  "estimate": 1.21e-07,
  "variance_estimate": 3.4e-15,
  "eval_excl_tuning": 200001,
  "eval_incl_tuning": 212345,
  "wall_time_sec": 6.1,
  "wall_time_incl_tuning_sec": 7.0,
  "bias": 3.65e-08,
  "squared_error": 1.33e-15,
  "rel_error": 0.432,
  "abs_log10_error": 0.156,
  "beta": 4.06,
  "sigma_t": 36.5,
  "q_tilt_tail_share": 0.012,
  "ess": 940.8,
  "acceptance_rate": 0.23
}
```

## 9) Reproducibility fields (always include)

In `metadata.json`, store:
- global seed strategy
- per-method seeds (and per-chain seeds for MCMC-IS)
- package versions (at least `numpy`, `python`, `perm_pval` version/hash)
- full method configs
- exact solver used for each scenario

This ensures numerical results and diagnostics are auditable and repeatable.
