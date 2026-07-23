# `perm_pval`

`perm_pval` is a Python research library for estimating permutation-test p-values, with emphasis on small-tail probabilities.

Implemented in this milestone:
- Core permutation test problem abstraction
- Two-sample statistics (difference in means, pooled and Welch t-statistics)
- Exact brute-force p-values for small `n`
- Exact DP solvers:
  - rank-sum / Mann-Whitney U via rank-sum DP (no ties)
  - linear-statistic DP for `offset + scale * sum(scores_i * y_i)` statistics
- Random sampling baseline (uniform label permutations)
- MCMC-IS with self-normalized importance sampling (SNIS)
  - includes OBM (overlapping batch means) variance / MCSE estimate for SNIS
- SAMC baseline implementation for rare-event/tail exploration
- Minimal experiment CLI and pytest coverage

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.stats.two_sample import difference_in_means
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.exact.linear_statistic_dp import LinearStatisticDPSolver
from perm_pval.methods.random_sampling import run_random_sampling
from perm_pval.methods.mcmc_is import run_hard_step_mcmc_is, run_mcmc_is
from perm_pval.methods.beta_tuning import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    make_short_chain_q_runner,
    tune_beta_to_target_q,
)

x = np.array([0.1, 0.3, 0.2, 1.5, 1.2, 1.7], dtype=float)
y_obs = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)

problem = PermutationTestProblem(
    X=x,
    y_obs=y_obs,
    statistic=difference_in_means,
    tail="right",
)

exact = BruteForceExactSolver(problem, max_permutations=10_000).compute()
# Alternative exact solver for difference in means (DP, score scaling)
exact_dp = LinearStatisticDPSolver.from_difference_in_means(
    problem,
    score_scale=10,
).compute()
mc = run_random_sampling(problem, n_samples=20_000, seed=7)
mcmc = run_mcmc_is(problem, beta=2.0, n_steps=12_000, burn_in=2_000, thin=5, seed=123)

print(exact.p_value, exact_dp.p_value, mc.estimate, mcmc.estimate)
```

## Assumptions and conventions

- Labels are binary with fixed counts (`1` = treated, `0` = control).
- The default constraint is fixed group sizes inferred from `y_obs`.
- Tail definitions:
  - `"right"`: `T >= t_obs`
  - `"left"`: `T <= t_obs`
  - `"two-sided"`: `|T| >= |t_obs|` (default two-sided convention in this codebase)
- MCMC-IS tilt (current implementation, right-tail only):
  - `g_beta(y) ∝ f(y) * exp(-beta * ((t_obs - T(y))/sigma_t)_+)`
  - Flat in the tail (`T(y) >= t_obs`) and exponentially downweighted below the tail threshold.
  - Importance weights are proportional to `exp(beta * ((t_obs - T(y))/sigma_t)_+)`.
- Hard-step MCMC-IS uses the same local proposal kernel with
  `pi_r(y) ∝ f(y) * {1 + (r - 1) 1_A(y)}`. If `f(A)=p0`, then
  `r = q(1 - p0) / (p0(1 - q))` gives tilted tail mass `pi_r(A)=q`.
- SAMC follows Yu et al. (2011) p-value evaluation setup:
  - right-tail partition with `bin_edges[-2] = t_obs`, `bin_edges[-1] = +inf`
  - stochastic-approximation updates with `gamma_t = t0 / max(t0, t)`
  - p-value estimate via Eq. (3.2): `exp(theta_m)(pi_m+pi0) / sum_j exp(theta_j)(pi_j+pi0)`
  - convergence diagnostic via relative frequency error Eq. (3.3)

## Beta initialization and tuning (scaled hinge)

`perm_pval.methods.beta_tuning` provides a pilot-based workflow:

```python
pilot_t = iid_pilot_statistics(problem, n_samples=5_000, seed=101)
sigma_t = estimate_scale_T(pilot_t, method="sd")

p0 = 1e-8
q_target = p0 ** 0.5
beta0 = init_beta_from_iid_pilot(
    pilot_T=pilot_t,
    T_obs=problem.t_obs,
    sigma_T=sigma_t,
    p0=p0,
    q_target=q_target,
)

runner = make_short_chain_q_runner(problem, sigma_T=sigma_t, thin=2, seed=202)
tuning = tune_beta_to_target_q(
    run_short_chain_fn=runner,
    init_state=problem.y_obs,
    beta0=beta0,
    q_target=q_target,
    n_steps=4_000,
    burn_in=1_000,
)
beta_hat = tuning["beta_hat"]

result = run_mcmc_is(
    problem,
    beta=beta_hat,
    sigma_t=sigma_t,
    n_steps=60_000,
    burn_in=10_000,
    thin=5,
    seed=303,
)
```

Use the tuning stage only to pick `beta`; then freeze `beta_hat` for the production chain.

## Experiment runner

Run one synthetic experiment and save JSON output:

```bash
python -m perm_pval.experiments.run_single \
  --n 14 --n-treated 7 --effect-size 1.25 \
  --beta 2.0 --mcmc-steps 30000 --mcmc-burn-in 5000 --mcmc-thin 5 \
  --mc-samples 50000 --seed 123 \
  --output results/single_run.json
```

The output includes config snapshot, seed, exact p-value (if feasible), and method diagnostics.
Use `--no-mcmc-estimate-variance` to disable OBM variance estimation, or
`--mcmc-obm-batch-size <int>` to override the default OBM batch size.

## Reproducible exact-scenario catalog

Generate and save fixed simulation datasets (`X.npy`, `y_obs.npy`) with exact p-values:

```bash
python -m perm_pval.experiments.generate_exact_scenarios \
  --output-dir results/exact_scenarios/v1
```

This writes:
- `results/exact_scenarios/v1/catalog.json`
- one folder per scenario, each containing:
  - `X.npy`
  - `y_obs.npy`
  - `metadata.json` (exact method, `t_obs`, exact p-value, tail hits, permutation count)

Included exact methods:
- Fisher exact test (2x2; hypergeometric tail) for treated-success count on binary outcomes
- `RankSumDPSolver`
- `LinearStatisticDPSolver`
- one `BruteForceExactSolver` benchmark for a non-DP statistic (`t_statistic_welch`)

Important: in the binary-outcome scenario, the one-sided Fisher exact p-value is equal to the
permutation p-value from full fixed-size-label enumeration.

## Module overview

- `perm_pval/core/problem.py`: `PermutationTestProblem`
- `perm_pval/stats/two_sample.py`: two-sample statistics
- `perm_pval/exact/brute_force.py`: exact brute-force solver
- `perm_pval/exact/rank_sum_dp.py`: exact rank-sum/Mann-Whitney U DP solver
- `perm_pval/exact/linear_statistic_dp.py`: exact linear-statistic DP solver
- `perm_pval/methods/random_sampling.py`: baseline Monte Carlo
- `perm_pval/methods/mcmc_is.py`: MCMC-IS + SNIS
- `perm_pval/methods/beta_tuning.py`: pilot-scale estimation + beta initialization/tuning
- `perm_pval/methods/samc.py`: SAMC baseline
- `perm_pval/diagnostics/`: MCMC/IS/SAMC diagnostics + plotting helpers
- `perm_pval/experiments/`: CLI for single run and simple sweep

## Notes on scalability

- Exact brute force is intentionally guarded by `max_permutations`.
- For larger `n`, use random sampling, MCMC-IS, and SAMC.
- Current implementations prioritize correctness and clear extension points.
