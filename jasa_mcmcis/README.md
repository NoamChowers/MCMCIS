# JASA MCMCIS Supplement

This is a compact, article-facing implementation of rare permutation-test
p-value estimation. It contains:

- MCMC importance sampling (MCMC-IS) for fixed-size binary-label permutation
  tests, using a tilted distribution and self-normalized importance weights.
- Hard-step MCMC-IS, using a known reference threshold `p0` to set a fixed
  tail multiplier.
- Stochastic Approximation Monte Carlo (SAMC) as a comparison method.
- The six frozen article scenarios plus 200 threshold-bin scenarios for
  cross-method comparisons.
- A small optional dynamic-programming exact solver for linear statistics,
  included for validation of the bundled scenarios.

Data generation and large experiment orchestration are intentionally not
included. The bundled data are the observed arrays, labels, and metadata needed
to reproduce method calls on the article scenarios.

## Install

From this directory:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Load A Scenario

```python
from jasa_mcmcis import load_scenario

scenario = load_scenario("gwas_additive_score_sig_n100")
problem = scenario.problem

print(problem.t_obs)
print(scenario.exact_p_value)
```

The bundled cross-method keys include the six article scenarios, 100
near-threshold variants, and 100 just-above-threshold variants:

```python
from jasa_mcmcis import (
    ABOVE_THRESHOLD_SCENARIO_KEYS,
    ARTICLE_SCENARIO_KEYS,
    CROSS_METHOD_SCENARIO_KEYS,
    NEAR_THRESHOLD_SCENARIO_KEYS,
)

print(ARTICLE_SCENARIO_KEYS)
print(NEAR_THRESHOLD_SCENARIO_KEYS[:3])
print(ABOVE_THRESHOLD_SCENARIO_KEYS[:3])
print(CROSS_METHOD_SCENARIO_KEYS)
```

## Run MCMC-IS

```python
from jasa_mcmcis import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    load_scenario,
    run_mcmc_is,
)

scenario = load_scenario("poisson_diffmeans_hep_sig_n200")
problem = scenario.problem

pilot_T = iid_pilot_statistics(problem, n_samples=20_000, seed=1)
sigma_t = estimate_scale_T(pilot_T)

q_target = scenario.exact_p_value ** (1 / 3)
beta = init_beta_from_iid_pilot(
    pilot_T,
    problem.t_obs,
    sigma_t,
    p0_reference=scenario.exact_p_value,
    q_target=q_target,
)

result = run_mcmc_is(
    problem,
    beta=beta,
    sigma_t=sigma_t,
    n_steps=50_000,
    burn_in=10_000,
    n_chains=2,
    proposal_size=5,
    seed=123,
)

print(result.estimate, result.mcse_obm, result.ess)
```

The hard-step variant uses `pi_r(y) ∝ f(y) * {1 + (r - 1) 1_A(y)}`. If
`f(A)=p0`, then `r = q(1 - p0) / (p0(1 - q))` targets `pi_r(A)=q`:

```python
from jasa_mcmcis import run_hard_step_mcmc_is

p0 = scenario.extra["known_significance_threshold"]
q = p0 ** (1 / 3)
hard_step = run_hard_step_mcmc_is(
    problem,
    p0=p0,
    q=q,
    n_steps=50_000,
    burn_in=10_000,
    n_chains=2,
    proposal_size=5,
    seed=123,
)
```

For the article's simple threshold-suite rule, use proposal size `2` for GWAS
scenarios and `5` for Poisson/HEP scenarios. The `proposal_size` argument also
accepts a fraction of the smaller group size.

## Run SAMC

```python
from jasa_mcmcis import load_scenario, run_samc

scenario = load_scenario("gwas_additive_score_sig_n100")

samc = run_samc(
    scenario.problem,
    n_steps=100_000,
    burn_in=20_000,
    n_bins=40,
    proposal_size=2,
    seed=123,
)

print(samc.estimate, samc.acceptance_rate)
```

## Exact DP Validation

The exact p-values are already stored in scenario metadata. The DP solver is
included because the bundled cross-method scenarios use linear statistics, and
exact checks are useful in a statistical-methods supplement.

```python
from jasa_mcmcis import LinearStatisticDPSolver, load_scenario

scenario = load_scenario("gwas_additive_score_above_n100")
exact = LinearStatisticDPSolver.from_scenario(scenario).compute()

print(exact.p_value)
print(scenario.exact_p_value)
```

This module is not required for running MCMC-IS or SAMC. It is a validation and
reproducibility aid, not part of the data-generation pipeline.

## API Summary

- `PermutationTestProblem(x, y_obs, statistic, tail="right")`: fixed-size
  permutation-test problem.
- `run_mcmc_is(problem, beta, sigma_t, n_steps, ...)`: tilted MCMC-IS estimator.
- `run_hard_step_mcmc_is(problem, p0, q, n_steps, ...)`: hard-step MCMC-IS
  estimator with `r` determined by the reference threshold and target tail mass.
- `run_samc(problem, n_steps, ...)`: SAMC estimator for right-tail tests.
- `iid_pilot_statistics`, `estimate_scale_T`, `init_beta_from_iid_pilot`:
  lightweight beta initialization helpers.
- `load_scenario`, `load_scenarios`, `available_scenarios`: access bundled
  article scenarios.
- `LinearStatisticDPSolver`: optional exact p-value validation for linear
  statistics.
