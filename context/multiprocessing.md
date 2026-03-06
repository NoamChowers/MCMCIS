# Multiprocessing Design

This note records the current multiprocessing strategy for the simulation notebooks and shared study helpers.

Scope:
- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/notebooks/cross_method_simulation.ipynb`
- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/notebooks/mcmcis_beta_diagnostics.ipynb`
- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/perm_pval/experiments/notebook_studies.py`

## 1) High-level rule

Use multiprocessing only at coarse-grained, independent job boundaries.

Do not parallelize:
- individual MCMC steps
- individual SAMC steps
- plotting
- file writes

Reason:
- those would add complexity without comparable wall-clock gains
- the method code is still easiest to reason about as one continued trajectory per run

## 2) Current worker boundaries

### Cross-method notebook

Current structure:

- iterate through methods
- for each method, parallelize over replicate runs

Concretely:

- `iid`: one worker per replicate
- `mcmc_is`: one worker per replicate
- `samc`: one worker per replicate

Important:
- MCMC-IS beta tuning is not done inside replicate workers
- beta workflow is built once in the parent process
- the fixed beta-selection budget is then passed into MCMC replicate workers

This preserves fair cross-method budget accounting.

### Beta notebook

Current structure:

- iterate through beta values
- for each beta, parallelize over replicate runs

Concretely:

- one worker per `(beta, replicate)` run, but submitted beta-by-beta

This matches the intended interpretation:
- conditional performance of MCMC-IS for a fixed beta value

## 3) Why this design

This is the best tradeoff between:
- wall-clock savings
- implementation simplicity
- preserving deterministic study semantics

Advantages:
- each worker handles a whole continued trajectory
- checkpoint logic stays unchanged inside the worker
- parent process still controls:
  - beta tuning
  - SAMC setup
  - aggregation
  - plotting
  - JSON/PNG writes

## 4) Current implementation

Implemented in:

- `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/perm_pval/experiments/notebook_studies.py`

Main pieces:

- `CrossMethodStudyConfig.n_jobs`
- `BetaSweepStudyConfig.n_jobs`
- module-level worker helpers for:
  - IID replicate runs
  - cross-method MCMC-IS replicate runs
  - SAMC replicate runs
  - beta-study MCMC-IS replicate runs

The workers are module-level on purpose so they are picklable under `spawn`.

## 5) Process model

The process pool uses:

- `multiprocessing.get_context("spawn")`

This is the safer choice on macOS / notebook environments.

## 6) Recommended default

In the notebooks, the recommended default is:

### Cross-method notebook

- `N_JOBS = min(N_REPEATS, os.cpu_count() or 1)`

### Beta notebook

- `N_JOBS = min(BETA_REPEATS, os.cpu_count() or 1)`

Reason:
- the actual unit of parallel work is the replicate run
- using more workers than replicates gives no benefit

## 7) Determinism

Results should be statistically identical between serial and parallel execution because:

- seeds are fixed per replicate
- beta workflow is built once in the parent
- rows are sorted before summarization/saving

Runtime fields are naturally different:
- wall-clock measurements can differ between serial and parallel execution

So regression tests should compare:
- estimates
- diagnostics
- summaries

but not raw runtime values.

## 8) Restricted-runtime fallback

Some environments block process-pool creation entirely.

In that case, the current code:

- catches process-pool construction failure
- emits a warning
- falls back to serial execution

This is intentional.

It keeps notebooks usable in:
- restricted sandboxes
- CI-like environments with limited semaphore support

without changing notebook code.

## 9) What to do next if more speed is needed

Only consider these after the current coarse-grained multiprocessing is insufficient:

1. parallelize across scenarios in the cross-method notebook
2. parallelize local beta scan candidates
3. parallelize multi-chain MCMC chains inside a single replicate

These are lower priority because they add more coordination complexity.
