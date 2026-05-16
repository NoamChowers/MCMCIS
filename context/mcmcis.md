# MCMC-IS (MCMCIS) beta initialization & tuning — implementation notes

Goal: implement **beta initialization** and **beta tuning** for a **smooth-hinge tilted** MCMC-IS sampler, with a clean separation between *tuning* (adaptive) and *production* (frozen).

---

## 1) Problem setup (permutation / resampling tests)

- Null distribution over states (permutations / resamples): \(f(y)\) (often uniform on the permutation space).
- Test statistic \(T(y)\) and observed threshold \(T_{\mathrm{obs}}\).
- Rare event set:
\[
A := \{y:\ T(y)\ge T_{\mathrm{obs}}\},\qquad p := \mathbb P_f(Y\in A)
\]

### Smooth-hinge (flat-tail) tilting family

We use a *flat-tail* smooth tilt that leaves the tail unchanged and smoothly downweights below-threshold states:
\[
\pi_\beta(y)\ \propto\ f(y)\,\exp\!\big(-\beta\,S_{\mathrm{scaled}}(y)\big),\qquad \beta\ge 0
\]
where
\[
S_{\mathrm{scaled}}(y) := \left(\frac{T_{\mathrm{obs}}-T(y)}{\sigma_T}\right)_+,
\qquad (u)_+ := \max(u,0)
\]
So \(S_{\mathrm{scaled}}(y)=0\) for \(y\in A\), hence the tail is *flat* under the tilt.

This matches the MCMCIS “continuous IS function” idea: the tilt equals 1 past the threshold and changes smoothly below it; see the form
\(g(\lambda(x),\beta_K)=\exp[\beta_K(\lambda(x)-\lambda^\*)\mathbf 1\{\lambda(x)\le \lambda^\*\}]\).

Implementation note:
- `run_mcmc_is(..., tilt_mode=\"smooth_hinge\")` uses this default tilt.
- `run_mcmc_is(..., tilt_mode=\"step\")` is also available for step-vs-smooth comparisons:
  \[
  \pi_\beta(y)\propto f(y)\exp\!\big(-\beta\,\mathbf 1\{T(y)<T_{\mathrm{obs}}\}\big).
  \]

---

## 2) What we tune: exceedance rate under \(\pi_\beta\)

Define the **exceedance rate**
\[
q_\beta := \pi_\beta(A)
\]

### Why target \(q_\beta \approx q_{\text{target}}\)?
- MCMCIS explicitly targets a tail sampling probability \(\tilde\pi\) (e.g. \(\tilde\pi=0.01\), \(\tilde\pi=0.1\) in plots/experiments).
- Your toy-model analysis suggests choices like \(q_{\text{target}}=p_0^{D_\alpha}\).
  In this document we use the starting-guess rule:
  \[
  q_{\text{target}} = p_0^{D_\alpha},\qquad D_\alpha=\tfrac14
  \]
  with the modeling assumption that \(p_0\) is the true p-value.

Operationally:
- \(q_\beta\) too small → rare tail visits; you revert toward IID inefficiency.
- \(q_\beta\) too large → normalizer/weights can become unstable (static variance dominates).

---

## 3) Separation of stages (critical)

### Tuning stage (adaptive)
- Adapt \(\beta\) using short runs/rounds (Shuli: \(J\) rounds, each \(K\) MH steps; discard burn-in within each round).
- Do **not** use adaptive-stage samples in the final estimator unless you implement a formally correct adaptive scheme.

### Production stage (frozen)
- Freeze \(\beta=\hat\beta\) and run a long chain targeting \(\pi_{\hat\beta}\).
- Compute the final SNIS estimate from production samples only.

---

## 4) Module: `beta_tuning.py`

### (A) Scale estimate \(\sigma_T\)

**Signature**
```python
def estimate_scale_T(pilot_T: np.ndarray, method: str = "sd") -> float:
    # pilot_T: IID pilot statistics under the base null f
    # method: "sd" (default) or "mad" (robust)
    # returns sigma_T used in S_scaled(y) = max((T_obs - T(y))/sigma_T, 0)
    ...
```

**Recommendations**
- `sd`: `np.std(pilot_T, ddof=1)`
- `mad`: `1.4826 * median(|T - median(T)|)` (normal-consistent MAD)

---

### (A1) Initial beta from IID pilot Laplace matching

**Signature**
```python
def init_beta_from_iid_pilot(
    pilot_T: np.ndarray,
    T_obs: float,
    sigma_T: float,
    p0: float,
    q_target: float,
    beta_max: float = 1e6,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> float:
    ...
```

Define shortfall sample:
\[
s_i = \left(\frac{T_{\mathrm{obs}}-T_i}{\sigma_T}\right)_+,\qquad T_i\sim f\ \text{(IID pilot)}
\]
and empirical Laplace transform:
\[
\widehat Z(\beta)=\frac1M\sum_{i=1}^M e^{-\beta s_i}.
\]
Set:
\[
Z_{\text{target}} = \frac{p_0}{q_{\text{target}}}.
\]
Then solve \(\widehat Z(\beta)\approx Z_{\text{target}}\) by bracketing + bisection.

Output is `beta0_laplace`, used as the **initial** value for tuning.

---
---

### (B) `tune_beta_to_target_q(...)` — bracket + bisection on empirical \(q_\beta\)

Provide a callback:
```python
run_short_chain_fn(beta, init_state, n_steps, burn_in) -> {
    "q_hat": float,
    "last_state": obj,
    "accept_rate": float,
}
```

Goal: find \(\beta\) such that \(\hat q(\beta)\approx q_{\text{target}}\) assuming monotonicity in \(\beta\).

**Signature**
```python
def tune_beta_to_target_q(
    run_short_chain_fn,
    init_state,
    beta0: float,
    q_target: float,
    n_steps: int,
    burn_in: int,
    bracket_factor: float = 2.0,
    tol_abs: float | None = None,
    tol_rel: float = 0.2,
    max_bracket_iter: int = 12,
    max_bisect_iter: int = 12,
    replicate: int = 1,
    reuse_state: bool = True,
) -> dict:
    ...
```

**Notes**
- Warm-starting with `last_state` can save burn-in.
- Always run production with frozen `beta_hat`.

---

## 5) MH acceptance ratio under smooth-hinge tilt

General MH:
\[
\alpha = \min\left(1,\ \frac{f(y')}{f(y)}\frac{g_\beta(y')}{g_\beta(y)}\frac{q(y\mid y')}{q(y'\mid y)}\right)
\]

Uniform \(f\) + symmetric proposal:
\[
\alpha = \min\left(1,\ \exp\left(-\beta\,[S_{\mathrm{scaled}}(y')-S_{\mathrm{scaled}}(y)]\right)\right)
\]

### 5.1 Proposal kernel (shared with SAMC)

Current implementation uses the **same localized-swap proposal family** in MCMC-IS and SAMC.

- A proposal swaps exactly `n_swap_pairs` treated/control pairs.
- The treated indices and control indices are sampled uniformly without replacement.
- The selected treated labels become controls, and the selected control labels become treated.
- Group sizes are preserved exactly.

Parameterization:

- `proposal_fraction` (default `0.075`) defines:
  \[
  n_{\text{swap\_pairs}}=\max\!\left(1,\ \text{round}\!\left(0.075\cdot \min(n_1,n_0)\right)\right)
  \]
- `proposal_swaps` (optional) overrides this with a fixed integer number of swap pairs.

Symmetry note:
- The kernel is treated as symmetric in implementation, so no explicit Hastings correction term is added.

---

## 6) End-to-end pipeline

1. IID pilot under \(f\): sample \(y_i\sim f\), compute \(T_i=T(y_i)\)
2. \(\sigma_T\leftarrow\) `estimate_scale_T(pilot_T)`
3. Choose \(q_{\text{target}}\) using the requested rule:
   \[
   q_{\text{target}}=p_0^{D_\alpha},\quad D_\alpha=\tfrac14
   \]
   with the assumption \(p_0\) is the true p-value
4. \(\beta_0\leftarrow\) `init_beta_from_iid_pilot(...)`
5. \(\hat\beta\leftarrow\) `tune_beta_to_target_q(...)`
6. Production: long MH at \(\hat\beta\), compute SNIS.

---

## 7) Two implementation modes in this repo

### 7.1 Tuning mode (search for beta, then freeze)

Use this when beta is not known in advance.

Minimal flow:
```python
pilot_t = iid_pilot_statistics(problem, n_samples=M, seed=seed)
sigma_t = estimate_scale_T(pilot_t, method="sd")
q_target = p0 ** 0.25  # D_alpha = 1/4
beta0 = init_beta_from_iid_pilot(
    pilot_T=pilot_t,
    T_obs=problem.t_obs,
    sigma_T=sigma_t,
    p0=p0,
    q_target=q_target,
)
runner = make_short_chain_q_runner(
    problem,
    sigma_T=sigma_t,
    thin=thin_tune,
    proposal_fraction=0.075,
    proposal_swaps=None,
    seed=seed + 1,
)
tuned = tune_beta_to_target_q(
    run_short_chain_fn=runner,
    init_state=problem.y_obs,
    beta0=beta0,
    q_target=q_target,
    n_steps=n_steps_tune,
    burn_in=burn_in_tune,
)
beta_used = tuned["beta_hat"]

res = run_mcmc_is(
    problem,
    beta=beta_used,
    sigma_t=sigma_t,
    n_steps=n_steps_prod,
    burn_in=burn_in_prod,
    thin=thin_prod,
    n_chains=n_chains,
    proposal_fraction=0.075,   # shared default with SAMC
    proposal_swaps=None,        # optional integer override
    seed=seed + 2,
)
```

Important: production estimator uses only frozen-beta production samples.

TODO:
- Investigate reusing fixed-beta local-scan samples in the final p-value estimator by carrying forward per-sample (or aggregated numerator/denominator) importance weights for the beta that generated each sample.
- This is different from reusing a state across beta values as if it were already stationary for a new beta. Reweighting can make past samples usable for estimating `p`, but does not by itself justify zero-burn-in continuation under a different production beta.
- Current implementation does not do this; non-selected beta/configuration runs are used for selection only, and only the selected configuration's terminal states are reused for production initialization.

### 7.2 Non-tuning mode (beta passed explicitly)

Use this for controlled diagnostics/ablation (e.g., beta sweep).

```python
res = run_mcmc_is(
    problem,
    beta=beta_explicit,   # used directly, no tuning
    sigma_t=sigma_t,
    n_steps=n_steps,
    burn_in=burn_in,
    thin=thin,
    n_chains=n_chains,
    proposal_fraction=0.075,   # shared default with SAMC
    proposal_swaps=None,        # optional integer override
    seed=seed,
)
```

`run_mcmc_is` does not initialize or tune beta internally; it uses the provided beta as-is.

### 7.3 Note on notebook fields

In `quickstart_perm_pval.ipynb`:
- `beta_formula_sqrt_log_1_over_p = sqrt(log(1/p))` is reported for reference.
- `beta0_laplace` is the initialization from pilot Laplace matching.
- `beta_hat_tuned` is tuning output.
- `beta_used` is what is passed to `run_mcmc_is` (unless `BETA_OVERRIDE` is set).

---

## 8) Tests & logging

**Unit tests**
- \(\widehat Z(\beta)\) decreases in \(\beta\)
- tuning returns a valid bracket when possible
- \(q_{\text{target}}\to 1\Rightarrow \hat\beta\to 0\)

**Logging**
- record `beta`, `q_hat`, `accept_rate` each tuning iteration
- warn if `accept_rate` extremely low/high

---
