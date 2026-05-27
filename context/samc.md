# SAMC for Efficient p-value Evaluation in Resampling Tests (Yu et al., 2011) — Implementation Notes

**Goal.** Estimate a small p-value for a resampling/permutation test
\[
p=\Pr_f\{\lambda(X)\ge \lambda^\*\}
\]
where \(X\) is a resample/permutation from the null \(f\) (often uniform over permutations), \(\lambda(\cdot)\) is the test statistic, and \(\lambda^\*\) is the observed threshold.

The paper adapts **Stochastic Approximation MCMC (SAMC)** to avoid the \(n\asymp 1/p\) cost of IID resampling when \(p\) is tiny.

---

## 1) IID baseline (why we need something else)

IID estimator:
\[
\hat p^{\mathrm{iid}}_n=\frac1n\sum_{t=1}^n \mathbf 1\{\lambda(X_t)\ge\lambda^\*\},\quad X_t\stackrel{iid}{\sim} f
\]
has \(\mathrm{Var}(\hat p_n^{\mathrm{iid}})=p(1-p)/n\approx p/n\).  
To get relative RMSE \(\mathrm{RMSE}(\hat p_n)/p\le \varepsilon\), need \(n\approx 1/(p\varepsilon^2)\).

The paper also gives a “CARE” rule-of-thumb: \(np=(0.674/\mathrm{CARE})^2\) for a target **median absolute relative error**.

---

## 2) SAMC in one sentence

**Partition states into statistic “bins” and adapt per-bin weights so the chain visits bins with a desired long-run frequency (typically uniform).**  
This enforces frequent exploration of extreme bins (including the tail).

---

## 3) SAMC setup

### 3.1 Partition by statistic levels
Pick cutpoints \(\lambda_1<\cdots<\lambda_{m-1}=\lambda^\*\) and define:
- \(E_1=\{x:\lambda(x)<\lambda_1\}\)
- \(E_i=\{x:\lambda_{i-1}\le \lambda(x)<\lambda_i\}\), \(i=2,\dots,m-1\)
- \(E_m=\{x:\lambda(x)\ge \lambda_{m-1}=\lambda^\*\}\) (**tail bin**)

Let \(J(x)\in\{1,\dots,m\}\) be the bin index.

**Implementation tip.** Too few bins → poor flattening; too many bins → many empty bins, unstable adaptation, noisy p-hat. Start moderate; merge/refine bins if needed.

### 3.2 Desired bin frequencies
Choose \(\pi=(\pi_1,\dots,\pi_m)\), usually **uniform**: \(\pi_i=1/m\).  
Target behavior: \(\Pr\{X_t\in E_i\}\approx \pi_i\) (for nonempty bins).

### 3.3 Adaptive target distribution
Write the null as \(f(x)\propto \psi(x)\) (often \(\psi\equiv 1\) under uniform permutation null).  
In this repository we use the SAMC target in **inverse-bin-weight form**:
\[
f_{\theta(t)}(x)\propto \sum_{i=1}^m \psi(x)\,e^{-\theta_i(t)}\,\mathbf 1\{x\in E_i\}
\]
i.e. within bin \(E_i\), unnormalized weight is \(\psi(x)e^{-\theta_i(t)}\).

This is equivalent to the \(e^{+\theta_i}\) form under a sign reparameterization \(\tilde\theta_i=-\theta_i\).  
The current implementation follows the \(e^{-\theta}\) convention.

---

## 4) One SAMC iteration (what you implement)

### Step size / gain schedule
Paper uses
\[
\gamma(t)=\frac{t_0}{\max(t_0,t)}
\]
with \(t_0\in[1000,5000]\). This behaves like \(1/t\) after \(t_0\), stabilizing \(\theta\).

### (a) MH move targeting \(f_{\theta(t)}\)
Given current \(x\):
1. Propose \(y\sim q(x,\cdot)\).
2. Compute MH ratio
\[
r=\frac{\psi(y)e^{-\theta_{J(y)}(t)}\,q(y,x)}{\psi(x)e^{-\theta_{J(x)}(t)}\,q(x,y)}
\]
3. Accept with prob \(\min(1,r)\).

**Permutation simplification.** If \(f\) is uniform and \(q\) is symmetric, then \(\psi(y)/\psi(x)=1\) and \(q(y,x)/q(x,y)=1\), so
\[
r=\exp\big(\theta_{J(x)}(t)-\theta_{J(y)}(t)\big)
\]
which is exactly the code path:
\[
\log r = \theta_{J(x)}-\theta_{J(y)}.
\]

### (b) Stochastic approximation update
For each bin \(i\):
\[
\theta_i(t+1)=\theta_i(t)+\gamma(t)\big(\mathbf 1\{x_{t+1}\in E_i\}-\pi_i\big)
\]
Interpretation: bins visited “too often” are pushed down; bins visited “too rarely” get pulled up, flattening the histogram.

---

## 5) p-value estimator from \(\theta\)

After \(T\) iterations, estimate the tail probability using \(\theta(T)\), with an “empty-bin” correction via \(\pi_0\):
\[
\widehat p
=
\frac{e^{\theta_m(T)}(\pi_m+\pi_0)}{\sum_{j=1}^m e^{\theta_j(T)}(\pi_j+\pi_0)}
\]
This is the Eq. (3.2)-style form used in the implementation.
The implementation keeps this corrected value as the primary estimate and also records
the corresponding no-correction estimate,
\[
\widehat p_{\mathrm{raw}}
=
\frac{e^{\theta_m(T)}\pi_m}{\sum_{j=1}^m e^{\theta_j(T)}\pi_j},
\]
so the impact of the empty-bin adjustment can be inspected. Empty-bin handling and
frequency diagnostics are computed from **post-burn-in** visit counts.

**Implementation shortcut.** If you skip \(\pi_0\) initially, choose bins so all bins get visited, or merge empty bins.

---

## 6) Proposal design (usually dominant)

SAMC still depends on a *good* proposal. The paper recommends changing only a small fraction per move (about **5–10%**).

### Current repository implementation (shared with MCMC-IS)

SAMC and MCMC-IS now use the same localized-swap proposal family:

- One proposal swaps exactly `n_swap_pairs` treated/control pairs.
- The treated indices and control indices are sampled uniformly without replacement.
- The selected treated labels are changed to control and the selected control labels are changed to treated.

Parameterization:

- `proposal_fraction` (default **0.075**) maps to
  \[
  n_{\text{swap\_pairs}}=\max\!\left(1,\ \text{round}\!\left(0.075\cdot \min(n_1,n_0)\right)\right)
  \]
- `proposal_swaps` (optional integer) overrides `proposal_fraction` and fixes the exact number of swap pairs per proposal.

This keeps group sizes fixed and provides consistent proposal behavior across SAMC and MCMC-IS.

Practical tuning:
- too local → slow bin crossings
- too global → low acceptance

---

## 7) Diagnostics / stopping

SAMC aims for a flat visit histogram. The paper monitors empirical bin frequencies and uses a relative frequency error criterion like
\[
\max_i |\varepsilon_f(E_i)|<20\%
\]
Implementation-friendly proxy: track post-burn-in counts \(N_i(T)\) and check \(N_i(T)/\sum_j N_j(T)\approx \pi_i\) for nonempty bins.

---

## 8) Minimal pseudocode

```text
Input: statistic λ(x), threshold λ*, bins E1..Em (λ_{m-1}=λ*), desired π_i, t0
Init: θ_i(0)=0, choose x0
for t=1..T:
  γ = t0 / max(t0, t)
  propose y ~ q(x, ·)
  r = [ψ(y) * exp(-θ_{J(y)}) * q(y,x)] / [ψ(x) * exp(-θ_{J(x)}) * q(x,y)]
  accept with prob min(1,r): x = y else keep x
  for i=1..m:
      θ_i = θ_i + γ * ( 1{ x ∈ E_i } - π_i )
end
p-hat = exp(θ_m)*(π_m+π0) / Σ_j exp(θ_j)*(π_j+π0)
```

Optional burn-in handling (as in code):
- run adaptation across all steps \(1,\dots,T\)
- accumulate \(N_i\) for diagnostics and \(\pi_0\) only after burn-in

---

## 9) Common failure modes (and fixes)

1. **Too many bins → many empty bins → unstable \(\theta\), noisy \(\hat p\)**  
   - **Symptom:** many \(E_i\) never visited; \(\theta\) drifts; p-hat changes wildly across time/runs  
   - **Fix:** reduce \(m\); merge adjacent bins; re-place cutpoints so bins have comparable mass; optionally adaptive binning

2. **Chain cannot move between bins (proposal too local)**  
   - **Symptom:** visit histogram concentrated in a few neighboring bins; slow transitions to the tail bin \(E_m\)  
   - **Fix:** increase move size slightly (e.g., swap a larger fraction of labels); use mixture proposals (local + occasional larger jump); ensure proposal makes the chain irreducible

3. **Very low acceptance (proposal too global)**  
   - **Symptom:** acceptance rate collapses; chain barely moves; histogram not flattening  
   - **Fix:** decrease move size; prefer 5–10% updates (as suggested in the paper); verify symmetry assumptions before using simplified acceptance ratios

4. **Histogram not flattening / SA not stabilizing**  
   - **Symptom:** empirical post-burn-in frequencies \(N_i(T)/\sum_j N_j(T)\) do not approach \(\pi_i\) even after long runs  
   - **Fix:** run longer; increase \(t_0\); check correctness of bin assignment \(J(x)\); adjust cutpoints; confirm \(\gamma(t)\) schedule implemented correctly

5. **Tail bin extremely rare / threshold too extreme for chosen partition**  
   - **Symptom:** \(E_m\) almost never visited; p-hat dominated by empty-bin handling  
   - **Fix:** refine bins near \(\lambda^\*\); consider more gradual bin spacing near the tail; ensure the proposal can actually reach configurations with \(\lambda\ge\lambda^\*\)

---
