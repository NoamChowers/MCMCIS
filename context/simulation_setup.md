# Simulation setup for permutation MCMC-IS (step vs smooth vs SAMC vs baselines)

This note proposes *implementation-oriented* simulation designs that (i) cover a range of null-statistic tail shapes, (ii) admit **ground-truth p-values** (exact or essentially exact), and (iii) resemble common application regimes (GWAS / particle physics / bioinformatics) while remaining tractable.

Related note:
- See `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/context/checkpoints.md` for how simulation checkpoints are interpreted in the current notebooks. In particular, cross-method checkpoints are total budgets, while beta-diagnostic checkpoints are production-only.
- See `/Users/noamchowers/Documents/University/Thesis/Code/MCMCIS/context/multiprocessing.md` for the current process-parallel execution strategy and the recommended `N_JOBS` defaults.

---


## 1) Group sizes: what to simulate (and why)

We need two regimes:

### Regime A (exact-enumerable permutation truth)
Choose sizes where \(\binom{n}{n_1}\) is manageable (or can be enumerated once per dataset).
- **Typical**: \(n_1=n_2\in\{10,12,15,20\}\)  
  - \(\binom{40}{20}\approx 1.38\times 10^{11}\) is too large, but \(\binom{30}{15}\approx 1.55\times 10^8\) is borderline; \(\binom{24}{12}\approx 2.7\times 10^6\) is easy.
- **Recommended exact-enumeration range**:
  - \(n_1=n_2=8\) (\(\binom{16}{8}=12870\))
  - \(n_1=n_2=10\) (\(\binom{20}{10}=184756\))
  - \(n_1=n_2=12\) (\(\binom{24}{12}=2.7\times 10^6\))
  - \(n_1=n_2=14\) (\(\binom{28}{14}=4.0\times 10^7\), doable with optimized code / caching)

These give *true permutation p-values* and let us validate correctness and variance scaling.

### Regime B (large-\(n\), “realistic”)
Here \(\binom{n}{n_1}\) is enormous; ground truth cannot be enumerated.
Use **analytic-null** truth (when available), or a **high-precision proxy**:
- \(n_1=n_2\in\{50,100,250,500\}\)
- This is where rare p-values (\(10^{-8}\)) become relevant.

---

## 2) Test statistics to include (variety of null tail behavior)

We want a menu that spans:
- approximately Gaussian tails (CLT-like),
- heavier tails / skewness,
- discrete / count-like,
- bounded / “uniform-ish” behaviors 

Below are recommended statistics with clear implementation and plausible application analogies.

### (S1) Difference in means (two-sample)
\[
T = |\bar X_1 - \bar X_2|
\]
- **Null families**:
  - Poisson(\(\gamma\)) (your earlier setup)
  - Normal(0,1)
  - Laplace(0,1) (heavier tails)
- **Analogies**: generic continuous traits; “GWAS-like” when standardized; broad baseline.

### (S2) Studentized mean difference (two-sample t-statistic)
\[
T = \frac{|\bar X_1 - \bar X_2|}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}
\]
- More “self-normalized”; often closer to Gaussian under the null than raw mean diff.
- **Analogies**: common in practice; improves comparability across scales.

### (S3) Wilcoxon rank-sum / Mann–Whitney U
- Statistic based on ranks (nonparametric).
- **Analogies**: robust bio-statistics; distribution is discrete and known.

### (S4) 2×2 association (Fisher exact / GWAS-style single-SNP)
Let \(Y\in\{0,1\}\) be phenotype and \(G\in\{0,1\}\) be genotype indicator (or allele count collapsed).
Use:
- Fisher exact p-value for a fixed margin contingency table
- or Pearson \(\chi^2\) / score statistic
- **Analogies**: GWAS and case-control tests.

### (S5) Difference in Poisson counts / Skellam-style
If counts are independent Poisson:
\[
T = |S_1 - S_2|
\]
where \(S_g=\sum_{i\in g} X_i\).
- **Analogies**: particle counts; photon counts.

### (S6) Max-type statistic (multiple testing proxy)
For \(d\) coordinates (e.g. genes / SNPs), compute per-coordinate two-sample statistic \(T_j\), then
\[
T = \max_{1\le j\le d} T_j
\]
- Use modest \(d\in\{20,50,100\}\) so ground truth remains feasible in Regime A (or analytic approximations in Regime B).
- **Analogies**: family-wise error, “look-elsewhere effect” in physics, multiple testing in GWAS.

---

## 3) Ground-truth p-values (exact / essentially exact)

We need a hierarchy of “truth methods” depending on the setting.

### Truth method T0: exact permutation enumeration (Regime A)
For small \(n\), enumerate all labelings (or sample without replacement exactly).
- Compute \(T\) for all \(\binom{n}{n_1}\) permutations and evaluate:
\[
p = \frac{\#\{\pi: T(\pi)\ge T_{\mathrm{obs}}\}}{\binom{n}{n_1}}
\]
- Works for S1–S3 and S6 (with small \(d\)).

### Truth method T1: exact null distribution (analytic)
Use known exact distributions:
- **S3 Wilcoxon**: exact null distribution available via dynamic programming / known recursion
- **S4 Fisher exact**: exact hypergeometric tail probability
- **S5 Skellam** (difference of Poissons) if modeling counts directly (not permutation); exact pmf exists
These give “gold truth” even at large \(n\) and tiny p-values (up to numerical precision).

### Truth method T2: exact conditional distribution via DP / convolution
For statistics that reduce to sums (e.g. S5 with fixed totals, or S1 with integer-valued \(X\)):
- For fixed observed sample values, the permutation distribution of a sum over chosen indices can be computed by subset-sum style DP (pseudo-polynomial in total sum).
- Useful when \(X_i\) are nonnegative integers and totals are moderate.

### Truth method T3: extremely accurate Monte Carlo (fallback)
If none of the above apply, use *very large IID sampling under \(f\)* to approximate \(p\) with a confidence interval narrow enough to treat as truth.
- Only feasible when p is not astronomically small (e.g. \(p\ge 10^{-6}\)), unless compute budget is huge.
- Prefer T0–T2 whenever possible.

---

## 4) Simulation “suites” (recommended experiments)

### Suite A: exact-permutation validation (correctness + variance scaling)
- Choose \(n_1=n_2\in\{10,12,14\}\)
- Generate one dataset under the null, compute \(T_{\mathrm{obs}}\) at a chosen quantile so that true \(p\in\{10^{-2},10^{-3},10^{-4}\}\).
- Compute exact permutation truth via enumeration (T0).
- Compare estimators:
  - IID permutations
  - Step tilt + MCMC-SNIS
  - Smooth-hinge tilt + MCMC-SNIS (with beta tuning)
  - SAMC (if implemented)
  - Any baseline RW-MH without tilt (should fail in tails)

### Suite B: GWAS-like exact truth (Fisher)
- Use 2×2 contingency tables with fixed margins.
- Define \(T\) as a score / \(\chi^2\) or odds-ratio based statistic, with p-value from Fisher exact (T1).
- Here the “state space” is all labelings consistent with margins; proposals can swap case/control labels.

### Suite C: physics-like Poisson counts (Skellam / conditional test)
- If modeling *counts* rather than labels, you can use analytic Skellam (T1).
- If you want *permutation* flavor: fix observed counts and permute group labels; for small \(n\) use enumeration (T0), for larger use DP when totals are moderate (T2).

### Suite D: max-statistic / multiple testing proxy
- Choose \(d\in\{20,50\}\), \(n_1=n_2\in\{10,12\}\)
- Use S6 with per-coordinate mean difference or t-statistic.
- Ground truth via enumeration (T0) if feasible; else approximate with a high-precision importance method as reference.

---

## 5) What “resembles real world” here (and what doesn’t)

- **GWAS resemblance**: Fisher exact / score tests, many tests (captured by S6)  
- **Physics resemblance**: Poisson counting, “5-sigma” tails (captured by S5, S6)  
- **Bioinformatics resemblance**: heavy-tailed expression / rank tests (captured by Laplace/S3), multiple testing (S6)

Caveat: real permutation spaces have high-dimensional local geometry; the toy model abstracts this. Simulations should therefore include:
- different proposal kernels (swap-1, swap-5%, block swap)
- different \(T\) shapes (smooth vs jagged vs discrete)

---

## 6) Concrete defaults (so implementation can start)

- **Relative RMSE target**: \(\varepsilon\in\{0.1,0.2\}\)
- **Replicates per setting**: 100 independent runs for variance estimates
- **Chain length budgeting**:
  - Pilot: \(M=10^4\) iid samples under \(f\) for \(\sigma_T\) and \(\beta_0\)
  - Tuning: 10–20 short chains of length 1e3–1e4
  - Production: choose \(n\) to hit target relRMSE based on estimated RelAVAR

- **q-target grid to test**:
  - \(\{10^{-3},10^{-2},10^{-1}\}\)
  - toy-model motivated: \(q_{\text{target}}=p_0^{D_\alpha}\) with \(p_0\in\{10^{-6},10^{-8}\}\)

---

## 7) What to report per run (minimum logging)

- \(\hat p\), estimated SE, relative error, runtime/#stat evals  
- for MCMC: acceptance rate, estimated \(q\), ESS proxy, autocorrelation at lag 1/10/100  
- tuning trace: \(\beta\) vs \(\hat q\) over tuning iterations

---

## 8) Checklist: ground truth feasibility

Before running a setting, choose a truth method:

- If \(\binom{n}{n_1}\) manageable → use **T0 enumeration**
- Else if a classical exact test exists (Fisher, Wilcoxon) → use **T1**
- Else if statistic is a sum with moderate integer totals → consider **T2**
- Else fallback **T3 high-precision IID** for not-too-small p
