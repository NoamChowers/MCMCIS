from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from perm_pval.core.proposals import propose_localized_swaps, resolve_n_swap_pairs
from perm_pval.core.problem import PermutationTestProblem
from perm_pval.diagnostics.samc import visitation_frequency


@dataclass
class SAMCResult:
    estimate: float
    estimate_no_empty_bin_correction: float
    empty_bin_correction_delta: float
    empty_bin_correction_ratio: float
    acceptance_rate: float
    n_steps: int
    burn_in: int
    bin_edges: np.ndarray
    visit_counts: np.ndarray
    visitation_frequency: np.ndarray
    target_visitation: np.ndarray
    theta_final: np.ndarray
    theta_trace: np.ndarray
    step_sizes: np.ndarray
    pvalue_estimator: str
    tail_bin_index: Optional[int]
    pi0_adjustment: float
    empty_bin_indices: np.ndarray
    relative_frequency_error: np.ndarray
    max_abs_relative_frequency_error: float
    convergence_reached: bool
    localized_swaps_per_proposal: int
    seed: Optional[int]
    n_retained_after_burn_in: int


def _bin_index(value: float, bin_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(bin_edges, value, side="right") - 1)
    return int(np.clip(idx, 0, bin_edges.size - 2))


def _default_stepsize(step: int, t0: float) -> float:
    return float(t0 / max(t0, step))


def _auto_bin_edges_paper_right(
    problem: PermutationTestProblem,
    rng: np.random.Generator,
    n_bins: int,
    n_pilot: int,
    lambda_min: Optional[float],
) -> np.ndarray:
    """
    Paper-style partition for right-tail p-value evaluation:
    E_1,...,E_{m-1} partition [lambda_0, t_obs), E_m = [t_obs, +inf).
    """
    if lambda_min is None:
        vals = []
        for _ in range(n_pilot):
            y = problem.sample_uniform_labels(rng)
            vals.append(problem.compute_stat(y))
        vals_arr = np.asarray(vals, dtype=float)
        lo = float(np.min(vals_arr))
    else:
        lo = float(lambda_min)

    if not np.isfinite(lo):
        raise ValueError("lambda_min must be finite.")
    if lo >= problem.t_obs:
        lo = float(problem.t_obs - 1.0)

    # m = n_bins regions:
    # first m-1 bins on [lo, t_obs), last bin [t_obs, +inf).
    finite_edges = np.linspace(lo, float(problem.t_obs), n_bins, dtype=float)
    return np.concatenate([finite_edges, np.asarray([np.inf], dtype=float)])


def _is_paper_partition_right(bin_edges: np.ndarray, t_obs: float, atol: float = 1e-12) -> bool:
    if bin_edges.ndim != 1 or bin_edges.size < 3:
        return False
    if not np.all(np.diff(bin_edges) > 0):
        return False
    if not np.isinf(bin_edges[-1]):
        return False
    return bool(np.isclose(bin_edges[-2], t_obs, atol=atol, rtol=0.0))


def _relative_sampling_frequency_error(visit_counts: np.ndarray) -> np.ndarray:
    """
    Relative frequency error from Yu et al. (2011), Eq. (3.3).
    """
    visits = np.asarray(visit_counts, dtype=float)
    total = float(np.sum(visits))
    if total <= 0.0:
        return np.zeros_like(visits, dtype=float)
    visited = visits > 0.0
    m = visits.size
    m0 = int(np.sum(~visited))
    nonempty = m - m0
    if nonempty <= 0:
        return np.zeros_like(visits, dtype=float)
    target_flat = 1.0 / nonempty
    freq = visits / total
    out = np.zeros_like(visits, dtype=float)
    out[visited] = ((freq[visited] - target_flat) / target_flat) * 100.0
    out[~visited] = 0.0
    return out


def _paper_pvalue_estimate(
    theta: np.ndarray,
    target: np.ndarray,
    visit_counts: np.ndarray,
    tail_bin_index: int,
) -> tuple[float, float, np.ndarray]:
    """
    Estimate p-value using Yu et al. (2011), Eq. (3.2):
        p_hat = exp(theta_m) (pi_m + pi0) / sum_j exp(theta_j) (pi_j + pi0)
    with pi0 correction for empty bins.
    """
    m = int(theta.size)
    visits = np.asarray(visit_counts, dtype=float)
    visited = visits > 0.0
    empty = ~visited
    m0 = int(np.sum(empty))
    nonempty = m - m0
    empty_idx = np.flatnonzero(empty).astype(np.int64)

    if nonempty <= 0:
        return 0.0, 0.0, empty_idx

    pi0 = float(np.sum(target[empty]) / nonempty)
    adjusted = np.zeros_like(target, dtype=float)
    adjusted[visited] = target[visited] + pi0

    log_terms = np.full(m, -np.inf, dtype=float)
    valid = adjusted > 0.0
    log_terms[valid] = theta[valid] + np.log(adjusted[valid])

    shift = float(np.max(log_terms[valid]))
    denom = float(np.sum(np.exp(log_terms[valid] - shift)))
    if denom <= 0.0:
        return 0.0, pi0, empty_idx

    if not valid[tail_bin_index]:
        numer = 0.0
    else:
        numer = float(np.exp(log_terms[tail_bin_index] - shift))
    return float(numer / denom), pi0, empty_idx


def _pvalue_estimate_no_empty_bin_correction(
    theta: np.ndarray,
    target: np.ndarray,
    tail_bin_index: int,
) -> float:
    """
    Estimate p-value from theta without Yu et al.'s empty-bin pi0 correction.
    """
    theta_arr = np.asarray(theta, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    valid = target_arr > 0.0
    if not np.any(valid):
        return 0.0

    log_terms = np.full(theta_arr.size, -np.inf, dtype=float)
    log_terms[valid] = theta_arr[valid] + np.log(target_arr[valid])
    shift = float(np.max(log_terms[valid]))
    denom = float(np.sum(np.exp(log_terms[valid] - shift)))
    if denom <= 0.0 or not valid[tail_bin_index]:
        return 0.0
    numer = float(np.exp(log_terms[tail_bin_index] - shift))
    return float(numer / denom)


def _correction_ratio(corrected: float, uncorrected: float) -> float:
    if uncorrected > 0.0:
        return float(corrected / uncorrected)
    if corrected > 0.0:
        return float("inf")
    return float("nan")


def run_samc(
    problem: PermutationTestProblem,
    *,
    n_steps: int,
    bin_edges: Optional[np.ndarray] = None,
    n_bins: int = 10,
    target_visitation: Optional[np.ndarray] = None,
    t0: float = 1_000.0,
    burn_in: int = 0,
    seed: Optional[int] = None,
    init: str = "random",
    trace_every: int = 10,
    lambda_min: Optional[float] = None,
    proposal_size: float | int = 0.075,
    convergence_tolerance: float = 20.0,
) -> SAMCResult:
    """
    Run Stochastic Approximation Monte Carlo (SAMC).

    This implementation follows Yu et al. (2011):
    - right-tail partition with ``bin_edges[-2] = t_obs`` and ``bin_edges[-1] = +inf``
    - SAMC updates with ``gamma_t = t0 / max(t0, t)``
    - p-value estimate via Eq. (3.2) with empty-bin adjustment ``pi0``
    - relative frequency error diagnostic via Eq. (3.3)

    Notes on conventions:
    - The target used here is the paper's standard inverse-bin-weight form, yielding
      MH log-ratio ``theta[J(current)] - theta[J(proposed)]`` for symmetric proposals.
    - ``burn_in`` is used for diagnostics and empty-bin handling (visit counts),
      while SAMC adaptation itself runs across all ``n_steps``.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if burn_in < 0 or burn_in >= n_steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps.")
    if trace_every <= 0:
        raise ValueError("trace_every must be positive.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive.")

    rng = np.random.default_rng(seed)

    if problem.tail != "right":
        raise NotImplementedError(
            "Current SAMC implementation follows Yu et al. (2011) right-tail formulation only."
        )

    n_swap_pairs = resolve_n_swap_pairs(
        problem.n_treated,
        problem.n_control,
        proposal_size=proposal_size,
    )

    if bin_edges is None:
        bin_edges = _auto_bin_edges_paper_right(
            problem,
            rng,
            n_bins=n_bins,
            n_pilot=max(200, 20 * problem.n),
            lambda_min=lambda_min,
        )
    else:
        bin_edges = np.asarray(bin_edges, dtype=float)
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bin_edges must be a 1D array with at least 2 values.")
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError("bin_edges must be strictly increasing.")
    if not _is_paper_partition_right(bin_edges, problem.t_obs):
        raise ValueError(
            "bin_edges must encode right-tail partition with "
            "bin_edges[-2] == problem.t_obs and bin_edges[-1] == +inf."
        )

    k = bin_edges.size - 1
    tail_bin_index = int(k - 1)

    if target_visitation is None:
        target = np.full(k, 1.0 / k, dtype=float)
    else:
        target = np.asarray(target_visitation, dtype=float)
        if target.shape != (k,):
            raise ValueError("target_visitation must have one entry per bin.")
        if np.any(target <= 0):
            raise ValueError("target_visitation must be strictly positive.")
        target = target / np.sum(target)

    if init == "observed":
        y = problem.y_obs.copy()
    elif init == "random":
        y = problem.sample_uniform_labels(rng)
    else:
        raise ValueError("init must be either 'observed' or 'random'.")

    t_cur = problem.compute_stat(y)
    b_cur = _bin_index(t_cur, bin_edges)

    theta = np.zeros(k, dtype=float)
    visit_counts = np.zeros(k, dtype=np.int64)
    theta_trace: list[np.ndarray] = []
    step_sizes = np.zeros(n_steps, dtype=float)

    accepted = 0
    proposals = 0

    for step in range(1, n_steps + 1):
        y_prop = propose_localized_swaps(y, rng, n_swap_pairs=n_swap_pairs)
        t_prop = problem.compute_stat(y_prop)
        b_prop = _bin_index(t_prop, bin_edges)

        proposals += 1
        log_alpha = theta[b_cur] - theta[b_prop]
        if log_alpha >= 0.0 or np.log(rng.random()) < log_alpha:
            y = y_prop
            t_cur = t_prop
            b_cur = b_prop
            accepted += 1

        if step > burn_in:
            visit_counts[b_cur] += 1
        gamma = _default_stepsize(step, t0=t0)
        step_sizes[step - 1] = gamma

        indicator = np.zeros(k, dtype=float)
        indicator[b_cur] = 1.0
        theta += gamma * (indicator - target)
        theta -= np.mean(theta)

        if step % trace_every == 0:
            theta_trace.append(theta.copy())

    acceptance_rate = accepted / proposals if proposals > 0 else 0.0
    visit_freq = visitation_frequency(visit_counts)
    rel_freq_error = _relative_sampling_frequency_error(visit_counts)
    max_abs_rel_freq_error = float(np.max(np.abs(rel_freq_error)))
    convergence_reached = bool(max_abs_rel_freq_error < convergence_tolerance)

    estimate, pi0_adjustment, empty_bin_indices = _paper_pvalue_estimate(
        theta=theta,
        target=target,
        visit_counts=visit_counts,
        tail_bin_index=tail_bin_index,
    )
    estimate_no_correction = _pvalue_estimate_no_empty_bin_correction(
        theta=theta,
        target=target,
        tail_bin_index=tail_bin_index,
    )
    pvalue_estimator = "samc_paper_eq_3_2"

    return SAMCResult(
        estimate=estimate,
        estimate_no_empty_bin_correction=float(estimate_no_correction),
        empty_bin_correction_delta=float(estimate - estimate_no_correction),
        empty_bin_correction_ratio=_correction_ratio(
            corrected=float(estimate),
            uncorrected=float(estimate_no_correction),
        ),
        acceptance_rate=float(acceptance_rate),
        n_steps=int(n_steps),
        burn_in=int(burn_in),
        bin_edges=bin_edges.copy(),
        visit_counts=visit_counts,
        visitation_frequency=visit_freq,
        target_visitation=target,
        theta_final=theta.copy(),
        theta_trace=np.asarray(theta_trace, dtype=float),
        step_sizes=step_sizes,
        pvalue_estimator=pvalue_estimator,
        tail_bin_index=tail_bin_index,
        pi0_adjustment=float(pi0_adjustment),
        empty_bin_indices=empty_bin_indices,
        relative_frequency_error=rel_freq_error,
        max_abs_relative_frequency_error=max_abs_rel_freq_error,
        convergence_reached=convergence_reached,
        localized_swaps_per_proposal=int(n_swap_pairs),
        seed=seed,
        n_retained_after_burn_in=int(n_steps - burn_in),
    )
