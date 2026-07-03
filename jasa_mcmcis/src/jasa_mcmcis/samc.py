from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jasa_mcmcis.diagnostics import visitation_frequency
from jasa_mcmcis.problem import PermutationTestProblem
from jasa_mcmcis.proposals import propose_swaps, resolve_n_swap_pairs


@dataclass(frozen=True)
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
    tail_bin_index: int
    pi0_adjustment: float
    empty_bin_indices: np.ndarray
    relative_frequency_error: np.ndarray
    max_abs_relative_frequency_error: float
    convergence_reached: bool
    proposal_swaps: int
    seed: int | None
    n_retained_after_burn_in: int


def _bin_index(value: float, bin_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(bin_edges, value, side="right") - 1)
    return int(np.clip(idx, 0, bin_edges.size - 2))


def _default_stepsize(step: int, t0: float) -> float:
    return float(t0 / max(float(t0), float(step)))


def _auto_right_tail_bin_edges(
    problem: PermutationTestProblem,
    rng: np.random.Generator,
    *,
    n_bins: int,
    lambda_min: float | None,
    n_pilot: int,
) -> np.ndarray:
    if lambda_min is None:
        vals = np.empty(int(n_pilot), dtype=float)
        for i in range(int(n_pilot)):
            vals[i] = problem.compute_stat(problem.sample_uniform_labels(rng))
        lo = float(np.min(vals))
    else:
        lo = float(lambda_min)
    if not np.isfinite(lo):
        raise ValueError("lambda_min must be finite.")
    if lo >= problem.t_obs:
        lo = float(problem.t_obs - 1.0)
    finite_edges = np.linspace(lo, float(problem.t_obs), int(n_bins), dtype=float)
    return np.concatenate([finite_edges, np.asarray([np.inf], dtype=float)])


def _is_right_tail_partition(bin_edges: np.ndarray, t_obs: float) -> bool:
    return bool(
        bin_edges.ndim == 1
        and bin_edges.size >= 3
        and np.all(np.diff(bin_edges) > 0)
        and np.isinf(bin_edges[-1])
        and np.isclose(bin_edges[-2], t_obs, atol=1e-12, rtol=0.0)
    )


def _relative_sampling_frequency_error(visit_counts: np.ndarray) -> np.ndarray:
    visits = np.asarray(visit_counts, dtype=float)
    total = float(np.sum(visits))
    if total <= 0.0:
        return np.zeros_like(visits)
    visited = visits > 0.0
    nonempty = int(np.sum(visited))
    if nonempty <= 0:
        return np.zeros_like(visits)
    target_flat = 1.0 / nonempty
    freq = visits / total
    out = np.zeros_like(visits)
    out[visited] = ((freq[visited] - target_flat) / target_flat) * 100.0
    return out


def _paper_pvalue_estimate(
    theta: np.ndarray,
    target: np.ndarray,
    visit_counts: np.ndarray,
    tail_bin_index: int,
) -> tuple[float, float, np.ndarray]:
    visited = np.asarray(visit_counts, dtype=float) > 0.0
    empty = ~visited
    empty_idx = np.flatnonzero(empty).astype(np.int64)
    nonempty = int(np.sum(visited))
    if nonempty <= 0:
        return 0.0, 0.0, empty_idx

    pi0 = float(np.sum(target[empty]) / nonempty)
    adjusted = np.zeros_like(target, dtype=float)
    adjusted[visited] = target[visited] + pi0
    valid = adjusted > 0.0
    if not np.any(valid):
        return 0.0, pi0, empty_idx

    log_terms = np.full(theta.size, -np.inf, dtype=float)
    log_terms[valid] = theta[valid] + np.log(adjusted[valid])
    shift = float(np.max(log_terms[valid]))
    denom = float(np.sum(np.exp(log_terms[valid] - shift)))
    numer = 0.0
    if valid[tail_bin_index]:
        numer = float(np.exp(log_terms[tail_bin_index] - shift))
    return float(numer / denom) if denom > 0.0 else 0.0, pi0, empty_idx


def _pvalue_no_empty_bin_correction(
    theta: np.ndarray,
    target: np.ndarray,
    tail_bin_index: int,
) -> float:
    valid = np.asarray(target, dtype=float) > 0.0
    if not np.any(valid) or not valid[tail_bin_index]:
        return 0.0
    log_terms = np.full(theta.size, -np.inf, dtype=float)
    log_terms[valid] = theta[valid] + np.log(target[valid])
    shift = float(np.max(log_terms[valid]))
    denom = float(np.sum(np.exp(log_terms[valid] - shift)))
    numer = float(np.exp(log_terms[tail_bin_index] - shift))
    return float(numer / denom) if denom > 0.0 else 0.0


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
    bin_edges: np.ndarray | None = None,
    n_bins: int = 10,
    target_visitation: np.ndarray | None = None,
    t0: float = 1_000.0,
    burn_in: int = 0,
    seed: int | None = None,
    init: str = "random",
    trace_every: int = 10,
    lambda_min: float | None = None,
    proposal_size: float | int = 0.075,
    convergence_tolerance: float = 20.0,
) -> SAMCResult:
    """
    Run SAMC for a right-tail permutation p-value.

    The automatic partition has the form
    ``[lambda_min, ..., t_obs), [t_obs, inf)`` and the estimate follows the
    Yu et al. empty-bin correction used in the article simulations.
    """
    if problem.tail != "right":
        raise NotImplementedError("SAMC currently implements the right-tail partition.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if burn_in < 0 or burn_in >= n_steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if trace_every <= 0:
        raise ValueError("trace_every must be positive.")
    if t0 <= 0.0 or not np.isfinite(t0):
        raise ValueError("t0 must be finite and positive.")
    if convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive.")

    rng = np.random.default_rng(seed)
    n_swap_pairs = resolve_n_swap_pairs(problem.n_treated, problem.n_control, proposal_size)
    if bin_edges is None:
        bin_edges = _auto_right_tail_bin_edges(
            problem,
            rng,
            n_bins=int(n_bins),
            lambda_min=lambda_min,
            n_pilot=max(200, 20 * problem.n),
        )
    else:
        bin_edges = np.asarray(bin_edges, dtype=float)
    if not _is_right_tail_partition(bin_edges, problem.t_obs):
        raise ValueError("bin_edges must end with [t_obs, +inf) for a right-tail SAMC run.")

    k = int(bin_edges.size - 1)
    tail_bin_index = k - 1
    if target_visitation is None:
        target = np.full(k, 1.0 / k, dtype=float)
    else:
        target = np.asarray(target_visitation, dtype=float)
        if target.shape != (k,):
            raise ValueError("target_visitation must have one entry per bin.")
        if np.any(target <= 0.0):
            raise ValueError("target_visitation must be strictly positive.")
        target = target / np.sum(target)

    if init == "observed":
        y = problem.y_obs.copy()
    elif init == "random":
        y = problem.sample_uniform_labels(rng)
    else:
        raise ValueError("init must be 'observed' or 'random'.")

    t_cur = problem.compute_stat(y)
    b_cur = _bin_index(t_cur, bin_edges)
    theta = np.zeros(k, dtype=float)
    visit_counts = np.zeros(k, dtype=np.int64)
    theta_trace: list[np.ndarray] = []
    step_sizes = np.zeros(int(n_steps), dtype=float)
    accepted = 0
    proposals = 0

    for step in range(1, int(n_steps) + 1):
        y_prop = propose_swaps(y, rng, n_swap_pairs)
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

        gamma = _default_stepsize(step, t0)
        step_sizes[step - 1] = gamma
        indicator = np.zeros(k, dtype=float)
        indicator[b_cur] = 1.0
        theta += gamma * (indicator - target)
        theta -= np.mean(theta)

        if step % trace_every == 0:
            theta_trace.append(theta.copy())

    freq = visitation_frequency(visit_counts)
    rel_error = _relative_sampling_frequency_error(visit_counts)
    max_abs_rel_error = float(np.max(np.abs(rel_error)))
    estimate, pi0, empty_idx = _paper_pvalue_estimate(
        theta=theta,
        target=target,
        visit_counts=visit_counts,
        tail_bin_index=tail_bin_index,
    )
    no_correction = _pvalue_no_empty_bin_correction(theta, target, tail_bin_index)

    return SAMCResult(
        estimate=float(estimate),
        estimate_no_empty_bin_correction=float(no_correction),
        empty_bin_correction_delta=float(estimate - no_correction),
        empty_bin_correction_ratio=_correction_ratio(float(estimate), float(no_correction)),
        acceptance_rate=float(accepted / proposals) if proposals else 0.0,
        n_steps=int(n_steps),
        burn_in=int(burn_in),
        bin_edges=bin_edges.copy(),
        visit_counts=visit_counts,
        visitation_frequency=freq,
        target_visitation=target,
        theta_final=theta.copy(),
        theta_trace=np.asarray(theta_trace, dtype=float),
        step_sizes=step_sizes,
        pvalue_estimator="samc_paper_eq_3_2",
        tail_bin_index=int(tail_bin_index),
        pi0_adjustment=float(pi0),
        empty_bin_indices=empty_idx,
        relative_frequency_error=rel_error,
        max_abs_relative_frequency_error=max_abs_rel_error,
        convergence_reached=bool(max_abs_rel_error < convergence_tolerance),
        proposal_swaps=int(n_swap_pairs),
        seed=seed,
        n_retained_after_burn_in=int(n_steps - burn_in),
    )
