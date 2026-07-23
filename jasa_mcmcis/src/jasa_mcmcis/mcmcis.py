from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from jasa_mcmcis.diagnostics import (
    WeightSummary,
    effective_sample_size,
    obm_long_run_variance,
    summarize_weights,
)
from jasa_mcmcis.problem import PermutationTestProblem
from jasa_mcmcis.proposals import propose_swaps, resolve_n_swap_pairs

TiltMode = Literal["smooth_hinge", "step"]


def hard_step_r_for_target_tail_mass(p0: float, q: float) -> float:
    """
    Tail multiplier for pi_r(y) ∝ f(y) * {1 + (r - 1) 1_A(y)}.

    If f(A) = p0, then pi_r(A) = q when
        r = q * (1 - p0) / (p0 * (1 - q)).
    """
    p0_f = float(p0)
    q_f = float(q)
    if not np.isfinite(p0_f) or not 0.0 < p0_f < 1.0:
        raise ValueError("p0 must be finite and lie in (0, 1).")
    if not np.isfinite(q_f) or not 0.0 < q_f < 1.0:
        raise ValueError("q must be finite and lie in (0, 1).")
    if q_f <= p0_f:
        raise ValueError("q must exceed p0 for a tail-upweighting hard-step tilt.")
    return float((q_f * (1.0 - p0_f)) / (p0_f * (1.0 - q_f)))


def hard_step_beta_for_target_tail_mass(p0: float, q: float) -> float:
    """Equivalent ``tilt_mode='step'`` beta, where beta = log(r)."""
    return float(np.log(hard_step_r_for_target_tail_mass(p0, q)))


@dataclass(frozen=True)
class ChainDiagnostics:
    acceptance_rate: float
    n_proposals: int
    n_accepted: int
    mean_stat: float
    sd_stat: float


@dataclass(frozen=True)
class MCMCISResult:
    """Self-normalized MCMC importance-sampling output."""

    estimate: float
    ess: float
    beta: float
    sigma_t: float
    tilt_mode: str
    proposal_swaps: int
    n_weighted_samples: int
    tail_hits_weighted_sample: int
    raw_tail_rate: float
    acceptance_rate: float
    chain_acceptance_rates: tuple[float, ...]
    mcse_obm: float | None
    variance_obm: float | None
    obm_batch_sizes: tuple[int, ...]
    weight_summary: WeightSummary
    seed: int | None
    chain_seeds: tuple[int, ...]
    t_samples: np.ndarray
    log_weights: np.ndarray
    tail_indicators: np.ndarray
    final_labels: tuple[np.ndarray, ...]
    chain_diagnostics: tuple[ChainDiagnostics, ...]


def _shortfall(
    problem: PermutationTestProblem,
    t: float,
    sigma_t: float,
    tilt_mode: TiltMode,
) -> float:
    if tilt_mode == "smooth_hinge":
        return float(max((problem.tail_threshold - problem.tail_value(t)) / sigma_t, 0.0))
    if tilt_mode == "step":
        return 0.0 if problem.is_in_tail(t) else 1.0
    raise ValueError("tilt_mode must be 'smooth_hinge' or 'step'.")


def _initial_state(
    problem: PermutationTestProblem,
    rng: np.random.Generator,
    init: str,
    init_state: np.ndarray | None,
) -> np.ndarray:
    if init_state is not None:
        return problem.validate_labels(init_state).copy()
    if init == "observed":
        return problem.y_obs.copy()
    if init == "random":
        return problem.sample_uniform_labels(rng)
    raise ValueError("init must be 'observed', 'random', or explicit labels.")


def _run_chain(
    problem: PermutationTestProblem,
    rng: np.random.Generator,
    *,
    beta: float,
    sigma_t: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    init: str,
    init_state: np.ndarray | None,
    tilt_mode: TiltMode,
    n_swap_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
    y = _initial_state(problem, rng, init, init_state)
    t_cur = problem.compute_stat(y)
    q_cur = _shortfall(problem, t_cur, sigma_t, tilt_mode)

    t_samples: list[float] = []
    q_samples: list[float] = []
    tail: list[int] = []
    accepted = 0
    proposals = 0

    for step in range(n_steps):
        y_prop = propose_swaps(y, rng, n_swap_pairs)
        t_prop = problem.compute_stat(y_prop)
        q_prop = _shortfall(problem, t_prop, sigma_t, tilt_mode)

        proposals += 1
        log_alpha = -beta * (q_prop - q_cur)
        if log_alpha >= 0.0 or np.log(rng.random()) < log_alpha:
            y = y_prop
            t_cur = t_prop
            q_cur = q_prop
            accepted += 1

        if step >= burn_in and ((step - burn_in) % thin == 0):
            t_samples.append(float(t_cur))
            q_samples.append(float(q_cur))
            tail.append(int(problem.is_in_tail(t_cur)))

    return (
        np.asarray(t_samples, dtype=float),
        np.asarray(q_samples, dtype=float),
        np.asarray(tail, dtype=np.int8),
        int(accepted),
        int(proposals),
        y.copy(),
    )


def _resolve_init_states(
    problem: PermutationTestProblem,
    n_chains: int,
    init: str | np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
) -> tuple[str, list[np.ndarray | None]]:
    if isinstance(init, str):
        if init not in {"observed", "random"}:
            raise ValueError("init must be 'observed', 'random', or explicit labels.")
        return init, [None] * n_chains
    if isinstance(init, (list, tuple)):
        if len(init) != n_chains:
            raise ValueError("explicit init list length must match n_chains.")
        return "random", [problem.validate_labels(np.asarray(v, dtype=np.int8)).copy() for v in init]
    if n_chains != 1:
        raise ValueError("a single explicit init state requires n_chains == 1.")
    return "random", [problem.validate_labels(np.asarray(init, dtype=np.int8)).copy()]


def _snis_obm_variance(
    weights: np.ndarray,
    indicators: np.ndarray,
    estimate: float,
    chain_lengths: list[int],
    batch_size: int | None,
) -> tuple[float | None, float | None, tuple[int, ...]]:
    n_total = int(weights.size)
    mean_w = float(np.mean(weights))
    if mean_w <= 0.0 or n_total < 4:
        return None, None, tuple()

    influence = weights * (indicators - estimate)
    start = 0
    var_mean_h = 0.0
    batch_sizes: list[int] = []
    for chain_len in chain_lengths:
        stop = start + int(chain_len)
        chain_h = influence[start:stop]
        start = stop
        sigma2, b = obm_long_run_variance(chain_h, batch_size=batch_size)
        batch_sizes.append(int(b))
        if np.isfinite(sigma2) and chain_len > 0:
            var_mean_h += (chain_len * sigma2) / (n_total * n_total)

    variance = float(var_mean_h / (mean_w * mean_w))
    return variance, float(np.sqrt(max(variance, 0.0))), tuple(batch_sizes)


def run_mcmc_is(
    problem: PermutationTestProblem,
    *,
    beta: float,
    n_steps: int,
    sigma_t: float = 1.0,
    burn_in: int = 0,
    thin: int = 1,
    n_chains: int = 1,
    seed: int | None = None,
    init: str | np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...] = "random",
    tilt_mode: TiltMode = "smooth_hinge",
    proposal_size: float | int = 0.075,
    estimate_variance: bool = True,
    obm_batch_size: int | None = None,
) -> MCMCISResult:
    """
    Estimate a permutation p-value with tilted MCMC and self-normalized IS.

    The tilted distribution is proportional to
    ``exp(-beta * q(y))`` over the fixed-size permutation space, where
    ``q(y)`` is either a scaled hinge shortfall from the tail or a one-step
    outside-tail penalty. Returned weights are proportional to
    ``exp(beta * q(y))``.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if burn_in < 0 or burn_in >= n_steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    if beta < 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and non-negative.")
    if sigma_t <= 0.0 or not np.isfinite(sigma_t):
        raise ValueError("sigma_t must be finite and positive.")
    if tilt_mode not in {"smooth_hinge", "step"}:
        raise ValueError("tilt_mode must be 'smooth_hinge' or 'step'.")

    n_swap_pairs = resolve_n_swap_pairs(
        problem.n_treated,
        problem.n_control,
        proposal_size=proposal_size,
    )
    init_mode, init_states = _resolve_init_states(problem, int(n_chains), init)

    seed_seq = np.random.SeedSequence(seed)
    child_seqs = seed_seq.spawn(int(n_chains))
    chain_seeds: list[int] = []
    t_by_chain: list[np.ndarray] = []
    q_by_chain: list[np.ndarray] = []
    tail_by_chain: list[np.ndarray] = []
    final_labels: list[np.ndarray] = []
    diagnostics: list[ChainDiagnostics] = []
    acceptance_rates: list[float] = []
    total_accepted = 0
    total_proposals = 0

    for chain_idx, child in enumerate(child_seqs):
        chain_seeds.append(int(child.generate_state(1)[0]))
        rng = np.random.default_rng(child)
        t_chain, q_chain, tail_chain, accepted, proposals, y_final = _run_chain(
            problem,
            rng,
            beta=float(beta),
            sigma_t=float(sigma_t),
            n_steps=int(n_steps),
            burn_in=int(burn_in),
            thin=int(thin),
            init=init_mode,
            init_state=init_states[chain_idx],
            tilt_mode=tilt_mode,
            n_swap_pairs=n_swap_pairs,
        )
        t_by_chain.append(t_chain)
        q_by_chain.append(q_chain)
        tail_by_chain.append(tail_chain)
        final_labels.append(y_final)

        acceptance = float(accepted / proposals)
        acceptance_rates.append(acceptance)
        total_accepted += int(accepted)
        total_proposals += int(proposals)
        diagnostics.append(
            ChainDiagnostics(
                acceptance_rate=acceptance,
                n_proposals=int(proposals),
                n_accepted=int(accepted),
                mean_stat=float(np.mean(t_chain)) if t_chain.size else float("nan"),
                sd_stat=float(np.std(t_chain)) if t_chain.size else float("nan"),
            )
        )

    t_samples = np.concatenate(t_by_chain)
    q_samples = np.concatenate(q_by_chain)
    indicators = np.concatenate(tail_by_chain).astype(np.int8)
    if t_samples.size == 0:
        raise RuntimeError("No samples were retained. Check n_steps, burn_in, and thin.")

    log_weights = float(beta) * q_samples
    shifted_weights = np.exp(log_weights - float(np.max(log_weights)))
    estimate = float(np.dot(shifted_weights, indicators) / np.sum(shifted_weights))
    ess = effective_sample_size(shifted_weights)
    variance = None
    mcse = None
    obm_sizes: tuple[int, ...] = tuple()
    if estimate_variance:
        variance, mcse, obm_sizes = _snis_obm_variance(
            shifted_weights,
            indicators,
            estimate,
            [int(x.size) for x in q_by_chain],
            obm_batch_size,
        )

    return MCMCISResult(
        estimate=estimate,
        ess=float(ess),
        beta=float(beta),
        sigma_t=float(sigma_t),
        tilt_mode=str(tilt_mode),
        proposal_swaps=int(n_swap_pairs),
        n_weighted_samples=int(t_samples.size),
        tail_hits_weighted_sample=int(np.sum(indicators)),
        raw_tail_rate=float(np.mean(indicators)),
        acceptance_rate=float(total_accepted / total_proposals),
        chain_acceptance_rates=tuple(acceptance_rates),
        mcse_obm=mcse,
        variance_obm=variance,
        obm_batch_sizes=obm_sizes,
        weight_summary=summarize_weights(shifted_weights),
        seed=seed,
        chain_seeds=tuple(chain_seeds),
        t_samples=t_samples,
        log_weights=log_weights,
        tail_indicators=indicators,
        final_labels=tuple(final_labels),
        chain_diagnostics=tuple(diagnostics),
    )


def run_hard_step_mcmc_is(
    problem: PermutationTestProblem,
    *,
    p0: float,
    q: float,
    n_steps: int,
    sigma_t: float = 1.0,
    burn_in: int = 0,
    thin: int = 1,
    n_chains: int = 1,
    seed: int | None = None,
    init: str | np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...] = "random",
    proposal_size: float | int = 0.075,
    estimate_variance: bool = True,
    obm_batch_size: int | None = None,
) -> MCMCISResult:
    """
    Run MCMC-IS with the hard-step tail tilt fixed by reference ``p0`` and target ``q``.
    """
    beta = hard_step_beta_for_target_tail_mass(p0, q)
    return run_mcmc_is(
        problem,
        beta=beta,
        sigma_t=sigma_t,
        n_steps=n_steps,
        burn_in=burn_in,
        thin=thin,
        n_chains=n_chains,
        seed=seed,
        init=init,
        tilt_mode="step",
        proposal_size=proposal_size,
        estimate_variance=estimate_variance,
        obm_batch_size=obm_batch_size,
    )
