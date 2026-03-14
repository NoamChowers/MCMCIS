from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from perm_pval.core.proposals import propose_localized_swaps, resolve_n_swap_pairs
from perm_pval.core.problem import PermutationTestProblem
from perm_pval.diagnostics.is_weights import (
    ISWeightSummary,
    effective_sample_size,
    summarize_weights,
)
from perm_pval.diagnostics.mcmc import integrated_autocorrelation_time, obm_long_run_variance


@dataclass
class MCMCChainDiagnostics:
    acceptance_rate: float
    n_proposals: int
    n_accepted: int
    mean_stat: float
    std_stat: float
    iact_stat: float


@dataclass
class MCMCISResult:
    estimate: float
    ess: float
    tilt_mode: str
    beta: float
    sigma_t: float
    snis_variance_obm: Optional[float]
    snis_mcse_obm: Optional[float]
    obm_batch_size_requested: int | None
    obm_chain_batch_sizes: list[int]
    obm_chain_long_run_variances: list[float]
    n_weighted_samples: int
    tail_hits_weighted_sample: int
    tail_share_raw_sample: float
    overall_acceptance_rate: float
    acceptance_rates: list[float]
    chain_diagnostics: list[MCMCChainDiagnostics]
    weight_summary: ISWeightSummary
    seed: Optional[int]
    chain_seeds: list[int]
    t_samples: np.ndarray
    log_weights: np.ndarray
    tail_indicators: np.ndarray
    final_states: list[np.ndarray]


def right_tail_deficit(t: float, t_obs: float) -> float:
    """
    (t_obs - t)_+ used in the hinge-style tilting scheme.
    """
    return float(max(t_obs - t, 0.0))


def right_tail_deficit_scaled(t: float, t_obs: float, sigma_t: float) -> float:
    """
    Scaled hinge shortfall:
        ((t_obs - t) / sigma_t)_+
    """
    return float(max((t_obs - t) / sigma_t, 0.0))


def right_tail_step_shortfall(t: float, t_obs: float) -> float:
    """
    Step shortfall:
        1{t < t_obs}
    """
    return 1.0 if t < t_obs else 0.0


def transformed_stat_for_tilt(t: float, tail: str) -> float:
    """
    Legacy helper kept for compatibility with older notebooks.
    """
    if tail == "right":
        return float(t)
    if tail == "left":
        return float(-t)
    return float(abs(t))


def make_beta_ladder(
    min_beta: float, max_beta: float, n_levels: int, spacing: Literal["linear", "geometric"] = "linear"
) -> np.ndarray:
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2.")
    if spacing == "linear":
        return np.linspace(min_beta, max_beta, n_levels, dtype=float)
    if min_beta <= 0 or max_beta <= 0:
        raise ValueError("geometric spacing requires positive min_beta and max_beta.")
    return np.geomspace(min_beta, max_beta, n_levels, dtype=float)


def _run_single_chain(
    problem: PermutationTestProblem,
    rng: np.random.Generator,
    beta: float,
    sigma_t: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    init_mode: str,
    init_state: np.ndarray | None,
    tilt_mode: Literal["smooth_hinge", "step"],
    n_swap_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
    if init_state is not None:
        y = problem.validate_labels(np.asarray(init_state, dtype=np.int8)).copy()
    elif init_mode == "observed":
        y = problem.y_obs.copy()
    elif init_mode == "random":
        y = problem.sample_uniform_labels(rng)
    else:
        raise ValueError("init must be either 'observed' or 'random'.")

    t_cur = problem.compute_stat(y)
    if tilt_mode == "smooth_hinge":
        q_cur = right_tail_deficit_scaled(t_cur, problem.t_obs, sigma_t)
    else:
        q_cur = right_tail_step_shortfall(t_cur, problem.t_obs)

    accepted = 0
    proposals = 0
    t_samples: list[float] = []
    q_samples: list[float] = []
    tail_indicators: list[int] = []

    for step in range(n_steps):
        y_prop = propose_localized_swaps(y, rng, n_swap_pairs=n_swap_pairs)
        t_prop = problem.compute_stat(y_prop)
        if tilt_mode == "smooth_hinge":
            q_prop = right_tail_deficit_scaled(t_prop, problem.t_obs, sigma_t)
        else:
            q_prop = right_tail_step_shortfall(t_prop, problem.t_obs)

        proposals += 1
        # log g(y) = const - beta * (t_obs - T(y))_+
        log_alpha = -beta * (q_prop - q_cur)
        if log_alpha >= 0.0 or np.log(rng.random()) < log_alpha:
            y = y_prop
            t_cur = t_prop
            q_cur = q_prop
            accepted += 1

        if step >= burn_in and ((step - burn_in) % thin == 0):
            t_samples.append(float(t_cur))
            q_samples.append(float(q_cur))
            tail_indicators.append(int(problem.is_in_tail(t_cur)))

    return (
        np.asarray(t_samples, dtype=float),
        np.asarray(q_samples, dtype=float),
        np.asarray(tail_indicators, dtype=np.int8),
        accepted,
        proposals,
        y.copy(),
    )


def run_mcmc_is(
    problem: PermutationTestProblem,
    *,
    beta: float,
    sigma_t: float = 1.0,
    n_steps: int,
    burn_in: int = 0,
    thin: int = 1,
    n_chains: int = 1,
    seed: Optional[int] = None,
    init: str | np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...] = "random",
    tilt_mode: Literal["smooth_hinge", "step"] = "smooth_hinge",
    proposal_size: float | int = 0.075,
    estimate_variance: bool = True,
    obm_batch_size: int | None = None,
) -> MCMCISResult:
    """
    MCMC-IS for right-tail permutation p-values.

    Supported tilt modes:

    1) smooth_hinge (default)

        g_beta(y) ∝ f(y) * exp(-beta * ((t_obs - T(y)) / sigma_t)_+)

       SNIS weights:
         w(y) ∝ exp(beta * ((t_obs - T(y)) / sigma_t)_+)

    2) step

        g_beta(y) ∝ f(y) * exp(-beta * 1{T(y) < t_obs})

       SNIS weights:
         w(y) ∝ exp(beta * 1{T(y) < t_obs})

    where f is uniform over valid labelings.

    Variance estimation (optional) uses OBM on the influence sequence
    h_i = w_i * (I_i - p_hat) and delta-method scaling by mean(w)^2.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if burn_in < 0 or burn_in >= n_steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if tilt_mode not in ("smooth_hinge", "step"):
        raise ValueError("tilt_mode must be one of {'smooth_hinge', 'step'}.")
    if not np.isfinite(sigma_t) or sigma_t <= 0.0:
        raise ValueError("sigma_t must be a finite positive scalar.")
    if problem.tail != "right":
        raise NotImplementedError(
            "Current MCMC-IS tilting implementation assumes a right-tail test and uses "
            "g_beta(y) ∝ f(y) * exp(-beta * ((t_obs - T(y)) / sigma_t)_+)."
        )

    n_swap_pairs = resolve_n_swap_pairs(
        problem.n_treated,
        problem.n_control,
        proposal_size=proposal_size,
    )

    init_mode = "random"
    init_states: list[np.ndarray | None]
    if isinstance(init, str):
        if init not in {"observed", "random"}:
            raise ValueError("init must be 'observed', 'random', or explicit chain state(s).")
        init_mode = init
        init_states = [None] * int(n_chains)
    elif isinstance(init, (list, tuple)):
        if len(init) != int(n_chains):
            raise ValueError("Explicit init state list must match n_chains.")
        init_states = [problem.validate_labels(np.asarray(state, dtype=np.int8)).copy() for state in init]
        init_mode = "random"
    else:
        if int(n_chains) != 1:
            raise ValueError("A single explicit init state can only be used when n_chains == 1.")
        init_states = [problem.validate_labels(np.asarray(init, dtype=np.int8)).copy()]
        init_mode = "random"

    seed_seq = np.random.SeedSequence(seed)
    child_seqs = seed_seq.spawn(n_chains)

    all_t_samples: list[np.ndarray] = []
    all_q_samples: list[np.ndarray] = []
    all_tail: list[np.ndarray] = []
    chain_diagnostics: list[MCMCChainDiagnostics] = []
    acceptance_rates: list[float] = []
    chain_seeds: list[int] = []
    final_states: list[np.ndarray] = []

    total_accepted = 0
    total_proposals = 0

    for chain_idx, ss in enumerate(child_seqs):
        chain_seed = int(ss.generate_state(1)[0])
        chain_seeds.append(chain_seed)
        rng = np.random.default_rng(ss)
        t_chain, q_chain, tail_chain, n_accepted, n_prop, y_final = _run_single_chain(
            problem=problem,
            rng=rng,
            beta=beta,
            sigma_t=sigma_t,
            n_steps=n_steps,
            burn_in=burn_in,
            thin=thin,
            init_mode=init_mode,
            init_state=init_states[chain_idx],
            tilt_mode=tilt_mode,
            n_swap_pairs=n_swap_pairs,
        )
        all_t_samples.append(t_chain)
        all_q_samples.append(q_chain)
        all_tail.append(tail_chain)
        final_states.append(y_final)

        acceptance = n_accepted / n_prop
        acceptance_rates.append(float(acceptance))
        total_accepted += n_accepted
        total_proposals += n_prop

        iact = integrated_autocorrelation_time(t_chain) if t_chain.size > 1 else float("nan")
        chain_diagnostics.append(
            MCMCChainDiagnostics(
                acceptance_rate=float(acceptance),
                n_proposals=int(n_prop),
                n_accepted=int(n_accepted),
                mean_stat=float(np.mean(t_chain)) if t_chain.size else float("nan"),
                std_stat=float(np.std(t_chain)) if t_chain.size else float("nan"),
                iact_stat=float(iact),
            )
        )

    t_samples = np.concatenate(all_t_samples) if all_t_samples else np.array([], dtype=float)
    q_samples = np.concatenate(all_q_samples) if all_q_samples else np.array([], dtype=float)
    tail_indicators = np.concatenate(all_tail) if all_tail else np.array([], dtype=np.int8)
    if t_samples.size == 0:
        raise RuntimeError("No MCMC samples collected. Check n_steps, burn_in, and thin.")

    # For g_beta(y) ∝ f(y) * exp(-beta * q(y)), we have:
    # w(y) ∝ f/g_beta ∝ exp(beta * q(y)).
    log_weights = beta * q_samples
    shift = float(np.max(log_weights))
    weights = np.exp(log_weights - shift)

    weight_sum = float(np.sum(weights))
    estimate = float(np.dot(weights, tail_indicators) / weight_sum)
    ess = float(effective_sample_size(weights))
    weight_summary = summarize_weights(weights)

    snis_variance_obm: Optional[float] = None
    snis_mcse_obm: Optional[float] = None
    obm_chain_batch_sizes: list[int] = []
    obm_chain_long_run_variances: list[float] = []

    if estimate_variance:
        n_total = int(weights.size)
        mean_w = float(np.mean(weights))
        # Influence-function sequence for SNIS ratio estimator.
        h_all = weights * (tail_indicators - estimate)

        if mean_w > 0.0 and n_total >= 4:
            chain_lengths = [int(arr.size) for arr in all_q_samples]
            start = 0
            var_mean_h = 0.0
            for m in chain_lengths:
                stop = start + m
                h_chain = h_all[start:stop]
                start = stop
                sigma2_chain, b_chain = obm_long_run_variance(
                    h_chain,
                    batch_size=obm_batch_size,
                )
                obm_chain_batch_sizes.append(int(b_chain))
                obm_chain_long_run_variances.append(float(sigma2_chain))
                if np.isfinite(sigma2_chain) and m > 0:
                    # Independent chains: var(mean(H_all)) aggregates by chain lengths.
                    var_mean_h += (m * sigma2_chain) / (n_total * n_total)

            snis_variance_obm = float(var_mean_h / (mean_w * mean_w))
            snis_mcse_obm = float(np.sqrt(max(snis_variance_obm, 0.0)))

    return MCMCISResult(
        estimate=estimate,
        ess=ess,
        tilt_mode=tilt_mode,
        beta=float(beta),
        sigma_t=float(sigma_t),
        snis_variance_obm=snis_variance_obm,
        snis_mcse_obm=snis_mcse_obm,
        obm_batch_size_requested=obm_batch_size,
        obm_chain_batch_sizes=obm_chain_batch_sizes,
        obm_chain_long_run_variances=obm_chain_long_run_variances,
        n_weighted_samples=int(t_samples.size),
        tail_hits_weighted_sample=int(np.sum(tail_indicators)),
        tail_share_raw_sample=float(np.mean(tail_indicators)),
        overall_acceptance_rate=float(total_accepted / total_proposals),
        acceptance_rates=acceptance_rates,
        chain_diagnostics=chain_diagnostics,
        weight_summary=weight_summary,
        seed=seed,
        chain_seeds=chain_seeds,
        t_samples=t_samples,
        log_weights=log_weights,
        tail_indicators=tail_indicators,
        final_states=final_states,
    )
