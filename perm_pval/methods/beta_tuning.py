from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

import numpy as np

from perm_pval.core.proposals import n_swap_pairs_from_fraction, propose_localized_swaps
from perm_pval.core.problem import PermutationTestProblem
from perm_pval.methods.mcmc_is import right_tail_deficit_scaled


def estimate_scale_T(pilot_T: np.ndarray, method: str = "sd") -> float:
    """
    Estimate scale for hinge shortfall normalization.

    Parameters
    ----------
    pilot_T
        Pilot statistic values sampled under base f.
    method
        "sd" (default) or robust "mad" (scaled by 1.4826).
    """
    t = np.asarray(pilot_T, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("pilot_T must be a 1D array with at least two samples.")

    if method == "sd":
        sigma = float(np.std(t, ddof=1))
    elif method == "mad":
        med = float(np.median(t))
        mad = float(np.median(np.abs(t - med)))
        sigma = 1.4826 * mad
    else:
        raise ValueError("method must be one of {'sd', 'mad'}.")

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Estimated sigma_T is non-positive; cannot scale hinge shortfall.")
    return sigma


def _scaled_shortfall(pilot_T: np.ndarray, T_obs: float, sigma_T: float) -> np.ndarray:
    t = np.asarray(pilot_T, dtype=float)
    if sigma_T <= 0.0 or not np.isfinite(sigma_T):
        raise ValueError("sigma_T must be finite and positive.")
    return np.maximum((T_obs - t) / sigma_T, 0.0)


def _z_hat_from_shortfall(shortfall: np.ndarray, beta: float) -> float:
    s = np.asarray(shortfall, dtype=float)
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    return float(np.mean(np.exp(-beta * s)))


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
    """
    Initialize beta via iid-pilot Laplace transform matching.

    Design identity:
        q_beta ≈ p0 / Z(beta),   Z(beta) = E_f[exp(-beta S_scaled)].
    """
    if not (p0 > 0.0 and np.isfinite(p0)):
        raise ValueError("p0 must be finite and positive.")
    if not (q_target > 0.0 and np.isfinite(q_target)):
        raise ValueError("q_target must be finite and positive.")
    if not (beta_max > 0.0 and np.isfinite(beta_max)):
        raise ValueError("beta_max must be finite and positive.")
    if not (tol > 0.0 and np.isfinite(tol)):
        raise ValueError("tol must be finite and positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    s = _scaled_shortfall(pilot_T=pilot_T, T_obs=T_obs, sigma_T=sigma_T)

    Z_target = p0 / q_target
    if Z_target >= 1.0:
        return 0.0
    if Z_target <= 0.0:
        raise ValueError("Z_target = p0/q_target must be strictly positive.")

    lo = 0.0
    hi = 1.0
    z_hi = _z_hat_from_shortfall(s, hi)
    while z_hi > Z_target and hi < beta_max:
        hi *= 2.0
        z_hi = _z_hat_from_shortfall(s, hi)

    if hi >= beta_max and z_hi > Z_target:
        warnings.warn(
            "init_beta_from_iid_pilot could not fully bracket target within beta_max; "
            "continuing bisection on [0, beta_max].",
            RuntimeWarning,
        )
        hi = float(beta_max)

    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        z_mid = _z_hat_from_shortfall(s, mid)
        rel_err = abs(z_mid - Z_target) / Z_target
        if rel_err < tol:
            break
        if z_mid > Z_target:
            lo = mid
        else:
            hi = mid
    return float(mid)


def tune_beta_to_target_q(
    run_short_chain_fn: Callable[[float, Any, int, int], dict[str, Any]],
    init_state: Any,
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
) -> dict[str, Any]:
    """
    Tune beta to match q_hat(beta) ≈ q_target using bracket + bisection.
    """
    if beta0 < 0.0:
        raise ValueError("beta0 must be non-negative.")
    if not (q_target > 0.0 and np.isfinite(q_target)):
        raise ValueError("q_target must be finite and positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if burn_in < 0 or burn_in >= n_steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps.")
    if bracket_factor <= 1.0:
        raise ValueError("bracket_factor must be > 1.")
    if not (tol_rel > 0.0 and np.isfinite(tol_rel)):
        raise ValueError("tol_rel must be finite and positive.")
    if tol_abs is not None and (tol_abs <= 0.0 or not np.isfinite(tol_abs)):
        raise ValueError("tol_abs must be finite and positive when provided.")
    if max_bracket_iter <= 0 or max_bisect_iter <= 0:
        raise ValueError("max_bracket_iter and max_bisect_iter must be positive.")
    if replicate <= 0:
        raise ValueError("replicate must be positive.")

    tol_used = float(tol_abs) if tol_abs is not None else float(tol_rel * q_target)
    history: list[dict[str, Any]] = []

    best_beta = float(beta0)
    best_q = float("nan")
    best_accept = float("nan")
    best_diff = float("inf")

    def _warn_accept(beta: float, accept_rate: float, stage: str) -> None:
        if not np.isfinite(accept_rate):
            return
        if accept_rate < 0.01:
            warnings.warn(
                f"Very low acceptance during beta tuning ({accept_rate:.4f}) at beta={beta:.6g} [{stage}].",
                RuntimeWarning,
            )
        if accept_rate > 0.99:
            warnings.warn(
                f"Very high acceptance during beta tuning ({accept_rate:.4f}) at beta={beta:.6g} [{stage}].",
                RuntimeWarning,
            )

    def _record(
        stage: str,
        iteration: int,
        beta: float,
        q_hat: float,
        accept_rate: float,
    ) -> None:
        nonlocal best_beta, best_q, best_accept, best_diff
        rec = {
            "stage": stage,
            "iter": int(iteration),
            "beta": float(beta),
            "q_hat": float(q_hat),
            "accept_rate": float(accept_rate),
        }
        history.append(rec)
        _warn_accept(beta, accept_rate, stage=stage)

        if np.isfinite(q_hat):
            diff = abs(q_hat - q_target)
            if diff < best_diff:
                best_diff = diff
                best_beta = float(beta)
                best_q = float(q_hat)
                best_accept = float(accept_rate)

    def _eval_once(beta: float, state: Any, stage: str, iteration: int) -> tuple[float, Any, float]:
        out = run_short_chain_fn(beta, state, n_steps, burn_in)
        q_hat = float(out["q_hat"])
        last_state = out.get("last_state", state)
        accept_rate = float(out.get("accept_rate", float("nan")))
        _record(stage=stage, iteration=iteration, beta=beta, q_hat=q_hat, accept_rate=accept_rate)
        return q_hat, last_state, accept_rate

    def _eval_replicates(
        beta: float,
        state: Any,
        stage: str,
        iteration: int,
    ) -> tuple[float, Any, float]:
        q_vals = []
        acc_vals = []
        state_curr = state
        for r in range(replicate):
            q_hat_r, last_state_r, acc_r = _eval_once(
                beta=beta,
                state=state_curr if reuse_state else init_state,
                stage=f"{stage}:rep{r+1}",
                iteration=iteration,
            )
            q_vals.append(q_hat_r)
            acc_vals.append(acc_r)
            if reuse_state:
                state_curr = last_state_r
        q_mean = float(np.mean(q_vals))
        acc_mean = float(np.mean(acc_vals))
        _record(stage=stage, iteration=iteration, beta=beta, q_hat=q_mean, accept_rate=acc_mean)
        return q_mean, state_curr, acc_mean

    # 1) Evaluate at beta0.
    q0, state0, _ = _eval_once(beta0, init_state, stage="init", iteration=0)
    if abs(q0 - q_target) <= tol_used:
        return {
            "beta_hat": float(beta0),
            "beta_L": float(beta0),
            "beta_U": float(beta0),
            "q_hat": float(q0),
            "q_L": float(q0),
            "q_U": float(q0),
            "history": history,
            "tol_used": tol_used,
            "bracket_succeeded": True,
            "best_beta_so_far": best_beta,
            "best_q_so_far": best_q,
            "best_accept_rate_so_far": best_accept,
        }

    beta_L: float
    beta_U: float
    q_L: float
    q_U: float
    bracket_succeeded = False

    # 2) Bracketing phase.
    if q0 < q_target:
        beta_L = float(beta0)
        q_L = float(q0)
        beta_U = float(beta0)
        q_U = float(q0)
        state_curr = state0
        for j in range(1, max_bracket_iter + 1):
            beta_U = 1.0 if beta_U == 0.0 else beta_U * bracket_factor
            q_new, state_new, _ = _eval_once(
                beta=beta_U,
                state=state_curr if reuse_state else init_state,
                stage="bracket_up",
                iteration=j,
            )
            q_U = q_new
            if reuse_state:
                state_curr = state_new
            if q_U >= q_target:
                bracket_succeeded = True
                break
            beta_L = beta_U
            q_L = q_U
    else:
        beta_U = float(beta0)
        q_U = float(q0)
        beta_L = float(beta0)
        q_L = float(q0)
        state_curr = state0
        for j in range(1, max_bracket_iter + 1):
            beta_candidate = beta_U / bracket_factor
            if beta_candidate < 0.0:
                beta_candidate = 0.0
            # Ensure we explicitly evaluate beta=0 in finite time when halving down.
            if beta_candidate <= np.finfo(float).eps * max(1.0, beta_U):
                beta_candidate = 0.0
            if j == max_bracket_iter:
                beta_candidate = 0.0
            q_new, state_new, _ = _eval_once(
                beta=beta_candidate,
                state=state_curr if reuse_state else init_state,
                stage="bracket_down",
                iteration=j,
            )
            beta_L = beta_candidate
            q_L = q_new
            if reuse_state:
                state_curr = state_new
            if q_L <= q_target:
                bracket_succeeded = True
                break
            beta_U = beta_candidate
            q_U = q_new
            if beta_U == 0.0:
                break

    if not bracket_succeeded:
        warnings.warn(
            "Could not bracket q_target within max_bracket_iter. Returning best-so-far beta.",
            RuntimeWarning,
        )
        return {
            "beta_hat": float(best_beta),
            "beta_L": float(beta_L),
            "beta_U": float(beta_U),
            "q_hat": float(best_q),
            "q_L": float(q_L),
            "q_U": float(q_U),
            "history": history,
            "tol_used": tol_used,
            "bracket_succeeded": False,
            "best_beta_so_far": best_beta,
            "best_q_so_far": best_q,
            "best_accept_rate_so_far": best_accept,
        }

    # 3) Bisection phase.
    beta_mid = 0.5 * (beta_L + beta_U)
    q_mid = float("nan")
    state_mid = state0
    for k in range(1, max_bisect_iter + 1):
        beta_mid = 0.5 * (beta_L + beta_U)
        q_mid, state_mid, _ = _eval_replicates(
            beta=beta_mid,
            state=state_mid if reuse_state else init_state,
            stage="bisect",
            iteration=k,
        )
        if q_mid < q_target:
            beta_L = beta_mid
            q_L = q_mid
        else:
            beta_U = beta_mid
            q_U = q_mid
        if abs(q_mid - q_target) <= tol_used:
            break

    return {
        "beta_hat": float(beta_mid),
        "beta_L": float(beta_L),
        "beta_U": float(beta_U),
        "q_hat": float(q_mid),
        "q_L": float(q_L),
        "q_U": float(q_U),
        "history": history,
        "tol_used": tol_used,
        "bracket_succeeded": True,
        "best_beta_so_far": best_beta,
        "best_q_so_far": best_q,
        "best_accept_rate_so_far": best_accept,
    }


def iid_pilot_statistics(
    problem: PermutationTestProblem,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw iid samples under base f (uniform valid permutations) and return T values.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(seed)
    out = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        y = problem.sample_uniform_labels(rng)
        out[i] = problem.compute_stat(y)
    return out


def make_short_chain_q_runner(
    problem: PermutationTestProblem,
    sigma_T: float,
    *,
    thin: int = 1,
    proposal_fraction: float = 0.075,
    proposal_swaps: int | None = None,
    seed: Optional[int] = None,
) -> Callable[[float, Any, int, int], dict[str, Any]]:
    """
    Build a deterministic callback compatible with tune_beta_to_target_q.
    """
    if problem.tail != "right":
        raise NotImplementedError("Current short-chain beta tuning helper assumes right-tail tests.")
    if sigma_T <= 0.0 or not np.isfinite(sigma_T):
        raise ValueError("sigma_T must be finite and positive.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if proposal_fraction <= 0.0:
        raise ValueError("proposal_fraction must be positive.")
    if proposal_swaps is not None and proposal_swaps <= 0:
        raise ValueError("proposal_swaps must be a positive integer when provided.")

    if proposal_swaps is not None:
        n_swap_pairs = int(proposal_swaps)
    else:
        n_swap_pairs = n_swap_pairs_from_fraction(
            problem.n_treated,
            problem.n_control,
            proposal_fraction=proposal_fraction,
        )

    seed_seq = np.random.SeedSequence(seed)

    def _runner(beta: float, init_state: Any, n_steps: int, burn_in: int) -> dict[str, Any]:
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if burn_in < 0 or burn_in >= n_steps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps.")

        ss = seed_seq.spawn(1)[0]
        rng = np.random.default_rng(ss)

        if init_state is None:
            y = problem.sample_uniform_labels(rng)
        else:
            y = problem.validate_labels(np.asarray(init_state, dtype=np.int8)).copy()

        t_cur = problem.compute_stat(y)
        s_cur = right_tail_deficit_scaled(t_cur, problem.t_obs, sigma_T)

        accepted = 0
        proposals = 0
        tail_hits = 0
        n_kept = 0

        for step in range(n_steps):
            y_prop = propose_localized_swaps(y, rng, n_swap_pairs=n_swap_pairs)
            t_prop = problem.compute_stat(y_prop)
            s_prop = right_tail_deficit_scaled(t_prop, problem.t_obs, sigma_T)

            proposals += 1
            log_alpha = -beta * (s_prop - s_cur)
            if log_alpha >= 0.0 or np.log(rng.random()) < log_alpha:
                y = y_prop
                t_cur = t_prop
                s_cur = s_prop
                accepted += 1

            if step >= burn_in and ((step - burn_in) % thin == 0):
                n_kept += 1
                tail_hits += int(problem.is_in_tail(t_cur))

        q_hat = float(tail_hits / n_kept) if n_kept > 0 else float("nan")
        accept_rate = float(accepted / proposals) if proposals > 0 else float("nan")
        return {
            "q_hat": q_hat,
            "last_state": y.copy(),
            "accept_rate": accept_rate,
        }

    return _runner
