from __future__ import annotations

import warnings

import numpy as np

from jasa_mcmcis.problem import PermutationTestProblem


def iid_pilot_statistics(
    problem: PermutationTestProblem,
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """Draw iid permutations under the uniform null and return statistic values."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(seed)
    out = np.empty(int(n_samples), dtype=float)
    for i in range(int(n_samples)):
        out[i] = problem.compute_stat(problem.sample_uniform_labels(rng))
    return out


def estimate_scale_T(pilot_T: np.ndarray, method: str = "sd") -> float:
    """Estimate the scale used to normalize the smooth-hinge shortfall."""
    t = np.asarray(pilot_T, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("pilot_T must be a one-dimensional array with at least two values.")
    if method == "sd":
        scale = float(np.std(t, ddof=1))
    elif method == "mad":
        med = float(np.median(t))
        scale = float(1.4826 * np.median(np.abs(t - med)))
    else:
        raise ValueError("method must be 'sd' or 'mad'.")
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("estimated scale is not finite and positive.")
    return scale


def _scaled_shortfall(
    pilot_T: np.ndarray,
    t_obs: float,
    sigma_t: float,
    tail: str = "right",
) -> np.ndarray:
    t = np.asarray(pilot_T, dtype=float)
    if sigma_t <= 0.0 or not np.isfinite(sigma_t):
        raise ValueError("sigma_t must be finite and positive.")
    if tail == "right":
        values = t
        threshold = float(t_obs)
    elif tail == "left":
        values = -t
        threshold = -float(t_obs)
    elif tail == "two-sided":
        values = np.abs(t)
        threshold = abs(float(t_obs))
    else:
        raise ValueError("tail must be 'right', 'left', or 'two-sided'.")
    return np.maximum((threshold - values) / sigma_t, 0.0)


def init_beta_from_iid_pilot(
    pilot_T: np.ndarray,
    t_obs: float,
    sigma_t: float,
    p0_reference: float,
    q_target: float,
    *,
    tail: str = "right",
    beta_max: float = 1e6,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> float:
    """
    Initialize the MCMC-IS tilt by matching an iid-pilot Laplace transform.

    The identity used is ``q_beta ~= p0 / Z(beta)`` with
    ``Z(beta) = E_f exp(-beta * shortfall)``. In the article simulations,
    ``p0_reference`` is the known significance threshold for simple MCMC-IS or
    the exact p-value for oracle comparisons.
    """
    if p0_reference <= 0.0 or not np.isfinite(p0_reference):
        raise ValueError("p0_reference must be finite and positive.")
    if q_target <= 0.0 or not np.isfinite(q_target):
        raise ValueError("q_target must be finite and positive.")
    if beta_max <= 0.0 or not np.isfinite(beta_max):
        raise ValueError("beta_max must be finite and positive.")
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    shortfall = _scaled_shortfall(pilot_T, t_obs, sigma_t, tail=tail)
    z_target = float(p0_reference / q_target)
    if z_target >= 1.0:
        return 0.0
    if z_target <= 0.0:
        raise ValueError("p0_reference / q_target must be positive.")

    def z_hat(beta: float) -> float:
        return float(np.mean(np.exp(-float(beta) * shortfall)))

    lo = 0.0
    hi = 1.0
    z_hi = z_hat(hi)
    while z_hi > z_target and hi < beta_max:
        hi *= 2.0
        z_hi = z_hat(hi)
    if hi >= beta_max and z_hi > z_target:
        warnings.warn(
            "Could not fully bracket the requested beta; bisecting on [0, beta_max].",
            RuntimeWarning,
        )
        hi = float(beta_max)

    mid = 0.5 * (lo + hi)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        z_mid = z_hat(mid)
        rel_err = abs(z_mid - z_target) / z_target
        if rel_err < tol:
            break
        if z_mid > z_target:
            lo = mid
        else:
            hi = mid
    return float(mid)
