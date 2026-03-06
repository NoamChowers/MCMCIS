from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)
    if max_lag is None:
        max_lag = min(n - 1, 100)
    max_lag = max(1, min(max_lag, n - 1))

    xc = x_arr - np.mean(x_arr)
    var = float(np.dot(xc, xc) / n)
    if var == 0.0:
        return np.ones(max_lag + 1, dtype=float)

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(xc[:-lag], xc[lag:]) / (n - lag))
        acf[lag] = cov / var
    return acf


def integrated_autocorrelation_time(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    Initial positive-sequence estimator:
    tau_int = 1 + 2 * sum_{k>=1} rho_k, truncated at first non-positive rho_k.
    """
    acf = autocorrelation(x, max_lag=max_lag)
    if acf.size == 0:
        return float("nan")
    tau = 1.0
    for rho in acf[1:]:
        if rho <= 0.0:
            break
        tau += 2.0 * float(rho)
    return float(max(tau, 1.0))


def default_obm_batch_size(n: int) -> int:
    """
    Standard default for OBM: floor(sqrt(n)), clipped into [2, n-1].
    """
    if n < 4:
        raise ValueError("OBM requires at least 4 samples.")
    b = int(np.floor(np.sqrt(n)))
    b = max(2, min(b, n - 1))
    return b


def obm_long_run_variance(x: np.ndarray, batch_size: int | None = None) -> tuple[float, int]:
    """
    Overlapping batch means estimator for long-run variance sigma^2 in
    sqrt(n) * (mean(x) - E[x]) -> N(0, sigma^2).

    Returns:
    - sigma2_hat: estimated long-run variance coefficient.
    - b: batch size used.
    """
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.size
    if n < 4:
        return float("nan"), 0

    if batch_size is None:
        b = default_obm_batch_size(n)
    else:
        b = int(batch_size)
    if b < 2 or b >= n:
        raise ValueError(f"Invalid OBM batch_size={b} for n={n}. Must satisfy 2 <= b < n.")

    mean_all = float(np.mean(x_arr))
    csum = np.empty(n + 1, dtype=float)
    csum[0] = 0.0
    csum[1:] = np.cumsum(x_arr)
    n_batches = n - b + 1
    batch_means = (csum[b:] - csum[:-b]) / b

    ss = float(np.sum((batch_means - mean_all) ** 2))
    denom = (n - b) * n_batches
    if denom <= 0:
        return float("nan"), b
    sigma2_hat = (n * b / denom) * ss
    return float(max(sigma2_hat, 0.0)), b


def obm_variance_of_mean(x: np.ndarray, batch_size: int | None = None) -> tuple[float, float, int]:
    """
    Estimate var(mean(x)) and MCSE via OBM.

    Returns:
    - var_mean_hat
    - mcse_hat
    - batch size used
    """
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.size
    sigma2_hat, b = obm_long_run_variance(x_arr, batch_size=batch_size)
    if not np.isfinite(sigma2_hat) or n <= 0:
        return float("nan"), float("nan"), b
    var_mean_hat = sigma2_hat / n
    return float(max(var_mean_hat, 0.0)), float(np.sqrt(max(var_mean_hat, 0.0))), b
