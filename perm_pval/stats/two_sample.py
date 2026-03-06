from __future__ import annotations

import numpy as np


def _as_1d_numeric_x(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("Two-sample scalar statistics currently require 1D X.")
    return x_arr


def _split_groups(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = _as_1d_numeric_x(x)
    y_arr = np.asarray(y, dtype=np.int8)
    mask = y_arr == 1
    x1 = x_arr[mask]
    x0 = x_arr[~mask]
    if x1.size == 0 or x0.size == 0:
        raise ValueError("Both groups must contain at least one sample.")
    return x1, x0


def difference_in_means(x: np.ndarray, y: np.ndarray) -> float:
    """Mean(x | y=1) - Mean(x | y=0)."""
    x1, x0 = _split_groups(x, y)
    return float(np.mean(x1) - np.mean(x0))


def t_statistic_pooled(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pooled two-sample t-statistic with equal-variance assumption.
    """
    x1, x0 = _split_groups(x, y)
    n1 = x1.size
    n0 = x0.size
    m1 = float(np.mean(x1))
    m0 = float(np.mean(x0))
    if n1 < 2 or n0 < 2:
        return float("nan")
    v1 = float(np.var(x1, ddof=1))
    v0 = float(np.var(x0, ddof=1))
    pooled_var = ((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)
    denom = np.sqrt(pooled_var * (1.0 / n1 + 1.0 / n0))
    diff = m1 - m0
    if denom == 0.0:
        if diff > 0:
            return float("inf")
        if diff < 0:
            return float("-inf")
        return 0.0
    return float(diff / denom)


def t_statistic_welch(x: np.ndarray, y: np.ndarray) -> float:
    """
    Welch two-sample t-statistic.
    """
    x1, x0 = _split_groups(x, y)
    n1 = x1.size
    n0 = x0.size
    m1 = float(np.mean(x1))
    m0 = float(np.mean(x0))
    if n1 < 2 or n0 < 2:
        return float("nan")
    v1 = float(np.var(x1, ddof=1))
    v0 = float(np.var(x0, ddof=1))
    denom = np.sqrt(v1 / n1 + v0 / n0)
    diff = m1 - m0
    if denom == 0.0:
        if diff > 0:
            return float("inf")
        if diff < 0:
            return float("-inf")
        return 0.0
    return float(diff / denom)
