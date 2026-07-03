from __future__ import annotations

import numpy as np


def _as_1d_numeric(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("This statistic requires a one-dimensional x array.")
    return x_arr


def _split_groups(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = _as_1d_numeric(x)
    y_arr = np.asarray(y, dtype=np.int8)
    treated = x_arr[y_arr == 1]
    control = x_arr[y_arr == 0]
    if treated.size == 0 or control.size == 0:
        raise ValueError("Both groups must be non-empty.")
    return treated, control


def treated_sum(x: np.ndarray, y: np.ndarray) -> float:
    """Additive score statistic: ``sum_i x_i y_i``."""
    x_arr = _as_1d_numeric(x)
    y_arr = np.asarray(y, dtype=np.int8)
    return float(np.dot(x_arr, y_arr))


def difference_in_means(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample statistic: ``mean(x | y=1) - mean(x | y=0)``."""
    treated, control = _split_groups(x, y)
    return float(np.mean(treated) - np.mean(control))


STATISTIC_REGISTRY = {
    "treated_sum": treated_sum,
    "difference_in_means": difference_in_means,
}
