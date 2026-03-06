from __future__ import annotations

import numpy as np


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    i = 0
    n = x.size
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mann-Whitney U for group y=1 against y=0.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=np.int8)
    if x_arr.ndim != 1:
        raise ValueError("mann_whitney_u currently requires 1D X.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have same length.")
    n1 = int(np.sum(y_arr == 1))
    if n1 == 0 or n1 == x_arr.size:
        raise ValueError("Both groups must be non-empty.")
    ranks = _average_ranks(x_arr)
    r1 = float(np.sum(ranks[y_arr == 1]))
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1)
