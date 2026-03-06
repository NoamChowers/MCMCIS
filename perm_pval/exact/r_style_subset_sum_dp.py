from __future__ import annotations

from math import comb

import numpy as np


def perm_exact_pval_diff_r_style(x: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    """
    Direct Python translation of the R routine `perm_exact_pval_diff`.

    Given integer, non-negative arrays X and Y, this computes the exact
    permutation p-value:

        p = P_f(S <= sum(X)),
        S = sum of n1 sampled entries from XY = concat(X, Y),
        n1 = len(X),

    under uniform sampling over all choose(N, n1) fixed-size labelings.

    Returns
    -------
    (p_value, tail_hits, n_permutations)
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("x and y must both be non-empty.")

    # Match the R table indexing assumptions exactly: integer, non-negative values.
    if not np.allclose(x_arr, np.round(x_arr), atol=0.0, rtol=0.0):
        raise ValueError("x must contain integer values for this DP translation.")
    if not np.allclose(y_arr, np.round(y_arr), atol=0.0, rtol=0.0):
        raise ValueError("y must contain integer values for this DP translation.")

    x_int = np.round(x_arr).astype(np.int64)
    y_int = np.round(y_arr).astype(np.int64)
    if np.any(x_int < 0) or np.any(y_int < 0):
        raise ValueError("x and y must be non-negative for this DP translation.")

    xy = np.concatenate([x_int, y_int])
    n = int(xy.size)
    n1 = int(x_int.size)
    s1 = int(np.sum(x_int))

    # Overflow-safe Python-int implementation of the same recursion.
    # dp[j][s] = count for subset size j and sum s after processing current prefix.
    dp_prev = [[0] * (s1 + 1) for _ in range(n1 + 1)]
    dp_prev[0][0] = 1
    if xy[0] <= s1:
        dp_prev[1][int(xy[0])] = 1

    for i in range(1, n):
        xi = int(xy[i])
        dp_cur = [row[:] for row in dp_prev]  # rtab[i,,] += rtab[i-1,,]
        # Keep strict '>' to mirror the R condition:
        # if (sl - XY[i] > 0) { ... }
        if s1 - xi > 0:
            m = min(n1, i + 1)
            for j in range(1, m + 1):
                src = dp_prev[j - 1]
                dst = dp_cur[j]
                for s in range(xi, s1 + 1):
                    dst[s] += src[s - xi]
        dp_prev = dp_cur

    tail_hits = int(sum(dp_prev[n1]))
    n_perm = int(comb(n, n1))
    p_value = float(tail_hits / n_perm)
    return p_value, tail_hits, n_perm
