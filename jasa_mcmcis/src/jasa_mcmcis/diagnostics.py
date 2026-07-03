from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightSummary:
    min_weight: float
    median_weight: float
    max_weight: float
    mean_weight: float
    cv: float


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative.")
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s1 <= 0.0 or s2 <= 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def summarize_weights(weights: np.ndarray) -> WeightSummary:
    w = np.asarray(weights, dtype=float)
    mean = float(np.mean(w))
    sd = float(np.std(w))
    return WeightSummary(
        min_weight=float(np.min(w)),
        median_weight=float(np.median(w)),
        max_weight=float(np.max(w)),
        mean_weight=mean,
        cv=float(sd / mean) if mean > 0.0 else float("inf"),
    )


def default_obm_batch_size(n: int) -> int:
    if n < 4:
        raise ValueError("OBM requires at least four samples.")
    return max(2, min(int(np.floor(np.sqrt(n))), n - 1))


def obm_long_run_variance(
    values: np.ndarray,
    batch_size: int | None = None,
) -> tuple[float, int]:
    """
    Overlapping batch-means estimate of the long-run variance coefficient.
    """
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    if n < 4:
        return float("nan"), 0
    b = default_obm_batch_size(n) if batch_size is None else int(batch_size)
    if b < 2 or b >= n:
        raise ValueError("batch_size must satisfy 2 <= batch_size < n.")

    centered_mean = float(np.mean(x))
    cumulative = np.empty(n + 1, dtype=float)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(x)
    batch_means = (cumulative[b:] - cumulative[:-b]) / b
    n_batches = n - b + 1
    denom = (n - b) * n_batches
    if denom <= 0:
        return float("nan"), b
    ss = float(np.sum((batch_means - centered_mean) ** 2))
    return float(max((n * b / denom) * ss, 0.0)), b


def visitation_frequency(visit_counts: np.ndarray) -> np.ndarray:
    visits = np.asarray(visit_counts, dtype=float)
    total = float(np.sum(visits))
    if total <= 0.0:
        return np.zeros_like(visits)
    return visits / total
