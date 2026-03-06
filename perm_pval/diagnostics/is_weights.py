from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ISWeightSummary:
    min_weight: float
    median_weight: float
    max_weight: float
    mean_weight: float
    cv: float


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    s1 = float(np.sum(w))
    if s1 == 0.0:
        return 0.0
    s2 = float(np.sum(w * w))
    if s2 == 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def summarize_weights(weights: np.ndarray) -> ISWeightSummary:
    w = np.asarray(weights, dtype=float)
    mean_w = float(np.mean(w))
    std_w = float(np.std(w))
    cv = float(std_w / mean_w) if mean_w > 0.0 else float("inf")
    return ISWeightSummary(
        min_weight=float(np.min(w)),
        median_weight=float(np.median(w)),
        max_weight=float(np.max(w)),
        mean_weight=mean_w,
        cv=cv,
    )
