from __future__ import annotations

import numpy as np


def visitation_frequency(visit_counts: np.ndarray) -> np.ndarray:
    visits = np.asarray(visit_counts, dtype=float)
    total = float(np.sum(visits))
    if total == 0.0:
        return np.zeros_like(visits, dtype=float)
    return visits / total


def samc_visitation_error(visit_counts: np.ndarray, target: np.ndarray) -> float:
    freq = visitation_frequency(visit_counts)
    target_arr = np.asarray(target, dtype=float)
    if target_arr.shape != freq.shape:
        raise ValueError("target and visit_counts must have matching shapes.")
    return float(np.sum(np.abs(freq - target_arr)))
