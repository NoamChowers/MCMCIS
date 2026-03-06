from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional

import numpy as np

from perm_pval.core.problem import PermutationTestProblem


@dataclass
class RandomSamplingResult:
    estimate: float
    standard_error: float
    n_samples: int
    tail_hits: int
    ci_low: float
    ci_high: float
    confidence_level: float
    seed: Optional[int]


def wilson_interval(successes: int, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")
    p = successes / n
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def run_random_sampling(
    problem: PermutationTestProblem,
    n_samples: int,
    *,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
) -> RandomSamplingResult:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(seed)

    tail_hits = 0
    for _ in range(n_samples):
        y = problem.sample_uniform_labels(rng)
        tail_hits += int(problem.is_in_tail_y(y))

    estimate = tail_hits / n_samples
    standard_error = float(np.sqrt(estimate * (1.0 - estimate) / n_samples))
    ci_low, ci_high = wilson_interval(tail_hits, n_samples, confidence_level=confidence_level)

    return RandomSamplingResult(
        estimate=float(estimate),
        standard_error=standard_error,
        n_samples=int(n_samples),
        tail_hits=int(tail_hits),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=float(confidence_level),
        seed=seed,
    )
