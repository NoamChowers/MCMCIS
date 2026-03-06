from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from perm_pval.core.problem import PermutationTestProblem


@dataclass
class ParallelTemperingResult:
    betas: np.ndarray
    swap_acceptance_rates: np.ndarray
    seed: Optional[int]


def run_parallel_tempering(
    problem: PermutationTestProblem,
    *,
    betas: np.ndarray,
    n_steps: int,
    seed: Optional[int] = None,
) -> ParallelTemperingResult:
    """
    Placeholder for future implementation.

    The current milestone focuses on a robust single-temperature MCMC-IS path.
    """
    raise NotImplementedError(
        "parallel tempering is not implemented in this milestone. "
        "Use methods.mcmc_is.run_mcmc_is for current tilted sampling."
    )
