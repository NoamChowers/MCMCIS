from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def resolve_n_swap_pairs(
    n_treated: int,
    n_control: int,
    proposal_size: float | int = 0.075,
) -> int:
    """
    Convert a proposal-size specification to a number of treated/control swaps.

    Integers are interpreted as an exact number of swap pairs. Floats are
    interpreted as a fraction of the smaller group size.
    """
    max_pairs = min(int(n_treated), int(n_control))
    if max_pairs <= 0:
        raise ValueError("Both groups must be non-empty.")
    if isinstance(proposal_size, bool):
        raise TypeError("proposal_size must be an int or a float.")
    if isinstance(proposal_size, Integral):
        n_pairs = int(proposal_size)
        if not 1 <= n_pairs <= max_pairs:
            raise ValueError("integer proposal_size must be between 1 and min group size.")
        return n_pairs
    if isinstance(proposal_size, Real):
        frac = float(proposal_size)
        if not 0.0 < frac <= 1.0:
            raise ValueError("float proposal_size must satisfy 0 < proposal_size <= 1.")
        return max(1, int(round(frac * max_pairs)))
    raise TypeError("proposal_size must be an int swap count or float fraction.")


def propose_swaps(
    y: np.ndarray,
    rng: np.random.Generator,
    n_swap_pairs: int,
) -> np.ndarray:
    """Propose a label vector by swapping treated and control indices."""
    if n_swap_pairs <= 0:
        raise ValueError("n_swap_pairs must be positive.")
    y_arr = np.asarray(y, dtype=np.int8)
    treated = np.flatnonzero(y_arr == 1)
    control = np.flatnonzero(y_arr == 0)
    max_pairs = min(treated.size, control.size)
    if max_pairs <= 0:
        raise ValueError("Both groups must be non-empty.")
    if n_swap_pairs > max_pairs:
        raise ValueError("n_swap_pairs cannot exceed the smaller group size.")

    swap_treated = rng.choice(treated, size=n_swap_pairs, replace=False)
    swap_control = rng.choice(control, size=n_swap_pairs, replace=False)
    y_prop = y_arr.copy()
    y_prop[swap_treated] = 0
    y_prop[swap_control] = 1
    return y_prop
