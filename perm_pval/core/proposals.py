from __future__ import annotations

from numbers import Integral, Real
from typing import Tuple

import numpy as np


def random_swap_move(y: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    """Choose one index from each group and return a swap move."""
    y_arr = np.asarray(y, dtype=np.int8)
    ones = np.flatnonzero(y_arr == 1)
    zeros = np.flatnonzero(y_arr == 0)
    if ones.size == 0 or zeros.size == 0:
        raise ValueError("Both groups must be non-empty to propose a swap move.")
    i = int(rng.choice(ones))
    j = int(rng.choice(zeros))
    return i, j


def apply_swap(y: np.ndarray, move: Tuple[int, int], *, in_place: bool = False) -> np.ndarray:
    """Apply swap move to labels."""
    i, j = move
    if in_place:
        out = y
    else:
        out = np.asarray(y, dtype=np.int8).copy()
    if out[i] == out[j]:
        raise ValueError("Swap move must exchange labels from different groups.")
    out[i], out[j] = out[j], out[i]
    return out


def n_swap_pairs_from_fraction(
    n_treated: int,
    n_control: int,
    proposal_fraction: float = 0.075,
) -> int:
    """
    Convert a proposal fraction to an integer number of treated/control swap pairs.
    """
    if proposal_fraction <= 0.0 or proposal_fraction > 1.0:
        raise ValueError("proposal_fraction must satisfy 0 < proposal_fraction <= 1.")
    max_pairs = min(int(n_treated), int(n_control))
    if max_pairs <= 0:
        raise ValueError("Both groups must be non-empty.")
    return max(1, int(round(proposal_fraction * max_pairs)))


def resolve_n_swap_pairs(
    n_treated: int,
    n_control: int,
    proposal_size: float | int = 0.075,
) -> int:
    """
    Resolve proposal size to an integer number of treated/control swap pairs.

    ``proposal_size`` is interpreted as:
    - ``int``: absolute number of swap pairs
    - ``float``: fraction of the smaller group size
    """
    max_pairs = min(int(n_treated), int(n_control))
    if max_pairs <= 0:
        raise ValueError("Both groups must be non-empty.")
    if isinstance(proposal_size, bool):
        raise TypeError("proposal_size must be an int swap count or float proposal fraction.")
    if isinstance(proposal_size, Integral):
        n_swap_pairs = int(proposal_size)
        if n_swap_pairs < 1 or n_swap_pairs > max_pairs:
            raise ValueError(
                "proposal_size must satisfy 1 <= proposal_size <= min group size when given as an int."
            )
        return n_swap_pairs
    if isinstance(proposal_size, Real):
        return n_swap_pairs_from_fraction(
            n_treated,
            n_control,
            proposal_fraction=float(proposal_size),
        )
    raise TypeError("proposal_size must be an int swap count or float proposal fraction.")


def propose_localized_swaps(
    y: np.ndarray,
    rng: np.random.Generator,
    n_swap_pairs: int,
) -> np.ndarray:
    """
    Propose a new labeling by swapping exactly ``n_swap_pairs`` treated/control pairs.

    The swapped treated and control indices are sampled without replacement, so
    ``n_swap_pairs`` has the same block-swap interpretation as Shuli's original
    implementation: an ``L``-swap proposal changes exactly ``L`` labels from
    each group.
    """
    if n_swap_pairs <= 0:
        raise ValueError("n_swap_pairs must be positive.")
    y_arr = np.asarray(y, dtype=np.int8)
    ones = np.flatnonzero(y_arr == 1)
    zeros = np.flatnonzero(y_arr == 0)
    if ones.size == 0 or zeros.size == 0:
        raise ValueError("Both groups must be non-empty to propose swaps.")
    max_pairs = min(ones.size, zeros.size)
    if n_swap_pairs > max_pairs:
        raise ValueError("n_swap_pairs must be no larger than the smaller group size.")

    swap_ones = rng.choice(ones, size=int(n_swap_pairs), replace=False)
    swap_zeros = rng.choice(zeros, size=int(n_swap_pairs), replace=False)
    y_prop = y_arr.copy()
    y_prop[swap_ones] = 0
    y_prop[swap_zeros] = 1
    return y_prop
