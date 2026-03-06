from __future__ import annotations

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
    if proposal_fraction <= 0.0:
        raise ValueError("proposal_fraction must be positive.")
    max_pairs = min(int(n_treated), int(n_control))
    if max_pairs <= 0:
        raise ValueError("Both groups must be non-empty.")
    return max(1, int(round(proposal_fraction * max_pairs)))


def propose_localized_swaps(
    y: np.ndarray,
    rng: np.random.Generator,
    n_swap_pairs: int,
) -> np.ndarray:
    """
    Propose a new labeling by applying ``n_swap_pairs`` random treated/control swaps.
    """
    if n_swap_pairs <= 0:
        raise ValueError("n_swap_pairs must be positive.")
    y_prop = np.asarray(y, dtype=np.int8).copy()
    for _ in range(n_swap_pairs):
        move = random_swap_move(y_prop, rng)
        y_prop = apply_swap(y_prop, move, in_place=True)
    return y_prop
