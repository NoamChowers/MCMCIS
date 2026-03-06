from __future__ import annotations

from typing import Optional

import numpy as np


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def spawn_rngs(seed: Optional[int], n_streams: int) -> list[np.random.Generator]:
    if n_streams <= 0:
        raise ValueError("n_streams must be positive.")
    seed_seq = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in seed_seq.spawn(n_streams)]
