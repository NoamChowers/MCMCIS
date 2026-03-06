from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import numpy as np

from perm_pval.core.proposals import apply_swap, random_swap_move

Tail = Literal["right", "left", "two-sided"]


@dataclass(frozen=True)
class FixedGroupConstraint:
    n_treated: int
    n_control: int


@dataclass
class PermutationTestProblem:
    """
    Central representation of a permutation-test instance.

    Assumptions in this implementation:
    - Labels are binary: 1 = treated, 0 = control.
    - Group sizes are fixed to those observed in y_obs.
    - two-sided tail convention uses |T| >= |t_obs|.
    """

    X: np.ndarray
    y_obs: np.ndarray
    statistic: Callable[[np.ndarray, np.ndarray], float]
    tail: Tail = "right"
    constraints: Optional[FixedGroupConstraint] = None
    t_obs: float = field(init=False)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X)
        self.y_obs = np.asarray(self.y_obs, dtype=np.int8)
        if self.X.shape[0] != self.y_obs.shape[0]:
            raise ValueError("X and y_obs must agree on first dimension length.")
        if self.tail not in ("right", "left", "two-sided"):
            raise ValueError("tail must be one of {'right', 'left', 'two-sided'}.")
        if not np.all((self.y_obs == 0) | (self.y_obs == 1)):
            raise ValueError("y_obs must contain only 0/1 labels.")

        n_treated = int(self.y_obs.sum())
        n_control = int(self.y_obs.size - n_treated)
        if n_treated == 0 or n_control == 0:
            raise ValueError("Both groups must be non-empty.")

        if self.constraints is None:
            self.constraints = FixedGroupConstraint(n_treated=n_treated, n_control=n_control)
        elif (
            self.constraints.n_treated != n_treated
            or self.constraints.n_control != n_control
        ):
            raise ValueError("Provided constraints do not match y_obs group counts.")

        self.t_obs = self.compute_stat(self.y_obs)

    @property
    def n(self) -> int:
        return int(self.y_obs.size)

    @property
    def n_treated(self) -> int:
        return int(self.constraints.n_treated)

    @property
    def n_control(self) -> int:
        return int(self.constraints.n_control)

    def validate_labels(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.int8)
        if y_arr.shape != self.y_obs.shape:
            raise ValueError("Label vector shape mismatch.")
        if not np.all((y_arr == 0) | (y_arr == 1)):
            raise ValueError("Labels must be binary (0/1).")
        if int(y_arr.sum()) != self.n_treated:
            raise ValueError("Labels violate fixed group-size constraints.")
        return y_arr

    def compute_stat(self, y: np.ndarray) -> float:
        y_arr = self.validate_labels(y)
        return float(self.statistic(self.X, y_arr))

    def is_in_tail(self, t: float) -> bool:
        if self.tail == "right":
            return bool(t >= self.t_obs)
        if self.tail == "left":
            return bool(t <= self.t_obs)
        return bool(abs(t) >= abs(self.t_obs))

    def is_in_tail_y(self, y: np.ndarray) -> bool:
        return self.is_in_tail(self.compute_stat(y))

    def sample_uniform_labels(self, rng: np.random.Generator) -> np.ndarray:
        y = np.zeros(self.n, dtype=np.int8)
        treated_idx = rng.choice(self.n, size=self.n_treated, replace=False)
        y[treated_idx] = 1
        return y

    def propose_local_move(self, y: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
        y_arr = self.validate_labels(y)
        return random_swap_move(y_arr, rng)

    def apply_move(
        self, y: np.ndarray, move: Tuple[int, int], *, in_place: bool = False
    ) -> np.ndarray:
        y_arr = self.validate_labels(y)
        return apply_swap(y_arr, move, in_place=in_place)

    def copy_with_y(self, y: np.ndarray) -> "PermutationTestProblem":
        y_arr = self.validate_labels(y)
        return PermutationTestProblem(
            X=self.X,
            y_obs=y_arr.copy(),
            statistic=self.statistic,
            tail=self.tail,
            constraints=FixedGroupConstraint(
                n_treated=self.n_treated,
                n_control=self.n_control,
            ),
        )
