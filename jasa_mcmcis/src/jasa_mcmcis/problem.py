from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

Tail = Literal["right", "left", "two-sided"]
Statistic = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class FixedGroupConstraint:
    """Fixed binary-label group sizes under the permutation null."""

    n_treated: int
    n_control: int


@dataclass
class PermutationTestProblem:
    """
    A fixed-size two-group permutation-test instance.

    Labels are binary: ``1`` is the treated or case group and ``0`` is the
    control group. The permutation null is uniform over all labels with the
    observed number of treated labels.
    """

    x: np.ndarray
    y_obs: np.ndarray
    statistic: Statistic
    tail: Tail = "right"
    constraints: FixedGroupConstraint | None = None
    t_obs: float = field(init=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x)
        self.y_obs = np.asarray(self.y_obs, dtype=np.int8)
        if self.x.shape[0] != self.y_obs.shape[0]:
            raise ValueError("x and y_obs must have the same first dimension.")
        if self.tail not in ("right", "left", "two-sided"):
            raise ValueError("tail must be 'right', 'left', or 'two-sided'.")
        if not np.all((self.y_obs == 0) | (self.y_obs == 1)):
            raise ValueError("y_obs must contain only 0/1 labels.")

        n_treated = int(np.sum(self.y_obs))
        n_control = int(self.y_obs.size - n_treated)
        if n_treated == 0 or n_control == 0:
            raise ValueError("Both groups must be non-empty.")

        if self.constraints is None:
            self.constraints = FixedGroupConstraint(n_treated, n_control)
        elif (
            self.constraints.n_treated != n_treated
            or self.constraints.n_control != n_control
        ):
            raise ValueError("constraints do not match y_obs group sizes.")

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
            raise ValueError("label vector shape mismatch.")
        if not np.all((y_arr == 0) | (y_arr == 1)):
            raise ValueError("labels must be binary.")
        if int(np.sum(y_arr)) != self.n_treated:
            raise ValueError("labels violate fixed group-size constraints.")
        return y_arr

    def compute_stat(self, y: np.ndarray) -> float:
        return float(self.statistic(self.x, self.validate_labels(y)))

    def tail_value(self, t: float) -> float:
        """Map a statistic value to a right-tail scale."""
        if self.tail == "right":
            return float(t)
        if self.tail == "left":
            return float(-t)
        return float(abs(t))

    @property
    def tail_threshold(self) -> float:
        return self.tail_value(self.t_obs)

    def is_in_tail(self, t: float) -> bool:
        return bool(self.tail_value(t) >= self.tail_threshold)

    def is_in_tail_y(self, y: np.ndarray) -> bool:
        return self.is_in_tail(self.compute_stat(y))

    def sample_uniform_labels(self, rng: np.random.Generator) -> np.ndarray:
        y = np.zeros(self.n, dtype=np.int8)
        treated = rng.choice(self.n, size=self.n_treated, replace=False)
        y[treated] = 1
        return y

    @property
    def X(self) -> np.ndarray:
        """Compatibility alias for older code using an uppercase covariate name."""
        return self.x
