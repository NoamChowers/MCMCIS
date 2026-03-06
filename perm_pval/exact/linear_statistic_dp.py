from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import DefaultDict, Optional

import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.base import ExactPValueResult, ExactPValueSolver


class LinearStatisticDPSolver(ExactPValueSolver):
    """
    Exact solver for linear statistics of the form:

        T(y) = offset + scale * sum_i scores_i * y_i

    with fixed number of treated labels (sum y_i = n_treated).

    The DP counts exact frequencies of the weighted subset sums.
    Scores are converted to integers via score_scale to ensure exact DP states.
    """

    def __init__(
        self,
        problem: PermutationTestProblem,
        *,
        scores: Optional[np.ndarray] = None,
        score_scale: int = 1,
        scale: float = 1.0,
        offset: float = 0.0,
        check_statistic_match: bool = True,
        atol: float = 1e-10,
    ):
        super().__init__(problem)
        if score_scale <= 0:
            raise ValueError("score_scale must be a positive integer.")
        self.score_scale = int(score_scale)
        self.scale = float(scale)
        self.offset = float(offset)
        self.check_statistic_match = check_statistic_match
        self.atol = float(atol)

        if scores is None:
            raw_scores = np.asarray(problem.X, dtype=float)
            if raw_scores.ndim != 1:
                raise ValueError("When scores is None, problem.X must be 1D.")
            self.scores = raw_scores
        else:
            self.scores = np.asarray(scores, dtype=float)

        if self.scores.ndim != 1:
            raise ValueError("scores must be a 1D array.")
        if self.scores.size != problem.n:
            raise ValueError("scores length must match problem.n.")

        scaled = self.scores * self.score_scale
        rounded = np.round(scaled)
        if not np.allclose(scaled, rounded, atol=self.atol, rtol=0.0):
            raise ValueError(
                "Scores are not integer-representable under provided score_scale within tolerance."
            )
        self.scores_int = rounded.astype(np.int64)

    @classmethod
    def from_difference_in_means(
        cls,
        problem: PermutationTestProblem,
        *,
        score_scale: int = 1,
        check_statistic_match: bool = True,
        atol: float = 1e-10,
    ) -> "LinearStatisticDPSolver":
        """
        Build an exact linear-stat solver for:
            T = mean(X|y=1) - mean(X|y=0)
        under fixed group sizes.
        """
        x = np.asarray(problem.X, dtype=float)
        if x.ndim != 1:
            raise ValueError("from_difference_in_means currently requires 1D X.")
        n1 = problem.n_treated
        n0 = problem.n_control
        # T = (1/n1 + 1/n0) * sum(x_i y_i) - (1/n0) * sum(x_i)
        scale = (1.0 / n1) + (1.0 / n0)
        offset = -(1.0 / n0) * float(np.sum(x))
        return cls(
            problem,
            scores=x,
            score_scale=score_scale,
            scale=scale,
            offset=offset,
            check_statistic_match=check_statistic_match,
            atol=atol,
        )

    def _distribution_weighted_sum(self, weights: np.ndarray, n_treated: int) -> dict[int, int]:
        states: list[DefaultDict[int, int]] = [defaultdict(int) for _ in range(n_treated + 1)]
        states[0][0] = 1
        for w in weights.tolist():
            for k in range(n_treated, 0, -1):
                prev = states[k - 1]
                if not prev:
                    continue
                cur = states[k]
                for s_prev, count in prev.items():
                    cur[s_prev + w] += count
        return dict(states[n_treated])

    def _sum_to_stat(self, sum_int: int) -> float:
        sum_unscaled = sum_int / self.score_scale
        return float(self.offset + self.scale * sum_unscaled)

    def _is_in_tail_for_solver(self, t_val: float, t_obs_solver: float) -> bool:
        if self.problem.tail == "right":
            return bool(t_val >= t_obs_solver - self.atol)
        if self.problem.tail == "left":
            return bool(t_val <= t_obs_solver + self.atol)
        return bool(abs(t_val) >= abs(t_obs_solver) - self.atol)

    def compute(self) -> ExactPValueResult:
        n = self.problem.n
        n_treated = self.problem.n_treated
        dist = self._distribution_weighted_sum(self.scores_int, n_treated=n_treated)

        observed_sum_int = int(np.sum(self.scores_int[self.problem.y_obs == 1]))
        t_obs_solver = self._sum_to_stat(observed_sum_int)
        if self.check_statistic_match and not np.isclose(
            t_obs_solver, self.problem.t_obs, atol=self.atol, rtol=0.0
        ):
            raise ValueError(
                "Problem statistic/t_obs does not match configured linear statistic mapping. "
                f"Expected observed={t_obs_solver}, got problem.t_obs={self.problem.t_obs}."
            )

        tail_hits = 0
        for sum_value, count in dist.items():
            t_val = self._sum_to_stat(sum_value)
            if self._is_in_tail_for_solver(t_val, t_obs_solver=t_obs_solver):
                tail_hits += count

        n_perm = comb(n, n_treated)
        p_value = tail_hits / n_perm
        return ExactPValueResult(
            p_value=float(p_value),
            tail_hits=int(tail_hits),
            n_permutations=int(n_perm),
            tail=self.problem.tail,
            t_obs=float(self.problem.t_obs),
        )
