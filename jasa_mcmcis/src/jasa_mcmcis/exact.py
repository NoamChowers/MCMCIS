from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import comb
from typing import DefaultDict

import numpy as np

from jasa_mcmcis.problem import PermutationTestProblem


@dataclass(frozen=True)
class ExactPValueResult:
    p_value: float
    tail_hits: int
    n_permutations: int
    tail: str
    t_obs: float


class LinearStatisticDPSolver:
    """
    Exact DP for fixed-size permutation tests with linear statistics.

    It covers statistics of the form
    ``offset + scale * sum_i scores_i y_i`` after converting ``scores`` to
    integer states through ``score_scale``. The six bundled article scenarios
    use either ``treated_sum`` or ``difference_in_means`` and are covered by
    this solver.
    """

    def __init__(
        self,
        problem: PermutationTestProblem,
        *,
        scores: np.ndarray | None = None,
        score_scale: int = 1,
        scale: float = 1.0,
        offset: float = 0.0,
        check_statistic_match: bool = True,
        atol: float = 1e-10,
    ):
        if score_scale <= 0:
            raise ValueError("score_scale must be positive.")
        self.problem = problem
        self.score_scale = int(score_scale)
        self.scale = float(scale)
        self.offset = float(offset)
        self.check_statistic_match = bool(check_statistic_match)
        self.atol = float(atol)
        raw_scores = np.asarray(problem.x if scores is None else scores, dtype=float)
        if raw_scores.ndim != 1 or raw_scores.size != problem.n:
            raise ValueError("scores must be a one-dimensional array of length problem.n.")
        scaled = raw_scores * self.score_scale
        rounded = np.round(scaled)
        if not np.allclose(scaled, rounded, atol=self.atol, rtol=0.0):
            raise ValueError("scores are not integer-representable under score_scale.")
        self.scores_int = rounded.astype(np.int64)

    @classmethod
    def from_treated_sum(
        cls,
        problem: PermutationTestProblem,
        *,
        score_scale: int = 1,
        check_statistic_match: bool = True,
        atol: float = 1e-10,
    ) -> "LinearStatisticDPSolver":
        return cls(
            problem,
            scores=problem.x,
            score_scale=score_scale,
            scale=1.0,
            offset=0.0,
            check_statistic_match=check_statistic_match,
            atol=atol,
        )

    @classmethod
    def from_difference_in_means(
        cls,
        problem: PermutationTestProblem,
        *,
        score_scale: int = 1,
        check_statistic_match: bool = True,
        atol: float = 1e-10,
    ) -> "LinearStatisticDPSolver":
        x = np.asarray(problem.x, dtype=float)
        if x.ndim != 1:
            raise ValueError("difference-in-means DP requires a one-dimensional x array.")
        n1 = problem.n_treated
        n0 = problem.n_control
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

    @classmethod
    def from_scenario(
        cls,
        scenario,
        *,
        score_scale: int = 1,
        check_statistic_match: bool = True,
        atol: float = 1e-10,
    ) -> "LinearStatisticDPSolver":
        if scenario.statistic_name == "treated_sum":
            return cls.from_treated_sum(
                scenario.problem,
                score_scale=score_scale,
                check_statistic_match=check_statistic_match,
                atol=atol,
            )
        if scenario.statistic_name == "difference_in_means":
            return cls.from_difference_in_means(
                scenario.problem,
                score_scale=score_scale,
                check_statistic_match=check_statistic_match,
                atol=atol,
            )
        raise ValueError(f"No linear DP mapping is registered for {scenario.statistic_name!r}.")

    def _distribution(self) -> dict[int, int]:
        states: list[DefaultDict[int, int]] = [
            defaultdict(int) for _ in range(self.problem.n_treated + 1)
        ]
        states[0][0] = 1
        for weight, count in sorted(Counter(int(w) for w in self.scores_int.tolist()).items()):
            max_take = min(int(count), self.problem.n_treated)
            choose_counts = [comb(int(count), j) for j in range(max_take + 1)]
            next_states: list[DefaultDict[int, int]] = [
                defaultdict(int) for _ in range(self.problem.n_treated + 1)
            ]
            for k_previous, previous in enumerate(states):
                if not previous:
                    continue
                max_group_take = min(max_take, self.problem.n_treated - k_previous)
                for partial_sum, state_count in previous.items():
                    for group_take in range(max_group_take + 1):
                        next_states[k_previous + group_take][partial_sum + group_take * weight] += (
                            int(state_count) * choose_counts[group_take]
                        )
            states = next_states
        return dict(states[self.problem.n_treated])

    def _sum_to_stat(self, sum_int: int) -> float:
        return float(self.offset + self.scale * (int(sum_int) / self.score_scale))

    def _in_tail(self, t_value: float, t_obs_solver: float) -> bool:
        if self.problem.tail == "right":
            return bool(t_value >= t_obs_solver - self.atol)
        if self.problem.tail == "left":
            return bool(t_value <= t_obs_solver + self.atol)
        return bool(abs(t_value) >= abs(t_obs_solver) - self.atol)

    def compute(self) -> ExactPValueResult:
        dist = self._distribution()
        observed_sum = int(np.sum(self.scores_int[self.problem.y_obs == 1]))
        t_obs_solver = self._sum_to_stat(observed_sum)
        if self.check_statistic_match and not np.isclose(
            t_obs_solver,
            self.problem.t_obs,
            atol=self.atol,
            rtol=0.0,
        ):
            raise ValueError(
                "configured linear statistic does not match problem.t_obs: "
                f"{t_obs_solver} != {self.problem.t_obs}"
            )

        tail_hits = 0
        for sum_value, count in dist.items():
            if self._in_tail(self._sum_to_stat(sum_value), t_obs_solver):
                tail_hits += int(count)
        n_perm = comb(self.problem.n, self.problem.n_treated)
        return ExactPValueResult(
            p_value=float(tail_hits / n_perm),
            tail_hits=int(tail_hits),
            n_permutations=int(n_perm),
            tail=str(self.problem.tail),
            t_obs=float(self.problem.t_obs),
        )
