from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import DefaultDict

import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.base import ExactPValueResult, ExactPValueSolver


class RankSumDPSolver(ExactPValueSolver):
    """
    Exact solver for rank-sum based statistics using dynamic programming.

    Supported statistic_type values:
    - "u": Mann-Whitney U for treated group (recommended)
    - "rank_sum": Wilcoxon rank sum (sum of treated ranks)

    Notes:
    - This implementation currently requires no ties in X values.
    - Labels are assumed binary with fixed treated count from the problem.
    """

    def __init__(
        self,
        problem: PermutationTestProblem,
        *,
        statistic_type: str = "u",
        check_statistic_match: bool = True,
        atol: float = 1e-10,
    ):
        super().__init__(problem)
        if statistic_type not in ("u", "rank_sum"):
            raise ValueError("statistic_type must be one of {'u', 'rank_sum'}.")
        self.statistic_type = statistic_type
        self.check_statistic_match = check_statistic_match
        self.atol = float(atol)

    def _integer_ranks_no_ties(self) -> np.ndarray:
        x = np.asarray(self.problem.X, dtype=float)
        if x.ndim != 1:
            raise ValueError("RankSumDPSolver currently requires 1D X.")
        if np.unique(x).size != x.size:
            raise ValueError(
                "RankSumDPSolver currently requires no ties in X for exact DP distribution."
            )
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(x.size, dtype=np.int64)
        ranks[order] = np.arange(1, x.size + 1, dtype=np.int64)
        return ranks

    def _distribution_rank_sum(self, ranks: np.ndarray, n_treated: int) -> dict[int, int]:
        states: list[DefaultDict[int, int]] = [defaultdict(int) for _ in range(n_treated + 1)]
        states[0][0] = 1
        for r in ranks.tolist():
            for k in range(n_treated, 0, -1):
                prev = states[k - 1]
                if not prev:
                    continue
                cur = states[k]
                for s_prev, count in prev.items():
                    cur[s_prev + r] += count
        return dict(states[n_treated])

    def _rank_sum_to_statistic(self, rank_sum: int, n_treated: int) -> float:
        if self.statistic_type == "rank_sum":
            return float(rank_sum)
        # U = R - n1(n1+1)/2
        u = rank_sum - n_treated * (n_treated + 1) / 2.0
        return float(u)

    def _is_in_tail_for_solver(self, t_val: float, t_obs_solver: float) -> bool:
        if self.problem.tail == "right":
            return bool(t_val >= t_obs_solver - self.atol)
        if self.problem.tail == "left":
            return bool(t_val <= t_obs_solver + self.atol)
        return bool(abs(t_val) >= abs(t_obs_solver) - self.atol)

    def compute(self) -> ExactPValueResult:
        n = self.problem.n
        n_treated = self.problem.n_treated
        ranks = self._integer_ranks_no_ties()
        dist = self._distribution_rank_sum(ranks, n_treated=n_treated)

        rank_sum_obs = int(np.sum(ranks[self.problem.y_obs == 1]))
        t_obs_solver = self._rank_sum_to_statistic(rank_sum_obs, n_treated=n_treated)
        if self.check_statistic_match and not np.isclose(
            t_obs_solver, self.problem.t_obs, atol=self.atol, rtol=0.0
        ):
            raise ValueError(
                "Problem statistic/t_obs does not match selected rank-sum statistic_type. "
                f"Expected observed={t_obs_solver}, got problem.t_obs={self.problem.t_obs}."
            )

        tail_hits = 0
        for rank_sum_value, count in dist.items():
            t_val = self._rank_sum_to_statistic(rank_sum_value, n_treated=n_treated)
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
