from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.base import ExactPValueResult, ExactPValueSolver


def iter_fixed_group_labelings(n: int, n_treated: int) -> Iterator[np.ndarray]:
    for treated_idx in combinations(range(n), n_treated):
        y = np.zeros(n, dtype=np.int8)
        y[list(treated_idx)] = 1
        yield y


class BruteForceExactSolver(ExactPValueSolver):
    """
    Exact p-value via exhaustive enumeration of all constrained labelings.
    """

    def __init__(self, problem: PermutationTestProblem, max_permutations: int = 1_000_000):
        super().__init__(problem)
        self.max_permutations = int(max_permutations)

    def compute(self) -> ExactPValueResult:
        n_perm = comb(self.problem.n, self.problem.n_treated)
        if n_perm > self.max_permutations:
            raise ValueError(
                f"Brute force requires {n_perm} permutations, which exceeds "
                f"max_permutations={self.max_permutations}. "
                "Increase the limit or use a sampling method."
            )

        tail_hits = 0
        for y in iter_fixed_group_labelings(self.problem.n, self.problem.n_treated):
            if self.problem.is_in_tail_y(y):
                tail_hits += 1
        p_value = tail_hits / n_perm
        return ExactPValueResult(
            p_value=float(p_value),
            tail_hits=int(tail_hits),
            n_permutations=int(n_perm),
            tail=self.problem.tail,
            t_obs=float(self.problem.t_obs),
        )


def exact_p_value_bruteforce(
    problem: PermutationTestProblem, max_permutations: int = 1_000_000
) -> ExactPValueResult:
    return BruteForceExactSolver(problem, max_permutations=max_permutations).compute()
