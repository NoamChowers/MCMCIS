from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from perm_pval.core.problem import PermutationTestProblem


@dataclass
class ExactPValueResult:
    p_value: float
    tail_hits: int
    n_permutations: int
    tail: str
    t_obs: float


class ExactPValueSolver(ABC):
    def __init__(self, problem: PermutationTestProblem):
        self.problem = problem

    @abstractmethod
    def compute(self) -> ExactPValueResult:
        raise NotImplementedError
