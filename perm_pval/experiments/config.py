from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationConfig:
    n: int = 14
    n_treated: int = 7
    effect_size: float = 1.0
    sigma: float = 1.0
    seed: int = 123
    tail: str = "right"


@dataclass
class ExactConfig:
    max_permutations: int = 200_000


@dataclass
class RandomSamplingConfig:
    n_samples: int = 50_000
    seed: Optional[int] = 17
    confidence_level: float = 0.95


@dataclass
class MCMCISConfig:
    beta: float = 2.0
    sigma_t: float = 1.0
    n_steps: int = 30_000
    burn_in: int = 5_000
    thin: int = 5
    n_chains: int = 2
    seed: Optional[int] = 37
    init: str = "random"
    estimate_variance: bool = True
    obm_batch_size: Optional[int] = None


@dataclass
class ExperimentConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    exact: ExactConfig = field(default_factory=ExactConfig)
    random_sampling: RandomSamplingConfig = field(default_factory=RandomSamplingConfig)
    mcmc_is: MCMCISConfig = field(default_factory=MCMCISConfig)
