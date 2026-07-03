"""Article-facing API for rare permutation p-value estimation."""

from jasa_mcmcis.beta import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
)
from jasa_mcmcis.exact import ExactPValueResult, LinearStatisticDPSolver
from jasa_mcmcis.mcmcis import MCMCISResult, run_mcmc_is
from jasa_mcmcis.problem import FixedGroupConstraint, PermutationTestProblem
from jasa_mcmcis.samc import SAMCResult, run_samc
from jasa_mcmcis.scenarios import (
    CROSS_METHOD_SCENARIO_KEYS,
    Scenario,
    available_scenarios,
    load_scenario,
    load_scenarios,
)
from jasa_mcmcis.statistics import difference_in_means, treated_sum

__all__ = [
    "CROSS_METHOD_SCENARIO_KEYS",
    "ExactPValueResult",
    "FixedGroupConstraint",
    "LinearStatisticDPSolver",
    "MCMCISResult",
    "PermutationTestProblem",
    "SAMCResult",
    "Scenario",
    "available_scenarios",
    "difference_in_means",
    "estimate_scale_T",
    "iid_pilot_statistics",
    "init_beta_from_iid_pilot",
    "load_scenario",
    "load_scenarios",
    "run_mcmc_is",
    "run_samc",
    "treated_sum",
]
