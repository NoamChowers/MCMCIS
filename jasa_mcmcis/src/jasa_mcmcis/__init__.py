"""Article-facing API for rare permutation p-value estimation."""

from jasa_mcmcis.beta import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
)
from jasa_mcmcis.exact import ExactPValueResult, LinearStatisticDPSolver
from jasa_mcmcis.mcmcis import (
    MCMCISResult,
    hard_step_beta_for_target_tail_mass,
    hard_step_r_for_target_tail_mass,
    run_hard_step_mcmc_is,
    run_mcmc_is,
)
from jasa_mcmcis.problem import FixedGroupConstraint, PermutationTestProblem
from jasa_mcmcis.samc import SAMCResult, run_samc
from jasa_mcmcis.scenarios import (
    ABOVE_THRESHOLD_SCENARIO_KEYS,
    ARTICLE_SCENARIO_KEYS,
    CROSS_METHOD_SCENARIO_KEYS,
    GWAS_ABOVE_THRESHOLD_SCENARIO_KEYS,
    GWAS_NEAR_THRESHOLD_SCENARIO_KEYS,
    HEP_ABOVE_THRESHOLD_SCENARIO_KEYS,
    HEP_NEAR_THRESHOLD_SCENARIO_KEYS,
    NEAR_THRESHOLD_SCENARIO_KEYS,
    Scenario,
    available_scenarios,
    load_scenario,
    load_scenarios,
)
from jasa_mcmcis.statistics import difference_in_means, treated_sum

__all__ = [
    "ABOVE_THRESHOLD_SCENARIO_KEYS",
    "CROSS_METHOD_SCENARIO_KEYS",
    "ARTICLE_SCENARIO_KEYS",
    "ExactPValueResult",
    "FixedGroupConstraint",
    "GWAS_ABOVE_THRESHOLD_SCENARIO_KEYS",
    "GWAS_NEAR_THRESHOLD_SCENARIO_KEYS",
    "HEP_ABOVE_THRESHOLD_SCENARIO_KEYS",
    "HEP_NEAR_THRESHOLD_SCENARIO_KEYS",
    "LinearStatisticDPSolver",
    "MCMCISResult",
    "NEAR_THRESHOLD_SCENARIO_KEYS",
    "PermutationTestProblem",
    "SAMCResult",
    "Scenario",
    "available_scenarios",
    "difference_in_means",
    "estimate_scale_T",
    "hard_step_beta_for_target_tail_mass",
    "hard_step_r_for_target_tail_mass",
    "iid_pilot_statistics",
    "init_beta_from_iid_pilot",
    "load_scenario",
    "load_scenarios",
    "run_hard_step_mcmc_is",
    "run_mcmc_is",
    "run_samc",
    "treated_sum",
]
