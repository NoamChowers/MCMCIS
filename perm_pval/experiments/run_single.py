from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.experiments.config import (
    ExactConfig,
    ExperimentConfig,
    MCMCISConfig,
    RandomSamplingConfig,
    SimulationConfig,
)
from perm_pval.methods.mcmc_is import run_mcmc_is
from perm_pval.methods.random_sampling import run_random_sampling
from perm_pval.stats.two_sample import difference_in_means


def generate_simulation_problem(config: SimulationConfig) -> PermutationTestProblem:
    rng = np.random.default_rng(config.seed)
    if config.n_treated <= 0 or config.n_treated >= config.n:
        raise ValueError("n_treated must satisfy 0 < n_treated < n.")

    y_obs = np.zeros(config.n, dtype=np.int8)
    y_obs[: config.n_treated] = 1
    rng.shuffle(y_obs)

    x = rng.normal(loc=0.0, scale=config.sigma, size=config.n)
    x = x + config.effect_size * y_obs

    return PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail=config.tail,
    )


def run_single_experiment(config: ExperimentConfig) -> dict[str, Any]:
    problem = generate_simulation_problem(config.simulation)

    exact_result = None
    try:
        exact_result = BruteForceExactSolver(
            problem, max_permutations=config.exact.max_permutations
        ).compute()
    except ValueError:
        exact_result = None

    random_result = run_random_sampling(
        problem,
        n_samples=config.random_sampling.n_samples,
        seed=config.random_sampling.seed,
        confidence_level=config.random_sampling.confidence_level,
    )
    mcmc_result = run_mcmc_is(
        problem,
        beta=config.mcmc_is.beta,
        sigma_t=config.mcmc_is.sigma_t,
        n_steps=config.mcmc_is.n_steps,
        burn_in=config.mcmc_is.burn_in,
        thin=config.mcmc_is.thin,
        n_chains=config.mcmc_is.n_chains,
        seed=config.mcmc_is.seed,
        init=config.mcmc_is.init,
        estimate_variance=config.mcmc_is.estimate_variance,
        obm_batch_size=config.mcmc_is.obm_batch_size,
    )

    return {
        "config": asdict(config),
        "observed_statistic": float(problem.t_obs),
        "exact": None if exact_result is None else asdict(exact_result),
        "random_sampling": asdict(random_result),
        "mcmc_is": {
            "estimate": mcmc_result.estimate,
            "ess": mcmc_result.ess,
            "tilt_mode": mcmc_result.tilt_mode,
            "beta": mcmc_result.beta,
            "sigma_t": mcmc_result.sigma_t,
            "snis_variance_obm": mcmc_result.snis_variance_obm,
            "snis_mcse_obm": mcmc_result.snis_mcse_obm,
            "obm_batch_size_requested": mcmc_result.obm_batch_size_requested,
            "obm_chain_batch_sizes": mcmc_result.obm_chain_batch_sizes,
            "obm_chain_long_run_variances": mcmc_result.obm_chain_long_run_variances,
            "n_weighted_samples": mcmc_result.n_weighted_samples,
            "tail_hits_weighted_sample": mcmc_result.tail_hits_weighted_sample,
            "tail_share_raw_sample": mcmc_result.tail_share_raw_sample,
            "overall_acceptance_rate": mcmc_result.overall_acceptance_rate,
            "acceptance_rates": mcmc_result.acceptance_rates,
            "chain_diagnostics": [asdict(d) for d in mcmc_result.chain_diagnostics],
            "weight_summary": asdict(mcmc_result.weight_summary),
            "seed": mcmc_result.seed,
            "chain_seeds": mcmc_result.chain_seeds,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single perm_pval experiment.")
    parser.add_argument("--n", type=int, default=14)
    parser.add_argument("--n-treated", type=int, default=7)
    parser.add_argument("--effect-size", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--tail", type=str, default="right", choices=["right", "left", "two-sided"])
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--max-permutations", type=int, default=200_000)
    parser.add_argument("--mc-samples", type=int, default=50_000)
    parser.add_argument("--mc-seed", type=int, default=17)

    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--mcmc-sigma-t", type=float, default=1.0)
    parser.add_argument("--mcmc-steps", type=int, default=30_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=5_000)
    parser.add_argument("--mcmc-thin", type=int, default=5)
    parser.add_argument("--mcmc-chains", type=int, default=2)
    parser.add_argument("--mcmc-seed", type=int, default=37)
    parser.add_argument("--mcmc-init", type=str, default="random", choices=["random", "observed"])
    parser.add_argument("--mcmc-obm-batch-size", type=int, default=None)
    parser.add_argument(
        "--mcmc-estimate-variance",
        dest="mcmc_estimate_variance",
        action="store_true",
        help="Enable OBM variance/MCSE estimation for MCMC-IS (default).",
    )
    parser.add_argument(
        "--no-mcmc-estimate-variance",
        dest="mcmc_estimate_variance",
        action="store_false",
        help="Disable OBM variance/MCSE estimation for MCMC-IS.",
    )
    parser.set_defaults(mcmc_estimate_variance=True)

    parser.add_argument("--output", type=str, default="")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = ExperimentConfig(
        simulation=SimulationConfig(
            n=args.n,
            n_treated=args.n_treated,
            effect_size=args.effect_size,
            sigma=args.sigma,
            seed=args.seed,
            tail=args.tail,
        ),
        exact=ExactConfig(max_permutations=args.max_permutations),
        random_sampling=RandomSamplingConfig(
            n_samples=args.mc_samples,
            seed=args.mc_seed,
            confidence_level=0.95,
        ),
        mcmc_is=MCMCISConfig(
            beta=args.beta,
            sigma_t=args.mcmc_sigma_t,
            n_steps=args.mcmc_steps,
            burn_in=args.mcmc_burn_in,
            thin=args.mcmc_thin,
            n_chains=args.mcmc_chains,
            seed=args.mcmc_seed,
            init=args.mcmc_init,
            estimate_variance=args.mcmc_estimate_variance,
            obm_batch_size=args.mcmc_obm_batch_size,
        ),
    )

    results = run_single_experiment(config)
    print(json.dumps(results, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
