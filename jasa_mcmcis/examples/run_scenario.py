from __future__ import annotations

from jasa_mcmcis import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    load_scenario,
    run_mcmc_is,
    run_samc,
)


def proposal_size_for_scenario(key: str) -> int:
    if key.startswith("gwas_"):
        return 2
    if key.startswith("poisson_"):
        return 5
    return 1


def main() -> None:
    scenario = load_scenario("gwas_additive_score_sig_n100")
    problem = scenario.problem
    proposal_size = proposal_size_for_scenario(scenario.key)

    pilot_T = iid_pilot_statistics(problem, n_samples=5_000, seed=11)
    sigma_t = estimate_scale_T(pilot_T)
    q_target = scenario.exact_p_value ** (1.0 / 3.0)
    beta = init_beta_from_iid_pilot(
        pilot_T,
        problem.t_obs,
        sigma_t,
        p0_reference=scenario.exact_p_value,
        q_target=q_target,
    )

    mcmcis = run_mcmc_is(
        problem,
        beta=beta,
        sigma_t=sigma_t,
        n_steps=50_000,
        burn_in=10_000,
        n_chains=2,
        proposal_size=proposal_size,
        seed=123,
    )
    samc = run_samc(
        problem,
        n_steps=100_000,
        burn_in=20_000,
        n_bins=40,
        proposal_size=proposal_size,
        seed=456,
    )

    print(f"scenario: {scenario.key}")
    print(f"exact p:  {scenario.exact_p_value:.6g}")
    print(f"MCMC-IS:  {mcmcis.estimate:.6g} (ESS {mcmcis.ess:.1f})")
    print(f"SAMC:     {samc.estimate:.6g}")


if __name__ == "__main__":
    main()
