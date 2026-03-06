import numpy as np

from perm_pval.methods.beta_tuning import (
    _scaled_shortfall,
    _z_hat_from_shortfall,
    init_beta_from_iid_pilot,
    tune_beta_to_target_q,
)


def test_z_hat_decreases_with_beta():
    pilot_t = np.array([-2.0, -1.0, 0.0, 0.5, 1.25], dtype=float)
    s = _scaled_shortfall(pilot_t, T_obs=0.5, sigma_T=1.0)

    betas = np.array([0.0, 0.2, 0.7, 1.5, 3.0], dtype=float)
    z_vals = np.array([_z_hat_from_shortfall(s, b) for b in betas], dtype=float)

    assert np.isclose(z_vals[0], 1.0, atol=1e-15)
    assert np.all(np.diff(z_vals) <= 1e-12)


def test_tune_beta_bracketing_contains_target_when_successful():
    def runner(beta: float, init_state, n_steps: int, burn_in: int) -> dict[str, float]:
        del init_state, n_steps, burn_in
        q_hat = float(1.0 - np.exp(-beta))
        return {"q_hat": q_hat, "last_state": None, "accept_rate": 0.25}

    q_target = 0.6
    result = tune_beta_to_target_q(
        run_short_chain_fn=runner,
        init_state=None,
        beta0=0.2,
        q_target=q_target,
        n_steps=50,
        burn_in=10,
        tol_abs=5e-3,
        max_bracket_iter=10,
        max_bisect_iter=14,
    )

    assert result["bracket_succeeded"] is True
    assert result["q_L"] <= q_target <= result["q_U"]


def test_tune_beta_near_one_target_can_push_beta_toward_zero():
    def runner(beta: float, init_state, n_steps: int, burn_in: int) -> dict[str, float]:
        del init_state, n_steps, burn_in
        q_hat = float(0.99 + 0.01 * (1.0 - np.exp(-beta)))
        return {"q_hat": q_hat, "last_state": None, "accept_rate": 0.3}

    result = tune_beta_to_target_q(
        run_short_chain_fn=runner,
        init_state=None,
        beta0=1.0,
        q_target=0.99,
        n_steps=40,
        burn_in=10,
        tol_abs=1e-6,
        max_bracket_iter=12,
        max_bisect_iter=20,
    )

    assert result["bracket_succeeded"] is True
    assert result["beta_hat"] < 1e-3


def test_init_beta_returns_zero_when_z_target_is_ge_one():
    pilot_t = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    beta0 = init_beta_from_iid_pilot(
        pilot_T=pilot_t,
        T_obs=0.25,
        sigma_T=0.1,
        p0=1e-3,
        q_target=1e-3,
    )
    assert beta0 == 0.0
