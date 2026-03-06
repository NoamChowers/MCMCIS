import numpy as np

from perm_pval.diagnostics.mcmc import obm_long_run_variance, obm_variance_of_mean


def test_obm_constant_sequence_zero_variance():
    x = np.ones(200, dtype=float)
    sigma2, b = obm_long_run_variance(x)
    var_mean, mcse, b2 = obm_variance_of_mean(x)
    assert b == b2
    assert sigma2 == 0.0
    assert var_mean == 0.0
    assert mcse == 0.0


def test_obm_iid_variance_scale_reasonable():
    rng = np.random.default_rng(1234)
    x = rng.normal(loc=0.0, scale=2.0, size=5000)
    var_mean_hat, mcse_hat, _ = obm_variance_of_mean(x)
    # For iid N(0, 4), var(mean) = 4/n. Use wide tolerance to avoid flaky tests.
    target = 4.0 / x.size
    assert var_mean_hat > 0.0
    assert mcse_hat > 0.0
    assert 0.2 * target <= var_mean_hat <= 5.0 * target


def test_obm_invalid_batch_size_raises():
    x = np.arange(20, dtype=float)
    raised = False
    try:
        obm_long_run_variance(x, batch_size=20)
    except ValueError:
        raised = True
    assert raised
