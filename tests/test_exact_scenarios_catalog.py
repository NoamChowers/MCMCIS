import numpy as np

from perm_pval.experiments.exact_scenarios import (
    _make_gwas_additive_score_scenario,
    _make_zero_inflated_poisson_diffmeans_scenario,
    load_saved_exact_scenarios,
    save_exact_scenarios,
)


def test_gwas_additive_score_scenario_is_nonextreme_and_exact():
    scenario = _make_gwas_additive_score_scenario()
    assert scenario.key == "gwas_additive_score_n40"
    assert scenario.statistic_name == "treated_sum"
    assert np.isclose(scenario.exact_p_value, 9.121811124035818e-07, atol=0.0, rtol=1e-15)
    assert scenario.tail_hits == 125741
    assert float(scenario.extra["observed_score"]) < float(scenario.extra["extreme_score"])


def test_zip_diffmeans_scenario_is_nonextreme_and_exact():
    scenario = _make_zero_inflated_poisson_diffmeans_scenario()
    assert scenario.key == "zip_diffmeans_righttail_n40"
    assert scenario.statistic_name == "difference_in_means"
    assert np.isclose(scenario.exact_p_value, 7.265329845975008e-05, atol=0.0, rtol=1e-15)
    assert scenario.tail_hits == 10015005
    assert float(scenario.extra["observed_treated_sum"]) < float(
        scenario.extra["extreme_treated_sum"]
    )


def test_catalog_roundtrip_supports_new_linear_dp_scenarios(tmp_path):
    scenarios = [
        _make_gwas_additive_score_scenario(),
        _make_zero_inflated_poisson_diffmeans_scenario(),
    ]
    save_exact_scenarios(scenarios, tmp_path)
    loaded = load_saved_exact_scenarios(tmp_path / "catalog.json")
    by_key = {s.key: s for s in loaded}

    assert sorted(by_key) == ["gwas_additive_score_n40", "zip_diffmeans_righttail_n40"]
    assert by_key["gwas_additive_score_n40"].statistic_name == "treated_sum"
    assert by_key["zip_diffmeans_righttail_n40"].statistic_name == "difference_in_means"
