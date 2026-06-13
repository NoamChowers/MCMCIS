import numpy as np

from perm_pval.experiments.exact_scenarios import (
    _make_gwas_additive_score_scenario,
    _make_poisson_diffmeans_righttail_scenario,
    _make_zero_inflated_poisson_diffmeans_scenario,
    build_exact_scenarios,
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


def test_gwas_additive_score_sig_scenario_is_null_rejecting_and_exact():
    scenario = _make_gwas_additive_score_scenario(
        key="gwas_additive_score_sig_n100",
        description="GWAS-like additive score with null-rejecting exact p-value.",
        n=100,
        n_treated=50,
        maf=0.25,
        seed=386,
        downgrade_swaps=4,
    )
    assert scenario.key == "gwas_additive_score_sig_n100"
    assert scenario.statistic_name == "treated_sum"
    assert np.isclose(scenario.exact_p_value, 4.3704692764348404e-08, atol=0.0, rtol=1e-15)
    assert scenario.tail_hits == 4409425215945901258536
    assert scenario.exact_p_value < 5e-8
    assert scenario.tail_hits > 1_000_000_000
    assert float(scenario.extra["observed_score"]) < float(scenario.extra["extreme_score"])


def test_gwas_additive_score_slight_above_scenario_is_edge_non_rejecting():
    scenario = _make_gwas_additive_score_scenario(
        key="gwas_additive_score_slight_above_n100",
        description="GWAS-like additive score with edge-above-threshold exact p-value.",
        n=100,
        n_treated=50,
        maf=0.08,
        seed=1065,
        downgrade_swaps=1,
    )
    assert scenario.key == "gwas_additive_score_slight_above_n100"
    assert scenario.statistic_name == "treated_sum"
    assert np.isclose(scenario.exact_p_value, 5.795410279092721e-08, atol=0.0, rtol=1e-15)
    assert scenario.tail_hits == 5847067352508480691528
    assert 1.1 <= scenario.exact_p_value / 5e-8 <= 1.2
    assert float(scenario.extra["observed_score"]) < float(scenario.extra["extreme_score"])


def test_gwas_additive_score_ultra_scenario_is_near_one_e_minus_ten():
    scenario = _make_gwas_additive_score_scenario(
        key="gwas_additive_score_ultra_n100",
        description="GWAS-like additive score with ultra-small exact p-value.",
        n=100,
        n_treated=50,
        maf=0.25,
        seed=386,
        downgrade_swaps=2,
    )
    assert scenario.key == "gwas_additive_score_ultra_n100"
    assert np.isclose(scenario.exact_p_value, 1.1440920625102207e-10, atol=0.0, rtol=1e-15)
    assert scenario.exact_p_value < 2e-10
    assert scenario.exact_p_value > 5e-11
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


def test_poisson_diffmeans_hep_slight_above_scenario_is_edge_non_rejecting():
    scenario = _make_poisson_diffmeans_righttail_scenario(
        key="poisson_diffmeans_hep_slight_above_n200",
        description="HEP-like Poisson count test with edge-above-threshold exact p-value.",
        n_pois2=100,
        n_pois3=100,
        lam_low=2.0,
        lam_high=3.0,
        seed=80,
    )
    assert scenario.key == "poisson_diffmeans_hep_slight_above_n200"
    assert scenario.statistic_name == "difference_in_means"
    assert np.isclose(scenario.exact_p_value, 3.5087455817982693e-07, atol=0.0, rtol=1e-15)
    assert scenario.tail_hits == 31771170073799822379139107549958183545318663514761606
    assert 1.1 <= scenario.exact_p_value / 3e-7 <= 1.2


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
    assert by_key["gwas_additive_score_n40"].portfolio["family"] == "gwas_additive_score"
    assert "exploratory_exact" in by_key["zip_diffmeans_righttail_n40"].portfolio["groups"]


def test_build_exact_scenarios_emits_portfolio_groups():
    scenarios = build_exact_scenarios()
    by_key = {scenario.key: scenario for scenario in scenarios}

    assert "gwas_additive_score_sig_n100" in by_key
    assert "gwas_additive_score_slight_above_n100" in by_key
    assert "poisson_diffmeans_hep_slight_above_n200" in by_key
    assert "hypergeom_1e7" in by_key
    assert "linear_stat_dp_cube_n40" in by_key
    assert "core_claim" in by_key["gwas_additive_score_sig_n100"].portfolio["groups"]
    assert by_key["gwas_additive_score_slight_above_n100"].portfolio["threshold_band"] == "slightly_above"
    assert by_key["poisson_diffmeans_hep_slight_above_n200"].portfolio["threshold_band"] == "slightly_above"
    assert "exploratory_exact" in by_key["hypergeom_1e7"].portfolio["groups"]
    assert "core_claim" in by_key["linear_stat_dp_n40"].portfolio["groups"]
    assert by_key["bruteforce_welch_nonextreme_n22"].portfolio["family"] == "welch_bruteforce"
