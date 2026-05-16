import numpy as np

from perm_pval.experiments.exact_scenarios import load_saved_exact_scenarios
from perm_pval.experiments.shuli_scenarios import (
    build_shuli_scenarios,
    load_shuli_data,
    save_shuli_scenarios,
)


def test_shuli_data_shape_and_row_sums():
    data = load_shuli_data()

    assert data.shape == (10, 100)
    assert [int(np.sum(data[i])) for i in range(data.shape[0])] == [
        222,
        292,
        205,
        297,
        175,
        304,
        216,
        354,
        109,
        194,
    ]


def test_build_shuli_scenarios_exact_values():
    scenarios = build_shuli_scenarios()

    expected = {
        "shuli_abs_sumdiff_exm0": (0.0032763814358856536, 70),
        "shuli_abs_sumdiff_exm1": (2.1511682472774435e-05, 92),
        "shuli_abs_sumdiff_exm2": (7.360743312184874e-09, 129),
        "shuli_abs_sumdiff_exm3": (4.1389404073661124e-07, 138),
        "shuli_abs_sumdiff_exm4": (9.70494037566255e-07, 85),
    }
    by_key = {scenario.key: scenario for scenario in scenarios}

    assert sorted(by_key) == sorted(expected)
    for key, (p_value, lambda_star) in expected.items():
        scenario = by_key[key]
        assert scenario.statistic_name == "absolute_sum_difference"
        assert scenario.problem.tail == "right"
        assert np.isclose(scenario.exact_p_value, p_value, atol=0.0, rtol=1e-15)
        assert int(scenario.extra["lambda_star"]) == lambda_star
        assert np.isclose(scenario.problem.t_obs, lambda_star, atol=0.0, rtol=0.0)

    assert by_key["shuli_abs_sumdiff_exm0"].extra["readme_example"] is True
    assert by_key["shuli_abs_sumdiff_exm4"].extra["readme_example"] is False
    assert "shuli_extra_data_pair" in by_key["shuli_abs_sumdiff_exm4"].portfolio["groups"]


def test_build_shuli_scenarios_can_keep_readme_examples_only():
    scenarios = build_shuli_scenarios(include_extra_pair=False)

    assert [scenario.key for scenario in scenarios] == [
        "shuli_abs_sumdiff_exm0",
        "shuli_abs_sumdiff_exm1",
        "shuli_abs_sumdiff_exm2",
        "shuli_abs_sumdiff_exm3",
    ]


def test_shuli_scenarios_roundtrip_standard_catalog(tmp_path):
    save_shuli_scenarios(tmp_path, include_extra_pair=False)
    loaded = load_saved_exact_scenarios(tmp_path / "catalog.json")
    by_key = {scenario.key: scenario for scenario in loaded}

    assert sorted(by_key) == [
        "shuli_abs_sumdiff_exm0",
        "shuli_abs_sumdiff_exm1",
        "shuli_abs_sumdiff_exm2",
        "shuli_abs_sumdiff_exm3",
    ]
    assert by_key["shuli_abs_sumdiff_exm2"].statistic_name == "absolute_sum_difference"
    assert np.isclose(by_key["shuli_abs_sumdiff_exm2"].problem.t_obs, 129.0)
    assert by_key["shuli_abs_sumdiff_exm2"].portfolio["family"] == "shuli_original_mcmcis"
