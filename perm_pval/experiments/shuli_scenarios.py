from __future__ import annotations

from collections import defaultdict
from math import comb
from pathlib import Path
from typing import DefaultDict

import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.experiments.exact_scenarios import (
    ExactScenario,
    absolute_sum_difference,
    save_exact_scenarios,
)


SHULI_DATA_TEXT = """\
1 1 0 2 3 4 2 2 0 1 2 2 1 5 7 0 3 2 2 1 4 4 1 1 5 1 5 0 1 3 2 2 3 3 0 0 7 4 2 2 1 1 0 3 4 0 3 1 0 1 4 0 3 3 4 2 6 3 3 2 3 1 2 2 2 2 0 2 2 1 1 2 3 4 1 4 1 4 5 4 1 2 2 4 2 4 2 2 3 0 3 1 1 2 3 1 2 3 3 0
4 2 3 6 3 1 6 5 2 6 3 2 6 2 3 3 2 1 1 1 5 1 4 6 2 2 2 2 2 1 1 6 1 3 3 4 6 2 1 2 1 1 2 3 2 2 2 3 3 0 0 8 4 7 2 4 2 1 3 3 3 6 3 4 4 3 1 1 2 2 3 1 4 3 5 3 3 3 1 2 7 5 5 1 3 3 2 6 2 2 2 3 1 3 2 3 4 1 5 4
3 0 1 3 2 2 1 2 2 2 6 4 3 1 1 1 4 3 1 2 4 2 2 3 1 1 3 3 0 2 2 2 2 2 3 2 3 1 1 0 2 3 3 2 0 5 2 1 3 4 3 0 3 4 1 2 2 1 2 1 2 1 1 0 2 1 5 1 1 2 1 2 1 2 2 1 3 3 4 2 2 1 2 1 1 2 1 1 5 5 2 0 4 2 2 3 2 1 1 3
3 1 4 1 1 3 3 3 1 1 1 0 1 3 4 3 1 4 3 3 5 7 2 3 2 7 4 5 5 1 1 2 2 2 0 2 4 4 4 4 0 3 2 5 5 1 4 5 3 1 1 5 1 1 4 4 1 2 3 5 3 2 4 3 4 4 2 3 6 1 3 4 4 3 2 3 3 1 3 4 6 1 2 5 3 6 2 6 5 2 3 2 2 5 2 6 3 2 6 0
2 1 5 2 1 3 3 3 1 1 2 2 1 3 2 3 1 5 2 2 2 1 0 1 1 1 2 1 2 2 3 2 2 2 3 0 1 2 2 1 1 2 3 1 0 4 1 1 4 0 0 4 3 2 5 0 0 2 2 0 1 2 1 2 0 1 2 1 2 1 1 0 0 2 1 0 3 4 1 0 3 3 2 1 1 2 1 3 2 1 1 2 1 2 4 1 3 3 2 1
5 1 5 1 2 4 6 2 3 8 4 1 3 5 2 5 2 2 5 2 6 3 2 4 0 1 0 3 1 1 4 4 2 1 3 2 4 3 6 5 3 2 4 4 5 6 1 5 1 3 2 4 3 2 5 2 3 3 2 5 3 2 2 1 4 2 0 1 1 4 3 2 4 2 1 6 3 2 5 4 1 1 2 3 5 1 5 4 4 0 2 4 6 6 5 4 2 3 6 0
6 3 1 0 5 0 1 2 2 4 1 4 1 4 1 4 0 1 2 0 3 1 4 1 1 3 0 1 3 1 3 3 0 1 1 1 3 1 3 1 5 1 5 1 2 4 1 4 4 4 0 1 1 3 4 0 1 3 2 4 3 2 3 3 5 2 1 6 2 0 3 1 2 2 2 0 2 1 0 4 2 2 3 6 1 1 0 2 2 1 1 5 2 1 1 6 4 2 0 3
2 6 4 1 5 2 6 2 2 5 4 9 4 2 4 5 2 4 2 1 4 3 2 2 4 4 6 5 6 4 2 2 4 0 5 4 8 2 5 3 2 4 3 1 2 5 3 4 1 3 5 2 1 5 3 8 1 6 1 1 3 6 0 4 2 5 3 1 0 2 2 5 4 10 5 2 7 6 2 4 3 3 3 2 4 3 3 7 6 1 7 6 3 2 3 4 7 0 5 1
0 1 2 1 0 3 1 2 1 0 1 1 0 1 1 2 2 0 1 0 2 2 0 0 0 1 1 1 3 1 0 2 0 0 1 2 0 0 2 3 1 2 0 2 1 0 3 0 0 2 2 3 2 0 1 1 2 4 1 1 0 0 2 2 0 1 0 1 1 1 1 0 0 2 2 2 1 2 1 0 1 0 1 0 1 1 4 3 1 1 0 1 1 1 1 1 1 1 0 1
5 1 1 1 1 2 2 4 4 1 3 3 0 2 2 0 2 1 3 0 2 2 1 1 5 2 1 4 2 2 3 1 1 2 3 0 3 0 1 5 2 0 1 1 1 4 2 2 1 1 0 2 2 0 1 2 0 0 1 1 0 2 1 4 0 3 2 2 0 0 2 4 2 1 2 2 3 2 0 2 5 4 3 3 1 2 3 2 3 1 4 3 3 2 3 4 2 3 1 5
"""

SHULI_README_EXAMPLE_IDS = (0, 1, 2, 3)


def load_shuli_data(data_path: Path | str | None = None) -> np.ndarray:
    """
    Load Shuli's original scenario data as a 2D integer array.

    When ``data_path`` is omitted, the embedded copy of ``shuli_code/data/data.txt``
    is used. Passing a path is useful for checking the local original file.
    """
    if data_path is None:
        rows = [
            [int(token) for token in line.split()]
            for line in SHULI_DATA_TEXT.strip().splitlines()
            if line.strip()
        ]
        data = np.asarray(rows, dtype=np.int64)
    else:
        data = np.loadtxt(Path(data_path), dtype=np.int64)

    if data.ndim != 2:
        raise ValueError("Shuli data must be a 2D matrix.")
    if data.shape[0] % 2 != 0:
        raise ValueError("Shuli data must have an even number of rows so rows can be paired.")
    return data


def _subset_sum_distribution(values: np.ndarray, n_selected: int) -> dict[int, int]:
    states: list[DefaultDict[int, int]] = [defaultdict(int) for _ in range(n_selected + 1)]
    states[0][0] = 1
    for value in np.asarray(values, dtype=np.int64).tolist():
        for k in range(n_selected, 0, -1):
            previous = states[k - 1]
            if not previous:
                continue
            current = states[k]
            for partial_sum, count in previous.items():
                current[partial_sum + int(value)] += count
    return dict(states[n_selected])


def exact_abs_sum_diff_pvalue(values: np.ndarray, n_selected: int, lambda_star: int) -> tuple[float, int, int]:
    """
    Exact p-value for Shuli's statistic over all fixed-size relabelings.
    """
    values_arr = np.asarray(values, dtype=np.int64)
    total = int(np.sum(values_arr))
    dist = _subset_sum_distribution(values_arr, n_selected=int(n_selected))
    tail_hits = sum(
        count
        for selected_sum, count in dist.items()
        if abs(total - 2 * int(selected_sum)) >= int(lambda_star)
    )
    n_permutations = comb(int(values_arr.size), int(n_selected))
    return float(tail_hits / n_permutations), int(tail_hits), int(n_permutations)


def _rarity_band(p_value: float) -> str:
    if p_value <= 1e-8:
        return "ultra_rare"
    if p_value <= 1e-6:
        return "extreme"
    if p_value <= 1e-4:
        return "very_rare"
    return "rare"


def _make_shuli_scenario(data: np.ndarray, exm_id: int, *, source: str) -> ExactScenario:
    x1 = np.asarray(data[2 * exm_id], dtype=np.int64)
    x2 = np.asarray(data[2 * exm_id + 1], dtype=np.int64)
    x = np.concatenate([x1, x2])
    y_obs = np.zeros(x.size, dtype=np.int8)
    y_obs[x1.size :] = 1

    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=absolute_sum_difference,
        tail="right",
    )

    sum_x1 = int(np.sum(x1))
    sum_x2 = int(np.sum(x2))
    lambda_star = int(abs(sum_x2 - sum_x1))
    if not np.isclose(problem.t_obs, lambda_star, atol=0.0, rtol=0.0):
        raise ValueError(f"Unexpected observed statistic for Shuli exm_id={exm_id}.")

    exact_p, tail_hits, n_permutations = exact_abs_sum_diff_pvalue(
        x,
        n_selected=x2.size,
        lambda_star=lambda_star,
    )
    readme_example = exm_id in SHULI_README_EXAMPLE_IDS
    groups = ["shuli_scenarios", "legacy_original", "exploratory_exact"]
    groups.append("shuli_readme_examples" if readme_example else "shuli_extra_data_pair")

    scenario = ExactScenario(
        key=f"shuli_abs_sumdiff_exm{exm_id}",
        description=(
            f"Shuli original MCMCIS data pair exm_id={exm_id}: rows "
            f"{2 * exm_id}-{2 * exm_id + 1}, n1=n2={x1.size}."
        ),
        problem=problem,
        statistic_name="absolute_sum_difference",
        exact_method="integer subset-sum DP for |sum(group1)-sum(group2)|",
        exact_p_value=float(exact_p),
        tail_hits=int(tail_hits),
        n_permutations=int(n_permutations),
        notes=(
            "Rows are paired using shuli_code/main.py's convention. The README usage examples "
            "list exm_id 0..3; the embedded data file also contains exm_id 4."
        ),
        extra={
            "source": source,
            "exm_id": int(exm_id),
            "row_pair": [int(2 * exm_id), int(2 * exm_id + 1)],
            "n_x1": int(x1.size),
            "n_x2": int(x2.size),
            "sum_x1": sum_x1,
            "sum_x2": sum_x2,
            "total_sum": int(sum_x1 + sum_x2),
            "lambda_star": lambda_star,
            "readme_example": bool(readme_example),
            "original_statistic": "|sum(X2) - sum(X1)|",
            "pvalue_definition": (
                "P(|sum(G2)-sum(G1)| >= lambda_star) over all fixed-size relabelings."
            ),
        },
        portfolio={
            "family": "shuli_original_mcmcis",
            "statistic_family": "absolute_linear_statistic",
            "data_family": "integer_counts",
            "rarity_band": _rarity_band(float(exact_p)),
            "expected_difficulty": "hard" if exact_p <= 1e-6 else "moderate",
            "sample_size_band": "large",
            "has_ties": True,
            "is_discrete": True,
            "groups": groups,
        },
    )
    return scenario


def build_shuli_scenarios(
    *,
    data_path: Path | str | None = None,
    include_extra_pair: bool = True,
) -> list[ExactScenario]:
    """
    Build a scenario catalog from Shuli's original paired data rows.

    By default all row pairs present in the data are included. Set
    ``include_extra_pair=False`` to keep only the README-documented exm_id 0..3.
    """
    data = load_shuli_data(data_path)
    n_pairs = int(data.shape[0] // 2)
    if not include_extra_pair:
        n_pairs = min(n_pairs, len(SHULI_README_EXAMPLE_IDS))

    source = "embedded:shuli_code/data/data.txt" if data_path is None else str(Path(data_path))
    return [_make_shuli_scenario(data, exm_id, source=source) for exm_id in range(n_pairs)]


def save_shuli_scenarios(
    output_dir: Path | str,
    *,
    data_path: Path | str | None = None,
    include_extra_pair: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Save Shuli's scenarios using the repository's standard exact-scenario format.
    """
    scenarios = build_shuli_scenarios(
        data_path=data_path,
        include_extra_pair=include_extra_pair,
    )
    return save_exact_scenarios(scenarios, Path(output_dir), overwrite=overwrite)
