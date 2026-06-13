from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.brute_force import BruteForceExactSolver
from perm_pval.exact.linear_statistic_dp import LinearStatisticDPSolver
from perm_pval.exact.rank_sum_dp import RankSumDPSolver
from perm_pval.stats.ranks import mann_whitney_u
from perm_pval.stats.two_sample import difference_in_means, t_statistic_welch


@dataclass
class ExactScenario:
    key: str
    description: str
    problem: PermutationTestProblem
    statistic_name: str
    exact_method: str
    exact_p_value: float
    tail_hits: int
    n_permutations: int
    notes: str
    extra: dict[str, Any]
    portfolio: dict[str, Any] = field(default_factory=dict)


def treated_successes(x: np.ndarray, y: np.ndarray) -> float:
    """
    Count of successes among treated labels (x must be 0/1).
    """
    return float(int(np.dot(np.asarray(x, dtype=np.int8), np.asarray(y, dtype=np.int8))))


def treated_sum(x: np.ndarray, y: np.ndarray) -> float:
    """
    Sum of numeric scores among treated labels.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=np.int8)
    return float(np.dot(x_arr, y_arr))


def absolute_sum_difference(x: np.ndarray, y: np.ndarray) -> float:
    """
    Absolute difference between fixed-size group sums.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=np.int8)
    return float(abs(np.sum(x_arr[y_arr == 1]) - np.sum(x_arr[y_arr == 0])))


def exact_hypergeom_right_tail(
    n: int,
    n_treated: int,
    total_successes: int,
    k_obs: int,
) -> tuple[float, int, int]:
    """
    Exact right-tail probability for treated success count under the permutation null.
    """
    total_states = comb(n, n_treated)
    k_max = min(total_successes, n_treated)
    k_min = max(0, n_treated - (n - total_successes))
    tail_hits = 0
    for k in range(max(k_obs, k_min), k_max + 1):
        tail_hits += comb(total_successes, k) * comb(n - total_successes, n_treated - k)
    return float(tail_hits / total_states), int(tail_hits), int(total_states)


def near_tail_labels_tiny_p_case() -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic n=40 setup used in notebook simulations.
    """
    n = 40
    x = np.arange(1, n + 1, dtype=float)
    treated_values = set([1, 16] + list(range(23, 41)))
    y = np.array([1 if int(v) in treated_values else 0 for v in x], dtype=np.int8)
    return x, y


def _rarity_band(exact_p_value: float) -> str:
    p = float(exact_p_value)
    if p <= 1e-8:
        return "ultra_rare"
    if p <= 1e-6:
        return "extreme"
    if p <= 1e-4:
        return "very_rare"
    return "rare"


def _sample_size_band(n: int) -> str:
    n_int = int(n)
    if n_int <= 24:
        return "small"
    if n_int <= 40:
        return "medium"
    return "large"


def _portfolio_metadata(
    *,
    scenario: ExactScenario,
    family: str,
    statistic_family: str,
    data_family: str,
    difficulty: str,
    groups: tuple[str, ...],
    has_ties: bool,
    is_discrete: bool,
) -> dict[str, Any]:
    return {
        "family": str(family),
        "statistic_family": str(statistic_family),
        "data_family": str(data_family),
        "rarity_band": _rarity_band(float(scenario.exact_p_value)),
        "expected_difficulty": str(difficulty),
        "sample_size_band": _sample_size_band(int(scenario.problem.n)),
        "has_ties": bool(has_ties),
        "is_discrete": bool(is_discrete),
        "groups": [str(group) for group in groups],
    }


def _apply_portfolio_metadata(
    scenario: ExactScenario,
    *,
    family: str,
    statistic_family: str,
    data_family: str,
    difficulty: str,
    groups: tuple[str, ...],
    has_ties: bool,
    is_discrete: bool,
) -> ExactScenario:
    scenario.portfolio = _portfolio_metadata(
        scenario=scenario,
        family=family,
        statistic_family=statistic_family,
        data_family=data_family,
        difficulty=difficulty,
        groups=groups,
        has_ties=has_ties,
        is_discrete=is_discrete,
    )
    return scenario


def _attach_application_threshold(
    scenario: ExactScenario,
    *,
    known_significance_threshold: float,
    setting_key: str,
    setting_label: str,
    threshold_band: str,
) -> ExactScenario:
    extra = dict(scenario.extra)
    extra["known_significance_threshold"] = float(known_significance_threshold)
    extra["application_setting_key"] = str(setting_key)
    extra["application_setting_label"] = str(setting_label)
    extra["threshold_band"] = str(threshold_band)
    scenario.extra = extra

    portfolio = dict(scenario.portfolio)
    groups = [str(v) for v in portfolio.get("groups", [])]
    for group in ("threshold_suite", str(setting_key), f"{setting_key}_{threshold_band}"):
        if group not in groups:
            groups.append(group)
    portfolio["groups"] = groups
    portfolio["known_significance_threshold"] = float(known_significance_threshold)
    portfolio["application_setting_key"] = str(setting_key)
    portfolio["application_setting_label"] = str(setting_label)
    portfolio["threshold_band"] = str(threshold_band)
    scenario.portfolio = portfolio
    return scenario


def _near_extreme_linear_labels(x: np.ndarray, n_treated: int, *, downgrade_swaps: int = 1) -> np.ndarray:
    """
    Start from the top-score labeling and downgrade it by a small number of swaps.

    This keeps the observed statistic near the right tail while avoiding the most
    extreme allocation.
    """
    if downgrade_swaps <= 0:
        raise ValueError("downgrade_swaps must be positive.")

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("_near_extreme_linear_labels requires a 1D score array.")
    if not (0 < n_treated < x_arr.size):
        raise ValueError("n_treated must satisfy 0 < n_treated < len(x).")

    order = np.argsort(x_arr, kind="mergesort")
    y = np.zeros(x_arr.size, dtype=np.int8)
    treated_idx = order[-n_treated:]
    y[treated_idx] = 1

    control_idx = order[:-n_treated]
    treated_sorted = treated_idx[np.argsort(x_arr[treated_idx], kind="mergesort")]
    control_sorted = control_idx[np.argsort(x_arr[control_idx], kind="mergesort")[::-1]]

    used_controls: set[int] = set()
    swaps_done = 0
    for ti in treated_sorted.tolist():
        for ci in control_sorted.tolist():
            if ci in used_controls:
                continue
            if x_arr[ti] > x_arr[ci]:
                y[ti] = 0
                y[ci] = 1
                used_controls.add(ci)
                swaps_done += 1
                break
        if swaps_done >= downgrade_swaps:
            break

    if swaps_done < downgrade_swaps:
        raise ValueError(
            "Could not construct a non-extreme near-tail labeling with the requested swaps."
        )
    return y


def _binary_hypergeom_problem(
    *,
    n: int,
    n_treated: int,
    m_success: int,
    k_obs: int,
) -> PermutationTestProblem:
    if not (0 <= m_success <= n):
        raise ValueError("m_success must be between 0 and n.")
    if not (0 <= n_treated <= n):
        raise ValueError("n_treated must be between 0 and n.")

    x = np.zeros(n, dtype=np.int8)
    x[:m_success] = 1

    y = np.zeros(n, dtype=np.int8)
    success_idx = np.arange(0, m_success)
    failure_idx = np.arange(m_success, n)

    if k_obs > success_idx.size:
        raise ValueError("k_obs cannot exceed number of successes.")
    n_fail_treated = n_treated - k_obs
    if n_fail_treated < 0 or n_fail_treated > failure_idx.size:
        raise ValueError("Invalid n_treated/k_obs for binary hypergeometric setup.")

    treated_idx = np.concatenate([success_idx[:k_obs], failure_idx[:n_fail_treated]])
    y[treated_idx] = 1
    return PermutationTestProblem(X=x, y_obs=y, statistic=treated_successes, tail="right")


def _make_hypergeom_scenario(
    *,
    key: str,
    description: str,
    n: int,
    n_treated: int,
    m_success: int,
    k_obs: int,
) -> ExactScenario:
    problem = _binary_hypergeom_problem(
        n=n,
        n_treated=n_treated,
        m_success=m_success,
        k_obs=k_obs,
    )
    p_hg, hits_hg, total_hg = exact_hypergeom_right_tail(
        n=n,
        n_treated=n_treated,
        total_successes=m_success,
        k_obs=int(problem.t_obs),
    )

    # Cross-check: same exact p-value via linear-stat DP on T(y)=sum x_i y_i.
    p_dp = LinearStatisticDPSolver(
        problem,
        scores=np.asarray(problem.X, dtype=float),
        score_scale=1,
        scale=1.0,
        offset=0.0,
    ).compute()

    if not np.isclose(p_hg, p_dp.p_value, atol=0.0, rtol=1e-15):
        raise ValueError(
            f"Fisher-exact and DP exact values disagree for scenario '{key}': "
            f"{p_hg} vs {p_dp.p_value}"
        )

    scenario = ExactScenario(
        key=key,
        description=description,
        problem=problem,
        statistic_name="treated_successes",
        exact_method="Fisher exact test (2x2; hypergeometric tail)",
        exact_p_value=p_hg,
        tail_hits=hits_hg,
        n_permutations=total_hg,
        notes=(
            "Fisher exact one-sided tail for a 2x2 table. In this binary permutation setup, "
            "it matches full permutation enumeration."
        ),
        extra={
            "n": n,
            "n_treated": n_treated,
            "m_success": m_success,
            "k_obs": k_obs,
            "dp_cross_check_p": float(p_dp.p_value),
        },
    )
    return _apply_portfolio_metadata(
        scenario,
        family="hypergeometric_tail",
        statistic_family="binary_count",
        data_family="binary",
        difficulty="moderate" if k_obs < m_success else "hard",
        groups=("exploratory_exact",) + (("core_claim",) if key in {"hypergeom_1e5", "hypergeom_1e7"} else tuple()) + (("stress_test",) if key == "hypergeom_1e7" else tuple()),
        has_ties=True,
        is_discrete=True,
    )


def _make_rank_sum_dp_scenario() -> ExactScenario:
    x, y = near_tail_labels_tiny_p_case()
    problem = PermutationTestProblem(X=x, y_obs=y, statistic=mann_whitney_u, tail="right")
    exact = RankSumDPSolver(problem, statistic_type="u").compute()
    scenario = ExactScenario(
        key="rank_sum_dp_n40",
        description="Rank-sum (Mann-Whitney U) with deterministic near-tail labels, n=40.",
        problem=problem,
        statistic_name="mann_whitney_u",
        exact_method="RankSumDPSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes="DP over rank-sum counts with no ties in X.",
        extra={},
    )
    return _apply_portfolio_metadata(
        scenario,
        family="rank_sum_dp",
        statistic_family="rank_based",
        data_family="continuous_ordered",
        difficulty="moderate",
        groups=("core_claim", "exploratory_exact"),
        has_ties=False,
        is_discrete=False,
    )


def _make_linear_dp_scenario() -> ExactScenario:
    x, y = near_tail_labels_tiny_p_case()
    x_nonlinear = x**2
    problem = PermutationTestProblem(
        X=x_nonlinear,
        y_obs=y,
        statistic=difference_in_means,
        tail="right",
    )
    exact = LinearStatisticDPSolver.from_difference_in_means(problem, score_scale=1).compute()
    scenario = ExactScenario(
        key="linear_stat_dp_n40",
        description="Difference in means on nonlinear numeric X=i^2, n=40.",
        problem=problem,
        statistic_name="difference_in_means",
        exact_method="LinearStatisticDPSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes="DP over treated weighted sums mapped exactly to difference in means.",
        extra={},
    )
    return _apply_portfolio_metadata(
        scenario,
        family="linear_stat_dp",
        statistic_family="linear_statistic",
        data_family="nonlinear_numeric",
        difficulty="hard",
        groups=("core_claim", "stress_test", "exploratory_exact"),
        has_ties=False,
        is_discrete=False,
    )


def _make_rank_sum_family_scenario(
    *,
    key: str,
    description: str,
    n: int,
    n_treated: int,
    downgrade_swaps: int,
) -> ExactScenario:
    x = np.arange(1, n + 1, dtype=float)
    y = _near_extreme_linear_labels(x, n_treated, downgrade_swaps=downgrade_swaps)
    problem = PermutationTestProblem(X=x, y_obs=y, statistic=mann_whitney_u, tail="right")
    exact = RankSumDPSolver(problem, statistic_type="u").compute()
    scenario = ExactScenario(
        key=key,
        description=description,
        problem=problem,
        statistic_name="mann_whitney_u",
        exact_method="RankSumDPSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes="Parametric rank-sum family member generated from near-extreme label downgrades.",
        extra={
            "n": int(n),
            "n_treated": int(n_treated),
            "downgrade_swaps": int(downgrade_swaps),
        },
    )
    return _apply_portfolio_metadata(
        scenario,
        family="rank_sum_dp",
        statistic_family="rank_based",
        data_family="continuous_ordered",
        difficulty="hard" if downgrade_swaps == 1 else "moderate",
        groups=("exploratory_exact",) + (("stress_test",) if downgrade_swaps == 1 else tuple()),
        has_ties=False,
        is_discrete=False,
    )


def _make_linear_family_scenario(
    *,
    key: str,
    description: str,
    n: int,
    n_treated: int,
    power: int,
    downgrade_swaps: int,
) -> ExactScenario:
    x = np.arange(1, n + 1, dtype=float) ** int(power)
    y = _near_extreme_linear_labels(x, n_treated, downgrade_swaps=downgrade_swaps)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y,
        statistic=difference_in_means,
        tail="right",
    )
    exact = LinearStatisticDPSolver.from_difference_in_means(problem, score_scale=1).compute()
    scenario = ExactScenario(
        key=key,
        description=description,
        problem=problem,
        statistic_name="difference_in_means",
        exact_method="LinearStatisticDPSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes="Linear-stat DP family member generated from powered scores and near-extreme label downgrades.",
        extra={
            "n": int(n),
            "n_treated": int(n_treated),
            "power": int(power),
            "downgrade_swaps": int(downgrade_swaps),
        },
    )
    return _apply_portfolio_metadata(
        scenario,
        family="linear_stat_dp",
        statistic_family="linear_statistic",
        data_family="nonlinear_numeric",
        difficulty="hard" if downgrade_swaps == 1 else "moderate",
        groups=("exploratory_exact",) + (("stress_test",) if downgrade_swaps == 1 else tuple()),
        has_ties=False,
        is_discrete=False,
    )


def _make_gwas_additive_score_scenario(
    *,
    key: str = "gwas_additive_score_n40",
    description: str = (
        "GWAS-like additive score: Binomial(2, maf=0.15) dosages, n=40, "
        "right-tail treated dosage sum with a near-extreme but non-max case set."
    ),
    n: int = 40,
    n_treated: int = 20,
    maf: float = 0.15,
    seed: int = 9,
    downgrade_swaps: int = 1,
) -> ExactScenario:
    rng = np.random.default_rng(seed)
    x = rng.binomial(2, maf, size=n).astype(float)
    y = _near_extreme_linear_labels(x, n_treated, downgrade_swaps=downgrade_swaps)
    problem = PermutationTestProblem(X=x, y_obs=y, statistic=treated_sum, tail="right")
    exact = LinearStatisticDPSolver(
        problem,
        scores=x,
        score_scale=1,
        scale=1.0,
        offset=0.0,
    ).compute()

    extreme_score = float(np.sum(np.sort(x)[-n_treated:]))
    observed_score = float(problem.t_obs)
    if not observed_score < extreme_score:
        raise ValueError(
            f"Scenario '{key}' must be non-extreme, but observed score={observed_score} "
            f"and extreme score={extreme_score}."
        )

    scenario = ExactScenario(
        key=key,
        description=description,
        problem=problem,
        statistic_name="treated_sum",
        exact_method="LinearStatisticDPSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes=(
            "Additive-dose score statistic T=sum_i G_i y_i with fixed case count. "
            "Observed labels are constructed by a one-swap downgrade from the maximal dosage sum."
        ),
        extra={
            "n": int(n),
            "n_treated": int(n_treated),
            "maf": float(maf),
            "seed": int(seed),
            "downgrade_swaps": int(downgrade_swaps),
            "observed_score": observed_score,
            "extreme_score": extreme_score,
            "n_heterozygous": int(np.sum(x == 1.0)),
            "n_homozygous_alt": int(np.sum(x == 2.0)),
            "total_dosage_sum": int(np.sum(x)),
        },
    )
    return _apply_portfolio_metadata(
        scenario,
        family="gwas_additive_score",
        statistic_family="linear_statistic",
        data_family="discrete_score",
        difficulty="hard" if downgrade_swaps == 1 else "moderate",
        groups=("exploratory_exact",)
        + (("core_claim",) if key in {"gwas_additive_score_n40", "gwas_additive_score_sig_n100"} else tuple())
        + (("stress_test",) if downgrade_swaps == 1 else tuple()),
        has_ties=True,
        is_discrete=True,
    )


def _make_skew_sparse_burden_scenario(
    *,
    key: str = "skew_sparse_burden_sum_tiny",
    description: str = (
        "Highly skewed sparse-burden benchmark: 78 zeros plus seven rare positive scores, "
        "n=85 with a tiny right-tail treated-sum event."
    ),
) -> ExactScenario:
    n = 85
    n_treated = 6
    x = np.asarray([0] * 78 + [1, 2, 2, 2, 4, 8, 16], dtype=float)
    y = np.zeros(n, dtype=np.int8)
    # Observed treated set gives score 33, which has exactly four right-tail hits.
    y[[78, 80, 81, 82, 83, 84]] = 1

    problem = PermutationTestProblem(X=x, y_obs=y, statistic=treated_sum, tail="right")
    exact = LinearStatisticDPSolver(
        problem,
        scores=x,
        score_scale=1,
        scale=1.0,
        offset=0.0,
    ).compute()

    if exact.tail_hits != 4:
        raise ValueError(
            f"Scenario '{key}' expected 4 tail hits but got {exact.tail_hits}."
        )

    scenario = ExactScenario(
        key=key,
        description=description,
        problem=problem,
        statistic_name="treated_sum",
        exact_method="LinearStatisticDPSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes=(
            "Sparse burden-style treated-sum benchmark with a deliberately highly skewed permutation null. "
            "Observed labels select six of seven rare positive scores, yielding a non-singleton tiny right tail."
        ),
        extra={
            "n": int(n),
            "n_treated": int(n_treated),
            "positive_scores": [1, 2, 2, 2, 4, 8, 16],
            "n_zeros": 78,
            "observed_score": float(problem.t_obs),
            "exact_tail_hits": int(exact.tail_hits),
        },
    )
    return _apply_portfolio_metadata(
        scenario,
        family="skew_sparse_burden",
        statistic_family="linear_statistic",
        data_family="sparse_discrete_score",
        difficulty="hard",
        groups=("exploratory_exact", "stress_test", "shape_stress"),
        has_ties=True,
        is_discrete=True,
    )


def _make_zero_inflated_poisson_diffmeans_scenario(
    *,
    key: str = "zip_diffmeans_righttail_n40",
    description: str = (
        "Zero-inflated Poisson benchmark: 80% zeros, otherwise Pois(4), n=40, "
        "right-tail difference in means with a near-extreme but non-max treated set."
    ),
    n: int = 40,
    n_treated: int = 20,
    zero_prob: float = 0.80,
    lam_nonzero: float = 4.0,
    seed: int = 4,
    downgrade_swaps: int = 1,
) -> ExactScenario:
    rng = np.random.default_rng(seed)
    is_zero = rng.random(n) < zero_prob
    x = np.where(is_zero, 0, rng.poisson(lam_nonzero, size=n)).astype(float)
    y = _near_extreme_linear_labels(x, n_treated, downgrade_swaps=downgrade_swaps)
    problem = PermutationTestProblem(
        X=x,
        y_obs=y,
        statistic=difference_in_means,
        tail="right",
    )
    exact = LinearStatisticDPSolver.from_difference_in_means(problem, score_scale=1).compute()

    extreme_treated_sum = float(np.sum(np.sort(x)[-n_treated:]))
    observed_treated_sum = float(np.sum(x[y == 1]))
    if not observed_treated_sum < extreme_treated_sum:
        raise ValueError(
            f"Scenario '{key}' must be non-extreme, but observed treated sum={observed_treated_sum} "
            f"and extreme treated sum={extreme_treated_sum}."
        )

    scenario = ExactScenario(
        key=key,
        description=description,
        problem=problem,
        statistic_name="difference_in_means",
        exact_method="LinearStatisticDPSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes=(
            "Zero-inflated count data with exact linear-stat DP truth. "
            "Observed labels are a one-swap downgrade from the maximal treated-count sum."
        ),
        extra={
            "n": int(n),
            "n_treated": int(n_treated),
            "zero_prob": float(zero_prob),
            "lambda_nonzero": float(lam_nonzero),
            "seed": int(seed),
            "downgrade_swaps": int(downgrade_swaps),
            "n_zeros": int(np.sum(x == 0.0)),
            "n_nonzero": int(np.sum(x > 0.0)),
            "observed_treated_sum": observed_treated_sum,
            "extreme_treated_sum": extreme_treated_sum,
        },
    )
    return _apply_portfolio_metadata(
        scenario,
        family="zero_inflated_count",
        statistic_family="linear_statistic",
        data_family="zero_inflated_count",
        difficulty="hard" if zero_prob >= 0.8 else "moderate",
        groups=("exploratory_exact",)
        + (("core_claim",) if key == "zip_diffmeans_righttail_n40" else tuple())
        + (("stress_test",) if zero_prob >= 0.8 else tuple()),
        has_ties=True,
        is_discrete=True,
    )


def _make_poisson_diffmeans_righttail_scenario(
    *,
    key: str = "poisson_diffmeans_righttail_tiny_n200",
    description: str = (
        "Poisson benchmark: treated~Pois(3), control~Pois(2), "
        "right-tail difference in means, n1=n2=100."
    ),
    n_pois2: int = 100,
    n_pois3: int = 100,
    lam_low: float = 2.0,
    lam_high: float = 3.0,
    seed: int = 20260228,
) -> ExactScenario:
    rng = np.random.default_rng(seed)
    x_low = rng.poisson(lam=lam_low, size=n_pois2)
    x_high = rng.poisson(lam=lam_high, size=n_pois3)
    # Keep generation order as requested (Pois(2), then Pois(3)),
    # but define treated group as the second (higher-mean) block.
    x = np.concatenate([x_low, x_high]).astype(float)
    y = np.array([0] * n_pois2 + [1] * n_pois3, dtype=np.int8)

    problem = PermutationTestProblem(
        X=x,
        y_obs=y,
        statistic=difference_in_means,
        tail="right",
    )
    exact = LinearStatisticDPSolver.from_difference_in_means(problem, score_scale=1).compute()
    scenario = ExactScenario(
        key=key,
        description=description,
        problem=problem,
        statistic_name="difference_in_means",
        exact_method="LinearStatisticDPSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes=(
            "Deterministic benchmark dataset generated from Poisson draws with fixed seed; "
            "right-tail test on mean(group1)-mean(group0)."
        ),
        extra={
            "n_pois2": int(n_pois2),
            "n_pois3": int(n_pois3),
            "lambda_low": float(lam_low),
            "lambda_high": float(lam_high),
            "treated_block": "pois3_second_block",
            "seed": int(seed),
            "mean_pois2": float(np.mean(x_low)),
            "mean_pois3": float(np.mean(x_high)),
        },
    )
    return _apply_portfolio_metadata(
        scenario,
        family="poisson_count",
        statistic_family="linear_statistic",
        data_family="count",
        difficulty="moderate",
        groups=("exploratory_exact",),
        has_ties=True,
        is_discrete=True,
    )


def _make_bruteforce_welch_scenario(
    *,
    key: str = "bruteforce_welch_nonextreme_n22",
    description: str = (
        "Brute-force benchmark (non-DP statistic): right-tail Welch t-statistic, "
        "n=22 with non-extreme observed labeling."
    ),
    n: int = 22,
    n_treated: int = 11,
    swap_from_extreme: int = 2,
) -> ExactScenario:
    x = np.arange(1, n + 1, dtype=float)
    y = np.zeros(n, dtype=np.int8)
    y[np.arange(n - n_treated, n)] = 1
    treated_idx = np.where(y == 1)[0]
    control_idx = np.where(y == 0)[0]
    if swap_from_extreme < 0 or swap_from_extreme > min(treated_idx.size, control_idx.size):
        raise ValueError("swap_from_extreme is out of range.")
    for s in range(swap_from_extreme):
        y[treated_idx[s]] = 0
        y[control_idx[-1 - s]] = 1

    problem = PermutationTestProblem(
        X=x,
        y_obs=y,
        statistic=t_statistic_welch,
        tail="right",
    )
    exact = BruteForceExactSolver(problem, max_permutations=comb(n, n_treated)).compute()
    if exact.tail_hits <= 1:
        raise ValueError(
            f"Scenario '{key}' has tail_hits={exact.tail_hits}. "
            "Increase swap_from_extreme to avoid single-tail edge case."
        )
    scenario = ExactScenario(
        key=key,
        description=description,
        problem=problem,
        statistic_name="t_statistic_welch",
        exact_method="BruteForceExactSolver",
        exact_p_value=float(exact.p_value),
        tail_hits=int(exact.tail_hits),
        n_permutations=int(exact.n_permutations),
        notes=(
            "Welch t-statistic is nonlinear in group variances and not covered by current linear/rank DP solvers. "
            "Exact p-value computed via full brute-force permutation enumeration."
        ),
        extra={
            "swap_from_extreme": int(swap_from_extreme),
            "n": int(n),
            "n_treated": int(n_treated),
        },
    )
    return _apply_portfolio_metadata(
        scenario,
        family="welch_bruteforce",
        statistic_family="nonlinear_statistic",
        data_family="ordered_numeric",
        difficulty="hard" if swap_from_extreme <= 2 else "moderate",
        groups=("exploratory_exact",)
        + (("core_claim",) if key == "bruteforce_welch_nonextreme_n22" else tuple())
        + (("stress_test",) if swap_from_extreme <= 2 else tuple()),
        has_ties=False,
        is_discrete=False,
    )


def build_exact_scenarios() -> list[ExactScenario]:
    scenarios: list[ExactScenario] = [
        _make_hypergeom_scenario(
            key="hypergeom_3e4",
            description="Binary treated-successes with exact p around 3e-4.",
            n=40,
            n_treated=20,
            m_success=15,
            k_obs=13,
        ),
        _make_hypergeom_scenario(
            key="hypergeom_1e5",
            description="Binary treated-successes with exact p around 1e-5.",
            n=40,
            n_treated=20,
            m_success=15,
            k_obs=14,
        ),
        _make_hypergeom_scenario(
            key="hypergeom_1e7",
            description="Binary treated-successes with exact p around 1e-7.",
            n=40,
            n_treated=20,
            m_success=15,
            k_obs=15,
        ),
        _make_gwas_additive_score_scenario(),
        _make_gwas_additive_score_scenario(
            key="gwas_additive_score_sig_n100",
            description=(
                "GWAS-like additive score: Binomial(2, maf=0.25) dosages, n=100, "
                "right-tail treated dosage sum with a larger permutation space and null-rejecting exact p-value."
            ),
            n=100,
            n_treated=50,
            maf=0.25,
            seed=386,
            downgrade_swaps=4,
        ),
        _make_gwas_additive_score_scenario(
            key="gwas_additive_score_ultra_n100",
            description=(
                "GWAS-like additive score: Binomial(2, maf=0.25) dosages, n=100, "
                "right-tail treated dosage sum with p-value far below the genome-wide threshold."
            ),
            n=100,
            n_treated=50,
            maf=0.25,
            seed=386,
            downgrade_swaps=2,
        ),
        _make_gwas_additive_score_scenario(
            key="gwas_additive_score_slight_above_n100",
            description=(
                "GWAS-like additive score: Binomial(2, maf=0.08) dosages, n=100, "
                "right-tail treated dosage sum with p-value slightly above the genome-wide threshold."
            ),
            n=100,
            n_treated=50,
            maf=0.08,
            seed=1065,
            downgrade_swaps=1,
        ),
        _make_gwas_additive_score_scenario(
            key="gwas_additive_score_above_n100",
            description=(
                "GWAS-like additive score: Binomial(2, maf=0.25) dosages, n=100, "
                "right-tail treated dosage sum with p-value above the genome-wide threshold."
            ),
            n=100,
            n_treated=50,
            maf=0.25,
            seed=386,
            downgrade_swaps=5,
        ),
        _make_gwas_additive_score_scenario(
            key="gwas_additive_score_swap2_n40",
            description=(
                "GWAS-like additive score with a two-swap downgrade from the maximal dosage sum, n=40."
            ),
            downgrade_swaps=2,
        ),
        _make_gwas_additive_score_scenario(
            key="gwas_additive_score_maf010_n40",
            description=(
                "GWAS-like additive score with Binomial(2, maf=0.10) dosages, n=40."
            ),
            maf=0.10,
            seed=21,
            downgrade_swaps=1,
        ),
        _make_skew_sparse_burden_scenario(),
        _make_rank_sum_dp_scenario(),
        _make_rank_sum_family_scenario(
            key="rank_sum_dp_swap2_n40",
            description="Rank-sum family member with a two-swap downgrade, n=40.",
            n=40,
            n_treated=20,
            downgrade_swaps=2,
        ),
        _make_rank_sum_family_scenario(
            key="rank_sum_dp_n32",
            description="Rank-sum family member with n=32 and a one-swap downgrade.",
            n=32,
            n_treated=16,
            downgrade_swaps=1,
        ),
        _make_linear_dp_scenario(),
        _make_linear_family_scenario(
            key="linear_stat_dp_cube_n40",
            description="Difference in means on cubic numeric scores X=i^3, n=40.",
            n=40,
            n_treated=20,
            power=3,
            downgrade_swaps=1,
        ),
        _make_linear_family_scenario(
            key="linear_stat_dp_swap2_n40",
            description="Difference in means on quadratic scores X=i^2 with a two-swap downgrade, n=40.",
            n=40,
            n_treated=20,
            power=2,
            downgrade_swaps=2,
        ),
        _make_zero_inflated_poisson_diffmeans_scenario(),
        _make_zero_inflated_poisson_diffmeans_scenario(
            key="zip_diffmeans_lesszero_n40",
            description=(
                "Zero-inflated Poisson benchmark with 65% zeros and near-extreme treated set, n=40."
            ),
            zero_prob=0.65,
            seed=14,
        ),
        _make_zero_inflated_poisson_diffmeans_scenario(
            key="zip_diffmeans_morezero_n40",
            description=(
                "Zero-inflated Poisson benchmark with 90% zeros and near-extreme treated set, n=40."
            ),
            zero_prob=0.90,
            seed=24,
        ),
        _make_poisson_diffmeans_righttail_scenario(),
        _make_poisson_diffmeans_righttail_scenario(
            key="poisson_diffmeans_hep_sig_n200",
            description=(
                "Poisson count benchmark resembling a high-energy-physics counting test: "
                "treated~Pois(3), control~Pois(2), n1=n2=100, with p-value near the 3e-7 discovery threshold."
            ),
            n_pois2=100,
            n_pois3=100,
            lam_low=2.0,
            lam_high=3.0,
            seed=5,
        ),
        _make_poisson_diffmeans_righttail_scenario(
            key="poisson_diffmeans_hep_slight_above_n200",
            description=(
                "Poisson count benchmark resembling a high-energy-physics counting test: "
                "treated~Pois(3), control~Pois(2), n1=n2=100, with p-value slightly above the 3e-7 discovery threshold."
            ),
            n_pois2=100,
            n_pois3=100,
            lam_low=2.0,
            lam_high=3.0,
            seed=80,
        ),
        _make_poisson_diffmeans_righttail_scenario(
            key="poisson_diffmeans_hep_above_n200",
            description=(
                "Poisson count benchmark resembling a high-energy-physics counting test: "
                "treated~Pois(3), control~Pois(2), n1=n2=100, with p-value above the 3e-7 discovery threshold."
            ),
            n_pois2=100,
            n_pois3=100,
            lam_low=2.0,
            lam_high=3.0,
            seed=1,
        ),
        _make_poisson_diffmeans_righttail_scenario(
            key="poisson_diffmeans_hep_ultra_n200",
            description=(
                "Poisson count benchmark resembling a high-energy-physics counting test: "
                "treated~Pois(3), control~Pois(2), n1=n2=100, with p-value far below the 3e-7 discovery threshold."
            ),
            n_pois2=100,
            n_pois3=100,
            lam_low=2.0,
            lam_high=3.0,
            seed=57,
        ),
        _make_bruteforce_welch_scenario(),
        _make_bruteforce_welch_scenario(
            key="bruteforce_welch_swap1_n22",
            description="Brute-force Welch t-statistic with a one-swap downgrade, n=22.",
            swap_from_extreme=1,
        ),
        _make_bruteforce_welch_scenario(
            key="bruteforce_welch_swap3_n22",
            description="Brute-force Welch t-statistic with a three-swap downgrade, n=22.",
            swap_from_extreme=3,
        ),
    ]
    threshold_tags = {
        "gwas_additive_score_ultra_n100": (5e-8, "gwas_threshold_suite", "GWAS-like additive score", "ultra"),
        "gwas_additive_score_sig_n100": (5e-8, "gwas_threshold_suite", "GWAS-like additive score", "near"),
        "gwas_additive_score_slight_above_n100": (5e-8, "gwas_threshold_suite", "GWAS-like additive score", "slightly_above"),
        "gwas_additive_score_above_n100": (5e-8, "gwas_threshold_suite", "GWAS-like additive score", "above"),
        "poisson_diffmeans_hep_ultra_n200": (3e-7, "hep_threshold_suite", "HEP-like Poisson count test", "ultra"),
        "poisson_diffmeans_hep_sig_n200": (3e-7, "hep_threshold_suite", "HEP-like Poisson count test", "near"),
        "poisson_diffmeans_hep_slight_above_n200": (3e-7, "hep_threshold_suite", "HEP-like Poisson count test", "slightly_above"),
        "poisson_diffmeans_hep_above_n200": (3e-7, "hep_threshold_suite", "HEP-like Poisson count test", "above"),
    }
    for scenario in scenarios:
        tag = threshold_tags.get(scenario.key)
        if tag is not None:
            threshold, setting_key, setting_label, threshold_band = tag
            _attach_application_threshold(
                scenario,
                known_significance_threshold=float(threshold),
                setting_key=str(setting_key),
                setting_label=str(setting_label),
                threshold_band=str(threshold_band),
            )
    return scenarios


def _scenario_metadata(s: ExactScenario) -> dict[str, Any]:
    return {
        "key": s.key,
        "description": s.description,
        "statistic_name": s.statistic_name,
        "tail": s.problem.tail,
        "permutation_p_value_definition": (
            "Right-tail probability from full permutation enumeration over all fixed-size labelings."
        ),
        "n": int(s.problem.n),
        "n_treated": int(s.problem.n_treated),
        "n_control": int(s.problem.n_control),
        "t_obs": float(s.problem.t_obs),
        "exact_method": s.exact_method,
        "exact_p_value": float(s.exact_p_value),
        "tail_hits": int(s.tail_hits),
        "n_permutations": int(s.n_permutations),
        "notes": s.notes,
        "extra": s.extra,
        "portfolio": s.portfolio,
    }


def save_exact_scenarios(
    scenarios: list[ExactScenario],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory '{output_dir}' is not empty. Use overwrite=True to replace files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # On overwrite, prune stale scenario directories that are no longer part of the catalog.
    if overwrite:
        keep_keys = {s.key for s in scenarios}
        for child in output_dir.iterdir():
            if child.is_dir() and child.name not in keep_keys:
                shutil.rmtree(child)

    catalog: list[dict[str, Any]] = []
    for s in scenarios:
        scenario_dir = output_dir / s.key
        scenario_dir.mkdir(parents=True, exist_ok=True)
        np.save(scenario_dir / "X.npy", np.asarray(s.problem.X))
        np.save(scenario_dir / "y_obs.npy", np.asarray(s.problem.y_obs, dtype=np.int8))

        meta = _scenario_metadata(s)
        (scenario_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        catalog.append(meta)

    run_meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_scenarios": len(catalog),
        "scenarios": catalog,
    }
    (output_dir / "catalog.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return output_dir


STATISTIC_REGISTRY = {
    "absolute_sum_difference": absolute_sum_difference,
    "treated_successes": treated_successes,
    "treated_sum": treated_sum,
    "mann_whitney_u": mann_whitney_u,
    "difference_in_means": difference_in_means,
    "t_statistic_welch": t_statistic_welch,
}


def load_saved_exact_scenarios(catalog_path: Path) -> list[ExactScenario]:
    """
    Load scenarios previously written by save_exact_scenarios().
    """
    catalog_path = Path(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    base_dir = catalog_path.parent

    scenarios: list[ExactScenario] = []
    for rec in catalog["scenarios"]:
        key = rec["key"]
        scenario_dir = base_dir / key
        x = np.load(scenario_dir / "X.npy")
        y = np.load(scenario_dir / "y_obs.npy").astype(np.int8)

        stat_name = str(rec["statistic_name"])
        if stat_name not in STATISTIC_REGISTRY:
            raise KeyError(
                f"Unknown statistic_name='{stat_name}' in catalog for scenario '{key}'. "
                f"Known: {sorted(STATISTIC_REGISTRY.keys())}"
            )
        stat_fn = STATISTIC_REGISTRY[stat_name]
        problem = PermutationTestProblem(
            X=x,
            y_obs=y,
            statistic=stat_fn,
            tail=str(rec.get("tail", "right")),
        )

        scenarios.append(
            ExactScenario(
                key=key,
                description=str(rec["description"]),
                problem=problem,
                statistic_name=stat_name,
                exact_method=str(rec["exact_method"]),
                exact_p_value=float(rec["exact_p_value"]),
                tail_hits=int(rec["tail_hits"]),
                n_permutations=int(rec["n_permutations"]),
                notes=str(rec.get("notes", "")),
                extra=dict(rec.get("extra", {})),
                portfolio=dict(rec.get("portfolio", {})),
            )
        )
    return scenarios
