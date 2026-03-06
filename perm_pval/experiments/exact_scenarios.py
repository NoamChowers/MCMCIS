from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
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


def treated_successes(x: np.ndarray, y: np.ndarray) -> float:
    """
    Count of successes among treated labels (x must be 0/1).
    """
    return float(int(np.dot(np.asarray(x, dtype=np.int8), np.asarray(y, dtype=np.int8))))


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

    return ExactScenario(
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


def _make_rank_sum_dp_scenario() -> ExactScenario:
    x, y = near_tail_labels_tiny_p_case()
    problem = PermutationTestProblem(X=x, y_obs=y, statistic=mann_whitney_u, tail="right")
    exact = RankSumDPSolver(problem, statistic_type="u").compute()
    return ExactScenario(
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
    return ExactScenario(
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
    return ExactScenario(
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
    return ExactScenario(
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


def build_exact_scenarios() -> list[ExactScenario]:
    scenarios: list[ExactScenario] = [
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
        _make_rank_sum_dp_scenario(),
        _make_linear_dp_scenario(),
        _make_poisson_diffmeans_righttail_scenario(),
        _make_bruteforce_welch_scenario(),
    ]
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
    "treated_successes": treated_successes,
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
            )
        )
    return scenarios
