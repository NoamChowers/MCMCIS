from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.experiments.notebook_studies import create_timestamped_run_dir


StatisticFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class ShapeScenario:
    key: str
    description: str
    x: np.ndarray
    y_obs: np.ndarray
    statistic_name: str
    statistic: StatisticFn
    expected_shape: str
    notes: str
    exact_tail_hits: int
    exact_p_value: float
    exact_p_note: str

    @property
    def problem(self) -> PermutationTestProblem:
        return PermutationTestProblem(
            X=np.asarray(self.x, dtype=float),
            y_obs=np.asarray(self.y_obs, dtype=np.int8),
            statistic=self.statistic,
            tail="right",
        )


def treated_burden_sum(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(np.asarray(x, dtype=float), np.asarray(y, dtype=np.int8)))


def treated_group_median(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=np.int8)
    return float(np.median(x_arr[y_arr == 1]))


def cluster_polarity_score(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=np.int8)
    treated_mean = float(np.mean(x_arr[y_arr == 1]))
    polarity = 1.0 if treated_mean >= 0.0 else -1.0
    return float(polarity + 0.001 * treated_mean)


def _sample_uniform_statistics(problem: PermutationTestProblem, *, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_samples), dtype=float)
    for i in range(int(n_samples)):
        y = problem.sample_uniform_labels(rng)
        vals[i] = problem.compute_stat(y)
    return vals


def _sample_skewness(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    centered = vals - float(np.mean(vals))
    scale = float(np.std(vals))
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    return float(np.mean(centered ** 3) / (scale ** 3))


def _summary(values: np.ndarray) -> dict[str, float | int | list[float]]:
    vals = np.asarray(values, dtype=float)
    return {
        "n_samples": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "skewness": float(_sample_skewness(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n_unique_rounded_4dp": int(np.unique(np.round(vals, 4)).size),
        "quantiles": [float(v) for v in np.quantile(vals, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])],
    }


def _build_shape_scenarios() -> list[ShapeScenario]:
    skew_n = 85
    skew_n_treated = 6
    skew_x = np.asarray([0] * (skew_n - 7) + [1, 2, 2, 2, 4, 8, 16], dtype=float)
    skew_y = np.zeros(skew_x.size, dtype=np.int8)
    skew_y[[skew_n - 7 + idx for idx in [0, 2, 3, 4, 5, 6]]] = 1

    bimodal_x = np.asarray(
        list(-np.arange(113, 99, -1)) + list(np.arange(100, 114)),
        dtype=float,
    )
    bimodal_y = np.zeros(bimodal_x.size, dtype=np.int8)
    bimodal_y[[13] + list(range(14, 28))] = 1
    bimodal_y[15] = 0

    skew_n_perm = comb(skew_n, skew_n_treated)
    bimodal_n_perm = comb(int(bimodal_x.size), int(np.sum(bimodal_y)))

    return [
        ShapeScenario(
            key="skew_sparse_burden_sum_tiny",
            description="Sparse burden-sum statistic with many zeros and seven rare positive scores",
            x=skew_x,
            y_obs=skew_y,
            statistic_name="treated_burden_sum",
            statistic=treated_burden_sum,
            expected_shape="highly right-skewed",
            notes=(
                "Designed as a rare-burden score with 78 zero observations and seven rare positive scores. "
                "The observed labeling selects six of the seven rare positives, giving a small but non-singleton right tail."
            ),
            exact_tail_hits=4,
            exact_p_value=4.0 / skew_n_perm,
            exact_p_note="Right-tail threshold equals treated sum 33; exactly four treated subsets attain at least this score.",
        ),
        ShapeScenario(
            key="bimodal_cluster_polarity_tiny",
            description="Cluster-polarity score on two sharply separated clusters",
            x=bimodal_x,
            y_obs=bimodal_y,
            statistic_name="cluster_polarity_score",
            statistic=cluster_polarity_score,
            expected_shape="bimodal",
            notes=(
                "Designed as a deliberately bimodal cluster-polarity statistic: the sign term creates two dominant "
                "modes, and the small mean term breaks ties within each mode. The observed labeling is one of the "
                "top positive-cluster subsets, giving a small but non-singleton right tail."
            ),
            exact_tail_hits=4,
            exact_p_value=4.0 / bimodal_n_perm,
            exact_p_note="Right-tail threshold corresponds to treated-sum 1290; exactly four treated subsets attain at least this score.",
        ),
    ]


def _plot_histogram(
    scenario: ShapeScenario,
    values: np.ndarray,
    *,
    save_path: Path,
) -> None:
    vals = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.hist(
        vals,
        bins=60,
        density=True,
        alpha=0.55,
        color="#4e79a7",
        edgecolor="none",
    )
    ax.axvline(float(np.mean(vals)), color="#f28e2b", linestyle="--", linewidth=1.3, label="sample mean")
    ax.set_title(f"{scenario.description}\nExpected shape: {scenario.expected_shape}")
    ax.set_xlabel(scenario.statistic_name)
    ax.set_ylabel("density")
    ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_figure(
    scenarios: list[ShapeScenario],
    samples_by_key: dict[str, np.ndarray],
    *,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    for ax, scenario in zip(axes, scenarios):
        vals = np.asarray(samples_by_key[scenario.key], dtype=float)
        ax.hist(
            vals,
            bins=60,
            density=True,
            alpha=0.55,
            color="#4e79a7",
            edgecolor="none",
        )
        ax.axvline(float(np.mean(vals)), color="#f28e2b", linestyle="--", linewidth=1.3)
        ax.set_title(f"{scenario.key}\n{scenario.expected_shape}")
        ax.set_xlabel(scenario.statistic_name)
        ax.set_ylabel("density")
    fig.suptitle("Permutation-null shape stress scenarios")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_irregular_shape_diagnostics(
    *,
    n_samples: int = 100_000,
    seed: int = 20260409,
    output_root: Path | None = None,
) -> dict[str, object]:
    scenarios = _build_shape_scenarios()
    root = Path(output_root) if output_root is not None else Path("results") / "irregular_shape_diagnostics"
    run_dir = create_timestamped_run_dir(root, "iid_shape")

    results: list[dict[str, object]] = []
    samples_by_key: dict[str, np.ndarray] = {}
    for idx, scenario in enumerate(scenarios):
        problem = scenario.problem
        values = _sample_uniform_statistics(problem, n_samples=int(n_samples), seed=int(seed + 1_000 * idx))
        samples_by_key[scenario.key] = values
        scenario_dir = run_dir / scenario.key
        _plot_histogram(scenario, values, save_path=scenario_dir / "iid_histogram.png")
        results.append(
            {
                "key": scenario.key,
                "description": scenario.description,
                "expected_shape": scenario.expected_shape,
                "statistic_name": scenario.statistic_name,
                "n": int(problem.n),
                "n_treated": int(problem.n_treated),
                "x_values": [float(v) for v in np.asarray(scenario.x, dtype=float)],
                "y_obs": [int(v) for v in np.asarray(scenario.y_obs, dtype=np.int8)],
                "notes": scenario.notes,
                "exact_tail_hits": int(scenario.exact_tail_hits),
                "n_permutations": int(comb(int(problem.n), int(problem.n_treated))),
                "exact_p_value": float(scenario.exact_p_value),
                "exact_p_note": scenario.exact_p_note,
                "sample_summary": _summary(values),
            }
        )

    _plot_combined_figure(scenarios, samples_by_key, save_path=run_dir / "combined_histograms.png")
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {
        "run_dir": run_dir,
        "scenarios": results,
    }


if __name__ == "__main__":
    payload = run_irregular_shape_diagnostics()
    print(json.dumps({"run_dir": str(payload["run_dir"])}, indent=2))
