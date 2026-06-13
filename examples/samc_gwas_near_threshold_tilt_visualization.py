from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPL_CONFIG_DIR = REPO_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.experiments.exact_scenarios import (
    ExactScenario,
    load_saved_exact_scenarios,
)
from perm_pval.methods.beta_tuning import iid_pilot_statistics
from perm_pval.methods.samc import run_samc


def load_scenario(catalog_path: Path, scenario_key: str) -> ExactScenario:
    scenarios = load_saved_exact_scenarios(catalog_path)
    by_key = {scenario.key: scenario for scenario in scenarios}
    if scenario_key not in by_key:
        raise KeyError(
            f"Unknown scenario key {scenario_key!r}. Known keys: {sorted(by_key)}"
        )
    return by_key[scenario_key]


def bin_index(value: float, bin_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(bin_edges, value, side="right") - 1)
    return int(np.clip(idx, 0, bin_edges.size - 2))


def make_bin_edges(
    problem: PermutationTestProblem,
    *,
    n_bins: int,
    lambda_min: float,
) -> np.ndarray:
    finite_edges = np.linspace(float(lambda_min), float(problem.t_obs), n_bins, dtype=float)
    return np.concatenate([finite_edges, np.asarray([np.inf], dtype=float)])


def pilot_lambda_min(
    problem: PermutationTestProblem,
    *,
    n_samples: int,
    seed: int,
) -> float:
    pilot_t = iid_pilot_statistics(problem, n_samples=n_samples, seed=seed)
    lambda_min = float(np.min(pilot_t))
    if lambda_min >= problem.t_obs:
        lambda_min = float(problem.t_obs - 1.0)
    return lambda_min


def weighted_sum_distribution(scores: np.ndarray, n_treated: int) -> dict[int, int]:
    scores_int = np.asarray(np.round(scores), dtype=np.int64)
    if not np.allclose(scores, scores_int, atol=1e-12, rtol=0.0):
        raise ValueError("This exact-bin helper expects integer-valued scores.")

    states = [defaultdict(int) for _ in range(n_treated + 1)]
    states[0][0] = 1
    for weight in scores_int.tolist():
        for k in range(n_treated, 0, -1):
            previous = states[k - 1]
            if not previous:
                continue
            current = states[k]
            for prev_sum, count in previous.items():
                current[prev_sum + weight] += count
    return dict(states[n_treated])


def exact_gwas_bin_masses(
    scenario: ExactScenario,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    if scenario.statistic_name != "treated_sum":
        raise ValueError(
            "This visualization expects a treated_sum/additive-score scenario; "
            f"got statistic_name={scenario.statistic_name!r}."
        )

    problem = scenario.problem
    dist = weighted_sum_distribution(np.asarray(problem.X, dtype=float), problem.n_treated)
    n_perm = comb(problem.n, problem.n_treated)
    counts = np.zeros(bin_edges.size - 1, dtype=object)
    for score_sum, count in dist.items():
        counts[bin_index(float(score_sum), bin_edges)] += int(count)

    masses = np.asarray([float(count / n_perm) for count in counts], dtype=float)
    return np.asarray(counts, dtype=object), masses, dist


def centered_on_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = arr.copy()
    out[mask] = arr[mask] - float(np.mean(arr[mask]))
    return out


def normalized_reachable(values: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    total = float(np.sum(values[reachable]))
    if total > 0.0:
        out[reachable] = values[reachable] / total
    return out


def safe_log10(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    mask = values > 0.0
    out[mask] = np.log10(values[mask])
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_bin_csv(
    path: Path,
    *,
    bin_edges: np.ndarray,
    exact_masses: np.ndarray,
    exact_counts: np.ndarray,
    theta: np.ndarray,
    tilted_mass: np.ndarray,
    visit_freq: np.ndarray,
    target: np.ndarray,
    reachable_target: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                "left_edge",
                "right_edge",
                "exact_count",
                "exact_mass",
                "reachable",
                "theta",
                "exp_minus_theta",
                "tilted_mass_reachable_normalized",
                "visit_frequency",
                "samc_target_unconditional",
                "samc_target_reachable_conditional",
            ],
        )
        writer.writeheader()
        for idx in range(exact_masses.size):
            writer.writerow(
                {
                    "bin": idx + 1,
                    "left_edge": float(bin_edges[idx]),
                    "right_edge": (
                        "inf" if np.isinf(bin_edges[idx + 1]) else float(bin_edges[idx + 1])
                    ),
                    "exact_count": str(exact_counts[idx]),
                    "exact_mass": float(exact_masses[idx]),
                    "reachable": bool(exact_masses[idx] > 0.0),
                    "theta": float(theta[idx]),
                    "exp_minus_theta": float(np.exp(-theta[idx])),
                    "tilted_mass_reachable_normalized": float(tilted_mass[idx]),
                    "visit_frequency": float(visit_freq[idx]),
                    "samc_target_unconditional": float(target[idx]),
                    "samc_target_reachable_conditional": float(reachable_target[idx]),
                }
            )


def plot_tilt_figure(
    *,
    scenario: ExactScenario,
    samc,
    exact_masses: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    target = np.asarray(samc.target_visitation, dtype=float)
    theta = np.asarray(samc.theta_final, dtype=float)
    reachable = exact_masses > 0.0
    n_reachable = int(np.sum(reachable))
    if n_reachable == 0:
        raise ValueError("No reachable exact bins under this partition.")

    reachable_target = normalized_reachable(target, reachable)
    theta_centered = centered_on_mask(theta, reachable)

    theta_star = np.full_like(theta, np.nan, dtype=float)
    theta_star[reachable] = np.log(exact_masses[reachable]) - np.log(
        reachable_target[reachable]
    )
    theta_star = centered_on_mask(theta_star, reachable)

    tilted_unnormalized = np.zeros_like(exact_masses, dtype=float)
    tilted_unnormalized[reachable] = exact_masses[reachable] * np.exp(-theta[reachable])
    tilted_mass = normalized_reachable(tilted_unnormalized, reachable)

    log_omega = safe_log10(exact_masses)
    log_tilt_factor = np.full_like(theta, np.nan, dtype=float)
    log_tilt_factor[reachable] = -theta[reachable] / np.log(10.0)
    log_tilted = safe_log10(tilted_mass)

    log_omega_centered = centered_on_mask(log_omega, reachable)
    log_tilt_factor_centered = centered_on_mask(log_tilt_factor, reachable)
    log_tilted_centered = centered_on_mask(log_tilted, reachable)

    x = np.arange(1, exact_masses.size + 1)
    colors = {
        "base": "#4477AA",
        "target": "#CCBB44",
        "samc": "#228833",
        "tilt": "#EE7733",
        "residual": "#332288",
        "empty": "#D0D0D0",
    }

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [
            ["mass", "theta"],
            ["cancel", "tilted"],
            ["visits", "trace"],
        ]
    )

    title = "SAMC tilt in the GWAS near-threshold scenario"
    subtitle = (
        f"{scenario.key}: exact p={scenario.exact_p_value:.3e}, "
        f"T_obs={scenario.problem.t_obs:.0f}, "
        f"steps={samc.n_steps:,}, bins={exact_masses.size}, "
        f"reachable bins={n_reachable}"
    )
    fig.suptitle(title + "\n" + subtitle, fontsize=14)

    ax = axes["mass"]
    floor = max(float(np.min(exact_masses[reachable])) * 0.25, 1e-14)
    ax.bar(x[reachable], exact_masses[reachable], color=colors["base"], width=0.9, label=r"exact $\omega_i$")
    ax.bar(x[~reachable], np.full(np.sum(~reachable), floor), color=colors["empty"], width=0.9, label="structurally empty")
    ax.axhline(float(target[0]), color=colors["target"], linestyle="--", linewidth=1.2, label="requested 1/100 target")
    ax.axhline(float(reachable_target[reachable][0]), color=colors["residual"], linestyle=":", linewidth=1.4, label="attainable reachable target")
    ax.set_yscale("log")
    ax.set_title("Original exact bin mass")
    ax.set_xlabel("SAMC bin index")
    ax.set_ylabel("probability, log scale")
    ax.legend(frameon=False, fontsize=9)

    ax = axes["theta"]
    ax.plot(x[reachable], theta_star[reachable], color="#AA3377", marker="o", markersize=3.0, linewidth=1.4, label=r"ideal centered $\log(\omega_i / p_i)$")
    ax.plot(x[reachable], theta_centered[reachable], color=colors["samc"], marker="s", markersize=2.8, linewidth=1.3, label=r"SAMC learned $\theta_i$")
    ax.scatter(x[~reachable], theta_centered[~reachable], color=colors["empty"], s=10, label="empty-bin learned theta")
    ax.axhline(0.0, color="0.75", linewidth=1.0)
    ax.set_title("Learned bias weights")
    ax.set_xlabel("SAMC bin index")
    ax.set_ylabel("centered log weight")
    ax.legend(frameon=False, fontsize=9)

    ax = axes["cancel"]
    ax.plot(x[reachable], log_omega_centered[reachable], color=colors["base"], marker="o", markersize=3.0, linewidth=1.2, label=r"centered $\log_{10}\omega_i$")
    ax.plot(x[reachable], log_tilt_factor_centered[reachable], color=colors["tilt"], marker="s", markersize=2.8, linewidth=1.2, label=r"centered $\log_{10}\exp(-\theta_i)$")
    ax.plot(x[reachable], log_tilted_centered[reachable], color=colors["samc"], marker="d", markersize=2.8, linewidth=1.2, label=r"centered $\log_{10}(\omega_i e^{-\theta_i})$")
    ax.axhline(0.0, color="0.75", linewidth=1.0)
    ax.set_title("The exponential tilt cancels log bin mass")
    ax.set_xlabel("SAMC bin index")
    ax.set_ylabel("centered log10 scale")
    ax.legend(frameon=False, fontsize=9)

    ax = axes["tilted"]
    ax.bar(x[reachable], tilted_mass[reachable], color="#66CCEE", width=0.9, label=r"normalized $\omega_i e^{-\theta_i}$")
    ax.plot(x[reachable], reachable_target[reachable], color=colors["target"], linewidth=1.4, label="reachable target")
    ax.set_title("Distribution implied by learned tilt")
    ax.set_xlabel("SAMC bin index")
    ax.set_ylabel("probability over reachable bins")
    ax.legend(frameon=False, fontsize=9)

    ax = axes["visits"]
    ax.bar(x, samc.visitation_frequency, color=colors["tilt"], width=0.9, label="empirical SAMC visits")
    ax.plot(x[reachable], reachable_target[reachable], color=colors["residual"], linewidth=1.4, label="reachable target")
    ax.axhline(float(target[0]), color=colors["target"], linestyle="--", linewidth=1.2, label="requested 1/100")
    ax.set_title("Post-burn-in visitation")
    ax.set_xlabel("SAMC bin index")
    ax.set_ylabel("frequency")
    ax.legend(frameon=False, fontsize=9)

    ax = axes["trace"]
    theta_trace = np.asarray(samc.theta_trace, dtype=float)
    if theta_trace.size:
        centered_trace = theta_trace - np.nanmean(theta_trace[:, reachable], axis=1, keepdims=True)
        lo, hi = np.nanpercentile(centered_trace, [2, 98])
        image = ax.imshow(
            centered_trace.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=lo,
            vmax=hi,
        )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(r"centered $\theta_i$")
    ax.set_title(r"$\theta_i$ trace heatmap")
    ax.set_xlabel("trace checkpoint")
    ax.set_ylabel("SAMC bin index")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    positive_tilt_error = float(
        np.max(np.abs(tilted_mass[reachable] - reachable_target[reachable]))
    )
    positive_visit_error = float(
        np.max(np.abs(samc.visitation_frequency[reachable] - reachable_target[reachable]))
    )
    return {
        "n_reachable_bins": n_reachable,
        "n_structurally_empty_bins": int(np.sum(~reachable)),
        "max_abs_tilted_mass_error_reachable": positive_tilt_error,
        "max_abs_visit_error_reachable": positive_visit_error,
        "tilted_mass": tilted_mass.tolist(),
        "reachable_target": reachable_target.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a realistic SAMC tilt visualization for the GWAS near-threshold "
            "exact scenario."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("results/exact_scenarios/v1/catalog.json"),
        help="Saved exact-scenario catalog.",
    )
    parser.add_argument(
        "--scenario-key",
        default="gwas_additive_score_sig_n100",
        help="Scenario key from the saved exact-scenario catalog.",
    )
    parser.add_argument("--n-steps", type=int, default=1_000_000)
    parser.add_argument("--burn-in", type=int, default=200_000)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--t0", type=float, default=1_000.0)
    parser.add_argument("--trace-every", type=int, default=1_000)
    parser.add_argument("--proposal-size", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-min-pilot", type=int, default=10_000)
    parser.add_argument("--lambda-min-seed", type=int, default=42)
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=None,
        help="Override lambda_min. Defaults to the standard pilot minimum.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/samc_gwas_near_threshold_tilt.png"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/samc_gwas_near_threshold_tilt_summary.json"),
    )
    parser.add_argument(
        "--bin-csv",
        type=Path,
        default=Path("results/samc_gwas_near_threshold_tilt_bins.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive.")
    if not (0 <= args.burn_in < args.n_steps):
        raise ValueError("--burn-in must satisfy 0 <= burn-in < n-steps.")

    scenario = load_scenario(args.catalog, args.scenario_key)
    problem = scenario.problem
    lambda_min = (
        float(args.lambda_min)
        if args.lambda_min is not None
        else pilot_lambda_min(
            problem,
            n_samples=int(args.lambda_min_pilot),
            seed=int(args.lambda_min_seed),
        )
    )
    bin_edges = make_bin_edges(problem, n_bins=int(args.n_bins), lambda_min=lambda_min)
    exact_counts, exact_masses, score_distribution = exact_gwas_bin_masses(
        scenario,
        bin_edges,
    )

    print(
        f"Running {args.scenario_key} with {args.n_steps:,} SAMC steps, "
        f"{args.n_bins} bins, lambda_min={lambda_min:.6g}"
    )
    samc = run_samc(
        problem,
        n_steps=int(args.n_steps),
        burn_in=int(args.burn_in),
        bin_edges=bin_edges,
        seed=int(args.seed),
        t0=float(args.t0),
        init="random",
        trace_every=int(args.trace_every),
        proposal_size=float(args.proposal_size),
    )

    plot_summary = plot_tilt_figure(
        scenario=scenario,
        samc=samc,
        exact_masses=exact_masses,
        output=args.output,
    )

    reachable_target = np.asarray(plot_summary["reachable_target"], dtype=float)
    tilted_mass = np.asarray(plot_summary["tilted_mass"], dtype=float)
    write_bin_csv(
        args.bin_csv,
        bin_edges=samc.bin_edges,
        exact_masses=exact_masses,
        exact_counts=exact_counts,
        theta=samc.theta_final,
        tilted_mass=tilted_mass,
        visit_freq=samc.visitation_frequency,
        target=samc.target_visitation,
        reachable_target=reachable_target,
    )

    summary = {
        "scenario_key": scenario.key,
        "description": scenario.description,
        "exact_p_value": float(scenario.exact_p_value),
        "known_significance_threshold": scenario.extra.get("known_significance_threshold"),
        "tail_hits": int(scenario.tail_hits),
        "n_permutations": int(scenario.n_permutations),
        "t_obs": float(problem.t_obs),
        "score_support_min": int(min(score_distribution)),
        "score_support_max": int(max(score_distribution)),
        "score_support_size": int(len(score_distribution)),
        "n_steps": int(samc.n_steps),
        "burn_in": int(samc.burn_in),
        "n_bins": int(exact_masses.size),
        "lambda_min": float(lambda_min),
        "acceptance_rate": float(samc.acceptance_rate),
        "samc_estimate": float(samc.estimate),
        "samc_estimate_no_empty_bin_correction": float(
            samc.estimate_no_empty_bin_correction
        ),
        "samc_max_abs_relative_frequency_error_reported": float(
            samc.max_abs_relative_frequency_error
        ),
        "n_empty_bins_after_burn_in": int(samc.empty_bin_indices.size),
        **{
            key: value
            for key, value in plot_summary.items()
            if key not in {"tilted_mass", "reachable_target"}
        },
        "output": str(args.output),
        "bin_csv": str(args.bin_csv),
    }
    write_json(args.summary_json, summary)

    print(f"Saved figure: {args.output}")
    print(f"Saved summary: {args.summary_json}")
    print(f"Saved per-bin data: {args.bin_csv}")
    print(f"Exact p-value: {scenario.exact_p_value:.6e}")
    print(f"SAMC p-hat: {samc.estimate:.6e}")
    print(f"Acceptance rate: {samc.acceptance_rate:.3f}")
    print(
        "Reachable bins / structurally empty bins: "
        f"{summary['n_reachable_bins']} / {summary['n_structurally_empty_bins']}"
    )
    print(
        "Max reachable-bin |tilted mass - target|: "
        f"{summary['max_abs_tilted_mass_error_reachable']:.4f}"
    )
    print(
        "Max reachable-bin |visit frequency - target|: "
        f"{summary['max_abs_visit_error_reachable']:.4f}"
    )


if __name__ == "__main__":
    main()
