from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
from perm_pval.exact.brute_force import iter_fixed_group_labelings
from perm_pval.methods.samc import run_samc
from perm_pval.stats.two_sample import difference_in_means


def build_demo_problem() -> PermutationTestProblem:
    x = np.arange(10, dtype=float)
    y_obs = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    return PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )


def bin_index(value: float, bin_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(bin_edges, value, side="right") - 1)
    return int(np.clip(idx, 0, bin_edges.size - 2))


def exact_bin_masses(
    problem: PermutationTestProblem,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = np.asarray(
        [
            problem.compute_stat(y)
            for y in iter_fixed_group_labelings(problem.n, problem.n_treated)
        ],
        dtype=float,
    )
    bins = np.asarray([bin_index(value, bin_edges) for value in stats], dtype=np.int64)
    counts = np.bincount(bins, minlength=bin_edges.size - 1)
    masses = counts / float(np.sum(counts))
    return stats, counts, masses


def centered(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr - float(np.mean(arr))


def normalized_from_log(log_values: np.ndarray) -> np.ndarray:
    shift = float(np.max(log_values))
    values = np.exp(log_values - shift)
    return values / float(np.sum(values))


def make_figure(
    *,
    problem: PermutationTestProblem,
    samc,
    exact_counts: np.ndarray,
    exact_masses: np.ndarray,
    output: Path,
) -> None:
    target = np.asarray(samc.target_visitation, dtype=float)
    if np.any(exact_counts == 0):
        empty = np.flatnonzero(exact_counts == 0) + 1
        raise ValueError(
            "The demo bins include unreachable exact-null bins "
            f"{empty.tolist()}; rerun with fewer bins."
        )

    theta_star = centered(np.log(exact_masses) - np.log(target))
    theta_samc = centered(samc.theta_final)
    tilted_by_samc = normalized_from_log(np.log(exact_masses) - theta_samc)

    bin_ids = np.arange(exact_masses.size)
    tick_labels = [str(i + 1) for i in bin_ids]
    tick_labels[-1] = "tail"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        "SAMC as a discrete exponential tilt over statistic bins",
        fontsize=14,
    )

    ax = axes[0, 0]
    width = 0.38
    ax.bar(
        bin_ids - width / 2,
        exact_masses,
        width=width,
        color="#4477AA",
        label="exact null mass",
    )
    ax.bar(
        bin_ids + width / 2,
        target,
        width=width,
        color="#CCBB44",
        label="SAMC target",
    )
    ax.set_title("Original bin masses")
    ax.set_ylabel("probability")
    ax.set_xticks(bin_ids, tick_labels)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(
        bin_ids,
        theta_star,
        marker="o",
        linewidth=2.0,
        color="#AA3377",
        label=r"ideal centered $\log(\omega_i / p_i)$",
    )
    ax.plot(
        bin_ids,
        theta_samc,
        marker="s",
        linewidth=1.8,
        color="#228833",
        label=r"SAMC learned $\theta_i$",
    )
    ax.axhline(0.0, color="0.75", linewidth=1.0)
    ax.set_title("Learned discrete tilt")
    ax.set_ylabel("centered log weight")
    ax.set_xticks(bin_ids, tick_labels)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.bar(
        bin_ids - width / 2,
        tilted_by_samc,
        width=width,
        color="#66CCEE",
        label=r"exact mass reweighted by $\exp(-\theta_i)$",
    )
    ax.bar(
        bin_ids + width / 2,
        target,
        width=width,
        color="#CCBB44",
        label="SAMC target",
    )
    ax.set_title("Distribution implied by the learned tilt")
    ax.set_xlabel("SAMC bin")
    ax.set_ylabel("probability")
    ax.set_xticks(bin_ids, tick_labels)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.bar(
        bin_ids - width / 2,
        samc.visitation_frequency,
        width=width,
        color="#EE7733",
        label="empirical SAMC visits",
    )
    ax.bar(
        bin_ids + width / 2,
        target,
        width=width,
        color="#CCBB44",
        label="SAMC target",
    )
    ax.set_title("Empirical SAMC visitation")
    ax.set_xlabel("SAMC bin")
    ax.set_ylabel("probability")
    ax.set_xticks(bin_ids, tick_labels)
    ax.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize SAMC as a discrete exponential tilt using the repository's "
            "SAMC sampler on a small exactly enumerable permutation problem."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/samc_discrete_tilt_visualization.png"),
        help="Path for the saved PNG figure.",
    )
    parser.add_argument("--n-steps", type=int, default=50_000)
    parser.add_argument("--burn-in", type=int, default=10_000)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--t0", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem = build_demo_problem()

    all_stats = np.asarray(
        [
            problem.compute_stat(y)
            for y in iter_fixed_group_labelings(problem.n, problem.n_treated)
        ],
        dtype=float,
    )
    bin_edges = np.concatenate(
        [
            np.linspace(float(np.min(all_stats)), float(problem.t_obs), args.n_bins),
            np.asarray([np.inf], dtype=float),
        ]
    )

    samc = run_samc(
        problem,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        bin_edges=bin_edges,
        seed=args.seed,
        t0=args.t0,
        init="random",
        trace_every=args.trace_every,
    )
    _, exact_counts, exact_masses = exact_bin_masses(problem, samc.bin_edges)
    make_figure(
        problem=problem,
        samc=samc,
        exact_counts=exact_counts,
        exact_masses=exact_masses,
        output=args.output,
    )

    print(f"Saved {args.output}")
    print(f"Exact tail mass: {exact_masses[-1]:.6f}")
    print(f"SAMC p-hat: {samc.estimate:.6f}")
    print(f"Acceptance rate: {samc.acceptance_rate:.3f}")
    print(
        "Max absolute relative visitation error: "
        f"{samc.max_abs_relative_frequency_error:.2f}%"
    )


if __name__ == "__main__":
    main()
