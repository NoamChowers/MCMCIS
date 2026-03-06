from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.exact.linear_statistic_dp import LinearStatisticDPSolver
from perm_pval.stats.two_sample import difference_in_means


def run_poisson_diff_in_means_righttail_simulation(
    *,
    n1: int,
    n2: int,
    lam1: float,
    lam2: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    x1 = rng.poisson(lam=lam1, size=n1)  # first block (typically lower mean)
    x2 = rng.poisson(lam=lam2, size=n2)  # second block (typically higher mean)

    x = np.concatenate([x1, x2]).astype(float)
    # Reverse direction so the right-tail test targets the higher-mean second block.
    y_obs = np.array([0] * n1 + [1] * n2, dtype=np.int8)

    # Right-tailed signed difference in means: mean(group1) - mean(group0).
    problem = PermutationTestProblem(
        X=x,
        y_obs=y_obs,
        statistic=difference_in_means,
        tail="right",
    )
    solver = LinearStatisticDPSolver.from_difference_in_means(
        problem,
        score_scale=1,
        check_statistic_match=True,
    )
    exact = solver.compute()

    metadata = {
        "n1": int(n1),
        "n2": int(n2),
        "lambda1": float(lam1),
        "lambda2": float(lam2),
        "seed": int(seed),
        "sum_pois2_block": int(np.sum(x1)),
        "sum_pois3_block": int(np.sum(x2)),
        "mean_pois2_block": float(np.mean(x1)),
        "mean_pois3_block": float(np.mean(x2)),
        "treated_block": "second_block",
        "observed_stat_diff_in_means": float(problem.t_obs),
        "tail": "right",
        "exact_method": "LinearStatisticDPSolver",
        "exact_p_value": float(exact.p_value),
        "tail_hits": int(exact.tail_hits),
        "n_permutations": int(exact.n_permutations),
        "equivalence_note": "Exact right-tail p-value for signed difference in means.",
    }
    return x, y_obs, metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Poisson two-group data and compute exact permutation p-value for "
            "right-tailed signed difference in means via linear-stat DP."
        )
    )
    parser.add_argument("--n1", type=int, default=100)
    parser.add_argument("--n2", type=int, default=100)
    parser.add_argument("--lam1", type=float, default=2.0)
    parser.add_argument("--lam2", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260228)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/simulations/poisson_diff_in_means_righttail_n100_n100_seed20260228",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory '{output_dir}' is not empty. Use --overwrite to replace files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y_obs, res = run_poisson_diff_in_means_righttail_simulation(
        n1=args.n1,
        n2=args.n2,
        lam1=args.lam1,
        lam2=args.lam2,
        seed=args.seed,
    )

    np.save(output_dir / "X.npy", x)
    np.save(output_dir / "y_obs.npy", y_obs)
    (output_dir / "metadata.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "observed_stat_diff_in_means": res["observed_stat_diff_in_means"],
        "exact_p_value": res["exact_p_value"],
        "tail_hits": res["tail_hits"],
        "n_permutations": res["n_permutations"],
    }, indent=2))


if __name__ == "__main__":
    main()
