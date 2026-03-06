from __future__ import annotations

import argparse
import json
from pathlib import Path

from perm_pval.experiments.config import ExperimentConfig
from perm_pval.experiments.run_single import run_single_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sweep over effect sizes and seeds.")
    parser.add_argument("--effect-sizes", type=str, default="0.0,0.5,1.0,1.5")
    parser.add_argument("--seeds", type=str, default="101,102,103")
    parser.add_argument("--output", type=str, default="results/sweep.jsonl")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    effect_sizes = [float(x.strip()) for x in args.effect_sizes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for effect in effect_sizes:
            for seed in seeds:
                config = ExperimentConfig()
                config.simulation.effect_size = effect
                config.simulation.seed = seed
                result = run_single_experiment(config)
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
