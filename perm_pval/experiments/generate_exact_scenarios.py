from __future__ import annotations

import argparse
import json
from pathlib import Path

from perm_pval.experiments.exact_scenarios import build_exact_scenarios, save_exact_scenarios


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible exact-permutation scenarios and save X, y_obs, and exact p-values."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/exact_scenarios/v1",
        help="Directory where scenario files and catalog.json are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scenarios = build_exact_scenarios()
    save_exact_scenarios(scenarios, output_dir, overwrite=args.overwrite)

    catalog = json.loads((output_dir / "catalog.json").read_text(encoding="utf-8"))
    print(f"Saved {catalog['n_scenarios']} scenarios to {output_dir}")
    for rec in catalog["scenarios"]:
        print(
            json.dumps(
                {
                    "key": rec["key"],
                    "exact_method": rec["exact_method"],
                    "exact_p_value": rec["exact_p_value"],
                    "tail_hits": rec["tail_hits"],
                    "n_permutations": rec["n_permutations"],
                }
            )
        )


if __name__ == "__main__":
    main()
