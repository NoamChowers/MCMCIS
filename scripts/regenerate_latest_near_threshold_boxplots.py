#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcmcis-matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from perm_pval.experiments.notebook_studies import (
    load_cross_method_saved_output,
    plot_near_threshold_checkpoint_boxplots,
)


DEFAULT_SCENARIOS: tuple[str, ...] = (
    "gwas_additive_score_sig_n100",
    "poisson_diffmeans_hep_sig_n200",
)

METHOD_LABELS: dict[str, str] = {
    "samc_swap_10pct": "SAMC",
    "mcmcis_per_seed_pilot": "MCMC-IS",
    "oracle_mcmcis": "Oracle MCMC-IS",
}

METHOD_COLORS: dict[str, str] = {
    "samc_swap_10pct": "#4c8c77",
    "mcmcis_per_seed_pilot": "#c48a3a",
    "oracle_mcmcis": "#b04a5a",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _read_csv_records(path: Path, *, label: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        df["label"] = label
    else:
        df["label"] = label
    if "replicate" not in df.columns:
        if "run_seed" in df.columns:
            df["replicate"] = df["run_seed"]
        elif "pilot_seed" in df.columns:
            df["replicate"] = df["pilot_seed"]
    return df.to_dict(orient="records")


def _oracle_records_for_scenario(saved_run_dir: Path, scenario_key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    saved = load_cross_method_saved_output(saved_run_dir / scenario_key)
    metadata = dict(saved["metadata"])
    threshold = metadata.get("scenario_portfolio", {}).get("known_significance_threshold")
    records: list[dict[str, Any]] = []
    for row in saved["records"]:
        if str(row.get("label")) != "oracle_mcmcis":
            continue
        enriched = dict(row)
        enriched["scenario"] = scenario_key
        enriched["exact_p"] = float(metadata["exact_p"])
        enriched["known_significance_threshold"] = float(threshold)
        records.append(enriched)
    return records, metadata


def _scenario_metadata(saved_run_dir: Path, scenario_key: str) -> dict[str, Any]:
    with (saved_run_dir / scenario_key / "metadata.json").open() as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=True) + "\n")


def _plot_set(
    *,
    records_by_scenario: dict[str, list[dict[str, Any]]],
    metadata_by_scenario: dict[str, dict[str, Any]],
    method_order: tuple[str, ...],
    output_dir: Path,
    checkpoints: tuple[int, ...],
    manifest_extra: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}
    tables: dict[str, str] = {}
    table_tex: dict[str, str] = {}
    all_counts: list[dict[str, Any]] = []

    for scenario_key in DEFAULT_SCENARIOS:
        metadata = metadata_by_scenario[scenario_key]
        records = records_by_scenario[scenario_key]
        threshold = float(metadata["scenario_portfolio"]["known_significance_threshold"])
        plot_path = output_dir / f"{scenario_key}_checkpoint_boxplots.png"
        table_path = output_dir / f"{scenario_key}_checkpoint_table.png"
        table_tex_path = output_dir / f"{scenario_key}_checkpoint_table.tex"
        count_rows = plot_near_threshold_checkpoint_boxplots(
            records,
            scenario_name=str(metadata["scenario_display"]),
            scenario_key=scenario_key,
            exact_p=float(metadata["exact_p"]),
            known_significance_threshold=threshold,
            checkpoints=list(checkpoints),
            method_order=list(method_order),
            method_labels=METHOD_LABELS,
            method_colors=METHOD_COLORS,
            n_control=int(metadata["n_control"]),
            n_treated=int(metadata["n_treated"]),
            threshold_count="above",
            save_path=plot_path,
            table_save_path=table_path,
            table_tex_save_path=table_tex_path,
        )
        plots[scenario_key] = str(plot_path)
        tables[scenario_key] = str(table_path)
        table_tex[scenario_key] = str(table_tex_path)
        for row in count_rows:
            enriched = dict(row)
            enriched["scenario"] = scenario_key
            enriched["scenario_display"] = str(metadata["scenario_display"])
            enriched["threshold_band"] = str(metadata["scenario_portfolio"].get("threshold_band", ""))
            all_counts.append(enriched)

    counts_path = output_dir / "threshold_crossing_counts.json"
    _write_json(counts_path, all_counts)
    manifest = {
        **manifest_extra,
        "checkpoints": [int(v) for v in checkpoints],
        "method_order": list(method_order),
        "method_labels": METHOD_LABELS,
        "plots": plots,
        "tables": tables,
        "table_tex": table_tex,
        "counts_path": str(counts_path),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Regenerate near-threshold boxplots from latest SAMC and MCMC-IS runs.")
    parser.add_argument(
        "--samc-csv",
        type=Path,
        default=root / "results" / "samc_swap_10pct" / "near_threshold_samc_swap_10pct_80x5m_sorted.csv",
    )
    parser.add_argument(
        "--mcmcis-csv",
        type=Path,
        default=root
        / "results"
        / "mcmcis_per_seed_pilot"
        / "near_threshold_mcmcis_per_seed_200k_pilot_80x5m_sorted.csv",
    )
    parser.add_argument(
        "--saved-cross-method-run",
        type=Path,
        default=root / "results" / "cross_method_notebook" / "20260529_185422_cross_method_oracle_beta_best_compare",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=root / "results" / "samc_swap_10pct",
    )
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1_000_000, 2_500_000, 5_000_000])
    parser.add_argument("--skip-oracle-variant", action="store_true", default=False)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    checkpoints = tuple(int(v) for v in args.checkpoints)

    samc_records = _read_csv_records(Path(args.samc_csv), label="samc_swap_10pct")
    mcmcis_records = _read_csv_records(Path(args.mcmcis_csv), label="mcmcis_per_seed_pilot")

    metadata_by_scenario = {
        scenario_key: _scenario_metadata(Path(args.saved_cross_method_run), scenario_key)
        for scenario_key in DEFAULT_SCENARIOS
    }

    latest_records: dict[str, list[dict[str, Any]]] = {}
    for scenario_key in DEFAULT_SCENARIOS:
        latest_records[scenario_key] = [
            row
            for row in samc_records + mcmcis_records
            if str(row.get("scenario")) == scenario_key
        ]

    latest_manifest = _plot_set(
        records_by_scenario=latest_records,
        metadata_by_scenario=metadata_by_scenario,
        method_order=("samc_swap_10pct", "mcmcis_per_seed_pilot"),
        output_dir=Path(args.output_base) / "near_threshold_visualizations_latest_samc_mcmcis",
        checkpoints=checkpoints,
        manifest_extra={
            "mode": "latest_samc_mcmcis",
            "source_samc_csv": str(Path(args.samc_csv)),
            "source_mcmcis_csv": str(Path(args.mcmcis_csv)),
        },
    )

    manifests = {"latest_samc_mcmcis": latest_manifest}
    if not bool(args.skip_oracle_variant):
        cross_records: dict[str, list[dict[str, Any]]] = {}
        for scenario_key in DEFAULT_SCENARIOS:
            oracle_records, _ = _oracle_records_for_scenario(Path(args.saved_cross_method_run), scenario_key)
            cross_records[scenario_key] = list(latest_records[scenario_key]) + oracle_records
        manifests["latest_cross_method_with_oracle"] = _plot_set(
            records_by_scenario=cross_records,
            metadata_by_scenario=metadata_by_scenario,
            method_order=("samc_swap_10pct", "mcmcis_per_seed_pilot", "oracle_mcmcis"),
            output_dir=Path(args.output_base) / "near_threshold_visualizations_latest_cross_method",
            checkpoints=checkpoints,
            manifest_extra={
                "mode": "latest_samc_mcmcis_plus_saved_oracle",
                "source_samc_csv": str(Path(args.samc_csv)),
                "source_mcmcis_csv": str(Path(args.mcmcis_csv)),
                "source_saved_oracle_run": str(Path(args.saved_cross_method_run)),
            },
        )

    print(json.dumps(manifests, indent=2, default=_json_default, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
