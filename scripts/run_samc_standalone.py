#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcmcis-matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from perm_pval.core.proposals import resolve_n_swap_pairs
from perm_pval.experiments.notebook_studies import (
    LoadedScenario,
    SAMCWorkflowConfig,
    _burn_in,
    _run_samc_cumulative_checkpoints,
    _sorted_unique_points,
    load_selected_scenarios,
    tune_samc_setup,
)


DEFAULT_SCENARIOS: tuple[str, ...] = (
    "gwas_additive_score_sig_n100",
    "poisson_diffmeans_hep_sig_n200",
)


@dataclass(frozen=True)
class SAMCJob:
    scenario: LoadedScenario
    scenario_index: int
    run_seed: int
    production_seed: int
    setup_seed: int
    setup_source: str
    checkpoints: tuple[int, ...]
    cfg: SAMCWorkflowConfig
    samc_setup: dict[str, Any]


def _project_root() -> Path:
    return PROJECT_ROOT


def _parse_int_list(raw: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip().replace("_", "")) for part in str(raw).split(",") if part.strip())
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("all values must be positive")
    return vals


def _parse_seeds(raw: str) -> tuple[int, ...]:
    text = str(raw).strip()
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("seed range must look like START:END")
        start, end = (int(part) for part in parts)
        if end < start:
            raise argparse.ArgumentTypeError("seed range END must be >= START")
        return tuple(range(start, end + 1))
    return _parse_int_list(text)


def _proposal_size_for_scenario(scenario: LoadedScenario, *, proposal_fraction: float) -> int:
    n_min = min(int(scenario.problem.n_treated), int(scenario.problem.n_control))
    return max(1, int(round(float(proposal_fraction) * float(n_min))))


def _completed_jobs(path: Path, *, final_checkpoint: int) -> set[tuple[str, int]]:
    if not Path(path).exists():
        return set()
    completed: set[tuple[str, int]] = set()
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row.get("checkpoint", 0)) == int(final_checkpoint) and str(row.get("status", "ok")) == "ok":
                completed.add((str(row["scenario"]), int(row["run_seed"])))
    return completed


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=_json_default)
    return value


def _known_threshold(scenario: LoadedScenario) -> float | None:
    threshold = scenario.extra.get("known_significance_threshold")
    if threshold is None:
        threshold = scenario.portfolio.get("known_significance_threshold")
    return None if threshold is None else float(threshold)


def _run_job(job: SAMCJob) -> list[dict[str, Any]]:
    started = time.perf_counter()
    n_swap_pairs = resolve_n_swap_pairs(
        job.scenario.problem.n_treated,
        job.scenario.problem.n_control,
        proposal_size=int(job.cfg.proposal_size),
    )
    rows = _run_samc_cumulative_checkpoints(
        job.scenario.problem,
        job.scenario.exact_p,
        checkpoints=tuple(int(v) for v in job.checkpoints),
        samc_setup=job.samc_setup,
        cfg=job.cfg,
        seed=int(job.production_seed),
    )
    job_elapsed = float(time.perf_counter() - started)
    threshold = _known_threshold(job.scenario)
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        checkpoint = int(enriched["checkpoint"])
        enriched.update(
            {
                "scenario": job.scenario.key,
                "description": job.scenario.description,
                "threshold_band": job.scenario.extra.get(
                    "threshold_band", job.scenario.portfolio.get("threshold_band")
                ),
                "application_setting_key": job.scenario.extra.get(
                    "application_setting_key", job.scenario.portfolio.get("application_setting_key")
                ),
                "label": "samc_swap_10pct",
                "config_id": "samc_swap_10pct",
                "replicate": int(job.run_seed),
                "run_seed": int(job.run_seed),
                "production_seed": int(job.production_seed),
                "setup_seed": int(job.setup_seed),
                "setup_source": str(job.setup_source),
                "setup_mode": "shared_per_scenario",
                "lambda_min": float(job.samc_setup["lambda_min"]),
                "n_bins": int(job.cfg.n_bins),
                "t0": float(job.cfg.t0),
                "trace_every": int(job.cfg.trace_every),
                "lambda_min_pilot": int(job.cfg.lambda_min_pilot),
                "convergence_tolerance": float(job.cfg.convergence_tolerance),
                "burn_in_fraction": float(job.cfg.burn_in_fraction),
                "burn_in": int(_burn_in(checkpoint, float(job.cfg.burn_in_fraction))),
                "proposal_fraction": float(int(job.cfg.proposal_size) / min(job.scenario.problem.n_treated, job.scenario.problem.n_control)),
                "proposal_size": int(job.cfg.proposal_size),
                "n_swap_pairs": int(n_swap_pairs),
                "exact_p": float(job.scenario.exact_p),
                "known_significance_threshold": threshold,
                "eval_incl_tuning": float(enriched.get("eval_excl_tuning", np.nan)),
                "job_elapsed_sec": job_elapsed,
                "status": "ok",
                "error": "",
            }
        )
        out.append(enriched)
    return out


def _fieldnames() -> list[str]:
    return [
        "scenario",
        "description",
        "threshold_band",
        "application_setting_key",
        "label",
        "config_id",
        "replicate",
        "run_seed",
        "production_seed",
        "setup_seed",
        "setup_source",
        "setup_mode",
        "lambda_min",
        "n_bins",
        "t0",
        "trace_every",
        "lambda_min_pilot",
        "convergence_tolerance",
        "burn_in_fraction",
        "burn_in",
        "proposal_fraction",
        "proposal_size",
        "n_swap_pairs",
        "exact_p",
        "known_significance_threshold",
        "method",
        "checkpoint",
        "estimate",
        "samc_estimate_no_empty_bin_correction",
        "samc_empty_bin_correction_delta",
        "samc_empty_bin_correction_ratio",
        "variance_estimate",
        "acceptance_rate",
        "samc_max_rel_freq_error",
        "samc_converged",
        "samc_pi0",
        "samc_empty_bins",
        "wall_time_sec",
        "eval_excl_tuning",
        "eval_incl_tuning",
        "samc_visit_total",
        "samc_tail_bin_freq",
        "abs_error",
        "rel_error",
        "log10_abs_error",
        "job_elapsed_sec",
        "status",
        "error",
    ]


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = _fieldnames()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _build_parser() -> argparse.ArgumentParser:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Run standalone SAMC with 10 percent local swap proposals.")
    parser.add_argument("--catalog-path", type=Path, default=root / "results" / "exact_scenarios" / "v1" / "catalog.json")
    parser.add_argument("--scenario-key", action="append", dest="scenario_keys", default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("1:80"))
    parser.add_argument("--chain-budget", type=int, default=5_000_000)
    parser.add_argument("--checkpoint-step", type=int, default=250_000)
    parser.add_argument("--proposal-fraction", type=float, default=0.10)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--t0", type=float, default=1_000.0)
    parser.add_argument("--trace-every", type=int, default=200)
    parser.add_argument("--lambda-min-pilot", type=int, default=10_000)
    parser.add_argument("--burn-in-fraction", type=float, default=0.20)
    parser.add_argument("--convergence-tolerance", type=float, default=20.0)
    parser.add_argument("--setup-seed-offset", type=int, default=30_000_000)
    parser.add_argument("--setup-source", choices=("saved-run", "fresh"), default="saved-run")
    parser.add_argument(
        "--setup-run-dir",
        type=Path,
        default=root / "results" / "cross_method_notebook" / "20260529_185422_cross_method_oracle_beta_best_compare",
    )
    parser.add_argument("--production-seed-offset", type=int, default=40_000_000)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--backend", choices=("auto", "process", "thread"), default="process")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "samc_swap_10pct")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser


def _run_jobs(
    jobs: list[SAMCJob],
    *,
    backend: str,
    workers: int,
    output_path: Path,
) -> int:
    completed = 0
    if workers == 1:
        for idx, job in enumerate(jobs, start=1):
            rows = _run_job(job)
            _append_rows(output_path, rows)
            completed += 1
            print(
                f"[{idx}/{len(jobs)}] {job.scenario.key} run_seed={job.run_seed} "
                f"proposal_size={job.cfg.proposal_size} done",
                flush=True,
            )
        return completed

    executor_cls: type[cf.Executor]
    if backend == "thread":
        executor_cls = cf.ThreadPoolExecutor
    else:
        executor_cls = cf.ProcessPoolExecutor

    try:
        with executor_cls(max_workers=workers) as executor:
            future_to_job = {executor.submit(_run_job, job): job for job in jobs}
            for idx, future in enumerate(cf.as_completed(future_to_job), start=1):
                job = future_to_job[future]
                rows = future.result()
                _append_rows(output_path, rows)
                completed += 1
                print(
                    f"[{idx}/{len(jobs)}] {job.scenario.key} run_seed={job.run_seed} "
                    f"proposal_size={job.cfg.proposal_size} done",
                    flush=True,
                )
    except PermissionError:
        if backend != "auto":
            raise
        print("Process backend unavailable; falling back to ThreadPoolExecutor.", flush=True)
        completed += _run_jobs(jobs, backend="thread", workers=workers, output_path=output_path)
    return completed


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    scenario_keys = tuple(args.scenario_keys) if args.scenario_keys else DEFAULT_SCENARIOS
    checkpoints = _sorted_unique_points(tuple(range(int(args.checkpoint_step), int(args.chain_budget) + 1, int(args.checkpoint_step))))
    if checkpoints[-1] != int(args.chain_budget):
        raise ValueError("--chain-budget must be divisible by --checkpoint-step")
    if int(args.jobs) <= 0:
        raise ValueError("--jobs must be positive")
    if not (0.0 < float(args.proposal_fraction) <= 1.0):
        raise ValueError("--proposal-fraction must be in (0, 1]")

    scenarios = load_selected_scenarios(
        catalog_path=Path(args.catalog_path),
        scenario_keys=scenario_keys,
        portfolio_group=None,
        min_tail_states=1,
    )
    scenario_by_key = {scenario.key: scenario for scenario in scenarios}

    output_name = args.output_name
    if output_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{stamp}_near_threshold_samc_swap_10pct_80x5m.csv"
    output_path = Path(args.output_dir) / output_name
    metadata_path = output_path.with_suffix(".metadata.json")

    completed_before = _completed_jobs(output_path, final_checkpoint=int(args.chain_budget)) if bool(args.resume) else set()

    setup_by_scenario: dict[str, dict[str, Any]] = {}
    cfg_by_scenario: dict[str, SAMCWorkflowConfig] = {}
    setup_seed_by_scenario: dict[str, int] = {}
    for scenario_index, scenario_key in enumerate(scenario_keys):
        scenario = scenario_by_key[str(scenario_key)]
        proposal_size = _proposal_size_for_scenario(scenario, proposal_fraction=float(args.proposal_fraction))
        cfg = SAMCWorkflowConfig(
            burn_in_fraction=float(args.burn_in_fraction),
            n_bins=int(args.n_bins),
            t0=float(args.t0),
            trace_every=int(args.trace_every),
            convergence_tolerance=float(args.convergence_tolerance),
            lambda_min_pilot=int(args.lambda_min_pilot),
            proposal_size=int(proposal_size),
        )
        if str(args.setup_source) == "saved-run":
            saved_metadata_path = Path(args.setup_run_dir) / str(scenario_key) / "metadata.json"
            metadata = json.loads(saved_metadata_path.read_text())
            setup = dict(metadata["samc_setup"])
            setup_seed = -1
            setup_source = str(saved_metadata_path)
        else:
            setup_seed = int(args.setup_seed_offset) + 1_000_000 * int(scenario_index)
            setup = tune_samc_setup(scenario.problem, cfg, seed=setup_seed)
            setup_source = "fresh_lambda_min_pilot"
        cfg_by_scenario[str(scenario_key)] = cfg
        setup_by_scenario[str(scenario_key)] = setup
        setup_seed_by_scenario[str(scenario_key)] = setup_seed
        setup_by_scenario[str(scenario_key)]["setup_source"] = setup_source

    jobs: list[SAMCJob] = []
    for scenario_index, scenario_key in enumerate(scenario_keys):
        scenario = scenario_by_key[str(scenario_key)]
        for seed in args.seeds:
            key = (str(scenario_key), int(seed))
            if key in completed_before:
                continue
            production_seed = int(args.production_seed_offset) + 1_000_000 * int(scenario_index) + int(seed)
            jobs.append(
                SAMCJob(
                    scenario=scenario,
                    scenario_index=int(scenario_index),
                    run_seed=int(seed),
                    production_seed=production_seed,
                    setup_seed=int(setup_seed_by_scenario[str(scenario_key)]),
                    setup_source=str(setup_by_scenario[str(scenario_key)].get("setup_source", "")),
                    checkpoints=checkpoints,
                    cfg=cfg_by_scenario[str(scenario_key)],
                    samc_setup=setup_by_scenario[str(scenario_key)],
                )
            )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "catalog_path": str(Path(args.catalog_path)),
        "scenario_keys": list(scenario_keys),
        "seeds": [int(v) for v in args.seeds],
        "chain_budget": int(args.chain_budget),
        "checkpoint_step": int(args.checkpoint_step),
        "estimation_points": [int(v) for v in checkpoints],
        "jobs_requested": int(args.jobs),
        "backend": str(args.backend),
        "proposal_fraction": float(args.proposal_fraction),
        "setup_source": str(args.setup_source),
        "setup_run_dir": str(Path(args.setup_run_dir)) if str(args.setup_source) == "saved-run" else None,
        "setup_mode": "shared_per_scenario",
        "counts_include_lambda_min_pilot": False,
        "n_jobs_total": int(len(scenario_keys) * len(tuple(args.seeds))),
        "n_jobs_already_completed": int(len(completed_before)),
        "n_jobs_to_run": int(len(jobs)),
        "output_csv": str(output_path),
        "config_by_scenario": {
            key: {
                "burn_in_fraction": float(cfg.burn_in_fraction),
                "n_bins": int(cfg.n_bins),
                "t0": float(cfg.t0),
                "trace_every": int(cfg.trace_every),
                "convergence_tolerance": float(cfg.convergence_tolerance),
                "lambda_min_pilot": int(cfg.lambda_min_pilot),
                "proposal_size": int(cfg.proposal_size),
                "proposal_fraction_realized": float(
                    int(cfg.proposal_size)
                    / min(scenario_by_key[key].problem.n_treated, scenario_by_key[key].problem.n_control)
                ),
            }
            for key, cfg in cfg_by_scenario.items()
        },
        "samc_setup_by_scenario": {
            key: {
                "setup_seed": int(setup_seed_by_scenario[key]),
                "setup_source": str(setup.get("setup_source", "")),
                "lambda_min": float(setup["lambda_min"]),
                "bin_edges_first": [float(v) for v in np.asarray(setup["bin_edges"], dtype=float)[:5]],
                "bin_edges_last": [float(v) for v in np.asarray(setup["bin_edges"], dtype=float)[-5:]],
            }
            for key, setup in setup_by_scenario.items()
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default))
    print(json.dumps(metadata, indent=2, default=_json_default), flush=True)

    if not jobs:
        print(json.dumps({"output_csv": str(output_path), "metadata": str(metadata_path), "n_jobs_run": 0}, indent=2))
        return 0

    started = time.perf_counter()
    workers = min(int(args.jobs), len(jobs))
    n_completed = _run_jobs(jobs, backend=str(args.backend), workers=workers, output_path=output_path)
    print(
        json.dumps(
            {
                "output_csv": str(output_path),
                "metadata": str(metadata_path),
                "n_jobs_run": int(n_completed),
                "elapsed_sec": float(time.perf_counter() - started),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
