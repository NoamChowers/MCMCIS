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
    MCMCWorkflowConfig,
    _run_mcmc_cumulative_checkpoints,
    _sorted_unique_points,
    load_selected_scenarios,
)


DEFAULT_SCENARIOS: tuple[str, ...] = (
    "gwas_additive_score_sig_n100",
    "poisson_diffmeans_hep_sig_n200",
)


@dataclass(frozen=True)
class PilotInitialization:
    scenario: str
    pilot_seed: int
    pilot_checkpoint: int
    beta: float
    sigma_t: float
    p0_reference: float
    p0_reference_source: str
    gamma: float
    q_target: float
    z_target: float
    pilot_tail_hits: int
    pilot_tail_fraction: float


@dataclass(frozen=True)
class MCMCJob:
    scenario: LoadedScenario
    init: PilotInitialization
    scenario_index: int
    production_seed: int
    checkpoints: tuple[int, ...]
    cfg: MCMCWorkflowConfig
    proposal_size: int


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


def _proposal_size_for_scenario(scenario: LoadedScenario) -> int:
    setting_key = str(scenario.extra.get("application_setting_key", ""))
    if setting_key == "gwas_threshold_suite":
        return 2
    if setting_key == "hep_threshold_suite":
        return 6
    band = str(scenario.portfolio.get("sample_size_band", "medium"))
    if band == "small":
        return 1
    if band == "large":
        return 2
    return 1


def _read_pilot_initializations(
    path: Path,
    *,
    scenario_keys: Iterable[str],
    seeds: Iterable[int],
    pilot_checkpoint: int,
) -> dict[tuple[str, int], PilotInitialization]:
    wanted_scenarios = {str(key) for key in scenario_keys}
    wanted_seeds = {int(seed) for seed in seeds}
    out: dict[tuple[str, int], PilotInitialization] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            scenario = str(row["scenario"])
            seed = int(row["seed"])
            checkpoint = int(row["checkpoint"])
            if scenario not in wanted_scenarios or seed not in wanted_seeds or checkpoint != int(pilot_checkpoint):
                continue
            if str(row.get("status", "ok")) != "ok":
                raise ValueError(f"Pilot initialization row is not ok for {scenario} seed={seed}: {row.get('error')}")
            out[(scenario, seed)] = PilotInitialization(
                scenario=scenario,
                pilot_seed=seed,
                pilot_checkpoint=checkpoint,
                beta=float(row["beta"]),
                sigma_t=float(row["sigma_t"]),
                p0_reference=float(row["p0_reference"]),
                p0_reference_source=str(row["p0_reference_source"]),
                gamma=float(row["gamma"]),
                q_target=float(row["q_target"]),
                z_target=float(row["z_target"]),
                pilot_tail_hits=int(row["pilot_tail_hits"]),
                pilot_tail_fraction=float(row["pilot_tail_fraction"]),
            )
    missing = sorted((scenario, seed) for scenario in wanted_scenarios for seed in wanted_seeds if (scenario, seed) not in out)
    if missing:
        preview = ", ".join(f"{scenario}:{seed}" for scenario, seed in missing[:10])
        more = "" if len(missing) <= 10 else f", ... +{len(missing) - 10} more"
        raise KeyError(f"Missing pilot initialization rows at checkpoint {pilot_checkpoint}: {preview}{more}")
    return out


def _completed_jobs(path: Path, *, final_checkpoint: int) -> set[tuple[str, int]]:
    if not Path(path).exists():
        return set()
    completed: set[tuple[str, int]] = set()
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row.get("checkpoint", 0)) == int(final_checkpoint) and str(row.get("status", "ok")) == "ok":
                completed.add((str(row["scenario"]), int(row["pilot_seed"])))
    return completed


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=_json_default)
    return value


def _run_job(job: MCMCJob) -> list[dict[str, Any]]:
    started = time.perf_counter()
    n_swap_pairs = resolve_n_swap_pairs(
        job.scenario.problem.n_treated,
        job.scenario.problem.n_control,
        proposal_size=int(job.proposal_size),
    )
    rows = _run_mcmc_cumulative_checkpoints(
        job.scenario.problem,
        job.scenario.exact_p,
        checkpoints=tuple(int(v) for v in job.checkpoints),
        beta=float(job.init.beta),
        sigma_t=float(job.init.sigma_t),
        cfg=job.cfg,
        seed=int(job.production_seed),
    )
    job_elapsed = float(time.perf_counter() - started)
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
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
                "label": "mcmcis_per_seed_pilot",
                "config_id": f"mcmcis_pilot_seed_{int(job.init.pilot_seed):03d}",
                "replicate": int(job.init.pilot_seed),
                "pilot_seed": int(job.init.pilot_seed),
                "production_seed": int(job.production_seed),
                "pilot_checkpoint": int(job.init.pilot_checkpoint),
                "pilot_beta": float(job.init.beta),
                "pilot_sigma_t": float(job.init.sigma_t),
                "pilot_p0_reference": float(job.init.p0_reference),
                "pilot_p0_reference_source": job.init.p0_reference_source,
                "pilot_gamma": float(job.init.gamma),
                "pilot_q_target": float(job.init.q_target),
                "pilot_z_target": float(job.init.z_target),
                "pilot_tail_hits": int(job.init.pilot_tail_hits),
                "pilot_tail_fraction": float(job.init.pilot_tail_fraction),
                "proposal_size": int(job.proposal_size),
                "n_swap_pairs": int(n_swap_pairs),
                "exact_p": float(job.scenario.exact_p),
                "known_significance_threshold": job.scenario.extra.get(
                    "known_significance_threshold",
                    job.scenario.portfolio.get("known_significance_threshold"),
                ),
                "source": "per_seed_200k_pilot_rule_p0_one_third",
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
        "pilot_seed",
        "production_seed",
        "pilot_checkpoint",
        "pilot_beta",
        "pilot_sigma_t",
        "pilot_p0_reference",
        "pilot_p0_reference_source",
        "pilot_gamma",
        "pilot_q_target",
        "pilot_z_target",
        "pilot_tail_hits",
        "pilot_tail_fraction",
        "proposal_size",
        "n_swap_pairs",
        "exact_p",
        "known_significance_threshold",
        "source",
        "method",
        "checkpoint",
        "estimate",
        "variance_estimate",
        "snis_mcse_obm",
        "tail_hits",
        "tail_share_raw",
        "ess",
        "acceptance_rate",
        "weight_cv",
        "beta",
        "sigma_t",
        "tilt_mode",
        "wall_time_sec",
        "eval_excl_tuning",
        "n_weighted_samples",
        "acceptance_rates",
        "abs_error",
        "rel_error",
        "log10_abs_error",
        "mcmc_chain_budget",
        "mcmc_reported_budget",
        "state_reused_init",
        "burn_in",
        "estimator_variant",
        "scan_n_weighted_samples",
        "production_n_weighted_samples",
        "pooled_scan_batch_count",
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
    parser = argparse.ArgumentParser(
        description="Run full-budget MCMC-IS chains from per-seed pilot initialization estimates."
    )
    parser.add_argument(
        "--pilot-csv",
        type=Path,
        default=root
        / "results"
        / "pilot_initialization_stability"
        / "near_threshold_pilot_initialization_stability_80x200k.csv",
    )
    parser.add_argument("--catalog-path", type=Path, default=root / "results" / "exact_scenarios" / "v1" / "catalog.json")
    parser.add_argument("--scenario-key", action="append", dest="scenario_keys", default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("1:80"))
    parser.add_argument("--pilot-checkpoint", type=int, default=200_000)
    parser.add_argument("--chain-budget", type=int, default=5_000_000)
    parser.add_argument("--checkpoint-step", type=int, default=250_000)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--backend", choices=("auto", "process", "thread"), default="process")
    parser.add_argument("--production-seed-offset", type=int, default=10_000_000)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "mcmcis_per_seed_pilot")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser


def _run_jobs(
    jobs: list[MCMCJob],
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
                f"[{idx}/{len(jobs)}] {job.scenario.key} pilot_seed={job.init.pilot_seed} "
                f"production_seed={job.production_seed} done",
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
                    f"[{idx}/{len(jobs)}] {job.scenario.key} pilot_seed={job.init.pilot_seed} "
                    f"production_seed={job.production_seed} done",
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

    scenarios = load_selected_scenarios(
        catalog_path=Path(args.catalog_path),
        scenario_keys=scenario_keys,
        portfolio_group=None,
        min_tail_states=1,
    )
    scenario_by_key = {scenario.key: scenario for scenario in scenarios}
    pilot_by_key = _read_pilot_initializations(
        Path(args.pilot_csv),
        scenario_keys=scenario_keys,
        seeds=args.seeds,
        pilot_checkpoint=int(args.pilot_checkpoint),
    )

    output_name = args.output_name
    if output_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{stamp}_near_threshold_mcmcis_per_seed_200k_pilot_80x5m.csv"
    output_path = Path(args.output_dir) / output_name
    metadata_path = output_path.with_suffix(".metadata.json")

    completed_before: set[tuple[str, int]] = set()
    if bool(args.resume):
        completed_before = _completed_jobs(output_path, final_checkpoint=int(args.chain_budget))

    jobs: list[MCMCJob] = []
    for scenario_index, scenario_key in enumerate(scenario_keys):
        scenario = scenario_by_key[str(scenario_key)]
        proposal_size = int(_proposal_size_for_scenario(scenario))
        cfg = MCMCWorkflowConfig(
            pilot_samples=0,
            tune_steps=0,
            chains=1,
            burn_in_fraction=0.20,
            thin=1,
            estimate_variance=True,
            obm_batch_size=None,
            chain_n_jobs=1,
            tilt_mode="smooth_hinge",
            proposal_size=proposal_size,
            local_scan_enabled=False,
        )
        for seed in args.seeds:
            key = (str(scenario_key), int(seed))
            if key in completed_before:
                continue
            init = pilot_by_key[key]
            production_seed = int(args.production_seed_offset) + 1_000_000 * int(scenario_index) + int(seed)
            jobs.append(
                MCMCJob(
                    scenario=scenario,
                    init=init,
                    scenario_index=int(scenario_index),
                    production_seed=production_seed,
                    checkpoints=checkpoints,
                    cfg=cfg,
                    proposal_size=proposal_size,
                )
            )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pilot_csv": str(Path(args.pilot_csv)),
        "catalog_path": str(Path(args.catalog_path)),
        "scenario_keys": list(scenario_keys),
        "seeds": [int(v) for v in args.seeds],
        "pilot_checkpoint": int(args.pilot_checkpoint),
        "chain_budget": int(args.chain_budget),
        "checkpoint_step": int(args.checkpoint_step),
        "estimation_points": [int(v) for v in checkpoints],
        "jobs_requested": int(args.jobs),
        "backend": str(args.backend),
        "production_seed_offset": int(args.production_seed_offset),
        "output_csv": str(output_path),
        "n_jobs_total": int(len(scenario_keys) * len(tuple(args.seeds))),
        "n_jobs_already_completed": int(len(completed_before)),
        "n_jobs_to_run": int(len(jobs)),
        "counts_include_pilot_budget": False,
        "proposal_size_by_scenario": {
            scenario.key: int(_proposal_size_for_scenario(scenario)) for scenario in scenarios
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)

    if not jobs:
        print(json.dumps({"output_csv": str(output_path), "metadata": str(metadata_path), "n_jobs_run": 0}, indent=2))
        return 0

    started = time.perf_counter()
    workers = min(int(args.jobs), len(jobs))
    n_completed = _run_jobs(jobs, backend=str(args.backend), workers=workers, output_path=output_path)
    done_payload = {
        "output_csv": str(output_path),
        "metadata": str(metadata_path),
        "n_jobs_run": int(n_completed),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    print(json.dumps(done_payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
