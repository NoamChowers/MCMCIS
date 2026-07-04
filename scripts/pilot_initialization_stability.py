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
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcmcis-matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from perm_pval.experiments.notebook_studies import LoadedScenario, load_selected_scenarios
from perm_pval.methods.beta_tuning import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
)


DEFAULT_SCENARIOS: tuple[str, ...] = (
    "gwas_additive_score_sig_n100",
    "poisson_diffmeans_hep_sig_n200",
)
DEFAULT_CHECKPOINTS: tuple[int, ...] = (5_000, 10_000, 25_000, 50_000, 100_000, 200_000)


@dataclass(frozen=True)
class JobSpec:
    scenario: LoadedScenario
    seed: int
    checkpoints: tuple[int, ...]
    pilot_size: int
    gamma: float
    p0_reference: float
    p0_reference_source: str
    sampler: str
    chunk_size: int


def _project_root() -> Path:
    return PROJECT_ROOT


def _parse_int_list(raw: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip().replace("_", "")) for part in raw.split(",") if part.strip())
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


def _latest_oracle_beta_run_dir(root: Path) -> Path:
    candidates = sorted(
        path for path in Path(root).iterdir() if path.is_dir() and path.name.endswith("_oracle_beta_search")
    )
    if not candidates:
        raise FileNotFoundError(f"No '*_oracle_beta_search' run found under {root}")
    return candidates[-1]


def _load_beta_reference_p0(beta_run_dir: Path, scenario_key: str) -> tuple[float, str]:
    scenario_dir = Path(beta_run_dir) / scenario_key
    best_config_path = scenario_dir / "best_config.json"
    metadata_path = scenario_dir / "metadata.json"
    for path in (best_config_path, metadata_path):
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if "simple_reference_p0" in payload:
            return float(payload["simple_reference_p0"]), f"beta_reference:{path.name}:simple_reference_p0"
        if "known_significance_threshold" in payload:
            return float(payload["known_significance_threshold"]), f"beta_reference:{path.name}:known_significance_threshold"
    raise FileNotFoundError(f"Could not resolve p0 reference for {scenario_key} under {scenario_dir}")


def _resolve_p0_reference(
    scenario: LoadedScenario,
    *,
    mode: str,
    beta_run_dir: Path | None,
) -> tuple[float, str]:
    if mode == "beta-reference":
        if beta_run_dir is None:
            raise ValueError("beta_run_dir is required for p0 mode 'beta-reference'")
        return _load_beta_reference_p0(beta_run_dir, scenario.key)
    if mode == "known-threshold":
        threshold = scenario.extra.get("known_significance_threshold")
        if threshold is None:
            threshold = scenario.portfolio.get("known_significance_threshold")
        if threshold is None:
            raise KeyError(f"Scenario {scenario.key} has no known_significance_threshold")
        return float(threshold), "known_significance_threshold"
    if mode == "exact-p":
        return float(scenario.exact_p), "exact_p"
    raise ValueError(f"Unknown p0 mode: {mode}")


def _linear_vectorized_pilot_statistics(
    scenario: LoadedScenario,
    *,
    n_samples: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray | None:
    problem = scenario.problem
    statistic_name = getattr(problem.statistic, "__name__", "")
    if statistic_name not in {"treated_sum", "difference_in_means"}:
        return None

    x = np.asarray(problem.X, dtype=float)
    if x.ndim != 1:
        return None

    n = int(problem.n)
    n_treated = int(problem.n_treated)
    n_control = int(problem.n_control)
    if n_treated <= 0 or n_control <= 0 or n_treated + n_control != n:
        return None

    rng = np.random.default_rng(int(seed))
    out = np.empty(int(n_samples), dtype=float)
    total_sum = float(np.sum(x))
    chunk = max(1, int(chunk_size))

    for start in range(0, int(n_samples), chunk):
        stop = min(start + chunk, int(n_samples))
        size = int(stop - start)
        ranks = rng.random((size, n), dtype=float)
        treated_idx = np.argpartition(ranks, kth=n_treated - 1, axis=1)[:, :n_treated]
        treated_sum = np.sum(x[treated_idx], axis=1)
        if statistic_name == "treated_sum":
            out[start:stop] = treated_sum
        else:
            out[start:stop] = treated_sum / float(n_treated) - (total_sum - treated_sum) / float(n_control)
    return out


def _sample_pilot_statistics(spec: JobSpec) -> tuple[np.ndarray, str]:
    if spec.sampler in {"auto", "linear-vectorized"}:
        vals = _linear_vectorized_pilot_statistics(
            spec.scenario,
            n_samples=int(spec.pilot_size),
            seed=int(spec.seed),
            chunk_size=int(spec.chunk_size),
        )
        if vals is not None:
            return vals, "linear-vectorized"
        if spec.sampler == "linear-vectorized":
            raise ValueError(f"Scenario {spec.scenario.key} is not supported by the vectorized linear sampler")
    vals = iid_pilot_statistics(spec.scenario.problem, n_samples=int(spec.pilot_size), seed=int(spec.seed))
    return vals, "generic"


def _tail_mask(values: np.ndarray, *, t_obs: float, tail: str) -> np.ndarray:
    if tail == "right":
        return values >= float(t_obs)
    if tail == "left":
        return values <= float(t_obs)
    if tail == "two-sided":
        return np.abs(values) >= abs(float(t_obs))
    raise ValueError(f"Unsupported tail: {tail}")


def _run_one_job(spec: JobSpec) -> list[dict[str, Any]]:
    started = time.perf_counter()
    t_values, sampler_used = _sample_pilot_statistics(spec)
    sample_elapsed = time.perf_counter() - started

    scenario = spec.scenario
    problem = scenario.problem
    q_target = float(spec.p0_reference ** spec.gamma)
    z_target = float(spec.p0_reference / q_target)
    rows: list[dict[str, Any]] = []

    for checkpoint in spec.checkpoints:
        row_started = time.perf_counter()
        prefix = np.asarray(t_values[: int(checkpoint)], dtype=float)
        status = "ok"
        error = ""
        beta_warning = ""
        sigma_t = float("nan")
        beta = float("nan")
        z_hat_at_beta = float("nan")
        implied_q_at_beta = float("nan")

        try:
            sigma_t = estimate_scale_T(prefix, method="sd")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                beta = init_beta_from_iid_pilot(
                    pilot_T=prefix,
                    T_obs=problem.t_obs,
                    sigma_T=sigma_t,
                    p0=float(spec.p0_reference),
                    q_target=q_target,
                    beta_max=1e6,
                )
            beta_warning = " | ".join(str(item.message) for item in caught)
            shortfall = np.maximum((float(problem.t_obs) - prefix) / float(sigma_t), 0.0)
            z_hat_at_beta = float(np.mean(np.exp(-float(beta) * shortfall)))
            implied_q_at_beta = float(spec.p0_reference / z_hat_at_beta) if z_hat_at_beta > 0.0 else float("inf")
        except Exception as exc:  # noqa: BLE001 - keep long simulation logging robust.
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        tail_hits = int(np.count_nonzero(_tail_mask(prefix, t_obs=problem.t_obs, tail=problem.tail)))
        row_elapsed = time.perf_counter() - row_started
        rows.append(
            {
                "scenario": scenario.key,
                "description": scenario.description,
                "threshold_band": scenario.extra.get("threshold_band", scenario.portfolio.get("threshold_band")),
                "application_setting_key": scenario.extra.get(
                    "application_setting_key", scenario.portfolio.get("application_setting_key")
                ),
                "seed": int(spec.seed),
                "pilot_size_total": int(spec.pilot_size),
                "checkpoint": int(checkpoint),
                "sampler": sampler_used,
                "p0_reference": float(spec.p0_reference),
                "p0_reference_source": spec.p0_reference_source,
                "gamma": float(spec.gamma),
                "q_target": q_target,
                "z_target": z_target,
                "exact_p": float(scenario.exact_p),
                "known_significance_threshold": scenario.extra.get(
                    "known_significance_threshold", scenario.portfolio.get("known_significance_threshold")
                ),
                "t_obs": float(problem.t_obs),
                "tail": str(problem.tail),
                "n": int(problem.n),
                "n_treated": int(problem.n_treated),
                "n_control": int(problem.n_control),
                "sigma_t": sigma_t,
                "beta": beta,
                "z_hat_at_beta": z_hat_at_beta,
                "implied_q_at_beta": implied_q_at_beta,
                "pilot_tail_hits": tail_hits,
                "pilot_tail_fraction": float(tail_hits / int(checkpoint)),
                "pilot_mean": float(np.mean(prefix)),
                "pilot_sd": float(np.std(prefix, ddof=1)),
                "pilot_min": float(np.min(prefix)),
                "pilot_max": float(np.max(prefix)),
                "pilot_q95": float(np.quantile(prefix, 0.95)),
                "pilot_q99": float(np.quantile(prefix, 0.99)),
                "sample_elapsed_sec": float(sample_elapsed),
                "checkpoint_elapsed_sec": float(row_elapsed),
                "job_elapsed_sec": float(time.perf_counter() - started),
                "status": status,
                "beta_warning": beta_warning,
                "error": error,
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure naive MCMC-IS pilot initialization stability across IID pilot budgets."
    )
    root = _project_root()
    parser.add_argument("--catalog-path", type=Path, default=root / "results" / "exact_scenarios" / "v1" / "catalog.json")
    parser.add_argument("--beta-results-root", type=Path, default=root / "results" / "mcmcis_beta_notebook")
    parser.add_argument("--beta-run-dir", type=Path, default=None)
    parser.add_argument("--p0-mode", choices=("beta-reference", "known-threshold", "exact-p"), default="beta-reference")
    parser.add_argument("--scenario-key", action="append", dest="scenario_keys", default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("1:80"))
    parser.add_argument("--checkpoints", type=_parse_int_list, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--pilot-size", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=1.0 / 3.0)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--backend", choices=("auto", "process", "thread"), default="auto")
    parser.add_argument("--sampler", choices=("auto", "linear-vectorized", "generic"), default="auto")
    parser.add_argument("--chunk-size", type=int, default=8_192)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "pilot_initialization_stability")
    parser.add_argument("--output-name", type=str, default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    scenario_keys = tuple(args.scenario_keys) if args.scenario_keys else DEFAULT_SCENARIOS
    checkpoints = tuple(sorted(set(int(v) for v in args.checkpoints)))
    if int(args.pilot_size) < max(checkpoints):
        raise ValueError("--pilot-size must be at least the largest checkpoint")
    if int(args.jobs) <= 0:
        raise ValueError("--jobs must be positive")

    beta_run_dir = Path(args.beta_run_dir) if args.beta_run_dir is not None else None
    if args.p0_mode == "beta-reference" and beta_run_dir is None:
        beta_run_dir = _latest_oracle_beta_run_dir(Path(args.beta_results_root))

    scenarios = load_selected_scenarios(
        catalog_path=Path(args.catalog_path),
        scenario_keys=scenario_keys,
        portfolio_group=None,
        min_tail_states=1,
    )
    p0_by_key = {
        scenario.key: _resolve_p0_reference(scenario, mode=str(args.p0_mode), beta_run_dir=beta_run_dir)
        for scenario in scenarios
    }

    specs = [
        JobSpec(
            scenario=scenario,
            seed=int(seed),
            checkpoints=checkpoints,
            pilot_size=int(args.pilot_size),
            gamma=float(args.gamma),
            p0_reference=float(p0_by_key[scenario.key][0]),
            p0_reference_source=str(p0_by_key[scenario.key][1]),
            sampler=str(args.sampler),
            chunk_size=int(args.chunk_size),
        )
        for scenario in scenarios
        for seed in args.seeds
    ]

    output_name = args.output_name
    if output_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{stamp}_near_threshold_pilot_initialization_stability.csv"
    output_path = Path(args.output_dir) / output_name
    metadata_path = output_path.with_suffix(".metadata.json")

    print(
        json.dumps(
            {
                "scenarios": list(scenario_keys),
                "seeds": [int(v) for v in args.seeds],
                "checkpoints": [int(v) for v in checkpoints],
                "pilot_size": int(args.pilot_size),
                "gamma": float(args.gamma),
                "q_rule": "q_target = p0_reference ** gamma",
                "p0_mode": str(args.p0_mode),
                "beta_run_dir": str(beta_run_dir) if beta_run_dir is not None else None,
                "p0_by_scenario": {
                    key: {"p0_reference": float(value), "source": source} for key, (value, source) in p0_by_key.items()
                },
                "jobs": int(args.jobs),
                "sampler": str(args.sampler),
                "backend": str(args.backend),
                "output_csv": str(output_path),
            },
            indent=2,
        ),
        flush=True,
    )

    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    workers = min(int(args.jobs), len(specs))
    if workers == 1:
        for idx, spec in enumerate(specs, start=1):
            all_rows.extend(_run_one_job(spec))
            print(f"[{idx}/{len(specs)}] {spec.scenario.key} seed={spec.seed} done", flush=True)
    else:
        executor_cls: type[cf.Executor]
        if args.backend == "thread":
            executor_cls = cf.ThreadPoolExecutor
        else:
            executor_cls = cf.ProcessPoolExecutor
        try:
            with executor_cls(max_workers=workers) as executor:
                future_to_spec = {executor.submit(_run_one_job, spec): spec for spec in specs}
                for idx, future in enumerate(cf.as_completed(future_to_spec), start=1):
                    spec = future_to_spec[future]
                    all_rows.extend(future.result())
                    print(f"[{idx}/{len(specs)}] {spec.scenario.key} seed={spec.seed} done", flush=True)
        except PermissionError:
            if args.backend != "auto":
                raise
            print("Process backend unavailable; falling back to ThreadPoolExecutor.", flush=True)
            with cf.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_spec = {executor.submit(_run_one_job, spec): spec for spec in specs}
                for idx, future in enumerate(cf.as_completed(future_to_spec), start=1):
                    spec = future_to_spec[future]
                    all_rows.extend(future.result())
                    print(f"[{idx}/{len(specs)}] {spec.scenario.key} seed={spec.seed} done", flush=True)

    all_rows = sorted(all_rows, key=lambda row: (str(row["scenario"]), int(row["seed"]), int(row["checkpoint"])))
    n_rows = _write_csv(output_path, all_rows)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": float(time.perf_counter() - started),
        "output_csv": str(output_path),
        "n_rows": int(n_rows),
        "n_jobs": int(workers),
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in {"seeds", "checkpoints"}
        },
        "seeds": [int(v) for v in args.seeds],
        "checkpoints": [int(v) for v in checkpoints],
        "scenario_keys": list(scenario_keys),
        "p0_by_scenario": {
            key: {"p0_reference": float(value), "source": source} for key, (value, source) in p0_by_key.items()
        },
        "job_spec_template": (
            {
                "seed": int(specs[0].seed),
                "checkpoints": [int(v) for v in specs[0].checkpoints],
                "pilot_size": int(specs[0].pilot_size),
                "gamma": float(specs[0].gamma),
                "sampler": str(specs[0].sampler),
                "chunk_size": int(specs[0].chunk_size),
            }
            if specs
            else None
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"output_csv": str(output_path), "metadata": str(metadata_path), "n_rows": n_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
