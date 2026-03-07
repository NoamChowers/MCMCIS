from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import time
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perm_pval.core.proposals import n_swap_pairs_from_fraction, propose_localized_swaps
from perm_pval.core.problem import PermutationTestProblem
from perm_pval.diagnostics.is_weights import effective_sample_size, summarize_weights
from perm_pval.diagnostics.mcmc import obm_long_run_variance
from perm_pval.diagnostics.samc import visitation_frequency
from perm_pval.experiments.exact_scenarios import ExactScenario, load_saved_exact_scenarios
from perm_pval.methods.beta_tuning import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    make_short_chain_q_runner,
    tune_beta_to_target_q,
)
from perm_pval.methods.mcmc_is import (
    right_tail_deficit_scaled,
    right_tail_step_shortfall,
    run_mcmc_is,
)
from perm_pval.methods.random_sampling import run_random_sampling, wilson_interval
from perm_pval.methods.samc import (
    _bin_index,
    _default_stepsize,
    _paper_pvalue_estimate,
    _relative_sampling_frequency_error,
    run_samc,
)


@dataclass(frozen=True)
class MCMCWorkflowConfig:
    use_true_p0_for_q_target: bool = True
    p0_guess: float = 1e-8
    d_alpha: float = 0.25
    pilot_samples: int = 20_000
    scale_method: str = "sd"
    beta_max_init: float = 1e6
    tune_steps: int = 2_000
    tune_burn_in_fraction: float = 0.20
    tune_thin: int = 2
    tune_bracket_factor: float = 2.0
    tune_tol_rel: float = 0.20
    tune_max_bracket: int = 12
    tune_max_bisect: int = 12
    tune_replicate: int = 1
    tune_reuse_state: bool = True
    beta_override: float | None = None
    chains: int = 2
    burn_in_fraction: float = 0.20
    thin: int = 1
    estimate_variance: bool = True
    obm_batch_size: int | None = None
    tilt_mode: str = "smooth_hinge"
    proposal_fraction: float = 0.075
    proposal_swaps: int | None = None
    local_scan_enabled: bool = True
    local_scan_q_multipliers: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
    local_scan_screen_total_steps: int = 12_000
    local_scan_finalist_count: int = 3
    local_scan_total_steps: int = 64_000
    local_scan_chains: int = 2
    local_scan_burn_in_fraction: float = 0.20
    local_scan_thin: int = 1
    local_scan_screen_min_tail_hits: int = 25
    local_scan_final_min_tail_hits: int = 20
    local_scan_min_ess: float = 10.0
    local_scan_min_ess_fraction: float = 1e-3
    local_scan_q_lower_factor: float = 0.25
    local_scan_q_upper_factor: float = 4.0
    local_scan_variance_near_min_ratio: float = 1.20


@dataclass(frozen=True)
class SAMCWorkflowConfig:
    burn_in_fraction: float = 0.20
    n_bins: int = 40
    t0: float = 1_000.0
    trace_every: int = 200
    convergence_tolerance: float = 20.0
    lambda_min_pilot: int = 10_000
    proposal_fraction: float = 0.075
    proposal_swaps: int | None = None


@dataclass(frozen=True)
class CrossMethodStudyConfig:
    estimation_points: tuple[int, ...]
    repeats: int = 5
    base_seed: int = 12_345
    iid_density_samples: int = 120_000
    min_tail_states: int = 2
    confidence_level: float = 0.95
    n_jobs: int = 1


@dataclass(frozen=True)
class BetaSweepStudyConfig:
    estimation_points: tuple[int, ...]
    repeats: int = 5
    beta_multipliers: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00, 1.25, 2.00)
    chains: int = 2
    burn_in_fraction: float = 0.20
    thin: int = 1
    estimate_variance: bool = True
    obm_batch_size: int | None = None
    tilt_mode: str = "smooth_hinge"
    proposal_fraction: float = 0.075
    proposal_swaps: int | None = None
    base_seed: int = 54_321
    n_jobs: int = 1


@dataclass(frozen=True)
class LoadedScenario:
    key: str
    description: str
    problem: PermutationTestProblem
    exact_p: float
    exact_tail_hits: int
    exact_n_perm: int
    exact_method: str
    notes: str
    extra: dict[str, Any]


def _sorted_unique_points(points: Iterable[int]) -> tuple[int, ...]:
    vals = sorted({int(v) for v in points})
    if not vals or vals[0] <= 0:
        raise ValueError("estimation_points must contain positive integers.")
    return tuple(vals)


def _to_loaded_scenario(s: ExactScenario) -> LoadedScenario:
    return LoadedScenario(
        key=s.key,
        description=s.description,
        problem=s.problem,
        exact_p=float(s.exact_p_value),
        exact_tail_hits=int(s.tail_hits),
        exact_n_perm=int(s.n_permutations),
        exact_method=str(s.exact_method),
        notes=str(s.notes),
        extra=dict(s.extra),
    )


def load_selected_scenarios(
    *,
    catalog_path: Path,
    scenario_keys: Iterable[str] | None = None,
    min_tail_states: int = 1,
) -> list[LoadedScenario]:
    all_scenarios = [_to_loaded_scenario(s) for s in load_saved_exact_scenarios(catalog_path)]
    by_key = {s.key: s for s in all_scenarios}

    if scenario_keys is None:
        selected_keys = [s.key for s in all_scenarios]
    else:
        selected_keys = [str(k) for k in scenario_keys]

    out: list[LoadedScenario] = []
    for key in selected_keys:
        if key not in by_key:
            raise KeyError(f"Unknown scenario key '{key}'. Known keys: {sorted(by_key)}")
        scenario = by_key[key]
        if scenario.exact_tail_hits < int(min_tail_states):
            raise ValueError(
                f"Scenario '{key}' has only {scenario.exact_tail_hits} tail state(s), "
                f"below min_tail_states={min_tail_states}."
            )
        out.append(scenario)
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default))
            handle.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def create_timestamped_run_dir(root: Path, prefix: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{ts}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _effective_n_jobs(n_jobs: int, n_tasks: int) -> int:
    if n_tasks <= 0:
        return 1
    if n_jobs is None:
        requested = 1
    else:
        requested = int(n_jobs)
    if requested <= 1:
        return 1
    return max(1, min(requested, n_tasks))


def _try_make_process_pool(n_jobs: int) -> cf.ProcessPoolExecutor | None:
    if int(n_jobs) <= 1:
        return None
    mp_ctx = mp.get_context("spawn")
    try:
        return cf.ProcessPoolExecutor(max_workers=int(n_jobs), mp_context=mp_ctx)
    except (PermissionError, NotImplementedError, OSError) as exc:
        warnings.warn(
            f"ProcessPoolExecutor unavailable in this runtime ({exc}). Falling back to serial execution.",
            RuntimeWarning,
        )
        return None


def samc_variance_proxy(p_hat: float, n_steps: int, burn_in: int) -> float:
    n_eff = max(int(n_steps - burn_in), 1)
    return float(max(p_hat * (1.0 - p_hat) / n_eff, 0.0))


def _kept_samples_per_chain(n_steps: int, burn_in: int, thin: int) -> int:
    if n_steps <= burn_in:
        return 0
    return int(1 + (n_steps - 1 - burn_in) // thin)


def _local_scan_steps(total_steps: int, n_chains: int, burn_in_fraction: float) -> tuple[int, int]:
    steps_per_chain = _steps_per_chain(total_steps, n_chains)
    burn_in = _burn_in(steps_per_chain, burn_in_fraction)
    return steps_per_chain, burn_in


def _local_scan_q_floor(total_steps: int, cfg: MCMCWorkflowConfig, min_tail_hits: int) -> float:
    steps_per_chain, burn_in = _local_scan_steps(
        total_steps,
        cfg.local_scan_chains,
        cfg.local_scan_burn_in_fraction,
    )
    kept_total = int(cfg.local_scan_chains * _kept_samples_per_chain(steps_per_chain, burn_in, cfg.local_scan_thin))
    if kept_total <= 0:
        return float("inf")
    return float(min_tail_hits / kept_total)


def _snis_design_variance_objective(
    *,
    log_weights: np.ndarray,
    tail_indicators: np.ndarray,
    p_reference: float,
    n_chains: int,
    chain_kept_samples: int,
    obm_batch_size: int | None,
) -> float:
    logw = np.asarray(log_weights, dtype=float)
    tail = np.asarray(tail_indicators, dtype=float)
    if logw.ndim != 1 or tail.ndim != 1 or logw.size != tail.size or logw.size < 4:
        return float("nan")
    if not np.isfinite(p_reference) or p_reference < 0.0:
        return float("nan")
    if n_chains <= 0 or chain_kept_samples <= 0:
        return float("nan")

    shift = float(np.max(logw))
    weights = np.exp(logw - shift)
    mean_w = float(np.mean(weights))
    n_total = int(weights.size)
    if not np.isfinite(mean_w) or mean_w <= 0.0:
        return float("nan")
    if n_chains * chain_kept_samples != n_total:
        return float("nan")

    h_all = weights * (tail - float(p_reference))
    var_mean_h = 0.0
    start = 0
    for _ in range(n_chains):
        stop = start + chain_kept_samples
        h_chain = h_all[start:stop]
        start = stop
        sigma2_chain, _ = obm_long_run_variance(
            h_chain,
            batch_size=obm_batch_size,
        )
        if np.isfinite(sigma2_chain) and chain_kept_samples > 0:
            var_mean_h += (chain_kept_samples * float(sigma2_chain)) / (n_total * n_total)
    if not np.isfinite(var_mean_h) or var_mean_h <= 0.0:
        return float("nan")
    return float(var_mean_h / (mean_w * mean_w))


def _local_scan_pool_indices(rows: list[dict[str, Any]]) -> tuple[list[int], bool]:
    eligible_indices = [idx for idx, row in enumerate(rows) if bool(row["guardrail_passed"])]
    if eligible_indices:
        return eligible_indices, True
    fallback_indices = [
        idx for idx, row in enumerate(rows)
        if np.isfinite(row["selection_objective_p0"]) and float(row["selection_objective_p0"]) > 0.0
    ]
    return fallback_indices, False


def _run_local_scan_stage(
    problem: PermutationTestProblem,
    cfg: MCMCWorkflowConfig,
    *,
    candidates: list[dict[str, float]],
    p0_reference: float,
    sigma_t: float,
    q_target: float,
    total_steps: int,
    seed: int,
    stage: str,
) -> dict[str, Any]:
    steps_per_chain, burn_in = _local_scan_steps(
        total_steps,
        cfg.local_scan_chains,
        cfg.local_scan_burn_in_fraction,
    )
    rows: list[dict[str, Any]] = []
    chain_kept_samples = _kept_samples_per_chain(steps_per_chain, burn_in, cfg.local_scan_thin)
    t_start = time.perf_counter()
    for idx, candidate in enumerate(candidates):
        beta = float(candidate["beta"])
        q_scan_target = float(candidate["q_scan_target"])
        res = run_mcmc_is(
            problem,
            beta=beta,
            sigma_t=sigma_t,
            n_steps=steps_per_chain,
            burn_in=burn_in,
            thin=cfg.local_scan_thin,
            n_chains=cfg.local_scan_chains,
            seed=seed + 10_000 * idx,
            init="random",
            tilt_mode=str(cfg.tilt_mode),
            proposal_fraction=cfg.proposal_fraction,
            proposal_swaps=cfg.proposal_swaps,
            estimate_variance=True,
            obm_batch_size=cfg.obm_batch_size,
        )
        var_hat = (
            float(res.snis_variance_obm)
            if res.snis_variance_obm is not None and np.isfinite(res.snis_variance_obm) and res.snis_variance_obm > 0.0
            else np.nan
        )
        q_hat = float(res.tail_share_raw_sample)
        q_log_ratio_abs = (
            float(abs(np.log(q_hat / q_target)))
            if q_hat > 0.0 and np.isfinite(q_hat) and q_target > 0.0 and np.isfinite(q_target)
            else float("inf")
        )
        ess = float(res.ess)
        weight_cv = float(res.weight_summary.cv)
        n_weighted_samples = int(res.n_weighted_samples)
        tail_hits_weighted_sample = int(res.tail_hits_weighted_sample)
        selection_objective_p0 = _snis_design_variance_objective(
            log_weights=np.asarray(res.log_weights, dtype=float),
            tail_indicators=np.asarray(res.tail_indicators, dtype=np.int8),
            p_reference=float(p0_reference),
            n_chains=int(cfg.local_scan_chains),
            chain_kept_samples=int(chain_kept_samples),
            obm_batch_size=cfg.obm_batch_size,
        )
        ess_guardrail_min = float(max(cfg.local_scan_min_ess, cfg.local_scan_min_ess_fraction * n_weighted_samples))
        guardrail_fail_reasons: list[str] = []
        if not np.isfinite(selection_objective_p0) or selection_objective_p0 <= 0.0:
            guardrail_fail_reasons.append("selection_objective_p0_nonpositive")
        if not np.isfinite(ess) or ess < ess_guardrail_min:
            guardrail_fail_reasons.append("ess_below_guardrail")
        rows.append(
            {
                "stage": stage,
                "beta": float(beta),
                "q_scan_target": q_scan_target,
                "q_scan_target_ratio": float(q_scan_target / q_target) if q_target > 0.0 else float("nan"),
                "estimate": float(res.estimate),
                "variance_estimate": var_hat,
                "selection_objective_p0": selection_objective_p0,
                "q_hat": q_hat,
                "q_log_ratio_abs": q_log_ratio_abs,
                "ess": ess,
                "n_weighted_samples": n_weighted_samples,
                "tail_hits_weighted_sample": tail_hits_weighted_sample,
                "ess_guardrail_min": ess_guardrail_min,
                "acceptance_rate": float(res.overall_acceptance_rate),
                "weight_cv": weight_cv,
                "guardrail_passed": int(not guardrail_fail_reasons),
                "guardrail_fail_reasons": guardrail_fail_reasons,
                "screen_rank": None,
                "advanced_to_final": None,
                "advanced_reason": None,
                "objective_ratio_to_best": None,
                "within_objective_tolerance": None,
                "variance_ratio_to_best": None,
                "within_variance_tolerance": None,
            }
        )

    return {
        "rows": rows,
        "steps_per_chain": int(steps_per_chain),
        "burn_in": int(burn_in),
        "eval_total": int(len(candidates) * _mcmc_eval_count(steps_per_chain, cfg.local_scan_chains)),
        "wall_time_sec": float(time.perf_counter() - t_start),
    }


def _count_short_chain_calls_from_history(history: list[dict[str, Any]]) -> int:
    n = 0
    for record in history:
        stage = str(record.get("stage", ""))
        if stage == "init" or stage.startswith("bracket_") or ":rep" in stage:
            n += 1
    return n


def _mcmc_eval_count(n_steps: int, n_chains: int) -> int:
    return int(n_chains * (n_steps + 1))


def _positive_for_plot(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    out[out <= 0.0] = np.nan
    if np.all(np.isnan(out)):
        return np.asarray([np.nan], dtype=float)
    return out


def _set_log_ylim(ax, arrays: list[np.ndarray], q_lo: float = 0.05, q_hi: float = 0.95, pad: float = 1.35) -> None:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a) & (a > 0.0)]
        if a.size:
            vals.append(a)
    if not vals:
        return
    flat = np.concatenate(vals)
    lo = float(np.quantile(flat, q_lo))
    hi = float(np.quantile(flat, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi <= lo:
        hi = lo * 1.10
    lo = max(lo / pad, np.min(flat) / pad)
    hi = hi * pad
    if lo > 0.0 and hi > lo:
        ax.set_ylim(lo, hi)


def _has_positive_finite(arrays: list[np.ndarray]) -> bool:
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(a) & (a > 0.0)):
            return True
    return False


def _stat_label(problem: PermutationTestProblem) -> str:
    return str(getattr(problem.statistic, "__name__", "statistic"))


def _steps_per_chain(total_steps: int, n_chains: int) -> int:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    return max(int(total_steps // n_chains), 1)


def _burn_in(n_steps: int, burn_in_fraction: float) -> int:
    if not (0.0 <= burn_in_fraction < 1.0):
        raise ValueError("burn_in_fraction must satisfy 0 <= burn_in_fraction < 1.")
    return min(int(burn_in_fraction * n_steps), max(n_steps - 1, 0))


def local_beta_scan(
    problem: PermutationTestProblem,
    cfg: MCMCWorkflowConfig,
    *,
    beta_center: float,
    pilot_t: np.ndarray,
    p0_for_qtarget: float,
    sigma_t: float,
    q_target: float,
    seed: int,
) -> dict[str, Any]:
    if cfg.local_scan_screen_total_steps <= 0:
        raise ValueError("local_scan_screen_total_steps must be positive.")
    if cfg.local_scan_finalist_count <= 0:
        raise ValueError("local_scan_finalist_count must be positive.")
    if cfg.local_scan_variance_near_min_ratio < 1.0:
        raise ValueError("local_scan_variance_near_min_ratio must be >= 1.")
    if not cfg.local_scan_enabled:
        return {
            "enabled": False,
            "selected_beta": float(beta_center),
            "rows": [],
            "scan_eval_total": 0,
            "scan_wall_time_sec": 0.0,
        }
    if beta_center <= 0.0:
        return {
            "enabled": True,
            "selected_beta": float(beta_center),
            "rows": [],
            "scan_eval_total": 0,
            "scan_wall_time_sec": 0.0,
            "selected_reason": "beta_center_nonpositive",
        }

    q_multipliers = tuple(float(mult) for mult in cfg.local_scan_q_multipliers)
    screen_q_floor = _local_scan_q_floor(
        int(cfg.local_scan_screen_total_steps),
        cfg,
        int(cfg.local_scan_screen_min_tail_hits),
    )
    final_q_floor = _local_scan_q_floor(
        int(cfg.local_scan_total_steps),
        cfg,
        int(cfg.local_scan_final_min_tail_hits),
    )
    q_candidates = [
        float(min(max(q_target * mult, final_q_floor), 1.0 - 1e-12))
        for mult in q_multipliers
        if mult > 0.0
    ]
    candidates: list[dict[str, float]] = []
    seen_betas: list[float] = []
    for q_candidate in q_candidates:
        beta_candidate = (
            float(beta_center)
            if np.isclose(q_candidate, q_target, rtol=1e-12, atol=1e-15)
            else float(
                init_beta_from_iid_pilot(
                    pilot_T=pilot_t,
                    T_obs=problem.t_obs,
                    sigma_T=sigma_t,
                    p0=p0_for_qtarget,
                    q_target=q_candidate,
                    beta_max=cfg.beta_max_init,
                )
            )
        )
        if beta_candidate <= 0.0 or not np.isfinite(beta_candidate):
            continue
        if any(np.isclose(beta_candidate, beta_prev, rtol=1e-10, atol=1e-12) for beta_prev in seen_betas):
            continue
        seen_betas.append(beta_candidate)
        candidates.append(
            {
                "beta": beta_candidate,
                "q_scan_target": float(q_candidate),
            }
        )
    candidates = sorted(candidates, key=lambda row: float(row["beta"]))
    t_start = time.perf_counter()

    screen = _run_local_scan_stage(
        problem,
        cfg,
        candidates=candidates,
        p0_reference=float(p0_for_qtarget),
        sigma_t=sigma_t,
        q_target=q_target,
        total_steps=int(cfg.local_scan_screen_total_steps),
        seed=seed,
        stage="screen",
    )
    screen_rows = screen["rows"]
    screen_pool_indices, screen_used_guardrails = _local_scan_pool_indices(screen_rows)
    ranked_screen_indices = sorted(
        screen_pool_indices,
        key=lambda idx: (
            float(screen_rows[idx]["selection_objective_p0"]),
            float(screen_rows[idx]["beta"]),
        ),
    )
    target_finalist_count = min(int(cfg.local_scan_finalist_count), len(candidates))
    if len(ranked_screen_indices) < target_finalist_count:
        supplemental_indices = [
            idx
            for idx, row in enumerate(screen_rows)
            if idx not in ranked_screen_indices
            and np.isfinite(row["selection_objective_p0"])
            and float(row["selection_objective_p0"]) > 0.0
        ]
        supplemental_indices = sorted(
            supplemental_indices,
            key=lambda idx: (
                float(screen_rows[idx]["selection_objective_p0"]),
                float(screen_rows[idx]["beta"]),
            ),
        )
        ranked_screen_indices.extend(supplemental_indices)
    if not ranked_screen_indices:
        ranked_screen_indices = list(range(len(screen_rows)))

    finalist_indices = [int(idx) for idx in ranked_screen_indices[:target_finalist_count]]
    if len(finalist_indices) < target_finalist_count:
        for idx in range(len(screen_rows)):
            if idx in finalist_indices:
                continue
            finalist_indices.append(int(idx))
            if len(finalist_indices) >= target_finalist_count:
                break

    finalist_candidates = [
        {
            "beta": float(screen_rows[idx]["beta"]),
            "q_scan_target": float(screen_rows[idx]["q_scan_target"]),
        }
        for idx in finalist_indices
    ]
    for rank, idx in enumerate(ranked_screen_indices):
        screen_rows[idx]["screen_rank"] = int(rank)
    for idx, row in enumerate(screen_rows):
        row["advanced_to_final"] = int(idx in finalist_indices)
        if idx in ranked_screen_indices[:target_finalist_count]:
            row["advanced_reason"] = "screen_objective_rank"

    final = _run_local_scan_stage(
        problem,
        cfg,
        candidates=finalist_candidates,
        p0_reference=float(p0_for_qtarget),
        sigma_t=sigma_t,
        q_target=q_target,
        total_steps=int(cfg.local_scan_total_steps),
        seed=seed + 1_000_000,
        stage="final",
    )
    rows = final["rows"]
    candidate_indices, final_used_guardrails = _local_scan_pool_indices(rows)
    if candidate_indices:
        best_objective = float(min(float(rows[idx]["selection_objective_p0"]) for idx in candidate_indices))
        objective_limit = float(best_objective * cfg.local_scan_variance_near_min_ratio)
        near_min_indices: list[int] = []
        for idx in candidate_indices:
            objective = float(rows[idx]["selection_objective_p0"])
            ratio = float(objective / best_objective) if best_objective > 0.0 else float("inf")
            within_tol = bool(objective <= objective_limit)
            rows[idx]["objective_ratio_to_best"] = ratio
            rows[idx]["within_objective_tolerance"] = int(within_tol)
            if within_tol:
                near_min_indices.append(idx)
        best_idx = min(
            near_min_indices,
            key=lambda idx: (
                float(rows[idx]["beta"]),
                float(rows[idx]["selection_objective_p0"]),
            ),
        )
        selected_beta = float(rows[best_idx]["beta"])
        selected_reason = (
            "two_stage_smallest_beta_within_objective_tolerance"
            if final_used_guardrails
            else "two_stage_smallest_beta_within_objective_tolerance_no_guardrail_survivors"
        )
    else:
        selected_beta = float(beta_center)
        selected_reason = "fallback_beta_center"

    return {
        "enabled": True,
        "selected_beta": selected_beta,
        "selected_reason": selected_reason,
        "steps_per_chain": int(final["steps_per_chain"]),
        "burn_in": int(final["burn_in"]),
        "screen_steps_per_chain": int(screen["steps_per_chain"]),
        "screen_burn_in": int(screen["burn_in"]),
        "screen_eval_total": int(screen["eval_total"]),
        "final_eval_total": int(final["eval_total"]),
        "scan_eval_total": int(screen["eval_total"] + final["eval_total"]),
        "scan_wall_time_sec": float(time.perf_counter() - t_start),
        "screen_wall_time_sec": float(screen["wall_time_sec"]),
        "final_wall_time_sec": float(final["wall_time_sec"]),
        "selection_metrics": {
            "selection_rule": {
                "type": "two_stage_smallest_beta_within_objective_tolerance",
                "objective": "design_point_obm_p0",
                "objective_near_min_ratio": float(cfg.local_scan_variance_near_min_ratio),
                "preference_within_tolerance": "smaller_beta",
            },
            "screening_rule": {
                "type": "low_budget_filter",
                "screen_total_steps": int(cfg.local_scan_screen_total_steps),
                "final_total_steps": int(cfg.local_scan_total_steps),
                "finalist_count": int(cfg.local_scan_finalist_count),
                "screen_ranking": "smallest_design_point_obm_objective",
                "screen_min_tail_hits": int(cfg.local_scan_screen_min_tail_hits),
                "final_min_tail_hits": int(cfg.local_scan_final_min_tail_hits),
                "screen_q_floor": float(screen_q_floor),
                "final_q_floor": float(final_q_floor),
                "screen_tail_hits_used_for_ranking": False,
                "screen_used_guardrails": bool(screen_used_guardrails),
                "final_used_guardrails": bool(final_used_guardrails),
            },
            "q_ladder": {
                "type": "pilot_mapped_q_targets",
                "q_target_center": float(q_target),
                "q_multipliers": [float(mult) for mult in q_multipliers],
                "q_candidates": [float(row["q_scan_target"]) for row in candidates],
            },
            "guardrails": {
                "selection_objective_p0_positive": True,
                "min_ess": float(cfg.local_scan_min_ess),
                "min_ess_fraction_of_weighted_samples": float(cfg.local_scan_min_ess_fraction),
            },
        },
        "screen_rows": screen_rows,
        "rows": rows,
    }


def build_beta_workflow(
    problem: PermutationTestProblem,
    exact_p: float,
    cfg: MCMCWorkflowConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    p0_for_qtarget = float(exact_p) if cfg.use_true_p0_for_q_target else float(cfg.p0_guess)
    q_target = float(p0_for_qtarget ** cfg.d_alpha)

    t_start = time.perf_counter()
    pilot_t = iid_pilot_statistics(problem, n_samples=cfg.pilot_samples, seed=seed)
    sigma_t = estimate_scale_T(pilot_t, method=cfg.scale_method)
    beta0_formula = float(np.sqrt(np.log(1.0 / exact_p)))
    beta0_laplace = init_beta_from_iid_pilot(
        pilot_T=pilot_t,
        T_obs=problem.t_obs,
        sigma_T=sigma_t,
        p0=p0_for_qtarget,
        q_target=q_target,
        beta_max=cfg.beta_max_init,
    )

    tune_burn_in = _burn_in(cfg.tune_steps, cfg.tune_burn_in_fraction)
    runner = make_short_chain_q_runner(
        problem,
        sigma_T=sigma_t,
        thin=cfg.tune_thin,
        proposal_fraction=cfg.proposal_fraction,
        proposal_swaps=cfg.proposal_swaps,
        seed=seed + 1,
    )
    tuning = tune_beta_to_target_q(
        run_short_chain_fn=runner,
        init_state=problem.y_obs,
        beta0=beta0_laplace,
        q_target=q_target,
        n_steps=cfg.tune_steps,
        burn_in=tune_burn_in,
        bracket_factor=cfg.tune_bracket_factor,
        tol_rel=cfg.tune_tol_rel,
        max_bracket_iter=cfg.tune_max_bracket,
        max_bisect_iter=cfg.tune_max_bisect,
        replicate=cfg.tune_replicate,
        reuse_state=cfg.tune_reuse_state,
    )
    tuning_wall_time_sec = float(time.perf_counter() - t_start)
    n_short_calls = _count_short_chain_calls_from_history(tuning["history"])
    pilot_eval_total = int(cfg.pilot_samples)
    tuning_chain_eval_total = int(n_short_calls * (cfg.tune_steps + 1))
    tuning_eval_total = int(pilot_eval_total + tuning_chain_eval_total)
    beta_tuned = float(tuning["beta_hat"])

    if cfg.beta_override is not None:
        beta_used = float(cfg.beta_override)
        scan = {
            "enabled": False,
            "selected_beta": beta_used,
            "rows": [],
            "scan_eval_total": 0,
            "scan_wall_time_sec": 0.0,
            "selected_reason": "beta_override",
        }
    else:
        scan = local_beta_scan(
            problem,
            cfg,
            beta_center=beta_tuned,
            pilot_t=pilot_t,
            p0_for_qtarget=p0_for_qtarget,
            sigma_t=float(sigma_t),
            q_target=q_target,
            seed=seed + 7,
        )
        beta_used = float(scan["selected_beta"])

    scan_eval_total = int(scan.get("scan_eval_total", 0))
    beta_selection_eval_total = int(tuning_eval_total + scan_eval_total)
    beta_selection_wall_time_sec = float(tuning_wall_time_sec + float(scan.get("scan_wall_time_sec", 0.0)))

    return {
        "beta0_formula": beta0_formula,
        "beta0_laplace": float(beta0_laplace),
        "beta_hat_tuned": beta_tuned,
        "beta_used": beta_used,
        "sigma_t": float(sigma_t),
        "p0_for_qtarget": p0_for_qtarget,
        "q_target": q_target,
        "q_hat_beta_hat": float(tuning["q_hat"]),
        "bracket_succeeded": bool(tuning["bracket_succeeded"]),
        "n_short_chain_calls": int(n_short_calls),
        "pilot_eval_total": pilot_eval_total,
        "tuning_chain_eval_total": tuning_chain_eval_total,
        "tuning_eval_total": int(tuning_eval_total),
        "scan_eval_total": scan_eval_total,
        "tuning_wall_time_sec": tuning_wall_time_sec,
        "beta_selection_eval_total": beta_selection_eval_total,
        "beta_selection_budget_breakdown": {
            "pilot_eval_total": pilot_eval_total,
            "tuning_chain_eval_total": tuning_chain_eval_total,
            "tuning_eval_total": int(tuning_eval_total),
            "screen_eval_total": int(scan.get("screen_eval_total", 0)),
            "final_eval_total": int(scan.get("final_eval_total", 0)),
            "scan_eval_total": scan_eval_total,
            "beta_selection_eval_total": beta_selection_eval_total,
        },
        "beta_selection_wall_time_sec": beta_selection_wall_time_sec,
        "local_scan": scan,
        "history_tail": tuning["history"][-5:],
        "tune_steps": int(cfg.tune_steps),
        "tune_burn_in": int(tune_burn_in),
        "tune_thin": int(cfg.tune_thin),
    }


def tune_samc_setup(problem: PermutationTestProblem, cfg: SAMCWorkflowConfig, *, seed: int) -> dict[str, Any]:
    pilot_t = iid_pilot_statistics(problem, n_samples=cfg.lambda_min_pilot, seed=seed)
    lambda_min = float(np.min(pilot_t))
    if lambda_min >= problem.t_obs:
        lambda_min = float(problem.t_obs - 1.0)
    finite_edges = np.linspace(lambda_min, float(problem.t_obs), cfg.n_bins, dtype=float)
    bin_edges = np.concatenate([finite_edges, np.asarray([np.inf], dtype=float)])
    return {"lambda_min": lambda_min, "bin_edges": bin_edges}


def plot_iid_stat_density(
    problem: PermutationTestProblem,
    scenario_name: str,
    *,
    exact_p: float,
    n_samples: int,
    seed: int,
    save_path: Path | None = None,
) -> dict[str, Any]:
    t_vals = iid_pilot_statistics(problem, n_samples=n_samples, seed=seed)
    summary = _iid_stat_summary_from_values(t_vals, t_obs=float(problem.t_obs))
    t_obs = float(problem.t_obs)

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8))
    hist_vals, bin_edges, _ = ax.hist(
        t_vals,
        bins=70,
        density=True,
        alpha=0.35,
        color="#4e79a7",
        edgecolor="none",
    )
    ax.axvline(t_obs, color="black", linestyle="--", linewidth=1.3, label=f"T_obs={t_obs:.4g}")

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    tail_mask = np.asarray([problem.is_in_tail(float(c)) for c in centers], dtype=bool)
    if np.any(tail_mask):
        widths = np.diff(bin_edges)
        ax.bar(
            bin_edges[:-1][tail_mask],
            hist_vals[tail_mask],
            width=widths[tail_mask],
            align="edge",
            color="#f28e2b",
            alpha=0.18,
            edgecolor="none",
            label="tail region",
            zorder=3,
        )

    ax.set_title(f"IID statistic density: {scenario_name}")
    ax.set_xlabel(_stat_label(problem))
    ax.set_ylabel("density")
    ax.legend(loc="best")
    fig.suptitle(f"Empirical null statistic shape (n={n_samples:,}, true p={exact_p:.3e})")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return summary


def _iid_stat_summary_from_values(t_vals: np.ndarray, *, t_obs: float) -> dict[str, Any]:
    vals = np.asarray(t_vals, dtype=float)
    return {
        "n_samples": int(vals.size),
        "t_obs": float(t_obs),
        "t_min": float(np.nanmin(vals)),
        "t_max": float(np.nanmax(vals)),
        "t_mean": float(np.nanmean(vals)),
        "t_sd": float(np.nanstd(vals, ddof=1)),
    }


def iid_stat_density_summary(
    problem: PermutationTestProblem,
    *,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    t_vals = iid_pilot_statistics(problem, n_samples=n_samples, seed=seed)
    return _iid_stat_summary_from_values(t_vals, t_obs=float(problem.t_obs))


def _annotate_error_fields(row: dict[str, Any], exact_p: float) -> dict[str, Any]:
    est = float(row["estimate"])
    row["exact_p"] = float(exact_p)
    row["bias"] = float(est - exact_p)
    row["squared_error"] = float((est - exact_p) ** 2)
    row["root_squared_error"] = float(np.sqrt((est - exact_p) ** 2))
    row["rel_error"] = float((est - exact_p) / exact_p)
    row["abs_log10_error"] = float(abs(np.log10(est) - np.log10(exact_p))) if est > 0.0 else np.nan
    return row


def _run_iid_cumulative_checkpoints(
    problem: PermutationTestProblem,
    exact_p: float,
    *,
    checkpoints: tuple[int, ...],
    seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    checkpoints = _sorted_unique_points(checkpoints)
    rng = np.random.default_rng(seed)
    max_checkpoint = int(checkpoints[-1])
    rows: list[dict[str, Any]] = []
    next_idx = 0
    tail_hits = 0
    t_start = time.perf_counter()
    for step in range(1, max_checkpoint + 1):
        y = problem.sample_uniform_labels(rng)
        tail_hits += int(problem.is_in_tail_y(y))
        if next_idx < len(checkpoints) and step == checkpoints[next_idx]:
            estimate = float(tail_hits / step)
            standard_error = float(np.sqrt(estimate * (1.0 - estimate) / step))
            ci_low, ci_high = wilson_interval(tail_hits, step, confidence_level=confidence_level)
            rows.append(
                _annotate_error_fields(
                    {
                        "method": "iid",
                        "checkpoint": int(step),
                        "estimate": estimate,
                        "variance_estimate": float(standard_error ** 2),
                        "tail_hits": int(tail_hits),
                        "tail_share_raw": estimate,
                        "zero_hits": int(tail_hits == 0),
                        "wall_time_sec": float(time.perf_counter() - t_start),
                        "eval_excl_tuning": float(step),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                    },
                    exact_p,
                )
            )
            next_idx += 1
            if next_idx >= len(checkpoints):
                break
    return rows


def _run_single_chain_full_trace(
    problem: PermutationTestProblem,
    rng: np.random.Generator,
    *,
    beta: float,
    sigma_t: float,
    n_steps: int,
    init: str,
    tilt_mode: str,
    n_swap_pairs: int,
    checkpoint_steps: tuple[int, ...],
) -> dict[str, Any]:
    if init == "observed":
        y = problem.y_obs.copy()
    elif init == "random":
        y = problem.sample_uniform_labels(rng)
    else:
        raise ValueError("init must be either 'observed' or 'random'.")

    t_cur = problem.compute_stat(y)
    if tilt_mode == "smooth_hinge":
        q_cur = right_tail_deficit_scaled(t_cur, problem.t_obs, sigma_t)
    else:
        q_cur = right_tail_step_shortfall(t_cur, problem.t_obs)

    t_trace = np.empty(n_steps, dtype=float)
    q_trace = np.empty(n_steps, dtype=float)
    tail_trace = np.empty(n_steps, dtype=np.int8)
    accepted_trace = np.zeros(n_steps, dtype=np.int8)
    elapsed_by_step: dict[int, float] = {}

    accepted = 0
    t_start = time.perf_counter()
    checkpoint_set = set(int(v) for v in checkpoint_steps)

    for step in range(1, n_steps + 1):
        y_prop = propose_localized_swaps(y, rng, n_swap_pairs=n_swap_pairs)
        t_prop = problem.compute_stat(y_prop)
        if tilt_mode == "smooth_hinge":
            q_prop = right_tail_deficit_scaled(t_prop, problem.t_obs, sigma_t)
        else:
            q_prop = right_tail_step_shortfall(t_prop, problem.t_obs)

        log_alpha = -beta * (q_prop - q_cur)
        if log_alpha >= 0.0 or np.log(rng.random()) < log_alpha:
            y = y_prop
            t_cur = t_prop
            q_cur = q_prop
            accepted += 1
            accepted_trace[step - 1] = 1

        t_trace[step - 1] = float(t_cur)
        q_trace[step - 1] = float(q_cur)
        tail_trace[step - 1] = int(problem.is_in_tail(t_cur))
        if step in checkpoint_set:
            elapsed_by_step[int(step)] = float(time.perf_counter() - t_start)

    return {
        "t_trace": t_trace,
        "q_trace": q_trace,
        "tail_trace": tail_trace,
        "accepted_trace": accepted_trace,
        "accepted_total": int(accepted),
        "elapsed_by_step": elapsed_by_step,
    }


def _mcmc_prefix_row(
    *,
    exact_p: float,
    checkpoint: int,
    steps_per_chain: int,
    burn_in: int,
    thin: int,
    beta: float,
    sigma_t: float,
    tilt_mode: str,
    estimate_variance: bool,
    obm_batch_size: int | None,
    traces: list[dict[str, Any]],
    n_chains: int,
) -> dict[str, Any]:
    t_chunks: list[np.ndarray] = []
    q_chunks: list[np.ndarray] = []
    tail_chunks: list[np.ndarray] = []
    retained_lengths: list[int] = []
    acceptance_rates: list[float] = []
    total_accepted = 0
    total_proposals = int(steps_per_chain * n_chains)
    wall_time_sec = 0.0

    for trace in traces:
        chain_t = trace["t_trace"][burn_in:steps_per_chain:thin]
        chain_q = trace["q_trace"][burn_in:steps_per_chain:thin]
        chain_tail = trace["tail_trace"][burn_in:steps_per_chain:thin]
        t_chunks.append(chain_t)
        q_chunks.append(chain_q)
        tail_chunks.append(chain_tail)
        retained_lengths.append(int(chain_q.size))

        chain_accepted = int(np.sum(trace["accepted_trace"][:steps_per_chain]))
        total_accepted += chain_accepted
        acceptance_rates.append(float(chain_accepted / steps_per_chain))
        wall_time_sec += float(trace["elapsed_by_step"][steps_per_chain])

    q_samples = np.concatenate(q_chunks)
    tail_indicators = np.concatenate(tail_chunks)
    log_weights = beta * q_samples
    shift = float(np.max(log_weights))
    weights = np.exp(log_weights - shift)

    weight_sum = float(np.sum(weights))
    estimate = float(np.dot(weights, tail_indicators) / weight_sum)
    ess = float(effective_sample_size(weights))
    weight_summary = summarize_weights(weights)

    snis_variance_obm = np.nan
    snis_mcse_obm = np.nan
    if estimate_variance:
        n_total = int(weights.size)
        mean_w = float(np.mean(weights))
        h_all = weights * (tail_indicators - estimate)
        if mean_w > 0.0 and n_total >= 4:
            start = 0
            var_mean_h = 0.0
            for m in retained_lengths:
                stop = start + m
                h_chain = h_all[start:stop]
                start = stop
                sigma2_chain, _ = obm_long_run_variance(h_chain, batch_size=obm_batch_size)
                if np.isfinite(sigma2_chain) and m > 0:
                    var_mean_h += (m * sigma2_chain) / (n_total * n_total)
            snis_variance_obm = float(var_mean_h / (mean_w * mean_w))
            snis_mcse_obm = float(np.sqrt(max(snis_variance_obm, 0.0)))

    return _annotate_error_fields(
        {
            "method": "mcmc_is",
            "checkpoint": int(checkpoint),
            "estimate": estimate,
            "variance_estimate": float(snis_variance_obm) if np.isfinite(snis_variance_obm) else np.nan,
            "snis_mcse_obm": float(snis_mcse_obm) if np.isfinite(snis_mcse_obm) else np.nan,
            "tail_hits": int(np.sum(tail_indicators)),
            "tail_share_raw": float(np.mean(tail_indicators)),
            "ess": ess,
            "acceptance_rate": float(total_accepted / total_proposals),
            "weight_cv": float(weight_summary.cv),
            "beta": float(beta),
            "sigma_t": float(sigma_t),
            "tilt_mode": str(tilt_mode),
            "wall_time_sec": float(wall_time_sec),
            "eval_excl_tuning": float(_mcmc_eval_count(steps_per_chain, n_chains)),
            "n_weighted_samples": int(weights.size),
            "acceptance_rates": acceptance_rates,
        },
        exact_p,
    )


def _run_mcmc_cumulative_checkpoints(
    problem: PermutationTestProblem,
    exact_p: float,
    *,
    checkpoints: tuple[int, ...],
    reported_checkpoints: tuple[int, ...] | None = None,
    beta: float,
    sigma_t: float,
    cfg: MCMCWorkflowConfig | BetaSweepStudyConfig,
    seed: int,
) -> list[dict[str, Any]]:
    checkpoints = _sorted_unique_points(checkpoints)
    if reported_checkpoints is None:
        reported_checkpoints = checkpoints
    reported_checkpoints = tuple(int(v) for v in reported_checkpoints)
    if len(reported_checkpoints) != len(checkpoints):
        raise ValueError("reported_checkpoints must match checkpoints length.")
    max_total_steps = int(checkpoints[-1])
    max_steps_per_chain = _steps_per_chain(max_total_steps, cfg.chains)
    steps_per_checkpoint = {int(cp): _steps_per_chain(int(cp), cfg.chains) for cp in checkpoints}
    unique_step_checkpoints = tuple(sorted(set(steps_per_checkpoint.values())))

    if cfg.proposal_swaps is not None:
        n_swap_pairs = int(cfg.proposal_swaps)
    else:
        n_swap_pairs = n_swap_pairs_from_fraction(
            problem.n_treated,
            problem.n_control,
            proposal_fraction=cfg.proposal_fraction,
        )

    seed_seq = np.random.SeedSequence(seed)
    traces: list[dict[str, Any]] = []
    for ss in seed_seq.spawn(cfg.chains):
        rng = np.random.default_rng(ss)
        traces.append(
            _run_single_chain_full_trace(
                problem,
                rng,
                beta=beta,
                sigma_t=sigma_t,
                n_steps=max_steps_per_chain,
                init="random",
                tilt_mode=str(cfg.tilt_mode),
                n_swap_pairs=n_swap_pairs,
                checkpoint_steps=unique_step_checkpoints,
            )
        )

    rows: list[dict[str, Any]] = []
    for checkpoint, report_checkpoint in zip(checkpoints, reported_checkpoints):
        steps_per_chain = steps_per_checkpoint[int(checkpoint)]
        burn_in = _burn_in(steps_per_chain, cfg.burn_in_fraction)
        row = _mcmc_prefix_row(
                exact_p=exact_p,
                checkpoint=int(report_checkpoint),
                steps_per_chain=steps_per_chain,
                burn_in=burn_in,
                thin=int(cfg.thin),
                beta=float(beta),
                sigma_t=float(sigma_t),
                tilt_mode=str(cfg.tilt_mode),
                estimate_variance=bool(cfg.estimate_variance),
                obm_batch_size=cfg.obm_batch_size,
                traces=traces,
                n_chains=int(cfg.chains),
            )
        row["mcmc_chain_budget"] = int(checkpoint)
        row["mcmc_reported_budget"] = int(report_checkpoint)
        rows.append(row)
    return rows


def _run_samc_cumulative_checkpoints(
    problem: PermutationTestProblem,
    exact_p: float,
    *,
    checkpoints: tuple[int, ...],
    samc_setup: dict[str, Any],
    cfg: SAMCWorkflowConfig,
    seed: int,
) -> list[dict[str, Any]]:
    checkpoints = _sorted_unique_points(checkpoints)
    max_steps = int(checkpoints[-1])
    bin_edges = np.asarray(samc_setup["bin_edges"], dtype=float)
    k = int(bin_edges.size - 1)
    tail_bin_index = int(k - 1)
    target = np.full(k, 1.0 / k, dtype=float)

    if cfg.proposal_swaps is not None:
        n_swap_pairs = int(cfg.proposal_swaps)
    else:
        n_swap_pairs = n_swap_pairs_from_fraction(
            problem.n_treated,
            problem.n_control,
            proposal_fraction=cfg.proposal_fraction,
        )

    rng = np.random.default_rng(seed)
    y = problem.sample_uniform_labels(rng)
    t_cur = problem.compute_stat(y)
    b_cur = _bin_index(t_cur, bin_edges)
    theta = np.zeros(k, dtype=float)
    bin_trace = np.empty(max_steps, dtype=np.int64)
    theta_at_checkpoint: dict[int, np.ndarray] = {}
    elapsed_by_checkpoint: dict[int, float] = {}
    accepted_by_checkpoint: dict[int, int] = {}

    accepted = 0
    t_start = time.perf_counter()
    checkpoint_set = set(int(v) for v in checkpoints)

    for step in range(1, max_steps + 1):
        y_prop = propose_localized_swaps(y, rng, n_swap_pairs=n_swap_pairs)
        t_prop = problem.compute_stat(y_prop)
        b_prop = _bin_index(t_prop, bin_edges)

        log_alpha = theta[b_cur] - theta[b_prop]
        if log_alpha >= 0.0 or np.log(rng.random()) < log_alpha:
            y = y_prop
            t_cur = t_prop
            b_cur = b_prop
            accepted += 1

        gamma = _default_stepsize(step, t0=cfg.t0)
        theta -= gamma * target
        theta[b_cur] += gamma
        theta -= np.mean(theta)
        bin_trace[step - 1] = b_cur

        if step in checkpoint_set:
            theta_at_checkpoint[int(step)] = theta.copy()
            elapsed_by_checkpoint[int(step)] = float(time.perf_counter() - t_start)
            accepted_by_checkpoint[int(step)] = int(accepted)

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        burn_in = _burn_in(int(checkpoint), cfg.burn_in_fraction)
        retained_bins = bin_trace[burn_in:int(checkpoint)]
        visit_counts = np.bincount(retained_bins, minlength=k).astype(np.int64)
        rel_freq_error = _relative_sampling_frequency_error(visit_counts)
        max_abs_rel_freq_error = float(np.max(np.abs(rel_freq_error)))
        estimate, pi0_adjustment, empty_bin_indices = _paper_pvalue_estimate(
            theta=theta_at_checkpoint[int(checkpoint)],
            target=target,
            visit_counts=visit_counts,
            tail_bin_index=tail_bin_index,
        )
        rows.append(
            _annotate_error_fields(
                {
                    "method": "samc",
                    "checkpoint": int(checkpoint),
                    "estimate": float(estimate),
                    "variance_estimate": samc_variance_proxy(float(estimate), int(checkpoint), burn_in),
                    "acceptance_rate": float(accepted_by_checkpoint[int(checkpoint)] / int(checkpoint)),
                    "samc_max_rel_freq_error": max_abs_rel_freq_error,
                    "samc_converged": int(max_abs_rel_freq_error < cfg.convergence_tolerance),
                    "samc_pi0": float(pi0_adjustment),
                    "samc_empty_bins": int(empty_bin_indices.size),
                    "wall_time_sec": float(elapsed_by_checkpoint[int(checkpoint)]),
                    "eval_excl_tuning": float(int(checkpoint) + 1),
                    "samc_visit_total": int(np.sum(visit_counts)),
                    "samc_tail_bin_freq": float(visitation_frequency(visit_counts)[tail_bin_index]),
                },
                exact_p,
            )
        )
    return rows


def _iid_replicate_worker(
    *,
    scenario_key: str,
    scenario_display: str,
    problem: PermutationTestProblem,
    exact_p: float,
    checkpoints: tuple[int, ...],
    rep: int,
    rep_seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    rows = _run_iid_cumulative_checkpoints(
        problem,
        exact_p,
        checkpoints=checkpoints,
        seed=rep_seed,
        confidence_level=confidence_level,
    )
    for row in rows:
        row["scenario"] = scenario_key
        row["scenario_display"] = scenario_display
        row["replicate"] = int(rep)
        row["beta_selection_budget"] = 0
        row["eval_incl_tuning"] = float(row["eval_excl_tuning"])
    return rows


def _mcmc_cross_replicate_worker(
    *,
    scenario_key: str,
    scenario_display: str,
    problem: PermutationTestProblem,
    exact_p: float,
    checkpoints: tuple[int, ...],
    mcmc_chain_checkpoints: tuple[int, ...],
    beta_workflow: dict[str, Any],
    beta_selection_budget: int,
    mcmc_cfg: MCMCWorkflowConfig,
    rep: int,
    rep_seed: int,
) -> list[dict[str, Any]]:
    rows = _run_mcmc_cumulative_checkpoints(
        problem,
        exact_p,
        checkpoints=mcmc_chain_checkpoints,
        reported_checkpoints=checkpoints,
        beta=float(beta_workflow["beta_used"]),
        sigma_t=float(beta_workflow["sigma_t"]),
        cfg=mcmc_cfg,
        seed=rep_seed + 1,
    )
    for row in rows:
        row["scenario"] = scenario_key
        row["scenario_display"] = scenario_display
        row["replicate"] = int(rep)
        row["beta_selection_budget"] = int(beta_selection_budget)
        row["eval_incl_tuning"] = float(int(row["mcmc_chain_budget"]) + beta_selection_budget)
    return rows


def _samc_replicate_worker(
    *,
    scenario_key: str,
    scenario_display: str,
    problem: PermutationTestProblem,
    exact_p: float,
    checkpoints: tuple[int, ...],
    samc_setup: dict[str, Any],
    samc_cfg: SAMCWorkflowConfig,
    rep: int,
    rep_seed: int,
) -> list[dict[str, Any]]:
    rows = _run_samc_cumulative_checkpoints(
        problem,
        exact_p,
        checkpoints=checkpoints,
        samc_setup=samc_setup,
        cfg=samc_cfg,
        seed=rep_seed + 2,
    )
    for row in rows:
        row["scenario"] = scenario_key
        row["scenario_display"] = scenario_display
        row["replicate"] = int(rep)
        row["beta_selection_budget"] = 0
        row["eval_incl_tuning"] = float(row["eval_excl_tuning"])
    return rows


def _beta_replicate_worker(
    *,
    problem: PermutationTestProblem,
    exact_p: float,
    checkpoints: tuple[int, ...],
    beta: float,
    sigma_t: float,
    beta_cfg: BetaSweepStudyConfig,
    rep: int,
    rep_seed: int,
    multiplier: float,
) -> list[dict[str, Any]]:
    beta_rows = _run_mcmc_cumulative_checkpoints(
        problem,
        exact_p,
        checkpoints=checkpoints,
        beta=beta,
        sigma_t=sigma_t,
        cfg=beta_cfg,
        seed=rep_seed,
    )
    for row in beta_rows:
        row["beta_multiplier"] = float(multiplier)
        row["beta"] = float(beta)
        row["replicate"] = int(rep)
        row["seed"] = int(rep_seed)
        row["q_tilt_tail_share"] = float(row.get("tail_share_raw", np.nan))
    return beta_rows


def summarize_records(
    records: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...] = ("checkpoint", "method"),
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record[field] for field in group_fields)
        groups.setdefault(key, []).append(record)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        sub = groups[key]
        est = np.asarray([row["estimate"] for row in sub], dtype=float)
        sq = np.asarray([row["squared_error"] for row in sub], dtype=float)
        var = np.asarray([row["variance_estimate"] for row in sub], dtype=float)
        exact_p = float(sub[0]["exact_p"])
        emp_var = float(np.var(est, ddof=1)) if est.size > 1 else np.nan
        var_finite = var[np.isfinite(var) & (var > 0.0)]
        mean_var_hat = float(np.mean(var_finite)) if var_finite.size else np.nan
        var_calib = (
            float(emp_var / mean_var_hat)
            if np.isfinite(emp_var) and np.isfinite(mean_var_hat) and mean_var_hat > 0.0
            else np.nan
        )
        abs_log = np.asarray([row.get("abs_log10_error", np.nan) for row in sub], dtype=float)
        abs_log = abs_log[np.isfinite(abs_log)]

        summary = {
            field: value
            for field, value in zip(group_fields, key)
        }
        summary.update(
            {
                "n_runs": int(est.size),
                "exact_p": exact_p,
                "mean_estimate": float(np.mean(est)),
                "median_estimate": float(np.median(est)),
                "bias": float(np.mean(est) - exact_p),
                "rel_bias": float((np.mean(est) - exact_p) / exact_p),
                "rmse": float(np.sqrt(np.mean(sq))),
                "mean_abs_log10_error": float(np.mean(abs_log)) if abs_log.size else np.nan,
                "empirical_var": emp_var,
                "mean_variance_estimate": mean_var_hat,
                "var_calibration_ratio": var_calib,
                "mean_wall_time_sec": float(np.mean([row["wall_time_sec"] for row in sub])),
                "mean_eval_excl_tuning": float(np.mean([row["eval_excl_tuning"] for row in sub])),
                "mean_eval_incl_tuning": (
                    float(np.mean([row.get("eval_incl_tuning", row["eval_excl_tuning"]) for row in sub]))
                ),
                "mean_q_tilt_tail_share": (
                    float(np.mean([row.get("tail_share_raw", np.nan) for row in sub if np.isfinite(row.get("tail_share_raw", np.nan))]))
                    if any(np.isfinite(row.get("tail_share_raw", np.nan)) for row in sub)
                    else np.nan
                ),
                "mean_ess": (
                    float(np.mean([row.get("ess", np.nan) for row in sub if np.isfinite(row.get("ess", np.nan))]))
                    if any(np.isfinite(row.get("ess", np.nan)) for row in sub)
                    else np.nan
                ),
                "mean_acceptance_rate": (
                    float(np.mean([row.get("acceptance_rate", np.nan) for row in sub if np.isfinite(row.get("acceptance_rate", np.nan))]))
                    if any(np.isfinite(row.get("acceptance_rate", np.nan)) for row in sub)
                    else np.nan
                ),
                "mean_zero_rate": (
                    float(np.mean([row.get("zero_hits", np.nan) for row in sub if np.isfinite(row.get("zero_hits", np.nan))]))
                    if any(np.isfinite(row.get("zero_hits", np.nan)) for row in sub)
                    else np.nan
                ),
                "mean_weight_cv": (
                    float(np.mean([row.get("weight_cv", np.nan) for row in sub if np.isfinite(row.get("weight_cv", np.nan))]))
                    if any(np.isfinite(row.get("weight_cv", np.nan)) for row in sub)
                    else np.nan
                ),
                "mean_samc_max_rel_freq_error": (
                    float(np.mean([row.get("samc_max_rel_freq_error", np.nan) for row in sub if np.isfinite(row.get("samc_max_rel_freq_error", np.nan))]))
                    if any(np.isfinite(row.get("samc_max_rel_freq_error", np.nan)) for row in sub)
                    else np.nan
                ),
            }
        )
        out.append(summary)
    return out


def run_cross_method_study(
    scenario: LoadedScenario,
    cross_cfg: CrossMethodStudyConfig,
    mcmc_cfg: MCMCWorkflowConfig,
    samc_cfg: SAMCWorkflowConfig,
) -> dict[str, Any]:
    checkpoints = _sorted_unique_points(cross_cfg.estimation_points)
    beta_workflow = build_beta_workflow(
        scenario.problem,
        scenario.exact_p,
        mcmc_cfg,
        seed=cross_cfg.base_seed + 10_000,
    )
    samc_setup = tune_samc_setup(scenario.problem, samc_cfg, seed=cross_cfg.base_seed + 20_000)
    beta_selection_budget = int(beta_workflow["beta_selection_eval_total"])
    mcmc_chain_checkpoints = tuple(int(cp - beta_selection_budget) for cp in checkpoints)
    if any(cp <= 0 for cp in mcmc_chain_checkpoints):
        raise ValueError(
            "Cross-method estimation_points must all exceed the fixed MCMC-IS beta-selection budget. "
            f"Received estimation_points={list(checkpoints)}, beta_selection_budget={beta_selection_budget}."
        )

    records: list[dict[str, Any]] = []
    repeat_jobs = [(int(rep), int(cross_cfg.base_seed + 1_000 * rep)) for rep in range(cross_cfg.repeats)]
    n_jobs = _effective_n_jobs(cross_cfg.n_jobs, len(repeat_jobs))

    executor = _try_make_process_pool(n_jobs) if n_jobs > 1 else None

    if executor is None:
        for rep, rep_seed in repeat_jobs:
            records.extend(
                _iid_replicate_worker(
                    scenario_key=scenario.key,
                    scenario_display=scenario.description,
                    problem=scenario.problem,
                    exact_p=scenario.exact_p,
                    checkpoints=checkpoints,
                    rep=rep,
                    rep_seed=rep_seed,
                    confidence_level=cross_cfg.confidence_level,
                )
            )
        for rep, rep_seed in repeat_jobs:
            records.extend(
                _mcmc_cross_replicate_worker(
                    scenario_key=scenario.key,
                    scenario_display=scenario.description,
                    problem=scenario.problem,
                    exact_p=scenario.exact_p,
                    checkpoints=checkpoints,
                    mcmc_chain_checkpoints=mcmc_chain_checkpoints,
                    beta_workflow=beta_workflow,
                    beta_selection_budget=beta_selection_budget,
                    mcmc_cfg=mcmc_cfg,
                    rep=rep,
                    rep_seed=rep_seed,
                )
            )
        for rep, rep_seed in repeat_jobs:
            records.extend(
                _samc_replicate_worker(
                    scenario_key=scenario.key,
                    scenario_display=scenario.description,
                    problem=scenario.problem,
                    exact_p=scenario.exact_p,
                    checkpoints=checkpoints,
                    samc_setup=samc_setup,
                    samc_cfg=samc_cfg,
                    rep=rep,
                    rep_seed=rep_seed,
                )
            )
    else:
        with executor:
            futures = [
                executor.submit(
                    _iid_replicate_worker,
                    scenario_key=scenario.key,
                    scenario_display=scenario.description,
                    problem=scenario.problem,
                    exact_p=scenario.exact_p,
                    checkpoints=checkpoints,
                    rep=rep,
                    rep_seed=rep_seed,
                    confidence_level=cross_cfg.confidence_level,
                )
                for rep, rep_seed in repeat_jobs
            ]
            for future in cf.as_completed(futures):
                records.extend(future.result())

            futures = [
                executor.submit(
                    _mcmc_cross_replicate_worker,
                    scenario_key=scenario.key,
                    scenario_display=scenario.description,
                    problem=scenario.problem,
                    exact_p=scenario.exact_p,
                    checkpoints=checkpoints,
                    mcmc_chain_checkpoints=mcmc_chain_checkpoints,
                    beta_workflow=beta_workflow,
                    beta_selection_budget=beta_selection_budget,
                    mcmc_cfg=mcmc_cfg,
                    rep=rep,
                    rep_seed=rep_seed,
                )
                for rep, rep_seed in repeat_jobs
            ]
            for future in cf.as_completed(futures):
                records.extend(future.result())

            futures = [
                executor.submit(
                    _samc_replicate_worker,
                    scenario_key=scenario.key,
                    scenario_display=scenario.description,
                    problem=scenario.problem,
                    exact_p=scenario.exact_p,
                    checkpoints=checkpoints,
                    samc_setup=samc_setup,
                    samc_cfg=samc_cfg,
                    rep=rep,
                    rep_seed=rep_seed,
                )
                for rep, rep_seed in repeat_jobs
            ]
            for future in cf.as_completed(futures):
                records.extend(future.result())

    records = sorted(records, key=lambda row: (int(row["replicate"]), str(row["method"]), int(row["checkpoint"])))
    summary = summarize_records(records)
    density_summary = iid_stat_density_summary(
        scenario.problem,
        n_samples=cross_cfg.iid_density_samples,
        seed=cross_cfg.base_seed + 30_000,
    )
    return {
        "scenario": scenario.key,
        "scenario_display": scenario.description,
        "exact_p": float(scenario.exact_p),
        "exact_method": scenario.exact_method,
        "exact_tail_hits": int(scenario.exact_tail_hits),
        "exact_n_perm": int(scenario.exact_n_perm),
        "estimation_points": list(checkpoints),
        "beta_workflow": beta_workflow,
        "mcmc_beta_selection_budget": beta_selection_budget,
        "samc_setup": {
            "lambda_min": float(samc_setup["lambda_min"]),
            "bin_edges": np.asarray(samc_setup["bin_edges"], dtype=float),
        },
        "records": records,
        "summary": summary,
        "iid_density_summary": density_summary,
    }


def plot_cross_method_max_budget(
    records: list[dict[str, Any]],
    *,
    scenario_name: str,
    exact_p: float,
    max_budget: int,
    beta_workflow: dict[str, Any] | None = None,
    save_path: Path | None = None,
) -> None:
    rows = [row for row in records if int(row["checkpoint"]) == int(max_budget)]
    methods = ["iid", "mcmc_is", "samc"]
    labels = ["IID", "MCMC-IS", "SAMC"]

    est_data = []
    rmse_data = []
    for method in methods:
        sub = [row for row in rows if row["method"] == method]
        est = np.asarray([row["estimate"] for row in sub], dtype=float)
        rse = np.asarray([row["root_squared_error"] for row in sub], dtype=float)
        if method == "iid":
            tail_hits = np.asarray([row.get("tail_hits", 0) for row in sub], dtype=int)
            est[tail_hits <= 0] = np.nan
            rse[tail_hits <= 0] = np.nan
        est_data.append(_positive_for_plot(est))
        rmse_data.append(_positive_for_plot(rse))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    axes[0].boxplot(est_data, tick_labels=labels, showfliers=False)
    if _has_positive_finite(est_data):
        axes[0].set_yscale("log")
    axes[0].axhline(exact_p, color="black", linestyle="--", linewidth=1.2, label=f"true p={exact_p:.2e}")
    _set_log_ylim(axes[0], est_data)
    axes[0].set_title("Estimator distribution")
    axes[0].set_ylabel("p_hat")
    axes[0].legend()

    axes[1].boxplot(rmse_data, tick_labels=labels, showfliers=False)
    if _has_positive_finite(rmse_data):
        axes[1].set_yscale("log")
    _set_log_ylim(axes[1], rmse_data)
    axes[1].set_title("RMSE across repeats")
    axes[1].set_ylabel("root squared error")

    title = f"Cross-method comparison: {scenario_name}\nmax iterations={max_budget:,}, true p={exact_p:.3e}"
    if beta_workflow is not None:
        title = (
            f"{title}\nMCMC-IS beta (laplace/tuned/used): "
            f"{beta_workflow['beta0_laplace']:.4g} / {beta_workflow['beta_hat_tuned']:.4g} / {beta_workflow['beta_used']:.4g}"
            f" | beta-selection budget (pilot+tuning+scan)={int(beta_workflow.get('beta_selection_eval_total', 0)):,}"
        )
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cross_method_convergence(
    summary: list[dict[str, Any]],
    *,
    scenario_name: str,
    exact_p: float,
    mcmc_beta_selection_budget: int = 0,
    save_path: Path | None = None,
) -> None:
    methods = ["iid", "mcmc_is", "samc"]
    labels = {"iid": "IID", "mcmc_is": "MCMC-IS", "samc": "SAMC"}
    colors = {"iid": "#4e79a7", "mcmc_is": "#f28e2b", "samc": "#59a14f"}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for method in methods:
        sub = sorted([row for row in summary if row["method"] == method], key=lambda row: row["checkpoint"])
        x = np.asarray([row["checkpoint"] for row in sub], dtype=float)
        mean_est = np.asarray([row["mean_estimate"] for row in sub], dtype=float)
        rmse = np.asarray([row["rmse"] for row in sub], dtype=float)

        axes[0].plot(x, mean_est, marker="o", label=labels[method], color=colors[method])
        axes[1].plot(x, rmse, marker="o", label=labels[method], color=colors[method])

    axes[0].axhline(exact_p, color="black", linestyle="--", linewidth=1.2, label=f"true p={exact_p:.2e}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("iterations")
    if _has_positive_finite([np.asarray([row["mean_estimate"] for row in summary if row["method"] == method], dtype=float) for method in methods]):
        axes[0].set_yscale("log")
    if _has_positive_finite([np.asarray([row["rmse"] for row in summary if row["method"] == method], dtype=float) for method in methods]):
        axes[1].set_yscale("log")
    axes[0].set_title("Mean estimate vs iterations")
    axes[0].set_ylabel("p_hat")
    axes[1].set_title("RMSE vs iterations")
    axes[1].set_ylabel("RMSE")
    axes[0].legend()
    fig.suptitle(
        f"Cross-method convergence: {scenario_name} (true p={exact_p:.3e})\n"
        f"MCMC-IS total budget includes fixed beta-selection budget (pilot+tuning+scan)={int(mcmc_beta_selection_budget):,}"
    )
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cross_method_diagnostics(
    summary: list[dict[str, Any]],
    *,
    scenario_name: str,
    mcmc_beta_selection_budget: int = 0,
    save_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    iid = sorted([row for row in summary if row["method"] == "iid"], key=lambda row: row["checkpoint"])
    mcmc = sorted([row for row in summary if row["method"] == "mcmc_is"], key=lambda row: row["checkpoint"])
    samc = sorted([row for row in summary if row["method"] == "samc"], key=lambda row: row["checkpoint"])

    axes[0, 0].plot(
        [row["checkpoint"] for row in iid],
        [row["mean_zero_rate"] for row in iid],
        marker="o",
        color="#4e79a7",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("IID zero-hit rate")
    axes[0, 0].set_xlabel("iterations")
    axes[0, 0].set_ylabel("share of runs")

    axes[0, 1].plot(
        [row["checkpoint"] for row in mcmc],
        [row["mean_q_tilt_tail_share"] for row in mcmc],
        marker="o",
        color="#f28e2b",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("MCMC-IS tilted-tail occupancy q")
    axes[0, 1].set_xlabel("iterations")
    axes[0, 1].set_ylabel("q_hat")

    axes[1, 0].plot(
        [row["checkpoint"] for row in mcmc],
        [row["mean_ess"] for row in mcmc],
        marker="o",
        color="#f28e2b",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("MCMC-IS ESS")
    axes[1, 0].set_xlabel("iterations")
    axes[1, 0].set_ylabel("ESS")

    axes[1, 1].plot(
        [row["checkpoint"] for row in samc],
        [row["mean_samc_max_rel_freq_error"] for row in samc],
        marker="o",
        color="#59a14f",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("SAMC max relative freq. error")
    axes[1, 1].set_xlabel("iterations")
    axes[1, 1].set_ylabel("percent")

    fig.suptitle(
        f"Cross-method diagnostics by checkpoint: {scenario_name}\n"
        f"MCMC-IS total budget includes fixed beta-selection budget (pilot+tuning+scan)={int(mcmc_beta_selection_budget):,}"
    )
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_beta_checkpoint_study(
    problem: PermutationTestProblem,
    exact_p: float,
    *,
    beta_center: float,
    sigma_t: float,
    beta_cfg: BetaSweepStudyConfig,
) -> dict[str, Any]:
    checkpoints = _sorted_unique_points(beta_cfg.estimation_points)
    rows: list[dict[str, Any]] = []
    n_jobs = _effective_n_jobs(beta_cfg.n_jobs, beta_cfg.repeats)

    executor = _try_make_process_pool(n_jobs) if n_jobs > 1 else None

    if executor is None:
        for beta_idx, multiplier in enumerate(beta_cfg.beta_multipliers):
            beta = float(beta_center * multiplier)
            for rep in range(beta_cfg.repeats):
                rep_seed = int(beta_cfg.base_seed + 10_000 * beta_idx + 100 * rep)
                rows.extend(
                    _beta_replicate_worker(
                        problem=problem,
                        exact_p=exact_p,
                        checkpoints=checkpoints,
                        beta=beta,
                        sigma_t=sigma_t,
                        beta_cfg=beta_cfg,
                        rep=rep,
                        rep_seed=rep_seed,
                        multiplier=multiplier,
                    )
                )
    else:
        with executor:
            for beta_idx, multiplier in enumerate(beta_cfg.beta_multipliers):
                beta = float(beta_center * multiplier)
                futures = [
                    executor.submit(
                        _beta_replicate_worker,
                        problem=problem,
                        exact_p=exact_p,
                        checkpoints=checkpoints,
                        beta=beta,
                        sigma_t=sigma_t,
                        beta_cfg=beta_cfg,
                        rep=rep,
                        rep_seed=int(beta_cfg.base_seed + 10_000 * beta_idx + 100 * rep),
                        multiplier=multiplier,
                    )
                    for rep in range(beta_cfg.repeats)
                ]
                for future in cf.as_completed(futures):
                    rows.extend(future.result())

    rows = sorted(rows, key=lambda row: (float(row["beta"]), int(row["replicate"]), int(row["checkpoint"])))
    summary = summarize_records(rows, group_fields=("checkpoint", "beta"))
    return {
        "records": rows,
        "summary": summary,
        "settings": {
            "estimation_points": list(checkpoints),
            "beta_center": float(beta_center),
            "sigma_t": float(sigma_t),
            "beta_multipliers": [float(v) for v in beta_cfg.beta_multipliers],
            "chains": int(beta_cfg.chains),
            "thin": int(beta_cfg.thin),
            "burn_in_fraction": float(beta_cfg.burn_in_fraction),
        },
    }


def plot_beta_sweep_max_budget(
    records: list[dict[str, Any]],
    *,
    scenario_name: str,
    exact_p: float,
    max_budget: int,
    save_path: Path | None = None,
) -> None:
    rows = [row for row in records if int(row["checkpoint"]) == int(max_budget)]
    betas = sorted({float(row["beta"]) for row in rows})
    labels = [f"{beta:.3g}" for beta in betas]

    def _box(metric: str) -> list[np.ndarray]:
        return [np.asarray([row.get(metric, np.nan) for row in rows if float(row["beta"]) == beta], dtype=float) for beta in betas]

    est_data = [_positive_for_plot(arr) for arr in _box("estimate")]
    var_data = [_positive_for_plot(arr) for arr in _box("variance_estimate")]
    q_data = _box("q_tilt_tail_share")
    ess_data = [_positive_for_plot(arr) for arr in _box("ess")]
    acc_data = _box("acceptance_rate")
    wcv_data = _box("weight_cv")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes[0, 0].boxplot(est_data, tick_labels=labels, showfliers=False)
    if _has_positive_finite(est_data):
        axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(exact_p, color="black", linestyle="--", linewidth=1.2, label=f"true p={exact_p:.2e}")
    _set_log_ylim(axes[0, 0], est_data)
    axes[0, 0].set_title("Estimator across beta")
    axes[0, 0].set_ylabel("p_hat")
    axes[0, 0].legend()

    axes[0, 1].boxplot(var_data, tick_labels=labels, showfliers=False)
    if _has_positive_finite(var_data):
        axes[0, 1].set_yscale("log")
    _set_log_ylim(axes[0, 1], var_data)
    axes[0, 1].set_title("Estimated variance across beta")
    axes[0, 1].set_ylabel("var_hat")

    axes[0, 2].boxplot(q_data, tick_labels=labels, showfliers=False)
    axes[0, 2].set_title("Tilted-tail occupancy q")
    axes[0, 2].set_ylabel("q_hat")

    axes[1, 0].boxplot(ess_data, tick_labels=labels, showfliers=False)
    if _has_positive_finite(ess_data):
        axes[1, 0].set_yscale("log")
    _set_log_ylim(axes[1, 0], ess_data)
    axes[1, 0].set_title("ESS across beta")
    axes[1, 0].set_ylabel("ESS")

    axes[1, 1].boxplot(acc_data, tick_labels=labels, showfliers=False)
    axes[1, 1].set_title("Acceptance across beta")
    axes[1, 1].set_ylabel("acceptance rate")

    axes[1, 2].boxplot(wcv_data, tick_labels=labels, showfliers=False)
    axes[1, 2].set_title("Weight CV across beta")
    axes[1, 2].set_ylabel("CV(weights)")

    for ax in axes.flat:
        ax.set_xlabel("beta")
    fig.suptitle(f"{scenario_name}: MCMC-IS beta diagnostics at max iterations={max_budget:,}")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_beta_sweep_convergence(
    summary: list[dict[str, Any]],
    *,
    scenario_name: str,
    exact_p: float,
    save_path: Path | None = None,
) -> None:
    betas = sorted({float(row["beta"]) for row in summary})
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(betas)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for color, beta in zip(colors, betas):
        sub = sorted([row for row in summary if float(row["beta"]) == beta], key=lambda row: row["checkpoint"])
        x = np.asarray([row["checkpoint"] for row in sub], dtype=float)
        mean_est = np.asarray([row["mean_estimate"] for row in sub], dtype=float)
        rmse = np.asarray([row["rmse"] for row in sub], dtype=float)
        mean_var = np.asarray([row["mean_variance_estimate"] for row in sub], dtype=float)
        q = np.asarray([row["mean_q_tilt_tail_share"] for row in sub], dtype=float)
        ess = np.asarray([row["mean_ess"] for row in sub], dtype=float)
        acc = np.asarray([row["mean_acceptance_rate"] for row in sub], dtype=float)
        label = f"{beta:.3g}"

        axes[0, 0].plot(x, mean_est, marker="o", color=color, label=label)
        axes[0, 1].plot(x, rmse, marker="o", color=color, label=label)
        axes[0, 2].plot(x, mean_var, marker="o", color=color, label=label)
        axes[1, 0].plot(x, q, marker="o", color=color, label=label)
        axes[1, 1].plot(x, ess, marker="o", color=color, label=label)
        axes[1, 2].plot(x, acc, marker="o", color=color, label=label)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("iterations")
    beta_est_arrays = [np.asarray([row["mean_estimate"] for row in summary if float(row["beta"]) == beta], dtype=float) for beta in betas]
    beta_rmse_arrays = [np.asarray([row["rmse"] for row in summary if float(row["beta"]) == beta], dtype=float) for beta in betas]
    beta_var_arrays = [np.asarray([row["mean_variance_estimate"] for row in summary if float(row["beta"]) == beta], dtype=float) for beta in betas]
    beta_ess_arrays = [np.asarray([row["mean_ess"] for row in summary if float(row["beta"]) == beta], dtype=float) for beta in betas]
    if _has_positive_finite(beta_est_arrays):
        axes[0, 0].set_yscale("log")
    if _has_positive_finite(beta_rmse_arrays):
        axes[0, 1].set_yscale("log")
    if _has_positive_finite(beta_var_arrays):
        axes[0, 2].set_yscale("log")
    if _has_positive_finite(beta_ess_arrays):
        axes[1, 1].set_yscale("log")
    axes[0, 0].axhline(exact_p, color="black", linestyle="--", linewidth=1.2, label=f"true p={exact_p:.2e}")
    axes[0, 0].set_title("Mean estimate vs iterations")
    axes[0, 0].set_ylabel("p_hat")
    axes[0, 1].set_title("RMSE vs iterations")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 2].set_title("Mean variance estimate vs iterations")
    axes[0, 2].set_ylabel("var_hat")
    axes[1, 0].set_title("Tilted-tail occupancy q")
    axes[1, 0].set_ylabel("q_hat")
    axes[1, 1].set_title("ESS vs iterations")
    axes[1, 1].set_ylabel("ESS")
    axes[1, 2].set_title("Acceptance vs iterations")
    axes[1, 2].set_ylabel("acceptance rate")
    axes[0, 0].legend(title="beta", fontsize=8)
    fig.suptitle(f"{scenario_name}: beta convergence diagnostics (true p={exact_p:.3e})")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_cross_method_outputs(
    scenario: LoadedScenario,
    study: dict[str, Any],
    *,
    output_dir: Path,
    cross_cfg: CrossMethodStudyConfig,
    mcmc_cfg: MCMCWorkflowConfig,
    samc_cfg: SAMCWorkflowConfig,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_cross_method_max_budget(
        study["records"],
        scenario_name=study["scenario_display"],
        exact_p=float(study["exact_p"]),
        max_budget=max(study["estimation_points"]),
        beta_workflow=study["beta_workflow"],
        save_path=output_dir / "cross_method_max_budget.png",
    )
    plot_cross_method_convergence(
        study["summary"],
        scenario_name=study["scenario_display"],
        exact_p=float(study["exact_p"]),
        mcmc_beta_selection_budget=int(study.get("mcmc_beta_selection_budget", 0)),
        save_path=output_dir / "cross_method_convergence.png",
    )
    plot_cross_method_diagnostics(
        study["summary"],
        scenario_name=study["scenario_display"],
        mcmc_beta_selection_budget=int(study.get("mcmc_beta_selection_budget", 0)),
        save_path=output_dir / "cross_method_diagnostics.png",
    )
    plot_iid_stat_density(
        scenario.problem,
        scenario.description,
        exact_p=float(study["exact_p"]),
        n_samples=int(cross_cfg.iid_density_samples),
        seed=int(cross_cfg.base_seed + 30_000),
        save_path=output_dir / "iid_density.png",
    )
    write_jsonl(output_dir / "run_records.jsonl", study["records"])
    write_json(output_dir / "summary.json", study["summary"])
    write_json(
        output_dir / "metadata.json",
        {
            "scenario": study["scenario"],
            "scenario_display": study["scenario_display"],
            "exact_p": study["exact_p"],
            "exact_method": study["exact_method"],
            "exact_tail_hits": study["exact_tail_hits"],
            "exact_n_perm": study["exact_n_perm"],
            "estimation_points": study["estimation_points"],
            "cross_config": cross_cfg,
            "mcmc_config": mcmc_cfg,
            "samc_config": samc_cfg,
            "beta_workflow": study["beta_workflow"],
            "samc_setup": study["samc_setup"],
            "iid_density_summary": study["iid_density_summary"],
        },
    )


def save_beta_sweep_outputs(
    study: dict[str, Any],
    *,
    output_dir: Path,
    scenario_name: str,
    exact_p: float,
    beta_cfg: BetaSweepStudyConfig,
    beta_workflow: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_beta_sweep_max_budget(
        study["records"],
        scenario_name=scenario_name,
        exact_p=exact_p,
        max_budget=max(study["settings"]["estimation_points"]),
        save_path=output_dir / "beta_max_budget.png",
    )
    plot_beta_sweep_convergence(
        study["summary"],
        scenario_name=scenario_name,
        exact_p=exact_p,
        save_path=output_dir / "beta_convergence.png",
    )
    write_jsonl(output_dir / "run_records.jsonl", study["records"])
    write_json(output_dir / "summary.json", study["summary"])
    write_json(
        output_dir / "metadata.json",
        {
            "scenario_display": scenario_name,
            "exact_p": exact_p,
            "beta_config": beta_cfg,
            "beta_workflow": beta_workflow,
            "settings": study["settings"],
        },
    )


def load_cross_method_saved_output(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    metadata = read_json(output_dir / "metadata.json")
    summary = read_json(output_dir / "summary.json")
    records = read_jsonl(output_dir / "run_records.jsonl")
    return {
        "output_dir": output_dir,
        "metadata": metadata,
        "summary": summary,
        "records": records,
    }


def load_beta_sweep_saved_output(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    metadata = read_json(output_dir / "metadata.json")
    summary = read_json(output_dir / "summary.json")
    records = read_jsonl(output_dir / "run_records.jsonl")
    return {
        "output_dir": output_dir,
        "metadata": metadata,
        "summary": summary,
        "records": records,
    }


def regenerate_cross_method_plots_from_saved(
    output_dir: Path,
    *,
    save_dir: Path | None = None,
) -> dict[str, Path]:
    saved = load_cross_method_saved_output(output_dir)
    metadata = saved["metadata"]
    save_dir = Path(save_dir) if save_dir is not None else Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    max_budget = int(max(metadata["estimation_points"]))
    beta_workflow = metadata.get("beta_workflow", {})
    mcmc_budget = int(
        metadata.get(
            "mcmc_beta_selection_budget",
            beta_workflow.get("beta_selection_eval_total", 0),
        )
    )

    out = {
        "cross_method_max_budget": save_dir / "cross_method_max_budget.png",
        "cross_method_convergence": save_dir / "cross_method_convergence.png",
        "cross_method_diagnostics": save_dir / "cross_method_diagnostics.png",
    }
    plot_cross_method_max_budget(
        saved["records"],
        scenario_name=str(metadata["scenario_display"]),
        exact_p=float(metadata["exact_p"]),
        max_budget=max_budget,
        beta_workflow=beta_workflow,
        save_path=out["cross_method_max_budget"],
    )
    plot_cross_method_convergence(
        saved["summary"],
        scenario_name=str(metadata["scenario_display"]),
        exact_p=float(metadata["exact_p"]),
        mcmc_beta_selection_budget=mcmc_budget,
        save_path=out["cross_method_convergence"],
    )
    plot_cross_method_diagnostics(
        saved["summary"],
        scenario_name=str(metadata["scenario_display"]),
        mcmc_beta_selection_budget=mcmc_budget,
        save_path=out["cross_method_diagnostics"],
    )
    return out


def regenerate_beta_sweep_plots_from_saved(
    output_dir: Path,
    *,
    save_dir: Path | None = None,
) -> dict[str, Path]:
    saved = load_beta_sweep_saved_output(output_dir)
    metadata = saved["metadata"]
    save_dir = Path(save_dir) if save_dir is not None else Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    max_budget = int(max(metadata["settings"]["estimation_points"]))
    out = {
        "beta_max_budget": save_dir / "beta_max_budget.png",
        "beta_convergence": save_dir / "beta_convergence.png",
    }
    plot_beta_sweep_max_budget(
        saved["records"],
        scenario_name=str(metadata["scenario_display"]),
        exact_p=float(metadata["exact_p"]),
        max_budget=max_budget,
        save_path=out["beta_max_budget"],
    )
    plot_beta_sweep_convergence(
        saved["summary"],
        scenario_name=str(metadata["scenario_display"]),
        exact_p=float(metadata["exact_p"]),
        save_path=out["beta_convergence"],
    )
    return out
