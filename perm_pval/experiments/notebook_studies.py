from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import time
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator

from perm_pval.core.proposals import propose_localized_swaps, resolve_n_swap_pairs
from perm_pval.core.problem import PermutationTestProblem
from perm_pval.diagnostics.is_weights import effective_sample_size, summarize_weights
from perm_pval.diagnostics.mcmc import obm_long_run_variance
from perm_pval.diagnostics.samc import visitation_frequency
from perm_pval.experiments.exact_scenarios import ExactScenario, load_saved_exact_scenarios
from perm_pval.methods.beta_tuning import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
)
from perm_pval.methods.mcmc_is import (
    hard_step_beta_for_target_tail_mass,
    hard_step_r_for_target_tail_mass,
    right_tail_deficit_scaled,
    right_tail_step_shortfall,
    run_mcmc_is,
)
from perm_pval.methods.random_sampling import run_random_sampling, wilson_interval
from perm_pval.methods.samc import (
    _bin_index,
    _correction_ratio,
    _default_stepsize,
    _paper_pvalue_estimate,
    _pvalue_estimate_no_empty_bin_correction,
    _relative_sampling_frequency_error,
    run_samc,
)


@dataclass(frozen=True)
class MCMCWorkflowConfig:
    use_true_p0_for_q_target: bool = True
    p0_guess: float = 1e-8
    d_alpha: float = 1.0 / 3.0
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
    chain_n_jobs: int = 1
    production_estimator_variant: str = "production_only"
    tilt_mode: str = "smooth_hinge"
    proposal_size: float | int = 0.075
    hard_step_enabled: bool = True
    hard_step_reference_p0: float | None = None
    hard_step_target_tail_mass: float | None = None
    hard_step_proposal_size: float | int | None = None
    local_scan_enabled: bool = True
    local_scan_strategy: str = "fixed_grid"
    local_scan_q_multipliers: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.25, 0.33)
    local_scan_coarse_q_multipliers: tuple[float, ...] = ()
    local_scan_swap_counts: tuple[int, ...] = (1, 2, 3)
    local_scan_objective: str = "varhat_qmatch_soft"
    local_scan_screen_total_steps: int = 6_000
    local_scan_screen_chains: int = 1
    local_scan_refine_total_steps: int | None = None
    local_scan_refine_chains: int | None = None
    local_scan_refine_top_k: int = 2
    local_scan_refine_radius: int = 1
    local_scan_refine_max_q_points: int = 6
    local_scan_finalist_count: int = 3
    local_scan_total_steps: int = 32_000
    local_scan_chains: int = 2
    local_scan_burn_in_fraction: float = 0.20
    local_scan_thin: int = 1
    local_scan_variance_near_min_ratio: float = 1.20
    local_scan_weight_cv_penalty_scale: float = 0.25


@dataclass(frozen=True)
class SAMCWorkflowConfig:
    burn_in_fraction: float = 0.20
    n_bins: int = 40
    t0: float = 1_000.0
    trace_every: int = 200
    convergence_tolerance: float = 20.0
    lambda_min_pilot: int = 10_000
    proposal_size: float | int = 0.075


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
    chain_n_jobs: int = 1
    tilt_mode: str = "smooth_hinge"
    proposal_size: float | int = 0.075
    base_seed: int = 54_321
    n_jobs: int = 1


DEFAULT_MCMC_OBJECTIVE_GRID_Q_MULTIPLIERS: tuple[float, ...] = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    5e-4,
    1e-3,
    3e-3,
    5e-3,
    1e-2,
    3e-2,
    5e-2,
    0.1,
    0.15,
    0.2,
    0.25,
    0.33,
)
DEFAULT_MCMC_OBJECTIVE_GRID_SWAP_COUNTS: tuple[int, ...] = (1, 2, 3, 4)
MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES: tuple[str, ...] = (
    "varhat",
    "varhat_qmatch_soft",
    "varhat_degeneracy_soft",
    "varhat_qmatch_degeneracy_soft",
)
MCMC_OBJECTIVE_GRID_ALL_OBJECTIVES: tuple[str, ...] = (
    "oracle_rmse",
    "oracle_abs_log10",
    *MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES,
)


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
    portfolio: dict[str, Any] = field(default_factory=dict)


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
        portfolio=dict(getattr(s, "portfolio", {})),
    )


def _valid_probability_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(val) and 0.0 < val < 1.0:
        return val
    return None


def build_hard_step_workflow(scenario: LoadedScenario, cfg: MCMCWorkflowConfig) -> dict[str, Any]:
    if not bool(cfg.hard_step_enabled):
        return {"enabled": False}
    if scenario.problem.tail != "right":
        raise NotImplementedError("Hard-step cross-method tilt currently follows the right-tail MCMC-IS kernel.")

    p0_source = "exact_p_fallback" if cfg.use_true_p0_for_q_target else "p0_guess_fallback"
    p0_ref = float(scenario.exact_p) if cfg.use_true_p0_for_q_target else float(cfg.p0_guess)

    if cfg.hard_step_reference_p0 is not None:
        override = _valid_probability_or_none(cfg.hard_step_reference_p0)
        if override is None:
            raise ValueError("hard_step_reference_p0 must be finite and lie in (0, 1).")
        p0_ref = float(override)
        p0_source = "hard_step_reference_p0"
    else:
        for source, container in (
            ("scenario_extra_known_significance_threshold", scenario.extra),
            ("scenario_portfolio_known_significance_threshold", scenario.portfolio),
        ):
            if not isinstance(container, dict):
                continue
            threshold = _valid_probability_or_none(container.get("known_significance_threshold"))
            if threshold is None:
                threshold = _valid_probability_or_none(container.get("canonical_threshold_p0"))
            if threshold is not None:
                p0_ref = float(threshold)
                p0_source = source
                break

    q_target = (
        float(cfg.hard_step_target_tail_mass)
        if cfg.hard_step_target_tail_mass is not None
        else float(p0_ref ** float(cfg.d_alpha))
    )
    r = hard_step_r_for_target_tail_mass(p0_ref, q_target)
    beta = hard_step_beta_for_target_tail_mass(p0_ref, q_target)
    proposal_size = cfg.proposal_size if cfg.hard_step_proposal_size is None else cfg.hard_step_proposal_size
    n_swap_pairs = resolve_n_swap_pairs(
        scenario.problem.n_treated,
        scenario.problem.n_control,
        proposal_size=proposal_size,
    )
    return {
        "enabled": True,
        "reference_p0": float(p0_ref),
        "reference_p0_source": p0_source,
        "target_tail_mass": float(q_target),
        "r": float(r),
        "beta": float(beta),
        "proposal_size": proposal_size,
        "n_swap_pairs": int(n_swap_pairs),
        "tilt_mode": "step",
        "target_tail_mass_rule": (
            "hard_step_target_tail_mass" if cfg.hard_step_target_tail_mass is not None else "p0_reference_power_d_alpha"
        ),
        "d_alpha": float(cfg.d_alpha),
    }


def load_selected_scenarios(
    *,
    catalog_path: Path,
    scenario_keys: Iterable[str] | None = None,
    portfolio_group: str | None = None,
    min_tail_states: int = 1,
) -> list[LoadedScenario]:
    all_scenarios = [_to_loaded_scenario(s) for s in load_saved_exact_scenarios(catalog_path)]
    by_key = {s.key: s for s in all_scenarios}

    if scenario_keys is None:
        if portfolio_group is None:
            selected_keys = [s.key for s in all_scenarios]
        else:
            selected_keys = [
                s.key
                for s in all_scenarios
                if str(portfolio_group) in set(str(v) for v in s.portfolio.get("groups", []))
            ]
            if not selected_keys:
                raise KeyError(
                    f"No scenarios found for portfolio_group='{portfolio_group}'. "
                    f"Available groups: {sorted({group for s in all_scenarios for group in s.portfolio.get('groups', [])})}"
                )
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


def _q_match_penalty(q_real: float, q_target: float, n_weighted_samples: int) -> float:
    if n_weighted_samples > 0:
        q_eps = float(max(1.0 / float(n_weighted_samples), 1e-12))
    else:
        q_eps = 1e-12
    if not np.isfinite(q_real) or not np.isfinite(q_target) or q_target <= 0.0:
        return float("inf")
    return float(1.0 + abs(np.log((q_real + q_eps) / (q_target + q_eps))))


def _weight_cv_penalty(weight_cv: float, scale: float) -> float:
    if not np.isfinite(weight_cv) or weight_cv < 0.0:
        return float("inf")
    return float(1.0 + float(scale) * np.log1p(weight_cv))


def _compute_varhat_objectives(
    *,
    variance_estimate: float,
    q_tilt_tail_share: float,
    q_target: float,
    n_weighted_samples: int,
    weight_cv: float,
    weight_cv_penalty_scale: float,
) -> dict[str, float]:
    base_varhat = float(variance_estimate) if _positive_finite(variance_estimate) else float("inf")
    p_q = _q_match_penalty(q_tilt_tail_share, q_target, n_weighted_samples)
    p_deg = _weight_cv_penalty(weight_cv, weight_cv_penalty_scale)
    return {
        "objective_varhat": base_varhat,
        "objective_varhat_qmatch_soft": (
            float(base_varhat * p_q) if np.isfinite(base_varhat) and np.isfinite(p_q) else float("inf")
        ),
        "objective_varhat_degeneracy_soft": (
            float(base_varhat * p_deg) if np.isfinite(base_varhat) and np.isfinite(p_deg) else float("inf")
        ),
        "objective_varhat_qmatch_degeneracy_soft": (
            float(base_varhat * p_q * p_deg)
            if np.isfinite(base_varhat) and np.isfinite(p_q) and np.isfinite(p_deg)
            else float("inf")
        ),
        "P_q": p_q,
        "P_deg": p_deg,
    }


def _scan_metric_key(objective_name: str) -> str:
    return f"objective_{str(objective_name)}"


def _build_q_scan_candidates(
    *,
    problem: PermutationTestProblem,
    cfg: MCMCWorkflowConfig,
    pilot_t: np.ndarray,
    p0_for_qtarget: float,
    sigma_t: float,
    q_target: float,
    q_multipliers: Iterable[float],
    swap_counts: Iterable[int],
    q_floor: float = 1e-12,
) -> list[dict[str, Any]]:
    q_grid = tuple(float(mult) for mult in q_multipliers)
    swap_grid = tuple(int(v) for v in swap_counts)
    max_swaps = min(int(problem.n_treated), int(problem.n_control))
    candidates: list[dict[str, Any]] = []
    for q_index, q_multiplier in enumerate(q_grid):
        if q_multiplier <= 0.0 or not np.isfinite(q_multiplier):
            raise ValueError("local_scan_q_multipliers must all be positive finite values.")
        q_candidate = float(max(q_target * q_multiplier, q_floor))
        try:
            beta_candidate = float(
                init_beta_from_iid_pilot(
                    pilot_T=pilot_t,
                    T_obs=problem.t_obs,
                    sigma_T=sigma_t,
                    p0=p0_for_qtarget,
                    q_target=q_candidate,
                    beta_max=cfg.beta_max_init,
                )
            )
        except Exception:
            beta_candidate = float("nan")
        for n_swap_pairs in swap_grid:
            if n_swap_pairs < 1 or n_swap_pairs > max_swaps:
                raise ValueError(
                    f"local_scan_swap_counts must lie in [1, {max_swaps}], received {n_swap_pairs}."
                )
            candidates.append(
                {
                    "config_id": f"q{q_index:02d}_s{n_swap_pairs}",
                    "label": f"q{q_index:02d}_s{n_swap_pairs}",
                    "q_index": int(q_index),
                    "q_multiplier": float(q_multiplier),
                    "q_scan_target": float(q_candidate),
                    "beta": float(beta_candidate),
                    "proposal_size": int(n_swap_pairs),
                    "n_swap_pairs": int(n_swap_pairs),
                    "status": (
                        "ok"
                        if np.isfinite(beta_candidate) and beta_candidate > 0.0
                        else "invalid_q_map"
                    ),
                }
            )
    return candidates


def _candidate_multiplier_key(q_multiplier: float) -> str:
    return f"{float(q_multiplier):.12g}"


def _select_candidate_subset_by_multipliers(
    candidates: list[dict[str, Any]],
    q_multipliers: Iterable[float],
) -> list[dict[str, Any]]:
    allowed = {_candidate_multiplier_key(v) for v in q_multipliers}
    return [
        dict(candidate)
        for candidate in candidates
        if _candidate_multiplier_key(float(candidate["q_multiplier"])) in allowed
    ]


def _rank_scan_rows(rows: list[dict[str, Any]], *, objective_name: str) -> list[int]:
    metric_key = _scan_metric_key(objective_name)
    finite_indices = [
        idx
        for idx, row in enumerate(rows)
        if np.isfinite(row.get(metric_key, np.inf))
    ]
    if finite_indices:
        return sorted(
            finite_indices,
            key=lambda idx: (
                float(rows[idx][metric_key]),
                int(rows[idx]["q_index"]),
                int(rows[idx]["n_swap_pairs"]),
                float(rows[idx]["beta"]),
            ),
        )
    return sorted(
        range(len(rows)),
        key=lambda idx: (
            int(rows[idx]["q_index"]),
            int(rows[idx]["n_swap_pairs"]),
            float(rows[idx]["beta"]) if np.isfinite(rows[idx].get("beta", np.nan)) else float("inf"),
        ),
    )


def _rank_scan_q_indices(rows: list[dict[str, Any]], *, objective_name: str) -> list[int]:
    metric_key = _scan_metric_key(objective_name)
    best_by_q: dict[int, dict[str, Any]] = {}
    for row in rows:
        q_index = int(row["q_index"])
        current = best_by_q.get(q_index)
        candidate_key = (
            float(row.get(metric_key, np.inf)),
            int(row["q_index"]),
            int(row["n_swap_pairs"]),
            float(row["beta"]) if np.isfinite(row.get("beta", np.nan)) else float("inf"),
        )
        if current is None:
            best_by_q[q_index] = dict(row)
            continue
        current_key = (
            float(current.get(metric_key, np.inf)),
            int(current["q_index"]),
            int(current["n_swap_pairs"]),
            float(current["beta"]) if np.isfinite(current.get("beta", np.nan)) else float("inf"),
        )
        if candidate_key < current_key:
            best_by_q[q_index] = dict(row)

    finite_q = [
        q_index
        for q_index, row in best_by_q.items()
        if np.isfinite(row.get(metric_key, np.inf))
    ]
    if finite_q:
        return sorted(
            finite_q,
            key=lambda q_index: (
                float(best_by_q[q_index][metric_key]),
                int(best_by_q[q_index]["q_index"]),
                int(best_by_q[q_index]["n_swap_pairs"]),
                float(best_by_q[q_index]["beta"]),
            ),
        )
    return sorted(best_by_q)


def _adaptive_refine_q_indices(
    rows: list[dict[str, Any]],
    *,
    objective_name: str,
    max_master_q_index: int,
    top_k: int,
    radius: int,
    max_q_points: int,
) -> list[int]:
    ranked_q_indices = _rank_scan_q_indices(rows, objective_name=objective_name)
    if not ranked_q_indices:
        return []

    anchor_indices = ranked_q_indices[: max(int(top_k), 1)]
    anchor_rank = {int(q_idx): rank for rank, q_idx in enumerate(anchor_indices)}
    candidates: dict[int, tuple[int, int, int]] = {}
    for q_idx in anchor_indices:
        for offset in range(-int(radius), int(radius) + 1):
            candidate_q = int(q_idx) + int(offset)
            if candidate_q < 0 or candidate_q > int(max_master_q_index):
                continue
            score = (abs(int(offset)), int(anchor_rank[int(q_idx)]), int(candidate_q))
            current = candidates.get(candidate_q)
            if current is None or score < current:
                candidates[candidate_q] = score

    ordered = [q_idx for q_idx, _ in sorted(candidates.items(), key=lambda item: item[1])]
    limit = max(int(max_q_points), len(anchor_indices), 1)
    if len(ordered) < limit:
        for q_idx in ranked_q_indices:
            if q_idx not in candidates:
                ordered.append(int(q_idx))
            if len(ordered) >= limit:
                break
    return ordered[:limit]


def _run_local_scan_stage(
    problem: PermutationTestProblem,
    cfg: MCMCWorkflowConfig,
    *,
    candidates: list[dict[str, Any]],
    sigma_t: float,
    total_steps: int,
    n_chains: int,
    seed: int,
    stage: str,
    return_sample_batches: bool = False,
    init_states_by_config: dict[str, list[np.ndarray]] | None = None,
) -> dict[str, Any]:
    steps_per_chain, burn_in = _local_scan_steps(
        total_steps,
        n_chains,
        cfg.local_scan_burn_in_fraction,
    )
    rows: list[dict[str, Any]] = []
    final_states: dict[str, list[np.ndarray]] = {}
    sample_batches: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for idx, candidate in enumerate(candidates):
        if str(candidate["status"]) != "ok":
            rows.append(
                {
                    **candidate,
                    "stage": stage,
                    "estimate": np.nan,
                    "variance_estimate": np.nan,
                    "selection_objective_p0": np.nan,
                    "q_hat": np.nan,
                    "q_tilt_tail_share": np.nan,
                    "ess": np.nan,
                    "n_weighted_samples": 0,
                    "tail_hits_weighted_sample": 0,
                    "acceptance_rate": np.nan,
                    "weight_cv": np.nan,
                    "state_reused_init": 0,
                    "state_reuse_mode": "fresh_observed",
                    "scan_burn_in": int(burn_in),
                    "objective_selected": float("inf"),
                    "screen_rank": None,
                    "advanced_to_final": None,
                    "advanced_reason": None,
                    "objective_ratio_to_best": None,
                    "within_objective_tolerance": None,
                    **_compute_varhat_objectives(
                        variance_estimate=float("nan"),
                        q_tilt_tail_share=float("nan"),
                        q_target=float(candidate["q_scan_target"]),
                        n_weighted_samples=0,
                        weight_cv=float("nan"),
                        weight_cv_penalty_scale=cfg.local_scan_weight_cv_penalty_scale,
                    ),
                }
            )
            continue

        candidate_key = str(candidate["config_id"])
        raw_init_states = None if init_states_by_config is None else init_states_by_config.get(candidate_key)
        if raw_init_states:
            if len(raw_init_states) >= int(n_chains):
                reused_init_states = [
                    np.asarray(raw_init_states[chain_idx], dtype=np.int8).copy()
                    for chain_idx in range(int(n_chains))
                ]
                candidate_burn_in = 0
                state_reuse_mode = "continued"
            else:
                # Expanding from fewer chains to more chains duplicates initial
                # states. Keep the usual burn-in so the new chains decorrelate.
                reused_init_states = [
                    np.asarray(raw_init_states[chain_idx % len(raw_init_states)], dtype=np.int8).copy()
                    for chain_idx in range(int(n_chains))
                ]
                candidate_burn_in = burn_in
                state_reuse_mode = "expanded_with_burn_in"
            candidate_init: str | list[np.ndarray] = reused_init_states
            state_reused_init = 1
        else:
            candidate_init = "observed"
            candidate_burn_in = burn_in
            state_reused_init = 0
            state_reuse_mode = "fresh_observed"

        res = run_mcmc_is(
            problem,
            beta=float(candidate["beta"]),
            sigma_t=sigma_t,
            n_steps=steps_per_chain,
            burn_in=candidate_burn_in,
            thin=cfg.local_scan_thin,
            n_chains=int(n_chains),
            seed=seed + 10_000 * idx,
            init=candidate_init,
            tilt_mode=str(cfg.tilt_mode),
            proposal_size=int(candidate["n_swap_pairs"]),
            estimate_variance=True,
            obm_batch_size=cfg.obm_batch_size,
        )
        var_hat = (
            float(res.snis_variance_obm)
            if res.snis_variance_obm is not None and np.isfinite(res.snis_variance_obm) and res.snis_variance_obm > 0.0
            else np.nan
        )
        selection_objective_p0 = _snis_design_variance_objective(
            log_weights=np.asarray(res.log_weights, dtype=float),
            tail_indicators=np.asarray(res.tail_indicators, dtype=np.int8),
            p_reference=float(candidate["q_scan_target"]),
            n_chains=int(n_chains),
            chain_kept_samples=int(_kept_samples_per_chain(steps_per_chain, candidate_burn_in, cfg.local_scan_thin)),
            obm_batch_size=cfg.obm_batch_size,
        )
        objective_values = _compute_varhat_objectives(
            variance_estimate=var_hat,
            q_tilt_tail_share=float(res.tail_share_raw_sample),
            q_target=float(candidate["q_scan_target"]),
            n_weighted_samples=int(res.n_weighted_samples),
            weight_cv=float(res.weight_summary.cv),
            weight_cv_penalty_scale=cfg.local_scan_weight_cv_penalty_scale,
        )
        rows.append(
            {
                **candidate,
                "stage": stage,
                "estimate": float(res.estimate),
                "variance_estimate": var_hat,
                "selection_objective_p0": float(selection_objective_p0),
                "q_hat": float(res.tail_share_raw_sample),
                "q_tilt_tail_share": float(res.tail_share_raw_sample),
                "ess": float(res.ess),
                "n_weighted_samples": int(res.n_weighted_samples),
                "tail_hits_weighted_sample": int(res.tail_hits_weighted_sample),
                "acceptance_rate": float(res.overall_acceptance_rate),
                "weight_cv": float(res.weight_summary.cv),
                "state_reused_init": int(state_reused_init),
                "state_reuse_mode": str(state_reuse_mode),
                "scan_burn_in": int(candidate_burn_in),
                "objective_selected": float(objective_values.get(_scan_metric_key(cfg.local_scan_objective), np.inf)),
                "screen_rank": None,
                "advanced_to_final": None,
                "advanced_reason": None,
                "objective_ratio_to_best": None,
                "within_objective_tolerance": None,
                **objective_values,
            }
        )
        final_states[str(candidate["config_id"])] = [np.asarray(state, dtype=np.int8).copy() for state in res.final_states]
        if return_sample_batches:
            sample_batches.append(
                {
                    "stage": str(stage),
                    "config_id": str(candidate["config_id"]),
                    "label": str(candidate["label"]),
                    "beta": float(candidate["beta"]),
                    "n_swap_pairs": int(candidate["n_swap_pairs"]),
                    "q_multiplier": float(candidate["q_multiplier"]),
                    "q_scan_target": float(candidate["q_scan_target"]),
                    "n_weighted_samples": int(res.n_weighted_samples),
                    "log_weights": np.asarray(res.log_weights, dtype=float).copy(),
                    "tail_indicators": np.asarray(res.tail_indicators, dtype=np.int8).copy(),
                }
            )

    return {
        "rows": rows,
        "steps_per_chain": int(steps_per_chain),
        "burn_in": int(burn_in),
        "eval_total": int(len(candidates) * _mcmc_eval_count(steps_per_chain, n_chains)),
        "wall_time_sec": float(time.perf_counter() - t_start),
        "final_states": final_states,
        "sample_batches": sample_batches,
    }


def _mcmc_eval_count(n_steps: int, n_chains: int) -> int:
    return int(n_chains * (n_steps + 1))


def _positive_for_plot(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    out[out <= 0.0] = np.nan
    if np.all(np.isnan(out)):
        return np.asarray([np.nan], dtype=float)
    return out


def _finite_for_plot(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
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


def _set_linear_ylim(
    ax,
    arrays: list[np.ndarray],
    *,
    include_values: Iterable[float] = (),
    pad: float = 0.08,
    anchor_zero: bool = False,
) -> None:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    extras = np.asarray([float(v) for v in include_values if np.isfinite(float(v))], dtype=float)
    if extras.size:
        vals.append(extras)
    if not vals:
        return
    flat = np.concatenate(vals)
    lo = float(np.min(flat))
    hi = float(np.max(flat))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    span = abs(hi - lo)
    scale = max(abs(lo), abs(hi), span, 1e-12)
    if anchor_zero and lo >= 0.0:
        lo = 0.0
    else:
        lo = lo - pad * scale
    if hi <= lo:
        hi = lo + max(scale, 1e-12)
    else:
        hi = hi + pad * scale
    ax.set_ylim(lo, hi)


def _has_positive_finite(arrays: list[np.ndarray]) -> bool:
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(a) & (a > 0.0)):
            return True
    return False


def _value_in_ylim(ax, value: float) -> bool:
    if not np.isfinite(float(value)):
        return False
    ymin, ymax = ax.get_ylim()
    lo, hi = (ymin, ymax) if ymin <= ymax else (ymax, ymin)
    return lo <= float(value) <= hi


_CROSS_METHOD_LABELS: dict[str, str] = {
    "iid": "IID",
    "mcmc_is": "MCMC-IS",
    "hard_step": "Hard-step IS",
    "samc": "SAMC",
}

_CROSS_METHOD_COLORS: dict[str, str] = {
    "iid": "#5b6c8f",
    "mcmc_is": "#c48a3a",
    "hard_step": "#b04a5a",
    "samc": "#4c8c77",
}

_CROSS_METHOD_ORDER: list[str] = ["iid", "mcmc_is", "hard_step", "samc"]

_CROSS_METHOD_TITLES: dict[str, str] = {
    "bruteforce_welch_nonextreme_n22": "Welch t-statistic",
    "gwas_additive_score_n40": "GWAS-like additive score",
    "gwas_additive_score_ultra_n100": "GWAS-like additive score\nClearly Significant",
    "gwas_additive_score_sig_n100": "GWAS-like additive score\nNear-Threshold",
    "gwas_additive_score_slight_above_n100": "GWAS-like additive score\nSlightly Non-Significant",
    "gwas_additive_score_above_n100": "GWAS-like additive score\nClearly Non-Significant",
    "hypergeom_1e7": "Hypergeometric count",
    "linear_stat_dp_n40": "Difference in means",
    "poisson_diffmeans_hep_ultra_n200": "HEP-like Poisson count\nClearly Significant",
    "poisson_diffmeans_hep_sig_n200": "HEP-like Poisson count\nNear-Threshold",
    "poisson_diffmeans_hep_slight_above_n200": "HEP-like Poisson count\nSlightly Non-Significant",
    "poisson_diffmeans_hep_above_n200": "HEP-like Poisson count\nClearly Non-Significant",
    "rank_sum_dp_n40": "Mann-Whitney U",
}


def _compact_plot_title(
    scenario_name: str,
    *,
    scenario_key: str | None = None,
    n_control: int | None = None,
    n_treated: int | None = None,
    exact_p: float | None = None,
    known_significance_threshold: float | None = None,
    n_runs: int | None = None,
) -> str:
    base = None
    if scenario_key is not None and scenario_key in _CROSS_METHOD_TITLES:
        base = _CROSS_METHOD_TITLES[str(scenario_key)]
    if base is None:
        base = " ".join(str(scenario_name).split()).rstrip(".")
    base = str(base).replace("\n", " - ")
    subtitle_parts: list[str] = []
    if known_significance_threshold is not None and np.isfinite(float(known_significance_threshold)) and float(known_significance_threshold) > 0.0:
        subtitle_parts.append(f"Significance threshold = {float(known_significance_threshold):.1e}")
    if exact_p is not None and np.isfinite(float(exact_p)) and float(exact_p) > 0.0:
        subtitle_parts.append(f"True p-value = {float(exact_p):.3e}")
    if n_runs is not None and int(n_runs) > 0:
        subtitle_parts.append(f"{int(n_runs)} runs/method")
    if subtitle_parts:
        return f"{base}\n" + " | ".join(subtitle_parts)
    return base


def _format_scientific_tick(value: int | float) -> str:
    val = float(value)
    if not np.isfinite(val) or val == 0.0:
        return str(value)
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / (10.0 ** exponent)
    rounded = round(mantissa, 1)
    if np.isclose(rounded, round(rounded)):
        mantissa_str = str(int(round(rounded)))
    else:
        mantissa_str = f"{rounded:.1f}".rstrip("0").rstrip(".")
    return f"{mantissa_str}e{exponent}"


def _format_budget_tick_compact(value: int | float) -> str:
    val = float(value)
    if not np.isfinite(val):
        return str(value)
    abs_val = abs(val)
    if abs_val >= 500_000:
        scaled = val / 1_000_000.0
        suffix = "M"
    elif abs_val >= 1_000:
        scaled = val / 1_000.0
        suffix = "K"
    else:
        scaled = val
        suffix = ""
    if np.isclose(scaled, round(scaled)):
        scaled_str = str(int(round(scaled)))
    else:
        scaled_str = f"{scaled:.1f}".rstrip("0").rstrip(".")
    return f"{scaled_str}{suffix}"


def _format_budget_millions(value: int | float) -> str:
    scaled = float(value) / 1_000_000.0
    if np.isfinite(scaled) and np.isclose(scaled, round(scaled)):
        return str(int(round(scaled)))
    if np.isfinite(scaled):
        return f"{scaled:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _format_budget_title(value: int | float, subject: str) -> str:
    return f"{subject} at {_format_budget_millions(value)} million iterations"


def _style_article_axis(
    ax,
    *,
    grid_axis: str = "y",
    add_minor_log_grid: bool = True,
) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, which="major", color="#dfe4e7", linewidth=0.85, alpha=0.95)
    if add_minor_log_grid:
        if ax.get_yscale() == "log" and grid_axis in {"y", "both"}:
            ax.grid(True, axis="y", which="minor", color="#eff2f4", linewidth=0.55, alpha=0.9)
        if ax.get_xscale() == "log" and grid_axis in {"x", "both"}:
            ax.grid(True, axis="x", which="minor", color="#eff2f4", linewidth=0.55, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color("#bcc4c9")
        ax.spines[spine_name].set_linewidth(0.85)
    ax.tick_params(colors="#333333", labelsize=10)


def _set_budget_ticks(
    ax,
    budgets: Iterable[int | float],
    *,
    max_labels: int = 6,
    formatter: Callable[[int | float], str] | None = None,
    label_every: int | None = None,
) -> None:
    ticks = sorted({int(v) for v in budgets if int(v) > 0})
    if not ticks:
        return
    if formatter is None:
        formatter = _format_scientific_tick
    label_ticks = ticks
    if label_every is not None and int(label_every) > 0:
        step = int(label_every)
        label_ticks = [tick for tick in ticks if tick % step == 0]
        if not label_ticks:
            label_ticks = ticks
        elif ticks[-1] not in label_ticks:
            label_ticks.append(ticks[-1])
        label_ticks = sorted(set(label_ticks))
    if max_labels > 0 and len(ticks) > max_labels:
        if label_every is None:
            step = max(int(np.ceil((len(ticks) - 1) / max(max_labels - 1, 1))), 1)
            label_ticks = ticks[::step]
            if label_ticks[-1] != ticks[-1]:
                label_ticks.append(ticks[-1])
            label_ticks = sorted(set(label_ticks))
    ax.set_xticks(label_ticks)
    ax.set_xticklabels([formatter(tick) for tick in label_ticks], fontsize=9)
    ax.minorticks_off()


def _resolve_group_sizes(
    *,
    scenario_key: str | None = None,
    n_control: int | None = None,
    n_treated: int | None = None,
) -> tuple[int | None, int | None]:
    if n_control is not None and n_treated is not None:
        return int(n_control), int(n_treated)
    if not scenario_key:
        return None, None
    catalog_path = Path("results") / "exact_scenarios" / "v1" / "catalog.json"
    if not catalog_path.exists():
        return None, None
    try:
        scenarios = load_saved_exact_scenarios(catalog_path)
    except Exception:
        return None, None
    for scenario in scenarios:
        if scenario.key == str(scenario_key):
            return int(scenario.problem.n_control), int(scenario.problem.n_treated)
    return None, None


def _styled_boxplot(
    ax,
    data: list[np.ndarray],
    *,
    labels: list[str],
    colors: list[str],
):
    artists = ax.boxplot(
        data,
        tick_labels=labels,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.45},
        whiskerprops={"color": "#6d7378", "linewidth": 1.0},
        capprops={"color": "#6d7378", "linewidth": 1.0},
        boxprops={"edgecolor": "#6d7378", "linewidth": 1.0},
    )
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    return artists


def _overlay_boxplot_points(
    ax,
    data: list[np.ndarray],
    *,
    colors: list[str],
) -> None:
    for idx, (arr, color) in enumerate(zip(data, colors), start=1):
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vals = np.sort(vals)
        if vals.size == 1:
            x = np.asarray([float(idx)], dtype=float)
        else:
            x = float(idx) + np.linspace(-0.09, 0.09, vals.size)
        ax.scatter(
            x,
            vals,
            s=24,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.95,
            zorder=3.5,
        )


def _record_group_label(row: dict[str, Any]) -> str:
    if "label" in row and row["label"] is not None:
        return str(row["label"])
    if "method" in row and row["method"] is not None:
        return str(row["method"])
    raise KeyError("record must contain either 'label' or 'method'.")


def _method_order_from_records(
    records: list[dict[str, Any]],
    method_order: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if method_order is not None:
        return [str(method) for method in method_order]
    seen: list[str] = []
    for row in records:
        label = _record_group_label(row)
        if label not in seen:
            seen.append(label)
    if set(seen).issubset(set(_CROSS_METHOD_ORDER)):
        return [method for method in _CROSS_METHOD_ORDER if method in seen]
    return seen


def _available_cross_method_order(rows: list[dict[str, Any]]) -> list[str]:
    present = {_record_group_label(row) for row in rows}
    ordered = [method for method in _CROSS_METHOD_ORDER if method in present]
    extras = sorted(present.difference(ordered))
    return [*ordered, *extras]


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
    pilot_t: np.ndarray,
    p0_for_qtarget: float,
    sigma_t: float,
    q_target: float,
    seed: int,
    return_sample_batches: bool = False,
) -> dict[str, Any]:
    if cfg.local_scan_screen_total_steps <= 0:
        raise ValueError("local_scan_screen_total_steps must be positive.")
    if cfg.local_scan_refine_total_steps is not None and int(cfg.local_scan_refine_total_steps) <= 0:
        raise ValueError("local_scan_refine_total_steps must be positive when provided.")
    if cfg.local_scan_finalist_count <= 0:
        raise ValueError("local_scan_finalist_count must be positive.")
    if cfg.local_scan_variance_near_min_ratio < 1.0:
        raise ValueError("local_scan_variance_near_min_ratio must be >= 1.")
    if not cfg.local_scan_enabled:
        return {
            "enabled": False,
            "selected_beta": float("nan"),
            "selected_proposal_size": cfg.proposal_size,
            "rows": [],
            "scan_eval_total": 0,
            "scan_wall_time_sec": 0.0,
        }
    t_start = time.perf_counter()
    candidates = _build_q_scan_candidates(
        problem=problem,
        cfg=cfg,
        pilot_t=pilot_t,
        p0_for_qtarget=float(p0_for_qtarget),
        sigma_t=float(sigma_t),
        q_target=float(q_target),
        q_multipliers=cfg.local_scan_q_multipliers,
        swap_counts=cfg.local_scan_swap_counts,
    )
    if not candidates:
        return {
            "enabled": True,
            "selected_beta": float("nan"),
            "selected_proposal_size": cfg.proposal_size,
            "rows": [],
            "screen_rows": [],
            "scan_eval_total": 0,
            "scan_wall_time_sec": 0.0,
            "selected_reason": "no_local_scan_candidates",
        }
    strategy = str(cfg.local_scan_strategy)
    refine_rows: list[dict[str, Any]] = []
    refine_eval_total = 0
    refine_wall_time_sec = 0.0
    refine_steps_per_chain = 0
    refine_burn_in = 0
    sample_batches: list[dict[str, Any]] = []

    if strategy == "adaptive_q":
        coarse_q_multipliers = tuple(
            float(v) for v in (
                cfg.local_scan_coarse_q_multipliers
                if cfg.local_scan_coarse_q_multipliers
                else cfg.local_scan_q_multipliers
            )
        )
        coarse_candidates = _select_candidate_subset_by_multipliers(candidates, coarse_q_multipliers)
        if not coarse_candidates:
            coarse_candidates = list(candidates)
        screen = _run_local_scan_stage(
            problem,
            cfg,
            candidates=coarse_candidates,
            sigma_t=sigma_t,
            total_steps=int(cfg.local_scan_screen_total_steps),
            n_chains=int(cfg.local_scan_screen_chains),
            seed=seed,
            stage="coarse",
            return_sample_batches=bool(return_sample_batches),
        )
        screen_rows = screen["rows"]
        ranked_screen_indices = _rank_scan_rows(screen_rows, objective_name=cfg.local_scan_objective)
        for rank, idx in enumerate(ranked_screen_indices):
            screen_rows[idx]["screen_rank"] = int(rank)
            screen_rows[idx]["advanced_to_final"] = 0
            screen_rows[idx]["advanced_reason"] = None

        max_master_q_index = max(int(candidate["q_index"]) for candidate in candidates)
        refine_q_indices = _adaptive_refine_q_indices(
            screen_rows,
            objective_name=cfg.local_scan_objective,
            max_master_q_index=max_master_q_index,
            top_k=int(cfg.local_scan_refine_top_k),
            radius=int(cfg.local_scan_refine_radius),
            max_q_points=int(cfg.local_scan_refine_max_q_points),
        )
        refine_candidates = [
            dict(candidate)
            for candidate in candidates
            if int(candidate["q_index"]) in set(int(v) for v in refine_q_indices)
        ]
        refine_total_steps = int(
            cfg.local_scan_refine_total_steps
            if cfg.local_scan_refine_total_steps is not None
            else cfg.local_scan_screen_total_steps
        )
        refine_chains = int(
            cfg.local_scan_refine_chains
            if cfg.local_scan_refine_chains is not None
            else cfg.local_scan_screen_chains
        )
        refine = _run_local_scan_stage(
            problem,
            cfg,
            candidates=refine_candidates,
            sigma_t=sigma_t,
            total_steps=refine_total_steps,
            n_chains=refine_chains,
            seed=seed + 500_000,
            stage="refine",
            return_sample_batches=bool(return_sample_batches),
            init_states_by_config=screen["final_states"],
        )
        refine_rows = refine["rows"]
        refine_eval_total = int(refine["eval_total"])
        refine_wall_time_sec = float(refine["wall_time_sec"])
        refine_steps_per_chain = int(refine["steps_per_chain"])
        refine_burn_in = int(refine["burn_in"])
        ranked_refine_indices = _rank_scan_rows(refine_rows, objective_name=cfg.local_scan_objective)
        target_finalist_count = min(int(cfg.local_scan_finalist_count), len(refine_rows))
        finalist_indices = [int(idx) for idx in ranked_refine_indices[:target_finalist_count]]
        finalist_candidates = [dict(refine_rows[idx]) for idx in finalist_indices]
        for rank, idx in enumerate(ranked_refine_indices):
            refine_rows[idx]["refine_rank"] = int(rank)
            refine_rows[idx]["advanced_to_final"] = int(idx in finalist_indices)
            if idx in finalist_indices:
                refine_rows[idx]["advanced_reason"] = "refine_objective_rank"
        final = _run_local_scan_stage(
            problem,
            cfg,
            candidates=finalist_candidates,
            sigma_t=sigma_t,
            total_steps=int(cfg.local_scan_total_steps),
            n_chains=int(cfg.local_scan_chains),
            seed=seed + 1_000_000,
            stage="final",
            return_sample_batches=bool(return_sample_batches),
            init_states_by_config=refine["final_states"],
        )
        sample_batches = (
            list(screen.get("sample_batches", []))
            + list(refine.get("sample_batches", []))
            + list(final.get("sample_batches", []))
        )
    elif strategy == "fixed_grid":
        screen = _run_local_scan_stage(
            problem,
            cfg,
            candidates=candidates,
            sigma_t=sigma_t,
            total_steps=int(cfg.local_scan_screen_total_steps),
            n_chains=int(cfg.local_scan_screen_chains),
            seed=seed,
            stage="screen",
            return_sample_batches=bool(return_sample_batches),
        )
        screen_rows = screen["rows"]
        ranked_screen_indices = _rank_scan_rows(screen_rows, objective_name=cfg.local_scan_objective)
        target_finalist_count = min(int(cfg.local_scan_finalist_count), len(screen_rows))
        finalist_indices = [int(idx) for idx in ranked_screen_indices[:target_finalist_count]]
        finalist_candidates = [dict(screen_rows[idx]) for idx in finalist_indices]
        for rank, idx in enumerate(ranked_screen_indices):
            screen_rows[idx]["screen_rank"] = int(rank)
            screen_rows[idx]["advanced_to_final"] = int(idx in finalist_indices)
            if idx in finalist_indices:
                screen_rows[idx]["advanced_reason"] = "screen_objective_rank"

        final = _run_local_scan_stage(
            problem,
            cfg,
            candidates=finalist_candidates,
            sigma_t=sigma_t,
            total_steps=int(cfg.local_scan_total_steps),
            n_chains=int(cfg.local_scan_chains),
            seed=seed + 1_000_000,
            stage="final",
            return_sample_batches=bool(return_sample_batches),
            init_states_by_config=screen["final_states"],
        )
        screen_rows = screen["rows"]
        sample_batches = list(screen.get("sample_batches", [])) + list(final.get("sample_batches", []))
    else:
        raise ValueError(f"Unknown local_scan_strategy: {strategy}")

    rows = final["rows"]
    ranked_final_indices = _rank_scan_rows(rows, objective_name=cfg.local_scan_objective)
    metric_key = _scan_metric_key(cfg.local_scan_objective)
    finite_final_indices = [idx for idx in ranked_final_indices if np.isfinite(rows[idx].get(metric_key, np.inf))]
    if finite_final_indices:
        best_objective = float(min(float(rows[idx][metric_key]) for idx in finite_final_indices))
        objective_limit = float(best_objective * cfg.local_scan_variance_near_min_ratio)
        near_min_indices: list[int] = []
        for idx in finite_final_indices:
            objective = float(rows[idx][metric_key])
            ratio = float(objective / best_objective) if best_objective > 0.0 else float("inf")
            within_tol = bool(objective <= objective_limit)
            rows[idx]["objective_ratio_to_best"] = ratio
            rows[idx]["within_objective_tolerance"] = int(within_tol)
            if within_tol:
                near_min_indices.append(idx)
        best_idx = min(
            near_min_indices,
            key=lambda idx: (
                int(rows[idx]["q_index"]),
                int(rows[idx]["n_swap_pairs"]),
                float(rows[idx]["beta"]),
                float(rows[idx][metric_key]),
            ),
        )
        selected_beta = float(rows[best_idx]["beta"])
        selected_proposal_size = int(rows[best_idx]["n_swap_pairs"])
        selected_q_multiplier = float(rows[best_idx]["q_multiplier"])
        production_init_states = final["final_states"].get(str(rows[best_idx]["config_id"]))
        selected_reason = "two_stage_smallest_stable_configuration"
    else:
        fallback_candidate = screen_rows[ranked_screen_indices[0]]
        selected_beta = float(fallback_candidate["beta"]) if np.isfinite(fallback_candidate.get("beta", np.nan)) else float("nan")
        selected_proposal_size = int(fallback_candidate["n_swap_pairs"])
        selected_q_multiplier = float(fallback_candidate["q_multiplier"])
        production_init_states = final["final_states"].get(str(fallback_candidate["config_id"]))
        selected_reason = "fallback_smallest_screen_configuration"

    return {
        "enabled": True,
        "selected_beta": selected_beta,
        "selected_proposal_size": int(selected_proposal_size),
        "selected_n_swap_pairs": int(selected_proposal_size),
        "selected_q_multiplier": float(selected_q_multiplier),
        "selected_reason": selected_reason,
        "steps_per_chain": int(final["steps_per_chain"]),
        "burn_in": int(final["burn_in"]),
        "screen_steps_per_chain": int(screen["steps_per_chain"]),
        "screen_burn_in": int(screen["burn_in"]),
        "screen_eval_total": int(screen["eval_total"]),
        "refine_steps_per_chain": int(refine_steps_per_chain),
        "refine_burn_in": int(refine_burn_in),
        "refine_eval_total": int(refine_eval_total),
        "final_eval_total": int(final["eval_total"]),
        "scan_eval_total": int(screen["eval_total"] + refine_eval_total + final["eval_total"]),
        "scan_wall_time_sec": float(time.perf_counter() - t_start),
        "screen_wall_time_sec": float(screen["wall_time_sec"]),
        "refine_wall_time_sec": float(refine_wall_time_sec),
        "final_wall_time_sec": float(final["wall_time_sec"]),
        "selection_metrics": {
            "selection_rule": {
                "type": "adaptive_q_smallest_stable_configuration" if strategy == "adaptive_q" else "two_stage_smallest_stable_configuration",
                "objective": str(cfg.local_scan_objective),
                "objective_near_min_ratio": float(cfg.local_scan_variance_near_min_ratio),
                "preference_within_tolerance": "smaller_q_then_smaller_swap_then_smaller_beta",
            },
            "screening_rule": {
                "type": "adaptive_coarse_refine_final" if strategy == "adaptive_q" else "discrete_q_swap_grid",
                "strategy": strategy,
                "screen_total_steps": int(cfg.local_scan_screen_total_steps),
                "screen_chains": int(cfg.local_scan_screen_chains),
                "refine_total_steps": (
                    int(cfg.local_scan_refine_total_steps)
                    if cfg.local_scan_refine_total_steps is not None
                    else int(cfg.local_scan_screen_total_steps)
                ),
                "refine_chains": (
                    int(cfg.local_scan_refine_chains)
                    if cfg.local_scan_refine_chains is not None
                    else int(cfg.local_scan_screen_chains)
                ),
                "refine_top_k": int(cfg.local_scan_refine_top_k),
                "refine_radius": int(cfg.local_scan_refine_radius),
                "refine_max_q_points": int(cfg.local_scan_refine_max_q_points),
                "final_total_steps": int(cfg.local_scan_total_steps),
                "final_chains": int(cfg.local_scan_chains),
                "finalist_count": int(cfg.local_scan_finalist_count),
                "screen_ranking": str(cfg.local_scan_objective),
                "reuse_state_across_scan_stages": True,
                "reuse_finalist_state_for_production": True,
            },
            "q_ladder": {
                "type": "adaptive_coarse_refine_grid" if strategy == "adaptive_q" else "fixed_discrete_grid",
                "q_target_center": float(q_target),
                "q_multipliers": [float(mult) for mult in cfg.local_scan_q_multipliers],
                "coarse_q_multipliers": [float(mult) for mult in cfg.local_scan_coarse_q_multipliers],
                "swap_counts": [int(v) for v in cfg.local_scan_swap_counts],
            },
        },
        "screen_rows": screen_rows,
        "refine_rows": refine_rows,
        "rows": rows,
        "production_init_states": production_init_states,
        "sample_batches": sample_batches,
    }


def _selected_scan_row(scan: dict[str, Any]) -> dict[str, Any]:
    rows = list(scan.get("rows", []))
    if not rows:
        return {}
    beta = float(scan.get("selected_beta", np.nan))
    proposal_size = int(scan.get("selected_proposal_size", -1))
    matches = [
        row
        for row in rows
        if int(row.get("n_swap_pairs", -999)) == proposal_size
        and np.isfinite(row.get("beta", np.nan))
        and abs(float(row["beta"]) - beta) <= 1e-10
    ]
    return dict(matches[0] if matches else rows[0])


def _pack_scan_sample_batches(sample_batches: list[dict[str, Any]]) -> dict[str, Any]:
    log_weight_blocks: list[np.ndarray] = []
    tail_blocks: list[np.ndarray] = []
    for batch in sample_batches:
        logw = np.asarray(batch.get("log_weights", []), dtype=float)
        tail = np.asarray(batch.get("tail_indicators", []), dtype=np.int8)
        if logw.size != tail.size:
            raise ValueError("scan sample batch log_weights and tail_indicators must have matching sizes")
        if logw.size:
            log_weight_blocks.append(logw)
            tail_blocks.append(tail)
    if log_weight_blocks:
        log_weights = np.concatenate(log_weight_blocks)
        tail_indicators = np.concatenate(tail_blocks)
    else:
        log_weights = np.asarray([], dtype=float)
        tail_indicators = np.asarray([], dtype=np.int8)
    return {
        "log_weights": log_weights,
        "tail_indicators": tail_indicators,
        "n_batches": int(len(sample_batches)),
        "n_weighted_samples": int(tail_indicators.size),
    }


def _selected_scan_sample_pack(scan: dict[str, Any]) -> dict[str, Any]:
    selected_beta = float(scan.get("selected_beta", np.nan))
    selected_proposal_size = int(scan.get("selected_proposal_size", -1))
    selected_batches = [
        batch
        for batch in list(scan.get("sample_batches", []))
        if int(batch.get("n_swap_pairs", -999)) == selected_proposal_size
        and np.isfinite(batch.get("beta", np.nan))
        and abs(float(batch["beta"]) - selected_beta) <= 1e-10
    ]
    return _pack_scan_sample_batches(selected_batches)


def _run_scan_budget_production_traces(
    problem: PermutationTestProblem,
    *,
    checkpoints: tuple[int, ...],
    beta: float,
    sigma_t: float,
    cfg: MCMCWorkflowConfig,
    seed: int,
    init_states: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    checkpoints = tuple(sorted({int(v) for v in checkpoints}))
    if not checkpoints:
        raise ValueError("checkpoints must contain at least one positive value")
    max_total_steps = int(checkpoints[-1])
    max_steps_per_chain = _steps_per_chain(max_total_steps, int(cfg.chains))
    steps_per_checkpoint = {int(cp): _steps_per_chain(int(cp), int(cfg.chains)) for cp in checkpoints}
    unique_step_checkpoints = tuple(sorted(set(steps_per_checkpoint.values())))
    n_swap_pairs = resolve_n_swap_pairs(
        problem.n_treated,
        problem.n_control,
        proposal_size=cfg.proposal_size,
    )
    seed_seq = np.random.SeedSequence(seed)
    spawned = seed_seq.spawn(int(cfg.chains))
    if init_states is None:
        normalized_init_states = [None] * int(cfg.chains)
    else:
        if len(init_states) != int(cfg.chains):
            raise ValueError("init_states length must match cfg.chains")
        normalized_init_states = [np.asarray(state, dtype=np.int8).copy() for state in init_states]

    traces: list[dict[str, Any]] = []
    for chain_idx, ss in enumerate(spawned):
        rng = np.random.default_rng(ss)
        traces.append(
            _run_single_chain_full_trace(
                problem,
                rng,
                beta=float(beta),
                sigma_t=float(sigma_t),
                n_steps=max_steps_per_chain,
                init="random" if normalized_init_states[chain_idx] is None else "observed",
                init_state=normalized_init_states[chain_idx],
                tilt_mode=str(cfg.tilt_mode),
                n_swap_pairs=n_swap_pairs,
                checkpoint_steps=unique_step_checkpoints,
            )
        )
    return traces, steps_per_checkpoint


def _production_prefix_payload(
    *,
    traces: list[dict[str, Any]],
    steps_per_chain: int,
    burn_in: int,
    thin: int,
    beta: float,
) -> dict[str, np.ndarray]:
    q_chunks: list[np.ndarray] = []
    tail_chunks: list[np.ndarray] = []
    for trace in traces:
        q_chunks.append(np.asarray(trace["q_trace"][burn_in:steps_per_chain:thin], dtype=float))
        tail_chunks.append(np.asarray(trace["tail_trace"][burn_in:steps_per_chain:thin], dtype=np.int8))
    q_samples = np.concatenate(q_chunks) if q_chunks else np.asarray([], dtype=float)
    tail_indicators = np.concatenate(tail_chunks) if tail_chunks else np.asarray([], dtype=np.int8)
    return {
        "log_weights": np.asarray(float(beta) * q_samples, dtype=float),
        "tail_indicators": np.asarray(tail_indicators, dtype=np.int8),
    }


def _pooled_scan_plus_production_row(
    *,
    exact_p: float,
    base_row: dict[str, Any],
    scan_sample_pack: dict[str, Any],
    production_payload: dict[str, np.ndarray],
    estimator_variant: str = "scan_plus_production",
) -> dict[str, Any]:
    scan_logw = np.asarray(scan_sample_pack.get("log_weights", []), dtype=float)
    scan_tail = np.asarray(scan_sample_pack.get("tail_indicators", []), dtype=np.int8)
    prod_logw = np.asarray(production_payload.get("log_weights", []), dtype=float)
    prod_tail = np.asarray(production_payload.get("tail_indicators", []), dtype=np.int8)

    log_weight_blocks = [arr for arr in (scan_logw, prod_logw) if arr.size]
    tail_blocks = [arr for arr in (scan_tail, prod_tail) if arr.size]
    if not log_weight_blocks or not tail_blocks:
        raise ValueError("pooled estimator requires at least one non-empty sample block")

    all_logw = np.concatenate(log_weight_blocks)
    all_tail = np.concatenate(tail_blocks)
    shift = float(np.max(all_logw))
    weights = np.exp(all_logw - shift)
    weight_sum = float(np.sum(weights))
    estimate = float(np.dot(weights, all_tail) / weight_sum)
    weight_summary = summarize_weights(weights)

    row = dict(base_row)
    row["estimate"] = float(estimate)
    row["variance_estimate"] = np.nan
    row["snis_mcse_obm"] = np.nan
    row["tail_hits"] = int(np.sum(all_tail))
    row["tail_share_raw"] = np.nan
    row["ess"] = float(effective_sample_size(weights))
    row["weight_cv"] = float(weight_summary.cv)
    row["n_weighted_samples"] = int(weights.size)
    row["estimator_variant"] = str(estimator_variant)
    row["scan_n_weighted_samples"] = int(scan_tail.size)
    row["production_n_weighted_samples"] = int(prod_tail.size)
    row["pooled_scan_batch_count"] = int(scan_sample_pack.get("n_batches", 0))
    return _annotate_error_fields(row, float(exact_p))


def run_scan_budget_repeat_job(
    *,
    scenario_key: str,
    scenario_display: str,
    family: str | None,
    statistic_family: str | None,
    sample_size_band: str | None,
    problem: PermutationTestProblem,
    exact_p: float,
    repeat_idx: int,
    pilot_seed: int,
    total_budget: int,
    base_mcmc_cfg: MCMCWorkflowConfig,
    rule_budgets: list[dict[str, Any]],
    production_estimate_variance: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    pilot_t = iid_pilot_statistics(
        problem,
        n_samples=int(base_mcmc_cfg.pilot_samples),
        seed=int(pilot_seed),
    )
    sigma_t = estimate_scale_T(pilot_t, method=str(base_mcmc_cfg.scale_method))
    p0_for_qtarget = float(exact_p) if base_mcmc_cfg.use_true_p0_for_q_target else float(base_mcmc_cfg.p0_guess)
    q_target = float(float(p0_for_qtarget) ** float(base_mcmc_cfg.d_alpha))

    scan_records: list[dict[str, Any]] = []
    all_scan_sample_packs: dict[str, dict[str, Any]] = {}
    selected_scan_sample_packs: dict[str, dict[str, Any]] = {}

    for rule_idx, rule_budget in enumerate(rule_budgets):
        cfg = replace(
            base_mcmc_cfg,
            proposal_size=int(rule_budget["proposal_size"]),
            local_scan_swap_counts=(int(rule_budget["proposal_size"]),),
            local_scan_finalist_count=int(rule_budget["finalist_count"]),
            local_scan_screen_total_steps=int(rule_budget["screen_total_steps"]),
            local_scan_refine_total_steps=(
                int(rule_budget["refine_total_steps"])
                if rule_budget.get("refine_total_steps") is not None
                else base_mcmc_cfg.local_scan_refine_total_steps
            ),
            local_scan_total_steps=int(rule_budget["final_total_steps"]),
        )
        scan_seed = int(pilot_seed) + 1_000 + 100 * int(rule_idx)
        scan = local_beta_scan(
            problem,
            cfg,
            pilot_t=pilot_t,
            p0_for_qtarget=p0_for_qtarget,
            sigma_t=float(sigma_t),
            q_target=float(q_target),
            seed=int(scan_seed),
            return_sample_batches=True,
        )
        row = _selected_scan_row(scan)
        scan_sample_pack = _pack_scan_sample_batches(list(scan.get("sample_batches", [])))
        selected_scan_pack = _selected_scan_sample_pack(scan)
        beta_selection_budget = int(cfg.pilot_samples) + int(scan.get("scan_eval_total", 0))
        production_budget = int(total_budget) - int(beta_selection_budget)
        record = {
            **dict(rule_budget),
            "scenario": str(scenario_key),
            "scenario_display": str(scenario_display),
            "family": family,
            "statistic_family": statistic_family,
            "sample_size_band": sample_size_band,
            "repeat": int(repeat_idx),
            "pilot_seed": int(pilot_seed),
            "scan_seed": int(scan_seed),
            "exact_p": float(exact_p),
            "sigma_t": float(sigma_t),
            "p0_for_qtarget": float(p0_for_qtarget),
            "q_target": float(q_target),
            "selected_beta": float(scan.get("selected_beta", np.nan)),
            "selected_q_multiplier": float(scan.get("selected_q_multiplier", np.nan)),
            "selected_proposal_size": int(scan.get("selected_proposal_size", rule_budget["proposal_size"])),
            "selected_reason": str(scan.get("selected_reason", "")),
            "selected_q_tilt_tail_share": float(row.get("q_tilt_tail_share", np.nan)),
            "selected_ess": float(row.get("ess", np.nan)),
            "selected_weight_cv": float(row.get("weight_cv", np.nan)),
            "selected_objective": float(row.get("objective_selected", np.nan)),
            "pilot_eval_total": int(cfg.pilot_samples),
            "screen_eval_total": int(scan.get("screen_eval_total", 0)),
            "refine_eval_total": int(scan.get("refine_eval_total", 0)),
            "final_eval_total": int(scan.get("final_eval_total", 0)),
            "scan_eval_total": int(scan.get("scan_eval_total", 0)),
            "scan_sample_batch_count": int(scan_sample_pack["n_batches"]),
            "scan_n_weighted_samples": int(scan_sample_pack["n_weighted_samples"]),
            "selected_scan_sample_batch_count": int(selected_scan_pack["n_batches"]),
            "selected_scan_n_weighted_samples": int(selected_scan_pack["n_weighted_samples"]),
            "beta_selection_budget": int(beta_selection_budget),
            "production_budget": int(production_budget),
            "total_budget": int(total_budget),
            "valid_for_production": int(production_budget > 0 and np.isfinite(scan.get("selected_beta", np.nan))),
        }
        scan_records.append(record)
        all_scan_sample_packs[str(rule_budget["rule_name"])] = scan_sample_pack
        selected_scan_sample_packs[str(rule_budget["rule_name"])] = selected_scan_pack

    valid_scan_records = [row for row in scan_records if int(row["valid_for_production"]) == 1]
    grouped_rows: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in valid_scan_records:
        grouped_rows.setdefault((int(row["selected_proposal_size"]), float(row["selected_beta"])), []).append(row)

    production_records: list[dict[str, Any]] = []
    for group_idx, ((proposal_size, beta), group_rows) in enumerate(
        sorted(grouped_rows.items(), key=lambda item: (int(item[0][0]), float(item[0][1])))
    ):
        checkpoints = tuple(sorted({int(row["production_budget"]) for row in group_rows if int(row["production_budget"]) > 0}))
        if not checkpoints:
            continue
        prod_cfg = replace(
            base_mcmc_cfg,
            estimate_variance=bool(production_estimate_variance),
            proposal_size=int(proposal_size),
            local_scan_swap_counts=(int(proposal_size),),
        )
        prod_seed = int(pilot_seed) + 1_000_000 + 10_000 * int(group_idx)
        traces, steps_per_checkpoint = _run_scan_budget_production_traces(
            problem,
            checkpoints=checkpoints,
            beta=float(beta),
            sigma_t=float(sigma_t),
            cfg=prod_cfg,
            seed=int(prod_seed),
            init_states=None,
        )

        rows_by_checkpoint: dict[int, dict[str, Any]] = {}
        payloads_by_checkpoint: dict[int, dict[str, np.ndarray]] = {}
        for checkpoint in checkpoints:
            steps_per_chain = int(steps_per_checkpoint[int(checkpoint)])
            burn_in = _burn_in(steps_per_chain, float(prod_cfg.burn_in_fraction))
            rows_by_checkpoint[int(checkpoint)] = _mcmc_prefix_row(
                exact_p=float(exact_p),
                checkpoint=int(checkpoint),
                steps_per_chain=steps_per_chain,
                burn_in=burn_in,
                thin=int(prod_cfg.thin),
                beta=float(beta),
                sigma_t=float(sigma_t),
                tilt_mode=str(prod_cfg.tilt_mode),
                estimate_variance=bool(prod_cfg.estimate_variance),
                obm_batch_size=prod_cfg.obm_batch_size,
                traces=traces,
                n_chains=int(prod_cfg.chains),
            )
            payloads_by_checkpoint[int(checkpoint)] = _production_prefix_payload(
                traces=traces,
                steps_per_chain=steps_per_chain,
                burn_in=burn_in,
                thin=int(prod_cfg.thin),
                beta=float(beta),
            )

        for scan_row in group_rows:
            checkpoint = int(scan_row["production_budget"])
            base_row = dict(rows_by_checkpoint[checkpoint])
            base_row.update(
                {
                    "scenario": str(scenario_key),
                    "repeat": int(repeat_idx),
                    "rule_name": str(scan_row["rule_name"]),
                    "rule_kind": str(scan_row["rule_kind"]),
                    "beta_selection_budget": int(scan_row["beta_selection_budget"]),
                    "production_budget": int(scan_row["production_budget"]),
                    "total_budget": int(scan_row["total_budget"]),
                    "eval_incl_scan": int(scan_row["beta_selection_budget"]) + int(scan_row["production_budget"]),
                    "selected_q_multiplier": float(scan_row["selected_q_multiplier"]),
                    "selected_beta": float(scan_row["selected_beta"]),
                    "selected_proposal_size": int(scan_row["selected_proposal_size"]),
                    "selected_scan_q_tilt_tail_share": float(scan_row["selected_q_tilt_tail_share"]),
                    "selected_scan_ess": float(scan_row["selected_ess"]),
                    "selected_scan_weight_cv": float(scan_row["selected_weight_cv"]),
                    "scan_sample_batch_count": int(scan_row.get("scan_sample_batch_count", 0)),
                    "available_scan_n_weighted_samples": int(scan_row.get("scan_n_weighted_samples", 0)),
                    "available_selected_scan_n_weighted_samples": int(scan_row.get("selected_scan_n_weighted_samples", 0)),
                    "prod_seed": int(prod_seed),
                }
            )

            prod_only_row = dict(base_row)
            prod_only_row["estimator_variant"] = "production_only"
            prod_only_row["scan_n_weighted_samples"] = 0
            prod_only_row["production_n_weighted_samples"] = int(prod_only_row.get("n_weighted_samples", 0))
            prod_only_row["pooled_scan_batch_count"] = 0
            production_records.append(prod_only_row)

            selected_scan_row = _pooled_scan_plus_production_row(
                exact_p=float(exact_p),
                base_row=base_row,
                scan_sample_pack=selected_scan_sample_packs[str(scan_row["rule_name"])],
                production_payload=payloads_by_checkpoint[checkpoint],
                estimator_variant="selected_scan_plus_production",
            )
            production_records.append(selected_scan_row)

            pooled_row = _pooled_scan_plus_production_row(
                exact_p=float(exact_p),
                base_row=base_row,
                scan_sample_pack=all_scan_sample_packs[str(scan_row["rule_name"])],
                production_payload=payloads_by_checkpoint[checkpoint],
                estimator_variant="all_scan_plus_production",
            )
            production_records.append(pooled_row)

    return {
        "scan_records": scan_records,
        "production_records": production_records,
    }


def build_beta_workflow(
    problem: PermutationTestProblem,
    exact_p: float,
    cfg: MCMCWorkflowConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    init_payload = build_beta_initialization(
        problem,
        exact_p,
        cfg,
        seed=seed,
    )
    pilot_t = np.asarray(init_payload["pilot_t"], dtype=float)
    sigma_t = float(init_payload["sigma_t"])
    p0_for_qtarget = float(init_payload["p0_for_qtarget"])
    q_target = float(init_payload["q_target"])
    beta0_formula = float(init_payload["beta0_formula"])
    beta0_laplace = float(init_payload["beta0_laplace"])
    pilot_eval_total = int(init_payload["pilot_eval_total"])
    tuning_chain_eval_total = 0
    tuning_eval_total = int(pilot_eval_total + tuning_chain_eval_total)
    tuning_wall_time_sec = float(init_payload["pilot_wall_time_sec"])
    beta_tuned = float(beta0_laplace)

    if cfg.beta_override is not None:
        beta_used = float(cfg.beta_override)
        scan = {
            "enabled": False,
            "selected_beta": beta_used,
            "selected_proposal_size": cfg.proposal_size,
            "rows": [],
            "scan_eval_total": 0,
            "scan_wall_time_sec": 0.0,
            "selected_reason": "beta_override",
            "production_init_states": None,
        }
    else:
        scan = local_beta_scan(
            problem,
            cfg,
            pilot_t=pilot_t,
            p0_for_qtarget=p0_for_qtarget,
            sigma_t=float(sigma_t),
            q_target=q_target,
            seed=seed + 7,
            return_sample_batches=bool(
                str(cfg.production_estimator_variant) in {
                    "selected_scan_plus_production",
                    "all_scan_plus_production",
                }
            ),
        )
        beta_used = float(scan["selected_beta"])

    scan_eval_total = int(scan.get("scan_eval_total", 0))
    beta_selection_eval_total = int(tuning_eval_total + scan_eval_total)
    beta_selection_wall_time_sec = float(tuning_wall_time_sec + float(scan.get("scan_wall_time_sec", 0.0)))
    proposal_size_used = scan.get("selected_proposal_size", cfg.proposal_size)

    return {
        "beta0_formula": beta0_formula,
        "beta0_laplace": float(beta0_laplace),
        "beta_hat_tuned": beta_tuned,
        "beta_used": beta_used,
        "proposal_size_used": proposal_size_used,
        "sigma_t": float(sigma_t),
        "p0_for_qtarget": p0_for_qtarget,
        "q_target": q_target,
        "q_hat_beta_hat": float("nan"),
        "bracket_succeeded": True,
        "n_short_chain_calls": 0,
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
            "refine_eval_total": int(scan.get("refine_eval_total", 0)),
            "final_eval_total": int(scan.get("final_eval_total", 0)),
            "scan_eval_total": scan_eval_total,
            "beta_selection_eval_total": beta_selection_eval_total,
        },
        "beta_selection_wall_time_sec": beta_selection_wall_time_sec,
        "local_scan": scan,
        "history_tail": [],
        "tune_steps": 0,
        "tune_burn_in": 0,
        "tune_thin": int(cfg.tune_thin),
        "production_init_states": scan.get("production_init_states"),
        "production_estimator_variant": str(cfg.production_estimator_variant),
        "scan_sample_pack": (
            _selected_scan_sample_pack(scan)
            if str(cfg.production_estimator_variant) == "selected_scan_plus_production"
            else _pack_scan_sample_batches(list(scan.get("sample_batches", [])))
            if str(cfg.production_estimator_variant) == "all_scan_plus_production"
            else None
        ),
        "beta_selection_strategy": "pilot_only_discrete_q_swap_grid",
    }


def build_beta_initialization(
    problem: PermutationTestProblem,
    exact_p: float,
    cfg: MCMCWorkflowConfig,
    *,
    seed: int,
    p0_reference: float | None = None,
) -> dict[str, Any]:
    if p0_reference is None:
        p0_for_qtarget = float(exact_p) if cfg.use_true_p0_for_q_target else float(cfg.p0_guess)
    else:
        p0_for_qtarget = float(p0_reference)
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
    return {
        "pilot_t": np.asarray(pilot_t, dtype=float),
        "sigma_t": float(sigma_t),
        "beta0_formula": beta0_formula,
        "beta0_laplace": float(beta0_laplace),
        "p0_for_qtarget": p0_for_qtarget,
        "q_target": q_target,
        "pilot_eval_total": int(cfg.pilot_samples),
        "pilot_wall_time_sec": float(time.perf_counter() - t_start),
    }


def run_mcmc_trial_candidate(
    problem: PermutationTestProblem,
    exact_p: float,
    *,
    beta: float,
    sigma_t: float,
    proposal_size: float | int,
    p0_reference: float,
    total_budget: int,
    chains: int,
    burn_in_fraction: float,
    thin: int,
    estimate_variance: bool,
    obm_batch_size: int | None,
    tilt_mode: str,
    seed: int,
) -> dict[str, Any]:
    steps_per_chain = _steps_per_chain(total_budget, chains)
    burn_in = _burn_in(steps_per_chain, burn_in_fraction)
    kept_per_chain = _kept_samples_per_chain(steps_per_chain, burn_in, thin)
    n_swap_pairs = resolve_n_swap_pairs(
        problem.n_treated,
        problem.n_control,
        proposal_size=proposal_size,
    )
    t_start = time.perf_counter()
    res = run_mcmc_is(
        problem,
        beta=float(beta),
        sigma_t=float(sigma_t),
        n_steps=steps_per_chain,
        burn_in=burn_in,
        thin=int(thin),
        n_chains=int(chains),
        seed=int(seed),
        init="random",
        tilt_mode=str(tilt_mode),
        proposal_size=proposal_size,
        estimate_variance=bool(estimate_variance),
        obm_batch_size=obm_batch_size,
    )
    selection_objective_p0 = _snis_design_variance_objective(
        log_weights=np.asarray(res.log_weights, dtype=float),
        tail_indicators=np.asarray(res.tail_indicators, dtype=np.int8),
        p_reference=float(p0_reference),
        n_chains=int(chains),
        chain_kept_samples=int(kept_per_chain),
        obm_batch_size=obm_batch_size,
    )
    row = _annotate_error_fields(
        {
            "method": "mcmc_is",
            "checkpoint": int(total_budget),
            "estimate": float(res.estimate),
            "variance_estimate": (
                float(res.snis_variance_obm)
                if res.snis_variance_obm is not None and np.isfinite(res.snis_variance_obm)
                else np.nan
            ),
            "snis_mcse_obm": (
                float(res.snis_mcse_obm)
                if res.snis_mcse_obm is not None and np.isfinite(res.snis_mcse_obm)
                else np.nan
            ),
            "tail_hits": int(res.tail_hits_weighted_sample),
            "tail_share_raw": float(res.tail_share_raw_sample),
            "q_tilt_tail_share": float(res.tail_share_raw_sample),
            "ess": float(res.ess),
            "acceptance_rate": float(res.overall_acceptance_rate),
            "weight_cv": float(res.weight_summary.cv),
            "beta": float(beta),
            "sigma_t": float(sigma_t),
            "tilt_mode": str(tilt_mode),
            "selection_objective_p0": float(selection_objective_p0),
            "proposal_size": int(n_swap_pairs),
            "n_swap_pairs": int(n_swap_pairs),
            "wall_time_sec": float(time.perf_counter() - t_start),
            "eval_excl_tuning": float(_mcmc_eval_count(steps_per_chain, chains)),
            "eval_incl_tuning": float(_mcmc_eval_count(steps_per_chain, chains)),
            "n_weighted_samples": int(res.n_weighted_samples),
            "steps_per_chain": int(steps_per_chain),
            "burn_in": int(burn_in),
            "chains": int(chains),
            "thin": int(thin),
            "zero_hits": int(res.tail_hits_weighted_sample == 0),
        },
        exact_p,
    )
    return row


def build_mcmc_objective_grid_candidates(
    problem: PermutationTestProblem,
    *,
    pilot_t: np.ndarray,
    sigma_t: float,
    p0_for_qtarget: float,
    q_target: float,
    q_multipliers: tuple[float, ...] = DEFAULT_MCMC_OBJECTIVE_GRID_Q_MULTIPLIERS,
    n_swap_pairs_values: tuple[int, ...] = DEFAULT_MCMC_OBJECTIVE_GRID_SWAP_COUNTS,
    beta_max: float,
    q_floor: float = 1e-12,
) -> list[dict[str, Any]]:
    q_grid = tuple(float(v) for v in q_multipliers)
    if not q_grid:
        raise ValueError("q_multipliers must be non-empty.")
    swap_values = tuple(int(v) for v in n_swap_pairs_values)
    if not swap_values:
        raise ValueError("n_swap_pairs_values must be non-empty.")
    max_swaps = min(int(problem.n_treated), int(problem.n_control))

    candidates: list[dict[str, Any]] = []
    for q_index, q_multiplier in enumerate(q_grid):
        if q_multiplier <= 0.0:
            raise ValueError("q_multipliers must all be positive.")
        q_raw = float(q_target * q_multiplier)
        q_trial = float(max(q_raw, float(q_floor)))
        for n_swap_pairs in swap_values:
            if n_swap_pairs < 1 or n_swap_pairs > max_swaps:
                raise ValueError(
                    f"n_swap_pairs must lie in [1, {max_swaps}], received {n_swap_pairs}."
                )
            config_id = f"q{q_index:02d}_s{n_swap_pairs}"
            candidate = {
                "config_id": config_id,
                "label": config_id,
                "q_index": int(q_index),
                "q_multiplier": float(q_multiplier),
                "q_target": float(q_target),
                "q_trial_raw": float(q_raw),
                "q_trial": float(q_trial),
                "q_floor": float(q_floor),
                "q_floor_applied": int(not np.isclose(q_trial, q_raw, rtol=1e-12, atol=1e-15)),
                "n_swap_pairs": int(n_swap_pairs),
                "proposal_size": int(n_swap_pairs),
                "sigma_t": float(sigma_t),
            }
            try:
                beta_trial = float(
                    init_beta_from_iid_pilot(
                        pilot_T=pilot_t,
                        T_obs=problem.t_obs,
                        sigma_T=sigma_t,
                        p0=p0_for_qtarget,
                        q_target=q_trial,
                        beta_max=beta_max,
                    )
                )
                if np.isfinite(beta_trial) and beta_trial > 0.0:
                    candidate["beta"] = float(beta_trial)
                    candidate["status"] = "ok"
                    candidate["invalid_reason"] = None
                else:
                    candidate["beta"] = np.nan
                    candidate["status"] = "invalid_q_map"
                    candidate["invalid_reason"] = (
                        "Pilot-based q-to-beta mapping produced a non-finite or non-positive beta."
                    )
            except Exception as exc:
                candidate["beta"] = np.nan
                candidate["status"] = "invalid_q_map"
                candidate["invalid_reason"] = str(exc)
            candidates.append(candidate)
    return candidates


def _positive_finite(value: Any) -> bool:
    return bool(np.isfinite(value) and float(value) > 0.0)


def _objective_grid_penalties(row: dict[str, Any]) -> dict[str, float]:
    q_real = float(row.get("q_tilt_tail_share", np.nan))
    q_trial = float(row.get("q_trial", np.nan))
    n_weighted_samples = int(row.get("n_weighted_samples", 0))
    weight_cv = float(row.get("weight_cv", np.nan))
    return {
        "P_q": _q_match_penalty(q_real, q_trial, n_weighted_samples),
        "P_deg": _weight_cv_penalty(weight_cv, 0.25),
    }


def score_mcmc_objective_grid_repeat_row(row: dict[str, Any]) -> dict[str, Any]:
    penalties = _objective_grid_penalties(row)
    varhat = float(row.get("variance_estimate", np.nan))
    base_varhat = float(varhat) if _positive_finite(varhat) else float("inf")

    scored = dict(row)
    scored.update(penalties)
    scored["objective_oracle_abs_log10"] = float(row.get("abs_log10_error", float("inf")))
    scored["objective_varhat"] = base_varhat
    scored["objective_varhat_qmatch_soft"] = (
        float(base_varhat * penalties["P_q"]) if np.isfinite(base_varhat) else float("inf")
    )
    scored["objective_varhat_degeneracy_soft"] = (
        float(base_varhat * penalties["P_deg"]) if np.isfinite(base_varhat) else float("inf")
    )
    scored["objective_varhat_qmatch_degeneracy_soft"] = (
        float(base_varhat * penalties["P_q"] * penalties["P_deg"])
        if np.isfinite(base_varhat)
        else float("inf")
    )
    return scored


def _invalid_mcmc_objective_grid_repeat_row(
    *,
    candidate: dict[str, Any],
    exact_p: float,
    sigma_t: float,
    total_budget: int,
    repeat_idx: int,
    seed: int,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    row = {
        "method": "mcmc_is",
        "checkpoint": int(total_budget),
        "estimate": np.nan,
        "variance_estimate": np.nan,
        "snis_mcse_obm": np.nan,
        "tail_hits": 0,
        "tail_share_raw": 0.0,
        "q_tilt_tail_share": 0.0,
        "ess": np.nan,
        "acceptance_rate": 0.0,
        "weight_cv": np.nan,
        "beta": float(candidate.get("beta", np.nan)),
        "sigma_t": float(sigma_t),
        "tilt_mode": "smooth_hinge",
        "selection_objective_p0": np.nan,
        "proposal_size": int(candidate["proposal_size"]),
        "n_swap_pairs": int(candidate["n_swap_pairs"]),
        "wall_time_sec": 0.0,
        "eval_excl_tuning": 0.0,
        "eval_incl_tuning": 0.0,
        "n_weighted_samples": 0,
        "steps_per_chain": 0,
        "burn_in": 0,
        "chains": 0,
        "thin": 1,
        "zero_hits": 1,
        "exact_p": float(exact_p),
        "squared_error": float("inf"),
        "abs_log10_error": float("inf"),
        "abs_rel_error": float("inf"),
        "config_id": str(candidate["config_id"]),
        "label": str(candidate["label"]),
        "q_index": int(candidate["q_index"]),
        "q_multiplier": float(candidate["q_multiplier"]),
        "q_target": float(candidate["q_target"]),
        "q_trial_raw": float(candidate["q_trial_raw"]),
        "q_trial": float(candidate["q_trial"]),
        "q_floor": float(candidate["q_floor"]),
        "q_floor_applied": int(candidate["q_floor_applied"]),
        "trial_repeat": int(repeat_idx),
        "trial_budget": int(total_budget),
        "seed": int(seed),
        "status": str(status),
        "invalid_reason": error_message,
    }
    return score_mcmc_objective_grid_repeat_row(row)


def _mcmc_objective_grid_repeat_worker(
    *,
    problem: PermutationTestProblem,
    exact_p: float,
    sigma_t: float,
    p0_reference: float,
    trial_budget: int,
    mcmc_cfg: MCMCWorkflowConfig,
    candidate: dict[str, Any],
    repeat_idx: int,
    seed: int,
) -> dict[str, Any]:
    if str(candidate["status"]) != "ok":
        return _invalid_mcmc_objective_grid_repeat_row(
            candidate=candidate,
            exact_p=exact_p,
            sigma_t=sigma_t,
            total_budget=trial_budget,
            repeat_idx=repeat_idx,
            seed=seed,
            status="invalid_q_map",
            error_message=str(candidate.get("invalid_reason")),
        )
    try:
        row = run_mcmc_trial_candidate(
            problem,
            exact_p,
            beta=float(candidate["beta"]),
            sigma_t=float(sigma_t),
            proposal_size=int(candidate["n_swap_pairs"]),
            p0_reference=float(p0_reference),
            total_budget=int(trial_budget),
            chains=int(mcmc_cfg.chains),
            burn_in_fraction=float(mcmc_cfg.burn_in_fraction),
            thin=int(mcmc_cfg.thin),
            estimate_variance=bool(mcmc_cfg.estimate_variance),
            obm_batch_size=mcmc_cfg.obm_batch_size,
            tilt_mode=str(mcmc_cfg.tilt_mode),
            seed=int(seed),
        )
    except Exception as exc:
        return _invalid_mcmc_objective_grid_repeat_row(
            candidate=candidate,
            exact_p=exact_p,
            sigma_t=sigma_t,
            total_budget=trial_budget,
            repeat_idx=repeat_idx,
            seed=seed,
            status="invalid_run",
            error_message=str(exc),
        )
    row.update(
        {
            "config_id": str(candidate["config_id"]),
            "label": str(candidate["label"]),
            "q_index": int(candidate["q_index"]),
            "q_multiplier": float(candidate["q_multiplier"]),
            "q_target": float(candidate["q_target"]),
            "q_trial_raw": float(candidate["q_trial_raw"]),
            "q_trial": float(candidate["q_trial"]),
            "q_floor": float(candidate["q_floor"]),
            "q_floor_applied": int(candidate["q_floor_applied"]),
            "trial_repeat": int(repeat_idx),
            "trial_budget": int(trial_budget),
            "seed": int(seed),
            "status": "ok",
            "invalid_reason": None,
        }
    )
    return score_mcmc_objective_grid_repeat_row(row)


def _objective_value_key(value: Any) -> float:
    if np.isfinite(value):
        return float(value)
    return float("inf")


def _aggregate_objective_values(values: np.ndarray) -> tuple[float, float, float, float, float]:
    if values.size == 0:
        return (float("inf"),) * 5
    finite = np.all(np.isfinite(values))
    if not finite:
        return (float("inf"),) * 5
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values, ddof=0)),
        float(np.min(values)),
        float(np.max(values)),
    )


def summarize_mcmc_objective_grid_configs(
    repeat_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not repeat_rows:
        raise ValueError("repeat_rows must be non-empty.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in repeat_rows:
        groups.setdefault(str(row["config_id"]), []).append(row)

    out: list[dict[str, Any]] = []
    for config_id in sorted(groups):
        rows = sorted(groups[config_id], key=lambda row: int(row["trial_repeat"]))
        first = rows[0]
        squared_error = np.asarray([float(row.get("squared_error", np.inf)) for row in rows], dtype=float)
        abs_log = np.asarray([float(row.get("abs_log10_error", np.inf)) for row in rows], dtype=float)
        oracle_rmse = float(np.sqrt(np.mean(squared_error))) if np.all(np.isfinite(squared_error)) else float("inf")
        oracle_abs = float(np.mean(abs_log)) if np.all(np.isfinite(abs_log)) else float("inf")
        statuses = [str(row.get("status", "ok")) for row in rows]
        n_ok = sum(status == "ok" for status in statuses)
        n_invalid_q_map = sum(status == "invalid_q_map" for status in statuses)
        n_invalid_run = sum(status == "invalid_run" for status in statuses)
        if n_ok == len(rows):
            config_status = "ok"
        elif n_invalid_q_map == len(rows):
            config_status = "invalid_q_map"
        elif n_invalid_run == len(rows):
            config_status = "invalid_run"
        else:
            config_status = "mixed"

        summary = {
            "config_id": str(first["config_id"]),
            "label": str(first["label"]),
            "q_index": int(first["q_index"]),
            "q_multiplier": float(first["q_multiplier"]),
            "q_target": float(first["q_target"]),
            "q_trial_raw": float(first["q_trial_raw"]),
            "q_trial": float(first["q_trial"]),
            "q_floor": float(first["q_floor"]),
            "q_floor_applied": int(first["q_floor_applied"]),
            "beta": float(first["beta"]) if np.isfinite(first.get("beta", np.nan)) else np.nan,
            "proposal_size": int(first["proposal_size"]),
            "n_swap_pairs": int(first["n_swap_pairs"]),
            "sigma_t": float(first["sigma_t"]),
            "status": config_status,
            "n_repeats": int(len(rows)),
            "n_ok": int(n_ok),
            "n_invalid_q_map": int(n_invalid_q_map),
            "n_invalid_run": int(n_invalid_run),
            "mean_oracle_rmse": float(oracle_rmse),
            "mean_oracle_abs_log10": float(oracle_abs),
        }
        for metric_name in ("tail_hits", "q_tilt_tail_share", "acceptance_rate", "ess", "weight_cv"):
            values = np.asarray([float(row.get(metric_name, np.nan)) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            summary[f"mean_{metric_name}"] = float(np.mean(finite)) if finite.size else np.nan
            summary[f"std_{metric_name}"] = float(np.std(finite, ddof=0)) if finite.size else np.nan
        for objective_name in MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES:
            values = np.asarray([float(row.get(f"objective_{objective_name}", np.inf)) for row in rows], dtype=float)
            mean_v, median_v, std_v, min_v, max_v = _aggregate_objective_values(values)
            summary[f"mean_objective_{objective_name}"] = mean_v
            summary[f"median_objective_{objective_name}"] = median_v
            summary[f"std_objective_{objective_name}"] = std_v
            summary[f"min_objective_{objective_name}"] = min_v
            summary[f"max_objective_{objective_name}"] = max_v
        out.append(summary)
    return sorted(out, key=lambda row: (int(row["q_index"]), int(row["n_swap_pairs"])))


def _select_objective_grid_winner(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
) -> dict[str, Any] | None:
    finite_rows = [row for row in rows if np.isfinite(row.get(metric_key, np.inf))]
    if not finite_rows:
        return None
    return min(
        finite_rows,
        key=lambda row: (
            float(row[metric_key]),
            int(row["q_index"]),
            int(row["n_swap_pairs"]),
            float(row["beta"]) if np.isfinite(row.get("beta", np.nan)) else float("inf"),
        ),
    )


def _objective_grid_similarity(
    *,
    q_index: int,
    n_swap_pairs: int,
    oracle_q_index: int,
    oracle_n_swap_pairs: int,
    q_grid_size: int,
    max_swap_distance: int,
) -> tuple[int, int, float]:
    q_distance = abs(int(q_index) - int(oracle_q_index))
    swap_distance = abs(int(n_swap_pairs) - int(oracle_n_swap_pairs))
    denominator = max(1, int(q_grid_size - 1) + int(max_swap_distance))
    similarity = 1.0 - float(q_distance + swap_distance) / float(denominator)
    return q_distance, swap_distance, similarity


def select_mcmc_objective_grid_winners(
    config_summary: list[dict[str, Any]],
    *,
    q_multipliers: tuple[float, ...],
    n_swap_pairs_values: tuple[int, ...],
) -> dict[str, Any]:
    if not config_summary:
        raise ValueError("config_summary must be non-empty.")
    max_swap_distance = max(n_swap_pairs_values) - min(n_swap_pairs_values) if n_swap_pairs_values else 0
    oracle_winner = _select_objective_grid_winner(config_summary, metric_key="mean_oracle_rmse")
    winner_rows: list[dict[str, Any]] = []
    objective_to_config: dict[str, str] = {}

    for objective_name in MCMC_OBJECTIVE_GRID_ALL_OBJECTIVES:
        metric_key = (
            "mean_oracle_rmse"
            if objective_name == "oracle_rmse"
            else "mean_oracle_abs_log10"
            if objective_name == "oracle_abs_log10"
            else f"mean_objective_{objective_name}"
        )
        selected = _select_objective_grid_winner(config_summary, metric_key=metric_key)
        row = {
            "objective_name": str(objective_name),
            "objective_kind": "oracle" if objective_name.startswith("oracle_") else "realistic",
            "metric_key": metric_key,
        }
        if selected is None:
            row.update(
                {
                    "config_id": None,
                    "label": None,
                    "q_index": None,
                    "q_multiplier": None,
                    "n_swap_pairs": None,
                    "beta": None,
                    "selected_objective_value": float("inf"),
                    "oracle_exact_match": np.nan,
                    "oracle_fuzzy_similarity": np.nan,
                    "oracle_q_index_distance": np.nan,
                    "oracle_swap_distance": np.nan,
                    "winner_status": "no_finite_winner",
                }
            )
        else:
            if oracle_winner is None:
                q_distance = swap_distance = 0
                similarity = np.nan
                exact_match = np.nan
            else:
                q_distance, swap_distance, similarity = _objective_grid_similarity(
                    q_index=int(selected["q_index"]),
                    n_swap_pairs=int(selected["n_swap_pairs"]),
                    oracle_q_index=int(oracle_winner["q_index"]),
                    oracle_n_swap_pairs=int(oracle_winner["n_swap_pairs"]),
                    q_grid_size=len(q_multipliers),
                    max_swap_distance=max_swap_distance,
                )
                exact_match = int(
                    int(selected["q_index"]) == int(oracle_winner["q_index"])
                    and int(selected["n_swap_pairs"]) == int(oracle_winner["n_swap_pairs"])
                )
            row.update(
                {
                    "config_id": str(selected["config_id"]),
                    "label": str(selected["label"]),
                    "q_index": int(selected["q_index"]),
                    "q_multiplier": float(selected["q_multiplier"]),
                    "n_swap_pairs": int(selected["n_swap_pairs"]),
                    "beta": float(selected["beta"]) if np.isfinite(selected.get("beta", np.nan)) else np.nan,
                    "selected_objective_value": float(selected[metric_key]),
                    "oracle_exact_match": exact_match,
                    "oracle_fuzzy_similarity": similarity,
                    "oracle_q_index_distance": q_distance,
                    "oracle_swap_distance": swap_distance,
                    "winner_status": str(selected.get("status", "ok")),
                }
            )
            objective_to_config[str(objective_name)] = str(selected["config_id"])
        winner_rows.append(row)

    return {
        "oracle_winner": next(row for row in winner_rows if row["objective_name"] == "oracle_rmse"),
        "objective_winners": winner_rows,
        "objective_to_config": objective_to_config,
    }


def summarize_mcmc_objective_grid_seed_noise(
    repeat_rows: list[dict[str, Any]],
    *,
    objective_winners: list[dict[str, Any]],
    q_multipliers: tuple[float, ...],
    n_swap_pairs_values: tuple[int, ...],
) -> dict[str, Any]:
    realistic_winners = {
        str(row["objective_name"]): row
        for row in objective_winners
        if str(row["objective_kind"]) == "realistic"
    }
    max_swap_distance = max(n_swap_pairs_values) - min(n_swap_pairs_values) if n_swap_pairs_values else 0
    repeats = sorted({int(row["trial_repeat"]) for row in repeat_rows})
    repeat_winner_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for objective_name in MCMC_OBJECTIVE_GRID_REALISTIC_OBJECTIVES:
        aggregate_winner = realistic_winners.get(str(objective_name))
        winner_counts: dict[str, int] = {}
        exact_matches: list[float] = []
        fuzzy_scores: list[float] = []

        for repeat_idx in repeats:
            sub = [row for row in repeat_rows if int(row["trial_repeat"]) == int(repeat_idx)]
            finite_rows = [row for row in sub if np.isfinite(row.get(f"objective_{objective_name}", np.inf))]
            if finite_rows:
                selected = min(
                    finite_rows,
                    key=lambda row: (
                        float(row[f"objective_{objective_name}"]),
                        int(row["q_index"]),
                        int(row["n_swap_pairs"]),
                        float(row["beta"]) if np.isfinite(row.get("beta", np.nan)) else float("inf"),
                    ),
                )
                config_id = str(selected["config_id"])
                winner_counts[config_id] = winner_counts.get(config_id, 0) + 1
                if aggregate_winner is not None and aggregate_winner.get("config_id") is not None:
                    q_distance, swap_distance, similarity = _objective_grid_similarity(
                        q_index=int(selected["q_index"]),
                        n_swap_pairs=int(selected["n_swap_pairs"]),
                        oracle_q_index=int(aggregate_winner["q_index"]),
                        oracle_n_swap_pairs=int(aggregate_winner["n_swap_pairs"]),
                        q_grid_size=len(q_multipliers),
                        max_swap_distance=max_swap_distance,
                    )
                    exact_match = int(config_id == str(aggregate_winner["config_id"]))
                else:
                    q_distance = np.nan
                    swap_distance = np.nan
                    similarity = np.nan
                    exact_match = np.nan
                repeat_winner_rows.append(
                    {
                        "objective_name": str(objective_name),
                        "trial_repeat": int(repeat_idx),
                        "config_id": config_id,
                        "q_index": int(selected["q_index"]),
                        "q_multiplier": float(selected["q_multiplier"]),
                        "n_swap_pairs": int(selected["n_swap_pairs"]),
                        "beta": float(selected["beta"]) if np.isfinite(selected.get("beta", np.nan)) else np.nan,
                        "objective_value": float(selected[f"objective_{objective_name}"]),
                        "aggregate_config_id": None if aggregate_winner is None else aggregate_winner.get("config_id"),
                        "aggregate_exact_match": exact_match,
                        "aggregate_fuzzy_similarity": similarity,
                        "aggregate_q_index_distance": q_distance,
                        "aggregate_swap_distance": swap_distance,
                    }
                )
                if np.isfinite(exact_match):
                    exact_matches.append(float(exact_match))
                if np.isfinite(similarity):
                    fuzzy_scores.append(float(similarity))
            else:
                repeat_winner_rows.append(
                    {
                        "objective_name": str(objective_name),
                        "trial_repeat": int(repeat_idx),
                        "config_id": None,
                        "q_index": None,
                        "q_multiplier": None,
                        "n_swap_pairs": None,
                        "beta": None,
                        "objective_value": float("inf"),
                        "aggregate_config_id": None if aggregate_winner is None else aggregate_winner.get("config_id"),
                        "aggregate_exact_match": np.nan,
                        "aggregate_fuzzy_similarity": np.nan,
                        "aggregate_q_index_distance": np.nan,
                        "aggregate_swap_distance": np.nan,
                    }
                )

        summary_rows.append(
            {
                "objective_name": str(objective_name),
                "aggregate_config_id": None if aggregate_winner is None else aggregate_winner.get("config_id"),
                "exact_match_rate": float(np.mean(exact_matches)) if exact_matches else np.nan,
                "mean_fuzzy_similarity": float(np.mean(fuzzy_scores)) if fuzzy_scores else np.nan,
                "winner_frequency": {
                    key: int(value)
                    for key, value in sorted(winner_counts.items(), key=lambda item: (-item[1], item[0]))
                },
            }
        )
    return {
        "summary": summary_rows,
        "repeat_winners": repeat_winner_rows,
    }


def run_mcmc_objective_grid_study(
    problem: PermutationTestProblem,
    exact_p: float,
    *,
    mcmc_cfg: MCMCWorkflowConfig,
    q_multipliers: tuple[float, ...] = DEFAULT_MCMC_OBJECTIVE_GRID_Q_MULTIPLIERS,
    n_swap_pairs_values: tuple[int, ...] = DEFAULT_MCMC_OBJECTIVE_GRID_SWAP_COUNTS,
    trial_repeats: int,
    trial_budget: int,
    base_seed: int,
    q_floor: float = 1e-12,
    n_jobs: int = 1,
) -> dict[str, Any]:
    p0_for_qtarget = float(exact_p) if mcmc_cfg.use_true_p0_for_q_target else float(mcmc_cfg.p0_guess)
    pilot_t = iid_pilot_statistics(problem, n_samples=mcmc_cfg.pilot_samples, seed=base_seed)
    sigma_t = estimate_scale_T(pilot_t, method=mcmc_cfg.scale_method)
    q_target = float(p0_for_qtarget ** float(mcmc_cfg.d_alpha))
    candidates = build_mcmc_objective_grid_candidates(
        problem,
        pilot_t=pilot_t,
        sigma_t=float(sigma_t),
        p0_for_qtarget=p0_for_qtarget,
        q_target=q_target,
        q_multipliers=tuple(float(v) for v in q_multipliers),
        n_swap_pairs_values=tuple(int(v) for v in n_swap_pairs_values),
        beta_max=float(mcmc_cfg.beta_max_init),
        q_floor=float(q_floor),
    )

    jobs = [
        (dict(candidate), int(rep), int(base_seed + 100_000 * cfg_idx + 1_000 * rep))
        for cfg_idx, candidate in enumerate(candidates)
        for rep in range(int(trial_repeats))
    ]
    repeat_rows: list[dict[str, Any]] = []
    n_workers = _effective_n_jobs(n_jobs, len(jobs))
    executor = _try_make_process_pool(n_workers) if n_workers > 1 else None

    if executor is None:
        for candidate, repeat_idx, seed in jobs:
            repeat_rows.append(
                _mcmc_objective_grid_repeat_worker(
                    problem=problem,
                    exact_p=exact_p,
                    sigma_t=float(sigma_t),
                    p0_reference=p0_for_qtarget,
                    trial_budget=int(trial_budget),
                    mcmc_cfg=mcmc_cfg,
                    candidate=candidate,
                    repeat_idx=int(repeat_idx),
                    seed=int(seed),
                )
            )
    else:
        with executor:
            futures = [
                executor.submit(
                    _mcmc_objective_grid_repeat_worker,
                    problem=problem,
                    exact_p=exact_p,
                    sigma_t=float(sigma_t),
                    p0_reference=p0_for_qtarget,
                    trial_budget=int(trial_budget),
                    mcmc_cfg=mcmc_cfg,
                    candidate=candidate,
                    repeat_idx=int(repeat_idx),
                    seed=int(seed),
                )
                for candidate, repeat_idx, seed in jobs
            ]
            for future in cf.as_completed(futures):
                repeat_rows.append(future.result())

    repeat_rows = sorted(
        repeat_rows,
        key=lambda row: (int(row["q_index"]), int(row["n_swap_pairs"]), int(row["trial_repeat"])),
    )
    config_summary = summarize_mcmc_objective_grid_configs(repeat_rows)
    winner_payload = select_mcmc_objective_grid_winners(
        config_summary,
        q_multipliers=tuple(float(v) for v in q_multipliers),
        n_swap_pairs_values=tuple(int(v) for v in n_swap_pairs_values),
    )
    seed_noise = summarize_mcmc_objective_grid_seed_noise(
        repeat_rows,
        objective_winners=winner_payload["objective_winners"],
        q_multipliers=tuple(float(v) for v in q_multipliers),
        n_swap_pairs_values=tuple(int(v) for v in n_swap_pairs_values),
    )
    return {
        "repeat_records": repeat_rows,
        "config_summary": config_summary,
        "oracle_winner": winner_payload["oracle_winner"],
        "objective_winners": winner_payload["objective_winners"],
        "objective_to_config": winner_payload["objective_to_config"],
        "objective_seed_noise": seed_noise["summary"],
        "repeat_winner_records": seed_noise["repeat_winners"],
        "study_context": {
            "exact_p": float(exact_p),
            "p0_for_qtarget": p0_for_qtarget,
            "q_target": q_target,
            "sigma_t": float(sigma_t),
            "pilot_samples": int(mcmc_cfg.pilot_samples),
            "q_multipliers": [float(v) for v in q_multipliers],
            "n_swap_pairs_values": [int(v) for v in n_swap_pairs_values],
            "trial_repeats": int(trial_repeats),
            "trial_budget": int(trial_budget),
            "q_floor": float(q_floor),
        },
    }


def _named_mcmc_replicate_worker(
    *,
    problem: PermutationTestProblem,
    exact_p: float,
    checkpoints: tuple[int, ...],
    sigma_t: float,
    template_cfg: MCMCWorkflowConfig | BetaSweepStudyConfig,
    config_spec: dict[str, Any],
    rep: int,
    rep_seed: int,
) -> list[dict[str, Any]]:
    resolved_swaps = resolve_n_swap_pairs(
        problem.n_treated,
        problem.n_control,
        proposal_size=config_spec["proposal_size"],
    )
    worker_cfg = BetaSweepStudyConfig(
        estimation_points=tuple(int(v) for v in checkpoints),
        repeats=1,
        beta_multipliers=(1.0,),
        chains=int(template_cfg.chains),
        burn_in_fraction=float(template_cfg.burn_in_fraction),
        thin=int(template_cfg.thin),
        estimate_variance=bool(template_cfg.estimate_variance),
        obm_batch_size=template_cfg.obm_batch_size,
        chain_n_jobs=int(getattr(template_cfg, "chain_n_jobs", 1)),
        tilt_mode=str(template_cfg.tilt_mode),
        proposal_size=config_spec["proposal_size"],
        base_seed=int(rep_seed),
        n_jobs=1,
    )
    rows = _run_mcmc_cumulative_checkpoints(
        problem,
        exact_p,
        checkpoints=tuple(int(v) for v in checkpoints),
        beta=float(config_spec["beta"]),
        sigma_t=float(sigma_t),
        cfg=worker_cfg,
        seed=int(rep_seed),
    )
    for row in rows:
        row["label"] = str(config_spec["label"])
        row["config_id"] = str(config_spec.get("config_id", config_spec["label"]))
        row["replicate"] = int(rep)
        row["seed"] = int(rep_seed)
        row["proposal_size"] = config_spec["proposal_size"]
        row["n_swap_pairs"] = int(resolved_swaps)
        row["source"] = str(config_spec.get("source", "grid"))
        row["selected_by_objectives"] = list(config_spec.get("selected_by_objectives", []))
    return rows


def run_named_mcmc_checkpoint_study(
    problem: PermutationTestProblem,
    exact_p: float,
    *,
    config_specs: list[dict[str, Any]],
    sigma_t: float,
    estimation_points: tuple[int, ...],
    repeats: int,
    base_seed: int,
    template_cfg: MCMCWorkflowConfig | BetaSweepStudyConfig,
    n_jobs: int = 1,
) -> dict[str, Any]:
    checkpoints = _sorted_unique_points(estimation_points)
    rows: list[dict[str, Any]] = []
    jobs = [
        (
            dict(spec),
            int(rep),
            int(base_seed + 100_000 * cfg_idx + 1_000 * rep),
        )
        for cfg_idx, spec in enumerate(config_specs)
        for rep in range(int(repeats))
    ]
    n_workers = _effective_n_jobs(n_jobs, len(jobs))
    executor = _try_make_process_pool(n_workers) if n_workers > 1 else None

    if executor is None:
        for spec, rep, rep_seed in jobs:
            rows.extend(
                _named_mcmc_replicate_worker(
                    problem=problem,
                    exact_p=exact_p,
                    checkpoints=checkpoints,
                    sigma_t=float(sigma_t),
                    template_cfg=template_cfg,
                    config_spec=spec,
                    rep=rep,
                    rep_seed=rep_seed,
                )
            )
    else:
        with executor:
            futures = [
                executor.submit(
                    _named_mcmc_replicate_worker,
                    problem=problem,
                    exact_p=exact_p,
                    checkpoints=checkpoints,
                    sigma_t=float(sigma_t),
                    template_cfg=template_cfg,
                    config_spec=spec,
                    rep=rep,
                    rep_seed=rep_seed,
                )
                for spec, rep, rep_seed in jobs
            ]
            for future in cf.as_completed(futures):
                rows.extend(future.result())

    rows = sorted(rows, key=lambda row: (str(row["label"]), int(row["replicate"]), int(row["checkpoint"])))
    summary = summarize_records(rows, group_fields=("checkpoint", "label"))
    return {
        "records": rows,
        "summary": summary,
        "settings": {
            "estimation_points": list(checkpoints),
            "repeats": int(repeats),
            "base_seed": int(base_seed),
            "sigma_t": float(sigma_t),
            "configs": list(config_specs),
        },
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
    init_state: np.ndarray | None = None,
    tilt_mode: str,
    n_swap_pairs: int,
    checkpoint_steps: tuple[int, ...],
) -> dict[str, Any]:
    if init_state is not None:
        y = problem.validate_labels(np.asarray(init_state, dtype=np.int8)).copy()
    elif init == "observed":
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
        "final_state": y.copy(),
    }


def _run_single_chain_full_trace_from_seed(
    *,
    problem: PermutationTestProblem,
    seed: int,
    beta: float,
    sigma_t: float,
    n_steps: int,
    init: str,
    init_state: np.ndarray | None,
    tilt_mode: str,
    n_swap_pairs: int,
    checkpoint_steps: tuple[int, ...],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    return _run_single_chain_full_trace(
        problem,
        rng,
        beta=beta,
        sigma_t=sigma_t,
        n_steps=n_steps,
        init=init,
        init_state=init_state,
        tilt_mode=tilt_mode,
        n_swap_pairs=n_swap_pairs,
        checkpoint_steps=checkpoint_steps,
    )


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
    init_states: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    reuse_init_state: bool = False,
    scan_sample_pack: dict[str, Any] | None = None,
    estimator_variant: str = "production_only",
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

    n_swap_pairs = resolve_n_swap_pairs(
        problem.n_treated,
        problem.n_control,
        proposal_size=cfg.proposal_size,
    )

    seed_seq = np.random.SeedSequence(seed)
    spawned = seed_seq.spawn(cfg.chains)
    normalized_init_states: list[np.ndarray | None]
    if init_states is None:
        normalized_init_states = [None] * int(cfg.chains)
    else:
        if len(init_states) != int(cfg.chains):
            raise ValueError("init_states length must match cfg.chains.")
        normalized_init_states = [np.asarray(state, dtype=np.int8).copy() for state in init_states]
    chain_jobs = []
    for chain_idx, ss in enumerate(spawned):
        chain_seed = int(ss.generate_state(1, dtype=np.uint64)[0])
        chain_jobs.append(
            {
                "chain_idx": int(chain_idx),
                "problem": problem,
                "seed": int(chain_seed),
                "beta": float(beta),
                "sigma_t": float(sigma_t),
                "n_steps": int(max_steps_per_chain),
                "init": "random" if normalized_init_states[chain_idx] is None else "observed",
                "init_state": normalized_init_states[chain_idx],
                "tilt_mode": str(cfg.tilt_mode),
                "n_swap_pairs": int(n_swap_pairs),
                "checkpoint_steps": unique_step_checkpoints,
            }
        )

    chain_n_jobs = max(1, min(int(getattr(cfg, "chain_n_jobs", 1)), len(chain_jobs)))
    executor = _try_make_process_pool(chain_n_jobs) if chain_n_jobs > 1 else None
    traces: list[dict[str, Any]] = [None] * len(chain_jobs)  # type: ignore[list-item]
    if executor is None:
        for job in chain_jobs:
            traces[int(job["chain_idx"])] = _run_single_chain_full_trace_from_seed(
                problem=job["problem"],
                seed=int(job["seed"]),
                beta=float(job["beta"]),
                sigma_t=float(job["sigma_t"]),
                n_steps=int(job["n_steps"]),
                init=str(job["init"]),
                init_state=job["init_state"],
                tilt_mode=str(job["tilt_mode"]),
                n_swap_pairs=int(job["n_swap_pairs"]),
                checkpoint_steps=tuple(int(v) for v in job["checkpoint_steps"]),
            )
    else:
        with executor:
            futures = {
                executor.submit(
                    _run_single_chain_full_trace_from_seed,
                    problem=job["problem"],
                    seed=int(job["seed"]),
                    beta=float(job["beta"]),
                    sigma_t=float(job["sigma_t"]),
                    n_steps=int(job["n_steps"]),
                    init=str(job["init"]),
                    init_state=job["init_state"],
                    tilt_mode=str(job["tilt_mode"]),
                    n_swap_pairs=int(job["n_swap_pairs"]),
                    checkpoint_steps=tuple(int(v) for v in job["checkpoint_steps"]),
                ): int(job["chain_idx"])
                for job in chain_jobs
            }
            for future in cf.as_completed(futures):
                traces[futures[future]] = future.result()

    rows: list[dict[str, Any]] = []
    for checkpoint, report_checkpoint in zip(checkpoints, reported_checkpoints):
        steps_per_chain = steps_per_checkpoint[int(checkpoint)]
        burn_in = 0 if reuse_init_state else _burn_in(steps_per_chain, cfg.burn_in_fraction)
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
        if (
            str(estimator_variant) in {"selected_scan_plus_production", "all_scan_plus_production"}
            and scan_sample_pack is not None
            and int(scan_sample_pack.get("n_weighted_samples", 0)) > 0
        ):
            row = _pooled_scan_plus_production_row(
                exact_p=float(exact_p),
                base_row=row,
                scan_sample_pack=scan_sample_pack,
                production_payload=_production_prefix_payload(
                    traces=traces,
                    steps_per_chain=steps_per_chain,
                    burn_in=burn_in,
                    thin=int(cfg.thin),
                    beta=float(beta),
                ),
                estimator_variant=str(estimator_variant),
            )
        else:
            row["estimator_variant"] = "production_only"
            row["scan_n_weighted_samples"] = 0
            row["production_n_weighted_samples"] = int(row.get("n_weighted_samples", 0))
            row["pooled_scan_batch_count"] = 0
        row["mcmc_chain_budget"] = int(checkpoint)
        row["mcmc_reported_budget"] = int(report_checkpoint)
        row["state_reused_init"] = int(reuse_init_state)
        row["burn_in"] = int(burn_in)
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

    n_swap_pairs = resolve_n_swap_pairs(
        problem.n_treated,
        problem.n_control,
        proposal_size=cfg.proposal_size,
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
        estimate_no_correction = _pvalue_estimate_no_empty_bin_correction(
            theta=theta_at_checkpoint[int(checkpoint)],
            target=target,
            tail_bin_index=tail_bin_index,
        )
        rows.append(
            _annotate_error_fields(
                {
                    "method": "samc",
                    "checkpoint": int(checkpoint),
                    "estimate": float(estimate),
                    "samc_estimate_no_empty_bin_correction": float(estimate_no_correction),
                    "samc_empty_bin_correction_delta": float(estimate - estimate_no_correction),
                    "samc_empty_bin_correction_ratio": _correction_ratio(
                        corrected=float(estimate),
                        uncorrected=float(estimate_no_correction),
                    ),
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
    worker_cfg = replace(
        mcmc_cfg,
        proposal_size=beta_workflow.get("proposal_size_used", mcmc_cfg.proposal_size),
    )
    rows = _run_mcmc_cumulative_checkpoints(
        problem,
        exact_p,
        checkpoints=mcmc_chain_checkpoints,
        reported_checkpoints=checkpoints,
        beta=float(beta_workflow["beta_used"]),
        sigma_t=float(beta_workflow["sigma_t"]),
        cfg=worker_cfg,
        seed=rep_seed + 1,
        init_states=beta_workflow.get("production_init_states"),
        reuse_init_state=beta_workflow.get("production_init_states") is not None,
        scan_sample_pack=beta_workflow.get("scan_sample_pack"),
        estimator_variant=str(beta_workflow.get("production_estimator_variant", "production_only")),
    )
    for row in rows:
        row["scenario"] = scenario_key
        row["scenario_display"] = scenario_display
        row["replicate"] = int(rep)
        row["beta_selection_budget"] = int(beta_selection_budget)
        row["eval_incl_tuning"] = float(int(row["mcmc_chain_budget"]) + beta_selection_budget)
        row["proposal_size"] = worker_cfg.proposal_size
        row["n_swap_pairs"] = int(
            resolve_n_swap_pairs(
                problem.n_treated,
                problem.n_control,
                proposal_size=worker_cfg.proposal_size,
            )
        )
    return rows


def _hard_step_cross_replicate_worker(
    *,
    scenario_key: str,
    scenario_display: str,
    problem: PermutationTestProblem,
    exact_p: float,
    checkpoints: tuple[int, ...],
    hard_step_workflow: dict[str, Any],
    mcmc_cfg: MCMCWorkflowConfig,
    rep: int,
    rep_seed: int,
) -> list[dict[str, Any]]:
    worker_cfg = replace(
        mcmc_cfg,
        proposal_size=hard_step_workflow["proposal_size"],
        tilt_mode="step",
        local_scan_enabled=False,
    )
    rows = _run_mcmc_cumulative_checkpoints(
        problem,
        exact_p,
        checkpoints=checkpoints,
        beta=float(hard_step_workflow["beta"]),
        sigma_t=1.0,
        cfg=worker_cfg,
        seed=rep_seed + 3,
    )
    for row in rows:
        row["method"] = "hard_step"
        row["scenario"] = scenario_key
        row["scenario_display"] = scenario_display
        row["replicate"] = int(rep)
        row["beta_selection_budget"] = 0
        row["eval_incl_tuning"] = float(row["eval_excl_tuning"])
        row["proposal_size"] = worker_cfg.proposal_size
        row["n_swap_pairs"] = int(hard_step_workflow["n_swap_pairs"])
        row["q_tilt_tail_share"] = float(row.get("tail_share_raw", np.nan))
        row["hard_step_reference_p0"] = float(hard_step_workflow["reference_p0"])
        row["hard_step_reference_p0_source"] = str(hard_step_workflow["reference_p0_source"])
        row["hard_step_target_tail_mass"] = float(hard_step_workflow["target_tail_mass"])
        row["hard_step_r"] = float(hard_step_workflow["r"])
        row["hard_step_beta"] = float(hard_step_workflow["beta"])
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


def known_threshold_p0_for_scenario(
    scenario: LoadedScenario,
    *,
    fallback_to_exact: bool = False,
) -> tuple[float, str]:
    for source, container in (
        ("scenario_extra_known_significance_threshold", scenario.extra),
        ("scenario_portfolio_known_significance_threshold", scenario.portfolio),
    ):
        if not isinstance(container, dict):
            continue
        threshold = _valid_probability_or_none(container.get("known_significance_threshold"))
        if threshold is None:
            threshold = _valid_probability_or_none(container.get("canonical_threshold_p0"))
        if threshold is not None:
            return float(threshold), source
    if fallback_to_exact:
        return float(scenario.exact_p), "exact_p_fallback"
    raise ValueError(
        f"Scenario '{scenario.key}' has no known significance threshold; "
        "set fallback_to_exact=True only for oracle/debug runs."
    )


def _threshold_grid_common_annotations(
    *,
    scenario: LoadedScenario,
    family: str,
    threshold_band: str,
    swap_fraction: float | int,
    gamma: float | None,
    p0_reference: float,
    p0_reference_source: str,
    q_target: float | None,
    seed: int,
    n_swap_pairs: int,
) -> dict[str, Any]:
    gamma_value = float(gamma) if gamma is not None else np.nan
    gamma_label = f"{float(gamma):.12g}" if gamma is not None else "none"
    return {
        "scenario": scenario.key,
        "scenario_display": scenario.description,
        "family": str(family),
        "threshold_band": str(threshold_band),
        "application_setting_key": scenario.extra.get("application_setting_key"),
        "known_significance_threshold": float(p0_reference),
        "p0_reference": float(p0_reference),
        "p0_reference_source": str(p0_reference_source),
        "p_over_p0": float(scenario.exact_p / p0_reference),
        "swap_fraction": float(swap_fraction),
        "proposal_size": swap_fraction,
        "n_swap_pairs": int(n_swap_pairs),
        "gamma": gamma_value,
        "gamma_label": gamma_label,
        "q_target": float(q_target) if q_target is not None else np.nan,
        "target_tail_mass": float(q_target) if q_target is not None else np.nan,
        "replicate": 0,
        "seed": int(seed),
    }


def _threshold_grid_mcmcis_scenario_worker(
    *,
    scenario: LoadedScenario,
    family: str,
    threshold_band: str,
    checkpoints: tuple[int, ...],
    gamma: float,
    swap_fraction: float | int,
    mcmc_cfg: MCMCWorkflowConfig,
    seed: int,
) -> list[dict[str, Any]]:
    p0_reference, p0_source = known_threshold_p0_for_scenario(scenario, fallback_to_exact=False)
    worker_cfg = replace(
        mcmc_cfg,
        use_true_p0_for_q_target=False,
        p0_guess=float(p0_reference),
        d_alpha=float(gamma),
        proposal_size=swap_fraction,
        tilt_mode="smooth_hinge",
        local_scan_enabled=False,
        hard_step_enabled=False,
    )
    init_payload = build_beta_initialization(
        scenario.problem,
        float(p0_reference),
        worker_cfg,
        seed=int(seed),
        p0_reference=float(p0_reference),
    )
    beta_selection_budget = int(worker_cfg.pilot_samples)
    checkpoint_pairs = [
        (int(cp), int(cp) - int(beta_selection_budget))
        for cp in checkpoints
        if int(cp) - int(beta_selection_budget) > 0
    ]
    if not checkpoint_pairs:
        return []

    reported_checkpoints = tuple(int(cp) for cp, _ in checkpoint_pairs)
    chain_checkpoints = tuple(int(chain_cp) for _, chain_cp in checkpoint_pairs)
    n_swap_pairs = resolve_n_swap_pairs(
        scenario.problem.n_treated,
        scenario.problem.n_control,
        proposal_size=swap_fraction,
    )
    rows = _run_mcmc_cumulative_checkpoints(
        scenario.problem,
        scenario.exact_p,
        checkpoints=chain_checkpoints,
        reported_checkpoints=reported_checkpoints,
        beta=float(init_payload["beta0_laplace"]),
        sigma_t=float(init_payload["sigma_t"]),
        cfg=worker_cfg,
        seed=int(seed) + 1,
    )
    annotations = _threshold_grid_common_annotations(
        scenario=scenario,
        family=family,
        threshold_band=threshold_band,
        swap_fraction=swap_fraction,
        gamma=float(gamma),
        p0_reference=float(p0_reference),
        p0_reference_source=p0_source,
        q_target=float(init_payload["q_target"]),
        seed=int(seed),
        n_swap_pairs=int(n_swap_pairs),
    )
    for row in rows:
        row.update(annotations)
        row["method"] = "mcmc_is_no_oracle"
        row["label"] = f"mcmc_is_no_oracle__swap_{float(swap_fraction):.2f}__gamma_{float(gamma):.12g}"
        row["beta_selection_budget"] = int(beta_selection_budget)
        row["eval_incl_tuning"] = float(row["eval_excl_tuning"] + beta_selection_budget)
        row["beta_init_pilot_samples"] = int(init_payload["pilot_eval_total"])
        row["beta_init_wall_time_sec"] = float(init_payload["pilot_wall_time_sec"])
        row["beta0_formula"] = float(init_payload["beta0_formula"])
        row["beta0_laplace"] = float(init_payload["beta0_laplace"])
        row["beta_used"] = float(init_payload["beta0_laplace"])
        row["p0_for_qtarget"] = float(init_payload["p0_for_qtarget"])
        row["q_tilt_tail_share"] = float(row.get("tail_share_raw", np.nan))
        row["oracle_mode"] = "known_threshold_p0"
    return rows


def _threshold_grid_hard_step_scenario_worker(
    *,
    scenario: LoadedScenario,
    family: str,
    threshold_band: str,
    checkpoints: tuple[int, ...],
    gamma: float,
    swap_fraction: float | int,
    mcmc_cfg: MCMCWorkflowConfig,
    seed: int,
) -> list[dict[str, Any]]:
    p0_reference, p0_source = known_threshold_p0_for_scenario(scenario, fallback_to_exact=False)
    q_target = float(float(p0_reference) ** float(gamma))
    r = hard_step_r_for_target_tail_mass(float(p0_reference), q_target)
    beta = hard_step_beta_for_target_tail_mass(float(p0_reference), q_target)
    worker_cfg = replace(
        mcmc_cfg,
        use_true_p0_for_q_target=False,
        p0_guess=float(p0_reference),
        d_alpha=float(gamma),
        proposal_size=swap_fraction,
        tilt_mode="step",
        local_scan_enabled=False,
        hard_step_enabled=True,
    )
    n_swap_pairs = resolve_n_swap_pairs(
        scenario.problem.n_treated,
        scenario.problem.n_control,
        proposal_size=swap_fraction,
    )
    rows = _run_mcmc_cumulative_checkpoints(
        scenario.problem,
        scenario.exact_p,
        checkpoints=checkpoints,
        beta=float(beta),
        sigma_t=1.0,
        cfg=worker_cfg,
        seed=int(seed) + 2,
    )
    annotations = _threshold_grid_common_annotations(
        scenario=scenario,
        family=family,
        threshold_band=threshold_band,
        swap_fraction=swap_fraction,
        gamma=float(gamma),
        p0_reference=float(p0_reference),
        p0_reference_source=p0_source,
        q_target=float(q_target),
        seed=int(seed),
        n_swap_pairs=int(n_swap_pairs),
    )
    for row in rows:
        row.update(annotations)
        row["method"] = "hard_step"
        row["label"] = f"hard_step__swap_{float(swap_fraction):.2f}__gamma_{float(gamma):.12g}"
        row["beta_selection_budget"] = 0
        row["eval_incl_tuning"] = float(row["eval_excl_tuning"])
        row["hard_step_reference_p0"] = float(p0_reference)
        row["hard_step_reference_p0_source"] = str(p0_source)
        row["hard_step_target_tail_mass"] = float(q_target)
        row["hard_step_r"] = float(r)
        row["hard_step_beta"] = float(beta)
        row["q_tilt_tail_share"] = float(row.get("tail_share_raw", np.nan))
        row["oracle_mode"] = "known_threshold_p0"
    return rows


def _threshold_grid_samc_scenario_worker(
    *,
    scenario: LoadedScenario,
    family: str,
    threshold_band: str,
    checkpoints: tuple[int, ...],
    swap_fraction: float | int,
    samc_cfg: SAMCWorkflowConfig,
    seed: int,
) -> list[dict[str, Any]]:
    p0_reference, p0_source = known_threshold_p0_for_scenario(scenario, fallback_to_exact=False)
    worker_cfg = replace(samc_cfg, proposal_size=swap_fraction)
    n_swap_pairs = resolve_n_swap_pairs(
        scenario.problem.n_treated,
        scenario.problem.n_control,
        proposal_size=swap_fraction,
    )
    samc_setup = tune_samc_setup(scenario.problem, worker_cfg, seed=int(seed))
    rows = _run_samc_cumulative_checkpoints(
        scenario.problem,
        scenario.exact_p,
        checkpoints=checkpoints,
        samc_setup=samc_setup,
        cfg=worker_cfg,
        seed=int(seed) + 3,
    )
    annotations = _threshold_grid_common_annotations(
        scenario=scenario,
        family=family,
        threshold_band=threshold_band,
        swap_fraction=swap_fraction,
        gamma=None,
        p0_reference=float(p0_reference),
        p0_reference_source=p0_source,
        q_target=None,
        seed=int(seed),
        n_swap_pairs=int(n_swap_pairs),
    )
    for row in rows:
        row.update(annotations)
        row["method"] = "samc"
        row["label"] = f"samc__swap_{float(swap_fraction):.2f}"
        row["beta_selection_budget"] = 0
        row["eval_incl_tuning"] = float(row["eval_excl_tuning"])
        row["samc_setup_eval_total"] = int(worker_cfg.lambda_min_pilot)
        row["eval_incl_setup"] = float(row["eval_excl_tuning"] + int(worker_cfg.lambda_min_pilot))
        row["samc_lambda_min"] = float(samc_setup["lambda_min"])
        row["samc_n_bins"] = int(worker_cfg.n_bins)
        row["oracle_mode"] = "not_applicable"
    return rows


def _threshold_grid_scenario_worker(job: dict[str, Any]) -> list[dict[str, Any]]:
    method = str(job["method"])
    if method == "samc":
        return _threshold_grid_samc_scenario_worker(
            scenario=job["scenario"],
            family=str(job["family"]),
            threshold_band=str(job["threshold_band"]),
            checkpoints=tuple(int(v) for v in job["checkpoints"]),
            swap_fraction=job["swap_fraction"],
            samc_cfg=job["samc_cfg"],
            seed=int(job["seed"]),
        )
    if method == "mcmc_is":
        return _threshold_grid_mcmcis_scenario_worker(
            scenario=job["scenario"],
            family=str(job["family"]),
            threshold_band=str(job["threshold_band"]),
            checkpoints=tuple(int(v) for v in job["checkpoints"]),
            gamma=float(job["gamma"]),
            swap_fraction=job["swap_fraction"],
            mcmc_cfg=job["mcmc_cfg"],
            seed=int(job["seed"]),
        )
    if method == "hard_step":
        return _threshold_grid_hard_step_scenario_worker(
            scenario=job["scenario"],
            family=str(job["family"]),
            threshold_band=str(job["threshold_band"]),
            checkpoints=tuple(int(v) for v in job["checkpoints"]),
            gamma=float(job["gamma"]),
            swap_fraction=job["swap_fraction"],
            mcmc_cfg=job["mcmc_cfg"],
            seed=int(job["seed"]),
        )
    raise ValueError(f"Unknown threshold-grid method: {method}")


def _read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default))
            handle.write("\n")


def _threshold_grid_record_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("threshold_band", "")),
        str(row.get("family", "")),
        float(row.get("swap_fraction", np.nan)),
        str(row.get("method", "")),
        str(row.get("gamma_label", "")),
        str(row.get("scenario", "")),
        int(row.get("checkpoint", 0)),
    )


def run_threshold_grid_method_block(
    scenarios: Iterable[LoadedScenario],
    *,
    method: str,
    family: str,
    threshold_band: str,
    checkpoints: Iterable[int],
    swap_fraction: float | int,
    gamma: float | None = None,
    mcmc_cfg: MCMCWorkflowConfig | None = None,
    samc_cfg: SAMCWorkflowConfig | None = None,
    base_seed: int = 246_810,
    n_jobs: int = 1,
    output_dir: Path | None = None,
    resume: bool = True,
    block_label: str | None = None,
) -> dict[str, Any]:
    method = str(method)
    if method not in {"samc", "mcmc_is", "hard_step"}:
        raise ValueError("method must be one of {'samc', 'mcmc_is', 'hard_step'}.")
    if method in {"mcmc_is", "hard_step"} and gamma is None:
        raise ValueError("gamma is required for MCMC-IS and hard-step blocks.")

    scenario_list = list(scenarios)
    checkpoints_tuple = _sorted_unique_points(checkpoints)
    mcmc_cfg = MCMCWorkflowConfig() if mcmc_cfg is None else mcmc_cfg
    samc_cfg = SAMCWorkflowConfig() if samc_cfg is None else samc_cfg
    output_dir = None if output_dir is None else Path(output_dir)
    final_records_path = output_dir / "block_records.jsonl" if output_dir is not None else None
    partial_records_path = output_dir / "block_records.partial.jsonl" if output_dir is not None else None
    summary_path = output_dir / "block_summary.json" if output_dir is not None else None
    metadata_path = output_dir / "block_metadata.json" if output_dir is not None else None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume and final_records_path is not None and final_records_path.exists():
            records = sorted(read_jsonl(final_records_path), key=_threshold_grid_record_sort_key)
            summary = summarize_records(
                records,
                group_fields=("threshold_band", "family", "swap_fraction", "method", "gamma_label", "checkpoint"),
            )
            metadata = read_json(metadata_path) if metadata_path is not None and metadata_path.exists() else {}
            metadata.update(
                {
                    "status": "loaded_complete",
                    "n_records": int(len(records)),
                    "n_scenarios": int(len(scenario_list)),
                }
            )
            return {
                "records": records,
                "summary": summary,
                "metadata": metadata,
                "output_dir": output_dir,
                "loaded_from": final_records_path,
            }
        if partial_records_path is not None and (not resume or not partial_records_path.exists()):
            partial_records_path.write_text("", encoding="utf-8")

    expected_rows_per_scenario = len(checkpoints_tuple)
    if method == "mcmc_is":
        expected_rows_per_scenario = sum(1 for cp in checkpoints_tuple if int(cp) > int(mcmc_cfg.pilot_samples))

    partial_rows: list[dict[str, Any]] = []
    completed_scenario_keys: set[str] = set()
    if resume and partial_records_path is not None and partial_records_path.exists():
        partial_rows = _read_jsonl_lenient(partial_records_path)
        counts_by_scenario: dict[str, int] = {}
        for row in partial_rows:
            key = str(row.get("scenario", ""))
            if key:
                counts_by_scenario[key] = counts_by_scenario.get(key, 0) + 1
        completed_scenario_keys = {
            key for key, count in counts_by_scenario.items() if count >= expected_rows_per_scenario
        }

    pending_scenarios = [s for s in scenario_list if s.key not in completed_scenario_keys]
    records = [row for row in partial_rows if str(row.get("scenario", "")) in completed_scenario_keys]
    metadata = {
        "status": "running",
        "block_label": block_label,
        "method": method,
        "family": family,
        "threshold_band": threshold_band,
        "swap_fraction": float(swap_fraction),
        "gamma": float(gamma) if gamma is not None else None,
        "checkpoints": list(checkpoints_tuple),
        "max_budget": int(checkpoints_tuple[-1]),
        "n_scenarios": int(len(scenario_list)),
        "n_pending_scenarios": int(len(pending_scenarios)),
        "n_completed_scenarios_from_partial": int(len(completed_scenario_keys)),
        "scenario_keys": [s.key for s in scenario_list],
        "base_seed": int(base_seed),
        "n_jobs_requested": int(n_jobs),
        "mcmc_config": mcmc_cfg,
        "samc_config": samc_cfg,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata_path is not None:
        write_json(metadata_path, metadata)

    jobs = [
        {
            "scenario": scenario,
            "method": method,
            "family": family,
            "threshold_band": threshold_band,
            "checkpoints": checkpoints_tuple,
            "gamma": gamma,
            "swap_fraction": swap_fraction,
            "mcmc_cfg": mcmc_cfg,
            "samc_cfg": samc_cfg,
            "seed": int(base_seed) + 10_000 * scenario_idx,
        }
        for scenario_idx, scenario in enumerate(pending_scenarios)
    ]
    effective_jobs = _effective_n_jobs(int(n_jobs), len(jobs))
    metadata["n_jobs_effective"] = int(effective_jobs)

    executor = _try_make_process_pool(effective_jobs) if effective_jobs > 1 else None
    t_start = time.perf_counter()
    if executor is None:
        for job in jobs:
            rows = _threshold_grid_scenario_worker(job)
            records.extend(rows)
            if partial_records_path is not None:
                _append_jsonl(partial_records_path, rows)
    else:
        with executor:
            futures = {executor.submit(_threshold_grid_scenario_worker, job): job for job in jobs}
            for future in cf.as_completed(futures):
                rows = future.result()
                records.extend(rows)
                if partial_records_path is not None:
                    _append_jsonl(partial_records_path, rows)

    records = sorted(records, key=_threshold_grid_record_sort_key)
    summary = summarize_records(
        records,
        group_fields=("threshold_band", "family", "swap_fraction", "method", "gamma_label", "checkpoint"),
    )
    metadata.update(
        {
            "status": "complete",
            "n_records": int(len(records)),
            "n_summary_rows": int(len(summary)),
            "wall_time_sec": float(time.perf_counter() - t_start),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if final_records_path is not None:
        write_jsonl(final_records_path, records)
    if summary_path is not None:
        write_json(summary_path, summary)
    if metadata_path is not None:
        write_json(metadata_path, metadata)
    return {
        "records": records,
        "summary": summary,
        "metadata": metadata,
        "output_dir": output_dir,
        "loaded_from": None,
    }


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
    hard_step_workflow = build_hard_step_workflow(scenario, mcmc_cfg)
    beta_selection_budget = int(beta_workflow["beta_selection_eval_total"])
    mcmc_checkpoint_pairs = [
        (int(cp), int(cp - beta_selection_budget))
        for cp in checkpoints
        if int(cp - beta_selection_budget) > 0
    ]
    mcmc_reported_checkpoints = tuple(int(cp) for cp, _ in mcmc_checkpoint_pairs)
    mcmc_chain_checkpoints = tuple(int(chain_cp) for _, chain_cp in mcmc_checkpoint_pairs)

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
        if mcmc_chain_checkpoints:
            for rep, rep_seed in repeat_jobs:
                records.extend(
                    _mcmc_cross_replicate_worker(
                        scenario_key=scenario.key,
                        scenario_display=scenario.description,
                        problem=scenario.problem,
                        exact_p=scenario.exact_p,
                        checkpoints=mcmc_reported_checkpoints,
                        mcmc_chain_checkpoints=mcmc_chain_checkpoints,
                        beta_workflow=beta_workflow,
                        beta_selection_budget=beta_selection_budget,
                        mcmc_cfg=mcmc_cfg,
                        rep=rep,
                        rep_seed=rep_seed,
                    )
                )
        if bool(hard_step_workflow.get("enabled", False)):
            for rep, rep_seed in repeat_jobs:
                records.extend(
                    _hard_step_cross_replicate_worker(
                        scenario_key=scenario.key,
                        scenario_display=scenario.description,
                        problem=scenario.problem,
                        exact_p=scenario.exact_p,
                        checkpoints=checkpoints,
                        hard_step_workflow=hard_step_workflow,
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

            if mcmc_chain_checkpoints:
                futures = [
                    executor.submit(
                        _mcmc_cross_replicate_worker,
                        scenario_key=scenario.key,
                        scenario_display=scenario.description,
                        problem=scenario.problem,
                        exact_p=scenario.exact_p,
                        checkpoints=mcmc_reported_checkpoints,
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

            if bool(hard_step_workflow.get("enabled", False)):
                futures = [
                    executor.submit(
                        _hard_step_cross_replicate_worker,
                        scenario_key=scenario.key,
                        scenario_display=scenario.description,
                        problem=scenario.problem,
                        exact_p=scenario.exact_p,
                        checkpoints=checkpoints,
                        hard_step_workflow=hard_step_workflow,
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
        "scenario_portfolio": dict(scenario.portfolio),
        "exact_p": float(scenario.exact_p),
        "exact_method": scenario.exact_method,
        "exact_tail_hits": int(scenario.exact_tail_hits),
        "exact_n_perm": int(scenario.exact_n_perm),
        "estimation_points": list(checkpoints),
        "beta_workflow": beta_workflow,
        "mcmc_beta_selection_budget": beta_selection_budget,
        "mcmc_reported_checkpoints": list(mcmc_reported_checkpoints),
        "hard_step_workflow": hard_step_workflow,
        "samc_setup": {
            "lambda_min": float(samc_setup["lambda_min"]),
            "bin_edges": np.asarray(samc_setup["bin_edges"], dtype=float),
        },
        "records": records,
        "summary": summary,
        "iid_density_summary": density_summary,
    }


def plot_named_method_max_budget(
    records: list[dict[str, Any]],
    *,
    scenario_name: str,
    exact_p: float,
    max_budget: int,
    method_order: list[str] | tuple[str, ...],
    method_labels: dict[str, str] | None = None,
    method_colors: dict[str, str] | None = None,
    scenario_key: str | None = None,
    n_control: int | None = None,
    n_treated: int | None = None,
    known_significance_threshold: float | None = None,
    save_path: Path | None = None,
) -> None:
    rows = [row for row in records if int(row["checkpoint"]) == int(max_budget)]
    methods = [str(method) for method in method_order]
    n_runs = max((sum(1 for row in rows if _record_group_label(row) == method) for method in methods), default=0)
    if method_labels is None:
        method_labels = {}
    if method_colors is None:
        method_colors = {}
    fallback_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(methods), 1)))
    labels = [str(method_labels.get(method, method)) for method in methods]
    colors = [
        str(method_colors.get(method, matplotlib.colors.to_hex(fallback_colors[idx])))
        for idx, method in enumerate(methods)
    ]

    est_data = []
    abs_error_data = []
    for method in methods:
        sub = [row for row in rows if _record_group_label(row) == method]
        est = np.asarray([row["estimate"] for row in sub], dtype=float)
        rse = np.asarray([row["root_squared_error"] for row in sub], dtype=float)
        est_data.append(_finite_for_plot(est))
        abs_error_data.append(_finite_for_plot(rse))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    _styled_boxplot(axes[0], est_data, labels=labels, colors=colors)
    _overlay_boxplot_points(axes[0], est_data, colors=colors)
    axes[0].axhline(exact_p, color="#000000", linestyle="--", linewidth=1.25, zorder=2.0)
    _set_linear_ylim(axes[0], est_data, include_values=[exact_p], anchor_zero=False)
    axes[0].set_title("Estimate", fontsize=12, pad=10)
    axes[0].set_ylabel(r"$\hat{p}$")
    if _value_in_ylim(axes[0], exact_p):
        axes[0].legend(
            handles=[Line2D([0], [0], color="#000000", linestyle="--", linewidth=1.25, label="Exact p")],
            frameon=False,
            fontsize=9.5,
            loc="upper right",
        )
    _style_article_axis(axes[0], grid_axis="y")

    _styled_boxplot(axes[1], abs_error_data, labels=labels, colors=colors)
    _overlay_boxplot_points(axes[1], abs_error_data, colors=colors)
    _set_linear_ylim(axes[1], abs_error_data, anchor_zero=True)
    axes[1].set_title("Absolute error", fontsize=12, pad=10)
    axes[1].set_ylabel(r"$|\hat{p} - p|$")
    _style_article_axis(axes[1], grid_axis="y")

    fig.suptitle(
        _compact_plot_title(
            scenario_name,
            scenario_key=scenario_key,
            n_control=n_control,
            n_treated=n_treated,
            exact_p=exact_p,
            known_significance_threshold=known_significance_threshold,
            n_runs=n_runs,
        ),
        fontsize=14,
        fontweight="semibold",
        y=0.99,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cross_method_max_budget(
    records: list[dict[str, Any]],
    *,
    scenario_name: str,
    scenario_key: str | None = None,
    exact_p: float,
    max_budget: int,
    n_control: int | None = None,
    n_treated: int | None = None,
    beta_workflow: dict[str, Any] | None = None,
    save_path: Path | None = None,
) -> None:
    plot_named_method_max_budget(
        records,
        scenario_name=scenario_name,
        scenario_key=scenario_key,
        exact_p=exact_p,
        max_budget=max_budget,
        method_order=_available_cross_method_order(records),
        method_labels=_CROSS_METHOD_LABELS,
        method_colors=_CROSS_METHOD_COLORS,
        n_control=n_control,
        n_treated=n_treated,
        save_path=save_path,
    )


def plot_named_method_convergence(
    summary: list[dict[str, Any]],
    *,
    scenario_name: str,
    exact_p: float,
    method_order: list[str] | tuple[str, ...],
    method_labels: dict[str, str] | None = None,
    method_colors: dict[str, str] | None = None,
    scenario_key: str | None = None,
    n_control: int | None = None,
    n_treated: int | None = None,
    known_significance_threshold: float | None = None,
    mcmc_beta_selection_budget: int = 0,
    x_label: str = "Total budget",
    x_scale: str = "linear",
    estimate_field: str = "mean_estimate",
    estimate_title: str = "Mean estimate",
    estimate_ylabel: str = r"mean $\hat{p}$",
    save_path: Path | None = None,
) -> None:
    methods = [str(method) for method in method_order]
    n_runs = max(
        (
            max(
                (int(row.get("n_runs", 0)) for row in summary if _record_group_label(row) == method),
                default=0,
            )
            for method in methods
        ),
        default=0,
    )
    if method_labels is None:
        method_labels = {}
    if method_colors is None:
        method_colors = {}
    budgets = sorted({int(row["checkpoint"]) for row in summary})
    fallback_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(methods), 1)))
    marker_cycle = ["o", "s", "D", "^", "P", "v"]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    for idx, method in enumerate(methods):
        sub = sorted([row for row in summary if _record_group_label(row) == method], key=lambda row: row["checkpoint"])
        x = np.asarray([row["checkpoint"] for row in sub], dtype=float)
        estimate_values = np.asarray([row[estimate_field] for row in sub], dtype=float)
        rmse = np.asarray([row["rmse"] for row in sub], dtype=float)
        color = str(method_colors.get(method, matplotlib.colors.to_hex(fallback_colors[idx])))
        label = str(method_labels.get(method, method))

        plot_kwargs = {
            "marker": marker_cycle[idx % len(marker_cycle)],
            "linestyle": "-",
            "markersize": 5.9,
            "linewidth": 2.15,
            "markeredgecolor": "white",
            "markeredgewidth": 0.85,
            "color": color,
            "alpha": 0.98,
            "zorder": 3.0 + 0.1 * idx,
        }
        axes[0].plot(x, estimate_values, label=label, **plot_kwargs)
        axes[1].plot(x, rmse, **plot_kwargs)

    axes[0].axhline(exact_p, color="#000000", linestyle="--", linewidth=1.25, zorder=2.0)
    if np.isfinite(float(exact_p)) and float(exact_p) > 0.0:
        rmse_ref_10 = 0.10 * float(exact_p)
        rmse_ref_05 = 0.05 * float(exact_p)
        axes[1].axhline(rmse_ref_10, color="#7a6f8f", linestyle=(0, (5.0, 2.8)), linewidth=1.05, zorder=1.8)
        axes[1].axhline(rmse_ref_05, color="#3f4459", linestyle=(0, (5.0, 2.8)), linewidth=1.35, zorder=1.9)
    else:
        rmse_ref_10 = np.nan
        rmse_ref_05 = np.nan
    est_arrays = [np.asarray([row[estimate_field] for row in summary if _record_group_label(row) == method], dtype=float) for method in methods]
    rmse_arrays = [np.asarray([row["rmse"] for row in summary if _record_group_label(row) == method], dtype=float) for method in methods]
    x_span = float(max(budgets) - min(budgets)) if len(budgets) > 1 else float(max(budgets[0], 1))
    for ax in axes:
        ax.set_xscale(str(x_scale))
        _set_budget_ticks(
            ax,
            budgets,
            max_labels=6 if str(x_scale) == "linear" else 4,
            formatter=_format_budget_tick_compact if str(x_scale) == "linear" else _format_scientific_tick,
            label_every=500_000 if str(x_scale) == "linear" else None,
        )
        if str(x_scale) == "linear" and budgets:
            pad = 0.025 * x_span if x_span > 0.0 else 0.15 * float(budgets[0])
            ax.set_xlim(float(min(budgets)) - pad, float(max(budgets)) + pad)
        ax.set_xlabel(str(x_label))
        ax.margins(x=0.05)
    _set_linear_ylim(axes[0], est_arrays, include_values=[exact_p], anchor_zero=False)
    _set_linear_ylim(axes[1], rmse_arrays, include_values=[rmse_ref_10, rmse_ref_05], anchor_zero=True)
    axes[0].set_title(str(estimate_title), fontsize=12.5, pad=10)
    axes[0].set_ylabel(str(estimate_ylabel))
    axes[1].set_title("RMSE", fontsize=12, pad=10)
    axes[1].set_ylabel("RMSE")
    for ax in axes:
        _style_article_axis(ax, grid_axis="both" if str(x_scale) == "linear" else "y")
    est_handles, est_labels = axes[0].get_legend_handles_labels()
    if _value_in_ylim(axes[0], exact_p):
        est_handles.append(Line2D([0], [0], color="#000000", linestyle="--", linewidth=1.25))
        est_labels.append("Exact p")
    axes[0].legend(est_handles, est_labels, frameon=False, fontsize=9.6, loc="upper right")
    rmse_handles: list[Line2D] = []
    rmse_labels: list[str] = []
    if _value_in_ylim(axes[1], rmse_ref_10):
        rmse_handles.append(Line2D([0], [0], color="#7a6f8f", linestyle=(0, (5.0, 2.8)), linewidth=1.05))
        rmse_labels.append("10% of true p")
    if _value_in_ylim(axes[1], rmse_ref_05):
        rmse_handles.append(Line2D([0], [0], color="#3f4459", linestyle=(0, (5.0, 2.8)), linewidth=1.35))
        rmse_labels.append("5% of true p")
    if rmse_handles:
        axes[1].legend(rmse_handles, rmse_labels, frameon=False, fontsize=9.2, loc="upper right")
    fig.suptitle(
        _compact_plot_title(
            scenario_name,
            scenario_key=scenario_key,
            n_control=n_control,
            n_treated=n_treated,
            exact_p=exact_p,
            known_significance_threshold=known_significance_threshold,
            n_runs=n_runs,
        ),
        fontsize=14,
        fontweight="semibold",
        y=0.99,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cross_method_convergence(
    summary: list[dict[str, Any]],
    *,
    scenario_name: str,
    scenario_key: str | None = None,
    exact_p: float,
    n_control: int | None = None,
    n_treated: int | None = None,
    mcmc_beta_selection_budget: int = 0,
    save_path: Path | None = None,
) -> None:
    plot_named_method_convergence(
        summary,
        scenario_name=scenario_name,
        scenario_key=scenario_key,
        exact_p=exact_p,
        method_order=_available_cross_method_order(summary),
        method_labels=_CROSS_METHOD_LABELS,
        method_colors=_CROSS_METHOD_COLORS,
        n_control=n_control,
        n_treated=n_treated,
        mcmc_beta_selection_budget=mcmc_beta_selection_budget,
        x_label="Total budget",
        save_path=save_path,
    )


def summarize_threshold_crossings(
    records: list[dict[str, Any]],
    *,
    known_significance_threshold: float,
    checkpoints: list[int] | tuple[int, ...],
    method_order: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Summarize down-crossings of a fixed threshold in saved trajectories."""
    threshold = float(known_significance_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("known_significance_threshold must be a positive finite value.")
    target_checkpoints = _sorted_unique_points(checkpoints)
    methods = _method_order_from_records(records, method_order)
    traces: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for idx, row in enumerate(records):
        method = _record_group_label(row)
        if method not in methods:
            continue
        run_id = row.get("replicate", row.get("seed", idx))
        traces.setdefault((method, str(run_id)), []).append(row)

    out: list[dict[str, Any]] = []
    for method in methods:
        method_traces = [
            sorted(rows, key=lambda item: int(item["checkpoint"]))
            for (trace_method, _), rows in traces.items()
            if trace_method == method
        ]
        for checkpoint in target_checkpoints:
            n_at_checkpoint = 0
            below_at_checkpoint = 0
            down_into_checkpoint = 0
            cumulative_down_events = 0
            runs_ever_down = 0

            for trace in method_traces:
                finite_points: list[tuple[int, float]] = []
                estimate_at_checkpoint: float | None = None
                for row in trace:
                    cp = int(row["checkpoint"])
                    est = float(row.get("estimate", np.nan))
                    if not np.isfinite(est):
                        continue
                    if cp <= int(checkpoint):
                        finite_points.append((cp, est))
                    if cp == int(checkpoint):
                        estimate_at_checkpoint = est

                if estimate_at_checkpoint is None:
                    continue
                n_at_checkpoint += 1
                if estimate_at_checkpoint <= threshold:
                    below_at_checkpoint += 1

                event_checkpoints: list[int] = []
                finite_points = sorted(finite_points, key=lambda item: item[0])
                for (_, prev_est), (curr_cp, curr_est) in zip(finite_points, finite_points[1:]):
                    if prev_est > threshold and curr_est <= threshold:
                        event_checkpoints.append(int(curr_cp))
                down_into_checkpoint += sum(1 for cp in event_checkpoints if cp == int(checkpoint))
                cumulative_down_events += len(event_checkpoints)
                if event_checkpoints:
                    runs_ever_down += 1

            out.append(
                {
                    "method": method,
                    "checkpoint": int(checkpoint),
                    "known_significance_threshold": threshold,
                    "n_runs_at_checkpoint": int(n_at_checkpoint),
                    "above_threshold_at_checkpoint": int(n_at_checkpoint - below_at_checkpoint),
                    "below_threshold_at_checkpoint": int(below_at_checkpoint),
                    "down_crossing_events_into_checkpoint": int(down_into_checkpoint),
                    "cumulative_down_crossing_events": int(cumulative_down_events),
                    "runs_ever_down_crossed_by_checkpoint": int(runs_ever_down),
                }
            )
    return out


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _latex_label_slug(text: str) -> str:
    chars = []
    for ch in str(text).lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {"_", "-", " ", "."}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "checkpoint_summary"


def _write_checkpoint_table_latex(
    path: Path,
    *,
    title: str,
    scenario_key: str | None,
    count_label: str,
    method_labels: list[str],
    checkpoint_labels: list[str],
    table_text: list[list[str]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tabular_cols = "l" + ("c" * len(checkpoint_labels))
    caption = (
        f"{title}: {count_label} and relative RMSE by checkpoint. "
        "Each cell reports count/trials | RRMSE."
    )
    label_key = scenario_key if scenario_key is not None else title
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{tab:{_latex_label_slug(label_key)}_checkpoint_summary}}",
        rf"\begin{{tabular}}{{{tabular_cols}}}",
        r"\toprule",
        "Method & " + " & ".join(_latex_escape(label) for label in checkpoint_labels) + r" \\",
        r"\midrule",
    ]
    for method_label, values in zip(method_labels, table_text):
        lines.append(
            _latex_escape(method_label)
            + " & "
            + " & ".join(_latex_escape(value) for value in values)
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_near_threshold_checkpoint_boxplots(
    records: list[dict[str, Any]],
    *,
    scenario_name: str,
    exact_p: float,
    known_significance_threshold: float,
    checkpoints: list[int] | tuple[int, ...] = (1_000_000, 2_500_000, 5_000_000),
    method_order: list[str] | tuple[str, ...] | None = None,
    method_labels: dict[str, str] | None = None,
    method_colors: dict[str, str] | None = None,
    scenario_key: str | None = None,
    n_control: int | None = None,
    n_treated: int | None = None,
    threshold_count: str = "above",
    save_path: Path | None = None,
    table_save_path: Path | None = None,
    table_tex_save_path: Path | None = None,
) -> list[dict[str, Any]]:
    target_checkpoints = _sorted_unique_points(checkpoints)
    methods = _method_order_from_records(records, method_order)
    if not methods:
        raise ValueError("records contain no methods to plot.")
    threshold = float(known_significance_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("known_significance_threshold must be a positive finite value.")
    threshold_count = str(threshold_count).lower()
    if threshold_count not in {"above", "below"}:
        raise ValueError("threshold_count must be either 'above' or 'below'.")
    if method_labels is None:
        method_labels = {}
    if method_colors is None:
        method_colors = {}

    count_rows = summarize_threshold_crossings(
        records,
        known_significance_threshold=threshold,
        checkpoints=target_checkpoints,
        method_order=methods,
    )
    for row in count_rows:
        method = str(row["method"])
        checkpoint = int(row["checkpoint"])
        values = np.asarray(
            [
                record.get("estimate", np.nan)
                for record in records
                if int(record["checkpoint"]) == checkpoint and _record_group_label(record) == method
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        rmse = float(np.sqrt(np.mean((values - float(exact_p)) ** 2))) if values.size else float("nan")
        row["rmse_at_checkpoint"] = rmse
        row["relative_rmse_at_checkpoint"] = (
            float(rmse / float(exact_p))
            if np.isfinite(rmse) and np.isfinite(float(exact_p)) and float(exact_p) > 0.0
            else float("nan")
        )
    count_lookup = {(str(row["method"]), int(row["checkpoint"])): row for row in count_rows}

    fallback_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(methods), 1)))
    colors_by_method = {
        method: str(method_colors.get(method, matplotlib.colors.to_hex(fallback_colors[idx])))
        for idx, method in enumerate(methods)
    }
    labels_by_method = {method: str(method_labels.get(method, method)) for method in methods}
    for method, label in list(labels_by_method.items()):
        if label == "Simple-init MCMC-IS":
            labels_by_method[method] = "MCMC-IS"

    data: list[np.ndarray] = []
    positions: list[float] = []
    box_colors: list[str] = []
    group_centers: list[float] = []
    n_methods = len(methods)
    group_width = float(n_methods + 1.0)

    for checkpoint_idx, checkpoint in enumerate(target_checkpoints):
        base = checkpoint_idx * group_width
        group_positions = []
        for method_idx, method in enumerate(methods):
            x = base + method_idx + 1.0
            group_positions.append(x)
            values = np.asarray(
                [
                    row.get("estimate", np.nan)
                    for row in records
                    if int(row["checkpoint"]) == int(checkpoint) and _record_group_label(row) == method
                ],
                dtype=float,
            )
            data.append(_finite_for_plot(values))
            positions.append(x)
            box_colors.append(colors_by_method[method])
        group_centers.append(float(np.mean(group_positions)))

    fig, ax = plt.subplots(figsize=(11.2, 6.25))
    artists = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 2.0},
        whiskerprops={"color": "#6d7378", "linewidth": 1.3},
        capprops={"color": "#6d7378", "linewidth": 1.3},
        boxprops={"edgecolor": "#6d7378", "linewidth": 1.2},
    )
    for patch, color in zip(artists["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for x, values, color in zip(positions, data, box_colors):
        finite_values = np.asarray(values, dtype=float)
        finite_values = np.sort(finite_values[np.isfinite(finite_values)])
        if finite_values.size == 0:
            continue
        if finite_values.size == 1:
            jitter = np.asarray([0.0], dtype=float)
        else:
            jitter = np.linspace(-0.12, 0.12, finite_values.size)
        ax.scatter(
            float(x) + jitter,
            finite_values,
            s=20,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.62,
            zorder=3.5,
        )

    ax.axhline(threshold, color="#7b2d26", linestyle="-", linewidth=1.8, zorder=2.2)
    ax.axhline(float(exact_p), color="#000000", linestyle="--", linewidth=1.7, zorder=2.1)
    for checkpoint_idx in range(1, len(target_checkpoints)):
        ax.axvline(checkpoint_idx * group_width - 0.5, color="#d7dde0", linewidth=1.05, zorder=1.0)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([_format_budget_tick_compact(cp) for cp in target_checkpoints])
    ax.set_xlabel("Checkpoint budget", fontsize=15)
    ax.set_ylabel(r"$\hat{p}$", fontsize=15)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _set_linear_ylim(ax, data, include_values=[threshold, float(exact_p)], anchor_zero=False)
    _style_article_axis(ax, grid_axis="y")
    ax.tick_params(axis="both", labelsize=13)
    ax.yaxis.get_offset_text().set_fontsize(13)

    handles = [
        Line2D(
            [0],
            [0],
            color=colors_by_method[method],
            marker="s",
            linestyle="",
            markersize=9.5,
            label=labels_by_method[method],
        )
        for method in methods
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#7b2d26", linestyle="-", linewidth=1.8, label="Known threshold"),
            Line2D([0], [0], color="#000000", linestyle="--", linewidth=1.7, label="True p-value"),
        ]
    )
    ax.legend(handles=handles, frameon=False, fontsize=12.4, ncol=min(len(handles), 5), loc="upper right")

    table_text = []
    count_key = f"{threshold_count}_threshold_at_checkpoint"
    count_label = "Above TH count" if threshold_count == "above" else "Below TH count"
    for method in methods:
        row_text = []
        for checkpoint in target_checkpoints:
            count_row = count_lookup[(method, int(checkpoint))]
            threshold_count_value = int(count_row[count_key])
            n_runs_at_checkpoint = int(count_row["n_runs_at_checkpoint"])
            rel_rmse = float(count_row["relative_rmse_at_checkpoint"])
            rel_rmse_label = f"{rel_rmse:.2f}" if np.isfinite(rel_rmse) else "nan"
            row_text.append(f"{threshold_count_value}/{n_runs_at_checkpoint} | {rel_rmse_label}")
        table_text.append(row_text)

    title = _CROSS_METHOD_TITLES.get(str(scenario_key), " ".join(str(scenario_name).split()).rstrip("."))
    title = str(title).replace("\n", " - ")
    ax.set_title(title, fontsize=17, fontweight="semibold", pad=12)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if table_save_path is not None:
        fig_table, table_ax = plt.subplots(figsize=(8.4, 1.95))
        table_ax.set_facecolor("white")
        table_ax.axis("off")
        table_title = f"{title}: {count_label} | RRMSE"
        table_ax.set_title(table_title, fontsize=14.5, fontweight="semibold", pad=8)
        table = table_ax.table(
            cellText=table_text,
            rowLabels=[labels_by_method[method] for method in methods],
            colLabels=[_format_budget_tick_compact(cp) for cp in target_checkpoints],
            cellLoc="center",
            rowLoc="right",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 0.76],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12.4)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("#d7dde0")
            cell.set_linewidth(0.9)
            cell.PAD = 0.045
            if row_idx == 0 or col_idx == -1:
                cell.set_text_props(fontweight="semibold")
        table_save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_table.savefig(table_save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_table)
    if table_tex_save_path is not None:
        _write_checkpoint_table_latex(
            table_tex_save_path,
            title=title,
            scenario_key=scenario_key,
            count_label=count_label,
            method_labels=[labels_by_method[method] for method in methods],
            checkpoint_labels=[_format_budget_tick_compact(cp) for cp in target_checkpoints],
            table_text=table_text,
        )
    return count_rows


def summarize_threshold_grid_gamma_swap_rrmse(
    records: list[dict[str, Any]],
    *,
    max_budget: int,
) -> list[dict[str, Any]]:
    def float_or_nan(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    rows = [row for row in records if int(row.get("checkpoint", -1)) == int(max_budget)]
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        family = str(row.get("family", ""))
        method = str(row.get("method", ""))
        swap_fraction = float(row.get("swap_fraction", np.nan))
        gamma_label = str(row.get("gamma_label", "none"))
        grouped.setdefault((family, method, swap_fraction, gamma_label), []).append(row)

    out: list[dict[str, Any]] = []
    for (family, method, swap_fraction, gamma_label), sub in sorted(grouped.items()):
        are_values: list[float] = []
        abs_log_values: list[float] = []
        zero_estimates = 0
        for row in sub:
            exact_p = float_or_nan(row.get("exact_p", np.nan))
            estimate = float_or_nan(row.get("estimate", np.nan))
            if np.isfinite(estimate) and estimate <= 0.0:
                zero_estimates += 1
            if np.isfinite(exact_p) and exact_p > 0.0 and np.isfinite(estimate):
                are_values.append(float(abs((estimate - exact_p) / exact_p)))
            abs_log = float_or_nan(row.get("abs_log10_error", np.nan))
            if np.isfinite(abs_log):
                abs_log_values.append(abs_log)
        vals = np.asarray(are_values, dtype=float)
        gamma_raw = float_or_nan(sub[0].get("gamma", np.nan)) if sub else np.nan
        median_are = float(np.median(vals)) if vals.size else np.nan
        mean_are = float(np.mean(vals)) if vals.size else np.nan
        p90_are = float(np.quantile(vals, 0.90)) if vals.size else np.nan
        cross_scenario_rrmse = float(np.sqrt(np.mean(vals * vals))) if vals.size else np.nan
        out.append(
            {
                "family": family,
                "method": method,
                "swap_fraction": float(swap_fraction),
                "gamma_label": gamma_label,
                "gamma": gamma_raw if gamma_label != "none" and np.isfinite(gamma_raw) else np.nan,
                "checkpoint": int(max_budget),
                "n_scenarios": int(vals.size),
                "median_scenario_are": median_are,
                "mean_scenario_are": mean_are,
                "p90_scenario_are": p90_are,
                "cross_scenario_rrmse": cross_scenario_rrmse,
                "median_scenario_rrmse": median_are,
                "relative_rmse": cross_scenario_rrmse,
                "mean_scenario_rrmse": mean_are,
                "p90_scenario_rrmse": p90_are,
                "mean_abs_log10_error": float(np.mean(abs_log_values)) if abs_log_values else np.nan,
                "zero_estimate_rate": float(zero_estimates / len(sub)) if sub else np.nan,
            }
        )
    return out


def plot_threshold_grid_gamma_swap_rrmse(
    records: list[dict[str, Any]],
    *,
    max_budget: int,
    families: tuple[str, ...] = ("gwas", "hep"),
    gamma_values: tuple[float, ...] = (0.25, 1.0 / 3.0, 0.4),
    swap_fractions: tuple[float, ...] = (0.05, 0.10),
    save_path: Path | None = None,
    table_save_path: Path | None = None,
) -> list[dict[str, Any]]:
    summary_rows = summarize_threshold_grid_gamma_swap_rrmse(records, max_budget=max_budget)
    lookup = {
        (
            str(row["family"]),
            str(row["method"]),
            round(float(row["swap_fraction"]), 10),
            str(row["gamma_label"]),
        ): float(row["median_scenario_are"])
        for row in summary_rows
    }
    gamma_labels = ["0.25", "1/3", "0.4"]
    method_specs = [
        ("mcmc_is_no_oracle", "Smooth MCMC-IS", "#2f6fb0"),
        ("hard_step", "Hard-step MCMC", "#b04a5a"),
    ]
    swap_specs = {
        0.05: {"label": "5% swaps", "marker": "o", "linestyle": "-"},
        0.10: {"label": "10% swaps", "marker": "s", "linestyle": "--"},
    }
    samc_specs = {
        0.05: {"label": "SAMC, 5% swaps", "linestyle": (0, (1.2, 2.0)), "color": "#4c8c77"},
        0.10: {"label": "SAMC, 10% swaps", "linestyle": (0, (5.0, 2.0)), "color": "#2e6f5f"},
    }
    family_titles = {
        "gwas": "Panel A: GWAS-like scenarios",
        "hep": "Panel B: HEP-like scenarios",
    }
    x = np.arange(len(gamma_values), dtype=float)

    fig, axes = plt.subplots(1, len(families), figsize=(12.2, 4.8), sharey=True)
    if len(families) == 1:
        axes = np.asarray([axes])
    handles_by_label: dict[str, Any] = {}

    for col_idx, (ax, family) in enumerate(zip(axes, families)):
        for method, method_label, color in method_specs:
            for swap_fraction in swap_fractions:
                spec = swap_specs.get(
                    round(float(swap_fraction), 10),
                    {"label": f"{100.0 * float(swap_fraction):.0f}% swaps", "marker": "D", "linestyle": "-"},
                )
                y = np.asarray(
                    [
                        lookup.get(
                            (
                                str(family),
                                method,
                                round(float(swap_fraction), 10),
                                f"{float(gamma):.12g}",
                            ),
                            np.nan,
                        )
                        for gamma in gamma_values
                    ],
                    dtype=float,
                )
                (line,) = ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=spec["linestyle"],
                    marker=spec["marker"],
                    markersize=6.2,
                    linewidth=2.1,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=f"{method_label}, {spec['label']}",
                    zorder=3,
                )
                handles_by_label[line.get_label()] = line

        for swap_fraction in swap_fractions:
            samc_y = lookup.get((str(family), "samc", round(float(swap_fraction), 10), "none"), np.nan)
            if not np.isfinite(samc_y):
                continue
            spec = samc_specs.get(
                round(float(swap_fraction), 10),
                {"label": f"SAMC, {100.0 * float(swap_fraction):.0f}% swaps", "linestyle": ":", "color": "#4c8c77"},
            )
            ref = ax.hlines(
                samc_y,
                xmin=float(x[0]) - 0.08,
                xmax=float(x[-1]) + 0.08,
                colors=str(spec["color"]),
                linestyles=spec["linestyle"],
                linewidth=1.9,
                label=str(spec["label"]),
                zorder=2,
            )
            handles_by_label[str(spec["label"])] = ref

        ax.set_title(family_titles.get(str(family), str(family)), fontsize=12.4, fontweight="semibold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(gamma_labels)
        ax.set_xlabel(r"Target-tail exponent $\gamma$ in $q=p_0^\gamma$")
        ax.set_yscale("log")
        _style_article_axis(ax, grid_axis="y", add_minor_log_grid=True)
        ax.margins(x=0.10)

    axes[0].set_ylabel("Median scenario ARE")
    finite_y = np.asarray(
        [float(row["median_scenario_are"]) for row in summary_rows if np.isfinite(float(row["median_scenario_are"]))],
        dtype=float,
    )
    finite_y = finite_y[finite_y > 0.0]
    if finite_y.size:
        y_min = max(float(np.min(finite_y)) / 1.7, 1e-4)
        y_max = float(np.max(finite_y)) * 1.7
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    fig.legend(
        list(handles_by_label.values()),
        list(handles_by_label.keys()),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.4,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Threshold-grid median absolute relative error at {_format_budget_tick_compact(max_budget)}",
        fontsize=14.5,
        fontweight="semibold",
        y=1.02,
    )
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if table_save_path is not None:
        write_json(Path(table_save_path), summary_rows)
    plt.close(fig)
    return summary_rows


def _threshold_grid_metric_label(metric: str) -> tuple[str, str]:
    metric_key = str(metric)
    if metric_key == "median":
        return "median_scenario_are", "Median scenario ARE"
    if metric_key == "mean":
        return "mean_scenario_are", "Mean scenario ARE"
    if metric_key in {"relative_rmse", "rmse"}:
        return "cross_scenario_rrmse", "Cross-scenario RRMSE"
    raise ValueError("metric must be one of {'median', 'mean', 'relative_rmse'}.")


def plot_threshold_grid_tilt_family_rrmse(
    records: list[dict[str, Any]],
    *,
    max_budget: int,
    metric: str = "median",
    families: tuple[str, ...] = ("gwas", "hep"),
    gamma_values: tuple[float, ...] = (0.25, 1.0 / 3.0, 0.4),
    swap_fractions: tuple[float, ...] = (0.05, 0.10),
    save_path: Path | None = None,
    table_save_path: Path | None = None,
) -> list[dict[str, Any]]:
    metric_field, metric_label = _threshold_grid_metric_label(metric)
    summary_rows = summarize_threshold_grid_gamma_swap_rrmse(records, max_budget=max_budget)
    summary_rows = [
        row
        for row in summary_rows
        if str(row["method"]) in {"mcmc_is_no_oracle", "hard_step"}
    ]
    lookup = {
        (
            str(row["family"]),
            str(row["method"]),
            round(float(row["swap_fraction"]), 10),
            str(row["gamma_label"]),
        ): float(row[metric_field])
        for row in summary_rows
    }
    gamma_labels = ["0.25", "1/3", "0.4"]
    method_specs = [
        ("mcmc_is_no_oracle", "Smooth MCMC-IS", "#2f6fb0"),
        ("hard_step", "Hard-step MCMC", "#b04a5a"),
    ]
    family_labels = {
        "gwas": "GWAS-like",
        "hep": "HEP-like",
    }
    x = np.arange(len(gamma_values), dtype=float)

    fig, axes = plt.subplots(
        len(families),
        len(swap_fractions),
        figsize=(7.8, 6.65),
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape((1, 1))
    elif axes.ndim == 1:
        axes = axes.reshape((len(families), len(swap_fractions)))
    handles_by_label: dict[str, Any] = {}

    for row_idx, family in enumerate(families):
        row_values: list[float] = []
        for swap_fraction in swap_fractions:
            for method, _, _ in method_specs:
                for gamma in gamma_values:
                    value = lookup.get(
                        (
                            str(family),
                            method,
                            round(float(swap_fraction), 10),
                            f"{float(gamma):.12g}",
                        ),
                        np.nan,
                    )
                    if np.isfinite(value) and value > 0.0:
                        row_values.append(float(value))
        for col_idx, swap_fraction in enumerate(swap_fractions):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(f"{100.0 * float(swap_fraction):.0f}% swaps", fontsize=11.8, pad=8)
            if col_idx == 0:
                ax.text(
                    -0.33,
                    0.5,
                    family_labels.get(str(family), str(family)),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11.6,
                    fontweight="semibold",
                )

            if row_idx == len(families) - 1:
                ax.set_xlabel(r"$\gamma$")
            if col_idx == 0:
                ax.set_ylabel(metric_label)

            ax.set_xticks(x)
            ax.set_xticklabels(gamma_labels)
            ax.set_yscale("log")
            _style_article_axis(ax, grid_axis="y", add_minor_log_grid=True)
            ax.margins(x=0.10)

            if row_values:
                y_min = max(float(np.min(row_values)) / 1.6, 1e-4)
                y_max = float(np.max(row_values)) * 1.6
                ax.set_ylim(y_min, y_max)

        for method, method_label, color in method_specs:
            for col_idx, swap_fraction in enumerate(swap_fractions):
                ax = axes[row_idx, col_idx]
                y = np.asarray(
                    [
                        lookup.get(
                            (
                                str(family),
                                method,
                                round(float(swap_fraction), 10),
                                f"{float(gamma):.12g}",
                            ),
                            np.nan,
                        )
                        for gamma in gamma_values
                    ],
                    dtype=float,
                )
                (line,) = ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle="-",
                    marker="o",
                    markersize=5.8,
                    linewidth=2.15,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=method_label,
                    zorder=3,
                )
                handles_by_label[line.get_label()] = line

    fig.legend(
        list(handles_by_label.values()),
        list(handles_by_label.keys()),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9.8,
        bbox_to_anchor=(0.5, -0.02),
    )
    metric_title = metric_label.replace("ARE", "absolute relative error")
    metric_title = metric_title[:1].lower() + metric_title[1:]
    fig.suptitle(
        _format_budget_title(max_budget, f"Tilt tuning: {metric_title}"),
        fontsize=12.2,
        y=0.99,
    )
    plt.tight_layout(rect=(0.04, 0.08, 1.0, 0.99))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if table_save_path is not None:
        write_json(Path(table_save_path), summary_rows)
    plt.close(fig)
    return summary_rows


def _threshold_grid_config_label(row: dict[str, Any]) -> str:
    swap_label = f"{100.0 * float(row['swap_fraction']):.0f}%"
    if str(row["method"]) == "samc":
        return swap_label
    gamma = float(row.get("gamma", np.nan))
    gamma_label = "1/3" if np.isclose(gamma, 1.0 / 3.0) else f"{gamma:.2g}"
    return f"{swap_label}, gamma={gamma_label}"


def _threshold_grid_figure_config_label(row: dict[str, Any]) -> str:
    swap_label = f"{100.0 * float(row['swap_fraction']):.0f}% swaps"
    if str(row["method"]) == "samc":
        return swap_label
    gamma = float(row.get("gamma", np.nan))
    gamma_label = "1/3" if np.isclose(gamma, 1.0 / 3.0) else f"{gamma:.2g}"
    return rf"$\gamma$={gamma_label}, {swap_label}"


def _threshold_grid_gamma_sort_value(row: dict[str, Any]) -> float:
    gamma = row.get("gamma", np.nan)
    if gamma is None:
        return np.inf
    try:
        gamma_f = float(gamma)
    except (TypeError, ValueError):
        return np.inf
    return gamma_f if np.isfinite(gamma_f) else np.inf


def _select_threshold_grid_best_practical_rows(
    records: list[dict[str, Any]],
    *,
    max_budget: int,
    families: tuple[str, ...],
    selection_metric: str,
) -> list[dict[str, Any]]:
    selection_field, _ = _threshold_grid_metric_label(selection_metric)
    summary_rows = summarize_threshold_grid_gamma_swap_rrmse(records, max_budget=max_budget)
    selected_rows: list[dict[str, Any]] = []
    for family in families:
        for method, display_method in (
            ("mcmc_is_no_oracle", "Smooth MCMC-IS"),
            ("samc", "SAMC"),
        ):
            candidates = [
                row
                for row in summary_rows
                if str(row["family"]) == str(family)
                and str(row["method"]) == method
                and np.isfinite(float(row[selection_field]))
            ]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda row: (
                    float(row[selection_field]),
                    float(row["swap_fraction"]),
                    _threshold_grid_gamma_sort_value(row),
                ),
            )
            out = dict(best)
            out["display_method"] = display_method
            out["config_label"] = _threshold_grid_config_label(best)
            out["selection_metric"] = str(selection_metric)
            out["selection_value"] = float(best[selection_field])
            selected_rows.append(out)
    return selected_rows


def _threshold_grid_record_matches_selection(
    record: dict[str, Any],
    row: dict[str, Any],
    *,
    max_budget: int,
) -> bool:
    if int(record.get("checkpoint", -1)) != int(max_budget):
        return False
    if str(record.get("family", "")) != str(row["family"]):
        return False
    if str(record.get("method", "")) != str(row["method"]):
        return False
    try:
        record_swap = round(float(record.get("swap_fraction", np.nan)), 10)
        row_swap = round(float(row["swap_fraction"]), 10)
    except (TypeError, ValueError):
        return False
    if record_swap != row_swap:
        return False
    return str(record.get("gamma_label", "none")) == str(row["gamma_label"])


def _threshold_grid_record_are(record: dict[str, Any]) -> float | None:
    exact_p = record.get("exact_p", np.nan)
    estimate = record.get("estimate", np.nan)
    try:
        exact_f = float(exact_p)
        estimate_f = float(estimate)
    except (TypeError, ValueError):
        return None
    if np.isfinite(exact_f) and exact_f > 0.0 and np.isfinite(estimate_f):
        return float(abs((estimate_f - exact_f) / exact_f))
    return None


def _threshold_grid_record_rrmse(record: dict[str, Any]) -> float | None:
    return _threshold_grid_record_are(record)


def _threshold_grid_equal_axis_ticks(upper: float) -> tuple[float, np.ndarray]:
    upper_f = float(upper)
    if not np.isfinite(upper_f) or upper_f <= 0.0:
        upper_f = 1.0
    ticks = np.asarray(MaxNLocator(nbins=5, min_n_ticks=4).tick_values(0.0, upper_f), dtype=float)
    ticks = np.asarray(
        [tick for tick in ticks if np.isfinite(tick) and 0.0 <= tick <= upper_f * (1.0 + 1e-9)],
        dtype=float,
    )
    if ticks.size == 0 or not np.isclose(float(ticks[0]), 0.0):
        ticks = np.concatenate(([0.0], ticks))
    if ticks.size < 2:
        ticks = np.asarray([0.0, upper_f], dtype=float)
    return upper_f, ticks


def _threshold_grid_selected_rrmse_values(
    records: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    max_budget: int,
) -> np.ndarray:
    values: list[float] = []
    for record in records:
        if not _threshold_grid_record_matches_selection(record, row, max_budget=max_budget):
            continue
        are = _threshold_grid_record_are(record)
        if are is not None:
            values.append(are)
    return np.asarray(values, dtype=float)


def _threshold_grid_selected_rrmse_by_scenario(
    records: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    max_budget: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        if not _threshold_grid_record_matches_selection(record, row, max_budget=max_budget):
            continue
        scenario = record.get("scenario")
        if scenario is None:
            continue
        are = _threshold_grid_record_are(record)
        if are is not None:
            grouped.setdefault(str(scenario), []).append(are)
    return {scenario: float(np.mean(vals)) for scenario, vals in grouped.items() if vals}


def bootstrap_threshold_grid_best_practical_diagnostics(
    records: list[dict[str, Any]],
    *,
    max_budget: int,
    families: tuple[str, ...] = ("gwas", "hep"),
    selection_metric: str = "median",
    config_selection_budget: int | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 20260724,
    save_path: Path | None = None,
) -> list[dict[str, Any]]:
    selection_budget = int(max_budget) if config_selection_budget is None else int(config_selection_budget)
    selected_rows = _select_threshold_grid_best_practical_rows(
        records,
        max_budget=selection_budget,
        families=families,
        selection_metric=selection_metric,
    )
    rows_by_family_method = {
        (str(row["family"]), str(row["method"])): row
        for row in selected_rows
    }
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []

    for family in families:
        smooth_row = rows_by_family_method.get((str(family), "mcmc_is_no_oracle"))
        samc_row = rows_by_family_method.get((str(family), "samc"))
        paired_points: list[dict[str, float | str]] = []
        if smooth_row is not None and samc_row is not None:
            smooth_by_scenario = _threshold_grid_selected_rrmse_by_scenario(
                records,
                smooth_row,
                max_budget=max_budget,
            )
            samc_by_scenario = _threshold_grid_selected_rrmse_by_scenario(
                records,
                samc_row,
                max_budget=max_budget,
            )
            for scenario in sorted(set(smooth_by_scenario).intersection(samc_by_scenario)):
                smooth_are = float(smooth_by_scenario[scenario])
                samc_are = float(samc_by_scenario[scenario])
                if np.isfinite(smooth_are) and np.isfinite(samc_are):
                    paired_points.append(
                        {
                            "scenario": scenario,
                            "smooth_are": smooth_are,
                            "samc_are": samc_are,
                        }
                    )

        smooth_values = np.asarray([float(point["smooth_are"]) for point in paired_points], dtype=float)
        samc_values = np.asarray([float(point["samc_are"]) for point in paired_points], dtype=float)
        n_scenarios = int(smooth_values.size)
        observed_smooth_rrmse = float(np.sqrt(np.mean(smooth_values * smooth_values))) if n_scenarios else np.nan
        observed_samc_rrmse = float(np.sqrt(np.mean(samc_values * samc_values))) if n_scenarios else np.nan
        observed_rrmse_ratio = (
            float(observed_smooth_rrmse / observed_samc_rrmse)
            if np.isfinite(observed_smooth_rrmse)
            and np.isfinite(observed_samc_rrmse)
            and observed_samc_rrmse > 0.0
            else np.nan
        )
        observed_win_count = int(np.sum(smooth_values < samc_values)) if n_scenarios else 0
        observed_win_rate = float(observed_win_count / n_scenarios) if n_scenarios else np.nan

        if n_scenarios and int(n_bootstrap) > 0:
            sample_indices = rng.integers(0, n_scenarios, size=(int(n_bootstrap), n_scenarios))
            sampled_smooth = smooth_values[sample_indices]
            sampled_samc = samc_values[sample_indices]
            bootstrap_win_rates = np.mean(sampled_smooth < sampled_samc, axis=1)
            bootstrap_smooth_rrmse = np.sqrt(np.mean(sampled_smooth * sampled_smooth, axis=1))
            bootstrap_samc_rrmse = np.sqrt(np.mean(sampled_samc * sampled_samc, axis=1))
            bootstrap_rrmse_ratios = np.divide(
                bootstrap_smooth_rrmse,
                bootstrap_samc_rrmse,
                out=np.full_like(bootstrap_smooth_rrmse, np.nan, dtype=float),
                where=bootstrap_samc_rrmse > 0.0,
            )
        else:
            bootstrap_win_rates = np.asarray([], dtype=float)
            bootstrap_rrmse_ratios = np.asarray([], dtype=float)

        finite_rrmse_ratios = bootstrap_rrmse_ratios[np.isfinite(bootstrap_rrmse_ratios)]
        finite_win_rates = bootstrap_win_rates[np.isfinite(bootstrap_win_rates)]
        out.append(
            {
                "family": str(family),
                "checkpoint": int(max_budget),
                "config_selection_checkpoint": int(selection_budget),
                "selection_metric": str(selection_metric),
                "smooth_config_label": smooth_row.get("config_label") if smooth_row is not None else None,
                "samc_config_label": samc_row.get("config_label") if samc_row is not None else None,
                "n_scenarios": n_scenarios,
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_seed": int(seed),
                "observed_rrmse_ratio_smooth_over_samc": observed_rrmse_ratio,
                "observed_smooth_cross_scenario_rrmse": observed_smooth_rrmse,
                "observed_samc_cross_scenario_rrmse": observed_samc_rrmse,
                "observed_win_rate": observed_win_rate,
                "observed_win_count": observed_win_count,
                "rrmse_ratio_bootstrap_mean": (
                    float(np.mean(finite_rrmse_ratios)) if finite_rrmse_ratios.size else np.nan
                ),
                "rrmse_ratio_bootstrap_median": (
                    float(np.median(finite_rrmse_ratios)) if finite_rrmse_ratios.size else np.nan
                ),
                "rrmse_ratio_ci95_low": (
                    float(np.quantile(finite_rrmse_ratios, 0.025)) if finite_rrmse_ratios.size else np.nan
                ),
                "rrmse_ratio_ci95_high": (
                    float(np.quantile(finite_rrmse_ratios, 0.975)) if finite_rrmse_ratios.size else np.nan
                ),
                "win_rate_bootstrap_mean": float(np.mean(finite_win_rates)) if finite_win_rates.size else np.nan,
                "win_rate_bootstrap_median": float(np.median(finite_win_rates)) if finite_win_rates.size else np.nan,
                "win_rate_ci95_low": (
                    float(np.quantile(finite_win_rates, 0.025)) if finite_win_rates.size else np.nan
                ),
                "win_rate_ci95_high": (
                    float(np.quantile(finite_win_rates, 0.975)) if finite_win_rates.size else np.nan
                ),
                "bootstrap_rrmse_ratios": bootstrap_rrmse_ratios.tolist(),
                "bootstrap_win_rates": bootstrap_win_rates.tolist(),
                "paired_points": paired_points,
            }
        )

    if save_path is not None:
        write_json(Path(save_path), out)
    return out


def plot_threshold_grid_best_practical_rrmse(
    records: list[dict[str, Any]],
    *,
    max_budget: int,
    families: tuple[str, ...] = ("gwas", "hep"),
    selection_metric: str = "median",
    save_path: Path | None = None,
    table_save_path: Path | None = None,
) -> list[dict[str, Any]]:
    selected_rows = _select_threshold_grid_best_practical_rows(
        records,
        max_budget=max_budget,
        families=families,
        selection_metric=selection_metric,
    )
    family_labels = {"gwas": "GWAS-like", "hep": "HEP-like"}
    colors = {
        "Smooth MCMC-IS": "#2f6fb0",
        "SAMC": "#4c8c77",
    }

    fig, axes = plt.subplots(1, len(families), figsize=(7.8, 3.95), sharey=False)
    if len(families) == 1:
        axes = np.asarray([axes])
    for row in selected_rows:
        values = _threshold_grid_selected_rrmse_values(records, row, max_budget=max_budget)
        values_for_plot = values[np.isfinite(values) & (values >= 0.0)]
        if values_for_plot.size == 0:
            values_for_plot = np.asarray([0.0], dtype=float)
        row["scenario_are_q25"] = float(np.quantile(values, 0.25)) if values.size else np.nan
        row["scenario_are_q75"] = float(np.quantile(values, 0.75)) if values.size else np.nan
        row["scenario_are_min"] = float(np.min(values)) if values.size else np.nan
        row["scenario_are_max"] = float(np.max(values)) if values.size else np.nan
        row["scenario_rrmse_q25"] = float(np.quantile(values, 0.25)) if values.size else np.nan
        row["scenario_rrmse_q75"] = float(np.quantile(values, 0.75)) if values.size else np.nan
        row["scenario_rrmse_min"] = float(np.min(values)) if values.size else np.nan
        row["scenario_rrmse_max"] = float(np.max(values)) if values.size else np.nan
        row["n_boxplot_scenarios"] = int(values.size)
        row["_plot_values"] = values_for_plot

    for col_idx, (ax, family) in enumerate(zip(axes, families)):
        family_rows = [
            row
            for row in selected_rows
            if str(row["family"]) == str(family)
        ]
        family_rows = sorted(
            family_rows,
            key=lambda row: 0 if str(row["method"]) == "mcmc_is_no_oracle" else 1,
        )
        data = [np.asarray(row["_plot_values"], dtype=float) for row in family_rows]
        labels = [
            (
                f"{'Smooth MCMC-IS' if str(row['method']) == 'mcmc_is_no_oracle' else 'SAMC'}\n"
                f"{_threshold_grid_figure_config_label(row)}"
            ).rstrip()
            for row in family_rows
        ]
        box_colors = [colors[str(row["display_method"])] for row in family_rows]
        artists = ax.boxplot(
            data,
            tick_labels=labels,
            widths=0.54,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            medianprops={"color": "#222222", "linewidth": 1.35},
            meanprops={
                "marker": "D",
                "markerfacecolor": "#171717",
                "markeredgecolor": "white",
                "markeredgewidth": 0.75,
                "markersize": 6.0,
            },
            whiskerprops={"color": "#6d7378", "linewidth": 0.95},
            capprops={"color": "#6d7378", "linewidth": 0.95},
            boxprops={"edgecolor": "#333333", "linewidth": 0.95},
        )
        for patch, color in zip(artists["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.76)
        ax.set_title(family_labels.get(str(family), str(family)), fontsize=11.8, pad=8)
        _style_article_axis(ax, grid_axis="y", add_minor_log_grid=False)
        ax.tick_params(axis="x", labelsize=9.1)
        ax.margins(x=0.10)
        displayed_y: list[float] = []
        for artist_key in ("whiskers", "caps", "medians", "means"):
            for artist in artists.get(artist_key, []):
                y_data = np.asarray(artist.get_ydata(), dtype=float)
                displayed_y.extend(float(v) for v in y_data if np.isfinite(v) and v >= 0.0)
        if displayed_y:
            y_max = max(float(np.max(displayed_y)) * 1.08, 1e-3)
            ax.set_ylim(0.0, y_max)
    axes[0].set_ylabel("Scenario ARE")
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#171717",
                markeredgecolor="white",
                markersize=6.5,
                label="Mean",
            )
        ],
        frameon=False,
        fontsize=9.2,
        loc="upper left",
    )

    for row in selected_rows:
        row.pop("_plot_values", None)

    fig.suptitle(
        _format_budget_title(max_budget, "Scenario-level absolute relative error distributions"),
        fontsize=12.2,
        y=0.99,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if table_save_path is not None:
        write_json(Path(table_save_path), selected_rows)
    plt.close(fig)
    return selected_rows


def plot_threshold_grid_best_practical_scenario_scatter(
    records: list[dict[str, Any]],
    *,
    max_budget: int,
    families: tuple[str, ...] = ("gwas", "hep"),
    selection_metric: str = "median",
    save_path: Path | None = None,
    table_save_path: Path | None = None,
) -> list[dict[str, Any]]:
    selected_rows = _select_threshold_grid_best_practical_rows(
        records,
        max_budget=max_budget,
        families=families,
        selection_metric=selection_metric,
    )
    rows_by_family_method = {
        (str(row["family"]), str(row["method"])): row
        for row in selected_rows
    }
    family_labels = {"gwas": "GWAS-like", "hep": "HEP-like"}

    fig, axes = plt.subplots(1, len(families), figsize=(7.8, 4.65), sharex=False, sharey=False)
    if len(families) == 1:
        axes = np.asarray([axes])
    summary_rows: list[dict[str, Any]] = []

    for col_idx, (ax, family) in enumerate(zip(axes, families)):
        smooth_row = rows_by_family_method.get((str(family), "mcmc_is_no_oracle"))
        samc_row = rows_by_family_method.get((str(family), "samc"))
        paired_points: list[dict[str, float | str]] = []
        if smooth_row is not None and samc_row is not None:
            smooth_by_scenario = _threshold_grid_selected_rrmse_by_scenario(
                records,
                smooth_row,
                max_budget=max_budget,
            )
            samc_by_scenario = _threshold_grid_selected_rrmse_by_scenario(
                records,
                samc_row,
                max_budget=max_budget,
            )
            for scenario in sorted(set(smooth_by_scenario).intersection(samc_by_scenario)):
                smooth_are = float(smooth_by_scenario[scenario])
                samc_are = float(samc_by_scenario[scenario])
                if np.isfinite(smooth_are) and np.isfinite(samc_are):
                    paired_points.append(
                        {
                            "scenario": scenario,
                            "smooth_are": smooth_are,
                            "samc_are": samc_are,
                            "smooth_rrmse": smooth_are,
                            "samc_rrmse": samc_are,
                        }
                    )

        smooth_values = np.asarray([float(p["smooth_are"]) for p in paired_points], dtype=float)
        samc_values = np.asarray([float(p["samc_are"]) for p in paired_points], dtype=float)
        ratio_values = np.asarray(
            [
                float(smooth / samc)
                for smooth, samc in zip(smooth_values, samc_values)
                if np.isfinite(smooth) and np.isfinite(samc) and samc > 0.0
            ],
            dtype=float,
        )
        smooth_cross_scenario_rrmse = (
            float(np.sqrt(np.mean(smooth_values * smooth_values))) if smooth_values.size else np.nan
        )
        samc_cross_scenario_rrmse = (
            float(np.sqrt(np.mean(samc_values * samc_values))) if samc_values.size else np.nan
        )
        cross_scenario_rrmse_ratio = (
            float(smooth_cross_scenario_rrmse / samc_cross_scenario_rrmse)
            if np.isfinite(smooth_cross_scenario_rrmse)
            and np.isfinite(samc_cross_scenario_rrmse)
            and samc_cross_scenario_rrmse > 0.0
            else np.nan
        )
        smooth_are_lower_count = int(np.sum(smooth_values < samc_values)) if smooth_values.size else 0
        win_prob = float(np.mean(smooth_values < samc_values)) if smooth_values.size else np.nan
        median_ratio = float(np.median(ratio_values)) if ratio_values.size else np.nan

        if smooth_values.size:
            raw_axis_max = max(float(np.max(samc_values)), float(np.max(smooth_values)), 1e-3) * 1.08
            axis_max, axis_ticks = _threshold_grid_equal_axis_ticks(raw_axis_max)
            ax.scatter(
                samc_values,
                smooth_values,
                s=36,
                color="#2f6fb0",
                alpha=0.58,
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
            )
        else:
            axis_max, axis_ticks = _threshold_grid_equal_axis_ticks(1.0)
        ax.plot(
            [0.0, axis_max],
            [0.0, axis_max],
            color="#545b61",
            linestyle=(0, (1.3, 2.3)),
            linewidth=1.05,
            zorder=2,
        )
        ax.set_xlim(0.0, axis_max)
        ax.set_ylim(0.0, axis_max)
        ax.xaxis.set_major_locator(FixedLocator(axis_ticks))
        ax.yaxis.set_major_locator(FixedLocator(axis_ticks))
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        smooth_config = _threshold_grid_figure_config_label(smooth_row) if smooth_row is not None else ""
        samc_config = _threshold_grid_figure_config_label(samc_row) if samc_row is not None else ""
        ax.set_xlabel(f"SAMC ARE\n{samc_config}".rstrip(), fontsize=9.4)
        if col_idx == 0:
            ax.set_ylabel(f"Smooth MCMC-IS ARE\n{smooth_config}".rstrip(), fontsize=9.4)
        ratio_text = "NA" if not np.isfinite(cross_scenario_rrmse_ratio) else f"{cross_scenario_rrmse_ratio:.2f}"
        win_text = (
            "NA"
            if smooth_values.size == 0
            else f"{smooth_are_lower_count}/{smooth_values.size}"
        )
        ax.set_title(
            f"{family_labels.get(str(family), str(family))}\n"
            f"Cross-scenario RRMSE ratio = {ratio_text}\n"
            f"Smooth ARE lower in {win_text} scenarios",
            fontsize=9.7,
            pad=8,
        )
        _style_article_axis(ax, grid_axis="both", add_minor_log_grid=False)

        summary_rows.append(
            {
                "family": str(family),
                "checkpoint": int(max_budget),
                "selection_metric": str(selection_metric),
                "smooth_config_label": smooth_row.get("config_label") if smooth_row is not None else None,
                "samc_config_label": samc_row.get("config_label") if samc_row is not None else None,
                "n_paired_scenarios": int(len(paired_points)),
                "cross_scenario_rrmse_ratio_smooth_over_samc": cross_scenario_rrmse_ratio,
                "smooth_cross_scenario_rrmse": smooth_cross_scenario_rrmse,
                "samc_cross_scenario_rrmse": samc_cross_scenario_rrmse,
                "n_smooth_are_lt_samc": smooth_are_lower_count,
                "median_are_ratio_smooth_over_samc": median_ratio,
                "prob_smooth_are_lt_samc": win_prob,
                "median_smooth_are": float(np.median(smooth_values)) if smooth_values.size else np.nan,
                "median_samc_are": float(np.median(samc_values)) if samc_values.size else np.nan,
                "median_rrmse_ratio_smooth_over_samc": median_ratio,
                "prob_smooth_rrmse_lt_samc": win_prob,
                "median_smooth_rrmse": float(np.median(smooth_values)) if smooth_values.size else np.nan,
                "median_samc_rrmse": float(np.median(samc_values)) if samc_values.size else np.nan,
                "paired_points": paired_points,
            }
        )

    fig.suptitle(
        _format_budget_title(max_budget, "Scenario-by-scenario absolute relative error comparison"),
        fontsize=12.2,
        y=0.99,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if table_save_path is not None:
        write_json(Path(table_save_path), summary_rows)
    plt.close(fig)
    return summary_rows


def _threshold_grid_ratio_point(record: dict[str, Any]) -> dict[str, float] | None:
    try:
        exact_p = float(record.get("exact_p", np.nan))
        estimate = float(record.get("estimate", np.nan))
    except (TypeError, ValueError):
        return None
    p0_candidates = (
        record.get("p0_reference", None),
        record.get("known_significance_threshold", None),
        record.get("hard_step_reference_p0", None),
    )
    p0 = np.nan
    for candidate in p0_candidates:
        try:
            candidate_f = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate_f) and candidate_f > 0.0:
            p0 = candidate_f
            break
    if not (np.isfinite(p0) and p0 > 0.0 and np.isfinite(exact_p) and np.isfinite(estimate)):
        return None
    try:
        p_over_p0 = float(record.get("p_over_p0", np.nan))
    except (TypeError, ValueError):
        p_over_p0 = np.nan
    if not np.isfinite(p_over_p0):
        p_over_p0 = exact_p / p0
    estimate_over_p0 = estimate / p0
    return {
        "p0_reference": float(p0),
        "p_over_p0": float(p_over_p0),
        "estimate_over_p0": float(estimate_over_p0),
        "ratio_error": float(estimate_over_p0 - p_over_p0),
        "abs_ratio_error": float(abs(estimate_over_p0 - p_over_p0)),
        "truth_below_threshold": bool(p_over_p0 < 1.0),
        "estimate_above_threshold": bool(estimate_over_p0 > 1.0),
        "false_negative": bool(p_over_p0 < 1.0 and estimate_over_p0 > 1.0),
    }


def plot_threshold_grid_estimate_vs_threshold_ratio(
    records: list[dict[str, Any]],
    *,
    max_budget: int,
    families: tuple[str, ...] = ("gwas", "hep"),
    selection_metric: str = "median",
    save_path: Path | None = None,
    table_save_path: Path | None = None,
) -> list[dict[str, Any]]:
    selected_rows = _select_threshold_grid_best_practical_rows(
        records,
        max_budget=max_budget,
        families=families,
        selection_metric=selection_metric,
    )
    rows_by_family_method = {
        (str(row["family"]), str(row["method"])): row
        for row in selected_rows
    }
    method_specs = [
        ("mcmc_is_no_oracle", "Smooth MCMC-IS", "#2f6fb0"),
        ("samc", "SAMC", "#4c8c77"),
    ]
    family_labels = {"gwas": "GWAS-like", "hep": "HEP-like"}
    point_rows: list[dict[str, Any]] = []
    points_by_panel: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for family in families:
        for method, display_method, _ in method_specs:
            selected_row = rows_by_family_method.get((str(family), method))
            panel_points: list[dict[str, Any]] = []
            if selected_row is not None:
                for record in records:
                    if not _threshold_grid_record_matches_selection(record, selected_row, max_budget=max_budget):
                        continue
                    point = _threshold_grid_ratio_point(record)
                    if point is None:
                        continue
                    out = {
                        "family": str(family),
                        "method": method,
                        "display_method": display_method,
                        "checkpoint": int(max_budget),
                        "selection_metric": str(selection_metric),
                        "config_label": selected_row.get("config_label"),
                        "scenario": record.get("scenario"),
                        "exact_p": float(record.get("exact_p", np.nan)),
                        "estimate": float(record.get("estimate", np.nan)),
                        **point,
                    }
                    panel_points.append(out)
                    point_rows.append(out)
            points_by_panel[(str(family), method)] = panel_points

    fig, axes = plt.subplots(len(families), len(method_specs), figsize=(8.7, 7.35), sharex=False, sharey=False)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape((len(families), len(method_specs)))

    for row_idx, family in enumerate(families):
        row_x: list[float] = []
        row_y: list[float] = []
        for method, _, _ in method_specs:
            for point in points_by_panel.get((str(family), method), []):
                row_x.append(float(point["p_over_p0"]))
                row_y.append(float(point["estimate_over_p0"]))
        finite_row_values = np.asarray(
            [value for value in (*row_x, *row_y, 1.0) if np.isfinite(float(value))],
            dtype=float,
        )
        if finite_row_values.size:
            row_min = max(0.0, float(np.min(finite_row_values)))
            row_max = float(np.max(finite_row_values))
            pad = max(0.025, 0.055 * max(row_max - row_min, 1e-6))
            axis_min = max(0.0, row_min - pad)
            axis_max = row_max + pad
        else:
            axis_min, axis_max = 0.0, 1.0
        ticks = np.asarray(MaxNLocator(nbins=5, min_n_ticks=4).tick_values(axis_min, axis_max), dtype=float)
        ticks = np.asarray(
            [tick for tick in ticks if np.isfinite(tick) and axis_min <= tick <= axis_max * (1.0 + 1e-9)],
            dtype=float,
        )

        for col_idx, (method, display_method, color) in enumerate(method_specs):
            ax = axes[row_idx, col_idx]
            panel_points = points_by_panel.get((str(family), method), [])
            x = np.asarray([float(point["p_over_p0"]) for point in panel_points], dtype=float)
            y = np.asarray([float(point["estimate_over_p0"]) for point in panel_points], dtype=float)
            if x.size:
                ax.scatter(
                    x,
                    y,
                    s=36,
                    color=color,
                    alpha=0.68,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=3,
                )
            ax.plot(
                [axis_min, axis_max],
                [axis_min, axis_max],
                color="#545b61",
                linestyle=(0, (1.3, 2.3)),
                linewidth=1.05,
                label="Exact",
                zorder=2,
            )
            ax.axhline(
                1.0,
                color="#7b2d26",
                linestyle="-",
                linewidth=1.15,
                label=r"$p_0$ threshold",
                zorder=2.2,
            )
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            if ticks.size:
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1)
            _style_article_axis(ax, grid_axis="both", add_minor_log_grid=False)
            if row_idx == 0:
                ax.set_title(display_method, fontsize=11.8, pad=8)
            if row_idx == len(families) - 1:
                ax.set_xlabel(r"$p/p_0$", fontsize=10.0, labelpad=5)
            if col_idx == 0:
                ax.set_ylabel(r"$\hat{p}/p_0$", fontsize=10.0, labelpad=7)
            if col_idx == 0:
                ax.text(
                    -0.39,
                    0.5,
                    family_labels.get(str(family), str(family)),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11.5,
                    fontweight="semibold",
                )
            fn_count = int(sum(bool(point.get("false_negative", False)) for point in panel_points))
            ax.text(
                0.03,
                0.96,
                f"FN: {fn_count}/{len(panel_points)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.8,
                color="#333333",
            )

    plt.tight_layout(rect=(0.15, 0.13, 1.0, 0.925))
    fig.canvas.draw()
    panel_boxes = [ax.get_position() for ax in axes.ravel()]
    panel_center_x = 0.5 * (min(box.x0 for box in panel_boxes) + max(box.x1 for box in panel_boxes))
    fig.text(
        panel_center_x,
        0.995,
        "Estimated versus exact p-values",
        ha="center",
        va="top",
        fontsize=12.2,
    )
    fig.text(
        panel_center_x,
        0.957,
        f"Results at {_format_budget_millions(max_budget)} million iterations",
        ha="center",
        va="top",
        fontsize=9.7,
        color="#333333",
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="#545b61", linestyle=(0, (1.3, 2.3)), linewidth=1.05, label=r"$\hat{p}=p$"),
            Line2D([0], [0], color="#7b2d26", linestyle="-", linewidth=1.15, label=r"$\hat{p}=p_0$"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9.6,
        bbox_to_anchor=(panel_center_x, 0.068),
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if table_save_path is not None:
        write_json(Path(table_save_path), point_rows)
    plt.close(fig)
    return point_rows


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
    hard_step = sorted([row for row in summary if row["method"] == "hard_step"], key=lambda row: row["checkpoint"])
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
        color=_CROSS_METHOD_COLORS["mcmc_is"],
        label=_CROSS_METHOD_LABELS["mcmc_is"],
    )
    if hard_step:
        axes[0, 1].plot(
            [row["checkpoint"] for row in hard_step],
            [row["mean_q_tilt_tail_share"] for row in hard_step],
            marker="s",
            color=_CROSS_METHOD_COLORS["hard_step"],
            label=_CROSS_METHOD_LABELS["hard_step"],
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Tilted-chain tail occupancy q")
    axes[0, 1].set_xlabel("iterations")
    axes[0, 1].set_ylabel("q_hat")
    axes[0, 1].legend(frameon=False, fontsize=8.8)

    axes[1, 0].plot(
        [row["checkpoint"] for row in mcmc],
        [row["mean_ess"] for row in mcmc],
        marker="o",
        color=_CROSS_METHOD_COLORS["mcmc_is"],
        label=_CROSS_METHOD_LABELS["mcmc_is"],
    )
    if hard_step:
        axes[1, 0].plot(
            [row["checkpoint"] for row in hard_step],
            [row["mean_ess"] for row in hard_step],
            marker="s",
            color=_CROSS_METHOD_COLORS["hard_step"],
            label=_CROSS_METHOD_LABELS["hard_step"],
        )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("IS ESS")
    axes[1, 0].set_xlabel("iterations")
    axes[1, 0].set_ylabel("ESS")
    axes[1, 0].legend(frameon=False, fontsize=8.8)

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
        scenario_key=study["scenario"],
        exact_p=float(study["exact_p"]),
        max_budget=max(study["estimation_points"]),
        n_control=int(scenario.problem.n_control),
        n_treated=int(scenario.problem.n_treated),
        beta_workflow=study["beta_workflow"],
        save_path=output_dir / "cross_method_max_budget.png",
    )
    plot_cross_method_convergence(
        study["summary"],
        scenario_name=study["scenario_display"],
        scenario_key=study["scenario"],
        exact_p=float(study["exact_p"]),
        n_control=int(scenario.problem.n_control),
        n_treated=int(scenario.problem.n_treated),
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
    beta_workflow_payload = dict(study["beta_workflow"])
    beta_workflow_payload.pop("production_init_states", None)
    beta_workflow_payload.pop("scan_sample_pack", None)
    local_scan_payload = beta_workflow_payload.get("local_scan")
    if isinstance(local_scan_payload, dict):
        local_scan_payload = dict(local_scan_payload)
        local_scan_payload.pop("sample_batches", None)
        beta_workflow_payload["local_scan"] = local_scan_payload
    write_jsonl(output_dir / "run_records.jsonl", study["records"])
    write_json(output_dir / "summary.json", study["summary"])
    write_json(
        output_dir / "metadata.json",
        {
            "scenario": study["scenario"],
            "scenario_display": study["scenario_display"],
            "scenario_portfolio": study.get("scenario_portfolio", {}),
            "exact_p": study["exact_p"],
            "exact_method": study["exact_method"],
            "exact_tail_hits": study["exact_tail_hits"],
            "exact_n_perm": study["exact_n_perm"],
            "n_treated": int(scenario.problem.n_treated),
            "n_control": int(scenario.problem.n_control),
            "n_total": int(scenario.problem.n),
            "estimation_points": study["estimation_points"],
            "mcmc_reported_checkpoints": study.get("mcmc_reported_checkpoints", study["estimation_points"]),
            "method_order": _available_cross_method_order(study["records"]),
            "method_labels": _CROSS_METHOD_LABELS,
            "method_colors": _CROSS_METHOD_COLORS,
            "cross_config": cross_cfg,
            "mcmc_config": mcmc_cfg,
            "samc_config": samc_cfg,
            "beta_workflow": beta_workflow_payload,
            "hard_step_workflow": study.get("hard_step_workflow", {"enabled": False}),
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


def save_mcmc_objective_grid_outputs(
    study: dict[str, Any],
    *,
    output_dir: Path,
    scenario_name: str,
    exact_p: float,
    notebook_config: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "repeat_records.jsonl", study["repeat_records"])
    write_json(
        output_dir / "config_summary.json",
        {
            "scenario_display": scenario_name,
            "exact_p": float(exact_p),
            "notebook_config": notebook_config,
            "study_context": study["study_context"],
            "config_summary": study["config_summary"],
        },
    )
    write_json(
        output_dir / "objective_winners.json",
        {
            "scenario_display": scenario_name,
            "exact_p": float(exact_p),
            "notebook_config": notebook_config,
            "study_context": study["study_context"],
            "oracle_winner": study["oracle_winner"],
            "objective_winners": study["objective_winners"],
            "objective_to_config": study["objective_to_config"],
        },
    )
    write_json(
        output_dir / "objective_seed_noise.json",
        {
            "scenario_display": scenario_name,
            "exact_p": float(exact_p),
            "notebook_config": notebook_config,
            "study_context": study["study_context"],
            "objective_seed_noise": study["objective_seed_noise"],
            "repeat_winner_records": study["repeat_winner_records"],
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


def _known_threshold_from_saved_metadata(metadata: dict[str, Any]) -> float | None:
    containers = [
        metadata.get("scenario_portfolio", {}),
        metadata.get("beta_reference", {}),
        metadata,
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("known_significance_threshold", "canonical_threshold_p0"):
            value = container.get(key)
            if value is None:
                continue
            threshold = float(value)
            if np.isfinite(threshold) and threshold > 0.0:
                return threshold
    return None


def _is_near_threshold_saved_metadata(metadata: dict[str, Any]) -> bool:
    portfolio = metadata.get("scenario_portfolio", {})
    if not isinstance(portfolio, dict):
        return False
    if str(portfolio.get("threshold_band", "")).lower() == "near":
        return True
    groups = {str(group).lower() for group in portfolio.get("groups", [])}
    return any(group.endswith("_near") or group.endswith("_threshold_suite_near") for group in groups)


def regenerate_near_threshold_checkpoint_visualizations_from_saved_run(
    run_dir: Path,
    *,
    save_dir: Path | None = None,
    checkpoints: list[int] | tuple[int, ...] = (1_000_000, 2_500_000, 5_000_000),
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    save_dir = Path(save_dir) if save_dir is not None else run_dir / "near_threshold_visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    plots: dict[str, Path] = {}
    tables: dict[str, Path] = {}
    table_tex: dict[str, Path] = {}
    all_counts: list[dict[str, Any]] = []
    for scenario_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        metadata_path = scenario_dir / "metadata.json"
        records_path = scenario_dir / "run_records.jsonl"
        summary_path = scenario_dir / "summary.json"
        if not (metadata_path.exists() and records_path.exists() and summary_path.exists()):
            continue
        metadata = read_json(metadata_path)
        if not _is_near_threshold_saved_metadata(metadata):
            continue
        threshold = _known_threshold_from_saved_metadata(metadata)
        if threshold is None:
            continue

        saved = load_cross_method_saved_output(scenario_dir)
        scenario_key = str(metadata.get("scenario", scenario_dir.name))
        plot_path = save_dir / f"{scenario_key}_checkpoint_boxplots.png"
        table_path = save_dir / f"{scenario_key}_checkpoint_table.png"
        table_tex_path = save_dir / f"{scenario_key}_checkpoint_table.tex"
        count_rows = plot_near_threshold_checkpoint_boxplots(
            saved["records"],
            scenario_name=str(metadata["scenario_display"]),
            scenario_key=scenario_key,
            exact_p=float(metadata["exact_p"]),
            known_significance_threshold=float(threshold),
            checkpoints=checkpoints,
            method_order=list(metadata.get("method_order", [])) or None,
            method_labels=dict(metadata.get("method_labels", {})),
            method_colors=dict(metadata.get("method_colors", {})),
            n_control=int(metadata["n_control"]) if metadata.get("n_control") is not None else None,
            n_treated=int(metadata["n_treated"]) if metadata.get("n_treated") is not None else None,
            save_path=plot_path,
            table_save_path=table_path,
            table_tex_save_path=table_tex_path,
        )
        plots[scenario_key] = plot_path
        tables[scenario_key] = table_path
        table_tex[scenario_key] = table_tex_path
        for row in count_rows:
            enriched = dict(row)
            enriched["scenario"] = scenario_key
            enriched["scenario_display"] = str(metadata["scenario_display"])
            enriched["threshold_band"] = str(metadata.get("scenario_portfolio", {}).get("threshold_band", ""))
            all_counts.append(enriched)

    counts_path = save_dir / "threshold_crossing_counts.json"
    write_json(counts_path, all_counts)
    manifest_path = save_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "run_dir": run_dir,
            "checkpoints": [int(v) for v in _sorted_unique_points(checkpoints)],
            "plots": plots,
            "tables": tables,
            "table_tex": table_tex,
            "counts_path": counts_path,
        },
    )
    return {
        "plots": plots,
        "tables": tables,
        "table_tex": table_tex,
        "counts": all_counts,
        "counts_path": counts_path,
        "manifest_path": manifest_path,
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


def load_mcmc_objective_grid_saved_output(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    config_summary_payload = read_json(output_dir / "config_summary.json")
    objective_winners_payload = read_json(output_dir / "objective_winners.json")
    objective_seed_noise_payload = read_json(output_dir / "objective_seed_noise.json")
    repeat_records = read_jsonl(output_dir / "repeat_records.jsonl")
    return {
        "output_dir": output_dir,
        "config_summary_payload": config_summary_payload,
        "objective_winners_payload": objective_winners_payload,
        "objective_seed_noise_payload": objective_seed_noise_payload,
        "repeat_records": repeat_records,
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
    n_control, n_treated = _resolve_group_sizes(
        scenario_key=str(metadata.get("scenario", "")) or None,
        n_control=metadata.get("n_control"),
        n_treated=metadata.get("n_treated"),
    )

    out = {
        "cross_method_max_budget": save_dir / "cross_method_max_budget.png",
        "cross_method_convergence": save_dir / "cross_method_convergence.png",
        "cross_method_diagnostics": save_dir / "cross_method_diagnostics.png",
    }
    plot_cross_method_max_budget(
        saved["records"],
        scenario_name=str(metadata["scenario_display"]),
        scenario_key=str(metadata.get("scenario", "")) or None,
        exact_p=float(metadata["exact_p"]),
        max_budget=max_budget,
        n_control=n_control,
        n_treated=n_treated,
        beta_workflow=beta_workflow,
        save_path=out["cross_method_max_budget"],
    )
    plot_cross_method_convergence(
        saved["summary"],
        scenario_name=str(metadata["scenario_display"]),
        scenario_key=str(metadata.get("scenario", "")) or None,
        exact_p=float(metadata["exact_p"]),
        n_control=n_control,
        n_treated=n_treated,
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
