from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from perm_pval.core.problem import PermutationTestProblem
from perm_pval.experiments.exact_scenarios import ExactScenario, load_saved_exact_scenarios
from perm_pval.methods.beta_tuning import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    make_short_chain_q_runner,
    tune_beta_to_target_q,
)
from perm_pval.methods.mcmc_is import run_mcmc_is
from perm_pval.methods.random_sampling import run_random_sampling
from perm_pval.methods.samc import run_samc


@dataclass(frozen=True)
class MCMCWorkflowConfig:
    use_true_p0_for_q_target: bool = True
    p0_guess: float = 1e-8
    d_alpha: float = 0.25
    pilot_samples: int = 20_000
    scale_method: str = "sd"
    beta_max_init: float = 1e6
    tune_steps: int = 12_000
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
    local_scan_multipliers: tuple[float, ...] = (0.70, 0.90, 1.00, 1.15, 1.35)
    local_scan_total_steps: int = 80_000
    local_scan_chains: int = 2
    local_scan_burn_in_fraction: float = 0.20
    local_scan_thin: int = 1


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


@dataclass(frozen=True)
class BetaSweepStudyConfig:
    estimation_points: tuple[int, ...]
    repeats: int = 5
    beta_multipliers: tuple[float, ...] = (0.70, 0.90, 1.00, 1.15, 1.35)
    chains: int = 2
    burn_in_fraction: float = 0.20
    thin: int = 1
    estimate_variance: bool = True
    obm_batch_size: int | None = None
    tilt_mode: str = "smooth_hinge"
    proposal_fraction: float = 0.075
    proposal_swaps: int | None = None
    base_seed: int = 54_321


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


def create_timestamped_run_dir(root: Path, prefix: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{ts}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def samc_variance_proxy(p_hat: float, n_steps: int, burn_in: int) -> float:
    n_eff = max(int(n_steps - burn_in), 1)
    return float(max(p_hat * (1.0 - p_hat) / n_eff, 0.0))


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
    sigma_t: float,
    seed: int,
) -> dict[str, Any]:
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

    betas = [
        float(beta_center * mult)
        for mult in cfg.local_scan_multipliers
        if float(beta_center * mult) > 0.0
    ]
    steps_per_chain = _steps_per_chain(cfg.local_scan_total_steps, cfg.local_scan_chains)
    burn_in = _burn_in(steps_per_chain, cfg.local_scan_burn_in_fraction)

    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for idx, beta in enumerate(betas):
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
        score = float(np.log(var_hat)) if np.isfinite(var_hat) and var_hat > 0.0 else float("inf")
        rows.append(
            {
                "beta": float(beta),
                "estimate": float(res.estimate),
                "variance_estimate": var_hat,
                "q_hat": float(res.tail_share_raw_sample),
                "ess": float(res.ess),
                "acceptance_rate": float(res.overall_acceptance_rate),
                "weight_cv": float(res.weight_summary.cv),
                "score": score,
            }
        )

    finite_rows = [row for row in rows if np.isfinite(row["score"])]
    if finite_rows:
        best = min(finite_rows, key=lambda row: row["score"])
        selected_beta = float(best["beta"])
        selected_reason = "min_estimated_variance"
    else:
        selected_beta = float(beta_center)
        selected_reason = "fallback_beta_center"

    return {
        "enabled": True,
        "selected_beta": selected_beta,
        "selected_reason": selected_reason,
        "steps_per_chain": int(steps_per_chain),
        "burn_in": int(burn_in),
        "scan_eval_total": int(len(betas) * _mcmc_eval_count(steps_per_chain, cfg.local_scan_chains)),
        "scan_wall_time_sec": float(time.perf_counter() - t_start),
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
    tuning_eval_total = int(cfg.pilot_samples + n_short_calls * (cfg.tune_steps + 1))
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
        scan = local_beta_scan(problem, cfg, beta_center=beta_tuned, sigma_t=float(sigma_t), seed=seed + 7)
        beta_used = float(scan["selected_beta"])

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
        "tuning_eval_total": int(tuning_eval_total),
        "tuning_wall_time_sec": tuning_wall_time_sec,
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


def run_cross_method_checkpoint(
    problem: PermutationTestProblem,
    exact_p: float,
    *,
    checkpoint: int,
    rep_seed: int,
    beta_workflow: dict[str, Any],
    samc_setup: dict[str, Any],
    cross_cfg: CrossMethodStudyConfig,
    mcmc_cfg: MCMCWorkflowConfig,
    samc_cfg: SAMCWorkflowConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoint = int(checkpoint)

    t0 = time.perf_counter()
    iid = run_random_sampling(problem, n_samples=checkpoint, seed=rep_seed, confidence_level=cross_cfg.confidence_level)
    iid_dt = float(time.perf_counter() - t0)
    rows.append(
        _annotate_error_fields(
            {
                "method": "iid",
                "checkpoint": checkpoint,
                "estimate": float(iid.estimate),
                "variance_estimate": float(iid.standard_error ** 2),
                "tail_hits": int(iid.tail_hits),
                "tail_share_raw": float(iid.estimate),
                "zero_hits": int(iid.tail_hits == 0),
                "wall_time_sec": iid_dt,
                "eval_excl_tuning": float(iid.n_samples),
                "ci_low": float(iid.ci_low),
                "ci_high": float(iid.ci_high),
            },
            exact_p,
        )
    )

    steps_per_chain = _steps_per_chain(checkpoint, mcmc_cfg.chains)
    burn_in = _burn_in(steps_per_chain, mcmc_cfg.burn_in_fraction)
    t0 = time.perf_counter()
    mcmc = run_mcmc_is(
        problem,
        beta=float(beta_workflow["beta_used"]),
        sigma_t=float(beta_workflow["sigma_t"]),
        n_steps=steps_per_chain,
        burn_in=burn_in,
        thin=mcmc_cfg.thin,
        n_chains=mcmc_cfg.chains,
        seed=rep_seed + 1,
        init="random",
        tilt_mode=str(mcmc_cfg.tilt_mode),
        proposal_fraction=mcmc_cfg.proposal_fraction,
        proposal_swaps=mcmc_cfg.proposal_swaps,
        estimate_variance=bool(mcmc_cfg.estimate_variance),
        obm_batch_size=mcmc_cfg.obm_batch_size,
    )
    mcmc_dt = float(time.perf_counter() - t0)
    rows.append(
        _annotate_error_fields(
            {
                "method": "mcmc_is",
                "checkpoint": checkpoint,
                "estimate": float(mcmc.estimate),
                "variance_estimate": (
                    float(mcmc.snis_variance_obm)
                    if mcmc.snis_variance_obm is not None and np.isfinite(mcmc.snis_variance_obm)
                    else np.nan
                ),
                "snis_mcse_obm": (
                    float(mcmc.snis_mcse_obm)
                    if mcmc.snis_mcse_obm is not None and np.isfinite(mcmc.snis_mcse_obm)
                    else np.nan
                ),
                "tail_hits": int(mcmc.tail_hits_weighted_sample),
                "tail_share_raw": float(mcmc.tail_share_raw_sample),
                "ess": float(mcmc.ess),
                "acceptance_rate": float(mcmc.overall_acceptance_rate),
                "weight_cv": float(mcmc.weight_summary.cv),
                "beta": float(mcmc.beta),
                "sigma_t": float(mcmc.sigma_t),
                "tilt_mode": str(mcmc.tilt_mode),
                "wall_time_sec": mcmc_dt,
                "eval_excl_tuning": float(_mcmc_eval_count(steps_per_chain, mcmc_cfg.chains)),
                "n_weighted_samples": int(mcmc.n_weighted_samples),
            },
            exact_p,
        )
    )

    samc_burn_in = _burn_in(checkpoint, samc_cfg.burn_in_fraction)
    t0 = time.perf_counter()
    samc = run_samc(
        problem,
        n_steps=checkpoint,
        burn_in=samc_burn_in,
        bin_edges=samc_setup["bin_edges"],
        seed=rep_seed + 2,
        init="random",
        t0=samc_cfg.t0,
        trace_every=samc_cfg.trace_every,
        proposal_fraction=samc_cfg.proposal_fraction,
        proposal_swaps=samc_cfg.proposal_swaps,
        convergence_tolerance=samc_cfg.convergence_tolerance,
    )
    samc_dt = float(time.perf_counter() - t0)
    rows.append(
        _annotate_error_fields(
            {
                "method": "samc",
                "checkpoint": checkpoint,
                "estimate": float(samc.estimate),
                "variance_estimate": samc_variance_proxy(float(samc.estimate), checkpoint, samc_burn_in),
                "acceptance_rate": float(samc.acceptance_rate),
                "samc_max_rel_freq_error": float(samc.max_abs_relative_frequency_error),
                "samc_converged": int(samc.convergence_reached),
                "samc_pi0": float(samc.pi0_adjustment),
                "samc_empty_bins": int(samc.empty_bin_indices.size),
                "wall_time_sec": samc_dt,
                "eval_excl_tuning": float(checkpoint + 1),
            },
            exact_p,
        )
    )

    return rows


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

    records: list[dict[str, Any]] = []
    for rep in range(cross_cfg.repeats):
        rep_seed = int(cross_cfg.base_seed + 1_000 * rep)
        for checkpoint in checkpoints:
            rows = run_cross_method_checkpoint(
                scenario.problem,
                scenario.exact_p,
                checkpoint=checkpoint,
                rep_seed=rep_seed,
                beta_workflow=beta_workflow,
                samc_setup=samc_setup,
                cross_cfg=cross_cfg,
                mcmc_cfg=mcmc_cfg,
                samc_cfg=samc_cfg,
            )
            for row in rows:
                row["scenario"] = scenario.key
                row["scenario_display"] = scenario.description
                row["replicate"] = int(rep)
                records.append(row)

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
    var_data = []
    rmse_data = []
    for method in methods:
        sub = [row for row in rows if row["method"] == method]
        est = np.asarray([row["estimate"] for row in sub], dtype=float)
        var_hat = np.asarray([row["variance_estimate"] for row in sub], dtype=float)
        rse = np.asarray([row["root_squared_error"] for row in sub], dtype=float)
        if method == "iid":
            tail_hits = np.asarray([row.get("tail_hits", 0) for row in sub], dtype=int)
            est[tail_hits <= 0] = np.nan
            var_hat[tail_hits <= 0] = np.nan
            rse[tail_hits <= 0] = np.nan
        est_data.append(_positive_for_plot(est))
        var_data.append(_positive_for_plot(var_hat))
        rmse_data.append(_positive_for_plot(rse))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    axes[0].boxplot(est_data, tick_labels=labels, showfliers=False)
    axes[0].set_yscale("log")
    axes[0].axhline(exact_p, color="black", linestyle="--", linewidth=1.2, label=f"true p={exact_p:.2e}")
    _set_log_ylim(axes[0], est_data)
    axes[0].set_title("Estimator distribution")
    axes[0].set_ylabel("p_hat")
    axes[0].legend()

    axes[1].boxplot(var_data, tick_labels=labels, showfliers=False)
    axes[1].set_yscale("log")
    _set_log_ylim(axes[1], var_data)
    axes[1].set_title("Variance estimate distribution")
    axes[1].set_ylabel("var_hat")

    axes[2].boxplot(rmse_data, tick_labels=labels, showfliers=False)
    axes[2].set_yscale("log")
    _set_log_ylim(axes[2], rmse_data)
    axes[2].set_title("RMSE across repeats")
    axes[2].set_ylabel("root squared error")

    title = f"Cross-method comparison: {scenario_name}\nmax iterations={max_budget:,}, true p={exact_p:.3e}"
    if beta_workflow is not None:
        title = (
            f"{title}\nMCMC-IS beta (laplace/tuned/used): "
            f"{beta_workflow['beta0_laplace']:.4g} / {beta_workflow['beta_hat_tuned']:.4g} / {beta_workflow['beta_used']:.4g}"
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
    save_path: Path | None = None,
) -> None:
    methods = ["iid", "mcmc_is", "samc"]
    labels = {"iid": "IID", "mcmc_is": "MCMC-IS", "samc": "SAMC"}
    colors = {"iid": "#4e79a7", "mcmc_is": "#f28e2b", "samc": "#59a14f"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for method in methods:
        sub = sorted([row for row in summary if row["method"] == method], key=lambda row: row["checkpoint"])
        x = np.asarray([row["checkpoint"] for row in sub], dtype=float)
        mean_est = np.asarray([row["mean_estimate"] for row in sub], dtype=float)
        rmse = np.asarray([row["rmse"] for row in sub], dtype=float)
        mean_var = np.asarray([row["mean_variance_estimate"] for row in sub], dtype=float)

        axes[0].plot(x, mean_est, marker="o", label=labels[method], color=colors[method])
        axes[1].plot(x, rmse, marker="o", label=labels[method], color=colors[method])
        axes[2].plot(x, mean_var, marker="o", label=labels[method], color=colors[method])

    axes[0].axhline(exact_p, color="black", linestyle="--", linewidth=1.2, label=f"true p={exact_p:.2e}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("iterations")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].set_title("Mean estimate vs iterations")
    axes[0].set_ylabel("p_hat")
    axes[1].set_title("RMSE vs iterations")
    axes[1].set_ylabel("RMSE")
    axes[2].set_title("Mean variance estimate vs iterations")
    axes[2].set_ylabel("var_hat")
    axes[0].legend()
    fig.suptitle(f"Cross-method convergence: {scenario_name} (true p={exact_p:.3e})")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cross_method_diagnostics(
    summary: list[dict[str, Any]],
    *,
    scenario_name: str,
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

    fig.suptitle(f"Cross-method diagnostics by checkpoint: {scenario_name}")
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

    for beta_idx, multiplier in enumerate(beta_cfg.beta_multipliers):
        beta = float(beta_center * multiplier)
        for rep in range(beta_cfg.repeats):
            rep_seed = int(beta_cfg.base_seed + 10_000 * beta_idx + 100 * rep)
            for checkpoint in checkpoints:
                steps_per_chain = _steps_per_chain(checkpoint, beta_cfg.chains)
                burn_in = _burn_in(steps_per_chain, beta_cfg.burn_in_fraction)
                t_start = time.perf_counter()
                res = run_mcmc_is(
                    problem,
                    beta=beta,
                    sigma_t=sigma_t,
                    n_steps=steps_per_chain,
                    burn_in=burn_in,
                    thin=beta_cfg.thin,
                    n_chains=beta_cfg.chains,
                    seed=rep_seed,
                    init="random",
                    tilt_mode=str(beta_cfg.tilt_mode),
                    proposal_fraction=beta_cfg.proposal_fraction,
                    proposal_swaps=beta_cfg.proposal_swaps,
                    estimate_variance=bool(beta_cfg.estimate_variance),
                    obm_batch_size=beta_cfg.obm_batch_size,
                )
                wall_time_sec = float(time.perf_counter() - t_start)
                row = {
                    "checkpoint": int(checkpoint),
                    "beta_multiplier": float(multiplier),
                    "beta": float(beta),
                    "replicate": int(rep),
                    "seed": int(rep_seed),
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
                    "q_tilt_tail_share": float(res.tail_share_raw_sample),
                    "ess": float(res.ess),
                    "acceptance_rate": float(res.overall_acceptance_rate),
                    "tail_hits": int(res.tail_hits_weighted_sample),
                    "n_weighted_samples": int(res.n_weighted_samples),
                    "weight_cv": float(res.weight_summary.cv),
                    "eval_excl_tuning": float(_mcmc_eval_count(steps_per_chain, beta_cfg.chains)),
                    "wall_time_sec": wall_time_sec,
                }
                _annotate_error_fields(row, exact_p)
                rows.append(row)

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
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(exact_p, color="black", linestyle="--", linewidth=1.2, label=f"true p={exact_p:.2e}")
    _set_log_ylim(axes[0, 0], est_data)
    axes[0, 0].set_title("Estimator across beta")
    axes[0, 0].set_ylabel("p_hat")
    axes[0, 0].legend()

    axes[0, 1].boxplot(var_data, tick_labels=labels, showfliers=False)
    axes[0, 1].set_yscale("log")
    _set_log_ylim(axes[0, 1], var_data)
    axes[0, 1].set_title("Estimated variance across beta")
    axes[0, 1].set_ylabel("var_hat")

    axes[0, 2].boxplot(q_data, tick_labels=labels, showfliers=False)
    axes[0, 2].set_title("Tilted-tail occupancy q")
    axes[0, 2].set_ylabel("q_hat")

    axes[1, 0].boxplot(ess_data, tick_labels=labels, showfliers=False)
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
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 2].set_yscale("log")
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
        save_path=output_dir / "cross_method_convergence.png",
    )
    plot_cross_method_diagnostics(
        study["summary"],
        scenario_name=study["scenario_display"],
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
