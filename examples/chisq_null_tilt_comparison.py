#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcmcis-matplotlib"))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D


def chisq_pdf(x: np.ndarray, *, df: float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    out = np.zeros_like(x_arr, dtype=float)
    positive = x_arr > 0.0
    half_df = 0.5 * float(df)
    log_norm = half_df * math.log(2.0) + math.lgamma(half_df)
    out[positive] = np.exp((half_df - 1.0) * np.log(x_arr[positive]) - 0.5 * x_arr[positive] - log_norm)
    if float(df) == 2.0:
        out[x_arr == 0.0] = 0.5
    return out


def density_bin_probs(bin_edges: np.ndarray, x_grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    idx = bin_index(x_grid, bin_edges)
    out = np.zeros(bin_edges.size - 1, dtype=float)
    for i in range(out.size):
        mask = idx == i
        if np.any(mask):
            out[i] = float(np.trapezoid(density[mask], x_grid[mask]))
    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out


def tail_prob_grid(*, threshold: float, x_grid: np.ndarray, density: np.ndarray) -> float:
    tail = x_grid >= float(threshold)
    total = float(np.trapezoid(density, x_grid))
    return float(np.trapezoid(density[tail], x_grid[tail]) / total)


def mcmcis_g(x: np.ndarray, *, z_threshold: float, beta: float, sigma: float) -> np.ndarray:
    shortfall = np.maximum((float(z_threshold) - np.asarray(x, dtype=float)) / float(sigma), 0.0)
    return np.exp(-float(beta) * shortfall)


def bin_index(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bin_edges, values, side="right") - 1
    return np.clip(idx, 0, bin_edges.size - 2).astype(int)


def samc_bin_edges(*, z_threshold: float, lambda_min: float, n_bins: int) -> np.ndarray:
    finite_edges = np.linspace(float(lambda_min), float(z_threshold), int(n_bins), dtype=float)
    return np.concatenate([finite_edges, np.asarray([np.inf], dtype=float)])


def tilted_tail_mass_grid(
    *,
    z_threshold: float,
    beta: float,
    sigma: float,
    x_grid: np.ndarray,
    df: float,
) -> float:
    f = chisq_pdf(x_grid, df=df)
    g = mcmcis_g(x_grid, z_threshold=z_threshold, beta=beta, sigma=sigma)
    z_norm = float(np.trapezoid(f * g, x_grid))
    tail = x_grid >= float(z_threshold)
    return float(np.trapezoid((f * g)[tail], x_grid[tail]) / z_norm)


def solve_beta_for_tail_occupancy(
    *,
    z_threshold: float,
    target_tail_occupancy: float,
    null_tail_probability: float,
    sigma: float,
    x_grid: np.ndarray,
    df: float,
) -> float:
    target = float(target_tail_occupancy)
    if not (float(null_tail_probability) < target < 1.0):
        raise ValueError("target_tail_occupancy must lie between the null tail probability and 1.")

    lo = 0.0
    hi = 1.0
    while tilted_tail_mass_grid(z_threshold=z_threshold, beta=hi, sigma=sigma, x_grid=x_grid, df=df) < target:
        hi *= 2.0
        if hi > 1_000.0:
            raise RuntimeError("Could not bracket MCMC-IS beta.")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        q_mid = tilted_tail_mass_grid(z_threshold=z_threshold, beta=mid, sigma=sigma, x_grid=x_grid, df=df)
        if q_mid < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def mcmcis_bin_probs(
    *,
    bin_edges: np.ndarray,
    z_threshold: float,
    beta: float,
    sigma: float,
    x_grid: np.ndarray,
    df: float,
) -> np.ndarray:
    f = chisq_pdf(x_grid, df=df)
    g = mcmcis_g(x_grid, z_threshold=z_threshold, beta=beta, sigma=sigma)
    density = f * g
    density /= float(np.trapezoid(density, x_grid))
    return density_bin_probs(bin_edges, x_grid, density)


def _format_bin_label(i: int, n_bins: int) -> str:
    return "Tail" if i == n_bins - 1 else f"B{i + 1}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def make_figure(
    *,
    z_threshold: float,
    chi_square_df: float,
    n_bins: int,
    lambda_min: float,
    mcmcis_gamma: float,
    sigma: float,
    g_xmin_multiplier: float,
    g_xmax_multiplier: float,
    output_png: Path,
    output_pdf: Path | None,
    summary_json: Path | None,
    g_output_png: Path | None,
    g_output_pdf: Path | None,
    pi_output_png: Path | None,
    pi_output_pdf: Path | None,
) -> dict[str, Any]:
    upper = max(float(z_threshold) * 3.0, float(z_threshold) + 12.0 * math.sqrt(2.0 * float(chi_square_df)))
    x = np.linspace(0.0, upper, 7000)
    x_integral = np.linspace(0.0, upper, 140_000)
    f = chisq_pdf(x, df=chi_square_df)
    f_integral = chisq_pdf(x_integral, df=chi_square_df)
    p0 = tail_prob_grid(threshold=z_threshold, x_grid=x_integral, density=f_integral)
    mcmcis_target = float(p0 ** float(mcmcis_gamma))
    beta = solve_beta_for_tail_occupancy(
        z_threshold=z_threshold,
        target_tail_occupancy=mcmcis_target,
        null_tail_probability=p0,
        sigma=sigma,
        x_grid=x_integral,
        df=chi_square_df,
    )

    g_mcmc = mcmcis_g(x, z_threshold=z_threshold, beta=beta, sigma=sigma)
    g_mcmc_integral = mcmcis_g(x_integral, z_threshold=z_threshold, beta=beta, sigma=sigma)
    z_mcmc = float(np.trapezoid(f_integral * g_mcmc_integral, x_integral))
    pi_mcmc = f * g_mcmc / z_mcmc

    edges = samc_bin_edges(z_threshold=z_threshold, lambda_min=lambda_min, n_bins=n_bins)
    null_bin_probs = density_bin_probs(edges, x_integral, f_integral)
    target_bin_probs = np.full(int(n_bins), 1.0 / float(n_bins), dtype=float)
    g_samc_by_bin = target_bin_probs / null_bin_probs
    idx = bin_index(x, edges)
    g_samc = g_samc_by_bin[idx]
    g_samc_integral = g_samc_by_bin[bin_index(x_integral, edges)]
    z_samc = float(np.trapezoid(f_integral * g_samc_integral, x_integral))
    pi_samc = f * g_samc / z_samc

    mcmc_bin_probs = mcmcis_bin_probs(
        bin_edges=edges,
        z_threshold=z_threshold,
        beta=beta,
        sigma=sigma,
        x_grid=x_integral,
        df=chi_square_df,
    )

    colors = {
        "null": "#4f6d8a",
        "mcmc": "#c48a3a",
        "samc": "#4c8c77",
        "threshold": "#7b2d26",
        "grid": "#d8dde1",
        "dark": "#222222",
    }

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.6), constrained_layout=True)
    fig.suptitle(
        rf"How SAMC and MCMC-IS tilt a $\chi^2_{{{chi_square_df:g}}}$ null toward a right tail",
        fontsize=18,
        fontweight="semibold",
    )

    ax = axes[0, 0]
    ax.plot(x, f, color=colors["null"], linewidth=2.2, label=rf"null density $f=\chi^2_{{{chi_square_df:g}}}$")
    tail_mask = x >= z_threshold
    ax.fill_between(x[tail_mask], 0.0, f[tail_mask], color=colors["threshold"], alpha=0.18, label="tail event")
    for edge in edges[1:-1]:
        ax.axvline(float(edge), color=colors["grid"], linewidth=0.9, zorder=0)
    ax.axvline(z_threshold, color=colors["threshold"], linewidth=2.0)
    ax.text(z_threshold + 0.15, max(f) * 0.82, rf"$t^\star={z_threshold:.1f}$", color=colors["threshold"])
    ax.set_title("1. Null distribution and statistic bins")
    ax.set_xlabel("test statistic t")
    ax.set_ylabel("density")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(0.0, max(f) * 1.08)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    g_mcmc_norm = g_mcmc / float(np.max(g_mcmc))
    g_samc_norm = g_samc / float(np.max(g_samc))
    ax.plot(x, g_mcmc_norm, color=colors["mcmc"], linewidth=2.4, label="MCMC-IS smooth hinge")
    ax.step(x, g_samc_norm, where="post", color=colors["samc"], linewidth=2.2, label="SAMC binwise tilt")
    ax.axvline(z_threshold, color=colors["threshold"], linewidth=1.7)
    ax.set_yscale("log")
    ax.set_title(r"2. Tilting functions $g(t)$, normalized")
    ax.set_xlabel("test statistic t")
    ax.set_ylabel("relative tilt, log scale")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(max(1e-4, min(np.nanmin(g_mcmc_norm), np.nanmin(g_samc_norm)) * 0.7), 1.5)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    ax.plot(x, f, color=colors["null"], linewidth=2.0, label="null f")
    ax.plot(x, pi_mcmc, color=colors["mcmc"], linewidth=2.35, label=rf"MCMC-IS $\pi$; tail={mcmcis_target:.2f}")
    ax.plot(x, pi_samc, color=colors["samc"], linewidth=2.25, label=rf"SAMC $\pi$; tail={1.0 / n_bins:.2f}")
    ax.fill_between(x[tail_mask], 0.0, pi_mcmc[tail_mask], color=colors["mcmc"], alpha=0.12)
    ax.fill_between(x[tail_mask], 0.0, pi_samc[tail_mask], color=colors["samc"], alpha=0.12)
    ax.axvline(z_threshold, color=colors["threshold"], linewidth=1.7)
    ax.set_title(r"3. Resulting importance distributions $\pi \propto f g$")
    ax.set_xlabel("test statistic t")
    ax.set_ylabel("density")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    bar_x = np.arange(n_bins)
    width = 0.25
    ax.bar(bar_x - width, null_bin_probs, width=width, color=colors["null"], alpha=0.86, label="null bin mass")
    ax.bar(bar_x, mcmc_bin_probs, width=width, color=colors["mcmc"], alpha=0.86, label="MCMC-IS bin mass")
    ax.bar(bar_x + width, target_bin_probs, width=width, color=colors["samc"], alpha=0.86, label="SAMC target mass")
    ax.axhline(1.0 / n_bins, color=colors["samc"], linestyle="--", linewidth=1.2)
    ax.set_title("4. Bin masses after tilting")
    ax.set_xlabel("SAMC statistic bin")
    ax.set_ylabel("probability")
    ax.set_xticks(bar_x)
    ax.set_xticklabels([_format_bin_label(i, n_bins) for i in range(n_bins)], rotation=0)
    ax.set_ylim(0.0, max(0.36, float(np.max([null_bin_probs.max(), mcmc_bin_probs.max(), target_bin_probs.max()])) * 1.25))
    ax.legend(frameon=False, loc="upper left")

    for ax in axes.flat:
        ax.grid(axis="y", color="#e6eaed", linewidth=0.9)
        ax.spines["left"].set_color("#bfc7cd")
        ax.spines["bottom"].set_color("#bfc7cd")

    formula_handles = [
        Line2D([0], [0], color=colors["mcmc"], linewidth=2.4),
        Line2D([0], [0], color=colors["samc"], linewidth=2.4),
    ]
    formula_labels = [
        rf"MCMC-IS: $g(t)=\exp[-\beta((t^\star-t)_+/\sigma)]$, $\beta={beta:.2f}$",
        rf"SAMC: $g(t)=1/(M\,P_f(B_i))$ for $t\in B_i$, $M={n_bins}$",
    ]
    fig.legend(
        formula_handles,
        formula_labels,
        frameon=False,
        loc="outside lower center",
        ncol=1,
        fontsize=11.5,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "output_png": output_png,
        "output_pdf": output_pdf,
        "g_output_png": g_output_png,
        "g_output_pdf": g_output_pdf,
        "pi_output_png": pi_output_png,
        "pi_output_pdf": pi_output_pdf,
        "null_distribution": f"chi_square(df={float(chi_square_df):g})",
        "chi_square_df": float(chi_square_df),
        "statistic_threshold": float(z_threshold),
        "null_tail_probability": float(p0),
        "mcmcis_gamma": float(mcmcis_gamma),
        "mcmcis_target_tail_occupancy": float(mcmcis_target),
        "mcmcis_beta": float(beta),
        "mcmcis_sigma": float(sigma),
        "mcmcis_normalizing_constant": float(z_mcmc),
        "oracle_tail_log_target_null_ratio": float(math.log(1.0 / p0)),
        "mcmcis_tail_occupancy_check": tilted_tail_mass_grid(
            z_threshold=z_threshold,
            beta=beta,
            sigma=sigma,
            x_grid=x_integral,
            df=chi_square_df,
        ),
        "samc_n_bins": int(n_bins),
        "samc_tail_occupancy": float(1.0 / n_bins),
        "samc_lambda_min": float(lambda_min),
        "samc_normalizing_constant": float(z_samc),
        "samc_tail_display": "held_at_terminal_pre_tail_level_in_focused_log_tilt_schematic",
        "g_xmin_multiplier": float(g_xmin_multiplier),
        "g_xmax_multiplier": float(g_xmax_multiplier),
        "samc_bin_edges": edges,
        "null_bin_probs": null_bin_probs,
        "mcmcis_bin_probs": mcmc_bin_probs,
        "samc_bin_probs": target_bin_probs,
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    if g_output_png is not None:
        make_g_focused_figure(
            x=x,
            z_threshold=z_threshold,
            beta=beta,
            sigma=sigma,
            n_bins=n_bins,
            g_mcmc=g_mcmc,
            g_samc=g_samc,
            z_mcmc=z_mcmc,
            z_samc=z_samc,
            null_tail_probability=p0,
            edges=edges,
            xmin=float(z_threshold) * float(g_xmin_multiplier),
            xmax=float(z_threshold) * float(g_xmax_multiplier),
            output_png=g_output_png,
            output_pdf=g_output_pdf,
        )
    if pi_output_png is not None:
        make_pi_focused_figure(
            x=x,
            z_threshold=z_threshold,
            f=f,
            pi_mcmc=pi_mcmc,
            pi_samc=pi_samc,
            xmax=float(z_threshold) * float(g_xmax_multiplier),
            output_png=pi_output_png,
            output_pdf=pi_output_pdf,
        )
    return summary


def make_g_focused_figure(
    *,
    x: np.ndarray,
    z_threshold: float,
    beta: float,
    sigma: float,
    n_bins: int,
    g_mcmc: np.ndarray,
    g_samc: np.ndarray,
    z_mcmc: float,
    z_samc: float,
    null_tail_probability: float,
    edges: np.ndarray,
    xmin: float,
    xmax: float,
    output_png: Path,
    output_pdf: Path | None,
) -> None:
    log_mcmc = np.log(np.asarray(g_mcmc, dtype=float) / float(z_mcmc))
    log_samc = np.log(np.asarray(g_samc, dtype=float) / float(z_samc))

    pre_tail = x < float(z_threshold)
    if np.any(pre_tail):
        last_pre_tail = int(np.flatnonzero(pre_tail)[-1])
        log_samc[x >= float(z_threshold)] = float(log_samc[last_pre_tail])
    log_oracle_tail = math.log(1.0 / float(null_tail_probability))

    colors = {
        "oracle": "#000000",
        "mcmc": "#c48a3a",
        "samc": "#4c8c77",
        "threshold": "#7b2d26",
    }

    plt.rcParams.update(
        {
            "font.size": 10.8,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.85, 3.85), constrained_layout=True)
    visible_xmin = max(float(np.min(x)), float(xmin))
    visible_xmax = min(float(np.max(x)), float(xmax))
    visible = (x >= visible_xmin) & (x <= visible_xmax)
    finite_visible = np.concatenate([log_mcmc[visible], log_samc[visible]])
    y_bottom = math.floor(float(np.min(finite_visible)) - 0.25)
    y_top = math.ceil(float(max(np.max(finite_visible), log_oracle_tail)) + 0.25)
    oracle_pre_tail = y_bottom + 0.32

    ax.step(
        [float(np.min(x)), float(z_threshold), visible_xmax],
        [oracle_pre_tail, log_oracle_tail, log_oracle_tail],
        where="post",
        color=colors["oracle"],
        linewidth=1.75,
        alpha=1.0,
        label="Step IS",
        zorder=3,
    )
    ax.plot(x, log_mcmc, color=colors["mcmc"], linewidth=2.7, label="MCMC-IS", zorder=4)
    ax.step(x, log_samc, where="post", color=colors["samc"], linewidth=2.35, label="SAMC", zorder=5)
    threshold_line = ax.axvline(
        float(z_threshold),
        color=colors["threshold"],
        linestyle="--",
        linewidth=1.75,
        alpha=0.98,
        zorder=10,
        label=r"$t_{\mathrm{obs}}$",
    )
    threshold_line.set_path_effects([pe.Stroke(linewidth=4.3, foreground="white"), pe.Normal()])
    ax.set_title("")
    ax.set_xlabel(r"Test statistic $t$")
    ax.set_ylabel(r"$\log\{g(t)\}$")
    ax.set_xlim(visible_xmin, visible_xmax)
    ax.set_ylim(y_bottom, y_top)
    ax.grid(axis="y", color="#e6eaed", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bfc7cd")
    ax.spines["bottom"].set_color("#bfc7cd")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("Step IS", "MCMC-IS", "SAMC", r"$t_{\mathrm{obs}}$")]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="upper left",
        handlelength=2.2,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=450, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def make_pi_focused_figure(
    *,
    x: np.ndarray,
    z_threshold: float,
    f: np.ndarray,
    pi_mcmc: np.ndarray,
    pi_samc: np.ndarray,
    xmax: float,
    output_png: Path,
    output_pdf: Path | None,
) -> None:
    colors = {
        "null": "#7c8790",
        "mcmc": "#c48a3a",
        "samc": "#4c8c77",
        "threshold": "#7b2d26",
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.35), constrained_layout=True)
    ax.plot(x, f, color=colors["null"], linewidth=2.0, alpha=0.62, label="null")
    ax.plot(x, pi_mcmc, color=colors["mcmc"], linewidth=3.0, label="MCMC-IS")
    ax.plot(x, pi_samc, color=colors["samc"], linewidth=2.6, label="SAMC")
    ax.axvline(
        float(z_threshold),
        color=colors["threshold"],
        linestyle="--",
        linewidth=1.8,
        zorder=1,
        label="threshold",
    )
    ax.set_title("Tilted Statistic Distributions", fontsize=15.5, fontweight="semibold", pad=10)
    ax.set_xlabel("test statistic")
    ax.set_ylabel("density")

    visible_xmax = min(float(np.max(x)), float(xmax))
    visible = x <= visible_xmax
    y_top = 1.08 * float(max(np.max(f[visible]), np.max(pi_mcmc[visible]), np.max(pi_samc[visible])))
    ax.set_xlim(float(np.min(x)), visible_xmax)
    ax.set_ylim(0.0, y_top)
    ax.grid(axis="y", color="#e6eaed", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bfc7cd")
    ax.spines["bottom"].set_color("#bfc7cd")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("null", "MCMC-IS", "SAMC", "threshold")]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="upper right",
        fontsize=11.5,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize SAMC and MCMC-IS tilts under a chi-squared null.")
    parser.add_argument("--chi-square-df", type=float, default=4.0)
    parser.add_argument(
        "--statistic-threshold",
        "--z-threshold",
        dest="statistic_threshold",
        type=float,
        default=18.0,
    )
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--mcmcis-gamma", type=float, default=1.0 / 3.0)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--g-xmin-multiplier", type=float, default=0.25)
    parser.add_argument("--g-xmax-multiplier", type=float, default=1.5)
    parser.add_argument("--output-png", type=Path, default=Path("results/chisq_null_tilt_comparison.png"))
    parser.add_argument("--output-pdf", type=Path, default=Path("results/chisq_null_tilt_comparison.pdf"))
    parser.add_argument("--g-output-png", type=Path, default=Path("results/chisq_null_tilt_functions.png"))
    parser.add_argument("--g-output-pdf", type=Path, default=Path("results/chisq_null_tilt_functions.pdf"))
    parser.add_argument("--pi-output-png", type=Path, default=Path("results/chisq_null_tilt_distributions.png"))
    parser.add_argument("--pi-output-pdf", type=Path, default=Path("results/chisq_null_tilt_distributions.pdf"))
    parser.add_argument("--summary-json", type=Path, default=Path("results/chisq_null_tilt_comparison_summary.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if int(args.n_bins) < 2:
        raise ValueError("--n-bins must be at least 2.")
    if float(args.chi_square_df) < 2.0:
        raise ValueError("--chi-square-df must be at least 2 for a finite lower-bound density in this visualization.")
    if float(args.lambda_min) >= float(args.statistic_threshold):
        raise ValueError("--lambda-min must be smaller than --statistic-threshold.")
    sigma = float(args.sigma) if args.sigma is not None else math.sqrt(2.0 * float(args.chi_square_df))
    if sigma <= 0.0:
        raise ValueError("--sigma must be positive.")
    if not (0.0 <= float(args.g_xmin_multiplier) < 1.0):
        raise ValueError("--g-xmin-multiplier must be in [0, 1).")
    if float(args.g_xmax_multiplier) <= 1.0:
        raise ValueError("--g-xmax-multiplier must be larger than 1.")
    summary = make_figure(
        z_threshold=float(args.statistic_threshold),
        chi_square_df=float(args.chi_square_df),
        n_bins=int(args.n_bins),
        lambda_min=float(args.lambda_min),
        mcmcis_gamma=float(args.mcmcis_gamma),
        sigma=sigma,
        g_xmin_multiplier=float(args.g_xmin_multiplier),
        g_xmax_multiplier=float(args.g_xmax_multiplier),
        output_png=Path(args.output_png),
        output_pdf=Path(args.output_pdf) if args.output_pdf is not None else None,
        summary_json=Path(args.summary_json) if args.summary_json is not None else None,
        g_output_png=Path(args.g_output_png) if args.g_output_png is not None else None,
        g_output_pdf=Path(args.g_output_pdf) if args.g_output_pdf is not None else None,
        pi_output_png=Path(args.pi_output_png) if args.pi_output_png is not None else None,
        pi_output_pdf=Path(args.pi_output_pdf) if args.pi_output_pdf is not None else None,
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
