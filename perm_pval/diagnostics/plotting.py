from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_trace(values: Sequence[float], *, title: str = "", ylabel: str = "Value"):
    vals = np.asarray(values, dtype=float)
    fig, ax = plt.subplots()
    ax.plot(np.arange(vals.size), vals, linewidth=1.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_histogram(values: Iterable[float], *, bins: int = 30, title: str = "", xlabel: str = "Value"):
    vals = np.asarray(list(values), dtype=float)
    fig, ax = plt.subplots()
    ax.hist(vals, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    return fig, ax
