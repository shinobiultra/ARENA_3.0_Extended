"""Plotting helpers for [7.4] Mini Natural Language Autoencoders.

Nothing in here is part of the method you're learning: it's plotting code. The verbaliser, the
reconstructor, the losses, the metrics and the baselines all live in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_factor_heatmap(table: pd.DataFrame, title: str, control_row: str | None = None):
    """
    Heatmap of a (factor x layer) table of FVE values. If `control_row` is given, that row is drawn
    below a dividing line so the control is visually separated from the real factors.
    """
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(1.1 * table.shape[1] + 3.5, 0.45 * table.shape[0] + 1.6))
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=max(0.6, float(np.nanmax(values))), aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] < 0.35 else "black")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    if control_row is not None and control_row in table.index:
        k = list(table.index).index(control_row)
        ax.axhline(k - 0.5, color="white", lw=2)
    ax.set_xlabel("layer (residual stream before block)")
    ax.set_ylabel("what the description says")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="held-out FVE")
    plt.tight_layout()
    plt.show()


def plot_bars(
    values: dict[str, float],
    title: str,
    ylabel: str,
    ylim: tuple[float, float] = (0, 1.1),
    chance: float | None = None,
    n_highlight: int = 1,
    fmt: str = "{:.0%}",
):
    """Bar chart of a method (the first `n_highlight` bars, in blue) against its controls (grey)."""
    fig, ax = plt.subplots(figsize=(1.25 * len(values) + 2.5, 3.8))
    names, vals = list(values), list(values.values())
    colors = ["tab:blue"] * n_highlight + ["tab:gray"] * (len(vals) - n_highlight)
    bars = ax.bar(names, vals, color=colors)
    span = ylim[1] - ylim[0]
    for bar, v in zip(bars, vals):
        y = max(v, ylim[0]) + 0.02 * span
        ax.text(bar.get_x() + bar.get_width() / 2, y, fmt.format(v), ha="center", fontsize=9)
    if chance is not None:
        ax.axhline(chance, color="k", ls=":", lw=1, label=f"chance ({fmt.format(chance)})")
        ax.legend(loc="upper right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()


def plot_curves(
    curves: dict[str, tuple[list[float], list[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    hlines: dict[str, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """One line per entry of `curves` (name -> (xs, ys)), plus optional labelled horizontal reference lines."""
    fig, ax = plt.subplots(figsize=(8, 4))
    styles = ["-o", "--s", ":^", "-.v", "-d", "--x"]
    for (name, (xs, ys)), style in zip(curves.items(), styles):
        ax.plot(xs, ys, style, label=name, markersize=4)
    for (name, y), color in zip((hlines or {}).items(), ["gray", "tab:red", "tab:purple", "tab:brown"]):
        ax.axhline(y, color=color, ls="--", lw=1, label=name)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_budget_sweep(
    sizes: list[int],
    fve_curves: dict[str, list[float]],
    behaviour_curves: dict[str, list[float]],
    title: str,
):
    """Two panels: FVE and transplant accuracy, against the size of the bottleneck."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    styles = ["-o", "--s", ":^", "-.v", "-d"]
    for ax, curves, ylabel in [
        (ax0, fve_curves, "held-out FVE"),
        (ax1, behaviour_curves, "transplant accuracy"),
    ]:
        for (name, ys), style in zip(curves.items(), styles):
            xs = sizes[: len(ys)]
            ax.plot(xs, ys, style, label=name, markersize=5)
        ax.set_xticks(sizes)
        ax.set_xlabel("bottleneck size (words for the NLA, dimensions for PCA)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax0.set_ylim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.25, color="k", ls=":", lw=1)
    ax0.set_title("How much variance does the bottleneck keep?", fontsize=10)
    ax1.set_title("Does the bottleneck keep what the model uses?", fontsize=10)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_fve_vs_behaviour(points: dict[str, tuple], title: str, ceiling: float | None = None):
    """
    Scatter plot of FVE (x) against transplant accuracy (y). `points` maps a label to (fve, accuracy, kind) or
    (fve, accuracy, kind, (dx, dy, ha)), where kind is one of "nla", "oracle", "baseline" or "control" (which sets
    the colour and marker), and the optional last element positions the text label (offset in points, alignment).
    """
    style = {
        "nla": dict(color="tab:blue", marker="o", s=80),
        "oracle": dict(color="tab:green", marker="D", s=60),
        "baseline": dict(color="tab:orange", marker="s", s=60),
        "control": dict(color="tab:gray", marker="X", s=70),
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for name, (x, y, kind, *offset) in points.items():
        label = {"nla": "NLA (verbaliser → text → reconstructor)", "oracle": "description oracle (class means)",
                 "baseline": "baseline bottleneck", "control": "control"}[kind]
        ax.scatter([max(x, -0.25)], [y], label=None if kind in seen else label, **style[kind])
        seen.add(kind)
        dx, dy, ha = offset[0] if offset else (6, 4, "left")
        ax.annotate(name, (max(x, -0.25), y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8)
    if ceiling is not None:
        ax.axhline(ceiling, color="k", ls="--", lw=1, label=f"true activation ({ceiling:.0%})")
    ax.axhline(0.25, color="k", ls=":", lw=1, label="chance (25%)")
    ax.set_xlim(-0.3, 1.05)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("held-out FVE  (how much of the activation's variance is reconstructed; clipped at -0.25)")
    ax.set_ylabel("transplant accuracy  (does the model use the reconstruction like the original?)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()
