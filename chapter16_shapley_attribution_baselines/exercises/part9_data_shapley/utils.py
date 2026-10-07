"""Plotting and data-loading helpers for [16.9] Data Shapley in One Training Run.

Nothing in here is part of the methods you're learning: it's plotting code and the boring bits of data
loading. Every Shapley estimator, gradient computation and training loop lives in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from matplotlib.patches import Patch
from sklearn.datasets import load_digits

# One colour per method, used consistently in every figure of the section
COLORS = {
    "data_shapley": "#2a78d6",  # retraining-based Data Shapley (exact or Monte Carlo)
    "first": "#eb6834",  # In-Run, first order
    "second": "#1baf7a",  # In-Run, second order
    "exact_local": "#4a3aa7",  # In-Run, exact local games
    "loo": "#e87ba4",  # leave-one-out
    "control": "#8c8c8c",  # random / shuffled controls
    "control_dark": "#3a3a3a",
    "flipped": "#e34948",  # known label flips
}


def load_digit_splits(seed: int = 0, n_train: int = 64, n_val: int = 24, n_test: int = 32) -> dict[str, dict]:
    """
    Loads the 8x8 handwritten digits that ship with scikit-learn (a copy of the UCI optical digits test set),
    keeps only 1s and 7s, and makes disjoint train / validation / test splits with `n_*` images of each digit.

    Labels are 0 for a "1" and 1 for a "7". Pixel values are scaled to [0, 1]. Returns a dict mapping split
    name to a dict with keys "images" (n, 8, 8), "x" (n, 64), "y" (n,) and "digit" (n,).
    """
    digits = load_digits()
    rng = np.random.default_rng(seed)
    parts = {name: {"images": [], "y": [], "digit": []} for name in ("train", "val", "test")}
    for label, digit in [(0, 1), (1, 7)]:
        idx = rng.permutation(np.flatnonzero(digits.target == digit))
        start = 0
        for name, size in [("train", n_train), ("val", n_val), ("test", n_test)]:
            chosen = idx[start : start + size]
            start += size
            parts[name]["images"].append(digits.images[chosen] / 16.0)
            parts[name]["y"] += [label] * size
            parts[name]["digit"] += [digit] * size
    splits = {}
    for name, part in parts.items():
        images = t.tensor(np.concatenate(part["images"]), dtype=t.float64)
        splits[name] = dict(
            images=images,
            x=images.flatten(1),
            y=t.tensor(part["y"], dtype=t.float64),
            digit=t.tensor(part["digit"]),
        )
    return splits


def plot_toy_problem(train_x, train_y, val_x, val_y, names: list[str]):
    """Scatter plot of the 8-point toy training set (coloured by label) and its validation set."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    train_x, val_x = train_x.cpu().numpy(), val_x.cpu().numpy()
    for label, color, name in [(1, COLORS["data_shapley"], "label 1"), (0, COLORS["first"], "label 0")]:
        mask = train_y.cpu().numpy() == label
        ax.scatter(*train_x[mask].T, s=90, color=color, label=f"training point, {name}", zorder=3, edgecolor="white")
        vmask = val_y.cpu().numpy() == label
        ax.scatter(*val_x[vmask].T, s=60, marker="x", color=color, label=f"validation point, {name}", zorder=2)
    # Points that share a location are listed together
    by_location = {}
    for i, point in enumerate(train_x):
        by_location.setdefault(tuple(point.round(3)), []).append(i)
    for point, idxs in by_location.items():
        text = "\n".join(f"{i}: {names[i]}" for i in idxs)
        ax.annotate(text, point, xytext=(8, -4), textcoords="offset points", fontsize=8, va="top")
    ax.axhline(0, color="#d7dadf", lw=0.8, zorder=0)
    ax.axvline(0, color="#d7dadf", lw=0.8, zorder=0)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_xlim(-2.6, 3.9)
    ax.set_title("The toy data valuation problem: 8 training points, 8 validation points")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_value_bars(
    values: dict[str, t.Tensor],
    labels: list[str],
    title: str,
    ylabel: str = "credit (validation loss improvement)",
    colors: list[str] | None = None,
    highlight: list[int] | None = None,
    figsize=(10, 4.2),
):
    """Grouped bar chart: one group per player (training point), one bar per method."""
    fig, ax = plt.subplots(figsize=figsize)
    n_methods = len(values)
    width = 0.8 / n_methods
    positions = np.arange(len(labels))
    colors = colors or list(COLORS.values())
    for j, (name, vals) in enumerate(values.items()):
        offset = (j - (n_methods - 1) / 2) * width
        ax.bar(positions + offset, vals.detach().cpu().numpy(), width * 0.92, label=name, color=colors[j])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for i in highlight or []:
        ax.get_xticklabels()[i].set_color(COLORS["flipped"])
        ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_mc_convergence(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    exact_cost: int,
    loo_point: tuple[int, float],
    title: str,
):
    """
    Log-log plot of estimation error against the number of trainings used.

    `curves` maps a name to (mean number of trainings, mean error, low error, high error), one entry per budget.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [COLORS["data_shapley"], "#eda100", "#008300"]  # MC, then the two truncated variants
    for (name, (calls, mean, low, high)), color in zip(curves.items(), colors):
        ax.plot(calls, mean, "-o", color=color, label=name, markersize=4, lw=2)
        ax.fill_between(calls, low, high, color=color, alpha=0.15)
    ax.axvline(exact_cost, color=COLORS["control_dark"], ls="--", lw=1.2, label=f"exact enumeration ({exact_cost} trainings, error 0)")
    ax.scatter(*loo_point, marker="D", s=60, color=COLORS["loo"], zorder=5, label=f"leave-one-out ({loo_point[0]} trainings)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of models trained (utility evaluations)")
    ax.set_ylabel("max |estimate - exact Data Shapley|")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_digit_gallery(images: t.Tensor, titles: list[str], red_border: list[bool], suptitle: str, ncols: int = 8):
    """Grid of 8x8 digit images; a thick red border marks images with a known flipped label."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.45 * ncols, 2.0 * nrows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, image, title, red in zip(np.atleast_1d(axes).flat, images, titles, red_border):
        ax.axis("on")
        ax.imshow(image.cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["flipped"] if red else "#cccccc")
            spine.set_linewidth(3 if red else 1)
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmaps(
    matrices: dict[str, t.Tensor], row_boundaries: list[int], row_names: list[str], col_boundaries: list[int],
    col_names: list[str], suptitle: str,
):
    """Side-by-side heatmaps of (train x validation) gradient dot products, with group separators."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(6.2 * len(matrices), 5.6))
    for ax, (name, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        m = matrix.detach().cpu().numpy()
        lim = np.abs(m).max()
        im = ax.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        for b in row_boundaries:
            ax.axhline(b - 0.5, color="black", lw=1)
        for b in col_boundaries:
            ax.axvline(b - 0.5, color="black", lw=1)
        starts = [0] + row_boundaries
        ends = row_boundaries + [m.shape[0]]
        ax.set_yticks([(s + e - 1) / 2 for s, e in zip(starts, ends)])
        ax.set_yticklabels(row_names, fontsize=8)
        cstarts, cends = [0] + col_boundaries, col_boundaries + [m.shape[1]]
        ax.set_xticks([(s + e - 1) / 2 for s, e in zip(cstarts, cends)])
        ax.set_xticklabels(col_names, fontsize=8)
        ax.set_xlabel("validation image")
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, label="gradient dot product  g_train · g_val")
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_timing(n_params: list[int], times: dict[str, list[float]], title: str):
    """Log-log plot of wall-clock time against the number of model parameters."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, values), color in zip(times.items(), [COLORS["control_dark"], COLORS["second"]]):
        ax.plot(n_params, np.array(values) * 1000, "-o", label=name, color=color, lw=2, markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of model parameters")
    ax.set_ylabel("time per similarity matrix (ms)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_accounting(steps_curves: dict[str, np.ndarray], scatter: dict[str, tuple[t.Tensor, t.Tensor]], flipped: t.Tensor, title: str):
    """
    Left: cumulative credit allocated by each method against the actual improvement in validation loss.
    Right: per-example credit of approximate methods against the exact-local credit.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.6))
    styles = {
        "actual improvement in validation loss": dict(color="black", lw=3, alpha=0.35),
        "In-Run, exact local games": dict(color=COLORS["exact_local"], lw=2, ls="--"),
        "In-Run, first order": dict(color=COLORS["first"], lw=2),
        "In-Run, second order": dict(color=COLORS["second"], lw=2, ls=":"),
    }
    for name, curve in steps_curves.items():
        ax0.plot(np.arange(len(curve)), curve, label=name, **styles.get(name, {}))
    ax0.set_xlabel("SGD step")
    ax0.set_ylabel("cumulative credit (validation loss improvement)")
    ax0.set_title("Does the allocated credit add up to what actually happened?")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8)
    lo = min(float(min(x.min(), y.min())) for x, y in scatter.values())
    hi = max(float(max(x.max(), y.max())) for x, y in scatter.values())
    ax1.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="y = x (perfect agreement)")
    colors = {"In-Run, first order": COLORS["first"], "In-Run, second order": COLORS["second"]}
    for name, (x, y) in scatter.items():
        x, y = x.cpu().numpy(), y.cpu().numpy()
        f = flipped.cpu().numpy()
        ax1.scatter(x[~f], y[~f], s=16, color=colors[name], alpha=0.7, label=f"{name} (clean label)")
        ax1.scatter(x[f], y[f], s=50, marker="X", color=colors[name], edgecolor=COLORS["flipped"], linewidth=1.2,
                    label=f"{name} (flipped label)")
    ax1.set_xlabel("credit from exact local games")
    ax1.set_ylabel("credit from Taylor approximation")
    ax1.set_title("Per-example credit: approximation vs exact local Shapley")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_credit_trajectories(cumulative: t.Tensor, flipped: t.Tensor, title: str):
    """One line per training example: its cumulative credit against SGD step. Flipped labels in red."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    c = cumulative.cpu().numpy()
    f = flipped.cpu().numpy()
    steps = np.arange(c.shape[0])
    ax.plot(steps, c[:, ~f], color=COLORS["control"], lw=0.6, alpha=0.5)
    ax.plot(steps, c[:, f], color=COLORS["flipped"], lw=1.2, alpha=0.9)
    ax.plot(steps, c[:, ~f].mean(1), color="black", lw=2.5, label="mean over clean-label examples")
    ax.plot(steps, c[:, f].mean(1), color=COLORS["flipped"], lw=3, label="mean over flipped-label examples")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("SGD step")
    ax.set_ylabel("cumulative In-Run credit")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(handles=[*ax.get_legend_handles_labels()[0],
                       Patch(color=COLORS["control"], label="individual clean-label examples"),
                       Patch(color=COLORS["flipped"], alpha=0.6, label="individual flipped-label examples")], fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_method_scatter(x: t.Tensor, y: t.Tensor, flipped: t.Tensor, xlabel: str, ylabel: str, title: str, ax=None):
    """Scatter of two attribution scores for every training example, with known label flips marked."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    x, y, f = x.cpu().numpy(), y.cpu().numpy(), flipped.cpu().numpy()
    ax.scatter(x[~f], y[~f], s=18, color=COLORS["data_shapley"], alpha=0.7, label="clean label")
    ax.scatter(x[f], y[f], s=60, marker="X", color=COLORS["flipped"], label="flipped label")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()


def plot_deletion_curves(ks: list[int], results: dict[int, dict[str, np.ndarray]], oracle: dict[int, float], title: str):
    """
    2 x n_seeds grid. Top row: remove the k LOWEST-credit training points. Bottom row: remove the k HIGHEST.
    `results[seed]` maps a curve name to an array of test-loss changes (one per k), except "random", which is a
    (n_repeats, len(ks)) array drawn as a mean line and a min-max band.
    """
    seeds = list(results)
    fig, axes = plt.subplots(2, len(seeds), figsize=(5.2 * len(seeds), 8), sharex=True)
    styles = {
        "In-Run, first order": dict(color=COLORS["first"], marker="o"),
        "In-Run, second order": dict(color=COLORS["second"], marker="s"),
        "Monte Carlo Data Shapley": dict(color=COLORS["data_shapley"], marker="^"),
        "leave-one-out": dict(color=COLORS["loo"], marker="D"),
        "In-Run, shuffled validation labels (control)": dict(color=COLORS["control_dark"], ls="--", marker="v"),
    }
    for col, seed in enumerate(seeds):
        for row, which in enumerate(["lowest", "highest"]):
            ax = axes[row, col]
            rand = results[seed]["random"]
            ax.fill_between(ks, rand.min(0), rand.max(0), color=COLORS["control"], alpha=0.25,
                            label="random points (min-max of 10 draws)")
            ax.plot(ks, rand.mean(0), color=COLORS["control"], lw=2, label="random points (mean)")
            for name, style in styles.items():
                ax.plot(ks, results[seed][f"{name}|{which}"], lw=1.8, markersize=4, label=name, **style)
            if which == "lowest":
                ax.scatter([12], [oracle[seed]], marker="*", s=220, color=COLORS["flipped"], zorder=5,
                           label="the 12 flipped labels (oracle)")
            ax.axhline(0, color="black", lw=0.8)
            ax.grid(alpha=0.3)
            ax.set_title(f"seed {seed}: remove the k {which}-credit points", fontsize=10)
            if col == 0:
                ax.set_ylabel("change in test loss after retraining\n(negative = deletion helped)")
            if row == 1:
                ax.set_xlabel("k (number of training points removed)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_lr_balance(lrs: list[float], rows: dict[str, list[float]], title: str):
    """Allocated-minus-actual credit against learning rate, one line per method."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"In-Run, first order": COLORS["first"], "In-Run, second order": COLORS["second"],
              "In-Run, exact local games": COLORS["exact_local"]}
    styles = {"In-Run, exact local games": dict(ls="--", marker="s")}
    for name, values in rows.items():
        ax.plot(lrs, values, label=name, color=colors.get(name), lw=2, **styles.get(name, dict(marker="o")))
    ax.axhline(1, color="black", lw=0.8, label="perfect accounting (ratio = 1)")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("total credit allocated / actual improvement")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.show()
