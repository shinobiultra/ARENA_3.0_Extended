"""Plotting helpers and word lists for [8.1] Activation Patching Refresher.

Nothing in here is part of the method you're learning: it's plotting code and the lists of names,
places and nouns we build prompts from. All of the patching logic (metrics, hooks, sweeps, frozen
runs, effect decompositions) lives in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from matplotlib.colors import TwoSlopeNorm

# Single-token (with a leading space) first names, from the IOI dataset of Wang et al. (2022)
NAMES = [
    "Aaron", "Adam", "Alan", "Alex", "Alice", "Amy", "Anderson", "Andre", "Andrew", "Andy", "Anna", "Anthony",
    "Arthur", "Austin", "Blake", "Brandon", "Brian", "Carter", "Charles", "Charlie", "Christian", "Christopher",
    "Clark", "Cole", "Collins", "Connor", "Crew", "Crystal", "Daniel", "David", "Dean", "Edward", "Elizabeth",
    "Emily", "Eric", "Eva", "Ford", "Frank", "George", "Georgia", "Graham", "Grant", "Henry", "Ian", "Jack",
    "Jacob", "Jake", "James", "Jamie", "Jane", "Jason", "Jay", "Jennifer", "Jeremy", "Jessica", "John",
    "Jonathan", "Jordan", "Joseph", "Joshua", "Justin", "Kate", "Kelly", "Kevin", "Kyle", "Laura", "Leon",
    "Lewis", "Lisa", "Louis", "Luke", "Madison", "Marco", "Marcus", "Maria", "Mark", "Martin", "Mary",
    "Matthew", "Max", "Michael", "Michelle", "Morgan", "Patrick", "Paul", "Peter", "Prince", "Rachel",
    "Richard", "River", "Robert", "Roman", "Rose", "Ruby", "Russell", "Ryan", "Sarah", "Scott", "Sean",
    "Simon", "Stephen", "Steven", "Sullivan", "Taylor", "Thomas", "Tyler", "Victoria", "Warren", "William",
]

PLACES = [
    "store", "garden", "restaurant", "school", "hospital", "office",
    "house", "station", "park", "beach", "market", "library",
]

OBJECTS = [
    "ring", "bone", "basketball", "computer", "necklace", "drink", "snack", "book",
    "letter", "gift", "ball", "bag", "apple", "cake", "pen", "key",
]

# Single-token nouns for the greater-than task (in the spirit of Hanna et al., 2023)
GT_NOUNS = [
    "war", "expedition", "project", "campaign", "voyage", "flight", "reign", "journey", "siege", "pilgrimage",
    "tour", "strike", "trial", "drought", "famine", "occupation", "conflict", "festival", "conference",
    "investigation", "dynasty", "empire", "rebellion", "revolution", "crisis", "partnership", "marriage",
    "friendship", "agreement", "treaty", "contract", "lease", "effort", "operation", "program", "plague",
    "epidemic", "renovation", "construction", "excavation",
]


def _np(x) -> np.ndarray:
    return x.detach().float().cpu().numpy() if isinstance(x, t.Tensor) else np.asarray(x, dtype=float)


def _diverging_norm(arrays: list[np.ndarray], vmax: float | None) -> TwoSlopeNorm:
    if vmax is None:
        vmax = max(float(np.nanmax(np.abs(a))) for a in arrays)
    vmax = max(vmax, 1e-6)
    return TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)


def _annotate(ax, arr: np.ndarray, threshold: float, fmt: str = "{:.2f}", fontsize: int = 6):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if abs(arr[i, j]) >= threshold:
                ax.text(j, i, fmt.format(arr[i, j]), ha="center", va="center", fontsize=fontsize, color="black")


def plot_metric_strips(values: dict[str, t.Tensor], title: str, xlabel: str, seed: int = 0):
    """One row of jittered points per condition (e.g. per corruption), with the mean marked by a black bar."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(values) + 1.2))
    names = list(values)
    for i, name in enumerate(names):
        v = _np(values[name])
        y = len(names) - 1 - i + rng.uniform(-0.25, 0.25, size=len(v))
        ax.scatter(v, y, s=10, alpha=0.5)
        ax.plot([v.mean()] * 2, [len(names) - 1 - i - 0.35, len(names) - 1 - i + 0.35], color="black", lw=2.5)
        ax.text(v.mean(), len(names) - 1 - i + 0.42, f"mean {v.mean():.2f}", ha="center", fontsize=8)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_layer_pos_maps(
    maps: dict[str, t.Tensor],
    labels: list[str],
    title: str,
    vmax: float | None = None,
    cbar_label: str = "patching score",
    annotate_threshold: float = 0.1,
    ylabel: str = "layer",
    ylabels: list[str] | None = None,
):
    """
    A row of (layer x position) heatmaps on a shared diverging colour scale centred at zero. Rows can be
    something other than layers (e.g. heads): pass `ylabel` and `ylabels`.
    """
    arrays = [_np(m) for m in maps.values()]
    norm = _diverging_norm(arrays, vmax)
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 1.5, 4.8), squeeze=False, layout="constrained")
    for ax, (name, arr) in zip(axes[0], zip(maps, arrays)):
        im = ax.imshow(arr, cmap="RdBu", norm=norm, aspect="auto")
        _annotate(ax, arr, annotate_threshold)
        ax.set_xticks(range(arr.shape[1]))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(arr.shape[0]))
        if ylabels is not None:
            ax.set_yticklabels(ylabels)
        ax.set_xlabel("position (labels from prompt 0)")
        ax.set_ylabel(ylabel)
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), label=cbar_label, shrink=0.85)
    fig.suptitle(title)
    plt.show()


def plot_head_maps(
    maps: dict[str, t.Tensor],
    title: str,
    vmax: float | None = None,
    ncols: int | None = None,
    cbar_label: str = "patching score",
    annotate_threshold: float = 0.15,
):
    """A grid of (layer x head) heatmaps on a shared diverging colour scale centred at zero."""
    arrays = [_np(m) for m in maps.values()]
    norm = _diverging_norm(arrays, vmax)
    n = len(maps)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols + 1.2, 3.4 * nrows + 0.8), squeeze=False, layout="constrained")
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        name, arr = list(maps)[k], arrays[k]
        im = ax.imshow(arr, cmap="RdBu", norm=norm)
        _annotate(ax, arr, annotate_threshold, fmt="{:.1f}", fontsize=5)
        ax.set_xticks(range(0, arr.shape[1], 2))
        ax.set_yticks(range(0, arr.shape[0], 2))
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), label=cbar_label, shrink=0.8)
    fig.suptitle(title)
    plt.show()


def plot_labelled_scatter(
    x: t.Tensor,
    y: t.Tensor,
    point_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    n_labels: int = 12,
    diagonal: bool = True,
    noise_floor: float | None = None,
    groups: dict[str, list[str]] | None = None,
    ax=None,
):
    """
    Scatter with the `n_labels` points furthest from the origin labelled. `groups` maps a group name (e.g.
    "name mover") to the point labels in that group; each group gets its own colour and legend entry.
    `noise_floor` shades the square |x|, |y| < noise_floor (e.g. the largest effect seen under a placebo).
    """
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6.3))
    x, y = _np(x).ravel(), _np(y).ravel()
    groups = groups or {}
    palette = ["tab:red", "tab:orange", "tab:purple", "tab:blue", "tab:green", "tab:cyan", "tab:brown", "tab:pink"]
    group_of = {lab: g for g, labs in groups.items() for lab in labs}
    in_group = np.array([lab in group_of for lab in point_labels], dtype=bool)
    ax.scatter(x[~in_group], y[~in_group], c="tab:gray", s=14, alpha=0.6, label="other heads" if groups else None)
    for k, (g, labs) in enumerate(groups.items()):
        idx = [i for i, lab in enumerate(point_labels) if lab in labs]
        ax.scatter(x[idx], y[idx], c=palette[k % len(palette)], s=30, alpha=0.9, label=g, edgecolor="black", lw=0.4)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    if diagonal:
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1, ls="--", label="y = x")
    if noise_floor is not None:
        ax.add_patch(plt.Rectangle((-noise_floor, -noise_floor), 2 * noise_floor, 2 * noise_floor,
                                   color="gold", alpha=0.3, label=f"placebo noise floor (±{noise_floor:.2f})"))
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    order = np.argsort(-np.hypot(x, y))
    for i in order[:n_labels]:
        ax.annotate(point_labels[i], (x[i], y[i]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if diagonal or noise_floor is not None or groups:
        ax.legend(fontsize=7, loc="best")
    if show:
        plt.tight_layout()
        plt.show()


def plot_correlation_matrix(matrix: np.ndarray, names: list[str], title: str):
    """Annotated heatmap of a symmetric correlation matrix."""
    matrix = _np(matrix)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2.5, 0.85 * len(names) + 1.8))
    im = ax.imshow(matrix, cmap="viridis", vmin=min(0, matrix.min()), vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if matrix[i, j] > 0.6 else "white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="Pearson correlation over the 144 heads")
    plt.tight_layout()
    plt.show()


def plot_total_vs_direct(
    groups: dict[str, list[tuple[float, float]]],
    title: str,
    xlabel: str = "direct effect on logit diff (downstream frozen)",
    ylabel: str = "total effect on logit diff (downstream recomputed)",
    annotate: dict[str, str] | None = None,
):
    """
    Scatter of effects, one colour per group. Each point is a (total effect, direct effect) pair, drawn
    with the direct effect on the x-axis. The line total = direct (no compensation) is drawn for reference.
    `annotate` maps a group name to a text label for its points.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ["o", "*", "s", "^", "D"]
    sizes = [25, 250, 60, 60, 60]
    colors = ["tab:gray", "tab:red", "tab:blue", "tab:green", "tab:purple"]
    all_vals = []
    for k, (name, pts) in enumerate(groups.items()):
        te = [p[0] for p in pts]
        de = [p[1] for p in pts]
        ax.scatter(de, te, marker=markers[k % 5], s=sizes[k % 5], color=colors[k % 5], label=name, alpha=0.8,
                   edgecolor="black" if k else None, zorder=3)
        all_vals += list(de) + list(te)
        if annotate and name in annotate:
            for d, tt in zip(de, te):
                ax.annotate(annotate[name], (d, tt), fontsize=9, xytext=(6, -12), textcoords="offset points")
    lo, hi = min(all_vals), max(all_vals)
    pad = 0.1 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", ls="--", lw=1, label="total = direct (no compensation)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_bars(
    values: dict[str, float],
    title: str,
    ylabel: str,
    colors: list[str] | None = None,
    errors: dict[str, tuple[float, float]] | None = None,
    rotation: int = 0,
    figsize: tuple[float, float] = (8, 3.8),
):
    """Bar chart. `errors` maps a bar name to a (low, high) interval drawn as an error bar."""
    fig, ax = plt.subplots(figsize=figsize)
    names, vals = list(values), [float(v) for v in values.values()]
    bars = ax.bar(names, vals, color=colors or ["tab:blue"] * len(vals))
    if errors:
        for i, name in enumerate(names):
            if name in errors:
                lo, hi = errors[name]
                ax.plot([i, i], [lo, hi], color="black", lw=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_per_prompt_effects(
    effects: t.Tensor, head_labels: list[str], cis: list[tuple[float, float]], title: str, ylabel: str, seed: int = 0
):
    """Jittered per-prompt effects for each head, with the mean and its bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    effects = _np(effects)
    fig, ax = plt.subplots(figsize=(1.1 * len(head_labels) + 2, 4.5))
    for i, row in enumerate(effects):
        ax.scatter(i + rng.uniform(-0.22, 0.22, size=len(row)), row, s=8, alpha=0.45, color="tab:blue")
        lo, hi = cis[i]
        ax.plot([i - 0.3, i + 0.3], [row.mean()] * 2, color="black", lw=2.5)
        ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, color="tab:red", alpha=0.35, lw=0))
    ax.scatter([], [], s=8, color="tab:blue", label="one prompt")
    ax.plot([], [], color="black", lw=2.5, label="mean over prompts")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="tab:red", alpha=0.35, label="95% bootstrap CI of the mean"))
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(head_labels)))
    ax.set_xticklabels(head_labels)
    ax.set_xlabel("head")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_year_probs(panels: dict[str, t.Tensor], start_years: t.Tensor, title: str):
    """
    For each panel, a heatmap of P(next token = YY') for YY' in 00..99 (x-axis), one row per prompt,
    with prompts sorted by their start year YY (shown as a red line).
    """
    order = start_years.argsort().cpu()
    years = start_years.cpu()[order].numpy()
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels) + 1, 4.8), squeeze=False)
    for ax, (name, probs) in zip(axes[0], panels.items()):
        arr = _np(probs)[order.numpy()]
        im = ax.imshow(arr, aspect="auto", cmap="Blues", vmin=0, vmax=np.quantile(arr, 0.995))
        ax.plot(years, np.arange(len(years)), color="tab:red", lw=1.2, label="start year YY of the prompt")
        ax.set_xlabel("predicted end year YY' (two-digit token)")
        ax.set_ylabel("prompt (sorted by start year YY)")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        fig.colorbar(im, ax=ax, label="probability")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_scatter_grid(
    x: t.Tensor,
    ys: dict[str, t.Tensor],
    point_labels: list[str],
    xlabel: str,
    title: str,
    n_labels: int = 6,
):
    """One labelled scatter per entry of `ys`, all against the same x, with the Pearson r in each title."""
    n = len(ys)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.4), squeeze=False)
    xv = _np(x).ravel()
    for ax, (name, y) in zip(axes[0], ys.items()):
        yv = _np(y).ravel()
        r = np.corrcoef(xv, yv)[0, 1]
        plot_labelled_scatter(xv, yv, point_labels, xlabel, name, f"{name}  (r = {r:.2f})", n_labels=n_labels, ax=ax)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
