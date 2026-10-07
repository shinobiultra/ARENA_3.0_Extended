"""Data-loading and plotting helpers for [10.1] Capstone Research Sprint.

Nothing in here is part of the method you're learning: it's the boring bits of data loading, and the plotting code.
The splits, the model, the interventions, the controls, the statistics and the assessment all live in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.datasets import load_digits

GREY = "#868e96"
BLUE = "#0b7285"
ORANGE = "#e67700"
GREEN = "#2b8a3e"
RED = "#c92a2a"


def load_digit_pair(digits: tuple[int, int]) -> tuple[t.Tensor, t.Tensor, np.ndarray]:
    """
    Returns the bundled scikit-learn handwritten digits (8x8 images, no download needed) for two digit classes.

    Returns:
        x: float64 tensor of shape (n, 64), pixel values scaled from 0..16 to [0, 1]
        y: int64 tensor of shape (n,), 0 for `digits[0]` and 1 for `digits[1]`
        rows: the row of each image in the full 1797-image dataset (so you can always find an image again)
    """
    if len(digits) != 2 or digits[0] == digits[1] or not all(0 <= d <= 9 for d in digits):
        raise ValueError(f"Expected two different digits between 0 and 9, got {digits}")
    data = load_digits()
    rows = np.flatnonzero(np.isin(data.target, digits))
    x = t.tensor(data.data[rows] / 16.0, dtype=t.float64)
    y = t.tensor((data.target[rows] == digits[1]).astype(np.int64))
    return x, y, rows


def plot_digit_grid(images: t.Tensor, titles: list[str], title: str, n_cols: int = 8):
    """Show a grid of 8x8 digit images (each row of `images` is one flattened image)."""
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.8 * n_rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, image, name in zip(np.ravel(axes), images, titles):
        ax.imshow(image.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(name, fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_training_curve(losses: list[float], title: str):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(np.arange(1, len(losses) + 1), losses, color=BLUE, label="training loss (full batch)")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_coverage(true_rates: list[float], coverage: dict[str, list[float]], n_pairs: int, nominal: float = 0.95):
    """Line plot of how often each interval method contains the true rate, in simulation."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for (name, values), style, color in zip(coverage.items(), ["-o", "-s", "-^"], [BLUE, ORANGE, GREEN]):
        ax.plot(true_rates, values, style, label=name, color=color)
    ax.axhline(nominal, color=GREY, ls="--", label=f"nominal coverage ({nominal:.0%})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("true swap rate used to simulate the data")
    ax.set_ylabel("fraction of simulated studies\nwhose interval contains the truth")
    ax.set_title(f"Do our intervals do what they claim? ({n_pairs} simulated pairs per study)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_decision_map(verdicts: dict[int, list[str]], threshold: float):
    """One strip per sample size: the verdict for every possible number of successes k = 0..n."""
    colors = {"ruled out": RED, "inconclusive": "#ced4da", "supported": GREEN}
    fig, ax = plt.subplots(figsize=(9, 0.8 + 0.7 * len(verdicts)))
    for row, (n, row_verdicts) in enumerate(verdicts.items()):
        for k, v in enumerate(row_verdicts):
            ax.barh(row, 1 / n, left=(k - 0.5) / n, color=colors[v], edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(verdicts)))
    ax.set_yticklabels([f"n = {n} pairs" for n in verdicts])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("observed swap rate k / n")
    ax.axvline(threshold, color="black", ls=":", lw=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors), loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_title(f"Which verdicts can the experiment reach? (claim: swap rate > {threshold:.0%}, 95% Clopper-Pearson)")
    plt.tight_layout()
    plt.show()


def _ranked_vs_random(ax, results: dict, coordinate_counts: list[int], title: str):
    # every result array has shape (n_draws, n_pairs); random controls have one row per random draw
    ranked = [results[f"units_{k}"]["recovered"].mean() for k in coordinate_counts]
    random_draws = np.array([results[f"random_units_{k}"]["recovered"].mean(1) for k in coordinate_counts])
    ax.plot(coordinate_counts, ranked, "-o", color=BLUE, label="top-k units (ranked on training data)")
    ax.plot(coordinate_counts, random_draws.mean(1), "--o", color=GREY, label="k random units (mean of draws)")
    ax.fill_between(
        coordinate_counts,
        np.quantile(random_draws, 0.05, axis=1),
        np.quantile(random_draws, 0.95, axis=1),
        color=GREY,
        alpha=0.2,
        label="k random units (5-95% of draws)",
    )
    for name, color, style in [
        ("full_patch", GREEN, ":"),
        ("direction", BLUE, "-."),
        ("shuffled", ORANGE, "--"),
        ("random_dirs", GREY, ":"),
    ]:
        ax.axhline(results[name]["recovered"].mean(), color=color, ls=style, lw=1.2, label=_LABELS[name])
    ax.set_xticks(coordinate_counts)
    ax.set_xlabel("number of hidden units replaced (k)")
    ax.set_ylabel("fraction of logit difference recovered")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.01, 0.6))


_LABELS = {
    "no_patch": "no patch",
    "full_patch": "full patch",
    "direction": "mean-diff direction",
    "shuffled": "shuffled-label direction",
    "random_dirs": "random directions (mean)",
}


def _label(name: str) -> str:
    if name in _LABELS:
        return _LABELS[name]
    if name.startswith("random_units_"):
        return f"{name.split('_')[-1]} random units"
    if name.startswith("units_"):
        return f"top-{name.split('_')[-1]} units"
    return name


def plot_dev_exploration(results: dict, coordinate_counts: list[int], order: list[str], selected: str | None = None):
    """Two panels on the development set: swap rate of every intervention, and the sparsity curve."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    rates = [results[name]["swap"].mean() for name in order]
    colors = [BLUE if (name.startswith("units_") or name == "direction") else GREY for name in order]
    colors = [ORANGE if name == "shuffled" else c for name, c in zip(order, colors)]
    colors = [GREEN if name == "full_patch" else c for name, c in zip(order, colors)]
    bars = ax0.bar(range(len(order)), rates, color=colors)
    for bar, rate in zip(bars, rates):
        ax0.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.2f}", ha="center", fontsize=7)
    if selected is not None:
        i = order.index(selected)
        ax0.annotate("selected", (i, rates[i] + 0.09), ha="center", fontsize=8, color=BLUE, weight="bold")
    ax0.set_xticks(range(len(order)))
    ax0.set_xticklabels([_label(n) for n in order], rotation=40, ha="right", fontsize=8)
    ax0.set_ylim(0, 1.18)
    ax0.set_ylabel("swap rate (fraction of pairs)")
    ax0.set_title("Development set: candidates (blue) and controls (grey/orange)")
    _ranked_vs_random(ax1, results, coordinate_counts, "Development set: how much do k units carry?")
    plt.tight_layout()
    plt.show()


def plot_signature(
    dev: dict,
    confirm: dict,
    coordinate_counts: list[int],
    order: list[str],
    selected: str,
    intervals: dict[str, tuple[float, float, float]],
    comparisons: dict[str, tuple[float, float, float]],
    threshold: float,
    suptitle: str,
):
    """The four-panel confirmation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # A: sparsity curve on the confirmation set
    _ranked_vs_random(axes[0, 0], confirm, coordinate_counts, "A. How much of the decision do k units carry?")

    # B: development vs confirmation swap rates
    ax = axes[0, 1]
    x = np.arange(len(order))
    ax.bar(x - 0.2, [dev[n]["swap"].mean() for n in order], 0.4, color="#adb5bd", label="development (exploration)")
    ax.bar(x + 0.2, [confirm[n]["swap"].mean() for n in order], 0.4, color=BLUE, label="confirmation (held out)")
    for i, n in enumerate(order):
        rate = confirm[n]["swap"].mean()
        ax.text(i + 0.2, rate + 0.02, f"{rate:.2f}", ha="center", fontsize=6, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(n) for n in order], rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("swap rate (fraction of pairs)")
    ax.set_title(f"B. Every intervention, both sets (selected on development: {_label(selected)})")
    ax.legend(fontsize=8, loc="upper left")

    # C: absolute swap rates with exact intervals, against the registered threshold
    ax = axes[1, 0]
    names = list(intervals)
    for i, name in enumerate(names):
        mean, low, high = intervals[name]
        color = RED if i == 0 else (ORANGE if name == "shuffled" else BLUE)
        ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt="o", capsize=5, color=color)
    ax.axvline(threshold, color=RED, ls="--", label=f"registered threshold ({threshold:.0%})")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([_label(n) + (" (H1, primary)" if i == 0 else "") for i, n in enumerate(names)])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("confirmation swap rate, with 95% Clopper-Pearson interval")
    ax.set_title("C. Is a single unit sufficient? (H1)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="center right")

    # D: paired bootstrap differences
    ax = axes[1, 1]
    names = list(comparisons)
    for i, name in enumerate(names):
        mean, low, high = comparisons[name]
        ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt="s", capsize=5, color=BLUE)
        label = f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"
        if mean < 0.5:
            ax.text(high + 0.04, i, label, ha="left", va="center", fontsize=8)
        else:
            ax.text(low - 0.04, i, label, ha="right", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(-0.15, 1.1)
    ax.set_xlabel("paired difference on confirmation pairs, with 95% bootstrap interval")
    ax.set_title("D. Does the method beat its controls? (H2, H3)")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_pairs(images: list[tuple[t.Tensor, t.Tensor]], row_labels: list[str], probs: np.ndarray, columns: list[str]):
    """Left: the recipient/donor images of each directed intervention. Right: P(donor's label) after each patch."""
    n = len(images)
    fig = plt.figure(figsize=(13, 0.75 * n + 1.5))
    grid = fig.add_gridspec(n, 4, width_ratios=[0.6, 0.6, 0.8, 6])
    for i, (recipient, donor) in enumerate(images):
        for j, (image, name) in enumerate([(recipient, "recipient"), (donor, "donor")]):
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(image.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=9)
    ax = fig.add_subplot(grid[:, 3])
    im = ax.imshow(probs, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if probs[i, j] < 0.5 else "black")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Probability the model assigns to the donor's label, after each intervention")
    fig.colorbar(im, ax=ax, shrink=0.8, label="P(donor's label)")
    plt.show()


def plot_hidden_code(
    h_sorted: np.ndarray,
    n_class0: int,
    digit_names: tuple[str, str],
    unit_order: list[int],
    cumulative_share: np.ndarray,
    measured_k: list[int],
    measured_recovered: list[float],
    random_expected: np.ndarray,
):
    """Left: hidden activations (images x units). Right: predicted vs measured fraction of logit difference."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    im = ax0.imshow(h_sorted, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax0.axhline(n_class0 - 0.5, color="black", lw=1.5)
    ax0.set_yticks([n_class0 / 2, n_class0 + (len(h_sorted) - n_class0) / 2])
    ax0.set_yticklabels([f"{digit_names[0]}s", f"{digit_names[1]}s"])
    ax0.set_xticks(range(len(unit_order)))
    ax0.set_xticklabels(unit_order, fontsize=6)
    ax0.set_xlabel("hidden unit (sorted by contribution to the logit difference, largest first)")
    ax0.set_title("Hidden activations of every training image")
    fig.colorbar(im, ax=ax0, label="tanh activation")
    ks = np.arange(1, len(cumulative_share) + 1)
    ax1.plot(ks, cumulative_share, "-", color=BLUE, label="predicted from class means (top-k units)")
    ax1.plot(ks, random_expected, "--", color=GREY, label="predicted for k random units (k / n_units x total)")
    ax1.plot(measured_k, measured_recovered, "o", color=ORANGE, markersize=8,
             label="measured on development pairs (top-k units)")
    ax1.axhline(0.5, color=RED, ls=":", lw=1, label="half of the logit difference")
    ax1.set_xlabel("number of hidden units replaced (k)")
    ax1.set_ylabel("fraction of logit difference")
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_title("Every unit carries a small, similar share")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_one_dimensional(
    variance_total: np.ndarray,
    variance_within: np.ndarray,
    cos_shuffled: np.ndarray,
    cos_random: np.ndarray,
    swap_shuffled: np.ndarray,
    swap_random: np.ndarray,
    registered: dict[str, tuple[float, float]],
):
    """Three panels: variance spectrum, alignment histograms, and alignment vs development swap rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ks = np.arange(1, len(variance_total) + 1)
    axes[0].plot(ks, variance_total, "-o", color=BLUE, markersize=4, label="all training activations")
    axes[0].plot(ks, variance_within, "-s", color=ORANGE, markersize=4, label="within-class (class means removed)")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("fraction of variance")
    axes[0].set_title("Hidden-layer variance spectrum")
    axes[0].set_xlim(0.5, 10.5)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    bins = np.linspace(0, 1, 26)
    axes[1].hist(cos_random, bins=bins, color=GREY, alpha=0.7, label="random directions")
    axes[1].hist(cos_shuffled, bins=bins, color=ORANGE, alpha=0.7, label="shuffled-label directions")
    axes[1].set_xlabel("|cosine| with the top principal component")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"{len(cos_shuffled)} re-drawn controls of each kind")
    axes[1].legend(fontsize=8)
    axes[2].scatter(cos_random, swap_random, color=GREY, alpha=0.5, s=14, label="random directions")
    axes[2].scatter(cos_shuffled, swap_shuffled, color=ORANGE, alpha=0.5, s=14, label="shuffled-label directions")
    # the registered directions often coincide, so draw them as concentric markers of decreasing size
    styles = zip(["*", "X", "P"], [BLUE, ORANGE, "black"], [260, 120, 50])
    for z, ((name, (c, s)), (marker, color, size)) in enumerate(zip(registered.items(), styles)):
        axes[2].scatter([c], [s], marker=marker, s=size, color=color, edgecolor="white", label=name, zorder=3 + z)
    axes[2].set_xlabel("|cosine| with the top principal component")
    axes[2].set_ylabel("development swap rate")
    axes[2].set_title("Alignment predicts whether a direction swaps")
    axes[2].set_ylim(-0.05, 1.08)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7, loc="center left", markerscale=0.7)
    plt.tight_layout()
    plt.show()


def plot_dose_response(scales: np.ndarray, curves: dict[str, np.ndarray], chosen: float, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (name, values), color in zip(curves.items(), [BLUE, ORANGE, GREY]):
        ax.plot(scales, values, "-o", markersize=3, color=color, label=name)
    ax.axvline(chosen, color=RED, ls=":", label=f"your scale = {chosen:g}")
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set_xlabel("scale (0 = no patch, 1 = replace the donor's projection)")
    ax.set_ylabel("mean P(donor's label)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
