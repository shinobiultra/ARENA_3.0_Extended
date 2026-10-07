"""Plotting helpers for [7.3] Mini Activation Oracles.

Nothing in here is part of the method you're learning: it's plotting code only. The dataset, the
injection mechanism, the oracle's training loop, the evaluation suite and all the controls live in
the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_COLORS = {"animal": "tab:blue", "city": "tab:orange", "sentiment": "tab:green", "number": "tab:red"}


def plot_probe_layers(
    real: dict[str, list[float]], shuffled: dict[str, list[float]], chance: dict[str, float], title: str
):
    """Probe accuracy against layer, one colour per question family. Shuffled-label controls are dashed."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for family, accs in real.items():
        color = FAMILY_COLORS.get(family)
        ax.plot(range(len(accs)), accs, "-o", color=color, markersize=4, label=f"{family} (real labels)")
        ax.plot(range(len(accs)), shuffled[family], "--", color=color, alpha=0.6, label=f"{family} (shuffled labels)")
        ax.axhline(chance[family], color=color, lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(range(len(next(iter(real.values())))))
    ax.set_xlabel("layer (residual stream after this many blocks)")
    ax.set_ylabel("held-out probe accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="center right")
    plt.tight_layout()
    plt.show()


def plot_training_loss(curves: dict[str, list[float]], floor: float, title: str, smooth: int = 20):
    """Training loss (moving average) with the best loss achievable without reading the activation."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, losses in curves.items():
        losses = np.array(losses)
        kernel = np.ones(smooth) / smooth
        smoothed = np.convolve(losses, kernel, mode="valid") if len(losses) >= smooth else losses
        ax.plot(range(len(smoothed)), smoothed, label=name)
    ax.axhline(floor, color="gray", ls="--", label=f"best text-only loss ({floor:.2f} nats)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"cross-entropy on answer token\n(moving average over {smooth} steps)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_condition_bars(results: dict[str, dict[str, float]], chance: dict[str, float], title: str):
    """Grouped bars: one group per question family, one bar per condition. Chance level drawn per group."""
    families = list(next(iter(results.values())))
    conditions = list(results)
    n = len(conditions)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 4.2))
    palette = ["tab:blue", "tab:purple", "#999999", "#bbbbbb", "#777777", "#555555", "#333333"]
    for i, cond in enumerate(conditions):
        xs = np.arange(len(families)) + (i - (n - 1) / 2) * width
        vals = [results[cond][f] for f in families]
        bars = ax.bar(xs, vals, width, label=cond, color=palette[i % len(palette)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{100 * v:.0f}", ha="center", fontsize=6.5)
    for j, f in enumerate(families):
        ax.hlines(chance[f], j - 0.45, j + 0.45, colors="black", linestyles="dashed", lw=1,
                  label="chance (uniform guess)" if j == 0 else None)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.show()


def plot_generalization_heatmaps(tables: dict[str, pd.DataFrame], title: str):
    """Side-by-side heatmaps (rows = evaluation split, columns = question family). NaN cells are shown as n/a."""
    fig, axes = plt.subplots(1, len(tables), figsize=(4.6 * len(tables) + 1.5, 3.8), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    for k, (ax, (name, df)) in enumerate(zip(axes, tables.items())):
        data = df.to_numpy(dtype=float)
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(data[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="black")
                else:
                    ax.text(j, i, f"{100 * data[i, j]:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if data[i, j] < 0.6 else "black")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(df.index)
        ax.tick_params(axis="y", labelleft=(k == 0))
        ax.set_title(name)
        ax.set_xlabel("question family")
    axes[0].set_ylabel("evaluation split")
    fig.colorbar(im, ax=axes.tolist(), label="top-1 accuracy (cells show %)")
    fig.suptitle(title)
    plt.show()


def plot_layer_transfer(
    oracle: dict[str, list[float]], probe: dict[str, list[float]], chance: dict[str, float], train_layer: int, title: str
):
    """Oracle accuracy vs source layer (solid) against a probe trained at each layer (dashed)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for family, accs in oracle.items():
        color = FAMILY_COLORS.get(family)
        ax.plot(range(len(accs)), accs, "-o", color=color, markersize=4, label=f"oracle: {family}")
        ax.plot(range(len(accs)), probe[family], "--", color=color, alpha=0.5, label=f"probe trained at that layer: {family}")
        ax.axhline(chance[family], color=color, lw=0.8, ls=":", alpha=0.6)
    ax.axvline(train_layer, color="gray", lw=6, alpha=0.15, label="layer the oracle was trained on")
    ax.set_xticks(range(len(next(iter(oracle.values())))))
    ax.set_xlabel("source layer of the injected activation")
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    plt.tight_layout()
    plt.show()


def plot_position_sweep(
    token_labels: list[str],
    curves: dict[str, dict[str, list[float]]],
    available: dict[str, int],
    chance: dict[str, float],
    title: str,
):
    """One panel per question family: accuracy at every source position, for several readers."""
    families = list(available)
    fig, axes = plt.subplots(1, len(families), figsize=(4.2 * len(families), 3.8), sharey=True)
    styles = ["-o", "-s", "--", ":"]
    for ax, family in zip(axes, families):
        for (name, accs), style in zip(curves.items(), styles):
            ax.plot(range(len(token_labels)), accs[family], style, markersize=4, label=name)
        ax.axhline(chance[family], color="black", lw=0.8, ls="--", label="chance")
        ax.axvspan(-0.5, available[family] - 0.5, color="gray", alpha=0.15, label="answer not yet in the prefix")
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"Q: {family}")
        ax.set_xlim(-0.5, len(token_labels) - 0.5)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("top-1 accuracy")
    axes[0].legend(fontsize=7, loc="center left")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confidence(confidences: dict[str, np.ndarray], title: str):
    """Histogram of the oracle's top-1 probability under several input conditions."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(0, 1, 26)
    styles = [dict(lw=4, alpha=0.5), dict(lw=1.5, ls="--"), dict(lw=2), dict(lw=2)]
    for (name, values), style in zip(confidences.items(), styles):
        ax.hist(values, bins=bins, histtype="step", label=f"{name} (mean {np.mean(values):.2f})", **style)
    ax.set_xlabel("probability of the oracle's top answer")
    ax.set_ylabel("number of questions")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_steering(alphas: list[float], curves: dict[str, list[float]], title: str, xlabel: str):
    """Fraction of answers against steering strength, for the concept direction and its controls."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    styles = ["-o", "--o", "-s", "--s", ":^", ":v"]
    for (name, values), style in zip(curves.items(), styles):
        ax.plot(alphas, values, style, markersize=4, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of test sentences")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_patching(layers: list[int], curves: dict[str, list[float]], title: str):
    """Fraction of readouts naming the patched-in (new) or original value, against the patch layer."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    styles = ["-o", "--s", "-^", "--v", ":d"]
    for (name, values), style in zip(curves.items(), styles):
        ax.plot(layers, values, style, markersize=5, label=name)
    ax.set_xticks(layers)
    ax.set_xlabel("layer at which the city token's residual stream is patched")
    ax.set_ylabel("fraction of test sentences")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
