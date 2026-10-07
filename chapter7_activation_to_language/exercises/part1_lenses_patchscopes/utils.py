"""Plotting and data-loading helpers for [7.1] Logit Lens, Tuned Lens & Patchscopes.

Nothing in here is part of the method you're learning: it's plotting code and the boring
bits of data loading. All of the lens / Patchscope logic lives in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer


def load_tinystories_tokens(
    model: HookedTransformer, n_sequences: int, seq_len: int = 64, seed: int = 0
) -> t.Tensor:
    """
    Returns a [n_sequences, seq_len] tensor of tokenized TinyStories text (each sequence starts with BOS).

    We use the validation split, which is a single ~20MB file.
    """
    path = hf_hub_download(
        "roneneldan/TinyStories",
        "data/validation-00000-of-00001-869c898b519ad725.parquet",
        repo_type="dataset",
    )
    texts = pd.read_parquet(path)["text"].tolist()
    rng = np.random.default_rng(seed)
    rows = []
    for i in rng.permutation(len(texts)):
        toks = model.to_tokens(texts[i], truncate=True)[0]
        if len(toks) >= seq_len:
            rows.append(toks[:seq_len])
        if len(rows) == n_sequences:
            break
    return t.stack(rows)


def plot_lens_ranks(
    curves: dict[str, list[float]],
    labels: list[str],
    title: str,
    ylabel: str = "median rank of answer (log scale)",
    logy: bool = True,
    ax=None,
):
    """Plot one line per entry in `curves`, with the layer labels on the x-axis."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    styles = ["-o", "--s", ":^", "-.v", "-d"]
    for (name, values), style in zip(curves.items(), styles):
        ax.plot(range(len(values)), values, style, label=name, markersize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    if show:
        plt.tight_layout()
        plt.show()


def plot_lens_comparison(
    logit_kl: list[float],
    tuned_kl: list[float],
    logit_agree: list[float],
    tuned_agree: list[float],
    bigram_agree: float | None = None,
):
    """Two panels: KL to the final distribution, and top-1 agreement with the final prediction."""
    layers = list(range(len(logit_kl)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.plot(layers, logit_kl, "-o", label="logit lens", markersize=4)
    ax0.plot(layers, tuned_kl, "-s", label="tuned lens", markersize=4)
    ax0.set_yscale("log")
    ax0.set_xlabel("layer (residual stream before block)")
    ax0.set_ylabel("KL(final || lens)  [nats, log scale]")
    ax0.set_title("How far is each lens from the model's final answer?")
    ax0.grid(alpha=0.3)
    ax0.legend()
    ax1.plot(layers, logit_agree, "-o", label="logit lens", markersize=4)
    ax1.plot(layers, tuned_agree, "-s", label="tuned lens", markersize=4)
    if bigram_agree is not None:
        ax1.axhline(bigram_agree, color="gray", ls="--", label="bigram baseline (layer-0 tuned lens)")
    ax1.set_xlabel("layer (residual stream before block)")
    ax1.set_ylabel("top-1 agreement with final prediction")
    ax1.set_ylim(0, 1)
    ax1.set_title("Does the lens predict the same token as the model?")
    ax1.grid(alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plt.show()


def plot_patchscope_grid(grid: t.Tensor, title: str, ceiling: float | None = None):
    """Heatmap of Patchscope accuracy, indexed [source_layer, target_layer]."""
    grid = grid.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, origin="lower")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{100 * grid[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] < 0.6 else "black")
    ax.set_xlabel("target layer  (where we patch the vector in)")
    ax.set_ylabel("source layer  (where we read the vector from)")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    subtitle = f"\n(model's own accuracy on the real prompts: {ceiling:.0%})" if ceiling is not None else ""
    ax.set_title(title + subtitle + "\n(cell values are accuracy in %)", fontsize=10)
    fig.colorbar(im, ax=ax, label="top-1 accuracy")
    plt.tight_layout()
    plt.show()


def plot_bars(values: dict[str, float], title: str, ylabel: str = "accuracy"):
    """Simple bar chart, used for comparing a method against its controls."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names, vals = list(values), list(values.values())
    bars = ax.bar(names, vals, color=["tab:blue"] + ["tab:gray"] * (len(vals) - 1))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()
