"""Plumbing for [8.5] Causal Scrubbing: model loading, data and plotting. The scrubbing itself lives in the notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from huggingface_hub import hf_hub_download
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig


def load_attn_only_2l(device: t.device | str) -> HookedTransformer:
    """The two-layer attention-only model from [1.2] (no layernorm, shortformer positional embeddings)."""
    cfg = HookedTransformerConfig(
        d_model=768, d_head=64, n_heads=12, n_layers=2, n_ctx=2048, d_vocab=50278, attention_dir="causal",
        attn_only=True, tokenizer_name="EleutherAI/gpt-neox-20b", seed=398, normalization_type=None,
        positional_embedding_type="shortformer",
    )
    model = HookedTransformer(cfg)
    weights = hf_hub_download(repo_id="callummcdougall/attn_only_2L_half", filename="attn_only_2L_half.pth")
    state = t.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.requires_grad_(False)
    return model


def repeated_random_tokens(bos: int, batch_shape: tuple[int, ...], half_len: int, generator: t.Generator, device) -> Tensor:
    """Samples from the task distribution: [BOS, s_1..s_n, s_1..s_n] for fresh random tokens s (ids 1000-9999)."""
    s = t.randint(1000, 10000, (*batch_shape, half_len), generator=generator)
    bos_col = t.full((*batch_shape, 1), bos)
    return t.cat([bos_col, s, s], dim=-1).to(device)


def plot_hypothesis_ladder(results: dict[str, tuple[float, float]], title: str, reference: dict[str, float] | None = None,
                           colours: dict[str, str] | None = None):
    """Horizontal bars of loss recovered (mean ± std over resamples), one per hypothesis, top to bottom."""
    names = list(results)
    means = np.array([results[n][0] for n in names])
    stds = np.array([results[n][1] for n in names])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.6))
    y = np.arange(len(names))[::-1]
    cols = [(colours or {}).get(n, "tab:blue") for n in names]
    ax.barh(y, means, xerr=stds, color=cols, alpha=0.85, capsize=3)
    for yi, m in zip(y, means):
        ax.text(max(m, 0) + 0.01, yi, f"{m:.2f}", va="center", fontsize=8)
    for name, value in (reference or {}).items():
        ax.axvline(value, ls="--", lw=1, color="gray")
        ax.text(value, len(names) - 0.4, name, fontsize=7, color="gray", ha="center")
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel("loss recovered (0 = everything resampled, 1 = the model)")
    ax.set_xlim(min(-0.05, means.min() - 0.05), 1.12)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_toy_losses(losses: dict[str, float], title: str):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = list(losses)
    ax.bar(names, [losses[n] for n in names], color=["tab:gray"] + ["tab:blue"] * (len(names) - 1))
    for i, n in enumerate(names):
        ax.text(i, losses[n], f"{losses[n]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean squared error")
    ax.set_title(title)
    plt.xticks(rotation=15, ha="right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_per_position(curves: dict[str, Tensor], positions: list[int], title: str):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, values in curves.items():
        ax.plot(positions, values.cpu().numpy(), marker=".", label=name)
    ax.set_xlabel("position (second half of the sequence)")
    ax.set_ylabel("loss (nats)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
