"""Plumbing for [8.8] Benchmarking Circuit Discovery on Compiled Transformers: an empty model to compile into, data,
a generic edge-graph forward pass (the [8.2]/[8.4] machinery for any small attention-only model), and plots."""

from __future__ import annotations

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig


def empty_attn_only_model(n_layers: int, n_heads: int, d_model: int, d_head: int, d_vocab: int, n_ctx: int, device) -> HookedTransformer:
    """An attention-only transformer with no layernorm and every weight set to zero, ready to have a program written in."""
    cfg = HookedTransformerConfig(
        n_layers=n_layers, d_model=d_model, d_head=d_head, n_heads=n_heads, d_vocab=d_vocab, n_ctx=n_ctx,
        attn_only=True, normalization_type=None, positional_embedding_type="standard", act_fn="relu", seed=0,
    )
    model = HookedTransformer(cfg).to(device)
    model.requires_grad_(False)
    with t.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def make_induction_data(n: int, vocab: int, half_len: int, generator: t.Generator, device) -> tuple[Tensor, Tensor]:
    """
    Clean prompts are [BOS=0, r_1..r_n, r_1..r_n] where the r_i are *distinct* tokens from 1..vocab-1 (so every token
    in the second half has exactly one earlier occurrence). Corrupted prompts are [BOS, s_1..s_2n] with s_i drawn
    uniformly from 1..vocab-1 (repeats allowed, no copied structure).
    """
    halves = t.stack([t.randperm(vocab - 1, generator=generator)[:half_len] + 1 for _ in range(n)])
    clean = t.cat([t.zeros(n, 1, dtype=t.long), halves, halves], dim=1)
    corrupt = t.cat([t.zeros(n, 1, dtype=t.long), t.randint(1, vocab, (n, 2 * half_len), generator=generator)], dim=1)
    return clean.to(device), corrupt.to(device)


class EdgeGraph:
    """
    The edge graph of an attention-only model, as in [8.4]: senders are the input (embedding + positional embedding)
    and every head; receivers are every head's query, key and value input, and the logits.

    `run(tokens, corrupt_out, keep)` runs the model on `tokens` where each valid edge (u, v) carries
    keep[u, v] * (u's current output) + (1 - keep[u, v]) * (u's output on the corrupted prompt). `keep` can be a
    boolean circuit or a float tensor (for gradients with respect to the edge mask). This is the [8.2]
    `run_with_circuit`, written for any small attention-only model.
    """

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.n_layers, self.n_heads = model.cfg.n_layers, model.cfg.n_heads
        L, H = self.n_layers, self.n_heads
        self.sender_names = ["input"] + [f"{l}.{h}" for l in range(L) for h in range(H)]
        self.receiver_names = [f"{l}.{h}<{c}>" for l in range(L) for c in "qkv" for h in range(H)] + ["logits"]
        self.logits = len(self.receiver_names) - 1
        mask = t.zeros(len(self.sender_names), len(self.receiver_names), dtype=t.bool)
        for v, r in enumerate(self.receiver_names):
            r_layer = L if r == "logits" else int(r.split(".")[0])
            for u, s in enumerate(self.sender_names):
                mask[u, v] = (-1 if s == "input" else int(s.split(".")[0])) < r_layer
        self.edge_mask = mask
        model.set_use_split_qkv_input(True)

    def sender(self, layer: int, head: int) -> int:
        return 1 + layer * self.n_heads + head

    def receiver(self, layer: int, c: str, head: int) -> int:
        return layer * 3 * self.n_heads + "qkv".index(c) * self.n_heads + head

    def index(self, sender: str, receiver: str) -> tuple[int, int]:
        return self.sender_names.index(sender), self.receiver_names.index(receiver)

    def mask_from_edges(self, edges: list[tuple[str, str]]) -> Tensor:
        circuit = t.zeros_like(self.edge_mask)
        for s, r in edges:
            circuit[self.index(s, r)] = True
        return circuit

    @t.no_grad()
    def sender_outputs(self, tokens: Tensor) -> Tensor:
        """(batch, seq, n_senders, d_model)."""
        _, cache = self.model.run_with_cache(tokens, names_filter=lambda n: n == "blocks.0.hook_resid_pre" or n.endswith("attn.hook_z"))
        outs = [cache["blocks.0.hook_resid_pre"][:, :, None]]
        for l in range(self.n_layers):
            outs.append(einops.einsum(cache[f"blocks.{l}.attn.hook_z"], self.model.W_O[l], "b p h e, h e d -> b p h d"))
        return t.cat(outs, dim=2)

    def run(self, tokens: Tensor, corrupt_out: Tensor, keep: Tensor) -> Tensor:
        model, H, device = self.model, self.n_heads, tokens.device
        valid = self.edge_mask.to(device).float()
        keep = keep.to(device).float() * valid
        cut = valid - keep
        state = {}

        def store_input(x, hook):
            state["current"] = t.zeros(*x.shape[:2], len(self.sender_names), x.shape[-1], device=device, dtype=x.dtype)
            state["current"][:, :, 0] = x
            return x

        def store_heads(z, hook):
            start = self.sender(hook.layer(), 0)
            state["current"][:, :, start : start + H] = einops.einsum(z, model.W_O[hook.layer()], "b p h e, h e d -> b p h d")
            return z

        def patch_heads(x, hook):
            cols = slice(self.receiver(hook.layer(), hook.name.split(".")[-1][len("hook_")], 0), None)
            cols = slice(cols.start, cols.start + H)
            delta = corrupt_out - state["current"]
            return x + einops.einsum(delta, cut[:, cols], "b p s d, s h -> b p h d")

        def patch_logits(x, hook):
            delta = corrupt_out - state["current"]
            return x + einops.einsum(delta, cut[:, self.logits], "b p s d, s -> b p d")

        hooks = [("blocks.0.hook_resid_pre", store_input)]
        for l in range(self.n_layers):
            hooks += [(f"blocks.{l}.hook_{c}_input", patch_heads) for c in "qkv"] + [(f"blocks.{l}.attn.hook_z", store_heads)]
        hooks += [(f"blocks.{self.n_layers - 1}.hook_resid_post", patch_logits)]
        return model.run_with_hooks(tokens, fwd_hooks=hooks)


# ---------------------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------------


def plot_patterns(patterns: dict[str, Tensor], tokens: list[str], title: str):
    """One attention pattern per panel (query rows, key columns)."""
    fig, axes = plt.subplots(1, len(patterns), figsize=(4.2 * len(patterns), 4.2))
    for ax, (name, p) in zip(np.atleast_1d(axes), patterns.items()):
        ax.imshow(p.cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.set_xticks(range(len(tokens)), tokens, fontsize=6, rotation=90)
        ax.set_yticks(range(len(tokens)), tokens, fontsize=6)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_auc_vs_noise(sigmas: list[float], aucs: dict[str, list[float]], title: str, costs: dict[str, str] | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    markers = ["o", "s", "^", "D", "v", "P"]
    for (name, values), m in zip(aucs.items(), markers):
        label = f"{name} ({costs[name]})" if costs and name in costs else name
        ax.plot(sigmas, values, marker=m, label=label)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="random ranking")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xticks(sigmas, [str(s) for s in sigmas])
    ax.set_xlabel("standard deviation of the noise added to every attention weight")
    ax.set_ylabel("ROC AUC against the compiled circuit")
    ax.set_ylim(0.3, 1.03)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_node_scores(panels: dict[str, Tensor], kinds: list[str], title: str):
    """Each panel is a (kind, layer, head) tensor of node scores, shown as kind x (layer.head) heatmaps."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.0))
    vmax = max(p.abs().max().item() for p in panels.values())
    for ax, (name, p) in zip(np.atleast_1d(axes), panels.items()):
        k, L, H = p.shape
        flat = p.reshape(k, L * H).cpu().numpy()
        im = ax.imshow(flat, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(k), kinds)
        ax.set_xticks(range(L * H), [f"{l}.{h}" for l in range(L) for h in range(H)], fontsize=7)
        for i in range(k):
            for j in range(L * H):
                if abs(flat[i, j]) > 0.05 * vmax:
                    ax.text(j, i, f"{flat[i, j]:.1f}", ha="center", va="center", fontsize=6)
        ax.set_title(name, fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(title)
    plt.show()
