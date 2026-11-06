"""Plumbing for [8.4] ACDC and Circuit Metrics: model loading, data, plotting, and the GPT-2 edge-patched forward
pass from [8.2]. None of the methods taught in the section live here."""

from __future__ import annotations

import einops
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
        attn_only=True, tokenizer_name="EleutherAI/gpt-neox-20b", seed=398, use_attn_result=True,
        normalization_type=None, positional_embedding_type="shortformer",
    )
    model = HookedTransformer(cfg)
    weights = hf_hub_download(repo_id="callummcdougall/attn_only_2L_half", filename="attn_only_2L_half.pth")
    model.load_state_dict(t.load(weights, map_location=device, weights_only=True))
    model.to(device)
    model.requires_grad_(False)
    return model


def make_induction_data(model: HookedTransformer, n_prompts: int, half_len: int, seed: int) -> tuple[Tensor, Tensor]:
    """
    Clean prompts are [BOS, r_1..r_n, r_1..r_n] for random tokens r; corrupted prompts are [BOS, s_1..s_2n] for
    fresh random tokens s, with nothing to repeat. Returns (clean_tokens, corrupt_tokens), each (n_prompts, 2n+1).
    """
    generator = t.Generator().manual_seed(seed)
    bos = t.full((n_prompts, 1), model.tokenizer.bos_token_id)
    repeated = t.randint(1000, 10000, (n_prompts, half_len), generator=generator)
    clean = t.cat([bos, repeated, repeated], dim=1)
    corrupt = t.cat([bos, t.randint(1000, 10000, (n_prompts, 2 * half_len), generator=generator)], dim=1)
    return clean.to(model.cfg.device), corrupt.to(model.cfg.device)


# ---------------------------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------------------------


def plot_edge_heatmap(values: Tensor, edge_mask: Tensor, sender_names: list[str], receiver_names: list[str], title: str):
    """Heatmap of a (sender, receiver) array on a log colour scale; invalid edges are blank."""
    v = values.detach().float().cpu().clone().abs()
    v[~edge_mask.cpu()] = float("nan")
    fig, ax = plt.subplots(figsize=(15, 5.2))
    floor = max(np.nanmin(np.where(v.numpy() > 0, v.numpy(), np.nan)), 1e-5)
    im = ax.imshow(np.log10(np.clip(v.numpy(), floor, None)), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(sender_names)), sender_names, fontsize=7)
    ax.set_xticks(range(len(receiver_names)), receiver_names, fontsize=6, rotation=90)
    ax.set_ylabel("sender")
    ax.set_xlabel("receiver")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log10(effect)", fraction=0.025, pad=0.01)
    plt.tight_layout()
    plt.show()


def _layout_2l(name: str, n_heads: int = 12) -> tuple[float, float]:
    if name == "embed":
        return (n_heads - 1) / 2, -1.0
    if name == "logits":
        return (n_heads - 1) / 2, 2.2
    layer, head = map(int, name.split("."))
    return float(head), float(layer) * 1.1


def draw_2l_circuit(circuit: Tensor, sender_names: list[str], receiver_names: list[str], title: str,
                    highlight: dict[str, str] | None = None, ax=None):
    """
    Draws a circuit in the two-layer model: embedding at the bottom, layer 0 and layer 1 heads in rows, logits on
    top. Edges into queries are dashed, into keys dotted, into values solid, and into the logits solid grey.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5.5))
    colours = {"q": "tab:orange", "k": "tab:purple", "v": "tab:green", "logits": "dimgray"}
    styles = {"q": "--", "k": ":", "v": "-", "logits": "-"}
    curvature = {"q": 0.2, "k": -0.2, "v": 0.05, "logits": 0.08}  # so that q and k edges between the same heads don't overlap
    used = set()
    for u, v in circuit.nonzero().tolist():
        s_name, r_name = sender_names[u], receiver_names[v]
        kind = r_name[-2] if "<" in r_name else "logits"
        r_node = r_name.split("<")[0]
        used |= {s_name, r_node}
        x0, y0 = _layout_2l(s_name)
        x1, y1 = _layout_2l(r_node)
        ax.annotate("", xy=(x1, y1 - 0.08), xytext=(x0, y0 + 0.08),
                    arrowprops=dict(arrowstyle="-|>", color=colours[kind], lw=1.6, linestyle=styles[kind], alpha=0.8,
                                    connectionstyle=f"arc3,rad={curvature[kind]}", shrinkA=5, shrinkB=5))
    for layer in range(2):
        for head in range(12):
            name = f"{layer}.{head}"
            x, y = _layout_2l(name)
            on = name in used
            ax.plot(x, y, "o", color="black" if on else "lightgray", markersize=11 if on else 6, zorder=3)
            ax.text(x, y - 0.17, name, ha="center", fontsize=8, color="black" if on else "gray")
            if highlight and name in highlight:
                ax.text(x, y + 0.13, highlight[name], ha="center", fontsize=7, color="tab:green" if on else "gray")
    for name in ["embed", "logits"]:
        x, y = _layout_2l(name)
        ax.plot(x, y, "s", color="black", markersize=12, zorder=3)
        ax.text(x + 0.35, y, name, va="center", fontsize=9)
    for kind in ["q", "k", "v", "logits"]:
        ax.plot([], [], color=colours[kind], linestyle=styles[kind], label=f"into {kind}" if kind != "logits" else "into logits")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-1, 12.5)
    ax.set_ylim(-1.5, 2.6)
    ax.axis("off")
    ax.set_title(title)
    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_size_vs_metric(curves: dict[str, tuple[list[int], list[float]]], title: str, ylabel: str,
                        random_band: tuple[list[int], list[float], list[float], list[float]] | None = None,
                        reference: dict[str, float] | None = None, logy: bool = False):
    """
    One line per method of (number of edges, metric). `random_band` is (sizes, low, median, high) for same-size
    random circuits; `reference` draws horizontal lines (e.g. the full model and the empty circuit).
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if random_band is not None:
        sizes, low, mid, high = random_band
        ax.fill_between(sizes, low, high, color="gray", alpha=0.25, label="random circuits of the same size (5-95%)")
        ax.plot(sizes, mid, color="gray", lw=1)
    markers = ["o", "s", "^", "D", "v"]
    for (name, (sizes, values)), m in zip(curves.items(), markers):
        ax.plot(sizes, values, marker=m, label=name)
    for name, value in (reference or {}).items():
        ax.axhline(value, ls="--", lw=1, color="black" if "full" in name else "tab:red", label=name)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("number of edges in the circuit (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_completeness(circuit_scores: Tensor, model_scores: Tensor, random_circuit_scores: Tensor | None, title: str):
    """Scatter of F(C \\ K) against F(M \\ K) for random subsets K; a complete circuit lies on the diagonal."""
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(circuit_scores, model_scores, s=18, alpha=0.75, label="K = random subset of the ACDC circuit")
    if random_circuit_scores is not None:
        ax.scatter(random_circuit_scores[0], random_circuit_scores[1], s=18, alpha=0.6, marker="x", color="tab:red",
                   label="same, for a random circuit of the same size")
    lo = min(0.0, float(t.cat([circuit_scores, model_scores]).min())) - 0.05
    ax.plot([lo, 1.05], [lo, 1.05], "k--", lw=0.8, label="complete: F(C\\K) = F(M\\K)")
    ax.set_xlabel("faithfulness of circuit minus K,  F(C \\ K)")
    ax.set_ylabel("faithfulness of full model minus K,  F(M \\ K)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_minimality(drops: dict[str, Tensor], title: str):
    """Sorted per-edge drops in faithfulness, one line per circuit."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, d in drops.items():
        d = d.sort(descending=True).values.numpy()
        ax.plot(np.arange(1, len(d) + 1), d, marker=".", label=f"{name} ({len(d)} edges)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("edges, sorted by how much removing them hurts")
    ax.set_ylabel("drop in faithfulness when removed alone")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_roc(curves: dict[str, tuple[list[float], list[float]]], title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for name, (fpr, tpr) in curves.items():
        order = np.argsort(fpr)
        ax.plot(np.array(fpr)[order], np.array(tpr)[order], marker=".", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------------------------
# GPT-2: the edge-patched forward pass you wrote in [8.2] (`run_with_circuit`), returning final-position logits
# ---------------------------------------------------------------------------------------------------------------


@t.no_grad()
def gpt2_circuit_logits(model: HookedTransformer, clean_tokens: Tensor, corrupt_out: Tensor, in_circuit: Tensor,
                        edge_mask: Tensor, graph) -> Tensor:
    """
    Final-position logits of GPT-2 run on `clean_tokens`, where every valid edge outside `in_circuit` carries the
    sender's corrupted output (`corrupt_out`, from [8.2]'s `sender_activations`) instead of its current output.
    `graph` is the [8.2] solutions module (for the sender/receiver index helpers). Requires
    `model.set_use_split_qkv_input(True)` and `model.set_use_hook_mlp_in(True)`.
    """
    n_heads, n_layers = model.cfg.n_heads, model.cfg.n_layers
    removed = (edge_mask & ~in_circuit).float().to(model.cfg.device)
    diff = t.zeros_like(corrupt_out)

    def store_input(x, hook):
        diff[:, :, 0] = x - corrupt_out[:, :, 0]

    def store_heads(z, hook):
        start = graph.head_sender(hook.layer(), 0)
        out = einops.einsum(z, model.W_O[hook.layer()], "b p h e, h e d -> b p h d")
        diff[:, :, start : start + n_heads] = out - corrupt_out[:, :, start : start + n_heads]

    def store_mlp(x, hook):
        u = graph.mlp_sender(hook.layer())
        diff[:, :, u] = x - corrupt_out[:, :, u]

    def patch_head_inputs(x, hook):
        start = graph.head_receiver(hook.layer(), hook.name.split(".")[-1][len("hook_")], 0)
        return x - einops.einsum(diff, removed[:, start : start + n_heads], "b p s d, s h -> b p h d")

    def patch_single_input(x, hook):
        v = graph.LOGITS_RECEIVER if hook.name.endswith("resid_post") else graph.mlp_receiver(hook.layer())
        return x - einops.einsum(diff, removed[:, v], "b p s d, s -> b p d")

    hooks = [("blocks.0.hook_resid_pre", store_input)]
    for layer in range(n_layers):
        hooks += [(f"blocks.{layer}.hook_{qkv}_input", patch_head_inputs) for qkv in "qkv"]
        hooks += [(f"blocks.{layer}.attn.hook_z", store_heads), (f"blocks.{layer}.hook_mlp_in", patch_single_input),
                  (f"blocks.{layer}.hook_mlp_out", store_mlp)]
    hooks += [(f"blocks.{n_layers - 1}.hook_resid_post", patch_single_input)]
    return model.run_with_hooks(clean_tokens, fwd_hooks=hooks)[:, -1]


def _gpt2_position(name: str, n_layers: int = 12) -> tuple[float, float]:
    if name == "input":
        return (5.5, -1.0)
    if name == "logits":
        return (5.5, n_layers)
    if name.startswith("m"):
        return (12.6, int(name[1:]) + 0.45)
    layer, head = name[1:].split("<")[0].split(".")
    return (int(head), int(layer))


def draw_gpt2_circuit(circuit: Tensor, sender_names: list[str], receiver_names: list[str], title: str,
                      highlight: dict[str, str] | None = None, n_layers: int = 12):
    """
    Draws a circuit in the [8.2] GPT-2 graph on a layer (y) x head (x) grid, with MLPs in the right-hand column.
    Edges into queries are orange dashed, into keys purple dotted, into values green, into MLPs and logits grey.
    Labels from `highlight` are green for nodes in the circuit and grey for nodes that aren't.
    """
    from matplotlib.lines import Line2D

    colours = {"q": "tab:orange", "k": "tab:purple", "v": "tab:green", "other": "dimgray"}
    styles = {"q": "--", "k": ":", "v": "-", "other": "-"}
    fig, ax = plt.subplots(figsize=(11, 8.5))
    used = set()
    for u, v in circuit.nonzero().tolist():
        s_name, r_name = sender_names[u], receiver_names[v]
        kind = r_name[-2] if "<" in r_name else "other"
        r_node = r_name.split("<")[0]
        used |= {s_name, r_node}
        x0, y0 = _gpt2_position(s_name, n_layers)
        x1, y1 = _gpt2_position(r_node, n_layers)
        ax.annotate("", xy=(x1, y1 - 0.12), xytext=(x0, y0 + 0.12),
                    arrowprops=dict(arrowstyle="-|>", color=colours[kind], lw=1.4, alpha=0.75, linestyle=styles[kind],
                                    connectionstyle="arc3,rad=0.12", shrinkA=4, shrinkB=4))
    for layer in range(n_layers):
        for head in range(12):
            name = f"a{layer}.{head}"
            on = name in used
            ax.plot(head, layer, "o", color="black" if on else "lightgray", markersize=7 if on else 4)
            if on:
                ax.text(head + 0.18, layer + 0.18, name[1:], fontsize=7)
        name = f"m{layer}"
        x, y = _gpt2_position(name, n_layers)
        on = name in used
        ax.plot(x, y, "s", color="black" if on else "lightgray", markersize=8 if on else 5)
        if on:
            ax.text(x + 0.2, y, f"MLP{layer}", fontsize=7, va="center")
    for name in ["input", "logits"]:
        x, y = _gpt2_position(name, n_layers)
        ax.plot(x, y, "D", color="black", markersize=9)
        ax.text(x + 0.25, y, name, fontsize=9, va="center")
    for name, label in (highlight or {}).items():
        x, y = _gpt2_position(name, n_layers)
        ax.text(x, y - 0.42, label, fontsize=6.5, color="darkgreen" if name in used else "silver", ha="center")
    legend = [Line2D([0], [0], color=colours[k], lw=1.5, ls=styles[k], label=f"into a {w}")
              for k, w in [("q", "query"), ("k", "key"), ("v", "value"), ("other", "MLP or the logits")]]
    ax.legend(handles=legend, loc="upper left", fontsize=8, bbox_to_anchor=(1.0, 1.0))
    ax.set_xticks(list(range(12)) + [12.6], [str(h) for h in range(12)] + ["MLP"])
    ax.set_yticks(range(n_layers))
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xlim(-0.8, 14)
    ax.set_ylim(-1.6, n_layers + 0.6)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()
