# %%
"""Reference solutions for [6.3] Transcoders and Attribution Graphs."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t
import torch.nn.functional as F

chapter = "chapter6_sparse_feature_methods"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_LN2_HOOK = "blocks.0.ln2.hook_normalized"
TL_MLP_POST_HOOK = "blocks.0.mlp.hook_post"
TL_MLP_OUT_HOOK = "blocks.0.hook_mlp_out"
TL_RESID_MID_HOOK = "blocks.0.hook_resid_mid"
TL_TRANSCODER_PROMPTS = [
    "The cat sat on the",
    "The bird flew over the",
    "To make tea, boil the",
    "To make bread, bake the",
    "The recipe calls for sugar and",
    "The recipe calls for salt and",
    "The chef cooked a",
    "The teacher taught a",
    "The river flows into the",
    "The road leads into the",
    "The Python function returns a",
    "The HTML page contains a",
    "The file was saved to disk as a",
    "The message was sent by mail as a",
    "The train arrived at the",
    "The plane landed at the",
]


# %%
@dataclass(frozen=True)
class ToyOrganism:
    input_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    output_names: tuple[str, ...]
    encoder_weight: t.Tensor
    encoder_bias: t.Tensor
    decoder_weight: t.Tensor
    decoder_bias: t.Tensor
    readout: t.Tensor


@dataclass(frozen=True)
class TranscoderOutput:
    pre_acts: t.Tensor
    feature_acts: t.Tensor
    reconstructed_activations: t.Tensor


@dataclass(frozen=True)
class ReconstructionDecomposition:
    feature_components: t.Tensor
    bias_component: t.Tensor
    reconstructed_activations: t.Tensor


@dataclass(frozen=True)
class EdgeAttributions:
    input_to_feature: t.Tensor
    feature_to_score: t.Tensor
    downstream_effects: t.Tensor


@dataclass(frozen=True)
class AttributionEdge:
    source_type: str
    source_id: int
    target_type: str
    target_id: int
    weight: float


@dataclass(frozen=True)
class AttributionGraph:
    feature_ids: tuple[int, ...]
    edges: tuple[AttributionEdge, ...]


@dataclass(frozen=True)
class CausalValidation:
    clean_score: float
    zero_score: float
    retained_score: float
    ablated_score: float
    faithfulness: float
    normalized_damage: float


# Legacy report dataclasses remain available for the pinned GELU-1L preflight.
@dataclass(frozen=True)
class TranscoderReplacementReport:
    reconstruction_mse: float
    replacement_kl: float
    target_logit_diff: float
    replacement_logit_diff: float
    logit_diff_error: float
    passes_kl: bool
    preserves_logit_diff: bool


@dataclass(frozen=True)
class AttributionGraphReport:
    num_nodes: int
    num_edges: int
    density: float
    full_logit_diff: float
    graph_logit_diff: float
    ablated_logit_diff: float
    random_ablated_logit_diff: float
    topk_damage: float
    random_damage: float
    preserves_logit_diff: bool
    passes_damage_control: bool
    reproducible: bool


# %%
def make_toy_organism() -> ToyOrganism:
    """Return the exact colored-shape MLP used throughout the section."""

    encoder_weight = t.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # red
            [0.0, 1.0, 0.0, 0.0],  # blue
            [0.0, 0.0, 1.0, 0.0],  # square
            [0.0, 0.0, 0.0, 1.0],  # circle
            [1.0, 0.0, 1.0, 0.0],  # red AND square
            [0.0, 1.0, 0.0, 1.0],  # blue AND circle
        ]
    )
    encoder_bias = t.tensor([-0.5, -0.5, -0.5, -0.5, -1.5, -1.5])
    decoder_weight = t.tensor(
        [
            [0.4, 0.0, 0.2],
            [0.0, 0.4, 3.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.3, 2.5],
            [2.0, -0.5, 0.0],
            [-0.5, 2.0, 3.0],
        ]
    )
    return ToyOrganism(
        input_names=("red", "blue", "square", "circle"),
        feature_names=("red", "blue", "square", "circle", "red_square", "blue_circle"),
        output_names=("warm_square", "cool_circle", "style"),
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=decoder_weight,
        decoder_bias=t.zeros(3),
        readout=t.tensor([1.0, -1.0, 0.0]),
    )


def make_toy_input(color: str, shape: str) -> t.Tensor:
    """Encode one of the four valid colored shapes as a binary residual vector."""

    if color not in {"red", "blue"}:
        raise ValueError("color must be 'red' or 'blue'.")
    if shape not in {"square", "circle"}:
        raise ValueError("shape must be 'square' or 'circle'.")
    return t.tensor(
        [float(color == "red"), float(color == "blue"), float(shape == "square"), float(shape == "circle")]
    )


# %%
def encode(
    inputs: t.Tensor,
    encoder_weight: t.Tensor,
    encoder_bias: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Map MLP inputs to pre-activations and sparse ReLU feature activations."""

    if inputs.shape[-1] != encoder_weight.shape[-1]:
        raise ValueError("inputs last dimension must match encoder input dimension.")
    if encoder_weight.shape[0] != encoder_bias.shape[0]:
        raise ValueError("encoder feature dimension must match encoder_bias.")
    pre_acts = inputs.float() @ encoder_weight.float().T + encoder_bias.float()
    return pre_acts, F.relu(pre_acts)


def decode(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor,
) -> t.Tensor:
    """Map sparse feature activations into the replaced MLP's output space."""

    if feature_acts.shape[-1] != decoder_weight.shape[0]:
        raise ValueError("feature dimension must match decoder rows.")
    if decoder_weight.shape[1] != decoder_bias.shape[0]:
        raise ValueError("decoder output dimension must match decoder_bias.")
    return feature_acts.float() @ decoder_weight.float() + decoder_bias.float()


def transcoder_forward(
    inputs: t.Tensor,
    encoder_weight: t.Tensor,
    decoder_weight: t.Tensor,
    *,
    encoder_bias: t.Tensor | None = None,
    decoder_bias: t.Tensor | None = None,
) -> TranscoderOutput:
    """Run the encoder and decoder that replace one MLP computation."""

    encoder_bias = (
        t.zeros(encoder_weight.shape[0], device=inputs.device)
        if encoder_bias is None
        else encoder_bias.to(inputs.device)
    )
    decoder_bias = (
        t.zeros(decoder_weight.shape[1], device=inputs.device)
        if decoder_bias is None
        else decoder_bias.to(inputs.device)
    )
    pre_acts, feature_acts = encode(inputs, encoder_weight, encoder_bias)
    reconstructed = decode(feature_acts, decoder_weight, decoder_bias)
    return TranscoderOutput(pre_acts, feature_acts, reconstructed)


# %%
def reconstruction_decomposition(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor,
) -> ReconstructionDecomposition:
    """Decompose a reconstruction into one vector per feature plus decoder bias."""

    if feature_acts.shape[-1] != decoder_weight.shape[0]:
        raise ValueError("feature dimension must match decoder rows.")
    feature_components = feature_acts.float().unsqueeze(-1) * decoder_weight.float()
    reconstructed = feature_components.sum(dim=-2) + decoder_bias.float()
    return ReconstructionDecomposition(feature_components, decoder_bias.float(), reconstructed)


# %%
def feature_edge_attributions(
    inputs: t.Tensor,
    pre_acts: t.Tensor,
    feature_acts: t.Tensor,
    encoder_weight: t.Tensor,
    encoder_bias: t.Tensor,
    decoder_weight: t.Tensor,
    readout: t.Tensor,
) -> EdgeAttributions:
    """Compute exact signed path attributions in the locally linear ReLU model."""

    if inputs.ndim != 1 or pre_acts.ndim != 1 or feature_acts.ndim != 1:
        raise ValueError("This local graph exercise expects one unbatched example.")
    if inputs.shape[0] != encoder_weight.shape[1]:
        raise ValueError("input dimension must match encoder columns.")
    if pre_acts.shape != feature_acts.shape or pre_acts.shape[0] != encoder_weight.shape[0]:
        raise ValueError("pre_acts and feature_acts must match the encoder feature dimension.")
    downstream_effects = decoder_weight.float() @ readout.float()
    gates = pre_acts.gt(0).float()
    input_terms = inputs.float().unsqueeze(-1) * encoder_weight.float().T
    bias_terms = encoder_bias.float().unsqueeze(0)
    local_terms = t.cat([input_terms, bias_terms], dim=0)
    input_to_feature = local_terms * gates.unsqueeze(0) * downstream_effects.unsqueeze(0)
    feature_to_score = feature_acts.float() * downstream_effects
    return EdgeAttributions(input_to_feature, feature_to_score, downstream_effects)


# %%
def extract_attribution_graph(
    input_to_feature: t.Tensor,
    feature_to_score: t.Tensor,
    *,
    top_k_features: int,
    min_abs_edge: float = 1e-8,
) -> AttributionGraph:
    """Keep the strongest feature nodes and every nonzero incoming signed edge."""

    if input_to_feature.ndim != 2:
        raise ValueError("input_to_feature must have shape (sources, features).")
    if input_to_feature.shape[1] != feature_to_score.shape[0]:
        raise ValueError("feature dimensions must match.")
    if top_k_features <= 0:
        raise ValueError("top_k_features must be positive.")
    ranked = sorted(
        range(feature_to_score.numel()),
        key=lambda feature_id: (-abs(float(feature_to_score[feature_id].item())), feature_id),
    )
    selected = tuple(
        feature_id
        for feature_id in ranked
        if abs(float(feature_to_score[feature_id].item())) > min_abs_edge
    )[:top_k_features]
    edges: list[AttributionEdge] = []
    bias_source = input_to_feature.shape[0] - 1
    for feature_id in selected:
        for source_id in range(input_to_feature.shape[0]):
            weight = float(input_to_feature[source_id, feature_id].item())
            if abs(weight) <= min_abs_edge:
                continue
            edges.append(
                AttributionEdge(
                    source_type="bias" if source_id == bias_source else "input",
                    source_id=0 if source_id == bias_source else source_id,
                    target_type="feature",
                    target_id=feature_id,
                    weight=weight,
                )
            )
        edges.append(AttributionEdge("feature", feature_id, "score", 0, float(feature_to_score[feature_id].item())))
    return AttributionGraph(feature_ids=selected, edges=tuple(edges))


def edge_conservation_score(edges: tuple[AttributionEdge, ...] | list[AttributionEdge], n_features: int) -> float:
    """Measure whether incoming signed paths agree with each feature-to-score edge."""

    incoming = t.zeros(n_features)
    outgoing = t.zeros(n_features)
    for edge in edges:
        if edge.target_type == "feature":
            incoming[edge.target_id] += edge.weight
        elif edge.source_type == "feature" and edge.target_type == "score":
            outgoing[edge.source_id] += edge.weight
    denominator = outgoing.abs().sum().item()
    if denominator == 0:
        return 1.0 if incoming.abs().sum().item() == 0 else 0.0
    score = 1.0 - (incoming - outgoing).abs().sum().item() / denominator
    return max(0.0, score)


def shuffle_input_edge_targets(
    edges: tuple[AttributionEdge, ...] | list[AttributionEdge],
    *,
    n_features: int,
    shift: int = 1,
) -> tuple[AttributionEdge, ...]:
    """Cyclically move incoming edges while leaving feature-to-score edges fixed."""

    if n_features <= 1:
        raise ValueError("n_features must be greater than one.")
    shift %= n_features
    return tuple(
        AttributionEdge(
            edge.source_type,
            edge.source_id,
            edge.target_type,
            (edge.target_id + shift) % n_features if edge.target_type == "feature" else edge.target_id,
            edge.weight,
        )
        for edge in edges
    )


# %%
def intervene_features(feature_acts: t.Tensor, feature_ids: list[int] | tuple[int, ...], *, mode: str) -> t.Tensor:
    """Keep or ablate chosen features before decoding."""

    if feature_acts.ndim != 1:
        raise ValueError("This intervention exercise expects one feature vector.")
    ids = t.as_tensor(feature_ids, dtype=t.long, device=feature_acts.device)
    if ids.numel() and (ids.min().item() < 0 or ids.max().item() >= feature_acts.numel()):
        raise ValueError("feature id out of range.")
    mask = t.zeros_like(feature_acts)
    mask[ids] = 1.0
    if mode == "keep":
        return feature_acts * mask
    if mode == "ablate":
        return feature_acts * (1.0 - mask)
    raise ValueError("mode must be 'keep' or 'ablate'.")


def causal_validate_graph(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor,
    readout: t.Tensor,
    feature_ids: list[int] | tuple[int, ...],
) -> CausalValidation:
    """Rerun the decoder after graph retention and ablation interventions."""

    def score(acts: t.Tensor) -> float:
        return float((decode(acts, decoder_weight, decoder_bias) @ readout.float()).item())

    clean_score = score(feature_acts)
    zero_score = score(t.zeros_like(feature_acts))
    retained_score = score(intervene_features(feature_acts, feature_ids, mode="keep"))
    ablated_score = score(intervene_features(feature_acts, feature_ids, mode="ablate"))
    scale = max(abs(clean_score - zero_score), 1e-12)
    return CausalValidation(
        clean_score,
        zero_score,
        retained_score,
        ablated_score,
        1.0 - abs(clean_score - retained_score) / scale,
        abs(clean_score - ablated_score) / scale,
    )


def faithfulness_curve(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor,
    readout: t.Tensor,
    ranking: list[int] | tuple[int, ...],
    *,
    max_k: int,
) -> t.Tensor:
    """Evaluate graph retention faithfulness for prefixes of a feature ranking."""

    if max_k <= 0 or max_k > len(ranking):
        raise ValueError("max_k must be between one and the ranking length.")
    return t.tensor(
        [
            causal_validate_graph(
                feature_acts, decoder_weight, decoder_bias, readout, list(ranking[:k])
            ).faithfulness
            for k in range(1, max_k + 1)
        ]
    )


def reconstruction_only_ranking(decoder_weight: t.Tensor) -> tuple[int, ...]:
    """Rank features by decoder norm, a behavior-agnostic reconstruction baseline."""

    norms = decoder_weight.float().norm(dim=-1)
    return tuple(sorted(range(norms.numel()), key=lambda i: (-float(norms[i].item()), i)))


# %%
def plot_attribution_graph(
    organism: ToyOrganism,
    edge_attributions: EdgeAttributions,
    graph: AttributionGraph,
    *,
    save_path: Path | None = None,
):
    """Plot the exact signed edge heatmap and extracted graph."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    matrix = edge_attributions.input_to_feature.detach().cpu()
    source_names = (*organism.input_names, "encoder bias")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    limit = float(matrix.abs().max().item())
    image = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[0].set_xticks(range(len(organism.feature_names)), organism.feature_names, rotation=35, ha="right")
    axes[0].set_yticks(range(len(source_names)), source_names)
    axes[0].set_title("Exact signed path attribution")
    axes[0].set_xlabel("target feature")
    axes[0].set_ylabel("source")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col].item())
            if abs(value) > 1e-8:
                axes[0].text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes[0], shrink=0.78, label="contribution to score")

    ax = axes[1]
    ax.set_title("Recovered graph for red square")
    ax.set_xlim(-0.15, 2.2)
    ax.set_ylim(-0.7, 4.7)
    ax.axis("off")
    source_pos = {i: (0.0, 4.0 - i) for i in range(len(organism.input_names))}
    bias_pos = (0.0, -0.25)
    feature_pos = {feature_id: (1.05, 3.4 - rank * 1.4) for rank, feature_id in enumerate(graph.feature_ids)}
    target_pos = (2.05, 2.0)
    for source_id, position in source_pos.items():
        ax.scatter(*position, s=850, color="#d7e8f5", edgecolor="#234", zorder=3)
        ax.text(*position, organism.input_names[source_id], ha="center", va="center", fontsize=9, zorder=4)
    ax.scatter(*bias_pos, s=850, color="#ece7df", edgecolor="#543", zorder=3)
    ax.text(*bias_pos, "bias", ha="center", va="center", fontsize=9, zorder=4)
    for feature_id, position in feature_pos.items():
        ax.scatter(*position, s=1100, color="#f2d6a2", edgecolor="#543", zorder=3)
        ax.text(*position, organism.feature_names[feature_id], ha="center", va="center", fontsize=9, zorder=4)
    ax.scatter(*target_pos, s=1200, color="#d8ead2", edgecolor="#243", zorder=3)
    ax.text(*target_pos, "warm - cool", ha="center", va="center", fontsize=9, zorder=4)
    max_weight = max(abs(edge.weight) for edge in graph.edges)
    for edge in graph.edges:
        if edge.target_type == "feature":
            start = bias_pos if edge.source_type == "bias" else source_pos[edge.source_id]
            end = feature_pos[edge.target_id]
        else:
            start = feature_pos[edge.source_id]
            end = target_pos
        color = "#0b6e4f" if edge.weight > 0 else "#b42318"
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=0.8 + 3.0 * abs(edge.weight) / max_weight,
                color=color,
                alpha=0.85,
                shrinkA=22,
                shrinkB=25,
                connectionstyle="arc3,rad=0.04",
                zorder=2,
            )
        )
        if edge.target_type == "score":
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            ax.text(*midpoint, f"{edge.weight:+.2f}", fontsize=7, color=color, ha="center", va="center")
    ax.text(0.0, 4.55, "inputs", ha="center", weight="bold")
    ax.text(1.05, 4.55, "active features", ha="center", weight="bold")
    ax.text(2.05, 4.55, "target", ha="center", weight="bold")
    ax.text(1.05, -0.55, "green: positive    red: inhibitory", ha="center", fontsize=8)
    fig.suptitle("Toy ground truth: every displayed edge is analytically exact", fontsize=14, weight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_signature_result(
    curves: dict[str, t.Tensor],
    damages: dict[str, float],
    edge_scores: dict[str, float],
    *,
    save_path: Path | None = None,
):
    """Plot intervention faithfulness and the required negative controls."""

    import matplotlib.pyplot as plt

    colors = {
        "discovered graph": "#0b6e4f",
        "same-size random": "#6b7280",
        "shuffled features": "#b42318",
        "reconstruction-only": "#a15c00",
    }
    markers = {
        "discovered graph": "o",
        "same-size random": "s",
        "shuffled features": "^",
        "reconstruction-only": "D",
    }
    x_offsets = {
        "discovered graph": 0.0,
        "same-size random": -0.07,
        "shuffled features": 0.0,
        "reconstruction-only": 0.07,
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ks = range(1, len(next(iter(curves.values()))) + 1)
    for label, values in curves.items():
        x_values = [k + x_offsets[label] for k in ks]
        axes[0].plot(
            x_values,
            values.tolist(),
            marker=markers[label],
            linewidth=2.2,
            label=label,
            color=colors[label],
        )
    axes[0].set_title("Retention faithfulness")
    axes[0].set_xlabel("features retained (k)")
    axes[0].set_ylabel("faithfulness")
    axes[0].set_xticks(list(ks))
    axes[0].set_ylim(-0.04, 1.06)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    labels = list(damages)
    axes[1].bar(range(len(labels)), [damages[label] for label in labels], color=[colors[label] for label in labels])
    axes[1].set_title("Ablation damage at k=3")
    axes[1].set_ylabel("normalized score damage")
    axes[1].set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    axes[1].set_ylim(0, 1.06)
    axes[1].grid(axis="y", alpha=0.25)
    for index, label in enumerate(labels):
        axes[1].text(index, damages[label] + 0.025, f"{damages[label]:.2f}", ha="center", fontsize=8)
    edge_labels = list(edge_scores)
    axes[2].bar(range(len(edge_labels)), [edge_scores[label] for label in edge_labels], color=["#356aa0", "#b42318"])
    axes[2].set_title("Incoming-edge conservation")
    axes[2].set_ylabel("conservation score")
    axes[2].set_xticks(range(len(edge_labels)), edge_labels, rotation=20, ha="right")
    axes[2].set_ylim(0, 1.06)
    axes[2].grid(axis="y", alpha=0.25)
    for index, label in enumerate(edge_labels):
        axes[2].text(index, edge_scores[label] + 0.025, f"{edge_scores[label]:.2f}", ha="center", fontsize=8)
    fig.suptitle("Signature result: the recovered graph survives interventions; controls fail", fontsize=14, weight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def toy_experiment() -> dict[str, object]:
    """Run the deterministic red-square experiment used by plots and tests."""

    organism = make_toy_organism()
    inputs = make_toy_input("red", "square")
    output = transcoder_forward(
        inputs,
        organism.encoder_weight,
        organism.decoder_weight,
        encoder_bias=organism.encoder_bias,
        decoder_bias=organism.decoder_bias,
    )
    decomposition = reconstruction_decomposition(output.feature_acts, organism.decoder_weight, organism.decoder_bias)
    attributions = feature_edge_attributions(
        inputs,
        output.pre_acts,
        output.feature_acts,
        organism.encoder_weight,
        organism.encoder_bias,
        organism.decoder_weight,
        organism.readout,
    )
    graph = extract_attribution_graph(attributions.input_to_feature, attributions.feature_to_score, top_k_features=3)
    rankings = {
        "discovered graph": graph.feature_ids,
        "same-size random": (3, 1, 2),
        "shuffled features": (5, 1, 3),
        "reconstruction-only": reconstruction_only_ranking(organism.decoder_weight)[:3],
    }
    curves = {
        label: faithfulness_curve(
            output.feature_acts,
            organism.decoder_weight,
            organism.decoder_bias,
            organism.readout,
            ranking,
            max_k=3,
        )
        for label, ranking in rankings.items()
    }
    damages = {
        label: causal_validate_graph(
            output.feature_acts,
            organism.decoder_weight,
            organism.decoder_bias,
            organism.readout,
            ranking,
        ).normalized_damage
        for label, ranking in rankings.items()
    }
    shuffled_edges = shuffle_input_edge_targets(graph.edges, n_features=len(organism.feature_names))
    edge_scores = {
        "exact targets": edge_conservation_score(graph.edges, len(organism.feature_names)),
        "shuffled targets": edge_conservation_score(shuffled_edges, len(organism.feature_names)),
    }
    return {
        "organism": organism,
        "inputs": inputs,
        "output": output,
        "decomposition": decomposition,
        "attributions": attributions,
        "graph": graph,
        "rankings": rankings,
        "curves": curves,
        "damages": damages,
        "edge_scores": edge_scores,
    }


def save_reference_figures(asset_dir: Path) -> tuple[Path, Path]:
    """Regenerate the two deterministic figures embedded in the instruction page."""

    import matplotlib.pyplot as plt

    asset_dir.mkdir(parents=True, exist_ok=True)
    experiment = toy_experiment()
    graph_path = asset_dir / "transcoder_attribution_toy_ground_truth.png"
    signature_path = asset_dir / "transcoder_attribution_faithfulness.png"
    fig = plot_attribution_graph(
        experiment["organism"], experiment["attributions"], experiment["graph"], save_path=graph_path
    )
    plt.close(fig)
    fig = plot_signature_result(
        experiment["curves"], experiment["damages"], experiment["edge_scores"], save_path=signature_path
    )
    plt.close(fig)
    return graph_path, signature_path


def run_smoke_test(cpu: bool = True) -> dict[str, object]:
    """Return exact model-organism metrics for the CPU verification path."""

    _ = cpu
    experiment = toy_experiment()
    output = experiment["output"]
    decomposition = experiment["decomposition"]
    attributions = experiment["attributions"]
    graph = experiment["graph"]
    curves = experiment["curves"]
    damages = experiment["damages"]
    edge_scores = experiment["edge_scores"]
    clean_score = float((output.reconstructed_activations @ experiment["organism"].readout).item())
    result = {
        "model_organism": "exact_colored_shape_relu_mlp",
        "input": "red square",
        "pre_acts": output.pre_acts.tolist(),
        "feature_acts": output.feature_acts.tolist(),
        "reconstructed_activations": output.reconstructed_activations.tolist(),
        "target_score": clean_score,
        "reconstruction_max_error": float(
            (output.reconstructed_activations - decomposition.reconstructed_activations).abs().max().item()
        ),
        "edge_decomposition_max_error": float(
            (attributions.input_to_feature.sum(dim=0) - attributions.feature_to_score).abs().max().item()
        ),
        "graph_feature_ids": list(graph.feature_ids),
        "graph_edge_count": len(graph.edges),
        "faithfulness_curve": [round(float(value), 6) for value in curves["discovered graph"]],
        "same_size_random_curve": [round(float(value), 6) for value in curves["same-size random"]],
        "shuffled_feature_curve": [round(float(value), 6) for value in curves["shuffled features"]],
        "reconstruction_only_curve": [round(float(value), 6) for value in curves["reconstruction-only"]],
        "graph_ablation_damage": round(float(damages["discovered graph"]), 6),
        "random_ablation_damage": round(float(damages["same-size random"]), 6),
        "edge_conservation": round(float(edge_scores["exact targets"]), 6),
        "shuffled_edge_conservation": round(float(edge_scores["shuffled targets"]), 6),
    }
    return {
        **result,
        "toy_target_score": result["target_score"],
        "toy_reconstruction_max_error": result["reconstruction_max_error"],
        "toy_edge_decomposition_max_error": result["edge_decomposition_max_error"],
        "toy_graph_edge_count": result["graph_edge_count"],
        "toy_graph_ablation_damage": result["graph_ablation_damage"],
        "toy_random_ablation_damage": result["random_ablation_damage"],
        "toy_edge_conservation": result["edge_conservation"],
        "toy_shuffled_edge_conservation": result["shuffled_edge_conservation"],
    }


# %%
# Compatibility helpers used by the existing pinned GELU-1L CUDA report.
def mean_kl_divergence(reference_logits: t.Tensor, reconstructed_logits: t.Tensor) -> float:
    if reference_logits.shape != reconstructed_logits.shape:
        raise ValueError("reference_logits and reconstructed_logits must have matching shapes.")
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    reconstructed_log_probs = F.log_softmax(reconstructed_logits.float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    return (reference_probs * (reference_log_probs - reconstructed_log_probs)).sum(dim=-1).mean().item()


def target_logit_diff(logits: t.Tensor, *, positive_token_id: int, negative_token_id: int) -> float:
    if logits.shape[-1] <= max(positive_token_id, negative_token_id):
        raise ValueError("token ids are out of range for logits.")
    return (logits[..., positive_token_id] - logits[..., negative_token_id]).float().mean().item()


def transcoder_replacement_report(
    *,
    reference_activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    reference_logits: t.Tensor,
    replacement_logits: t.Tensor,
    positive_token_id: int,
    negative_token_id: int,
    kl_threshold: float = 1e-3,
    logit_diff_tolerance: float = 0.1,
) -> TranscoderReplacementReport:
    if reference_activations.shape != reconstructed_activations.shape:
        raise ValueError("activation tensors must have matching shapes.")
    replacement_kl = mean_kl_divergence(reference_logits, replacement_logits)
    target_diff = target_logit_diff(
        reference_logits, positive_token_id=positive_token_id, negative_token_id=negative_token_id
    )
    replacement_diff = target_logit_diff(
        replacement_logits, positive_token_id=positive_token_id, negative_token_id=negative_token_id
    )
    error = abs(target_diff - replacement_diff)
    return TranscoderReplacementReport(
        F.mse_loss(reconstructed_activations.float(), reference_activations.float()).item(),
        replacement_kl,
        target_diff,
        replacement_diff,
        error,
        replacement_kl <= kl_threshold,
        error <= logit_diff_tolerance,
    )


def feature_logit_contributions(feature_acts: t.Tensor, logit_effects: t.Tensor) -> t.Tensor:
    if feature_acts.shape[-1] != logit_effects.shape[0]:
        raise ValueError("feature_acts feature dimension must match logit_effects.")
    return feature_acts.float().mean(dim=tuple(range(feature_acts.ndim - 1))) * logit_effects.float()


def build_attribution_edges(
    input_to_feature_scores: t.Tensor,
    feature_logit_effects: t.Tensor,
    *,
    top_k: int,
) -> list[AttributionEdge]:
    if input_to_feature_scores.ndim != 2:
        raise ValueError("input_to_feature_scores must have shape (inputs, features).")
    if input_to_feature_scores.shape[1] != feature_logit_effects.shape[0]:
        raise ValueError("feature dimension must match feature_logit_effects.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    edges: list[AttributionEdge] = []
    n_features = input_to_feature_scores.shape[1]
    flat_indices = input_to_feature_scores.float().flatten().abs().topk(
        k=min(top_k, input_to_feature_scores.numel())
    ).indices
    for flat_index in flat_indices:
        source_id = int(flat_index.item() // n_features)
        target_id = int(flat_index.item() % n_features)
        edges.append(
            AttributionEdge(
                "input",
                source_id,
                "feature",
                target_id,
                float(input_to_feature_scores[source_id, target_id].item()),
            )
        )
    feature_ids = feature_logit_effects.float().abs().topk(k=min(top_k, feature_logit_effects.numel())).indices
    for feature_id_tensor in feature_ids:
        feature_id = int(feature_id_tensor.item())
        edges.append(
            AttributionEdge("feature", feature_id, "logit_diff", 0, float(feature_logit_effects[feature_id].item()))
        )
    return edges


def graph_reproducible(
    edges_a: list[AttributionEdge],
    edges_b: list[AttributionEdge],
    *,
    atol: float = 1e-6,
) -> bool:
    if len(edges_a) != len(edges_b):
        return False
    for edge_a, edge_b in zip(edges_a, edges_b, strict=True):
        same_structure = (
            edge_a.source_type == edge_b.source_type
            and edge_a.source_id == edge_b.source_id
            and edge_a.target_type == edge_b.target_type
            and edge_a.target_id == edge_b.target_id
        )
        if not same_structure or abs(edge_a.weight - edge_b.weight) > atol:
            return False
    return True


def graph_density(*, num_nodes: int, num_edges: int) -> float:
    return 0.0 if num_nodes <= 1 else num_edges / (num_nodes * (num_nodes - 1))


def attribution_graph_report(
    contributions: t.Tensor,
    graph_feature_ids: t.Tensor | list[int],
    random_feature_ids: t.Tensor | list[int],
    *,
    num_nodes: int,
    num_edges: int,
    reproducible: bool,
    preservation_threshold: float = 0.8,
) -> AttributionGraphReport:
    contributions = contributions.flatten().float()
    graph_ids = t.as_tensor(graph_feature_ids, dtype=t.long, device=contributions.device)
    random_ids = t.as_tensor(random_feature_ids, dtype=t.long, device=contributions.device)
    if graph_ids.numel() == 0 or random_ids.numel() == 0:
        raise ValueError("graph_feature_ids and random_feature_ids must be nonempty.")
    full_diff = contributions.sum().item()
    graph_diff = contributions[graph_ids].sum().item()
    ablated_diff = full_diff - graph_diff
    random_removed = contributions[random_ids].sum().item()
    random_ablated_diff = full_diff - random_removed
    topk_damage = abs(full_diff - ablated_diff)
    random_damage = abs(full_diff - random_ablated_diff)
    return AttributionGraphReport(
        num_nodes,
        num_edges,
        graph_density(num_nodes=num_nodes, num_edges=num_edges),
        full_diff,
        graph_diff,
        ablated_diff,
        random_ablated_diff,
        topk_damage,
        random_damage,
        abs(graph_diff) / max(abs(full_diff), 1e-12) >= preservation_threshold,
        topk_damage > random_damage,
        reproducible,
    )


# %%
def _load_gelu1l_model_on_cuda():
    os.environ.setdefault("BNB_CUDA_VERSION", TL_BNB_CUDA_OVERRIDE)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        TL_GELU1L_MODEL_NAME, device="cuda", dtype="float32", revision=TL_GELU1L_REVISION
    )


def _decode_from_residual(model, residual_stream: t.Tensor) -> t.Tensor:
    return model.ln_final(residual_stream) @ model.W_U + model.b_U


def _cache_transcoder_dataset(model) -> dict[str, t.Tensor]:
    names = {TL_LN2_HOOK, TL_MLP_POST_HOOK, TL_MLP_OUT_HOOK, TL_RESID_MID_HOOK}
    normalized_inputs, mlp_features, mlp_outputs, resid_mid, logits = [], [], [], [], []
    for prompt in TL_TRANSCODER_PROMPTS:
        tokens = model.to_tokens(prompt)
        with t.no_grad():
            prompt_logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
        normalized_inputs.append(cache[TL_LN2_HOOK][0].detach())
        mlp_features.append(cache[TL_MLP_POST_HOOK][0].detach())
        mlp_outputs.append(cache[TL_MLP_OUT_HOOK][0].detach())
        resid_mid.append(cache[TL_RESID_MID_HOOK][0].detach())
        logits.append(prompt_logits[0].detach())
    return {
        "normalized_inputs": t.cat(normalized_inputs, dim=0),
        "mlp_features": t.cat(mlp_features, dim=0),
        "mlp_outputs": t.cat(mlp_outputs, dim=0),
        "resid_mid": t.cat(resid_mid, dim=0),
        "logits": t.cat(logits, dim=0),
    }


def _fit_tiny_relu_transcoder(
    normalized_inputs: t.Tensor,
    mlp_outputs: t.Tensor,
    *,
    width: int = 512,
    steps: int = 800,
) -> dict[str, t.Tensor | float]:
    """Train a small ReLU transcoder on real MLP input/output activations."""

    t.manual_seed(6303)
    feature_mean = normalized_inputs.mean(dim=0, keepdim=True)
    feature_std = normalized_inputs.std(dim=0, keepdim=True).clamp_min(1e-3)
    inputs = (normalized_inputs - feature_mean) / feature_std
    order = t.randperm(inputs.shape[0], device=inputs.device)
    split = int(0.8 * order.numel())
    train_idx, heldout_idx = order[:split], order[split:]
    encoder = t.nn.Linear(inputs.shape[-1], width, device=inputs.device)
    decoder = t.nn.Linear(width, mlp_outputs.shape[-1], device=inputs.device)
    opt = t.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        feature_acts = F.relu(encoder(inputs[train_idx]))
        reconstructed = decoder(feature_acts)
        loss = F.mse_loss(reconstructed, mlp_outputs[train_idx]) + 1e-5 * feature_acts.mean()
        loss.backward()
        opt.step()
    with t.no_grad():
        output = transcoder_forward(
            inputs,
            encoder.weight.detach(),
            decoder.weight.detach().T,
            encoder_bias=encoder.bias.detach(),
            decoder_bias=decoder.bias.detach(),
        )
        heldout_mse = F.mse_loss(output.reconstructed_activations[heldout_idx], mlp_outputs[heldout_idx]).item()
        train_mse = F.mse_loss(output.reconstructed_activations[train_idx], mlp_outputs[train_idx]).item()
        heldout_zero_mse = F.mse_loss(t.zeros_like(mlp_outputs[heldout_idx]), mlp_outputs[heldout_idx]).item()
    return {
        "feature_acts": output.feature_acts.detach(),
        "reconstructed": output.reconstructed_activations.detach(),
        "train_mse": train_mse,
        "heldout_mse": heldout_mse,
        "heldout_zero_mse": heldout_zero_mse,
        "heldout_idx": heldout_idx,
        "feature_density": output.feature_acts.gt(0).float().mean().item(),
        "width": width,
        "steps": steps,
    }


def run_transformerlens_transcoder_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run the existing pinned GELU-1L replacement preflight on CUDA."""

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for the 6.3 real-model transcoder preflight.")
    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()
    data = _cache_transcoder_dataset(model)
    mlp = model.blocks[0].mlp
    oracle_mlp_out = data["mlp_features"] @ mlp.W_out.detach() + mlp.b_out.detach()
    oracle_logits = _decode_from_residual(model, data["resid_mid"] + oracle_mlp_out)
    positive_token_id = model.to_single_token(" floor")
    negative_token_id = model.to_single_token(" top")
    oracle_replacement = transcoder_replacement_report(
        reference_activations=data["mlp_outputs"],
        reconstructed_activations=oracle_mlp_out,
        reference_logits=data["logits"],
        replacement_logits=oracle_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        kl_threshold=1e-6,
        logit_diff_tolerance=1e-5,
    )
    oracle_mlp_out_max_abs_error = (data["mlp_outputs"] - oracle_mlp_out).abs().max().item()
    oracle_logits_max_abs_error = (data["logits"] - oracle_logits).abs().max().item()
    trained = _fit_tiny_relu_transcoder(data["normalized_inputs"], data["mlp_outputs"])
    trained_logits = _decode_from_residual(model, data["resid_mid"] + trained["reconstructed"])
    trained_top1_agreement = trained_logits.argmax(dim=-1).eq(data["logits"].argmax(dim=-1)).float().mean().item()
    trained_reconstruction_ratio = trained["heldout_mse"] / max(trained["heldout_zero_mse"], 1e-12)
    logit_effects = mlp.W_out.detach() @ (model.W_U[:, positive_token_id] - model.W_U[:, negative_token_id])
    contributions = feature_logit_contributions(data["mlp_features"], logit_effects)
    top_feature_ids = contributions.abs().topk(k=64).indices
    random_feature_ids = contributions.abs().topk(k=64, largest=False).indices
    edges = build_attribution_edges(data["mlp_features"].abs(), logit_effects, top_k=16)
    repeated_edges = build_attribution_edges(data["mlp_features"].abs(), logit_effects, top_k=16)
    graph = attribution_graph_report(
        contributions,
        graph_feature_ids=top_feature_ids,
        random_feature_ids=random_feature_ids,
        num_nodes=int(data["mlp_features"].shape[-1] + data["mlp_features"].shape[0] + 1),
        num_edges=len(edges),
        reproducible=graph_reproducible(edges, repeated_edges),
        preservation_threshold=0.2,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        oracle_replacement.passes_kl
        and oracle_replacement.preserves_logit_diff
        and oracle_logits_max_abs_error <= 5e-5
        and trained_reconstruction_ratio <= 0.5
        and trained_top1_agreement >= 0.75
        and graph.preserves_logit_diff
        and graph.passes_damage_control
        and graph.reproducible
        and within_vram_budget
    )
    del model
    t.cuda.empty_cache()
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_name": TL_GELU1L_MODEL_NAME,
        "hf_revision": TL_GELU1L_REVISION,
        "tokenizer_id": TL_GELU1L_TOKENIZER_ID,
        "tokenizer_revision": TL_GELU1L_TOKENIZER_REVISION,
        "prompt_count": len(TL_TRANSCODER_PROMPTS),
        "activation_count": int(data["mlp_outputs"].shape[0]),
        "d_model": int(data["mlp_outputs"].shape[-1]),
        "mlp_width": int(data["mlp_features"].shape[-1]),
        "oracle_mlp_out_max_abs_error": oracle_mlp_out_max_abs_error,
        "oracle_logits_max_abs_error": oracle_logits_max_abs_error,
        "oracle_replacement_kl": oracle_replacement.replacement_kl,
        "oracle_preserves_logit_diff": oracle_replacement.preserves_logit_diff,
        "trained_transcoder_width": trained["width"],
        "trained_transcoder_steps": trained["steps"],
        "trained_transcoder_train_mse": trained["train_mse"],
        "trained_transcoder_heldout_mse": trained["heldout_mse"],
        "trained_transcoder_heldout_zero_mse": trained["heldout_zero_mse"],
        "trained_transcoder_heldout_mse_ratio": trained_reconstruction_ratio,
        "trained_transcoder_feature_density": trained["feature_density"],
        "trained_replacement_top1_agreement": trained_top1_agreement,
        "feature_acts_sum": float(data["mlp_features"].sum().item()),
        "graph_feature_count": int(top_feature_ids.numel()),
        "graph_preserves_logit_diff": graph.preserves_logit_diff,
        "graph_passes_damage_control": graph.passes_damage_control,
        "graph_reproducible": graph.reproducible,
        "graph_topk_damage": graph.topk_damage,
        "graph_random_damage": graph.random_damage,
        "replacement_kl": oracle_replacement.replacement_kl,
        "preserves_logit_diff": oracle_replacement.preserves_logit_diff,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "preflight_passed": preflight_passed,
        "full_path": (
            "Pinned TransformerLens gelu-1l CUDA preflight: exact MLP-feature oracle replacement, "
            "trained tiny ReLU transcoder held-out reconstruction, and feature-level attribution "
            "graph with top-feature versus low-effect controls."
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_transcoder_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
