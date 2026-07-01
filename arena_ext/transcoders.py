"""Toy transcoder and attribution-graph utilities for sparse feature notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
import torch.nn.functional as F

from arena_ext.features import mean_kl_divergence


@dataclass(frozen=True)
class TranscoderOutput:
    feature_acts: t.Tensor
    reconstructed_activations: t.Tensor


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
class AttributionEdge:
    source_type: str
    source_id: int
    target_type: str
    target_id: int
    weight: float


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


def transcoder_forward(
    inputs: t.Tensor,
    encoder_weight: t.Tensor,
    decoder_weight: t.Tensor,
    *,
    encoder_bias: t.Tensor | None = None,
    decoder_bias: t.Tensor | None = None,
) -> TranscoderOutput:
    """Run a one-hidden-layer ReLU transcoder."""

    if inputs.shape[-1] != encoder_weight.shape[-1]:
        raise ValueError("inputs last dimension must match encoder_weight input dimension.")
    if encoder_weight.shape[0] != decoder_weight.shape[0]:
        raise ValueError("encoder features must match decoder rows.")

    pre_acts = inputs.float() @ encoder_weight.float().T
    if encoder_bias is not None:
        pre_acts = pre_acts + encoder_bias.to(pre_acts.device)
    feature_acts = pre_acts.clamp_min(0)
    reconstructed = feature_acts @ decoder_weight.float()
    if decoder_bias is not None:
        reconstructed = reconstructed + decoder_bias.to(reconstructed.device)
    return TranscoderOutput(
        feature_acts=feature_acts,
        reconstructed_activations=reconstructed,
    )


def target_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    """Mean target-minus-negative logit difference."""

    if logits.shape[-1] <= max(positive_token_id, negative_token_id):
        raise ValueError("token ids are out of range for logits.")
    diff = logits[..., positive_token_id] - logits[..., negative_token_id]
    return diff.float().mean().item()


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
    """Check whether replacing an MLP with a transcoder preserves behavior."""

    if reference_activations.shape != reconstructed_activations.shape:
        raise ValueError("activation tensors must have matching shapes.")
    replacement_kl = mean_kl_divergence(reference_logits, replacement_logits)
    target_diff = target_logit_diff(
        reference_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    replacement_diff = target_logit_diff(
        replacement_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    error = abs(target_diff - replacement_diff)
    return TranscoderReplacementReport(
        reconstruction_mse=F.mse_loss(
            reconstructed_activations.float(),
            reference_activations.float(),
        ).item(),
        replacement_kl=replacement_kl,
        target_logit_diff=target_diff,
        replacement_logit_diff=replacement_diff,
        logit_diff_error=error,
        passes_kl=replacement_kl <= kl_threshold,
        preserves_logit_diff=error <= logit_diff_tolerance,
    )


def feature_logit_contributions(feature_acts: t.Tensor, logit_effects: t.Tensor) -> t.Tensor:
    """Estimate feature contributions to a target logit difference."""

    if feature_acts.shape[-1] != logit_effects.shape[0]:
        raise ValueError("feature_acts feature dimension must match logit_effects.")
    reduce_dims = tuple(range(feature_acts.ndim - 1))
    mean_feature_acts = feature_acts.float().mean(dim=reduce_dims)
    return mean_feature_acts * logit_effects.float()


def build_attribution_edges(
    input_to_feature_scores: t.Tensor,
    feature_logit_effects: t.Tensor,
    *,
    top_k: int,
) -> list[AttributionEdge]:
    """Build a small input-feature-logit attribution graph."""

    if input_to_feature_scores.ndim != 2:
        raise ValueError("input_to_feature_scores must have shape (inputs, features).")
    if input_to_feature_scores.shape[1] != feature_logit_effects.shape[0]:
        raise ValueError("feature dimension must match feature_logit_effects.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    edges: list[AttributionEdge] = []
    flat_scores = input_to_feature_scores.float().flatten()
    k_inputs = min(top_k, flat_scores.numel())
    values, flat_indices = flat_scores.abs().topk(k=k_inputs)
    n_features = input_to_feature_scores.shape[1]
    for value, flat_index in zip(values, flat_indices, strict=True):
        source_id = int(flat_index.item() // n_features)
        target_id = int(flat_index.item() % n_features)
        signed_weight = input_to_feature_scores[source_id, target_id].item()
        edges.append(
            AttributionEdge(
                source_type="input",
                source_id=source_id,
                target_type="feature",
                target_id=target_id,
                weight=float(signed_weight if value >= 0 else -value.item()),
            )
        )

    k_outputs = min(top_k, feature_logit_effects.numel())
    output_values, feature_ids = feature_logit_effects.float().abs().topk(k=k_outputs)
    for value, feature_id in zip(output_values, feature_ids, strict=True):
        signed_weight = feature_logit_effects[feature_id].item()
        edges.append(
            AttributionEdge(
                source_type="feature",
                source_id=int(feature_id.item()),
                target_type="logit_diff",
                target_id=0,
                weight=float(signed_weight if value >= 0 else -value.item()),
            )
        )
    return edges


def graph_density(*, num_nodes: int, num_edges: int) -> float:
    """Directed graph density without self-loops."""

    if num_nodes <= 1:
        return 0.0
    return num_edges / (num_nodes * (num_nodes - 1))


def graph_reproducible(
    edges_a: list[AttributionEdge],
    edges_b: list[AttributionEdge],
    *,
    atol: float = 1e-6,
) -> bool:
    """Check whether two graph edge lists match in structure and weight."""

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
    """Report graph preservation and node-ablation controls."""

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
    preserved_fraction = abs(graph_diff) / max(abs(full_diff), 1e-12)

    return AttributionGraphReport(
        num_nodes=num_nodes,
        num_edges=num_edges,
        density=graph_density(num_nodes=num_nodes, num_edges=num_edges),
        full_logit_diff=full_diff,
        graph_logit_diff=graph_diff,
        ablated_logit_diff=ablated_diff,
        random_ablated_logit_diff=random_ablated_diff,
        topk_damage=topk_damage,
        random_damage=random_damage,
        preserves_logit_diff=preserved_fraction >= preservation_threshold,
        passes_damage_control=topk_damage > random_damage,
        reproducible=reproducible,
    )
