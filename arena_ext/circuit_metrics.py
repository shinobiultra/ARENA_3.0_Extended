"""Circuit discovery metrics for automated circuit notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch as t

from arena_ext.activation_language import prediction_accuracy


@dataclass(frozen=True)
class ActivationPatchingSweep:
    patch_scores: t.Tensor
    best_index: int
    best_score: float


@dataclass(frozen=True)
class ACDCPruningReport:
    kept_edges: tuple[str, ...]
    removed_edges: tuple[str, ...]
    threshold: float
    num_kept: int


@dataclass(frozen=True)
class CircuitFaithfulnessReport:
    full_metric: float
    corrupt_metric: float
    circuit_metric: float
    preserved_fraction: float
    passes_faithfulness: bool


@dataclass(frozen=True)
class CircuitMinimalityReport:
    circuit_metric: float
    ablated_metric: float
    metric_damage: float
    passes_minimality: bool


@dataclass(frozen=True)
class CircuitCompletenessReport:
    circuit_metric: float
    expanded_metric: float
    omitted_node_gain: float
    passes_completeness: bool


@dataclass(frozen=True)
class RandomCircuitBaselineReport:
    circuit_metric: float
    random_metric: float
    margin: float
    circuit_beats_random: bool


@dataclass(frozen=True)
class OODTemplateReport:
    per_template_accuracy: dict[int, float]
    worst_template_accuracy: float
    passes_ood: bool


@dataclass(frozen=True)
class CircuitMethodComparisonReport:
    exact_top_edges: tuple[str, ...]
    method_top_edges: dict[str, tuple[str, ...]]
    topk_overlap: dict[str, float]
    score_correlations: dict[str, float]
    circuit_sizes: dict[str, int]
    best_matching_method: str
    passes_comparison: bool


@dataclass(frozen=True)
class ToyCircuitEvaluationReport:
    ground_truth_edges: tuple[str, ...]
    discovered_edges: tuple[str, ...]
    exact_match: bool
    full_metric: float
    corrupt_metric: float
    circuit_metric: float
    random_metric: float
    preserved_fraction: float
    minimality_damage: float
    completeness_gain: float
    random_margin: float
    passes: bool


def _require_finite_tensor(tensor: t.Tensor, name: str) -> None:
    if not t.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite.")


def _require_finite_scalar(value: float, name: str) -> None:
    if not t.isfinite(t.tensor(value)).item():
        raise ValueError(f"{name} must be finite.")


def answer_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    """Return mean positive-minus-negative logit difference."""

    if logits.ndim < 1:
        raise ValueError("logits must have a vocabulary dimension.")
    vocab_size = logits.shape[-1]
    if vocab_size == 0:
        raise ValueError("logits vocabulary dimension must be nonempty.")
    if positive_token_id == negative_token_id:
        raise ValueError("positive_token_id and negative_token_id must differ.")
    if not 0 <= positive_token_id < vocab_size:
        raise ValueError("positive_token_id is out of range.")
    if not 0 <= negative_token_id < vocab_size:
        raise ValueError("negative_token_id is out of range.")
    _require_finite_tensor(logits, "logits")
    diff = logits[..., positive_token_id] - logits[..., negative_token_id]
    return diff.float().mean().item()


def activation_patching_sweep(
    *,
    clean_metric: float,
    corrupt_metric: float,
    patched_metrics: t.Tensor,
) -> ActivationPatchingSweep:
    """Convert per-component patched metrics into normalized recovery scores."""

    denominator = clean_metric - corrupt_metric
    for name, value in {
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
    }.items():
        _require_finite_scalar(value, name)
    if denominator == 0:
        raise ValueError("clean_metric and corrupt_metric must differ.")
    if patched_metrics.ndim != 1:
        raise ValueError("patched_metrics must be rank-1.")
    if patched_metrics.numel() == 0:
        raise ValueError("patched_metrics must be nonempty.")
    _require_finite_tensor(patched_metrics, "patched_metrics")
    patch_scores = (patched_metrics.float() - corrupt_metric) / denominator
    best_index = int(patch_scores.argmax().item())
    return ActivationPatchingSweep(
        patch_scores=patch_scores,
        best_index=best_index,
        best_score=float(patch_scores[best_index].item()),
    )


def toy_acdc_graph() -> dict[str, object]:
    """Return a tiny graph with two true causal edges and two decoys."""

    return {
        "edge_names": [
            "color_input -> answer_logit",
            "shape_input -> answer_logit",
            "background_input -> answer_logit",
            "position_input -> answer_logit",
        ],
        "clean_edge_values": t.tensor([2.0, 1.5, 0.2, -0.1]),
        "corrupt_edge_values": t.tensor([0.0, 0.0, 0.2, -0.1]),
        "ground_truth_edges": (
            "color_input -> answer_logit",
            "shape_input -> answer_logit",
        ),
        "same_size_random_edges": (
            "background_input -> answer_logit",
            "position_input -> answer_logit",
        ),
        "threshold": 0.35,
    }


def _validate_edge_values(
    clean_edge_values: t.Tensor,
    corrupt_edge_values: t.Tensor,
    edge_names: list[str],
) -> tuple[t.Tensor, t.Tensor]:
    clean = clean_edge_values.flatten().float()
    corrupt = corrupt_edge_values.flatten().float()
    if clean.numel() == 0:
        raise ValueError("edge values must be nonempty.")
    if clean.shape != corrupt.shape:
        raise ValueError("clean and corrupt edge values must have the same shape.")
    if clean.numel() != len(edge_names):
        raise ValueError("edge values and edge_names must align.")
    if any(not name for name in edge_names):
        raise ValueError("edge_names must be nonempty strings.")
    _require_finite_tensor(clean, "clean_edge_values")
    _require_finite_tensor(corrupt, "corrupt_edge_values")
    return clean, corrupt


def patch_toy_graph_edges(
    clean_edge_values: t.Tensor,
    corrupt_edge_values: t.Tensor,
    edge_names: list[str],
    patched_edges: tuple[str, ...] | list[str],
) -> float:
    """Patch named clean edge contributions into a corrupt toy graph."""

    clean, corrupt = _validate_edge_values(clean_edge_values, corrupt_edge_values, edge_names)
    name_to_index = {name: index for index, name in enumerate(edge_names)}
    patched = corrupt.clone()
    for edge in patched_edges:
        if edge not in name_to_index:
            raise ValueError(f"unknown edge: {edge}")
        index = name_to_index[edge]
        patched[index] = clean[index]
    return float(patched.sum().item())


def exact_toy_edge_patch_scores(
    clean_edge_values: t.Tensor,
    corrupt_edge_values: t.Tensor,
    edge_names: list[str],
) -> ActivationPatchingSweep:
    """Score each toy edge by exact single-edge patch recovery."""

    clean, corrupt = _validate_edge_values(clean_edge_values, corrupt_edge_values, edge_names)
    clean_metric = float(clean.sum().item())
    corrupt_metric = float(corrupt.sum().item())
    patched_metrics = t.tensor(
        [
            patch_toy_graph_edges(clean, corrupt, edge_names, [edge_name])
            for edge_name in edge_names
        ],
        dtype=t.float32,
    )
    return activation_patching_sweep(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        patched_metrics=patched_metrics,
    )


def evaluate_toy_circuit(
    clean_edge_values: t.Tensor,
    corrupt_edge_values: t.Tensor,
    edge_names: list[str],
    ground_truth_edges: tuple[str, ...],
    random_edges: tuple[str, ...],
    *,
    threshold: float = 0.35,
) -> ToyCircuitEvaluationReport:
    """Recover and evaluate a known toy circuit from exact edge patch scores."""

    clean, corrupt = _validate_edge_values(clean_edge_values, corrupt_edge_values, edge_names)
    sweep = exact_toy_edge_patch_scores(clean, corrupt, edge_names)
    pruning = acdc_pruning_report(sweep.patch_scores, edge_names, threshold=threshold)
    discovered_edges = pruning.kept_edges
    full_metric = float(clean.sum().item())
    corrupt_metric = float(corrupt.sum().item())
    circuit_metric = patch_toy_graph_edges(clean, corrupt, edge_names, discovered_edges)
    random_metric = patch_toy_graph_edges(clean, corrupt, edge_names, random_edges)

    if not discovered_edges:
        ablated_metric = corrupt_metric
    else:
        ablated_metric = max(
            patch_toy_graph_edges(
                clean,
                corrupt,
                edge_names,
                tuple(edge for edge in discovered_edges if edge != edge_to_remove),
            )
            for edge_to_remove in discovered_edges
        )

    omitted_edges = [edge for edge in edge_names if edge not in discovered_edges]
    if omitted_edges:
        top_omitted = max(
            omitted_edges,
            key=lambda edge: float(sweep.patch_scores[edge_names.index(edge)].item()),
        )
        expanded_metric = patch_toy_graph_edges(
            clean,
            corrupt,
            edge_names,
            (*discovered_edges, top_omitted),
        )
    else:
        expanded_metric = circuit_metric

    faithfulness = circuit_faithfulness_report(
        full_metric=full_metric,
        corrupt_metric=corrupt_metric,
        circuit_metric=circuit_metric,
        min_preserved_fraction=0.99,
    )
    minimality = circuit_minimality_report(
        circuit_metric=circuit_metric,
        ablated_metric=ablated_metric,
        min_metric_damage=1.0,
    )
    completeness = circuit_completeness_report(
        circuit_metric=circuit_metric,
        expanded_metric=expanded_metric,
        max_omitted_node_gain=1e-6,
    )
    random_baseline = random_circuit_baseline_report(
        circuit_metric=circuit_metric,
        random_metric=random_metric,
        min_margin=1.0,
    )
    exact_match = set(discovered_edges) == set(ground_truth_edges)
    passes = (
        exact_match
        and faithfulness.passes_faithfulness
        and minimality.passes_minimality
        and completeness.passes_completeness
        and random_baseline.circuit_beats_random
    )
    return ToyCircuitEvaluationReport(
        ground_truth_edges=tuple(ground_truth_edges),
        discovered_edges=tuple(discovered_edges),
        exact_match=exact_match,
        full_metric=full_metric,
        corrupt_metric=corrupt_metric,
        circuit_metric=circuit_metric,
        random_metric=random_metric,
        preserved_fraction=faithfulness.preserved_fraction,
        minimality_damage=minimality.metric_damage,
        completeness_gain=completeness.omitted_node_gain,
        random_margin=random_baseline.margin,
        passes=passes,
    )


def acdc_pruning_report(
    edge_scores: t.Tensor,
    edge_names: list[str],
    *,
    threshold: float,
) -> ACDCPruningReport:
    """Keep edges whose score survives an ACDC-style threshold."""

    scores = edge_scores.flatten().float()
    if scores.numel() == 0:
        raise ValueError("edge_scores must be nonempty.")
    if scores.numel() != len(edge_names):
        raise ValueError("edge_scores and edge_names must align.")
    if any(not name for name in edge_names):
        raise ValueError("edge_names must be nonempty strings.")
    _require_finite_tensor(scores, "edge_scores")
    _require_finite_scalar(threshold, "threshold")
    kept = []
    removed = []
    for score, name in zip(scores.tolist(), edge_names):
        if score >= threshold:
            kept.append(name)
        else:
            removed.append(name)
    return ACDCPruningReport(
        kept_edges=tuple(kept),
        removed_edges=tuple(removed),
        threshold=threshold,
        num_kept=len(kept),
    )


def circuit_faithfulness_report(
    *,
    full_metric: float,
    corrupt_metric: float,
    circuit_metric: float,
    min_preserved_fraction: float = 0.75,
) -> CircuitFaithfulnessReport:
    """Check how much clean-vs-corrupt behavior the circuit preserves."""

    for name, value in {
        "full_metric": full_metric,
        "corrupt_metric": corrupt_metric,
        "circuit_metric": circuit_metric,
        "min_preserved_fraction": min_preserved_fraction,
    }.items():
        _require_finite_scalar(value, name)
    denominator = full_metric - corrupt_metric
    if denominator == 0:
        raise ValueError("full_metric and corrupt_metric must differ.")
    if min_preserved_fraction < 0:
        raise ValueError("min_preserved_fraction must be nonnegative.")
    preserved_fraction = (circuit_metric - corrupt_metric) / denominator
    return CircuitFaithfulnessReport(
        full_metric=full_metric,
        corrupt_metric=corrupt_metric,
        circuit_metric=circuit_metric,
        preserved_fraction=preserved_fraction,
        passes_faithfulness=preserved_fraction >= min_preserved_fraction,
    )


def circuit_minimality_report(
    *,
    circuit_metric: float,
    ablated_metric: float,
    min_metric_damage: float = 0.5,
) -> CircuitMinimalityReport:
    """Check whether removing circuit nodes damages behavior."""

    for name, value in {
        "circuit_metric": circuit_metric,
        "ablated_metric": ablated_metric,
        "min_metric_damage": min_metric_damage,
    }.items():
        _require_finite_scalar(value, name)
    if min_metric_damage < 0:
        raise ValueError("min_metric_damage must be nonnegative.")
    metric_damage = circuit_metric - ablated_metric
    return CircuitMinimalityReport(
        circuit_metric=circuit_metric,
        ablated_metric=ablated_metric,
        metric_damage=metric_damage,
        passes_minimality=metric_damage >= min_metric_damage,
    )


def circuit_completeness_report(
    *,
    circuit_metric: float,
    expanded_metric: float,
    max_omitted_node_gain: float = 0.2,
) -> CircuitCompletenessReport:
    """Check whether adding top omitted nodes improves little."""

    for name, value in {
        "circuit_metric": circuit_metric,
        "expanded_metric": expanded_metric,
        "max_omitted_node_gain": max_omitted_node_gain,
    }.items():
        _require_finite_scalar(value, name)
    if max_omitted_node_gain < 0:
        raise ValueError("max_omitted_node_gain must be nonnegative.")
    omitted_node_gain = expanded_metric - circuit_metric
    return CircuitCompletenessReport(
        circuit_metric=circuit_metric,
        expanded_metric=expanded_metric,
        omitted_node_gain=omitted_node_gain,
        passes_completeness=omitted_node_gain <= max_omitted_node_gain,
    )


def random_circuit_baseline_report(
    *,
    circuit_metric: float,
    random_metric: float,
    min_margin: float = 0.5,
) -> RandomCircuitBaselineReport:
    """Check that a discovered circuit beats a same-size random circuit."""

    for name, value in {
        "circuit_metric": circuit_metric,
        "random_metric": random_metric,
        "min_margin": min_margin,
    }.items():
        _require_finite_scalar(value, name)
    if min_margin < 0:
        raise ValueError("min_margin must be nonnegative.")
    margin = circuit_metric - random_metric
    return RandomCircuitBaselineReport(
        circuit_metric=circuit_metric,
        random_metric=random_metric,
        margin=margin,
        circuit_beats_random=margin >= min_margin,
    )


def ood_template_report(
    logits: t.Tensor,
    answer_ids: t.Tensor,
    template_ids: t.Tensor,
    *,
    min_accuracy: float = 0.75,
) -> OODTemplateReport:
    """Report circuit answer accuracy on held-out prompt templates."""

    if logits.shape[:-1] != answer_ids.shape:
        raise ValueError("answer_ids must match logits leading dimensions.")
    if answer_ids.shape != template_ids.shape:
        raise ValueError("answer_ids and template_ids must match.")
    if answer_ids.numel() == 0:
        raise ValueError("answer_ids must be nonempty.")
    if logits.shape[-1] == 0:
        raise ValueError("logits vocabulary dimension must be nonempty.")
    if not 0.0 <= min_accuracy <= 1.0:
        raise ValueError("min_accuracy must be between 0 and 1.")
    _require_finite_tensor(logits, "logits")
    if ((answer_ids < 0) | (answer_ids >= logits.shape[-1])).any():
        raise ValueError("answer_ids must be valid vocabulary indices.")

    per_template: dict[int, float] = {}
    for template_id in template_ids.unique(sorted=True):
        mask = template_ids.eq(template_id)
        accuracy = prediction_accuracy(logits[mask], answer_ids[mask])
        per_template[int(template_id.item())] = accuracy
    worst_accuracy = min(per_template.values()) if per_template else 0.0
    return OODTemplateReport(
        per_template_accuracy=per_template,
        worst_template_accuracy=worst_accuracy,
        passes_ood=worst_accuracy >= min_accuracy,
    )


def _top_edge_names(scores: t.Tensor, edge_names: list[str], *, top_k: int) -> tuple[str, ...]:
    flat_scores = scores.flatten().float()
    if flat_scores.numel() == 0:
        raise ValueError("scores must be nonempty.")
    if flat_scores.numel() != len(edge_names):
        raise ValueError("scores and edge_names must align.")
    _require_finite_tensor(flat_scores, "scores")
    k = min(top_k, flat_scores.numel())
    top_indices = flat_scores.topk(k=k).indices.tolist()
    return tuple(edge_names[int(index)] for index in top_indices)


def _pearson_correlation(left: t.Tensor, right: t.Tensor) -> float:
    left = left.flatten().float()
    right = right.flatten().float()
    if left.shape != right.shape:
        raise ValueError("score tensors must have matching shapes.")
    if left.numel() < 2:
        raise ValueError("at least two scores are required for correlation.")
    _require_finite_tensor(left, "left")
    _require_finite_tensor(right, "right")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((left_centered @ right_centered / denominator).item())


def circuit_method_comparison_report(
    exact_scores: t.Tensor,
    method_scores: Mapping[str, t.Tensor],
    edge_names: list[str],
    *,
    top_k: int,
    min_topk_overlap: float = 0.5,
    min_score_correlation: float = 0.5,
) -> CircuitMethodComparisonReport:
    """Compare approximate circuit-discovery methods against exact patching.

    This is deliberately method-agnostic: callers can pass attribution patching,
    EAP, EAP-IG, or ACDC scores as long as larger means more important.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0.0 <= min_topk_overlap <= 1.0:
        raise ValueError("min_topk_overlap must be between 0 and 1.")
    if not -1.0 <= min_score_correlation <= 1.0:
        raise ValueError("min_score_correlation must be between -1 and 1.")
    if not method_scores:
        raise ValueError("method_scores must contain at least one method.")
    exact_flat = exact_scores.flatten().float()
    if exact_flat.numel() == 0:
        raise ValueError("exact_scores must be nonempty.")
    if exact_flat.numel() != len(edge_names):
        raise ValueError("exact_scores and edge_names must align.")
    _require_finite_tensor(exact_flat, "exact_scores")

    exact_top_edges = _top_edge_names(exact_flat, edge_names, top_k=top_k)
    exact_top_set = set(exact_top_edges)
    method_top_edges: dict[str, tuple[str, ...]] = {}
    topk_overlap: dict[str, float] = {}
    score_correlations: dict[str, float] = {}
    circuit_sizes: dict[str, int] = {"exact": len(exact_top_edges)}

    for method_name, scores in method_scores.items():
        method_flat = scores.flatten().float()
        if method_flat.shape != exact_flat.shape:
            raise ValueError(f"{method_name} scores must match exact_scores shape.")
        _require_finite_tensor(method_flat, f"{method_name} scores")
        top_edges = _top_edge_names(method_flat, edge_names, top_k=top_k)
        overlap = len(exact_top_set.intersection(top_edges)) / len(exact_top_edges)
        method_top_edges[method_name] = top_edges
        topk_overlap[method_name] = overlap
        score_correlations[method_name] = _pearson_correlation(exact_flat, method_flat)
        circuit_sizes[method_name] = len(top_edges)

    best_matching_method = max(
        method_scores,
        key=lambda name: (topk_overlap[name], score_correlations[name]),
    )
    passes_comparison = all(
        topk_overlap[name] >= min_topk_overlap
        and score_correlations[name] >= min_score_correlation
        for name in method_scores
    )
    return CircuitMethodComparisonReport(
        exact_top_edges=exact_top_edges,
        method_top_edges=method_top_edges,
        topk_overlap=topk_overlap,
        score_correlations=score_correlations,
        circuit_sizes=circuit_sizes,
        best_matching_method=best_matching_method,
        passes_comparison=passes_comparison,
    )
