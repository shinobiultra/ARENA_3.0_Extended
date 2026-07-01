"""Circuit discovery metrics for automated circuit notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch as t

from arena_ext.activation_language import prediction_accuracy


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


def _require_finite_tensor(tensor: t.Tensor, name: str) -> None:
    if not t.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite.")


def _require_finite_scalar(value: float, name: str) -> None:
    if not t.isfinite(t.tensor(value)).item():
        raise ValueError(f"{name} must be finite.")


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
