"""Representation geometry utilities for direction and manifold notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t


CausalDirection = Literal["increase", "decrease"]


@dataclass(frozen=True)
class PCAProjection:
    projected: t.Tensor
    components: t.Tensor
    explained_variance_ratio: t.Tensor


@dataclass(frozen=True)
class GeometryLabelPredictionReport:
    heldout_accuracy: float
    predicts_heldout_labels: bool


@dataclass(frozen=True)
class WhiteNoiseControlReport:
    real_accuracy: float
    noise_accuracy: float
    margin: float
    survives_white_noise_control: bool


@dataclass(frozen=True)
class GeometryStabilityReport:
    neighbor_sets: tuple[tuple[int, ...], ...]
    mean_pairwise_jaccard: float
    stable_across_seeds: bool


@dataclass(frozen=True)
class DirectionCausalEffectReport:
    baseline_mean: float
    intervened_mean: float
    random_control_mean: float
    observed_delta: float
    random_delta: float
    has_causal_effect: bool


@dataclass(frozen=True)
class KNNLabelPredictionReport:
    k: int
    heldout_accuracy: float
    predicts_heldout_labels: bool


@dataclass(frozen=True)
class NeighborhoodPreservationReport:
    k: int
    mean_neighbor_overlap: float
    preserves_neighborhoods: bool


@dataclass(frozen=True)
class VisualizationSweepReport:
    seed_count: int
    setting_count: int
    run_count: int
    min_heldout_knn_accuracy: float
    mean_heldout_knn_accuracy: float
    min_trustworthiness: float
    mean_trustworthiness: float
    min_neighborhood_preservation: float
    mean_neighborhood_preservation: float
    random_label_accuracy_max: float
    random_token_accuracy_max: float
    passes_visualization_controls: bool


def pca_svd_projection(
    activations: t.Tensor,
    *,
    n_components: int = 2,
) -> PCAProjection:
    """Project activations with centered PCA using SVD."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if n_components > min(activations.shape):
        raise ValueError("n_components cannot exceed min(examples, d_model).")

    activations_float = activations.float()
    centered = activations_float - activations_float.mean(dim=0, keepdim=True)
    _, singular_values, vh = t.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    projected = centered @ components.T
    variance = singular_values.pow(2)
    total_variance = variance.sum()
    if total_variance.item() == 0:
        explained = t.zeros(
            n_components,
            dtype=activations_float.dtype,
            device=activations.device,
        )
    else:
        explained = variance[:n_components] / total_variance
    return PCAProjection(
        projected=projected,
        components=components,
        explained_variance_ratio=explained,
    )


def nearest_centroid_predictions(
    train_points: t.Tensor,
    train_labels: t.Tensor,
    heldout_points: t.Tensor,
) -> t.Tensor:
    """Classify held-out points by nearest class centroid."""

    if train_points.ndim != 2 or heldout_points.ndim != 2:
        raise ValueError("points must have shape (examples, dimensions).")
    if train_points.shape[-1] != heldout_points.shape[-1]:
        raise ValueError("train and heldout point dimensions must match.")
    labels = train_labels.flatten().long()
    if labels.numel() != train_points.shape[0]:
        raise ValueError("train_labels must have one label per train point.")

    unique_labels = labels.unique(sorted=True)
    centroids = []
    for label in unique_labels:
        centroids.append(train_points[labels.eq(label)].float().mean(dim=0))
    centroid_tensor = t.stack(centroids)
    distances = t.cdist(heldout_points.float(), centroid_tensor)
    predicted_indices = distances.argmin(dim=-1)
    return unique_labels[predicted_indices]


def knn_label_predictions(
    train_points: t.Tensor,
    train_labels: t.Tensor,
    heldout_points: t.Tensor,
    *,
    k: int = 1,
) -> t.Tensor:
    """Classify held-out points by k-nearest train labels."""

    if k <= 0:
        raise ValueError("k must be positive.")
    if train_points.ndim != 2 or heldout_points.ndim != 2:
        raise ValueError("points must have shape (examples, dimensions).")
    if train_points.shape[-1] != heldout_points.shape[-1]:
        raise ValueError("train and heldout point dimensions must match.")
    labels = train_labels.flatten().long()
    if labels.numel() != train_points.shape[0]:
        raise ValueError("train_labels must have one label per train point.")
    if k > train_points.shape[0]:
        raise ValueError("k cannot exceed the number of train points.")

    distances = t.cdist(heldout_points.float(), train_points.float())
    nearest = distances.topk(k, largest=False).indices
    nearest_labels = labels[nearest]
    unique_labels = labels.unique(sorted=True)
    predictions = []
    for row in nearest_labels:
        votes = t.stack([(row == label).sum() for label in unique_labels])
        predictions.append(unique_labels[votes.argmax()])
    return t.stack(predictions)


def heldout_knn_accuracy_report(
    train_points: t.Tensor,
    train_labels: t.Tensor,
    heldout_points: t.Tensor,
    heldout_labels: t.Tensor,
    *,
    k: int = 1,
    min_accuracy: float = 0.8,
) -> KNNLabelPredictionReport:
    """Check whether a low-dimensional geometry predicts held-out labels by kNN."""

    predictions = knn_label_predictions(train_points, train_labels, heldout_points, k=k)
    labels = heldout_labels.flatten().long()
    if predictions.shape != labels.shape:
        raise ValueError("heldout_labels must have one label per heldout point.")
    accuracy = predictions.eq(labels).float().mean().item()
    return KNNLabelPredictionReport(
        k=k,
        heldout_accuracy=accuracy,
        predicts_heldout_labels=accuracy >= min_accuracy,
    )


def neighborhood_preservation_report(
    high_dim_points: t.Tensor,
    low_dim_points: t.Tensor,
    *,
    k: int = 3,
    min_overlap: float = 0.5,
) -> NeighborhoodPreservationReport:
    """Compare high-dimensional and low-dimensional nearest-neighbor sets."""

    if k <= 0:
        raise ValueError("k must be positive.")
    if high_dim_points.ndim != 2 or low_dim_points.ndim != 2:
        raise ValueError("points must have shape (examples, dimensions).")
    if high_dim_points.shape[0] != low_dim_points.shape[0]:
        raise ValueError("high_dim_points and low_dim_points must have the same examples.")
    if k >= high_dim_points.shape[0]:
        raise ValueError("k must be smaller than the number of examples.")

    high_distances = t.cdist(high_dim_points.float(), high_dim_points.float())
    low_distances = t.cdist(low_dim_points.float(), low_dim_points.float())
    high_neighbors = high_distances.topk(k + 1, largest=False).indices[:, 1:]
    low_neighbors = low_distances.topk(k + 1, largest=False).indices[:, 1:]
    overlaps = []
    for high_row, low_row in zip(high_neighbors, low_neighbors, strict=True):
        high_set = set(int(index) for index in high_row.tolist())
        low_set = set(int(index) for index in low_row.tolist())
        overlaps.append(len(high_set & low_set) / k)
    mean_overlap = sum(overlaps) / len(overlaps)
    return NeighborhoodPreservationReport(
        k=k,
        mean_neighbor_overlap=mean_overlap,
        preserves_neighborhoods=mean_overlap >= min_overlap,
    )


def visualization_sweep_report(
    *,
    seed_count: int,
    setting_count: int,
    heldout_knn_accuracies: list[float],
    trustworthiness_scores: list[float],
    neighborhood_preservation_scores: list[float],
    random_label_accuracies: list[float],
    random_token_accuracies: list[float],
    min_seed_count: int = 5,
    min_setting_count: int = 3,
    min_heldout_accuracy: float = 0.8,
    min_trustworthiness: float = 0.8,
    min_neighborhood_preservation: float = 0.5,
    max_random_label_accuracy: float = 0.3,
    max_random_token_accuracy: float = 0.3,
) -> VisualizationSweepReport:
    """Aggregate a visualization reducer sweep into an acceptance report."""

    run_count = len(heldout_knn_accuracies)
    if run_count == 0:
        raise ValueError("visualization sweeps must include at least one run.")
    metric_lists = (
        trustworthiness_scores,
        neighborhood_preservation_scores,
        random_label_accuracies,
        random_token_accuracies,
    )
    if not all(len(values) == run_count for values in metric_lists):
        raise ValueError("all visualization metric lists must have the same length.")

    min_accuracy = min(heldout_knn_accuracies)
    mean_accuracy = sum(heldout_knn_accuracies) / run_count
    min_trust = min(trustworthiness_scores)
    mean_trust = sum(trustworthiness_scores) / run_count
    min_preservation = min(neighborhood_preservation_scores)
    mean_preservation = sum(neighborhood_preservation_scores) / run_count
    random_label_max = max(random_label_accuracies)
    random_token_max = max(random_token_accuracies)
    passes = (
        seed_count >= min_seed_count
        and setting_count >= min_setting_count
        and min_accuracy >= min_heldout_accuracy
        and min_trust >= min_trustworthiness
        and min_preservation >= min_neighborhood_preservation
        and random_label_max <= max_random_label_accuracy
        and random_token_max <= max_random_token_accuracy
    )
    return VisualizationSweepReport(
        seed_count=seed_count,
        setting_count=setting_count,
        run_count=run_count,
        min_heldout_knn_accuracy=min_accuracy,
        mean_heldout_knn_accuracy=mean_accuracy,
        min_trustworthiness=min_trust,
        mean_trustworthiness=mean_trust,
        min_neighborhood_preservation=min_preservation,
        mean_neighborhood_preservation=mean_preservation,
        random_label_accuracy_max=random_label_max,
        random_token_accuracy_max=random_token_max,
        passes_visualization_controls=passes,
    )


def template_center_activations(
    activations_by_template: t.Tensor,
) -> t.Tensor:
    """Remove each prompt template's mean direction from its activation batch."""

    if activations_by_template.ndim != 3:
        raise ValueError(
            "activations_by_template must have shape (templates, examples, d_model)."
        )
    return activations_by_template.float() - activations_by_template.float().mean(
        dim=1,
        keepdim=True,
    )


def geometry_label_prediction_report(
    train_points: t.Tensor,
    train_labels: t.Tensor,
    heldout_points: t.Tensor,
    heldout_labels: t.Tensor,
    *,
    min_accuracy: float = 0.8,
) -> GeometryLabelPredictionReport:
    """Check whether geometry predicts held-out labels."""

    predictions = nearest_centroid_predictions(train_points, train_labels, heldout_points)
    labels = heldout_labels.flatten().long()
    if predictions.shape != labels.shape:
        raise ValueError("heldout_labels must have one label per heldout point.")
    accuracy = predictions.eq(labels).float().mean().item()
    return GeometryLabelPredictionReport(
        heldout_accuracy=accuracy,
        predicts_heldout_labels=accuracy >= min_accuracy,
    )


def white_noise_control_report(
    *,
    real_accuracy: float,
    noise_accuracy: float,
    min_margin: float = 0.2,
) -> WhiteNoiseControlReport:
    """Check that real geometry beats a white-noise geometry control."""

    margin = real_accuracy - noise_accuracy
    return WhiteNoiseControlReport(
        real_accuracy=real_accuracy,
        noise_accuracy=noise_accuracy,
        margin=margin,
        survives_white_noise_control=margin >= min_margin,
    )


def geometry_stability_report(
    neighbor_sets: list[list[int]],
    *,
    min_jaccard: float = 0.5,
) -> GeometryStabilityReport:
    """Check whether nearest-neighbor sets are stable across seeds."""

    if not neighbor_sets:
        raise ValueError("neighbor_sets must be nonempty.")
    normalized = tuple(tuple(int(index) for index in neighbors) for neighbors in neighbor_sets)
    overlaps = []
    for i, left in enumerate(normalized):
        for right in normalized[i + 1 :]:
            left_set = set(left)
            right_set = set(right)
            if not left_set and not right_set:
                overlaps.append(1.0)
            else:
                overlaps.append(len(left_set & right_set) / max(len(left_set), len(right_set)))
    mean_jaccard = sum(overlaps) / len(overlaps) if overlaps else 1.0
    return GeometryStabilityReport(
        neighbor_sets=normalized,
        mean_pairwise_jaccard=mean_jaccard,
        stable_across_seeds=mean_jaccard >= min_jaccard,
    )


def direction_causal_effect_report(
    baseline_scores: t.Tensor,
    intervened_scores: t.Tensor,
    random_control_scores: t.Tensor,
    *,
    expected_direction: CausalDirection = "increase",
    min_effect: float = 0.2,
    min_random_margin: float = 0.1,
) -> DirectionCausalEffectReport:
    """Check that a representation direction has causal effect over random control."""

    if baseline_scores.shape != intervened_scores.shape:
        raise ValueError("baseline and intervened scores must match.")
    if baseline_scores.shape != random_control_scores.shape:
        raise ValueError("baseline and random control scores must match.")
    baseline_mean = baseline_scores.float().mean().item()
    intervened_mean = intervened_scores.float().mean().item()
    random_control_mean = random_control_scores.float().mean().item()
    observed_delta = intervened_mean - baseline_mean
    random_delta = random_control_mean - baseline_mean
    if expected_direction == "increase":
        directional_effect = observed_delta >= min_effect
    elif expected_direction == "decrease":
        directional_effect = -observed_delta >= min_effect
    else:
        raise ValueError("expected_direction must be 'increase' or 'decrease'.")
    has_causal_effect = directional_effect and abs(observed_delta) > (
        abs(random_delta) + min_random_margin
    )
    return DirectionCausalEffectReport(
        baseline_mean=baseline_mean,
        intervened_mean=intervened_mean,
        random_control_mean=random_control_mean,
        observed_delta=observed_delta,
        random_delta=random_delta,
        has_causal_effect=has_causal_effect,
    )
