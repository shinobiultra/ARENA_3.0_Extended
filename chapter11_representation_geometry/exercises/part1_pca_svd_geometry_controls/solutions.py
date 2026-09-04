# %%
"""Reference solutions for [11.1] PCA, SVD, and Geometry Controls."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t

chapter = "chapter11_representation_geometry"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

PYTHIA_WEEKDAY_MODEL_ID = "EleutherAI/pythia-70m-deduped"
PYTHIA_WEEKDAY_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
WEEKDAY_TRAIN_TEMPLATE = "Today is {}"
WEEKDAY_HELDOUT_TEMPLATE = "The calendar says {}"
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTH_TRAIN_TEMPLATE = "The report was written in {}"
MONTH_HELDOUT_TEMPLATE = "The event happened during {}"
WEEKDAY_VIS_TRAIN_TEMPLATES = (
    "Today is {}",
    "Yesterday was {}",
    "Tomorrow will be {}",
    "The meeting is on {}",
    "The schedule says {}",
)
WEEKDAY_VIS_HELDOUT_TEMPLATES = (
    "The calendar says {}",
    "The reminder mentions {}",
    "I wrote down {}",
)
MONTH_VIS_TRAIN_TEMPLATES = (
    "The report was written in {}",
    "The invoice arrived in {}",
    "The launch happened in {}",
    "The archive is from {}",
    "The fiscal period is {}",
)
MONTH_VIS_HELDOUT_TEMPLATES = (
    "The event happened during {}",
    "The memo is dated {}",
    "The deadline falls in {}",
)
WEEKDAY_RANDOM_TOKEN_CONTROL_LABELS = (
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "silver",
)
MONTH_RANDOM_TOKEN_CONTROL_LABELS = (
    "circle",
    "square",
    "triangle",
    "hexagon",
    "spiral",
    "stripe",
    "dot",
    "line",
    "cube",
    "cone",
    "prism",
    "wave",
)
VISUALIZATION_SWEEP_SEEDS = (0, 1, 2, 3, 4)
VISUALIZATION_SWEEP_SETTINGS = (
    {"n_neighbors": 3, "min_dist": 0.0},
    {"n_neighbors": 5, "min_dist": 0.1},
    {"n_neighbors": 8, "min_dist": 0.3},
)


# %%
CausalDirection = Literal["increase", "decrease"]


@dataclass(frozen=True)
class PCAProjection:
    projected: t.Tensor
    components: t.Tensor
    explained_variance_ratio: t.Tensor


@dataclass(frozen=True)
class TrainHeldoutPCAProjection:
    train_projected: t.Tensor
    heldout_projected: t.Tensor
    train_mean: t.Tensor
    components: t.Tensor
    explained_variance_ratio: t.Tensor


@dataclass(frozen=True)
class RidgeCoordinateProbe:
    weights: t.Tensor
    heldout_predictions: t.Tensor


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


def pca_svd_train_heldout_projection(
    train_activations: t.Tensor,
    heldout_activations: t.Tensor,
    *,
    n_components: int = 2,
) -> TrainHeldoutPCAProjection:
    """Fit PCA on training activations, then transform held-out activations."""

    if train_activations.ndim != 2 or heldout_activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    if train_activations.shape[1] != heldout_activations.shape[1]:
        raise ValueError("train and heldout activation dimensions must match.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if n_components > min(train_activations.shape):
        raise ValueError("n_components cannot exceed min(train_examples, d_model).")

    train_float = train_activations.float()
    heldout_float = heldout_activations.float()
    train_mean = train_float.mean(dim=0, keepdim=True)
    train_centered = train_float - train_mean
    _, singular_values, vh = t.linalg.svd(train_centered, full_matrices=False)
    components = vh[:n_components]
    variance = singular_values.pow(2)
    total_variance = variance.sum()
    if total_variance.item() == 0:
        explained = t.zeros(
            n_components,
            dtype=train_float.dtype,
            device=train_float.device,
        )
    else:
        explained = variance[:n_components] / total_variance
    return TrainHeldoutPCAProjection(
        train_projected=train_centered @ components.T,
        heldout_projected=(heldout_float - train_mean) @ components.T,
        train_mean=train_mean,
        components=components,
        explained_variance_ratio=explained,
    )


def ridge_coordinate_probe(
    train_activations: t.Tensor,
    train_coordinates: t.Tensor,
    heldout_activations: t.Tensor,
    *,
    l2: float = 1e-3,
) -> RidgeCoordinateProbe:
    """Fit a bias-aware ridge probe on train data and predict held-out coordinates."""

    if train_activations.ndim != 2 or heldout_activations.ndim != 2:
        raise ValueError("activations must have shape (examples, features).")
    if train_coordinates.ndim != 2:
        raise ValueError("train_coordinates must have shape (examples, targets).")
    if train_activations.shape[0] != train_coordinates.shape[0]:
        raise ValueError("train activations and coordinates must have equal examples.")
    if train_activations.shape[1] != heldout_activations.shape[1]:
        raise ValueError("train and heldout activation dimensions must match.")
    if l2 < 0:
        raise ValueError("l2 must be non-negative.")

    train_float = train_activations.float()
    heldout_float = heldout_activations.float()
    targets_float = train_coordinates.float()
    train_design = t.cat(
        [train_float, t.ones(train_float.shape[0], 1, device=train_float.device)],
        dim=1,
    )
    heldout_design = t.cat(
        [heldout_float, t.ones(heldout_float.shape[0], 1, device=heldout_float.device)],
        dim=1,
    )
    regularizer = t.eye(
        train_design.shape[1],
        dtype=train_design.dtype,
        device=train_design.device,
    )
    regularizer[-1, -1] = 0.0
    weights = t.linalg.solve(
        train_design.T @ train_design + l2 * regularizer,
        train_design.T @ targets_float,
    )
    return RidgeCoordinateProbe(
        weights=weights,
        heldout_predictions=heldout_design @ weights,
    )


def make_toy_calendar_ring(
    *,
    label_count: int = 7,
    train_template_count: int = 4,
    heldout_template_count: int = 2,
    d_model: int = 32,
    template_strength: float = 5.0,
    noise_scale: float = 0.03,
    seed: int = 0,
) -> dict:
    """Construct a known cyclic representation with strong template nuisance offsets."""

    if label_count < 3:
        raise ValueError("label_count must be at least three.")
    if train_template_count <= 0 or heldout_template_count <= 0:
        raise ValueError("template counts must be positive.")
    if d_model < 2:
        raise ValueError("d_model must be at least two.")
    if template_strength < 0 or noise_scale < 0:
        raise ValueError("strength and noise scales must be non-negative.")

    generator = t.Generator().manual_seed(seed)
    angles = 2 * t.pi * t.arange(label_count, dtype=t.float32) / label_count
    ring_coordinates = t.stack([angles.cos(), angles.sin()], dim=-1)
    basis, _ = t.linalg.qr(t.randn(d_model, 2, generator=generator), mode="reduced")
    signal = ring_coordinates @ basis.T

    def make_grid(template_count: int) -> t.Tensor:
        offsets = template_strength * t.randn(
            template_count, 1, d_model, generator=generator
        )
        noise = noise_scale * t.randn(
            template_count, label_count, d_model, generator=generator
        )
        return signal.unsqueeze(0) + offsets + noise

    train_raw_grid = make_grid(train_template_count)
    heldout_raw_grid = make_grid(heldout_template_count)
    label_ids = t.arange(label_count, dtype=t.long)
    return {
        "ring_coordinates": ring_coordinates,
        "train_raw_grid": train_raw_grid,
        "heldout_raw_grid": heldout_raw_grid,
        "train_raw": train_raw_grid.flatten(0, 1),
        "heldout_raw": heldout_raw_grid.flatten(0, 1),
        "train_centered": template_center_activations(train_raw_grid).flatten(0, 1),
        "heldout_centered": template_center_activations(heldout_raw_grid).flatten(0, 1),
        "train_labels": label_ids.repeat(train_template_count),
        "heldout_labels": label_ids.repeat(heldout_template_count),
        "train_template_ids": t.arange(train_template_count).repeat_interleave(label_count),
        "heldout_template_ids": t.arange(heldout_template_count).repeat_interleave(label_count),
        "train_targets": ring_coordinates.repeat(train_template_count, 1),
        "heldout_targets": ring_coordinates.repeat(heldout_template_count, 1),
    }


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
    activations_float = activations_by_template.float()
    return activations_float - activations_float.mean(dim=1, keepdim=True)


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
                overlaps.append(len(left_set & right_set) / len(left_set | right_set))
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


# %%
def pca_smoke_test() -> dict:
    activations = t.tensor(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
        ]
    )
    projection = pca_svd_projection(activations, n_components=1)
    return {
        "projected_shape": list(projection.projected.shape),
        "components_shape": list(projection.components.shape),
        "explained_variance_ratio": [
            round(value, 6)
            for value in projection.explained_variance_ratio.tolist()
        ],
    }


def prediction_smoke_test() -> dict:
    train_points = t.tensor([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.2, 0.0]])
    train_labels = t.tensor([0, 0, 1, 1])
    heldout_points = t.tensor([[0.1, 0.0], [2.1, 0.0]])
    heldout_labels = t.tensor([0, 1])
    return geometry_label_prediction_report(
        train_points,
        train_labels,
        heldout_points,
        heldout_labels,
        min_accuracy=1.0,
    ).__dict__


def noise_control_smoke_test() -> dict:
    return white_noise_control_report(
        real_accuracy=0.9,
        noise_accuracy=0.5,
        min_margin=0.2,
    ).__dict__


def stability_smoke_test() -> dict:
    return geometry_stability_report(
        [[1, 2, 3], [1, 2, 4], [1, 2, 3]],
        min_jaccard=0.5,
    ).__dict__


def causal_direction_smoke_test() -> dict:
    baseline = t.tensor([0.2, 0.3])
    intervened = t.tensor([0.8, 0.7])
    random_control = t.tensor([0.35, 0.25])
    return direction_causal_effect_report(
        baseline,
        intervened,
        random_control,
        expected_direction="increase",
        min_effect=0.4,
        min_random_margin=0.2,
    ).__dict__


def template_centering_smoke_test() -> dict:
    activations = t.tensor(
        [
            [[10.0, 1.0], [12.0, 3.0]],
            [[-5.0, 2.0], [-1.0, 4.0]],
        ]
    )
    centered = template_center_activations(activations)
    return {
        "shape": list(centered.shape),
        "max_template_mean_abs": centered.mean(dim=1).abs().max().item(),
        "first_template_centered": centered[0].tolist(),
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "pca": pca_smoke_test(),
        "prediction": prediction_smoke_test(),
        "noise_control": noise_control_smoke_test(),
        "stability": stability_smoke_test(),
        "causal_direction": causal_direction_smoke_test(),
        "template_centering": template_centering_smoke_test(),
    }


def _final_token_hidden_states(
    *,
    model,
    tokenizer,
    prompts: tuple[str, ...],
    device: t.device,
) -> t.Tensor:
    """Return last-layer hidden states at each prompt's final non-padding token."""

    encoded = tokenizer(list(prompts), padding=True, return_tensors="pt").to(device)
    with t.inference_mode():
        output = model(**encoded, output_hidden_states=True)
    final_token_indices = encoded.attention_mask.sum(dim=1) - 1
    batch_indices = t.arange(len(prompts), device=device)
    return output.hidden_states[-1][batch_indices, final_token_indices].float().detach().cpu()


def _template_activation_grid(
    *,
    model,
    tokenizer,
    label_names: tuple[str, ...],
    templates: tuple[str, ...],
    device: t.device,
) -> t.Tensor:
    """Return hidden states with shape (templates, labels, d_model)."""

    return t.stack(
        [
            _final_token_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=tuple(template.format(label) for label in label_names),
                device=device,
            )
            for template in templates
        ]
    )


def _template_centered_activation_table(
    *,
    model,
    tokenizer,
    label_names: tuple[str, ...],
    templates: tuple[str, ...],
    device: t.device,
) -> tuple[t.Tensor, t.Tensor]:
    """Extract and template-center hidden states for several prompt templates."""

    grid = _template_activation_grid(
        model=model,
        tokenizer=tokenizer,
        label_names=label_names,
        templates=templates,
        device=device,
    )
    labels = t.arange(len(label_names), dtype=t.long).repeat(len(templates))
    return template_center_activations(grid).flatten(0, 1), labels


def _extract_calendar_task_activations(
    *,
    model,
    tokenizer,
    label_names: tuple[str, ...],
    train_templates: tuple[str, ...],
    heldout_templates: tuple[str, ...],
    random_token_labels: tuple[str, ...],
    device: t.device,
) -> dict:
    train_raw_grid = _template_activation_grid(
        model=model,
        tokenizer=tokenizer,
        label_names=label_names,
        templates=train_templates,
        device=device,
    )
    heldout_raw_grid = _template_activation_grid(
        model=model,
        tokenizer=tokenizer,
        label_names=label_names,
        templates=heldout_templates,
        device=device,
    )
    random_token_raw_grid = _template_activation_grid(
        model=model,
        tokenizer=tokenizer,
        label_names=random_token_labels,
        templates=train_templates,
        device=device,
    )
    label_ids = t.arange(len(label_names), dtype=t.long)
    random_token_ids = t.arange(len(random_token_labels), dtype=t.long)
    return {
        "label_names": label_names,
        "train_templates": train_templates,
        "heldout_templates": heldout_templates,
        "random_token_names": random_token_labels,
        "train_raw_grid": train_raw_grid,
        "heldout_raw_grid": heldout_raw_grid,
        "random_token_raw_grid": random_token_raw_grid,
        "train_raw": train_raw_grid.flatten(0, 1),
        "heldout_raw": heldout_raw_grid.flatten(0, 1),
        "random_token_raw": random_token_raw_grid.flatten(0, 1),
        "train_centered": template_center_activations(train_raw_grid).flatten(0, 1),
        "heldout_centered": template_center_activations(heldout_raw_grid).flatten(0, 1),
        "random_token_centered": template_center_activations(
            random_token_raw_grid
        ).flatten(0, 1),
        "train_labels": label_ids.repeat(len(train_templates)),
        "heldout_labels": label_ids.repeat(len(heldout_templates)),
        "random_token_labels": random_token_ids.repeat(len(train_templates)),
        "train_template_ids": t.arange(len(train_templates)).repeat_interleave(
            len(label_names)
        ),
        "heldout_template_ids": t.arange(len(heldout_templates)).repeat_interleave(
            len(label_names)
        ),
        "train_prompts": tuple(
            template.format(label)
            for template in train_templates
            for label in label_names
        ),
        "heldout_prompts": tuple(
            template.format(label)
            for template in heldout_templates
            for label in label_names
        ),
    }


def load_pythia_calendar_activations(
    *,
    max_vram_gb: float = 24.0,
    model_id: str = PYTHIA_WEEKDAY_MODEL_ID,
    revision: str = PYTHIA_WEEKDAY_REVISION,
) -> dict:
    """Load one pinned Pythia checkpoint and extract real calendar activations."""

    if not t.cuda.is_available():
        raise RuntimeError("11.1 Pythia calendar activations require CUDA.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=t.float32,
    ).to(device)
    model.eval()
    tasks = {
        "weekday": _extract_calendar_task_activations(
            model=model,
            tokenizer=tokenizer,
            label_names=WEEKDAYS,
            train_templates=WEEKDAY_VIS_TRAIN_TEMPLATES,
            heldout_templates=WEEKDAY_VIS_HELDOUT_TEMPLATES,
            random_token_labels=WEEKDAY_RANDOM_TOKEN_CONTROL_LABELS,
            device=device,
        ),
        "month": _extract_calendar_task_activations(
            model=model,
            tokenizer=tokenizer,
            label_names=MONTHS,
            train_templates=MONTH_VIS_TRAIN_TEMPLATES,
            heldout_templates=MONTH_VIS_HELDOUT_TEMPLATES,
            random_token_labels=MONTH_RANDOM_TOKEN_CONTROL_LABELS,
            device=device,
        ),
    }
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    del model
    t.cuda.empty_cache()
    return {
        "model_id": model_id,
        "revision": revision,
        "cuda_version": t.version.cuda,
        "device": t.cuda.get_device_name(0),
        "generation_used": False,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "tasks": tasks,
    }


def _umap_visualization_sweep(
    *,
    train_activations: t.Tensor,
    train_labels: t.Tensor,
    heldout_activations: t.Tensor,
    heldout_labels: t.Tensor,
    random_token_activations: t.Tensor,
    random_token_labels: t.Tensor,
    task_name: str,
    include_selected_projection: bool = False,
) -> dict:
    """Run a seeded UMAP sweep with kNN, random-label, and random-token controls."""

    from sklearn.manifold import trustworthiness
    import umap

    heldout_accuracies = []
    trustworthiness_scores = []
    neighborhood_scores = []
    random_label_accuracies = []
    random_token_accuracies = []
    run_summaries = []
    selected_projection = None
    combined_high_dim = t.cat([train_activations, heldout_activations], dim=0)
    for seed in VISUALIZATION_SWEEP_SEEDS:
        for setting in VISUALIZATION_SWEEP_SETTINGS:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=setting["n_neighbors"],
                min_dist=setting["min_dist"],
                metric="cosine",
                random_state=seed,
                transform_seed=seed,
                n_jobs=1,
            )
            train_low = t.tensor(
                reducer.fit_transform(train_activations.numpy()),
                dtype=t.float32,
            )
            heldout_low = t.tensor(
                reducer.transform(heldout_activations.numpy()),
                dtype=t.float32,
            )
            combined_low = t.cat([train_low, heldout_low], dim=0)
            k = 3
            knn = heldout_knn_accuracy_report(
                train_low,
                train_labels,
                heldout_low,
                heldout_labels,
                k=k,
                min_accuracy=0.8,
            )
            trust = float(
                trustworthiness(
                    combined_high_dim.numpy(),
                    combined_low.numpy(),
                    n_neighbors=5,
                    metric="cosine",
                )
            )
            preservation = neighborhood_preservation_report(
                combined_high_dim,
                combined_low,
                k=5,
                min_overlap=0.55,
            )
            shuffled_labels = train_labels[
                t.randperm(train_labels.numel(), generator=t.Generator().manual_seed(seed))
            ]
            random_label = heldout_knn_accuracy_report(
                train_low,
                shuffled_labels,
                heldout_low,
                heldout_labels,
                k=k,
                min_accuracy=0.0,
            )
            random_token_reducer = umap.UMAP(
                n_components=2,
                n_neighbors=setting["n_neighbors"],
                min_dist=setting["min_dist"],
                metric="cosine",
                random_state=seed,
                transform_seed=seed,
                n_jobs=1,
            )
            random_token_train_low = t.tensor(
                random_token_reducer.fit_transform(random_token_activations.numpy()),
                dtype=t.float32,
            )
            heldout_from_random_token_low = t.tensor(
                random_token_reducer.transform(heldout_activations.numpy()),
                dtype=t.float32,
            )
            random_token = heldout_knn_accuracy_report(
                random_token_train_low,
                random_token_labels,
                heldout_from_random_token_low,
                heldout_labels,
                k=k,
                min_accuracy=0.0,
            )
            if include_selected_projection and seed == 2 and setting == {
                "n_neighbors": 5,
                "min_dist": 0.1,
            }:
                selected_projection = {
                    "seed": seed,
                    "n_neighbors": setting["n_neighbors"],
                    "min_dist": setting["min_dist"],
                    "train_coordinates": train_low.tolist(),
                    "heldout_coordinates": heldout_low.tolist(),
                    "shuffled_train_labels": shuffled_labels.tolist(),
                    "random_token_train_coordinates": random_token_train_low.tolist(),
                    "heldout_from_random_token_coordinates": (
                        heldout_from_random_token_low.tolist()
                    ),
                }
            heldout_accuracies.append(knn.heldout_accuracy)
            trustworthiness_scores.append(trust)
            neighborhood_scores.append(preservation.mean_neighbor_overlap)
            random_label_accuracies.append(random_label.heldout_accuracy)
            random_token_accuracies.append(random_token.heldout_accuracy)
            run_summaries.append(
                {
                    "seed": seed,
                    "n_neighbors": setting["n_neighbors"],
                    "min_dist": setting["min_dist"],
                    "heldout_knn_accuracy": knn.heldout_accuracy,
                    "trustworthiness": trust,
                    "neighborhood_preservation": preservation.mean_neighbor_overlap,
                    "random_label_accuracy": random_label.heldout_accuracy,
                    "random_token_accuracy": random_token.heldout_accuracy,
                }
            )
    summary = visualization_sweep_report(
        seed_count=len(VISUALIZATION_SWEEP_SEEDS),
        setting_count=len(VISUALIZATION_SWEEP_SETTINGS),
        heldout_knn_accuracies=heldout_accuracies,
        trustworthiness_scores=trustworthiness_scores,
        neighborhood_preservation_scores=neighborhood_scores,
        random_label_accuracies=random_label_accuracies,
        random_token_accuracies=random_token_accuracies,
        min_heldout_accuracy=0.8,
        min_trustworthiness=0.9,
        min_neighborhood_preservation=0.55,
        max_random_label_accuracy=0.35,
        max_random_token_accuracy=0.35,
    )
    result = {
        "task_name": task_name,
        "train_example_count": int(train_activations.shape[0]),
        "heldout_example_count": int(heldout_activations.shape[0]),
        "random_token_train_example_count": int(random_token_activations.shape[0]),
        "sweep": summary.__dict__,
        "runs": run_summaries,
        "passes_visualization_controls": summary.passes_visualization_controls,
    }
    if include_selected_projection:
        if selected_projection is None:
            raise RuntimeError("the configured signature UMAP projection was not produced")
        result["selected_projection"] = selected_projection
    return result


def _template_centered_identity_geometry_from_task(
    task: dict,
    *,
    noise_seed: int,
    min_centered_accuracy: float,
) -> dict:
    """Evaluate the first train/held-out prompt pair from an extracted task."""

    labels = t.arange(len(task["label_names"]), dtype=t.long)
    train_activations = task["train_raw_grid"][0]
    heldout_activations = task["heldout_raw_grid"][0]
    raw_report = geometry_label_prediction_report(
        train_activations,
        labels,
        heldout_activations,
        labels,
        min_accuracy=min_centered_accuracy,
    )
    centered_train = train_activations - train_activations.mean(dim=0, keepdim=True)
    centered_heldout = heldout_activations - heldout_activations.mean(dim=0, keepdim=True)
    centered_report = geometry_label_prediction_report(
        centered_train,
        labels,
        centered_heldout,
        labels,
        min_accuracy=min_centered_accuracy,
    )
    permuted_report = geometry_label_prediction_report(
        centered_train,
        (labels + 1) % len(task["label_names"]),
        centered_heldout,
        labels,
        min_accuracy=0.0,
    )
    generator = t.Generator().manual_seed(noise_seed)
    noise_train = t.randn(centered_train.shape, generator=generator)
    noise_heldout = t.randn(centered_heldout.shape, generator=generator)
    noise_report = geometry_label_prediction_report(
        noise_train,
        labels,
        noise_heldout,
        labels,
        min_accuracy=0.0,
    )
    noise_control = white_noise_control_report(
        real_accuracy=centered_report.heldout_accuracy,
        noise_accuracy=noise_report.heldout_accuracy,
        min_margin=0.4,
    )
    projection = pca_svd_projection(
        t.cat([centered_train, centered_heldout], dim=0),
        n_components=2,
    )
    similarity = t.nn.functional.normalize(centered_train, dim=-1) @ t.nn.functional.normalize(
        centered_heldout,
        dim=-1,
    ).T
    matched_pair_accuracy = similarity.argmax(dim=-1).eq(labels).float().mean().item()
    return {
        "label_count": len(task["label_names"]),
        "train_template": task["train_templates"][0],
        "heldout_template": task["heldout_templates"][0],
        "raw_heldout_accuracy": raw_report.heldout_accuracy,
        "raw_predicts_heldout_labels": raw_report.predicts_heldout_labels,
        "centered_heldout_accuracy": centered_report.heldout_accuracy,
        "centered_predicts_heldout_labels": centered_report.predicts_heldout_labels,
        "permuted_label_accuracy": permuted_report.heldout_accuracy,
        "noise_accuracy": noise_report.heldout_accuracy,
        "white_noise_margin": noise_control.margin,
        "survives_white_noise_control": noise_control.survives_white_noise_control,
        "matched_pair_accuracy": matched_pair_accuracy,
        "pca_explained_variance_ratio": projection.explained_variance_ratio.tolist(),
    }


def _calendar_signature_task_result(
    task_name: str,
    task: dict,
    visualization: dict,
) -> dict:
    """Build notebook-ready coordinates and metrics from one calendar task."""

    raw_pca = pca_svd_train_heldout_projection(
        task["train_raw"],
        task["heldout_raw"],
        n_components=2,
    )
    centered_pca = pca_svd_train_heldout_projection(
        task["train_centered"],
        task["heldout_centered"],
        n_components=2,
    )
    raw_knn = heldout_knn_accuracy_report(
        raw_pca.train_projected,
        task["train_labels"],
        raw_pca.heldout_projected,
        task["heldout_labels"],
        k=3,
        min_accuracy=0.0,
    )
    centered_knn = heldout_knn_accuracy_report(
        centered_pca.train_projected,
        task["train_labels"],
        centered_pca.heldout_projected,
        task["heldout_labels"],
        k=3,
        min_accuracy=0.0,
    )
    train_centroids = task["train_centered"].reshape(
        len(task["train_templates"]), len(task["label_names"]), -1
    ).mean(dim=0)
    heldout_centroids = task["heldout_centered"].reshape(
        len(task["heldout_templates"]), len(task["label_names"]), -1
    ).mean(dim=0)
    cosine_similarity = t.nn.functional.normalize(train_centroids, dim=-1) @ (
        t.nn.functional.normalize(heldout_centroids, dim=-1).T
    )
    matched_pair_accuracy = (
        cosine_similarity.argmax(dim=-1)
        .eq(t.arange(len(task["label_names"])))
        .float()
        .mean()
        .item()
    )
    return {
        "task_name": task_name,
        "label_names": list(task["label_names"]),
        "train_templates": list(task["train_templates"]),
        "heldout_templates": list(task["heldout_templates"]),
        "random_token_names": list(task["random_token_names"]),
        "train_prompts": list(task["train_prompts"]),
        "heldout_prompts": list(task["heldout_prompts"]),
        "train_labels": task["train_labels"].tolist(),
        "heldout_labels": task["heldout_labels"].tolist(),
        "random_token_labels": task["random_token_labels"].tolist(),
        "train_template_ids": task["train_template_ids"].tolist(),
        "heldout_template_ids": task["heldout_template_ids"].tolist(),
        "raw_pca": {
            "train_coordinates": raw_pca.train_projected.tolist(),
            "heldout_coordinates": raw_pca.heldout_projected.tolist(),
            "explained_variance_ratio": raw_pca.explained_variance_ratio.tolist(),
            "heldout_knn_accuracy": raw_knn.heldout_accuracy,
        },
        "centered_pca": {
            "train_coordinates": centered_pca.train_projected.tolist(),
            "heldout_coordinates": centered_pca.heldout_projected.tolist(),
            "explained_variance_ratio": centered_pca.explained_variance_ratio.tolist(),
            "heldout_knn_accuracy": centered_knn.heldout_accuracy,
        },
        "cosine_similarity": cosine_similarity.tolist(),
        "matched_pair_accuracy": matched_pair_accuracy,
        "visualization": visualization,
        "activations": {
            "train_raw": task["train_raw"],
            "heldout_raw": task["heldout_raw"],
            "train_centered": task["train_centered"],
            "heldout_centered": task["heldout_centered"],
            "random_token_centered": task["random_token_centered"],
        },
    }


def _evaluate_pythia_calendar_activations(
    activation_dataset: dict,
    *,
    max_vram_gb: float,
    min_centered_accuracy: float,
    include_signature_data: bool,
) -> dict:
    weekday_task = activation_dataset["tasks"]["weekday"]
    month_task = activation_dataset["tasks"]["month"]
    weekday_geometry = _template_centered_identity_geometry_from_task(
        weekday_task,
        noise_seed=0,
        min_centered_accuracy=min_centered_accuracy,
    )
    month_geometry = _template_centered_identity_geometry_from_task(
        month_task,
        noise_seed=1,
        min_centered_accuracy=min_centered_accuracy,
    )
    visualizations = {}
    for task_name, task in (("weekday", weekday_task), ("month", month_task)):
        visualizations[task_name] = _umap_visualization_sweep(
            train_activations=task["train_centered"],
            train_labels=task["train_labels"],
            heldout_activations=task["heldout_centered"],
            heldout_labels=task["heldout_labels"],
            random_token_activations=task["random_token_centered"],
            random_token_labels=task["random_token_labels"],
            task_name=task_name,
            include_selected_projection=include_signature_data,
        )
    weekday_visualization = visualizations["weekday"]
    month_visualization = visualizations["month"]
    within_vram_budget = activation_dataset["peak_vram_gb"] <= max_vram_gb
    preflight_passed = (
        not weekday_geometry["raw_predicts_heldout_labels"]
        and weekday_geometry["centered_predicts_heldout_labels"]
        and weekday_geometry["permuted_label_accuracy"] <= weekday_geometry["noise_accuracy"]
        and weekday_geometry["survives_white_noise_control"]
        and weekday_geometry["matched_pair_accuracy"] >= min_centered_accuracy
        and month_geometry["raw_heldout_accuracy"] <= 0.75
        and month_geometry["centered_predicts_heldout_labels"]
        and month_geometry["permuted_label_accuracy"] <= month_geometry["noise_accuracy"]
        and month_geometry["survives_white_noise_control"]
        and month_geometry["matched_pair_accuracy"] >= min_centered_accuracy
        and weekday_visualization["passes_visualization_controls"]
        and month_visualization["passes_visualization_controls"]
        and within_vram_budget
    )
    result = {
        "cuda_available": True,
        "cuda_version": activation_dataset["cuda_version"],
        "device": activation_dataset["device"],
        "model_id": activation_dataset["model_id"],
        "revision": activation_dataset["revision"],
        "claim_scope": "template_centered_weekday_and_month_hidden_state_geometry_preflight",
        "generation_used": activation_dataset["generation_used"],
        "calendar_task_count": 2,
        "weekday_count": len(WEEKDAYS),
        "month_count": len(MONTHS),
        "train_template": WEEKDAY_TRAIN_TEMPLATE,
        "heldout_template": WEEKDAY_HELDOUT_TEMPLATE,
        "raw_heldout_accuracy": weekday_geometry["raw_heldout_accuracy"],
        "raw_predicts_heldout_labels": weekday_geometry["raw_predicts_heldout_labels"],
        "centered_heldout_accuracy": weekday_geometry["centered_heldout_accuracy"],
        "centered_predicts_heldout_labels": weekday_geometry[
            "centered_predicts_heldout_labels"
        ],
        "permuted_label_accuracy": weekday_geometry["permuted_label_accuracy"],
        "noise_accuracy": weekday_geometry["noise_accuracy"],
        "white_noise_margin": weekday_geometry["white_noise_margin"],
        "survives_white_noise_control": weekday_geometry["survives_white_noise_control"],
        "matched_pair_accuracy": weekday_geometry["matched_pair_accuracy"],
        "pca_explained_variance_ratio": weekday_geometry[
            "pca_explained_variance_ratio"
        ],
        "weekday_geometry": weekday_geometry,
        "month_geometry": month_geometry,
        "weekday_visualization": weekday_visualization,
        "month_visualization": month_visualization,
        "visualization_seed_count": len(VISUALIZATION_SWEEP_SEEDS),
        "visualization_setting_count": len(VISUALIZATION_SWEEP_SETTINGS),
        "weekday_visualization_passed": weekday_visualization[
            "passes_visualization_controls"
        ],
        "month_visualization_passed": month_visualization[
            "passes_visualization_controls"
        ],
        "peak_vram_gb": activation_dataset["peak_vram_gb"],
        "within_vram_budget": within_vram_budget,
        "preflight_passed": preflight_passed,
    }
    if include_signature_data:
        result["signature"] = {
            "weekday": _calendar_signature_task_result(
                "weekday", weekday_task, weekday_visualization
            ),
            "month": _calendar_signature_task_result(
                "month", month_task, month_visualization
            ),
        }
    return result


def pythia_weekday_geometry_preflight(
    *,
    max_vram_gb: float = 24.0,
    model_id: str = PYTHIA_WEEKDAY_MODEL_ID,
    revision: str = PYTHIA_WEEKDAY_REVISION,
    min_centered_accuracy: float = 0.8,
) -> dict:
    """Run the compact, report-safe Pythia calendar geometry preflight."""

    activations = load_pythia_calendar_activations(
        max_vram_gb=max_vram_gb,
        model_id=model_id,
        revision=revision,
    )
    return _evaluate_pythia_calendar_activations(
        activations,
        max_vram_gb=max_vram_gb,
        min_centered_accuracy=min_centered_accuracy,
        include_signature_data=False,
    )


def run_pythia_calendar_signature_result(
    *,
    max_vram_gb: float = 24.0,
    model_id: str = PYTHIA_WEEKDAY_MODEL_ID,
    revision: str = PYTHIA_WEEKDAY_REVISION,
    min_centered_accuracy: float = 0.8,
) -> dict:
    """Run the live learner-facing result, including activations and plot coordinates."""

    activations = load_pythia_calendar_activations(
        max_vram_gb=max_vram_gb,
        model_id=model_id,
        revision=revision,
    )
    return _evaluate_pythia_calendar_activations(
        activations,
        max_vram_gb=max_vram_gb,
        min_centered_accuracy=min_centered_accuracy,
        include_signature_data=True,
    )

def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("11.1 GPU verification requires CUDA.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    activations = t.tensor(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
        ],
        device=device,
    )
    projection = pca_svd_projection(activations, n_components=1)
    train_points = t.tensor(
        [[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.2, 0.0]],
        device=device,
    )
    train_labels = t.tensor([0, 0, 1, 1], device=device)
    heldout_points = t.tensor([[0.1, 0.0], [2.1, 0.0]], device=device)
    heldout_labels = t.tensor([0, 1], device=device)
    prediction = geometry_label_prediction_report(
        train_points,
        train_labels,
        heldout_points,
        heldout_labels,
        min_accuracy=1.0,
    )
    t.cuda.synchronize()
    toy_peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    weekday_preflight = pythia_weekday_geometry_preflight(max_vram_gb=max_vram_gb)
    peak_vram_gb = max(toy_peak_vram_gb, weekday_preflight["peak_vram_gb"])
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "explained_variance_ratio": float(projection.explained_variance_ratio[0].item()),
        "heldout_accuracy": prediction.heldout_accuracy,
        "predicts_heldout_labels": prediction.predicts_heldout_labels,
        "template_centering_max_mean_abs": template_centering_smoke_test()[
            "max_template_mean_abs"
        ],
        "pythia_weekday_preflight_passed": weekday_preflight["preflight_passed"],
        "pythia_weekday_raw_heldout_accuracy": weekday_preflight["raw_heldout_accuracy"],
        "pythia_weekday_centered_heldout_accuracy": weekday_preflight[
            "centered_heldout_accuracy"
        ],
        "pythia_weekday_permuted_label_accuracy": weekday_preflight[
            "permuted_label_accuracy"
        ],
        "pythia_weekday_noise_accuracy": weekday_preflight["noise_accuracy"],
        "pythia_weekday_white_noise_margin": weekday_preflight["white_noise_margin"],
        "pythia_weekday_matched_pair_accuracy": weekday_preflight["matched_pair_accuracy"],
        "pythia_calendar_preflight_passed": weekday_preflight["preflight_passed"],
        "pythia_calendar_task_count": weekday_preflight["calendar_task_count"],
        "pythia_month_centered_heldout_accuracy": weekday_preflight["month_geometry"][
            "centered_heldout_accuracy"
        ],
        "pythia_month_raw_heldout_accuracy": weekday_preflight["month_geometry"][
            "raw_heldout_accuracy"
        ],
        "pythia_month_permuted_label_accuracy": weekday_preflight["month_geometry"][
            "permuted_label_accuracy"
        ],
        "pythia_month_noise_accuracy": weekday_preflight["month_geometry"][
            "noise_accuracy"
        ],
        "pythia_month_white_noise_margin": weekday_preflight["month_geometry"][
            "white_noise_margin"
        ],
        "pythia_month_matched_pair_accuracy": weekday_preflight["month_geometry"][
            "matched_pair_accuracy"
        ],
        "pythia_visualization_seed_count": weekday_preflight[
            "visualization_seed_count"
        ],
        "pythia_visualization_setting_count": weekday_preflight[
            "visualization_setting_count"
        ],
        "pythia_weekday_visualization_passed": weekday_preflight[
            "weekday_visualization_passed"
        ],
        "pythia_weekday_umap_min_heldout_knn_accuracy": weekday_preflight[
            "weekday_visualization"
        ]["sweep"]["min_heldout_knn_accuracy"],
        "pythia_weekday_umap_mean_heldout_knn_accuracy": weekday_preflight[
            "weekday_visualization"
        ]["sweep"]["mean_heldout_knn_accuracy"],
        "pythia_weekday_umap_min_trustworthiness": weekday_preflight[
            "weekday_visualization"
        ]["sweep"]["min_trustworthiness"],
        "pythia_weekday_umap_min_neighborhood_preservation": weekday_preflight[
            "weekday_visualization"
        ]["sweep"]["min_neighborhood_preservation"],
        "pythia_weekday_umap_random_label_accuracy_max": weekday_preflight[
            "weekday_visualization"
        ]["sweep"]["random_label_accuracy_max"],
        "pythia_weekday_umap_random_token_accuracy_max": weekday_preflight[
            "weekday_visualization"
        ]["sweep"]["random_token_accuracy_max"],
        "pythia_month_visualization_passed": weekday_preflight[
            "month_visualization_passed"
        ],
        "pythia_month_umap_min_heldout_knn_accuracy": weekday_preflight[
            "month_visualization"
        ]["sweep"]["min_heldout_knn_accuracy"],
        "pythia_month_umap_mean_heldout_knn_accuracy": weekday_preflight[
            "month_visualization"
        ]["sweep"]["mean_heldout_knn_accuracy"],
        "pythia_month_umap_min_trustworthiness": weekday_preflight[
            "month_visualization"
        ]["sweep"]["min_trustworthiness"],
        "pythia_month_umap_min_neighborhood_preservation": weekday_preflight[
            "month_visualization"
        ]["sweep"]["min_neighborhood_preservation"],
        "pythia_month_umap_random_label_accuracy_max": weekday_preflight[
            "month_visualization"
        ]["sweep"]["random_label_accuracy_max"],
        "pythia_month_umap_random_token_accuracy_max": weekday_preflight[
            "month_visualization"
        ]["sweep"]["random_token_accuracy_max"],
        "pythia_weekday_generation_used": weekday_preflight["generation_used"],
        "pythia_weekday_peak_vram_gb": weekday_preflight["peak_vram_gb"],
        "pythia_weekday_preflight": weekday_preflight,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb
        and weekday_preflight["within_vram_budget"],
        "full_path": "Run PCA/SVD and geometry controls on cached model activations.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
