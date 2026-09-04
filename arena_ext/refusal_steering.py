"""Refusal-direction and safe-steering utilities for alignment notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t


SteeringDirection = Literal["increase", "decrease"]


@dataclass(frozen=True)
class RefusalSeparationReport:
    refusal_mean_score: float
    non_refusal_mean_score: float
    margin: float
    accuracy: float
    separates_refusal: bool


@dataclass(frozen=True)
class SteeringEffectReport:
    baseline_refusal_rate: float
    steered_refusal_rate: float
    refusal_rate_delta: float
    changes_refusal_rate: bool


@dataclass(frozen=True)
class CapabilityDegradationReport:
    baseline_capability: float
    steered_capability: float
    degradation: float
    degradation_small: bool


@dataclass(frozen=True)
class RandomDirectionControlReport:
    target_direction_delta: float
    random_direction_delta: float
    margin: float
    random_direction_fails: bool


@dataclass(frozen=True)
class LabelShuffleControlReport:
    true_accuracy: float
    shuffled_accuracy: float
    accuracy_gap: float
    true_margin: float
    shuffled_margin: float
    label_shuffle_fails: bool


def _maximally_mismatched_labels(labels: t.Tensor) -> t.Tensor:
    """Return a same-count label assignment that breaks the original grouping."""

    positive_indices = labels.nonzero(as_tuple=False).flatten()
    negative_indices = (~labels).nonzero(as_tuple=False).flatten()
    n_positive = positive_indices.numel()

    shuffled_positive_indices = negative_indices[:n_positive]
    if shuffled_positive_indices.numel() < n_positive:
        remaining = n_positive - shuffled_positive_indices.numel()
        shuffled_positive_indices = t.cat(
            [shuffled_positive_indices, positive_indices[:remaining]],
            dim=0,
        )

    shuffled = t.zeros_like(labels)
    shuffled[shuffled_positive_indices] = True
    return shuffled


@dataclass(frozen=True)
class DirectionComparisonReport:
    method_scores: dict[str, float]
    best_method: str
    best_score: float


def _require_finite_tensor(name: str, tensor: t.Tensor) -> None:
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be nonempty.")
    if not t.isfinite(tensor.float()).all():
        raise ValueError(f"{name} must contain only finite values.")


def _require_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not t.isfinite(t.tensor(scalar)).item():
        raise ValueError(f"{name} must be finite.")
    return scalar


def _require_binary_labels(labels: t.Tensor) -> t.Tensor:
    _require_finite_tensor("labels", labels)
    flattened = labels.flatten()
    if flattened.dtype == t.bool:
        return flattened
    if not flattened.eq(0).logical_or(flattened.eq(1)).all():
        raise ValueError("labels must be boolean or binary 0/1 values.")
    return flattened.bool()


def _require_nonzero_direction(direction: t.Tensor) -> t.Tensor:
    _require_finite_tensor("direction", direction)
    norm = direction.float().norm()
    if norm.item() == 0:
        raise ValueError("direction has zero norm.")
    return direction.float() / norm


def mean_difference_direction(
    refusal_activations: t.Tensor,
    non_refusal_activations: t.Tensor,
) -> t.Tensor:
    """Return a normalized refusal-minus-non-refusal direction."""

    if refusal_activations.ndim != 2 or non_refusal_activations.ndim != 2:
        raise ValueError("activation tensors must have shape (examples, d_model).")
    if refusal_activations.shape[-1] != non_refusal_activations.shape[-1]:
        raise ValueError("activation dimensions must match.")
    _require_finite_tensor("refusal_activations", refusal_activations)
    _require_finite_tensor("non_refusal_activations", non_refusal_activations)
    direction = refusal_activations.float().mean(dim=0)
    direction = direction - non_refusal_activations.float().mean(dim=0)
    try:
        return _require_nonzero_direction(direction)
    except ValueError as exc:
        raise ValueError("mean-difference direction has zero norm.") from exc


def refusal_direction_scores(
    activations: t.Tensor,
    direction: t.Tensor,
) -> t.Tensor:
    """Project activations onto a candidate refusal direction."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    if direction.shape != (activations.shape[-1],):
        raise ValueError("direction must have shape (d_model,).")
    _require_finite_tensor("activations", activations)
    unit_direction = _require_nonzero_direction(direction)
    return activations.float() @ unit_direction


def refusal_separation_report(
    activations: t.Tensor,
    refusal_labels: t.Tensor,
    direction: t.Tensor,
    *,
    min_accuracy: float = 0.9,
) -> RefusalSeparationReport:
    """Check whether a direction separates refusal from non-refusal examples."""

    labels = _require_binary_labels(refusal_labels)
    if activations.shape[0] != labels.numel():
        raise ValueError("labels must have one entry per activation.")
    min_accuracy = _require_finite_scalar("min_accuracy", min_accuracy)
    scores = refusal_direction_scores(activations, direction)
    refusal_scores = scores[labels]
    non_refusal_scores = scores[~labels]
    if refusal_scores.numel() == 0 or non_refusal_scores.numel() == 0:
        raise ValueError("both refusal and non-refusal examples are required.")
    refusal_mean = refusal_scores.mean().item()
    non_refusal_mean = non_refusal_scores.mean().item()
    threshold = 0.5 * (refusal_mean + non_refusal_mean)
    predictions = scores >= threshold
    accuracy = predictions.eq(labels).float().mean().item()
    margin = refusal_mean - non_refusal_mean
    return RefusalSeparationReport(
        refusal_mean_score=refusal_mean,
        non_refusal_mean_score=non_refusal_mean,
        margin=margin,
        accuracy=accuracy,
        separates_refusal=accuracy >= min_accuracy and margin > 0,
    )


def steering_effect_report(
    baseline_refusal_scores: t.Tensor,
    steered_refusal_scores: t.Tensor,
    *,
    threshold: float = 0.5,
    expected_direction: SteeringDirection = "increase",
    min_rate_delta: float = 0.2,
) -> SteeringEffectReport:
    """Check whether steering changes refusal rate on a safe benchmark."""

    if baseline_refusal_scores.shape != steered_refusal_scores.shape:
        raise ValueError("baseline and steered scores must have matching shape.")
    _require_finite_tensor("baseline_refusal_scores", baseline_refusal_scores)
    _require_finite_tensor("steered_refusal_scores", steered_refusal_scores)
    threshold = _require_finite_scalar("threshold", threshold)
    min_rate_delta = _require_finite_scalar("min_rate_delta", min_rate_delta)
    baseline_rate = baseline_refusal_scores.float().ge(threshold).float().mean().item()
    steered_rate = steered_refusal_scores.float().ge(threshold).float().mean().item()
    rate_delta = steered_rate - baseline_rate
    if expected_direction == "increase":
        changes_rate = rate_delta >= min_rate_delta
    elif expected_direction == "decrease":
        changes_rate = -rate_delta >= min_rate_delta
    else:
        raise ValueError("expected_direction must be 'increase' or 'decrease'.")
    return SteeringEffectReport(
        baseline_refusal_rate=baseline_rate,
        steered_refusal_rate=steered_rate,
        refusal_rate_delta=rate_delta,
        changes_refusal_rate=changes_rate,
    )


def capability_degradation_report(
    baseline_capability_scores: t.Tensor,
    steered_capability_scores: t.Tensor,
    *,
    max_degradation: float = 0.1,
) -> CapabilityDegradationReport:
    """Check that steering does not greatly degrade general capability."""

    if baseline_capability_scores.shape != steered_capability_scores.shape:
        raise ValueError("baseline and steered capability scores must match.")
    _require_finite_tensor("baseline_capability_scores", baseline_capability_scores)
    _require_finite_tensor("steered_capability_scores", steered_capability_scores)
    max_degradation = _require_finite_scalar("max_degradation", max_degradation)
    baseline = baseline_capability_scores.float().mean().item()
    steered = steered_capability_scores.float().mean().item()
    degradation = baseline - steered
    return CapabilityDegradationReport(
        baseline_capability=baseline,
        steered_capability=steered,
        degradation=degradation,
        degradation_small=degradation <= max_degradation,
    )


def random_direction_control_report(
    *,
    target_direction_delta: float,
    random_direction_delta: float,
    min_margin: float = 0.2,
    expected_direction: SteeringDirection | None = None,
) -> RandomDirectionControlReport:
    """Check that the refusal direction beats a random direction control."""

    target_direction_delta = _require_finite_scalar(
        "target_direction_delta", target_direction_delta
    )
    random_direction_delta = _require_finite_scalar(
        "random_direction_delta", random_direction_delta
    )
    min_margin = _require_finite_scalar("min_margin", min_margin)
    if expected_direction == "increase":
        margin = target_direction_delta - max(random_direction_delta, 0.0)
    elif expected_direction == "decrease":
        margin = -target_direction_delta - max(-random_direction_delta, 0.0)
    elif expected_direction is None:
        margin = abs(target_direction_delta) - abs(random_direction_delta)
    else:
        raise ValueError("expected_direction must be 'increase', 'decrease', or None.")
    return RandomDirectionControlReport(
        target_direction_delta=target_direction_delta,
        random_direction_delta=random_direction_delta,
        margin=margin,
        random_direction_fails=margin >= min_margin,
    )


def label_shuffle_control_report(
    activations: t.Tensor,
    refusal_labels: t.Tensor,
    *,
    min_accuracy_gap: float = 0.25,
) -> LabelShuffleControlReport:
    """Check that shuffled labels do not recover the same refusal direction."""

    labels = _require_binary_labels(refusal_labels)
    if activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    if activations.shape[0] != labels.numel():
        raise ValueError("labels must have one entry per activation.")
    _require_finite_tensor("activations", activations)
    min_accuracy_gap = _require_finite_scalar("min_accuracy_gap", min_accuracy_gap)
    if labels.sum().item() == 0 or (~labels).sum().item() == 0:
        raise ValueError("both refusal and non-refusal examples are required.")

    true_direction = mean_difference_direction(activations[labels], activations[~labels])
    true_report = refusal_separation_report(activations, labels, true_direction, min_accuracy=0.0)

    shuffled_labels = _maximally_mismatched_labels(labels)
    if shuffled_labels.equal(labels) or shuffled_labels.sum().item() == 0 or (~shuffled_labels).sum().item() == 0:
        shuffled_labels = labels.roll(shifts=1)
    shuffled_direction = mean_difference_direction(
        activations[shuffled_labels],
        activations[~shuffled_labels],
    )
    shuffled_report = refusal_separation_report(
        activations,
        labels,
        shuffled_direction,
        min_accuracy=0.0,
    )
    accuracy_gap = true_report.accuracy - shuffled_report.accuracy
    return LabelShuffleControlReport(
        true_accuracy=true_report.accuracy,
        shuffled_accuracy=shuffled_report.accuracy,
        accuracy_gap=accuracy_gap,
        true_margin=true_report.margin,
        shuffled_margin=shuffled_report.margin,
        label_shuffle_fails=accuracy_gap >= min_accuracy_gap,
    )


def direction_comparison_report(
    method_scores: dict[str, float],
) -> DirectionComparisonReport:
    """Return the best-scoring refusal-direction construction method."""

    if not method_scores:
        raise ValueError("method_scores must be nonempty.")
    for method, score in method_scores.items():
        _require_finite_scalar(f"method_scores[{method!r}]", score)
    best_method = max(method_scores, key=method_scores.__getitem__)
    return DirectionComparisonReport(
        method_scores=dict(method_scores),
        best_method=best_method,
        best_score=method_scores[best_method],
    )
