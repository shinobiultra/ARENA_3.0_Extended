"""White-box monitor utilities for alignment notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t


@dataclass(frozen=True)
class MonitorDashboardRow:
    prompt: str
    model_output: str
    active_features: tuple[str, ...]
    refusal_score: float
    hallucination_score: float
    cot_faithfulness_score: float


@dataclass(frozen=True)
class MonitorCalibrationReport:
    auroc: float
    calibrated: bool


@dataclass(frozen=True)
class MissedFailureReport:
    caught_failure_indices: tuple[int, ...]
    num_caught_failures: int
    catches_black_box_miss: bool


@dataclass(frozen=True)
class FalsePositiveDocumentationReport:
    false_positive_indices: tuple[int, ...]
    num_false_positives: int
    documented: bool


@dataclass(frozen=True)
class FeatureExplanationValidationReport:
    heldout_accuracy: float
    explanations_validated: bool


def _require_finite_tensor(name: str, tensor: t.Tensor) -> None:
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not t.isfinite(tensor.float()).all():
        raise ValueError(f"{name} must contain only finite values.")


def _require_finite_scalar(name: str, value: float) -> None:
    value_tensor = t.tensor(value, dtype=t.float32)
    if not t.isfinite(value_tensor):
        raise ValueError(f"{name} must be finite.")


def _require_unit_interval(name: str, value: float) -> None:
    _require_finite_scalar(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _require_binary_tensor(name: str, tensor: t.Tensor) -> None:
    _require_finite_tensor(name, tensor)
    if tensor.dtype == t.bool:
        return
    values_are_binary = tensor.eq(0) | tensor.eq(1)
    if not values_are_binary.all():
        raise ValueError(f"{name} must contain only binary 0/1 values.")


def monitor_dashboard_row(
    *,
    prompt: str,
    model_output: str,
    active_features: list[str],
    refusal_score: float,
    hallucination_score: float,
    cot_faithfulness_score: float,
) -> MonitorDashboardRow:
    """Bundle one white-box monitor dashboard row."""

    if not prompt.strip():
        raise ValueError("prompt must be non-empty.")
    if not model_output.strip():
        raise ValueError("model_output must be non-empty.")
    active_features_tuple = tuple(active_features)
    if not all(isinstance(feature, str) and feature.strip() for feature in active_features_tuple):
        raise ValueError("active_features must contain non-empty feature names.")
    for name, value in {
        "refusal_score": refusal_score,
        "hallucination_score": hallucination_score,
        "cot_faithfulness_score": cot_faithfulness_score,
    }.items():
        _require_finite_scalar(name, value)

    return MonitorDashboardRow(
        prompt=prompt,
        model_output=model_output,
        active_features=active_features_tuple,
        refusal_score=refusal_score,
        hallucination_score=hallucination_score,
        cot_faithfulness_score=cot_faithfulness_score,
    )


def binary_auroc(scores: t.Tensor, labels: t.Tensor) -> float:
    """Compute AUROC for binary labels without external dependencies."""

    scores = scores.flatten().float()
    labels = labels.flatten()
    _require_finite_tensor("scores", scores)
    _require_binary_tensor("labels", labels)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have matching shape.")
    labels = labels.bool()
    positive_scores = scores[labels]
    negative_scores = scores[~labels]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        raise ValueError("both positive and negative labels are required.")

    comparisons = positive_scores[:, None] - negative_scores[None, :]
    wins = comparisons.gt(0).float().sum().item()
    ties = comparisons.eq(0).float().sum().item()
    total_pairs = positive_scores.numel() * negative_scores.numel()
    return (wins + 0.5 * ties) / total_pairs


def monitor_calibration_report(
    monitor_scores: t.Tensor,
    failure_labels: t.Tensor,
    *,
    min_auroc: float = 0.8,
) -> MonitorCalibrationReport:
    """Check whether monitor scores separate failures by AUROC."""

    _require_unit_interval("min_auroc", min_auroc)
    auroc = binary_auroc(monitor_scores, failure_labels)
    return MonitorCalibrationReport(
        auroc=auroc,
        calibrated=auroc >= min_auroc,
    )


def missed_failure_report(
    white_box_predictions: t.Tensor,
    black_box_predictions: t.Tensor,
    failure_labels: t.Tensor,
) -> MissedFailureReport:
    """Find failures caught by white-box monitor but missed by black-box baseline."""

    white = white_box_predictions.flatten()
    black = black_box_predictions.flatten()
    labels = failure_labels.flatten()
    _require_binary_tensor("white_box_predictions", white)
    _require_binary_tensor("black_box_predictions", black)
    _require_binary_tensor("failure_labels", labels)
    if white.shape != black.shape or white.shape != labels.shape:
        raise ValueError("prediction and label tensors must have matching shape.")
    white = white.bool()
    black = black.bool()
    labels = labels.bool()
    caught_mask = labels & white & ~black
    indices = tuple(int(index.item()) for index in caught_mask.nonzero().flatten())
    return MissedFailureReport(
        caught_failure_indices=indices,
        num_caught_failures=len(indices),
        catches_black_box_miss=len(indices) > 0,
    )


def false_positive_documentation_report(
    monitor_predictions: t.Tensor,
    failure_labels: t.Tensor,
    documentation: dict[int, str] | None = None,
) -> FalsePositiveDocumentationReport:
    """Check whether monitor false positives have written notes."""

    predictions = monitor_predictions.flatten()
    labels = failure_labels.flatten()
    _require_binary_tensor("monitor_predictions", predictions)
    _require_binary_tensor("failure_labels", labels)
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have matching shape.")
    predictions = predictions.bool()
    labels = labels.bool()
    false_positive_mask = predictions & ~labels
    indices = tuple(int(index.item()) for index in false_positive_mask.nonzero().flatten())
    documentation = documentation or {}
    documented = all(bool(documentation.get(index, "").strip()) for index in indices)
    return FalsePositiveDocumentationReport(
        false_positive_indices=indices,
        num_false_positives=len(indices),
        documented=documented,
    )


def feature_explanation_validation_report(
    explanation_predictions: t.Tensor,
    heldout_labels: t.Tensor,
    *,
    min_accuracy: float = 0.8,
) -> FeatureExplanationValidationReport:
    """Validate feature explanations on held-out prompts."""

    _require_unit_interval("min_accuracy", min_accuracy)
    predictions = explanation_predictions.flatten()
    labels = heldout_labels.flatten()
    _require_binary_tensor("explanation_predictions", predictions)
    _require_binary_tensor("heldout_labels", labels)
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have matching shape.")
    predictions = predictions.bool()
    labels = labels.bool()
    accuracy = predictions.eq(labels).float().mean().item()
    return FeatureExplanationValidationReport(
        heldout_accuracy=accuracy,
        explanations_validated=accuracy >= min_accuracy,
    )
