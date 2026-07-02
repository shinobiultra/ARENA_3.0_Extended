"""Safe proxy drift-detection utilities for alignment notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t

from arena_ext.activation_language import prediction_accuracy


ProxyDriftKind = Literal[
    "sycophantic",
    "overconfident",
    "json_only",
    "style_drift",
    "refusal_overgeneralizing",
]


@dataclass(frozen=True)
class DriftDetectorReport:
    detector_accuracy: float
    predicts_heldout_drift: bool


@dataclass(frozen=True)
class CrosscoderDriftAlignmentReport:
    correlation: float
    aligns_with_behavior_delta: bool


@dataclass(frozen=True)
class DriftMitigationReport:
    baseline_drift_score: float
    mitigated_drift_score: float
    drift_reduction: float
    capability_loss: float
    mitigation_passes: bool


@dataclass(frozen=True)
class EarlyWarningReport:
    white_box_detection_step: int
    black_box_detection_step: int
    white_box_catches_earlier: bool


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


def _require_correlation_threshold(name: str, value: float) -> None:
    _require_finite_scalar(name, value)
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between -1 and 1.")


def _require_binary_detector_inputs(logits: t.Tensor, labels: t.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError("detector_logits must have shape (batch, 2).")
    if labels.shape != (logits.shape[0],):
        raise ValueError("drift_labels must have shape (batch,).")
    label_values = labels.long()
    if not labels.float().eq(label_values.float()).all():
        raise ValueError("drift_labels must contain integer class ids.")
    if not label_values.ge(0).logical_and(label_values.le(1)).all():
        raise ValueError("drift_labels must contain only 0 or 1.")


def safe_proxy_drift_kinds() -> tuple[ProxyDriftKind, ...]:
    """Return benign proxy drift categories for course exercises."""

    return (
        "sycophantic",
        "overconfident",
        "json_only",
        "style_drift",
        "refusal_overgeneralizing",
    )


def drift_detector_report(
    detector_logits: t.Tensor,
    drift_labels: t.Tensor,
    *,
    min_accuracy: float = 0.8,
) -> DriftDetectorReport:
    """Check whether a white-box detector predicts held-out drift labels."""

    _require_finite_tensor("detector_logits", detector_logits)
    _require_finite_tensor("drift_labels", drift_labels)
    _require_binary_detector_inputs(detector_logits, drift_labels)
    _require_unit_interval("min_accuracy", min_accuracy)
    accuracy = prediction_accuracy(detector_logits, drift_labels.long())
    return DriftDetectorReport(
        detector_accuracy=accuracy,
        predicts_heldout_drift=accuracy >= min_accuracy,
    )


def _pearson_correlation(left: t.Tensor, right: t.Tensor) -> float:
    left = left.flatten().float()
    right = right.flatten().float()
    if left.shape != right.shape:
        raise ValueError("correlation inputs must have matching shape.")
    if left.numel() < 2:
        raise ValueError("at least two values are required for correlation.")
    _require_finite_tensor("left", left)
    _require_finite_tensor("right", right)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if denominator.item() == 0:
        raise ValueError("correlation is undefined for constant inputs.")
    return float((left_centered @ right_centered / denominator).item())


def crosscoder_drift_alignment_report(
    model_specific_feature_scores: t.Tensor,
    behavior_delta_scores: t.Tensor,
    *,
    min_correlation: float = 0.8,
) -> CrosscoderDriftAlignmentReport:
    """Check whether model-specific features align with behavior deltas."""

    _require_correlation_threshold("min_correlation", min_correlation)
    correlation = _pearson_correlation(
        model_specific_feature_scores,
        behavior_delta_scores,
    )
    return CrosscoderDriftAlignmentReport(
        correlation=correlation,
        aligns_with_behavior_delta=correlation >= min_correlation,
    )


def drift_mitigation_report(
    baseline_drift_scores: t.Tensor,
    mitigated_drift_scores: t.Tensor,
    baseline_capability_scores: t.Tensor,
    mitigated_capability_scores: t.Tensor,
    *,
    min_drift_reduction: float = 0.2,
    max_capability_loss: float = 0.1,
) -> DriftMitigationReport:
    """Check whether mitigation reduces benign drift without large capability loss."""

    if baseline_drift_scores.shape != mitigated_drift_scores.shape:
        raise ValueError("drift score tensors must match.")
    if baseline_capability_scores.shape != mitigated_capability_scores.shape:
        raise ValueError("capability score tensors must match.")
    _require_finite_tensor("baseline_drift_scores", baseline_drift_scores)
    _require_finite_tensor("mitigated_drift_scores", mitigated_drift_scores)
    _require_finite_tensor("baseline_capability_scores", baseline_capability_scores)
    _require_finite_tensor("mitigated_capability_scores", mitigated_capability_scores)
    _require_finite_scalar("min_drift_reduction", min_drift_reduction)
    _require_finite_scalar("max_capability_loss", max_capability_loss)
    baseline_drift = baseline_drift_scores.float().mean().item()
    mitigated_drift = mitigated_drift_scores.float().mean().item()
    drift_reduction = baseline_drift - mitigated_drift
    baseline_capability = baseline_capability_scores.float().mean().item()
    mitigated_capability = mitigated_capability_scores.float().mean().item()
    capability_loss = baseline_capability - mitigated_capability
    return DriftMitigationReport(
        baseline_drift_score=baseline_drift,
        mitigated_drift_score=mitigated_drift,
        drift_reduction=drift_reduction,
        capability_loss=capability_loss,
        mitigation_passes=(
            drift_reduction >= min_drift_reduction
            and capability_loss <= max_capability_loss
        ),
    )


def early_warning_report(
    *,
    white_box_detection_step: int,
    black_box_detection_step: int,
) -> EarlyWarningReport:
    """Check whether a white-box detector catches drift earlier than black-box eval."""

    if white_box_detection_step < 0 or black_box_detection_step < 0:
        raise ValueError("detection steps must be nonnegative.")
    return EarlyWarningReport(
        white_box_detection_step=white_box_detection_step,
        black_box_detection_step=black_box_detection_step,
        white_box_catches_earlier=white_box_detection_step < black_box_detection_step,
    )
