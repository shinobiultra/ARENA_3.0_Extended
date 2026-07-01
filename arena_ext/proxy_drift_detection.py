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
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if denominator.item() == 0:
        return 0.0
    return float((left_centered @ right_centered / denominator).item())


def crosscoder_drift_alignment_report(
    model_specific_feature_scores: t.Tensor,
    behavior_delta_scores: t.Tensor,
    *,
    min_correlation: float = 0.8,
) -> CrosscoderDriftAlignmentReport:
    """Check whether model-specific features align with behavior deltas."""

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
