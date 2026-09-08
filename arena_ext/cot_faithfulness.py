"""Chain-of-thought faithfulness utilities for alignment notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t

from arena_ext.activation_language import prediction_accuracy


CoTCondition = Literal["no_cot", "faithful_cot", "biased_cot", "posthoc"]


@dataclass(frozen=True)
class PreFinalAnswerProbeReport:
    hidden_answer_accuracy: float
    final_answer_agreement: float
    predicts_hidden_answer: bool


@dataclass(frozen=True)
class HiddenAnswerPatchingReport:
    original_answer: int
    patched_answer: int
    changed_output: bool


@dataclass(frozen=True)
class CoTTextBaselineReport:
    detector_recall: float
    text_only_recall: float
    text_only_misses_cases: bool


@dataclass(frozen=True)
class FeatureDetectorReport:
    feature_accuracy: float
    baseline_accuracy: float
    improves_detection: bool


@dataclass(frozen=True)
class CoTConditionComparisonReport:
    condition_accuracies: dict[str, float]
    biased_gap: float
    posthoc_gap: float


def _require_finite_tensor(name: str, tensor: t.Tensor) -> None:
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not t.isfinite(tensor.float()).all():
        raise ValueError(f"{name} must contain only finite values.")


def _require_finite_scalar(name: str, value: float) -> None:
    value_tensor = t.tensor(value, dtype=t.float32)
    if not t.isfinite(value_tensor):
        raise ValueError(f"{name} must be finite.")


def pre_final_answer_probe_report(
    probe_logits: t.Tensor,
    hidden_answer_ids: t.Tensor,
    final_answer_ids: t.Tensor,
    *,
    min_hidden_accuracy: float = 0.8,
) -> PreFinalAnswerProbeReport:
    """Check whether a probe predicts hidden answers before the final token."""

    if hidden_answer_ids.shape != final_answer_ids.shape:
        raise ValueError("hidden and final answer ids must have matching shape.")
    _require_finite_tensor("probe_logits", probe_logits)
    _require_finite_tensor("hidden_answer_ids", hidden_answer_ids)
    _require_finite_tensor("final_answer_ids", final_answer_ids)
    _require_finite_scalar("min_hidden_accuracy", min_hidden_accuracy)
    hidden_accuracy = prediction_accuracy(probe_logits, hidden_answer_ids)
    predictions = probe_logits.argmax(dim=-1)
    final_agreement = predictions.eq(final_answer_ids).float().mean().item()
    return PreFinalAnswerProbeReport(
        hidden_answer_accuracy=hidden_accuracy,
        final_answer_agreement=final_agreement,
        predicts_hidden_answer=hidden_accuracy >= min_hidden_accuracy,
    )


def hidden_answer_patching_report(
    original_answer_logits: t.Tensor,
    patched_answer_logits: t.Tensor,
) -> HiddenAnswerPatchingReport:
    """Check whether patching hidden-answer state changes the output answer."""

    if original_answer_logits.ndim != 1 or patched_answer_logits.ndim != 1:
        raise ValueError("answer logits must be rank-1 tensors.")
    if original_answer_logits.shape != patched_answer_logits.shape:
        raise ValueError("answer logits must have matching shape.")
    _require_finite_tensor("original_answer_logits", original_answer_logits)
    _require_finite_tensor("patched_answer_logits", patched_answer_logits)
    original_answer = int(original_answer_logits.argmax().item())
    patched_answer = int(patched_answer_logits.argmax().item())
    return HiddenAnswerPatchingReport(
        original_answer=original_answer,
        patched_answer=patched_answer,
        changed_output=original_answer != patched_answer,
    )


def _binary_recall(predictions: t.Tensor, labels: t.Tensor) -> float:
    labels = labels.flatten().bool()
    predictions = predictions.flatten().bool()
    if labels.shape != predictions.shape:
        raise ValueError("predictions and labels must match.")
    positives = labels.sum().item()
    if positives == 0:
        raise ValueError("at least one positive label is required.")
    true_positives = predictions.logical_and(labels).float().sum().item()
    return true_positives / positives


def cot_text_baseline_report(
    detector_predictions: t.Tensor,
    text_only_predictions: t.Tensor,
    unfaithful_labels: t.Tensor,
) -> CoTTextBaselineReport:
    """Check whether a text-only CoT baseline misses unfaithful cases."""

    _require_finite_tensor("detector_predictions", detector_predictions)
    _require_finite_tensor("text_only_predictions", text_only_predictions)
    _require_finite_tensor("unfaithful_labels", unfaithful_labels)
    detector_recall = _binary_recall(detector_predictions, unfaithful_labels)
    text_only_recall = _binary_recall(text_only_predictions, unfaithful_labels)
    return CoTTextBaselineReport(
        detector_recall=detector_recall,
        text_only_recall=text_only_recall,
        text_only_misses_cases=text_only_recall < detector_recall,
    )


def feature_detector_report(
    feature_scores: t.Tensor,
    baseline_scores: t.Tensor,
    unfaithful_labels: t.Tensor,
    *,
    threshold: float = 0.5,
) -> FeatureDetectorReport:
    """Compare a feature-level unfaithfulness detector against a baseline."""

    _require_finite_tensor("feature_scores", feature_scores)
    _require_finite_tensor("baseline_scores", baseline_scores)
    _require_finite_tensor("unfaithful_labels", unfaithful_labels)
    _require_finite_scalar("threshold", threshold)
    labels = unfaithful_labels.flatten().bool()
    feature_predictions = feature_scores.flatten().float() >= threshold
    baseline_predictions = baseline_scores.flatten().float() >= threshold
    if feature_predictions.shape != labels.shape or baseline_predictions.shape != labels.shape:
        raise ValueError("scores and labels must have matching shape.")
    feature_accuracy = feature_predictions.eq(labels).float().mean().item()
    baseline_accuracy = baseline_predictions.eq(labels).float().mean().item()
    return FeatureDetectorReport(
        feature_accuracy=feature_accuracy,
        baseline_accuracy=baseline_accuracy,
        improves_detection=feature_accuracy > baseline_accuracy,
    )


def cot_condition_comparison_report(
    condition_correct: dict[CoTCondition, t.Tensor],
) -> CoTConditionComparisonReport:
    """Compare answer accuracy across no-CoT, faithful, biased, and post-hoc cases."""

    required = {"no_cot", "faithful_cot", "biased_cot", "posthoc"}
    if set(condition_correct) != required:
        raise ValueError("condition_correct must include all CoT conditions.")
    for condition, values in condition_correct.items():
        _require_finite_tensor(condition, values)
    accuracies = {
        condition: values.float().mean().item()
        for condition, values in condition_correct.items()
    }
    faithful_accuracy = accuracies["faithful_cot"]
    posthoc_accuracy = accuracies["posthoc"]
    return CoTConditionComparisonReport(
        condition_accuracies=accuracies,
        biased_gap=posthoc_accuracy - accuracies["biased_cot"],
        posthoc_gap=faithful_accuracy - posthoc_accuracy,
    )
