"""Toy crosscoder and model-diffing utilities for sparse feature notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t
import torch.nn.functional as F

from arena_ext.features import roc_auc_binary


FeatureOwner = Literal["shared", "model_a", "model_b"]


@dataclass(frozen=True)
class CrosscoderOutput:
    shared_acts: t.Tensor
    model_a_specific_acts: t.Tensor
    model_b_specific_acts: t.Tensor
    reconstructed_model_a: t.Tensor
    reconstructed_model_b: t.Tensor


@dataclass(frozen=True)
class CrosscoderReconstructionReport:
    model_a_mse: float
    model_b_mse: float
    shared_active_fraction: float
    model_a_passes: bool
    model_b_passes: bool
    shared_reconstructs_both: bool


@dataclass(frozen=True)
class FeatureSpecificityReport:
    feature_id: int
    model_a_mean: float
    model_b_mean: float
    specificity: float
    owner: FeatureOwner


@dataclass(frozen=True)
class BehaviorDeltaPredictionReport:
    feature_id: int
    auc: float
    positive_mean: float
    negative_mean: float
    passes_threshold: bool


@dataclass(frozen=True)
class CrosscoderAblationReport:
    baseline_delta: float
    ablated_delta: float
    random_ablated_delta: float
    delta_reduction: float
    random_reduction: float
    passes_control: bool


def decode_crosscoder(
    shared_acts: t.Tensor,
    model_a_specific_acts: t.Tensor,
    model_b_specific_acts: t.Tensor,
    shared_decoder_a: t.Tensor,
    shared_decoder_b: t.Tensor,
    model_a_decoder: t.Tensor,
    model_b_decoder: t.Tensor,
) -> CrosscoderOutput:
    """Decode shared and model-specific features into two model activation spaces."""

    if shared_acts.shape[-1] != shared_decoder_a.shape[0]:
        raise ValueError("shared feature dimension must match model A shared decoder rows.")
    if shared_acts.shape[-1] != shared_decoder_b.shape[0]:
        raise ValueError("shared feature dimension must match model B shared decoder rows.")
    if model_a_specific_acts.shape[-1] != model_a_decoder.shape[0]:
        raise ValueError("model A feature dimension must match model A decoder rows.")
    if model_b_specific_acts.shape[-1] != model_b_decoder.shape[0]:
        raise ValueError("model B feature dimension must match model B decoder rows.")
    if shared_decoder_a.shape[1] != model_a_decoder.shape[1]:
        raise ValueError("model A decoder output dimensions must match.")
    if shared_decoder_b.shape[1] != model_b_decoder.shape[1]:
        raise ValueError("model B decoder output dimensions must match.")

    reconstructed_a = shared_acts.float() @ shared_decoder_a.float()
    reconstructed_a = reconstructed_a + model_a_specific_acts.float() @ model_a_decoder.float()
    reconstructed_b = shared_acts.float() @ shared_decoder_b.float()
    reconstructed_b = reconstructed_b + model_b_specific_acts.float() @ model_b_decoder.float()
    return CrosscoderOutput(
        shared_acts=shared_acts,
        model_a_specific_acts=model_a_specific_acts,
        model_b_specific_acts=model_b_specific_acts,
        reconstructed_model_a=reconstructed_a,
        reconstructed_model_b=reconstructed_b,
    )


def crosscoder_reconstruction_report(
    model_a_activations: t.Tensor,
    model_b_activations: t.Tensor,
    output: CrosscoderOutput,
    *,
    mse_threshold: float = 1e-6,
) -> CrosscoderReconstructionReport:
    """Check whether crosscoder reconstructions preserve both model activation spaces."""

    model_a_mse = F.mse_loss(
        output.reconstructed_model_a.float(),
        model_a_activations.float(),
    ).item()
    model_b_mse = F.mse_loss(
        output.reconstructed_model_b.float(),
        model_b_activations.float(),
    ).item()
    shared_active_fraction = output.shared_acts.gt(0).float().mean().item()
    model_a_passes = model_a_mse <= mse_threshold
    model_b_passes = model_b_mse <= mse_threshold
    return CrosscoderReconstructionReport(
        model_a_mse=model_a_mse,
        model_b_mse=model_b_mse,
        shared_active_fraction=shared_active_fraction,
        model_a_passes=model_a_passes,
        model_b_passes=model_b_passes,
        shared_reconstructs_both=model_a_passes and model_b_passes,
    )


def feature_specificity_report(
    model_a_feature_acts: t.Tensor,
    model_b_feature_acts: t.Tensor,
    feature_id: int,
    *,
    shared_threshold: float = 0.1,
) -> FeatureSpecificityReport:
    """Classify one feature as shared, model-A-specific, or model-B-specific."""

    if model_a_feature_acts.shape != model_b_feature_acts.shape:
        raise ValueError("model feature activation tensors must have matching shapes.")
    if feature_id < 0 or feature_id >= model_a_feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")
    model_a_mean = model_a_feature_acts[..., feature_id].float().mean().item()
    model_b_mean = model_b_feature_acts[..., feature_id].float().mean().item()
    specificity = model_b_mean - model_a_mean
    if abs(specificity) <= shared_threshold:
        owner: FeatureOwner = "shared"
    elif specificity > 0:
        owner = "model_b"
    else:
        owner = "model_a"
    return FeatureSpecificityReport(
        feature_id=feature_id,
        model_a_mean=model_a_mean,
        model_b_mean=model_b_mean,
        specificity=specificity,
        owner=owner,
    )


def classify_features_by_specificity(
    model_a_feature_acts: t.Tensor,
    model_b_feature_acts: t.Tensor,
    *,
    shared_threshold: float = 0.1,
) -> list[FeatureOwner]:
    """Classify every feature by cross-model activation specificity."""

    return [
        feature_specificity_report(
            model_a_feature_acts,
            model_b_feature_acts,
            feature_id,
            shared_threshold=shared_threshold,
        ).owner
        for feature_id in range(model_a_feature_acts.shape[-1])
    ]


def behavior_delta_prediction_report(
    feature_scores: t.Tensor,
    behavior_delta_labels: t.Tensor,
    *,
    feature_id: int,
    min_auc: float = 0.8,
) -> BehaviorDeltaPredictionReport:
    """Check whether a feature predicts examples with a behavior delta."""

    scores = feature_scores.flatten().float()
    labels = behavior_delta_labels.flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("feature_scores and behavior_delta_labels must have equal length.")
    auc = roc_auc_binary(scores, labels)
    signed_auc = max(auc, 1.0 - auc)
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("Need at least one positive and one negative example.")
    return BehaviorDeltaPredictionReport(
        feature_id=feature_id,
        auc=signed_auc,
        positive_mean=positives.mean().item(),
        negative_mean=negatives.mean().item(),
        passes_threshold=signed_auc >= min_auc,
    )


def crosscoder_ablation_report(
    baseline_behavior_deltas: t.Tensor,
    ablated_behavior_deltas: t.Tensor,
    random_ablated_behavior_deltas: t.Tensor,
) -> CrosscoderAblationReport:
    """Check whether ablating model-specific features reduces behavior deltas."""

    baseline_delta = baseline_behavior_deltas.float().mean().item()
    ablated_delta = ablated_behavior_deltas.float().mean().item()
    random_delta = random_ablated_behavior_deltas.float().mean().item()
    baseline_abs = abs(baseline_delta)
    ablated_abs = abs(ablated_delta)
    random_abs = abs(random_delta)
    delta_reduction = baseline_abs - ablated_abs
    random_reduction = baseline_abs - random_abs
    return CrosscoderAblationReport(
        baseline_delta=baseline_delta,
        ablated_delta=ablated_delta,
        random_ablated_delta=random_delta,
        delta_reduction=delta_reduction,
        random_reduction=random_reduction,
        passes_control=delta_reduction > random_reduction,
    )


def toy_behavior_delta_scores(
    model_a_scores: t.Tensor,
    model_b_scores: t.Tensor,
) -> t.Tensor:
    """Return model-B minus model-A behavior scores for paired prompts."""

    if model_a_scores.shape != model_b_scores.shape:
        raise ValueError("model_a_scores and model_b_scores must have matching shapes.")
    return model_b_scores.float() - model_a_scores.float()
