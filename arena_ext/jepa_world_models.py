"""JEPA and world-model interpretability utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t


@dataclass(frozen=True)
class JEPAPredictionReport:
    mean_cosine: float
    mse: float
    predicts_target: bool


@dataclass(frozen=True)
class WorldStateProbeReport:
    accuracy: float
    predicts_state: bool


@dataclass(frozen=True)
class TransitionConsistencyReport:
    mean_cosine: float
    transition_consistent: bool


@dataclass(frozen=True)
class ObjectPermanenceReport:
    visible_mean: float
    occluded_mean: float
    absent_mean: float
    occluded_absent_gap: float
    preserves_occluded_object: bool


@dataclass(frozen=True)
class LossDecreaseReport:
    initial_loss: float
    final_loss: float
    baseline_loss: float
    relative_reduction: float
    beats_baseline: bool
    loss_decreases: bool


@dataclass(frozen=True)
class CollapseDiagnosticsReport:
    finite_features: bool
    feature_std: float
    effective_rank: float
    non_collapsed: bool


@dataclass(frozen=True)
class LatentRolloutReport:
    rollout_loss: float
    copy_baseline_loss: float
    shuffled_action_loss: float
    beats_copy_baseline: bool
    shuffled_action_fails: bool
    rollout_passes: bool


@dataclass(frozen=True)
class CausalLatentPatchReport:
    object_patch_effect: float
    random_patch_effect: float
    patch_random_gap: float
    causal_patch_passes: bool


def _paired_cosine(left: t.Tensor, right: t.Tensor, *, eps: float = 1e-8) -> t.Tensor:
    left_float = left.float()
    right_float = right.float()
    numerator = (left_float * right_float).sum(dim=-1)
    denominator = left_float.norm(dim=-1) * right_float.norm(dim=-1)
    return numerator / denominator.clamp_min(eps)


def jepa_prediction_report(
    predicted_targets: t.Tensor,
    target_embeddings: t.Tensor,
    *,
    min_cosine: float = 0.8,
    max_mse: float = 0.05,
) -> JEPAPredictionReport:
    """Check whether a JEPA-style predictor matches target embeddings."""

    if predicted_targets.shape != target_embeddings.shape:
        raise ValueError("predicted_targets and target_embeddings must match.")
    if predicted_targets.ndim != 2:
        raise ValueError("embeddings must have shape (examples, d_model).")

    mean_cosine = _paired_cosine(predicted_targets, target_embeddings).mean().item()
    mse = (predicted_targets.float() - target_embeddings.float()).pow(2).mean().item()
    return JEPAPredictionReport(
        mean_cosine=mean_cosine,
        mse=mse,
        predicts_target=mean_cosine >= min_cosine and mse <= max_mse,
    )


def world_state_probe_report(
    probe_logits: t.Tensor,
    labels: t.Tensor,
    *,
    min_accuracy: float = 0.9,
) -> WorldStateProbeReport:
    """Check whether a latent world-model state predicts held-out labels."""

    if probe_logits.ndim != 2:
        raise ValueError("probe_logits must have shape (examples, classes).")
    flattened_labels = labels.flatten().long()
    if flattened_labels.numel() != probe_logits.shape[0]:
        raise ValueError("labels must have one value per example.")

    predictions = probe_logits.argmax(dim=-1)
    accuracy = predictions.eq(flattened_labels).float().mean().item()
    return WorldStateProbeReport(
        accuracy=accuracy,
        predicts_state=accuracy >= min_accuracy,
    )


def transition_consistency_report(
    state_embeddings: t.Tensor,
    action_deltas: t.Tensor,
    next_state_embeddings: t.Tensor,
    *,
    min_cosine: float = 0.8,
) -> TransitionConsistencyReport:
    """Check whether state plus action delta predicts the next latent state."""

    if state_embeddings.shape != action_deltas.shape:
        raise ValueError("state_embeddings and action_deltas must match.")
    if state_embeddings.shape != next_state_embeddings.shape:
        raise ValueError("state and next_state embeddings must match.")

    predicted_next = state_embeddings.float() + action_deltas.float()
    mean_cosine = _paired_cosine(predicted_next, next_state_embeddings).mean().item()
    return TransitionConsistencyReport(
        mean_cosine=mean_cosine,
        transition_consistent=mean_cosine >= min_cosine,
    )


def object_permanence_report(
    visible_scores: t.Tensor,
    occluded_scores: t.Tensor,
    absent_scores: t.Tensor,
    *,
    min_occluded_score: float = 0.6,
    min_absent_gap: float = 0.3,
) -> ObjectPermanenceReport:
    """Check whether occluded objects stay represented more than absent objects."""

    visible_mean = visible_scores.float().mean().item()
    occluded_mean = occluded_scores.float().mean().item()
    absent_mean = absent_scores.float().mean().item()
    occluded_absent_gap = occluded_mean - absent_mean
    preserves_occluded_object = (
        occluded_mean >= min_occluded_score
        and occluded_absent_gap >= min_absent_gap
    )
    return ObjectPermanenceReport(
        visible_mean=visible_mean,
        occluded_mean=occluded_mean,
        absent_mean=absent_mean,
        occluded_absent_gap=occluded_absent_gap,
        preserves_occluded_object=preserves_occluded_object,
    )


def loss_decrease_report(
    initial_loss: float,
    final_loss: float,
    baseline_loss: float,
    *,
    min_relative_reduction: float = 0.25,
    max_final_to_baseline: float = 0.8,
) -> LossDecreaseReport:
    """Check whether a trained latent predictor improves over its initial and baseline losses."""

    initial = float(initial_loss)
    final = float(final_loss)
    baseline = float(baseline_loss)
    relative_reduction = (initial - final) / max(initial, 1e-12)
    beats_baseline = final <= baseline * max_final_to_baseline
    return LossDecreaseReport(
        initial_loss=initial,
        final_loss=final,
        baseline_loss=baseline,
        relative_reduction=relative_reduction,
        beats_baseline=beats_baseline,
        loss_decreases=relative_reduction >= min_relative_reduction and beats_baseline,
    )


def collapse_diagnostics_report(
    features: t.Tensor,
    *,
    min_feature_std: float = 0.1,
    min_effective_rank: float = 3.0,
) -> CollapseDiagnosticsReport:
    """Check that a latent representation is finite and not collapsed to one direction."""

    if features.ndim < 2:
        raise ValueError("features must have at least example and feature dimensions.")
    flat = features.float().reshape(features.shape[0], -1)
    finite_features = bool(t.isfinite(flat).all().item())
    feature_std = flat.std().item()
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular_values = t.linalg.svdvals(centered)
    spectrum = singular_values.pow(2)
    probabilities = spectrum / spectrum.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    effective_rank = entropy.exp().item()
    return CollapseDiagnosticsReport(
        finite_features=finite_features,
        feature_std=feature_std,
        effective_rank=effective_rank,
        non_collapsed=(
            finite_features
            and feature_std >= min_feature_std
            and effective_rank >= min_effective_rank
        ),
    )


def latent_rollout_report(
    rollout_loss: float,
    copy_baseline_loss: float,
    shuffled_action_loss: float,
    *,
    max_rollout_to_copy: float = 0.8,
    max_rollout_to_shuffled: float = 0.8,
) -> LatentRolloutReport:
    """Check that an action-conditioned latent rollout beats copy and shuffled-action baselines."""

    rollout = float(rollout_loss)
    copy = float(copy_baseline_loss)
    shuffled = float(shuffled_action_loss)
    beats_copy = rollout <= copy * max_rollout_to_copy
    shuffled_fails = rollout <= shuffled * max_rollout_to_shuffled
    return LatentRolloutReport(
        rollout_loss=rollout,
        copy_baseline_loss=copy,
        shuffled_action_loss=shuffled,
        beats_copy_baseline=beats_copy,
        shuffled_action_fails=shuffled_fails,
        rollout_passes=beats_copy and shuffled_fails,
    )


def causal_latent_patch_report(
    object_patch_effects: t.Tensor,
    random_patch_effects: t.Tensor,
    *,
    min_object_patch_effect: float = 0.2,
    min_patch_random_gap: float = 0.1,
) -> CausalLatentPatchReport:
    """Compare targeted latent-token patches against same-size random-token patches."""

    if object_patch_effects.numel() == 0 or random_patch_effects.numel() == 0:
        raise ValueError("patch effect tensors must be non-empty.")
    object_effect = object_patch_effects.float().mean().item()
    random_effect = random_patch_effects.float().mean().item()
    gap = object_effect - random_effect
    return CausalLatentPatchReport(
        object_patch_effect=object_effect,
        random_patch_effect=random_effect,
        patch_random_gap=gap,
        causal_patch_passes=(
            object_effect >= min_object_patch_effect and gap >= min_patch_random_gap
        ),
    )
