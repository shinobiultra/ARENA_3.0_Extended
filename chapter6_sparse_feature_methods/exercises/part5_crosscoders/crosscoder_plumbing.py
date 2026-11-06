"""Exact record and ROC plumbing; crosscoder methods stay in the lesson."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t


FeatureOwner = Literal["shared", "model_a", "model_b"]


@dataclass(frozen=True)
class CrosscoderInterventionReport:
    target_latent: int
    baseline_abs_delta: float
    target_ablated_abs_delta: float
    same_norm_random_abs_delta: float
    orthogonal_abs_delta: float
    target_reduction: float
    same_norm_random_reduction: float
    orthogonal_reduction: float
    passes_controls: bool


@dataclass(frozen=True)
class PlantedCrosscoderData:
    """A paired activation dataset with known shared and model-specific features."""

    model_a_activations: t.Tensor
    model_b_activations: t.Tensor
    true_latents: t.Tensor
    behavior_labels: t.Tensor
    behavior_deltas: t.Tensor
    true_decoder_a: t.Tensor
    true_decoder_b: t.Tensor
    feature_names: tuple[str, ...]
    feature_owners: tuple[FeatureOwner, ...]
    prompts: tuple[str, ...]
    train_idx: t.Tensor
    heldout_idx: t.Tensor


@dataclass(frozen=True)
class PlantedFeatureMatchReport:
    feature_names: tuple[str, ...]
    true_owners: tuple[FeatureOwner, ...]
    matched_latents: tuple[int, ...]
    predicted_owners: tuple[FeatureOwner, ...]
    correlations: tuple[float, ...]
    ownership_accuracy: float
    min_correlation: float
    target_model_b_feature_id: int
    target_model_b_latent: int


@dataclass(frozen=True)
class SparseCrosscoderForward:
    model_a_feature_acts: t.Tensor
    model_b_feature_acts: t.Tensor
    reconstructed_model_a: t.Tensor
    reconstructed_model_b: t.Tensor


@dataclass(frozen=True)
class SparseCrosscoderLoss:
    reconstruction_loss: t.Tensor
    l1_loss: t.Tensor
    total_loss: t.Tensor


@dataclass(frozen=True)
class SparseCrosscoderReconstructionReport:
    model_a_mse: float
    model_b_mse: float
    heldout_reconstruction_mse: float
    zero_reconstruction_mse: float
    model_a_l0: float
    model_b_l0: float
    beats_zero_baseline: bool


@dataclass
class TrainedSparseCrosscoder:
    model: "SparseCrosscoder"
    history: list[dict[str, float]]
    train_idx: t.Tensor
    heldout_idx: t.Tensor


def roc_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Compute binary ROC AUC using rank statistics."""

    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same number of elements.")
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires at least one positive and one negative example.")

    order = scores.argsort()
    sorted_scores = scores[order]
    ranks_sorted = t.empty_like(sorted_scores)
    _, counts = t.unique_consecutive(sorted_scores, return_counts=True)
    start = 0
    for count in counts.tolist():
        end = start + count
        average_rank = (start + 1 + end) / 2
        ranks_sorted[start:end] = average_rank
        start = end
    ranks = t.empty_like(scores)
    ranks[order] = ranks_sorted
    pos_rank_sum = ranks[labels].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc.item())

