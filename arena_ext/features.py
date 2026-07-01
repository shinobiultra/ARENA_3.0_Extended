"""Sparse feature validation utilities for ARENA extension notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t
import torch.nn.functional as F


@dataclass(frozen=True)
class SAEReconstructionMetrics:
    l0: float
    feature_density_mean: float
    dead_feature_fraction: float
    reconstruction_mse: float
    reconstruction_kl: float | None = None
    loss_recovered: float | None = None


@dataclass(frozen=True)
class FeatureDetectionReport:
    auc: float
    positive_mean: float
    negative_mean: float
    separation: float
    threshold_accuracy: float


@dataclass(frozen=True)
class SteeringComparisonReport:
    baseline_mean: float
    steered_mean: float
    random_mean: float
    steered_delta: float
    random_delta: float
    passes_control: bool


def feature_density(feature_acts: t.Tensor, threshold: float = 0.0) -> t.Tensor:
    """Return per-feature firing rate over all non-feature dimensions."""

    if feature_acts.ndim < 2:
        raise ValueError("feature_acts must have at least batch and feature dimensions.")
    fired = feature_acts > threshold
    reduce_dims = tuple(range(feature_acts.ndim - 1))
    return fired.float().mean(dim=reduce_dims)


def l0(feature_acts: t.Tensor, threshold: float = 0.0) -> float:
    """Return average number of active features per activation vector."""

    return (feature_acts > threshold).float().sum(dim=-1).mean().item()


def dead_feature_fraction(feature_acts: t.Tensor, threshold: float = 0.0) -> float:
    """Return fraction of features that never fire above threshold."""

    densities = feature_density(feature_acts, threshold=threshold)
    return densities.eq(0).float().mean().item()


def mean_kl_divergence(reference_logits: t.Tensor, reconstructed_logits: t.Tensor) -> float:
    """Compute mean KL(reference || reconstructed) over all non-vocab positions."""

    if reference_logits.shape != reconstructed_logits.shape:
        raise ValueError("reference_logits and reconstructed_logits must have matching shapes.")
    ref_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    recon_log_probs = F.log_softmax(reconstructed_logits.float(), dim=-1)
    ref_probs = ref_log_probs.exp()
    return (ref_probs * (ref_log_probs - recon_log_probs)).sum(dim=-1).mean().item()


def loss_recovered(
    *,
    clean_loss: float,
    reconstructed_loss: float,
    zero_ablation_loss: float,
) -> float:
    """Return the standard SAE loss-recovered score."""

    denominator = zero_ablation_loss - clean_loss
    if denominator == 0:
        raise ValueError("zero_ablation_loss and clean_loss must differ.")
    return (zero_ablation_loss - reconstructed_loss) / denominator


def compute_sae_reconstruction_metrics(
    *,
    activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    feature_acts: t.Tensor,
    threshold: float = 0.0,
    reference_logits: t.Tensor | None = None,
    reconstructed_logits: t.Tensor | None = None,
    clean_loss: float | None = None,
    reconstructed_loss: float | None = None,
    zero_ablation_loss: float | None = None,
) -> SAEReconstructionMetrics:
    """Compute common SAE reconstruction and sparsity metrics."""

    if activations.shape != reconstructed_activations.shape:
        raise ValueError("activations and reconstructed_activations must have matching shapes.")

    reconstruction_kl = None
    if reference_logits is not None or reconstructed_logits is not None:
        if reference_logits is None or reconstructed_logits is None:
            raise ValueError("Provide both reference_logits and reconstructed_logits, or neither.")
        reconstruction_kl = mean_kl_divergence(reference_logits, reconstructed_logits)

    recovered = None
    if clean_loss is not None or reconstructed_loss is not None or zero_ablation_loss is not None:
        if clean_loss is None or reconstructed_loss is None or zero_ablation_loss is None:
            raise ValueError(
                "Provide clean_loss, reconstructed_loss, and zero_ablation_loss together."
            )
        recovered = loss_recovered(
            clean_loss=clean_loss,
            reconstructed_loss=reconstructed_loss,
            zero_ablation_loss=zero_ablation_loss,
        )

    return SAEReconstructionMetrics(
        l0=l0(feature_acts, threshold=threshold),
        feature_density_mean=feature_density(feature_acts, threshold=threshold).mean().item(),
        dead_feature_fraction=dead_feature_fraction(feature_acts, threshold=threshold),
        reconstruction_mse=F.mse_loss(
            reconstructed_activations.float(),
            activations.float(),
        ).item(),
        reconstruction_kl=reconstruction_kl,
        loss_recovered=recovered,
    )


def direct_logit_attribution(
    decoder_vectors: t.Tensor,
    unembedding: t.Tensor,
    token_ids: t.Tensor | list[int] | None = None,
) -> t.Tensor:
    """Project decoder vectors into vocabulary-logit space.

    Args:
        decoder_vectors: Tensor of shape ``(features, d_model)``.
        unembedding: Tensor of shape ``(d_model, vocab)``.
        token_ids: Optional token ids to select from the vocab dimension.
    """

    if decoder_vectors.ndim != 2 or unembedding.ndim != 2:
        raise ValueError("decoder_vectors and unembedding must both be rank-2 tensors.")
    if decoder_vectors.shape[-1] != unembedding.shape[0]:
        raise ValueError("decoder_vectors last dimension must match unembedding first dimension.")
    logit_effects = decoder_vectors.float() @ unembedding.float()
    if token_ids is None:
        return logit_effects
    return logit_effects[:, t.as_tensor(token_ids, device=logit_effects.device)]


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


def feature_detection_report(
    scores: t.Tensor,
    labels: t.Tensor,
    threshold: float | None = None,
) -> FeatureDetectionReport:
    """Evaluate whether a feature separates positives from matched negatives."""

    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same number of elements.")

    positives = scores[labels]
    negatives = scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("Need at least one positive and one negative example.")

    if threshold is None:
        threshold = 0.5 * (positives.mean().item() + negatives.mean().item())
    predictions = scores >= threshold
    threshold_accuracy = predictions.eq(labels).float().mean().item()

    positive_mean = positives.mean().item()
    negative_mean = negatives.mean().item()
    return FeatureDetectionReport(
        auc=roc_auc_binary(scores, labels),
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        separation=positive_mean - negative_mean,
        threshold_accuracy=threshold_accuracy,
    )


def top_activating_examples(
    feature_acts: t.Tensor,
    feature_id: int,
    k: int = 10,
) -> tuple[t.Tensor, t.Tensor]:
    """Return flattened indices and values for the top activations of one feature."""

    if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")
    scores = feature_acts[..., feature_id].flatten()
    k = min(k, scores.numel())
    values, indices = scores.topk(k)
    return indices, values


def ablate_features(
    feature_acts: t.Tensor,
    feature_ids: t.Tensor | list[int],
    replacement: Literal["zero", "mean"] = "zero",
) -> t.Tensor:
    """Ablate selected feature activations."""

    ablated = feature_acts.clone()
    feature_ids_tensor = t.as_tensor(feature_ids, device=feature_acts.device, dtype=t.long)
    if replacement == "zero":
        ablated[..., feature_ids_tensor] = 0
    elif replacement == "mean":
        means = feature_acts[..., feature_ids_tensor].mean(
            dim=tuple(range(feature_acts.ndim - 1)),
            keepdim=True,
        )
        ablated[..., feature_ids_tensor] = means
    else:
        raise ValueError("replacement must be 'zero' or 'mean'.")
    return ablated


def apply_decoder_steering(
    activations: t.Tensor,
    decoder_vectors: t.Tensor,
    feature_ids: t.Tensor | list[int],
    coefficients: t.Tensor | list[float] | float,
    *,
    positions: Literal["all", "last"] = "last",
) -> t.Tensor:
    """Add decoder-vector steering directions to activations."""

    feature_ids_tensor = t.as_tensor(feature_ids, device=activations.device, dtype=t.long)
    selected = decoder_vectors.to(
        device=activations.device,
        dtype=activations.dtype,
    )[feature_ids_tensor]
    coeffs = t.as_tensor(coefficients, device=activations.device, dtype=activations.dtype)
    if coeffs.ndim == 0:
        coeffs = coeffs.expand(selected.shape[0])
    if coeffs.numel() != selected.shape[0]:
        raise ValueError("coefficients must be scalar or match number of feature_ids.")
    steering_vector = (coeffs[:, None] * selected).sum(dim=0)

    steered = activations.clone()
    if positions == "all":
        steered = steered + steering_vector
    elif positions == "last":
        steered[..., -1, :] = steered[..., -1, :] + steering_vector
    else:
        raise ValueError("positions must be 'all' or 'last'.")
    return steered


def steering_comparison_report(
    baseline_scores: t.Tensor,
    steered_scores: t.Tensor,
    random_control_scores: t.Tensor,
) -> SteeringComparisonReport:
    """Compare target-score change from feature steering against random-feature steering."""

    baseline_mean = baseline_scores.float().mean().item()
    steered_mean = steered_scores.float().mean().item()
    random_mean = random_control_scores.float().mean().item()
    steered_delta = steered_mean - baseline_mean
    random_delta = random_mean - baseline_mean
    return SteeringComparisonReport(
        baseline_mean=baseline_mean,
        steered_mean=steered_mean,
        random_mean=random_mean,
        steered_delta=steered_delta,
        random_delta=random_delta,
        passes_control=abs(steered_delta) > abs(random_delta),
    )
