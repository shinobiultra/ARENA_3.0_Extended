"""Toy SAE variant utilities for sparse feature method notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
import torch.nn.functional as F

from arena_ext.features import (
    SAEReconstructionMetrics,
    compute_sae_reconstruction_metrics,
    feature_density,
    l0,
    roc_auc_binary,
)


@dataclass(frozen=True)
class ToySuperpositionBatch:
    feature_acts: t.Tensor
    activations: t.Tensor
    dictionary: t.Tensor


@dataclass(frozen=True)
class SAEVariantMetrics:
    name: str
    l0: float
    feature_density_mean: float
    dead_feature_fraction: float
    reconstruction_mse: float


@dataclass(frozen=True)
class DictionaryRecoveryReport:
    mean_best_cosine: float
    recovered_fraction: float
    duplicate_fraction: float
    best_learned_for_true: t.Tensor


@dataclass(frozen=True)
class FeatureAUCReport:
    feature_id: int
    auc: float


def make_toy_superposition_batch(
    *,
    batch: int = 256,
    n_features: int = 8,
    d_model: int = 4,
    feature_probability: float = 0.2,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> ToySuperpositionBatch:
    """Create sparse planted features mixed through a random dictionary."""

    if batch <= 0 or n_features <= 0 or d_model <= 0:
        raise ValueError("batch, n_features, and d_model must be positive.")
    if not 0 <= feature_probability <= 1:
        raise ValueError("feature_probability must be in [0, 1].")
    if noise_scale < 0:
        raise ValueError("noise_scale must be nonnegative.")

    generator = t.Generator().manual_seed(seed)
    dictionary = F.normalize(t.randn(n_features, d_model, generator=generator), dim=-1)
    active = t.rand(batch, n_features, generator=generator) < feature_probability
    magnitudes = 0.5 + t.rand(batch, n_features, generator=generator)
    feature_acts = active.float() * magnitudes
    activations = feature_acts @ dictionary
    if noise_scale:
        activations = activations + noise_scale * t.randn(
            activations.shape,
            generator=generator,
        )
    return ToySuperpositionBatch(
        feature_acts=feature_acts,
        activations=activations,
        dictionary=dictionary,
    )


def relu_l1_encode(pre_acts: t.Tensor, *, l1_coefficient: float = 0.0) -> t.Tensor:
    """ReLU encoder with a simple soft-threshold standing in for L1 pressure."""

    if l1_coefficient < 0:
        raise ValueError("l1_coefficient must be nonnegative.")
    return (pre_acts - l1_coefficient).clamp_min(0)


def topk_encode(pre_acts: t.Tensor, *, k: int) -> t.Tensor:
    """Keep the top-k nonnegative feature activations per example."""

    if pre_acts.ndim < 1:
        raise ValueError("pre_acts must have at least one dimension.")
    if k < 0 or k > pre_acts.shape[-1]:
        raise ValueError("k must be between 0 and the number of features.")
    relu_acts = pre_acts.clamp_min(0)
    if k == 0:
        return t.zeros_like(relu_acts)
    values, indices = relu_acts.topk(k=k, dim=-1)
    encoded = t.zeros_like(relu_acts)
    return encoded.scatter(-1, indices, values)


def gated_encode(
    pre_acts: t.Tensor,
    gate_logits: t.Tensor,
    *,
    gate_threshold: float = 0.0,
) -> t.Tensor:
    """Gated SAE-style encoder decoupling detection from magnitude."""

    if pre_acts.shape != gate_logits.shape:
        raise ValueError("pre_acts and gate_logits must have matching shapes.")
    return pre_acts.clamp_min(0) * gate_logits.gt(gate_threshold)


def jumprelu_encode(pre_acts: t.Tensor, *, threshold: float) -> t.Tensor:
    """JumpReLU encoder: activations fire only after crossing a threshold."""

    return t.where(pre_acts > threshold, pre_acts, t.zeros_like(pre_acts))


def decode_features(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor | None = None,
) -> t.Tensor:
    """Decode sparse feature activations back into model activations."""

    if feature_acts.shape[-1] != decoder_weight.shape[0]:
        raise ValueError("feature dimension must match decoder rows.")
    reconstructed = feature_acts.float() @ decoder_weight.float()
    if decoder_bias is not None:
        reconstructed = reconstructed + decoder_bias.to(reconstructed.device)
    return reconstructed


def sae_variant_metrics(
    name: str,
    *,
    activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    feature_acts: t.Tensor,
) -> SAEVariantMetrics:
    """Compact metrics for comparing SAE variants."""

    metrics: SAEReconstructionMetrics = compute_sae_reconstruction_metrics(
        activations=activations,
        reconstructed_activations=reconstructed_activations,
        feature_acts=feature_acts,
    )
    return SAEVariantMetrics(
        name=name,
        l0=metrics.l0,
        feature_density_mean=metrics.feature_density_mean,
        dead_feature_fraction=metrics.dead_feature_fraction,
        reconstruction_mse=metrics.reconstruction_mse,
    )


def dictionary_recovery_report(
    learned_decoder: t.Tensor,
    true_dictionary: t.Tensor,
    *,
    threshold: float = 0.8,
) -> DictionaryRecoveryReport:
    """Check whether learned decoder directions recover planted feature directions."""

    if learned_decoder.ndim != 2 or true_dictionary.ndim != 2:
        raise ValueError("learned_decoder and true_dictionary must be rank-2 tensors.")
    if learned_decoder.shape[1] != true_dictionary.shape[1]:
        raise ValueError("decoder and dictionary dimensions must match.")
    learned = F.normalize(learned_decoder.float(), dim=-1)
    true = F.normalize(true_dictionary.float(), dim=-1)
    cosine = learned @ true.T
    best_cosine, best_learned = cosine.max(dim=0)
    recovered = best_cosine >= threshold

    best_true_for_learned = cosine.argmax(dim=-1)
    unique_best = best_true_for_learned.unique().numel()
    duplicate_fraction = 1.0 - unique_best / learned_decoder.shape[0]

    return DictionaryRecoveryReport(
        mean_best_cosine=best_cosine.mean().item(),
        recovered_fraction=recovered.float().mean().item(),
        duplicate_fraction=duplicate_fraction,
        best_learned_for_true=best_learned,
    )


def best_feature_auc(feature_acts: t.Tensor, labels: t.Tensor) -> FeatureAUCReport:
    """Return the feature with the strongest binary-label separation."""

    if feature_acts.ndim != 2:
        raise ValueError("feature_acts must have shape (examples, features).")
    if labels.shape != (feature_acts.shape[0],):
        raise ValueError("labels must have shape (examples,).")

    best_id = 0
    best_auc = -1.0
    for feature_id in range(feature_acts.shape[1]):
        auc = roc_auc_binary(feature_acts[:, feature_id], labels)
        score = max(auc, 1.0 - auc)
        if score > best_auc:
            best_id = feature_id
            best_auc = score
    return FeatureAUCReport(feature_id=best_id, auc=best_auc)


def density_is_nondegenerate(
    feature_acts: t.Tensor,
    *,
    min_active_fraction: float = 0.05,
    max_active_fraction: float = 0.95,
) -> bool:
    """Check that some features fire, but not all features fire everywhere."""

    densities = feature_density(feature_acts)
    return bool(
        densities.gt(min_active_fraction).any()
        and densities.lt(max_active_fraction).any()
        and l0(feature_acts) > 0
    )
