"""Activation patching utilities for automated circuit notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t


@dataclass(frozen=True)
class PatchingRecoveryReport:
    clean_metric: float
    corrupt_metric: float
    patched_metric: float
    recovered_fraction: float
    passes_recovery: bool


@dataclass(frozen=True)
class ActivationPatchingSweep:
    patch_scores: t.Tensor
    best_index: int
    best_score: float


@dataclass(frozen=True)
class PatchingLocalizationReport:
    top_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    topk_overlap: float
    localizes_target: bool


@dataclass(frozen=True)
class RandomPatchControlReport:
    top_patch_score: float
    random_patch_score: float
    max_random_patch_score: float
    top_beats_random: bool
    top_beats_max_random: bool


def answer_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    """Return mean positive-minus-negative logit difference."""

    if logits.ndim < 1:
        raise ValueError("logits must have a vocabulary dimension.")
    vocab_size = logits.shape[-1]
    if vocab_size == 0:
        raise ValueError("logits vocabulary dimension must be nonempty.")
    if positive_token_id == negative_token_id:
        raise ValueError("positive_token_id and negative_token_id must differ.")
    if not 0 <= positive_token_id < vocab_size:
        raise ValueError("positive_token_id is out of range.")
    if not 0 <= negative_token_id < vocab_size:
        raise ValueError("negative_token_id is out of range.")
    if not t.isfinite(logits).all():
        raise ValueError("logits must be finite.")
    diff = logits[..., positive_token_id] - logits[..., negative_token_id]
    return diff.float().mean().item()


def patch_activation_slice(
    clean_activations: t.Tensor,
    corrupt_activations: t.Tensor,
    *,
    component_index: int,
    component_dim: int = 0,
) -> t.Tensor:
    """Patch one clean activation slice into the corrupt activation tensor."""

    if clean_activations.shape != corrupt_activations.shape:
        raise ValueError("clean and corrupt activations must have matching shape.")
    if not 0 <= component_dim < clean_activations.ndim:
        raise ValueError("component_dim is out of range.")
    if not 0 <= component_index < clean_activations.shape[component_dim]:
        raise ValueError("component_index is out of range.")

    patched = corrupt_activations.clone()
    slices = [slice(None)] * clean_activations.ndim
    slices[component_dim] = component_index
    patched[tuple(slices)] = clean_activations[tuple(slices)]
    return patched


def recovery_fraction(
    *,
    clean_metric: float,
    corrupt_metric: float,
    patched_metric: float,
) -> float:
    """Return how much of the clean-corrupt gap a patch recovers."""

    denominator = clean_metric - corrupt_metric
    metrics_are_finite = all(
        t.isfinite(t.tensor(value)).item()
        for value in (clean_metric, corrupt_metric, patched_metric)
    )
    if not metrics_are_finite:
        raise ValueError("clean, corrupt, and patched metrics must be finite.")
    if denominator == 0:
        raise ValueError("clean_metric and corrupt_metric must differ.")
    return (patched_metric - corrupt_metric) / denominator


def patching_recovery_report(
    clean_logits: t.Tensor,
    corrupt_logits: t.Tensor,
    patched_logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
    min_recovered_fraction: float = 0.5,
) -> PatchingRecoveryReport:
    """Measure logit-diff recovery after patching clean activations."""

    if not 0.0 <= min_recovered_fraction <= 1.0:
        raise ValueError("min_recovered_fraction must be between 0 and 1.")
    clean_metric = answer_logit_diff(
        clean_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    corrupt_metric = answer_logit_diff(
        corrupt_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    patched_metric = answer_logit_diff(
        patched_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    recovered = recovery_fraction(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        patched_metric=patched_metric,
    )
    return PatchingRecoveryReport(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        patched_metric=patched_metric,
        recovered_fraction=recovered,
        passes_recovery=recovered >= min_recovered_fraction,
    )


def activation_patching_sweep(
    *,
    clean_metric: float,
    corrupt_metric: float,
    patched_metrics: t.Tensor,
) -> ActivationPatchingSweep:
    """Convert per-component patched metrics into recovery scores."""

    if patched_metrics.ndim != 1:
        raise ValueError("patched_metrics must be rank-1.")
    if patched_metrics.numel() == 0:
        raise ValueError("patched_metrics must be nonempty.")
    if not t.isfinite(patched_metrics).all():
        raise ValueError("patched_metrics must be finite.")
    denominator = clean_metric - corrupt_metric
    metrics_are_finite = all(
        t.isfinite(t.tensor(value)).item()
        for value in (clean_metric, corrupt_metric)
    )
    if not metrics_are_finite:
        raise ValueError("clean_metric and corrupt_metric must be finite.")
    if denominator == 0:
        raise ValueError("clean_metric and corrupt_metric must differ.")
    patch_scores = (patched_metrics.float() - corrupt_metric) / denominator
    best_index = int(patch_scores.argmax().item())
    return ActivationPatchingSweep(
        patch_scores=patch_scores,
        best_index=best_index,
        best_score=float(patch_scores[best_index].item()),
    )


def patching_localization_report(
    patch_scores: t.Tensor,
    target_indices: list[int],
    *,
    top_k: int = 2,
    min_overlap: float = 0.5,
) -> PatchingLocalizationReport:
    """Check whether top patching scores recover known target components."""

    if patch_scores.ndim != 1:
        raise ValueError("patch_scores must be rank-1.")
    if patch_scores.numel() == 0:
        raise ValueError("patch_scores must be nonempty.")
    if not t.isfinite(patch_scores).all():
        raise ValueError("patch_scores must be finite.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1.")
    if len(target_indices) == 0:
        raise ValueError("target_indices must be nonempty.")
    target_tensor = t.tensor(target_indices, dtype=t.long, device=patch_scores.device)
    if target_tensor.min().item() < 0 or target_tensor.max().item() >= patch_scores.numel():
        raise ValueError("target index is out of range.")

    k = min(top_k, patch_scores.numel())
    top_indices = tuple(int(index) for index in patch_scores.topk(k=k).indices.tolist())
    target_tuple = tuple(int(index) for index in target_indices)
    top_set = set(top_indices)
    target_set = set(target_tuple)
    denominator = min(k, len(target_set))
    overlap = len(top_set & target_set) / denominator
    return PatchingLocalizationReport(
        top_indices=top_indices,
        target_indices=target_tuple,
        topk_overlap=overlap,
        localizes_target=overlap >= min_overlap,
    )


def random_patch_control_report(
    patch_scores: t.Tensor,
    random_indices: list[int],
    *,
    top_k: int = 2,
) -> RandomPatchControlReport:
    """Compare top patching score against a random-component control."""

    if patch_scores.ndim != 1:
        raise ValueError("patch_scores must be rank-1.")
    if patch_scores.numel() == 0:
        raise ValueError("patch_scores must be nonempty.")
    if not t.isfinite(patch_scores).all():
        raise ValueError("patch_scores must be finite.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if len(random_indices) == 0:
        raise ValueError("random_indices must be nonempty.")

    k = min(top_k, patch_scores.numel())
    top_patch_score = patch_scores.topk(k=k).values.mean().item()
    random_tensor = t.tensor(random_indices, dtype=t.long, device=patch_scores.device)
    if random_tensor.min().item() < 0 or random_tensor.max().item() >= patch_scores.numel():
        raise ValueError("random index is out of range.")
    random_scores = patch_scores[random_tensor].float()
    random_patch_score = random_scores.mean().item()
    max_random_patch_score = random_scores.max().item()
    return RandomPatchControlReport(
        top_patch_score=top_patch_score,
        random_patch_score=random_patch_score,
        max_random_patch_score=max_random_patch_score,
        top_beats_random=top_patch_score > random_patch_score,
        top_beats_max_random=top_patch_score > max_random_patch_score,
    )
