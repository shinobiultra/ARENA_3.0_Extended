"""Finetuning, LoRA, DoRA, and adapter-interpretability utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t


@dataclass(frozen=True)
class AdapterDeltaReport:
    rank: int
    alpha: float
    update_norm: float
    nonzero_update: bool


@dataclass(frozen=True)
class DoRAWeightReport:
    target_norms: t.Tensor
    row_norms: t.Tensor
    max_norm_error: float
    norm_preserved: bool


@dataclass(frozen=True)
class IntruderDimensionReport:
    projection_fraction: float
    intruder_detected: bool


@dataclass(frozen=True)
class AdapterMechanismReport:
    accuracy_delta: float
    mechanism_delta: float
    accuracy_improved: bool
    mechanism_preserved: bool
    adapter_acceptable: bool


def lora_delta(lora_a: t.Tensor, lora_b: t.Tensor, *, alpha: float = 1.0) -> t.Tensor:
    """Return the LoRA weight delta using A=(rank,in), B=(out,rank)."""

    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("lora_a and lora_b must both be matrices.")
    rank = lora_a.shape[0]
    if rank == 0:
        raise ValueError("LoRA rank must be positive.")
    if lora_b.shape[1] != rank:
        raise ValueError("lora_b second dimension must match LoRA rank.")
    return (alpha / rank) * (lora_b.float() @ lora_a.float())


def adapter_delta_report(
    lora_a: t.Tensor,
    lora_b: t.Tensor,
    *,
    alpha: float = 1.0,
    min_update_norm: float = 1e-6,
) -> AdapterDeltaReport:
    """Summarize the rank and norm of a LoRA update."""

    delta = lora_delta(lora_a, lora_b, alpha=alpha)
    update_norm = delta.norm().item()
    return AdapterDeltaReport(
        rank=lora_a.shape[0],
        alpha=alpha,
        update_norm=update_norm,
        nonzero_update=update_norm >= min_update_norm,
    )


def dora_recompose_weight(
    base_weight: t.Tensor,
    adapter_delta: t.Tensor,
    magnitude: t.Tensor,
    *,
    eps: float = 1e-8,
) -> t.Tensor:
    """Return a DoRA-style recomposed weight with learned row magnitudes."""

    if base_weight.shape != adapter_delta.shape:
        raise ValueError("base_weight and adapter_delta must match.")
    if magnitude.ndim != 1 or magnitude.shape[0] != base_weight.shape[0]:
        raise ValueError("magnitude must have shape (out_features,).")

    direction = base_weight.float() + adapter_delta.float()
    unit_direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    return unit_direction * magnitude.float().unsqueeze(-1)


def dora_weight_report(
    base_weight: t.Tensor,
    adapter_delta: t.Tensor,
    magnitude: t.Tensor,
    *,
    max_allowed_norm_error: float = 1e-5,
) -> DoRAWeightReport:
    """Check whether DoRA recomposition preserves target row magnitudes."""

    recomposed = dora_recompose_weight(base_weight, adapter_delta, magnitude)
    row_norms = recomposed.norm(dim=-1)
    norm_errors = (row_norms - magnitude.float()).abs()
    max_norm_error = norm_errors.max().item()
    return DoRAWeightReport(
        target_norms=magnitude.float(),
        row_norms=row_norms,
        max_norm_error=max_norm_error,
        norm_preserved=max_norm_error <= max_allowed_norm_error,
    )


def intruder_dimension_report(
    adapter_delta: t.Tensor,
    protected_direction: t.Tensor,
    *,
    max_projection_fraction: float = 0.2,
) -> IntruderDimensionReport:
    """Check whether an adapter update projects onto a protected direction."""

    if adapter_delta.ndim != 2:
        raise ValueError("adapter_delta must have shape (out_features, in_features).")
    if protected_direction.ndim != 1:
        raise ValueError("protected_direction must have shape (in_features,).")
    if adapter_delta.shape[-1] != protected_direction.shape[0]:
        raise ValueError("adapter and protected direction dimensions must match.")

    direction = protected_direction.float()
    direction = direction / direction.norm().clamp_min(1e-8)
    projected = adapter_delta.float() @ direction
    adapter_norm = adapter_delta.float().norm().clamp_min(1e-8).item()
    projection_fraction = projected.norm().item() / adapter_norm
    return IntruderDimensionReport(
        projection_fraction=projection_fraction,
        intruder_detected=projection_fraction > max_projection_fraction,
    )


def adapter_mechanism_report(
    *,
    adapter_accuracy: float,
    baseline_accuracy: float,
    adapter_mechanism_score: float,
    baseline_mechanism_score: float,
    min_accuracy_gain: float = 0.05,
    min_mechanism_delta: float = -0.02,
) -> AdapterMechanismReport:
    """Check that an adapter improves accuracy without breaking the mechanism."""

    accuracy_delta = adapter_accuracy - baseline_accuracy
    mechanism_delta = adapter_mechanism_score - baseline_mechanism_score
    accuracy_improved = accuracy_delta >= min_accuracy_gain
    mechanism_preserved = mechanism_delta >= min_mechanism_delta
    return AdapterMechanismReport(
        accuracy_delta=accuracy_delta,
        mechanism_delta=mechanism_delta,
        accuracy_improved=accuracy_improved,
        mechanism_preserved=mechanism_preserved,
        adapter_acceptable=accuracy_improved and mechanism_preserved,
    )
