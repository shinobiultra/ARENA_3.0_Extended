"""Attribution patching and EAP utilities for automated circuit notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t


@dataclass(frozen=True)
class ScoreCorrelationReport:
    correlation: float
    passes_threshold: bool


@dataclass(frozen=True)
class TopKOverlapReport:
    exact_top_indices: tuple[int, ...]
    approx_top_indices: tuple[int, ...]
    topk_overlap: float
    passes_threshold: bool


@dataclass(frozen=True)
class RuntimeImprovementReport:
    exact_runtime_s: float
    approx_runtime_s: float
    speedup: float
    passes_speedup: bool


@dataclass(frozen=True)
class FalseNegativeReport:
    false_negative_indices: tuple[int, ...]
    num_false_negatives: int
    documented: bool


def _require_finite_tensor(tensor: t.Tensor, name: str) -> None:
    if not t.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite.")


def _require_finite_scalar(value: float, name: str) -> None:
    if not t.isfinite(t.tensor(value)).item():
        raise ValueError(f"{name} must be finite.")


def attribution_patch_scores(
    clean_activations: t.Tensor,
    corrupt_activations: t.Tensor,
    corrupt_gradients: t.Tensor,
    *,
    component_dim: int = 0,
) -> t.Tensor:
    """Approximate patching effects with (clean - corrupt) dot gradient."""

    if clean_activations.shape != corrupt_activations.shape:
        raise ValueError("clean and corrupt activations must have matching shape.")
    if clean_activations.shape != corrupt_gradients.shape:
        raise ValueError("gradients must match activation shape.")
    if not 0 <= component_dim < clean_activations.ndim:
        raise ValueError("component_dim is out of range.")
    _require_finite_tensor(clean_activations, "clean_activations")
    _require_finite_tensor(corrupt_activations, "corrupt_activations")
    _require_finite_tensor(corrupt_gradients, "corrupt_gradients")
    contribution = clean_activations.float() - corrupt_activations.float()
    contribution = contribution * corrupt_gradients.float()
    reduce_dims = tuple(dim for dim in range(contribution.ndim) if dim != component_dim)
    return contribution.sum(dim=reduce_dims)


def integrated_gradient_patch_scores(
    clean_activations: t.Tensor,
    corrupt_activations: t.Tensor,
    path_gradients: t.Tensor,
    *,
    component_dim: int = 0,
) -> t.Tensor:
    """Approximate patching effects using averaged path gradients."""

    if path_gradients.ndim != clean_activations.ndim + 1:
        raise ValueError("path_gradients must have shape (steps, *activation_shape).")
    if path_gradients.shape[0] == 0:
        raise ValueError("path_gradients must contain at least one step.")
    if path_gradients.shape[1:] != clean_activations.shape:
        raise ValueError("path gradient activation dimensions must match activations.")
    _require_finite_tensor(path_gradients, "path_gradients")
    mean_gradient = path_gradients.float().mean(dim=0)
    return attribution_patch_scores(
        clean_activations,
        corrupt_activations,
        mean_gradient,
        component_dim=component_dim,
    )


def edge_attribution_scores(
    upstream_activation_delta: t.Tensor,
    downstream_gradients: t.Tensor,
) -> t.Tensor:
    """Return EAP-style upstream-by-downstream edge attribution scores."""

    if upstream_activation_delta.ndim != 2 or downstream_gradients.ndim != 2:
        raise ValueError("inputs must have shape (components, d_model).")
    if upstream_activation_delta.shape[0] == 0 or downstream_gradients.shape[0] == 0:
        raise ValueError("inputs must contain at least one component.")
    if upstream_activation_delta.shape[-1] == 0:
        raise ValueError("hidden dimension must be nonempty.")
    if upstream_activation_delta.shape[-1] != downstream_gradients.shape[-1]:
        raise ValueError("upstream and downstream hidden dimensions must match.")
    _require_finite_tensor(upstream_activation_delta, "upstream_activation_delta")
    _require_finite_tensor(downstream_gradients, "downstream_gradients")
    return upstream_activation_delta.float() @ downstream_gradients.float().T


def score_correlation_report(
    exact_scores: t.Tensor,
    approx_scores: t.Tensor,
    *,
    min_correlation: float = 0.8,
) -> ScoreCorrelationReport:
    """Report Pearson correlation between exact and approximate patch scores."""

    exact = exact_scores.flatten().float()
    approx = approx_scores.flatten().float()
    if not -1.0 <= min_correlation <= 1.0:
        raise ValueError("min_correlation must be between -1 and 1.")
    if exact.shape != approx.shape:
        raise ValueError("exact_scores and approx_scores must have matching shape.")
    if exact.numel() < 2:
        raise ValueError("at least two scores are required for correlation.")
    _require_finite_tensor(exact, "exact_scores")
    _require_finite_tensor(approx, "approx_scores")
    exact_centered = exact - exact.mean()
    approx_centered = approx - approx.mean()
    denominator = exact_centered.norm() * approx_centered.norm()
    if denominator.item() == 0:
        correlation = 0.0
    else:
        correlation = float((exact_centered @ approx_centered / denominator).item())
    return ScoreCorrelationReport(
        correlation=correlation,
        passes_threshold=correlation >= min_correlation,
    )


def topk_overlap_report(
    exact_scores: t.Tensor,
    approx_scores: t.Tensor,
    *,
    top_k: int = 3,
    min_overlap: float = 0.5,
) -> TopKOverlapReport:
    """Report top-k overlap between exact and approximate patch scores."""

    exact = exact_scores.flatten().float()
    approx = approx_scores.flatten().float()
    if exact.shape != approx.shape:
        raise ValueError("exact_scores and approx_scores must have matching shape.")
    if exact.numel() == 0:
        raise ValueError("score tensors must be nonempty.")
    _require_finite_tensor(exact, "exact_scores")
    _require_finite_tensor(approx, "approx_scores")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1.")
    k = min(top_k, exact.numel())
    exact_top = tuple(int(index) for index in exact.topk(k=k).indices.tolist())
    approx_top = tuple(int(index) for index in approx.topk(k=k).indices.tolist())
    overlap = len(set(exact_top) & set(approx_top)) / k
    return TopKOverlapReport(
        exact_top_indices=exact_top,
        approx_top_indices=approx_top,
        topk_overlap=overlap,
        passes_threshold=overlap >= min_overlap,
    )


def runtime_improvement_report(
    *,
    exact_runtime_s: float,
    approx_runtime_s: float,
    min_speedup: float = 2.0,
) -> RuntimeImprovementReport:
    """Report exact-patching runtime divided by approximate-patching runtime."""

    _require_finite_scalar(exact_runtime_s, "exact_runtime_s")
    _require_finite_scalar(approx_runtime_s, "approx_runtime_s")
    _require_finite_scalar(min_speedup, "min_speedup")
    if exact_runtime_s <= 0 or approx_runtime_s <= 0:
        raise ValueError("runtimes must be positive.")
    if min_speedup <= 0:
        raise ValueError("min_speedup must be positive.")
    speedup = exact_runtime_s / approx_runtime_s
    return RuntimeImprovementReport(
        exact_runtime_s=exact_runtime_s,
        approx_runtime_s=approx_runtime_s,
        speedup=speedup,
        passes_speedup=speedup >= min_speedup,
    )


def false_negative_report(
    exact_scores: t.Tensor,
    approx_scores: t.Tensor,
    *,
    exact_threshold: float,
    approx_threshold: float,
    documentation: dict[int, str] | None = None,
) -> FalseNegativeReport:
    """Find exact-important components missed by approximate scores."""

    exact = exact_scores.flatten().float()
    approx = approx_scores.flatten().float()
    if exact.shape != approx.shape:
        raise ValueError("exact_scores and approx_scores must have matching shape.")
    _require_finite_tensor(exact, "exact_scores")
    _require_finite_tensor(approx, "approx_scores")
    _require_finite_scalar(exact_threshold, "exact_threshold")
    _require_finite_scalar(approx_threshold, "approx_threshold")
    important = exact >= exact_threshold
    missed = approx < approx_threshold
    false_negative_indices = (important & missed).nonzero(as_tuple=False).flatten()
    indices = tuple(int(index.item()) for index in false_negative_indices)
    documentation = documentation or {}
    documented = all(bool(documentation.get(index, "").strip()) for index in indices)
    return FalseNegativeReport(
        false_negative_indices=indices,
        num_false_negatives=len(indices),
        documented=documented,
    )
