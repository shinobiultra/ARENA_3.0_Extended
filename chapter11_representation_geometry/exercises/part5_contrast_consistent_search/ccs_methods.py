"""Visible methods for the CCS learner under development.

Core fitting consumes paired activations, never labels. Semantic orientation is
a separate operation on a labelled calibration split. The loss is the original
Burns et al. objective, including its asymmetric confidence term.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _check_pairs(plus: Tensor, minus: Tensor) -> None:
    if plus.ndim != 2 or plus.shape != minus.shape or min(plus.shape) == 0:
        raise ValueError("Paired activations must have matching nonempty (examples, features) shapes.")
    if plus.dtype != minus.dtype or plus.device != minus.device:
        raise ValueError("Both branches must have the same dtype and device.")
    if not plus.is_floating_point() or not torch.isfinite(plus).all() or not torch.isfinite(minus).all():
        raise ValueError("Activations must be finite floating-point tensors.")


def fit_pair_centering(plus: Tensor, minus: Tensor) -> dict[str, Tensor]:
    """Fit separate branch means on training data; there is no scale normalization."""
    _check_pairs(plus, minus)
    return {"plus_mean": plus.detach().mean(0), "minus_mean": minus.detach().mean(0)}


def apply_pair_centering(
    plus: Tensor, minus: Tensor, centering: dict[str, Tensor]
) -> tuple[Tensor, Tensor]:
    """Apply fitted means to any split, without refitting on that split."""
    _check_pairs(plus, minus)
    if any(centering[key].shape != (plus.shape[1],) for key in ("plus_mean", "minus_mean")):
        raise ValueError("A training mean must have one value per feature.")
    return plus - centering["plus_mean"], minus - centering["minus_mean"]


def ccs_loss(p_plus: Tensor, p_minus: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Original loss, consistency and confidence, averaged over examples only.

    Inputs are (examples,) or (examples, independent_reporters). Returning one
    loss per reporter makes batching equivalent to fitting each independently.
    """
    if p_plus.ndim not in (1, 2) or p_plus.shape != p_minus.shape or p_plus.shape[0] == 0:
        raise ValueError("Probabilities must have matching nonempty shapes; broadcasting is forbidden.")
    if any(not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any() for p in (p_plus, p_minus)):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    consistency = (p_plus + p_minus - 1).square().mean(0)
    confidence = torch.minimum(p_plus, p_minus).square().mean(0)
    return consistency + confidence, consistency, confidence


def paired_probabilities(
    plus: Tensor, minus: Tensor, weight: Tensor, bias: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return averaged positive-answer scores and the two branch probabilities.

    A vector weight gives one reporter; (reporters, features) gives many.
    A score is not semantically oriented until a separate orientation is chosen.
    """
    _check_pairs(plus, minus)
    if weight.ndim not in (1, 2) or weight.shape[-1] != plus.shape[-1]:
        raise ValueError("Reporter weights must end in the activation feature dimension.")
    expected_bias = () if weight.ndim == 1 else (weight.shape[0],)
    if bias.shape != expected_bias:
        raise ValueError("Bias must be scalar for one reporter or one value per reporter.")
    p_plus = (plus @ weight.T + bias).sigmoid() if weight.ndim == 2 else (plus @ weight + bias).sigmoid()
    p_minus = (minus @ weight.T + bias).sigmoid() if weight.ndim == 2 else (minus @ weight + bias).sigmoid()
    return (p_plus + 1 - p_minus) / 2, p_plus, p_minus


def fit_ccs_restarts(
    plus: Tensor,
    minus: Tensor,
    *,
    seeds: Sequence[int] = tuple(range(10)),
    steps: int = 1000,
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
) -> dict[str, Tensor | list[int]]:
    """Fit independent affine reporters with AdamW and retain every restart.

    Inputs must already have training-fitted centering applied. Independent
    parameter rows are optimized together; sum, rather than mean, over their
    losses preserves each reporter's gradient. The final loss is recomputed
    after the final update. No labels or validation scores enter this function.
    """
    _check_pairs(plus, minus)
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one unique restart seed.")
    if steps < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Steps and learning rate must be positive; weight decay must be nonnegative.")
    plus, minus = plus.detach(), minus.detach()
    width = plus.shape[1]
    bound = 1 / math.sqrt(width)
    initial_weights, initial_biases = [], []
    for seed in seeds:
        generator = torch.Generator(device=plus.device).manual_seed(seed)
        initial_weights.append(
            torch.empty(width, dtype=plus.dtype, device=plus.device).uniform_(-bound, bound, generator=generator)
        )
        initial_biases.append(
            torch.empty((), dtype=plus.dtype, device=plus.device).uniform_(-bound, bound, generator=generator)
        )
    weight = torch.stack(initial_weights).requires_grad_()
    bias = torch.stack(initial_biases).requires_grad_()
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=weight_decay)
    history = []
    for step in range(steps + 1):
        _, p_plus, p_minus = paired_probabilities(plus, minus, weight, bias)
        loss, consistency, confidence = ccs_loss(p_plus, p_minus)
        history.append(torch.stack((loss, consistency, confidence), dim=-1).detach().cpu())
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
    return {
        "weights": weight.detach().clone(),
        "biases": bias.detach().clone(),
        "history": torch.stack(history),
        "seeds": seeds,
    }


def select_restart(fitted: dict) -> dict:
    """Choose the first minimum final training-loss restart, without labels."""
    history = fitted["history"]
    if history.ndim != 3 or history.shape[-1] != 3 or not torch.isfinite(history).all():
        raise ValueError("Expected finite (steps + 1, restarts, 3) loss history.")
    count = len(fitted["seeds"])
    if history.shape[1] != count or fitted["weights"].shape[0] != count or fitted["biases"].shape != (count,):
        raise ValueError("Restart parameters, seeds and history must agree.")
    index = int(history[-1, :, 0].argmin())
    return {
        "index": index,
        "seed": fitted["seeds"][index],
        "weight": fitted["weights"][index].clone(),
        "bias": fitted["biases"][index].clone(),
        "training_loss": float(history[-1, index, 0]),
    }


def _check_scores_labels(scores: Tensor, labels: Tensor) -> None:
    if scores.ndim != 1 or scores.shape != labels.shape or len(scores) == 0:
        raise ValueError("Provide equally sized nonempty score and label vectors.")
    if not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must lie in [0, 1].")
    if ((labels != 0) & (labels != 1)).any():
        raise ValueError("Labels must be binary.")


def calibrate_orientation(scores: Tensor, calibration_labels: Tensor) -> int:
    """Choose a global orientation using a separate labelled calibration split.

    A tie preserves +1. This is supervision for orientation, not CCS fitting.
    Exact 0.5 scores predict zero under either orientation.
    """
    _check_scores_labels(scores, calibration_labels)
    raw = ((scores > 0.5) == calibration_labels).sum()
    flipped = (((1 - scores) > 0.5) == calibration_labels).sum()
    return -1 if flipped > raw else 1


def binary_metrics(scores: Tensor, labels: Tensor, orientation: int = 1) -> dict[str, float | int]:
    """Evaluate a fixed orientation. The test-oracle score is explicitly separate."""
    _check_scores_labels(scores, labels)
    if orientation not in (-1, 1):
        raise ValueError("Orientation must be +1 or -1.")
    probability = scores if orientation == 1 else 1 - scores
    correct = ((probability > 0.5) == labels).sum().item()
    other_correct = (((1 - probability) > 0.5) == labels).sum().item()
    epsilon = max(1e-12, torch.finfo(probability.dtype).eps)
    clipped = probability.clamp(epsilon, 1 - epsilon)
    labels_float = labels.to(probability)
    cross_entropy = -(labels_float * clipped.log() + (1 - labels_float) * (1 - clipped).log()).mean()
    return {
        "count": len(scores),
        "correct": correct,
        "accuracy": correct / len(scores),
        "test_oracle_orientation_accuracy": max(correct, other_correct) / len(scores),
        "cross_entropy": float(cross_entropy),
        "brier": float((probability - labels_float).square().mean()),
    }


def fit_difference_pca(plus: Tensor, minus: Tensor) -> dict[str, Tensor]:
    """Fit a first-PC baseline to centred paired differences, without labels."""
    _check_pairs(plus, minus)
    differences = (plus - minus).detach()
    mean = differences.mean(0)
    _, singular, vh = torch.linalg.svd(differences - mean, full_matrices=False)
    if float(singular[0]) == 0:
        raise ValueError("The paired differences have no variance.")
    weight = vh[0]
    # Fix the numerical SVD sign; this is not a semantic orientation.
    weight = weight * weight[weight.abs().argmax()].sign()
    return {"weight": weight, "mean": mean, "singular_values": singular}
