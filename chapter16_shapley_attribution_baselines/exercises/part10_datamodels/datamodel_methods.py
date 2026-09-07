"""Readable reference functions for the student-implemented Datamodels method."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sample_subsets(n: int, size: int, count: int, seed: int) -> Tensor:
    """Uniform fixed-size membership masks, one independent permutation per row."""
    if not 0 < size < n or count < 1:
        raise ValueError("Use a nonempty proper subset and a positive sample count.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    masks = torch.zeros(count, n, dtype=torch.bool)
    for row in range(count):
        included = torch.randperm(n, generator=generator)[:size]
        masks[row, included] = True
    return masks


def correct_class_margins(logits: Tensor, labels: Tensor) -> Tensor:
    """True-class logit minus the largest other logit; preserve leading axes."""
    if logits.ndim < 2 or logits.shape[-1] < 2 or labels.shape != (logits.shape[-2],):
        raise ValueError("Expected logits [..., queries, classes] and labels [queries].")
    if labels.dtype != torch.int64 or (labels < 0).any() or (labels >= logits.shape[-1]).any():
        raise ValueError("Labels must index valid classes.")
    index = labels.expand(*logits.shape[:-1]).unsqueeze(-1)
    correct = logits.gather(-1, index).squeeze(-1)
    incorrect = logits.scatter(-1, index, -torch.inf).amax(-1)
    return correct - incorrect


def soft_threshold(values: Tensor, threshold: Tensor | float) -> Tensor:
    """Proximal map for a nonnegative elementwise L1 penalty."""
    threshold = torch.as_tensor(threshold, dtype=values.dtype, device=values.device)
    if not torch.isfinite(threshold).all() or (threshold < 0).any():
        raise ValueError("Thresholds must be finite and nonnegative.")
    return values.sign() * (values.abs() - threshold).clamp_min(0)


def lasso_objective(x: Tensor, y: Tensor, weights: Tensor, bias: Tensor, penalty: Tensor) -> Tensor:
    """One objective per target: mean squared error / 2 plus lambda * L1."""
    residual = x @ weights + bias - y
    return residual.square().mean(0) / 2 + penalty * weights.abs().sum(0)


def lasso_kkt(gram: Tensor, cross: Tensor, weights: Tensor, penalty: Tensor) -> Tensor:
    """Maximum stationarity violation per target for centered Lasso."""
    gradient = gram @ weights - cross
    active = (gradient + penalty * weights.sign()).abs()
    inactive = (gradient.abs() - penalty).clamp_min(0)
    return torch.where(weights != 0, active, inactive).amax(0)


def fit_lasso(
    x: Tensor,
    y: Tensor,
    penalty: Tensor,
    initial: Tensor | None = None,
    tolerance: float = 1e-8,
    max_steps: int = 20000,
) -> dict[str, Tensor | int]:
    """Accelerated proximal gradient with target-wise adaptive restart.

    Center using training means only. The Gram matrix is shared by all targets;
    each target still has its own coefficients, penalty and acceleration state.
    A checked KKT residual, not an iteration count, certifies convergence.
    """
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 1:
        raise ValueError("Expected x [models, examples] and y [models, targets].")
    if penalty.shape != (y.shape[1],) or (penalty < 0).any() or not torch.isfinite(penalty).all():
        raise ValueError("Provide one finite nonnegative penalty per target.")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Regression inputs must be finite.")
    if not math.isfinite(tolerance) or tolerance <= 0 or max_steps < 1:
        raise ValueError("Use a positive finite tolerance and positive step limit.")
    x_mean, y_mean = x.mean(0), y.mean(0)
    xc, yc = x - x_mean, y - y_mean
    gram, cross = xc.T @ xc / len(x), xc.T @ yc / len(x)
    lipschitz = torch.linalg.eigvalsh(gram)[-1]
    shape = (x.shape[1], y.shape[1])
    weights = x.new_zeros(shape) if initial is None else initial.clone()
    if weights.shape != shape or not torch.isfinite(weights).all():
        raise ValueError("Initial coefficients have the wrong shape or nonfinite entries.")
    if lipschitz <= torch.finfo(x.dtype).eps:
        # Constant columns convey no information once the intercept is fitted.
        weights.zero_()
        residual = lasso_kkt(gram, cross, weights, penalty)
        return dict(
            weights=weights,
            bias=y_mean,
            kkt=residual,
            steps=0,
            history=torch.cat([x.new_zeros(1), residual.max().reshape(1)])[None],
        )
    extrapolated = weights.clone()
    momentum_time = x.new_ones(y.shape[1])
    history = []
    for step in range(1, max_steps + 1):
        gradient = gram @ extrapolated - cross
        updated = soft_threshold(extrapolated - gradient / lipschitz, penalty / lipschitz)
        next_time = (1 + torch.sqrt(1 + 4 * momentum_time.square())) / 2
        restart = ((extrapolated - updated) * (updated - weights)).sum(0) > 0
        acceleration = torch.where(restart, 0, (momentum_time - 1) / next_time)
        extrapolated = updated + acceleration * (updated - weights)
        momentum_time = torch.where(restart, 1, next_time)
        weights = updated
        if step == 1 or step % 10 == 0 or step == max_steps:
            residual = lasso_kkt(gram, cross, weights, penalty)
            history.append(torch.stack([x.new_tensor(step), residual.max()]))
            if (residual <= tolerance).all():
                return dict(
                    weights=weights,
                    bias=y_mean - x_mean @ weights,
                    kkt=residual,
                    steps=step,
                    history=torch.stack(history),
                )
    raise RuntimeError(f"Lasso did not converge: maximum KKT residual {residual.max().item():.3g}.")


def fit_datamodel_path(
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    fractions: Tensor,
) -> dict[str, Tensor]:
    """Fit every path point on train rows, then select per target on validation."""
    if fractions.ndim != 1 or len(fractions) < 1 or not torch.isfinite(fractions).all():
        raise ValueError("Use a nonempty finite path of penalty fractions.")
    if (fractions <= 0).any() or (fractions > 1).any() or (fractions[1:] > fractions[:-1]).any():
        raise ValueError("Penalty fractions must descend within (0, 1].")
    if x_val.shape[1:] != x_train.shape[1:] or y_val.shape != (len(x_val), y_train.shape[1]):
        raise ValueError("Validation masks and targets must align with the training axes.")
    xc, yc = x_train - x_train.mean(0), y_train - y_train.mean(0)
    lambda_max = (xc.T @ yc / len(x_train)).abs().amax(0)
    weights, biases, residuals, steps, validation, penalties = [], [], [], [], [], []
    initial = None
    for fraction in fractions:
        penalty = fraction * lambda_max
        fit = fit_lasso(x_train, y_train, penalty, initial=initial)
        initial = fit["weights"]
        weights.append(fit["weights"])
        biases.append(fit["bias"])
        residuals.append(fit["kkt"])
        steps.append(fit["steps"])
        penalties.append(penalty)
        prediction = x_val @ fit["weights"] + fit["bias"]
        validation.append((prediction - y_val).square().mean(0))
    weights, biases = torch.stack(weights), torch.stack(biases)
    validation = torch.stack(validation)
    selected = validation.argmin(0)  # First path index wins exact ties.
    targets = torch.arange(y_train.shape[1], device=x_train.device)
    return dict(
        weights=weights,
        bias=biases,
        penalty=torch.stack(penalties),
        kkt=torch.stack(residuals),
        steps=torch.tensor(steps),
        validation_mse=validation,
        selected=selected,
        selected_weights=weights[selected, :, targets].T,
        selected_bias=biases[selected, targets],
    )


def select_supportive(scores: Tensor, budget: int) -> Tensor:
    """Indices with largest scores, breaking ties by the original source order."""
    if scores.ndim != 1 or not torch.isfinite(scores).all() or not 0 <= budget < len(scores):
        raise ValueError("Provide finite scores and a budget leaving at least one example.")
    return torch.argsort(scores, descending=True, stable=True)[:budget].sort().values


def matched_random_removals(labels: Tensor, selected: Tensor, count: int, seed: int) -> Tensor:
    """Random index sets matching the selected set's count in every class."""
    if labels.ndim != 1 or labels.dtype != torch.int64 or selected.dtype != torch.int64:
        raise ValueError("Labels and selected indices must be integer vectors.")
    if selected.ndim != 1 or len(selected.unique()) != len(selected) or count < 1:
        raise ValueError("Use distinct selected indices and a positive control count.")
    if (selected < 0).any() or (selected >= len(labels)).any():
        raise ValueError("Selected indices are outside the training pool.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sets = []
    for _ in range(count):
        pieces = []
        for label in labels.unique(sorted=True):
            candidates = torch.nonzero(labels == label).flatten()
            number = int((labels[selected] == label).sum())
            permutation = torch.randperm(len(candidates), generator=generator)
            pieces.append(candidates[permutation[:number]])
        sets.append(torch.cat(pieces).sort().values)
    return torch.stack(sets)


def centered_weight_similarity(weights: Tensor) -> Tensor:
    """Cosine similarity of target datamodels, invariant to target-wise offsets."""
    centered = weights - weights.mean(0, keepdim=True)
    norms = centered.norm(dim=0)
    normalized = centered / norms.clamp_min(torch.finfo(weights.dtype).eps)
    return normalized.T @ normalized
