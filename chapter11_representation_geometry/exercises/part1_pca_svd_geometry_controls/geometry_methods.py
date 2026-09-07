"""Transparent geometry primitives for the rewritten learner (development).

These bodies will appear in the notebook's own solution dropdowns. This module
is a reference for numerical tests, never the learner's hidden experiment.
"""
from __future__ import annotations

import torch as t


def fit_pca(x: t.Tensor, components: int):
    """Return training mean, orthonormal column basis and total-variance fractions."""
    if x.ndim != 2 or x.shape[0] < 2 or not 1 <= components <= min(x.shape):
        raise ValueError('PCA needs at least two rows and a feasible component count.')
    mean = x.mean(0)
    _, singular, vh = t.linalg.svd(x - mean, full_matrices=False)
    energy = singular.square()
    if energy.sum() <= 0:
        raise ValueError('Constant observations have no explained-variance ratio.')
    return mean, vh[:components].T, energy[:components] / energy.sum()


def center_template_grid(grid: t.Tensor):
    """Subtract each template's complete class-balanced batch mean.

This uses all examples in that batch. On a held-out template it is an explicit
transductive operation, not a preprocessing transform fitted on training data.
"""
    if grid.ndim != 3 or grid.shape[1] < 2:
        raise ValueError('Expected [templates, classes, width] with at least two classes.')
    return grid - grid.mean(1, keepdim=True)


def fit_ridge_coordinates(x: t.Tensor, targets: t.Tensor, ridge: float = 0.):
    """Affine least squares; intercept is not penalized, penalty is ridge * ||W||²."""
    if x.ndim != 2 or targets.ndim != 2 or x.shape[0] != targets.shape[0] or ridge < 0:
        raise ValueError('Use paired observation matrices and nonnegative ridge.')
    mean, bias = x.mean(0), targets.mean(0)
    centered, values = x - mean, targets - bias
    if ridge == 0:
        weights = t.linalg.lstsq(centered, values, driver='gelsd' if x.device.type == 'cpu' else 'gels').solution
    else:
        weights = t.linalg.solve(centered.T @ centered + ridge * t.eye(x.shape[1], device=x.device, dtype=x.dtype),
                                 centered.T @ values)
    return mean, weights, bias


def circular_coordinates(labels: t.Tensor, period: int):
    """Monday and January have phase zero; adjacent classes differ by 2π/period."""
    if period < 3:
        raise ValueError('A circular family needs at least three distinct classes.')
    angle = labels * (2 * t.pi / period)
    return t.stack((angle.cos(), angle.sin()), -1)


def angular_predictions(coordinates: t.Tensor, period: int):
    """Nearest unit-circle class by angle; an exactly zero vector abstains as -1."""
    if coordinates.shape[-1] != 2 or period < 3:
        raise ValueError('Use two coordinates and period >= 3.')
    angle = t.atan2(coordinates[..., 1], coordinates[..., 0])
    prediction = t.round(angle * period / (2 * t.pi)).long().remainder(period)
    return t.where(coordinates.square().sum(-1) > 0, prediction, -1)


def coordinate_r2(targets: t.Tensor, predictions: t.Tensor):
    """Joint coordinate R², relative to the observed target mean; may be negative."""
    if targets.shape != predictions.shape:
        raise ValueError('Prediction and target inventories must match.')
    denominator = (targets - targets.mean(0)).square().sum()
    if denominator <= 0:
        raise ValueError('R² is undefined when targets have no variation.')
    return 1 - (targets - predictions).square().sum() / denominator


def cyclic_distance_agreement(class_points: t.Tensor):
    """Correlation of pair distances with chronological circle chord distances.

Class identities, not individual prompt rows, are the permutation unit. A high
score can support periodic order; ID classification alone cannot.
"""
    count = class_points.shape[0]
    if count < 4:
        raise ValueError('Use at least four ordered class centroids.')
    target = circular_coordinates(t.arange(count, device=class_points.device, dtype=class_points.dtype), count)
    observed, expected = t.pdist(class_points), t.pdist(target)
    a, b = observed - observed.mean(), expected - expected.mean()
    denominator = a.norm() * b.norm()
    if denominator <= 0:
        raise ValueError('Distance agreement is undefined for a constant distance pattern.')
    return (a @ b) / denominator


def cyclic_permutation_test(class_points: t.Tensor, permutations: t.Tensor):
    """One-sided Monte Carlo null including the observed arrangement via +1."""
    count = class_points.shape[0]
    if permutations.ndim != 2 or permutations.shape[1] != count or not len(permutations):
        raise ValueError('Each null must permute every class exactly once.')
    expected = t.arange(count, device=permutations.device).expand_as(permutations)
    if not t.equal(permutations.sort(-1).values, expected):
        raise ValueError('Each null must permute every class exactly once.')
    observed = cyclic_distance_agreement(class_points)
    null = t.stack([cyclic_distance_agreement(class_points[order]) for order in permutations])
    probability = (1 + (null >= observed - 1e-7).sum()) / (1 + len(null))
    return observed, null, probability


def knn_predictions(train: t.Tensor, labels: t.Tensor, test: t.Tensor, k: int = 3):
    """Euclidean kNN; deterministic nearest-row and lowest-label tie breaking."""
    if not 1 <= k <= len(train) or labels.shape != (len(train),):
        raise ValueError('Use one label per training row and a valid neighbor count.')
    classes, encoded = labels.unique(sorted=True, return_inverse=True)
    neighbors = t.cdist(test, train).argsort(dim=-1, stable=True)[:, :k]
    votes = t.nn.functional.one_hot(encoded[neighbors], len(classes)).sum(1)
    return classes[votes.argmax(-1)]


def neighborhood_scores(high: t.Tensor, low: t.Tensor, k: int = 5):
    """Trustworthiness (intrusions) and continuity (missing true neighbors)."""
    count = len(high)
    if len(low) != count or not 1 <= k < count / 2:
        raise ValueError('Use aligned rows and 1 <= k < n/2.')
    def ranks(points):
        distances = t.cdist(points, points)
        distances.fill_diagonal_(float('inf'))
        order = distances.argsort(dim=-1, stable=True)
        rank = t.empty_like(order)
        rank.scatter_(1, order, t.arange(1, count + 1, device=points.device).expand(count, -1))
        return rank
    original, projected = ranks(high), ranks(low)
    intrusion = t.where(projected <= k, (original - k).clamp_min(0), 0).sum()
    missing = t.where(original <= k, (projected - k).clamp_min(0), 0).sum()
    factor = 2 / (count * k * (2 * count - 3 * k - 1))
    return 1 - factor * intrusion, 1 - factor * missing


def fit_procrustes(source: t.Tensor, target: t.Tensor):
    """Fit centered row-vector orthogonal alignment; reflections are permitted."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError('Align the same observations in equal-dimensional spaces.')
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vh = t.linalg.svd((source - source_mean).T @ (target - target_mean))
    return source_mean, target_mean, u @ vh


def replace_coordinates(activations: t.Tensor, mean: t.Tensor, basis: t.Tensor,
                        weights: t.Tensor, bias: t.Tensor, target: t.Tensor):
    """Minimum-L2 change satisfying the fitted PCA-coordinate probe.

The PCA basis must have orthonormal columns, and the probe must have full
column rank. Nothing outside the probe's column space is changed. Satisfying
this equation alone does not establish the model uses these coordinates.
"""
    if basis.ndim != 2 or weights.ndim != 2 or basis.shape[1] != weights.shape[0]:
        raise ValueError('Basis [width,k] and weights [k,coordinates] must align.')
    if not t.allclose(basis.T @ basis, t.eye(basis.shape[1], device=basis.device, dtype=basis.dtype), atol=1e-5, rtol=1e-5):
        raise ValueError('Minimum-norm coordinates require an orthonormal PCA basis.')
    if t.linalg.matrix_rank(weights) != weights.shape[1]:
        raise ValueError('Every requested coordinate must be independently controllable.')
    prediction = (activations - mean) @ basis @ weights + bias
    correction = (target - prediction) @ t.linalg.pinv(weights) @ basis.T
    return activations + correction


def replace_subspace(activations: t.Tensor, donor: t.Tensor, basis: t.Tensor):
    """Copy only donor coordinates in an orthonormal column basis."""
    if not t.allclose(basis.T @ basis, t.eye(basis.shape[1], device=basis.device, dtype=basis.dtype), atol=1e-5, rtol=1e-5):
        raise ValueError('A donor subspace needs an orthonormal column basis.')
    return activations + (donor - activations) @ basis @ basis.T


def mean_ablate_coordinates(mean: t.Tensor, basis: t.Tensor, weights: t.Tensor,
                           bias: t.Tensor, target: t.Tensor):
    """Set fitted coordinates; replace the entire complement by its training mean.

Unlike replace_coordinates, this deliberately discards receiver information
outside the fitted plane. Compare against mean ablation alone and a pure edit.
"""
    receiver = mean.expand(*target.shape[:-1], mean.shape[-1])
    return replace_coordinates(receiver, mean, basis, weights, bias, target)


def author_raw_circle_replacement(mean: t.Tensor, basis: t.Tensor,
                                  weights: t.Tensor, target: t.Tensor):
    """Literal row-vector QR formula in the paper authors' intervention code.

Reference: JoshEngels/MultiDimensionalFeatures, commit ff8c7eee8f13d78805f590ced6f3fc2227dd1daa,
intervention/circle_finding_utils.py: patch_in_subspace_activations (MIT).
This sets raw plane coordinates and mean-ablates the complement. It is not
the affine centered-coordinate constraint used in mean_ablate_coordinates.
"""
    if t.linalg.matrix_rank(weights) != weights.shape[1]:
        raise ValueError('The circular probe needs independent coordinate columns.')
    q, r = t.linalg.qr(weights, mode='reduced')
    plane = basis @ q
    return mean + target @ t.linalg.inv(r) @ plane.T - mean @ plane @ plane.T


def capture_residual_positions(model, boundaries, encoded, positions):
    """Capture one semantic position at each supplied native residual boundary."""
    saved = {}
    handles = []
    def make_hook(index):
        def hook(module, args):
            saved[index] = gather_token_positions(args[0], positions, encoded['attention_mask']).detach().float().cpu()
        return hook
    try:
        for index, boundary in enumerate(boundaries):
            handles.append(boundary.register_forward_pre_hook(make_hook(index)))
        logits = model(**encoded, use_cache=False).logits
        if set(saved) != set(range(len(boundaries))):
            raise ValueError('Not every requested residual boundary ran.')
        return logits, t.stack([saved[i] for i in range(len(boundaries))], dim=1)
    finally:
        for handle in handles:
            handle.remove()


def gather_token_positions(hidden: t.Tensor, positions: t.Tensor, attention_mask: t.Tensor):
    """Read one explicitly aligned real token per row; reject padding positions."""
    batch, length = hidden.shape[:2]
    if positions.shape != (batch,) or attention_mask.shape != (batch, length):
        raise ValueError('Supply one position per row and the matching padding mask.')
    if (positions < 0).any() or (positions >= length).any():
        raise ValueError('Selected token position is outside the sequence.')
    row = t.arange(batch, device=hidden.device)
    if not attention_mask[row, positions].bool().all():
        raise ValueError('Never treat padding as a semantic token.')
    return hidden[row, positions]


def positions_from_spans(offset_mapping: t.Tensor, attention_mask: t.Tensor, spans: t.Tensor):
    """Locate a whole, single-token semantic span using actual tokenizer offsets."""
    left, right = offset_mapping.unbind(-1)
    overlaps = (left < spans[:, 1, None]) & (right > spans[:, 0, None]) & attention_mask.bool()
    if not overlaps.sum(-1).eq(1).all():
        raise ValueError('Each semantic word must overlap exactly one real token.')
    positions = overlaps.long().argmax(-1)
    row = t.arange(len(spans), device=spans.device)
    if ((left[row, positions] > spans[:, 0]) | (right[row, positions] < spans[:, 1])).any():
        raise ValueError('The selected token does not contain the whole semantic word.')
    return positions


def score_calendar_logits(logits: t.Tensor, original_ids: t.Tensor,
                          desired_ids: t.Tensor, calendar_ids: t.Tensor):
    """Keep full-vocabulary correctness separate from the restricted-choice task."""
    if logits.ndim != 2 or original_ids.shape != (len(logits),) or desired_ids.shape != (len(logits),):
        raise ValueError('Every model row needs an original and a desired answer token.')
    if not t.isfinite(logits).all():
        raise ValueError('Nonfinite model logits cannot support an answer claim.')
    row = t.arange(len(logits), device=logits.device)
    top = logits.argmax(-1)
    restricted = calendar_ids[logits[:, calendar_ids].argmax(-1)]
    return {'top_token_ids': top, 'restricted_token_ids': restricted,
            'original_full_correct': top.eq(original_ids), 'desired_full_correct': top.eq(desired_ids),
            'original_restricted_correct': restricted.eq(original_ids), 'desired_restricted_correct': restricted.eq(desired_ids),
            'desired_minus_original_logit': logits[row, desired_ids] - logits[row, original_ids]}


def forward_with_residual_replacement(model, block, encoded, positions, replacement):
    """Replace the input residual to one native decoder block for one forward pass.

`positions` names the semantic token independently in every padded row. The
hook always gets removed, even if the model raises; caller controls inference
and model mode. Returning logits keeps scoring choices visible to the learner.
"""
    def hook(module, args):
        hidden = args[0]
        current = gather_token_positions(hidden, positions, encoded['attention_mask'])
        if current.shape != replacement.shape:
            raise ValueError('Replacement must match [batch, residual width].')
        modified = hidden.clone()
        modified[t.arange(len(hidden), device=hidden.device), positions] = replacement.to(hidden)
        return (modified, *args[1:])
    handle = block.register_forward_pre_hook(hook)
    try:
        return model(**encoded, use_cache=False).logits
    finally:
        handle.remove()
