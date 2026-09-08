"""Visible reference implementations for the concept-erasure learner.

Rows are examples. Fitting is performed in float64 in the experiments; no
regularization or singular-value cutoff is chosen using final examples.
"""
from __future__ import annotations

import torch as t
import torch.nn.functional as F


def centered_statistics(x: t.Tensor, z: t.Tensor) -> dict:
    """Population second moments of paired rows, preserving their affine means."""
    if x.ndim != 2 or z.ndim != 2 or len(x) != len(z) or len(x) < 2:
        raise ValueError("Use at least two paired rows in X and Z.")
    if not x.is_floating_point() or not z.is_floating_point():
        raise ValueError("Use floating representations and floating one-hot labels.")
    if x.dtype != z.dtype or x.device != z.device or not x.shape[1] or not z.shape[1]:
        raise ValueError("Use nonempty features with matching dtype and device.")
    if not t.isfinite(x).all() or not t.isfinite(z).all():
        raise ValueError("Moments require finite data.")
    mean_x, mean_z = x.mean(0), z.mean(0)
    xc, zc = x - mean_x, z - mean_z
    return dict(mean_x=mean_x, mean_z=mean_z,
                covariance=xc.T @ xc / len(x), cross_covariance=xc.T @ zc / len(x))


def psd_whitener(covariance: t.Tensor, rcond: float | None = None) -> dict:
    """Whiten on the empirical support and preserve an explicit numerical rank."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not len(covariance):
        raise ValueError("The covariance must be a nonempty square matrix.")
    if not covariance.is_floating_point() or not t.isfinite(covariance).all():
        raise ValueError("Use a finite real floating covariance.")
    eps = t.finfo(covariance.dtype).eps
    relative = len(covariance) * eps if rcond is None else rcond
    if relative < 0 or not t.isfinite(t.tensor(relative)):
        raise ValueError("The relative eigenvalue cutoff must be finite and nonnegative.")
    scale = float(covariance.abs().max())
    if not t.allclose(covariance, covariance.T, atol=10 * len(covariance) * eps * scale, rtol=0):
        raise ValueError("The covariance must be symmetric.")
    eigenvalues, vectors = t.linalg.eigh((covariance + covariance.T) / 2)
    if eigenvalues.min() < -10 * len(covariance) * eps * eigenvalues.abs().max():
        raise ValueError("The covariance must be positive semidefinite.")
    eigenvalues = eigenvalues.clamp_min(0)
    threshold = eigenvalues.max() * relative
    keep = eigenvalues > threshold
    roots = t.where(keep, eigenvalues.sqrt(), 0)
    inverse_roots = t.zeros_like(roots)
    inverse_roots[keep] = 1 / roots[keep]
    return dict(whiten=(vectors * inverse_roots) @ vectors.T,
                unwhiten=(vectors * roots) @ vectors.T,
                rank=int(keep.sum()), eigenvalues=eigenvalues, threshold=float(threshold))


def fit_orthogonal_eraser(x: t.Tensor, z: t.Tensor, svd_atol: float = 1e-10) -> dict:
    """Remove the cross-covariance span, rather than a fitted classifier's normal."""
    if svd_atol < 0 or not t.isfinite(t.tensor(svd_atol)):
        raise ValueError("Use a finite nonnegative singular-value cutoff.")
    stats = centered_statistics(x, z)
    u, singular, _ = t.linalg.svd(stats["cross_covariance"], full_matrices=False)
    basis = u[:, singular > svd_atol]
    return dict(mean=stats["mean_x"], left=basis, right=basis.T,
                rank=basis.shape[1], cross_singular_values=singular)


def fit_leace(x: t.Tensor, z: t.Tensor, *, covariance_rcond: float | None = None,
              svd_atol: float = 1e-10, shrinkage: float = 0.0) -> dict:
    """Fit the whiten-project-unwhiten eraser as two thin factors.

    shrinkage=0 is the empirical least-squares solution. Positive shrinkage
    uses a spherical covariance surrogate; it is a different fitting objective.
    """
    if not 0 <= shrinkage <= 1 or svd_atol < 0 or not t.isfinite(t.tensor(svd_atol)):
        raise ValueError("Use shrinkage in [0, 1] and a finite nonnegative SVD cutoff.")
    stats = centered_statistics(x, z)
    covariance = stats["covariance"]
    eye = t.eye(len(covariance), dtype=x.dtype, device=x.device)
    surrogate = (1 - shrinkage) * covariance + shrinkage * covariance.trace() / len(covariance) * eye
    geometry = psd_whitener(surrogate, covariance_rcond)
    whiten, unwhiten = geometry["whiten"], geometry["unwhiten"]
    u, singular, _ = t.linalg.svd(whiten @ stats["cross_covariance"], full_matrices=False)
    basis = u[:, singular > svd_atol]
    return dict(mean=stats["mean_x"], left=unwhiten @ basis, right=basis.T @ whiten,
                rank=basis.shape[1], covariance_rank=geometry["rank"],
                covariance_threshold=geometry["threshold"],
                cross_singular_values=singular, shrinkage=shrinkage)


def apply_eraser(x: t.Tensor, eraser: dict) -> t.Tensor:
    """Apply an affine low-rank edit to row vectors without an O(d squared) map."""
    left, right, mean = eraser["left"], eraser["right"], eraser["mean"]
    if (x.ndim != 2 or mean.shape != (x.shape[1],) or left.ndim != 2 or right.ndim != 2
            or left.shape[0] != x.shape[1] or right.shape != (left.shape[1], x.shape[1])):
        raise ValueError("Use X[n,d], mean[d], left[d,k], and right[k,d].")
    if any(a.dtype != x.dtype or a.device != x.device for a in (left, right, mean)):
        raise ValueError("Apply the eraser in its fitted dtype and device.")
    return x - ((x - mean) @ right.T) @ left.T


def fit_binary_probe(x: t.Tensor, labels: t.Tensor, *, l2: float = 1e-3,
                     steps: int = 150) -> dict:
    """Refit a convex logistic probe from training rows only, including an intercept."""
    if x.ndim != 2 or labels.shape != (len(x),) or len(x) < 2:
        raise ValueError("Use one binary label for each representation row.")
    if not t.all((labels == 0) | (labels == 1)) or labels.unique().numel() != 2:
        raise ValueError("Training must contain both binary classes.")
    if not x.is_floating_point() or not t.isfinite(x).all() or l2 < 0 or steps < 1:
        raise ValueError("Use finite floating activations and a valid optimization budget.")
    mean = x.mean(0)
    scale = (x - mean).square().mean(0).sqrt()
    scale = t.where(scale > t.finfo(x.dtype).eps, scale, 1)
    standardized = (x - mean) / scale
    target = labels.to(dtype=x.dtype, device=x.device)
    w = t.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    b = target.mean().logit().detach().requires_grad_(True)
    optimizer = t.optim.LBFGS([w, b], max_iter=steps, tolerance_grad=1e-10,
                             tolerance_change=1e-12, line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(standardized @ w + b, target)
        objective = loss + l2 * w.square().sum() / 2
        objective.backward()
        history.append(float(objective.detach()))
        return objective

    optimizer.step(closure)
    coefficient = w.detach() / scale
    intercept = b.detach() - mean @ coefficient
    return dict(coefficient=coefficient, intercept=intercept, training_mean=mean,
                training_scale=scale, objective_history=history, l2=l2)


def evaluate_binary_probe(x: t.Tensor, labels: t.Tensor, probe: dict) -> dict:
    """Keep convex log loss and thresholded accuracy as different measurements."""
    logits = x @ probe["coefficient"] + probe["intercept"]
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits))
    return dict(log_loss=float(loss), accuracy=float(((logits > 0) == labels.bool()).double().mean()),
                probabilities=logits.sigmoid(), logits=logits)


def matched_random_edit(x: t.Tensor, mean: t.Tensor, direction: t.Tensor,
                        target_eraser: dict) -> t.Tensor:
    """A control with each example's target edit norm and a fixed random direction.

    This magnitude-matched control is nonlinear in X. It is not claimed to be
    an affine eraser or to satisfy a zero-cross-covariance constraint.
    """
    if direction.shape != (x.shape[1],) or mean.shape != direction.shape:
        raise ValueError("Use one random direction and the training mean.")
    if not t.allclose(direction.norm(), direction.new_tensor(1.), atol=1e-10, rtol=1e-10):
        raise ValueError("The random direction must have unit length.")
    magnitude = (apply_eraser(x, target_eraser) - x).norm(dim=-1)
    projection = (x - mean) @ direction
    sign = t.where(projection >= 0, 1., -1.)
    return x - (sign * magnitude)[:, None] * direction


def compare_model_outputs(logits: t.Tensor, native_logits: t.Tensor, targets: t.Tensor) -> dict:
    """Separate answer correctness, confidence, distribution change and native errors."""
    if (logits.ndim != 2 or logits.shape != native_logits.shape or not len(logits)
            or targets.shape != (len(logits),)):
        raise ValueError("Use two equally shaped full-vocabulary batches and one target per row.")
    if not t.isfinite(logits).all() or not t.isfinite(native_logits).all():
        raise ValueError("Native comparisons require finite logits.")
    log_p, log_q = logits.log_softmax(-1), native_logits.log_softmax(-1)
    correct, native_correct = logits.argmax(-1) == targets, native_logits.argmax(-1) == targets
    return dict(rows=len(logits), accuracy=float(correct.double().mean()), correct=int(correct.sum()),
                native_correct_count=int(native_correct.sum()),
                preserved_native_correct=int((correct & native_correct).sum()),
                answer_loss=float(F.cross_entropy(logits, targets)),
                answer_probability=float(log_p[t.arange(len(logits)), targets].exp().mean()),
                kl_from_native=float((log_q.exp() * (log_q - log_p)).sum(-1).mean()),
                changed_winner_fraction=float((logits.argmax(-1) != native_logits.argmax(-1)).double().mean()))
