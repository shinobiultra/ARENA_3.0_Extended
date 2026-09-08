"""Visible mathematical answers for the influence / TracIn learner.

All losses use binary targets in {0, 1}. A constant input coordinate represents
an intercept; when present it is regularized just like every other parameter.
"""
from __future__ import annotations

from collections.abc import Callable
import math

import torch
import torch.nn.functional as F
from torch import Tensor


def logistic_losses(theta: Tensor, x: Tensor, y: Tensor, l2: float = 0.0) -> Tensor:
    """Unreduced per-example losses, optionally including R(theta) in each row."""
    if theta.ndim != 1 or x.ndim != 2 or x.shape[1] != theta.numel() or y.shape != (len(x),):
        raise ValueError("Expected theta [parameters], x [examples, parameters], y [examples].")
    if not ((y == 0) | (y == 1)).all() or not math.isfinite(l2) or l2 < 0:
        raise ValueError("Use binary labels and a finite nonnegative regularizer.")
    logits = x @ theta
    return F.softplus(logits) - y * logits + 0.5 * l2 * theta.square().sum()


def logistic_gradients(theta: Tensor, x: Tensor, y: Tensor, l2: float = 0.0) -> Tensor:
    """Per-example parameter gradients; their mean is the objective gradient."""
    logistic_losses(theta, x, y, l2)  # Validate the common public contract.
    residual = torch.sigmoid(x @ theta) - y
    return residual[:, None] * x + l2 * theta[None, :]


def weighted_objective(theta: Tensor, x: Tensor, y: Tensor, weights: Tensor, l2: float) -> Tensor:
    """One independent normalized objective per row of theta [fits, parameters]."""
    if theta.ndim != 2 or x.ndim != 2 or theta.shape[1] != x.shape[1]:
        raise ValueError("Expected theta [fits, parameters] and x [examples, parameters].")
    if weights.shape != (len(theta), len(x)) or y.shape != (len(x),):
        raise ValueError("Expected weights [fits, examples] and y [examples].")
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(1) <= 0).any():
        raise ValueError("Every fit needs finite nonnegative weights with positive total mass.")
    if not ((y == 0) | (y == 1)).all() or not math.isfinite(l2) or l2 < 0:
        raise ValueError("Use binary targets and a finite nonnegative regularizer.")
    probabilities = weights / weights.sum(1, keepdim=True)
    logits = theta @ x.T
    return ((F.softplus(logits) - y[None, :] * logits) * probabilities).sum(1) + 0.5 * l2 * theta.square().sum(1)


def objective_derivatives(theta: Tensor, x: Tensor, y: Tensor, weights: Tensor, l2: float) -> tuple[Tensor, Tensor]:
    """Gradient and dense Hessian of each independently weighted objective."""
    weighted_objective(theta, x, y, weights, l2)
    mass = weights / weights.sum(1, keepdim=True)
    probabilities = torch.sigmoid(theta @ x.T)
    gradient = ((probabilities - y[None, :]) * mass) @ x + l2 * theta
    curvature = probabilities * (1 - probabilities) * mass
    hessian = torch.einsum("np,bn,nq->bpq", x, curvature, x)
    hessian = hessian + l2 * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return gradient, hessian


def logistic_hvp(theta: Tensor, x: Tensor, y: Tensor, weights: Tensor, vector: Tensor, l2: float) -> Tensor:
    """An objective Hessian-vector product without materializing a Hessian."""
    if theta.ndim != 1 or vector.shape != theta.shape or weights.shape != (len(x),):
        raise ValueError("Use one parameter vector, one matching direction and one example-weight vector.")
    weighted_objective(theta[None], x, y, weights[None], l2)
    probabilities = torch.sigmoid(x @ theta)
    curvature = weights / weights.sum() * probabilities * (1 - probabilities)
    return x.T @ (curvature * (x @ vector)) + l2 * vector


def conjugate_gradient(
    matvec: Callable[[Tensor], Tensor], rhs: Tensor, rtol: float = 1e-10,
    atol: float = 1e-12, max_steps: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Solve an SPD system, returning the actual residual norms at every step."""
    if rhs.ndim != 1 or not torch.isfinite(rhs).all():
        raise ValueError("The right-hand side must be one finite vector.")
    if not all(math.isfinite(t) and t >= 0 for t in (rtol, atol)):
        raise ValueError("Residual tolerances must be finite and nonnegative.")
    max_steps = 2 * rhs.numel() if max_steps is None else max_steps
    if max_steps < 1:
        raise ValueError("Allow at least one CG step.")
    answer = torch.zeros_like(rhs)
    residual = rhs - matvec(answer)
    direction = residual.clone()
    squared_norm = residual @ residual
    threshold = atol + rtol * torch.linalg.vector_norm(rhs)
    history = [torch.linalg.vector_norm(residual)]
    if history[-1] <= threshold:
        return answer, torch.stack(history)
    for _ in range(max_steps):
        product = matvec(direction)
        curvature = direction @ product
        if not torch.isfinite(curvature) or curvature <= 0:
            raise ValueError("CG encountered nonpositive curvature; its SPD assumption failed.")
        alpha = squared_norm / curvature
        answer = answer + alpha * direction
        residual = residual - alpha * product
        # A recurrence residual can drift: success is checked against A @ answer.
        actual_norm = torch.linalg.vector_norm(rhs - matvec(answer))
        history.append(actual_norm)
        if actual_norm <= threshold:
            return answer, torch.stack(history)
        next_squared_norm = residual @ residual
        direction = residual + (next_squared_norm / squared_norm) * direction
        squared_norm = next_squared_norm
    raise RuntimeError(f"CG did not converge: residual {float(history[-1]):.3g} exceeds {float(threshold):.3g}.")


def upweight_influence(hessian: Tensor, train_gradients: Tensor, query_gradients: Tensor, damping: float = 0.0) -> Tensor:
    """d(query loss)/d epsilon for F + epsilon * one regularized training loss."""
    p = hessian.shape[0]
    if hessian.shape != (p, p) or train_gradients.ndim != 2 or query_gradients.ndim != 2:
        raise ValueError("Use a square Hessian and two example-by-parameter gradient matrices.")
    if train_gradients.shape[1] != p or query_gradients.shape[1] != p:
        raise ValueError("Every gradient must refer to the same parameter coordinates.")
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("Damping must be finite and nonnegative.")
    if not torch.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-12):
        raise ValueError("The objective Hessian must be symmetric.")
    system = hessian + damping * torch.eye(p, dtype=hessian.dtype, device=hessian.device)
    torch.linalg.cholesky(system)  # Fail explicitly if the local SPD premise fails.
    response = torch.linalg.solve(system, query_gradients.T)
    return -train_gradients @ response


def tracin_checkpoint_scores(train_gradients: Tensor, query_gradients: Tensor, rates: Tensor) -> Tensor:
    """Weighted gradient products; positive means first-order loss reduction.

The input is [checkpoints, examples, parameters]. Rates are actual checkpoint
learning-rate weights. A checkpoint subset covers only its selected visits;
this function does not rescale scores to pretend they sum to observed loss.
"""
    if train_gradients.ndim != 3 or query_gradients.ndim != 3:
        raise ValueError("Keep checkpoint, example and parameter axes distinct.")
    k, _, p = train_gradients.shape
    if query_gradients.shape[0] != k or query_gradients.shape[2] != p or rates.shape != (k,) or k == 0:
        raise ValueError("Checkpoint counts, parameter coordinates and rate counts must match.")
    if not torch.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Use finite nonnegative learning-rate weights.")
    return torch.einsum("knp,kqp,k->nq", train_gradients, query_gradients, rates)


def ghost_linear_products(left_inputs: Tensor, left_deltas: Tensor, right_inputs: Tensor, right_deltas: Tensor, bias: bool = True) -> Tensor:
    """Exact gradient products for one affine layer with one invocation per example.

Deltas differentiate scalar losses with respect to preactivations. This is not
a formula for tied parameters used at several token positions; those require
all pairs of positions. Shared regularizers are also separate gradient terms.
"""
    arrays = (left_inputs, left_deltas, right_inputs, right_deltas)
    if any(a.ndim != 2 for a in arrays):
        raise ValueError("Inputs and preactivation derivatives must be two-dimensional.")
    if len(left_inputs) != len(left_deltas) or len(right_inputs) != len(right_deltas):
        raise ValueError("Every example needs its own input and preactivation derivative.")
    if left_inputs.shape[1] != right_inputs.shape[1] or left_deltas.shape[1] != right_deltas.shape[1]:
        raise ValueError("Both sides must describe the same affine layer.")
    return (left_inputs @ right_inputs.T + int(bias)) * (left_deltas @ right_deltas.T)


def fit_weighted_logistic(
    x: Tensor, y: Tensor, weights: Tensor, l2: float = 0.02,
    tolerance: float = 1e-10, max_steps: int = 60,
) -> dict[str, Tensor]:
    """Independently refit every row of weights from zero using Newton/Armijo.

Batching shares matrix operations, not fitted parameters or loss reductions.
Each fit keeps its own line search and convergence decision. No failed fit is
silently accepted. Regularization stays fixed after normalization of weights.
"""
    if weights.ndim != 2 or l2 <= 0 or not math.isfinite(l2):
        raise ValueError("Use weights [fits, examples] and a strictly positive regularizer.")
    if tolerance <= 0 or not math.isfinite(tolerance) or max_steps < 1:
        raise ValueError("Use a positive residual tolerance and iteration budget.")
    theta = torch.zeros(weights.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
    states, residuals = [], []
    for _ in range(max_steps + 1):
        loss = weighted_objective(theta, x, y, weights, l2)
        gradient, hessian = objective_derivatives(theta, x, y, weights, l2)
        norms = torch.linalg.vector_norm(gradient, ord=float("inf"), dim=1)
        states.append(theta.clone())
        residuals.append(norms)
        active = norms > tolerance
        if not active.any():
            return {"theta": theta, "states": torch.stack(states), "gradient_norms": torch.stack(residuals), "objective": loss}
        if len(states) > max_steps:
            break
        step = torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
        step[~active] = 0
        decrement = (gradient * step).sum(1)
        rates = torch.ones(len(theta), dtype=x.dtype, device=x.device)
        accepted = ~active
        proposed = theta.clone()
        for _ in range(30):
            trial = theta - rates[:, None] * step
            trial_loss = weighted_objective(trial, x, y, weights, l2)
            # Near the optimum, the predicted decrease can be below one ulp.
            # Allow objective roundoff while still requiring the final gradient
            # tolerance; otherwise a vanishing step can stall a correct refit.
            roundoff = 8 * torch.finfo(x.dtype).eps * loss.abs().clamp_min(1)
            sufficient = trial_loss <= loss - 1e-4 * rates * decrement + roundoff
            newly_accepted = sufficient & ~accepted
            proposed[newly_accepted] = trial[newly_accepted]
            accepted |= sufficient
            if accepted.all():
                break
            rates[~accepted] *= 0.5
        if not accepted.all():
            raise RuntimeError("Newton line search failed; retain the failure instead of accepting an unfinished refit.")
        theta = proposed
    raise RuntimeError(f"Newton did not converge for every fit; largest gradient is {float(norms.max()):.3g}.")


def sgd_trace(x: Tensor, y: Tensor, orders: Tensor, epoch_rates: Tensor, l2: float) -> dict[str, Tensor]:
    """Record the actual single-example optimization path, starting at zero."""
    if orders.ndim != 2 or orders.shape[1] != len(x) or epoch_rates.shape != (len(orders),):
        raise ValueError("Provide one permutation and one learning rate for every epoch.")
    expected = torch.arange(len(x), device=orders.device).expand_as(orders)
    if not torch.equal(orders.sort(dim=1).values, expected):
        raise ValueError("Each epoch must visit every training example exactly once.")
    if not torch.isfinite(epoch_rates).all() or (epoch_rates <= 0).any():
        raise ValueError("Epoch learning rates must be finite and strictly positive.")
    used_ids = orders.flatten()
    rates = epoch_rates.repeat_interleave(len(x))
    states = torch.zeros(len(used_ids) + 1, x.shape[1], dtype=x.dtype, device=x.device)
    with torch.no_grad():
        for t, index in enumerate(used_ids.tolist()):
            gradient = logistic_gradients(states[t], x[index:index + 1], y[index:index + 1], l2)[0]
            states[t + 1] = states[t] - rates[t] * gradient
    return {"states": states, "used_ids": used_ids, "rates": rates}


def trace_loss_reductions(
    states: Tensor, used_ids: Tensor, rates: Tensor, train_x: Tensor, train_y: Tensor,
    query_x: Tensor, query_y: Tensor, l2: float, chunk_size: int = 128,
) -> dict[str, Tensor]:
    """Actual loss accounting and its Taylor approximation along recorded SGD.

The exact allocation is well-defined here because each update uses one row.
The calculation includes that update's regularizer contribution. The raw-only
control deliberately omits it. No tensor of per-step per-query full gradients
is needed: a logistic query gradient is its residual times its input vector.
"""
    steps = len(used_ids)
    if states.ndim != 2 or len(states) != steps + 1 or rates.shape != (steps,):
        raise ValueError("Every update needs a before/after state, training ID and learning rate.")
    if used_ids.dtype != torch.int64 or (used_ids < 0).any() or (used_ids >= len(train_x)).any():
        raise ValueError("Recorded training IDs must be valid integer row indices.")
    if chunk_size < 1 or not torch.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Use a positive chunk size and positive finite learning rates.")
    exact_steps, first_order_steps, raw_steps = [], [], []
    for start in range(0, steps, chunk_size):
        stop = min(start + chunk_size, steps)
        before, after = states[start:stop], states[start + 1:stop + 1]
        x = train_x[used_ids[start:stop]]
        y = train_y[used_ids[start:stop]]
        raw_gradient = (torch.sigmoid((before * x).sum(1)) - y)[:, None] * x
        gradient = raw_gradient + l2 * before
        if not torch.allclose(after - before, -rates[start:stop, None] * gradient, rtol=1e-9, atol=1e-12):
            raise ValueError("Recorded states do not follow the stated single-example SGD update.")
        before_logits, after_logits = before @ query_x.T, after @ query_x.T
        before_loss = F.softplus(before_logits) - query_y[None, :] * before_logits
        after_loss = F.softplus(after_logits) - query_y[None, :] * after_logits
        query_residual = torch.sigmoid(before_logits) - query_y[None, :]
        exact_steps.append(before_loss - after_loss)
        first_order_steps.append(rates[start:stop, None] * query_residual * (gradient @ query_x.T))
        raw_steps.append(rates[start:stop, None] * query_residual * (raw_gradient @ query_x.T))
    exact_steps = torch.cat(exact_steps)
    first_order_steps = torch.cat(first_order_steps)
    raw_steps = torch.cat(raw_steps)
    shape = (len(train_x), len(query_x))
    def collect(values):
        return torch.zeros(shape, dtype=states.dtype, device=states.device).index_add_(0, used_ids, values)
    return {
        "exact": collect(exact_steps), "first_order": collect(first_order_steps), "raw_only": collect(raw_steps),
        "exact_steps": exact_steps, "first_order_steps": first_order_steps, "raw_only_steps": raw_steps,
    }


def class_matched_removals(helpfulness: Tensor, labels: Tensor, budget: int) -> Tensor:
    """Select low helpfulness, with equal observed-class counts and stable ties."""
    if helpfulness.ndim != 1 or labels.shape != helpfulness.shape or not torch.isfinite(helpfulness).all():
        raise ValueError("Provide one finite helpfulness score and observed label per training row.")
    if not ((labels == 0) | (labels == 1)).all() or budget < 0 or budget % 2:
        raise ValueError("Binary class matching requires an even nonnegative deletion budget.")
    selected = []
    for label in (0, 1):
        ids = torch.nonzero(labels == label).flatten()
        if len(ids) < budget // 2:
            raise ValueError("The budget exceeds the available rows in an observed class.")
        order = torch.argsort(helpfulness[ids], stable=True)
        selected.append(ids[order[:budget // 2]])
    return torch.cat(selected).sort().values
