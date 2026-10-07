"""Visible reference answers for the nonlinear SGD accounting exercises."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def init_mlp(n_inputs: int, width: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Initialize an explicit float64 tanh classifier on the CPU."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n_inputs, width, generator=generator, dtype=torch.float64) / math.sqrt(n_inputs),
        torch.zeros(width, dtype=torch.float64),
        torch.randn(width, 1, generator=generator, dtype=torch.float64) / math.sqrt(width),
        torch.zeros(1, dtype=torch.float64),
    )


def mlp_forward(params: tuple[torch.Tensor, ...], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits [batch] and hidden activations [batch, width]."""
    w1, b1, w2, b2 = params
    hidden = torch.tanh(x @ w1 + b1)
    return (hidden @ w2 + b2).squeeze(-1), hidden


def mlp_factors(params: tuple[torch.Tensor, ...], x: torch.Tensor, y: torch.Tensor):
    """Return each affine layer's input and unreduced per-example output derivative.

    Each pair has shapes [batch, input_width], [batch, output_width].
    Losses are individual binary cross entropies, with no batch averaging here.
    """
    logits, hidden = mlp_forward(params, x)
    delta2 = (logits.sigmoid() - y.to(logits.dtype))[:, None]
    delta1 = (delta2 @ params[2].T) * (1 - hidden.square())
    return ((x, delta1), (hidden, delta2))


def per_example_gradients(factors) -> torch.Tensor:
    """Materialize the slow oracle [batch, parameters], weights then bias per layer."""
    blocks = []
    for inputs, derivatives in factors:
        blocks.extend(((inputs[:, :, None] * derivatives[:, None, :]).flatten(1), derivatives))
    return torch.cat(blocks, dim=1)


def ghost_dot_products(left, right) -> torch.Tensor:
    """Compute all cross-example gradient inner products without parameter gradients.

    Assumes separate affine weights and biases per layer and independent examples;
    tied parameters, batch-dependent normalization and sequence positions need a
    different derivation. The +1 accounts for each layer's bias gradients.
    """
    if not left or len(left) != len(right):
        raise ValueError("Both networks must supply the same nonempty layer inventory.")
    result = None
    for (a, d), (other_a, other_d) in zip(left, right):
        if (a.ndim != 2 or d.ndim != 2 or other_a.ndim != 2 or other_d.ndim != 2
                or len(a) != len(d) or len(other_a) != len(other_d)
                or a.shape[1] != other_a.shape[1] or d.shape[1] != other_d.shape[1]):
            raise ValueError("Layer inputs and output derivatives have incompatible shapes.")
        block = (a @ other_a.T + 1) * (d @ other_d.T)
        if result is not None and result.shape != block.shape:
            raise ValueError("Layer batch dimensions must agree.")
        result = block if result is None else result + block
    return result


def sgd_step(params, factors, learning_rate: float, denominator: int, keep=None):
    """Update with a fixed original-batch denominator, including for subset controls."""
    if denominator <= 0 or not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("Require a positive denominator and finite nonnegative step size.")
    if len(params) != 2 * len(factors):
        raise ValueError("Each affine layer must have a weight and a bias.")
    updated = []
    for layer, (inputs, derivatives) in enumerate(factors):
        weighted = derivatives if keep is None else derivatives * keep.to(derivatives.dtype)[:, None]
        gradient_w = inputs.T @ weighted / denominator
        gradient_b = weighted.sum(dim=0) / denominator
        updated.extend((params[2 * layer] - learning_rate * gradient_w,
                        params[2 * layer + 1] - learning_rate * gradient_b))
    return tuple(value.detach() for value in updated)


def local_coalition_values(params, train_x, train_y, val_x, val_y, learning_rate):
    """Return all 2**B local loss improvements at the SAME starting checkpoint.

    Removing members never changes the denominator B. Utility sign is positive
    for improvement, the opposite of the paper's loss-change convention.
    """
    size = len(train_x)
    if not 1 <= size <= 8:
        raise ValueError("Exact local enumeration is restricted to batches of one to eight.")
    factors = mlp_factors(params, train_x, train_y)
    before = F.binary_cross_entropy_with_logits(mlp_forward(params, val_x)[0], val_y.double())
    values = []
    for mask in range(1 << size):
        keep = torch.tensor([bool(mask & (1 << i)) for i in range(size)])
        candidate = sgd_step(params, factors, learning_rate, size, keep)
        after = F.binary_cross_entropy_with_logits(mlp_forward(candidate, val_x)[0], val_y.double())
        values.append(before - after)
    return torch.stack(values)


def batch_shapley(values: torch.Tensor) -> torch.Tensor:
    """Exact weighted-subset Shapley for a dense bit-mask indexed local game."""
    count = values.numel()
    if values.ndim != 1 or count < 2 or count & (count - 1):
        raise ValueError("Need a one-dimensional complete nonempty-player Boolean game.")
    size = count.bit_length() - 1
    result = torch.zeros(size, dtype=values.dtype, device=values.device)
    for player in range(size):
        for mask in range(count):
            if not mask & (1 << player):
                weight = 1 / (size * math.comb(size - 1, mask.bit_count()))
                result[player] += weight * (values[mask | (1 << player)] - values[mask])
    return result


def first_order_scores(similarities: torch.Tensor, learning_rate: float, batch_size: int):
    """Positive loss-improvement credit; columns are individual validation examples."""
    if similarities.ndim != 2 or similarities.shape[1] == 0 or batch_size <= 0:
        raise ValueError("Need train-by-validation similarities and a positive batch size.")
    return learning_rate / batch_size * similarities.mean(dim=1)


def second_order_scores(train_gradients, validation_gradient, validation_hessian,
                        learning_rate: float, batch_size: int):
    """Allocate the quadratic local game, including self and cross-gradient terms."""
    scale = learning_rate / batch_size
    first = scale * (train_gradients @ validation_gradient)
    interaction = train_gradients @ validation_hessian @ train_gradients.sum(dim=0)
    return first - 0.5 * scale**2 * interaction


def make_schedule(n_examples: int, batch_size: int, epochs: int, seed: int):
    """Fixed full-size minibatches; reuse exact row order for deletion controls."""
    if n_examples <= 0 or batch_size <= 0 or n_examples % batch_size or epochs <= 0:
        raise ValueError("Require positive sizes and complete equal-size batches.")
    generator = torch.Generator().manual_seed(seed)
    return [row.clone() for _ in range(epochs)
            for row in torch.randperm(n_examples, generator=generator).reshape(-1, batch_size)]


def run_accountant(initial, train_x, train_y, val_x, val_y, schedule, learning_rate,
                   *, exact=False, removed=()):
    """Train a fresh fixed-schedule trajectory and accumulate visible ghost scores.

    Validation examples determine attribution but never the SGD update. Removed
    examples contribute zero update; original batch size and step count stay fixed.
    """
    params = tuple(value.detach().clone() for value in initial)
    score = torch.zeros(len(train_x), dtype=train_x.dtype)
    exact_score = torch.zeros_like(score)
    keep_global = torch.ones(len(train_x), dtype=torch.bool)
    keep_global[list(removed)] = False
    losses, allocations, exact_allocations = [], [], []
    losses.append(float(F.binary_cross_entropy_with_logits(mlp_forward(params, val_x)[0], val_y.double())))
    if exact and removed:
        raise ValueError("Exact full-batch accounting and deletion are separate experiments.")
    for batch in schedule:
        factors = mlp_factors(params, train_x[batch], train_y[batch])
        validation = mlp_factors(params, val_x, val_y)
        local = first_order_scores(ghost_dot_products(factors, validation), learning_rate, len(batch))
        local = local * keep_global[batch]
        score.index_add_(0, batch, local)
        row = torch.zeros_like(score).index_add_(0, batch, local)
        allocations.append(row)
        if exact:
            values = local_coalition_values(params, train_x[batch], train_y[batch], val_x, val_y, learning_rate)
            exact_local = batch_shapley(values)
            exact_score.index_add_(0, batch, exact_local)
            exact_allocations.append(torch.zeros_like(score).index_add_(0, batch, exact_local))
        params = sgd_step(params, factors, learning_rate, len(batch), keep_global[batch])
        losses.append(float(F.binary_cross_entropy_with_logits(mlp_forward(params, val_x)[0], val_y.double())))
    return dict(params=params, scores=score, exact_scores=exact_score if exact else None,
                losses=torch.tensor(losses, dtype=train_x.dtype), allocations=torch.stack(allocations),
                exact_allocations=torch.stack(exact_allocations) if exact else None)
