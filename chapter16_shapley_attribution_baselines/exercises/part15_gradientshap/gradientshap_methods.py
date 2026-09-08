"""Visible learner methods for Expected Gradients and noisy GradientShap paths.

Expected Gradients averages continuous donor-to-query path contributions; it is
not exact Shapley for the discrete feature-coalition game. Finite Monte Carlo
means have ordinary estimator standard error and need not be exactly complete.
When noise perturbs the query, the summed population target becomes
E[f(x + noise)] - E[f(background)], rather than the clean prediction difference.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import random
from collections.abc import Iterator

import numpy as np
import torch as t
from torch import Tensor, nn


__all__ = [
    "sample_path_schedule",
    "interpolate_paths",
    "input_gradients",
    "expected_gradient_samples",
    "summarize_expected_gradients",
    "expected_gradients",
    "gauss_legendre_pairs",
    "noisy_expected_gradient_samples",
    "ranked_coalition_curves",
]


def _eg_float_tensor(name: str, value: Tensor, ndim: int | None = None) -> Tensor:
    """Check the lesson's CPU-float64 tensor contract without changing the value."""
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if value.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not t.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _eg_index_tensor(name: str, value: Tensor, ndim: int | None = None) -> Tensor:
    """Check a CPU int64 index tensor without coercing or aliasing it."""
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != t.int64:
        raise TypeError(f"{name} must be a CPU int64 tensor.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if value.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    return value


def _eg_positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _eg_seed_words(seed: int, source_id: int, stream_tag: str) -> int:
    """Derive one local 128-bit PCG64 seed from the declared lesson namespace."""
    name = f"arena16.15/gradientshap/v1/{seed}/{source_id}/{stream_tag}"
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:16], "little")


def _eg_model(model: nn.Module) -> nn.Module:
    """Validate the CPU-float64 state used by the lesson's scalar models."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != t.float64:
            raise TypeError("Every model parameter must be a CPU float64 tensor.")
        if not t.isfinite(parameter).all():
            raise ValueError("Every model parameter must be finite.")
    for buffer in model.buffers():
        if buffer.device.type != "cpu":
            raise TypeError("Every model buffer must be on CPU.")
        if buffer.is_floating_point():
            if buffer.dtype != t.float64:
                raise TypeError("Every floating model buffer must be float64.")
            if not t.isfinite(buffer).all():
                raise ValueError("Every floating model buffer must be finite.")
    return model


def _eg_model_output(model: nn.Module, inputs: Tensor) -> Tensor:
    """Return one scalar per row; callers control whether gradients are enabled."""
    output = model(inputs)
    if type(output) is not Tensor:
        raise TypeError("The model must return a torch.Tensor.")
    if output.shape == (len(inputs), 1):
        output = output[:, 0]
    elif output.shape != (len(inputs),):
        raise ValueError("The model must return one scalar output per input row.")
    if output.device.type != "cpu" or output.dtype != t.float64:
        raise TypeError("The model must return CPU float64 outputs.")
    if not t.isfinite(output).all():
        raise ValueError("The model returned a nonfinite value.")
    return output


@contextlib.contextmanager
def _eg_preserve_execution_state(model: nn.Module) -> Iterator[None]:
    """Restore global RNG, default dtype, module modes, buffers, and parameter grads."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = t.random.get_rng_state()
    default_dtype = t.get_default_dtype()
    module_modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    gradients = [
        (
            parameter,
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in model.parameters()
    ]
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        t.random.set_rng_state(torch_state)
        t.set_default_dtype(default_dtype)
        for module, training in module_modes:
            module.training = training
        with t.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
            for parameter, original, saved in gradients:
                if original is None:
                    parameter.grad = None
                else:
                    parameter.grad = original
                    original.copy_(saved)


def _eg_validate_paths(
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    noise: Tensor | None = None,
) -> tuple[int, int, int]:
    _eg_float_tensor("x", x, 2)
    _eg_float_tensor("background", background, 2)
    _eg_index_tensor("indices", indices, 2)
    _eg_float_tensor("alpha", alpha, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    if indices.shape != alpha.shape or indices.shape[0] != len(x):
        raise ValueError("indices and alpha must both have shape (rows of x, samples).")
    if indices.min() < 0 or indices.max() >= len(background):
        raise ValueError("Every donor index must select a supplied background row.")
    if alpha.min() < 0 or alpha.max() > 1:
        raise ValueError("Every alpha must lie in the closed interval [0, 1].")
    n_rows, samples = indices.shape
    n_features = x.shape[1]
    if noise is not None:
        _eg_float_tensor("noise", noise, 3)
        if noise.shape != (n_rows, samples, n_features):
            raise ValueError("noise must have shape (rows of x, samples, input features).")
    return n_rows, samples, n_features


def sample_path_schedule(
    source_ids: Tensor,
    background_rows: int,
    samples: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Draw donor indices and alphas from independent local streams for each source ID."""
    _eg_index_tensor("source_ids", source_ids, 1)
    background_rows = _eg_positive_int("background_rows", background_rows)
    samples = _eg_positive_int("samples", samples)
    if type(seed) is not int or seed < 0 or seed >= 2**64:
        raise ValueError("seed must be an integer in [0, 2**64).")
    if (source_ids < 0).any():
        raise ValueError("source_ids must be nonnegative integers.")
    if len(t.unique(source_ids)) != len(source_ids):
        raise ValueError("source_ids must be unique so every row names one sampling stream.")

    indices = np.empty((len(source_ids), samples), dtype=np.int64)
    alpha = np.empty((len(source_ids), samples), dtype=np.float64)
    for row, source_id in enumerate(source_ids.tolist()):
        donor_rng = np.random.Generator(
            np.random.PCG64(_eg_seed_words(seed, source_id, "baseline"))
        )
        alpha_rng = np.random.Generator(
            np.random.PCG64(_eg_seed_words(seed, source_id, "alpha"))
        )
        indices[row] = donor_rng.integers(0, background_rows, size=samples, dtype=np.int64)
        alpha[row] = alpha_rng.random(samples, dtype=np.float64)
    return t.tensor(indices, dtype=t.int64), t.tensor(alpha, dtype=t.float64)


def interpolate_paths(
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return sampled path points and query-minus-donor deltas, both shaped N,K,D."""
    _eg_validate_paths(x, background, indices, alpha, noise)
    donors = background[indices]
    query = x[:, None, :] if noise is None else x[:, None, :] + noise
    delta = query - donors
    points = donors + alpha[:, :, None] * delta
    return points, delta


def input_gradients(
    model: nn.Module,
    points: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Differentiate a deterministic scalar function applied independently to each row.

    The sum-of-outputs gradient is valid only under that batch-independence assumption.
    This function preserves the caller's tensors, RNG states, model modes, buffers, and
    parameter gradients. A scalar function constant with respect to its input gets zero
    input gradients.
    """
    model = _eg_model(model)
    _eg_float_tensor("points", points, 3)
    forward_batch = _eg_positive_int("forward_batch", forward_batch)
    flat_points = points.reshape(-1, points.shape[-1])
    flat_gradients = t.empty_like(flat_points)

    with _eg_preserve_execution_state(model):
        for start in range(0, len(flat_points), forward_batch):
            chunk = flat_points[start : start + forward_batch].detach().clone().requires_grad_(True)
            with t.enable_grad():
                output = _eg_model_output(model, chunk)
                if output.requires_grad:
                    gradient = t.autograd.grad(output.sum(), chunk, allow_unused=True)[0]
                else:
                    gradient = None
            if gradient is None:
                gradient = t.zeros_like(chunk)
            if not t.isfinite(gradient).all():
                raise ValueError("The model produced a nonfinite input gradient.")
            flat_gradients[start : start + len(chunk)] = gradient.detach()
    return flat_gradients.reshape(points.shape)


def expected_gradient_samples(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Return clean Expected Gradients Monte Carlo contributions, shape N,K,D."""
    points, delta = interpolate_paths(x, background, indices, alpha)
    gradients = input_gradients(model, points, forward_batch)
    return gradients * delta


def summarize_expected_gradients(
    samples: Tensor,
    budgets: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    """Return ordinary sample SE, not SHAP 0.49.1's variance/sqrt(K) proxy."""
    _eg_float_tensor("samples", samples, 3)
    if type(budgets) is not tuple or not budgets:
        raise TypeError("budgets must be a nonempty tuple of sample counts.")
    if any(type(budget) is not int for budget in budgets):
        raise TypeError("Every budget must be an integer sample count.")
    if any(budget < 2 or budget > samples.shape[1] for budget in budgets):
        raise ValueError("Every budget must be between 2 and the available sample count.")
    if any(left >= right for left, right in zip(budgets, budgets[1:])):
        raise ValueError("budgets must be strictly increasing nested prefixes.")

    means, standard_errors = [], []
    for budget in budgets:
        prefix = samples[:, :budget]
        means.append(prefix.mean(dim=1))
        standard_errors.append(prefix.std(dim=1, correction=1) / math.sqrt(budget))
    return t.stack(means), t.stack(standard_errors)


def expected_gradients(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    forward_batch: int = 32_768,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return full-background base, clean Expected Gradients, and ordinary estimator SE."""
    _, samples, _ = _eg_validate_paths(x, background, indices, alpha)
    if samples < 2:
        raise ValueError(
            "Expected Gradients needs at least two samples to estimate standard error."
        )
    contributions = expected_gradient_samples(
        model, x, background, indices, alpha, forward_batch
    )
    means, standard_errors = summarize_expected_gradients(contributions, (samples,))
    model = _eg_model(model)
    with _eg_preserve_execution_state(model), t.no_grad():
        base = _eg_model_output(model, background).mean().detach().clone()
    return base, means[0], standard_errors[0]


def gauss_legendre_pairs(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    n_nodes: int,
    row_batch: int = 4,
    forward_batch: int = 32_768,
) -> Tensor:
    """Approximate every query-donor Integrated Gradients path by finite quadrature.

    ReLU path gradients jump, so finite-node quadrature remains a numerical control
    rather than exact path truth.
    """
    model = _eg_model(model)
    _eg_float_tensor("x", x, 2)
    _eg_float_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    n_nodes = _eg_positive_int("n_nodes", n_nodes)
    row_batch = _eg_positive_int("row_batch", row_batch)
    forward_batch = _eg_positive_int("forward_batch", forward_batch)

    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(n_nodes)
    nodes = t.tensor((raw_nodes + 1) / 2, dtype=t.float64)
    weights = t.tensor(raw_weights / 2, dtype=t.float64)
    pairs = t.empty((len(x), len(background), x.shape[1]), dtype=t.float64)

    for start in range(0, len(x), row_batch):
        query = x[start : start + row_batch]
        delta = query[:, None, :] - background[None, :, :]
        points = background[None, :, None, :] + nodes[None, None, :, None] * delta[:, :, None, :]
        packed = points.reshape(len(query), len(background) * n_nodes, x.shape[1])
        gradients = input_gradients(model, packed, forward_batch).reshape(points.shape)
        mean_gradient = (gradients * weights[None, None, :, None]).sum(dim=2)
        pairs[start : start + len(query)] = mean_gradient * delta
    return pairs


def noisy_expected_gradient_samples(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    noise: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Use x+noise in point and delta, targeting the noise-smoothed query output."""
    points, delta = interpolate_paths(x, background, indices, alpha, noise)
    gradients = input_gradients(model, points, forward_batch)
    return gradients * delta


def ranked_coalition_curves(
    values: Tensor,
    phi: Tensor,
    orders: Tensor | None = None,
) -> dict[str, Tensor]:
    """Gather raw insertion and deletion prefixes from complete coalition tables."""
    _eg_float_tensor("values", values, 2)
    _eg_float_tensor("phi", phi, 2)
    n_rows, size = values.shape
    n_features = size.bit_length() - 1
    if n_features < 1 or size != 2**n_features or phi.shape != (n_rows, n_features):
        raise ValueError("values must be (N,2**D) and phi must be (N,D), with D >= 1.")

    if orders is None:
        direction = t.where(values[:, -1] - values[:, 0] >= 0, 1.0, -1.0)[:, None]
        orders = t.argsort(direction * phi, dim=1, descending=True, stable=True)
    else:
        _eg_index_tensor("orders", orders, 2)
        if orders.shape != (n_rows, n_features):
            raise ValueError("orders must have shape (rows, features).")
        expected = t.arange(n_features, dtype=t.int64).expand(n_rows, n_features)
        if not t.equal(orders.sort(dim=1).values, expected):
            raise ValueError("Every order row must contain each feature exactly once.")
        orders = orders.clone()

    insertion_masks = t.zeros((n_rows, n_features + 1), dtype=t.int64)
    deletion_masks = t.full((n_rows, n_features + 1), size - 1, dtype=t.int64)
    for step in range(n_features):
        bit = 1 << orders[:, step]
        insertion_masks[:, step + 1] = insertion_masks[:, step] | bit
        deletion_masks[:, step + 1] = deletion_masks[:, step] ^ bit
    return {
        "orders": orders,
        "insertion_masks": insertion_masks,
        "deletion_masks": deletion_masks,
        "insertion": values.gather(1, insertion_masks),
        "deletion": values.gather(1, deletion_masks),
    }
