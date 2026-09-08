"""Visible learner methods for DeepSHAP on scalar affine/ReLU networks."""

from __future__ import annotations

import math

import torch as t
from torch import Tensor, nn


def _method_tensor(name: str, value: Tensor, ndim: int | None = None) -> Tensor:
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


def _layers(model: nn.Sequential) -> list[nn.Module]:
    """Validate the deliberately narrow graph supported by the taught rules."""
    if type(model) is not nn.Sequential:
        raise TypeError("model must be an exact nn.Sequential instance.")
    layers = list(model)
    if not layers:
        raise ValueError("model cannot be empty.")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("Every layer must be a distinct module; shared modules are unsupported.")
    for layer in layers:
        if type(layer) not in (nn.Linear, nn.ReLU):
            raise TypeError(
                f"Unsupported module {type(layer).__name__}; "
                "use exact nn.Linear and nn.ReLU modules."
            )
        if type(layer) is nn.ReLU and layer.inplace:
            raise ValueError(
                "In-place ReLU is unsupported because paired states must be preserved."
            )
        for parameter in layer.parameters(recurse=False):
            if parameter.device.type != "cpu" or parameter.dtype != t.float64:
                raise TypeError("Every model parameter must be a CPU float64 tensor.")
            if not t.isfinite(parameter).all():
                raise ValueError("Every model parameter must be finite.")
    if type(layers[-1]) is not nn.Linear or layers[-1].out_features != 1:
        raise ValueError("model must end with an exact scalar-output nn.Linear layer.")
    return layers


def _model_output(model: nn.Sequential, x: Tensor) -> Tensor:
    output = model(x)
    if output.shape != (len(x), 1):
        raise ValueError("The model must return one scalar output per input row.")
    if not t.isfinite(output).all():
        raise ValueError("The model returned a nonfinite value.")
    return output[:, 0]


def _check_widths(layers: list[nn.Module], n_features: int) -> None:
    width = n_features
    for index, layer in enumerate(layers):
        if type(layer) is nn.Linear:
            if layer.in_features != width:
                raise ValueError(
                    f"Layer {index} expects {layer.in_features} values, but the previous "
                    f"state has width {width}."
                )
            width = layer.out_features


def _epsilon(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("epsilon must be a finite nonnegative number.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("epsilon must be a finite nonnegative number.")
    return value


def paired_forward(
    model: nn.Sequential, x: Tensor, background: Tensor
) -> tuple[list[Tensor], list[Tensor]]:
    """Save the input and every layer output for queries and references."""
    layers = _layers(model)
    _method_tensor("x", x, 2)
    _method_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    _check_widths(layers, x.shape[1])

    with t.no_grad():
        x_value, b_value = x.clone(), background.clone()
        x_states, b_states = [x_value], [b_value]
        for layer in layers:
            x_value, b_value = layer(x_value), layer(b_value)
            x_states.append(x_value)
            b_states.append(b_value)
        if x_states[-1].shape != (len(x), 1) or b_states[-1].shape != (len(background), 1):
            raise ValueError("The model must return one scalar output per input row.")
        if not t.isfinite(x_states[-1]).all() or not t.isfinite(b_states[-1]).all():
            raise ValueError("The model returned a nonfinite value.")
    return x_states, b_states


def rescale_slope(zx: Tensor, zb: Tensor, epsilon: float = 1e-6) -> Tensor:
    """ReLU secants, using the reference derivative for small denominators."""
    _method_tensor("zx", zx)
    _method_tensor("zb", zb)
    epsilon = _epsilon(epsilon)
    try:
        zx, zb = t.broadcast_tensors(zx, zb)
    except RuntimeError as error:
        raise ValueError("zx and zb must be broadcast-compatible.") from error

    delta = zx - zb
    use_derivative = delta.eq(0) | delta.abs().lt(epsilon)
    safe_delta = t.where(use_derivative, t.ones_like(delta), delta)
    secant = (t.relu(zx) - t.relu(zb)) / safe_delta
    return t.where(use_derivative, (zb > 0).to(zb.dtype), secant)


@t.no_grad()
def rescale_multipliers(
    model: nn.Sequential,
    x: Tensor,
    background: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Propagate scalar-output Rescale multipliers to every input feature."""
    layers = _layers(model)
    epsilon = _epsilon(epsilon)
    x_states, b_states = paired_forward(model, x, background)
    multiplier = t.ones((len(x), len(background), 1), dtype=t.float64)

    for index in range(len(layers) - 1, -1, -1):
        layer = layers[index]
        if type(layer) is nn.Linear:
            multiplier = t.einsum("nro,oi->nri", multiplier, layer.weight)
        else:
            slope = rescale_slope(
                x_states[index][:, None, :],
                b_states[index][None, :, :],
                epsilon,
            )
            multiplier = multiplier * slope
    return multiplier


@t.no_grad()
def single_reference(
    model: nn.Sequential,
    x: Tensor,
    reference: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Explain each query relative to one complete reference vector."""
    _method_tensor("x", x, 2)
    _method_tensor("reference", reference, 1)
    if x.shape[1] != reference.shape[0]:
        raise ValueError("reference must contain one value for every input feature.")
    multiplier = rescale_multipliers(model, x, reference[None, :], epsilon)[:, 0]
    return multiplier * (x - reference)


@t.no_grad()
def background_deepshap(
    model: nn.Sequential,
    x: Tensor,
    background: Tensor,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Average complete reference explanations and return E[f(background)]."""
    multiplier = rescale_multipliers(model, x, background, epsilon)
    pair_values = multiplier * (x[:, None, :] - background[None, :, :])
    with t.no_grad():
        base = _model_output(model, background).mean()
    return base, pair_values.mean(dim=1)


def coalition_masks(d: int) -> Tensor:
    """Return all Boolean coalitions in integer order, with feature zero as bit zero."""
    if type(d) is not int or d < 1:
        raise ValueError("d must be a positive integer.")
    subset = t.arange(2**d, dtype=t.int64)
    feature = t.arange(d, dtype=t.int64)
    return subset[:, None].bitwise_and(1 << feature).ne(0)


def coalition_values(
    model: nn.Sequential,
    x: Tensor,
    background: Tensor,
    masks: Tensor,
    row_batch: int = 8,
    forward_batch: int = 65_536,
) -> Tensor:
    """Evaluate every supplied intact-donor hybrid, then average over donors."""
    layers = _layers(model)
    _method_tensor("x", x, 2)
    _method_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    _check_widths(layers, x.shape[1])
    if type(masks) is not Tensor or masks.device.type != "cpu" or masks.dtype != t.bool:
        raise TypeError("masks must be a CPU Boolean tensor.")
    if masks.ndim != 2 or masks.shape[1] != x.shape[1] or len(masks) == 0:
        raise ValueError("masks must have shape (nonempty coalitions, input features).")
    for name, size in (("row_batch", row_batch), ("forward_batch", forward_batch)):
        if type(size) is not int or size < 1:
            raise ValueError(f"{name} must be a positive integer.")

    result = t.empty((len(x), len(masks)), dtype=t.float64)
    with t.no_grad():
        for start in range(0, len(x), row_batch):
            query = x[start : start + row_batch]
            hybrids = t.where(
                masks[None, :, None, :],
                query[:, None, None, :],
                background[None, None, :, :],
            )
            flat = hybrids.reshape(-1, x.shape[1])
            predictions = t.cat(
                [
                    _model_output(model, flat[i : i + forward_batch])
                    for i in range(0, len(flat), forward_batch)
                ]
            )
            result[start : start + len(query)] = predictions.reshape(
                len(query), len(masks), len(background)
            ).mean(dim=2)
    return result


def exact_shapley(values: Tensor) -> Tensor:
    """Contract a complete (..., 2**D) game with exact Shapley weights."""
    _method_tensor("values", values)
    if values.ndim < 1:
        raise ValueError("values must have a final coalition axis.")
    size = values.shape[-1]
    d = size.bit_length() - 1
    if d < 1 or size != 2**d:
        raise ValueError("The final values axis must contain all 2**D coalitions for D >= 1.")

    result = t.zeros((*values.shape[:-1], d), dtype=t.float64)
    subset_ids = t.arange(size, dtype=t.int64)
    for feature in range(d):
        absent = subset_ids[subset_ids.bitwise_and(1 << feature).eq(0)]
        weights = t.tensor(
            [1 / (d * math.comb(d - 1, int(subset).bit_count())) for subset in absent],
            dtype=t.float64,
        )
        marginal = values[..., absent | (1 << feature)] - values[..., absent]
        result[..., feature] = (marginal * weights).sum(dim=-1)
    return result


def reveal_cancel_groups(
    activation,
    reference: Tensor,
    positive: Tensor,
    negative: Tensor,
) -> tuple[Tensor, Tensor]:
    """Two-player Shapley allocation of positive/negative groups through one activation."""
    if not callable(activation):
        raise TypeError("activation must be callable.")
    for name, value in (("reference", reference), ("positive", positive), ("negative", negative)):
        _method_tensor(name, value)
    try:
        reference, positive, negative = t.broadcast_tensors(reference, positive, negative)
    except RuntimeError as error:
        raise ValueError(
            "reference, positive and negative must be broadcast-compatible."
        ) from error
    if (positive < 0).any() or (negative > 0).any():
        raise ValueError("positive must be nonnegative and negative must be nonpositive.")

    def activate(value: Tensor) -> Tensor:
        output = activation(value)
        if type(output) is not Tensor or output.shape != value.shape:
            raise TypeError("activation must return a tensor with the same shape as its input.")
        if output.device.type != "cpu" or output.dtype != t.float64 or not t.isfinite(output).all():
            raise ValueError("activation must return finite CPU float64 values.")
        return output

    base = activate(reference)
    positive_first = activate(reference + positive)
    negative_first = activate(reference + negative)
    together = activate(reference + positive + negative)
    positive_value = 0.5 * (positive_first - base + together - negative_first)
    negative_value = 0.5 * (negative_first - base + together - positive_first)
    return positive_value, negative_value


def ranked_replacement_curves(
    values: Tensor,
    phi: Tensor,
    orders: Tensor | None = None,
) -> dict[str, Tensor]:
    """Read insertion/deletion prefixes from a complete per-row coalition table."""
    _method_tensor("values", values, 2)
    _method_tensor("phi", phi, 2)
    n, size = values.shape
    d = size.bit_length() - 1
    if d < 1 or size != 2**d or phi.shape != (n, d):
        raise ValueError("values must be (N,2**D) and phi must be (N,D), with D >= 1.")

    if orders is None:
        direction = t.where(values[:, -1] - values[:, 0] >= 0, 1.0, -1.0)[:, None]
        orders = t.argsort(direction * phi, dim=1, descending=True, stable=True)
    else:
        if type(orders) is not Tensor or orders.device.type != "cpu" or orders.dtype != t.int64:
            raise TypeError("orders must be a CPU int64 tensor.")
        if orders.shape != (n, d):
            raise ValueError("orders must have shape (rows, features).")
        expected = t.arange(d, dtype=t.int64).expand(n, d)
        if not t.equal(orders.sort(dim=1).values, expected):
            raise ValueError("Every order row must contain each feature exactly once.")
        orders = orders.clone()

    insertion_masks = t.zeros((n, d + 1), dtype=t.int64)
    deletion_masks = t.full((n, d + 1), size - 1, dtype=t.int64)
    for step in range(d):
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


def gradient_delta(model: nn.Sequential, x: Tensor, background: Tensor) -> Tensor:
    """Point gradient at x times x minus the mean background; not GradientSHAP."""
    layers = _layers(model)
    _method_tensor("x", x, 2)
    _method_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    _check_widths(layers, x.shape[1])

    query = x.detach().clone().requires_grad_(True)
    with t.enable_grad():
        output = _model_output(model, query)
        gradient = t.autograd.grad(output.sum(), query)[0]
    if not t.isfinite(gradient).all():
        raise ValueError("The model produced a nonfinite input gradient.")
    return (gradient * (query - background.mean(dim=0))).detach()
