"""Visible exact methods for Shapley-Taylor and Faith-Shap interactions.

Games and coefficient vectors use ascending integer bitmasks on their final
axis. Entry zero is the intercept. The algebraic methods operate on complete
CPU-float64 games; ``masked_model_game`` builds such games from intact donor
rows without fitting or changing the supplied model.
"""

from __future__ import annotations

import contextlib
import math
import random
from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor, nn


__all__ = [
    "mobius_transform",
    "shapley_taylor_interactions",
    "faith_design_matrix",
    "faith_kernel_weights",
    "faith_shap_interactions",
    "faith_shap_closed_form",
    "reconstruct_game",
    "faithfulness_error",
    "masked_model_game",
    "contextual_pair_synergy",
]


def _interaction_n_players(n_players: int, *, minimum: int = 1) -> int:
    if type(n_players) is not int or n_players < minimum:
        qualifier = "at least two" if minimum == 2 else "positive"
        raise ValueError(f"n_players must be a {qualifier} integer.")
    return n_players


def _interaction_order(name: str, value: int, n_players: int) -> int:
    if type(value) is not int or not 1 <= value <= n_players:
        raise ValueError(f"{name} must be an integer from 1 through n_players.")
    return value


def _interaction_tensor(name: str, value: Tensor, n_players: int) -> Tensor:
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if value.ndim < 1 or value.shape[-1] != 2**n_players:
        raise ValueError(f"{name}'s final dimension must have length 2**n_players.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _interaction_counts(n_players: int) -> Tensor:
    return torch.tensor(
        [mask.bit_count() for mask in range(2**n_players)], dtype=torch.int64
    )


def _interaction_float_input(name: str, value: Tensor, ndim: int) -> Tensor:
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if value.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _interaction_model(model: nn.Module) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float64:
            raise TypeError("Every model parameter must be a CPU float64 tensor.")
        if not torch.isfinite(parameter).all():
            raise ValueError("Every model parameter must be finite.")
    for buffer in model.buffers():
        if buffer.device.type != "cpu":
            raise TypeError("Every model buffer must be on CPU.")
        if buffer.is_floating_point():
            if buffer.dtype != torch.float64:
                raise TypeError("Every floating model buffer must be float64.")
            if not torch.isfinite(buffer).all():
                raise ValueError("Every floating model buffer must be finite.")
    return model


def _interaction_model_output(model: nn.Module, inputs: Tensor) -> Tensor:
    output = model(inputs)
    if type(output) is not Tensor:
        raise TypeError("The model must return a torch.Tensor.")
    if output.shape == (len(inputs), 1):
        output = output[:, 0]
    elif output.shape != (len(inputs),):
        raise ValueError("The model must return one scalar output per input row.")
    if output.device.type != "cpu" or output.dtype != torch.float64:
        raise TypeError("The model must return CPU float64 outputs.")
    if not torch.isfinite(output).all():
        raise ValueError("The model returned a nonfinite value.")
    return output


@contextlib.contextmanager
def _interaction_preserve_model_state(model: nn.Module) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    default_dtype = torch.get_default_dtype()
    modes = [(module, module.training) for module in model.modules()]
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
        torch.random.set_rng_state(torch_state)
        torch.set_default_dtype(default_dtype)
        for module, training in modes:
            module.training = training
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
            for parameter, original, saved in gradients:
                if original is None:
                    parameter.grad = None
                else:
                    parameter.grad = original
                    original.copy_(saved)


def mobius_transform(game_values: Tensor, *, n_players: int) -> Tensor:
    """Return a full bitmask-indexed Möbius coefficient tensor."""
    n_players = _interaction_n_players(n_players)
    _interaction_tensor("game_values", game_values, n_players)
    mobius = game_values.clone()
    masks = torch.arange(2**n_players, dtype=torch.int64)
    for player in range(n_players):
        present = masks.bitwise_and(1 << player).ne(0)
        lower = masks[present].bitwise_xor(1 << player)
        mobius[..., present] -= mobius[..., lower]
    return mobius


def shapley_taylor_interactions(
    mobius: Tensor, *, n_players: int, order: int
) -> Tensor:
    """Allocate Möbius dividends into an efficient order-truncated expansion."""
    n_players = _interaction_n_players(n_players)
    order = _interaction_order("order", order, n_players)
    _interaction_tensor("mobius", mobius, n_players)
    counts = _interaction_counts(n_players)
    result = torch.zeros_like(mobius)
    result[..., 0] = mobius[..., 0]
    lower = (counts > 0) & (counts < order)
    result[..., lower] = mobius[..., lower]
    for interaction in range(1, 2**n_players):
        if interaction.bit_count() != order:
            continue
        for superset in range(interaction, 2**n_players):
            if superset & interaction == interaction:
                result[..., interaction] += mobius[..., superset] / math.comb(
                    superset.bit_count(), order
                )
    return result


def faith_design_matrix(
    *, n_players: int, max_order: int
) -> tuple[Tensor, Tensor]:
    """Return float64 X[coalition_mask, term] and int64 term_masks."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order("max_order", max_order, n_players)
    masks = torch.arange(2**n_players, dtype=torch.int64)
    counts = _interaction_counts(n_players)
    term_masks = masks[counts <= max_order]
    design = masks[:, None].bitwise_and(term_masks[None, :]).eq(term_masks).to(torch.float64)
    return design, term_masks


def faith_kernel_weights(*, n_players: int) -> Tensor:
    """Return normalized interior per-coalition weights; endpoints are zero."""
    n_players = _interaction_n_players(n_players, minimum=2)
    weights = torch.zeros(2**n_players, dtype=torch.float64)
    for mask in range(1, 2**n_players - 1):
        size = mask.bit_count()
        weights[mask] = (n_players - 1) / (
            math.comb(n_players, size) * size * (n_players - size)
        )
    return weights / weights.sum()


def faith_shap_interactions(
    game_values: Tensor, *, n_players: int, max_order: int
) -> Tensor:
    """Solve exact equality-constrained weighted least squares."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order("max_order", max_order, n_players)
    _interaction_tensor("game_values", game_values, n_players)
    if max_order == n_players:
        # All coalition monomials are present: the unique zero-error interpolant
        # is the Möbius expansion. Avoid an unnecessary ill-conditioned KKT solve.
        return mobius_transform(game_values, n_players=n_players)
    design, term_masks = faith_design_matrix(
        n_players=n_players, max_order=max_order
    )
    weights = faith_kernel_weights(n_players=n_players)
    interior = design[1:-1]
    weighted_design = weights[1:-1, None] * interior
    gram = interior.T @ weighted_design
    constraints = torch.stack((design[0], design[-1]))
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    kkt = torch.cat(
        (
            torch.cat((gram, constraints.T), dim=1),
            torch.cat((constraints, zeros), dim=1),
        ),
        dim=0,
    )

    values = game_values.reshape(-1, 2**n_players)
    weighted_targets = values[:, 1:-1] @ weighted_design
    endpoint_targets = torch.stack((values[:, 0], values[:, -1]), dim=1)
    right_hand_side = torch.cat((weighted_targets, endpoint_targets), dim=1)
    solution = torch.linalg.solve(kkt, right_hand_side.T).T
    result = torch.zeros_like(values)
    result[:, term_masks] = solution[:, : len(term_masks)]
    return result.reshape(game_values.shape)


def faith_shap_closed_form(
    mobius: Tensor, *, n_players: int, max_order: int
) -> Tensor:
    """Compute the independent primary-paper Möbius formula."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order("max_order", max_order, n_players)
    _interaction_tensor("mobius", mobius, n_players)
    result = torch.zeros_like(mobius)
    result[..., 0] = mobius[..., 0]
    for interaction in range(1, 2**n_players):
        size = interaction.bit_count()
        if size > max_order:
            continue
        correction = torch.zeros_like(mobius[..., 0])
        for superset in range(2**n_players):
            superset_size = superset.bit_count()
            if superset_size > max_order and superset & interaction == interaction:
                correction += (
                    math.comb(superset_size - 1, max_order)
                    / math.comb(superset_size + max_order - 1, max_order + size)
                    * mobius[..., superset]
                )
        scale = (
            (-1) ** (max_order - size)
            * size
            / (max_order + size)
            * math.comb(max_order, size)
        )
        result[..., interaction] = mobius[..., interaction] + scale * correction
    return result


def reconstruct_game(coefficients: Tensor, *, n_players: int) -> Tensor:
    """Apply the subset zeta transform to interaction coefficients."""
    n_players = _interaction_n_players(n_players)
    _interaction_tensor("coefficients", coefficients, n_players)
    values = coefficients.clone()
    masks = torch.arange(2**n_players, dtype=torch.int64)
    for player in range(n_players):
        present = masks.bitwise_and(1 << player).ne(0)
        lower = masks[present].bitwise_xor(1 << player)
        values[..., present] += values[..., lower]
    return values


def faithfulness_error(
    game_values: Tensor, coefficients: Tensor, *, n_players: int
) -> Tensor:
    """Return normalized Shapley-kernel interior MSE for each leading item."""
    n_players = _interaction_n_players(n_players, minimum=2)
    _interaction_tensor("game_values", game_values, n_players)
    _interaction_tensor("coefficients", coefficients, n_players)
    if game_values.shape != coefficients.shape:
        raise ValueError("game_values and coefficients must have the same shape.")
    residual = game_values - reconstruct_game(coefficients, n_players=n_players)
    weights = faith_kernel_weights(n_players=n_players)
    return (weights * residual.square()).sum(dim=-1)


@contextlib.contextmanager
def _interaction_game_batches(
    model: nn.Module,
    x: Tensor,
    reference: Tensor,
    *,
    coalition_batch_size: int = 4096,
) -> Iterator[tuple[int, int, int, Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]]]:
    """Supply validated row/mask/donor chunks inside a state-preserving context.

    Each chunk contains query rows, intact donor rows, present-feature flags,
    and flattened row/coalition output slots. It does not build hybrids, call
    the model, or average predictions; those are the learner's operations.
    """
    model = _interaction_model(model)
    _interaction_float_input("x", x, 2)
    if type(reference) is not Tensor:
        raise TypeError("reference must be a torch.Tensor.")
    if reference.ndim not in (1, 2):
        raise ValueError("reference must have shape [features] or [donors, features].")
    _interaction_float_input("reference", reference, reference.ndim)
    if x.shape[1] != reference.shape[-1]:
        raise ValueError("x and reference must have the same feature dimension.")
    if type(coalition_batch_size) is not int or coalition_batch_size < 1:
        raise ValueError("coalition_batch_size must be a positive integer.")

    bank = reference[None, :] if reference.ndim == 1 else reference
    rows, features = x.shape
    coalitions, donors = 2**features, len(bank)
    total = rows * coalitions * donors
    bits = 1 << torch.arange(features, dtype=torch.int64)

    def batches() -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
        for start in range(0, total, coalition_batch_size):
            flat = torch.arange(
                start, min(start + coalition_batch_size, total), dtype=torch.int64
            )
            donor_index = flat.remainder(donors)
            coalition_index = flat.div(donors, rounding_mode="floor").remainder(
                coalitions
            )
            row_index = flat.div(donors * coalitions, rounding_mode="floor")
            present = coalition_index[:, None].bitwise_and(bits[None, :]).ne(0)
            slots = row_index * coalitions + coalition_index
            yield x[row_index], bank[donor_index], present, slots

    with _interaction_preserve_model_state(model):
        model.eval()
        with torch.no_grad():
            yield rows, coalitions, donors, batches()


def masked_model_game(
    model: nn.Module,
    x: Tensor,
    reference: Tensor,
    *,
    coalition_batch_size: int = 4096,
) -> Tensor:
    """Return games using one reference [D] or an equal-weight bank [B,D]."""
    with _interaction_game_batches(
        model, x, reference, coalition_batch_size=coalition_batch_size
    ) as (rows, coalitions, donors, batches):
        sums = torch.zeros(rows * coalitions, dtype=torch.float64)
        for query_rows, donor_rows, present, slots in batches:
            inputs = torch.where(present, query_rows, donor_rows)
            output = _interaction_model_output(model, inputs)
            sums.index_add_(0, slots, output)
        return sums.div(donors).reshape(rows, coalitions).detach()


def contextual_pair_synergy(
    game_values: Tensor,
    pair: tuple[int, int],
    *,
    n_players: int,
    context_masks: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return sorted valid context masks and Delta_pair v(context)."""
    n_players = _interaction_n_players(n_players, minimum=2)
    _interaction_tensor("game_values", game_values, n_players)
    if (
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(player) is not int for player in pair)
    ):
        raise TypeError("pair must contain exactly two integer player indices.")
    first, second = pair
    if first == second:
        raise ValueError("pair must contain two different players.")
    if not 0 <= first < n_players or not 0 <= second < n_players:
        raise ValueError("pair players must lie in [0, n_players).")
    pair_mask = (1 << first) | (1 << second)

    if context_masks is None:
        contexts = torch.tensor(
            [
                mask
                for mask in range(2**n_players)
                if mask & pair_mask == 0
            ],
            dtype=torch.int64,
        )
    else:
        if type(context_masks) is not Tensor:
            raise TypeError("context_masks must be a torch.Tensor.")
        if context_masks.device.type != "cpu" or context_masks.dtype != torch.int64:
            raise TypeError("context_masks must be a CPU int64 tensor.")
        if context_masks.ndim != 1 or context_masks.numel() == 0:
            raise ValueError("context_masks must be a nonempty one-dimensional tensor.")
        if context_masks.min() < 0 or context_masks.max() >= 2**n_players:
            raise ValueError("Every context mask must index the complete game.")
        if context_masks.bitwise_and(pair_mask).ne(0).any():
            raise ValueError("Context masks must be disjoint from the requested pair.")
        contexts = context_masks.sort().values

    with_first = contexts.bitwise_or(1 << first)
    with_second = contexts.bitwise_or(1 << second)
    with_pair = contexts.bitwise_or(pair_mask)
    synergy = (
        game_values[..., with_pair]
        - game_values[..., with_first]
        - game_values[..., with_second]
        + game_values[..., contexts]
    )
    return contexts, synergy
