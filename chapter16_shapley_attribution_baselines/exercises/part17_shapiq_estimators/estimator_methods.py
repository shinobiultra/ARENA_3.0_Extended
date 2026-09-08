"""Visible teaching implementations of SHAP-IQ and pairwise KernelSHAP-IQ.

These functions consume only queried values. Complete tables belong exclusively
to the explicitly named exact reference. Float64 CPU keeps the small linear
systems inspectable; sampling uses a local generator and never enumerates 2**n.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import torch as t
from torch import Tensor


def _players(n: int) -> int:
    if type(n) is not int or not 4 <= n <= 20:
        raise ValueError("n_players must be an integer between 4 and 20.")
    return n


def _masks(masks: Tensor, n: int, name: str = "masks") -> Tensor:
    if not isinstance(masks, Tensor) or masks.device.type != "cpu" or masks.dtype != t.int64:
        raise TypeError(f"{name} must be a CPU int64 tensor.")
    if masks.ndim != 1 or not len(masks) or bool(((masks < 0) | (masks >= 2**n)).any()):
        raise ValueError(f"{name} must be a nonempty vector of valid coalition masks.")
    return masks


def _values(values: Tensor, length: int, name: str) -> Tensor:
    if not isinstance(values, Tensor) or values.device.type != "cpu" or values.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if values.ndim < 1 or values.shape[-1] != length or not bool(t.isfinite(values).all()):
        raise ValueError(f"{name} must be finite and have final axis {length}.")
    return values


def interaction_masks(n_players: int, order: int = 2) -> Tensor:
    """Return lexicographically ordered feature sets as masks (provided indexing)."""
    n = _players(n_players)
    if type(order) is not int or not 1 <= order <= n:
        raise ValueError("order must lie between one and n_players.")
    return t.tensor(
        [sum(1 << i for i in members) for members in itertools.combinations(range(n), order)],
        dtype=t.int64,
    )


def boundary_masks(n_players: int) -> Tensor:
    """Query sizes 0, 1, n-1 and n exactly: 2*n+2 unique values."""
    n = _players(n_players)
    full = 2**n - 1
    return t.tensor(sorted({0, full, *(1 << i for i in range(n)), *(full ^ (1 << i) for i in range(n))}), dtype=t.int64)


def coalition_design(masks: Tensor, *, n_players: int) -> Tensor:
    """Return X[queried coalition, feature] in float64."""
    n = _players(n_players)
    _masks(masks, n)
    bits = 1 << t.arange(n, dtype=t.int64)
    return masks[:, None].bitwise_and(bits).ne(0).to(t.float64)


def sii_query_weights(masks: Tensor, *, n_players: int, order: int = 2) -> Tensor:
    """The SHAP-IQ gamma coefficient of each queried value in every order-s SII.

    If k=|T intersection S|, gamma(T,S)=(-1)**(s-k) /
    ((n-s+1) * choose(n-s, |T|-k)). No model evaluations occur here.
    """
    n = _players(n_players)
    terms = interaction_masks(n, order)
    design = coalition_design(masks, n_players=n)
    term_design = coalition_design(terms, n_players=n)
    overlap = (design @ term_design.T).to(t.int64)
    outside = design.sum(-1).to(t.int64)[:, None] - overlap
    denominator = t.tensor([(n - order + 1) * math.comb(n - order, r) for r in range(n - order + 1)], dtype=t.float64)
    sign = t.where((order - overlap) % 2 == 0, 1.0, -1.0).to(t.float64)
    return sign / denominator[outside]


def sample_coalitions(
    *, n_players: int, draws: int, seed: int, proposal: str = "kernel"
) -> tuple[Tensor, Tensor]:
    """Draw iid interior coalitions, retaining duplicate draws and per-set p(T).

    Draw size first, then a uniformly random subset of that size. Kernel means
    p(T) proportional to 1/choose(n-2, |T|-1), restricted to 2..n-2.
    Uniform means uniform over individual sets in that same support.
    """
    n = _players(n_players)
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("draws >= 2 and a nonnegative integer seed are required.")
    if proposal not in ("kernel", "uniform"):
        raise ValueError("proposal must be 'kernel' or 'uniform'.")
    sizes = t.arange(2, n - 1, dtype=t.int64)
    if proposal == "kernel":
        size_mass = 1.0 / (sizes.to(t.float64) * (n - sizes))
    else:
        size_mass = t.tensor([math.comb(n, int(k)) for k in sizes], dtype=t.float64)
    size_probability = size_mass / size_mass.sum()
    generator = t.Generator(device="cpu").manual_seed(seed)
    chosen = t.multinomial(size_probability, draws, replacement=True, generator=generator)
    selected_sizes = sizes[chosen]
    # Each row's random permutation gives a uniform subset, without 2**n storage.
    permutations = t.rand((draws, n), generator=generator, dtype=t.float64).argsort(dim=-1)
    take = t.arange(n)[None, :] < selected_sizes[:, None]
    masks = ((1 << permutations) * take).sum(-1)
    combinations = t.tensor([math.comb(n, int(k)) for k in sizes], dtype=t.float64)
    probabilities = (size_probability / combinations)[chosen]
    return masks, probabilities


def _queried_inputs(boundary_values, sampled_values, sampled_masks, probabilities, n):
    edges = boundary_masks(n)
    _values(boundary_values, len(edges), "boundary_values")
    _masks(sampled_masks, n, "sampled_masks")
    _values(sampled_values, len(sampled_masks), "sampled_values")
    _values(probabilities, len(sampled_masks), "probabilities")
    if probabilities.ndim != 1 or not bool(((probabilities > 0) & (probabilities <= 1)).all()):
        raise ValueError("probabilities must be a positive per-draw vector bounded by one.")
    sizes = coalition_design(sampled_masks, n_players=n).sum(-1)
    if len(sampled_masks) < 2 or bool(((sizes < 2) | (sizes > n - 2)).any()):
        raise ValueError("At least two draws from sizes 2..n-2 are required.")
    if boundary_values.shape[:-1] != sampled_values.shape[:-1]:
        raise ValueError("Boundary and sampled values must share their leading axes.")
    return edges


def shapiq_estimate(
    boundary_values: Tensor,
    sampled_values: Tensor,
    sampled_masks: Tensor,
    probabilities: Tensor,
    *, n_players: int, order: int = 2,
) -> tuple[Tensor, Tensor]:
    """Return the SHAP-IQ point estimate and iid sample standard error.

    The first boundary is empty. Subtract its value from every queried value:
    this preserves offset invariance for each realized sample, not just in mean.
    SE concerns iid draws from a fixed game, not uncertainty about the model/data.
    """
    n = _players(n_players)
    edges = _queried_inputs(boundary_values, sampled_values, sampled_masks, probabilities, n)
    baseline = boundary_values[..., :1]
    exact = (boundary_values - baseline) @ sii_query_weights(edges, n_players=n, order=order)
    weights = sii_query_weights(sampled_masks, n_players=n, order=order) / probabilities[:, None]
    contributions = (sampled_values - baseline)[..., :, None] * weights
    estimate = exact + contributions.mean(dim=-2)
    standard_error = (contributions.var(dim=-2, correction=1) / len(sampled_masks)).sqrt()
    return estimate, standard_error


def constrained_wls(
    design: Tensor, values: Tensor, weights: Tensor,
    constraints: Tensor, targets: Tensor, *, rtol: float = 1e-12,
) -> dict[str, Any]:
    """Minimize ||sqrt(W)(X beta-y)|| subject to C beta=d, using a nullspace.

    Rank-deficient sampled designs return the minimum-norm free solution and an
    explicit identified=False flag. They never silently acquire a ridge penalty.
    """
    if design.ndim != 2 or constraints.ndim != 2 or constraints.shape[1] != design.shape[1]:
        raise ValueError("design and constraints must have the same coefficient axis.")
    rows, columns = design.shape
    for name, value, length in (("design", design, columns), ("values", values, rows), ("weights", weights, rows), ("constraints", constraints, columns), ("targets", targets, len(constraints))):
        _values(value, length, name)
    if weights.ndim != 1 or bool((weights < 0).any()) or not bool((weights > 0).any()):
        raise ValueError("weights must be nonnegative with a positive entry.")
    if values.shape[:-1] != targets.shape[:-1] or not 0 < rtol < 1:
        raise ValueError("Target batches must match; rtol must lie between zero and one.")
    _, singular, vh = t.linalg.svd(constraints, full_matrices=True)
    rank = int((singular > rtol * singular.max()).sum()) if len(singular) else 0
    particular = targets @ t.linalg.pinv(constraints, rtol=rtol).T
    error = particular @ constraints.T - targets
    scale = max(1.0, float(targets.abs().max()))
    if float(error.abs().max()) > 100 * rtol * scale:
        raise ValueError("The equality constraints are inconsistent.")
    nullspace = vh[rank:].T
    free = nullspace.shape[1]
    if free:
        full_weighted_design = weights.sqrt()[:, None] * design
        weighted_design = full_weighted_design @ nullspace
        residual = (values - particular @ design.T) * weights.sqrt()
        u, singular_values, free_vh = t.linalg.svd(weighted_design, full_matrices=False)
        # Compare with the original design's scale: a theoretically zero X@Z can
        # contain roundoff. Scaling only by X@Z would invert that numerical noise.
        cutoff = rtol * t.linalg.matrix_norm(full_weighted_design, ord=2)
        retained = singular_values > cutoff
        free_solution = ((residual @ u[:, retained]) / singular_values[retained]) @ free_vh[retained]
        coefficients = particular + free_solution @ nullspace.T
        free_rank = int(retained.sum())
    else:
        coefficients, free_rank = particular, 0
        singular_values = t.empty(0, dtype=t.float64)
    return {
        "coefficients": coefficients,
        "constraint_rank": rank,
        "free_parameters": free,
        "free_rank": free_rank,
        "identified": free_rank == free,
        "singular_values": singular_values,
        "constraint_residual": coefficients @ constraints.T - targets,
    }


def pair_residual_design(masks: Tensor, *, n_players: int) -> Tensor:
    """Bernoulli pair design: -1/2 if exactly one pair member is present, else 0."""
    design = coalition_design(masks, n_players=n_players)
    pairs = t.tensor(list(itertools.combinations(range(n_players), 2)), dtype=t.int64)
    left, right = design[:, pairs[:, 0]], design[:, pairs[:, 1]]
    return left * right - (left + right) / 2


def kernel_weights(masks: Tensor, *, n_players: int, order: int) -> Tensor:
    """Finite per-coalition KernelSHAP-IQ weights; boundary markers are zero."""
    n = _players(n_players)
    if order not in (1, 2):
        raise ValueError("This lesson proves and implements orders one and two only.")
    sizes = coalition_design(masks, n_players=n).sum(-1).to(t.int64)
    return t.tensor([1 / math.comb(n - 2 * order, int(k) - order) if order <= k <= n - order else 0.0 for k in sizes], dtype=t.float64)


def kernelshap_iq(
    boundary_values: Tensor, sampled_values: Tensor,
    sampled_masks: Tensor, probabilities: Tensor,
    *, n_players: int,
) -> dict[str, Any]:
    """Sequential pairwise KernelSHAP-IQ with the exact boundary-weight limit.

    First fit Shapley main effects with their endpoint sum constraint. Next fit
    the pair design to that residual. Equal infinite weights on singleton and
    complement rows imply C beta = v0(full)-v0(single)-v0(complement), where C is
    feature/pair incidence. This minimizes boundary error; it does NOT set every
    boundary residual to zero. Interior WLS resolves the remaining directions.
    """
    n = _players(n_players)
    edges = _queried_inputs(boundary_values, sampled_values, sampled_masks, probabilities, n)
    baseline = boundary_values[..., :1]
    edge_values, draw_values = boundary_values - baseline, sampled_values - baseline
    masks = t.cat((edges, sampled_masks))
    values = t.cat((edge_values, draw_values), dim=-1)
    adjustment = 1.0 / (len(sampled_masks) * probabilities)
    first_weights = kernel_weights(masks, n_players=n, order=1) * t.cat((t.ones(len(edges), dtype=t.float64), adjustment))
    design = coalition_design(masks, n_players=n)
    first = constrained_wls(design, values, first_weights, t.ones((1, n), dtype=t.float64), edge_values[..., -1:])
    residual = draw_values - first["coefficients"] @ coalition_design(sampled_masks, n_players=n).T
    single_positions = t.searchsorted(edges, 1 << t.arange(n))
    complement_positions = t.searchsorted(edges, (2**n - 1) ^ (1 << t.arange(n)))
    targets = edge_values[..., -1:] - edge_values[..., single_positions] - edge_values[..., complement_positions]
    incidence = coalition_design(interaction_masks(n, 2), n_players=n).T
    second = constrained_wls(pair_residual_design(sampled_masks, n_players=n), residual, kernel_weights(sampled_masks, n_players=n, order=2) * adjustment, incidence, targets)
    return {"shapley": first["coefficients"], "pairs": second["coefficients"], "first_stage": first, "pair_stage": second}


def exact_sii(game_values: Tensor, *, n_players: int, order: int = 2) -> Tensor:
    """Independent complete-table derivative reference; never used by an estimator."""
    n = _players(n_players)
    if n > 12:
        raise ValueError("Exact reference is limited to n <= 12.")
    _values(game_values, 2**n, "game_values")
    scores = []
    for term in interaction_masks(n, order).tolist():
        score = t.zeros_like(game_values[..., 0])
        subsets = [part for part in range(2**n) if part & term == part]
        for context in range(2**n):
            if context & term:
                continue
            derivative = sum(((-1) ** (order - part.bit_count())) * game_values[..., context | part] for part in subsets)
            weight = 1 / ((n - order + 1) * math.comb(n - order, context.bit_count()))
            score = score + weight * derivative
        scores.append(score)
    return t.stack(scores, dim=-1)


def query_sampled_game(
    game: Callable[[Tensor], Tensor], sampled_masks: Tensor, *, n_players: int,
) -> dict[str, Any]:
    """Query each required coalition once while preserving iid duplicate draws."""
    n = _players(n_players)
    _masks(sampled_masks, n)
    edges = boundary_masks(n)
    requested = t.cat((edges, sampled_masks))
    unique, inverse = t.unique(requested, sorted=True, return_inverse=True)
    observed = game(unique)
    _values(observed, len(unique), "game output")
    restored = observed[..., inverse]
    return {
        "boundary_values": restored[..., :len(edges)],
        "sampled_values": restored[..., len(edges):],
        "queried_masks": unique,
        "queried_values": observed,
        "draw_requests": len(sampled_masks),
        "boundary_queries": len(edges),
        "total_requests": len(requested),
        "unique_queries": len(unique),
    }
