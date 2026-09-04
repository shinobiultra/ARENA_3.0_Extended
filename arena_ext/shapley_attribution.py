"""Exact Shapley-value helpers for attribution-baseline notebooks."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch as t


Coalition = frozenset[int]


@dataclass(frozen=True)
class ShapleyEfficiencyReport:
    shapley_sum: float
    total_value_delta: float
    efficiency_error: float
    satisfies_efficiency: bool


@dataclass(frozen=True)
class PermutationParityReport:
    max_abs_error: float
    matches_exact: bool


@dataclass(frozen=True)
class InteractionGapReport:
    shapley_total: float
    leave_one_out_total: float
    overcount: float
    detects_interaction_overcount: bool


@dataclass(frozen=True)
class KernelSHAPApproximationReport:
    shapley_values: t.Tensor
    exact_values: t.Tensor
    max_abs_error: float
    approximates_exact: bool


@dataclass(frozen=True)
class PartitionSHAPReport:
    group_values: t.Tensor
    player_values: t.Tensor
    exact_values: t.Tensor
    max_abs_error: float
    recovers_exact: bool


@dataclass(frozen=True)
class PairwiseInteractionReport:
    pair_interactions: t.Tensor
    target_pair: tuple[int, int]
    target_interaction: float
    max_spurious_interaction: float
    recovers_interaction: bool


@dataclass(frozen=True)
class ShapiqInteractionParityReport:
    exact_pair_interactions: t.Tensor
    shapiq_pair_interactions: t.Tensor
    max_abs_error: float
    matches_shapiq: bool
    shapiq_available: bool


@dataclass(frozen=True)
class TokenShapleySamplingReport:
    tokens: tuple[str, ...]
    exact_values: t.Tensor
    sampled_values: t.Tensor
    max_abs_error: float
    top_token: str
    sampled_top_token: str
    rank_matches: bool
    approximates_exact: bool


@dataclass(frozen=True)
class TokenBaselineReport:
    full_score: float
    baseline_score: float
    total_delta: float
    shapley_sum: float
    efficiency_error: float
    satisfies_efficiency: bool


@dataclass(frozen=True)
class VLMModalitySHAPReport:
    modality_values: t.Tensor
    baseline_score: float
    image_only_score: float
    text_only_score: float
    full_score: float
    synergy: float
    detects_synergy: bool
    satisfies_efficiency: bool


@dataclass(frozen=True)
class VLMRegionSHAPReport:
    region_values: t.Tensor
    region_names: tuple[str, ...]
    target_region: str
    target_value: float
    max_background_value: float
    localizes_target: bool
    satisfies_efficiency: bool


@dataclass(frozen=True)
class ShapleyPatchingComparisonReport:
    shapley_values: t.Tensor
    patching_effects: t.Tensor
    max_abs_error: float
    shapley_top_feature: int
    patching_top_feature: int
    top_feature_agrees: bool
    agrees_with_shapley: bool


@dataclass(frozen=True)
class InteractionPatchingFailureReport:
    shapley_values: t.Tensor
    patching_effects: t.Tensor
    shapley_total: float
    patching_total: float
    overcount: float
    documents_overcount: bool


@dataclass(frozen=True)
class DataShapleyReport:
    exact_values: t.Tensor
    full_utility: float
    baseline_utility: float
    harmful_index: int
    harmful_value: float
    helpful_index: int
    helpful_value: float
    harmful_removal_delta: float
    helpful_addition_utility: float
    deletion_test_passes: bool
    addition_test_passes: bool


@dataclass(frozen=True)
class MonteCarloDataShapleyReport:
    exact_values: t.Tensor
    sampled_values: t.Tensor
    max_abs_error: float
    top_example_matches: bool
    harmful_example_matches: bool
    approximates_exact: bool


@dataclass(frozen=True)
class InRunDataShapleyReport:
    exact_values: t.Tensor
    in_run_scores: t.Tensor
    pearson_correlation: float
    harmful_index: int
    in_run_harmful_index: int
    helpful_index: int
    in_run_helpful_index: int
    identifies_harmful: bool
    identifies_helpful: bool
    correlates_with_exact: bool


@dataclass(frozen=True)
class AttributionAgreementReport:
    shapley_values: t.Tensor
    patching_effects: t.Tensor
    mechanistic_scores: t.Tensor
    spearman_correlation: float
    topk_overlap: float
    shapley_top_feature: int
    mechanistic_top_feature: int
    deletion_drop: float
    random_baseline_drop: float
    agrees_with_mechanistic: bool


@dataclass(frozen=True)
class InteractionAgreementReport:
    shapley_values: t.Tensor
    pair_interactions: t.Tensor
    mechanistic_pair: tuple[int, int]
    recovered_pair_interaction: float
    max_single_feature_value: float
    ordinary_shapley_misses: bool
    interaction_recovers_pair: bool


def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every coalition for `num_players` players."""

    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    players = range(num_players)
    coalitions = []
    for size in range(num_players + 1):
        coalitions.extend(frozenset(group) for group in itertools.combinations(players, size))
    return tuple(coalitions)


def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    """Normalize coalition keys and require a complete value table."""

    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")
    return values


def coalition_values_from_function(
    num_players: int,
    value_fn: Callable[[Coalition], float],
) -> dict[Coalition, float]:
    """Evaluate a set function on every coalition."""

    return {coalition: float(value_fn(coalition)) for coalition in all_coalitions(num_players)}


def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values by summing weighted marginal effects."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    factorial = math.factorial
    denominator = factorial(num_players)
    shapley = t.zeros(num_players, dtype=t.float64)
    for player in range(num_players):
        others = [item for item in range(num_players) if item != player]
        for size in range(num_players):
            weight = factorial(size) * factorial(num_players - size - 1) / denominator
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                with_player = coalition | {player}
                shapley[player] += weight * (values[with_player] - values[coalition])
    return shapley


def permutation_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute Shapley values by averaging exact permutation marginals."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    totals = t.zeros(num_players, dtype=t.float64)
    permutations = tuple(itertools.permutations(range(num_players)))
    for ordering in permutations:
        coalition: Coalition = frozenset()
        for player in ordering:
            with_player = coalition | {player}
            totals[player] += values[with_player] - values[coalition]
            coalition = with_player
    return totals / len(permutations)


def sampled_permutation_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    num_samples: int,
    seed: int = 0,
) -> t.Tensor:
    """Estimate Shapley values by sampling random player orderings."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    rng = random.Random(seed)
    totals = t.zeros(num_players, dtype=t.float64)
    players = tuple(range(num_players))
    for _ in range(num_samples):
        ordering = rng.sample(players, k=num_players)
        coalition: Coalition = frozenset()
        for player in ordering:
            with_player = coalition | {player}
            totals[player] += values[with_player] - values[coalition]
            coalition = with_player
    return totals / num_samples


def leave_one_out_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Return full-coalition leave-one-out effects for each player."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    full = frozenset(range(num_players))
    scores = []
    for player in range(num_players):
        scores.append(values[full] - values[full - {player}])
    return t.tensor(scores, dtype=t.float64)


def activation_patching_effects(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Model full-activation patch effects as full-minus-ablated values."""

    return leave_one_out_values(coalition_values, num_players=num_players)


def shapley_patching_comparison_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> ShapleyPatchingComparisonReport:
    """Compare exact Shapley values to full-minus-ablated patch effects."""

    shapley = exact_shapley_values(coalition_values, num_players=num_players)
    patching = activation_patching_effects(coalition_values, num_players=num_players)
    max_error = float((shapley - patching).abs().max().item())
    shapley_top = int(shapley.argmax().item())
    patching_top = int(patching.argmax().item())
    return ShapleyPatchingComparisonReport(
        shapley_values=shapley,
        patching_effects=patching,
        max_abs_error=max_error,
        shapley_top_feature=shapley_top,
        patching_top_feature=patching_top,
        top_feature_agrees=shapley_top == patching_top,
        agrees_with_shapley=max_error <= tolerance,
    )


def interaction_patching_failure_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    min_overcount: float = 0.5,
) -> InteractionPatchingFailureReport:
    """Document patching overcount on interaction-heavy coalition games."""

    shapley = exact_shapley_values(coalition_values, num_players=num_players)
    patching = activation_patching_effects(coalition_values, num_players=num_players)
    shapley_total = float(shapley.sum().item())
    patching_total = float(patching.sum().item())
    overcount = patching_total - shapley_total
    return InteractionPatchingFailureReport(
        shapley_values=shapley,
        patching_effects=patching,
        shapley_total=shapley_total,
        patching_total=patching_total,
        overcount=overcount,
        documents_overcount=overcount >= min_overcount,
    )


def toy_data_shapley_problem() -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Return a tiny one-dimensional training/validation attribution problem."""

    train_x = t.ones(4, 1, dtype=t.float64)
    train_y = t.tensor([1.0, 1.0, 1.0, -1.0], dtype=t.float64)
    val_x = t.ones(1, 1, dtype=t.float64)
    val_y = t.ones(1, dtype=t.float64)
    return train_x, train_y, val_x, val_y


def one_step_linear_utility(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    coalition: Coalition,
    *,
    learning_rate: float = 0.5,
) -> float:
    """Utility from one full-batch gradient step on the selected examples."""

    train_x = train_x.double()
    train_y = train_y.double()
    val_x = val_x.double()
    val_y = val_y.double()
    weight = t.zeros(train_x.shape[1], dtype=t.float64)
    baseline_loss = ((val_x @ weight - val_y) ** 2).mean()
    if not coalition:
        return 0.0
    indices = t.tensor(sorted(coalition), dtype=t.long)
    selected_x = train_x[indices]
    selected_y = train_y[indices]
    train_error = selected_x @ weight - selected_y
    gradient = (2 * train_error.unsqueeze(-1) * selected_x).mean(dim=0)
    updated_weight = weight - learning_rate * gradient
    updated_loss = ((val_x @ updated_weight - val_y) ** 2).mean()
    return float((baseline_loss - updated_loss).item())


def data_coalition_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    learning_rate: float = 0.5,
) -> dict[Coalition, float]:
    """Build a complete training-example coalition table."""

    num_examples = int(train_x.shape[0])
    return coalition_values_from_function(
        num_examples,
        lambda coalition: one_step_linear_utility(
            train_x,
            train_y,
            val_x,
            val_y,
            coalition,
            learning_rate=learning_rate,
        ),
    )


def exact_data_shapley_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    learning_rate: float = 0.5,
) -> t.Tensor:
    """Compute exact Data Shapley values for a tiny training run."""

    values = data_coalition_values(
        train_x,
        train_y,
        val_x,
        val_y,
        learning_rate=learning_rate,
    )
    return exact_shapley_values(values, num_players=int(train_x.shape[0]))


def monte_carlo_data_shapley_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    learning_rate: float = 0.5,
    num_samples: int = 256,
    seed: int = 0,
) -> t.Tensor:
    """Estimate Data Shapley by sampled training-example permutations."""

    values = data_coalition_values(
        train_x,
        train_y,
        val_x,
        val_y,
        learning_rate=learning_rate,
    )
    return sampled_permutation_shapley_values(
        values,
        num_players=int(train_x.shape[0]),
        num_samples=num_samples,
        seed=seed,
    )


def in_run_first_order_data_scores(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
) -> t.Tensor:
    """Compute one-run first-order gradient-dot data scores from initialization."""

    train_x = train_x.double()
    train_y = train_y.double()
    val_x = val_x.double()
    val_y = val_y.double()
    weight = t.zeros(train_x.shape[1], dtype=t.float64)
    val_error = val_x @ weight - val_y
    val_gradient = (2 * val_error.unsqueeze(-1) * val_x).mean(dim=0)
    train_error = train_x @ weight - train_y
    train_gradients = 2 * train_error.unsqueeze(-1) * train_x
    return train_gradients @ val_gradient


def pearson_correlation(first: t.Tensor, second: t.Tensor) -> float:
    """Return Pearson correlation for two nonconstant vectors."""

    first = first.double().flatten()
    second = second.double().flatten()
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((first_centered @ second_centered / denominator).item())


def average_ranks(scores: t.Tensor) -> t.Tensor:
    """Return ascending average ranks, with ties assigned their mean rank."""

    scores = scores.double().flatten()
    ranks = t.empty_like(scores)
    sorted_indices = t.argsort(scores, stable=True)
    sorted_scores = scores[sorted_indices]
    start = 0
    while start < int(scores.numel()):
        end = start + 1
        while end < int(scores.numel()) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + end - 1) / 2
        ranks[sorted_indices[start:end]] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(first: t.Tensor, second: t.Tensor) -> float:
    """Return Spearman rank correlation with average ranks for ties."""

    return pearson_correlation(average_ranks(first), average_ranks(second))


def topk_overlap_fraction(first: t.Tensor, second: t.Tensor, *, k: int) -> float:
    """Return overlap fraction between the top-k indices of two score vectors."""

    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, int(first.numel()), int(second.numel()))
    first_top = set(t.topk(first.double().flatten(), k=k).indices.tolist())
    second_top = set(t.topk(second.double().flatten(), k=k).indices.tolist())
    return len(first_top & second_top) / k


def attribution_agreement_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    mechanistic_scores: t.Tensor,
    num_players: int,
    topk: int = 1,
    min_correlation: float = 0.99,
) -> AttributionAgreementReport:
    """Compare Shapley and patching scores to known mechanistic scores."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    shapley = exact_shapley_values(values, num_players=num_players)
    patching = activation_patching_effects(values, num_players=num_players)
    mech = mechanistic_scores.flatten().double()
    if int(mech.numel()) != num_players:
        raise ValueError("mechanistic_scores must have one score per player.")
    correlation = spearman_rank_correlation(shapley, mech)
    overlap = topk_overlap_fraction(shapley, mech, k=topk)
    shapley_top = int(shapley.argmax().item())
    mech_top = int(mech.argmax().item())
    full = frozenset(range(num_players))
    deletion_drop = values[full] - values[full - {shapley_top}]
    random_candidates = [player for player in range(num_players) if player != shapley_top]
    random_baseline = (
        sum(values[full] - values[full - {player}] for player in random_candidates)
        / len(random_candidates)
        if random_candidates
        else 0.0
    )
    return AttributionAgreementReport(
        shapley_values=shapley,
        patching_effects=patching,
        mechanistic_scores=mech,
        spearman_correlation=correlation,
        topk_overlap=overlap,
        shapley_top_feature=shapley_top,
        mechanistic_top_feature=mech_top,
        deletion_drop=deletion_drop,
        random_baseline_drop=random_baseline,
        agrees_with_mechanistic=correlation >= min_correlation and overlap == 1.0,
    )


def xor_game() -> dict[Coalition, float]:
    """Return a two-player XOR/parity value table."""

    return coalition_values_from_function(2, lambda coalition: len(coalition) == 1)


def interaction_agreement_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float] | None = None,
    *,
    mechanistic_pair: tuple[int, int] = (0, 1),
    single_feature_tolerance: float = 1e-9,
    min_pair_interaction: float = 1.0,
) -> InteractionAgreementReport:
    """Show when interactions agree with a mechanistic pair but Shapley does not."""

    values = xor_game() if coalition_values is None else coalition_values
    num_players = max(max(coalition, default=-1) for coalition in values) + 1
    shapley = exact_shapley_values(values, num_players=num_players)
    interactions = pairwise_shapley_interactions(values, num_players=num_players)
    first, second = mechanistic_pair
    pair_value = float(abs(interactions[first, second]).item())
    max_single = float(shapley.abs().max().item())
    return InteractionAgreementReport(
        shapley_values=shapley,
        pair_interactions=interactions,
        mechanistic_pair=mechanistic_pair,
        recovered_pair_interaction=pair_value,
        max_single_feature_value=max_single,
        ordinary_shapley_misses=max_single <= single_feature_tolerance,
        interaction_recovers_pair=pair_value >= min_pair_interaction,
    )


def data_shapley_report(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    learning_rate: float = 0.5,
) -> DataShapleyReport:
    """Summarize exact Data Shapley and deletion/addition checks."""

    values = data_coalition_values(
        train_x,
        train_y,
        val_x,
        val_y,
        learning_rate=learning_rate,
    )
    exact = exact_shapley_values(values, num_players=int(train_x.shape[0]))
    full = frozenset(range(int(train_x.shape[0])))
    harmful_index = int(exact.argmin().item())
    helpful_index = int(exact.argmax().item())
    harmful_removed = values[full - {harmful_index}]
    helpful_alone = values[frozenset({helpful_index})]
    removal_delta = harmful_removed - values[full]
    return DataShapleyReport(
        exact_values=exact,
        full_utility=values[full],
        baseline_utility=values[frozenset()],
        harmful_index=harmful_index,
        harmful_value=float(exact[harmful_index].item()),
        helpful_index=helpful_index,
        helpful_value=float(exact[helpful_index].item()),
        harmful_removal_delta=removal_delta,
        helpful_addition_utility=helpful_alone,
        deletion_test_passes=removal_delta > 0.0,
        addition_test_passes=helpful_alone > values[frozenset()],
    )


def monte_carlo_data_shapley_report(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    learning_rate: float = 0.5,
    num_samples: int = 512,
    seed: int = 0,
    tolerance: float = 0.08,
) -> MonteCarloDataShapleyReport:
    """Compare sampled Data Shapley estimates to exact values."""

    exact = exact_data_shapley_values(
        train_x,
        train_y,
        val_x,
        val_y,
        learning_rate=learning_rate,
    )
    sampled = monte_carlo_data_shapley_values(
        train_x,
        train_y,
        val_x,
        val_y,
        learning_rate=learning_rate,
        num_samples=num_samples,
        seed=seed,
    )
    max_error = float((exact - sampled).abs().max().item())
    sampled_top = int(sampled.argmax().item())
    sampled_harmful = int(sampled.argmin().item())
    exact_top_value = float(exact.max().item())
    exact_harmful_value = float(exact.min().item())
    return MonteCarloDataShapleyReport(
        exact_values=exact,
        sampled_values=sampled,
        max_abs_error=max_error,
        top_example_matches=abs(float(exact[sampled_top].item()) - exact_top_value)
        <= tolerance,
        harmful_example_matches=abs(float(exact[sampled_harmful].item()) - exact_harmful_value)
        <= tolerance,
        approximates_exact=max_error <= tolerance,
    )


def in_run_data_shapley_report(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    min_correlation: float = 0.9,
) -> InRunDataShapleyReport:
    """Compare first-order in-run scores against exact Data Shapley."""

    exact = exact_data_shapley_values(train_x, train_y, val_x, val_y)
    scores = in_run_first_order_data_scores(train_x, train_y, val_x, val_y)
    correlation = pearson_correlation(exact, scores)
    harmful = int(exact.argmin().item())
    helpful = int(exact.argmax().item())
    in_run_harmful = int(scores.argmin().item())
    in_run_helpful = int(scores.argmax().item())
    return InRunDataShapleyReport(
        exact_values=exact,
        in_run_scores=scores,
        pearson_correlation=correlation,
        harmful_index=harmful,
        in_run_harmful_index=in_run_harmful,
        helpful_index=helpful,
        in_run_helpful_index=in_run_helpful,
        identifies_harmful=harmful == in_run_harmful,
        identifies_helpful=helpful == in_run_helpful,
        correlates_with_exact=correlation >= min_correlation,
    )


def kernelshap_kernel_weight(coalition_size: int, num_players: int) -> float:
    """Return the KernelSHAP weight for a non-empty, non-full coalition."""

    if coalition_size <= 0 or coalition_size >= num_players:
        raise ValueError("KernelSHAP finite weights require 0 < coalition_size < num_players.")
    return (num_players - 1) / (
        math.comb(num_players, coalition_size)
        * coalition_size
        * (num_players - coalition_size)
    )


def kernelshap_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact KernelSHAP from the full coalition table by weighted regression."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    empty = frozenset()
    full = frozenset(range(num_players))
    rows = []
    targets = []
    weights = []
    baseline = values[empty]
    total_delta = values[full] - baseline
    for coalition in all_coalitions(num_players):
        size = len(coalition)
        if size == 0 or size == num_players:
            continue
        rows.append([1.0 if player in coalition else 0.0 for player in range(num_players)])
        targets.append(values[coalition] - baseline)
        weights.append(kernelshap_kernel_weight(size, num_players))
    design = t.tensor(rows, dtype=t.float64)
    target = t.tensor(targets, dtype=t.float64)
    weight_vector = t.tensor(weights, dtype=t.float64)
    hessian = design.T @ (design * weight_vector.unsqueeze(-1))
    rhs = design.T @ (target * weight_vector)
    constraint = t.ones((1, num_players), dtype=t.float64)
    kkt = t.cat(
        [
            t.cat([hessian, constraint.T], dim=1),
            t.cat([constraint, t.zeros((1, 1), dtype=t.float64)], dim=1),
        ],
        dim=0,
    )
    constrained_rhs = t.cat([rhs, t.tensor([total_delta], dtype=t.float64)])
    return t.linalg.solve(kkt, constrained_rhs)[:num_players]


def kernelshap_approximation_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> KernelSHAPApproximationReport:
    """Check KernelSHAP against exact Shapley on a complete small game."""

    kernel_values = kernelshap_values(coalition_values, num_players=num_players)
    exact = exact_shapley_values(coalition_values, num_players=num_players)
    max_error = float((kernel_values - exact).abs().max().item())
    return KernelSHAPApproximationReport(
        shapley_values=kernel_values,
        exact_values=exact,
        max_abs_error=max_error,
        approximates_exact=max_error <= tolerance,
    )


def grouped_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
) -> dict[Coalition, float]:
    """Collapse player coalitions into group coalitions for Owen values."""

    group_count = len(groups)
    player_count = sum(len(group) for group in groups)
    values = normalize_coalition_values(coalition_values, num_players=player_count)
    group_values = {}
    for group_coalition in all_coalitions(group_count):
        players: set[int] = set()
        for group_index in group_coalition:
            players.update(groups[group_index])
        group_values[group_coalition] = values[frozenset(players)]
    return group_values


def partition_shap_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
) -> t.Tensor:
    """Compute exact Owen values for a fixed partition of players."""

    if not groups:
        raise ValueError("groups must be nonempty.")
    flattened = [player for group in groups for player in group]
    if sorted(flattened) != list(range(len(flattened))):
        raise ValueError("groups must partition players 0..n-1 exactly once.")

    values = normalize_coalition_values(coalition_values, num_players=len(flattened))
    group_count = len(groups)
    player_values = t.zeros(len(flattened), dtype=t.float64)
    for group_index, group in enumerate(groups):
        outside_group_indices = [idx for idx in range(group_count) if idx != group_index]
        group_denominator = math.factorial(group_count)
        local_denominator = math.factorial(len(group))
        for local_index, player in enumerate(group):
            others_in_group = [
                candidate
                for idx, candidate in enumerate(group)
                if idx != local_index
            ]
            for outside_size in range(group_count):
                outside_weight = (
                    math.factorial(outside_size)
                    * math.factorial(group_count - outside_size - 1)
                    / group_denominator
                )
                for outside_groups in itertools.combinations(outside_group_indices, outside_size):
                    outside_players: set[int] = set()
                    for outside_group in outside_groups:
                        outside_players.update(groups[outside_group])
                    for inside_size in range(len(group)):
                        inside_weight = (
                            math.factorial(inside_size)
                            * math.factorial(len(group) - inside_size - 1)
                            / local_denominator
                        )
                        for inside_players in itertools.combinations(
                            others_in_group,
                            inside_size,
                        ):
                            coalition = frozenset(outside_players | set(inside_players))
                            player_values[player] += outside_weight * inside_weight * (
                                values[coalition | {player}] - values[coalition]
                            )
    return player_values


def partition_shap_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
    tolerance: float = 1e-9,
) -> PartitionSHAPReport:
    """Compare partition/Owen values to exact Shapley values."""

    num_players = sum(len(group) for group in groups)
    player_values = partition_shap_values(coalition_values, groups=groups)
    exact = exact_shapley_values(coalition_values, num_players=num_players)
    group_values = exact_shapley_values(
        grouped_coalition_values(coalition_values, groups=groups),
        num_players=len(groups),
    )
    max_error = float((player_values - exact).abs().max().item())
    return PartitionSHAPReport(
        group_values=group_values,
        player_values=player_values,
        exact_values=exact,
        max_abs_error=max_error,
        recovers_exact=max_error <= tolerance,
    )


def pairwise_shapley_interactions(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact second-order Shapley interaction indices for all pairs."""

    if num_players < 2:
        raise ValueError("pairwise interactions require at least two players.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    interactions = t.zeros((num_players, num_players), dtype=t.float64)
    factorial = math.factorial
    denominator = factorial(num_players - 1)
    for first, second in itertools.combinations(range(num_players), 2):
        others = [player for player in range(num_players) if player not in (first, second)]
        score = 0.0
        for size in range(num_players - 1):
            weight = factorial(size) * factorial(num_players - size - 2) / denominator
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                delta = (
                    values[coalition | {first, second}]
                    - values[coalition | {first}]
                    - values[coalition | {second}]
                    + values[coalition]
                )
                score += weight * delta
        interactions[first, second] = score
        interactions[second, first] = score
    return interactions


def pairwise_interaction_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    target_pair: tuple[int, int] = (0, 1),
    expected_target: float = 1.0,
    tolerance: float = 1e-9,
) -> PairwiseInteractionReport:
    """Check that a game has one intended second-order interaction."""

    interactions = pairwise_shapley_interactions(coalition_values, num_players=num_players)
    first, second = target_pair
    if first == second:
        raise ValueError("target_pair must contain two different players.")
    if not (0 <= first < num_players and 0 <= second < num_players):
        raise ValueError("target_pair players must be valid player indices.")
    target_value = float(interactions[first, second].item())
    spurious = []
    for row, col in itertools.combinations(range(num_players), 2):
        if frozenset((row, col)) != frozenset(target_pair):
            spurious.append(abs(float(interactions[row, col].item())))
    max_spurious = max(spurious, default=0.0)
    return PairwiseInteractionReport(
        pair_interactions=interactions,
        target_pair=target_pair,
        target_interaction=target_value,
        max_spurious_interaction=max_spurious,
        recovers_interaction=(
            abs(target_value - expected_target) <= tolerance
            and max_spurious <= tolerance
        ),
    )


def interaction_game(
    num_players: int,
    *,
    pair: tuple[int, int] = (0, 1),
    pair_weight: float = 1.0,
    additive_weights: t.Tensor | None = None,
) -> dict[Coalition, float]:
    """Return a game with one pair interaction plus optional additive effects."""

    if num_players < 2:
        raise ValueError("interaction_game requires at least two players.")
    first, second = pair
    if first == second:
        raise ValueError("pair must contain two different players.")
    if not (0 <= first < num_players and 0 <= second < num_players):
        raise ValueError("pair players must be valid player indices.")
    if additive_weights is None:
        weights = t.zeros(num_players, dtype=t.float64)
    else:
        weights = additive_weights.flatten().double()
        if int(weights.numel()) != num_players:
            raise ValueError("additive_weights must have one weight per player.")

    def value_fn(coalition: Coalition) -> float:
        additive = weights[list(coalition)].sum().item() if coalition else 0.0
        interaction = pair_weight if first in coalition and second in coalition else 0.0
        return additive + interaction

    return coalition_values_from_function(num_players, value_fn)


def shapiq_pairwise_interactions(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    index: str = "SII",
) -> t.Tensor:
    """Compute pairwise interactions through shapiq on a complete value table."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    try:
        import numpy as np
        import shapiq
    except ImportError as exc:  # pragma: no cover - exercised by report flag.
        raise RuntimeError("shapiq is required for shapiq parity checks.") from exc

    def game(coalitions):
        array = np.asarray(coalitions, dtype=bool)
        if array.ndim == 1:
            array = array[None, :]
        outputs = []
        for row in array:
            coalition = frozenset(
                player for player, present in enumerate(row.tolist()) if present
            )
            outputs.append(values[coalition])
        return np.asarray(outputs, dtype=float)

    explainer = shapiq.AgnosticExplainer(
        game,
        n_players=num_players,
        index=index,
        max_order=2,
        random_state=0,
    )
    interaction_values = explainer.explain(budget=2**num_players, random_state=0)
    lookup = interaction_values.interaction_lookup
    raw_values = interaction_values.values
    matrix = t.zeros((num_players, num_players), dtype=t.float64)
    for first, second in itertools.combinations(range(num_players), 2):
        lookup_key = (first, second)
        reverse_key = (second, first)
        if lookup_key in lookup:
            value_index = lookup[lookup_key]
        else:
            value_index = lookup[reverse_key]
        value = float(raw_values[value_index])
        matrix[first, second] = value
        matrix[second, first] = value
    return matrix


def shapiq_interaction_parity_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    index: str = "SII",
    tolerance: float = 1e-6,
) -> ShapiqInteractionParityReport:
    """Compare closed-form pairwise interactions to shapiq on a small game."""

    exact = pairwise_shapley_interactions(coalition_values, num_players=num_players)
    try:
        observed = shapiq_pairwise_interactions(
            coalition_values,
            num_players=num_players,
            index=index,
        )
        available = True
    except RuntimeError:
        observed = t.full_like(exact, float("nan"))
        available = False
    max_error = (
        float((exact - observed).abs().max().item()) if available else float("inf")
    )
    return ShapiqInteractionParityReport(
        exact_pair_interactions=exact,
        shapiq_pair_interactions=observed,
        max_abs_error=max_error,
        matches_shapiq=available and max_error <= tolerance,
        shapiq_available=available,
    )


def token_coalition_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = "[MASK]",
) -> dict[Coalition, float]:
    """Build a complete position-coalition table for masked token attribution."""

    token_tuple = tuple(tokens)
    num_tokens = len(token_tuple)
    if num_tokens <= 0:
        raise ValueError("tokens must be nonempty.")
    values = {}
    for coalition in all_coalitions(num_tokens):
        masked_tokens = tuple(
            token if index in coalition else mask_token
            for index, token in enumerate(token_tuple)
        )
        values[coalition] = float(score_fn(masked_tokens))
    return values


def exact_token_shapley_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = "[MASK]",
) -> t.Tensor:
    """Compute exact Shapley attributions over token positions."""

    values = token_coalition_values(tokens, score_fn, mask_token=mask_token)
    return exact_shapley_values(values, num_players=len(tuple(tokens)))


def sampled_token_shapley_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = "[MASK]",
    num_samples: int = 256,
    seed: int = 0,
) -> t.Tensor:
    """Estimate token Shapley values by sampled permutation TokenSHAP."""

    values = token_coalition_values(tokens, score_fn, mask_token=mask_token)
    return sampled_permutation_shapley_values(
        values,
        num_players=len(tuple(tokens)),
        num_samples=num_samples,
        seed=seed,
    )


def token_shapley_sampling_report(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = "[MASK]",
    num_samples: int = 256,
    seed: int = 0,
    tolerance: float = 0.15,
) -> TokenShapleySamplingReport:
    """Compare sampled TokenSHAP values with exact token Shapley values."""

    token_tuple = tuple(tokens)
    exact = exact_token_shapley_values(token_tuple, score_fn, mask_token=mask_token)
    sampled = sampled_token_shapley_values(
        token_tuple,
        score_fn,
        mask_token=mask_token,
        num_samples=num_samples,
        seed=seed,
    )
    max_error = float((exact - sampled).abs().max().item())
    top_index = int(exact.argmax().item())
    sampled_top_index = int(sampled.argmax().item())
    return TokenShapleySamplingReport(
        tokens=token_tuple,
        exact_values=exact,
        sampled_values=sampled,
        max_abs_error=max_error,
        top_token=token_tuple[top_index],
        sampled_top_token=token_tuple[sampled_top_index],
        rank_matches=top_index == sampled_top_index,
        approximates_exact=max_error <= tolerance,
    )


def token_baseline_report(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = "[MASK]",
    tolerance: float = 1e-9,
) -> TokenBaselineReport:
    """Check token Shapley efficiency against masked baseline and full prompt."""

    token_tuple = tuple(tokens)
    baseline_tokens = tuple(mask_token for _ in token_tuple)
    exact = exact_token_shapley_values(token_tuple, score_fn, mask_token=mask_token)
    full_score = float(score_fn(token_tuple))
    baseline_score = float(score_fn(baseline_tokens))
    total_delta = full_score - baseline_score
    shapley_sum = float(exact.sum().item())
    error = abs(shapley_sum - total_delta)
    return TokenBaselineReport(
        full_score=full_score,
        baseline_score=baseline_score,
        total_delta=total_delta,
        shapley_sum=shapley_sum,
        efficiency_error=error,
        satisfies_efficiency=error <= tolerance,
    )


def keyword_interaction_token_score(
    tokens: Sequence[str],
    *,
    target_token: str = "Paris",
    context_token: str = "capital",
    target_weight: float = 1.0,
    interaction_weight: float = 2.0,
) -> float:
    """Toy prompt score with one target token and one context-target interaction."""

    token_set = set(tokens)
    target_present = target_token in token_set
    context_present = context_token in token_set
    score = target_weight if target_present else 0.0
    if target_present and context_present:
        score += interaction_weight
    return score


def vlm_modality_game(
    *,
    image_weight: float = 1.0,
    text_weight: float = 0.5,
    synergy_weight: float = 2.0,
) -> dict[Coalition, float]:
    """Return a two-player image/text value table with multimodal synergy."""

    def value_fn(coalition: Coalition) -> float:
        image_present = 0 in coalition
        text_present = 1 in coalition
        score = 0.0
        if image_present:
            score += image_weight
        if text_present:
            score += text_weight
        if image_present and text_present:
            score += synergy_weight
        return score

    return coalition_values_from_function(2, value_fn)


def vlm_modality_shap_report(
    *,
    image_weight: float = 1.0,
    text_weight: float = 0.5,
    synergy_weight: float = 2.0,
    min_synergy: float = 1.0,
    tolerance: float = 1e-9,
) -> VLMModalitySHAPReport:
    """Compute modality Shapley values and detect image/text synergy."""

    values = vlm_modality_game(
        image_weight=image_weight,
        text_weight=text_weight,
        synergy_weight=synergy_weight,
    )
    modality_values = exact_shapley_values(values, num_players=2)
    baseline = values[frozenset()]
    image_only = values[frozenset({0})]
    text_only = values[frozenset({1})]
    full = values[frozenset({0, 1})]
    synergy = full - image_only - text_only + baseline
    efficiency_error = abs(float(modality_values.sum().item()) - (full - baseline))
    return VLMModalitySHAPReport(
        modality_values=modality_values,
        baseline_score=baseline,
        image_only_score=image_only,
        text_only_score=text_only,
        full_score=full,
        synergy=synergy,
        detects_synergy=synergy >= min_synergy,
        satisfies_efficiency=efficiency_error <= tolerance,
    )


def vlm_region_game(
    *,
    object_weight: float = 2.0,
    ocr_weight: float = 0.75,
    object_ocr_interaction: float = 0.5,
) -> dict[Coalition, float]:
    """Return a structured image-region game for object/OCR/background regions."""

    def value_fn(coalition: Coalition) -> float:
        object_present = 0 in coalition
        background_present = 1 in coalition
        ocr_present = 2 in coalition
        _ = background_present
        score = 0.0
        if object_present:
            score += object_weight
        if ocr_present:
            score += ocr_weight
        if object_present and ocr_present:
            score += object_ocr_interaction
        return score

    return coalition_values_from_function(3, value_fn)


def vlm_region_shap_report(
    *,
    region_names: tuple[str, ...] = ("object", "background", "ocr_text"),
    target_region: str = "object",
    min_margin: float = 0.5,
    tolerance: float = 1e-9,
) -> VLMRegionSHAPReport:
    """Compute structured region Shapley values and check target localization."""

    if len(region_names) != 3:
        raise ValueError("region_names must name object, background, and OCR regions.")
    values = vlm_region_game()
    region_values = exact_shapley_values(values, num_players=3)
    if target_region not in region_names:
        raise ValueError("target_region must be one of region_names.")
    target_index = region_names.index(target_region)
    target_value = float(region_values[target_index].item())
    background_values = [
        abs(float(value.item()))
        for index, value in enumerate(region_values)
        if index != target_index
    ]
    max_background = max(background_values, default=0.0)
    efficiency = shapley_efficiency_report(values, num_players=3, tolerance=tolerance)
    return VLMRegionSHAPReport(
        region_values=region_values,
        region_names=region_names,
        target_region=target_region,
        target_value=target_value,
        max_background_value=max_background,
        localizes_target=target_value >= max_background + min_margin,
        satisfies_efficiency=efficiency.satisfies_efficiency,
    )


def shapley_efficiency_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> ShapleyEfficiencyReport:
    """Check that Shapley values sum to full-minus-empty value."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    shapley = exact_shapley_values(values, num_players=num_players)
    full = frozenset(range(num_players))
    empty = frozenset()
    shapley_sum = float(shapley.sum().item())
    total_delta = values[full] - values[empty]
    error = abs(shapley_sum - total_delta)
    return ShapleyEfficiencyReport(
        shapley_sum=shapley_sum,
        total_value_delta=total_delta,
        efficiency_error=error,
        satisfies_efficiency=error <= tolerance,
    )


def permutation_parity_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> PermutationParityReport:
    """Compare closed-form exact Shapley values to permutation averaging."""

    exact = exact_shapley_values(coalition_values, num_players=num_players)
    permutation = permutation_shapley_values(coalition_values, num_players=num_players)
    max_error = float((exact - permutation).abs().max().item())
    return PermutationParityReport(
        max_abs_error=max_error,
        matches_exact=max_error <= tolerance,
    )


def interaction_gap_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    min_overcount: float = 0.5,
) -> InteractionGapReport:
    """Compare Shapley total credit with leave-one-out total credit."""

    shapley_total = float(exact_shapley_values(coalition_values, num_players=num_players).sum())
    loo_total = float(leave_one_out_values(coalition_values, num_players=num_players).sum())
    overcount = loo_total - shapley_total
    return InteractionGapReport(
        shapley_total=shapley_total,
        leave_one_out_total=loo_total,
        overcount=overcount,
        detects_interaction_overcount=overcount >= min_overcount,
    )


def additive_game(weights: t.Tensor) -> dict[Coalition, float]:
    """Return coalition values for an additive linear game."""

    weights = weights.flatten().double()
    return coalition_values_from_function(
        int(weights.numel()),
        lambda coalition: weights[list(coalition)].sum().item() if coalition else 0.0,
    )


def conjunction_game(num_players: int) -> dict[Coalition, float]:
    """Return a Boolean AND game where only the full coalition has value one."""

    full = frozenset(range(num_players))
    return coalition_values_from_function(num_players, lambda coalition: coalition == full)
