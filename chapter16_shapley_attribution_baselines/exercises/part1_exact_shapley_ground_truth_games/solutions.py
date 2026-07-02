# %%
"""Reference solutions for [16.1] Exact Shapley on Ground-Truth Games."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import itertools
import math
import sys
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.shapley_neural_game import (
    NEURAL_GAME_NUM_PLAYERS,
    NEURAL_GAME_STEPS,
    binary_feature_table,
    coalition_table_from_model,
    coalition_table_from_true_game,
    shuffled_targets,
    train_neural_game,
    true_neural_game_scores,
)

MAIN = __name__ == "__main__"

Coalition = frozenset[int]

NEURAL_GAME_MAX_SHAPLEY_ERROR = 1e-4
NEURAL_GAME_MAX_FIT_MSE = 1e-8
NEURAL_GAME_RANDOM_CONTROL_MIN_ERROR = 1.0
NEURAL_GAME_RANDOM_CONTROL_MAX_COSINE = 0.25


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


def leave_one_out_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Return full-coalition leave-one-out effects for each player."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    full = frozenset(range(num_players))
    return t.tensor(
        [values[full] - values[full - {player}] for player in range(num_players)],
        dtype=t.float64,
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


# %%
def additive_smoke_test() -> dict:
    weights = t.tensor([1.0, 2.0, -0.5])
    values = additive_game(weights)
    shapley = exact_shapley_values(values, num_players=3)
    return {
        "shapley": shapley.tolist(),
        "efficiency": shapley_efficiency_report(values, num_players=3).__dict__,
    }


def conjunction_smoke_test() -> dict:
    values = conjunction_game(3)
    shapley = exact_shapley_values(values, num_players=3)
    return {
        "shapley": shapley.tolist(),
        "efficiency": shapley_efficiency_report(values, num_players=3).__dict__,
    }


def permutation_parity_smoke_test() -> dict:
    values = conjunction_game(3)
    return permutation_parity_report(values, num_players=3).__dict__


def interaction_failure_smoke_test() -> dict:
    values = conjunction_game(2)
    return interaction_gap_report(values, num_players=2, min_overcount=0.5).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "additive": additive_smoke_test(),
        "conjunction": conjunction_smoke_test(),
        "permutation_parity": permutation_parity_smoke_test(),
        "interaction_failure": interaction_failure_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.1 GPU preflight requires CUDA; no CPU fallback is accepted.")

    return run_neural_exact_shapley_preflight(max_vram_gb=max_vram_gb)


def run_neural_exact_shapley_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()

    inputs = binary_feature_table(device)
    true_targets = true_neural_game_scores(inputs)
    trained = train_neural_game(inputs, true_targets)

    true_values = coalition_table_from_true_game(device)
    model_values = coalition_table_from_model(trained.model, device)
    true_shapley = exact_shapley_values(true_values, num_players=NEURAL_GAME_NUM_PLAYERS)
    model_shapley = exact_shapley_values(model_values, num_players=NEURAL_GAME_NUM_PLAYERS)
    max_abs_error = float((model_shapley - true_shapley).abs().max().item())
    efficiency = shapley_efficiency_report(model_values, num_players=NEURAL_GAME_NUM_PLAYERS)

    shuffled = train_neural_game(inputs, shuffled_targets(true_targets))
    shuffled_values = coalition_table_from_model(shuffled.model, device)
    shuffled_shapley = exact_shapley_values(shuffled_values, num_players=NEURAL_GAME_NUM_PLAYERS)
    shuffled_error = float((shuffled_shapley - true_shapley).abs().max().item())
    shuffled_cosine = float(
        t.nn.functional.cosine_similarity(
            shuffled_shapley.float(),
            true_shapley.float(),
            dim=0,
        ).item()
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
        trained.fit_mse <= NEURAL_GAME_MAX_FIT_MSE
        and max_abs_error <= NEURAL_GAME_MAX_SHAPLEY_ERROR
        and efficiency.satisfies_efficiency
        and shuffled_error >= NEURAL_GAME_RANDOM_CONTROL_MIN_ERROR
        and shuffled_cosine <= NEURAL_GAME_RANDOM_CONTROL_MAX_COSINE
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "preflight_passed": preflight_passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_family": "cuda_trained_neural_coalition_game_mlp",
        "num_players": NEURAL_GAME_NUM_PLAYERS,
        "coalition_count": 2**NEURAL_GAME_NUM_PLAYERS,
        "complete_finite_domain_evaluated": True,
        "training_example_count": int(inputs.shape[0]),
        "training_steps": NEURAL_GAME_STEPS,
        "fit_mse": trained.fit_mse,
        "fit_max_abs_error": trained.fit_max_abs_error,
        "true_shapley": true_shapley.tolist(),
        "model_shapley": model_shapley.tolist(),
        "neural_shapley_max_abs_error": max_abs_error,
        "efficiency_error": efficiency.efficiency_error,
        "satisfies_efficiency": efficiency.satisfies_efficiency,
        "shuffled_control_fit_mse": shuffled.fit_mse,
        "shuffled_control_shapley": shuffled_shapley.tolist(),
        "shuffled_control_error": shuffled_error,
        "shuffled_control_cosine": shuffled_cosine,
        "shuffled_control_rejected": shuffled_error >= NEURAL_GAME_RANDOM_CONTROL_MIN_ERROR
        and shuffled_cosine <= NEURAL_GAME_RANDOM_CONTROL_MAX_COSINE,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "ood_generalization_claimed": False,
        "generalization_scope": "complete_binary_feature_table",
        "full_path": "Train a CUDA MLP on a complete binary feature game, compute exact Shapley values from real model ablations, and reject a shuffled-label control.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
