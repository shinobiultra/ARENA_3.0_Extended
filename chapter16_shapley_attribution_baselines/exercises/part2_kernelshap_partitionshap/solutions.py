# %%


MAIN = __name__ == "__main__"

# %%

from collections.abc import Callable, Mapping, Sequence

from dataclasses import dataclass

import itertools

import math

import sys

from pathlib import Path

from typing import Literal

import matplotlib.pyplot as plt

import numpy as np

import torch as t

chapter = "chapter16_shapley_attribution_baselines"

section = "part2_kernelshap_partitionshap"

root_dir = Path(__file__).resolve().parents[3]

exercises_dir = root_dir / chapter / "exercises"

assets_dir = root_dir / chapter / "exercises" / section / "media"

if MAIN:
    for path in (root_dir, exercises_dir):
        if str(path) not in sys.path:
            sys.path.append(str(path))

if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partitionshap import tests

Coalition = frozenset[int]

if MAIN:
    t.set_grad_enabled(False)

if MAIN:
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    })

EXACT = "#B43A32"

PAIRED = "#167D8D"

UNIFORM = "#D39A2C"

CONTROL = "#5F6875"

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

# %%

def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    return tuple(
        frozenset(group)
        for size in range(num_players + 1)
        for group in itertools.combinations(range(num_players), size)
    )

def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    normalized = {
        frozenset(key): float(value) for key, value in coalition_values.items()
    }
    expected = set(all_coalitions(num_players))
    missing = expected - set(normalized)
    extra = set(normalized) - expected
    if missing or extra:
        raise ValueError(
            f"coalition table has missing={len(missing)} and extra={len(extra)} coalitions"
        )
    return {coalition: normalized[coalition] for coalition in all_coalitions(num_players)}

def coalition_values_from_function(
    num_players: int,
    value_fn: Callable[[Coalition], float],
) -> dict[Coalition, float]:
    return {
        coalition: float(value_fn(coalition))
        for coalition in all_coalitions(num_players)
    }

def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    result = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        for coalition in all_coalitions(num_players):
            if player in coalition:
                continue
            size = len(coalition)
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            result[player] += weight * (
                values[coalition | {player}] - values[coalition]
            )
    return result

def additive_game(weights: t.Tensor) -> dict[Coalition, float]:
    weights = weights.detach().flatten().double()
    return coalition_values_from_function(
        int(weights.numel()),
        lambda coalition: weights[list(coalition)].sum().item()
        if coalition else 0.0,
    )

def conjunction_game(num_players: int) -> dict[Coalition, float]:
    full = frozenset(range(num_players))
    return coalition_values_from_function(
        num_players, lambda coalition: coalition == full
    )

# %%

PLAYER_NAMES = ("embed", "attend", "mlp", "router", "memory", "readout")

GAME_DIVIDENDS = {
    frozenset({0}): 0.6,
    frozenset({1}): -0.2,
    frozenset({2}): 0.5,
    frozenset({3}): 0.1,
    frozenset({4}): 0.4,
    frozenset({5}): -0.1,
    frozenset({0, 1, 2}): 1.8,
    frozenset({3, 4, 5}): -0.9,
}

def game_from_dividends(
    num_players: int,
    dividends: Mapping[Coalition | tuple[int, ...], float],
) -> dict[Coalition, float]:
    normalized = {
        frozenset(term): float(value) for term, value in dividends.items()
    }
    if any(not term for term in normalized):
        raise ValueError("The empty dividend is a baseline, not player credit.")
    if any(player < 0 or player >= num_players for term in normalized for player in term):
        raise ValueError("Dividend contains an invalid player.")
    return coalition_values_from_function(
        num_players,
        lambda coalition: sum(
            value for term, value in normalized.items()
            if term.issubset(coalition)
        ),
    )

def shapley_from_dividends(
    num_players: int,
    dividends: Mapping[Coalition | tuple[int, ...], float],
) -> t.Tensor:
    result = t.zeros(num_players, dtype=t.float64)
    for raw_term, raw_value in dividends.items():
        term = frozenset(raw_term)
        if not term:
            raise ValueError("The empty dividend is a baseline, not player credit.")
        for player in term:
            result[player] += float(raw_value) / len(term)
    return result

if MAIN:
    tests.test_exact_dividend_game_oracle(
        game_from_dividends,
        shapley_from_dividends,
    )

if MAIN:
    game_values = game_from_dividends(6, GAME_DIVIDENDS)

if MAIN:
    exact_oracle = shapley_from_dividends(6, GAME_DIVIDENDS)

if MAIN:
    exact_formula = exact_shapley_values(game_values, num_players=6)

if MAIN:
    oracle_parity_error = (exact_formula - exact_oracle).abs().max().item()

if MAIN:
    print("Exact oracle =", [round(value, 3) for value in exact_oracle.tolist()])

if MAIN:
    print(
        f"Full-game value = {game_values[frozenset(range(6))]:.3f}; "
        f"oracle parity error = {oracle_parity_error:.2e}"
    )

# %%

def kernelshap_kernel_weight(coalition_size: int, num_players: int) -> float:
    if not 0 < coalition_size < num_players:
        raise ValueError("Need 0 < coalition_size < num_players.")
    return (num_players - 1) / (
        math.comb(num_players, coalition_size)
        * coalition_size
        * (num_players - coalition_size)
    )

def kernelshap_regression_rows(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    coalitions: Sequence[Coalition] | None = None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    rows = (
        tuple(frozenset(coalition) for coalition in coalitions)
        if coalitions is not None
        else tuple(
            coalition for coalition in all_coalitions(num_players)
            if 0 < len(coalition) < num_players
        )
    )
    if not rows or any(len(coalition) in (0, num_players) for coalition in rows):
        raise ValueError("Regression rows must be non-empty interior coalitions.")
    X = t.tensor(
        [
            [float(player in coalition) for player in range(num_players)]
            for coalition in rows
        ],
        dtype=t.float64,
    )
    y = t.tensor(
        [values[coalition] - values[frozenset()] for coalition in rows],
        dtype=t.float64,
    )
    weights = t.tensor(
        [
            kernelshap_kernel_weight(len(coalition), num_players)
            for coalition in rows
        ],
        dtype=t.float64,
    )
    return X, y, weights

if MAIN:
    tests.test_kernelshap_kernel_weight_uses_finite_coalition_formula(
        kernelshap_kernel_weight
    )

if MAIN:
    full_X, full_y, full_weights = kernelshap_regression_rows(
        game_values, num_players=6
    )

if MAIN:
    assert full_X.shape == (62, 6)

if MAIN:
    assert kernelshap_kernel_weight(1, 6) > kernelshap_kernel_weight(3, 6)

if MAIN:
    print(
        f"Kernel regression rows: X={tuple(full_X.shape)}, "
        f"y={tuple(full_y.shape)}, w={tuple(full_weights.shape)}"
    )

if MAIN:
    print("Kernel design-row tests passed!")

# %%

def kernelshap_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    coalitions: Sequence[Coalition] | None = None,
) -> t.Tensor:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    X, y, weights = kernelshap_regression_rows(
        values,
        num_players=num_players,
        coalitions=coalitions,
    )
    gram = X.T @ (weights[:, None] * X)
    rhs = X.T @ (weights * y)
    kkt = t.zeros(
        (num_players + 1, num_players + 1), dtype=t.float64
    )
    kkt[:num_players, :num_players] = gram
    kkt[:num_players, -1] = 1.0
    kkt[-1, :num_players] = 1.0
    delta = (
        values[frozenset(range(num_players))]
        - values[frozenset()]
    )
    constrained_rhs = t.cat(
        [rhs, t.tensor([delta], dtype=t.float64)]
    )
    if X.shape[0] == 2**num_players - 2:
        return t.linalg.solve(kkt, constrained_rhs)[:num_players]
    # CPU SVD gives the declared minimum-norm KKT solution reliably.
    # CUDA supports only gels, so solve this small system on CPU and return
    # the coefficients to the original device.
    solution = t.linalg.lstsq(
        kkt.cpu(), constrained_rhs.cpu(), driver="gelsd"
    ).solution
    return solution[:num_players].to(kkt.device)

def kernelshap_approximation_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> KernelSHAPApproximationReport:
    estimate = kernelshap_values(
        coalition_values, num_players=num_players
    )
    exact = exact_shapley_values(
        coalition_values, num_players=num_players
    )
    error = float((estimate - exact).abs().max().item())
    return KernelSHAPApproximationReport(
        shapley_values=estimate,
        exact_values=exact,
        max_abs_error=error,
        approximates_exact=error <= tolerance,
    )

if MAIN:
    tests.test_kernelshap_approximation_report_matches_exact_additive_game(
        kernelshap_approximation_report,
        additive_game,
    )

if MAIN:
    tests.test_kernelshap_interaction_report_splits_conjunction_credit(
        kernelshap_approximation_report,
        conjunction_game,
    )

if MAIN:
    full_kernel = kernelshap_values(game_values, num_players=6)

if MAIN:
    full_kernel_error = (full_kernel - exact_oracle).abs().max().item()

if MAIN:
    full_kernel_efficiency_error = abs(
        full_kernel.sum().item() - game_values[frozenset(range(6))]
    )

if MAIN:
    print(f"Full-table KernelSHAP max error: {full_kernel_error:.2e}")

if MAIN:
    print(f"Efficiency error: {full_kernel_efficiency_error:.2e}")

# %%

def sample_interior_coalitions(
    num_players: int,
    *,
    budget: int,
    seed: int,
    strategy: Literal["paired_kernel", "uniform"],
) -> tuple[Coalition, ...]:
    interior = [
        coalition for coalition in all_coalitions(num_players)
        if 0 < len(coalition) < num_players
    ]
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if budget >= len(interior):
        return tuple(interior)
    generator = t.Generator().manual_seed(seed)
    if strategy == "uniform":
        indices = t.randperm(
            len(interior), generator=generator
        )[:budget]
        return tuple(interior[index] for index in indices.tolist())
    if strategy != "paired_kernel":
        raise ValueError("strategy must be 'paired_kernel' or 'uniform'.")
    if budget % 2:
        raise ValueError("paired_kernel requires an even budget.")

    full = frozenset(range(num_players))
    seen: set[Coalition] = set()
    pairs = []
    for coalition in interior:
        if coalition in seen:
            continue
        complement = full - coalition
        pairs.append((coalition, complement))
        seen.update((coalition, complement))
    pair_weights = t.tensor(
        [
            kernelshap_kernel_weight(len(pair[0]), num_players)
            for pair in pairs
        ],
        dtype=t.float64,
    )
    pair_ids = t.multinomial(
        pair_weights,
        budget // 2,
        replacement=False,
        generator=generator,
    )
    return tuple(
        coalition
        for index in pair_ids.tolist()
        for coalition in pairs[index]
    )

if MAIN:
    tests.test_coalition_sampler_controls(sample_interior_coalitions)

if MAIN:
    tests.test_sampled_kernelshap_convergence_control(
        game_from_dividends,
        shapley_from_dividends,
        sample_interior_coalitions,
        kernelshap_values,
    )

# %%

def validate_partition(
    groups: tuple[tuple[int, ...], ...],
    num_players: int,
) -> None:
    players = [player for group in groups for player in group]
    if (
        not groups
        or any(not group for group in groups)
        or sorted(players) != list(range(num_players))
        or len(players) != len(set(players))
    ):
        raise ValueError(
            "partition players must cover each player exactly once"
        )

def grouped_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
) -> dict[Coalition, float]:
    num_players = len(
        {player for group in groups for player in group}
    )
    validate_partition(groups, num_players)
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    return coalition_values_from_function(
        len(groups),
        lambda group_coalition: values[
            frozenset(
                player
                for group_index in group_coalition
                for player in groups[group_index]
            )
        ],
    )

if MAIN:
    tests.test_grouped_coalition_values_expand_groups_before_lookup(
        grouped_coalition_values,
        additive_game,
    )

if MAIN:
    for invalid_groups in (((0, 1), (1, 2)), ((0, 1),)):
        try:
            validate_partition(invalid_groups, 3)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid partition should fail.")

if MAIN:
    print("Partition validation tests passed!")

# %%

def partition_shap_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
) -> t.Tensor:
    num_players = sum(len(group) for group in groups)
    validate_partition(groups, num_players)
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    result = t.zeros(num_players, dtype=t.float64)
    num_groups = len(groups)
    for group_index, group in enumerate(groups):
        other_groups = [
            index for index in range(num_groups)
            if index != group_index
        ]
        for player in group:
            inside_players = [item for item in group if item != player]
            for outside_size in range(num_groups):
                outside_weight = (
                    math.factorial(outside_size)
                    * math.factorial(num_groups - outside_size - 1)
                    / math.factorial(num_groups)
                )
                for outside_indices in itertools.combinations(
                    other_groups, outside_size
                ):
                    outside = (
                        set().union(
                            *(set(groups[index]) for index in outside_indices)
                        )
                        if outside_indices else set()
                    )
                    for inside_size in range(len(group)):
                        inside_weight = (
                            math.factorial(inside_size)
                            * math.factorial(len(group) - inside_size - 1)
                            / math.factorial(len(group))
                        )
                        for inside_before in itertools.combinations(
                            inside_players, inside_size
                        ):
                            before = frozenset(
                                outside | set(inside_before)
                            )
                            result[player] += (
                                outside_weight
                                * inside_weight
                                * (
                                    values[before | {player}]
                                    - values[before]
                                )
                            )
    return result

def partition_shap_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
    tolerance: float = 1e-9,
) -> PartitionSHAPReport:
    num_players = sum(len(group) for group in groups)
    player_values = partition_shap_values(
        coalition_values, groups=groups
    )
    group_values = exact_shapley_values(
        grouped_coalition_values(coalition_values, groups=groups),
        num_players=len(groups),
    )
    exact_values = exact_shapley_values(
        coalition_values, num_players=num_players
    )
    error = float(
        (player_values - exact_values).abs().max().item()
    )
    return PartitionSHAPReport(
        group_values=group_values,
        player_values=player_values,
        exact_values=exact_values,
        max_abs_error=error,
        recovers_exact=error <= tolerance,
    )

if MAIN:
    tests.test_partition_shap_report_recovers_additive_groups(
        partition_shap_report,
        additive_game,
    )

if MAIN:
    tests.test_partition_shap_report_rejects_invalid_grouping(
        partition_shap_report,
        additive_game,
    )

if MAIN:
    tests.test_partition_shap_report_splits_interaction_group_symmetrically(
        partition_shap_report,
        conjunction_game,
    )

if MAIN:
    tests.test_partition_shap_report_handles_cross_group_interaction_without_irrelevant_credit(
        partition_shap_report,
        coalition_values_from_function,
    )

if MAIN:
    singleton_partition = tuple((player,) for player in range(6))

if MAIN:
    singleton_values = partition_shap_values(
        game_values, groups=singleton_partition
    )

if MAIN:
    t.testing.assert_close(singleton_values, exact_oracle, atol=1e-12, rtol=0)

if MAIN:
    print("PartitionSHAP semantic tests passed!")

# %%

if MAIN:
    tests.test_partition_alignment_and_random_control(
        game_from_dividends,
        shapley_from_dividends,
        partition_shap_values,
    )

if MAIN:
    aligned_groups = ((0, 1, 2), (3, 4, 5))

if MAIN:
    misaligned_groups = ((0, 3, 4), (1, 2, 5))

if MAIN:
    aligned_values = partition_shap_values(game_values, groups=aligned_groups)

if MAIN:
    misaligned_values = partition_shap_values(game_values, groups=misaligned_groups)

if MAIN:
    aligned_error = (aligned_values - exact_oracle).abs().max().item()

if MAIN:
    misaligned_error = (misaligned_values - exact_oracle).abs().max().item()

if MAIN:
    random_partition_errors = []

if MAIN:
    random_partitions = []

if MAIN:
    for seed in range(100):
        generator = t.Generator().manual_seed(seed)
        ordering = t.randperm(6, generator=generator).tolist()
        groups = (
            tuple(sorted(ordering[:3])),
            tuple(sorted(ordering[3:])),
        )
        values = partition_shap_values(game_values, groups=groups)
        random_partitions.append(groups)
        random_partition_errors.append(
            (values - exact_oracle).abs().max().item()
        )

if MAIN:
    nonexact_random_partitions = sum(
        error > 1e-9 for error in random_partition_errors
    )

if MAIN:
    print(f"Aligned error = {aligned_error:.3f}")

if MAIN:
    print(f"Fixed misaligned error = {misaligned_error:.3f}")

if MAIN:
    print(
        f"{nonexact_random_partitions} / 100 seeded random balanced partitions "
        "differ from exact Shapley"
    )

# %%

if MAIN:
    budgets = t.tensor([12, 16, 24, 32, 48, 62])

if MAIN:
    seeds = range(50)

if MAIN:
    sampling_errors = {}

if MAIN:
    for strategy in ("paired_kernel", "uniform"):
        matrix = t.zeros((len(budgets), len(tuple(seeds))), dtype=t.float64)
        for budget_index, budget in enumerate(budgets.tolist()):
            for seed in seeds:
                sample = sample_interior_coalitions(
                    6,
                    budget=budget,
                    seed=seed,
                    strategy=strategy,
                )
                estimate = kernelshap_values(
                    game_values,
                    num_players=6,
                    coalitions=sample,
                )
                matrix[budget_index, seed] = (
                    estimate - exact_oracle
                ).abs().max()
        sampling_errors[strategy] = matrix

if MAIN:
    interior = tuple(
        coalition for coalition in all_coalitions(6)
        if 0 < len(coalition) < 6
    )

if MAIN:
    randomized_value_errors = []

if MAIN:
    for seed in seeds:
        generator = t.Generator().manual_seed(100 + seed)
        shuffled = dict(game_values)
        interior_values = t.tensor(
            [game_values[coalition] for coalition in interior],
            dtype=t.float64,
        )
        permutation = t.randperm(len(interior), generator=generator)
        for coalition, value in zip(interior, interior_values[permutation].tolist()):
            shuffled[coalition] = value
        randomized_estimate = kernelshap_values(
            shuffled, num_players=6
        )
        randomized_value_errors.append(
            (randomized_estimate - exact_oracle).abs().max().item()
        )

if MAIN:
    paired_mean = sampling_errors["paired_kernel"].mean(dim=1)

if MAIN:
    paired_std = sampling_errors["paired_kernel"].std(dim=1)

if MAIN:
    uniform_mean = sampling_errors["uniform"].mean(dim=1)

if MAIN:
    uniform_std = sampling_errors["uniform"].std(dim=1)

if MAIN:
    randomized_mean = float(np.mean(randomized_value_errors))

if MAIN:
    assert full_kernel_error < 1e-12

if MAIN:
    assert paired_mean[2] < uniform_mean[2]

if MAIN:
    assert aligned_error < 1e-12

if MAIN:
    assert abs(misaligned_error - 0.3) < 1e-12

if MAIN:
    assert randomized_mean > 0.5

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))

if MAIN:
    axes[0].plot(budgets, paired_mean, marker="o", color=PAIRED, label="paired-kernel sample")

if MAIN:
    axes[0].fill_between(budgets, paired_mean - paired_std, paired_mean + paired_std, color=PAIRED, alpha=0.16)

if MAIN:
    axes[0].plot(budgets, uniform_mean, marker="o", color=UNIFORM, label="uniform sample")

if MAIN:
    axes[0].fill_between(budgets, uniform_mean - uniform_std, uniform_mean + uniform_std, color=UNIFORM, alpha=0.16)

if MAIN:
    axes[0].axhline(randomized_mean, color=CONTROL, linestyle="--", label="shuffled-value control")

if MAIN:
    axes[0].scatter([62], [full_kernel_error], color=EXACT, marker="*", s=150, label="full-table parity")

if MAIN:
    axes[0].set_xlabel("interior coalition budget")

if MAIN:
    axes[0].set_ylabel("max absolute player error")

if MAIN:
    axes[0].set_title("KernelSHAP convergence (50 seeds)")

if MAIN:
    axes[0].legend(frameon=False, fontsize=7)

if MAIN:
    x = np.arange(6)

if MAIN:
    width = 0.25

if MAIN:
    axes[1].bar(x - width, exact_oracle, width, color=EXACT, label="exact Shapley")

if MAIN:
    axes[1].bar(x, aligned_values, width, color=PAIRED, label="aligned partition")

if MAIN:
    axes[1].bar(x + width, misaligned_values, width, color=UNIFORM, label="misaligned partition")

if MAIN:
    axes[1].axhline(0, color="#B8BDC5", linewidth=1)

if MAIN:
    axes[1].set_xticks(x, PLAYER_NAMES, rotation=25)

if MAIN:
    axes[1].set_ylabel("player attribution")

if MAIN:
    axes[1].set_title("Partition changes interaction credit")

if MAIN:
    axes[1].legend(frameon=False, fontsize=7)

if MAIN:
    bins = np.linspace(-0.01, 0.325, 12)

if MAIN:
    axes[2].hist(random_partition_errors, bins=bins, color=CONTROL, alpha=0.78, label="100 random balanced partitions")

if MAIN:
    axes[2].axvline(aligned_error, color=PAIRED, linewidth=2, label="aligned control")

if MAIN:
    axes[2].axvline(misaligned_error, color=UNIFORM, linestyle="--", linewidth=2, label="fixed misaligned")

if MAIN:
    axes[2].set_xlabel("max absolute error vs exact Shapley")

if MAIN:
    axes[2].set_ylabel("seed count")

if MAIN:
    axes[2].set_title("Random partition control")

if MAIN:
    axes[2].legend(frameon=False, fontsize=7)

if MAIN:
    fig.suptitle(
        "KernelSHAP needs sampling controls; PartitionSHAP needs partition controls",
        fontsize=13,
        fontweight="bold",
    )

if MAIN:
    fig.tight_layout()

if MAIN:
    signature_path = assets_dir / "kernelshap_partition_controls_signature.png"

if MAIN:
    fig.savefig(signature_path, dpi=170, bbox_inches="tight")

if MAIN:
    plt.show()

if MAIN:
    print(
        f"Signature metrics: full_error={full_kernel_error:.2e}, "
        f"budget24_paired={paired_mean[2]:.3f}, "
        f"budget24_uniform={uniform_mean[2]:.3f}, "
        f"shuffled_value_error={randomized_mean:.3f}, "
        f"aligned_error={aligned_error:.3f}, "
        f"misaligned_error={misaligned_error:.3f}, "
        f"random_nonexact={nonexact_random_partitions}/100"
    )

# %%

if MAIN:
    PLAY_BUDGET = 24

if MAIN:
    PLAY_SEED = 4

if MAIN:
    PLAY_STRATEGY: Literal["paired_kernel", "uniform"] = "paired_kernel"

if MAIN:
    PLAY_GROUPS = ((0, 3, 4), (1, 2, 5))

if MAIN:
    play_sample = sample_interior_coalitions(
        6,
        budget=PLAY_BUDGET,
        seed=PLAY_SEED,
        strategy=PLAY_STRATEGY,
    )

if MAIN:
    play_kernel = kernelshap_values(
        game_values, num_players=6, coalitions=play_sample
    )

if MAIN:
    play_partition = partition_shap_values(
        game_values, groups=PLAY_GROUPS
    )

if MAIN:
    print(
        f"KernelSHAP ({PLAY_STRATEGY}, budget={PLAY_BUDGET}, seed={PLAY_SEED}):",
        [round(value, 3) for value in play_kernel.tolist()],
    )

if MAIN:
    print(
        "Kernel max error:",
        round((play_kernel - exact_oracle).abs().max().item(), 3),
    )

if MAIN:
    print("Partition:", PLAY_GROUPS)

if MAIN:
    print(
        "Partition values:",
        [round(value, 3) for value in play_partition.tolist()],
    )

if MAIN:
    print(
        "Partition max error:",
        round((play_partition - exact_oracle).abs().max().item(), 3),
    )

# %%

from chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partition_shap_controls import solutions as reference

def run_smoke_test(cpu: bool = True) -> dict:
    return reference.run_smoke_test(cpu=cpu)

def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return reference.run_gpu_test(max_vram_gb=max_vram_gb)

def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return reference.run_full_experiment(max_vram_gb=max_vram_gb)

if MAIN:
    verification_summary = run_smoke_test(cpu=True)

if MAIN:
    verification_summary

# %%

def compare_kernelshap_play(*, dividends, budget: int, seed: int, groups) -> dict:
    """Recompute both sampled estimators and partition controls on one exact game."""
    values = game_from_dividends(6, dividends)
    oracle = shapley_from_dividends(6, dividends)
    full = kernelshap_values(values, num_players=6)
    estimates = {"Full interior table": full}
    samples = {}
    for strategy, label in (("paired_kernel", "Paired-kernel sample"), ("uniform", "Uniform sample")):
        samples[strategy] = sample_interior_coalitions(6, budget=budget, seed=seed, strategy=strategy)
        estimates[label] = kernelshap_values(values, num_players=6, coalitions=samples[strategy])
    estimates["Requested Owen groups"] = partition_shap_values(values, groups=groups)
    estimates["Aligned Owen groups"] = partition_shap_values(values, groups=((0, 1, 2), (3, 4, 5)))
    estimates["Singleton Owen groups"] = partition_shap_values(values, groups=tuple((i,) for i in range(6)))
    return {
        "dividends": dividends, "budget": budget, "seed": seed, "groups": groups,
        "values": values, "oracle": oracle, "samples": samples, "estimates": estimates,
        "errors": {name: float((estimate - oracle).abs().max()) for name, estimate in estimates.items()},
    }


def check_kernelshap_play(result: dict) -> None:
    """Independent ordering oracle and endpoint/pairing checks for the live play."""
    # This enumerates all player orderings, independently of dividend splitting.
    permutation_credit = np.zeros(6, dtype=np.float64)
    for ordering in itertools.permutations(range(6)):
        before = frozenset()
        for player in ordering:
            after = before | {player}
            permutation_credit[player] += result["values"][after] - result["values"][before]
            before = after
    expected = t.tensor(permutation_credit / math.factorial(6), dtype=t.float64)
    t.testing.assert_close(result["oracle"], expected, atol=1e-12, rtol=0)
    t.testing.assert_close(result["estimates"]["Full interior table"], expected, atol=1e-12, rtol=0)
    total = result["values"][frozenset(range(6))] - result["values"][frozenset()]
    for estimate in result["estimates"].values():
        assert abs(float(estimate.sum()) - total) < 1e-10
    paired = result["samples"]["paired_kernel"]
    assert len(paired) == len(set(paired)) == min(result["budget"], 62)
    assert all((frozenset(range(6)) - coalition) in paired for coalition in paired)
    print("Permutation oracle, full-table parity, efficiency and complement controls passed.")


if MAIN:
    # Change one knob, commit a forecast, then rerun only this CPU play cell.
    PLAY_PLOT_BUDGET = 24
    PLAY_PLOT_SEED = 4
    PLAY_PLOT_STRATEGY = "paired_kernel"
    PLAY_PLOT_GROUPS = ((0, 3, 4), (1, 2, 5))
    if PLAY_PLOT_STRATEGY not in ("paired_kernel", "uniform"):
        raise ValueError("Choose paired_kernel or uniform for the highlighted estimate.")
    plotted_play = compare_kernelshap_play(
        dividends=GAME_DIVIDENDS, budget=PLAY_PLOT_BUDGET,
        seed=PLAY_PLOT_SEED, groups=PLAY_PLOT_GROUPS,
    )
    check_kernelshap_play(plotted_play)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), constrained_layout=True)
    axes[0].plot(range(6), plotted_play["oracle"], color=EXACT, marker="*", linewidth=2.5, label="Exact dividend/permutation oracle")
    selected_label = "Paired-kernel sample" if PLAY_PLOT_STRATEGY == "paired_kernel" else "Uniform sample"
    for name, estimate in plotted_play["estimates"].items():
        axes[0].plot(range(6), estimate, marker="o", linewidth=2.5 if name == selected_label else 1, alpha=.85, label=name)
    axes[0].set_xticks(range(6), PLAYER_NAMES, rotation=20)
    axes[0].set(xlabel="Player", ylabel="Attribution", title="Live CPU play: credit and all controls")
    axes[0].legend(fontsize=7)
    names = list(plotted_play["errors"])
    axes[1].bar(range(len(names)), list(plotted_play["errors"].values()), color=[CONTROL, PAIRED, UNIFORM, "#8B6C9B", PAIRED, CONTROL])
    axes[1].set_xticks(range(len(names)), names, rotation=25, ha="right")
    axes[1].set(ylabel="Maximum absolute player error", title="Full-table, sampler and partition comparisons")
    fig.suptitle(f"One exact game: budget={PLAY_PLOT_BUDGET}, seed={PLAY_PLOT_SEED}; highlighted={PLAY_PLOT_STRATEGY}")
    (root_dir / chapter / "exercises" / section / "media").mkdir(exist_ok=True)
    fig.savefig(root_dir / chapter / "exercises" / section / "media/kernelshap_live_play.png", dpi=160)
    plt.show()
    print("Live CPU play max errors:", {name: round(error, 6) for name, error in plotted_play["errors"].items()})

# %%
