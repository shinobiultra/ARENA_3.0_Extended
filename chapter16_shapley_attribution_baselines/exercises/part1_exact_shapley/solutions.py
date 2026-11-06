# %%


MAIN = __name__ == "__main__"

# %%

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import itertools
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t

chapter = "chapter16_shapley_attribution_baselines"
section = "part1_exact_shapley"
root_dir = Path(__file__).resolve().parents[3]
exercises_dir = root_dir / chapter / "exercises"
assets_dir = exercises_dir / section / "media"
for path in (root_dir, exercises_dir):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import part1_exact_shapley.tests as tests

Coalition = frozenset[int]
if MAIN:
    t.set_grad_enabled(False)
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    })
EXACT = "#B43A32"
ORACLE = "#D39A2C"
BASELINE = "#167D8D"
CONTROL = "#5F6875"


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
            "coalition table has "
            f"missing coalitions={len(missing)} and extra coalitions={len(extra)}"
        )
    return {
        coalition: normalized[coalition]
        for coalition in all_coalitions(num_players)
    }


def coalition_values_from_function(
    num_players: int,
    value_fn: Callable[[Coalition], float],
) -> dict[Coalition, float]:
    return {
        coalition: float(value_fn(coalition))
        for coalition in all_coalitions(num_players)
    }
if MAIN:
    tests.test_all_coalitions_enumerates_the_power_set(all_coalitions)
    tests.test_complete_table_normalization(
        all_coalitions,
        normalize_coalition_values,
        coalition_values_from_function,
    )

# %%

PLAYER_NAMES = ("input", "memory", "router", "readout")
DIRECT_DIVIDENDS = {
    frozenset({0}): 0.8,
    frozenset({1}): -0.2,
    frozenset({2}): 0.4,
    frozenset({3}): 0.1,
}
INTERACTION_DIVIDENDS = {
    frozenset({0, 1}): 1.2,
    frozenset({1, 2}): -0.6,
    frozenset({2, 3}): 0.9,
    frozenset({0, 1, 2}): 1.5,
}
BASELINE_VALUE = 0.3

def scaled_dividends(interaction_scale: float = 1.0) -> dict[Coalition, float]:
    return DIRECT_DIVIDENDS | {
        term: float(interaction_scale) * value
        for term, value in INTERACTION_DIVIDENDS.items()
    }
def game_from_dividends(
    num_players: int,
    dividends: Mapping[Coalition | tuple[int, ...], float],
    baseline: float = 0.0,
) -> dict[Coalition, float]:
    normalized = {
        frozenset(term): float(value) for term, value in dividends.items()
    }
    if any(not term for term in normalized):
        raise ValueError("Use baseline for the empty dividend.")
    if any(player < 0 or player >= num_players for term in normalized for player in term):
        raise ValueError("Dividend contains a player outside the game.")
    return coalition_values_from_function(
        num_players,
        lambda coalition: float(baseline)
        + sum(
            value for term, value in normalized.items()
            if term.issubset(coalition)
        ),
    )


def shapley_from_dividends(
    num_players: int,
    dividends: Mapping[Coalition | tuple[int, ...], float],
) -> t.Tensor:
    result = t.zeros(num_players, dtype=t.float64)
    for raw_term, raw_dividend in dividends.items():
        term = frozenset(raw_term)
        if not term:
            raise ValueError("The empty dividend is a baseline, not player credit.")
        for player in term:
            result[player] += float(raw_dividend) / len(term)
    return result
if MAIN:
    tests.test_dividend_game_and_analytic_oracle(
        game_from_dividends,
        shapley_from_dividends,
    )
    organism_dividends = scaled_dividends(1.0)
    organism_values = game_from_dividends(
        4, organism_dividends, baseline=BASELINE_VALUE
    )
    analytic_oracle = shapley_from_dividends(4, organism_dividends)
    full_coalition = frozenset(range(4))
    print(
        f"Empty value = {organism_values[frozenset()]:.3f}; "
        f"full value = {organism_values[full_coalition]:.3f}; "
        f"total delta = {organism_values[full_coalition] - organism_values[frozenset()]:.3f}"
    )
    print("Analytic oracle =", [round(value, 3) for value in analytic_oracle.tolist()])
    print("\nComplete coalition table:")
    for coalition, value in organism_values.items():
        names = ", ".join(PLAYER_NAMES[player] for player in sorted(coalition)) or "empty"
        print(f"  {str(sorted(coalition)):<14} {names:<31} value={value:5.2f}")

# %%

def marginal_contribution_rows(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    player: int,
) -> tuple[tuple[Coalition, float, float], ...]:
    if player < 0 or player >= num_players:
        raise ValueError("player must index this game.")
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    rows = []
    denominator = math.factorial(num_players)
    for coalition in all_coalitions(num_players):
        if player in coalition:
            continue
        size = len(coalition)
        weight = (
            math.factorial(size)
            * math.factorial(num_players - size - 1)
            / denominator
        )
        marginal = values[coalition | {player}] - values[coalition]
        rows.append((coalition, marginal, weight))
    return tuple(rows)


def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    result = t.zeros(num_players, dtype=t.float64)
    for player in range(num_players):
        result[player] = sum(
            marginal * weight
            for _, marginal, weight in marginal_contribution_rows(
                coalition_values,
                num_players=num_players,
                player=player,
            )
        )
    return result


def additive_game(weights: t.Tensor) -> dict[Coalition, float]:
    weights = weights.detach().flatten().double()
    return coalition_values_from_function(
        int(weights.numel()),
        lambda coalition: weights[list(coalition)].sum().item()
        if coalition else 0.0,
    )
if MAIN:
    tests.test_weighted_marginal_rows(marginal_contribution_rows)
    tests.test_additive_game_and_exact_shapley_recover_weights(
        additive_game,
        exact_shapley_values,
    )
    tests.test_exact_shapley_requires_a_complete_coalition_table(
        exact_shapley_values
    )
    exact_values = exact_shapley_values(organism_values, num_players=4)
    print("Exact Shapley =", [round(value, 3) for value in exact_values.tolist()])

# %%

if MAIN:
    player_one_rows = marginal_contribution_rows(
        organism_values, num_players=4, player=1
    )
    print("Player 1 marginal contributions by predecessor context:")
    print("  coalition        marginal   weight   weighted")
    for coalition, marginal, weight in player_one_rows:
        print(
            f"  {str(sorted(coalition)):<16} {marginal:>7.2f}   "
            f"{weight:>6.3f}   {marginal * weight:>8.3f}"
        )
    print("  weighted sum:", round(sum(m * w for _, m, w in player_one_rows), 3))

# %%

def conjunction_game(num_players: int) -> dict[Coalition, float]:
    full = frozenset(range(num_players))
    return coalition_values_from_function(
        num_players, lambda coalition: coalition == full
    )


def shapley_efficiency_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> ShapleyEfficiencyReport:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    shapley_sum = exact_shapley_values(
        values, num_players=num_players
    ).sum().item()
    delta = (
        values[frozenset(range(num_players))]
        - values[frozenset()]
    )
    error = abs(shapley_sum - delta)
    return ShapleyEfficiencyReport(
        shapley_sum=shapley_sum,
        total_value_delta=delta,
        efficiency_error=error,
        satisfies_efficiency=error <= tolerance,
    )
if MAIN:
    tests.test_conjunction_game_splits_symmetric_credit_and_checks_efficiency(
        conjunction_game,
        exact_shapley_values,
        shapley_efficiency_report,
    )
    efficiency = shapley_efficiency_report(organism_values, num_players=4)
    oracle_error = (exact_values - analytic_oracle).abs().max().item()
    dummy_values = game_from_dividends(
        5, organism_dividends, baseline=BASELINE_VALUE
    )
    dummy_attribution = exact_shapley_values(dummy_values, num_players=5)[4].item()
    assert oracle_error < 1e-12
    assert abs(dummy_attribution) < 1e-12
    print(f"Oracle max error: {oracle_error:.2e}")
    print(f"Dummy player attribution: {dummy_attribution:.3f}")
    print(f"Efficiency error: {efficiency.efficiency_error:.2e}")

# %%

def permutation_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    result = t.zeros(num_players, dtype=t.float64)
    for ordering in itertools.permutations(range(num_players)):
        before = frozenset()
        for player in ordering:
            result[player] += (
                values[before | {player}] - values[before]
            )
            before = before | {player}
    return result / math.factorial(num_players)


def permutation_parity_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> PermutationParityReport:
    exact = exact_shapley_values(
        coalition_values, num_players=num_players
    )
    permutation = permutation_shapley_values(
        coalition_values, num_players=num_players
    )
    error = float((exact - permutation).abs().max().item())
    return PermutationParityReport(
        max_abs_error=error,
        matches_exact=error <= tolerance,
    )
if MAIN:
    tests.test_permutation_parity_report_matches_exact_formula(
        conjunction_game,
        permutation_parity_report,
    )
    permutation_values = permutation_shapley_values(
        organism_values, num_players=4
    )
    permutation_error = (permutation_values - exact_values).abs().max().item()
    print(
        f"Permutation max error: {permutation_error:.2e} "
        f"across {math.factorial(4)} orderings"
    )

# %%

def leave_one_out_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    full = frozenset(range(num_players))
    return t.tensor(
        [
            values[full] - values[full - {player}]
            for player in range(num_players)
        ],
        dtype=t.float64,
    )


def interaction_gap_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    min_overcount: float = 0.5,
) -> InteractionGapReport:
    shapley_total = exact_shapley_values(
        coalition_values, num_players=num_players
    ).sum().item()
    leave_one_out_total = leave_one_out_values(
        coalition_values, num_players=num_players
    ).sum().item()
    overcount = leave_one_out_total - shapley_total
    return InteractionGapReport(
        shapley_total=shapley_total,
        leave_one_out_total=leave_one_out_total,
        overcount=overcount,
        detects_interaction_overcount=overcount >= min_overcount,
    )
if MAIN:
    tests.test_interaction_gap_report_catches_leave_one_out_overcount(
        conjunction_game,
        interaction_gap_report,
    )
    leave_one_out = leave_one_out_values(organism_values, num_players=4)
    interaction_gap = interaction_gap_report(organism_values, num_players=4)
    print("Leave-one-out =", [round(value, 3) for value in leave_one_out.tolist()])
    print(
        f"Shapley total = {interaction_gap.shapley_total:.3f}; "
        f"leave-one-out total = {interaction_gap.leave_one_out_total:.3f}; "
        f"overcount = {interaction_gap.overcount:.3f}"
    )

# %%

def interaction_scale_sweep(scales: t.Tensor) -> dict[str, t.Tensor]:
    scales = scales.detach().flatten().double()
    efficiency_errors = []
    oracle_errors = []
    overcounts = []
    for scale in scales.tolist():
        dividends = scaled_dividends(scale)
        values = game_from_dividends(
            4, dividends, baseline=BASELINE_VALUE
        )
        exact = exact_shapley_values(values, num_players=4)
        oracle = shapley_from_dividends(4, dividends)
        efficiency_errors.append(
            shapley_efficiency_report(
                values, num_players=4
            ).efficiency_error
        )
        oracle_errors.append((exact - oracle).abs().max().item())
        overcounts.append(
            leave_one_out_values(values, num_players=4).sum().item()
            - exact.sum().item()
        )
    return {
        "shapley_efficiency_error": t.tensor(
            efficiency_errors, dtype=t.float64
        ),
        "oracle_max_error": t.tensor(
            oracle_errors, dtype=t.float64
        ),
        "leave_one_out_overcount": t.tensor(
            overcounts, dtype=t.float64
        ),
    }
if MAIN:
    tests.test_interaction_scale_sweep(interaction_scale_sweep)
    sweep_scales = t.linspace(0.0, 2.0, 9, dtype=t.float64)
    sweep = interaction_scale_sweep(sweep_scales)
    print(f"Additive control overcount = {sweep['leave_one_out_overcount'][0]:.3f}")
    print(f"Scale-1 overcount = {sweep['leave_one_out_overcount'][4]:.3f}")
    print(f"Scale-2 overcount = {sweep['leave_one_out_overcount'][-1]:.3f}")

# %%

if MAIN:
    player_zero_rows = marginal_contribution_rows(
        organism_values, num_players=4, player=0
    )
    additive_values = game_from_dividends(
        4, DIRECT_DIVIDENDS, baseline=BASELINE_VALUE
    )
    additive_player_zero_rows = marginal_contribution_rows(
        additive_values, num_players=4, player=0
    )

    assert oracle_error < 1e-12
    assert permutation_error < 1e-12
    assert efficiency.efficiency_error < 1e-12
    assert abs(interaction_gap.overcount - 4.5) < 1e-12
    assert abs(sweep["leave_one_out_overcount"][-1].item() - 9.0) < 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))
    x = np.arange(len(PLAYER_NAMES))
    width = 0.24
    axes[0].bar(x - width, analytic_oracle, width, color=ORACLE, label="dividend oracle")
    axes[0].bar(x, exact_values, width, color=EXACT, label="exact Shapley")
    axes[0].bar(x + width, leave_one_out, width, color=BASELINE, label="leave-one-out")
    axes[0].axhline(0, color="#B8BDC5", linewidth=1)
    axes[0].set_xticks(x, PLAYER_NAMES)
    axes[0].set_ylabel("attribution")
    axes[0].set_title("Known credit at scale 1")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        sweep_scales,
        sweep["leave_one_out_overcount"],
        marker="o",
        color=BASELINE,
        label="leave-one-out surplus",
    )
    axes[1].plot(
        sweep_scales,
        sweep["shapley_efficiency_error"],
        marker="o",
        color=EXACT,
        label="Shapley efficiency error",
    )
    axes[1].scatter([0], [0], s=100, facecolors="none", edgecolors=CONTROL, linewidths=2, label="additive control")
    axes[1].set_xlabel("interaction scale")
    axes[1].set_ylabel("attribution total minus game delta")
    axes[1].set_title("Interactions expose overcounting")
    axes[1].legend(frameon=False, fontsize=8)

    context_sizes = [len(coalition) for coalition, _, _ in player_zero_rows]
    marginals = [marginal for _, marginal, _ in player_zero_rows]
    additive_marginals = [marginal for _, marginal, _ in additive_player_zero_rows]
    offsets = np.linspace(-0.10, 0.10, len(context_sizes))
    axes[2].scatter(
        np.array(context_sizes) + offsets,
        marginals,
        color=EXACT,
        s=45,
        label="interaction game contexts",
    )
    axes[2].scatter(
        np.array(context_sizes) - offsets,
        additive_marginals,
        color=CONTROL,
        marker="x",
        s=45,
        label="additive control contexts",
    )
    axes[2].axhline(exact_values[0].item(), color=ORACLE, linestyle="--", label="Shapley weighted mean")
    axes[2].scatter([3], [leave_one_out[0].item()], color=BASELINE, marker="*", s=160, label="leave-one-out context")
    axes[2].set_xticks([0, 1, 2, 3])
    axes[2].set_xlabel("players preceding player 0")
    axes[2].set_ylabel("player 0 marginal contribution")
    axes[2].set_title("Why one context is not an average")
    axes[2].legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Exact Shapley conserves interaction credit; leave-one-out does not",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    signature_path = assets_dir / "exact_shapley_ground_truth_signature.png"
    fig.savefig(signature_path, dpi=170, bbox_inches="tight")
    plt.show()

    print(
        f"Signature metrics: oracle_error={oracle_error:.2e}, "
        f"permutation_error={permutation_error:.2e}, "
        f"efficiency_error={efficiency.efficiency_error:.2e}, "
        f"leave_one_out_overcount={interaction_gap.overcount:.3f}, "
        f"player0_marginal_range=[{min(marginals):.3f}, {max(marginals):.3f}]"
    )

# %%

if MAIN:
    PLAY_INTERACTION_SCALE = 0.5
    PLAY_PLAYER = 1

    play_dividends = scaled_dividends(PLAY_INTERACTION_SCALE)
    play_values = game_from_dividends(4, play_dividends, baseline=BASELINE_VALUE)
    play_shapley = exact_shapley_values(play_values, num_players=4)
    play_oracle = shapley_from_dividends(4, play_dividends)
    play_loo = leave_one_out_values(play_values, num_players=4)
    play_rows = marginal_contribution_rows(
        play_values, num_players=4, player=PLAY_PLAYER
    )
    t.testing.assert_close(play_shapley, play_oracle, atol=1e-12, rtol=0)
    print(f"Interaction scale: {PLAY_INTERACTION_SCALE:.2f}")
    print(f"Player: {PLAY_PLAYER} ({PLAYER_NAMES[PLAY_PLAYER]})")
    print(f"Shapley: {play_shapley[PLAY_PLAYER]:.3f}")
    print(f"Leave-one-out: {play_loo[PLAY_PLAYER]:.3f}")
    print(
        "Context marginals:",
        [round(marginal, 3) for _, marginal, _ in play_rows],
    )

# %%

def run_gpu_test(max_vram_gb: float = 24.0):
    from part1_exact_shapley_ground_truth_games.solutions import run_gpu_test as _run_gpu_test
    return _run_gpu_test(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0):
    return run_gpu_test(max_vram_gb=max_vram_gb)

# %%
