# %%


from __future__ import annotations
# Exact original literal defaults needed while Python defines functions.
FEATURES = 8
EMPIRICAL_ORDER = 2
MAIN = __name__ == "__main__"
from pathlib import Path as _MigrationPath
import sys as _migration_sys
from IPython.display import display
_MIGRATION_ROOT = _MigrationPath(__file__).resolve().parents[3]
if str(_MIGRATION_ROOT) not in _migration_sys.path:
    _migration_sys.path.insert(0, str(_MIGRATION_ROOT))
if str(_MIGRATION_ROOT / 'chapter16_shapley_attribution_baselines/exercises') not in _migration_sys.path:
    _migration_sys.path.insert(0, str(_MIGRATION_ROOT / 'chapter16_shapley_attribution_baselines/exercises'))

# %%

import itertools as toy_itertools

import math as toy_math

import random as toy_random

import sys as toy_sys

from collections.abc import Mapping as toy_Mapping

from dataclasses import dataclass

from pathlib import Path as toy_Path

import matplotlib.pyplot as toy_plt

import numpy as toy_np

import pandas as toy_pd

import torch as toy_t

if MAIN:
    toy_chapter = 'chapter16_shapley_attribution_baselines'

if MAIN:
    section = 'part3_shapley_interactions_shapiq'

if MAIN:
    toy_root_dir = next((p for p in [toy_Path.cwd(), *toy_Path.cwd().parents] if (p / toy_chapter).exists()))

if MAIN:
    exercises_dir = toy_root_dir / toy_chapter / 'exercises'

if MAIN:
    section_dir = exercises_dir / section

if MAIN:
    assets_dir = toy_root_dir / toy_chapter / 'instructions' / 'assets'

if MAIN:
    if str(toy_root_dir) not in toy_sys.path:
        toy_sys.path.append(str(toy_root_dir))

if MAIN:
    if str(exercises_dir) not in toy_sys.path:
        toy_sys.path.append(str(exercises_dir))

if MAIN:
    import chapter16_shapley_attribution_baselines.exercises.part3_shapley_interactions_shapiq.tests as toy_tests

if MAIN:
    Coalition = frozenset[int]

@dataclass(frozen=True)
class InteractionRecoveryReport:
    predicted_pair: tuple[int, int]
    target_rank: int
    target_value: float
    max_off_target_interaction: float
    mean_abs_error: float

if MAIN:
    PLAYER_NAMES = ['new_device', 'foreign_login', 'high_amount', 'many_attempts']

if MAIN:
    NUM_PLAYERS = len(PLAYER_NAMES)

if MAIN:
    TARGET_PAIR = (0, 1)

if MAIN:
    ADDITIVE_WEIGHTS = toy_t.tensor([0.2, 0.2, 5.0, 4.0], dtype=toy_t.float64)

if MAIN:
    PAIR_WEIGHT = 3.0

if MAIN:
    TRIPLE = (0, 1, 2)

if MAIN:
    TRIPLE_WEIGHT = 2.0

# %%

def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    if num_players <= 0:
        raise ValueError('num_players must be positive.')
    return tuple((frozenset(group) for size in range(num_players + 1) for group in toy_itertools.combinations(range(num_players), size)))


def normalize_coalition_values(coalition_values: toy_Mapping[Coalition | tuple[int, ...], float], *, num_players: int) -> dict[Coalition, float]:
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    missing = set(all_coalitions(num_players)) - set(values)
    if missing:
        raise ValueError(f'coalition table is missing {len(missing)} coalitions.')
    return values


def polynomial_game(additive_weights: toy_t.Tensor, interaction_terms: toy_Mapping[Coalition | tuple[int, ...], float]) -> dict[Coalition, float]:
    weights = additive_weights.flatten().double()
    num_players = int(weights.numel())
    if num_players <= 0:
        raise ValueError('additive_weights must contain at least one player.')
    terms = {frozenset(term): float(value) for term, value in interaction_terms.items()}
    for term in terms:
        if len(term) < 2:
            raise ValueError('interaction terms must contain at least two players.')
        if min(term) < 0 or max(term) >= num_players:
            raise ValueError('interaction term contains an invalid player index.')
    values: dict[Coalition, float] = {}
    for coalition in all_coalitions(num_players):
        additive = weights[list(coalition)].sum().item() if coalition else 0.0
        interaction = sum((value for term, value in terms.items() if term <= coalition))
        values[coalition] = additive + interaction
    return values

if MAIN:
    toy_tests.test_polynomial_game_enumerates_exact_ground_truth(polynomial_game)

# %%

def discrete_second_difference(coalition_values: toy_Mapping[Coalition | tuple[int, ...], float], coalition: Coalition | tuple[int, ...], pair: tuple[int, int], *, num_players: int) -> float:
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    context = frozenset(coalition)
    first, second = pair
    if first == second:
        raise ValueError('pair must contain two different players.')
    if not (0 <= first < num_players and 0 <= second < num_players):
        raise ValueError('pair contains an invalid player index.')
    if first in context or second in context:
        raise ValueError('coalition context must exclude both players in pair.')
    return values[context | {first, second}] - values[context | {first}] - values[context | {second}] + values[context]

if MAIN:
    toy_tests.test_discrete_second_difference_isolates_synergy_from_additive_effects(polynomial_game, discrete_second_difference)

# %%

def pairwise_shapley_interactions(coalition_values: toy_Mapping[Coalition | tuple[int, ...], float], *, num_players: int) -> toy_t.Tensor:
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    interactions = toy_t.zeros((num_players, num_players), dtype=toy_t.float64)
    denominator = toy_math.factorial(num_players - 1)
    for first, second in toy_itertools.combinations(range(num_players), 2):
        others = [player for player in range(num_players) if player not in (first, second)]
        score = 0.0
        for size in range(num_players - 1):
            weight = toy_math.factorial(size) * toy_math.factorial(num_players - size - 2) / denominator
            for group in toy_itertools.combinations(others, size):
                score += weight * discrete_second_difference(values, frozenset(group), (first, second), num_players=num_players)
        interactions[first, second] = score
        interactions[second, first] = score
    return interactions

if MAIN:
    toy_tests.test_pairwise_sii_recovers_pair_hidden_by_large_additive_effects(polynomial_game, pairwise_shapley_interactions)

# %%

def exact_shapley_values(coalition_values: toy_Mapping[Coalition | tuple[int, ...], float], *, num_players: int) -> toy_t.Tensor:
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    result = toy_t.zeros(num_players, dtype=toy_t.float64)
    denominator = toy_math.factorial(num_players)
    for player in range(num_players):
        others = [other for other in range(num_players) if other != player]
        for size in range(num_players):
            weight = toy_math.factorial(size) * toy_math.factorial(num_players - size - 1) / denominator
            for group in toy_itertools.combinations(others, size):
                coalition = frozenset(group)
                result[player] += weight * (values[coalition | {player}] - values[coalition])
    return result

if MAIN:
    toy_tests.test_exact_shapley_values_make_main_effect_ranking_misleading(polynomial_game, exact_shapley_values)

# %%

def sampled_pair_interaction(coalition_values: toy_Mapping[Coalition | tuple[int, ...], float], *, num_players: int, pair: tuple[int, int], budget: int, seed: int) -> float:
    if budget <= 0:
        raise ValueError('budget must be positive.')
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    first, second = pair
    if first == second or not (0 <= first < num_players and 0 <= second < num_players):
        raise ValueError('pair must contain two different valid players.')
    others = [player for player in range(num_players) if player not in pair]
    marker = object()
    rng = toy_random.Random(seed)
    deltas: list[float] = []
    for _ in range(budget):
        units: list[int | object] = [*others, marker]
        rng.shuffle(units)
        predecessor = frozenset((int(player) for player in units[:units.index(marker)] if player is not marker))
        deltas.append(discrete_second_difference(values, predecessor, pair, num_players=num_players))
    return sum(deltas) / budget


def permute_coalition_values_within_sizes(coalition_values: toy_Mapping[Coalition | tuple[int, ...], float], *, num_players: int, seed: int) -> dict[Coalition, float]:
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    rng = toy_random.Random(seed)
    permuted: dict[Coalition, float] = {}
    for size in range(num_players + 1):
        coalitions = sorted((coalition for coalition in values if len(coalition) == size), key=lambda coalition: tuple(sorted(coalition)))
        observed = [values[coalition] for coalition in coalitions]
        rng.shuffle(observed)
        permuted.update(zip(coalitions, observed, strict=True))
    return permuted

if MAIN:
    toy_tests.test_permutation_sampling_is_reproducible_and_converges(polynomial_game, pairwise_shapley_interactions, sampled_pair_interaction)

if MAIN:
    toy_tests.test_within_size_value_permutation_breaks_target_semantics(polynomial_game, pairwise_shapley_interactions, permute_coalition_values_within_sizes)

# %%

def shapiq_pairwise_interactions(coalition_values: toy_Mapping[Coalition | tuple[int, ...], float], *, num_players: int, index: str='SII') -> toy_t.Tensor:
    import shapiq
    values = normalize_coalition_values(coalition_values, num_players=num_players)

    def game(coalitions):
        array = toy_np.asarray(coalitions, dtype=bool)
        if array.ndim == 1:
            array = array[None, :]
        outputs = []
        for row in array:
            coalition = frozenset((player for player, present in enumerate(row.tolist()) if present))
            outputs.append(values[coalition])
        return toy_np.asarray(outputs, dtype=float)
    explainer = shapiq.AgnosticExplainer(game, n_players=num_players, index=index, max_order=2, random_state=0)
    result = explainer.explain(budget=2 ** num_players, random_state=0)
    matrix = toy_t.zeros((num_players, num_players), dtype=toy_t.float64)
    for first, second in toy_itertools.combinations(range(num_players), 2):
        key = (first, second)
        reverse = (second, first)
        value_index = result.interaction_lookup[key] if key in result.interaction_lookup else result.interaction_lookup[reverse]
        matrix[first, second] = float(result.values[value_index])
        matrix[second, first] = matrix[first, second]
    return matrix

if MAIN:
    toy_tests.test_shapiq_pairwise_matrix_matches_student_exact_sii(polynomial_game, pairwise_shapley_interactions, shapiq_pairwise_interactions)

# %%

def interaction_recovery_report(observed: toy_t.Tensor, expected: toy_t.Tensor, *, target_pair: tuple[int, int]) -> InteractionRecoveryReport:
    if observed.ndim != 2 or observed.shape[0] != observed.shape[1]:
        raise ValueError('observed must be a square interaction matrix.')
    if expected.shape != observed.shape:
        raise ValueError('expected must have the same shape as observed.')
    first, second = target_pair
    if first == second or not (0 <= first < observed.shape[0] and 0 <= second < observed.shape[0]):
        raise ValueError('target_pair must contain two different valid players.')
    pairs = list(toy_itertools.combinations(range(observed.shape[0]), 2))
    ranked = sorted(pairs, key=lambda pair: abs(float(observed[pair].item())), reverse=True)
    off_target = [abs(float(observed[pair].item())) for pair in pairs if frozenset(pair) != frozenset(target_pair)]
    upper = toy_t.triu_indices(observed.shape[0], observed.shape[1], offset=1)
    mean_abs_error = float((observed[upper[0], upper[1]] - expected[upper[0], upper[1]]).abs().mean().item())
    canonical_target = tuple(sorted(target_pair))
    return InteractionRecoveryReport(predicted_pair=ranked[0], target_rank=ranked.index(canonical_target) + 1, target_value=float(observed[target_pair].item()), max_off_target_interaction=max(off_target, default=0.0), mean_abs_error=mean_abs_error)

if MAIN:
    toy_tests.test_interaction_recovery_report_tracks_rank_spurious_terms_and_error(interaction_recovery_report)

# %%

if MAIN:
    planted_values = polynomial_game(ADDITIVE_WEIGHTS, {TARGET_PAIR: PAIR_WEIGHT, TRIPLE: TRIPLE_WEIGHT})

if MAIN:
    additive_values = polynomial_game(ADDITIVE_WEIGHTS, {})

if MAIN:
    planted_sii = pairwise_shapley_interactions(planted_values, num_players=NUM_PLAYERS)

if MAIN:
    additive_sii = pairwise_shapley_interactions(additive_values, num_players=NUM_PLAYERS)

if MAIN:
    main_effects = exact_shapley_values(planted_values, num_players=NUM_PLAYERS)

if MAIN:
    shapiq_sii = shapiq_pairwise_interactions(planted_values, num_players=NUM_PLAYERS)

if MAIN:
    shapiq_max_error = float((shapiq_sii - planted_sii).abs().max().item())

if MAIN:
    recovery = interaction_recovery_report(planted_sii, planted_sii, target_pair=TARGET_PAIR)

if MAIN:
    permuted_values = permute_coalition_values_within_sizes(planted_values, num_players=NUM_PLAYERS, seed=0)

if MAIN:
    permuted_sii = pairwise_shapley_interactions(permuted_values, num_players=NUM_PLAYERS)

if MAIN:
    budgets = toy_np.array([2, 4, 8, 16, 32, 64, 128, 256])

if MAIN:
    seeds = range(40)

if MAIN:
    true_exact_target = float(pairwise_shapley_interactions(planted_values, num_players=NUM_PLAYERS)[TARGET_PAIR])

def sampling_error_band(values):
    errors = toy_np.array([[abs(sampled_pair_interaction(values, num_players=NUM_PLAYERS, pair=TARGET_PAIR, budget=int(budget), seed=seed) - true_exact_target) for seed in seeds] for budget in budgets])
    return (errors.mean(axis=1), toy_np.quantile(errors, 0.1, axis=1), toy_np.quantile(errors, 0.9, axis=1))

if MAIN:
    sample_mean, sample_low, sample_high = sampling_error_band(planted_values)

if MAIN:
    control_mean, control_low, control_high = sampling_error_band(permuted_values)

if MAIN:
    summary = toy_pd.DataFrame([{'condition': 'exact pair + three-way game', 'target SII': float(planted_sii[TARGET_PAIR]), 'largest other |SII|': recovery.max_off_target_interaction, 'target rank': recovery.target_rank}, {'condition': 'matched additive control', 'target SII': float(additive_sii[TARGET_PAIR]), 'largest other |SII|': float(additive_sii.abs().max()), 'target rank': 'tie'}, {'condition': 'within-size value permutation', 'target SII': float(permuted_sii[TARGET_PAIR]), 'largest other |SII|': float(permuted_sii[toy_t.triu_indices(NUM_PLAYERS, NUM_PLAYERS, offset=1).unbind()].abs().max()), 'target rank': 'control'}])

if MAIN:
    display(summary)

if MAIN:
    display(toy_pd.DataFrame({'player': PLAYER_NAMES, 'individual Shapley value': main_effects.tolist(), 'member of planted pair': [True, True, False, False]}))

if MAIN:
    fig, axes = toy_plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

if MAIN:
    matrix_limit = 4.0

if MAIN:
    for ax, matrix, title in [(axes[0, 0], planted_sii, 'Exact SII: pair + three-way term'), (axes[0, 1], additive_sii, 'Matched additive control')]:
        image = ax.imshow(matrix.numpy(), cmap='coolwarm', vmin=-matrix_limit, vmax=matrix_limit)
        ax.set_xticks(range(NUM_PLAYERS), PLAYER_NAMES, rotation=24, ha='right')
        ax.set_yticks(range(NUM_PLAYERS), PLAYER_NAMES)
        ax.set_title(title, loc='left', fontweight='bold')
        for toy_row in range(NUM_PLAYERS):
            for column in range(NUM_PLAYERS):
                toy_value = matrix[toy_row, column].item()
                ax.text(column, toy_row, f'{toy_value:.2f}', ha='center', va='center', color='white' if abs(toy_value) > 1.5 else '#17202a')

if MAIN:
    fig.colorbar(image, ax=axes[0, :], shrink=0.82, label='second-order SII')

if MAIN:
    colors = ['#16828f', '#16828f', '#d29500', '#d29500']

if MAIN:
    axes[1, 0].bar(PLAYER_NAMES, main_effects.numpy(), color=colors)

if MAIN:
    axes[1, 0].set_ylabel('individual Shapley value')

if MAIN:
    axes[1, 0].set_title('Misleading main-effect control', loc='left', fontweight='bold')

if MAIN:
    axes[1, 0].tick_params(axis='x', rotation=24)

if MAIN:
    for index, toy_value in enumerate(main_effects.tolist()):
        axes[1, 0].text(index, toy_value + 0.1, f'{toy_value:.1f}', ha='center', fontweight='bold')

if MAIN:
    axes[1, 0].plot([0, 0, 1, 1], [2.55, 2.7, 2.7, 2.55], color='#16828f', linewidth=2)

if MAIN:
    axes[1, 0].text(0.5, 2.77, 'planted pair', ha='center', color='#16828f', fontweight='bold')

if MAIN:
    axes[1, 1].plot(budgets, sample_mean, marker='o', color='#16828f', label='correct coalition labels')

if MAIN:
    axes[1, 1].fill_between(budgets, sample_low, sample_high, color='#16828f', alpha=0.18)

if MAIN:
    axes[1, 1].plot(budgets, control_mean, marker='s', color='#c44e52', label='within-size value permutation')

if MAIN:
    axes[1, 1].fill_between(budgets, control_low, control_high, color='#c44e52', alpha=0.14)

if MAIN:
    axes[1, 1].axhline(0.0, color='#17202a', linestyle='--', label='exact enumeration')

if MAIN:
    axes[1, 1].set_xscale('log', base=2)

if MAIN:
    axes[1, 1].set_xticks(budgets, [str(value) for value in budgets])

if MAIN:
    axes[1, 1].set_xlabel('sampled merged-player permutations')

if MAIN:
    axes[1, 1].set_ylabel('mean absolute error to true target SII')

if MAIN:
    axes[1, 1].set_title('Sampling convergence and permutation control', loc='left', fontweight='bold')

if MAIN:
    axes[1, 1].legend(loc='upper right')

if MAIN:
    fig.suptitle('Pair interactions recover structure that individual credit hides', fontsize=17, fontweight='bold')

if MAIN:
    signature_path = assets_dir / 'shapley_interactions_exact_signature.png'

if MAIN:
    fig.savefig(signature_path, dpi=170, bbox_inches='tight')

if MAIN:
    toy_plt.show()

if MAIN:
    assert recovery.predicted_pair == TARGET_PAIR and recovery.target_rank == 1

if MAIN:
    assert abs(recovery.target_value - 4.0) < 1e-09

if MAIN:
    assert abs(recovery.max_off_target_interaction - 1.0) < 1e-09

if MAIN:
    assert abs(float(planted_sii[0, 2]) - 1.0) < 1e-09

if MAIN:
    assert abs(float(planted_sii[1, 2]) - 1.0) < 1e-09

if MAIN:
    assert float(additive_sii.abs().max()) < 1e-09

if MAIN:
    assert set(main_effects.topk(2).indices.tolist()) == {2, 3}

if MAIN:
    assert shapiq_max_error < 1e-06

if MAIN:
    assert sample_mean[-1] < 0.06

if MAIN:
    assert control_mean[-1] > 3.5

if MAIN:
    print(f'shapiq max |error|: {shapiq_max_error:.3e}')

if MAIN:
    print(f'sampling MAE at budget 256: {sample_mean[-1]:.6f}')

if MAIN:
    print(f'permuted-control MAE at budget 256: {control_mean[-1]:.6f}')

# %%

if MAIN:
    PLAY_ADDITIVE_WEIGHTS = toy_t.tensor([0.2, 0.2, 5.0, 4.0], dtype=toy_t.float64)

if MAIN:
    PLAY_PAIR = (0, 1)

if MAIN:
    PLAY_PAIR_WEIGHT = 3.0

if MAIN:
    PLAY_TRIPLE = (0, 1, 2)

if MAIN:
    PLAY_TRIPLE_WEIGHT = 2.0

if MAIN:
    PLAY_BUDGET = 32

if MAIN:
    PLAY_SEED = 7

if MAIN:
    play_terms = {PLAY_PAIR: PLAY_PAIR_WEIGHT}

if MAIN:
    if PLAY_TRIPLE_WEIGHT != 0.0:
        play_terms[PLAY_TRIPLE] = PLAY_TRIPLE_WEIGHT

if MAIN:
    play_values = polynomial_game(PLAY_ADDITIVE_WEIGHTS, play_terms)

if MAIN:
    play_sii = pairwise_shapley_interactions(play_values, num_players=NUM_PLAYERS)

if MAIN:
    play_main = exact_shapley_values(play_values, num_players=NUM_PLAYERS)

if MAIN:
    play_sample = sampled_pair_interaction(play_values, num_players=NUM_PLAYERS, pair=PLAY_PAIR, budget=PLAY_BUDGET, seed=PLAY_SEED)

if MAIN:
    display(toy_pd.DataFrame([{'pair': f'{PLAYER_NAMES[first]} x {PLAYER_NAMES[second]}', 'exact SII': float(play_sii[first, second])} for first, second in toy_itertools.combinations(range(NUM_PLAYERS), 2)]))

if MAIN:
    print('individual Shapley values:', dict(zip(PLAYER_NAMES, play_main.tolist(), strict=True)))

if MAIN:
    print(f'sampled target-pair SII (PLAY_BUDGET={PLAY_BUDGET!r}): {play_sample:.4f}')

# %%

if MAIN:
    planted_sii = pairwise_shapley_interactions(planted_values, num_players=NUM_PLAYERS)

if MAIN:
    anomaly_rows = []

if MAIN:
    for toy_pair in [(0, 1), (0, 2), (1, 2)]:
        remaining = [player for player in range(NUM_PLAYERS) if player not in toy_pair]
        empty_delta = discrete_second_difference(planted_values, frozenset(), toy_pair, num_players=NUM_PLAYERS)
        revealing_context = frozenset((next((p for p in (0, 1, 2) if p not in toy_pair), remaining[0]),))
        context_delta = discrete_second_difference(planted_values, revealing_context, toy_pair, num_players=NUM_PLAYERS)
        anomaly_rows.append({'pair': f'{PLAYER_NAMES[toy_pair[0]]} x {PLAYER_NAMES[toy_pair[1]]}', 'empty-context delta': empty_delta, 'revealing-context delta': context_delta, 'pairwise SII': float(planted_sii[toy_pair])})

if MAIN:
    display(toy_pd.DataFrame(anomaly_rows))

# %%

if MAIN:
    neural_target_values = polynomial_game(toy_t.tensor([1.2, -0.7, 1.6, 0.9], dtype=toy_t.float64), {(0, 2): 2.2, (1, 3): -1.5})

if MAIN:
    neural_target_sii = pairwise_shapley_interactions(neural_target_values, num_players=4)

if MAIN:
    display(toy_pd.DataFrame([{'pair': pair, 'exact target SII': float(neural_target_sii[pair])} for pair in toy_itertools.combinations(range(4), 2)]))

# %%

if MAIN:
    verification_report_path = section_dir / 'verification_report.json'

def run_gpu_test(max_vram_gb: float=24.0) -> dict:
    from part3_shapley_interactions_shapiq.solutions import run_gpu_test as run
    return run(max_vram_gb=max_vram_gb)

def run_full_experiment(max_vram_gb: float=24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)

# %%

from fractions import Fraction

if MAIN:
    cold_open = {'Shapley-Taylor': (Fraction(0), Fraction(1, 6)), 'Faith-Shap': (Fraction(-1, 5), Fraction(3, 10)), 'Shapley / pair SII': (Fraction(1, 4), Fraction(1, 3))}

if MAIN:
    for name, (singleton, allocation_pair) in cold_open.items():
        print(f'{name:20s} singleton={str(singleton):>4s}  pair={str(allocation_pair):>4s}')

if MAIN:
    print('ST efficient:', 4 * cold_open['Shapley-Taylor'][0] + 6 * cold_open['Shapley-Taylor'][1])

if MAIN:
    print('Faith efficient:', 4 * cold_open['Faith-Shap'][0] + 6 * cold_open['Faith-Shap'][1])

# %%

import contextlib

import hashlib

import math as allocation_math

import os as allocation_os

import random as allocation_random

from collections.abc import Iterator, Mapping as allocation_Mapping

from pathlib import Path as allocation_Path

from types import SimpleNamespace as allocation_SimpleNamespace

from typing import Any as allocation_Any

import matplotlib.pyplot as allocation_plt

import numpy as allocation_np

import pandas as allocation_pd

import torch

import torch as allocation_t

from torch import Tensor as allocation_Tensor, nn

from IPython.display import display as allocation_display

if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import interaction_methods_tests as allocation_tests

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_methods import _interaction_counts, _interaction_float_input, _interaction_game_batches, _interaction_model, _interaction_model_output, _interaction_n_players, _interaction_order, _interaction_preserve_model_state, _interaction_tensor

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_diagnostics import GAME_NAMES, PAIR_INDICES, _callable, _diagnostic, _ordinary_shapley_from_mobius, _pairwise_sii_from_mobius, _zeta, interaction_diagnostics

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_plots import plot_allrow_pair_heatmaps, plot_background_changes_the_game, plot_contextual_pair_interventions, plot_dividend_play, plot_donor_context_play, plot_exact_allocation_ledger, plot_fixed_row_interaction_graph, plot_heldout_pair_order_capture, plot_native_game_and_controls, plot_order_two_reconstruction, plot_reconstruction_reason

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_data import background_order, load_manifest, load_split, tensor_mapping_sha256

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_reference import _preserve_execution_state, _reference_check_impl, _validate_exact_suite, validate_reference_payload

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.run_native import persist_completed, persist_prepared, write_notebook_summary

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_study import _METHOD_FIRST_LEAVES, _SHARED_GAME_LEAVES, _background_records, _clone_state, _combine_game_blocks, _derived_seed, _direct_full_context_four, _empirical_reference_plan, _evaluate_backgrounds, _evaluate_primary, _exact_suite, _fit_one_mlp, _fit_ridge, _float_tensor, _group_weighted_mean_abs, _int_tensor, _localization_statistics, _logical_hybrid_row, _pair_ranks, _performance, _require_disjoint_ids, _require_methods, _rows, _storage_plan, _training_normalization, _validate_fitted, _within_size_permutations, coalition_masks, exact_dividend_play, nearest_training_diagnostics, pair_indices

if MAIN:
    DEVICE = torch.device('cpu')

if MAIN:
    DTYPE = torch.float64

if MAIN:
    torch.set_default_dtype(DTYPE)

if MAIN:
    torch.set_num_threads(4)

if MAIN:
    FIGURE_DIR = allocation_Path(allocation_os.environ.get('ARENA_INTERACTION_OUTPUT_DIRECTORY', './figures'))

if MAIN:
    native_directory = FIGURE_DIR / 'native'

def allocation_show_figure(figure, stem: str):
    """Display one result and retain the exact PNG used by the draft execution."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f'{stem}.png'
    figure.savefig(path, dpi=160, bbox_inches='tight')
    allocation_display(figure)
    allocation_plt.close(figure)
    return path

# %%

def mobius_transform(game_values: allocation_Tensor, *, n_players: int) -> allocation_Tensor:
    """Return a full bitmask-indexed Möbius coefficient tensor."""
    n_players = _interaction_n_players(n_players)
    _interaction_tensor('game_values', game_values, n_players)
    mobius = game_values.clone()
    masks = torch.arange(2 ** n_players, dtype=torch.int64)
    for player in range(n_players):
        present = masks.bitwise_and(1 << player).ne(0)
        lower = masks[present].bitwise_xor(1 << player)
        mobius[..., present] -= mobius[..., lower]
    return mobius

if MAIN:
    allocation_tests.test_mobius_transform(fn=mobius_transform)

# %%

def _fixture_mobius() -> allocation_Tensor:
    mobius = torch.zeros((len(GAME_NAMES), 16), dtype=torch.float64)
    mobius[0, 0] = 2.0
    mobius[1, 0] = -0.5
    mobius[1, [1, 2, 4, 8]] = torch.tensor([1.0, -2.0, 1.5, 0.25], dtype=torch.float64)
    mobius[2, 3] = 1.0
    mobius[3, 10] = -0.75
    mobius[4, 7] = 1.0
    mobius[5, 15] = 1.0
    mobius[6, 0] = 1 / 3
    mobius[6, [1, 2, 4, 8]] = torch.tensor([0.5, -0.25, 0.75, -1.0], dtype=torch.float64)
    for mask, value in {3: 2 / 3, 9: -0.5, 6: 0.2, 12: 0.375}.items():
        mobius[6, mask] = value
    mobius[7, 0] = -0.2
    mobius[7, 1] = 1 / 3
    mobius[7, 6] = -0.4
    mobius[7, 13] = 0.75
    mobius[7, 15] = -0.5
    return mobius

def exact_game_fixture() -> dict[str, allocation_Any]:
    """Return the declared analytic games and exact SII comparators."""
    mobius = _fixture_mobius()
    games = _zeta(mobius, 4)
    return {'evidence_kind': 'analytic_exact', 'label': 'exact teaching games', 'n_players': 4, 'order': 2, 'feature_names': ('A', 'B', 'C', 'D'), 'game_names': GAME_NAMES, 'game_values': games, 'expected_mobius': mobius, 'ordinary_shapley': _ordinary_shapley_from_mobius(mobius, 4), 'sii_pairs': _pairwise_sii_from_mobius(mobius, 4), 'pair_indices': PAIR_INDICES, 'single_masks': torch.tensor([1, 2, 4, 8], dtype=torch.int64), 'pair_masks': torch.tensor([1 << first | 1 << second for first, second in PAIR_INDICES], dtype=torch.int64), 'exact_fraction_labels': {'pure_triple': {'stii_singleton': '0', 'stii_contained_pair': '1/3', 'fsii_singleton': '-1/6', 'fsii_contained_pair': '1/2', 'shapley_singleton': '1/3', 'sii_contained_pair': '1/2'}, 'pure_four': {'stii_singleton': '0', 'stii_pair': '1/6', 'fsii_singleton': '-1/5', 'fsii_pair': '3/10', 'shapley_singleton': '1/4', 'sii_pair': '1/3'}}}

# %%

def diagnose_mobius_games(mobius_transform_fn: Callable[..., allocation_Tensor], diagnostic: dict[str, allocation_Any] | None=None) -> dict[str, allocation_Any]:
    """Run the learner Möbius transform on all eight exact fixtures."""
    _callable('mobius_transform_fn', mobius_transform_fn)
    result = _diagnostic(diagnostic)
    observed = mobius_transform_fn(result['game_values'], n_players=4)
    torch.testing.assert_close(observed, result['expected_mobius'], rtol=0, atol=1e-12)
    result['mobius'] = observed
    result['mobius_max_abs_error'] = float((observed - result['expected_mobius']).abs().max().item())
    result['stages'] = tuple(result.get('stages', ())) + ('mobius',)
    return result

# %%

if MAIN:
    exact_diagnostics = diagnose_mobius_games(mobius_transform, diagnostic=exact_game_fixture())

if MAIN:
    print('max |Möbius error|:', exact_diagnostics['mobius_max_abs_error'])

if MAIN:
    for game_name, allocation_row in zip(exact_diagnostics['game_names'], exact_diagnostics['mobius'], strict=True):
        sparse = {f'{mask:04b}': round(float(value), 6) for mask, value in enumerate(allocation_row) if abs(float(value)) > 1e-12}
        print(f'{game_name:16s}', sparse)

# %%

def shapley_taylor_interactions(mobius: allocation_Tensor, *, n_players: int, order: int) -> allocation_Tensor:
    """Allocate Möbius dividends into an efficient order-truncated expansion."""
    n_players = _interaction_n_players(n_players)
    order = _interaction_order('order', order, n_players)
    _interaction_tensor('mobius', mobius, n_players)
    counts = _interaction_counts(n_players)
    result = torch.zeros_like(mobius)
    result[..., 0] = mobius[..., 0]
    lower = (counts > 0) & (counts < order)
    result[..., lower] = mobius[..., lower]
    for interaction in range(1, 2 ** n_players):
        if interaction.bit_count() != order:
            continue
        for superset in range(interaction, 2 ** n_players):
            if superset & interaction == interaction:
                result[..., interaction] += mobius[..., superset] / allocation_math.comb(superset.bit_count(), order)
    return result

if MAIN:
    allocation_tests.test_shapley_taylor_interactions(fn=shapley_taylor_interactions)

# %%

def diagnose_shapley_taylor(diagnostic: dict[str, allocation_Any], shapley_taylor_fn: Callable[..., allocation_Tensor]) -> dict[str, allocation_Any]:
    """Add student Shapley-Taylor allocations and exact fraction checks."""
    _callable('shapley_taylor_fn', shapley_taylor_fn)
    result = _diagnostic(diagnostic)
    if 'mobius' not in result:
        raise ValueError('Run diagnose_mobius_games before Shapley-Taylor diagnostics.')
    coefficients = shapley_taylor_fn(result['mobius'], n_players=4, order=2)
    if coefficients.shape != result['game_values'].shape:
        raise AssertionError('Shapley-Taylor must return one coefficient per mask.')
    pure_triple = GAME_NAMES.index('pure_triple')
    pure_four = GAME_NAMES.index('pure_four')
    singles, pairs = (result['single_masks'], result['pair_masks'])
    torch.testing.assert_close(coefficients[pure_four, singles], torch.zeros(4, dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(coefficients[pure_four, pairs], torch.full((6,), 1 / 6, dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(coefficients[pure_triple, singles], torch.zeros(4, dtype=torch.float64), rtol=0, atol=1e-12)
    contained = torch.tensor([0, 1, 3], dtype=torch.int64)
    torch.testing.assert_close(coefficients[pure_triple, pairs[contained]], torch.full((3,), 1 / 3, dtype=torch.float64), rtol=0, atol=1e-12)
    efficiency = coefficients.sum(dim=-1) - result['game_values'][:, -1]
    torch.testing.assert_close(efficiency, torch.zeros_like(efficiency), rtol=0, atol=1e-12)
    result['stii'] = coefficients
    result['stii_efficiency_residual'] = efficiency
    result['stages'] = tuple(result.get('stages', ())) + ('shapley_taylor',)
    return result

# %%

if MAIN:
    exact_diagnostics = diagnose_shapley_taylor(exact_diagnostics, shapley_taylor_interactions)

if MAIN:
    print('max |efficiency residual|:', float(exact_diagnostics['stii_efficiency_residual'].abs().max()))

# %%

def faith_design_matrix(*, n_players: int, max_order: int) -> tuple[allocation_Tensor, allocation_Tensor]:
    """Return float64 X[coalition_mask, term] and int64 term_masks."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order('max_order', max_order, n_players)
    masks = torch.arange(2 ** n_players, dtype=torch.int64)
    counts = _interaction_counts(n_players)
    term_masks = masks[counts <= max_order]
    design = masks[:, None].bitwise_and(term_masks[None, :]).eq(term_masks).to(torch.float64)
    return (design, term_masks)


def faith_kernel_weights(*, n_players: int) -> allocation_Tensor:
    """Return normalized interior per-coalition weights; endpoints are zero."""
    n_players = _interaction_n_players(n_players, minimum=2)
    weights = torch.zeros(2 ** n_players, dtype=torch.float64)
    for mask in range(1, 2 ** n_players - 1):
        size = mask.bit_count()
        weights[mask] = (n_players - 1) / (allocation_math.comb(n_players, size) * size * (n_players - size))
    return weights / weights.sum()

if MAIN:
    allocation_tests.test_faith_design_matrix(fn=faith_design_matrix)

if MAIN:
    allocation_tests.test_faith_kernel_weights(fn=faith_kernel_weights)

# %%

def diagnose_faith_design(diagnostic: dict[str, allocation_Any], faith_design_matrix_fn: Callable[..., tuple[allocation_Tensor, allocation_Tensor]], faith_kernel_weights_fn: Callable[..., allocation_Tensor]) -> dict[str, allocation_Any]:
    """Check the visible Faith design and per-coalition kernel semantics."""
    _callable('faith_design_matrix_fn', faith_design_matrix_fn)
    _callable('faith_kernel_weights_fn', faith_kernel_weights_fn)
    result = _diagnostic(diagnostic)
    design, term_masks = faith_design_matrix_fn(n_players=3, max_order=2)
    weights = faith_kernel_weights_fn(n_players=3)
    weights_n4 = faith_kernel_weights_fn(n_players=4)
    if design.shape != (8, 7) or term_masks.shape != (7,):
        raise AssertionError('For n=3, order=2 the Faith design must have shape [8,7].')
    expected_terms = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)
    torch.testing.assert_close(term_masks, expected_terms, rtol=0, atol=0)
    expected = torch.arange(8, dtype=torch.int64)[:, None].bitwise_and(term_masks[None, :]).eq(term_masks).to(torch.float64)
    torch.testing.assert_close(design, expected, rtol=0, atol=0)
    if weights.dtype != torch.float64 or weights.shape != (8,):
        raise AssertionError('Faith kernel weights must be float64 with one value per coalition.')
    torch.testing.assert_close(weights[[0, -1]], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-12)
    expected_n4 = torch.zeros(16, dtype=torch.float64)
    for mask in range(1, 15):
        size = mask.bit_count()
        expected_n4[mask] = 3 / (allocation_math.comb(4, size) * size * (4 - size))
    expected_n4 /= expected_n4.sum()
    torch.testing.assert_close(weights_n4, expected_n4, rtol=0, atol=1e-12)
    result['faith_design'] = design
    result['faith_term_masks'] = term_masks
    result['faith_kernel_weights'] = weights
    result['faith_kernel_weights_n4'] = weights_n4
    result['stages'] = tuple(result.get('stages', ())) + ('faith_design',)
    return result

# %%

if MAIN:
    exact_diagnostics = diagnose_faith_design(exact_diagnostics, faith_design_matrix, faith_kernel_weights)

if MAIN:
    print('n=3 design shape:', tuple(exact_diagnostics['faith_design'].shape))

if MAIN:
    print('n=3 kernel sum:', float(exact_diagnostics['faith_kernel_weights'].sum()))

# %%

def faith_shap_interactions(game_values: allocation_Tensor, *, n_players: int, max_order: int) -> allocation_Tensor:
    """Solve exact equality-constrained weighted least squares."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order('max_order', max_order, n_players)
    _interaction_tensor('game_values', game_values, n_players)
    if max_order == n_players:
        return mobius_transform(game_values, n_players=n_players)
    design, term_masks = faith_design_matrix(n_players=n_players, max_order=max_order)
    weights = faith_kernel_weights(n_players=n_players)
    interior = design[1:-1]
    weighted_design = weights[1:-1, None] * interior
    gram = interior.T @ weighted_design
    constraints = torch.stack((design[0], design[-1]))
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    kkt = torch.cat((torch.cat((gram, constraints.T), dim=1), torch.cat((constraints, zeros), dim=1)), dim=0)
    values = game_values.reshape(-1, 2 ** n_players)
    weighted_targets = values[:, 1:-1] @ weighted_design
    endpoint_targets = torch.stack((values[:, 0], values[:, -1]), dim=1)
    right_hand_side = torch.cat((weighted_targets, endpoint_targets), dim=1)
    solution = torch.linalg.solve(kkt, right_hand_side.T).T
    result = torch.zeros_like(values)
    result[:, term_masks] = solution[:, :len(term_masks)]
    return result.reshape(game_values.shape)

if MAIN:
    allocation_tests.test_faith_shap_interactions(fn=faith_shap_interactions)

# %%

def faith_shap_closed_form(mobius: allocation_Tensor, *, n_players: int, max_order: int) -> allocation_Tensor:
    """Compute the independent primary-paper Möbius formula."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order('max_order', max_order, n_players)
    _interaction_tensor('mobius', mobius, n_players)
    result = torch.zeros_like(mobius)
    result[..., 0] = mobius[..., 0]
    for interaction in range(1, 2 ** n_players):
        size = interaction.bit_count()
        if size > max_order:
            continue
        correction = torch.zeros_like(mobius[..., 0])
        for superset in range(2 ** n_players):
            superset_size = superset.bit_count()
            if superset_size > max_order and superset & interaction == interaction:
                correction += allocation_math.comb(superset_size - 1, max_order) / allocation_math.comb(superset_size + max_order - 1, max_order + size) * mobius[..., superset]
        scale = (-1) ** (max_order - size) * size / (max_order + size) * allocation_math.comb(max_order, size)
        result[..., interaction] = mobius[..., interaction] + scale * correction
    return result

if MAIN:
    allocation_tests.test_faith_shap_closed_form(fn=faith_shap_closed_form)

# %%

def diagnose_faith_solver(diagnostic: dict[str, allocation_Any], faith_shap_fn: Callable[..., allocation_Tensor], faith_closed_form_fn: Callable[..., allocation_Tensor]) -> dict[str, allocation_Any]:
    """Add Faith-Shap coefficients and independent closed-form parity."""
    _callable('faith_shap_fn', faith_shap_fn)
    _callable('faith_closed_form_fn', faith_closed_form_fn)
    result = _diagnostic(diagnostic)
    if 'mobius' not in result or 'faith_design' not in result:
        raise ValueError('Run Möbius and Faith-design diagnostics before the Faith solver.')
    coefficients = faith_shap_fn(result['game_values'], n_players=4, max_order=2)
    closed = faith_closed_form_fn(result['mobius'], n_players=4, max_order=2)
    torch.testing.assert_close(coefficients, closed, rtol=0, atol=1e-10)
    pure_triple = GAME_NAMES.index('pure_triple')
    pure_four = GAME_NAMES.index('pure_four')
    singles, pairs = (result['single_masks'], result['pair_masks'])
    torch.testing.assert_close(coefficients[pure_four, singles], torch.full((4,), -1 / 5, dtype=torch.float64), rtol=0, atol=1e-10)
    torch.testing.assert_close(coefficients[pure_four, pairs], torch.full((6,), 3 / 10, dtype=torch.float64), rtol=0, atol=1e-10)
    torch.testing.assert_close(coefficients[pure_triple, singles[:3]], torch.full((3,), -1 / 6, dtype=torch.float64), rtol=0, atol=1e-10)
    contained = torch.tensor([0, 1, 3], dtype=torch.int64)
    torch.testing.assert_close(coefficients[pure_triple, pairs[contained]], torch.full((3,), 1 / 2, dtype=torch.float64), rtol=0, atol=1e-10)
    efficiency = coefficients.sum(dim=-1) - result['game_values'][:, -1]
    torch.testing.assert_close(efficiency, torch.zeros_like(efficiency), rtol=0, atol=1e-10)
    result['fsii'] = coefficients
    result['fsii_closed_form'] = closed
    result['fsii_closed_form_max_abs_error'] = float((coefficients - closed).abs().max().item())
    result['fsii_efficiency_residual'] = efficiency
    result['stages'] = tuple(result.get('stages', ())) + ('faith_solver',)
    return result

# %%

if MAIN:
    exact_diagnostics = diagnose_faith_solver(exact_diagnostics, faith_shap_interactions, faith_shap_closed_form)

if MAIN:
    print('max |KKT - closed form|:', exact_diagnostics['fsii_closed_form_max_abs_error'])

# %%

def reconstruct_game(coefficients: allocation_Tensor, *, n_players: int) -> allocation_Tensor:
    """Apply the subset zeta transform to interaction coefficients."""
    n_players = _interaction_n_players(n_players)
    _interaction_tensor('coefficients', coefficients, n_players)
    values = coefficients.clone()
    masks = torch.arange(2 ** n_players, dtype=torch.int64)
    for player in range(n_players):
        present = masks.bitwise_and(1 << player).ne(0)
        lower = masks[present].bitwise_xor(1 << player)
        values[..., present] += values[..., lower]
    return values


def faithfulness_error(game_values: allocation_Tensor, coefficients: allocation_Tensor, *, n_players: int) -> allocation_Tensor:
    """Return normalized Shapley-kernel interior MSE for each leading item."""
    n_players = _interaction_n_players(n_players, minimum=2)
    _interaction_tensor('game_values', game_values, n_players)
    _interaction_tensor('coefficients', coefficients, n_players)
    if game_values.shape != coefficients.shape:
        raise ValueError('game_values and coefficients must have the same shape.')
    residual = game_values - reconstruct_game(coefficients, n_players=n_players)
    weights = faith_kernel_weights(n_players=n_players)
    return (weights * residual.square()).sum(dim=-1)

if MAIN:
    allocation_tests.test_reconstruct_game(fn=reconstruct_game)

if MAIN:
    allocation_tests.test_faithfulness_error(fn=faithfulness_error)

# %%

def diagnose_reconstruction(diagnostic: dict[str, allocation_Any], reconstruct_game_fn: Callable[..., allocation_Tensor], faithfulness_error_fn: Callable[..., allocation_Tensor]) -> dict[str, allocation_Any]:
    """Add same-game reconstructions and the declared Faith objective."""
    _callable('reconstruct_game_fn', reconstruct_game_fn)
    _callable('faithfulness_error_fn', faithfulness_error_fn)
    result = _diagnostic(diagnostic)
    if 'stii' not in result or 'fsii' not in result:
        raise ValueError('Run both interaction-index diagnostics before reconstruction.')
    reconstructions = torch.stack([reconstruct_game_fn(result['stii'], n_players=4), reconstruct_game_fn(result['fsii'], n_players=4)])
    residual = result['game_values'].unsqueeze(0) - reconstructions
    kernel_mse = torch.stack([faithfulness_error_fn(result['game_values'], result['stii'], n_players=4), faithfulness_error_fn(result['game_values'], result['fsii'], n_players=4)])
    if torch.any(kernel_mse[1] > kernel_mse[0] + 1e-10):
        raise AssertionError('Faith-Shap must minimize the declared kernel objective.')
    sizes = torch.tensor([mask.bit_count() for mask in range(16)], dtype=torch.int64)
    rmse_by_size = torch.stack([residual[..., sizes == size].square().mean(dim=-1).sqrt() for size in range(5)], dim=-1)
    degree_two = GAME_NAMES.index('mixed_degree2')
    torch.testing.assert_close(reconstructions[:, degree_two], result['game_values'][degree_two].expand(2, -1), rtol=0, atol=1e-10)
    result['reconstruction'] = reconstructions
    result['residual'] = residual
    result['kernel_mse'] = kernel_mse
    result['rmse_by_size'] = rmse_by_size
    result['method_labels'] = ('Shapley-Taylor', 'Faith-Shap')
    result['stages'] = tuple(result.get('stages', ())) + ('reconstruction',)
    return result

# %%

if MAIN:
    exact_diagnostics = diagnose_reconstruction(exact_diagnostics, reconstruct_game, faithfulness_error)

if MAIN:
    allocation_figure = plot_exact_allocation_ledger(exact_diagnostics)

if MAIN:
    allocation_show_figure(allocation_figure, '01_exact_allocation_ledger')

# %%

if MAIN:
    allocation_figure = plot_reconstruction_reason(exact_diagnostics)

if MAIN:
    allocation_show_figure(allocation_figure, '02_why_faith_moves_singletons')

# %%

def masked_model_game(model: nn.Module, x: allocation_Tensor, reference: allocation_Tensor, *, coalition_batch_size: int=4096) -> allocation_Tensor:
    """Return games using one reference [D] or an equal-weight bank [B,D]."""
    with _interaction_game_batches(model, x, reference, coalition_batch_size=coalition_batch_size) as (rows, coalitions, donors, batches):
        sums = torch.zeros(rows * coalitions, dtype=torch.float64)
        for query_rows, donor_rows, present, slots in batches:
            inputs = torch.where(present, query_rows, donor_rows)
            output = _interaction_model_output(model, inputs)
            sums.index_add_(0, slots, output)
        return sums.div(donors).reshape(rows, coalitions).detach()

if MAIN:
    allocation_tests.test_masked_model_game(fn=masked_model_game)

# %%

class _AdditiveDiagnosticModel(nn.Module):

    def forward(self, inputs: allocation_Tensor) -> allocation_Tensor:
        return 1.0 + 0.5 * inputs[:, 0] + 0.25 * inputs[:, 1].square() + torch.relu(inputs[:, 2])

def _additive_formula(inputs: allocation_Tensor) -> allocation_Tensor:
    return 1.0 + 0.5 * inputs[..., 0] + 0.25 * inputs[..., 1].square() + torch.relu(inputs[..., 2])

def diagnose_model_game(masked_model_game_fn: Callable[..., allocation_Tensor]) -> dict[str, allocation_Any]:
    """Exercise one donor-averaged model game with an additive nonlinear control."""
    _callable('masked_model_game_fn', masked_model_game_fn)
    model = _AdditiveDiagnosticModel().to(dtype=torch.float64)
    query = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
    donors = torch.tensor([[0.0, 0.0, -1.0], [1.0, -2.0, 1.0]], dtype=torch.float64)
    observed = masked_model_game_fn(model, query, donors, coalition_batch_size=3)
    masks = torch.arange(8, dtype=torch.int64)
    present = masks[:, None].bitwise_and(1 << torch.arange(3, dtype=torch.int64)).ne(0)
    hybrid = torch.where(present[:, None, :], query[:, None, :], donors[None, :, :])
    expected = _additive_formula(hybrid).mean(dim=1).unsqueeze(0)
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-12)
    mobius = observed.clone()
    for player in range(3):
        selected = masks.bitwise_and(1 << player).ne(0)
        mobius[..., selected] -= mobius[..., masks[selected].bitwise_xor(1 << player)]
    pair_masks = torch.tensor([3, 5, 6], dtype=torch.int64)
    torch.testing.assert_close(mobius[..., pair_masks], torch.zeros((1, 3), dtype=torch.float64), rtol=0, atol=1e-12)
    return {'evidence_kind': 'analytic_exact', 'label': 'nonlinear additive donor-game control', 'feature_names': ('linear', 'square', 'ReLU'), 'query': query, 'donors': donors, 'game_values': observed, 'expected_game_values': expected, 'mobius': mobius, 'pair_max_abs': float(mobius[..., pair_masks].abs().max().item()), 'units': 'fixture output units'}

# %%

if MAIN:
    model_game_diagnostic = diagnose_model_game(masked_model_game)

if MAIN:
    print('max |pair Möbius term|:', model_game_diagnostic['pair_max_abs'])

# %%

def contextual_pair_synergy(game_values: allocation_Tensor, pair: tuple[int, int], *, n_players: int, context_masks: allocation_Tensor | None=None) -> tuple[allocation_Tensor, allocation_Tensor]:
    """Return sorted valid context masks and Delta_pair v(context)."""
    n_players = _interaction_n_players(n_players, minimum=2)
    _interaction_tensor('game_values', game_values, n_players)
    if type(pair) is not tuple or len(pair) != 2 or any((type(player) is not int for player in pair)):
        raise TypeError('pair must contain exactly two integer player indices.')
    first, second = pair
    if first == second:
        raise ValueError('pair must contain two different players.')
    if not 0 <= first < n_players or not 0 <= second < n_players:
        raise ValueError('pair players must lie in [0, n_players).')
    pair_mask = 1 << first | 1 << second
    if context_masks is None:
        contexts = torch.tensor([mask for mask in range(2 ** n_players) if mask & pair_mask == 0], dtype=torch.int64)
    else:
        if type(context_masks) is not allocation_Tensor:
            raise TypeError('context_masks must be a torch.Tensor.')
        if context_masks.device.type != 'cpu' or context_masks.dtype != torch.int64:
            raise TypeError('context_masks must be a CPU int64 tensor.')
        if context_masks.ndim != 1 or context_masks.numel() == 0:
            raise ValueError('context_masks must be a nonempty one-dimensional tensor.')
        if context_masks.min() < 0 or context_masks.max() >= 2 ** n_players:
            raise ValueError('Every context mask must index the complete game.')
        if context_masks.bitwise_and(pair_mask).ne(0).any():
            raise ValueError('Context masks must be disjoint from the requested pair.')
        contexts = context_masks.sort().values
    with_first = contexts.bitwise_or(1 << first)
    with_second = contexts.bitwise_or(1 << second)
    with_pair = contexts.bitwise_or(pair_mask)
    synergy = game_values[..., with_pair] - game_values[..., with_first] - game_values[..., with_second] + game_values[..., contexts]
    return (contexts, synergy)

if MAIN:
    allocation_tests.test_contextual_pair_synergy(fn=contextual_pair_synergy)

# %%

def diagnose_pair_context(contextual_pair_synergy_fn: Callable[..., tuple[allocation_Tensor, allocation_Tensor]]) -> dict[str, allocation_Any]:
    """Show where a pure triple's pair derivative switches on."""
    _callable('contextual_pair_synergy_fn', contextual_pair_synergy_fn)
    fixture = exact_game_fixture()
    game = fixture['game_values'][GAME_NAMES.index('pure_triple')]
    contexts, synergy = contextual_pair_synergy_fn(game, (0, 1), n_players=4)
    expected_contexts = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
    expected_synergy = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(contexts, expected_contexts, rtol=0, atol=0)
    torch.testing.assert_close(synergy, expected_synergy, rtol=0, atol=1e-12)
    full_context = 12
    four_masks = torch.tensor([full_context, full_context | 1, full_context | 2, full_context | 3], dtype=torch.int64)
    return {'evidence_kind': 'analytic_exact', 'label': 'pure-triple contextual pair diagnostic', 'pair': (0, 1), 'contexts': contexts, 'synergy': synergy, 'full_context_mask': full_context, 'four_masks': four_masks, 'four_values': game[four_masks], 'pair_sii': float(fixture['sii_pairs'][GAME_NAMES.index('pure_triple'), 0].item())}

# %%

if MAIN:
    pair_context_diagnostic = diagnose_pair_context(contextual_pair_synergy)

if MAIN:
    print('contexts:', pair_context_diagnostic['contexts'].tolist())

if MAIN:
    print('pair derivatives:', pair_context_diagnostic['synergy'].tolist())

# %%

if MAIN:
    allocation_methods = allocation_SimpleNamespace(mobius_transform=mobius_transform, shapley_taylor_interactions=shapley_taylor_interactions, faith_design_matrix=faith_design_matrix, faith_kernel_weights=faith_kernel_weights, faith_shap_interactions=faith_shap_interactions, faith_shap_closed_form=faith_shap_closed_form, reconstruct_game=reconstruct_game, faithfulness_error=faithfulness_error, masked_model_game=masked_model_game, contextual_pair_synergy=contextual_pair_synergy)

# %%

if MAIN:
    PLAY_N = 4

if MAIN:
    PLAY_MEMBERS = (0, 1, 2, 3)

if MAIN:
    PLAY_VALUE = 1.0

if MAIN:
    PLAY_ORDER = 2

if MAIN:
    dividend_play = exact_dividend_play(methods=allocation_methods, n_players=PLAY_N, members=PLAY_MEMBERS, value=PLAY_VALUE, order=PLAY_ORDER)

if MAIN:
    explained_play = dividend_play['explained']

if MAIN:
    retained_masks = [mask for mask in range(1, 2 ** PLAY_N) if mask.bit_count() <= PLAY_ORDER]

if MAIN:
    for label, values in (('Shapley-Taylor', explained_play['stii_coefficients']), ('Faith-Shap', explained_play['fsii_coefficients'])):
        nonzero = {f'{mask:0{PLAY_N}b}': round(float(values[mask]), 6) for mask in retained_masks if abs(float(values[mask])) > 1e-12}
        print(f'{label:15s}', nonzero)

# %%

if MAIN:
    allocation_figure = plot_dividend_play(dividend_play)

if MAIN:
    allocation_show_figure(allocation_figure, '10_dividend_play')

# %%

def reference_check(game_values: allocation_np.ndarray | torch.Tensor, student_stii: allocation_np.ndarray | torch.Tensor, student_fsii: allocation_np.ndarray | torch.Tensor, student_sii_pairs: allocation_np.ndarray | torch.Tensor, *, n_players: int, order: int, expected_game_count: int | None=None) -> dict[str, allocation_Any]:
    """Validate every supplied complete game with independent and direct package references."""
    with _preserve_execution_state():
        return _reference_check_impl(game_values, student_stii, student_fsii, student_sii_pairs, n_players=n_players, order=order, expected_game_count=expected_game_count)

def validate_exact_reference_suite(prepared: allocation_Mapping[str, allocation_Any]) -> dict[str, allocation_Any]:
    """Revalidate the retained 8 fixed, 45 random-order, and 5 boundary cases."""
    with _preserve_execution_state():
        if not isinstance(prepared, allocation_Mapping) or prepared.get('phase') != 'prepared_before_audit':
            raise ValueError('prepared must be the sealed pre-audit study payload.')
        return _validate_exact_suite(prepared)

def finalize_study_reference(prepared: allocation_Mapping[str, allocation_Any], empirical_reference: allocation_Mapping[str, allocation_Any]) -> dict[str, allocation_Any]:
    """Validate and augment an existing all-2,992 empirical result without package calls."""
    with _preserve_execution_state():
        payload_validation = validate_reference_payload(empirical_reference, expected_game_count=2992)
        exact_validation = validate_exact_reference_suite(prepared)
        result = dict(empirical_reference)
        result['counts'] = dict(empirical_reference['counts'])
        result['exact_suite_validation'] = exact_validation
        result['payload_validation'] = payload_validation
        result['counts'].update({'primary_games': 1904, 'nonprimary_clean0_games': 1088, 'models': 7, 'rows_per_model': 272, 'background_variants': 4, 'fixed_unique_exact_games': exact_validation['fixed_unique_games'], 'random_unique_exact_games': exact_validation['random_unique_games'], 'random_game_order_cases': exact_validation['random_game_order_cases'], 'random_STII_plus_FSII_case_calls': exact_validation['random_STII_plus_FSII_case_calls'], 'boundary_order_n_cases': exact_validation['boundary_order_n_cases'], 'exact_direct_shapiq_exact_computers': exact_validation['direct_shapiq_exact_computers'], 'exact_direct_shapiq_index_calls': exact_validation['direct_shapiq_index_calls'], 'all_reference_direct_shapiq_exact_computers': 2992 + exact_validation['direct_shapiq_exact_computers'], 'all_reference_direct_shapiq_index_calls': 2992 * 3 + exact_validation['direct_shapiq_index_calls'], 'all_reference_student_full_axis_coordinates': result['counts']['student_full_axis_coordinates'] + exact_validation['student_full_axis_coordinates'], 'all_reference_independent_vs_package_relevant_coordinates': result['counts']['independent_vs_package_relevant_coordinates'] + exact_validation['independent_vs_package_relevant_coordinates'], 'all_reference_raw_package_returned_coordinates': result['counts']['raw_package_returned_coordinates'] + exact_validation['raw_package_returned_coordinates']})
        if result.get('status') != 'PASS' or result['counts']['all_reference_direct_shapiq_exact_computers'] != 3050 or result['counts']['all_reference_direct_shapiq_index_calls'] != 9150:
            raise AssertionError('The complete exact plus empirical reference scope is invalid.')
        return result

# %%

if MAIN:
    reference_methods = allocation_SimpleNamespace(reference_check=reference_check)

# %%

if MAIN:
    FEATURES = 8

if MAIN:
    HIDDEN = 32

if MAIN:
    EMPIRICAL_ORDER = 2

if MAIN:
    TRAINING_STEPS = 800

if MAIN:
    LEARNING_RATE = 0.003

if MAIN:
    WEIGHT_DECAY = 0.0001

if MAIN:
    RIDGE_PENALTY = 0.01

if MAIN:
    FORWARD_BATCH = 32768

if MAIN:
    DISTANCE_BATCH = 2048

if MAIN:
    MODEL_KEYS = ('clean_0', 'clean_1', 'clean_2', 'shuffled_0', 'shuffled_1', 'shuffled_2', 'ridge')

if MAIN:
    MODEL_SEEDS = (161410, 161411, 161412)

if MAIN:
    SHUFFLE_SEEDS = (161420, 161421, 161422)

if MAIN:
    BACKGROUND_SPECS = {'primary64': (161460, 64), 'nested16': (161460, 16), 'nested128': (161460, 128), 'independent64_a': (161461, 64), 'independent64_b': (161462, 64)}

if MAIN:
    METHOD_KEYS = ('sii', 'stii', 'fsii')

if MAIN:
    METHOD_NAMES = ('mobius_transform', 'shapley_taylor_interactions', 'faith_design_matrix', 'faith_kernel_weights', 'faith_shap_interactions', 'faith_shap_closed_form', 'reconstruct_game', 'faithfulness_error', 'masked_model_game', 'contextual_pair_synergy')

if MAIN:
    FEATURE_NAMES = ('Cement', 'Slag', 'Fly ash', 'Water', 'Superplasticizer', 'Coarse aggregate', 'Fine aggregate', 'Age')

if MAIN:
    FEATURE_UNITS = ('kg/m3',) * 7 + ('days',)

if MAIN:
    TARGET_NAME = 'Compressive strength'

if MAIN:
    TARGET_UNIT = 'MPa'

if MAIN:
    RNG_NAMESPACE = 'arena16.16/shapley-interactions/v1'

if MAIN:
    RANDOM_EXACT_SEED = 161610

if MAIN:
    WITHIN_SIZE_SEED = 161611

if MAIN:
    RANDOM_PAIR_SEED = 161612

if MAIN:
    WITHIN_SIZE_REPLICATES = 128

if MAIN:
    RANDOM_PAIR_ORDERS = 256

if MAIN:
    SUPPORT_DONORS = 8

if MAIN:
    DEFAULT_PLAY = {'model_key': 'clean_0', 'source_id': 0, 'donor_source_id': 173, 'background_key': 'primary64', 'pair': (0, 1), 'context_mask': 252}

# %%

def _train_mlp(model: allocation_t.nn.Sequential, x: allocation_t.Tensor, y: allocation_t.Tensor) -> tuple[dict[str, allocation_t.Tensor], allocation_t.Tensor]:
    """Run the visible fixed 800-step full-batch Adam training loop."""
    optimizer = allocation_t.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    model.train()
    for _ in range(TRAINING_STEPS):
        optimizer.zero_grad()
        loss = (model(x).squeeze(-1) - y).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().clone())
    return (_clone_state(model.state_dict()), allocation_t.stack(losses))

def fit_known_models(train: allocation_Mapping[str, allocation_Any]) -> dict[str, allocation_Any]:
    """Reproduce all seven known predictors."""
    x, y, ids, groups = _rows(train, 'train')
    allocation_t.set_num_threads(4)
    allocation_t.use_deterministic_algorithms(True)
    normalization = _training_normalization(train)
    models: dict[str, allocation_Any] = {}
    for index, seed in enumerate(MODEL_SEEDS):
        models[f'clean_{index}'] = _fit_one_mlp(train, normalization, seed, None)
    for index, seed in enumerate(MODEL_SEEDS):
        models[f'shuffled_{index}'] = _fit_one_mlp(train, normalization, seed, SHUFFLE_SEEDS[index])
    models['ridge'] = _fit_ridge(train, normalization)
    for record in models.values():
        record['state_hashes'] = {'initial_state': tensor_mapping_sha256(record['initial_state']), 'standardized_state': tensor_mapping_sha256(record['standardized_state']), 'raw_folded_state': tensor_mapping_sha256(record['state'])}
    return {'phase': 'fitted_before_interactions', 'models': models, 'normalization': normalization, 'train_x_raw': x.clone(), 'train_y_raw': y.clone(), 'train_ids': ids.clone(), 'train_recipe_group_ids': groups.clone(), 'feature_names': FEATURE_NAMES, 'feature_units': FEATURE_UNITS, 'target_name': TARGET_NAME, 'target_unit': TARGET_UNIT}

def verify_known_reproduction(fitted: allocation_Mapping[str, allocation_Any], manifest: allocation_Mapping[str, allocation_Any]) -> dict[str, allocation_Any]:
    """Hard-fail unless all 21 known state hashes and fit identities agree."""
    _validate_fitted(fitted)
    expected = manifest.get('known_16_14_evidence', {}).get('model_state_hashes', {})
    if tuple(expected) != MODEL_KEYS:
        raise ValueError('Manifest must declare all seven known model hashes.')
    fields = {'initial_state': 'initial_state', 'standardized_state': 'standardized_state', 'raw_folded_state': 'state'}
    observed: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    for key in MODEL_KEYS:
        record = fitted['models'][key]
        observed[key] = {label: tensor_mapping_sha256(record[field]) for label, field in fields.items()}
        for label in fields:
            if observed[key][label] != expected[key][label]:
                mismatches.append(f'{key}.{label}')
        if record.get('state_hashes') != observed[key]:
            mismatches.append(f'{key}.retained_state_hashes')
    train_ids = allocation_t.tensor(manifest.get('splits', {}).get('train', []), dtype=allocation_t.int64)
    if not allocation_t.equal(train_ids, fitted['train_ids']):
        mismatches.append('train_ids')
    recipe = manifest.get('model_recipe', {})
    recipe_matches = recipe.get('model_keys') == list(MODEL_KEYS) and recipe.get('model_seeds') == list(MODEL_SEEDS) and (recipe.get('shuffle_seeds') == list(SHUFFLE_SEEDS)) and (recipe.get('training_steps') == TRAINING_STEPS) and (recipe.get('learning_rate') == LEARNING_RATE) and (recipe.get('weight_decay') == WEIGHT_DECAY) and (recipe.get('ridge_penalty') == RIDGE_PENALTY)
    if not recipe_matches:
        mismatches.append('training_recipe')
    if mismatches:
        raise AssertionError('Known-model reproduction mismatch: ' + ', '.join(mismatches))
    return {'status': 'PASS', 'known_section': '16.14/16.15', 'model_state_hashes': observed, 'expected_model_state_hashes': expected, 'mismatches': (), 'state_comparisons': 21, 'training_recipe_match': True, 'train_ids_match': True, 'non_independent_replication': True}

# %%

if MAIN:
    manifest = load_manifest()

if MAIN:
    train = load_split('train')

if MAIN:
    explain = load_split('explain')

# %%

if MAIN:
    query_row = explain['x'][allocation_np.flatnonzero(explain['ids'] == 0).item()]

if MAIN:
    donor_row = train['x'][allocation_np.flatnonzero(train['ids'] == 173).item()]

if MAIN:
    preview_present = 252 & 1 << allocation_np.arange(8) != 0

if MAIN:
    allocation_display(allocation_pd.DataFrame({'feature': FEATURE_NAMES, 'unit': FEATURE_UNITS, 'query (source 0)': query_row, 'donor (source 173)': donor_row, 'hybrid 11111100': allocation_np.where(preview_present, query_row, donor_row), 'taken from': allocation_np.where(preview_present, 'query', 'donor')}).set_index('feature'))

# %%

if MAIN:
    fitted = fit_known_models(train)

if MAIN:
    reproduction = verify_known_reproduction(fitted, manifest)

if MAIN:
    fitted = {**fitted, 'reproduction': reproduction}

if MAIN:
    print(f"All {reproduction['state_comparisons']} known model-state comparisons passed.")

# %%

def donor_replacement_games(model: allocation_t.nn.Module, queries: allocation_t.Tensor, donors: allocation_t.Tensor, *, methods: allocation_Any, donor_source_ids: allocation_t.Tensor | None=None, coalition_batch_size: int=4096) -> dict[str, allocation_t.Tensor]:
    """Retain every single-donor game, then average the declared donor bank."""
    _require_methods(methods)
    x = _float_tensor(queries, 'queries', 2)
    background = _float_tensor(donors, 'donors', 2)
    if x.shape[1] != FEATURES or background.shape[1] != FEATURES or len(background) == 0:
        raise ValueError('queries and donors must contain eight-feature rows.')
    if not isinstance(coalition_batch_size, int) or coalition_batch_size <= 0:
        raise ValueError('coalition_batch_size must be positive.')
    if donor_source_ids is None:
        donor_source_ids = allocation_t.arange(len(background), dtype=allocation_t.int64)
    donor_source_ids = _int_tensor(donor_source_ids, 'donor_source_ids', 1)
    if donor_source_ids.shape != (len(background),) or len(allocation_t.unique(donor_source_ids)) != len(background):
        raise ValueError('donor_source_ids must uniquely identify every donor.')
    hybrid = allocation_t.empty((len(x), 2 ** FEATURES, len(background)), dtype=allocation_t.float64)
    for donor_index, donor in enumerate(background):
        game = methods.masked_model_game(model, x, donor, coalition_batch_size=coalition_batch_size)
        game = _float_tensor(game, 'single_donor_game', 2)
        if game.shape != (len(x), 2 ** FEATURES):
            raise ValueError('masked_model_game returned an unexpected shape.')
        hybrid[..., donor_index] = game
    return {'hybrid_prediction': hybrid, 'coalition_values': hybrid.mean(-1), 'donor_source_ids': donor_source_ids.clone()}

def evaluate_model_games(model: allocation_t.nn.Module, queries: allocation_t.Tensor, donors: allocation_t.Tensor, *, methods: allocation_Any, donor_source_ids: allocation_t.Tensor, coalition_batch_size: int=4096) -> dict[str, allocation_t.Tensor]:
    """Construct one retained donor game and compute every exact order-two output."""
    donor = donor_replacement_games(model, queries, donors, methods=methods, donor_source_ids=donor_source_ids, coalition_batch_size=coalition_batch_size)
    explained = explain_complete_games(donor['coalition_values'], methods=methods)
    return {**donor, **explained}

# %%

def pairwise_sii(game_values: allocation_t.Tensor, *, n_players: int) -> allocation_t.Tensor:
    """Compute the exact pair SII comparator in lexicographic pair order."""
    values = _float_tensor(game_values, 'game_values')
    if values.ndim < 1 or values.shape[-1] != 2 ** n_players or n_players < 2:
        raise ValueError('game_values must end in one complete game axis.')
    output = []
    for left, right in pair_indices(n_players).tolist():
        pair_mask = 1 << left | 1 << right
        contexts = [mask for mask in range(2 ** n_players) if mask & pair_mask == 0]
        score = allocation_t.zeros(values.shape[:-1], dtype=allocation_t.float64)
        denominator = allocation_math.factorial(n_players - 1)
        for context in contexts:
            size = int(context).bit_count()
            weight = allocation_math.factorial(size) * allocation_math.factorial(n_players - size - 2) / denominator
            delta = values[..., context | pair_mask] - values[..., context | 1 << left] - values[..., context | 1 << right] + values[..., context]
            score = score + weight * delta
        output.append(score)
    return allocation_t.stack(output, dim=-1)

def all_pair_context_synergies(game_values: allocation_t.Tensor, *, n_players: int, methods: allocation_Any) -> dict[str, allocation_t.Tensor]:
    """Apply the learner's local derivative to all pairs and valid contexts."""
    _require_methods(methods)
    values = _float_tensor(game_values, 'game_values')
    pairs = pair_indices(n_players)
    context_rows = []
    synergy_rows = []
    full_rows = []
    for left, right in pairs.tolist():
        contexts, synergy = methods.contextual_pair_synergy(values, (left, right), n_players=n_players)
        contexts = _int_tensor(contexts, 'context_masks', 1)
        synergy = _float_tensor(synergy, 'context_synergy')
        if synergy.shape != values.shape[:-1] + (len(contexts),):
            raise ValueError('contextual_pair_synergy returned an unexpected shape.')
        expected_full = 2 ** n_players - 1 ^ 1 << left ^ 1 << right
        positions = allocation_t.nonzero(contexts == expected_full).flatten()
        if len(positions) != 1:
            raise ValueError('The full pair context is absent or repeated.')
        context_rows.append(contexts)
        synergy_rows.append(synergy)
        full_rows.append(synergy[..., int(positions[0])])
    context_masks = allocation_t.stack(context_rows)
    context_synergy = allocation_t.stack(synergy_rows, dim=-2)
    return {'pair_indices': pairs, 'context_masks': context_masks, 'context_synergy': context_synergy, 'full_context_synergy': allocation_t.stack(full_rows, dim=-1), 'mean_abs_context_synergy': context_synergy.abs().mean(-1)}

def explain_complete_games(game_values: allocation_t.Tensor, *, methods: allocation_Any, n_players: int=FEATURES, order: int=EMPIRICAL_ORDER, include_contexts: bool=True) -> dict[str, allocation_t.Tensor]:
    """Apply every visible interaction method to a batch of complete games."""
    _require_methods(methods)
    values = _float_tensor(game_values, 'game_values')
    if values.ndim < 1 or values.shape[-1] != 2 ** n_players:
        raise ValueError('game_values must end in one complete bitmask axis.')
    mobius = methods.mobius_transform(values, n_players=n_players)
    mobius = _float_tensor(mobius, 'mobius')
    stii = methods.shapley_taylor_interactions(mobius, n_players=n_players, order=order)
    fsii = methods.faith_shap_interactions(values, n_players=n_players, max_order=order)
    closed = methods.faith_shap_closed_form(mobius, n_players=n_players, max_order=order)
    stii = _float_tensor(stii, 'stii_coefficients')
    fsii = _float_tensor(fsii, 'fsii_coefficients')
    closed = _float_tensor(closed, 'fsii_closed_form')
    expected_shape = values.shape
    if stii.shape != expected_shape or fsii.shape != expected_shape or closed.shape != expected_shape:
        raise ValueError('Interaction coefficient tensors must preserve the game shape.')
    ordinary_full = methods.shapley_taylor_interactions(mobius, n_players=n_players, order=1)
    singleton_masks = 1 << allocation_t.arange(n_players, dtype=allocation_t.int64)
    pairs = pair_indices(n_players)
    pair_masks = 1 << pairs[:, 0] | 1 << pairs[:, 1]
    sii = pairwise_sii(values, n_players=n_players)
    recon = allocation_t.stack([methods.reconstruct_game(stii, n_players=n_players), methods.reconstruct_game(fsii, n_players=n_players)])
    recon = _float_tensor(recon, 'reconstruction')
    residual = values.unsqueeze(0) - recon
    sizes = coalition_masks(n_players).sum(-1)
    rmse = allocation_t.stack([allocation_t.sqrt(residual[..., sizes == size].square().mean(-1)) for size in range(n_players + 1)], dim=-1)
    kernel_mse = allocation_t.stack([methods.faithfulness_error(values, stii, n_players=n_players), methods.faithfulness_error(values, fsii, n_players=n_players)])
    result = {'coalition_values': values, 'mobius': mobius, 'ordinary_shapley': ordinary_full[..., singleton_masks], 'sii_pairs': sii, 'stii_coefficients': stii, 'fsii_coefficients': fsii, 'fsii_closed_form': closed, 'fsii_closed_form_error': (fsii - closed).abs().amax(-1), 'pair_values': allocation_t.stack([sii, stii[..., pair_masks], fsii[..., pair_masks]], dim=-2), 'reconstruction': recon, 'residual': residual, 'kernel_mse': kernel_mse, 'rmse_by_size': rmse, 'efficiency_residual': allocation_t.stack([stii.sum(-1) - values[..., -1], fsii.sum(-1) - values[..., -1]]), 'pair_indices': pairs, 'pair_masks': pair_masks}
    if include_contexts:
        result.update(all_pair_context_synergies(values, n_players=n_players, methods=methods))
    return result

# %%

def explanation_pair_rankings(pair_values: allocation_t.Tensor, recipe_group_ids: allocation_t.Tensor) -> dict[str, allocation_t.Tensor]:
    """Rank all 28 pairs from explanation groups without dropping any pair."""
    values = _float_tensor(pair_values, 'pair_values', 4)
    if values.shape[2:] != (len(METHOD_KEYS), allocation_math.comb(FEATURES, 2)):
        raise ValueError('pair_values must have shape [models, rows, 3, 28].')
    strengths = allocation_t.stack([_group_weighted_mean_abs(model_values, recipe_group_ids) for model_values in values])
    orders = allocation_t.argsort(strengths, dim=-1, descending=True, stable=True)
    return {'method_keys': METHOD_KEYS, 'pair_strength': strengths, 'pair_order': orders}

def fixed_random_pair_orders() -> dict[str, allocation_Any]:
    """Return all 256 content-independent random pair permutations."""
    orders = []
    seeds = []
    for replicate in range(RANDOM_PAIR_ORDERS):
        derived = _derived_seed(RANDOM_PAIR_SEED, replicate)
        seeds.append(derived)
        orders.append(allocation_np.random.Generator(allocation_np.random.PCG64(derived)).permutation(28))
    result = {'seed': RANDOM_PAIR_SEED, 'derived_seeds_hex': tuple((f'{value:032x}' for value in seeds)), 'orders': allocation_t.tensor(allocation_np.stack(orders), dtype=allocation_t.int64)}
    declared = load_manifest().get('rng_contract', {}).get('random_pair_order', {})
    if declared.get('derived_seeds_hex') != list(result['derived_seeds_hex']):
        raise ValueError('Random pair-order seeds differ from the manifest.')
    if declared.get('orders_value_sha256') != tensor_mapping_sha256({'orders': result['orders']}):
        raise ValueError('Random pair orders differ from the manifest.')
    return result

def prepare_study(fitted: allocation_Mapping[str, allocation_Any], train: allocation_Mapping[str, allocation_Any], explain: allocation_Mapping[str, allocation_Any], *, methods: allocation_Any, reference: allocation_Any | None=None, manifest: allocation_Mapping[str, allocation_Any] | None=None) -> dict[str, allocation_Any]:
    """Compute and seal every explanation-side decision before audit access."""
    _require_methods(methods)
    manifest = load_manifest() if manifest is None else manifest
    reproduction = verify_known_reproduction(fitted, manifest)
    train_x, train_y, train_ids, train_groups = _rows(train, 'train')
    explain_x, explain_y, explain_ids, explain_groups = _rows(explain, 'explain')
    _require_disjoint_ids(train=train_ids, explain=explain_ids)
    if not allocation_t.equal(train_x, fitted['train_x_raw']) or not allocation_t.equal(train_y, fitted['train_y_raw']):
        raise ValueError('Training values changed after fitting.')
    if not allocation_t.equal(train_ids, fitted['train_ids']) or not allocation_t.equal(train_groups, fitted['train_recipe_group_ids']):
        raise ValueError('Training identities changed after fitting.')
    expected_explain = allocation_t.tensor(manifest.get('splits', {}).get('explain', []), dtype=allocation_t.int64)
    if not allocation_t.equal(explain_ids, expected_explain):
        raise ValueError('Explanation source IDs differ from the pinned split.')
    backgrounds = _background_records(train_x, train_ids, manifest)
    primary = _evaluate_primary(fitted, explain_x, backgrounds['primary64'], methods=methods)
    variants = _evaluate_backgrounds(fitted, explain_x, backgrounds, primary, methods=methods)
    rankings = explanation_pair_rankings(primary['pair_values'], explain_groups)
    direct = _direct_full_context_four(fitted, explain_x, backgrounds['primary64']['x_raw'])
    direct['table_synergy'] = primary['full_context_synergy']
    direct['lookup_difference'] = direct['synergy'] - direct['table_synergy']
    design_contract = {}
    for order in (1, 2, 3):
        design, term_masks = methods.faith_design_matrix(n_players=FEATURES, max_order=order)
        design_contract[order] = {'design': _float_tensor(design, f'faith_design_{order}', 2), 'term_masks': _int_tensor(term_masks, f'term_masks_{order}', 1)}
    design_contract['kernel_weights'] = _float_tensor(methods.faith_kernel_weights(n_players=FEATURES), 'kernel_weights', 1)
    return {'phase': 'prepared_before_audit', 'audit_used_for_selection': False, 'reproduction': reproduction, 'feature_names': FEATURE_NAMES, 'feature_units': FEATURE_UNITS, 'target_name': TARGET_NAME, 'target_unit': TARGET_UNIT, 'model_keys': MODEL_KEYS, 'method_keys': METHOD_KEYS, 'mask_ids': allocation_t.arange(2 ** FEATURES, dtype=allocation_t.int64), 'coalition_masks': coalition_masks(FEATURES), 'coalition_sizes': coalition_masks(FEATURES).sum(-1).to(allocation_t.int64), 'pair_indices': pair_indices(FEATURES), 'context_masks': primary['context_masks'], 'design_contract': design_contract, 'train_ids': train_ids, 'train_recipe_group_ids': train_groups, 'explain_source_ids': explain_ids, 'explain_recipe_group_ids': explain_groups, 'explain_x_raw': explain_x, 'explain_targets': explain_y, 'backgrounds': backgrounds, 'exact_games': _exact_suite(methods, reference), 'primary64': primary, 'background_variants': variants, 'direct_full_context': direct, 'references': {'exact_suite_checked_in_preparation': reference is not None, 'empirical': _empirical_reference_plan('prepared_before_audit')}, 'rankings': rankings, 'random_pair_orders': fixed_random_pair_orders(), 'fixed_play_defaults': dict(DEFAULT_PLAY), 'source_provenance': dict(manifest.get('source_provenance', {})), 'storage_plan': _storage_plan(manifest, 'prepared_before_audit')}

# %%

if MAIN:
    prepared = prepare_study(fitted, train, explain, methods=allocation_methods, reference=reference_methods, manifest=manifest)

if MAIN:
    persist_prepared(native_directory, fitted, prepared)

if MAIN:
    assert prepared['phase'] == 'prepared_before_audit'

if MAIN:
    assert prepared['audit_used_for_selection'] is False

# %%

if MAIN:
    audit = load_split('audit')

# %%

def pair_capture_curves(pair_orders: allocation_t.Tensor, audit_full_context_synergy: allocation_t.Tensor, audit_recipe_group_ids: allocation_t.Tensor, random_orders: allocation_t.Tensor) -> dict[str, allocation_t.Tensor]:
    """Evaluate explanation-derived and random pair orders on every audit group."""
    orders = _int_tensor(pair_orders, 'pair_orders', 3)
    synergy = _float_tensor(audit_full_context_synergy, 'audit_full_context_synergy', 3)
    random = _int_tensor(random_orders, 'random_orders', 2)
    if orders.shape != (len(MODEL_KEYS), len(METHOD_KEYS), 28):
        raise ValueError('pair_orders must have shape [7,3,28].')
    if synergy.shape[0] != len(MODEL_KEYS) or synergy.shape[2] != 28:
        raise ValueError('audit_full_context_synergy must have shape [7,rows,28].')
    if random.shape != (RANDOM_PAIR_ORDERS, 28):
        raise ValueError('random_orders must have shape [256,28].')
    strength = allocation_t.stack([_group_weighted_mean_abs(model_values, audit_recipe_group_ids) for model_values in synergy])
    denominator = strength.sum(-1)
    zero = denominator == 0
    safe = allocation_t.where(zero, allocation_t.ones_like(denominator), denominator)
    observed = allocation_t.zeros((len(MODEL_KEYS), len(METHOD_KEYS), 29), dtype=allocation_t.float64)
    random_curves = allocation_t.zeros((len(MODEL_KEYS), RANDOM_PAIR_ORDERS, 29), dtype=allocation_t.float64)
    for model_index in range(len(MODEL_KEYS)):
        for method_index in range(len(METHOD_KEYS)):
            ordered = strength[model_index, orders[model_index, method_index]]
            observed[model_index, method_index, 1:] = ordered.cumsum(0) / safe[model_index]
        random_strength = strength[model_index][random]
        random_curves[model_index, :, 1:] = random_strength.cumsum(-1) / safe[model_index]
    observed[zero, :] = 0.0
    random_curves[zero, :] = 0.0
    return {'audit_full_context_strength': strength, 'capture_curves': observed, 'random_capture_curves': random_curves, 'zero_denominator': zero}

def within_size_shuffle_control(game_values: allocation_t.Tensor, observed_pair_values: allocation_t.Tensor, source_ids: allocation_t.Tensor, recipe_group_ids: allocation_t.Tensor, *, methods: allocation_Any) -> dict[str, allocation_Any]:
    """Build the fixed structural null while preserving endpoints and size marginals."""
    _require_methods(methods)
    values = _float_tensor(game_values, 'game_values', 2)
    if values.shape != (len(source_ids), 256):
        raise ValueError('The within-size control requires all audit D=8 games.')
    schedule = _within_size_permutations(source_ids)
    index = schedule['permutation'].to(allocation_t.int64)
    shuffled = allocation_t.gather(values[None, :, :].expand(WITHIN_SIZE_REPLICATES, -1, -1), 2, index)
    flat = shuffled.reshape(-1, 256)
    mobius = methods.mobius_transform(flat, n_players=FEATURES)
    stii = methods.shapley_taylor_interactions(mobius, n_players=FEATURES, order=2)
    fsii = methods.faith_shap_interactions(flat, n_players=FEATURES, max_order=2)
    pairs = pair_indices(FEATURES)
    pair_masks = 1 << pairs[:, 0] | 1 << pairs[:, 1]
    pair_values = allocation_t.stack([pairwise_sii(flat, n_players=FEATURES), stii[..., pair_masks], fsii[..., pair_masks]], dim=-2).reshape(WITHIN_SIZE_REPLICATES, len(values), len(METHOD_KEYS), 28)
    return {**schedule, 'method_keys': METHOD_KEYS, 'pair_values': pair_values, 'observed': _localization_statistics(observed_pair_values, recipe_group_ids), 'null': _localization_statistics(pair_values, recipe_group_ids), 'interpretation': 'structural null; not expected to have zero pair interaction'}

def background_sensitivity(variants: allocation_Mapping[str, allocation_Mapping[str, allocation_Any]]) -> dict[str, allocation_Any]:
    """Compare every clean_0 finite-background game with primary64."""
    if tuple(variants) != tuple(BACKGROUND_SPECS):
        raise ValueError('Background variants changed order or membership.')
    primary = variants['primary64']
    result = {}
    for name, item in variants.items():
        pair_movement = item['pair_values'] - primary['pair_values']
        result[name] = {'base_movement': item['coalition_values'][..., 0] - primary['coalition_values'][..., 0], 'prediction_movement': item['coalition_values'][..., -1] - primary['coalition_values'][..., -1], 'ordinary_shapley_movement': item['ordinary_shapley'] - primary['ordinary_shapley'], 'pair_movement': pair_movement, 'absolute_pair_movement': pair_movement.abs(), 'pair_sign_change': allocation_t.signbit(item['pair_values']) != allocation_t.signbit(primary['pair_values']), 'pair_rank_change': _pair_ranks(item['pair_values']) - _pair_ranks(primary['pair_values']), 'full_context_movement': item['full_context_synergy'] - primary['full_context_synergy']}
    return result

def evaluate_study(fitted: allocation_Mapping[str, allocation_Any], prepared: allocation_Mapping[str, allocation_Any], audit: allocation_Mapping[str, allocation_Any], *, methods: allocation_Any, reference: allocation_Any | None=None, manifest: allocation_Mapping[str, allocation_Any] | None=None) -> dict[str, allocation_Any]:
    """Evaluate every prespecified audit outcome without changing preparation."""
    _require_methods(methods)
    if prepared.get('phase') != 'prepared_before_audit' or prepared.get('audit_used_for_selection') is not False:
        raise ValueError('A sealed pre-audit preparation payload is required.')
    manifest = load_manifest() if manifest is None else manifest
    reproduction = verify_known_reproduction(fitted, manifest)
    audit_x, audit_y, audit_ids, audit_groups = _rows(audit, 'audit')
    expected_audit = allocation_t.tensor(manifest.get('splits', {}).get('audit', []), dtype=allocation_t.int64)
    if not allocation_t.equal(audit_ids, expected_audit):
        raise ValueError('Audit source IDs differ from the pinned split.')
    _require_disjoint_ids(train=fitted['train_ids'], explain=prepared['explain_source_ids'], audit=audit_ids)
    backgrounds = prepared['backgrounds']
    primary_audit = _evaluate_primary(fitted, audit_x, backgrounds['primary64'], methods=methods)
    variants_audit = _evaluate_backgrounds(fitted, audit_x, backgrounds, primary_audit, methods=methods)
    primary_all = _combine_game_blocks(prepared['primary64'], primary_audit, model_axis=True)
    variants_all = {name: _combine_game_blocks(prepared['background_variants'][name], variants_audit[name], model_axis=False) for name in BACKGROUND_SPECS}
    direct_audit = _direct_full_context_four(fitted, audit_x, backgrounds['primary64']['x_raw'])
    direct_audit['table_synergy'] = primary_audit['full_context_synergy']
    direct_audit['lookup_difference'] = direct_audit['synergy'] - direct_audit['table_synergy']
    direct_all = {leaf: allocation_t.cat([prepared['direct_full_context'][leaf], direct_audit[leaf]], dim=1) for leaf in ('four_values', 'synergy', 'table_synergy', 'lookup_difference')}
    random_orders = prepared['random_pair_orders']['orders']
    curves = pair_capture_curves(prepared['rankings']['pair_order'], primary_audit['full_context_synergy'], audit_groups, random_orders)
    null = within_size_shuffle_control(primary_audit['coalition_values'][0], primary_audit['pair_values'][0], audit_ids, audit_groups, methods=methods)
    all_x = allocation_t.cat([prepared['explain_x_raw'], audit_x])
    all_ids = allocation_t.cat([prepared['explain_source_ids'], audit_ids])
    all_groups = allocation_t.cat([prepared['explain_recipe_group_ids'], audit_groups])
    support = nearest_training_diagnostics(fitted['train_x_raw'], fitted['train_ids'], all_x, backgrounds['primary64']['x_raw'][:SUPPORT_DONORS], fitted['normalization'])
    row_split = allocation_t.cat([allocation_t.zeros(len(prepared['explain_source_ids']), dtype=allocation_t.int64), allocation_t.ones(len(audit_ids), dtype=allocation_t.int64)])
    return {'phase': 'evaluated_after_audit', 'audit_used_for_selection': False, 'reproduction': reproduction, 'feature_names': FEATURE_NAMES, 'feature_units': FEATURE_UNITS, 'target_name': TARGET_NAME, 'target_unit': TARGET_UNIT, 'model_keys': MODEL_KEYS, 'method_keys': METHOD_KEYS, 'source_ids': all_ids, 'row_split': row_split, 'split_labels': ('explanation', 'audit'), 'recipe_group_ids': all_groups, 'audit_source_ids': audit_ids, 'audit_recipe_group_ids': audit_groups, 'audit_x_raw': audit_x, 'audit_targets': audit_y, 'primary64_audit': primary_audit, 'background_variants_audit': variants_audit, 'all_rows': {'primary64': primary_all, 'background_variants': variants_all}, 'direct_full_context': direct_all, 'performance': {'explanation': _performance(fitted, prepared['explain_x_raw'], prepared['explain_targets']), 'audit': _performance(fitted, audit_x, audit_y)}, 'rankings': prepared['rankings'], 'pair_capture': {**curves, 'random_pair_orders': random_orders, 'method_keys': METHOD_KEYS}, 'random_controls': {'within_size': null}, 'background_sensitivity': background_sensitivity(variants_all), 'support': support, 'references': {'prepared': prepared['references'], 'empirical': _empirical_reference_plan('evaluated_after_audit'), 'adapter_supplied': reference is not None}, 'fixed_play_defaults': dict(DEFAULT_PLAY), 'source_provenance': dict(prepared.get('source_provenance', {})), 'storage_plan': _storage_plan(manifest, 'evaluated_after_audit')}

# %%

if MAIN:
    evaluated = evaluate_study(fitted, prepared, audit, methods=allocation_methods, manifest=manifest)

if MAIN:
    assert evaluated['phase'] == 'evaluated_after_audit'

if MAIN:
    assert evaluated['audit_used_for_selection'] is False

# %%

if MAIN:
    primary_reference = evaluated['all_rows']['primary64']

if MAIN:
    variant_references = evaluated['all_rows']['background_variants']

if MAIN:
    reference_blocks = [primary_reference] + [variant_references[name] for name in ('nested16', 'nested128', 'independent64_a', 'independent64_b')]

def flatten_reference_leaf(leaf: str) -> allocation_Tensor:
    return torch.cat([block[leaf].reshape(-1, block[leaf].shape[-1]) for block in reference_blocks])

if MAIN:
    empirical_reference = reference_check(flatten_reference_leaf('coalition_values'), flatten_reference_leaf('stii_coefficients'), flatten_reference_leaf('fsii_coefficients'), flatten_reference_leaf('sii_pairs'), n_players=8, order=2, expected_game_count=2992)

if MAIN:
    reference_branches = []

if MAIN:
    branch_start = 0

if MAIN:
    for name, block in zip(('primary64', 'nested16', 'nested128', 'independent64_a', 'independent64_b'), reference_blocks, strict=True):
        allocation_games = block['coalition_values'].numel() // 256
        reference_branches.append({'name': name, 'model_keys': MODEL_KEYS if name == 'primary64' else ('clean_0',), 'start': branch_start, 'stop': branch_start + allocation_games, 'games': allocation_games})
        branch_start += allocation_games

if MAIN:
    reference_result = finalize_study_reference(prepared, empirical_reference)

if MAIN:
    reference_result['branches'] = tuple(reference_branches)

if MAIN:
    print(reference_result['status'])

# %%

def empirical_context_play(evaluated: allocation_Mapping[str, allocation_Any], methods: allocation_Any, *, model_key: str='clean_0', source_id: int=0, background_key: str='primary64', pair: tuple[int, int]=(0, 1), context_mask: int=252, donor_source_id: int | None=None) -> dict[str, allocation_Any]:
    """Inspect a real retained row, donor bank, pair, and coalition context."""
    _require_methods(methods)
    if evaluated.get('phase') != 'evaluated_after_audit':
        raise ValueError('Use the complete evaluated payload.')
    if model_key not in MODEL_KEYS or background_key not in BACKGROUND_SPECS:
        raise ValueError('Unknown model or background key.')
    if background_key != 'primary64' and model_key != 'clean_0':
        raise ValueError('Alternate backgrounds are retained only for clean_0.')
    ids = _int_tensor(evaluated['source_ids'], 'source_ids', 1)
    positions = allocation_t.nonzero(ids == int(source_id)).flatten()
    if len(positions) != 1:
        raise ValueError('source_id is absent or repeated.')
    row_index = int(positions[0])
    left, right = pair
    if left == right or min(pair) < 0 or max(pair) >= FEATURES:
        raise ValueError('pair must contain two distinct valid players.')
    pair_mask = 1 << left | 1 << right
    if context_mask < 0 or context_mask >= 256 or context_mask & pair_mask:
        raise ValueError('context_mask must be valid and disjoint from the pair.')
    if background_key == 'primary64':
        primary = evaluated['all_rows']['primary64']
        model_index = MODEL_KEYS.index(model_key)
        block = {key: (value[:, model_index] if key in _METHOD_FIRST_LEAVES else value[model_index]) if isinstance(value, allocation_t.Tensor) and key not in _SHARED_GAME_LEAVES else value for key, value in primary.items() if key != 'model_keys'}
        hybrid_meta = primary['hybrid_prediction']
        block['hybrid_prediction'] = {**hybrid_meta, 'explain': hybrid_meta['explain'][model_index], 'audit': hybrid_meta['audit'][model_index]}
    else:
        block = evaluated['all_rows']['background_variants'][background_key]
    hybrid_row = _logical_hybrid_row(block, row_index, evaluated['row_split'])
    donor_ids = block['donor_source_ids']
    if donor_source_id is None:
        game = block['coalition_values'][row_index]
        donor_values = hybrid_row
        selected_donor_ids = donor_ids
        game_label = 'equal_weight_donor_bank'
    else:
        donor_positions = allocation_t.nonzero(donor_ids == int(donor_source_id)).flatten()
        if len(donor_positions) != 1:
            raise ValueError('donor_source_id is not in the selected background.')
        donor_index = int(donor_positions[0])
        game = hybrid_row[:, donor_index]
        donor_values = hybrid_row[:, donor_index:donor_index + 1]
        selected_donor_ids = donor_ids[donor_index:donor_index + 1]
        game_label = 'single_donor_game'
    masks = allocation_t.tensor([context_mask, context_mask | 1 << left, context_mask | 1 << right, context_mask | pair_mask], dtype=allocation_t.int64)
    four_by_donor = donor_values[masks]
    explained = explain_complete_games(game, methods=methods, n_players=FEATURES, order=2)
    pair_row = allocation_t.nonzero((pair_indices(FEATURES) == allocation_t.tensor(pair)).all(-1)).flatten()
    if len(pair_row) != 1:
        reverse = allocation_t.tensor((right, left), dtype=allocation_t.int64)
        pair_row = allocation_t.nonzero((pair_indices(FEATURES) == reverse).all(-1)).flatten()
    pair_index = int(pair_row[0])
    return {'kind': 'empirical_context_play', 'model_key': model_key, 'source_id': int(source_id), 'split': 'audit_reuse' if int(evaluated['row_split'][row_index]) else 'explanation', 'background_key': background_key, 'game_label': game_label, 'target_unit': TARGET_UNIT, 'donor_source_ids': selected_donor_ids, 'pair': tuple(pair), 'context_mask': int(context_mask), 'four_masks': masks, 'four_values_by_donor': four_by_donor, 'four_values': four_by_donor.mean(-1), 'local_synergy_by_donor': four_by_donor[3] - four_by_donor[1] - four_by_donor[2] + four_by_donor[0], 'local_synergy': game[masks[3]] - game[masks[1]] - game[masks[2]] + game[masks[0]], 'pair_values': explained['pair_values'][..., pair_index], 'method_keys': METHOD_KEYS, 'game_values': game.clone(), 'interprets_model_input_intervention_only': True}

# %%

def native_play(evaluated: allocation_Mapping[str, allocation_Any], *, methods: allocation_Any) -> dict[str, allocation_Any]:
    """Assemble the fixed content-blind native result without live model objects."""
    if evaluated.get('phase') != 'evaluated_after_audit':
        raise ValueError('Run the complete frozen evaluation before native_play.')
    empirical = empirical_context_play(evaluated, methods, model_key=DEFAULT_PLAY['model_key'], source_id=DEFAULT_PLAY['source_id'], background_key=DEFAULT_PLAY['background_key'], pair=DEFAULT_PLAY['pair'], context_mask=DEFAULT_PLAY['context_mask'], donor_source_id=None)
    donor = empirical_context_play(evaluated, methods, model_key=DEFAULT_PLAY['model_key'], source_id=DEFAULT_PLAY['source_id'], background_key=DEFAULT_PLAY['background_key'], pair=DEFAULT_PLAY['pair'], context_mask=DEFAULT_PLAY['context_mask'], donor_source_id=DEFAULT_PLAY['donor_source_id'])
    ids = _int_tensor(evaluated['source_ids'], 'source_ids', 1)
    row_positions = allocation_t.nonzero(ids == DEFAULT_PLAY['source_id']).flatten()
    if len(row_positions) != 1:
        raise ValueError('The fixed source row is absent or repeated.')
    row_index = int(row_positions[0])
    primary = evaluated['all_rows']['primary64']
    clean_index = MODEL_KEYS.index('clean_0')
    ridge_index = MODEL_KEYS.index('ridge')
    fixed_row = {'model_key': 'clean_0', 'source_id': DEFAULT_PLAY['source_id'], 'background_key': 'primary64', 'order': EMPIRICAL_ORDER, 'donor_count': int(primary['donor_source_ids'].numel()), 'target_unit': TARGET_UNIT, 'pair_indices': primary['pair_indices'], 'stii_coefficients': primary['stii_coefficients'][clean_index, row_index], 'fsii_coefficients': primary['fsii_coefficients'][clean_index, row_index], 'sii_pairs': primary['sii_pairs'][clean_index, row_index], 'ridge_pair_max_abs': primary['pair_values'][ridge_index, row_index].abs().amax()}
    return {'phase': 'native_play_from_frozen_evaluation', 'fixed_defaults': dict(DEFAULT_PLAY), 'exact_dividend': exact_dividend_play(methods=methods), 'empirical_background_average': empirical, 'empirical_fixed_donor': donor, 'fixed_row': fixed_row, 'overview': {'feature_names': evaluated['feature_names'], 'feature_units': evaluated['feature_units'], 'target_name': evaluated['target_name'], 'target_unit': evaluated['target_unit'], 'model_keys': evaluated['model_keys'], 'method_keys': evaluated['method_keys'], 'source_ids': evaluated['source_ids'], 'row_split': evaluated['row_split'], 'performance': evaluated['performance'], 'rankings': evaluated['rankings'], 'pair_capture': evaluated['pair_capture'], 'background_sensitivity': evaluated['background_sensitivity'], 'direct_full_context': evaluated['direct_full_context']}}

# %%

if MAIN:
    diagnostics = interaction_diagnostics(prepared, evaluated, reference_result)

if MAIN:
    allocation_play = native_play(evaluated, methods=allocation_methods)

if MAIN:
    publication_summary = persist_completed(native_directory, fitted, prepared, evaluated, reference_result, diagnostics, allocation_play, timings={})

if MAIN:
    write_notebook_summary(FIGURE_DIR, native_directory, publication_summary)

if MAIN:
    print('Complete games, controls and reference checks retained. Read the figures below.')

# %%

if MAIN:
    allocation_figure = plot_native_game_and_controls(evaluated)

if MAIN:
    allocation_show_figure(allocation_figure, '03_native_game_and_controls')

# %%

if MAIN:
    allocation_figure = plot_fixed_row_interaction_graph(allocation_play, prepared['feature_names'])

if MAIN:
    allocation_show_figure(allocation_figure, '04_fixed_row_interaction_graph')

# %%

if MAIN:
    allocation_figure = plot_allrow_pair_heatmaps(evaluated, prepared['feature_names'])

if MAIN:
    allocation_show_figure(allocation_figure, '05_allrow_pair_heatmaps')

# %%

if MAIN:
    allocation_figure = plot_heldout_pair_order_capture(evaluated, prepared['feature_names'])

if MAIN:
    allocation_show_figure(allocation_figure, '06_heldout_pair_order_capture')

# %%

if MAIN:
    allocation_figure = plot_order_two_reconstruction(evaluated)

if MAIN:
    allocation_show_figure(allocation_figure, '07_order_two_reconstruction')

# %%

if MAIN:
    allocation_figure = plot_background_changes_the_game(evaluated, prepared['feature_names'])

if MAIN:
    allocation_show_figure(allocation_figure, '08_background_changes_the_game')

# %%

if MAIN:
    allocation_figure = plot_contextual_pair_interventions(evaluated, allocation_play, prepared['feature_names'])

if MAIN:
    allocation_show_figure(allocation_figure, '09_contextual_pair_interventions')

# %%

if MAIN:
    allocation_MODEL_KEY = 'clean_0'

if MAIN:
    allocation_SOURCE_ID = 0

if MAIN:
    BACKGROUND_KEY = 'primary64'

if MAIN:
    PAIR = (0, 1)

if MAIN:
    CONTEXT_MASK = 252

if MAIN:
    DONOR_SOURCE_ID = None

if MAIN:
    context_play = empirical_context_play(evaluated, methods=allocation_methods, model_key=allocation_MODEL_KEY, source_id=allocation_SOURCE_ID, background_key=BACKGROUND_KEY, pair=PAIR, context_mask=CONTEXT_MASK, donor_source_id=DONOR_SOURCE_ID)

if MAIN:
    allocation_figure = plot_donor_context_play(context_play, prepared['feature_names'])

if MAIN:
    allocation_show_figure(allocation_figure, '11_donor_context_play')

# %%

import itertools as estimator_itertools

import math as estimator_math

import os as estimator_os

import sys as estimator_sys

from collections.abc import Callable

from pathlib import Path as estimator_Path

from types import SimpleNamespace as estimator_SimpleNamespace

from typing import Any as estimator_Any

import numpy as estimator_np

import pandas as estimator_pd

import matplotlib.pyplot as estimator_plt

import torch as estimator_t

from torch import Tensor as estimator_Tensor

from IPython.display import display as estimator_display

if MAIN:
    estimator_chapter = 'chapter16_shapley_attribution_baselines'

if MAIN:
    estimator_root_dir = next((p for p in [estimator_Path.cwd(), *estimator_Path.cwd().parents] if (p / estimator_chapter).exists()))

if MAIN:
    if str(estimator_root_dir) not in estimator_sys.path:
        estimator_sys.path.insert(0, str(estimator_root_dir))

if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods_tests as estimator_tests
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_study as study, estimator_plots as plots

from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.estimator_methods import _players, _masks, _values, _queried_inputs, interaction_masks, boundary_masks

if MAIN:
    estimator_t.set_num_threads(4)

if MAIN:
    OUTPUT_DIR = estimator_Path(estimator_os.environ.get('ARENA_ESTIMATOR_OUTPUT_DIRECTORY', './estimator_outputs'))

if MAIN:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def estimator_show_figure(figure, name):
    figure.savefig(OUTPUT_DIR / f'{name}.png', dpi=130, bbox_inches='tight')
    estimator_display(figure)
    estimator_plt.close(figure)

# %%

if MAIN:
    game_rows = []

if MAIN:
    for mask in range(16):
        a, b, c, d = [float(bool(mask & 1 << i)) for i in range(4)]
        estimator_value = 0.2 * a + 0.2 * b + 5 * c + 4 * d + 3 * a * b + 2 * a * b * c
        game_rows.append({'mask': f'{mask:04b}', 'A': a, 'B': b, 'C': c, 'D': d, 'value': estimator_value})

if MAIN:
    estimator_display(estimator_pd.DataFrame(game_rows).set_index('mask'))

if MAIN:
    print('Query {A,B}:', game_rows[3]['value'])

# %%

def coalition_design(masks: estimator_Tensor, *, n_players: int) -> estimator_Tensor:
    """Return X[queried coalition, feature] in float64."""
    n = _players(n_players)
    _masks(masks, n)
    bits = 1 << estimator_t.arange(n, dtype=estimator_t.int64)
    return masks[:, None].bitwise_and(bits).ne(0).to(estimator_t.float64)

if MAIN:
    estimator_tests.test_coalition_design(fn=coalition_design)

# %%

def sii_query_weights(masks: estimator_Tensor, *, n_players: int, order: int=2) -> estimator_Tensor:
    """The SHAP-IQ gamma coefficient of each queried value in every order-s SII.

    If k=|T intersection S|, gamma(T,S)=(-1)**(s-k) /
    ((n-s+1) * choose(n-s, |T|-k)). No model evaluations occur here.
    """
    n = _players(n_players)
    terms = interaction_masks(n, order)
    design = coalition_design(masks, n_players=n)
    term_design = coalition_design(terms, n_players=n)
    overlap = (design @ term_design.T).to(estimator_t.int64)
    outside = design.sum(-1).to(estimator_t.int64)[:, None] - overlap
    denominator = estimator_t.tensor([(n - order + 1) * estimator_math.comb(n - order, r) for r in range(n - order + 1)], dtype=estimator_t.float64)
    sign = estimator_t.where((order - overlap) % 2 == 0, 1.0, -1.0).to(estimator_t.float64)
    return sign / denominator[outside]

if MAIN:
    estimator_tests.test_sii_query_weights(fn=sii_query_weights)

# %%

if MAIN:
    estimator_games = plots.analytic_games(coalition_design)

if MAIN:
    estimator_figure = plots.plot_shared_query(estimator_games, sii_query_weights, query_mask=3)

if MAIN:
    estimator_show_figure(estimator_figure, '01_shared_query')

# %%

if MAIN:
    QUERY_MASK = 3

if MAIN:
    GAME_NAME = 'mixed'

if MAIN:
    estimator_figure = plots.plot_shared_query(estimator_games, sii_query_weights, query_mask=QUERY_MASK, game_name=GAME_NAME)

if MAIN:
    estimator_show_figure(estimator_figure, '02_query_play')

# %%

def sample_coalitions(*, n_players: int, draws: int, seed: int, proposal: str='kernel') -> tuple[estimator_Tensor, estimator_Tensor]:
    """Draw iid interior coalitions, retaining duplicate draws and per-set p(T).

    Draw size first, then a uniformly random subset of that size. Kernel means
    p(T) proportional to 1/choose(n-2, |T|-1), restricted to 2..n-2.
    Uniform means uniform over individual sets in that same support.
    """
    n = _players(n_players)
    if type(draws) is not int or draws < 2 or type(seed) is not int or (seed < 0):
        raise ValueError('draws >= 2 and a nonnegative integer seed are required.')
    if proposal not in ('kernel', 'uniform'):
        raise ValueError("proposal must be 'kernel' or 'uniform'.")
    sizes = estimator_t.arange(2, n - 1, dtype=estimator_t.int64)
    if proposal == 'kernel':
        size_mass = 1.0 / (sizes.to(estimator_t.float64) * (n - sizes))
    else:
        size_mass = estimator_t.tensor([estimator_math.comb(n, int(k)) for k in sizes], dtype=estimator_t.float64)
    size_probability = size_mass / size_mass.sum()
    generator = estimator_t.Generator(device='cpu').manual_seed(seed)
    chosen = estimator_t.multinomial(size_probability, draws, replacement=True, generator=generator)
    selected_sizes = sizes[chosen]
    permutations = estimator_t.rand((draws, n), generator=generator, dtype=estimator_t.float64).argsort(dim=-1)
    take = estimator_t.arange(n)[None, :] < selected_sizes[:, None]
    masks = ((1 << permutations) * take).sum(-1)
    combinations = estimator_t.tensor([estimator_math.comb(n, int(k)) for k in sizes], dtype=estimator_t.float64)
    probabilities = (size_probability / combinations)[chosen]
    return (masks, probabilities)

if MAIN:
    estimator_tests.test_sample_coalitions(fn=sample_coalitions)

# %%

if MAIN:
    estimator_figure = plots.plot_sampling(sample_coalitions)

if MAIN:
    estimator_show_figure(estimator_figure, '03_sampling_and_cost')

# %%

def shapiq_estimate(boundary_values: estimator_Tensor, sampled_values: estimator_Tensor, sampled_masks: estimator_Tensor, probabilities: estimator_Tensor, *, n_players: int, order: int=2) -> tuple[estimator_Tensor, estimator_Tensor]:
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
    return (estimate, standard_error)

if MAIN:
    estimator_tests.test_shapiq_estimate(fn=shapiq_estimate)

# %%

def query_sampled_game(game: Callable[[estimator_Tensor], estimator_Tensor], sampled_masks: estimator_Tensor, *, n_players: int) -> dict[str, estimator_Any]:
    """Query each required coalition once while preserving iid duplicate draws."""
    n = _players(n_players)
    _masks(sampled_masks, n)
    edges = boundary_masks(n)
    requested = estimator_t.cat((edges, sampled_masks))
    unique, inverse = estimator_t.unique(requested, sorted=True, return_inverse=True)
    observed = game(unique)
    _values(observed, len(unique), 'game output')
    restored = observed[..., inverse]
    return {'boundary_values': restored[..., :len(edges)], 'sampled_values': restored[..., len(edges):], 'queried_masks': unique, 'queried_values': observed, 'draw_requests': len(sampled_masks), 'boundary_queries': len(edges), 'total_requests': len(requested), 'unique_queries': len(unique)}

if MAIN:
    estimator_tests.test_query_sampled_game(fn=query_sampled_game)

# %%

def constrained_wls(design: estimator_Tensor, values: estimator_Tensor, weights: estimator_Tensor, constraints: estimator_Tensor, targets: estimator_Tensor, *, rtol: float=1e-12) -> dict[str, estimator_Any]:
    """Minimize ||sqrt(W)(X beta-y)|| subject to C beta=d, using a nullspace.

    Rank-deficient sampled designs return the minimum-norm free solution and an
    explicit identified=False flag. They never silently acquire a ridge penalty.
    """
    if design.ndim != 2 or constraints.ndim != 2 or constraints.shape[1] != design.shape[1]:
        raise ValueError('design and constraints must have the same coefficient axis.')
    rows, columns = design.shape
    for name, value, length in (('design', design, columns), ('values', values, rows), ('weights', weights, rows), ('constraints', constraints, columns), ('targets', targets, len(constraints))):
        _values(value, length, name)
    if weights.ndim != 1 or bool((weights < 0).any()) or (not bool((weights > 0).any())):
        raise ValueError('weights must be nonnegative with a positive entry.')
    if values.shape[:-1] != targets.shape[:-1] or not 0 < rtol < 1:
        raise ValueError('Target batches must match; rtol must lie between zero and one.')
    _, singular, vh = estimator_t.linalg.svd(constraints, full_matrices=True)
    rank = int((singular > rtol * singular.max()).sum()) if len(singular) else 0
    particular = targets @ estimator_t.linalg.pinv(constraints, rtol=rtol).T
    error = particular @ constraints.T - targets
    scale = max(1.0, float(targets.abs().max()))
    if float(error.abs().max()) > 100 * rtol * scale:
        raise ValueError('The equality constraints are inconsistent.')
    nullspace = vh[rank:].T
    free = nullspace.shape[1]
    if free:
        full_weighted_design = weights.sqrt()[:, None] * design
        weighted_design = full_weighted_design @ nullspace
        residual = (values - particular @ design.T) * weights.sqrt()
        u, singular_values, free_vh = estimator_t.linalg.svd(weighted_design, full_matrices=False)
        cutoff = rtol * estimator_t.linalg.matrix_norm(full_weighted_design, ord=2)
        retained = singular_values > cutoff
        free_solution = residual @ u[:, retained] / singular_values[retained] @ free_vh[retained]
        coefficients = particular + free_solution @ nullspace.T
        free_rank = int(retained.sum())
    else:
        coefficients, free_rank = (particular, 0)
        singular_values = estimator_t.empty(0, dtype=estimator_t.float64)
    return {'coefficients': coefficients, 'constraint_rank': rank, 'free_parameters': free, 'free_rank': free_rank, 'identified': free_rank == free, 'singular_values': singular_values, 'constraint_residual': coefficients @ constraints.T - targets}

if MAIN:
    estimator_tests.test_constrained_wls(fn=constrained_wls)

# %%

def pair_residual_design(masks: estimator_Tensor, *, n_players: int) -> estimator_Tensor:
    """Bernoulli pair design: -1/2 if exactly one pair member is present, else 0."""
    design = coalition_design(masks, n_players=n_players)
    pairs = estimator_t.tensor(list(estimator_itertools.combinations(range(n_players), 2)), dtype=estimator_t.int64)
    left, right = (design[:, pairs[:, 0]], design[:, pairs[:, 1]])
    return left * right - (left + right) / 2


def kernel_weights(masks: estimator_Tensor, *, n_players: int, order: int) -> estimator_Tensor:
    """Finite per-coalition KernelSHAP-IQ weights; boundary markers are zero."""
    n = _players(n_players)
    if order not in (1, 2):
        raise ValueError('This lesson proves and implements orders one and two only.')
    sizes = coalition_design(masks, n_players=n).sum(-1).to(estimator_t.int64)
    return estimator_t.tensor([1 / estimator_math.comb(n - 2 * order, int(k) - order) if order <= k <= n - order else 0.0 for k in sizes], dtype=estimator_t.float64)

if MAIN:
    estimator_tests.test_pair_residual_design(fn=pair_residual_design)

if MAIN:
    estimator_tests.test_kernel_weights(fn=kernel_weights)

# %%

def kernelshap_iq(boundary_values: estimator_Tensor, sampled_values: estimator_Tensor, sampled_masks: estimator_Tensor, probabilities: estimator_Tensor, *, n_players: int) -> dict[str, estimator_Any]:
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
    edge_values, draw_values = (boundary_values - baseline, sampled_values - baseline)
    masks = estimator_t.cat((edges, sampled_masks))
    values = estimator_t.cat((edge_values, draw_values), dim=-1)
    adjustment = 1.0 / (len(sampled_masks) * probabilities)
    first_weights = kernel_weights(masks, n_players=n, order=1) * estimator_t.cat((estimator_t.ones(len(edges), dtype=estimator_t.float64), adjustment))
    design = coalition_design(masks, n_players=n)
    first = constrained_wls(design, values, first_weights, estimator_t.ones((1, n), dtype=estimator_t.float64), edge_values[..., -1:])
    residual = draw_values - first['coefficients'] @ coalition_design(sampled_masks, n_players=n).T
    single_positions = estimator_t.searchsorted(edges, 1 << estimator_t.arange(n))
    complement_positions = estimator_t.searchsorted(edges, 2 ** n - 1 ^ 1 << estimator_t.arange(n))
    targets = edge_values[..., -1:] - edge_values[..., single_positions] - edge_values[..., complement_positions]
    incidence = coalition_design(interaction_masks(n, 2), n_players=n).T
    second = constrained_wls(pair_residual_design(sampled_masks, n_players=n), residual, kernel_weights(sampled_masks, n_players=n, order=2) * adjustment, incidence, targets)
    return {'shapley': first['coefficients'], 'pairs': second['coefficients'], 'first_stage': first, 'pair_stage': second}

if MAIN:
    estimator_tests.test_kernelshap_iq(fn=kernelshap_iq)

# %%

if MAIN:
    estimator_figure = plots.plot_boundary_limit(kernelshap_iq)

if MAIN:
    estimator_show_figure(estimator_figure, '04_boundary_limit')

# %%

if MAIN:
    estimator_methods = estimator_SimpleNamespace(coalition_design=coalition_design, sii_query_weights=sii_query_weights, sample_coalitions=sample_coalitions, shapiq_estimate=shapiq_estimate, query_sampled_game=query_sampled_game, constrained_wls=constrained_wls, pair_residual_design=pair_residual_design, kernel_weights=kernel_weights, kernelshap_iq=kernelshap_iq)

# %%

if MAIN:
    payload = study.run_study(estimator_methods, OUTPUT_DIR / 'native')

if MAIN:
    model_games = payload['games']

if MAIN:
    print('All 21 known model-state checks passed.')

if MAIN:
    print('Complete reference:', model_games['exact_reference_coalition_queries'], 'coalition queries;')

if MAIN:
    print(model_games['exact_reference_donor_predictions'], 'underlying donor predictions.')

if MAIN:
    source_zero = int(estimator_t.nonzero(model_games['source_ids'] == 0).item())

if MAIN:
    estimator_display(estimator_pd.DataFrame({'feature': model_games['feature_names'], 'source 0 recipe': model_games['queries'][source_zero].numpy(), 'first intact donor': model_games['donors'][0].numpy()}).set_index('feature'))

# %%

if MAIN:
    estimator_figure = plots.plot_model_controls(payload)

if MAIN:
    estimator_show_figure(estimator_figure, '05_model_controls')

# %%

if MAIN:
    estimator_figure = plots.plot_convergence(payload)

if MAIN:
    estimator_show_figure(estimator_figure, '06_query_budget_errors')

# %%

if MAIN:
    estimator_figure = plots.plot_signed_comparison(payload)

if MAIN:
    estimator_show_figure(estimator_figure, '07_signed_pair_comparison')

# %%

if MAIN:
    estimator_figure = plots.plot_variance(payload)

if MAIN:
    estimator_show_figure(estimator_figure, '08_sampling_variance')

# %%

if MAIN:
    estimator_figure = plots.plot_query_costs(payload)

if MAIN:
    estimator_show_figure(estimator_figure, '09_unique_queries_and_rank')

# %%

if MAIN:
    estimator_MODEL_KEY = 'clean_0'

if MAIN:
    estimator_SOURCE_ID = 0

if MAIN:
    DRAWS = 128

if MAIN:
    SEED = 161750

if MAIN:
    PROPOSAL = 'kernel'

if MAIN:
    estimator_play = plots.make_native_play(payload, estimator_methods, model_key=estimator_MODEL_KEY, source_id=estimator_SOURCE_ID, draws=DRAWS, seed=SEED, proposal=PROPOSAL)

if MAIN:
    print('Unique queried masks:', estimator_play['receipt']['queried_masks'].tolist())

if MAIN:
    print('Pair free rank:', estimator_play['kernelshap_iq']['pair_stage']['free_rank'], '/ 20')

if MAIN:
    if estimator_play['exploratory_audit_reuse']:
        print('Exploratory audit reuse: this play does not change the fixed study.')

if MAIN:
    estimator_figure = plots.plot_native_play(estimator_play)

if MAIN:
    estimator_show_figure(estimator_figure, '10_native_query_play')

# %%
