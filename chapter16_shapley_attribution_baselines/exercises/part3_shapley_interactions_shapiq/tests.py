from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

import torch as t

from arena_ext import shapley_attribution as reference


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part3_shapley_interactions_shapiq import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _verification_report() -> dict:
    return json.loads((_section_dir() / "verification_report.json").read_text())


def _gpu_report() -> dict:
    return _verification_report()["metrics"]["gpu_test"]


def _as_dict(report: object) -> dict[str, Any]:
    return report if isinstance(report, dict) else report.__dict__


def _assert_close(actual: Any, expected: Any, *, name: str, atol: float = 1e-6) -> None:
    if hasattr(actual, "tolist"):
        actual = actual.tolist()
    if hasattr(expected, "tolist"):
        expected = expected.tolist()
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{name} should be a dictionary-like report."
        assert actual.keys() == expected.keys(), (
            f"{name} should expose the same fields as the independent reference."
        )
        for key, expected_value in expected.items():
            _assert_close(actual[key], expected_value, name=f"{name}.{key}", atol=atol)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{name} should be a list-like value."
        assert len(actual) == len(expected), f"{name} should have length {len(expected)}."
        for index, expected_value in enumerate(expected):
            _assert_close(actual[index], expected_value, name=f"{name}[{index}]", atol=atol)
        return
    if isinstance(expected, float):
        assert abs(float(actual) - expected) <= atol, (
            f"{name} should be {expected}, got {actual}."
        )
        return
    assert actual == expected, f"{name} should be {expected!r}, got {actual!r}."


def _assert_report_close(actual: object, expected: object, *, msg: str, atol: float = 1e-6) -> None:
    _assert_close(_as_dict(actual), _as_dict(expected), name=msg, atol=atol)


def test_additive_game_enumerates_complete_zero_interaction_table(
    additive_game: Callable | None = None,
    pairwise_shapley_interactions: Callable | None = None,
):
    additive_game = additive_game or _solutions().additive_game
    pairwise_shapley_interactions = (
        pairwise_shapley_interactions or _solutions().pairwise_shapley_interactions
    )
    weights = t.tensor([1.0, -2.0, 0.5])
    values = additive_game(weights)
    expected_values = reference.additive_game(weights)
    assert values == expected_values, (
        "additive_game should enumerate every coalition with the sum of selected weights."
    )
    interactions = pairwise_shapley_interactions(values, num_players=3)
    expected_interactions = reference.pairwise_shapley_interactions(
        expected_values,
        num_players=3,
    )
    assert t.allclose(interactions, expected_interactions, atol=1e-9), (
        "Additive games should have zero pairwise Shapley interaction terms."
    )
    assert interactions.shape == (3, 3), (
        "pairwise_shapley_interactions should return a square player-by-player matrix."
    )
    print("All tests in `test_additive_game_enumerates_complete_zero_interaction_table` passed!")


def test_interaction_game_recovers_target_pair_delta(
    interaction_game: Callable | None = None,
    pairwise_shapley_interactions: Callable | None = None,
):
    interaction_game = interaction_game or _solutions().interaction_game
    pairwise_shapley_interactions = (
        pairwise_shapley_interactions or _solutions().pairwise_shapley_interactions
    )
    values = interaction_game(
        3,
        pair=(0, 1),
        pair_weight=1.0,
        additive_weights=t.tensor([0.5, -1.0, 2.0]),
    )
    expected_values = reference.interaction_game(
        3,
        pair=(0, 1),
        pair_weight=1.0,
        additive_weights=t.tensor([0.5, -1.0, 2.0]),
    )
    assert values == expected_values, (
        "interaction_game should keep additive effects separate from the target pair bonus."
    )
    interactions = pairwise_shapley_interactions(values, num_players=3)
    assert abs(float(interactions[0, 1]) - 1.0) < 1e-9, (
        "The planted (0, 1) pair interaction should be recovered exactly."
    )
    assert abs(float(interactions[0, 2])) < 1e-9 and abs(float(interactions[1, 2])) < 1e-9, (
        "Off-target pairs should remain zero in the single-interaction control game."
    )
    assert t.allclose(interactions, interactions.T, atol=1e-9), (
        "The pairwise interaction matrix should be symmetric."
    )
    assert t.allclose(t.diag(interactions), t.zeros(3, dtype=interactions.dtype), atol=1e-9), (
        "The diagonal should stay zero because self-interactions are not defined."
    )
    print("All tests in `test_interaction_game_recovers_target_pair_delta` passed!")


def test_pairwise_interaction_report_matches_reference_and_rejects_spurious_pairs(
    pairwise_interaction_report: Callable | None = None,
):
    pairwise_interaction_report = (
        pairwise_interaction_report or _solutions().pairwise_interaction_report
    )
    values = reference.interaction_game(
        3,
        pair=(0, 1),
        pair_weight=1.0,
        additive_weights=t.tensor([0.5, -1.0, 2.0]),
    )
    report = pairwise_interaction_report(values, num_players=3)
    expected = reference.pairwise_interaction_report(values, num_players=3)
    _assert_report_close(report, expected, msg="Pairwise interaction report", atol=1e-9)
    assert report.recovers_interaction, (
        "The report should pass only when the target pair is correct and all other pairs are near zero."
    )
    noisy_values = dict(values)
    noisy_values[frozenset((0, 2))] = noisy_values[frozenset((0, 2))] + 0.5
    noisy_report = pairwise_interaction_report(noisy_values, num_players=3)
    assert not noisy_report.recovers_interaction, (
        "Adding an off-target pair effect should fail the spurious-interaction control."
    )
    print("All tests in `test_pairwise_interaction_report_matches_reference_and_rejects_spurious_pairs` passed!")


def test_shapiq_interaction_parity_report_matches_exact_sii(
    shapiq_interaction_parity_report: Callable | None = None,
):
    shapiq_interaction_parity_report = (
        shapiq_interaction_parity_report or _solutions().shapiq_interaction_parity_report
    )
    values = reference.interaction_game(
        3,
        pair=(0, 1),
        pair_weight=1.0,
        additive_weights=t.tensor([0.5, -1.0, 2.0]),
    )
    report = shapiq_interaction_parity_report(values, num_players=3)
    expected = reference.shapiq_interaction_parity_report(values, num_players=3)
    _assert_report_close(report, expected, msg="shapiq SII parity report", atol=1e-5)
    assert report.shapiq_available, (
        "The uv environment should include shapiq; missing shapiq is a real failure, not a skipped test."
    )
    assert report.matches_shapiq and report.max_abs_error < 1e-5, (
        "Closed-form pairwise interactions should match shapiq SII on the complete toy table."
    )
    print("All tests in `test_shapiq_interaction_parity_report_matches_exact_sii` passed!")


def test_shapiq_pairwise_matrix_matches_student_exact_sii(
    polynomial_game: Callable | None = None,
    pairwise_shapley_interactions: Callable | None = None,
    shapiq_pairwise_interactions: Callable | None = None,
):
    solutions = _solutions()
    polynomial_game = polynomial_game or solutions.polynomial_game
    pairwise_shapley_interactions = (
        pairwise_shapley_interactions or solutions.pairwise_shapley_interactions
    )
    shapiq_pairwise_interactions = (
        shapiq_pairwise_interactions or solutions.shapiq_pairwise_interactions
    )
    values = polynomial_game(
        t.tensor([0.2, 0.2, 5.0, 4.0], dtype=t.float64),
        {(0, 1): 3.0, (0, 1, 2): 2.0},
    )
    exact = pairwise_shapley_interactions(values, num_players=4)
    observed = shapiq_pairwise_interactions(values, num_players=4)
    t.testing.assert_close(
        observed,
        exact,
        atol=1e-6,
        rtol=0.0,
        msg="Pinned shapiq SII should match the student implementation on all six pairs.",
    )
    print("All tests in `test_shapiq_pairwise_matrix_matches_student_exact_sii` passed!")


def test_polynomial_game_enumerates_exact_ground_truth(
    polynomial_game: Callable | None = None,
):
    polynomial_game = polynomial_game or _solutions().polynomial_game
    values = polynomial_game(
        t.tensor([0.2, 0.2, 5.0, 4.0], dtype=t.float64),
        {(0, 1): 3.0, (0, 1, 2): 2.0},
    )
    assert len(values) == 16, (
        "A four-player exact game should contain all 2**4 coalition values."
    )
    assert values[frozenset()] == 0.0, (
        "The empty coalition should be the zero baseline for this game."
    )
    assert abs(values[frozenset((0, 1))] - 3.4) < 1e-9, (
        "The target pair should include both additive terms and the planted synergy."
    )
    assert abs(values[frozenset((0, 1, 2))] - 10.4) < 1e-9, (
        "The three-player coalition should include the planted higher-order term."
    )
    assert abs(values[frozenset((0, 1, 2, 3))] - 14.4) < 1e-9, (
        "The full coalition should sum all additive, pair, and higher-order terms."
    )
    print("All tests in `test_polynomial_game_enumerates_exact_ground_truth` passed!")


def test_discrete_second_difference_isolates_synergy_from_additive_effects(
    polynomial_game: Callable | None = None,
    discrete_second_difference: Callable | None = None,
):
    solutions = _solutions()
    polynomial_game = polynomial_game or solutions.polynomial_game
    discrete_second_difference = (
        discrete_second_difference or solutions.discrete_second_difference
    )
    weights = t.tensor([0.2, 0.2, 5.0, 4.0], dtype=t.float64)
    interacting = polynomial_game(weights, {(0, 1): 3.0, (0, 1, 2): 2.0})
    additive = polynomial_game(weights, {})
    for context in (frozenset(), frozenset((2,)), frozenset((3,)), frozenset((2, 3))):
        observed = discrete_second_difference(
            interacting,
            context,
            (0, 1),
            num_players=4,
        )
        expected = 5.0 if 2 in context else 3.0
        assert abs(observed - expected) < 1e-9, (
            "The target pair's second difference should expose when the three-way term is active."
        )
        control = discrete_second_difference(
            additive,
            context,
            (0, 1),
            num_players=4,
        )
        assert abs(control) < 1e-9, (
            "The matched additive control should have zero second difference."
        )
    print(
        "All tests in "
        "`test_discrete_second_difference_isolates_synergy_from_additive_effects` passed!"
    )


def test_pairwise_sii_recovers_pair_hidden_by_large_additive_effects(
    polynomial_game: Callable | None = None,
    pairwise_shapley_interactions: Callable | None = None,
):
    solutions = _solutions()
    polynomial_game = polynomial_game or solutions.polynomial_game
    pairwise_shapley_interactions = (
        pairwise_shapley_interactions or solutions.pairwise_shapley_interactions
    )
    values = polynomial_game(
        t.tensor([0.2, 0.2, 5.0, 4.0], dtype=t.float64),
        {(0, 1): 3.0, (0, 1, 2): 2.0},
    )
    interactions = pairwise_shapley_interactions(values, num_players=4)
    expected = t.zeros((4, 4), dtype=t.float64)
    expected[0, 1] = expected[1, 0] = 4.0
    expected[0, 2] = expected[2, 0] = 1.0
    expected[1, 2] = expected[2, 1] = 1.0
    t.testing.assert_close(
        interactions,
        expected,
        atol=1e-9,
        rtol=0.0,
        msg="Exact SII should recover the pair term plus the contextual three-way contribution.",
    )
    print("All tests in `test_pairwise_sii_recovers_pair_hidden_by_large_additive_effects` passed!")


def test_exact_shapley_values_make_main_effect_ranking_misleading(
    polynomial_game: Callable | None = None,
    exact_shapley_values: Callable | None = None,
):
    solutions = _solutions()
    polynomial_game = polynomial_game or solutions.polynomial_game
    exact_shapley_values = exact_shapley_values or solutions.exact_shapley_values
    values = polynomial_game(
        t.tensor([0.2, 0.2, 5.0, 4.0], dtype=t.float64),
        {(0, 1): 3.0, (0, 1, 2): 2.0},
    )
    main_effects = exact_shapley_values(values, num_players=4)
    t.testing.assert_close(
        main_effects,
        t.tensor([71 / 30, 71 / 30, 17 / 3, 4.0], dtype=t.float64),
        atol=1e-9,
        rtol=0.0,
        msg="Individual values should divide both pair and three-way dividends symmetrically.",
    )
    assert set(main_effects.topk(2).indices.tolist()) == {2, 3}, (
        "The two largest individual Shapley values should be additive decoys, not the interacting pair."
    )
    print("All tests in `test_exact_shapley_values_make_main_effect_ranking_misleading` passed!")


def test_permutation_sampling_is_reproducible_and_converges(
    polynomial_game: Callable | None = None,
    pairwise_shapley_interactions: Callable | None = None,
    sampled_pair_interaction: Callable | None = None,
):
    solutions = _solutions()
    polynomial_game = polynomial_game or solutions.polynomial_game
    pairwise_shapley_interactions = (
        pairwise_shapley_interactions or solutions.pairwise_shapley_interactions
    )
    sampled_pair_interaction = (
        sampled_pair_interaction or solutions.sampled_pair_interaction
    )
    values = polynomial_game(
        t.tensor([0.2, 0.2, 5.0, 4.0], dtype=t.float64),
        {(0, 1): 3.0, (0, 1, 2): 2.0},
    )
    exact = float(pairwise_shapley_interactions(values, num_players=4)[0, 1])
    first = sampled_pair_interaction(
        values,
        num_players=4,
        pair=(0, 1),
        budget=4096,
        seed=0,
    )
    second = sampled_pair_interaction(
        values,
        num_players=4,
        pair=(0, 1),
        budget=4096,
        seed=0,
    )
    assert exact == 4.0, (
        "The triple term should contribute 1.0 to the target pair's second-order SII."
    )
    assert first == second, "A fixed seed should reproduce the same sampled estimate."
    assert abs(first - exact) < 0.02, (
        "A 4096-permutation estimate should converge close to exact SII on the contextual game."
    )
    print("All tests in `test_permutation_sampling_is_reproducible_and_converges` passed!")


def test_within_size_value_permutation_breaks_target_semantics(
    polynomial_game: Callable | None = None,
    pairwise_shapley_interactions: Callable | None = None,
    permute_coalition_values_within_sizes: Callable | None = None,
):
    solutions = _solutions()
    polynomial_game = polynomial_game or solutions.polynomial_game
    pairwise_shapley_interactions = (
        pairwise_shapley_interactions or solutions.pairwise_shapley_interactions
    )
    permute_coalition_values_within_sizes = (
        permute_coalition_values_within_sizes
        or solutions.permute_coalition_values_within_sizes
    )
    values = polynomial_game(
        t.tensor([0.2, 0.2, 5.0, 4.0], dtype=t.float64),
        {(0, 1): 3.0, (0, 1, 2): 2.0},
    )
    permuted = permute_coalition_values_within_sizes(values, num_players=4, seed=0)
    for size in range(5):
        original_values = sorted(value for coalition, value in values.items() if len(coalition) == size)
        permuted_values = sorted(
            value for coalition, value in permuted.items() if len(coalition) == size
        )
        assert original_values == permuted_values, (
            "The control should preserve the value distribution at every coalition size."
        )
    permuted_target = float(pairwise_shapley_interactions(permuted, num_players=4)[0, 1])
    assert abs(permuted_target - 0.1) < 1e-9, (
        "The fixed within-size permutation should reduce the true target interaction from 4.0 to 0.1."
    )
    print("All tests in `test_within_size_value_permutation_breaks_target_semantics` passed!")


def test_interaction_recovery_report_tracks_rank_spurious_terms_and_error(
    interaction_recovery_report: Callable | None = None,
):
    interaction_recovery_report = (
        interaction_recovery_report or _solutions().interaction_recovery_report
    )
    expected = t.zeros((4, 4), dtype=t.float64)
    expected[0, 1] = expected[1, 0] = 3.0
    observed = expected.clone()
    observed[2, 3] = observed[3, 2] = 0.25
    report = interaction_recovery_report(observed, expected, target_pair=(0, 1))
    assert report.predicted_pair == (0, 1) and report.target_rank == 1, (
        "The planted pair should remain the strongest recovered interaction."
    )
    assert report.target_value == 3.0, "The report should expose the target interaction value."
    assert report.max_off_target_interaction == 0.25, (
        "The report should expose the largest off-target interaction."
    )
    assert abs(report.mean_abs_error - (0.25 / 6)) < 1e-9, (
        "Matrix MAE should average over the six unique unordered feature pairs."
    )
    print(
        "All tests in "
        "`test_interaction_recovery_report_tracks_rank_spurious_terms_and_error` passed!"
    )


def test_neural_game_value_table_contains_planted_interactions(
    binary_feature_table: Callable | None = None,
    true_neural_game_scores: Callable | None = None,
    coalition_table_from_true_game: Callable | None = None,
    pairwise_shapley_interactions: Callable | None = None,
):
    solutions = _solutions()
    binary_feature_table = binary_feature_table or solutions.binary_feature_table
    true_neural_game_scores = true_neural_game_scores or solutions.true_neural_game_scores
    coalition_table_from_true_game = (
        coalition_table_from_true_game or solutions.coalition_table_from_true_game
    )
    pairwise_shapley_interactions = (
        pairwise_shapley_interactions or solutions.pairwise_shapley_interactions
    )
    device = t.device("cpu")
    inputs = binary_feature_table(device)
    scores = true_neural_game_scores(inputs).squeeze(-1)
    values = coalition_table_from_true_game(device)
    interactions = pairwise_shapley_interactions(
        values,
        num_players=solutions.NEURAL_GAME_NUM_PLAYERS,
    )
    assert inputs.shape == (16, 4), (
        "The finite neural game should enumerate all 16 binary feature coalitions."
    )
    assert scores.shape == (16,), (
        "true_neural_game_scores should return one scalar score per binary input."
    )
    assert abs(float(interactions[0, 2]) - 2.2) < 1e-6, (
        "The true game should contain the planted positive (0, 2) interaction."
    )
    assert abs(float(interactions[1, 3]) + 1.5) < 1e-6, (
        "The true game should contain the planted negative (1, 3) interaction."
    )
    assert abs(float(interactions[0, 1])) < 1e-6, (
        "Unplanted feature pairs should remain near zero in the exact true table."
    )
    print("All tests in `test_neural_game_value_table_contains_planted_interactions` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["additive_interactions"]["max_abs_interaction"] < 1e-9, (
        "The notebook contract should include the additive zero-interaction control."
    )
    assert result["target_pair"]["recovers_interaction"], (
        "The notebook contract should include the exact target-pair recovery control."
    )
    assert result["shapiq_parity"]["shapiq_available"], (
        "The notebook contract should require shapiq to be installed."
    )
    assert result["shapiq_parity"]["matches_shapiq"], (
        "The notebook contract should include shapiq parity on the toy coalition table."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_matches_shapley_interaction_contract(
    gpu: Mapping[str, Any] | None = None,
):
    gpu = dict(gpu or _gpu_report())
    assert gpu["cuda_available"] and gpu["preflight_passed"], (
        "The committed report should be produced by a passing CUDA preflight."
    )
    assert gpu["device"] == "NVIDIA GeForce RTX 5090 Laptop GPU", (
        "The report should record the local RTX 5090 Laptop GPU used for validation."
    )
    assert gpu["model_family"] == "cuda_trained_neural_coalition_game_mlp", (
        "16.3 should validate a trained finite neural coalition game, not a placeholder."
    )
    assert gpu["num_players"] == 4 and gpu["coalition_count"] == 16, (
        "The CUDA report should cover the complete four-feature coalition table."
    )
    assert gpu["fit_mse"] <= 1e-8 and gpu["fit_max_abs_error"] <= 1e-4, (
        "The trained model should fit the finite target game before interactions are trusted."
    )
    assert gpu["positive_interaction_pair"] == [0, 2], (
        "The report should identify the planted positive interaction pair."
    )
    assert gpu["negative_interaction_pair"] == [1, 3], (
        "The report should identify the planted negative interaction pair."
    )
    assert gpu["positive_interaction_value"] > 0 and gpu["negative_interaction_value"] < 0, (
        "The recovered interaction signs should match the planted game."
    )
    assert gpu["interaction_max_abs_error"] <= 1e-4, (
        "Model-derived pair interactions should match the exact true-game interactions."
    )
    assert gpu["max_spurious_interaction"] <= 1e-4, (
        "Unplanted feature pairs should stay near zero in the trained-model table."
    )
    assert gpu["shapiq_available"] and gpu["shapiq_matches"], (
        "The trained-model coalition table should also pass shapiq SII parity."
    )
    assert gpu["shuffled_control_interaction_error"] >= 1.0 and gpu["shuffled_control_rejected"], (
        "The shuffled-label trained-model control should be rejected."
    )
    assert gpu["peak_vram_gb"] < 1.0 and gpu["within_vram_budget"], (
        "The finite-game CUDA preflight should stay well inside the 1 GB expected budget."
    )
    print("All tests in `test_committed_gpu_report_matches_shapley_interaction_contract` passed!")
