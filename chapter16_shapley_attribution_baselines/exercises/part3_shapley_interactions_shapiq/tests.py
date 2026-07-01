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
