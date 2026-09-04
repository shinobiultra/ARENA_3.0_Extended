from collections.abc import Callable
import json
from pathlib import Path

import torch as t

from arena_ext import shapley_attribution as reference


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part1_exact_shapley_ground_truth_games import (
        solutions,
    )

    return solutions


def _assert_close(actual: float, expected: float, *, msg: str, atol: float = 1e-9) -> None:
    assert abs(actual - expected) <= atol, f"{msg} Expected {expected}, got {actual}."


def _assert_tensor_close(actual: t.Tensor, expected: t.Tensor, *, msg: str) -> None:
    assert actual.shape == expected.shape, (
        f"{msg} should have shape {tuple(expected.shape)}, got {tuple(actual.shape)}."
    )
    assert t.allclose(actual.double(), expected.double(), atol=1e-9, rtol=0), (
        f"{msg} should match the independent reference implementation."
    )


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} should expose the same fields as the independent reference."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, float):
            _assert_close(
                float(actual_value),
                expected_value,
                msg=f"{msg} field {key!r}",
                atol=1e-9,
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_all_coalitions_enumerates_the_power_set(
    all_coalitions: Callable | None = None,
):
    all_coalitions = all_coalitions or _solutions().all_coalitions
    coalitions = all_coalitions(3)
    expected = reference.all_coalitions(3)
    assert coalitions == expected, (
        "Coalitions should be ordered by size and match the exact power-set reference."
    )
    assert len(coalitions) == 8 and frozenset() in coalitions and frozenset({0, 1, 2}) in coalitions, (
        "A 3-player complete table should contain the empty set and the full coalition."
    )
    try:
        all_coalitions(0)
    except ValueError as exc:
        assert "num_players" in str(exc), (
            "Invalid player counts should raise a helpful num_players error."
        )
    else:
        raise AssertionError("num_players=0 should raise ValueError.")
    print("All tests in `test_all_coalitions_enumerates_the_power_set` passed!")


def test_additive_game_and_exact_shapley_recover_weights(
    additive_game: Callable | None = None,
    exact_shapley_values: Callable | None = None,
):
    additive_game = additive_game or _solutions().additive_game
    exact_shapley_values = exact_shapley_values or _solutions().exact_shapley_values
    weights = t.tensor([1.0, 2.0, -0.5])
    values = additive_game(weights)
    shapley = exact_shapley_values(values, num_players=3)
    expected_values = reference.additive_game(weights)
    expected_shapley = reference.exact_shapley_values(expected_values, num_players=3)

    assert values == expected_values, (
        "The additive game should assign each coalition the sum of its feature weights."
    )
    _assert_tensor_close(shapley, expected_shapley, msg="Additive exact Shapley values")
    assert shapley.tolist() == [1.0, 2.0, -0.5], (
        "Exact Shapley should recover every additive feature weight, including negative weights."
    )
    print("All tests in `test_additive_game_and_exact_shapley_recover_weights` passed!")


def test_exact_shapley_requires_a_complete_coalition_table(
    exact_shapley_values: Callable | None = None,
):
    exact_shapley_values = exact_shapley_values or _solutions().exact_shapley_values
    incomplete = {frozenset(): 0.0, frozenset({0}): 1.0}
    try:
        exact_shapley_values(incomplete, num_players=2)
    except ValueError as exc:
        assert "missing" in str(exc) and "coalition" in str(exc), (
            "Incomplete coalition tables should raise an explicit missing-coalition error."
        )
    else:
        raise AssertionError("Incomplete coalition tables should raise ValueError.")
    print("All tests in `test_exact_shapley_requires_a_complete_coalition_table` passed!")


def test_conjunction_game_splits_symmetric_credit_and_checks_efficiency(
    conjunction_game: Callable | None = None,
    exact_shapley_values: Callable | None = None,
    shapley_efficiency_report: Callable | None = None,
):
    conjunction_game = conjunction_game or _solutions().conjunction_game
    exact_shapley_values = exact_shapley_values or _solutions().exact_shapley_values
    shapley_efficiency_report = (
        shapley_efficiency_report or _solutions().shapley_efficiency_report
    )
    values = conjunction_game(3)
    shapley = exact_shapley_values(values, num_players=3)
    report = shapley_efficiency_report(values, num_players=3)
    expected = reference.shapley_efficiency_report(values, num_players=3)

    _assert_tensor_close(
        shapley,
        t.tensor([1 / 3, 1 / 3, 1 / 3], dtype=t.float64),
        msg="Symmetric conjunction Shapley values",
    )
    _assert_report_close(report, expected, msg="Conjunction efficiency report")
    assert report.satisfies_efficiency, (
        "The conjunction Shapley values should sum to v(full) - v(empty)."
    )
    print("All tests in `test_conjunction_game_splits_symmetric_credit_and_checks_efficiency` passed!")


def test_permutation_parity_report_matches_exact_formula(
    conjunction_game: Callable | None = None,
    permutation_parity_report: Callable | None = None,
):
    conjunction_game = conjunction_game or _solutions().conjunction_game
    permutation_parity_report = (
        permutation_parity_report or _solutions().permutation_parity_report
    )
    values = conjunction_game(3)
    report = permutation_parity_report(values, num_players=3)
    expected = reference.permutation_parity_report(values, num_players=3)

    _assert_report_close(report, expected, msg="Permutation parity report")
    assert report.matches_exact and report.max_abs_error == 0.0, (
        "A complete three-player game should match exact permutation averaging with zero error."
    )
    print("All tests in `test_permutation_parity_report_matches_exact_formula` passed!")


def test_interaction_gap_report_catches_leave_one_out_overcount(
    conjunction_game: Callable | None = None,
    interaction_gap_report: Callable | None = None,
):
    conjunction_game = conjunction_game or _solutions().conjunction_game
    interaction_gap_report = interaction_gap_report or _solutions().interaction_gap_report
    values = conjunction_game(2)
    report = interaction_gap_report(values, num_players=2, min_overcount=0.5)
    expected = reference.interaction_gap_report(values, num_players=2, min_overcount=0.5)

    _assert_report_close(report, expected, msg="Interaction gap report")
    assert report.shapley_total == 1.0 and report.leave_one_out_total == 2.0, (
        "In a two-player AND game, leave-one-out assigns one point to each feature even though total value is one."
    )
    assert report.detects_interaction_overcount, (
        "The report should flag leave-one-out overcounting on interaction-heavy games."
    )
    print("All tests in `test_interaction_gap_report_catches_leave_one_out_overcount` passed!")


def test_additive_smoke_test(additive_smoke_test: Callable | None = None):
    additive_smoke_test = additive_smoke_test or _solutions().additive_smoke_test
    result = additive_smoke_test()
    assert result["shapley"] == [1.0, 2.0, -0.5], (
        "The additive smoke test should expose exact feature-weight Shapley values."
    )
    assert result["efficiency"]["satisfies_efficiency"], (
        "The additive smoke test should include a passing efficiency check."
    )
    print("All tests in `test_additive_smoke_test` passed!")


def test_conjunction_smoke_test(conjunction_smoke_test: Callable | None = None):
    conjunction_smoke_test = conjunction_smoke_test or _solutions().conjunction_smoke_test
    result = conjunction_smoke_test()
    assert all(abs(value - (1 / 3)) < 1e-9 for value in result["shapley"]), (
        "The conjunction smoke test should split one unit of credit equally among three players."
    )
    assert result["efficiency"]["satisfies_efficiency"], (
        "The conjunction smoke test should include a passing efficiency check."
    )
    print("All tests in `test_conjunction_smoke_test` passed!")


def test_permutation_parity_smoke_test(
    permutation_parity_smoke_test: Callable | None = None,
):
    permutation_parity_smoke_test = (
        permutation_parity_smoke_test or _solutions().permutation_parity_smoke_test
    )
    result = permutation_parity_smoke_test()
    assert result["matches_exact"], (
        "Permutation averaging should exactly match the closed-form Shapley formula."
    )
    assert result["max_abs_error"] < 1e-9, (
        "Permutation parity should have negligible maximum absolute error."
    )
    print("All tests in `test_permutation_parity_smoke_test` passed!")


def test_interaction_failure_smoke_test(
    interaction_failure_smoke_test: Callable | None = None,
):
    interaction_failure_smoke_test = (
        interaction_failure_smoke_test or _solutions().interaction_failure_smoke_test
    )
    result = interaction_failure_smoke_test()
    assert result["shapley_total"] == 1.0, (
        "The interaction smoke test should keep total Shapley credit equal to game value."
    )
    assert result["leave_one_out_total"] == 2.0, (
        "The interaction smoke test should show leave-one-out overcounts the AND game."
    )
    assert result["detects_interaction_overcount"], (
        "The interaction smoke test should flag the overcounting failure mode."
    )
    print("All tests in `test_interaction_failure_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["additive"]["efficiency"]["satisfies_efficiency"], (
        "The notebook contract should include a passing additive efficiency check."
    )
    assert result["conjunction"]["efficiency"]["satisfies_efficiency"], (
        "The notebook contract should include a passing conjunction efficiency check."
    )
    assert result["permutation_parity"]["matches_exact"], (
        "The notebook contract should include a permutation-parity check."
    )
    assert result["interaction_failure"]["detects_interaction_overcount"], (
        "The notebook contract should include the leave-one-out interaction failure check."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_records_exact_shapley_preflight():
    report_path = Path(__file__).with_name("verification_report.json")
    report = json.loads(report_path.read_text())
    gpu = report["metrics"]["gpu_test"]
    claim_scope = report["claim_scope"].lower()

    assert report["accepted"] and report["tests_passed"], (
        "The committed 16.1 verification report should be accepted and test-backed."
    )
    assert report["known_failures"] == [], (
        "Course-ready exact Shapley should not ship known report failures."
    )
    assert gpu["cuda_available"] and gpu["preflight_passed"], (
        "The report should record a real CUDA preflight, not a CPU fallback."
    )
    assert gpu["model_family"] == "cuda_trained_neural_coalition_game_mlp", (
        "The report should identify the trained neural coalition game."
    )
    assert gpu["num_players"] == 4 and gpu["coalition_count"] == 16, (
        "16.1 should evaluate the complete four-feature binary coalition table."
    )
    assert gpu["complete_finite_domain_evaluated"] is True, (
        "The CUDA report should document that all 16 binary feature-table inputs were evaluated."
    )
    assert gpu["ood_generalization_claimed"] is False, (
        "16.1 should explicitly avoid claiming OOD generalization from a finite table."
    )
    assert gpu["generalization_scope"] == "complete_binary_feature_table", (
        "The generated metrics should scope generalization to the complete finite table."
    )
    assert gpu["training_example_count"] == 16 and gpu["training_steps"] == 1200, (
        "The report should pin the complete training table and training budget."
    )
    assert gpu["fit_mse"] <= 1e-8 and gpu["fit_max_abs_error"] <= 1e-4, (
        "The neural game should fit the finite coalition table before attribution."
    )
    assert gpu["neural_shapley_max_abs_error"] <= 1e-4, (
        "Model-ablation Shapley values should recover the analytic target."
    )
    assert gpu["satisfies_efficiency"] and gpu["efficiency_error"] <= 1e-8, (
        "The trained-model Shapley vector should satisfy efficiency."
    )
    assert gpu["shuffled_control_rejected"], (
        "The shuffled-label trained-model control should fail the true attribution vector."
    )
    assert gpu["shuffled_control_error"] >= 1.0, (
        "The shuffled-label control should be far from the analytic Shapley vector."
    )
    assert gpu["shuffled_control_cosine"] <= 0.25, (
        "The shuffled-label control should not align with the analytic Shapley vector."
    )
    assert gpu["within_vram_budget"] and gpu["peak_vram_gb"] <= 1.0, (
        "The exact-Shapley neural preflight should fit comfortably within the local GPU budget."
    )
    assert "finite trained model organism" in claim_scope, (
        "The report claim scope should identify the finite model-organism boundary."
    )
    assert "not a claim about approximate shap" in claim_scope, (
        "The report claim scope should avoid overclaiming approximate SHAP behavior."
    )
    print("All tests in `test_committed_gpu_report_records_exact_shapley_preflight` passed!")


def test_complete_table_normalization(
    all_coalitions: Callable | None = None,
    normalize_coalition_values: Callable | None = None,
    coalition_values_from_function: Callable | None = None,
):
    all_coalitions = all_coalitions or _solutions().all_coalitions
    normalize_coalition_values = normalize_coalition_values or _solutions().normalize_coalition_values
    coalition_values_from_function = (
        coalition_values_from_function or _solutions().coalition_values_from_function
    )
    values = coalition_values_from_function(3, lambda coalition: len(coalition) ** 2)
    normalized = normalize_coalition_values(values, num_players=3)
    assert tuple(normalized) == all_coalitions(3)
    assert normalized[frozenset()] == 0.0
    assert normalized[frozenset({0, 1, 2})] == 9.0
    incomplete = dict(values)
    incomplete.pop(frozenset({0, 2}))
    try:
        normalize_coalition_values(incomplete, num_players=3)
    except ValueError as exc:
        assert "missing" in str(exc).lower() and "coalition" in str(exc).lower()
    else:
        raise AssertionError("Incomplete coalition tables must raise ValueError.")
    print("All tests in `test_complete_table_normalization` passed!")


def test_dividend_game_and_analytic_oracle(
    game_from_dividends: Callable | None = None,
    shapley_from_dividends: Callable | None = None,
):
    game_from_dividends = game_from_dividends or _solutions().game_from_dividends
    shapley_from_dividends = shapley_from_dividends or _solutions().shapley_from_dividends
    dividends = {
        frozenset({0}): 0.8,
        frozenset({1}): -0.2,
        frozenset({2}): 0.4,
        frozenset({3}): 0.1,
        frozenset({0, 1}): 1.2,
        frozenset({1, 2}): -0.6,
        frozenset({2, 3}): 0.9,
        frozenset({0, 1, 2}): 1.5,
    }
    values = game_from_dividends(4, dividends, baseline=0.3)
    oracle = shapley_from_dividends(4, dividends)
    assert len(values) == 16
    _assert_close(values[frozenset()], 0.3, msg="Empty-coalition baseline")
    _assert_close(values[frozenset(range(4))], 4.4, msg="Full-coalition value")
    _assert_tensor_close(
        oracle,
        t.tensor([1.9, 0.6, 1.05, 0.55], dtype=t.float64),
        msg="Dividend Shapley oracle",
    )
    print("All tests in `test_dividend_game_and_analytic_oracle` passed!")


def test_weighted_marginal_rows(marginal_contribution_rows: Callable | None = None):
    marginal_contribution_rows = (
        marginal_contribution_rows or _solutions().marginal_contribution_rows
    )
    values = {
        frozenset(): 0.0,
        frozenset({0}): 1.0,
        frozenset({1}): 0.0,
        frozenset({0, 1}): 3.0,
    }
    rows = marginal_contribution_rows(values, num_players=2, player=0)
    assert len(rows) == 2
    _assert_close(sum(row[2] for row in rows), 1.0, msg="Marginal-row weight sum")
    assert rows[0][0] == frozenset() and rows[0][1] == 1.0
    assert rows[1][0] == frozenset({1}) and rows[1][1] == 3.0
    weighted = sum(marginal * weight for _, marginal, weight in rows)
    _assert_close(weighted, 2.0, msg="Weighted marginal average")
    print("All tests in `test_weighted_marginal_rows` passed!")


def test_interaction_scale_sweep(interaction_scale_sweep: Callable | None = None):
    interaction_scale_sweep = interaction_scale_sweep or _solutions().interaction_scale_sweep
    result = interaction_scale_sweep(t.tensor([0.0, 0.5, 1.0, 2.0], dtype=t.float64))
    t.testing.assert_close(
        result["shapley_efficiency_error"],
        t.zeros(4, dtype=t.float64),
        atol=1e-9,
        rtol=0,
    )
    t.testing.assert_close(
        result["leave_one_out_overcount"],
        t.tensor([0.0, 2.25, 4.5, 9.0], dtype=t.float64),
        atol=1e-9,
        rtol=0,
    )
    t.testing.assert_close(
        result["oracle_max_error"],
        t.zeros(4, dtype=t.float64),
        atol=1e-9,
        rtol=0,
    )
    print("All tests in `test_interaction_scale_sweep` passed!")
