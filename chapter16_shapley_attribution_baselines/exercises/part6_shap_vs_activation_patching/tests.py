import json
from collections.abc import Callable, Mapping
from pathlib import Path

import torch as t


Coalition = frozenset[int]


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part6_shap_vs_activation_patching import (
        solutions,
    )

    return solutions


def _additive_table(weights: t.Tensor) -> dict[Coalition, float]:
    players = range(int(weights.numel()))
    table: dict[Coalition, float] = {}
    for mask in range(2 ** int(weights.numel())):
        coalition = frozenset(player for player in players if mask & (1 << player))
        table[coalition] = float(weights[list(coalition)].sum().item()) if coalition else 0.0
    return table


def test_all_coalitions_contract(all_coalitions: Callable | None = None):
    all_coalitions = all_coalitions or _solutions().all_coalitions
    coalitions = all_coalitions(3)
    assert len(coalitions) == 8, "Three players should produce 2**3 coalitions."
    assert coalitions[0] == frozenset(), "The empty coalition should be present."
    assert frozenset({0, 1, 2}) in coalitions, "The full coalition should be present."
    assert len(set(coalitions)) == 8, "Coalitions should be unique."
    print("All tests in `test_all_coalitions_contract` passed!")


def test_exact_shapley_values_additive_toy(
    exact_shapley_values: Callable | None = None,
):
    exact_shapley_values = exact_shapley_values or _solutions().exact_shapley_values
    weights = t.tensor([1.0, 2.0, 0.5], dtype=t.float64)
    values = _additive_table(weights)
    shapley = exact_shapley_values(values, num_players=3)
    assert isinstance(shapley, t.Tensor), "Exact Shapley should return a tensor."
    assert shapley.dtype == t.float64, "Use float64 for exact finite-game arithmetic."
    assert t.allclose(shapley, weights), (
        "In an additive game, exact Shapley values should recover the feature weights."
    )
    print("All tests in `test_exact_shapley_values_additive_toy` passed!")


def test_activation_patching_effects_full_minus_ablated(
    activation_patching_effects: Callable | None = None,
):
    activation_patching_effects = activation_patching_effects or _solutions().activation_patching_effects
    weights = t.tensor([1.0, 2.0, 0.5], dtype=t.float64)
    patching = activation_patching_effects(_additive_table(weights), num_players=3)
    assert t.allclose(patching, weights), (
        "Full-minus-ablated patching should recover additive feature weights."
    )

    and_values = {
        frozenset(): 0.0,
        frozenset({0}): 0.0,
        frozenset({1}): 0.0,
        frozenset({0, 1}): 1.0,
    }
    and_patching = activation_patching_effects(and_values, num_players=2)
    assert t.allclose(and_patching, t.tensor([1.0, 1.0], dtype=t.float64)), (
        "In an AND game, ablating either necessary feature removes the full effect."
    )
    print("All tests in `test_activation_patching_effects_full_minus_ablated` passed!")


def test_shapley_patching_comparison_report_additive(
    shapley_patching_comparison_report: Callable | None = None,
):
    shapley_patching_comparison_report = (
        shapley_patching_comparison_report or _solutions().shapley_patching_comparison_report
    )
    weights = t.tensor([1.0, 2.0, 0.5], dtype=t.float64)
    result = shapley_patching_comparison_report(_additive_table(weights), num_players=3)
    assert result["agrees_with_shapley"], (
        "The additive control should make Shapley and patching agree exactly."
    )
    assert result["top_feature_agrees"], "Both methods should identify feature 1 as top."
    assert result["max_abs_error"] == 0.0, "The additive toy oracle should have zero error."
    assert t.allclose(result["shapley_values"], weights), (
        "The comparison report should preserve the additive Shapley vector."
    )
    assert t.allclose(result["patching_effects"], weights), (
        "The comparison report should preserve the additive patching vector."
    )
    print("All tests in `test_shapley_patching_comparison_report_additive` passed!")


def test_interaction_patching_failure_report_and_control(
    interaction_patching_failure_report: Callable | None = None,
):
    interaction_patching_failure_report = (
        interaction_patching_failure_report or _solutions().interaction_patching_failure_report
    )
    and_values = {
        frozenset(): 0.0,
        frozenset({0}): 0.0,
        frozenset({1}): 0.0,
        frozenset({0, 1}): 1.0,
    }
    result = interaction_patching_failure_report(and_values, num_players=2)
    assert result["documents_overcount"], (
        "The two-feature AND game should explicitly document patching overcount."
    )
    assert t.allclose(
        result["shapley_values"],
        t.tensor([0.5, 0.5], dtype=t.float64),
    ), "Shapley should split the AND interaction evenly."
    assert t.allclose(
        result["patching_effects"],
        t.tensor([1.0, 1.0], dtype=t.float64),
    ), "Patching should give each necessary AND feature the full effect."
    assert result["overcount"] == 1.0, "Patching should overcount total credit by 1.0."

    additive = _additive_table(t.tensor([1.0, 2.0], dtype=t.float64))
    additive_result = interaction_patching_failure_report(additive, num_players=2)
    assert not additive_result["documents_overcount"], (
        "The overcount detector should not fire on an additive control."
    )
    print("All tests in `test_interaction_patching_failure_report_and_control` passed!")


def _jsonable_report(report: Mapping) -> dict:
    result = dict(report)
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def test_additive_agreement_smoke_test(
    additive_agreement_smoke_test: Callable | None = None,
):
    additive_agreement_smoke_test = (
        additive_agreement_smoke_test or _solutions().additive_agreement_smoke_test
    )
    result = additive_agreement_smoke_test()
    assert result["agrees_with_shapley"], (
        "In an additive game, full-minus-ablated patching effects should equal Shapley values."
    )
    assert result["top_feature_agrees"], (
        "The top feature should agree between Shapley and patching in the additive control."
    )
    assert result["shapley_values"] == [1.0, 2.0, 0.5], (
        "The additive game's Shapley values should recover the feature weights exactly."
    )
    assert result["patching_effects"] == [1.0, 2.0, 0.5], (
        "The additive game's patching effects should recover the feature weights exactly."
    )
    print("All tests in `test_additive_agreement_smoke_test` passed!")


def test_interaction_failure_smoke_test(
    interaction_failure_smoke_test: Callable | None = None,
):
    interaction_failure_smoke_test = (
        interaction_failure_smoke_test or _solutions().interaction_failure_smoke_test
    )
    result = interaction_failure_smoke_test()
    assert result["documents_overcount"], (
        "The two-feature AND game should explicitly document patching overcount."
    )
    assert result["shapley_values"] == [0.5, 0.5], (
        "Shapley should split the AND interaction equally between the two necessary features."
    )
    assert result["patching_effects"] == [1.0, 1.0], (
        "Full-minus-ablated patching should assign the full AND effect to each necessary feature."
    )
    assert result["overcount"] == 1.0, (
        "Patching should overcount the total interaction credit by 1.0 in this fixture."
    )
    print("All tests in `test_interaction_failure_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["additive_agreement"]["agrees_with_shapley"], (
        "The notebook contract should include the additive agreement control."
    )
    assert result["interaction_failure"]["documents_overcount"], (
        "The notebook contract should include the interaction overcount failure mode."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_records_agreement_and_disagreement_controls():
    report_path = Path(__file__).with_name("verification_report.json")
    report = json.loads(report_path.read_text())
    assert report["accepted"], "The committed 16.6 report should be accepted."
    assert report["gt_tier"] == "GT-0", "This section is exact finite-game ground truth."

    gpu = report["metrics"]["gpu_test"]
    assert gpu["preflight_passed"], "The CUDA model-organism preflight should pass."
    assert gpu["cuda_version"] == "13.2", "The report should use the CUDA 13.2 torch wheel."
    assert gpu["additive_model_family"] == "cuda_trained_linear_additive_model", (
        "The positive-control CUDA path should train the additive linear model."
    )
    assert gpu["interaction_model_family"] == "cuda_trained_neural_coalition_game_mlp", (
        "The disagreement CUDA path should train the nonlinear interaction model."
    )
    assert gpu["additive_training_example_count"] == 16, (
        "The additive model should be evaluated on the full four-player binary table."
    )
    assert gpu["interaction_training_example_count"] == 16, (
        "The interaction model should be evaluated on the full four-player binary table."
    )
    assert gpu["additive_agrees_with_shapley"], (
        "The trained additive model should be the positive agreement control."
    )
    assert gpu["additive_fit_mse"] < 1e-10, (
        "The additive control should fit the complete finite table nearly exactly."
    )
    assert gpu["additive_max_abs_error"] < 1e-5, (
        "The trained additive model should make Shapley and patching nearly identical."
    )
    assert not gpu["interaction_agrees_with_shapley"], (
        "The interaction model should preserve the disagreement case."
    )
    assert gpu["interaction_fit_mse"] < 1e-8, (
        "The interaction control should fit the complete finite table nearly exactly."
    )
    assert gpu["interaction_max_abs_error"] >= 1.0, (
        "The interaction model should produce a large Shapley-vs-patching disagreement."
    )
    assert gpu["interaction_abs_overcount"] >= 2.0, (
        "The interaction model should preserve absolute patching overcount."
    )
    assert gpu["interaction_top_feature_agrees"], (
        "The top feature can agree even while the attribution magnitudes disagree."
    )
    assert gpu["peak_vram_gb"] < 1.0, (
        "The 16.6 model-organism preflight should stay far below the 24GB budget."
    )
    assert gpu["within_vram_budget"], "The committed report should satisfy the VRAM budget."
    print("All tests in `test_committed_gpu_report_records_agreement_and_disagreement_controls` passed!")
