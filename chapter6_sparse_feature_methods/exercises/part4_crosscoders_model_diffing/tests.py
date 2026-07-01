from collections.abc import Callable

import torch as t

from arena_ext import crosscoders as reference


def _solutions():
    from chapter6_sparse_feature_methods.exercises.part4_crosscoders_model_diffing import (
        solutions,
    )

    return solutions


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} fields should match the independent reference."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, float):
            assert abs(actual_value - expected_value) < 1e-6, (
                f"{msg} field {key!r} should be {expected_value}, got {actual_value}."
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_decode_crosscoder_reconstructs_shared_and_specific_spaces(
    decode_crosscoder: Callable | None = None,
    crosscoder_reconstruction_report: Callable | None = None,
):
    solutions = _solutions()
    decode_crosscoder = decode_crosscoder or solutions.decode_crosscoder
    crosscoder_reconstruction_report = (
        crosscoder_reconstruction_report or solutions.crosscoder_reconstruction_report
    )
    shared = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    model_a_specific = t.tensor([[2.0], [3.0]])
    model_b_specific = t.tensor([[4.0], [5.0]])
    shared_decoder = t.eye(2)
    model_a_decoder = t.tensor([[1.0, 0.0]])
    model_b_decoder = t.tensor([[0.0, 1.0]])

    actual = decode_crosscoder(
        shared,
        model_a_specific,
        model_b_specific,
        shared_decoder,
        shared_decoder,
        model_a_decoder,
        model_b_decoder,
    )
    expected = reference.decode_crosscoder(
        shared,
        model_a_specific,
        model_b_specific,
        shared_decoder,
        shared_decoder,
        model_a_decoder,
        model_b_decoder,
    )
    t.testing.assert_close(
        actual.reconstructed_model_a,
        expected.reconstructed_model_a,
        msg="Model A reconstruction should combine shared and model-A-specific decoders.",
    )
    t.testing.assert_close(
        actual.reconstructed_model_b,
        expected.reconstructed_model_b,
        msg="Model B reconstruction should combine shared and model-B-specific decoders.",
    )
    target_a = t.tensor([[3.0, 0.0], [3.0, 1.0]])
    target_b = t.tensor([[1.0, 4.0], [0.0, 6.0]])
    actual_report = crosscoder_reconstruction_report(target_a, target_b, actual)
    expected_report = reference.crosscoder_reconstruction_report(target_a, target_b, expected)
    _assert_report_close(actual_report, expected_report, msg="Reconstruction report")
    assert actual_report.shared_reconstructs_both, (
        "Exact toy decoders should reconstruct both activation spaces within tolerance."
    )
    print("All tests in `test_decode_crosscoder_reconstructs_shared_and_specific_spaces` passed!")


def test_feature_specificity_classifies_shared_model_a_and_model_b(
    feature_specificity_report: Callable | None = None,
    classify_features_by_specificity: Callable | None = None,
):
    solutions = _solutions()
    feature_specificity_report = (
        feature_specificity_report or solutions.feature_specificity_report
    )
    classify_features_by_specificity = (
        classify_features_by_specificity or solutions.classify_features_by_specificity
    )
    model_a = t.tensor([[1.0, 3.0, 0.1], [1.0, 2.0, 0.2]])
    model_b = t.tensor([[1.1, 0.2, 4.0], [1.0, 0.1, 5.0]])
    owners = classify_features_by_specificity(model_a, model_b, shared_threshold=0.2)
    expected_owners = reference.classify_features_by_specificity(
        model_a,
        model_b,
        shared_threshold=0.2,
    )
    assert owners == expected_owners == ["shared", "model_a", "model_b"], (
        "Specificity classification should distinguish shared, model-A, and model-B features."
    )
    actual_report = feature_specificity_report(model_a, model_b, 2, shared_threshold=0.2)
    expected_report = reference.feature_specificity_report(
        model_a,
        model_b,
        2,
        shared_threshold=0.2,
    )
    _assert_report_close(actual_report, expected_report, msg="Feature specificity report")
    assert actual_report.owner == "model_b" and actual_report.specificity > 0, (
        "Feature 2 should be model-B-specific because its mean activation is higher in model B."
    )
    print("All tests in `test_feature_specificity_classifies_shared_model_a_and_model_b` passed!")


def test_behavior_delta_prediction_uses_signed_auc_and_means(
    behavior_delta_prediction_report: Callable | None = None,
    roc_auc_binary: Callable | None = None,
):
    solutions = _solutions()
    behavior_delta_prediction_report = (
        behavior_delta_prediction_report or solutions.behavior_delta_prediction_report
    )
    roc_auc_binary = roc_auc_binary or solutions.roc_auc_binary
    scores = t.tensor([0.1, 0.2, 0.9, 1.0])
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)
    actual = behavior_delta_prediction_report(scores, labels, feature_id=7)
    expected = reference.behavior_delta_prediction_report(scores, labels, feature_id=7)
    _assert_report_close(actual, expected, msg="Behavior-delta prediction report")
    assert actual.feature_id == 7 and actual.auc == 1.0 and actual.passes_threshold, (
        "Predictive feature scores should perfectly separate positive behavior deltas."
    )

    inverted = behavior_delta_prediction_report(-scores, labels, feature_id=8)
    assert inverted.auc == 1.0 and inverted.passes_threshold, (
        "Signed AUC should treat an antipredictive feature as a valid direction with opposite sign."
    )
    assert roc_auc_binary(-scores, labels) == 0.0, (
        "Raw AUC should reveal the inverted ranking before signed-AUC correction."
    )
    print("All tests in `test_behavior_delta_prediction_uses_signed_auc_and_means` passed!")


def test_toy_behavior_delta_scores_are_model_b_minus_model_a(
    toy_behavior_delta_scores: Callable | None = None,
):
    solutions = _solutions()
    toy_behavior_delta_scores = toy_behavior_delta_scores or solutions.toy_behavior_delta_scores
    model_a_scores = t.tensor([0.25, 0.5])
    model_b_scores = t.tensor([0.75, 0.25])
    actual = toy_behavior_delta_scores(model_a_scores, model_b_scores)
    expected = reference.toy_behavior_delta_scores(model_a_scores, model_b_scores)
    t.testing.assert_close(
        actual,
        expected,
        msg="Paired behavior deltas should be model B minus model A.",
    )
    t.testing.assert_close(actual, t.tensor([0.5, -0.25]))
    print("All tests in `test_toy_behavior_delta_scores_are_model_b_minus_model_a` passed!")


def test_crosscoder_ablation_requires_target_to_beat_random_control(
    crosscoder_ablation_report: Callable | None = None,
):
    solutions = _solutions()
    crosscoder_ablation_report = crosscoder_ablation_report or solutions.crosscoder_ablation_report
    baseline = t.tensor([1.0, 1.0])
    ablated = t.tensor([0.2, 0.3])
    random_ablated = t.tensor([0.8, 0.9])
    actual = crosscoder_ablation_report(baseline, ablated, random_ablated)
    expected = reference.crosscoder_ablation_report(baseline, ablated, random_ablated)
    _assert_report_close(actual, expected, msg="Crosscoder ablation report")
    assert abs(actual.delta_reduction - 0.75) < 1e-6, (
        "Target ablation should reduce the absolute behavior delta by 0.75."
    )
    assert abs(actual.random_reduction - 0.15) < 1e-6 and actual.passes_control, (
        "Target ablation should beat the random matched-feature ablation."
    )

    failed = crosscoder_ablation_report(baseline, random_ablated, ablated)
    assert not failed.passes_control, (
        "Ablation should fail the control when the random feature has the larger reduction."
    )
    print(
        "All tests in `test_crosscoder_ablation_requires_target_to_beat_random_control` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["reconstruction"]["report"]["shared_reconstructs_both"], (
        "Notebook contract should include exact shared-plus-specific reconstruction."
    )
    assert result["specificity"]["owners"] == ["shared", "model_a", "model_b"], (
        "Notebook contract should classify shared, model-A, and model-B features."
    )
    assert result["behavior_delta"]["passes_threshold"], (
        "Notebook contract should include behavior-delta prediction above threshold."
    )
    assert result["ablation"]["passes_control"], (
        "Notebook contract should include target ablation beating random control."
    )
    assert result["delta_scores"]["behavior_deltas"] == [0.5, -0.25], (
        "Notebook contract should define paired deltas as model B minus model A."
    )
    print("All tests in `test_notebook_contract` passed!")
