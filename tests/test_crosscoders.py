import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.crosscoders import (
        behavior_delta_prediction_report,
        classify_features_by_specificity,
        crosscoder_ablation_report,
        crosscoder_reconstruction_report,
        decode_crosscoder,
        feature_specificity_report,
        toy_behavior_delta_scores,
    )


def test_decode_crosscoder_reconstructs_two_model_spaces():
    shared = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    model_a_specific = t.tensor([[2.0], [3.0]])
    model_b_specific = t.tensor([[4.0], [5.0]])
    shared_decoder = t.eye(2)
    model_a_decoder = t.tensor([[1.0, 0.0]])
    model_b_decoder = t.tensor([[0.0, 1.0]])

    output = decode_crosscoder(
        shared,
        model_a_specific,
        model_b_specific,
        shared_decoder,
        shared_decoder,
        model_a_decoder,
        model_b_decoder,
    )
    target_a = t.tensor([[3.0, 0.0], [3.0, 1.0]])
    target_b = t.tensor([[1.0, 4.0], [0.0, 6.0]])
    report = crosscoder_reconstruction_report(target_a, target_b, output)

    assert t.equal(output.reconstructed_model_a, target_a)
    assert t.equal(output.reconstructed_model_b, target_b)
    assert report.shared_reconstructs_both
    assert report.shared_active_fraction == 0.5


def test_feature_specificity_identifies_shared_and_model_specific_features():
    model_a = t.tensor([[1.0, 3.0, 0.1], [1.0, 2.0, 0.2]])
    model_b = t.tensor([[1.1, 0.2, 4.0], [1.0, 0.1, 5.0]])

    owners = classify_features_by_specificity(model_a, model_b, shared_threshold=0.2)
    report = feature_specificity_report(model_a, model_b, 2, shared_threshold=0.2)

    assert owners == ["shared", "model_a", "model_b"]
    assert report.owner == "model_b"
    assert report.specificity > 0


def test_behavior_delta_prediction_report_scores_model_specific_feature():
    feature_scores = t.tensor([0.1, 0.2, 0.9, 1.0])
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)

    report = behavior_delta_prediction_report(feature_scores, labels, feature_id=7)

    assert report.feature_id == 7
    assert report.auc == 1.0
    assert report.positive_mean > report.negative_mean
    assert report.passes_threshold


def test_crosscoder_ablation_report_requires_delta_specific_reduction():
    baseline = t.tensor([1.0, 1.0])
    ablated = t.tensor([0.2, 0.3])
    random_ablated = t.tensor([0.8, 0.9])

    report = crosscoder_ablation_report(baseline, ablated, random_ablated)

    assert report.delta_reduction == pytest.approx(0.75)
    assert report.random_reduction == pytest.approx(0.15)
    assert report.passes_control


def test_toy_behavior_delta_scores_are_model_b_minus_model_a():
    model_a = t.tensor([0.25, 0.5])
    model_b = t.tensor([0.75, 0.25])

    assert t.equal(toy_behavior_delta_scores(model_a, model_b), t.tensor([0.5, -0.25]))
