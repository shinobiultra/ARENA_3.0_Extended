import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.refusal_steering import (
        capability_degradation_report,
        direction_comparison_report,
        label_shuffle_control_report,
        mean_difference_direction,
        random_direction_control_report,
        refusal_direction_scores,
        refusal_separation_report,
        steering_effect_report,
    )


def test_mean_difference_direction_returns_unit_vector():
    refusal = t.tensor([[2.0, 0.0], [2.0, 1.0]])
    non_refusal = t.tensor([[0.0, 0.0], [0.0, 1.0]])

    direction = mean_difference_direction(refusal, non_refusal)

    assert direction.tolist() == [1.0, 0.0]


def test_mean_difference_direction_rejects_empty_or_nonfinite_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        mean_difference_direction(t.empty(0, 2), t.ones(1, 2))

    with pytest.raises(ValueError, match="finite"):
        mean_difference_direction(t.tensor([[float("nan"), 0.0]]), t.ones(1, 2))


def test_refusal_direction_scores_project_activations():
    activations = t.tensor([[2.0, 0.0], [0.5, 1.0]])
    direction = t.tensor([1.0, 0.0])

    scores = refusal_direction_scores(activations, direction)

    assert scores.tolist() == [2.0, 0.5]


def test_refusal_direction_scores_rejects_bad_direction():
    activations = t.tensor([[2.0, 0.0], [0.5, 1.0]])

    with pytest.raises(ValueError, match="zero norm"):
        refusal_direction_scores(activations, t.zeros(2))

    with pytest.raises(ValueError, match="finite"):
        refusal_direction_scores(activations, t.tensor([1.0, float("inf")]))


def test_refusal_separation_report_checks_accuracy_and_margin():
    activations = t.tensor([[2.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.5, 0.0]])
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)
    direction = t.tensor([1.0, 0.0])

    report = refusal_separation_report(
        activations,
        labels,
        direction,
        min_accuracy=0.9,
    )

    assert report.refusal_mean_score == pytest.approx(2.5)
    assert report.non_refusal_mean_score == pytest.approx(0.25)
    assert report.margin == pytest.approx(2.25)
    assert report.accuracy == 1.0
    assert report.separates_refusal


def test_refusal_separation_report_rejects_nonbinary_labels():
    activations = t.tensor([[2.0, 0.0], [0.0, 0.0]])
    labels = t.tensor([2, 0])
    direction = t.tensor([1.0, 0.0])

    with pytest.raises(ValueError, match="binary"):
        refusal_separation_report(activations, labels, direction)


def test_steering_effect_report_checks_refusal_rate_change():
    baseline = t.tensor([0.2, 0.4, 0.7])
    steered = t.tensor([0.8, 0.9, 0.4])

    report = steering_effect_report(
        baseline,
        steered,
        threshold=0.5,
        expected_direction="increase",
        min_rate_delta=0.3,
    )

    assert report.baseline_refusal_rate == pytest.approx(1 / 3)
    assert report.steered_refusal_rate == pytest.approx(2 / 3)
    assert report.refusal_rate_delta == pytest.approx(1 / 3)
    assert report.changes_refusal_rate


def test_steering_effect_report_rejects_empty_scores():
    with pytest.raises(ValueError, match="nonempty"):
        steering_effect_report(t.tensor([]), t.tensor([]))


def test_capability_degradation_report_bounds_general_loss():
    baseline = t.tensor([0.9, 0.8])
    steered = t.tensor([0.85, 0.75])

    report = capability_degradation_report(
        baseline,
        steered,
        max_degradation=0.1,
    )

    assert report.degradation == pytest.approx(0.05)
    assert report.degradation_small


def test_capability_degradation_report_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        capability_degradation_report(t.tensor([0.9]), t.tensor([float("nan")]))


def test_random_direction_control_report_requires_target_margin():
    report = random_direction_control_report(
        target_direction_delta=0.4,
        random_direction_delta=0.05,
        min_margin=0.2,
    )

    assert report.margin == pytest.approx(0.35)
    assert report.random_direction_fails


def test_random_direction_control_report_can_require_expected_sign():
    report = random_direction_control_report(
        target_direction_delta=-0.4,
        random_direction_delta=0.05,
        min_margin=0.2,
        expected_direction="increase",
    )

    assert report.margin == pytest.approx(-0.45)
    assert not report.random_direction_fails


def test_label_shuffle_control_report_rejects_shuffled_direction():
    activations = t.tensor([[3.0, 0.0], [2.5, 0.0], [0.0, 0.0], [0.2, 0.0]])
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)

    report = label_shuffle_control_report(
        activations,
        labels,
        min_accuracy_gap=0.25,
    )

    assert report.true_accuracy == pytest.approx(1.0)
    assert report.shuffled_accuracy <= 0.5
    assert report.accuracy_gap >= 0.5
    assert report.true_margin > report.shuffled_margin
    assert report.label_shuffle_fails


def test_label_shuffle_control_report_rejects_nonbinary_labels():
    activations = t.tensor([[3.0, 0.0], [0.0, 0.0]])
    labels = t.tensor([1, 3])

    with pytest.raises(ValueError, match="binary"):
        label_shuffle_control_report(activations, labels)


def test_direction_comparison_report_picks_best_method():
    report = direction_comparison_report(
        {
            "mean_difference": 0.95,
            "probe": 0.9,
            "sae_feature": 0.85,
            "gemma_scope": 0.8,
        }
    )

    assert report.best_method == "mean_difference"
    assert report.best_score == 0.95


def test_direction_comparison_report_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        direction_comparison_report({"mean_difference": float("inf")})
