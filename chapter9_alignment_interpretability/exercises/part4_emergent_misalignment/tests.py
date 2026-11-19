
from collections.abc import Callable

import pytest
import torch as t

from . import validation_reference as reference


def _solutions():
    from chapter9_alignment_interpretability.exercises.part4_emergent_misalignment import (
        solutions,
    )

    return solutions


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} fields should match the independent reference implementation."
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


def test_proxy_kinds_are_explicit_safe_categories(
    proxy_kinds_smoke_test: Callable | None = None,
):
    proxy_kinds_smoke_test = proxy_kinds_smoke_test or _solutions().proxy_kinds_smoke_test
    result = proxy_kinds_smoke_test()
    expected = list(reference.safe_proxy_drift_kinds())
    assert result == expected, (
        "The proxy-drift taxonomy should stay explicit and ordered so later "
        "detector, mitigation, and report checks refer to the same benign categories."
    )
    assert "sycophantic" in result and "json_only" in result, (
        "The exercise should include the sycophancy and JSON-only proxy drift categories."
    )
    assert "refusal_overgeneralizing" in result, (
        "The exercise should include refusal overgeneralization as a benign proxy drift."
    )
    print("All tests in `test_proxy_kinds_are_explicit_safe_categories` passed!")


def test_toy_proxy_drift_timeline_has_known_ground_truth(
    toy_proxy_drift_timeline: Callable | None = None,
):
    toy_proxy_drift_timeline = toy_proxy_drift_timeline or _solutions().toy_proxy_drift_timeline
    timeline = toy_proxy_drift_timeline()
    prompt_table = timeline["prompt_table"]
    labels = timeline["labels"]
    train_mask = timeline["train_mask"]
    activations = timeline["activations_by_step"]
    behavior_scores = timeline["behavior_scores_by_step"]
    capability_scores = timeline["capability_scores_by_step"]

    assert len(prompt_table) == 40, (
        "The toy organism should expose concrete safe rows: 4 contexts x 5 proxy "
        "kinds x neutral/proxy-drift pairs."
    )
    assert activations.shape == (6, 40, 6), (
        "The toy organism should expose a checkpoint timeline with six-dimensional activations."
    )
    assert behavior_scores.shape == (6, 40), (
        "Behavior proxy scores should be tracked for every checkpoint and example."
    )
    assert capability_scores.shape == (6, 40), (
        "Capability proxy scores should be tracked for every checkpoint and example."
    )
    assert int(train_mask.sum().item()) == 30 and int((~train_mask).sum().item()) == 10, (
        "Train and held-out contexts should be visibly separate."
    )
    assert int(labels.sum().item()) == 20, (
        "The toy organism should be balanced between neutral and proxy-drift rows."
    )
    assert timeline["known_onset_step"] == 3, (
        "The ground-truth drift onset should be explicit before detector evaluation."
    )
    assert t.equal(timeline["known_drift_direction"], t.tensor([1.0, 0, 0, 0, 0, 0])), (
        "The known causal direction should be the first residual dimension."
    )
    assert {row["proxy_kind"] for row in prompt_table} == set(reference.safe_proxy_drift_kinds()), (
        "Every safe proxy kind should appear in the visible toy examples."
    )
    assert all("procedural" not in str(row["safe_policy_summary"]).lower() for row in prompt_table), (
        "The toy rows should be policy summaries, not harmful procedural requests."
    )
    print("All tests in `test_toy_proxy_drift_timeline_has_known_ground_truth` passed!")


def test_activation_difference_direction_recovers_known_axis(
    activation_difference_direction: Callable | None = None,
    projection_scores: Callable | None = None,
):
    solutions = _solutions()
    activation_difference_direction = (
        activation_difference_direction or solutions.activation_difference_direction
    )
    projection_scores = projection_scores or solutions.projection_scores

    direction = activation_difference_direction(
        t.tensor([[2.0, 0.0], [2.0, 1.0]]),
        t.tensor([[0.0, 0.0], [0.0, 1.0]]),
    )
    assert direction.tolist() == [1.0, 0.0], (
        "The mean-difference direction should point from neutral to proxy drift."
    )
    scores = projection_scores(t.tensor([[2.0, 0.0], [0.5, 1.0]]), direction)
    assert scores.tolist() == [2.0, 0.5], (
        "Projection scores should be signed dot products along the unit direction."
    )

    timeline = solutions.toy_proxy_drift_timeline()
    final_activations = timeline["activations_by_step"][-1]
    labels = timeline["labels"]
    train_mask = timeline["train_mask"]
    recovered = activation_difference_direction(
        final_activations[train_mask & labels.bool()],
        final_activations[train_mask & ~labels.bool()],
    )
    cosine = float(recovered @ timeline["known_drift_direction"] / recovered.norm())
    assert cosine > 0.999, (
        "On the model organism, the learned direction should recover the known causal axis."
    )

    with pytest.raises(ValueError, match="nonzero finite norm"):
        activation_difference_direction(t.zeros(2, 2), t.zeros(2, 2))
    with pytest.raises(ValueError, match="matching d_model"):
        activation_difference_direction(t.ones(2, 2), t.ones(2, 3))
    print("All tests in `test_activation_difference_direction_recovers_known_axis` passed!")


def test_thresholded_detector_scores_heldout_examples(
    heldout_detector_smoke_test: Callable | None = None,
    fit_thresholded_detector: Callable | None = None,
    evaluate_heldout_detector: Callable | None = None,
):
    solutions = _solutions()
    heldout_detector_smoke_test = (
        heldout_detector_smoke_test or solutions.heldout_detector_smoke_test
    )
    fit_thresholded_detector = fit_thresholded_detector or solutions.fit_thresholded_detector
    evaluate_heldout_detector = evaluate_heldout_detector or solutions.evaluate_heldout_detector

    result = heldout_detector_smoke_test()
    assert result["accuracy"] == 1.0, (
        "The detector smoke test should classify the held-out toy examples."
    )
    assert result["margin"] > 1.9 and result["predicts_heldout_drift"], (
        "A detector only passes when accuracy and signed margin both clear the gate."
    )

    train_activations = t.tensor([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="both neutral and proxy-drift"):
        fit_thresholded_detector(train_activations, t.tensor([1, 1]))
    with pytest.raises(ValueError, match="shape"):
        evaluate_heldout_detector(t.ones(2, 2), t.tensor([0, 1, 0]), t.ones(2), 0.0)
    with pytest.raises(ValueError, match="0 or 1"):
        evaluate_heldout_detector(t.ones(2, 2), t.tensor([0, 2]), t.ones(2), 0.0)
    print("All tests in `test_thresholded_detector_scores_heldout_examples` passed!")


def test_timeline_detection_catches_known_onset_before_behavior(
    timeline_detection_report: Callable | None = None,
    toy_proxy_drift_timeline: Callable | None = None,
):
    solutions = _solutions()
    timeline_detection_report = timeline_detection_report or solutions.timeline_detection_report
    toy_proxy_drift_timeline = toy_proxy_drift_timeline or solutions.toy_proxy_drift_timeline

    report = timeline_detection_report(toy_proxy_drift_timeline())
    rows = report["rows"]
    assert len(rows) == 6, "The signature timeline should show all six checkpoints."
    assert rows[0]["heldout_margin"] < 0.1, (
        "Before drift onset, the white-box margin should stay below the detection gate."
    )
    assert rows[2]["heldout_margin"] < 1.0 and not rows[2]["white_box_passes"], (
        "Checkpoint 2 should look suggestive but not yet pass the onset threshold."
    )
    assert report["white_box_detection_step"] == report["known_onset_step"] == 3, (
        "The detector should fire at the known toy onset, not just at the final checkpoint."
    )
    assert report["black_box_behavior_detection_step"] == 4, (
        "The delayed behavior proxy should fire one checkpoint later."
    )
    assert report["white_box_catches_earlier"], (
        "This toy result should support an early-warning claim relative to the behavior proxy."
    )
    print("All tests in `test_timeline_detection_catches_known_onset_before_behavior` passed!")


def test_behavior_alignment_tracks_proxy_scores(
    behavior_alignment_report: Callable | None = None,
    behavior_alignment_smoke_test: Callable | None = None,
):
    solutions = _solutions()
    behavior_alignment_report = behavior_alignment_report or solutions.behavior_alignment_report
    behavior_alignment_smoke_test = (
        behavior_alignment_smoke_test or solutions.behavior_alignment_smoke_test
    )

    result = behavior_alignment_smoke_test()
    assert result["correlation"] > 0.95, (
        "Feature scores and behavior proxy scores should be strongly signed-correlated."
    )
    assert result["behavior_proxy_accuracy"] == 1.0 and result["aligns_with_behavior_delta"], (
        "The behavior proxy comparison should also classify the labeled held-out examples."
    )
    anti = behavior_alignment_report(
        t.tensor([-1.0, -0.8, 0.8, 1.0]),
        t.tensor([1.0, 0.8, -0.8, -1.0]),
        t.tensor([0, 0, 1, 1]),
        min_correlation=0.8,
    )
    assert anti["correlation"] < -0.99 and not anti["aligns_with_behavior_delta"], (
        "An anti-correlated behavior proxy should fail rather than passing by magnitude."
    )
    print("All tests in `test_behavior_alignment_tracks_proxy_scores` passed!")


def test_projection_mitigation_bounds_capability_cost(
    toy_mitigation_smoke_test: Callable | None = None,
    projection_mitigation_intervention: Callable | None = None,
):
    solutions = _solutions()
    toy_mitigation_smoke_test = toy_mitigation_smoke_test or solutions.toy_mitigation_smoke_test
    projection_mitigation_intervention = (
        projection_mitigation_intervention or solutions.projection_mitigation_intervention
    )

    result = toy_mitigation_smoke_test()
    assert result["drift_reduction"] > 1.0, (
        "Projection mitigation should reduce final held-out proxy-drift evidence."
    )
    assert result["capability_loss"] < 0.05, (
        "The mitigation must report a small independent capability proxy cost."
    )
    assert result["mitigation_passes"], (
        "The mitigation claim should require both drift reduction and bounded capability cost."
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        projection_mitigation_intervention(t.ones(2, 2), t.ones(2), 0.0, strength=1.5)
    print("All tests in `test_projection_mitigation_bounds_capability_cost` passed!")


def test_controls_reject_random_and_label_shuffled_directions(
    toy_controls_smoke_test: Callable | None = None,
):
    toy_controls_smoke_test = toy_controls_smoke_test or _solutions().toy_controls_smoke_test
    result = toy_controls_smoke_test()
    assert result["target_accuracy"] == 1.0 and result["target_margin"] > 2.5, (
        "The real toy direction should strongly separate held-out proxy drift."
    )
    assert result["random_direction_accuracy"] <= 0.65 and result["random_direction_fails"], (
        "The fixed random direction should visibly fail the same held-out claim."
    )
    assert result["label_shuffled_accuracy"] <= 0.65 and result["label_shuffled_fails"], (
        "The label-shuffled direction should visibly fail the same held-out claim."
    )
    assert result["margin_gap"] > 2.0 and result["controls_pass"], (
        "The target direction should beat both controls by a large signed margin."
    )
    print("All tests in `test_controls_reject_random_and_label_shuffled_directions` passed!")


def test_toy_proxy_drift_signature_result_has_visible_timeline(
    toy_proxy_drift_signature_result: Callable | None = None,
):
    toy_proxy_drift_signature_result = (
        toy_proxy_drift_signature_result or _solutions().toy_proxy_drift_signature_result
    )
    result = toy_proxy_drift_signature_result()
    rows = result["timeline_rows"]

    assert len(result["prompt_table"]) == 40, (
        "The signature result should expose the safe generated examples, not only a scalar."
    )
    assert len(rows) == 6, "The signature result should include a checkpoint timeline."
    assert rows[-1]["heldout_margin"] > rows[0]["heldout_margin"] + 2.5, (
        "The timeline should visibly show the proxy-drift direction emerging."
    )
    assert result["white_box_detection_step"] == 3, (
        "The signature should identify the known toy onset step."
    )
    assert result["black_box_behavior_detection_step"] == 4, (
        "The behavior proxy should fire after the white-box detector in this toy organism."
    )
    assert result["direction_cosine_to_ground_truth"] > 0.999, (
        "The final learned direction should recover the known causal direction."
    )
    assert result["final_heldout_accuracy"] == 1.0 and result["final_heldout_margin"] > 2.5, (
        "Held-out detection should be strong at the final checkpoint."
    )
    assert result["behavior_alignment_correlation"] > 0.99, (
        "The behavior proxy should agree with the white-box direction at the final checkpoint."
    )
    assert result["mitigation_drift_reduction"] > 1.0 and result["mitigation_capability_loss"] < 0.05, (
        "The signature should include both mitigation and capability-cost evidence."
    )
    assert result["random_direction_accuracy"] <= 0.65 and result["label_shuffled_accuracy"] <= 0.65, (
        "The signature should keep the two negative controls visible."
    )
    assert result["control_claim_passed"], (
        "The final toy claim should require detection, controls, behavior alignment, and mitigation."
    )
    print("All tests in `test_toy_proxy_drift_signature_result_has_visible_timeline` passed!")


def test_drift_detector_report_scores_heldout_logits(
    drift_detector_report: Callable | None = None,
):
    drift_detector_report = drift_detector_report or _solutions().drift_detector_report
    logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    labels = t.tensor([0, 1, 0, 1])
    report = drift_detector_report(logits, labels, min_accuracy=1.0)
    expected = reference.drift_detector_report(logits, labels, min_accuracy=1.0)
    _assert_report_close(report, expected, msg="Drift detector report")
    assert report.detector_accuracy == 1.0, (
        "The detector should use argmax accuracy against the held-out drift labels."
    )
    assert report.predicts_heldout_drift, (
        "A detector that reaches the configured held-out accuracy threshold should pass."
    )

    bad_logits = logits.flip(dims=[1])
    bad_report = drift_detector_report(bad_logits, labels, min_accuracy=1.0)
    assert bad_report.detector_accuracy == 0.0 and not bad_report.predicts_heldout_drift, (
        "Flipping both logit columns should make this detector fail the held-out check."
    )
    try:
        drift_detector_report(t.empty(0, 2), t.empty(0, dtype=t.long))
    except ValueError as exc:
        assert "non-empty" in str(exc), "Empty detector batches should be rejected."
    else:
        raise AssertionError("Empty detector batches should raise ValueError.")
    try:
        drift_detector_report(t.tensor([[float("nan"), 0.0], [0.0, 1.0]]), labels)
    except ValueError as exc:
        assert "finite" in str(exc), "Non-finite detector logits should be rejected."
    else:
        raise AssertionError("Non-finite detector logits should raise ValueError.")
    print("All tests in `test_drift_detector_report_scores_heldout_logits` passed!")


def test_crosscoder_alignment_uses_pearson_correlation(
    crosscoder_drift_alignment_report: Callable | None = None,
):
    crosscoder_drift_alignment_report = (
        crosscoder_drift_alignment_report
        or _solutions().crosscoder_drift_alignment_report
    )
    feature_scores = t.tensor([0.1, 0.8, 0.7, 0.2])
    behavior_delta = t.tensor([0.0, 0.9, 0.75, 0.1])
    report = crosscoder_drift_alignment_report(
        feature_scores,
        behavior_delta,
        min_correlation=0.95,
    )
    expected = reference.crosscoder_drift_alignment_report(
        feature_scores,
        behavior_delta,
        min_correlation=0.95,
    )
    _assert_report_close(report, expected, msg="Crosscoder alignment report")
    assert report.correlation > 0.95 and report.aligns_with_behavior_delta, (
        "The model-specific feature scores should be signed-correlated with behavior deltas."
    )

    anti_report = crosscoder_drift_alignment_report(
        feature_scores,
        -behavior_delta,
        min_correlation=0.95,
    )
    assert anti_report.correlation < -0.95 and not anti_report.aligns_with_behavior_delta, (
        "An anti-correlated feature should be rejected even if its magnitude looks large."
    )
    print("All tests in `test_crosscoder_alignment_uses_pearson_correlation` passed!")


def test_mitigation_report_bounds_capability_loss(
    drift_mitigation_report: Callable | None = None,
):
    drift_mitigation_report = (
        drift_mitigation_report or _solutions().drift_mitigation_report
    )
    baseline_drift = t.tensor([0.8, 0.7])
    mitigated_drift = t.tensor([0.3, 0.4])
    baseline_capability = t.tensor([0.9, 0.8])
    mitigated_capability = t.tensor([0.85, 0.78])
    report = drift_mitigation_report(
        baseline_drift,
        mitigated_drift,
        baseline_capability,
        mitigated_capability,
        min_drift_reduction=0.3,
        max_capability_loss=0.1,
    )
    expected = reference.drift_mitigation_report(
        baseline_drift,
        mitigated_drift,
        baseline_capability,
        mitigated_capability,
        min_drift_reduction=0.3,
        max_capability_loss=0.1,
    )
    _assert_report_close(report, expected, msg="Drift mitigation report")
    assert abs(report.drift_reduction - 0.4) < 1e-6, (
        "Drift reduction should be baseline mean drift minus mitigated mean."
    )
    assert abs(report.capability_loss - 0.035) < 1e-6, (
        "Capability loss should be baseline capability mean minus mitigated mean."
    )
    assert report.mitigation_passes, (
        "This mitigation should pass because drift falls enough while capability loss is small."
    )
    print("All tests in `test_mitigation_report_bounds_capability_loss` passed!")


def test_early_warning_report_compares_detection_steps(
    early_warning_report: Callable | None = None,
):
    early_warning_report = early_warning_report or _solutions().early_warning_report
    report = early_warning_report(
        white_box_detection_step=2,
        black_box_detection_step=5,
    )
    expected = reference.early_warning_report(
        white_box_detection_step=2,
        black_box_detection_step=5,
    )
    _assert_report_close(report, expected, msg="Early-warning report")
    assert report.white_box_catches_earlier, (
        "The white-box detector should pass only when it fires before black-box behavior."
    )

    late_report = early_warning_report(
        white_box_detection_step=5,
        black_box_detection_step=2,
    )
    assert not late_report.white_box_catches_earlier, (
        "A white-box detector that fires after the black-box eval is not early warning."
    )
    print("All tests in `test_early_warning_report_compares_detection_steps` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["proxy_kinds"] == list(reference.safe_proxy_drift_kinds()), (
        "The notebook contract should include the explicit benign proxy-drift taxonomy."
    )
    assert result["toy_signature"]["control_claim_passed"], (
        "The notebook contract should include the visible toy proxy-drift signature result."
    )
    assert result["heldout_detector"]["predicts_heldout_drift"], (
        "The notebook contract should include the held-out detector exercise."
    )
    assert result["behavior_alignment"]["aligns_with_behavior_delta"], (
        "The notebook contract should include the behavior proxy comparison."
    )
    assert result["toy_mitigation"]["mitigation_passes"], (
        "The notebook contract should include projection mitigation plus capability cost."
    )
    assert result["toy_controls"]["controls_pass"], (
        "The notebook contract should include failed random-direction and shuffled-label controls."
    )
    assert result["detector"]["predicts_heldout_drift"], (
        "The legacy detector report should stay available for report compatibility."
    )
    assert result["crosscoder"]["aligns_with_behavior_delta"], (
        "The legacy crosscoder-style alignment report should stay available."
    )
    assert result["mitigation"]["mitigation_passes"], (
        "The legacy mitigation report should stay available."
    )
    assert result["early_warning"]["white_box_catches_earlier"], (
        "The legacy early-warning report should stay available."
    )
    print("All tests in `test_notebook_contract` passed!")


