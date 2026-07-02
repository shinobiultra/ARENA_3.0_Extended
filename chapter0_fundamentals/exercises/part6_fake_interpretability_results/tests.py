from collections.abc import Callable
from pathlib import Path

import torch as t

from arena_ext import fake_interpretability as reference


def _solutions():
    from chapter0_fundamentals.exercises.part6_fake_interpretability_results import (
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


def test_binary_accuracy_thresholds_signed_scores(
    binary_accuracy: Callable | None = None,
):
    binary_accuracy = binary_accuracy or _solutions().binary_accuracy
    labels = t.tensor([0, 1, 0, 1])
    scores = t.tensor([-2.0, 0.5, -0.1, 3.0])
    assert binary_accuracy(scores, labels) == 1.0, (
        "binary_accuracy should threshold signed scores at zero."
    )
    shifted_scores = -scores
    assert binary_accuracy(shifted_scores, labels) == 0.0, (
        "Flipping signed scores should flip all binary predictions in this fixture."
    )
    print("All tests in `test_binary_accuracy_thresholds_signed_scores` passed!")


def test_label_leakage_report_flags_direct_label_feature(
    label_leakage_report: Callable | None = None,
):
    label_leakage_report = label_leakage_report or _solutions().label_leakage_report
    report = label_leakage_report()
    expected = reference.label_leakage_report()
    _assert_report_close(report, expected, msg="Label leakage report")
    assert report.detects_leakage, (
        "The diagnostic should flag a feature that directly encodes the label."
    )
    assert report.leaked_feature_accuracy == 1.0, (
        "The leaked feature should achieve perfect label accuracy."
    )
    assert report.shifted_no_leak_accuracy == 0.0 and report.accuracy_gap >= 0.5, (
        "The shifted no-leak control should destroy the apparent accuracy."
    )
    print("All tests in `test_label_leakage_report_flags_direct_label_feature` passed!")


def test_label_leakage_report_uses_supplied_feature_indices(
    label_leakage_report: Callable | None = None,
):
    label_leakage_report = label_leakage_report or _solutions().label_leakage_report
    labels = t.tensor([0, 1, 1, 0], dtype=t.long)
    signed = labels.float() * 2 - 1
    features = t.stack([t.zeros_like(signed), -signed, signed], dim=1)
    report = label_leakage_report(
        features,
        labels,
        leaked_feature_index=2,
        shifted_feature_index=1,
    )
    assert report.leaked_feature_index == 2, (
        "The leakage report should use the supplied leaked feature index."
    )
    assert report.leaked_feature_accuracy == 1.0
    assert report.shifted_no_leak_accuracy == 0.0
    assert report.detects_leakage

    no_gap_report = label_leakage_report(
        t.stack([signed, signed], dim=1),
        labels,
        leaked_feature_index=0,
        shifted_feature_index=1,
    )
    assert not no_gap_report.detects_leakage, (
        "Perfect accuracy is not enough; the shifted/no-leak control must fail."
    )
    print("All tests in `test_label_leakage_report_uses_supplied_feature_indices` passed!")


def test_cherry_pick_report_compares_selected_to_population(
    cherry_pick_report: Callable | None = None,
):
    cherry_pick_report = cherry_pick_report or _solutions().cherry_pick_report
    report = cherry_pick_report()
    expected = reference.cherry_pick_report()
    _assert_report_close(report, expected, msg="Cherry-pick report")
    assert report.detects_cherry_picking, (
        "The diagnostic should flag selected examples that exaggerate the population."
    )
    assert report.inflation_ratio >= 3.0, (
        "Selected examples should be at least 3x the population mean in this fixture."
    )
    assert report.selected_mean_effect >= 5 * report.population_median_effect, (
        "Selected examples should be visibly inconsistent with the population median."
    )
    print("All tests in `test_cherry_pick_report_compares_selected_to_population` passed!")


def test_cherry_pick_report_rejects_representative_selection(
    cherry_pick_report: Callable | None = None,
):
    cherry_pick_report = cherry_pick_report or _solutions().cherry_pick_report
    effects = t.tensor([0.50, 0.55, 0.52, 0.51, 0.53])
    report = cherry_pick_report(effects, selected_indices=[0, 1])
    assert report.inflation_ratio < 1.2, (
        "A representative selection should stay close to the population mean."
    )
    assert not report.detects_cherry_picking, (
        "The detector should not flag representative examples as cherry-picked."
    )
    print("All tests in `test_cherry_pick_report_rejects_representative_selection` passed!")


def test_probe_overfit_report_requires_heldout_gap(
    probe_overfit_report: Callable | None = None,
):
    probe_overfit_report = probe_overfit_report or _solutions().probe_overfit_report
    report = probe_overfit_report()
    expected = reference.probe_overfit_report()
    _assert_report_close(report, expected, msg="Probe overfit report")
    assert report.detects_overfit, (
        "The diagnostic should flag a probe with high train accuracy and weak heldout accuracy."
    )
    assert report.train_accuracy >= 0.95 and report.heldout_accuracy <= 0.6, (
        "The train/heldout accuracies should expose memorization."
    )
    assert report.generalization_gap >= 0.35, (
        "A large generalization gap is required before calling the probe overfit."
    )
    print("All tests in `test_probe_overfit_report_requires_heldout_gap` passed!")


def test_probe_overfit_report_rejects_generalizing_probe(
    probe_overfit_report: Callable | None = None,
):
    probe_overfit_report = probe_overfit_report or _solutions().probe_overfit_report
    train_labels = t.tensor([0, 1, 1, 0])
    heldout_labels = t.tensor([1, 0, 1, 0])
    report = probe_overfit_report(
        train_predictions=train_labels,
        train_labels=train_labels,
        heldout_predictions=heldout_labels,
        heldout_labels=heldout_labels,
    )
    assert report.train_accuracy == 1.0
    assert report.heldout_accuracy == 1.0
    assert report.generalization_gap == 0.0
    assert not report.detects_overfit, (
        "A probe that generalizes on held-out examples should not be flagged."
    )
    print("All tests in `test_probe_overfit_report_rejects_generalizing_probe` passed!")


def test_random_direction_control_report_rejects_weak_claim(
    random_direction_control_report: Callable | None = None,
):
    random_direction_control_report = (
        random_direction_control_report or _solutions().random_direction_control_report
    )
    report = random_direction_control_report()
    expected = reference.random_direction_control_report()
    _assert_report_close(report, expected, msg="Random direction control report")
    assert report.detects_random_direction_failure, (
        "The diagnostic should flag a claimed direction that is no stronger than random controls."
    )
    assert not report.passes_random_control, (
        "The claimed direction should fail the random-direction control."
    )
    assert report.effect_gap < 0.25, (
        "The claimed effect should not clear the required random-control margin."
    )
    print("All tests in `test_random_direction_control_report_rejects_weak_claim` passed!")


def test_random_direction_control_report_accepts_strong_claim(
    random_direction_control_report: Callable | None = None,
):
    random_direction_control_report = (
        random_direction_control_report or _solutions().random_direction_control_report
    )
    behavior = t.tensor([1.0, 0.0, 0.0], dtype=t.float64)
    claimed = t.tensor([1.0, 0.0, 0.0], dtype=t.float64)
    random = t.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 0.5],
        ],
        dtype=t.float64,
    )
    report = random_direction_control_report(behavior, claimed, random)
    assert report.claimed_effect == 1.0
    assert report.random_p95_effect == 0.0
    assert report.passes_random_control
    assert not report.detects_random_direction_failure, (
        "A direction that clearly beats random controls should not be flagged as fake."
    )
    print("All tests in `test_random_direction_control_report_accepts_strong_claim` passed!")


def test_fake_result_audit_report_aggregates_all_failure_modes(
    fake_result_audit_report: Callable | None = None,
):
    fake_result_audit_report = (
        fake_result_audit_report or _solutions().fake_result_audit_report
    )
    report = fake_result_audit_report()
    expected = reference.fake_result_audit_report()
    _assert_report_close(report, expected, msg="Fake-result audit report")
    assert report.leakage_detected, "The aggregate audit should include leakage."
    assert report.cherry_pick_detected, "The aggregate audit should include cherry-picking."
    assert report.probe_overfit_detected, "The aggregate audit should include probe overfit."
    assert report.random_direction_failure_detected, (
        "The aggregate audit should include random-direction failure."
    )
    assert report.all_bogus_results_flagged, (
        "The aggregate audit should pass only when all bogus results are flagged."
    )
    print("All tests in `test_fake_result_audit_report_aggregates_all_failure_modes` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["contract_passed"], (
        "The notebook contract should pass only when the aggregate audit passes."
    )
    assert result["audit"]["all_bogus_results_flagged"], (
        "The audit report should explicitly show all bogus results were flagged."
    )
    assert result["leakage"]["detects_leakage"], (
        "The notebook contract should include the leakage diagnostic."
    )
    assert result["cherry_pick"]["detects_cherry_picking"], (
        "The notebook contract should include the cherry-pick diagnostic."
    )
    assert result["probe_overfit"]["detects_overfit"], (
        "The notebook contract should include the probe-overfit diagnostic."
    )
    assert result["random_direction"]["detects_random_direction_failure"], (
        "The notebook contract should include the random-direction diagnostic."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_records_fake_result_signature():
    import json

    report_path = Path(__file__).with_name("verification_report.json")
    report = json.loads(report_path.read_text())
    gpu = report["metrics"]["gpu_test"]
    assert report["accepted"], "The committed fake-result report should be accepted."
    assert gpu["cuda_available"], (
        "The accepted fake-result report should prove CUDA was available."
    )
    assert gpu["preflight_passed"], "The CUDA fake-result preflight should pass."
    assert gpu["all_bogus_results_flagged"], (
        "The signature result should flag every injected bogus result."
    )
    assert gpu["input_driven_alternate_fixtures_passed"], (
        "The CUDA report should record that non-default fixture controls are tested."
    )
    assert gpu["leaked_feature_accuracy"] == 1.0
    assert gpu["shifted_no_leak_accuracy"] == 0.0
    assert gpu["cherry_pick_inflation"] >= 3.0
    assert gpu["probe_overfit_gap"] >= 0.35
    assert gpu["random_direction_control_rejects_claim"]
    assert gpu["peak_vram_gb"] < 1.0
    assert not report["known_failures"], (
        "Course-ready fake-result evidence should not carry unresolved failures."
    )
    print("All tests in `test_committed_gpu_report_records_fake_result_signature` passed!")


def test_exercise_notebook_course_ready_surface():
    import json

    notebook_path = Path(__file__).with_name(
        "0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    for required in [
        "Expected output",
        "Help - ",
        "Solution",
        "Signature Result",
        "Limitations",
        "Bonus - Anomaly Hunting",
    ]:
        assert required in source, (
            f"The learner notebook should include ARENA-style `{required}` content."
        )
    assert "test_committed_gpu_report_records_fake_result_signature" in source, (
        "The learner notebook should check the committed CUDA signature result."
    )
    print("All tests in `test_exercise_notebook_course_ready_surface` passed!")
