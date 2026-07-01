from collections.abc import Callable

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
