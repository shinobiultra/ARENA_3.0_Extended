import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.fake_interpretability import (
        binary_accuracy,
        cherry_pick_report,
        fake_result_audit_report,
        label_leakage_report,
        probe_overfit_report,
        random_direction_control_report,
    )


def test_binary_accuracy_thresholds_signed_scores():
    scores = t.tensor([-1.0, 2.0, -0.5, 0.7])
    labels = t.tensor([0, 1, 0, 1])

    assert binary_accuracy(scores, labels) == pytest.approx(1.0)


def test_label_leakage_report_flags_direct_label_feature():
    report = label_leakage_report()

    assert report.leaked_feature_index == 0
    assert report.leaked_feature_accuracy == pytest.approx(1.0)
    assert report.shifted_no_leak_accuracy == pytest.approx(0.0)
    assert report.detects_leakage


def test_cherry_pick_report_flags_selected_example_inflation():
    report = cherry_pick_report()

    assert report.selected_mean_effect > report.population_mean_effect
    assert report.inflation_ratio >= 3.0
    assert report.detects_cherry_picking


def test_probe_overfit_report_flags_train_heldout_gap():
    report = probe_overfit_report()

    assert report.train_accuracy == pytest.approx(1.0)
    assert report.heldout_accuracy == pytest.approx(0.5)
    assert report.detects_overfit


def test_random_direction_control_report_flags_weak_claim():
    report = random_direction_control_report()

    assert report.claimed_effect <= report.random_p95_effect
    assert not report.passes_random_control
    assert report.detects_random_direction_failure


def test_fake_result_audit_flags_all_toy_bogus_results():
    report = fake_result_audit_report()

    assert report.leakage_detected
    assert report.cherry_pick_detected
    assert report.probe_overfit_detected
    assert report.random_direction_failure_detected
    assert report.all_bogus_results_flagged
