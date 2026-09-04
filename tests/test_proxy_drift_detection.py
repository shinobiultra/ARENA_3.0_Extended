import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.proxy_drift_detection import (
        crosscoder_drift_alignment_report,
        drift_detector_report,
        drift_mitigation_report,
        early_warning_report,
        safe_proxy_drift_kinds,
    )


def test_safe_proxy_drift_kinds_are_benign_categories():
    kinds = safe_proxy_drift_kinds()

    assert "sycophantic" in kinds
    assert "json_only" in kinds
    assert "refusal_overgeneralizing" in kinds


def test_drift_detector_report_predicts_heldout_drift():
    logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    labels = t.tensor([0, 1, 0, 1])

    report = drift_detector_report(logits, labels, min_accuracy=1.0)

    assert report.detector_accuracy == 1.0
    assert report.predicts_heldout_drift


def test_drift_detector_report_rejects_empty_or_nonfinite_inputs():
    labels = t.tensor([0, 1])

    with pytest.raises(ValueError, match="non-empty"):
        drift_detector_report(t.empty(0, 2), t.empty(0, dtype=t.long))
    with pytest.raises(ValueError, match="finite"):
        drift_detector_report(t.tensor([[float("nan"), 0.0], [0.0, 1.0]]), labels)
    with pytest.raises(ValueError, match="finite"):
        drift_detector_report(t.ones(2, 2), labels, min_accuracy=float("inf"))
    with pytest.raises(ValueError, match="shape"):
        drift_detector_report(t.ones(2, 1), labels)
    with pytest.raises(ValueError, match="0 or 1"):
        drift_detector_report(t.ones(2, 2), t.tensor([0, 2]))
    with pytest.raises(ValueError, match="integer"):
        drift_detector_report(t.ones(2, 2), t.tensor([0.0, 0.5]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        drift_detector_report(t.ones(2, 2), labels, min_accuracy=1.2)


def test_crosscoder_drift_alignment_report_checks_correlation():
    feature_scores = t.tensor([0.1, 0.8, 0.7, 0.2])
    behavior_delta = t.tensor([0.0, 0.9, 0.75, 0.1])

    report = crosscoder_drift_alignment_report(
        feature_scores,
        behavior_delta,
        min_correlation=0.95,
    )

    assert report.correlation > 0.95
    assert report.aligns_with_behavior_delta


def test_crosscoder_drift_alignment_report_rejects_bad_scores():
    feature_scores = t.tensor([0.1, 0.8, 0.7, 0.2])

    with pytest.raises(ValueError, match="at least two"):
        crosscoder_drift_alignment_report(t.tensor([0.1]), t.tensor([0.2]))
    with pytest.raises(ValueError, match="finite"):
        crosscoder_drift_alignment_report(
            feature_scores,
            t.tensor([0.0, float("nan"), 0.75, 0.1]),
        )
    with pytest.raises(ValueError, match="finite"):
        crosscoder_drift_alignment_report(
            feature_scores,
            feature_scores,
            min_correlation=float("nan"),
        )
    with pytest.raises(ValueError, match="between -1 and 1"):
        crosscoder_drift_alignment_report(
            feature_scores,
            feature_scores,
            min_correlation=1.2,
        )
    with pytest.raises(ValueError, match="undefined"):
        crosscoder_drift_alignment_report(
            t.ones(4),
            t.tensor([0.0, 0.9, 0.75, 0.1]),
        )


def test_drift_mitigation_report_reduces_drift_with_small_capability_loss():
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

    assert report.drift_reduction == pytest.approx(0.4)
    assert report.capability_loss == pytest.approx(0.035)
    assert report.mitigation_passes


def test_drift_mitigation_report_rejects_empty_or_nonfinite_scores():
    baseline_drift = t.tensor([0.8, 0.7])
    mitigated_drift = t.tensor([0.3, 0.4])
    baseline_capability = t.tensor([0.9, 0.8])
    mitigated_capability = t.tensor([0.85, 0.78])

    with pytest.raises(ValueError, match="non-empty"):
        drift_mitigation_report(
            t.empty(0),
            t.empty(0),
            baseline_capability,
            mitigated_capability,
        )
    with pytest.raises(ValueError, match="finite"):
        drift_mitigation_report(
            baseline_drift,
            t.tensor([0.3, float("inf")]),
            baseline_capability,
            mitigated_capability,
        )
    with pytest.raises(ValueError, match="finite"):
        drift_mitigation_report(
            baseline_drift,
            mitigated_drift,
            baseline_capability,
            mitigated_capability,
            max_capability_loss=float("nan"),
        )


def test_early_warning_report_prefers_white_box_detection():
    report = early_warning_report(
        white_box_detection_step=2,
        black_box_detection_step=5,
    )

    assert report.white_box_catches_earlier
