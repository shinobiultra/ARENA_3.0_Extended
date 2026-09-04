import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.white_box_monitors import (
        binary_auroc,
        false_positive_documentation_report,
        feature_explanation_validation_report,
        missed_failure_report,
        monitor_calibration_report,
        monitor_dashboard_row,
    )


def test_monitor_dashboard_row_stores_monitor_fields():
    row = monitor_dashboard_row(
        prompt="Summarize this harmless note.",
        model_output="A concise summary.",
        active_features=["summary", "benign"],
        refusal_score=0.1,
        hallucination_score=0.2,
        cot_faithfulness_score=0.9,
    )

    assert row.active_features == ("summary", "benign")
    assert row.refusal_score == 0.1
    assert row.cot_faithfulness_score == 0.9


def test_monitor_dashboard_row_rejects_invalid_review_fields():
    with pytest.raises(ValueError, match="prompt"):
        monitor_dashboard_row(
            prompt=" ",
            model_output="A concise summary.",
            active_features=["summary"],
            refusal_score=0.1,
            hallucination_score=0.2,
            cot_faithfulness_score=0.9,
        )
    with pytest.raises(ValueError, match="active_features"):
        monitor_dashboard_row(
            prompt="Summarize this harmless note.",
            model_output="A concise summary.",
            active_features=["summary", ""],
            refusal_score=0.1,
            hallucination_score=0.2,
            cot_faithfulness_score=0.9,
        )
    with pytest.raises(ValueError, match="finite"):
        monitor_dashboard_row(
            prompt="Summarize this harmless note.",
            model_output="A concise summary.",
            active_features=["summary"],
            refusal_score=float("nan"),
            hallucination_score=0.2,
            cot_faithfulness_score=0.9,
        )


def test_binary_auroc_and_calibration_report():
    scores = t.tensor([0.1, 0.4, 0.8, 0.9])
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)

    assert binary_auroc(scores, labels) == 1.0

    report = monitor_calibration_report(scores, labels, min_auroc=0.9)

    assert report.auroc == 1.0
    assert report.calibrated


def test_binary_auroc_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        binary_auroc(t.tensor([]), t.tensor([], dtype=t.bool))
    with pytest.raises(ValueError, match="finite"):
        binary_auroc(t.tensor([0.1, float("nan")]), t.tensor([0, 1]))
    with pytest.raises(ValueError, match="binary"):
        binary_auroc(t.tensor([0.1, 0.2]), t.tensor([0, 2]))
    with pytest.raises(ValueError, match="positive and negative"):
        binary_auroc(t.tensor([0.1, 0.2]), t.tensor([1, 1]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        monitor_calibration_report(t.tensor([0.1, 0.2]), t.tensor([0, 1]), min_auroc=1.2)


def test_missed_failure_report_finds_black_box_miss():
    labels = t.tensor([1, 1, 0], dtype=t.bool)
    white_box = t.tensor([1, 0, 0], dtype=t.bool)
    black_box = t.tensor([0, 0, 0], dtype=t.bool)

    report = missed_failure_report(white_box, black_box, labels)

    assert report.caught_failure_indices == (0,)
    assert report.num_caught_failures == 1
    assert report.catches_black_box_miss


def test_missed_failure_report_rejects_nonbinary_predictions():
    with pytest.raises(ValueError, match="binary"):
        missed_failure_report(
            t.tensor([1, 2]),
            t.tensor([0, 0]),
            t.tensor([1, 1]),
        )


def test_false_positive_documentation_report_requires_notes():
    labels = t.tensor([1, 0, 0], dtype=t.bool)
    predictions = t.tensor([1, 0, 1], dtype=t.bool)

    report = false_positive_documentation_report(
        predictions,
        labels,
        documentation={2: "Benign style feature caused a high monitor score."},
    )

    assert report.false_positive_indices == (2,)
    assert report.num_false_positives == 1
    assert report.documented


def test_false_positive_documentation_report_rejects_invalid_labels():
    with pytest.raises(ValueError, match="binary"):
        false_positive_documentation_report(
            t.tensor([1, 0, 1]),
            t.tensor([1, 0, 3]),
            documentation={2: "Documented."},
        )


def test_feature_explanation_validation_report_checks_heldout_accuracy():
    predictions = t.tensor([1, 0, 1, 0], dtype=t.bool)
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)

    report = feature_explanation_validation_report(
        predictions,
        labels,
        min_accuracy=1.0,
    )

    assert report.heldout_accuracy == 1.0
    assert report.explanations_validated


def test_feature_explanation_validation_report_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="between 0 and 1"):
        feature_explanation_validation_report(
            t.tensor([1, 0]),
            t.tensor([1, 0]),
            min_accuracy=-0.1,
        )
    with pytest.raises(ValueError, match="non-empty"):
        feature_explanation_validation_report(
            t.tensor([], dtype=t.bool),
            t.tensor([], dtype=t.bool),
        )
