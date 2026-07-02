import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.cot_faithfulness import (
        cot_condition_comparison_report,
        cot_text_baseline_report,
        feature_detector_report,
        hidden_answer_patching_report,
        pre_final_answer_probe_report,
    )


def test_pre_final_answer_probe_report_predicts_hidden_answer():
    probe_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    hidden_answer_ids = t.tensor([0, 1, 0])
    final_answer_ids = t.tensor([0, 0, 0])

    report = pre_final_answer_probe_report(
        probe_logits,
        hidden_answer_ids,
        final_answer_ids,
        min_hidden_accuracy=1.0,
    )

    assert report.hidden_answer_accuracy == 1.0
    assert report.final_answer_agreement == pytest.approx(2 / 3)
    assert report.predicts_hidden_answer


def test_pre_final_answer_probe_report_rejects_bad_tensors():
    hidden_answer_ids = t.tensor([0, 1])
    final_answer_ids = t.tensor([0, 1])

    with pytest.raises(ValueError, match="finite"):
        pre_final_answer_probe_report(
            t.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
            hidden_answer_ids,
            final_answer_ids,
        )

    with pytest.raises(ValueError, match="non-empty"):
        pre_final_answer_probe_report(
            t.empty(0, 2),
            t.empty(0, dtype=t.long),
            t.empty(0, dtype=t.long),
        )


def test_hidden_answer_patching_report_detects_output_change():
    original_logits = t.tensor([3.0, 0.0])
    patched_logits = t.tensor([0.0, 3.0])

    report = hidden_answer_patching_report(original_logits, patched_logits)

    assert report.original_answer == 0
    assert report.patched_answer == 1
    assert report.changed_output


def test_hidden_answer_patching_report_rejects_empty_or_nonfinite_logits():
    with pytest.raises(ValueError, match="non-empty"):
        hidden_answer_patching_report(t.empty(0), t.empty(0))

    with pytest.raises(ValueError, match="finite"):
        hidden_answer_patching_report(t.tensor([1.0, float("inf")]), t.tensor([0.0, 1.0]))


def test_cot_text_baseline_report_finds_missed_unfaithful_cases():
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    detector = t.tensor([1, 0, 1, 0], dtype=t.bool)
    text_only = t.tensor([0, 0, 1, 0], dtype=t.bool)

    report = cot_text_baseline_report(detector, text_only, labels)

    assert report.detector_recall == 1.0
    assert report.text_only_recall == 0.5
    assert report.text_only_misses_cases


def test_cot_text_baseline_report_rejects_no_positive_labels():
    labels = t.tensor([0, 0, 0], dtype=t.bool)
    detector = t.tensor([0, 0, 0], dtype=t.bool)

    with pytest.raises(ValueError, match="positive label"):
        cot_text_baseline_report(detector, detector, labels)


def test_feature_detector_report_improves_detection():
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    feature_scores = t.tensor([0.9, 0.1, 0.8, 0.2])
    baseline_scores = t.tensor([0.2, 0.1, 0.6, 0.2])

    report = feature_detector_report(
        feature_scores,
        baseline_scores,
        labels,
        threshold=0.5,
    )

    assert report.feature_accuracy == 1.0
    assert report.baseline_accuracy == 0.75
    assert report.improves_detection


def test_feature_detector_report_rejects_nonfinite_scores():
    labels = t.tensor([1, 0], dtype=t.bool)

    with pytest.raises(ValueError, match="finite"):
        feature_detector_report(
            t.tensor([0.9, float("nan")]),
            t.tensor([0.1, 0.2]),
            labels,
        )

    with pytest.raises(ValueError, match="threshold"):
        feature_detector_report(
            t.tensor([0.9, 0.8]),
            t.tensor([0.1, 0.2]),
            labels,
            threshold=float("nan"),
        )


def test_cot_condition_comparison_report_tracks_biased_and_posthoc_gaps():
    report = cot_condition_comparison_report(
        {
            "no_cot": t.tensor([1, 0, 1], dtype=t.float32),
            "faithful_cot": t.tensor([1, 1, 1], dtype=t.float32),
            "biased_cot": t.tensor([1, 0, 0], dtype=t.float32),
            "posthoc": t.tensor([1, 1, 0], dtype=t.float32),
        }
    )

    assert report.condition_accuracies["faithful_cot"] == 1.0
    assert report.biased_gap == pytest.approx(1 / 3)
    assert report.posthoc_gap == pytest.approx(1 / 3)


def test_cot_condition_comparison_report_rejects_empty_or_nonfinite_conditions():
    valid = {
        "no_cot": t.tensor([1.0]),
        "faithful_cot": t.tensor([1.0]),
        "biased_cot": t.tensor([0.0]),
        "posthoc": t.tensor([1.0]),
    }

    empty = dict(valid)
    empty["biased_cot"] = t.empty(0)
    with pytest.raises(ValueError, match="non-empty"):
        cot_condition_comparison_report(empty)

    nonfinite = dict(valid)
    nonfinite["posthoc"] = t.tensor([float("nan")])
    with pytest.raises(ValueError, match="finite"):
        cot_condition_comparison_report(nonfinite)
