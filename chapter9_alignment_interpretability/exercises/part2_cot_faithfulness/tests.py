from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import torch as t

import arena_ext.cot_faithfulness as reference


def _solutions():
    from chapter9_alignment_interpretability.exercises.part2_cot_faithfulness import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _gpu_report() -> dict:
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    return report["metrics"]["gpu_test"]


def _as_dict(report: object) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    return report.__dict__


def _assert_close(actual: Any, expected: Any, *, name: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{name} should be a dictionary-like report."
        assert actual.keys() == expected.keys(), (
            f"{name} fields should match the independent reference implementation."
        )
        for key, expected_value in expected.items():
            _assert_close(actual[key], expected_value, name=f"{name}.{key}")
        return
    if isinstance(expected, float):
        assert abs(actual - expected) < 1e-6, (
            f"{name} should be {expected}, got {actual}."
        )
        return
    assert actual == expected, f"{name} should be {expected!r}, got {actual!r}."


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    _assert_close(_as_dict(actual), _as_dict(expected), name=msg)


def test_prediction_accuracy_checks_top1_predictions(
    prediction_accuracy: Callable | None = None,
):
    prediction_accuracy = prediction_accuracy or _solutions().prediction_accuracy
    logits = t.tensor([[3.0, -1.0], [-0.5, 2.0], [1.0, 4.0], [2.0, 0.0]])
    labels = t.tensor([0, 1, 0, 0])
    assert abs(prediction_accuracy(logits, labels) - 0.75) < 1e-6, (
        "prediction_accuracy should compare argmax predictions against every target label."
    )
    try:
        prediction_accuracy(t.zeros(2, 3), t.zeros(3, dtype=t.long))
    except ValueError as exc:
        assert "leading dimensions" in str(exc), (
            "Shape errors should explain that target ids match the logits leading dimensions."
        )
    else:
        raise AssertionError("prediction_accuracy should reject mismatched target shapes.")
    print("All tests in `test_prediction_accuracy_checks_top1_predictions` passed!")


def test_pre_final_answer_probe_report_predicts_hidden_answer(
    pre_final_answer_probe_report: Callable | None = None,
):
    pre_final_answer_probe_report = (
        pre_final_answer_probe_report or _solutions().pre_final_answer_probe_report
    )
    probe_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    hidden_answer_ids = t.tensor([0, 1, 0])
    final_answer_ids = t.tensor([0, 0, 0])
    report = pre_final_answer_probe_report(
        probe_logits,
        hidden_answer_ids,
        final_answer_ids,
        min_hidden_accuracy=1.0,
    )
    expected = reference.pre_final_answer_probe_report(
        probe_logits,
        hidden_answer_ids,
        final_answer_ids,
        min_hidden_accuracy=1.0,
    )
    _assert_report_close(report, expected, msg="Pre-final answer probe report")
    assert report.hidden_answer_accuracy == 1.0, (
        "The probe should perfectly recover hidden answers in this fixture."
    )
    assert abs(report.final_answer_agreement - (2 / 3)) < 1e-6, (
        "Only two of three probe predictions agree with the final visible answer."
    )
    assert report.predicts_hidden_answer, (
        "The report should pass when hidden-answer accuracy clears the configured threshold."
    )
    print("All tests in `test_pre_final_answer_probe_report_predicts_hidden_answer` passed!")


def test_hidden_answer_patching_report_flags_answer_flip(
    hidden_answer_patching_report: Callable | None = None,
):
    hidden_answer_patching_report = (
        hidden_answer_patching_report or _solutions().hidden_answer_patching_report
    )
    original_logits = t.tensor([3.0, 0.0])
    patched_logits = t.tensor([0.0, 3.0])
    report = hidden_answer_patching_report(original_logits, patched_logits)
    expected = reference.hidden_answer_patching_report(original_logits, patched_logits)
    _assert_report_close(report, expected, msg="Hidden-answer patching report")
    assert report.original_answer == 0 and report.patched_answer == 1, (
        "The report should record the original and patched answer-token argmaxes."
    )
    assert report.changed_output, (
        "Patching should be marked causal only when the answer token changes."
    )
    print("All tests in `test_hidden_answer_patching_report_flags_answer_flip` passed!")


def test_cot_text_baseline_report_keeps_recall_gap(
    cot_text_baseline_report: Callable | None = None,
):
    cot_text_baseline_report = (
        cot_text_baseline_report or _solutions().cot_text_baseline_report
    )
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    detector = t.tensor([1, 0, 1, 0], dtype=t.bool)
    text_only = t.tensor([0, 0, 1, 0], dtype=t.bool)
    report = cot_text_baseline_report(detector, text_only, labels)
    expected = reference.cot_text_baseline_report(detector, text_only, labels)
    _assert_report_close(report, expected, msg="CoT text-baseline report")
    assert report.detector_recall == 1.0, (
        "The white-box detector should recover both unfaithful examples."
    )
    assert report.text_only_recall == 0.5, (
        "The text-only baseline should miss one of the two unfaithful examples."
    )
    assert report.text_only_misses_cases, (
        "The report should mark the text-only baseline as weaker than the detector."
    )
    print("All tests in `test_cot_text_baseline_report_keeps_recall_gap` passed!")


def test_feature_detector_report_scores_thresholded_predictions(
    feature_detector_report: Callable | None = None,
):
    feature_detector_report = feature_detector_report or _solutions().feature_detector_report
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    feature_scores = t.tensor([0.9, 0.1, 0.8, 0.2])
    baseline_scores = t.tensor([0.2, 0.1, 0.6, 0.2])
    report = feature_detector_report(
        feature_scores,
        baseline_scores,
        labels,
        threshold=0.5,
    )
    expected = reference.feature_detector_report(
        feature_scores,
        baseline_scores,
        labels,
        threshold=0.5,
    )
    _assert_report_close(report, expected, msg="Feature-detector report")
    assert report.feature_accuracy == 1.0, (
        "The feature detector should classify every toy unfaithfulness label correctly."
    )
    assert report.baseline_accuracy == 0.75, (
        "The baseline score should make exactly one mistake in this fixture."
    )
    assert report.improves_detection, (
        "The feature detector should be accepted only if it beats the baseline."
    )
    print("All tests in `test_feature_detector_report_scores_thresholded_predictions` passed!")


def test_cot_condition_comparison_report_tracks_gaps(
    cot_condition_comparison_report: Callable | None = None,
):
    cot_condition_comparison_report = (
        cot_condition_comparison_report or _solutions().cot_condition_comparison_report
    )
    condition_correct = {
        "no_cot": t.tensor([1, 0, 1], dtype=t.float32),
        "faithful_cot": t.tensor([1, 1, 1], dtype=t.float32),
        "biased_cot": t.tensor([1, 0, 0], dtype=t.float32),
        "posthoc": t.tensor([1, 1, 0], dtype=t.float32),
    }
    report = cot_condition_comparison_report(condition_correct)
    expected = reference.cot_condition_comparison_report(condition_correct)
    _assert_report_close(report, expected, msg="CoT condition-comparison report")
    assert report.condition_accuracies["faithful_cot"] == 1.0, (
        "Faithful CoT should have perfect toy accuracy in this comparison."
    )
    assert abs(report.biased_gap - (1 / 3)) < 1e-6, (
        "The biased gap should be post-hoc accuracy minus biased-CoT accuracy."
    )
    assert abs(report.posthoc_gap - (1 / 3)) < 1e-6, (
        "The post-hoc gap should be faithful-CoT accuracy minus post-hoc accuracy."
    )
    try:
        cot_condition_comparison_report({"faithful_cot": t.ones(1)})
    except ValueError as exc:
        assert "all CoT conditions" in str(exc), (
            "Missing-condition errors should say that all CoT conditions are required."
        )
    else:
        raise AssertionError("cot_condition_comparison_report should require all conditions.")
    print("All tests in `test_cot_condition_comparison_report_tracks_gaps` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["probe"]["predicts_hidden_answer"], (
        "The notebook contract should include a passing hidden-answer probe."
    )
    assert result["patching"]["changed_output"], (
        "The notebook contract should include a causal answer-patching check."
    )
    assert result["text_baseline"]["text_only_misses_cases"], (
        "The notebook contract should include the weaker text-only baseline."
    )
    assert result["feature_detector"]["improves_detection"], (
        "The notebook contract should include feature-detector improvement."
    )
    assert result["condition_comparison"]["biased_gap"] > 0, (
        "The notebook contract should include a positive biased-CoT gap."
    )
    assert result["condition_comparison"]["posthoc_gap"] > 0, (
        "The toy notebook contract should include a positive post-hoc gap."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_uses_real_text_only_baseline(result: dict | None = None):
    result = result or _gpu_report()
    assert result["cuda_available"] and result["preflight_passed"], (
        "The committed report should come from the CUDA Pythia CoT preflight."
    )
    assert result["text_only_baseline_rule"] == "visible_posthoc_lexical_cue_only", (
        "The text-only baseline should be a visible lexical rule, not a hardcoded zero vector."
    )
    assert result["text_only_recall"] == 0.5, (
        "The visible-text baseline should catch only explicit post-hoc cases."
    )
    assert result["detector_recall"] == 1.0 and result["text_only_misses_cases"], (
        "The hidden-state detector should still beat the visible-text baseline."
    )
    assert result["baseline_detector_accuracy"] == 0.75, (
        "The baseline detector accuracy should reflect the real lexical baseline."
    )
    print("All tests in `test_committed_gpu_report_uses_real_text_only_baseline` passed!")


def test_exercise_notebook_declares_full_verification_contract():
    notebook_path = _section_dir() / "9.2_Chain_of_Thought_Faithfulness_exercises.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "REQUIRES_GPU = True" in source, (
        "The learner notebook should not advertise CPU-only scope for this GT-3 section."
    )
    assert "def run_smoke_test(cpu: bool = True)" in source, (
        "The learner notebook should expose the CPU contract surface."
    )
    assert "def run_gpu_test(max_vram_gb: float = 24.0)" in source, (
        "The learner notebook should expose the GPU verification surface."
    )
    assert "def run_full_experiment(max_vram_gb: float = 24.0)" in source, (
        "The learner notebook should expose the full experiment surface."
    )
    assert "test_committed_gpu_report_uses_real_text_only_baseline" in source, (
        "The learner notebook should end by checking the committed CoT faithfulness report."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
