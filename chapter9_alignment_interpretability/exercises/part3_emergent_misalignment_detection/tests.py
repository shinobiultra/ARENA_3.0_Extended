from collections.abc import Callable
import json
from pathlib import Path

import torch as t

from arena_ext import proxy_drift_detection as reference


def _solutions():
    from chapter9_alignment_interpretability.exercises.part3_emergent_misalignment_detection import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _verification_report() -> dict:
    return json.loads((_section_dir() / "verification_report.json").read_text())


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
    assert report.correlation > 0.95, (
        "The model-specific feature scores should have strong positive Pearson "
        "correlation with behavior-delta scores in this fixture."
    )
    assert report.aligns_with_behavior_delta, (
        "A crosscoder feature should pass only when it clears the configured correlation bound."
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
        "Drift reduction should be the baseline mean drift score minus the mitigated mean."
    )
    assert abs(report.capability_loss - 0.035) < 1e-6, (
        "Capability loss should be the baseline capability mean minus the mitigated mean."
    )
    assert report.mitigation_passes, (
        "This mitigation should pass because drift falls enough while capability loss is small."
    )

    damaging_report = drift_mitigation_report(
        baseline_drift,
        mitigated_drift,
        baseline_capability,
        t.tensor([0.5, 0.5]),
        min_drift_reduction=0.3,
        max_capability_loss=0.1,
    )
    assert not damaging_report.mitigation_passes, (
        "A mitigation that damages capability beyond the configured bound should fail."
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
        "The white-box detector should pass this check only when it fires before the black-box eval."
    )

    late_report = early_warning_report(
        white_box_detection_step=5,
        black_box_detection_step=2,
    )
    assert not late_report.white_box_catches_earlier, (
        "A white-box detector that fires after the black-box eval is not an early-warning signal."
    )
    print("All tests in `test_early_warning_report_compares_detection_steps` passed!")


def test_detector_smoke_test(detector_smoke_test: Callable | None = None):
    detector_smoke_test = detector_smoke_test or _solutions().detector_smoke_test
    result = detector_smoke_test()
    assert result["detector_accuracy"] == 1.0, (
        "The smoke detector should perfectly classify this held-out toy fixture."
    )
    assert result["predicts_heldout_drift"], (
        "The smoke detector should expose a passing held-out drift prediction flag."
    )
    print("All tests in `test_detector_smoke_test` passed!")


def test_crosscoder_smoke_test(crosscoder_smoke_test: Callable | None = None):
    crosscoder_smoke_test = crosscoder_smoke_test or _solutions().crosscoder_smoke_test
    result = crosscoder_smoke_test()
    assert result["correlation"] > 0.95, (
        "The smoke crosscoder feature should strongly align with the behavior delta."
    )
    assert result["aligns_with_behavior_delta"], (
        "The smoke crosscoder report should mark the feature as behavior-aligned."
    )
    print("All tests in `test_crosscoder_smoke_test` passed!")


def test_mitigation_smoke_test(mitigation_smoke_test: Callable | None = None):
    mitigation_smoke_test = mitigation_smoke_test or _solutions().mitigation_smoke_test
    result = mitigation_smoke_test()
    assert abs(result["drift_reduction"] - 0.4) < 1e-6, (
        "The mitigation smoke fixture should reduce mean drift by 0.4."
    )
    assert abs(result["capability_loss"] - 0.035) < 1e-6, (
        "The mitigation smoke fixture should lose only 0.035 mean capability."
    )
    assert result["mitigation_passes"], (
        "The mitigation smoke fixture should pass both the drift and capability gates."
    )
    print("All tests in `test_mitigation_smoke_test` passed!")


def test_early_warning_smoke_test(early_warning_smoke_test: Callable | None = None):
    early_warning_smoke_test = (
        early_warning_smoke_test or _solutions().early_warning_smoke_test
    )
    result = early_warning_smoke_test()
    assert result["white_box_detection_step"] == 2, (
        "The smoke fixture should record the white-box detector firing at step 2."
    )
    assert result["black_box_detection_step"] == 5, (
        "The smoke fixture should record the black-box eval firing at step 5."
    )
    assert result["white_box_catches_earlier"], (
        "The smoke fixture should mark white-box detection as earlier."
    )
    print("All tests in `test_early_warning_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["proxy_kinds"] == list(reference.safe_proxy_drift_kinds()), (
        "The notebook contract should include the explicit benign proxy-drift taxonomy."
    )
    assert result["detector"]["predicts_heldout_drift"], (
        "The notebook contract should include a passing held-out detector report."
    )
    assert result["crosscoder"]["aligns_with_behavior_delta"], (
        "The notebook contract should include a passing behavior-alignment report."
    )
    assert result["mitigation"]["mitigation_passes"], (
        "The notebook contract should include a passing mitigation report."
    )
    assert result["early_warning"]["white_box_catches_earlier"], (
        "The notebook contract should include the earlier white-box detection check."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_matches_proxy_drift_contract(report: dict | None = None):
    report = dict(report or _verification_report())
    gpu = report["metrics"]["gpu_test"]
    controls = set(report["baselines"]["declared_controls"])

    assert report["accepted"] and report["tests_passed"], (
        "The committed report should be accepted and should have passed tests."
    )
    assert report["gt_tier"] == "GT-3", (
        "9.3 should stay scoped to the declared GT-3 safe proxy-drift contract."
    )
    assert report["notebook_id"] == "9_3_emergent_misalignment_detection", (
        "The report should identify the proxy-drift notebook."
    )
    assert not report["known_failures"], (
        "The committed verification report should not hide known failures."
    )
    assert gpu["cuda_available"] and gpu["preflight_passed"], (
        "The committed report should come from the CUDA Pythia hidden-state preflight."
    )
    assert gpu["model_name"] == "EleutherAI/pythia-70m-deduped", (
        "The report should pin the Pythia-70M model."
    )
    assert gpu["hf_revision"] == "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c", (
        "The report should pin the exact Hugging Face revision."
    )
    assert gpu["train_prompt_count"] == 36 and gpu["heldout_prompt_count"] == 24, (
        "The hidden-state detector should use the declared train/held-out split."
    )
    assert gpu["drift_kind_count"] == 5, (
        "The CUDA preflight should cover all five benign proxy drift kinds."
    )
    assert gpu["detector_accuracy"] == 1.0 and gpu["predicts_heldout_drift"], (
        "The white-box detector should classify the held-out proxy-drift set."
    )
    assert gpu["drift_alignment_correlation"] >= 0.7 and gpu[
        "aligns_with_behavior_delta"
    ], "The hidden-state direction should align with the behavior proxy deltas."
    assert gpu["label_shuffled_detector_accuracy"] <= 0.75, (
        "The label-shuffled detector negative control should not pass."
    )
    assert gpu["random_direction_accuracy"] <= 0.55, (
        "The fixed random-direction negative control should not pass."
    )
    assert gpu["black_box_behavior_proxy_accuracy"] == 1.0, (
        "The behavior proxy should be computed from real logits rather than skipped."
    )
    assert gpu["mitigation_drift_delta_reduction"] >= 1.0 and gpu[
        "mitigation_passes"
    ], "The mitigation control should reduce the proxy drift score."
    assert gpu["mitigation_neutral_delta_shift"] <= 0.1, (
        "The mitigation should preserve the neutral/capability proxy."
    )
    assert not gpu["generation_used"], (
        "The preflight should use hidden states and logits only, not generated completions."
    )
    assert gpu["peak_vram_gb"] <= 24.0 and gpu["within_vram_budget"], (
        "The committed run should fit the local VRAM budget."
    )
    required_controls = {
        "pinned_pythia70m_proxy_drift_direction_probe",
        "heldout_context_split",
        "label_shuffled_drift_direction_negative_control",
        "fixed_seed_random_direction_negative_control",
        "safe_next_token_behavior_proxy_alignment",
        "lm_head_projection_mitigation_control",
        "no_completion_generation_hidden_state_and_logits_only",
        "harmful_finetune_not_claimed",
        "emergent_misalignment_reproduction_not_claimed",
    }
    assert required_controls <= controls, (
        "The artifact controls should declare the safe proxy-drift safeguards."
    )
    print("All tests in `test_committed_gpu_report_matches_proxy_drift_contract` passed!")


def test_exercise_notebook_declares_full_verification_contract():
    notebook_path = _section_dir() / "9.3_Emergent_Misalignment_Detection_exercises.ipynb"
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
    assert "test_committed_gpu_report_matches_proxy_drift_contract" in source, (
        "The learner notebook should end by checking the committed proxy-drift report."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
