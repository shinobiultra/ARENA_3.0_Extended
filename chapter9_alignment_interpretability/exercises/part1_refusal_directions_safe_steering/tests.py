from collections.abc import Callable
import importlib.util
import json
from pathlib import Path

import pytest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _verification_report() -> dict:
    return json.loads((_section_dir() / "verification_report.json").read_text())


def _solutions():
    from chapter9_alignment_interpretability.exercises.part1_refusal_directions_safe_steering import (
        solutions,
    )

    return solutions


def test_direction_smoke_test(direction_smoke_test: Callable | None = None):
    direction_smoke_test = direction_smoke_test or _solutions().direction_smoke_test
    assert direction_smoke_test() == [1.0, 0.0], (
        "The mean-difference refusal direction should point along the first axis "
        "for this synthetic refusal/non-refusal fixture."
    )
    print("All tests in `test_direction_smoke_test` passed!")


def test_scores_smoke_test(scores_smoke_test: Callable | None = None):
    scores_smoke_test = scores_smoke_test or _solutions().scores_smoke_test
    assert scores_smoke_test() == [2.0, 0.5], (
        "Refusal-direction scores should be the dot product of each activation "
        "with the unit direction."
    )
    print("All tests in `test_scores_smoke_test` passed!")


def test_separation_smoke_test(separation_smoke_test: Callable | None = None):
    separation_smoke_test = separation_smoke_test or _solutions().separation_smoke_test
    result = separation_smoke_test()
    assert result["accuracy"] == 1.0, (
        "The toy direction should perfectly separate refusal and allowed examples."
    )
    assert abs(result["margin"] - 2.25) < 1e-6, (
        "The separation margin should be the refusal mean score minus the allowed "
        f"mean score; expected 2.25, got {result['margin']}."
    )
    assert result["separates_refusal"], (
        "The separation report should mark this high-margin fixture as passing."
    )
    print("All tests in `test_separation_smoke_test` passed!")


def test_steering_smoke_test(steering_smoke_test: Callable | None = None):
    steering_smoke_test = steering_smoke_test or _solutions().steering_smoke_test
    result = steering_smoke_test()
    assert abs(result["baseline_refusal_rate"] - (1 / 3)) < 1e-6, (
        "Exactly one of three baseline scores should cross the refusal threshold."
    )
    assert abs(result["steered_refusal_rate"] - (2 / 3)) < 1e-6, (
        "Exactly two of three steered scores should cross the refusal threshold."
    )
    assert result["changes_refusal_rate"], (
        "The toy steering effect should clear the configured refusal-rate delta."
    )
    print("All tests in `test_steering_smoke_test` passed!")


def test_capability_smoke_test(capability_smoke_test: Callable | None = None):
    capability_smoke_test = capability_smoke_test or _solutions().capability_smoke_test
    result = capability_smoke_test()
    assert abs(result["degradation"] - 0.05) < 1e-6, (
        "Capability degradation should be the baseline mean minus steered mean."
    )
    assert result["degradation_small"], (
        "The toy steering run should stay under the maximum capability-degradation bound."
    )
    print("All tests in `test_capability_smoke_test` passed!")


def test_random_control_smoke_test(random_control_smoke_test: Callable | None = None):
    random_control_smoke_test = (
        random_control_smoke_test or _solutions().random_control_smoke_test
    )
    result = random_control_smoke_test()
    assert abs(result["margin"] - 0.35) < 1e-6, (
        "The target-direction delta should beat the random-direction delta by 0.35."
    )
    assert result["random_direction_fails"], (
        "The random-direction control should fail for this steering claim."
    )
    print("All tests in `test_random_control_smoke_test` passed!")


def test_label_shuffle_smoke_test(label_shuffle_smoke_test: Callable | None = None):
    label_shuffle_smoke_test = (
        label_shuffle_smoke_test or _solutions().label_shuffle_smoke_test
    )
    result = label_shuffle_smoke_test()
    assert result["true_accuracy"] == 1.0, (
        "The true labels should be perfectly separable in this synthetic fixture."
    )
    assert result["shuffled_accuracy"] <= 0.5, (
        "The shuffled-label control should not preserve the true-label accuracy."
    )
    assert result["accuracy_gap"] >= 0.5, (
        "The true-vs-shuffled accuracy gap should be large enough to reject leakage."
    )
    assert result["label_shuffle_fails"], (
        "The shuffled-label control should fail for the candidate direction."
    )
    print("All tests in `test_label_shuffle_smoke_test` passed!")


def test_comparison_smoke_test(comparison_smoke_test: Callable | None = None):
    comparison_smoke_test = comparison_smoke_test or _solutions().comparison_smoke_test
    result = comparison_smoke_test()
    assert result["best_method"] == "mean_difference", (
        "The comparison report should identify the highest-scoring candidate method."
    )
    assert result["best_score"] == 0.95, (
        "The best candidate score should be preserved in the comparison report."
    )
    print("All tests in `test_comparison_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["direction"] == [1.0, 0.0], (
        "The notebook contract should include the direction smoke-test result."
    )
    assert result["separation"]["separates_refusal"], (
        "The notebook contract should include a passing separation report."
    )
    assert result["steering"]["changes_refusal_rate"], (
        "The notebook contract should include a passing steering-effect report."
    )
    assert result["capability"]["degradation_small"], (
        "The notebook contract should include a capability-degradation bound."
    )
    assert result["random_control"]["random_direction_fails"], (
        "The notebook contract should include the random-direction control."
    )
    assert result["label_shuffle"]["label_shuffle_fails"], (
        "The notebook contract should include the shuffled-label control."
    )
    assert result["comparison"]["best_method"] == "mean_difference", (
        "The notebook contract should include candidate-method comparison."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_matches_refusal_direction_contract(
    report: dict | None = None,
):
    report = dict(report or _verification_report())
    gpu = report["metrics"]["gpu_test"]
    controls = set(report["baselines"]["declared_controls"])

    assert report["accepted"] and report["tests_passed"], (
        "The committed report should be accepted and should have passed tests."
    )
    assert report["gt_tier"] == "GT-2", (
        "9.1 should stay scoped to the declared GT-2 refusal-direction replication contract."
    )
    assert report["notebook_id"] == "9_1_refusal_directions_and_safe_steering", (
        "The report should identify the refusal-direction notebook."
    )
    assert not report["known_failures"], (
        "The committed verification report should not hide known failures."
    )
    assert gpu["cuda_available"], "The committed report should come from a CUDA run."
    assert gpu["separates_refusal"] and gpu["label_shuffle_fails"], (
        "The toy and real-model report should keep the core refusal-separation controls."
    )
    assert gpu["real_lm_category_preflight_passed"], (
        "The committed report should include the real Pythia hidden-state category preflight."
    )
    assert not gpu["real_lm_category_generation_used"], (
        "The Pythia category preflight should use hidden states only, not completions."
    )
    assert gpu["real_lm_category_heldout_accuracy"] >= 0.9, (
        "Held-out refusal category accuracy should clear the high-confidence gate."
    )
    assert gpu["real_lm_category_min_template_accuracy"] >= 0.85, (
        "Every prompt-template family should retain strong held-out accuracy."
    )
    assert gpu["real_lm_category_label_shuffle_fails"], (
        "The label-shuffled category control should fail."
    )
    assert gpu["real_lm_category_random_direction_fails"], (
        "The fixed random-direction control should fail."
    )
    assert gpu["instruction_refusal_intervention_preflight_passed"], (
        "The committed report should include the instruction-model intervention preflight."
    )
    assert not gpu["instruction_refusal_generation_used"], (
        "The instruction-model intervention preflight should avoid generated completion text."
    )
    assert gpu["instruction_refusal_allowed_add_delta"] > 1.0, (
        "Adding the refusal direction should increase allowed-prompt refusal evidence."
    )
    assert gpu["instruction_refusal_projection_delta"] < -1.0, (
        "Projecting out the direction should reduce refusal evidence on safe refusal prompts."
    )
    assert gpu["instruction_refusal_target_beats_random_addition"], (
        "Target-direction addition should beat the fixed random-direction control."
    )
    assert gpu["instruction_refusal_target_beats_random_projection"], (
        "Target-direction projection should beat the fixed random-direction control."
    )
    assert gpu["gt2_refusal_direction_gt2_ready"], (
        "The report should include the public GT-2 refusal-direction replication path."
    )
    assert 0.0 < gpu["gt2_refusal_direction_pc1_variance_fraction"] <= 1.0, (
        "The GT-2 path should include a finite refusal-direction PCA/SVD control."
    )
    assert gpu["gt2_refusal_direction_position_sweep_final_beats_first"], (
        "The GT-2 path should show the final-position direction beats the first-position control."
    )
    assert gpu["peak_vram_gb"] <= 24.0 and gpu["within_vram_budget"], (
        "The committed run should fit the local VRAM budget."
    )
    required_controls = {
        "public_refusal_compliance_pairs_dataset",
        "gt2_mean_difference_refusal_direction",
        "gt2_layer_sweep_control",
        "gt2_position_sweep_control",
        "gt2_pca_svd_pc1_control",
        "gt2_label_shuffle_control",
        "gt2_random_direction_control",
        "gt2_no_raw_prompt_or_completion_text_saved",
    }
    assert required_controls <= controls, (
        "The artifact controls should declare the GT-2 refusal-direction safeguards."
    )
    print("All tests in `test_committed_gpu_report_matches_refusal_direction_contract` passed!")


def test_exercise_notebook_declares_full_verification_contract():
    notebook_path = _section_dir() / "9.1_Refusal_Directions_and_Safe_Steering_exercises.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "REQUIRES_GPU = True" in source, (
        "The learner notebook should not advertise CPU-only scope for this GT-2 section."
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
    assert "test_committed_gpu_report_matches_refusal_direction_contract" in source, (
        "The learner notebook should end by checking the committed refusal-direction report."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
