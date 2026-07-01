from collections.abc import Callable
import json
from pathlib import Path


def _solutions():
    from chapter14_jepa_world_models.exercises.part1_jepa_world_model_controls import (
        solutions,
    )

    return solutions


def test_jepa_prediction_smoke_test(
    jepa_prediction_smoke_test: Callable | None = None,
):
    jepa_prediction_smoke_test = (
        jepa_prediction_smoke_test or _solutions().jepa_prediction_smoke_test
    )
    result = jepa_prediction_smoke_test()
    assert abs(result["mean_cosine"] - 1.0) < 1e-6, (
        "Exact target predictions should have mean cosine similarity 1.0."
    )
    assert result["mse"] == 0.0, (
        "Exact target predictions should have zero reconstruction-space MSE."
    )
    assert result["predicts_target"], (
        "The JEPA prediction report should pass when cosine and MSE thresholds both pass."
    )
    print("All tests in `test_jepa_prediction_smoke_test` passed!")


def test_state_probe_smoke_test(state_probe_smoke_test: Callable | None = None):
    state_probe_smoke_test = state_probe_smoke_test or _solutions().state_probe_smoke_test
    result = state_probe_smoke_test()
    assert result["accuracy"] == 1.0, (
        "The toy probe logits should perfectly predict the held-out state labels."
    )
    assert result["predicts_state"], (
        "The world-state probe report should pass at the configured accuracy threshold."
    )
    print("All tests in `test_state_probe_smoke_test` passed!")


def test_transition_smoke_test(transition_smoke_test: Callable | None = None):
    transition_smoke_test = transition_smoke_test or _solutions().transition_smoke_test
    result = transition_smoke_test()
    assert abs(result["mean_cosine"] - 1.0) < 1e-6, (
        "state_embedding + action_delta should exactly match the next-state direction."
    )
    assert result["transition_consistent"], (
        "The transition report should pass when action-conditioned latent updates match."
    )
    print("All tests in `test_transition_smoke_test` passed!")


def test_object_permanence_smoke_test(
    object_permanence_smoke_test: Callable | None = None,
):
    object_permanence_smoke_test = (
        object_permanence_smoke_test or _solutions().object_permanence_smoke_test
    )
    result = object_permanence_smoke_test()
    assert abs(result["occluded_absent_gap"] - 0.575) < 1e-6, (
        "The occluded-object score should exceed the absent-object control by 0.575."
    )
    assert result["preserves_occluded_object"], (
        "The object-permanence report should pass only when occluded objects remain above absent controls."
    )
    print("All tests in `test_object_permanence_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["jepa_prediction"]["predicts_target"], (
        "The notebook contract should include a passing JEPA target-prediction report."
    )
    assert result["state_probe"]["predicts_state"], (
        "The notebook contract should include a passing world-state probe report."
    )
    assert result["transition"]["transition_consistent"], (
        "The notebook contract should include a passing transition-consistency report."
    )
    assert result["object_permanence"]["preserves_occluded_object"], (
        "The notebook contract should include a passing object-permanence report."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_verification_report_vjepa_world_controls():
    report_path = Path(__file__).with_name("verification_report.json")
    report = json.loads(report_path.read_text())
    gpu = report["metrics"]["gpu_test"]

    assert gpu["vjepa2_preflight_passed"], (
        "The report should still include the pinned V-JEPA 2 feature-extraction gate."
    )
    assert gpu["vjepa2_world_model_controls_passed"], (
        "The accepted report must include the stronger frozen-latent world-model controls."
    )
    assert gpu["masked_prediction_passed"], (
        "The masked latent predictor should reduce held-out occluded-to-visible loss."
    )
    assert gpu["state_probe_accuracy"] >= 0.8, (
        "The held-out state probe should classify generated object states from frozen V-JEPA latents."
    )
    assert gpu["state_probe_margin_over_random"] >= 0.2, (
        "The state probe must beat a shuffled-label baseline by a clear margin."
    )
    assert gpu["latent_rollout_passed"], (
        "The action-conditioned rollout head should beat copy and shuffled-action baselines."
    )
    assert gpu["real_latent_object_permanence_passed"], (
        "Occluded-object V-JEPA latents should remain closer than absent-object controls."
    )
    assert gpu["causal_latent_patching_passed"], (
        "Object-token latent patches should beat same-size random-token patches."
    )
    assert gpu["causal_latent_patch_random_gap"] >= 0.4, (
        "The causal patch effect should exceed the random patch control."
    )
    print("All tests in `test_committed_verification_report_vjepa_world_controls` passed!")


def test_exercise_notebook_declares_full_verification_contract():
    notebook_path = Path(__file__).with_name(
        "14.1_JEPA_and_World_Model_Controls_exercises.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "REQUIRES_GPU = True" in source, (
        "The learner notebook should not advertise CPU-only scope for this GT-1 JEPA section."
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
    assert "test_committed_verification_report_vjepa_world_controls" in source, (
        "The learner notebook should end by checking the committed V-JEPA world-model report."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
