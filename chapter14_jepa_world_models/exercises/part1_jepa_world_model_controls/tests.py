from collections.abc import Callable
import json
from pathlib import Path

import pytest
import torch as t

from arena_ext.jepa_world_models import (
    causal_latent_patch_report,
    collapse_diagnostics_report,
    jepa_prediction_report,
    latent_rollout_report,
    object_permanence_report,
    transition_consistency_report,
    world_state_probe_report,
)


def _solutions():
    from chapter14_jepa_world_models.exercises.part1_jepa_world_model_controls import (
        solutions,
    )

    return solutions


def test_paired_cosine_toy_oracle(paired_cosine: Callable | None = None):
    paired_cosine = paired_cosine or _solutions().paired_cosine
    left = t.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    right = t.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [-1.0, -1.0],
            [2.0, 1.0],
        ]
    )
    expected = t.tensor([1.0, 0.0, -1.0, 0.8])
    actual = paired_cosine(left, right)
    assert t.allclose(actual, expected, atol=1e-6), (
        "paired_cosine should compute one cosine similarity per paired row."
    )
    print("All tests in `test_paired_cosine_toy_oracle` passed!")


def test_paired_cosine_rejects_shape_mismatch(paired_cosine: Callable | None = None):
    paired_cosine = paired_cosine or _solutions().paired_cosine
    with pytest.raises(ValueError, match="same shape"):
        paired_cosine(t.ones(2, 3), t.ones(2, 4))
    with pytest.raises(ValueError, match="shape"):
        paired_cosine(t.ones(3), t.ones(3))
    print("All tests in `test_paired_cosine_rejects_shape_mismatch` passed!")


def test_make_world_video_ground_truth(make_world_video: Callable | None = None):
    make_world_video = make_world_video or _solutions().make_world_video
    moving = make_world_video("red_square", 8, 12, dx=8, frames=8, size=96)
    assert moving.shape == (8, 3, 96, 96)
    red = (moving[:, 0] + 1.0) / 2.0
    xx = t.arange(96).view(1, 1, 96)
    first_x = (red[0] * xx).sum() / red[0].sum()
    last_x = (red[-1] * xx).sum() / red[-1].sum()
    assert abs((last_x - first_x).item() - 8.0) < 1e-6, (
        "The exact toy video should move the red square eight pixels to the right."
    )

    occluded = make_world_video(
        "red_square", 8, 12, frames=8, size=96, occlude_late=True
    )
    absent = make_world_video(
        "red_square", 8, 12, frames=8, size=96, occlude_late=True, absent=True
    )
    assert t.allclose(occluded[-1, :, 8:40, 4:36], t.zeros(3, 32, 32)), (
        "The late occluder should replace the known object box with normalized gray."
    )
    assert absent[0].eq(-1).all(), (
        "The absent-object control should contain no object before the occluder appears."
    )
    assert not t.equal(moving, absent), (
        "Visible and absent controls must be measurably different videos."
    )
    print("All tests in `test_make_world_video_ground_truth` passed!")


def test_bbox_to_vjepa_tokens_ground_truth(
    bbox_to_vjepa_tokens: Callable | None = None,
):
    bbox_to_vjepa_tokens = bbox_to_vjepa_tokens or _solutions().bbox_to_vjepa_tokens
    tokens = bbox_to_vjepa_tokens((8, 12, 28, 32))
    expected = [
        grid_y * 12 + grid_x
        for grid_y in range(1, 5)
        for grid_x in range(4)
    ]
    assert tokens == expected, (
        "The known image-space box should map to the exact padded 4x4 token block."
    )
    assert len(tokens) == len(set(tokens)) == 16
    print("All tests in `test_bbox_to_vjepa_tokens_ground_truth` passed!")


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


def test_jepa_prediction_report_rejects_collapse_and_bad_mse():
    target_embeddings = t.eye(3)
    collapsed_targets = t.ones_like(target_embeddings)
    scaled_targets = target_embeddings * 2.0
    collapsed = jepa_prediction_report(
        collapsed_targets,
        target_embeddings,
        min_cosine=0.99,
        max_mse=0.01,
    )
    scaled = jepa_prediction_report(
        scaled_targets,
        target_embeddings,
        min_cosine=0.99,
        max_mse=0.01,
    )
    assert not collapsed.predicts_target, (
        "A collapsed predictor can have finite outputs but should fail the cosine target check."
    )
    assert not scaled.predicts_target and scaled.mean_cosine > 0.99, (
        "Cosine alone is not enough: a scaled target should fail the MSE threshold."
    )
    print("All tests in `test_jepa_prediction_report_rejects_collapse_and_bad_mse` passed!")


def test_collapse_diagnostics_smoke_test(
    collapse_diagnostics_smoke_test: Callable | None = None,
):
    collapse_diagnostics_smoke_test = (
        collapse_diagnostics_smoke_test or _solutions().collapse_diagnostics_smoke_test
    )
    result = collapse_diagnostics_smoke_test()
    assert result["structured"]["non_collapsed"], (
        "Structured toy features should pass the non-collapse diagnostic."
    )
    assert result["collapsed_control_rejected"], (
        "Identical features should be rejected as collapsed representations."
    )
    print("All tests in `test_collapse_diagnostics_smoke_test` passed!")


def test_collapse_diagnostics_rejects_identical_features(
    collapse_diagnostics_report_fn: Callable | None = None,
):
    report_fn = collapse_diagnostics_report_fn or collapse_diagnostics_report
    collapsed = report_fn(
        t.ones(8, 4),
        min_feature_std=0.1,
        min_effective_rank=2.0,
    )
    assert collapsed.finite_features, (
        "The collapsed-control tensor should still be finite; the failure should come from collapse."
    )
    assert collapsed.feature_std == 0.0, (
        "Identical toy features should have exactly zero feature variance."
    )
    assert not collapsed.non_collapsed, (
        "A representation with zero variance is white-noise-equivalent evidence for this section."
    )
    print("All tests in `test_collapse_diagnostics_rejects_identical_features` passed!")


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


def test_state_probe_control_smoke_test(
    state_probe_control_smoke_test: Callable | None = None,
):
    state_probe_control_smoke_test = (
        state_probe_control_smoke_test or _solutions().state_probe_control_smoke_test
    )
    result = state_probe_control_smoke_test()
    assert result["probe"]["predicts_state"], (
        "The aligned toy probe should predict held-out world labels."
    )
    assert result["shuffled_control_rejected"], (
        "The same logits with shuffled labels should fail; otherwise the probe is not evidence."
    )
    assert result["accuracy_margin"] >= 1.0, (
        "The toy probe should have a visible margin over the shuffled-label control."
    )
    print("All tests in `test_state_probe_control_smoke_test` passed!")


def test_state_probe_report_rejects_shuffled_labels(
    world_state_probe_report_fn: Callable | None = None,
):
    report_fn = world_state_probe_report_fn or world_state_probe_report
    logits = t.tensor([[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0]])
    labels = t.tensor([0, 0, 1, 1])
    shuffled_labels = t.tensor([1, 1, 0, 0])
    aligned = report_fn(logits, labels, min_accuracy=0.9)
    shuffled = report_fn(logits, shuffled_labels, min_accuracy=0.9)
    assert aligned.predicts_state, (
        "The aligned probe report should pass before the shuffled-label control is tested."
    )
    assert shuffled.accuracy == 0.0 and not shuffled.predicts_state, (
        "Probe evidence must disappear when labels are deliberately shuffled."
    )
    print("All tests in `test_state_probe_report_rejects_shuffled_labels` passed!")


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


def test_transition_report_rejects_missing_action_delta():
    failed = transition_consistency_report(
        state_embeddings=t.tensor([[1.0, 0.0], [0.0, 1.0]]),
        action_deltas=t.zeros(2, 2),
        next_state_embeddings=t.tensor([[0.0, 1.0], [1.0, 0.0]]),
        min_cosine=0.99,
    )
    assert failed.mean_cosine == 0.0, (
        "With zero action deltas and orthogonal next states, the toy transition cosine should be zero."
    )
    assert not failed.transition_consistent, (
        "A static latent state should not pass a transition test when the action delta is missing."
    )
    print("All tests in `test_transition_report_rejects_missing_action_delta` passed!")


def test_rollout_control_smoke_test(rollout_control_smoke_test: Callable | None = None):
    rollout_control_smoke_test = (
        rollout_control_smoke_test or _solutions().rollout_control_smoke_test
    )
    result = rollout_control_smoke_test()
    assert result["rollout"]["rollout_passes"], (
        "The action-conditioned toy rollout should beat copy and shuffled-action baselines."
    )
    assert result["copy_and_shuffled_controls_rejected"], (
        "A rollout that does not beat copy or shuffled actions should be rejected."
    )
    print("All tests in `test_rollout_control_smoke_test` passed!")


def test_latent_rollout_report_rejects_copy_and_shuffled_controls(
    latent_rollout_report_fn: Callable | None = None,
):
    report_fn = latent_rollout_report_fn or latent_rollout_report
    good = report_fn(
        rollout_loss=0.10,
        copy_baseline_loss=1.0,
        shuffled_action_loss=0.9,
        max_rollout_to_copy=0.8,
        max_rollout_to_shuffled=0.8,
    )
    bad = report_fn(
        rollout_loss=0.75,
        copy_baseline_loss=0.8,
        shuffled_action_loss=0.7,
        max_rollout_to_copy=0.8,
        max_rollout_to_shuffled=0.8,
    )
    assert good.rollout_passes, (
        "The positive rollout toy case should beat copy and shuffled-action baselines."
    )
    assert not bad.rollout_passes, (
        "A latent rollout is not a world-model result unless it beats copy and shuffled-action controls."
    )
    print("All tests in `test_latent_rollout_report_rejects_copy_and_shuffled_controls` passed!")


def test_causal_latent_patch_report_rejects_random_control(
    causal_latent_patch_report_fn: Callable | None = None,
):
    report_fn = causal_latent_patch_report_fn or causal_latent_patch_report
    targeted = report_fn(
        t.tensor([0.82, 0.88, 0.91]),
        t.tensor([0.01, 0.03, 0.02]),
        min_object_patch_effect=0.5,
        min_patch_random_gap=0.4,
    )
    matched_random = report_fn(
        t.tensor([0.82, 0.88, 0.91]),
        t.tensor([0.75, 0.84, 0.87]),
        min_object_patch_effect=0.5,
        min_patch_random_gap=0.4,
    )
    assert targeted.causal_patch_passes and targeted.patch_random_gap > 0.8
    assert not matched_random.causal_patch_passes, (
        "A target patch is not specific when a same-size random patch has the same effect."
    )
    print(
        "All tests in `test_causal_latent_patch_report_rejects_random_control` passed!"
    )


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


def test_object_permanence_control_smoke_test(
    object_permanence_control_smoke_test: Callable | None = None,
):
    object_permanence_control_smoke_test = (
        object_permanence_control_smoke_test
        or _solutions().object_permanence_control_smoke_test
    )
    result = object_permanence_control_smoke_test()
    assert result["preserved"]["preserves_occluded_object"], (
        "The positive toy case should keep occluded-object scores above absent controls."
    )
    assert result["absent_like_rejected"], (
        "Occluded scores close to absent scores should be rejected."
    )
    assert result["different_object_rejected"], (
        "Different-object similarity should not count as object permanence."
    )
    print("All tests in `test_object_permanence_control_smoke_test` passed!")


def test_object_permanence_report_rejects_absent_and_different_object_controls(
    object_permanence_report_fn: Callable | None = None,
):
    report_fn = object_permanence_report_fn or object_permanence_report
    absent_like = report_fn(
        visible_scores=t.tensor([0.95, 0.9]),
        occluded_scores=t.tensor([0.52, 0.48]),
        absent_scores=t.tensor([0.46, 0.44]),
        min_occluded_score=0.6,
        min_absent_gap=0.4,
    )
    different_object = report_fn(
        visible_scores=t.tensor([0.95, 0.9]),
        occluded_scores=t.tensor([0.78, 0.76]),
        absent_scores=t.tensor([0.72, 0.7]),
        min_occluded_score=0.6,
        min_absent_gap=0.4,
    )
    assert not absent_like.preserves_occluded_object, (
        "Occluded evidence should fail when it is effectively absent-object evidence."
    )
    assert not different_object.preserves_occluded_object, (
        "A high score that is not specific to the original object should fail the absent/different-object control."
    )
    print(
        "All tests in "
        "`test_object_permanence_report_rejects_absent_and_different_object_controls` passed!"
    )


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
    assert result["collapse"]["collapsed_control_rejected"], (
        "The notebook contract should include a collapsed-representation negative control."
    )
    assert result["state_probe_control"]["shuffled_control_rejected"], (
        "The notebook contract should include a shuffled-label probe control."
    )
    assert result["rollout_control"]["copy_and_shuffled_controls_rejected"], (
        "The notebook contract should include copy and shuffled-action rollout controls."
    )
    assert result["object_permanence"]["preserves_occluded_object"], (
        "The notebook contract should include a passing object-permanence report."
    )
    assert result["object_permanence_control"]["different_object_rejected"], (
        "The notebook contract should include a different-object permanence control."
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
    assert gpu["vjepa2_world_feature_shape"] == [200, 1024], (
        "The report should keep the generated-video world-control suite shape explicit."
    )
    assert gpu["vjepa2_token_feature_shape"] == [200, 144, 1024], (
        "The report should keep the V-JEPA token grid available for patching checks."
    )
    assert gpu["peak_vram_gb"] < 24.0, (
        "The accepted local V-JEPA 2 path should fit the requested 24GB machine."
    )
    print("All tests in `test_committed_verification_report_vjepa_world_controls` passed!")


def validate_vjepa2_signature_visual_payload(signature_result: dict) -> None:
    """Validate the real videos, latents, and patch effects plotted by the notebook."""

    assert signature_result["preflight_passed"]
    payload = signature_result.get("visual_payload")
    assert payload is not None, "The live signature result should retain visual evidence."
    cases = payload["cases"]
    assert [case["case_id"] for case in cases] == [
        "red_square_right",
        "blue_circle_right",
    ]
    for case in cases:
        assert case["frame_indices"] == [0, 3, 4, 7]
        assert case["action"] == (8, 0)
        for key in ("visible_frames", "next_frames", "occluded_frames", "absent_frames"):
            frames = case[key]
            assert frames.shape == (4, 3, 96, 96)
            assert t.isfinite(frames).all()
        assert not t.equal(case["visible_frames"], case["absent_frames"])

    assert payload["pooled_features"].shape == (200, 1024)
    assert payload["labels"].shape == payload["x_buckets"].shape == (200,)
    object_effects = payload["object_patch_effects"]
    random_effects = payload["random_patch_effects"]
    assert object_effects.shape == random_effects.shape == (16,)
    assert object_effects.mean().item() >= random_effects.mean().item() + 0.4
    assert signature_result["state_probe_margin_over_random"] >= 0.2
    assert signature_result["latent_rollout_passed"]
    assert signature_result["real_latent_object_permanence_passed"]
    assert signature_result["causal_latent_patching_passed"]
    print("All tests in `validate_vjepa2_signature_visual_payload` passed!")


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
    assert "Expected output" in source, (
        "The learner notebook should show expected outputs, not just hidden assertions."
    )
    assert "Help - " in source, (
        "The learner notebook should include interpretation/help dropdowns."
    )
    assert "Signature Result" in source, (
        "The learner notebook should expose the section's signature result."
    )
    assert "What this does not show" in source, (
        "The learner notebook should state claim boundaries."
    )
    assert "## Try It Yourself" in source, (
        "The learner notebook should let students perturb a real video or control."
    )
    assert "run_vjepa2_world_model_signature_result" in source, (
        "The learner notebook should generate the live V-JEPA 2 evidence panel."
    )
    assert "jepa_world_model_live_signature.png" in source, (
        "The learner notebook should display the real video-latent signature figure."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
