from collections.abc import Callable, Mapping
import json
from pathlib import Path

import torch as t

from arena_ext import vlm_interpretability as reference


def _solutions():
    from chapter12_vlm_interpretability.exercises.part1_clip_siglip_vlm_controls import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _verification_report() -> dict:
    return json.loads((_section_dir() / "verification_report.json").read_text())


def _gpu_report() -> dict:
    return _verification_report()["metrics"]["gpu_test"]


def _as_list(value: object) -> list:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)  # type: ignore[arg-type]


def _assert_close(actual: float, expected: float, *, msg: str, atol: float = 1e-6) -> None:
    assert abs(actual - expected) <= atol, f"{msg} Expected {expected}, got {actual}."


def _bbox_overlap_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def test_same_size_random_region_control_is_non_overlapping():
    solutions = _solutions()
    object_bbox = tuple(solutions.SHAPE_OBJECT_BBOX)
    random_bbox = tuple(solutions.SHAPE_RANDOM_CONTROL_BBOX)
    assert (
        object_bbox[2] - object_bbox[0] == random_bbox[2] - random_bbox[0]
        and object_bbox[3] - object_bbox[1] == random_bbox[3] - random_bbox[1]
    ), "The random-region control should match the object patch size."
    assert _bbox_overlap_area(object_bbox, random_bbox) == 0, (
        "The random-region control should not overlap the object patch. "
        f"object={object_bbox}, random={random_bbox}"
    )
    assert solutions._same_size_random_control_bbox(object_bbox, seed=7) != object_bbox, (
        "The deterministic random-control helper should never return the object bbox itself."
    )
    print("All tests in `test_same_size_random_region_control_is_non_overlapping` passed!")


def test_contrastive_smoke_test(
    contrastive_smoke_test: Callable[[], dict] | None = None,
):
    contrastive_smoke_test = (
        contrastive_smoke_test or _solutions().contrastive_smoke_test
    )
    result = contrastive_smoke_test()
    expected = reference.contrastive_alignment_report(
        reference.clip_contrastive_logits(t.eye(3), t.eye(3), logit_scale=5.0),
        min_accuracy=1.0,
        min_positive_margin=4.0,
    )
    assert result["image_to_text_accuracy"] == expected.image_to_text_accuracy, (
        "Identity image embeddings should retrieve their matching text embeddings."
    )
    assert result["text_to_image_accuracy"] == expected.text_to_image_accuracy, (
        "Identity text embeddings should retrieve their matching image embeddings."
    )
    _assert_close(
        result["mean_positive_margin"],
        expected.mean_positive_margin,
        msg="The positive-pair margin should match the independent reference.",
    )
    assert result["aligned"], (
        "The report should pass only when both retrieval directions and the margin pass."
    )
    print("All tests in `test_contrastive_smoke_test` passed!")


def test_siglip_smoke_test(siglip_smoke_test: Callable[[], dict] | None = None):
    siglip_smoke_test = siglip_smoke_test or _solutions().siglip_smoke_test
    result = siglip_smoke_test()
    expected_loss = reference.siglip_pairwise_loss(
        t.tensor([[4.0, -4.0], [-3.0, 3.0]]),
        t.eye(2),
    ).item()
    _assert_close(
        result["loss"],
        expected_loss,
        msg="SigLIP pairwise loss should use positive labels as +1 and negatives as -1.",
    )
    assert result["loss"] < 0.05, (
        "Confident correct positive and negative pairs should produce a low loss."
    )
    print("All tests in `test_siglip_smoke_test` passed!")


def test_token_attribution_smoke_test(
    token_attribution_smoke_test: Callable[[], dict] | None = None,
):
    token_attribution_smoke_test = (
        token_attribution_smoke_test or _solutions().token_attribution_smoke_test
    )
    result = token_attribution_smoke_test()
    expected = reference.visual_token_attribution_report(
        t.tensor([[0.0, 0.0], [3.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        t.tensor([1.0, 0.0]),
        top_k=2,
        min_top_token_mass=0.8,
    )
    assert _as_list(result["top_token_indices"]) == expected.top_token_indices.tolist(), (
        "The two visual tokens aligned with the text direction should be selected."
    )
    _assert_close(
        result["top_token_mass"],
        expected.top_token_mass,
        msg="Top-token mass should be computed over positive attribution mass.",
    )
    assert result["localized"], (
        "A localized object claim should put enough attribution mass on the top tokens."
    )
    print("All tests in `test_token_attribution_smoke_test` passed!")


def test_hallucination_smoke_test(
    hallucination_smoke_test: Callable[[], dict] | None = None,
):
    hallucination_smoke_test = (
        hallucination_smoke_test or _solutions().hallucination_smoke_test
    )
    result = hallucination_smoke_test()
    _assert_close(
        result["text_prior_gap"],
        0.6,
        msg="Text-prior gap should be text_prior_score minus visual_evidence_score.",
    )
    assert result["flags_hallucination"], (
        "High object score plus weak visual evidence and strong text prior should be flagged."
    )
    print("All tests in `test_hallucination_smoke_test` passed!")


def test_arbitration_smoke_test(
    arbitration_smoke_test: Callable[[], dict] | None = None,
):
    arbitration_smoke_test = (
        arbitration_smoke_test or _solutions().arbitration_smoke_test
    )
    result = arbitration_smoke_test()
    _assert_close(
        result["visual_margin"],
        0.6,
        msg="Visual margin should compare the visual answer against the text-prior answer.",
    )
    assert result["trusts_visual_evidence"], (
        "The controlled conflict should pass only when the visual answer wins by the margin."
    )
    print("All tests in `test_arbitration_smoke_test` passed!")


def test_synthetic_scene_schema_smoke_test(
    synthetic_scene_schema_smoke_test: Callable[[], dict] | None = None,
):
    synthetic_scene_schema_smoke_test = (
        synthetic_scene_schema_smoke_test
        or _solutions().synthetic_scene_schema_smoke_test
    )
    result = synthetic_scene_schema_smoke_test()
    first_scene = result["first_scene"]
    assert result["num_scenes"] == 4, (
        "Two colors times two shapes should produce four controlled scenes."
    )
    assert first_scene["question"] == "What color is the cube?", (
        "The scene schema should expose the image-dependent question."
    )
    assert first_scene["answer"] != first_scene["counterfactual_answer"], (
        "Each scene needs a counterfactual answer for controlled evaluation."
    )
    assert result["has_spurious_text_control"], (
        "The scene table should include misleading text controls."
    )
    assert result["has_counterfactual_answers"], (
        "Every generated scene should expose a counterfactual label."
    )
    print("All tests in `test_synthetic_scene_schema_smoke_test` passed!")


def test_clothing_geometry_smoke_test(
    clothing_geometry_smoke_test: Callable[..., dict] | None = None,
):
    clothing_geometry_smoke_test = (
        clothing_geometry_smoke_test or _solutions().clothing_geometry_smoke_test
    )
    result = clothing_geometry_smoke_test()
    assert result["scene_count"] == 8, (
        "The clothing ladder should cover garment, color, and style factors."
    )
    assert result["first_scene"]["question"] == "What color is the formal shirt?", (
        "The first clothing scene should ask the expected color question."
    )
    assert result["has_spurious_text_control"], (
        "Clothing scenes should include corrupted text-prior controls."
    )
    assert result["garment_accuracy"] == 1.0, (
        "Held-out garment type should be linearly recoverable in the toy geometry."
    )
    assert result["color_accuracy"] == 1.0, (
        "Held-out color should be linearly recoverable from image factors."
    )
    assert result["style_accuracy"] == 1.0, (
        "Held-out style should be linearly recoverable in the toy geometry."
    )
    assert result["text_prior_color_agreement"] == 0.0, (
        "Image-factor predictions should reject the corrupted text-prior colors."
    )
    assert result["random_color_agreement"] == 0.5, (
        "Image-factor predictions should reject the deterministic permuted-label control."
    )
    assert result["random_labels_distinct_from_text_prior"], (
        "The random-label control must be distinct from the flipped text-prior control."
    )
    assert result["random_labels_distinct_from_true_labels"], (
        "The random-label control must be distinct from the true color labels."
    )
    assert result["predicts_clothing_factors"], (
        "All three image factors should pass the required accuracy threshold."
    )
    assert result["rejects_text_prior"] and result["rejects_random_labels"], (
        "A usable VLM geometry control must reject text-prior and random labels."
    )
    print("All tests in `test_clothing_geometry_smoke_test` passed!")


def test_controlled_baselines_smoke_test(
    controlled_baselines_smoke_test: Callable[[], dict] | None = None,
):
    controlled_baselines_smoke_test = (
        controlled_baselines_smoke_test or _solutions().controlled_baselines_smoke_test
    )
    result = controlled_baselines_smoke_test()
    assert result["joint_accuracy"] == 1.0, (
        "The joint image-text path should solve the image-dependent labels."
    )
    assert result["image_only_accuracy"] == 1.0, (
        "The image-only baseline should solve this controlled visual task."
    )
    assert result["text_only_accuracy"] <= 0.5, (
        "The text-only prior should fail on image-dependent questions."
    )
    assert result["joint_beats_text_only"], (
        "Joint VLM evidence should beat the text-only prior."
    )
    assert result["text_only_fails_image_questions"], (
        "A passing control should explicitly mark text-only failure."
    )
    print("All tests in `test_controlled_baselines_smoke_test` passed!")


def test_visual_region_patch_smoke_test(
    visual_region_patch_smoke_test: Callable[..., dict] | None = None,
):
    visual_region_patch_smoke_test = (
        visual_region_patch_smoke_test or _solutions().visual_region_patch_smoke_test
    )
    result = visual_region_patch_smoke_test()
    assert result["object_patch_effect"] > result["background_patch_effect"], (
        "Patching object tokens should matter more than patching background tokens."
    )
    assert result["object_patch_effect"] > result["random_patch_effect"], (
        "Patching object tokens should beat a same-size random-token control."
    )
    assert result["object_beats_background"], (
        "The object-vs-background patch gap should pass the configured threshold."
    )
    assert result["object_beats_random"], (
        "The object-vs-random patch gap should pass the configured threshold."
    )
    assert result["random_control_same_size"], (
        "The random-token patch control should use the same number of rows as the object patch."
    )
    assert result["flips_answer"], (
        "Replacing the object-region contribution should flip the target margin."
    )
    print("All tests in `test_visual_region_patch_smoke_test` passed!")


def test_visual_sequence_patch_smoke_test(
    visual_sequence_patch_smoke_test: Callable[..., dict] | None = None,
):
    visual_sequence_patch_smoke_test = (
        visual_sequence_patch_smoke_test
        or _solutions().visual_sequence_patch_smoke_test
    )
    result = visual_sequence_patch_smoke_test()
    assert result["object_patch_flips_answer"], (
        "Object-token activation patching should flip the target/counterfactual margin."
    )
    assert result["background_patch_preserves_answer"], (
        "Background-token activation patching should preserve the clean answer."
    )
    assert result["random_patch_preserves_answer"], (
        "Same-size random-token activation patching should preserve the clean answer."
    )
    assert result["full_sequence_patch_flips_answer"], (
        "Full visual-sequence patching should flip to the corrupt answer."
    )
    assert result["full_sequence_patch_matches_corrupt"], (
        "Full visual-sequence patching should match the corrupt visual sequence margin."
    )
    assert result["passes_activation_patching_controls"], (
        "The hidden-token patching report should require object, background, random, "
        "and full-sequence controls."
    )
    print("All tests in `test_visual_sequence_patch_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable[..., dict] | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["contrastive"]["aligned"], (
        "The notebook contract should include the contrastive alignment check."
    )
    assert result["siglip"]["loss"] < 0.05, (
        "The notebook contract should include the SigLIP pairwise-loss check."
    )
    assert result["token_attribution"]["localized"], (
        "The notebook contract should include localized visual-token attribution."
    )
    assert result["hallucination"]["flags_hallucination"], (
        "The notebook contract should include the hallucination control."
    )
    assert result["arbitration"]["trusts_visual_evidence"], (
        "The notebook contract should include modality arbitration."
    )
    assert result["synthetic_scene_schema"]["has_counterfactual_answers"], (
        "The notebook contract should include counterfactual scene labels."
    )
    assert result["clothing_geometry"]["predicts_clothing_factors"], (
        "The notebook contract should include clothing factor geometry."
    )
    assert result["controlled_baselines"]["joint_beats_text_only"], (
        "The notebook contract should include image-grounded baselines."
    )
    assert result["visual_region_patch"]["object_beats_background"], (
        "The notebook contract should include object-region patch controls."
    )
    assert result["visual_sequence_patch"]["passes_activation_patching_controls"], (
        "The notebook contract should include hidden visual-token patch controls."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_exercise_notebook_declares_full_verification_contract():
    notebook_path = (
        _section_dir() / "12.1_CLIP_SigLIP_and_VLM_Controls_exercises.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "REQUIRES_GPU = True" in source, (
        "The learner notebook should not advertise CPU-only scope for this GT-1 VLM section."
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
    assert "test_committed_verification_report_real_model_controls" in source, (
        "The learner notebook should end by checking the committed real-model report."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")


def test_committed_verification_report_real_model_controls(
    report: Mapping[str, object] | None = None,
):
    report = dict(report or _verification_report())
    gpu = report["metrics"]["gpu_test"]  # type: ignore[index]
    assert report["accepted"] and report["tests_passed"], (
        "The committed report should be accepted and should have passed tests."
    )
    assert report["gt_tier"] == "GT-1", (
        "12.1 should stay scoped to the declared GT-1 evidence contract."
    )
    assert not report["known_failures"], (
        "The committed verification report should not hide known failures."
    )
    assert gpu["cuda_available"], "The committed report should come from a CUDA run."
    assert gpu["real_clip_rendered_shape_preflight_passed"], (
        "Pinned CLIP rendered-shape retrieval and object patching should pass."
    )
    assert gpu["real_siglip_rendered_shape_preflight_passed"], (
        "Pinned SigLIP rendered-shape retrieval and object patching should pass."
    )
    assert gpu["real_qwen25_vl_generation_preflight_passed"], (
        "Pinned Qwen2.5-VL rendered-shape generation should pass."
    )
    assert gpu["real_qwen25_vl_answers"] == gpu["real_qwen25_vl_expected_answers"], (
        "The generative VLM check should answer the rendered shape questions exactly."
    )
    assert gpu["object_beats_background"] and gpu["object_beats_random"], (
        "The synthetic patching ladder should beat background and random controls."
    )
    assert gpu["random_patch_same_size"], (
        "The synthetic random-token patch control should be the same size as the object patch."
    )
    assert gpu["clothing_random_labels_distinct_from_text_prior"], (
        "The committed clothing control should not reuse the flipped text-prior labels as random labels."
    )
    assert gpu["real_clip_object_beats_random"] and gpu["real_siglip_object_beats_random"], (
        "Real CLIP/SigLIP object patches should beat same-size random-region controls."
    )
    assert gpu["real_clip_visual_token_activation_patching_preflight_passed"], (
        "Pinned CLIP should pass real hidden visual-token activation patching."
    )
    assert gpu["real_siglip_visual_token_activation_patching_preflight_passed"], (
        "Pinned SigLIP should pass real hidden visual-token activation patching."
    )
    assert (
        gpu["real_clip_activation_patch_object_flips_answer"]
        and gpu["real_siglip_activation_patch_object_flips_answer"]
    ), "Object-token activation patching should flip both real-model answers."
    assert (
        gpu["real_clip_activation_patch_background_preserves_answer"]
        and gpu["real_siglip_activation_patch_background_preserves_answer"]
    ), "Background-token activation patching should preserve both real-model answers."
    assert (
        gpu["real_clip_activation_patch_random_preserves_answer"]
        and gpu["real_siglip_activation_patch_random_preserves_answer"]
    ), "Same-size random-token activation patching should preserve both answers."
    assert (
        gpu["real_clip_activation_patch_full_sequence_matches_corrupt"]
        and gpu["real_siglip_activation_patch_full_sequence_matches_corrupt"]
    ), "Full visual-sequence activation patching should match the corrupt sequence."
    assert (
        gpu["real_clip_activation_patch_random_control_same_size"]
        and gpu["real_siglip_activation_patch_random_control_same_size"]
    ), "Activation-patching random controls should match object-token count."
    assert (
        not gpu["real_clip_activation_patch_random_control_overlaps_object"]
        and not gpu["real_siglip_activation_patch_random_control_overlaps_object"]
    ), "Activation-patching random controls should not overlap object tokens."
    assert (
        gpu["real_clip_activation_patch_hook_point"] == "vision_model.embeddings"
        and gpu["real_siglip_activation_patch_hook_point"] == "vision_model.embeddings"
    ), "Activation patching should declare the real hidden-token hook point."
    assert (
        gpu["real_clip_random_patch_same_size_as_object"]
        and gpu["real_siglip_random_patch_same_size_as_object"]
    ), "Real CLIP/SigLIP random-region controls should match object patch size."
    assert (
        gpu["real_clip_random_patch_overlap_area"] == 0
        and not gpu["real_clip_random_patch_overlaps_object"]
    ), "Real CLIP random-region control should not overlap the object patch."
    assert (
        gpu["real_siglip_random_patch_overlap_area"] == 0
        and not gpu["real_siglip_random_patch_overlaps_object"]
    ), "Real SigLIP random-region control should not overlap the object patch."
    assert gpu["within_vram_budget"] and gpu["peak_vram_gb"] <= 24.0, (
        "The real-model path should fit inside the declared VRAM budget."
    )
    print("All tests in `test_committed_verification_report_real_model_controls` passed!")
