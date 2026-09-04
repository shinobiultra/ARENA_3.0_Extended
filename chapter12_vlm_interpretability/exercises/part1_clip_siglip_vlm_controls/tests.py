import ast
from collections.abc import Callable, Mapping
import json
from pathlib import Path

import torch as t


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


def test_extract_contrastive_embeddings_normalizes_both_towers(
    extract_contrastive_embeddings: Callable[..., tuple[t.Tensor, t.Tensor]] | None = None,
):
    solutions = _solutions()
    extract_contrastive_embeddings = (
        extract_contrastive_embeddings or solutions.extract_contrastive_embeddings
    )

    class ExactDualEncoder(t.nn.Module):
        def get_image_features(self, *, pixel_values: t.Tensor) -> t.Tensor:
            return pixel_values @ t.tensor([[3.0, 0.0], [0.0, 4.0]])

        def get_text_features(
            self,
            *,
            input_ids: t.Tensor,
            attention_mask: t.Tensor | None = None,
        ) -> t.Tensor:
            if attention_mask is None:
                raise AssertionError("The text tower should receive the attention mask.")
            return input_ids.float() * attention_mask.float()

    batch = {
        "pixel_values": t.eye(2),
        "input_ids": t.tensor([[3, 0], [0, 4]]),
        "attention_mask": t.ones(2, 2),
    }
    image, text = extract_contrastive_embeddings(ExactDualEncoder(), batch)
    assert t.allclose(image, t.eye(2)), (
        f"The exact image tower should normalize to the basis vectors; got {image}."
    )
    assert t.allclose(text, t.eye(2)), (
        f"The exact text tower should normalize to the basis vectors; got {text}."
    )
    print("All tests in `test_extract_contrastive_embeddings_normalizes_both_towers` passed!")


def test_bidirectional_retrieval_metrics_uses_both_directions(
    bidirectional_retrieval_metrics: Callable[..., dict[str, object]] | None = None,
):
    solutions = _solutions()
    bidirectional_retrieval_metrics = (
        bidirectional_retrieval_metrics or solutions.bidirectional_retrieval_metrics
    )
    image = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    text = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = bidirectional_retrieval_metrics(image, text, logit_scale=5.0)
    assert result["image_to_text_accuracy"] == 1.0, (
        "Each image should retrieve its paired caption in the exact basis oracle."
    )
    assert result["text_to_image_accuracy"] == 1.0, (
        "Each caption should retrieve its paired image in the exact basis oracle."
    )
    _assert_close(
        float(result["mean_positive_margin"]),
        5.0,
        msg="The diagonal-to-off-diagonal retrieval margin is exact.",
    )
    print("All tests in `test_bidirectional_retrieval_metrics_uses_both_directions` passed!")


def test_hook_cache_and_hidden_patch_are_causal(
    capture_module_output: Callable[..., tuple[object, t.Tensor]] | None = None,
    patch_hidden_token_rows: Callable[..., t.Tensor] | None = None,
):
    solutions = _solutions()
    capture_module_output = capture_module_output or solutions.capture_module_output
    patch_hidden_token_rows = patch_hidden_token_rows or solutions.patch_hidden_token_rows
    module = t.nn.Linear(2, 2, bias=False)
    with t.no_grad():
        module.weight.copy_(t.eye(2))
    values = t.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]])
    result, cached = capture_module_output(module, lambda: module(values))
    assert isinstance(result, t.Tensor) and t.allclose(cached, values), (
        "The hook should capture the real module output from exactly one forward pass."
    )
    corrupt = -values
    patched = patch_hidden_token_rows(cached, corrupt, (1,))
    assert t.allclose(patched[:, 1], corrupt[:, 1]), (
        "The selected hidden token should be replaced by its corrupt counterpart."
    )
    assert t.allclose(patched[:, (0, 2)], cached[:, (0, 2)]), (
        "Unselected hidden tokens should remain exactly clean."
    )
    assert not module._forward_hooks, "Hook handles must be removed after capture."
    print("All tests in `test_hook_cache_and_hidden_patch_are_causal` passed!")


def test_causal_patch_metrics_separates_object_from_controls(
    causal_patch_metrics: Callable[..., dict[str, object]] | None = None,
):
    solutions = _solutions()
    causal_patch_metrics = causal_patch_metrics or solutions.causal_patch_metrics
    clean = t.tensor([[6.0, 1.0], [0.0, 5.0]])
    corrupt = t.tensor([[0.0, 5.0], [6.0, 0.0]])
    result = causal_patch_metrics(
        clean,
        corrupt,
        {
            "object": corrupt,
            "background": clean,
            "same_size_random": clean,
            "full_sequence": corrupt,
        },
        target_indices=t.tensor([0, 1]),
        counterfactual_indices=t.tensor([1, 0]),
    )
    rows = {row["condition"]: row for row in result["rows"]}
    assert rows["object"]["all_flip"] and rows["full_sequence"]["all_flip"], (
        "Object and full-sequence patches should flip both exact-oracle examples."
    )
    assert not rows["background"]["all_flip"] and not rows["same_size_random"]["all_flip"], (
        "Matched control patches should preserve both exact-oracle examples."
    )
    assert rows["object"]["mean_effect"] > rows["background"]["mean_effect"], (
        "The causal object effect should exceed the background control."
    )
    print("All tests in `test_causal_patch_metrics_separates_object_from_controls` passed!")


def test_trim_generated_tokens_removes_prompt_prefix(
    trim_generated_tokens: Callable[..., list[t.Tensor]] | None = None,
):
    solutions = _solutions()
    trim_generated_tokens = trim_generated_tokens or solutions.trim_generated_tokens
    prompts = t.tensor([[10, 11, 12], [20, 21, 0]])
    generated = t.tensor([[10, 11, 12, 31, 32], [20, 21, 0, 41, 42]])
    trimmed = trim_generated_tokens(prompts, generated)
    assert [row.tolist() for row in trimmed] == [[31, 32], [41, 42]], (
        "Qwen answer decoding should remove the entire padded prompt width from each row."
    )
    print("All tests in `test_trim_generated_tokens_removes_prompt_prefix` passed!")


def test_contrastive_smoke_test(
    contrastive_smoke_test: Callable[[], dict] | None = None,
):
    contrastive_smoke_test = (
        contrastive_smoke_test or _solutions().contrastive_smoke_test
    )
    result = contrastive_smoke_test()
    assert result["image_to_text_accuracy"] == 1.0, (
        "Identity image embeddings should retrieve their matching text embeddings."
    )
    assert result["text_to_image_accuracy"] == 1.0, (
        "Identity text embeddings should retrieve their matching image embeddings."
    )
    _assert_close(
        result["mean_positive_margin"],
        5.0,
        msg="The positive-pair margin should compare the diagonal to the strongest negative.",
    )
    assert result["aligned"], (
        "The report should pass only when both retrieval directions and the margin pass."
    )
    print("All tests in `test_contrastive_smoke_test` passed!")


def test_siglip_smoke_test(siglip_smoke_test: Callable[[], dict] | None = None):
    siglip_smoke_test = siglip_smoke_test or _solutions().siglip_smoke_test
    result = siglip_smoke_test()
    signed_margins = t.tensor([4.0, 4.0, 3.0, 3.0])
    expected_loss = t.nn.functional.softplus(-signed_margins).mean().item()
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
    assert _as_list(result["token_scores"]) == [0.0, 3.0, 2.0, 0.0], (
        "Attribution scores should be dot products against the normalized text direction."
    )
    assert _as_list(result["top_token_indices"]) == [1, 2], (
        "The two visual tokens aligned with the text direction should be selected."
    )
    _assert_close(
        result["top_token_mass"],
        1.0,
        msg="Top-token mass should be computed over positive attribution mass.",
    )
    assert result["localized"], (
        "A localized object claim should put enough attribution mass on the top tokens."
    )
    print("All tests in `test_token_attribution_smoke_test` passed!")


def test_contrastive_report_rejects_wrong_pairing():
    solutions = _solutions()
    logits = t.tensor(
        [
            [0.0, 4.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 5.0],
        ]
    )
    report = solutions.contrastive_alignment_report(
        logits,
        min_accuracy=1.0,
        min_positive_margin=1.0,
    )
    assert report.image_to_text_accuracy < 1.0, (
        "Image-to-text accuracy should drop when images prefer off-diagonal captions."
    )
    assert report.text_to_image_accuracy < 1.0, (
        "Text-to-image accuracy should drop when captions prefer off-diagonal images."
    )
    assert not report.aligned, (
        "The report must fail when the highest-scoring pairs are off diagonal."
    )
    print("All tests in `test_contrastive_report_rejects_wrong_pairing` passed!")


def test_siglip_pairwise_loss_rejects_shape_mismatch():
    solutions = _solutions()
    try:
        solutions.siglip_pairwise_loss(t.zeros(2, 2), t.zeros(2, 3))
    except ValueError as exc:
        assert "same shape" in str(exc), (
            "The error should explain that SigLIP logits and labels need matching shapes."
        )
    else:
        raise AssertionError("SigLIP loss should reject mismatched logits and labels.")
    print("All tests in `test_siglip_pairwise_loss_rejects_shape_mismatch` passed!")


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


def test_toy_clip_rendered_batch_is_visible_and_nonblank():
    solutions = _solutions()
    batch = solutions.build_toy_clip_batch(
        colors=("red", "blue"),
        shapes=("square", "circle"),
        image_size=48,
    )
    assert batch.image_tensors.shape == (4, 3, 48, 48), (
        "The toy CLIP batch should produce a visible image grid, not just feature tensors."
    )
    nonwhite_pixels = batch.image_tensors.lt(0.98).any(dim=1).sum(dim=(1, 2))
    assert nonwhite_pixels.min().item() > 200, (
        "Each rendered image should contain a substantial colored object."
    )
    assert batch.captions[0] == "a red square", (
        "Captions should be concrete image-text pairs the student can inspect."
    )
    assert batch.image_features.shape[0] == batch.text_features.shape[0] == 4, (
        "Rendered images and captions should have paired feature rows."
    )
    print("All tests in `test_toy_clip_rendered_batch_is_visible_and_nonblank` passed!")


def test_toy_clip_signature_result_has_controls(
    toy_clip_signature_result: Callable[..., dict] | None = None,
):
    toy_clip_signature_result = (
        toy_clip_signature_result or _solutions().toy_clip_signature_result
    )
    result = toy_clip_signature_result(device="cpu", steps=250, seed=0)
    assert result["scene_count"] == 12, (
        "The signature result should use a nontrivial colored-shape grid."
    )
    assert result["image_grid_shape"] == [12, 3, 48, 48], (
        "The signature result should expose the image grid dimensions."
    )
    assert result["loss_end"] < 0.1 * result["loss_start"], (
        "The tiny CLIP training loss should visibly fall."
    )
    assert result["image_to_text_accuracy"] == 1.0, (
        "Trained toy CLIP should retrieve the correct caption for every image."
    )
    assert result["text_to_image_accuracy"] == 1.0, (
        "Trained toy CLIP should retrieve the correct image for every caption."
    )
    assert result["mean_positive_margin"] > 1.0, (
        "The paired retrieval diagonal should beat its strongest distractors."
    )
    assert all(row["target_rank"] == 1 for row in result["retrieval_rows"]), (
        "Every visible retrieval row should put the target caption at rank 1."
    )
    assert result["random_caption_accuracy"] <= 0.25, (
        "Deranged-caption retrieval should fail instead of looking like a pass."
    )
    assert not result["random_caption_aligned"], (
        "The random-caption control should explicitly fail the alignment report."
    )
    assert result["conflict_caption_accuracy"] <= 0.25, (
        "Counterfactual color captions should fail as a conflict control."
    )
    assert result["control_claim_passed"], (
        "The learner-facing claim should require success plus failed controls."
    )
    print("All tests in `test_toy_clip_signature_result_has_controls` passed!")


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
    assert result["toy_clip_signature"]["control_claim_passed"], (
        "The notebook contract should include the visible toy CLIP signature result."
    )
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


def test_clip_siglip_core_notebook_contract(
    run_smoke_test: Callable[..., dict] | None = None,
):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["toy_clip_signature"]["control_claim_passed"], (
        "The core lesson should be grounded in a visible toy CLIP result."
    )
    assert result["contrastive"]["aligned"], (
        "The core lesson should include the contrastive alignment check."
    )
    assert result["siglip"]["loss"] < 0.05, (
        "The core lesson should include the SigLIP pairwise-loss check."
    )
    assert result["synthetic_scene_schema"]["has_counterfactual_answers"], (
        "The core lesson should expose counterfactual image-text labels."
    )
    assert result["synthetic_scene_schema"]["has_spurious_text_control"], (
        "The core lesson should expose misleading text controls."
    )
    assert result["token_attribution"]["localized"], (
        "The core lesson should include localized visual-token attribution."
    )
    print("All tests in `test_clip_siglip_core_notebook_contract` passed!")


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
    required = [
        "def extract_contrastive_embeddings",
        "def bidirectional_retrieval_metrics",
        "def bbox_to_visual_tokens",
        "def capture_module_output",
        "def patch_hidden_token_rows",
        "def run_with_activation_patch",
        "def causal_patch_metrics",
        "def generate_qwen_answers",
        "run_real_contrastive_study",
        "Set RUN_REAL_MODELS=True to regenerate",
    ]
    for needle in required:
        assert needle in source, f"The learner notebook is missing the visible method surface: {needle}"
    assert source.count("### Exercise ") == 10, (
        "12.1 should present ten graded exercises from toy ground truth through real-model generation."
    )
    assert source.count("<summary>Expected output</summary>") >= 10, (
        "Every graded exercise should state its expected output."
    )
    assert source.count("<summary>Help -") >= 10, (
        "Every graded exercise should include reasoning-oriented help."
    )
    assert source.count("<summary>Solution</summary>") >= 10, (
        "Every graded exercise should include a visible solution dropdown."
    )
    assert source.count("<summary>Interpretation</summary>") >= 10, (
        "Every graded exercise should explain how to interpret the result."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")


def test_solution_notebook_keeps_real_vlm_methods_visible():
    notebook_path = _section_dir() / "12.1_CLIP_SigLIP_and_VLM_Controls_solutions.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(code_source)
    taught = {
        "clip_contrastive_logits",
        "contrastive_alignment_report",
        "siglip_pairwise_loss",
        "train_toy_clip_projectors",
        "extract_contrastive_embeddings",
        "bidirectional_retrieval_metrics",
        "patch_image_region",
        "bbox_to_visual_tokens",
        "capture_module_output",
        "patch_hidden_token_rows",
        "run_with_activation_patch",
        "causal_patch_metrics",
        "trim_generated_tokens",
        "generate_qwen_answers",
    }
    definitions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert taught <= definitions.keys(), (
        f"Solved notebook is missing visible implementations: {taught - definitions.keys()}"
    )
    for name in taught:
        assert not any(
            isinstance(child, ast.Raise)
            and isinstance(child.exc, ast.Call)
            and isinstance(child.exc.func, ast.Name)
            and child.exc.func.id == "NotImplementedError"
            for child in ast.walk(definitions[name])
        ), f"{name} remains a placeholder in the solved notebook."
    assert "reference_solutions" not in code_source and "reference." not in code_source, (
        "The solved notebook must run taught VLM methods directly, not delegate to solutions.py."
    )
    print("All tests in `test_solution_notebook_keeps_real_vlm_methods_visible` passed!")


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
