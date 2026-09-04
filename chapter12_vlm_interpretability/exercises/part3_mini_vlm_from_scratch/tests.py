from __future__ import annotations

import ast
from collections.abc import Callable
import json
from pathlib import Path

import torch as t


def _solutions():
    from chapter12_vlm_interpretability.exercises.part3_mini_vlm_from_scratch import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def test_rendered_scene_is_visible_and_has_bbox(
    render_controlled_scene: Callable[..., tuple[t.Tensor, tuple[int, int, int, int]]] | None = None,
):
    solutions = _solutions()
    render_controlled_scene = render_controlled_scene or solutions.render_controlled_scene
    image, bbox = render_controlled_scene("red", "square", "center")
    assert image.shape == (3, solutions.IMAGE_SIZE, solutions.IMAGE_SIZE), (
        f"A rendered scene should have CHW shape (3, {solutions.IMAGE_SIZE}, "
        f"{solutions.IMAGE_SIZE}); got {tuple(image.shape)}."
    )
    assert bbox == (12, 12, 36, 36), (
        f"The centered object should have bbox (12, 12, 36, 36); got {bbox}."
    )
    nonwhite = image.lt(0.98).any(dim=0).sum().item()
    assert nonwhite == solutions.OBJECT_SIZE * solutions.OBJECT_SIZE, (
        "A square scene should contain the exact colored object area."
    )
    assert image[:, :8, :8].mean().item() > 0.99, (
        "The top-left background should stay white for object-vs-background controls."
    )
    print("All tests in `test_rendered_scene_is_visible_and_has_bbox` passed!")


def test_train_and_heldout_styles_are_disjoint():
    solutions = _solutions()
    train = solutions.build_vqa_batch("train")
    heldout = solutions.build_vqa_batch("heldout")
    train_pairs = {(example.color, example.shape) for example in train.examples}
    heldout_pairs = {(example.color, example.shape) for example in heldout.examples}
    expected_pairs = {
        (color, shape) for color in solutions.COLORS for shape in solutions.SHAPES
    }
    assert train_pairs == heldout_pairs == expected_pairs, (
        "Train and held-out splits should contain the same complete color-shape support; "
        f"train={train_pairs}, heldout={heldout_pairs}, expected={expected_pairs}."
    )
    train_styles = {example.style for example in train.examples}
    heldout_styles = {example.style for example in heldout.examples}
    assert train_styles == {solutions.TRAIN_STYLE}, (
        f"Training examples should use only {solutions.TRAIN_STYLE!r}; got {train_styles}."
    )
    assert heldout_styles == {solutions.HELDOUT_STYLE}, (
        f"Held-out examples should use only {solutions.HELDOUT_STYLE!r}; got {heldout_styles}."
    )
    assert solutions.TRAIN_STYLE != solutions.HELDOUT_STYLE, (
        "Train and held-out rendering styles must differ for the visual-style generalization test."
    )
    assert len(heldout.examples) >= 20, (
        f"The held-out evaluation needs at least 20 examples; got {len(heldout.examples)}."
    )
    train_positions = {example.position for example in train.examples}
    heldout_positions = {example.position for example in heldout.examples}
    expected_positions = set(solutions.POSITIONS)
    assert train_positions == expected_positions, (
        f"Training examples should cover every object position; got {train_positions}."
    )
    assert heldout_positions == expected_positions, (
        f"Held-out examples should cover every object position; got {heldout_positions}."
    )
    print("All tests in `test_train_and_heldout_styles_are_disjoint` passed!")


def test_visual_token_cache_detaches_and_preserves_patch_grid(
    encode_visual_token_cache: Callable[..., t.Tensor] | None = None,
):
    solutions = _solutions()
    encode_visual_token_cache = encode_visual_token_cache or solutions.encode_visual_token_cache
    batch = solutions.build_vqa_batch("train")
    encoder = solutions.FrozenPatchEncoder()
    cache = encode_visual_token_cache(encoder, batch.images)
    expected_shape = (
        len(batch.examples),
        solutions.PATCH_GRID * solutions.PATCH_GRID,
        solutions.VISION_FEATURE_DIM,
    )
    assert cache.shape == expected_shape, (
        f"The visual cache should preserve batch, patch-grid, and feature axes; "
        f"expected {expected_shape}, got {tuple(cache.shape)}."
    )
    assert not cache.requires_grad, "The frozen vision-token cache should be detached."
    object_mass = cache[..., 3]
    assert object_mass.max().item() > 0.9, (
        f"At least one object patch should have occupancy above 0.9; got {object_mass.max().item():.4f}."
    )
    assert object_mass.min().item() == 0.0, (
        f"Pure background patches should have zero occupancy; got {object_mass.min().item():.4f}."
    )
    print("All tests in `test_visual_token_cache_detaches_and_preserves_patch_grid` passed!")


def test_patch_indices_from_bbox_matches_known_grid(
    patch_indices_from_bbox: Callable[..., tuple[int, ...]] | None = None,
):
    solutions = _solutions()
    patch_indices_from_bbox = patch_indices_from_bbox or solutions.patch_indices_from_bbox
    indices = patch_indices_from_bbox((12, 12, 36, 36))
    assert indices == (
        7,
        8,
        9,
        10,
        13,
        14,
        15,
        16,
        19,
        20,
        21,
        22,
        25,
        26,
        27,
        28,
    ), (
        "The center object should map to the 4x4 patch-token block it overlaps."
    )
    top_indices = patch_indices_from_bbox((12, 0, 36, 24))
    expected_top = (1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16)
    assert top_indices == expected_top, (
        f"The top object should map to the expected 3x4 token block; got {top_indices}."
    )
    print("All tests in `test_patch_indices_from_bbox_matches_known_grid` passed!")


def test_patch_visual_tokens_replaces_only_selected_rows(
    patch_visual_tokens: Callable[[t.Tensor, t.Tensor, tuple[int, ...]], t.Tensor] | None = None,
):
    solutions = _solutions()
    patch_visual_tokens = patch_visual_tokens or solutions.patch_visual_tokens
    clean = t.zeros(2, 5, 3)
    corrupt = t.ones(2, 5, 3)
    patched = patch_visual_tokens(clean, corrupt, (1, 3))
    assert t.allclose(patched[:, 1], t.ones(2, 3)), (
        "Selected row 1 should be replaced by the corrupt cache values."
    )
    assert t.allclose(patched[:, 3], t.ones(2, 3)), (
        "Selected row 3 should be replaced by the corrupt cache values."
    )
    assert t.allclose(patched[:, 0], t.zeros(2, 3)), (
        "Unselected row 0 should preserve the clean cache values."
    )
    assert t.allclose(clean, t.zeros_like(clean)), "Patching should not mutate the clean cache."
    try:
        patch_visual_tokens(clean, corrupt, (1, 1))
    except ValueError as exc:
        assert "unique" in str(exc), (
            f"Duplicate-index errors should explain the uniqueness requirement; got {exc!s}."
        )
    else:
        raise AssertionError("Duplicate patch indices should be rejected.")
    print("All tests in `test_patch_visual_tokens_replaces_only_selected_rows` passed!")


def test_multimodal_sequence_has_visual_prefix_and_question_token(
    build_multimodal_sequence: Callable[..., t.Tensor] | None = None,
):
    solutions = _solutions()
    build_multimodal_sequence = build_multimodal_sequence or solutions.build_multimodal_sequence
    model = solutions.MiniVLM(d_model=24, num_heads=4)
    cache = t.randn(3, solutions.PATCH_GRID * solutions.PATCH_GRID, solutions.VISION_FEATURE_DIM)
    question_ids = t.tensor([0, 1, 0])
    sequence = build_multimodal_sequence(model, cache, question_ids)
    expected_shape = (3, solutions.PATCH_GRID * solutions.PATCH_GRID + 1, 24)
    assert sequence.shape == expected_shape, (
        f"The multimodal sequence should append one question token; expected {expected_shape}, "
        f"got {tuple(sequence.shape)}."
    )
    assert not t.allclose(sequence[:, -1], sequence[:, 0]), (
        "The final token should be the question token, not another visual token."
    )
    print("All tests in `test_multimodal_sequence_has_visual_prefix_and_question_token` passed!")


def test_mini_vlm_uses_real_multi_head_cross_attention():
    solutions = _solutions()
    model = solutions.MiniVLM(d_model=24, num_heads=4)
    cache = t.randn(3, solutions.PATCH_GRID**2, solutions.VISION_FEATURE_DIM)
    questions = t.tensor([0, 1, 0])
    logits, activations = model(cache, questions, return_cache=True)
    assert logits.shape == (3, len(solutions.ANSWER_VOCAB)), (
        f"MiniVLM logits should have one row per example and one column per answer; "
        f"got {tuple(logits.shape)}."
    )
    assert model.num_heads == 4 and model.head_dim == 6, (
        f"d_model=24 with four heads should give head_dim=6; got "
        f"num_heads={model.num_heads}, head_dim={model.head_dim}."
    )
    expected_value_shape = (3, solutions.PATCH_GRID**2, 24)
    assert activations["value_0"].shape == expected_value_shape, (
        f"Layer-0 values should retain every visual-token row; got {tuple(activations['value_0'].shape)}."
    )
    assert activations["value_1"].shape == expected_value_shape, (
        f"Layer-1 values should retain every visual-token row; got {tuple(activations['value_1'].shape)}."
    )
    print("All tests in `test_mini_vlm_uses_real_multi_head_cross_attention` passed!")


def test_vqa_accuracy_and_answer_margin(
    vqa_accuracy: Callable[[t.Tensor, t.Tensor], float] | None = None,
    answer_margin: Callable[[t.Tensor, t.Tensor, t.Tensor], t.Tensor] | None = None,
):
    solutions = _solutions()
    vqa_accuracy = vqa_accuracy or solutions.vqa_accuracy
    answer_margin = answer_margin or solutions.answer_margin
    logits = t.tensor([[4.0, 1.0, -1.0], [0.0, 3.0, 2.0]])
    labels = t.tensor([0, 2])
    counters = t.tensor([1, 1])
    accuracy = vqa_accuracy(logits, labels)
    assert accuracy == 0.5, f"The two-example fixture should have accuracy 0.5; got {accuracy}."
    margins = answer_margin(logits, labels, counters)
    assert t.allclose(margins, t.tensor([3.0, -1.0])), (
        f"Target-minus-counterfactual margins should be [3.0, -1.0]; got {margins.tolist()}."
    )
    print("All tests in `test_vqa_accuracy_and_answer_margin` passed!")


def test_toy_ground_truth_patch_report_has_exact_controls(
    toy_ground_truth_patch_report: Callable[..., dict[str, object]] | None = None,
):
    solutions = _solutions()
    toy_ground_truth_patch_report = (
        toy_ground_truth_patch_report or solutions.toy_ground_truth_patch_report
    )
    result = toy_ground_truth_patch_report()
    assert result["clean_margin"] > 0, (
        f"The exact clean oracle should prefer the target answer; got margin {result['clean_margin']}."
    )
    assert result["corrupt_margin"] < 0, (
        f"The exact corrupt oracle should prefer the counterfactual; got margin {result['corrupt_margin']}."
    )
    assert result["object_patch_flips"], (
        "Exact toy object patching should flip the answer margin."
    )
    assert result["background_patch_preserves"], (
        "Exact toy background patching should preserve the clean answer margin."
    )
    assert result["random_patch_preserves"], (
        "Exact toy same-size random-region patching should preserve the clean answer margin."
    )
    assert result["object_beats_background"], (
        "Exact object-token patching should have a larger causal effect than background patching."
    )
    print("All tests in `test_toy_ground_truth_patch_report_has_exact_controls` passed!")


def check_training_and_baselines(training, baselines: dict[str, float]) -> None:
    train_accuracy = (
        float(training["train_accuracy"])
        if isinstance(training, dict)
        else training.train_accuracy
    )
    heldout_accuracy = (
        float(training["heldout_accuracy"])
        if isinstance(training, dict)
        else training.heldout_accuracy
    )
    assert train_accuracy >= 0.95, (
        f"The trained MiniVLM should reach at least 0.95 training accuracy; got {train_accuracy:.4f}."
    )
    assert heldout_accuracy >= 0.95, (
        f"Muted-style held-out accuracy should reach at least 0.95; got {heldout_accuracy:.4f}."
    )
    assert baselines["joint_accuracy"] >= 0.95, (
        f"Joint image-question accuracy should reach at least 0.95; got {baselines['joint_accuracy']:.4f}."
    )
    assert baselines["text_only_accuracy"] <= 0.45, (
        "Question text without pixels should fail image-dependent VQA."
    )
    assert baselines["image_only_accuracy"] <= 0.60, (
        "Pixels without the question should not reliably choose color vs shape answers."
    )
    assert baselines["random_visual_accuracy"] <= 0.50, (
        "Shuffled visual caches should fail image-dependent VQA; "
        f"got accuracy {baselines['random_visual_accuracy']:.4f}."
    )
    print("Training and modality-control checks passed!")


def check_patch_report(report: dict[str, object]) -> None:
    assert report["object_patch_flips"], "Object-token patching should flip the answer."
    assert report["background_patch_preserves"], (
        "A matched background patch should preserve the clean answer."
    )
    assert report["random_region_preserves"], (
        "A same-size random-region patch should preserve the clean answer."
    )
    assert report["full_sequence_matches_corrupt"], (
        "Patching every visual row should reproduce the corrupt forward pass."
    )
    assert report["object_patch_margin"] < 0, (
        f"Object patching should make the signed margin negative; got {report['object_patch_margin']}."
    )
    assert report["background_patch_margin"] > 0, (
        f"Background patching should leave the signed margin positive; got {report['background_patch_margin']}."
    )
    print("Object/background/random/full-sequence patch checks passed!")


def test_trained_mini_vlm_signature_result_has_controls(
    run_signature_result: Callable[..., dict[str, object]] | None = None,
):
    solutions = _solutions()
    run_signature_result = run_signature_result or solutions.run_signature_result
    result = run_signature_result(steps=260, device="cpu")
    baselines = result["baselines"]
    color_patch = result["color_patch"]
    shape_patch = result["shape_patch"]
    assert result["accepted"], "The trained MiniVLM signature result should pass its claim gate."
    check_training_and_baselines(result, baselines)
    check_patch_report(color_patch)
    check_patch_report(shape_patch)
    print("All tests in `test_trained_mini_vlm_signature_result_has_controls` passed!")


def test_exercise_notebook_exposes_arena_learner_surface():
    notebook_path = _section_dir() / "12.3_Mini_VLM_from_Scratch_exercises.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        "By the end of this notebook",
        "## Core Question",
        "## Cold Open",
        "### Exercise - build the visual-token cache",
        "### Exercise - map a bounding box to visual-token indices",
        "### Exercise - exact toy object patching",
        "### Exercise - train and evaluate the MiniVLM",
        "## Signature Result",
        "## Try It Yourself",
        "## Bonus: Hunt an Anomaly",
        "mini_vlm_signature_result.png",
        "mini_vlm_layer_position_patching_heatmap.png",
    ]
    for needle in required:
        assert needle in source, f"Notebook is missing learner-surface marker: {needle}"
    assert source.count("### Exercise -") == 8, "12.3 should have exactly 8 graded exercises."
    assert source.count("<summary>Expected output</summary>") >= 8, (
        "Each graded exercise should include a visible expected-output dropdown."
    )
    assert source.count("<summary>Help -") >= 8, (
        "Each graded exercise should include a reasoning-oriented help dropdown."
    )
    assert source.count("<summary>Interpretation</summary>") >= 8, (
        "Each graded exercise should explain how to interpret its result."
    )
    assert source.count("<summary>Solution</summary>") >= 8, (
        "Each graded exercise should include a full solution dropdown."
    )
    print("All tests in `test_exercise_notebook_exposes_arena_learner_surface` passed!")


def test_solution_notebook_exposes_taught_implementations():
    notebook_path = _section_dir() / "12.3_Mini_VLM_from_Scratch_solutions.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(code)
    taught = {
        "encode_visual_token_cache",
        "build_multimodal_sequence",
        "vqa_accuracy",
        "answer_margin",
        "patch_indices_from_bbox",
        "patch_visual_tokens",
        "toy_ground_truth_patch_report",
        "train_mini_vlm",
        "modality_baseline_report",
        "forward_with_visual_patch",
        "trained_patch_report",
        "patching_effect_heatmap",
    }
    definitions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert taught <= definitions.keys(), f"Missing inline solution bodies: {taught - definitions.keys()}"
    for name in taught:
        node = definitions[name]
        assert not any(
            isinstance(child, ast.Raise)
            and isinstance(child.exc, ast.Call)
            and isinstance(child.exc.func, ast.Name)
            and child.exc.func.id == "NotImplementedError"
            for child in ast.walk(node)
        ), f"{name} is still a placeholder in the solved notebook."
        assert not any(
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "reference"
            and child.attr == name
            for child in ast.walk(node)
        ), f"{name} delegates the taught method to hidden reference code."
    assert "NotImplementedError" not in code, (
        "The solved notebook should not contain any remaining implementation placeholders."
    )
    print("All tests in `test_solution_notebook_exposes_taught_implementations` passed!")
