from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest
import torch as t
from torch import Tensor
import yaml


def _solutions():
    from chapter12_vlm_interpretability.exercises.part2_object_hallucination_arbitration import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _section_dir().parents[2]


def test_build_arbitration_benchmark(
    build_arbitration_benchmark: Callable[..., list] | None = None,
):
    s = _solutions()
    build = build_arbitration_benchmark or s.build_arbitration_benchmark
    examples = build()
    assert len(examples) == 192
    assert {example.condition for example in examples} == set(s.CONDITIONS)
    assert {example.split for example in examples} == {"development", "validation", "test"}
    assert all(sum(example.split == split for example in examples) == 64 for split in s.SPLITS)
    for split in s.SPLITS:
        split_examples = [example for example in examples if example.split == split]
        observed = {(example.target_color, example.target_shape) for example in split_examples}
        assert observed == set(s.TARGETS), "Every split must contain the full color-shape factorial."
        for target in s.TARGETS:
            rows = [
                example
                for example in split_examples
                if (example.target_color, example.target_shape) == target
            ]
            assert sum(row.label for row in rows) == len(rows) // 2
            assert sum(row.prior == "claims_present" for row in rows if row.label == 1) == 1
            assert sum(row.prior == "claims_present" for row in rows if row.label == 0) == 1


def test_encode_prompt_contains_query_and_prior_not_answer(
    encode_prompt: Callable[[str], Tensor] | None = None,
):
    s = _solutions()
    fn = encode_prompt or s.encode_prompt
    positive = fn(s.build_prompt("green", "circle", "claims_present"))
    negative = fn(s.build_prompt("green", "circle", "claims_absent"))
    assert positive.tolist() == [2, 1, s.PRIOR_PRESENT]
    assert negative.tolist() == [2, 1, s.PRIOR_ABSENT]
    assert positive.shape == (3,), "There is no fourth token carrying the target answer."


def test_counterfactual_changes_only_pixels_and_evaluation_label():
    s = _solutions()
    example = next(
        item
        for item in s.build_arbitration_benchmark()
        if item.condition == "counterfactual_present"
    )
    counterfactual = s.make_counterfactual(example)
    assert counterfactual.prompt == example.prompt
    assert counterfactual.target_bbox == example.target_bbox
    assert counterfactual.distractor_bbox == example.distractor_bbox
    assert counterfactual.label != example.label
    assert not t.equal(counterfactual.image, example.image)


def test_model_inference_is_independent_of_example_label_metadata():
    s = _solutions()
    example = s.build_arbitration_benchmark()[0]
    mislabeled = replace(example, object_present=not example.object_present)
    model = s.PixelPromptGroundTruthVLM()
    images_a, prompts_a, _ = s.batch_inputs([example])
    images_b, prompts_b, _ = s.batch_inputs([mislabeled])
    assert t.equal(images_a, images_b) and t.equal(prompts_a, prompts_b)
    assert t.equal(model(images_a, prompts_a), model(images_b, prompts_b))


def test_pixel_prompt_organism_recognizes_full_factorial_and_wrong_shape():
    s = _solutions()
    examples = [
        example
        for example in s.build_arbitration_benchmark()
        if example.split == "test" and example.condition in {"neutral_present", "neutral_absent"}
    ]
    images, prompts, labels = s.batch_inputs(examples)
    predictions = s.PixelPromptGroundTruthVLM()(images, prompts).argmax(dim=-1)
    assert t.equal(predictions, labels)
    assert {(example.target_color, example.target_shape) for example in examples} == set(s.TARGETS)


def test_bbox_to_patch_indices(
    bbox_to_patch_indices: Callable[..., tuple[int, ...]] | None = None,
):
    s = _solutions()
    fn = bbox_to_patch_indices or s.bbox_to_patch_indices
    assert fn((16, 16, 48, 48)) == (7, 8, 13, 14)
    with pytest.raises(ValueError, match="inside"):
        fn((-1, 0, 16, 16))


def test_matched_controls_are_true_background_equal_distinct_and_nonoverlapping(
    matched_control_regions: Callable[..., object] | None = None,
):
    s = _solutions()
    fn = matched_control_regions or s.matched_control_regions
    recipients = [
        example
        for example in s.build_arbitration_benchmark()
        if example.condition == "counterfactual_present"
    ]
    for example in recipients:
        controls = fn(example, seed=11)
        controls.validate()
        true_background = set(s.geometry_background_indices(example))
        assert set(controls.background) <= true_background
        assert set(controls.random) <= true_background
        assert controls.background != controls.random
        assert set(controls.background).isdisjoint(controls.random)
        if example.distractor_bbox is not None:
            distractor = set(s.bbox_to_patch_indices(example.distractor_bbox))
            assert distractor.isdisjoint(controls.background)
            assert distractor.isdisjoint(controls.random)


def test_cache_has_pixel_and_prompt_routing_stages(
    capture_arbitration_cache: Callable[..., tuple[Tensor, Tensor]] | None = None,
):
    s = _solutions()
    examples = s.build_arbitration_benchmark()
    present = next(
        example
        for example in examples
        if example.split == "test" and example.condition == "neutral_present"
    )
    conflict = next(
        example
        for example in examples
        if example.split == "test" and example.condition == "conflict_absent"
    )
    images, prompts, _ = s.batch_inputs([present, conflict])
    capture = capture_arbitration_cache or s.capture_arbitration_cache
    logits, cache = capture(s.PixelPromptGroundTruthVLM(), images, prompts)
    assert cache.shape == (s.N_LAYERS + 1, 2, s.SEQ_LEN, s.D_MODEL)
    assert cache[1, 0, s.ANSWER_TOKEN, 0] == s.VISUAL_STRENGTH
    assert cache[2, 1, s.ANSWER_TOKEN, 1] == s.PRIOR_STRENGTH
    assert s.answer_margin(logits).tolist() == [4.0, 2.0]


def test_binary_metrics_are_semantic(
    binary_metrics: Callable[[Tensor, Tensor], dict[str, float]] | None = None,
):
    s = _solutions()
    fn = binary_metrics or s.binary_metrics
    logits = t.tensor([[2.0, -1.0], [-1.0, 2.0], [-2.0, 1.0], [1.0, -2.0]])
    labels = t.tensor([0, 0, 1, 1])
    result = fn(logits, labels)
    assert result == {"accuracy": 0.5, "hallucination_rate": 0.5, "miss_rate": 0.5}


def test_replace_token_activations_is_exact_and_stub_sensitive(
    replace_token_activations: Callable[..., Tensor] | None = None,
):
    s = _solutions()
    fn = replace_token_activations or s.replace_token_activations
    recipient = t.arange(30, dtype=t.float32).reshape(1, 5, 6)
    donor = recipient + 100
    patched = fn(recipient, donor, (1, 3))
    assert t.equal(patched[:, (1, 3)], donor[:, (1, 3)])
    assert t.equal(patched[:, (0, 2, 4)], recipient[:, (0, 2, 4)])
    assert t.equal(recipient, t.arange(30, dtype=t.float32).reshape(1, 5, 6))
    half = fn(recipient, donor, (1,), strength=0.5)
    assert t.equal(half[:, 1], (recipient[:, 1] + donor[:, 1]) / 2)
    with pytest.raises(ValueError, match="unique"):
        fn(recipient, donor, (1, 1))


def test_counterfactual_patch_sweep_is_causal_and_controlled(
    counterfactual_patch_sweep: Callable[..., pd.DataFrame] | None = None,
):
    s = _solutions()
    fn = counterfactual_patch_sweep or s.counterfactual_patch_sweep
    recipient = next(
        example
        for example in s.build_arbitration_benchmark()
        if example.split == "test" and example.condition == "counterfactual_present"
    )
    donor = s.make_counterfactual(recipient)
    report = fn(s.PixelPromptGroundTruthVLM(), recipient, donor)
    layer_zero = report[report.layer == 0].set_index("condition")
    assert layer_zero.loc["object_patch", "answer_flip"]
    assert layer_zero.loc["object_patch", "directional_effect"] == 8.0
    assert layer_zero.loc["background_patch", "directional_effect"] == 0.0
    assert layer_zero.loc["same_size_random", "directional_effect"] == 0.0
    assert layer_zero.loc["full_visual_patch", "directional_effect"] == 8.0
    assert layer_zero.loc["object_patch", "token_count"] == layer_zero.loc[
        "background_patch", "token_count"
    ]
    assert report[report.layer > 0].query("condition == 'object_patch'").directional_effect.eq(0).all()


def test_validation_selects_layer_and_test_remains_untouched(
    select_layer_on_validation: Callable[..., tuple[int, pd.DataFrame]] | None = None,
):
    s = _solutions()
    fn = select_layer_on_validation or s.select_layer_on_validation
    examples = s.build_arbitration_benchmark()
    validation_pairs = s._counterfactual_pairs(examples, "validation")
    selected, curve = fn(s.PixelPromptGroundTruthVLM(), validation_pairs)
    assert selected == 0
    assert curve.loc[curve.layer == 0, "selection_score"].item() == 8.0
    with pytest.raises(ValueError, match="validation"):
        fn(s.PixelPromptGroundTruthVLM(), s._counterfactual_pairs(examples, "test"))


def test_parameterized_playground_changes_strength_and_region():
    s = _solutions()
    recipient = s._counterfactual_pairs(s.build_arbitration_benchmark(), "test")[0][0]
    model = s.PixelPromptGroundTruthVLM()
    object_patch = s.intervention_playground(
        model, recipient, layer=0, region="object_patch", strength=1.0
    )
    background = s.intervention_playground(
        model, recipient, layer=0, region="background_patch", strength=1.0
    )
    assert object_patch["prediction"] == "no" and object_patch["answer_flip"]
    assert background["prediction"] == "yes" and not background["answer_flip"]
    assert s.intervention_playground(
        model, recipient, layer=0, region="object_patch", strength=0.0
    )["prediction"] == "yes"


def test_toy_signature_and_smoke_fixture():
    s = _solutions()
    result = s.toy_signature_result()
    assert result["accepted"]
    assert result["metrics"]["joint_accuracy"] == 0.75
    assert result["metrics"]["conflict_accuracy"] == 0.0
    assert result["metrics"]["hallucination_rate"] == 0.25
    fixture = json.loads((_section_dir() / "expected_outputs/smoke_test.json").read_text())
    assert fixture == s.run_smoke_test()


def test_vlm_geometry_controls_exclude_object_and_distractor():
    s = _solutions()
    example = s._counterfactual_pairs(s.build_arbitration_benchmark(), "test")[0][0]
    positions = t.arange(100, 136)
    controls = s.geometry_matched_vlm_controls(
        example, positions, t.tensor([1, 12, 12]), merge_size=2, seed=4
    )
    controls.validate()
    target = set(
        s.bbox_to_vlm_token_indices(
            example.target_bbox, positions, t.tensor([1, 12, 12]), merge_size=2
        ).tolist()
    )
    distractor = set(
        s.bbox_to_vlm_token_indices(
            example.distractor_bbox, positions, t.tensor([1, 12, 12]), merge_size=2
        ).tolist()
    )
    assert target == set(controls.object)
    assert distractor.isdisjoint(controls.background)
    assert distractor.isdisjoint(controls.random)


def _valid_real_preflight_fixture():
    # Synthetic unit-test evidence for validating the acceptance calculation only.
    # This fixture is never serialized as an actual Qwen run.
    s = _solutions()
    examples = s.build_arbitration_benchmark()
    test = [example for example in examples if example.split == "test"]
    margins = [1.0 if example.label else -1.0 for example in test]
    rows = []
    for recipient, _ in s._counterfactual_pairs(examples, "test"):
        for condition, margin in (
            ("recipient", -1.0), ("donor", 1.0), ("object_patch", 0.0),
            ("background_patch", -0.9), ("same_size_random", -0.9), ("full_visual_patch", 0.2),
        ):
            rows.append({
                "example_id": recipient.example_id, "split": "test", "layer": 12,
                "condition": condition, "margin": margin,
                "prediction": "yes" if margin >= 0 else "no", "token_count": 4,
                "directional_effect": margin + 1.0, "answer_flip": margin >= 0,
            })
    return {
        "model_id": s.MODEL_ID, "revision": s.MODEL_REVISION,
        "test_example_count": 64, "validation_pair_count": 8, "test_pair_count": 8,
        "selected_layer": 12, "candidate_layers": [3, 8, 12, 20],
        "layer_selection_split": "validation", "reporting_split": "test",
        "joint_accuracy": 1.0, "conflict_accuracy": 1.0, "hallucination_rate": 0.0,
        "image_question_accuracy": 1.0, "text_only_accuracy": 0.5,
        "joint_margins": margins, "image_question_margins": margins,
        "text_only_margins": [-1.0] * 64, "test_patch_rows": rows,
        "test_patch_summary": s.summarize_patch_rows(pd.DataFrame(rows), layer=12),
        "peak_vram_gb": 10.0, "wall_clock_seconds": 900.0,
    }


def test_real_acceptance_requires_behavior_and_directional_controls():
    s = _solutions()
    result = _valid_real_preflight_fixture()
    assert s.evaluate_real_acceptance(result) == (True, [])
    result["conflict_accuracy"] = 0.0
    accepted, failures = s.evaluate_real_acceptance(result)
    assert not accepted and any("conflict_accuracy" in failure for failure in failures)
    result = _valid_real_preflight_fixture()
    result["test_patch_summary"]["object_patch_directional_effect_mean"] = 0.0
    accepted, failures = s.evaluate_real_acceptance(result)
    assert not accepted and any("object_patch" in failure for failure in failures)


def test_real_acceptance_rejects_wrong_identity_nonfinite_and_impossible_rates():
    s = _solutions()
    for key, value in (
        ("model_id", "another/model"), ("revision", "main"),
        ("joint_accuracy", float("nan")), ("hallucination_rate", float("inf")),
        ("joint_accuracy", 1.1), ("text_only_accuracy", -0.2),
        ("wall_clock_seconds", -1.0), ("test_example_count", 64.5),
    ):
        result = _valid_real_preflight_fixture()
        result[key] = value
        assert not s.evaluate_real_acceptance(result)[0], f"Reject invalid {key}={value}."
    for key in ("joint_margins", "image_question_margins", "text_only_margins"):
        result = _valid_real_preflight_fixture()
        result[key] = [float("nan")] * 64
        assert not s.evaluate_real_acceptance(result)[0], f"Reject nonfinite raw {key}."


def test_real_acceptance_recomputes_raw_rows_and_denominators():
    s = _solutions()
    result = _valid_real_preflight_fixture()
    result["test_patch_rows"][2]["directional_effect"] = float("nan")
    assert not s.evaluate_real_acceptance(result)[0], "Nonfinite row effects must never pass."
    result = _valid_real_preflight_fixture()
    result["test_patch_rows"][2]["margin"] = -1.0
    assert not s.evaluate_real_acceptance(result)[0], "Effect and flip must follow the observed margin."
    result = _valid_real_preflight_fixture()
    result["test_patch_rows"].pop()
    assert not s.evaluate_real_acceptance(result)[0], "Every example must have every required control."
    result = _valid_real_preflight_fixture()
    result["joint_margins"] = result["joint_margins"][:-1]
    assert not s.evaluate_real_acceptance(result)[0], "Do not silently shrink the held-out denominator."
    result = _valid_real_preflight_fixture()
    result["test_patch_rows"][0]["split"] = "validation"
    assert not s.evaluate_real_acceptance(result)[0], "Test reports cannot include validation rows."


def test_capture_module_output_removes_hook():
    s = _solutions()
    module = t.nn.Linear(2, 2, bias=False)
    values = t.ones(1, 2)
    result, captured = s.capture_module_output(module, lambda: module(values))
    assert t.equal(result, captured)
    assert not module._forward_hooks


def _notebook_source(path: Path, *, code_only: bool = False) -> str:
    notebook = json.loads(path.read_text())
    cells = notebook["cells"]
    if code_only:
        cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    return "\n".join("".join(cell.get("source", [])) for cell in cells)


def test_exercise_notebook_has_arena_surface_and_immediate_stub_tests():
    path = _section_dir() / "12.5_Object_Hallucination_and_Modality_Arbitration_exercises.ipynb"
    source = _notebook_source(path)
    required = (
        "# [12.5] Object Hallucination and Modality Arbitration",
        "## Core Question",
        "## Cold Open",
        "pixel-and-prompt ground-truth organism",
        "## Signature Result",
        "## Try It Yourself",
        "## Real VLM Escalation",
        "validation",
        "untouched test",
        "## Limitations",
    )
    for marker in required:
        assert marker in source, f"Missing learner-surface marker: {marker}"
    assert source.count("### Exercise ") == 8
    assert source.count("<summary>Expected output</summary>") == 8
    assert source.count("<summary>Help</summary>") == 8
    assert source.count("<summary>Solution</summary>") == 8
    assert source.count("<summary>Interpretation</summary>") == 8
    assert source.count("# Immediate test: fails while the stub is present") == 8
    assert "LAYER = 0" in source and "REGION = \"object_patch\"" in source


def test_solution_notebook_keeps_taught_causal_methods_inline_and_is_executed():
    path = _section_dir() / "12.5_Object_Hallucination_and_Modality_Arbitration_solutions.ipynb"
    notebook = json.loads(path.read_text())
    code = _notebook_source(path, code_only=True)
    tree = ast.parse(code)
    definitions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    taught = {
        "build_arbitration_benchmark",
        "encode_prompt",
        "bbox_to_patch_indices",
        "matched_control_regions",
        "PixelPromptGroundTruthVLM",
        "replace_token_activations",
        "counterfactual_patch_sweep",
        "select_layer_on_validation",
    }
    assert taught <= definitions.keys()
    for name in taught:
        node = definitions[name]
        assert not any(isinstance(child, ast.Pass) for child in ast.walk(node))
        assert not any(
            isinstance(child, ast.Raise)
            and isinstance(child.exc, ast.Call)
            and isinstance(child.exc.func, ast.Name)
            and child.exc.func.id == "NotImplementedError"
            for child in ast.walk(node)
        )
    assert "solutions." not in code and "reference." not in code
    assert all(cell.get("execution_count") is not None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert any(cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_instruction_page_places_images_beside_prose_and_uses_honest_number():
    page = _repo_root() / "chapter12_vlm_interpretability/instructions/pages/05_[12.5]_Object_Hallucination_and_Modality_Arbitration.md"
    source = page.read_text()
    assert source.startswith("# [12.5] Object Hallucination and Modality Arbitration")
    assert '<div class="arena-side-by-side" style="display:grid' in source
    assert "vlm_hallucination_cold_open.png" in source
    assert "vlm_hallucination_signature.png" in source
    assert "12.2" not in source


def test_local_report_schema_and_manifest_are_strict_and_fresh():
    from jsonschema import Draft202012Validator
    from arena_ext.verification import verify_report_input_manifest

    section = _section_dir()
    schema = json.loads((section / "verification_report.schema.json").read_text())
    report = json.loads((section / "verification_report.json").read_text())
    manifest = yaml.safe_load((section / "artifacts.lock.yml").read_text())
    assert schema["additionalProperties"] is False
    assert schema["properties"]["report_inputs"]["additionalProperties"] is False
    assert schema["properties"]["metrics"]["properties"]["cpu_smoke"]["additionalProperties"] is False
    Draft202012Validator(schema).validate(report)
    assert set(manifest) == {
        "notebook_id", "section", "title", "exercise_metadata", "gt_tier",
        "evidence_level", "claim_scope", "required_gpu_gb", "max_allowed_gpu_gb",
        "models", "datasets", "expected_metrics", "controls", "safety_notes",
    }
    assert manifest["notebook_id"] == "12_5_object_hallucination_arbitration"
    assert manifest["section"] == "12.5"
    assert manifest["exercise_metadata"]["EXPECTED_RUNTIME"] == "4-6 hours learner; 12-25 minutes CUDA verification"
    assert verify_report_input_manifest(_repo_root(), report["report_inputs"]) == []


def test_real_path_is_pinned_cuda_only_validation_selected_and_has_no_invalid_equality_gate():
    source = (_section_dir() / "solutions.py").read_text()
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    required = {
        "load_pinned_real_vlm",
        "prepare_real_vlm_batch",
        "collect_real_vlm_evidence",
        "patch_real_vlm_layer",
        "real_activation_patch_controls",
        "select_real_layer_on_validation",
        "evaluate_real_acceptance",
        "run_real_vlm_benchmark",
        "run_gpu_test",
        "run_full_experiment",
    }
    assert required <= functions
    s = _solutions()
    assert s.MODEL_REVISION == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert "CPU fallback is forbidden" in source
    assert 'device_map={"": "cuda:0"}' in source
    assert "full_visual_counterfactual_error" not in source
    assert '"layer_selection_split": "validation"' in source
    assert '"reporting_split": "test"' in source


def test_dropdowns_complete_learner_without_missing_helpers():
    import re

    section = _section_dir()
    learner = json.loads((section / "12.5_Object_Hallucination_and_Modality_Arbitration_exercises.ipynb").read_text())
    solved = json.loads((section / "12.5_Object_Hallucination_and_Modality_Arbitration_solutions.ipynb").read_text())
    completed = 0
    for index, (cell, answer_cell) in enumerate(zip(learner["cells"], solved["cells"], strict=True)):
        if cell["cell_type"] != "code":
            for block in re.findall(r"```python\n(.*?)```", "".join(cell["source"]), re.S):
                ast.parse(block)
            continue
        source = "".join(cell["source"])
        answer_source = "".join(answer_cell["source"])
        if source == answer_source:
            ast.parse(source)
            continue
        before = "".join(learner["cells"][index - 1]["source"])
        dropdown = re.search(r"<summary>Solution</summary>(.*?)</details>", before, re.S)
        assert dropdown, f"Cell {index} needs its answer next to the exercise."
        answer = "\n".join(re.findall(r"```python\n(.*?)```", dropdown[1], re.S))
        answer_tree = ast.parse(answer)
        assert ast.dump(answer_tree) == ast.dump(ast.parse(answer_source))
        given = ast.parse(source)
        names = {node.name for node in given.body if isinstance(node, ast.FunctionDef)}
        expected = {node.name for node in answer_tree.body if isinstance(node, ast.FunctionDef)}
        assert names == expected, f"Cell {index} omits provided dependencies: {expected - names}."
        completed += 1
    assert completed == 8
