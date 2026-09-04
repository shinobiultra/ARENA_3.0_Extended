from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

import pytest
import torch as t


def _solutions():
    from chapter9_alignment_interpretability.exercises.part2_cot_faithfulness import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _assert_close(actual: Any, expected: float, *, name: str) -> None:
    assert abs(float(actual) - expected) < 1e-6, f"{name} should be {expected}, got {actual}."


def test_prediction_accuracy_checks_top1_predictions(
    prediction_accuracy: Callable | None = None,
):
    prediction_accuracy = prediction_accuracy or _solutions().prediction_accuracy
    logits = t.tensor([[3.0, -1.0], [-0.5, 2.0], [1.0, 4.0], [2.0, 0.0]])
    labels = t.tensor([0, 1, 0, 0])
    _assert_close(
        prediction_accuracy(logits, labels),
        0.75,
        name="prediction_accuracy",
    )
    with pytest.raises(ValueError, match="leading dimensions"):
        prediction_accuracy(t.zeros(2, 3), t.zeros(3, dtype=t.long))
    with pytest.raises(ValueError, match="finite"):
        prediction_accuracy(t.tensor([[float("nan"), 0.0]]), t.tensor([0]))
    with pytest.raises(ValueError, match="answer ids"):
        prediction_accuracy(t.zeros(1, 2), t.tensor([2]))
    print("All tests in `test_prediction_accuracy_checks_top1_predictions` passed!")


def test_answer_logit_diff_tracks_b_minus_a(answer_logit_diff: Callable | None = None):
    answer_logit_diff = answer_logit_diff or _solutions().answer_logit_diff
    logits = t.tensor([[2.0, 5.0], [4.0, 1.0]])
    diff = answer_logit_diff(logits)
    assert t.allclose(diff, t.tensor([3.0, -3.0])), (
        "answer_logit_diff should compute logit(B) - logit(A) for every example."
    )
    with pytest.raises(ValueError, match=r"\[A, B\]"):
        answer_logit_diff(t.zeros(2, 3))
    print("All tests in `test_answer_logit_diff_tracks_b_minus_a` passed!")


def test_mean_difference_probe_recovers_toy_hidden_answer(
    fit_mean_difference_probe: Callable | None = None,
    probe_logits_from_direction: Callable | None = None,
):
    s = _solutions()
    fit_mean_difference_probe = fit_mean_difference_probe or s.fit_mean_difference_probe
    probe_logits_from_direction = probe_logits_from_direction or s.probe_logits_from_direction
    batch = s.make_toy_cot_batch(num_pairs=8)
    final_position = batch.position_names.index("final_prompt")
    hidden_states = batch.activations[:, 2, final_position, :]
    probe = fit_mean_difference_probe(hidden_states, batch.hidden_answer_ids)
    logits = probe_logits_from_direction(hidden_states, probe)
    assert s.prediction_accuracy(logits, batch.hidden_answer_ids) == 1.0, (
        "The mean-difference probe should exactly recover the toy hidden-answer direction."
    )
    shuffled_probe = fit_mean_difference_probe(hidden_states, batch.hidden_answer_ids.roll(1))
    shuffled_logits = probe_logits_from_direction(hidden_states, shuffled_probe)
    assert s.prediction_accuracy(shuffled_logits, batch.hidden_answer_ids) <= 0.5, (
        "Rolling the labels should destroy the toy probe's predictive signal."
    )
    with pytest.raises(ValueError, match="both answer classes"):
        fit_mean_difference_probe(hidden_states[:4], t.zeros(4, dtype=t.long))
    print("All tests in `test_mean_difference_probe_recovers_toy_hidden_answer` passed!")


def test_layer_position_heatmap_finds_toy_answer_stream(
    layer_position_probe_heatmap: Callable | None = None,
):
    s = _solutions()
    layer_position_probe_heatmap = layer_position_probe_heatmap or s.layer_position_probe_heatmap
    batch = s.make_toy_cot_batch(num_pairs=12)
    train = t.arange(0, 12)
    eval_idx = t.arange(12, 24)
    heatmap = layer_position_probe_heatmap(
        batch.activations[train],
        batch.hidden_answer_ids[train],
        batch.activations[eval_idx],
        batch.hidden_answer_ids[eval_idx],
    )
    final_position = batch.position_names.index("final_prompt")
    rationale_position = batch.position_names.index("rationale_answer")
    assert heatmap.shape == (4, 4), (
        "The heatmap should score every toy layer-position pair."
    )
    assert heatmap[2, final_position] == 1.0, (
        "Layer 2 at the final prompt is an exact hidden-answer readout location."
    )
    assert heatmap[3, final_position] == 1.0, (
        "Layer 3 at the final prompt should preserve the exact hidden-answer signal."
    )
    assert heatmap[0, rationale_position] <= 0.5, (
        "The early rationale state carries the visible rationale, not hidden-answer ground truth."
    )
    print("All tests in `test_layer_position_heatmap_finds_toy_answer_stream` passed!")


def test_replace_position_in_layer_output_patches_tensor_and_tuple(
    replace_position_in_layer_output: Callable | None = None,
):
    replace_position_in_layer_output = (
        replace_position_in_layer_output or _solutions().replace_position_in_layer_output
    )
    hidden = t.zeros(2, 4, 3)
    donor = t.tensor([1.0, 2.0, 3.0])
    patched = replace_position_in_layer_output(hidden, donor, token_position=2)
    assert t.allclose(patched[:, 2, :], donor.expand(2, -1)), (
        "The hook helper should replace only the requested token position with the donor state."
    )
    assert t.allclose(hidden, t.zeros_like(hidden)), "The hook helper should clone before editing."

    tuple_output = (hidden, "cache")
    tuple_patched = replace_position_in_layer_output(tuple_output, donor, token_position=-1)
    assert isinstance(tuple_patched, tuple), (
        "Tuple-returning transformer blocks should remain tuples after patching."
    )
    assert tuple_patched[1] == "cache", (
        "Non-hidden-state tuple entries should pass through the patch unchanged."
    )
    assert t.allclose(tuple_patched[0][:, -1, :], donor.expand(2, -1)), (
        "Negative token indices should patch the corresponding position in tuple outputs."
    )
    with pytest.raises(ValueError, match="outside"):
        replace_position_in_layer_output(hidden, donor, token_position=9)
    print("All tests in `test_replace_position_in_layer_output_patches_tensor_and_tuple` passed!")


def test_toy_forward_patch_has_exact_causal_ground_truth(
    toy_forward_patch_answer_logits: Callable | None = None,
):
    s = _solutions()
    toy_forward_patch_answer_logits = toy_forward_patch_answer_logits or s.toy_forward_patch_answer_logits
    batch = s.make_toy_cot_batch(num_pairs=12)
    final_position = batch.position_names.index("final_prompt")
    rationale_position = batch.position_names.index("rationale_answer")
    target = t.tensor([12, 14, 16])
    donor = target + 1
    target_ids = batch.hidden_answer_ids[target]
    donor_ids = batch.hidden_answer_ids[donor]
    clean_logits = s.toy_answer_logits(batch)[target]
    clean_margin = s.signed_margin_toward_donor(clean_logits, target_ids, donor_ids)
    patched_logits = toy_forward_patch_answer_logits(
        batch,
        target,
        donor,
        layer_index=2,
        position_index=final_position,
    )
    irrelevant_logits = toy_forward_patch_answer_logits(
        batch,
        target,
        donor,
        layer_index=2,
        position_index=rationale_position,
    )
    patched_margin = s.signed_margin_toward_donor(patched_logits, target_ids, donor_ids)
    irrelevant_margin = s.signed_margin_toward_donor(irrelevant_logits, target_ids, donor_ids)
    assert patched_logits.argmax(dim=-1).eq(donor_ids).all(), (
        "Patching the known causal state should flip every toy answer to the donor answer."
    )
    assert (patched_margin - clean_margin).min() > 5.0, (
        "The exact causal patch should create a large donor-margin intervention effect."
    )
    assert t.allclose(irrelevant_margin, clean_margin), (
        "Patching the rationale position should leave the toy answer margin unchanged."
    )
    print("All tests in `test_toy_forward_patch_has_exact_causal_ground_truth` passed!")


def test_patch_control_summary_requires_target_beats_controls(
    patch_control_summary: Callable | None = None,
):
    patch_control_summary = patch_control_summary or _solutions().patch_control_summary
    effects = {
        "target_patch": t.tensor([2.0, 2.5, 3.0]),
        "text_only": t.tensor([0.0, 0.0, 0.0]),
        "label_shuffled": t.tensor([0.1, 0.0, -0.1]),
        "random_direction": t.tensor([0.2, 0.1, 0.0]),
        "random_donor": t.tensor([0.5, 0.2, 0.1]),
        "irrelevant_position": t.tensor([0.0, 0.0, 0.0]),
    }
    flips = {
        name: (values > 1.0)
        for name, values in effects.items()
    }
    report = patch_control_summary(effects, flips, min_target_control_gap=1.0)
    assert report.target_beats_controls, (
        "The constructed target effect should clear the predeclared control gap."
    )
    assert report.max_control_name == "random_donor", (
        "The summary should identify the largest non-target intervention control."
    )
    assert report.flip_rates["target_patch"] == 1.0, (
        "All constructed target patches should cross the donor-flip threshold."
    )
    with pytest.raises(ValueError, match="target_patch"):
        patch_control_summary({"text_only": t.zeros(2)}, {"text_only": t.zeros(2)})
    print("All tests in `test_patch_control_summary_requires_target_beats_controls` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["accepted"] and result["contract_passed"] and result["tests_passed"], (
        "The exact toy notebook contract should satisfy every declared acceptance gate."
    )
    assert result["target_probe_accuracy"] == 1.0, (
        "The target toy state should be perfectly decodable on held-out examples."
    )
    assert result["label_shuffled_probe_accuracy"] <= 0.5, (
        "The shuffled-label probe should fail at or below chance."
    )
    assert result["patch_control_flip_rates"]["target_patch"] == 1.0, (
        "The known causal patch should flip every selected toy answer."
    )
    assert result["patch_control_flip_rates"]["irrelevant_position"] == 0.0, (
        "The irrelevant-position control should never flip the toy answer."
    )
    assert result["patch_target_control_gap"] >= 1.0, (
        "The toy target patch should visibly outperform the strongest matched control."
    )
    assert len(result["qualitative_examples"]) >= 3, (
        "The notebook contract should expose concrete target-donor examples for inspection."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_live_gpu_signature_result(result: Mapping[str, Any] | None = None):
    if result is None:
        pytest.skip("Pass a live run_gpu_test result from the parent CUDA verification run.")
    assert result["cuda_available"] and result["experiment_completed"], (
        "The live signature result must come from a completed CUDA experiment."
    )
    assert result["model_name"] == "EleutherAI/pythia-70m-deduped", (
        "The real-model result must use the course's pinned Pythia checkpoint."
    )
    assert result["hf_revision"] == "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c", (
        "The checkpoint revision must remain pinned so the negative result is reproducible."
    )
    assert result["heldout_prompt_count"] >= 20, (
        "The Pythia claim boundary requires a nontrivial held-out prompt set."
    )
    assert result["hidden_answer_accuracy"] >= 0.7, (
        "The hidden-answer probe should clear its predeclared held-out threshold."
    )
    assert result["label_shuffled_probe_accuracy"] <= 0.65, (
        "The shuffled-label control should remain substantially weaker than the probe."
    )
    assert result["text_only_recall"] >= result["detector_recall"], (
        "This pinned run is negative because text-only recall matches or exceeds the detector."
    )
    assert not result["text_only_misses_cases"], (
        "The private answer is text-visible here, so text-only should miss no target cases."
    )
    assert not result["target_beats_controls"], (
        "The pinned Pythia target patch should not be reported as beating matched controls."
    )
    assert result["patch_control_means"]["target_patch"] <= result["patch_control_means"]["random_donor"], (
        "The target intervention must not exceed the matched random-donor control in this negative run."
    )
    assert result["patch_target_control_gap"] <= 0.01, (
        "The target-control gap should stay at the locked negative-result boundary."
    )
    assert result["negative_result_detected"], (
        "The verification path should explicitly classify the Pythia evidence as negative."
    )
    assert not result["real_model_claim_supported"], (
        "Failed causal and text-only controls must block a real-model faithfulness claim."
    )
    assert not result["preflight_passed"], (
        "The preflight should fail honestly when the real-model claim boundaries are not met."
    )
    assert "true forward-pass residual activation patching" in result["full_path"], (
        "The evidence record should state that real hook-based forward patching was executed."
    )
    assert len(result["probe_heatmap"]) >= 2, (
        "The real-model result should expose a layer-position heatmap, not only a scalar."
    )
    assert len(result["qualitative_examples"]) >= 4, (
        "The real-model result should retain enough prompts for qualitative inspection."
    )
    print("All tests in `test_live_gpu_signature_result` passed!")


def test_exercise_notebook_declares_arena_pedagogy():
    notebook_path = _section_dir() / "9.2_Chain_of_Thought_Faithfulness_exercises.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    required_strings = [
        "By the end of this notebook",
        "exact toy ground truth",
        "Exercise 1",
        "Exercise 2",
        "Exercise 3",
        "Exercise 4",
        "Exercise 5",
        "Exercise 6",
        "layer-position heatmap",
        "true forward-pass residual activation patching",
        "text-only",
        "label-shuffled",
        "random-direction",
        "random-donor",
        "irrelevant-position",
        "Try It Yourself",
        "Anomaly hunt",
        "def run_smoke_test(cpu: bool = True)",
        "def run_gpu_test(max_vram_gb: float = 24.0)",
        "test_live_gpu_signature_result",
    ]
    for required in required_strings:
        assert required in source, f"The learner notebook should include {required!r}."
    assert source.count("<summary>Expected output</summary>") >= 6, (
        "Each hard exercise should state the output a learner should expect."
    )
    assert source.count("<summary>Help") >= 6, (
        "Each hard exercise should provide local help without revealing the solution immediately."
    )
    assert source.count("<summary>Solution") >= 6, (
        "Each hard exercise should include a collapsible reference solution."
    )
    print("All tests in `test_exercise_notebook_declares_arena_pedagogy` passed!")
