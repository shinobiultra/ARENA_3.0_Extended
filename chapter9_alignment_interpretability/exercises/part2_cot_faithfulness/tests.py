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
    assert t.allclose(diff, t.tensor([3.0, -3.0]))
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
    assert s.prediction_accuracy(logits, batch.hidden_answer_ids) == 1.0
    shuffled_probe = fit_mean_difference_probe(hidden_states, batch.hidden_answer_ids.roll(1))
    shuffled_logits = probe_logits_from_direction(hidden_states, shuffled_probe)
    assert s.prediction_accuracy(shuffled_logits, batch.hidden_answer_ids) <= 0.5
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
    assert heatmap.shape == (4, 4)
    assert heatmap[2, final_position] == 1.0
    assert heatmap[3, final_position] == 1.0
    assert heatmap[0, rationale_position] <= 0.5
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
    assert t.allclose(patched[:, 2, :], donor.expand(2, -1))
    assert t.allclose(hidden, t.zeros_like(hidden)), "The hook helper should clone before editing."

    tuple_output = (hidden, "cache")
    tuple_patched = replace_position_in_layer_output(tuple_output, donor, token_position=-1)
    assert isinstance(tuple_patched, tuple)
    assert tuple_patched[1] == "cache"
    assert t.allclose(tuple_patched[0][:, -1, :], donor.expand(2, -1))
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
    assert patched_logits.argmax(dim=-1).eq(donor_ids).all()
    assert (patched_margin - clean_margin).min() > 5.0
    assert t.allclose(irrelevant_margin, clean_margin)
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
    assert report.target_beats_controls
    assert report.max_control_name == "random_donor"
    assert report.flip_rates["target_patch"] == 1.0
    with pytest.raises(ValueError, match="target_patch"):
        patch_control_summary({"text_only": t.zeros(2)}, {"text_only": t.zeros(2)})
    print("All tests in `test_patch_control_summary_requires_target_beats_controls` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["accepted"] and result["contract_passed"] and result["tests_passed"]
    assert result["target_probe_accuracy"] == 1.0
    assert result["label_shuffled_probe_accuracy"] <= 0.5
    assert result["patch_control_flip_rates"]["target_patch"] == 1.0
    assert result["patch_control_flip_rates"]["irrelevant_position"] == 0.0
    assert result["patch_target_control_gap"] >= 1.0
    assert len(result["qualitative_examples"]) >= 3
    print("All tests in `test_notebook_contract` passed!")


def test_live_gpu_signature_result(result: Mapping[str, Any] | None = None):
    if result is None:
        pytest.skip("Pass a live run_gpu_test result from the parent CUDA verification run.")
    assert result["cuda_available"] and result["experiment_completed"]
    assert result["model_name"] == "EleutherAI/pythia-70m-deduped"
    assert result["hf_revision"] == "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
    assert result["heldout_prompt_count"] >= 20
    assert result["hidden_answer_accuracy"] >= 0.7
    assert result["label_shuffled_probe_accuracy"] <= 0.65
    assert result["text_only_recall"] >= result["detector_recall"]
    assert not result["text_only_misses_cases"]
    assert not result["target_beats_controls"]
    assert result["patch_control_means"]["target_patch"] <= result["patch_control_means"]["random_donor"]
    assert result["patch_target_control_gap"] <= 0.01
    assert result["negative_result_detected"]
    assert not result["real_model_claim_supported"]
    assert not result["preflight_passed"]
    assert "true forward-pass residual activation patching" in result["full_path"]
    assert len(result["probe_heatmap"]) >= 2
    assert len(result["qualitative_examples"]) >= 4
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
    assert source.count("<summary>Expected output</summary>") >= 6
    assert source.count("<summary>Help") >= 6
    assert source.count("<summary>Solution") >= 6
    print("All tests in `test_exercise_notebook_declares_arena_pedagogy` passed!")
