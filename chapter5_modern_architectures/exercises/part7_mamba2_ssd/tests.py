"""Focused semantic tests for the Mamba-2 / SSD learner implementation."""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import torch as t

try:
    from . import solutions as reference
except ImportError:
    import solutions as reference


def test_segment_sums_exact(segment_sums_fn=reference.segment_sums) -> None:
    values = t.tensor([1.0, 2.0, 4.0], dtype=t.float64)
    actual = segment_sums_fn(values)
    expected = t.tensor(
        [[0.0, -t.inf, -t.inf], [2.0, 0.0, -t.inf], [6.0, 4.0, 0.0]],
        dtype=t.float64,
    )
    assert t.equal(actual, expected), (
        "segment sums must include transitions after the source and through the "
        "destination, with zero on the diagonal and -inf above it"
    )
    print("All tests in `test_segment_sums_exact` passed!")


def test_expand_grouped_parameters(
    expand_fn=reference.expand_grouped_parameters,
) -> None:
    B = t.tensor([[[[1.0], [7.0]]]])
    C = -B
    B_heads, C_heads = expand_fn(B, C, n_heads=4)
    assert B_heads.shape == (1, 1, 4, 1), (
        "two groups mapped to four heads must produce one parameter row per head"
    )
    assert t.equal(B_heads.flatten(), t.tensor([1.0, 1.0, 7.0, 7.0])), (
        "each group must be shared by a contiguous block of heads"
    )
    assert t.equal(C_heads, -B_heads), "B and C must use the same head-to-group map"
    with pytest.raises(ValueError, match="divisible"):
        expand_fn(B, C, n_heads=3)
    print("All tests in `test_expand_grouped_parameters` passed!")


def test_semiseparable_matrix_hand_oracle(
    matrix_fn=reference.semiseparable_matrix,
) -> None:
    log_decay = t.log(t.tensor([[[0.5], [0.25], [0.2]]], dtype=t.float64))
    B = t.ones(1, 3, 1, 1, dtype=t.float64)
    C = t.ones_like(B)
    actual = matrix_fn(log_decay, B, C)[0, 0]
    expected = t.tensor(
        [[1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [0.05, 0.2, 1.0]],
        dtype=t.float64,
    )
    assert t.allclose(actual, expected, atol=1e-15, rtol=0), (
        "the (t,s) entry must multiply transitions s+1 through t; the source "
        "transition is not applied to its own write"
    )
    print("All tests in `test_semiseparable_matrix_hand_oracle` passed!")


def test_recurrent_ssd_hand_oracle(recurrent_fn=reference.ssd_recurrent) -> None:
    X = t.tensor([[[[2.0]], [[3.0]], [[5.0]]]], dtype=t.float64)
    log_decay = t.log(t.tensor([[[0.9], [0.5], [0.25]]], dtype=t.float64))
    B = t.ones(1, 3, 1, 1, dtype=t.float64)
    C = t.ones_like(B)
    output, state = recurrent_fn(X, log_decay, B, C)
    expected = t.tensor([2.0, 4.0, 6.0], dtype=t.float64)
    assert t.equal(output.flatten(), expected), (
        "the literal recurrence should produce states 2, 0.5*2+3=4, and "
        "0.25*4+5=6"
    )
    assert state.item() == 6.0, "the returned cache must be the post-token state"
    ablated, _ = recurrent_fn(X, log_decay, B, C, carry_scale=0.0)
    assert t.equal(ablated.flatten(), X.flatten()), (
        "zeroing recurrent carry must preserve each token's current write while "
        "removing all earlier-token information"
    )
    print("All tests in `test_recurrent_ssd_hand_oracle` passed!")


def test_quadratic_ssd_parity_and_causality(
    recurrent_fn=reference.ssd_recurrent,
    quadratic_fn=reference.ssd_quadratic,
) -> None:
    X, A, B, C = reference.make_ssd_inputs(length=11, seed=3)
    recurrent, _ = recurrent_fn(X, A, B, C)
    quadratic = quadratic_fn(X, A, B, C)
    assert t.allclose(recurrent, quadratic, atol=1e-11, rtol=1e-11), (
        "the attention-dual matrix must reproduce the independently coded recurrence"
    )
    changed = X.clone()
    changed[:, -1] += 10_000
    changed_quadratic = quadratic_fn(changed, A, B, C)
    assert t.equal(changed_quadratic[:, :-1], quadratic[:, :-1]), (
        "changing a future value must not alter any earlier SSD output"
    )
    print("All tests in `test_quadratic_ssd_parity_and_causality` passed!")


def test_chunked_ssd_parity_state_and_gradients(
    recurrent_fn=reference.ssd_recurrent,
    chunked_fn=reference.ssd_chunked,
) -> None:
    X, A, B, C = reference.make_ssd_inputs(length=19, seed=5)
    X.requires_grad_(True)
    initial = t.randn(2, 4, 3, 5, dtype=t.float64)
    recurrent, recurrent_state = recurrent_fn(X, A, B, C, initial)
    chunked, chunked_state = chunked_fn(X, A, B, C, 6, initial)
    assert t.allclose(recurrent, chunked, atol=2e-11, rtol=2e-11), (
        "chunked SSD must match recurrence even when length is not divisible by "
        "chunk size and the initial state is nonzero"
    )
    assert t.allclose(recurrent_state, chunked_state, atol=2e-11, rtol=2e-11), (
        "padding must leave the final recurrent cache unchanged"
    )
    chunked.square().mean().backward()
    assert X.grad is not None and bool(t.isfinite(X.grad).all()), (
        "the educational chunked contractions must preserve finite gradients"
    )
    print("All tests in `test_chunked_ssd_parity_state_and_gradients` passed!")


def test_official_listing_one_parity(
    parity_fn=reference.official_ssd_reference_parity,
) -> None:
    metrics = parity_fn()
    assert metrics["output_max_abs_diff"] < 1e-11, (
        "our chunked output must match the official Mamba repository's SSD Listing 1"
    )
    assert metrics["final_state_max_abs_diff"] < 1e-11, (
        "our final boundary state must match the official Listing 1 cache"
    )
    print("All tests in `test_official_listing_one_parity` passed!")


def test_mamba2_block_shape_and_causality(
    block_cls=reference.Mamba2BlockFromScratch,
    config_cls=reference.Mamba2Config,
) -> None:
    t.manual_seed(11)
    block = block_cls(config_cls()).eval()
    inputs = t.randn(2, 17, block.config.d_model)
    output = block(inputs)
    assert output.shape == inputs.shape, "a residual Mamba-2 block must preserve [B,L,D]"
    perturbed = inputs.clone()
    perturbed[:, -1] += 1_000
    changed = block(perturbed)
    assert t.allclose(output[:, :-1], changed[:, :-1], atol=0, rtol=0), (
        "the projection, depthwise convolution, and SSD path must all be causal"
    )
    print("All tests in `test_mamba2_block_shape_and_causality` passed!")


def test_mamba2_step_cache_parity(
    block_cls=reference.Mamba2BlockFromScratch,
    config_cls=reference.Mamba2Config,
) -> None:
    t.manual_seed(13)
    block = block_cls(config_cls()).eval()
    inputs = t.randn(2, 19, block.config.d_model)
    with t.inference_mode():
        full = block(inputs)
        cache = block.allocate_cache(2, device=inputs.device, dtype=inputs.dtype)
        pieces = []
        for position in range(inputs.shape[1]):
            output, cache = block.step(inputs[:, position], cache)
            pieces.append(output)
        stepped = t.cat(pieces, dim=1)
    assert t.allclose(full, stepped, atol=2e-6, rtol=2e-6), (
        "one-token decoding must reproduce the full convolution and SSD trajectory"
    )
    assert cache.conv_state.shape[-1] == block.config.d_conv, (
        "the convolution cache must retain exactly one kernel width"
    )
    print("All tests in `test_mamba2_step_cache_parity` passed!")


def test_delayed_recall_batch_semantics(
    batch_fn=reference.make_delayed_recall_batch,
) -> None:
    generator = t.Generator().manual_seed(0)
    inputs, labels = batch_fn(128, 32, generator=generator)
    recovered = inputs[:, 0, 0].gt(0).long()
    assert t.equal(recovered, labels), "the unshuffled target must equal the WRITE value"
    assert bool(inputs[:, 0, 1].eq(1).all()), "WRITE markers belong only at the first token"
    assert bool(inputs[:, -1, 2].eq(1).all()), "READ markers belong at the final token"
    shuffled_inputs, shuffled = batch_fn(
        4096,
        32,
        generator=t.Generator().manual_seed(0),
        shuffled_labels=True,
    )
    accidental_agreement = shuffled.eq(shuffled_inputs[:, 0, 0].gt(0)).float().mean()
    assert abs(float(accidental_agreement) - 0.5) < 0.03, (
        "the shuffled-label control must be statistically independent of the WRITE value"
    )
    print("All tests in `test_delayed_recall_batch_semantics` passed!")


@pytest.fixture(scope="module")
def recall_result() -> reference.RecallExperiment:
    return reference.run_delayed_recall_experiment()


def test_delayed_recall_scientific_controls(recall_result) -> None:
    result = recall_result
    assert result.mamba_accuracy >= 0.95, (
        "the trained SSD model must solve held-out delayed recall before we interpret it"
    )
    assert result.state_ablated_accuracy <= 0.65, (
        "zeroing recurrent carry should destroy the long-range solution"
    )
    assert result.local_conv_accuracy <= 0.65, (
        "a one-layer local convolution must remain near chance beyond its receptive field"
    )
    assert result.shuffled_label_accuracy <= 0.65, (
        "the same SSD capacity trained on shuffled labels must not learn the true task"
    )
    assert result.ood_length_accuracy >= 0.65, (
        "the learned recurrent mechanism should retain useful signal at twice the train length"
    )
    print("All tests in `test_delayed_recall_scientific_controls` passed!")


def test_recurrent_quadratic_chunked_gradient_parity(gradient_fn=reference.exact_gradient_parity_metrics) -> None:
    metrics = gradient_fn()
    assert max(metrics.values()) < 1e-11, (
        "all three schedules must differentiate the same mathematical operator"
    )
    print("All tests in `test_recurrent_quadratic_chunked_gradient_parity` passed!")


def test_tiny_stacked_lm_full_vs_recurrent_cache(lm_class=reference.TinyStackedMamba2LM) -> None:
    t.manual_seed(19)
    model = lm_class(vocab_size=47, n_layers=2).eval()
    input_ids = t.randint(0, 47, (2, 16))
    with t.inference_mode():
        full = model(input_ids)
        recurrent, caches = model.forward_recurrent(input_ids)
    assert full.shape == (2, 16, 47), 'The stacked LM must return one vocabulary-logit vector per batch token.'
    assert len(caches) == 2, 'Return one independent recurrent cache per layer.'
    assert t.allclose(full, recurrent, atol=2e-6, rtol=2e-6), (
        "stacking blocks must preserve the one-token cache contract through logits"
    )
    print("All tests in `test_tiny_stacked_lm_full_vs_recurrent_cache` passed!")


def test_parameter_matched_attention_and_mamba1_controls(
    count_fn=reference.parameter_matched_control_counts,
    attention_class=reference.TinyCausalAttentionControl,
    mamba1_class=reference.TinyMamba1Control,
) -> None:
    counts = count_fn()
    target = counts["Mamba-2"]
    assert abs(counts["causal attention"] - target) / target < 0.10, 'Match the Transformer parameter budget to Mamba-2 within ten percent.'
    assert abs(counts["Mamba-1"] - target) / target < 0.10, 'Match the Mamba-1 parameter budget to Mamba-2 within ten percent.'
    for module in (
        attention_class(),
        mamba1_class(),
    ):
        output = module(t.randn(2, 17, 12))
        assert output.shape == (2, 17, 12), 'Every matched architecture must return the requested batch, length, and vocabulary axes.'
        assert bool(t.isfinite(output).all()), 'Matched models must return finite logits before training comparisons.'
    print("All tests in `test_parameter_matched_attention_and_mamba1_controls` passed!")


def test_released_mamba2_full_mapping_hidden_logits_and_tokens() -> None:
    parity = reference.verify_released_mamba2_parity("cpu")
    assert parity.mapped_tensor_count == 219, 'Load all 219 checkpoint tensors before claiming released-model parity.'
    assert parity.hidden_state_count == 25, 'Compare embeddings and all 24 layer outputs, not only final logits.'
    assert parity.hidden_max_abs_diff < 0.002, 'Hidden-state maximum error exceeds 0.002; inspect tensor mapping and IEEE float32 dot precision.'
    assert parity.hidden_mean_abs_diff < 5e-5, 'Mean hidden-state error exceeds the locked float32 parity bound.'
    assert parity.logits_max_abs_diff < 5e-4, 'Maximum logit error exceeds 0.0005; do not relax this to hide a kernel mismatch.'
    assert parity.logits_mean_abs_diff < 5e-5, 'Mean logit error exceeds the locked 0.00005 bound.'
    assert parity.top1_agreement == 1.0, 'Every checked prompt must keep the reference next-token argmax.'
    assert parity.reference_top_tokens == parity.learner_top_tokens, 'The returned token IDs must agree as well as the numerical parity summary.'
    print("All tests in `test_released_mamba2_full_mapping_hidden_logits_and_tokens` passed!")


def test_cuda_hooks_are_strict() -> None:
    gpu_signature = inspect.signature(reference.run_gpu_test)
    full_signature = inspect.signature(reference.run_full_experiment)
    assert str(gpu_signature) == "(max_vram_gb: 'float' = 24.0) -> 'dict[str, Any]'", (
        "run_gpu_test must expose the standard bounded release signature"
    )
    assert str(full_signature) == "(max_vram_gb: 'float' = 24.0) -> 'dict[str, Any]'", (
        "run_full_experiment must expose the standard bounded release signature"
    )
    source = inspect.getsource(reference._require_cuda)
    assert "raise RuntimeError" in source and "fallback" in source, (
        "CUDA hooks must fail closed when CUDA is unavailable; they may not run on CPU"
    )
    print("All tests in `test_cuda_hooks_are_strict` passed!")


def test_recall_training_updates_parameters(train_fn=reference.train_recall_model) -> None:
    class ReadWriteBit(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = t.nn.Linear(1, 2, bias=False)
            t.nn.init.zeros_(self.linear.weight)

        def forward(self, inputs):
            return self.linear(inputs[:, 0, :1])

    model = ReadWriteBit()
    losses = train_fn(model, steps=8, sequence_length=8, batch_size=128, learning_rate=0.05, seed=7)
    assert len(losses) == 8 and all(math.isfinite(loss) for loss in losses), 'Every requested optimizer step must return a finite measured training loss.'
    assert abs(losses[0] - math.log(2)) < 1e-6, "Zero logits have exact CE log(2)."
    assert losses[-1] < 0.6 * losses[0], "The optimizer must learn the directly readable WRITE bit."
    assert model.linear.weight[0, 0] < 0 < model.linear.weight[1, 0], (
        "Cross entropy must train toward the true WRITE label, not its complement."
    )
    print("All tests in `test_recall_training_updates_parameters` passed!")


def test_recall_experiment_uses_requested_length_and_equal_budgets() -> None:
    """Observe the actual orchestration without retraining all three models."""
    from unittest.mock import patch

    training = []
    evaluation = []

    def record_training(model, **kwargs):
        training.append(kwargs)
        return [0.5] * kwargs["steps"]

    def record_evaluation(model, **kwargs):
        evaluation.append(kwargs)
        return 0.5

    with patch.object(reference, "train_recall_model", record_training), patch.object(
        reference, "evaluate_recall_model", record_evaluation
    ):
        reference.run_delayed_recall_experiment(steps=17, sequence_length=48)
    assert [call["steps"] for call in training] == [17, 17, 17], 'Train the main architecture and both controls for the same requested step count.'
    assert [call["sequence_length"] for call in evaluation] == [48, 48, 48, 48, 96], 'Evaluate all in-distribution conditions at the requested length and OOD at twice that length.'
    print("All tests in `test_recall_experiment_uses_requested_length_and_equal_budgets` passed!")


def test_notebook_contract() -> None:
    section = Path(__file__).resolve().parent
    exercises = section / "5.7_Mamba-2_and_SSD_exercises.ipynb"
    solutions = section / "5.7_Mamba-2_and_SSD_solutions.ipynb"
    assert exercises.exists() and solutions.exists(), "Generate both notebook surfaces before validation"
    learner = json.loads(exercises.read_text(encoding="utf-8"))
    solved = json.loads(solutions.read_text(encoding="utf-8"))
    learner_text = "\n".join("".join(cell.get("source", [])) for cell in learner["cells"])
    solved_text = "\n".join("".join(cell.get("source", [])) for cell in solved["cells"])
    for needle in [
        "Core Question",
        "By the end of this notebook",
        "Expected output",
        "Help - common bug",
        "<summary>Solution",
        "Try It Yourself",
        "Signature Result",
        "Anomaly Hunt",
        "Limitations",
    ]:
        assert needle in learner_text, f"learner notebook is missing the ARENA surface: {needle}"
    for function_name in [
        "segment_sums",
        "semiseparable_matrix",
        "ssd_recurrent",
        "ssd_quadratic",
        "ssd_chunked",
        "Mamba2BlockFromScratch",
        "run_delayed_recall_experiment",
        "TinyStackedMamba2LM",
        "TinyCausalAttentionControl",
        "TinyMamba1Control",
        "ReleasedMamba2LMFromScratch",
        "verify_released_mamba2_parity",
    ]:
        assert f"def {function_name}" in learner_text or f"class {function_name}" in learner_text, (
            f"the learner must see and implement {function_name} in the notebook"
        )
    executed = [
        cell
        for cell in solved["cells"]
        if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()
    ]
    assert executed and all(cell.get("execution_count") is not None for cell in executed), (
        "every nonempty solution code cell must be executed and retain its output state"
    )
    learner_code = "\n".join("".join(cell["source"]) for cell in learner["cells"]
                             if cell["cell_type"] == "code")
    assert "verification_report.json" not in learner_code, (
        "the scientific lesson must generate its result live rather than load a report"
    )
    assert "run_gpu_test(max_vram_gb: float = 24.0)" in solved_text, (
        "the solved notebook must expose the strict CUDA release hook"
    )
    print("All tests in `test_notebook_contract` passed!")


def test_solution_dropdowns_are_executable_python() -> None:
    section = Path(__file__).resolve().parent
    paths = list(section.glob("*.ipynb"))
    paths += [section.parents[1] / "instructions/pages/07_[5.7]_Mamba-2_and_Structured_State-Space_Duality.md"]
    for path in paths:
        text = path.read_text()
        if path.suffix == ".ipynb":
            text = "\n".join("".join(cell["source"]) for cell in json.loads(text)["cells"]
                             if cell["cell_type"] == "markdown")
        dropdowns = re.findall(r"<details>\s*<summary>Solution</summary>(.*?)</details>", text, re.S)
        assert len(dropdowns) == 11, f"{path.name}: expected eleven complete solutions"
        for index, dropdown in enumerate(dropdowns, 1):
            blocks = re.findall(r"```python\n(.*?)```", dropdown, re.S)
            assert blocks, f"Exercise {index} needs copyable Python"
            tree = ast.parse("\n".join(blocks), filename=f"{path.name}:solution-{index}")
            assert any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body), (
                f"Exercise {index} must supply implementation, not a pointer to another file"
            )
    print("All tests in `test_solution_dropdowns_are_executable_python` passed!")


def test_generated_surfaces_match_canonical_sources() -> None:
    try:
        from .build_notebooks import render, SECTION, STEM, PAGE
    except ImportError:
        from build_notebooks import render, SECTION, STEM, PAGE
    learner, solved, page = render()
    assert PAGE.read_text() == page, 'Regenerate the instruction page from the same lesson as the notebooks.'
    for notebook, suffix in [(learner, "exercises"), (solved, "solutions")]:
        stored = json.loads((SECTION / f"{STEM}_{suffix}.ipynb").read_text())
        assert len(stored["cells"]) == len(notebook.cells), 'Generated and stored notebooks must contain the same cells.'
        for actual, expected in zip(stored["cells"], notebook.cells, strict=True):
            assert "".join(actual["source"]) == expected.source, 'Stored notebook code and prose are stale relative to the lesson and implementations.'
    print("All tests in `test_generated_surfaces_match_canonical_sources` passed!")


def test_dropdown_completion_supplies_every_dependency() -> None:
    """A learner copying every answer must get the solved code, including helpers."""
    section = Path(__file__).resolve().parent
    learner = json.loads((section / "5.7_Mamba-2_and_SSD_exercises.ipynb").read_text())
    solved = json.loads((section / "5.7_Mamba-2_and_SSD_solutions.ipynb").read_text())
    for index, (exercise, solution) in enumerate(zip(learner["cells"], solved["cells"], strict=True)):
        if not exercise.get("metadata", {}).get("exercise_functions"):
            continue
        previous = "".join(learner["cells"][index - 1]["source"])
        dropdown = re.search(r"<summary>Solution</summary>(.*?)</details>", previous, re.S)[1]
        answer = "\n".join(re.findall(r"```python\n(.*?)```", dropdown, re.S))
        expected = "".join(solution["source"])
        assert ast.dump(ast.parse(answer)) == ast.dump(ast.parse(expected)), (
            f"cell {index}: the copyable answer must supply the full implementation and dependencies"
        )
    print("All tests in `test_dropdown_completion_supplies_every_dependency` passed!")
