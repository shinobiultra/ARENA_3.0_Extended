"""Immediate semantic tests for [15.3] QLoRA from Scratch.

Notebook-facing ``check_*`` functions require the learner implementation as an
argument. The zero-argument ``test_*`` wrappers exercise the visible reference
implementation under pytest; no learner test silently substitutes it.
"""

from __future__ import annotations

import inspect
import json
import math
import ast
import hashlib
from dataclasses import replace
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import bitsandbytes as bnb
import bitsandbytes.functional as bnb_functional
import pytest
import torch as t
from torch import nn

from . import solutions


def _actual_nf4_linear(in_features: int = 64, out_features: int = 12) -> bnb.nn.Linear4bit:
    dense = nn.Linear(in_features, out_features, bias=True)
    layer = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=True,
        compute_dtype=t.float32,
        compress_statistics=True,
        quant_type="nf4",
    )
    layer.load_state_dict(dense.state_dict())
    return layer.to("cpu")


def check_nf4_codebook(fn: Callable[..., t.Tensor]) -> None:
    expected = bnb_functional.get_4bit_type("nf4", device="cpu")
    actual = fn()
    assert t.equal(actual, expected)
    assert actual.numel() == 16 and actual[7].item() == 0.0


def test_nf4_codebook_matches_bitsandbytes_exactly() -> None:
    check_nf4_codebook(solutions.nf4_codebook)


def check_blockwise_absmax(fn: Callable[..., t.Tensor]) -> None:
    values = t.tensor([-1.0, 2.0, -3.0, 0.5, 7.0])
    assert t.equal(fn(values, block_size=4), t.tensor([3.0, 7.0]))


def test_blockwise_absmax_handles_padding() -> None:
    check_blockwise_absmax(solutions.blockwise_absmax)


def check_manual_nf4(
    quantize: Callable[..., solutions.NF4Tensor],
    dequantize: Callable[..., t.Tensor],
) -> None:
    values = t.linspace(-2, 2, 64) + 0.01 * t.sin(t.arange(64))
    manual = dequantize(quantize(values, block_size=64))
    packed, state = bnb_functional.quantize_4bit(
        values,
        blocksize=64,
        compress_statistics=False,
        quant_type="nf4",
    )
    actual = bnb_functional.dequantize_4bit(packed, state).reshape_as(values)
    assert t.equal(manual, actual)
    zeros = t.zeros(3, 5)
    assert t.equal(zeros, dequantize(quantize(zeros, block_size=4)))


def test_manual_nf4_matches_real_bitsandbytes_dequantization() -> None:
    check_manual_nf4(solutions.quantize_nf4, solutions.dequantize_nf4)


def check_double_quantization(
    quantize: Callable[..., solutions.DoubleQuantizedScales],
    dequantize: Callable[[solutions.DoubleQuantizedScales], t.Tensor],
    account: Callable[..., dict[str, Any]],
) -> None:
    scales = t.tensor([0.2, 0.7, 1.2, 0.35, 0.9, 0.1, 0.55, 0.4])
    reconstructed = dequantize(quantize(scales, block_size=4))
    assert (scales - reconstructed).abs().max().item() < 0.003
    accounting = account(numel=1_048_576)
    assert accounting["single_bits_per_parameter"] == 4.5
    assert abs(accounting["double_bits_per_parameter"] - 4.126983642578125) < 1e-12
    assert accounting["paper_amortized_double_bits_per_parameter"] < accounting["double_bits_per_parameter"]
    assert accounting["metadata_savings_bits"] > 0


def test_double_quantization_and_bit_accounting_are_explicit() -> None:
    check_double_quantization(
        solutions.double_quantize_scales,
        solutions.dequantize_scales,
        solutions.nf4_storage_accounting,
    )


def check_linear4bit_storage(fn: Callable[[nn.Module], dict[str, Any]]) -> None:
    layer = _actual_nf4_linear(66, 7)
    accounting = fn(layer)
    assert accounting["original_numel"] == 462
    assert accounting["packed_weight_bytes"] == math.ceil(462 / 2)
    assert accounting["packed_weight_dtype"] == "torch.uint8"
    assert accounting["quant_state_tensors"]["quant_state.code"]["shape"] == [16]
    assert accounting["quant_state_tensors"]["quant_state.state2.code"]["shape"] == [256]
    assert all(item["all_finite"] for item in accounting["quant_state_tensors"].values())
    assert accounting["actual_tensor_bytes_including_codebooks"] > accounting["tail_aware_formula_bytes_excluding_codebooks"]


def test_actual_linear4bit_storage_includes_codebooks_and_packed_tail() -> None:
    check_linear4bit_storage(solutions.linear4bit_storage_accounting)


def check_quant_state_audit(fn: Callable[..., dict[str, Any]]) -> None:
    layer = _actual_nf4_linear()
    report = fn(layer, expected_compute_dtype=t.float32, expected_storage_dtype=t.uint8)
    assert report["nf4_codebook_exact"]
    assert report["dynamic_codebook_exact"]
    assert report["all_state_tensors_finite"]
    assert report["dequantized_weight_finite"]


def test_nested_quant_state_audit_checks_values_and_finiteness() -> None:
    check_quant_state_audit(solutions.audit_linear4bit)


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("nf4_code", "canonical NF4"),
        ("dynamic_code", "signed dynamic map"),
        ("offset", "non-finite"),
        ("nested_absmax", "non-finite"),
    ],
)
def test_nested_quant_state_audit_rejects_corrupt_real_state(field: str, expected_message: str) -> None:
    layer = _actual_nf4_linear()
    state = layer.weight.quant_state
    if field == "nf4_code":
        state.code[3] += 0.125
    elif field == "dynamic_code":
        state.state2.code[17] += 0.125
    elif field == "offset":
        state.offset.fill_(float("nan"))
    else:
        state.state2.absmax.fill_(float("inf"))
    with pytest.raises(RuntimeError, match=expected_message):
        solutions.audit_linear4bit(
            layer,
            expected_compute_dtype=t.float32,
            expected_storage_dtype=t.uint8,
        )


def check_ste(fn: Callable[..., t.Tensor]) -> None:
    weight = t.tensor([[0.2, -0.4], [0.7, -1.0]], requires_grad=True)
    output = fn(weight, block_size=2)
    expected = solutions.dequantize_nf4(solutions.quantize_nf4(weight.detach(), block_size=2))
    assert t.equal(output.detach(), expected)
    output.sum().backward()
    assert t.equal(weight.grad, t.ones_like(weight))


def test_ste_has_quantized_forward_and_identity_backward() -> None:
    check_ste(solutions.ste_nf4)


def check_toy_qlora_forward(fn: Callable[..., t.Tensor]) -> None:
    base = t.tensor([[0.4, 1.0], [0.0, -1.0]])
    a = t.tensor([[0.3, -0.2], [0.4, 0.1]], requires_grad=True)
    b = t.tensor([[0.1, -0.4], [0.2, 0.05]], requires_grad=True)
    inputs = t.tensor([[1.0, -2.0], [-0.5, 0.25]], requires_grad=True)
    weights = t.tensor([[0.7, -0.2], [0.1, 0.3]])
    scale = 5.0 / 2
    codebook = bnb_functional.get_4bit_type("nf4", device="cpu")
    # Each two-element block has unit absmax. 0.4 maps to the known NF4 entry 12.
    quantized = t.tensor([[codebook[12], 1.0], [0.0, -1.0]])
    expected = inputs.detach() @ quantized.T + scale * inputs.detach() @ a.detach().T @ b.detach().T
    output = fn(inputs, base, a, b, alpha=5.0, block_size=2)
    t.testing.assert_close(output, expected)
    (output * weights).sum().backward()
    t.testing.assert_close(a.grad, scale * b.detach().T @ weights.T @ inputs.detach())
    t.testing.assert_close(b.grad, scale * weights.T @ inputs.detach() @ a.detach().T)
    t.testing.assert_close(inputs.grad, weights @ quantized + scale * weights @ b.detach() @ a.detach())


def test_toy_qlora_forward_preserves_adapter_gradient() -> None:
    check_toy_qlora_forward(solutions.toy_qlora_forward)


def check_real_qlora_wrapper(cls: type[solutions.QLoRALinear]) -> None:
    t.manual_seed(0)
    base = _actual_nf4_linear()
    wrapper = cls(base, rank=4, alpha=10.0)
    inputs = t.randn(5, 64)
    assert t.equal(wrapper(inputs), base(inputs))
    assert not wrapper.base.weight.requires_grad
    assert wrapper.lora_a.requires_grad and wrapper.lora_b.requires_grad
    packed_before = base.weight.detach().clone()
    with t.no_grad():
        wrapper.lora_a.copy_(t.linspace(-0.2, 0.3, 4 * 64).reshape(4, 64))
        wrapper.lora_b.copy_(t.linspace(-0.1, 0.2, 12 * 4).reshape(12, 4))
    expected_delta = 2.5 * wrapper.lora_b.detach() @ wrapper.lora_a.detach()
    t.testing.assert_close(wrapper.delta_weight(), expected_delta)
    expected = base(inputs).detach() + inputs @ expected_delta.T
    actual = wrapper(inputs)
    t.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    gradient = t.ones(5, 12)
    t.testing.assert_close(wrapper.lora_a.grad, 2.5 * wrapper.lora_b.detach().T @ gradient.T @ inputs)
    t.testing.assert_close(wrapper.lora_b.grad, 2.5 * gradient.T @ inputs @ wrapper.lora_a.detach().T)
    assert base.weight.grad is None and base.bias.grad is None
    t.testing.assert_close(base.weight, packed_before, rtol=0, atol=0)


def test_real_bnb_qlora_starts_as_exact_quantized_base_and_freezes_weight() -> None:
    check_real_qlora_wrapper(solutions.QLoRALinear)


def test_update_extraction_accepts_notebook_defined_adapter() -> None:
    # Execute the actual class as a learner would, with a distinct class identity.
    namespace = dict(vars(solutions))
    namespace['__name__'] = 'completed_learner'
    exec(compile(inspect.getsource(solutions.QLoRALinear), '<learner>', 'exec'), namespace)
    learner_cls = namespace['QLoRALinear']
    assert learner_cls is not solutions.QLoRALinear
    base = _actual_nf4_linear(64, 12)
    wrapper = learner_cls(base, rank=2, alpha=5.0)
    with t.no_grad():
        wrapper.lora_a.fill_(0.2)
        wrapper.lora_b.fill_(0.3)
    model = nn.Module()
    model.proj = wrapper
    updates = solutions.extract_selected_updates(model, {'proj': t.zeros(12, 64)}, ('proj',))
    t.testing.assert_close(updates['proj'], t.full((12, 64), 0.3))
    assert solutions.freeze_except_adapters(model) == 2 * (12 + 64)
    assert not wrapper.base.weight.requires_grad and not wrapper.base.bias.requires_grad
    error = solutions.selected_quantization_error(model, {'proj': t.zeros(12, 64)}, ('proj',))
    assert math.isfinite(error['mse']) and error['mse'] > 0
    solutions.make_same_norm_random_adapter(model, target_norm=3.25, seed=7)
    t.testing.assert_close(wrapper.delta_weight().norm(), t.tensor(3.25))


def test_forward_grader_rejects_an_adapter_only_imitation() -> None:
    def omits_base(inputs, base_weight, lora_a, lora_b, *, alpha, block_size):
        return inputs @ lora_a.T @ lora_b.T

    with pytest.raises(AssertionError):
        check_toy_qlora_forward(omits_base)


def check_dequantized_base_parity(fn: Callable[[nn.Module, t.Tensor], float]) -> None:
    assert fn(_actual_nf4_linear(), t.randn(7, 64)) == 0.0


def test_real_bnb_dequantized_base_parity() -> None:
    check_dequantized_base_parity(solutions.dequantized_base_max_diff)


def test_real_bnb_adapter_gradients_leave_packed_weight_untouched() -> None:
    report = solutions.cpu_bnb_adapter_oracle(seed=0)
    assert report["dequantized_base_max_diff"] == 0.0
    assert report["base_weight_has_no_gradient"]
    assert report["lora_a_gradient_norm"] > 0 and report["lora_b_gradient_norm"] > 0


def check_inject_qlora(
    inject: Callable[..., dict[str, solutions.QLoRALinear]],
    wrapper_cls: type[solutions.QLoRALinear],
) -> None:
    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = _actual_nf4_linear(64, 64)
            self.v_proj = _actual_nf4_linear(64, 64)

    model = Attention()
    inserted = inject(model, ["q_proj"], rank=2, alpha=2.0)
    assert isinstance(model.q_proj, wrapper_cls)
    assert isinstance(model.v_proj, bnb.nn.Linear4bit)
    assert list(inserted) == ["q_proj"]
    assert all(parameter.requires_grad == ("lora_" in name) for name, parameter in model.named_parameters())


def test_inject_qlora_replaces_only_named_actual_4bit_layer() -> None:
    check_inject_qlora(solutions.inject_qlora, solutions.QLoRALinear)


def test_qlora_rejects_dense_fp4_and_unmaterialized_substitutes() -> None:
    with pytest.raises(TypeError, match="actual bitsandbytes Linear4bit"):
        solutions.QLoRALinear(nn.Linear(8, 8), rank=2, alpha=2.0)

    fp4 = bnb.nn.Linear4bit(64, 8, quant_type="fp4", compute_dtype=t.float32)
    fp4.load_state_dict(nn.Linear(64, 8).state_dict())
    with pytest.raises(TypeError, match="requires NF4"):
        solutions.QLoRALinear(fp4.to("cpu"), rank=2, alpha=2.0)

    unmaterialized = bnb.nn.Linear4bit(
        64,
        8,
        quant_type="nf4",
        compute_dtype=t.float32,
        compress_statistics=True,
    )
    with pytest.raises(TypeError, match="materialized nested-NF4"):
        solutions.QLoRALinear(unmaterialized, rank=2, alpha=2.0)


def check_nf4_model_audit(fn: Callable[..., dict[str, Any]]) -> None:
    class TinyQuantizedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = _actual_nf4_linear(64, 8)
            self.v_proj = _actual_nf4_linear(64, 8)

    model = TinyQuantizedModel()
    report = fn(
        model,
        ("q_proj", "v_proj"),
        expected_compute_dtype=t.float32,
        expected_storage_dtype=t.uint8,
    )
    assert report["intended_module_count"] == report["materialized_module_count"] == 2
    assert report["all_intended_materialized"] and report["all_nested"]
    assert report["all_state_tensors_finite"] and report["all_dequantized_weights_finite"]
    assert report["all_nf4_codebooks_exact"] and report["all_dynamic_codebooks_exact"]

    model.v_proj = bnb.nn.Linear4bit(
        64,
        8,
        compute_dtype=t.float32,
        compress_statistics=True,
        quant_type="nf4",
    )
    with pytest.raises(RuntimeError, match="unmaterialized"):
        fn(
            model,
            ("q_proj", "v_proj"),
            expected_compute_dtype=t.float32,
            expected_storage_dtype=t.uint8,
        )


def test_nested_quant_state_audit_checks_every_intended_module() -> None:
    check_nf4_model_audit(solutions.audit_nf4_model)


def _assert_crossed(examples: list[solutions.PromptExample]) -> None:
    subjects = {example.subject for example in examples}
    templates = {example.template_id for example in examples}
    counts = Counter((example.subject, example.template_id, example.label) for example in examples)
    assert set(counts.values()) == {1}
    assert set(counts) == {
        (subject, template_id, label)
        for subject in subjects
        for template_id in templates
        for label in (0, 1)
    }


def check_counterbalanced_data(
    make_examples: Callable[[], tuple[list[solutions.PromptExample], list[solutions.PromptExample]]],
    make_ood: Callable[[], list[solutions.PromptExample]],
    make_controls: Callable[..., dict[str, list[solutions.PromptExample]]],
    audit: Callable[[list[solutions.PromptExample]], dict[str, Any]],
    baseline: Callable[[list[solutions.PromptExample], list[solutions.PromptExample]], dict[str, Any]],
) -> None:
    train, heldout = make_examples()
    ood = make_ood()
    assert (len(train), len(heldout), len(ood)) == (60, 40, 40)
    _assert_crossed(train)
    _assert_crossed(heldout)
    _assert_crossed(ood)
    assert {item.subject for item in ood}.isdisjoint({item.subject for item in train + heldout})
    assert {item.template_id for item in train}.isdisjoint({item.template_id for item in heldout + ood})
    assert {item.template_id for item in heldout}.isdisjoint({item.template_id for item in ood})
    for split in (train, heldout, ood):
        report = audit(split)
        assert report["complete_subject_template_label_cross"]
        assert report["subjects_balanced"] and report["templates_balanced"]

    controls = make_controls(heldout)
    assert set(controls) == {"marker_swap", "marker_removed"}
    assert len(controls["marker_swap"]) == len(controls["marker_removed"]) == len(heldout)
    for source, swapped, removed in zip(
        heldout,
        controls["marker_swap"],
        controls["marker_removed"],
        strict=True,
    ):
        assert swapped.subject == removed.subject == source.subject
        assert swapped.template_id == removed.template_id == source.template_id
        assert swapped.label == 1 - source.label and swapped.marker == solutions.MARKERS[swapped.label]
        assert source.marker not in removed.prompt and all(marker not in removed.prompt for marker in solutions.MARKERS)
    assert baseline(train, heldout)["accuracy"] == 0.5
    assert baseline(train, ood)["accuracy"] == 0.5


def test_names_labels_templates_and_ood_are_counterbalanced_with_controls() -> None:
    check_counterbalanced_data(
        solutions.make_codebook_examples,
        solutions.make_unseen_subject_examples,
        solutions.make_marker_control_examples,
        solutions.counterbalance_audit,
        solutions.name_only_baseline,
    )


def check_random_labels(
    make_examples: Callable[[], tuple[list[solutions.PromptExample], list[solutions.PromptExample]]],
    randomize: Callable[..., list[solutions.PromptExample]],
) -> None:
    train, _ = make_examples()
    shuffled = randomize(train, seed=7)
    assert [item.prompt for item in shuffled] == [item.prompt for item in train]
    assert sum(item.label for item in shuffled) == sum(item.label for item in train)
    assert any(left.label != right.label for left, right in zip(train, shuffled, strict=True))


def test_random_label_control_preserves_inputs_and_balance() -> None:
    check_random_labels(solutions.make_codebook_examples, solutions.randomize_training_labels)


def check_training_loop(fn: Callable[..., solutions.TrainTrace]) -> None:
    """Check two optimizer steps against explicit loss, gradient, and Adam math."""

    class TinyTokenizer:
        eos_token = "<eos>"

        def encode(self, text: str, **_: Any) -> list[int]:
            return [{" red": 0, " blue": 1, " girl": 2, " sky": 3,
                     " friend": 2, " happy": 3}[text]]

        def __call__(self, prompts: list[str], **_: Any) -> dict[str, t.Tensor]:
            rows = [0 if p == "A" else 1 if p == "B" else 3 if p.startswith("Tom") else 2
                    for p in prompts]
            ids = t.tensor(rows)[:, None]
            return {"input_ids": ids, "attention_mask": t.ones_like(ids)}

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit_table = nn.Parameter(t.tensor([
                [0.4, -0.2, 0.1, 0.3], [-0.1, 0.7, 0.2, -0.4],
                [0.3, -0.5, 0.8, -0.1], [0.9, 0.2, -0.3, 0.6],
            ]))
            self.frozen_bias = nn.Parameter(t.tensor([0.2, -0.1, 0.1, -0.2]), requires_grad=False)

        def forward(self, input_ids: t.Tensor, **_: Any) -> Any:
            return SimpleNamespace(logits=self.logit_table[input_ids] + self.frozen_bias)

    model = TinyModel()
    expected = model.logit_table.detach().clone()
    bias = model.frozen_bias.detach().clone()
    generator = t.Generator().manual_seed(17)
    first_moment, second_moment = t.zeros_like(expected), t.zeros_like(expected)
    expected_losses = []
    # Independent closed-form gradients: no autograd, production loss helper,
    # or optimizer implementation is used to construct these expected updates.
    for step in (1, 2):
        rows = t.randint(2, (3,), generator=generator)
        targets = 1 - rows  # A -> blue, B -> red; do not infer labels from row IDs.
        logits = expected[rows, :2] + bias[:2]
        probabilities = logits.softmax(-1)
        loss = -(logits.log_softmax(-1)[t.arange(3), targets]).mean()
        gradient = t.zeros_like(expected)
        dlogits = probabilities.clone()
        dlogits[t.arange(3), targets] -= 1
        for row, delta in zip(rows, dlogits, strict=True):
            gradient[row, :2] += delta / 3
        replay_rows, replay_targets = t.tensor([2, 2, 3, 2]), t.tensor([2, 3, 2, 3])
        replay_logits = expected[replay_rows] + bias
        loss += 0.37 * -(replay_logits.log_softmax(-1)[t.arange(4), replay_targets]).mean()
        replay_gradient = replay_logits.softmax(-1)
        replay_gradient[t.arange(4), replay_targets] -= 1
        for row, delta in zip(replay_rows, replay_gradient, strict=True):
            gradient[row] += 0.37 * delta / 4
        gradient *= min(1.0, 1.0 / (float(gradient.norm()) + 1e-6))
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient.square()
        expected -= 0.03 * (first_moment / (1 - 0.9**step)) / (
            (second_moment / (1 - 0.999**step)).sqrt() + 1e-8
        )
        expected_losses.append(float(loss))
    examples = [solutions.PromptExample("A", 1, "a", "oracle"),
                solutions.PromptExample("B", 0, "b", "oracle")]
    trace = fn(model, TinyTokenizer(), examples, steps=2, learning_rate=0.03,
               batch_size=3, seed=17, protected_replay_weight=0.37)
    t.testing.assert_close(model.logit_table, expected, rtol=1e-5, atol=2e-7)
    t.testing.assert_close(t.tensor(trace.losses), t.tensor(expected_losses))
    assert t.equal(model.frozen_bias, bias) and model.frozen_bias.grad is None
    assert (trace.steps, trace.examples_seen) == (2, 6)
    assert (trace.trainable_parameter_count, trace.trainable_parameter_bytes) == (16, 64)
    assert trace.learning_rate == 0.03
    assert trace.optimizer_state_bytes == 132
    assert trace.optimizer_state_dtypes == {"torch.float32": 33}


def test_training_loop_matches_two_analytic_adam_steps() -> None:
    check_training_loop(solutions.train_on_codebook)


def test_training_grader_rejects_omitted_replay() -> None:
    def without_replay(*args: Any, **kwargs: Any) -> solutions.TrainTrace:
        return solutions.train_on_codebook(*args, **{**kwargs, "protected_replay_weight": 0.0})
    with pytest.raises(AssertionError):
        check_training_loop(without_replay)


def test_training_loop_updates_real_tinystories_adapter() -> None:
    model = solutions.load_dense_model(device=t.device("cpu"))
    solutions.inject_dense_lora(model, solutions.TARGET_MODULES, rank=2, alpha=2.0)
    tokenizer = solutions._load_tokenizer()
    train, _ = solutions.make_codebook_examples()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    trace = solutions.train_on_codebook(
        model,
        tokenizer,
        train,
        steps=1,
        learning_rate=1e-2,
        batch_size=2,
        seed=0,
    )
    assert trace.steps == 1 and trace.examples_seen == 2
    assert trace.trainable_parameter_count == 512
    assert trace.trainable_parameter_bytes == 2048
    assert trace.optimizer_state_bytes > 0 and trace.optimizer_state_dtypes
    assert len(trace.losses) == 1 and math.isfinite(trace.losses[0])
    assert any(
        not t.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def check_spectrum(fn: Callable[[t.Tensor], solutions.SpectrumMetrics]) -> None:
    low = fn(t.tensor([[1.0, 2.0], [2.0, 4.0]]))
    full = fn(t.eye(2))
    assert low.effective_rank < 1.01 and low.singular_values[1] < 1e-5
    assert full.effective_rank > 1.99
    zero = fn(t.zeros(3, 2))
    assert zero.effective_rank == zero.stable_rank == zero.numerical_rank == 0


def test_spectrum_recovers_low_and_full_rank() -> None:
    check_spectrum(solutions.summarize_spectrum)


def check_outlier_stress(fn: Callable[..., dict[str, Any]]) -> None:
    stress = fn(seed=0)
    small = stress["outlier"]["8"]["non_outlier_mse"]
    large = stress["outlier"]["128"]["non_outlier_mse"]
    assert large > 10 * small
    assert stress["clean"]["128"]["non_outlier_mse"] < large


def test_outlier_stress_exposes_block_coupling() -> None:
    check_outlier_stress(solutions.quantization_stress_test)


def test_three_seed_uncertainty_uses_student_t() -> None:
    stats = solutions._summary_stats([0.2, 0.5, 0.8])
    assert stats["n"] == 3
    assert math.isclose(stats["t_critical_95"], 4.302652729911275)
    assert stats["ci95_half_width"] > 1.96 * stats["sample_std"] / math.sqrt(3)
    assert "Student-t" in stats["uncertainty_label"]
    with pytest.raises(ValueError, match="exactly three measured"):
        solutions._summary_stats([0.2, 0.8])


def check_learner_bundle(bundle: solutions.LearnerFunctions) -> None:
    report = solutions.audit_implementation_provenance(
        bundle,
        require_distinct_from_reference=True,
    )
    assert report["all_callable"]
    assert report["all_code_fingerprints_present"]
    assert report["all_distinct_from_reference"]
    assert report["policy_satisfied"]


def test_reference_bundle_is_rejected_by_strict_learner_provenance() -> None:
    report = solutions.audit_implementation_provenance(
        solutions.reference_learner_functions(),
        require_distinct_from_reference=True,
    )
    assert report["all_callable"] and report["all_code_fingerprints_present"]
    assert not report["all_distinct_from_reference"]
    assert not report["policy_satisfied"]


def test_provenance_records_forward_and_rejects_simple_delegation() -> None:
    class ChangedForward(solutions.QLoRALinear):
        def forward(self, inputs: t.Tensor) -> t.Tensor:
            return inputs
    record = solutions._callable_provenance(ChangedForward)
    assert "forward" in record["fingerprinted_methods"]
    assert record["code_sha256"] != solutions._callable_provenance(solutions.QLoRALinear)["code_sha256"]
    callback = solutions.nf4_codebook
    def thin_wrapper(**kwargs: Any) -> t.Tensor:
        return callback(**kwargs)
    bundle = replace(solutions.reference_learner_functions(), nf4_codebook=thin_wrapper)
    report = solutions.audit_implementation_provenance(bundle, require_distinct_from_reference=True)
    assert report["functions"]["nf4_codebook"]["direct_reference_dependencies"] == ["nf4_codebook"]
    assert not report["no_direct_reference_delegation"]
    assert not report["policy_satisfied"]


def test_completed_learner_dropdowns_reproduce_solution_code() -> None:
    from .build_notebooks import complete_learner
    section = Path(__file__).resolve().parent
    learner = json.loads((section / "15.3_QLoRA_from_Scratch_exercises.ipynb").read_text())
    solved = json.loads((section / "15.3_QLoRA_from_Scratch_solutions.ipynb").read_text())
    completed = complete_learner(learner)
    for actual, expected in zip(completed["cells"], solved["cells"], strict=True):
        if actual["cell_type"] == "code":
            source = "".join(actual["source"])
            assert "raise NotImplementedError" not in source
            assert ast.dump(ast.parse(source)) == ast.dump(ast.parse("".join(expected["source"])))


def test_cpu_and_cuda_release_apis_are_explicit_and_fail_closed() -> None:
    cpu = inspect.signature(solutions.run_cpu_release)
    cuda = inspect.signature(solutions.run_cuda_release)
    assert cpu.parameters["implementation"].default is inspect.Parameter.empty
    assert cuda.parameters["implementation"].default is inspect.Parameter.empty
    assert "implementation" not in inspect.signature(solutions.run_gpu_test).parameters
    assert "require_distinct_from_reference=True" in inspect.getsource(solutions.run_cuda_release)
    loader = inspect.getsource(solutions.load_nf4_model)
    guard = inspect.getsource(solutions._cuda_metadata)
    assert "CUDA is required" in guard and "no CPU fallback" in guard
    assert 'device.type != "cuda"' in loader
    assert "load_in_4bit=True" in loader
    assert 'bnb_4bit_quant_type="nf4"' in loader
    assert "bnb_4bit_use_double_quant=True" in loader


def test_smoke_acceptance_is_derived_from_measured_cpu_checks() -> None:
    source = inspect.getsource(solutions.run_smoke_test)
    assert '"tests_passed": True' not in source
    assert '"accepted": True' not in source
    result = solutions.run_smoke_test(cpu=True)
    assert result["cpu"] is True
    assert result["tests_passed"] == all(result["checks"].values())
    assert result["accepted"] == result["tests_passed"]
    assert result["nested_nf4_codebook_values_exact"]
    assert result["nested_dynamic_codebook_values_exact"]
    assert result["nested_quant_state_all_finite"]
    assert (result["train_examples"], result["heldout_examples"], result["ood_examples"]) == (60, 40, 40)


def test_non_cpu_smoke_dispatches_to_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    measured = {"cpu": False, "accepted": False}
    monkeypatch.setattr(solutions, "run_gpu_test", lambda: measured)
    assert solutions.run_smoke_test(cpu=False) is measured


def test_cuda_acceptance_uses_measured_seeds_provenance_and_total_state() -> None:
    source = inspect.getsource(solutions._comparison_acceptance)
    for required in (
        "measured_seeds",
        "seed_payloads_match_run",
        "implementation_provenance_verified",
        "student_t_uncertainty_for_n3",
        "nested_quant_state_values_exact",
        "nested_quant_state_all_finite",
        "marker_swap_accuracy_min",
        "marker_swap_flip_rate_min",
        "marker_swap_pair_accuracy_min",
        "marker_removed_accuracy_max",
        "name_only_template_transfer_is_chance",
        "total_training_state_accounting_exact",
        "qlora_total_training_state_smaller_than_bf16_lora",
        "two_label_conditional_nll",
    ):
        assert required in source
    assert '"cuda_available": True' not in source
    assert '"full_path": True' not in source
    assert '"accepted": True' not in source


def test_notebook_contract_exposes_learner_method_controls_and_strict_apis() -> None:
    section = Path(__file__).resolve().parent
    notebooks = [
        section / "15.3_QLoRA_from_Scratch_exercises.ipynb",
        section / "15.3_QLoRA_from_Scratch_solutions.ipynb",
    ]
    combined = "".join(path.read_text() for path in notebooks)
    for required in (
        "NF4",
        "expected_dynamic_codebook",
        "audit_linear4bit",
        "corrupt",
        "counterbalance_audit",
        "marker_swap",
        "marker_removed",
        "name_only_baseline",
        "two-label conditional NLL",
        "Student-t",
        "resident_training_state",
        "LearnerFunctions",
        "run_cpu_release",
        "run_cuda_release",
        "Try It Yourself",
        "Bonus Anomaly Hunt",
    ):
        assert required.lower() in combined.lower()
    assert "reference.run_cpu_reference" not in combined
    assert "reference.run_gpu_test" not in combined
    assert "tests.check_learner_bundle(learner_functions)" in combined


def test_notebook_learner_checks_always_receive_explicit_functions() -> None:
    section = Path(__file__).resolve().parent
    for notebook_name in (
        "15.3_QLoRA_from_Scratch_exercises.ipynb",
        "15.3_QLoRA_from_Scratch_solutions.ipynb",
    ):
        cells = json.loads((section / notebook_name).read_text())["cells"]
        code = "\n".join(
            "".join(cell["source"])
            for cell in cells
            if cell["cell_type"] == "code"
        )
        assert "tests.test_" not in code
        for line in code.splitlines():
            if "tests.check_" in line:
                assert "(" in line and not line.rstrip().endswith("()")


def test_exercise_has_real_stubs_and_solution_has_no_placeholders() -> None:
    section = Path(__file__).resolve().parent
    exercise = (section / "15.3_QLoRA_from_Scratch_exercises.ipynb").read_text()
    solution_notebook = json.loads((section / "15.3_QLoRA_from_Scratch_solutions.ipynb").read_text())
    solution = "\n".join("".join(cell["source"]) for cell in solution_notebook["cells"])
    assert exercise.count("raise NotImplementedError") >= 12
    assert "raise NotImplementedError" not in solution
    for forbidden in ("mock", "fake cuda", "dense fallback", "placeholder"):
        assert forbidden not in solution.lower()


def test_try_it_yourself_has_executable_cpu_play_cells() -> None:
    section = Path(__file__).resolve().parent
    for notebook_name in (
        "15.3_QLoRA_from_Scratch_exercises.ipynb",
        "15.3_QLoRA_from_Scratch_solutions.ipynb",
    ):
        cells = json.loads((section / notebook_name).read_text())["cells"]
        heading = next(
            index
            for index, cell in enumerate(cells)
            if cell["cell_type"] == "markdown" and "## Try It Yourself" in "".join(cell["source"])
        )
        play_source = "".join(cells[heading + 1]["source"])
        counterfactual_source = "".join(cells[heading + 2]["source"])
        assert cells[heading + 1]["cell_type"] == cells[heading + 2]["cell_type"] == "code"
        assert "PLAY_BLOCK_SIZE" in play_source and "PLAY_OUTLIER_VALUE" in play_source
        assert "make_marker_control_examples" in counterfactual_source


def test_report_schema_uses_strict_release_metric_names() -> None:
    schema = json.loads((Path(__file__).resolve().parent / "verification_report.schema.json").read_text())
    assert set(schema["required"]) == {
        "notebook_id",
        "date_run",
        "git_commit",
        "report_inputs",
        "gt_tier",
        "evidence_level",
        "claim_scope",
        "gpu_name",
        "peak_vram_gb",
        "wall_clock_seconds",
        "models",
        "datasets",
        "tests_passed",
        "metrics",
        "baselines",
        "negative_controls",
        "ood_tests",
        "known_failures",
        "safety_notes",
        "accepted",
    }
    gpu_required = schema["properties"]["metrics"]["properties"]["gpu_test"]["required"]
    for field in (
        "measured_seeds",
        "implementation_provenance",
        "qlora_r8_two_label_conditional_nll",
        "qlora_r8_marker_swap_flip_rate",
        "qlora_r8_marker_swap_pair_accuracy",
        "qlora_r8_resident_training_state_mib",
        "checks",
        "accepted",
    ):
        assert field in gpu_required


def test_reference_evidence_is_current() -> None:
    section = Path(__file__).resolve().parent
    root = section.parents[2]
    report = json.loads((section / "expected_outputs/reference_metrics.json").read_text())
    for name, expected in report["source_sha256"].items():
        assert hashlib.sha256((section / name).read_bytes()).hexdigest() == expected, name
    for name, expected in report["figure_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    assert report["all_code_cells_executed_without_errors"]
    assert report["completed_learner_code_cells"] >= 21
    assert report["actual_test_run"]["returncode"] == 0
    if report["cuda"] is not None:
        cuda = report["cuda"]
        assert cuda["training"]["measured_seeds"] == [0, 1, 2]
        assert cuda["accepted"] == all(cuda["checks"].values())
        assert cuda["implementation_provenance"]["no_direct_reference_delegation"]
