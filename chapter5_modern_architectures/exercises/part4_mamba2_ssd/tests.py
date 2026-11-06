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


def test_recall_orchestration(experiment_fn=reference.run_delayed_recall_experiment) -> None:
    """Observe the actual orchestration without retraining all three models."""
    from unittest.mock import patch

    training = []
    evaluation = []

    def record_training(model, **kwargs):
        training.append(kwargs)
        return [0.5] * kwargs["steps"]

    def record_evaluation(model, **kwargs):
        evaluation.append(kwargs)
        return len(evaluation) / 10

    with patch.dict(experiment_fn.__globals__, {"train_recall_model": record_training, "evaluate_recall_model": record_evaluation}):
        result = experiment_fn(steps=17, sequence_length=48)
    assert [call["steps"] for call in training] == [17, 17, 17], 'Train the main architecture and both controls for the same requested step count.'
    assert [call["sequence_length"] for call in evaluation] == [48, 48, 48, 48, 96], 'Evaluate all in-distribution conditions at the requested length and OOD at twice that length.'
    assert [call["seed"] for call in training] == [0, 1, 2]
    assert [call.get("shuffled_labels", False) for call in training] == [False, False, True]
    assert evaluation[1]["carry_scale"] == 0.0
    assert [result.mamba_accuracy, result.state_ablated_accuracy, result.local_conv_accuracy, result.shuffled_label_accuracy, result.ood_length_accuracy] == [.1, .2, .3, .4, .5]
    print("All tests in `test_recall_orchestration` passed!")



def test_scaling_storage_without_benchmark(scaling_fn=reference.measure_ssd_scaling, result_cls=reference.ScalingResult):
    """Exact byte counts for L=5/8/9, Q=4; replace only the timer."""
    from unittest.mock import patch
    with patch.dict(scaling_fn.__globals__, {"_median_runtime_ms": lambda *args: 7.0}):
        result = scaling_fn((5, 8, 9), chunk_size=4, repeats=1)
    assert isinstance(result, result_cls)
    assert result.lengths == [5, 8, 9]
    assert result.quadratic_matrix_mb == [200 / 2**20, 512 / 2**20, 648 / 2**20], "Two heads, four-byte elements, and L squared entries."
    assert result.chunk_local_matrix_mb == [256 / 2**20, 256 / 2**20, 384 / 2**20], "Pad L to a multiple of Q before counting chunk-local entries."
    assert result.recurrent_ms == result.quadratic_ms == result.chunked_ms == [7., 7., 7.]
    print("All tests in `test_scaling_storage_without_benchmark` passed!")


def test_exact_metrics_on_hand_oracle(metric_fn=reference.exact_parity_metrics):
    from unittest.mock import patch
    # Inject deliberately different paths: a metric cannot pass by returning zeros.
    x = t.tensor([1., -2.]); state = t.tensor([3.])
    calls = []
    def inputs(**kwargs):
        calls.append(kwargs)
        return (None,) * 4
    with patch.dict(metric_fn.__globals__, {
        "make_ssd_inputs": inputs,
        "ssd_recurrent": lambda *args: (x, state),
        "ssd_quadratic": lambda *args: t.tensor([1.5, -2.]),
        "ssd_chunked": lambda *args: (t.tensor([1., -3.]), t.tensor([5.])),
    }):
        result = metric_fn(length=3, chunk_size=2, device="cpu")
    assert result == {"recurrent_quadratic_max_abs_diff": .5, "recurrent_chunked_max_abs_diff": 1., "final_state_max_abs_diff": 2.}
    assert calls == [{"length": 3, "device": "cpu"}]
    print("All tests in `test_exact_metrics_on_hand_oracle` passed!")


def test_released_mapping_on_synthetic_tensors(mapping_fn=reference.native_to_released_state_dict):
    native = {"backbone.embedding.weight": t.tensor([[1., -2.]], dtype=t.float16)}
    native.update({f"untouched.{i}": t.tensor([i], dtype=t.float16) for i in range(218)})
    before = {key: value.clone() for key, value in native.items()}
    actual = mapping_fn(native)
    assert set(actual) == (set(native) - {"backbone.embedding.weight"}) | {"backbone.embeddings.weight"}
    assert all(v.dtype == t.float32 for v in actual.values())
    for key, value in before.items():
        mapped = "backbone.embeddings.weight" if key == "backbone.embedding.weight" else key
        t.testing.assert_close(actual[mapped], value.float(), rtol=0, atol=0)
        t.testing.assert_close(native[key], value, rtol=0, atol=0)
    with pytest.raises(AssertionError, match="219"):
        mapping_fn({"backbone.embedding.weight": native["backbone.embedding.weight"]})
    print("All tests in `test_released_mapping_on_synthetic_tensors` passed!")


def test_released_gate_order(norm_cls=reference.ReleasedGatedRMSNorm):
    # exp(log(3))=3, so SiLU(log(3))=.75*log(3). The second gate is zero.
    layer = norm_cls(2, eps=0.)
    with t.no_grad(): layer.weight.copy_(t.tensor([2., 3.]))
    actual = layer(t.tensor([[3., 4.]]), t.tensor([[math.log(3.), 0.]]))
    t.testing.assert_close(actual, t.tensor([[2 * math.sqrt(2), 0.]]), rtol=1e-6, atol=1e-6)
    print("All tests in `test_released_gate_order` passed!")


def test_released_mixer_on_tiny_zero_projection(mixer_cls=reference.ReleasedMamba2MixerFromScratch, config_cls=reference.ReleasedMamba2Config):
    config = config_cls(vocab_size=3, d_model=2, n_layers=1, d_inner=2, n_heads=1, head_dim=2, d_state=1, n_groups=1, d_conv=2, chunk_size=2)
    mixer = mixer_cls(config)
    with t.no_grad():
        for parameter in mixer.parameters(): parameter.zero_()
        mixer.norm.weight.fill_(1.)
        mixer.out_proj.weight.copy_(t.eye(2))
    inputs = t.tensor([[[3., 4.], [1., 2.], [-2., 5.]]])
    actual = mixer(inputs)
    assert actual.shape == inputs.shape
    t.testing.assert_close(actual, t.zeros_like(inputs), rtol=0, atol=0)
    print("All tests in `test_released_mixer_on_tiny_zero_projection` passed!")


def test_released_lm_on_tiny_configuration(lm_cls=reference.ReleasedMamba2LMFromScratch, config_cls=reference.ReleasedMamba2Config):
    # No layer is needed to hand-check embedding -> final RMS -> head orientation.
    config = config_cls(vocab_size=2, d_model=2, n_layers=0, d_inner=2, n_heads=1, head_dim=2, d_state=1, eps=0.)
    model = lm_cls(config)
    with t.no_grad():
        model.backbone.embeddings.weight.copy_(t.tensor([[3., 4.], [0., 2.]]))
        model.backbone.norm_f.weight.fill_(1.)
        model.lm_head.weight.copy_(t.eye(2))
    actual, hidden = model(t.tensor([[0, 1]]))
    expected = t.tensor([[[3 / math.sqrt(12.5), 4 / math.sqrt(12.5)], [0., math.sqrt(2.)]]])
    assert len(hidden) == 1
    t.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    t.testing.assert_close(hidden[0], expected, rtol=1e-6, atol=1e-6)
    print("All tests in `test_released_lm_on_tiny_configuration` passed!")


def test_checkpoint_parity_assembly_without_weights(parity_fn=reference.verify_released_mamba2_parity, result_cls=reference.ReleasedCheckpointParity):
    from types import SimpleNamespace
    from unittest.mock import patch
    import sys
    seen = []
    class Fake(t.nn.Module):
        def __init__(self, *args): super().__init__()
        def load_state_dict(self, state, **kwargs): seen.append((state, kwargs))
        def forward(self, ids, **kwargs):
            assert tuple(tuple(row) for row in ids.tolist()) == ((46,31834,3210,476),(510,5798,2206,327))
            logits = t.tensor([[[0., 4.]], [[5., 0.]]])
            if kwargs:
                return SimpleNamespace(logits=logits + .25, hidden_states=(t.tensor([1., 4.]), t.tensor([3., 4.])))
            return logits, (t.tensor([1., 2.]), t.tensor([3., 4.]))
    with patch.dict(sys.modules, {"transformers": SimpleNamespace(Mamba2ForCausalLM=Fake)}), patch.dict(parity_fn.__globals__, {
        "_load_released_state_dict": lambda: {"sentinel": t.tensor(0.)},
        "ReleasedMamba2LMFromScratch": Fake,
        "ReleasedMamba2Config": lambda: None,
        "_released_hf_config": lambda: None,
    }), patch.object(t.cuda, "is_available", return_value=False):
        result = parity_fn("cpu")
    assert isinstance(result, result_cls)
    assert len(seen) == 2 and all(kwargs == {"strict": True, "assign": True} for _, kwargs in seen)
    assert result.mapped_tensor_count == 1 and result.hidden_state_count == 2
    assert result.hidden_max_abs_diff == 2. and result.hidden_mean_abs_diff == 1.
    assert result.logits_max_abs_diff == result.logits_mean_abs_diff == .25
    assert result.top1_agreement == 1. and result.reference_top_tokens == result.learner_top_tokens == (1, 0)
    print("All tests in `test_checkpoint_parity_assembly_without_weights` passed!")


def test_tiny_recall_carry_intervention(model_cls=reference.TinyMamba2RecallModel):
    """A prefix outside the convolution window reaches READ only through state."""
    with t.random.fork_rng():
        t.manual_seed(0)
        model = model_cls().eval()
    inputs = t.zeros(1, 8, 4)
    inputs[:, 0, 0] = 1.
    inputs[:, 0, 1] = inputs[:, -1, 2] = 1.
    changed = inputs.clone(); changed[:, 0, 0] = -1.
    with t.no_grad():
        normal, different = model(inputs), model(changed)
        ablated = model(inputs, carry_scale=0.)
        ablated_changed = model(changed, carry_scale=0.)
    assert normal.shape == (1, 2)
    assert float((normal - different).abs().max()) > 1e-7, "The retained state must transmit a prefix outside the local convolution window."
    t.testing.assert_close(ablated, ablated_changed, rtol=0, atol=0)
    print("All tests in `test_tiny_recall_carry_intervention` passed!")


def test_released_mixer_literal_scalar_oracle(mixer_cls=reference.ReleasedMamba2MixerFromScratch, config_cls=reference.ReleasedMamba2Config):
    """Two value channels, one state/head, two tokens; independent Python arithmetic."""
    config = config_cls(vocab_size=3, d_model=1, n_layers=1, d_inner=2, n_heads=1, head_dim=2, d_state=1, n_groups=1, d_conv=1, chunk_size=3, eps=1e-5)
    model = mixer_cls(config)
    with t.no_grad():
        model.in_proj.weight.copy_(t.tensor([[.2], [-.3], [2.], [-1.], [.4], [-.6], [.1]]))
        model.conv1d.weight.fill_(1.);model.conv1d.bias.zero_()
        model.dt_bias.fill_(-.7);model.A_log.fill_(math.log(.2));model.D.fill_(.3)
        model.norm.weight.copy_(t.tensor([1.2, .8]));model.out_proj.weight.copy_(t.tensor([[1., 2.]]))
    state = [0., 0.];expected = []
    silu = lambda v: v / (1 + math.exp(-v))
    for u in (1., -.5):
        x = [silu(2*u), silu(-u)];b,c = silu(.4*u), silu(-.6*u)
        dt = math.log1p(math.exp(.1*u-.7))
        state = [math.exp(-.2*dt)*old + dt*value*b for old,value in zip(state,x)]
        y = [s*c + .3*value for s,value in zip(state,x)]
        gated = [value*silu(z*u) for value,z in zip(y,(.2,-.3))]
        scale = math.sqrt(sum(v*v for v in gated)/2 + 1e-5)
        normalized = [v*w/scale for v,w in zip(gated,(1.2,.8))]
        expected.append(normalized[0] + 2*normalized[1])
    actual = model(t.tensor([[[1.], [-.5]]]))
    t.testing.assert_close(actual.flatten(), t.tensor(expected), rtol=3e-6, atol=3e-6)
    print("All tests in `test_released_mixer_literal_scalar_oracle` passed!")
