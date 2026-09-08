"""Small independent numerical oracles for the actual image-to-word model."""
from __future__ import annotations

import pytest
import torch as t
import torch.nn.functional as F
from torch import nn

from . import language_vlm as reference
from .language_data import VOCAB


def test_learned_patches_match_explicit_pixel_windows(cls=None):
    t.manual_seed(13)
    module = (cls or reference.LearnedPatchEmbedding)(image_size=8, patch_size=2, width=5).double()
    pixels = t.rand(2, 3, 8, 8, dtype=t.float64, requires_grad=True)
    actual = module(pixels)
    windows = [(2 * pixels[:, :, row:row + 2, column:column + 2] - 1).flatten(1)
               for row in range(0, 8, 2) for column in range(0, 8, 2)]
    expected = t.stack([window @ module.projection.weight.flatten(1).T + module.projection.bias
                        for window in windows], dim=1)
    t.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    probe = t.randn_like(actual)
    leaves = (pixels, module.projection.weight, module.projection.bias)
    actual_grad = t.autograd.grad(actual, leaves, probe, retain_graph=True)
    expected_grad = t.autograd.grad(expected, leaves, probe)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        t.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)


def attention_oracle(block, hidden, valid, causal):
    batch, positions, width = hidden.shape
    normalized = F.layer_norm(hidden, (width,), block.norm1.weight, block.norm1.bias, block.norm1.eps)
    projected = F.linear(normalized, block.qkv.weight)
    q, k, v = [part.reshape(batch, positions, block.heads, width // block.heads).transpose(1, 2)
               for part in projected.chunk(3, dim=-1)]
    allowed = valid[:, None, None, :].expand(batch, 1, positions, positions)
    if causal:
        allowed = allowed & t.ones(positions, positions, dtype=t.bool).tril()
    context = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
    residual = hidden + F.linear(context.transpose(1, 2).reshape(batch, positions, width), block.attention_out.weight)
    normalized = F.layer_norm(residual, (width,), block.norm2.weight, block.norm2.bias, block.norm2.eps)
    result = residual + F.linear(F.gelu(F.linear(normalized, block.fc1.weight, block.fc1.bias)),
                                  block.fc2.weight, block.fc2.bias)
    return result.masked_fill(~valid.unsqueeze(-1), 0)


def test_word_vision_attention_values_and_gradients(cls=None):
    for causal in (False, True):
        t.manual_seed(14)
        block = (cls or reference.WordVisionBlock)(width=12, heads=3).double()
        hidden = t.randn(2, 5, 12, dtype=t.float64, requires_grad=True)
        valid = t.tensor([[True] * 5, [True, True, True, False, False]])
        actual = block(hidden, valid, causal=causal)
        expected = attention_oracle(block, hidden, valid, causal)
        t.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        probe = t.randn_like(actual)
        leaves = (hidden, *block.parameters())
        actual_grad = t.autograd.grad(actual, leaves, probe, retain_graph=True)
        expected_grad = t.autograd.grad(expected, leaves, probe)
        for left, right in zip(actual_grad, expected_grad, strict=True):
            t.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)


def test_word_attention_future_and_padding_invariance(cls=None):
    t.manual_seed(15)
    block = (cls or reference.WordVisionBlock)(width=12, heads=3).double()
    hidden = t.randn(2, 5, 12, dtype=t.float64)
    valid = t.tensor([[True] * 5, [True, True, True, False, False]])
    changed = hidden.clone()
    changed[0, 4] += 10 * t.randn(12, dtype=t.float64)
    changed[1, 3:] += 10 * t.randn(2, 12, dtype=t.float64)
    before, after = block(hidden, valid, causal=True), block(changed, valid, causal=True)
    t.testing.assert_close(before[0, :4], after[0, :4], atol=1e-12, rtol=1e-12)
    t.testing.assert_close(before[1], after[1], atol=1e-12, rtol=1e-12)
    assert not t.allclose(block(hidden, valid, causal=False)[0, :4], block(changed, valid, causal=False)[0, :4])


def test_vision_encoder_positions_and_blocks_match_explicit_oracle(cls=None):
    t.manual_seed(31)
    encoder = (cls or reference.PatchVisionEncoder)(image_size=8, patch_size=2, width=12, heads=3, layers=2).double()
    pixels = t.rand(2, 3, 8, 8, dtype=t.float64)
    cache = {}
    actual = encoder(pixels, cache=cache)
    patch = encoder.patch_embedding
    windows = [(2 * pixels[:, :, row:row + 2, column:column + 2] - 1).flatten(1)
               for row in range(0, 8, 2) for column in range(0, 8, 2)]
    expected = t.stack([window @ patch.projection.weight.flatten(1).T + patch.projection.bias
                        for window in windows], dim=1)
    t.testing.assert_close(cache["patches"], expected, atol=1e-12, rtol=1e-12)
    expected = expected + encoder.position
    valid = t.ones(2, 16, dtype=t.bool)
    for block in encoder.blocks:
        expected = attention_oracle(block, expected, valid, causal=False)
    expected = F.layer_norm(expected, (12,), encoder.final_norm.weight, encoder.final_norm.bias, encoder.final_norm.eps)
    t.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    t.testing.assert_close(cache["vision"], expected, atol=1e-11, rtol=1e-11)
    assert "position" in dict(encoder.named_parameters())


def test_multimodal_word_positions_and_padding_are_explicit(fn=None):
    fn = fn or reference.build_multimodal_word_sequence
    model = nn.Module()
    model.num_visual_tokens, model.max_text_tokens = 4, 6
    model.word_embedding = nn.Embedding(len(VOCAB), 3).double()
    model.position = nn.Parameter(t.arange(36., dtype=t.float64).reshape(1, 12, 3))
    visual = t.arange(24., dtype=t.float64).reshape(2, 4, 3).requires_grad_()
    ids = t.tensor([[VOCAB["<bos>"], VOCAB["what"], VOCAB["<answer>"]], [VOCAB["<bos>"], VOCAB["<answer>"], 0]])
    valid = t.tensor([[True, True, True], [True, True, False]])
    actual, actual_valid = fn(model, visual, ids, valid)
    expected = t.stack([t.stack([
        model.word_embedding.weight[VOCAB["<image>"]], *visual[row].unbind(),
        model.word_embedding.weight[VOCAB["</image>"]],
        *(model.word_embedding.weight[token] for token in ids[row]),
    ]) for row in range(2)]) + model.position[:, :9]
    t.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual_valid.tolist() == [[True] * 9, [True] * 8 + [False]]
    gradient = t.autograd.grad(actual.sum(), visual)[0]
    t.testing.assert_close(gradient, t.ones_like(visual), atol=0, rtol=0)
    with pytest.raises(ValueError, match="context"):
        fn(model, visual, t.zeros(2, 7, dtype=t.long), t.ones(2, 7, dtype=t.bool))


def test_teacher_forcing_shift_mask_and_analytic_gradient(fn=None):
    fn = fn or reference.teacher_forced_answer_loss
    t.manual_seed(16)
    logits = t.randn(2, 6, 9, dtype=t.float64, requires_grad=True)
    ids = t.tensor([[1, 4, 3, 6, 2, 0], [1, 5, 7, 3, 8, 2]])
    mask = t.tensor([[False, False, False, True, True, False], [False, False, False, False, True, True]])
    expected_terms = [-logits[row, target - 1].log_softmax(-1)[ids[row, target]]
                      for row, target in ((0, 3), (0, 4), (1, 4), (1, 5))]
    actual = fn(logits, ids, mask)
    t.testing.assert_close(actual, t.stack(expected_terms).mean(), atol=1e-12, rtol=1e-12)
    gradient = t.autograd.grad(actual, logits)[0]
    expected_gradient = t.zeros_like(logits)
    for row, target in ((0, 3), (0, 4), (1, 4), (1, 5)):
        expected_gradient[row, target - 1] = logits[row, target - 1].softmax(-1).detach() / 4
        expected_gradient[row, target - 1, ids[row, target]] -= .25
    t.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)


def test_activation_row_replacement_is_exact_and_differentiable(fn=None):
    fn = fn or reference.replace_activation_rows
    original = t.arange(24., dtype=t.float64).reshape(2, 3, 4).requires_grad_()
    donor = (-t.arange(24., dtype=t.float64)).reshape(2, 3, 4).requires_grad_()
    mask = t.tensor([[False, True, False], [True, False, True]])
    expected = original.detach().clone()
    expected[mask] = donor.detach()[mask]
    actual = fn(original, donor, mask)
    t.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not t.equal(original.detach(), actual.detach())
    gradient, donor_gradient = t.autograd.grad(actual.sum(), (original, donor))
    t.testing.assert_close(gradient, (~mask).unsqueeze(-1).expand_as(original).double())
    t.testing.assert_close(donor_gradient, mask.unsqueeze(-1).expand_as(donor).double())


def small_model(cls=None):
    t.manual_seed(17)
    return (cls or reference.WordMiniVLM)(image_size=8, patch_size=2, vision_width=12, width=12,
                                         heads=3, vision_layers=1, decoder_layers=2, max_text_tokens=16).double()


def test_word_model_native_patch_and_whole_sequence_parity(cls=None):
    model = small_model(cls)
    clean, corrupt = t.rand(2, 3, 8, 8, dtype=t.float64), t.rand(2, 3, 8, 8, dtype=t.float64)
    ids = t.tensor([[1, 6, 7, 5], [1, 6, 8, 5]])
    valid = t.ones_like(ids, dtype=t.bool)
    with t.no_grad():
        native_corrupt, donor_cache = model(corrupt, ids, valid, return_cache=True)
        modules = {"patches": model.vision.patch_embedding, "vision": model.vision.final_norm,
                   "projected": model.connector, "decoder_0": model.decoder[0], "decoder_1": model.decoder[1]}
        for stage, module in modules.items():
            donor = donor_cache[stage]
            rows = t.zeros(donor.shape[:2], dtype=t.bool)
            rows[0, 2], rows[1, 4] = True, True
            def hook(_module, _inputs, output):
                result = output.clone()
                result[rows] = donor[rows]
                return result
            handle = module.register_forward_hook(hook)
            try:
                expected = model(clean, ids, valid)
            finally:
                handle.remove()
            actual = model(clean, ids, valid, patch=reference.ActivationPatch(stage, rows, donor))
            t.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
            full = reference.ActivationPatch(stage, t.ones_like(rows), donor)
            t.testing.assert_close(model(clean, ids, valid, patch=full), native_corrupt, atol=1e-11, rtol=1e-11)
        # Later visual rows alone exclude question states that already read the image.
        rows = t.zeros(donor_cache["decoder_0"].shape[:2], dtype=t.bool)
        rows[:, 1:1 + model.num_visual_tokens] = True
        visual_only = model(clean, ids, valid, patch=reference.ActivationPatch("decoder_0", rows, donor_cache["decoder_0"]))
        assert not t.allclose(visual_only, native_corrupt, atol=1e-7, rtol=1e-7)


class ScriptedWords(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.calls, self.fail = 0, fail

    def forward(self, pixels, ids, valid):
        self.calls += 1
        if self.fail:
            raise RuntimeError("deliberate forward failure")
        output = t.full((*ids.shape, len(VOCAB)), -20.)
        output[:, :, VOCAB["yellow"]] = 15.
        for row in range(len(ids)):
            sequence = (["red", "<eos>"] if row == 0 else ["a", "blue", "circle", "<eos>"])
            step = min(self.calls - 1, len(sequence) - 1)
            last = int(valid[row].sum()) - 1
            if self.calls > 1:
                previous = sequence[min(self.calls - 2, len(sequence) - 1)]
                assert ids[row, last] == VOCAB[previous], "Generation must append to the actual non-pad prefix."
            output[row, last, VOCAB[sequence[step]]] = 20.
        return output


def test_greedy_words_stop_each_example_and_restore_mode(fn=None):
    fn = fn or reference.greedy_word_generation
    ids = t.tensor([[1, 5, 0], [1, 6, 5]])
    valid = t.tensor([[True, True, False], [True, True, True]])
    original_ids, original_valid = ids.clone(), valid.clone()
    model = ScriptedWords().train()
    result = fn(model, t.zeros(2, 3, 8, 8), ids, valid, max_new_tokens=6)
    assert result.lengths.tolist() == [2, 4]
    assert result.finished.tolist() == [True, True]
    assert result.tokens[0].tolist() == [VOCAB["red"], VOCAB["<eos>"], 0, 0, 0, 0]
    assert result.tokens[1, :4].tolist() == [VOCAB[word] for word in ("a", "blue", "circle", "<eos>")]
    assert model.training and model.calls == 4
    assert t.equal(ids, original_ids) and t.equal(valid, original_valid)
    failed_model = ScriptedWords(fail=True).train()
    with pytest.raises(RuntimeError, match="deliberate"):
        fn(failed_model, t.zeros(2, 3, 8, 8), ids, valid)
    assert failed_model.training


def test_full_word_loss_reaches_both_towers_and_connector(cls=None):
    model = small_model(cls)
    ids = t.tensor([[1, 6, 7, 5, VOCAB["red"], 2], [1, 6, 8, 5, VOCAB["square"], 2]])
    valid = t.ones_like(ids, dtype=t.bool)
    targets = t.tensor([[False, False, False, False, True, True]] * 2)
    logits = model(t.rand(2, 3, 8, 8, dtype=t.float64), ids, valid)
    reference.teacher_forced_answer_loss(logits, ids, targets).backward()
    for parameter in (model.vision.patch_embedding.projection.weight, model.vision.blocks[0].qkv.weight,
                      model.connector[0].weight, model.word_embedding.weight,
                      model.decoder[0].qkv.weight, model.unembed.weight):
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
