"""Exact geometry, unfiltered nulls and a known image-to-word causal oracle."""
from __future__ import annotations

import torch as t
from torch import nn

from . import language_patching as reference
from .language_data import VOCAB
from .language_vlm import greedy_word_generation


def test_pixel_rows_and_unrestricted_controls_do_not_exclude_objects(row_fn=None, control_fn=None):
    row_fn = row_fn or reference.visual_rows_from_pixel_mask
    control_fn = control_fn or reference.sample_visual_row_controls
    pixels = t.zeros(2, 8, 8, dtype=t.bool)
    pixels[0, 1:3, 3:5] = True
    pixels[1, 5:7, 1:3] = True
    actual = row_fn(pixels, patch_size=2)
    expected = t.zeros(2, 16, dtype=t.bool)
    expected[0, [1, 2, 5, 6]] = True
    expected[1, [8, 9, 12, 13]] = True
    assert t.equal(actual, expected)
    controls = control_fn(actual, samples=16, seed=701)
    for name, samples in controls.items():
        assert len(samples) == 16
        for mask in samples:
            assert t.equal(mask.sum(1), actual.sum(1))
            if name == "background":
                assert not (mask & actual).any()
    assert any((mask & actual).any() for mask in controls["unrestricted_random"])


class ImageWordOracle(nn.Module):
    """A first word reads the image; later word states retain that information."""
    num_visual_tokens = 2

    def forward(self, pixels, ids, valid, *, patch=None, return_cache=False):
        visual = pixels[:, 0, 0, 0, None, None].expand(-1, 2, 1).clone()
        if patch is not None and patch.stage == "vision":
            visual[patch.row_mask] = patch.donor[patch.row_mask]
        hidden = visual.mean(1, keepdim=True).expand(-1, 4 + ids.shape[1], -1).clone()
        if patch is not None and patch.stage == "decoder_0":
            hidden[patch.row_mask] = patch.donor[patch.row_mask]
        logits = t.full((*ids.shape, len(VOCAB)), -20.)
        tokens = t.where(hidden[:, 4:, 0] > 0, VOCAB["red"], VOCAB["blue"])
        tokens = t.where((ids == VOCAB["red"]) | (ids == VOCAB["blue"]), VOCAB["<eos>"], tokens)
        logits.scatter_(-1, tokens.unsqueeze(-1), 20.)
        cache = {"vision": visual, "decoder_0": hidden}
        return (logits, cache) if return_cache else logits


def test_generation_full_sequence_includes_image_informed_text_states(fn=None):
    fn = fn or reference.generate_with_visual_patch
    model = ImageWordOracle().train()
    clean, donor = t.ones(2, 3, 8, 8), -t.ones(2, 3, 8, 8)
    ids = t.tensor([[VOCAB["<bos>"], VOCAB["<answer>"]]] * 2)
    valid = t.ones_like(ids, dtype=t.bool)
    rows = t.ones(2, 2, dtype=t.bool)
    native = greedy_word_generation(model, donor, ids, valid, max_new_tokens=4)
    early = fn(model, clean, donor, ids, valid, stage="vision", visual_rows=rows, max_new_tokens=4)
    late_visual = fn(model, clean, donor, ids, valid, stage="decoder_0", visual_rows=rows, max_new_tokens=4)
    late_full = fn(model, clean, donor, ids, valid, stage="decoder_0", visual_rows=rows,
                   full_sequence=True, max_new_tokens=4)
    assert (native.tokens[:, 0] == VOCAB["blue"]).all()
    assert t.equal(early.tokens, native.tokens) and t.equal(late_full.tokens, native.tokens)
    assert (late_visual.tokens[:, 0] == VOCAB["red"]).all()
    assert model.training


def test_targeted_flip_denominators_exclude_failed_endpoints(fn=None):
    fn = fn or reference.conditional_targeted_flips
    result = fn(t.tensor([True, True, False, True]), t.tensor([True, False, True, True]),
                t.tensor([True, True, True, False]))
    assert result == {"examples": 4, "eligible": 2, "targeted_flips": 1,
                      "conditional_flip_rate": .5, "unconditional_donor_answer_rate": .75}
    result = fn(t.tensor([False]), t.tensor([True]), t.tensor([True]))
    assert result["eligible"] == 0 and result["conditional_flip_rate"] is None
