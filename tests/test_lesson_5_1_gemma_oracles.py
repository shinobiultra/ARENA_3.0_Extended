"""5.1 Gemma from scratch: exact attention-mask and projection-layout checks, no model loads."""
from __future__ import annotations

import importlib

import pytest



def test_exact_mask_and_negative_control_without_model():
    import torch as t

    lesson = importlib.import_module("chapter5_modern_architectures.exercises.part1_gemma_from_scratch.solutions")
    kwargs = dict(query_length=2, key_length=5, sliding_window=3, device=t.device("cpu"))
    correct = lesson.build_causal_attention_mask(past_length=3, **kwargs)
    expected = t.tensor([[[[False, True, True, True, False], [False, False, True, True, True]]]])
    assert t.equal(correct, expected)
    wrong = lesson.build_causal_attention_mask(past_length=0, **kwargs)
    with pytest.raises(AssertionError):
        t.testing.assert_close(wrong, expected)
    assert int((correct != wrong).sum()) == 9


def test_convention_basis_inverse_is_not_assumed_and_input_is_unmodified():
    import torch as t

    lesson = importlib.import_module("chapter5_modern_architectures.exercises.part1_gemma_from_scratch.solutions")
    weights = t.arange(48).reshape(16, 3)
    before = weights.clone()
    converted = lesson.hf_split_to_interleaved_projection(weights, 2)
    t.testing.assert_close(weights, before)
    assert not t.equal(lesson.hf_split_to_interleaved_projection(converted, 2), weights)
    for weight, heads in [(t.zeros(3, 2), 1), (t.zeros(4), 1), (t.zeros(4, 2), 0)]:
        with pytest.raises(ValueError):
            lesson.hf_split_to_interleaved_projection(weight, heads)
