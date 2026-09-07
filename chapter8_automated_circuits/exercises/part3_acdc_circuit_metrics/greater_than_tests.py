"""Independent probability/gradient and year-token boundary tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

from . import greater_than as reference
from .real_edges import GPT2_REVISION


def test_greater_than_probabilities_and_gradient(fn=None):
    fn = fn or reference.greater_than_probability_difference
    # Put numeric suffixes in a non-contiguous permuted vocabulary, with extra words.
    generator = t.Generator().manual_seed(61)
    ids = t.randperm(117, generator=generator)[:100]
    logits = t.randn(3, 117, dtype=t.float64, generator=generator, requires_grad=True)
    starts = t.tensor([0, 37, 99])
    probabilities = logits.softmax(-1).detach()
    expected = []
    signs = t.zeros_like(logits)
    for row, start in enumerate(starts.tolist()):
        score = 0.
        for suffix, token in enumerate(ids.tolist()):
            sign = 1 if suffix > start else -1
            score += sign * float(probabilities[row, token])
            signs[row, token] = sign
        expected.append(score)
    result = fn(logits, ids, starts)
    t.testing.assert_close(result, t.tensor(expected, dtype=t.float64))
    gradient = t.autograd.grad(result.mean(), logits)[0]
    exact_gradient = probabilities * (signs - t.tensor(expected, dtype=t.float64)[:, None]) / 3
    t.testing.assert_close(gradient, exact_gradient, atol=1e-10, rtol=1e-7)
    assert result[-1] < 0, "At 99 there are no valid later suffixes."
    with pytest.raises(ValueError):
        fn(logits, ids, t.tensor([0, 37, 100]))


def test_non_numeric_mass_is_not_renormalized(fn=None):
    fn = fn or reference.greater_than_probability_difference
    ids = t.arange(100)
    logits = t.full((1, 101), -100., dtype=t.float64)
    logits[0, 75] = 0.
    low_other = fn(logits, ids, t.tensor([50]))
    logits[0, 100] = 0.
    high_other = fn(logits, ids, t.tensor([50]))
    assert float(low_other) == pytest.approx(1.)
    assert float(high_other) == pytest.approx(.5), "Other-vocabulary mass dilutes this task metric."


@pytest.mark.requires_model
def test_greater_than_matched_tokens_and_split_instances():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=GPT2_REVISION, local_files_only=True)
    tokenizer.add_bos_token = False
    tokenizer.pad_token = tokenizer.eos_token
    model = SimpleNamespace(tokenizer=tokenizer)
    discovery = reference.prepare_greater_than_data(model, count=32, seed=11)
    development = reference.prepare_greater_than_data(model, count=32, seed=12, split="development")
    assert not (set(discovery["prompts"]) & set(development["prompts"]))
    for data in (discovery, development):
        assert len(set(data["prompts"])) == 32
        for row, prompt in enumerate(data["prompts"]):
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            end = int(data["end_positions"][row])
            assert end == len(tokens) - 1
            assert data["clean_tokens"][row, :end + 1].tolist() == tokens
            changes = (data["clean_tokens"][row] != data["corrupt_tokens"][row]).nonzero().flatten().tolist()
            assert changes == [data["changed_positions"][row]]
            assert tokenizer.decode([int(data["corrupt_tokens"][row, changes[0]])]) == "01"
            assert tokenizer.decode([tokens[changes[0]]]) == f"{int(data['start_suffixes'][row]):02d}"
        assert data["suffix_ids"].unique().numel() == 100


def test_wrong_conditional_probability_rejected():
    def wrong(logits, ids, starts):
        numeric = logits[:, ids].softmax(-1)
        later = t.arange(100)[None, :] > starts[:, None]
        return (numeric * t.where(later, 1., -1.)).sum(-1)
    with pytest.raises(AssertionError):
        test_non_numeric_mass_is_not_renormalized(wrong)
