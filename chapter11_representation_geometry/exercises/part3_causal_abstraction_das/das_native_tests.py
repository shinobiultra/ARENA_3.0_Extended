"""Fast native plumbing regression on an explicitly random tiny GPT-2.

This checks library semantics, not learned behavior. The separate native-checks
artifact records the same path on the actual pinned pretrained GPT-2 weights.
"""

import re
import pytest
import torch as t
from transformers import GPT2Config, GPT2LMHeadModel

from .das_native import capture_native, continue_native, full_forward_intervention


class IntegerTokenizer:
    def __call__(self, texts, **kwargs):
        lengths = [len(text.split()) for text in texts]
        maximum = max(lengths)
        ids, offsets, masks = [], [], []
        for text, length in zip(texts, lengths):
            pad = maximum - length
            ids.append([0] * pad + [int(word) for word in text.split()])
            offsets.append([(0, 0)] * pad + [match.span() for match in re.finditer(r"\d+", text)])
            masks.append([0] * pad + [1] * length)
        return dict(
            input_ids=t.tensor(ids),
            attention_mask=t.tensor(masks),
            offset_mapping=t.tensor(offsets),
        )


def tiny_fixture():
    with t.random.fork_rng():
        t.manual_seed(381)
        config = GPT2Config(
            n_layer=3,
            n_head=3,
            n_embd=12,
            vocab_size=31,
            n_positions=32,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).double().eval().requires_grad_(False)
    rows = [dict(prompt="1 2 3 4", span=[4, 5]), dict(prompt="5 6 7", span=[2, 3])]
    cache = capture_native(model, IntegerTokenizer(), rows, layers=(0, 1, 2), batch_size=1)
    return model, cache


def test_cached_native_logits_and_gradients(continue_function=None):
    if continue_function is None:
        continue_function = continue_native
    model, cache = tiny_fixture()
    indices = t.tensor([1, 0, 1])  # Repeated recipients must work during sampling.
    for layer in (0, 1, 2):
        for mode in ("subject", "end"):
            replacement = (cache.activations(layer, mode)[indices].flip(0) + 0.07).requires_grad_()
            full = full_forward_intervention(model, cache, layer, indices, replacement, mode)
            tail = continue_function(model, cache, layer, indices, replacement, mode)
            t.testing.assert_close(tail, full, atol=1e-12, rtol=1e-12)
            full_grad = t.autograd.grad(full[:, 3].sum(), replacement)[0]
            tail_grad = t.autograd.grad(tail[:, 3].sum(), replacement)[0]
            t.testing.assert_close(full_grad, tail_grad, atol=1e-12, rtol=1e-12)
            assert tail_grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    print("Native continuation logits and gradient tests passed!")


def test_native_noop_and_hook_cleanup():
    model, cache = tiny_fixture()
    indices = t.arange(2)
    with t.no_grad():
        noop = continue_native(model, cache, 1, indices, cache.activations(1, "subject"))
    t.testing.assert_close(noop, cache.logits, atol=1e-12, rtol=1e-12)
    hooks = len(model.transformer.h[1]._forward_pre_hooks)
    with pytest.raises((RuntimeError, ValueError)):
        full_forward_intervention(model, cache, 1, indices, t.zeros(2, 5))
    assert len(model.transformer.h[1]._forward_pre_hooks) == hooks
    model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        continue_native(model, cache, 1, indices, cache.activations(1, "subject"))
