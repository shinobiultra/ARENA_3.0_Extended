"""Independent semantic checks, callable from each learner implementation cell."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch as t

from . import direction_methods as reference


def test_fit_mean_direction(fn=reference.fit_mean_direction):
    x = t.tensor([[4., 1.], [6., 3.], [0., 1.], [2., 3.]], dtype=t.float64)
    y = t.tensor([True, True, False, False])
    fitted = fn(x, y)
    t.testing.assert_close(fitted["difference"], t.tensor([4., 0.], dtype=t.float64))
    t.testing.assert_close(fitted["unit"], t.tensor([1., 0.], dtype=t.float64))
    assert fitted["threshold"] == 3 and fitted["norm"] == 4, 'The class means differ by length 4 and their projected midpoint is 3; use both training classes.'
    translated = fn(x + t.tensor([10., -4.]), y)
    t.testing.assert_close(translated["unit"], fitted["unit"])
    assert translated["threshold"] == 13, 'Translating training activations by +10 along the direction must move the threshold from 3 to 13.'
    with pytest.raises(ValueError, match="Identical"):
        fn(t.ones(4, 2), y)
    print("Training-only direction and threshold tests passed!")


def test_pairwise_auroc(fn=reference.pairwise_auroc):
    labels = t.tensor([True, True, False, False])
    assert fn(t.tensor([3., 1., 1., 0.]), labels) == .875, 'A positive-negative tie contributes half a win: this four-pair AUROC must be 0.875.'
    assert fn(t.ones(4), labels) == .5, 'Constant scores have AUROC 0.5 regardless of which class comes first.'
    assert fn(t.tensor([0., 0., 1., 1.]), labels) == 0, 'Ranking every positive below every negative must give AUROC 0, not an absolute or flipped score.'
    assert fn(t.tensor([1., 1., 0., 0.]), labels) == 1, 'Ranking every positive above every negative must give AUROC 1.'
    with pytest.raises(ValueError):
        fn(t.ones(4), t.ones(4, dtype=t.bool))
    print("AUROC tie and ordering tests passed!")


def test_evaluate_frozen_direction(fn=reference.evaluate_frozen_direction):
    fitted = {"unit": t.tensor([1., 0.]), "threshold": t.tensor(3.)}
    x = t.tensor([[14., 0.], [16., 0.], [10., 0.], [12., 0.]])
    labels = t.tensor([True, True, False, False])
    out = fn(x, labels, fitted)
    # Perfect ranking but a translated test distribution defeats the fixed threshold.
    assert out["accuracy"] == .5 and out["auroc"] == 1, 'The shifted examples have perfect ranking but only 50% fixed-threshold accuracy; do not fit on evaluation labels.'
    assert bool(out["predictions"].all()) and fitted["threshold"] == 3, 'All shifted scores exceed the original threshold 3; leave the fitted threshold unchanged.'
    flipped = fn(x, ~labels, fitted)
    t.testing.assert_close(out["scores"], flipped["scores"])
    assert t.equal(out["predictions"], flipped["predictions"]), 'Changing evaluation labels must never change the predictions from a frozen direction.'
    print("Frozen evaluation rejects threshold leakage!")


def test_direction_null_ensemble(fn=reference.direction_null_ensemble):
    x = t.randn(24, 7, generator=t.Generator().manual_seed(1), dtype=t.float64)
    labels = t.arange(24) < 12
    x[labels, 0] += 3
    nulls = fn(x, labels, samples=32, seed=901)
    repeated = fn(x, labels, samples=32, seed=901)
    assert len(nulls["random"]) == len(nulls["permuted"]) == 32, 'Retain exactly all 32 random and all 32 training-permutation draws.'
    has_mixed_labels = False
    for row, other, random in zip(nulls["permuted"], repeated["permuted"], nulls["random"], strict=True):
        permutation = row["permutation"]
        assert t.equal(permutation.sort().values, t.arange(24)), 'A permutation must contain every training row exactly once.'
        assert t.equal(row["labels"], labels[permutation]), 'Apply the saved permutation to training labels before refitting each null direction.'
        assert t.equal(permutation, other["permutation"]), 'The same seed must reproduce the same permutation at each draw.'
        has_mixed_labels |= not t.equal(row["labels"], labels) and not t.equal(row["labels"], ~labels)
        positive = x[row["labels"]].mean(0); negative = x[~row["labels"]].mean(0)
        unit = (positive - negative) / (positive - negative).norm()
        t.testing.assert_close(row["fitted"]["unit"], unit)
        t.testing.assert_close(row["fitted"]["threshold"], ((positive + negative) / 2) @ unit)
        t.testing.assert_close(random["difference"].norm(), nulls["target"]["norm"])
        t.testing.assert_close(row["fitted"]["difference"].norm(), nulls["target"]["norm"])
    assert has_mixed_labels, "Flipping all labels is not a permutation ensemble"
    print("All training permutations and matched-norm random draws checked!")


def test_direction_geometry(fn=reference.direction_geometry):
    x = t.tensor([[1., 10.], [1., -10.], [-1., 10.], [-1., -10.]], dtype=t.float64)
    fitted = {"unit": t.tensor([1., 0.], dtype=t.float64)}
    out = fn(x, fitted)
    t.testing.assert_close(out["variance_fraction"], t.tensor([100/101, 1/101], dtype=t.float64))
    assert out["pc1_alignment"] < 1e-12, 'The largest centered PCA variance is a nuisance axis here, orthogonal to the class direction.'  # Strong PCA nuisance need not be the class direction.
    translated = fn(x + 7, fitted)
    t.testing.assert_close(out["singular_values"], translated["singular_values"])
    print("PCA nuisance-axis and centering tests passed!")


def test_edit_residual(fn=reference.edit_residual):
    x = t.tensor([[[3., 5., 7.], [4., 1., 2.]]], dtype=t.float64, requires_grad=True)
    vector = t.tensor([3., 4., 0.], dtype=t.float64)
    original = x.detach().clone(); unit = vector / 5
    out = fn(x, vector, mode="ablate", strength=1.)
    matrix = t.eye(3, dtype=t.float64) - unit[:, None] * unit[None, :]
    t.testing.assert_close(out, x @ matrix)
    t.testing.assert_close(t.autograd.grad(out.square().sum(), x)[0], 2 * (x @ matrix) @ matrix)
    t.testing.assert_close(fn(out, vector, mode="ablate", strength=1.), out)
    t.testing.assert_close(fn(x, vector, mode="add", strength=.5), x + .5 * vector)
    t.testing.assert_close(fn(x, vector, mode="ablate", strength=0.), x)
    t.testing.assert_close(x.detach(), original)
    assert out.dtype == x.dtype, 'Residual editing must preserve the activation dtype.'
    print("Projection geometry, non-mutation and gradients passed!")


def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    with t.random.fork_rng():
        t.manual_seed(12)
        config = Qwen2Config(vocab_size=29, hidden_size=12, intermediate_size=24,
                            num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=3,
                            max_position_embeddings=64, tie_word_embeddings=False,
                            attention_dropout=0., pad_token_id=0, eos_token_id=2)
        config._attn_implementation = "sdpa"
        return Qwen2ForCausalLM(config).eval().double()


def tiny_batch():
    return {"input_ids": t.tensor([[0, 0, 3, 4, 5], [3, 6, 7, 8, 9]]),
            "attention_mask": t.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])}


def test_cache_qwen_residuals(fn=reference.cache_qwen_residuals):
    model, encoded = tiny_qwen(), tiny_batch()
    original_hooks = len(model.model.layers[0]._forward_pre_hooks)
    cached = fn(model, encoded, offsets=(-2, -1), batch_size=1)
    assert cached.shape == (2, 2, 3, 12), 'Cache axes must be layer, example, requested suffix positions plus first valid token, and hidden width.'
    with t.inference_mode():
        positions = encoded["attention_mask"].cumsum(1).sub(1).clamp_min(0)
        native = model(**encoded, position_ids=positions, output_hidden_states=True)
    for layer in range(2):
        for row, first in ((0, 2), (1, 0)):
            t.testing.assert_close(cached[layer, row], native.hidden_states[layer][row, [3, 4, first]].float(), atol=1e-6, rtol=1e-5)
    assert len(model.model.layers[0]._forward_pre_hooks) == original_hooks, 'Remove only your temporary caching hooks after the forward pass.'
    with pytest.raises(ValueError, match="left-padded"):
        fn(model, {key: value.flip(1) for key, value in encoded.items()})
    print("Native layer-input cache and padding checks passed!")


def test_qwen_direction_hooks(fn=reference.qwen_direction_hooks):
    model, batch = tiny_qwen(), tiny_batch()
    vector = t.arange(1, 13, dtype=t.float64); vector /= vector.norm()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with t.inference_mode():
        baseline = model(**batch).logits
        for mode in ("add", "ablate"):
            with fn(model, vector, mode=mode, strength=0., layer=1):
                t.testing.assert_close(model(**batch).logits[batch["attention_mask"].bool()], baseline[batch["attention_mask"].bool()])
    unrelated_calls = []
    external = model.model.layers[0].register_forward_pre_hook(lambda *_: unrelated_calls.append(True))
    with pytest.raises(RuntimeError, match="intentional"):
        with fn(model, vector, mode="add", strength=1., layer=1):
            raise RuntimeError("intentional")
    assert len(model.model.layers[0]._forward_pre_hooks) == 1, 'After an intervention error, the unrelated hook must remain and all intervention hooks must be removed.'
    with t.inference_mode():
        t.testing.assert_close(model(**batch).logits[batch["attention_mask"].bool()], baseline[batch["attention_mask"].bool()])
    assert unrelated_calls, 'The unrelated hook must still execute on the next model call.'
    external.remove()
    for name, value in model.state_dict().items():
        assert t.equal(value, before[name]), 'Activation intervention hooks must not modify model parameters.'
    print("Zero-strength identity, unrelated hooks, cleanup and parameter preservation passed!")


def test_full_ablation_matches_weight_oracle(fn=reference.qwen_direction_hooks):
    model, batch = tiny_qwen(), tiny_batch()
    oracle = copy.deepcopy(model)
    vector = t.arange(1, 13, dtype=t.float64); vector /= vector.norm()
    projection = t.eye(12, dtype=t.float64) - vector[:, None] * vector[None, :]
    # Independent linear-algebra oracle on a disposable untied tiny model.
    with t.no_grad():
        oracle.model.embed_tokens.weight.copy_(oracle.model.embed_tokens.weight @ projection)
        for block in oracle.model.layers:
            for module in (block.self_attn.o_proj, block.mlp.down_proj):
                module.weight.copy_(projection @ module.weight)
                if module.bias is not None:
                    module.bias.copy_(projection @ module.bias)
        expected = oracle(**batch).logits
        with fn(model, vector, mode="ablate", strength=1.):
            actual = model(**batch).logits
    valid = batch["attention_mask"].bool()
    assert bool(t.isfinite(actual[valid]).all() and t.isfinite(expected[valid]).all()), 'Compare finite logits only at valid tokens; NaNs cannot establish projection-oracle parity.'
    t.testing.assert_close(actual[valid], expected[valid], atol=2e-7, rtol=2e-5)
    print("Every-write ablation matches an independent weight-projection oracle!")


def test_greedy_direction_generation(fn=reference.greedy_direction_generation):
    model, batch = tiny_qwen(), tiny_batch()
    # Avoid early EOS so all decoding calls and cache positions are exercised.
    with reference.qwen_direction_hooks(model, t.arange(1, 13).double(), mode="add", strength=.03, layer=1) as visits:
        cached = fn(model, batch, max_new_tokens=5, eos_token_id=100, pad_token_id=0, use_cache=True)
    assert len(visits) == 5 and [row["sequence_length"] for row in visits] == [5, 1, 1, 1, 1], 'Visit the hook at prefill and every generated token: cached sequence lengths should be 5,1,1,1,1.'
    for mode in ("add", "ablate"):
        with reference.qwen_direction_hooks(model, t.arange(1, 13).double(), mode=mode, strength=.03, layer=1):
            cached = fn(model, batch, max_new_tokens=5, eos_token_id=100, pad_token_id=0, use_cache=True)
        with reference.qwen_direction_hooks(model, t.arange(1, 13).double(), mode=mode, strength=.03, layer=1):
            uncached = fn(model, batch, max_new_tokens=5, eos_token_id=100, pad_token_id=0, use_cache=False)
        assert t.equal(cached.tokens, uncached.tokens), 'Persistent cached and uncached intervention generation must produce identical token IDs.'
    assert cached.lengths.tolist() == [5, 5] and not bool(cached.finished.any()), 'An unreachable EOS must produce five tokens per example with no finished rows.'
    print("Persistent cached/uncached generation and token positions agree!")


def test_generation_individual_eos(fn=reference.greedy_direction_generation):
    class Scripted(t.nn.Module):
        device = t.device("cpu")
        def __init__(self):
            super().__init__(); self.calls = []
        def forward(self, input_ids, attention_mask, **kwargs):
            self.calls.append(attention_mask.clone())
            last = input_ids[:, -1]
            next_token = t.where(last == 8, 5, t.where(last == 5, 6, 2))
            logits = t.zeros(len(last), 1, 10)
            logits[t.arange(len(last)), 0, next_token] = 1
            return SimpleNamespace(logits=logits, past_key_values=None)
    model = Scripted().train()
    output = fn(model, {"input_ids": t.tensor([[7], [8]]), "attention_mask": t.ones(2, 1, dtype=t.long)},
                max_new_tokens=5, eos_token_id=2, pad_token_id=0)
    assert output.tokens.tolist() == [[2, 0, 0, 0, 0], [5, 6, 2, 0, 0]], 'Each row must emit its own EOS, then PAD; the second row continues until its third token.'
    assert output.lengths.tolist() == [1, 3] and bool(output.finished.all()), 'Record per-example generated lengths including EOS: one and three tokens here.'
    assert model.training and model.calls[-1][0].tolist() == [1, 1, 0], 'Restore the model training state and mark already-finished row padding invalid in later attention masks.'
    print("Individual EOS lengths and completed-row padding passed!")


def test_refusal_marker_flags(fn=reference.refusal_marker_flags):
    samples = ["I cannot help with that.", "I’m sorry, I cannot provide this.", "The sum is 5.",
               "I cannot wait to celebrate your birthday!", "That request is outside my remit."]
    # The final two expose a false positive and a false negative of this rubric.
    assert fn(samples).tolist() == [True, True, False, True, False], 'The fixed English marker rubric intentionally has both a false positive and a false negative; preserve them.'
    print("Marker matching and its explicit counterexamples passed!")
