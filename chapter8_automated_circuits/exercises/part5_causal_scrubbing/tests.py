from typing import Callable

import torch as t
from transformer_lens import HookedTransformer

import part5_causal_scrubbing.utils as utils


def test_resample_matching(resample_matching: Callable):
    g = t.Generator().manual_seed(0)
    pool = t.randint(0, 10, (5000, 3), generator=g)
    x = t.randint(0, 10, (300, 3), generator=g)
    parity = lambda rows: (rows[:, 0] + rows[:, 1]) % 2
    out = resample_matching(x, parity, pool, g)
    assert out.shape == x.shape, f"Expected shape {tuple(x.shape)}, got {tuple(out.shape)}"
    assert (parity(out) == parity(x)).all(), "Every resampled row must have the same feature value as the row it replaces"
    pool_rows = {tuple(r) for r in pool.tolist()}
    assert all(tuple(r) in pool_rows for r in out.tolist()), "Resampled rows must come from the pool"
    assert (out != x).any(dim=1).float().mean() > 0.8, "Rows should be resampled, not copied from x"
    # with a feature that's constant, every pool row is a candidate: the first column should look uniform
    out = resample_matching(x.repeat(20, 1), lambda rows: t.zeros(len(rows), dtype=t.long), pool, g)
    counts = t.bincount(out[:, 0], minlength=10).float()
    assert counts.min() > 0.6 * counts.mean(), "With a constant feature, resampled rows should be uniform over the pool"
    print("All tests in `test_resample_matching` passed!")


def test_loss_recovered(loss_recovered: Callable):
    assert abs(loss_recovered(scrubbed=1.0, original=1.0, random=5.0) - 1.0) < 1e-9
    assert abs(loss_recovered(scrubbed=5.0, original=1.0, random=5.0)) < 1e-9
    assert abs(loss_recovered(scrubbed=2.0, original=1.0, random=5.0) - 0.75) < 1e-9
    assert loss_recovered(scrubbed=0.5, original=1.0, random=5.0) > 1, "A scrubbed loss below the original gives more than 1"
    print("All tests in `test_loss_recovered` passed!")


@t.no_grad()
def test_layer0_outputs(layer0_outputs: Callable, model: HookedTransformer, tokens: t.Tensor):
    out = layer0_outputs(model, tokens)
    assert set(out) == {"embed"} | {f"0.{h}" for h in range(model.cfg.n_heads)}, f"Unexpected keys {sorted(out)}"
    _, cache = model.run_with_cache(tokens, names_filter="blocks.1.hook_resid_pre")
    total = sum(out.values()) + model.b_O[0]
    t.testing.assert_close(total, cache["blocks.1.hook_resid_pre"], atol=1e-3, rtol=1e-4,
                           msg="The embedding plus every layer-0 head's output plus b_O[0] should equal the input to layer 1")
    stacked = layer0_outputs(model, t.stack([tokens[:2], tokens[2:4]]))
    assert stacked["0.7"].shape == (2, 2, tokens.shape[1], model.cfg.d_model), "Extra leading batch dimensions should be preserved"
    t.testing.assert_close(stacked["0.7"][1], out["0.7"][2:4], atol=1e-4, rtol=1e-4)
    print("All tests in `test_layer0_outputs` passed!")


@t.no_grad()
def test_head1_forward(head1_forward: Callable, model: HookedTransformer, tokens: t.Tensor):
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in ("blocks.1.hook_resid_pre", "blocks.1.attn.hook_z"))
    resid = cache["blocks.1.hook_resid_pre"]
    for head in [4, 10, 2]:
        ours = head1_forward(model, head, resid, resid, resid)
        expected = cache["blocks.1.attn.hook_z"][:, :, head] @ model.W_O[1, head]
        t.testing.assert_close(ours, expected, atol=1e-3, rtol=1e-4,
                               msg=f"With the true residual stream as all three inputs, head 1.{head}'s output should match the model's")
    # different inputs for q, k and v must be used for the right things
    shuffled = resid[t.randperm(len(resid))]
    a = head1_forward(model, 4, resid, resid, shuffled)
    b = head1_forward(model, 4, resid, resid, resid)
    assert not t.allclose(a, b), "Changing the value input should change the output"
    print("All tests in `test_head1_forward` passed!")


@t.no_grad()
def test_sender_restricted_input(sender_restricted_input: Callable, layer0_outputs: Callable, model: HookedTransformer,
                                 tokens: t.Tensor, half_len: int):
    g = t.Generator().manual_seed(0)
    _, cache = model.run_with_cache(tokens, names_filter="blocks.1.hook_resid_pre")
    full = sender_restricted_input(model, tokens, None, g)
    t.testing.assert_close(full, cache["blocks.1.hook_resid_pre"], atol=1e-3, rtol=1e-4,
                           msg="With senders=None (all senders read the real input), you should get the real input to layer 1")
    g1, g2 = t.Generator().manual_seed(1), t.Generator().manual_seed(1)
    ours = sender_restricted_input(model, tokens, ["embed", "0.7"], g1)
    resample = utils.repeated_random_tokens(model.tokenizer.bos_token_id, tokens.shape[:-1], half_len, g2, tokens.device)
    real, other = layer0_outputs(model, tokens), layer0_outputs(model, resample)
    expected = model.b_O[0] + real["embed"] + real["0.7"] + sum(other[f"0.{h}"] for h in range(model.cfg.n_heads) if h != 7)
    t.testing.assert_close(ours, expected, atol=1e-3, rtol=1e-4,
                           msg="Senders not in the list should read one resample drawn with utils.repeated_random_tokens(...)")
    print("All tests in `test_sender_restricted_input` passed!")


def test_per_position_resample(per_position_resample: Callable, tokens: t.Tensor, bos: int, half_len: int):
    g = t.Generator().manual_seed(0)
    offsets = [0, 2]
    xs = per_position_resample(tokens, offsets, bos, half_len, g)
    seq = tokens.shape[1]
    assert xs.shape == (seq, *tokens.shape), f"Expected shape (seq, batch, seq) = {(seq, *tokens.shape)}, got {tuple(xs.shape)}"
    for j in range(seq):
        for d in offsets:
            if j - d >= 0:
                assert t.equal(xs[j, :, j - d], tokens[:, j - d]), f"Copy {j} should agree with x at position {j - d}"
    assert (xs[:, :, 0] == bos).all(), "Position 0 is always BOS"
    keep = t.zeros(seq, seq, dtype=t.bool)
    for j in range(seq):
        for d in offsets:
            if j - d >= 0:
                keep[j, j - d] = True
    fresh = ~keep
    fresh[:, 0] = False  # BOS
    agree = (xs == tokens[None]).float().mean(1)  # (copy, position): how often the copy agrees with x
    assert agree[fresh].mean() < 0.05, "Positions that aren't copied from x should come from fresh samples"
    print("All tests in `test_per_position_resample` passed!")


@t.no_grad()
def test_scrubbed_loss(scrubbed_loss: Callable, model: HookedTransformer, tokens: t.Tensor, original_loss: float):
    g = t.Generator().manual_seed(0)
    everything = {"logits": ["embed"] + [f"{layer}.{h}" for layer in range(2) for h in range(model.cfg.n_heads)]}
    kept = scrubbed_loss(model, tokens, everything, g)
    assert abs(kept - original_loss) < 1e-3, f"A hypothesis that keeps everything should reproduce the original loss {original_loss:.4f}, got {kept:.4f}"
    nothing = scrubbed_loss(model, tokens, {}, g)
    assert nothing > original_loss + 3, f"Resampling everything should make the loss much worse (got {nothing:.3f})"
    print("All tests in `test_scrubbed_loss` passed!")
