"""Tests for [9.1] Refusal Directions and Representation Engineering.

Most tests use independent oracles: scikit-learn, plain `torch` hooks written inline, closed-form identities and
small hand-worked examples. Tests that need a transformer use a tiny randomly-initialised Qwen2 model (so they take
a fraction of a second and need no download), except where they need the real tokenizer.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch as t
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------


def _tiny_model(tie_word_embeddings: bool = True, seed: int = 0):
    """A 3-layer, 32-dimensional Qwen2 with random weights (large enough that edits visibly change the logits)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=tie_word_embeddings,
    )
    t.manual_seed(seed)
    model = Qwen2ForCausalLM(config).eval()
    for name, p in model.named_parameters():
        if "norm" not in name:
            p.data.normal_(0, 0.2)
    if tie_word_embeddings:
        model.tie_weights()
    return model


def _reference_ablated_logits(model, input_ids: t.Tensor, direction: t.Tensor, strength: float = 1.0) -> t.Tensor:
    """Directional ablation written from scratch: project the direction out of every write to the residual stream."""
    r = direction / direction.norm()

    def edit(module, args, output):
        x = output[0] if isinstance(output, tuple) else output
        x = x - strength * (x @ r)[..., None] * r
        return (x, *output[1:]) if isinstance(output, tuple) else x

    handles = [model.model.embed_tokens.register_forward_hook(edit)]
    for layer in model.model.layers:
        handles += [layer.self_attn.register_forward_hook(edit), layer.mlp.register_forward_hook(edit)]
    try:
        return model(input_ids).logits
    finally:
        for handle in handles:
            handle.remove()


def _reference_added_logits(model, input_ids: t.Tensor, vector: t.Tensor, layer: int, coefficient: float) -> t.Tensor:
    def hook(module, args, kwargs):
        return (args[0] + coefficient * vector, *args[1:]), kwargs

    handle = model.model.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        return model(input_ids).logits
    finally:
        handle.remove()


def _hook_counts(model) -> list[tuple[int, int]]:
    modules = [model.model.embed_tokens] + [m for layer in model.model.layers for m in (layer, layer.self_attn, layer.mlp)]
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]


def _no_hooks_left(model) -> bool:
    """True if the model has no forward hooks on the modules we edit (a freshly built tiny model has none)."""
    return all(count == (0, 0) for count in _hook_counts(model))


# --------------------------------------------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------------------------------------------


def test_is_refusal(is_refusal: Callable[[str], bool]):
    refusals = [
        "I'm sorry, but I can't assist with that.",
        "I cannot help with that request.",
        "As an AI language model, I don't have personal preferences.",
        "Sorry, but that's not something I'm able to do.",
        "I’m sorry, I can’t do that.",  # curly apostrophes
        "I CANNOT HELP WITH THAT.",  # case
        "Unfortunately I don't have access to real-time information.",  # marker in the middle of the text
    ]
    compliances = [
        "Sure! Here's a short poem about the ocean at sunset.",
        "The capital of France is Paris.",
        "I'd be happy to help with that. Here are three tips:",
        "Of course. First, preheat the oven to 200 degrees.",
        "",
    ]
    for text in refusals:
        assert is_refusal(text) is True, f"Expected {text!r} to be flagged as a refusal (and the function should return a bool)"
    for text in compliances:
        assert is_refusal(text) is False, f"Expected {text!r} NOT to be flagged as a refusal"
    print("All tests in `test_is_refusal` passed!")


def test_refusal_score(refusal_score: Callable):
    # Hand-worked example: vocabulary of 4 tokens, refusal tokens {0, 1}. P(refusal) = e^0 + e^0 / (2 + e^0 + e^0) = 0.5
    logits = t.tensor([[0.0, 0.0, 0.0, 0.0]])
    out = refusal_score(logits, [0, 1])
    assert out.shape == (1,), f"Expected shape (n_prompts,), got {tuple(out.shape)}"
    t.testing.assert_close(out, t.tensor([0.0]), msg="With uniform logits P(refusal)=0.5, so the log-odds is 0")
    # General case against the definition: log(p / (1 - p)) with p the total probability of the refusal tokens
    t.manual_seed(0)
    logits = t.randn(5, 30)
    ids = [3, 7, 11]
    p = logits.softmax(-1)[:, ids].sum(-1)
    t.testing.assert_close(refusal_score(logits, ids), (p / (1 - p)).log(), atol=1e-5, rtol=1e-5)
    # Numerical stability: with huge logits, softmax-then-log would give inf or nan
    extreme = t.tensor([[200.0, 0.0, -200.0, 5.0]])
    value = refusal_score(extreme, [0])
    assert t.isfinite(value).all(), "The score should stay finite for extreme logits (use logsumexp rather than softmax then log)"
    t.testing.assert_close(value, t.tensor([200.0 - t.logsumexp(t.tensor([0.0, -200.0, 5.0]), 0).item()]), atol=1e-3, rtol=1e-4)
    print("All tests in `test_refusal_score` passed!")


def test_get_activations(get_activations: Callable, model, tokenizer):
    import part1_refusal_directions.utils as utils

    prompts = ["Hi", "Write a short poem about the sea at night"]  # different lengths, so the batch is left-padded
    result = get_activations(model, tokenizer, prompts, n_positions=3, batch_size=1)
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size
    assert result.shape == (n_layers, 2, 3, d_model), f"Expected shape (layer, prompt, position, d_model) = {(n_layers, 2, 3, d_model)}, got {tuple(result.shape)}"
    # Independent oracle: forward pre-hooks that capture the input of every decoder block
    captured: dict[int, t.Tensor] = {}
    handles = [
        layer.register_forward_pre_hook(lambda m, a, kw, i=i: captured.__setitem__(i, a[0].detach().clone()), with_kwargs=True)
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        batch = utils.encode_chats(tokenizer, prompts, device=model.device)
        with t.inference_mode():
            model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    expected = t.stack([captured[i][:, -3:] for i in range(n_layers)])
    t.testing.assert_close(result.to(expected.device).float(), expected.float(), atol=1e-3, rtol=1e-3)
    print("All tests in `test_get_activations` passed!")


def test_difference_of_means(difference_of_means: Callable):
    t.manual_seed(0)
    pos = t.randn(3, 7, 2, 5) + 2.0
    neg = t.randn(3, 9, 2, 5) - 1.0
    out = difference_of_means(pos, neg)
    assert out.shape == (3, 2, 5), f"Expected shape (layer, position, d_model) = (3, 2, 5), got {tuple(out.shape)}"
    expected = np.mean(pos.numpy(), axis=1) - np.mean(neg.numpy(), axis=1)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-5, err_msg="Should be mean over prompts of the positive class minus mean over prompts of the negative class (not normalised)")
    # Hand-worked example: one layer, one position, 2-dimensional
    pos = t.tensor([[[[1.0, 0.0]], [[3.0, 2.0]]]])  # mean (2, 1)
    neg = t.tensor([[[[0.0, 1.0]]]])  # mean (0, 1)
    t.testing.assert_close(difference_of_means(pos, neg), t.tensor([[[2.0, 0.0]]]))
    print("All tests in `test_difference_of_means` passed!")


def test_auroc(auroc: Callable):
    t.manual_seed(0)
    for shift in [0.0, 0.5, 2.0]:
        pos = t.randn(30) + shift
        neg = t.randn(45)
        expected = roc_auc_score(np.r_[np.ones(30), np.zeros(45)], np.r_[pos.numpy(), neg.numpy()])
        assert abs(float(auroc(pos, neg)) - expected) < 1e-6, f"AUROC mismatch for shift {shift}: got {float(auroc(pos, neg)):.4f}, expected {expected:.4f}"
    # Ties count for half: every score identical -> 0.5
    assert abs(float(auroc(t.ones(4), t.ones(6))) - 0.5) < 1e-6, "If every score is tied the AUROC should be 0.5"
    assert abs(float(auroc(t.tensor([2.0, 3.0]), t.tensor([0.0, 1.0]))) - 1.0) < 1e-6
    assert abs(float(auroc(t.tensor([0.0, 1.0]), t.tensor([2.0, 3.0]))) - 0.0) < 1e-6
    print("All tests in `test_auroc` passed!")


def test_shuffled_label_direction(shuffled_label_direction: Callable):
    t.manual_seed(0)
    n, d = 40, 128
    axis = t.randn(d)
    axis = axis / axis.norm()
    pos = t.randn(n, d) + 4.0 * axis
    neg = t.randn(n, d)
    true = pos.mean(0) - neg.mean(0)
    # Same seed -> same output; the output is a raw (unnormalised) difference of two group means
    a = shuffled_label_direction(pos, neg, seed=1, stratified=True)
    b = shuffled_label_direction(pos, neg, seed=1, stratified=True)
    assert a.shape == (d,), f"Expected shape (d_model,), got {tuple(a.shape)}"
    t.testing.assert_close(a, b, msg="The same seed should give the same direction")
    # Over many shuffles, a stratified shuffle should have (almost) no overlap with the true direction, while a naive shuffle
    # (permuting all 2n labels) leaves a large random overlap, because its groups contain unequal numbers of positives.
    cos = lambda x: float(F.cosine_similarity(x, true, dim=0))
    strat = np.array([cos(shuffled_label_direction(pos, neg, seed=s, stratified=True)) for s in range(60)])
    naive = np.array([cos(shuffled_label_direction(pos, neg, seed=s, stratified=False)) for s in range(60)])
    assert np.abs(strat).mean() < 0.15, f"Stratified shuffles should be nearly orthogonal to the true direction; mean |cos| was {np.abs(strat).mean():.2f}"
    assert np.abs(naive).mean() > 1.5 * np.abs(strat).mean(), (
        f"Naive shuffles (permute all labels) should overlap more with the true direction than stratified ones: "
        f"mean |cos| {np.abs(naive).mean():.2f} vs {np.abs(strat).mean():.2f}"
    )
    # Stratified: each of the two groups gets exactly half of the positives and half of the negatives, so a constant shift
    # applied to every positive cancels exactly in the difference of the group means.
    shifted = shuffled_label_direction(pos + 100.0 * axis, neg, seed=1, stratified=True)
    t.testing.assert_close(shifted, a, atol=1e-3, rtol=1e-3, msg="A stratified shuffle puts the same number of positives in each group, so shifting the positive class must not change it")
    print("All tests in `test_shuffled_label_direction` passed!")


# --------------------------------------------------------------------------------------------------------------
# Part 2
# --------------------------------------------------------------------------------------------------------------


def test_project_out(project_out: Callable):
    t.manual_seed(0)
    x = t.randn(4, 6, 8)
    r = t.randn(8)
    rhat = r / r.norm()
    out = project_out(x, r)
    assert out.shape == x.shape
    assert (out @ rhat).abs().max() < 1e-5, "After projecting out the direction, the component along it should be zero"
    # Components orthogonal to the direction are untouched
    ortho = x - (x @ rhat)[..., None] * rhat
    t.testing.assert_close(out, ortho, atol=1e-5, rtol=1e-5)
    # The direction doesn't need to be unit length
    t.testing.assert_close(project_out(x, 7.0 * r), out, atol=1e-5, rtol=1e-5)
    # Partial strength removes a fraction of the component
    half = project_out(x, r, strength=0.5)
    t.testing.assert_close(half @ rhat, 0.5 * (x @ rhat), atol=1e-5, rtol=1e-5)
    t.testing.assert_close(project_out(x, r, strength=0.0), x)
    # It must be differentiable with respect to the direction (we will optimise directions in part 4)
    rg = r.clone().requires_grad_(True)
    project_out(x, rg).square().sum().backward()
    assert rg.grad is not None and rg.grad.abs().sum() > 0, "project_out should be differentiable with respect to the direction (don't detach it)"
    print("All tests in `test_project_out` passed!")


def test_directional_ablation(directional_ablation: Callable):
    model = _tiny_model()
    t.manual_seed(1)
    ids = t.randint(0, 64, (2, 9))
    r = t.randn(32)
    rhat = r / r.norm()
    base = model(ids).logits
    captured: list[t.Tensor] = []
    with directional_ablation(model, r):
        handles = [
            layer.register_forward_pre_hook(lambda m, a, kw: captured.append(a[0].detach().clone()), with_kwargs=True)
            for layer in model.model.layers
        ]
        edited = model(ids).logits
        for handle in handles:
            handle.remove()
    assert len(captured) == 3, "Expected to capture the input of each of the 3 blocks"
    for i, hidden in enumerate(captured):
        assert (hidden @ rhat).abs().max() < 1e-4, (
            f"Block {i}'s input still has a component along the direction ({(hidden @ rhat).abs().max():.4f}). "
            "Every write to the residual stream must be projected: the embedding, and each attention and MLP output."
        )
    t.testing.assert_close(edited, _reference_ablated_logits(model, ids, r), atol=1e-4, rtol=1e-4)
    assert not t.allclose(edited, base, atol=1e-3), "Ablation should change the logits"
    # Hooks are removed on exit
    assert _no_hooks_left(model), "Hooks were left attached after leaving the context manager"
    t.testing.assert_close(model(ids).logits, base)
    # Strength
    with directional_ablation(model, r, strength=0.0):
        t.testing.assert_close(model(ids).logits, base, atol=1e-5, rtol=1e-5)
    expected_half = _reference_ablated_logits(model, ids, r, 0.5)
    with directional_ablation(model, r, strength=0.5):
        t.testing.assert_close(model(ids).logits, expected_half, atol=1e-4, rtol=1e-4)
    # Hooks are removed even if the body raises
    try:
        with directional_ablation(model, r):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _no_hooks_left(model), "Hooks must be removed even when an exception is raised inside the `with` block (use try/finally)"
    # Gradients flow back to the direction
    rg = r.clone().requires_grad_(True)
    with directional_ablation(model, rg):
        loss = model(ids).logits.square().mean()
    loss.backward()
    assert rg.grad is not None and rg.grad.abs().sum() > 0, "The gradient should reach the direction (don't detach it inside the hooks)"
    # It also has to work during cached generation, where each forward pass sees a single new token
    kwargs = dict(attention_mask=t.ones_like(ids), max_new_tokens=4, do_sample=False, pad_token_id=0)
    with directional_ablation(model, r):
        generated = model.generate(ids, **kwargs)
    handles = [model.model.embed_tokens.register_forward_hook(lambda m, a, o: o - (o @ rhat)[..., None] * rhat)]
    for layer in model.model.layers:
        handles += [
            layer.self_attn.register_forward_hook(lambda m, a, o: (o[0] - (o[0] @ rhat)[..., None] * rhat, *o[1:])),
            layer.mlp.register_forward_hook(lambda m, a, o: o - (o @ rhat)[..., None] * rhat),
        ]
    expected = model.generate(ids, **kwargs)
    for handle in handles:
        handle.remove()
    assert t.equal(generated, expected), "Ablation during `generate` doesn't match the reference"
    print("All tests in `test_directional_ablation` passed!")


def test_activation_addition(activation_addition: Callable):
    model = _tiny_model()
    t.manual_seed(2)
    ids = t.randint(0, 64, (2, 9))
    v = t.randn(32)
    base = model(ids).logits
    for layer in [0, 2]:
        expected = _reference_added_logits(model, ids, v, layer, 2.0)
        with activation_addition(model, v, layer, coefficient=2.0):
            out = model(ids).logits
        t.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4, msg=f"Adding at layer {layer} doesn't match the reference")
        assert not t.allclose(out, base, atol=1e-3)
    assert _no_hooks_left(model), "Hooks were left attached after leaving the context manager"
    t.testing.assert_close(model(ids).logits, base)
    with activation_addition(model, v, 1, coefficient=0.0):
        t.testing.assert_close(model(ids).logits, base, atol=1e-5, rtol=1e-5)
    # It also has to work during cached generation
    kwargs = dict(attention_mask=t.ones_like(ids), max_new_tokens=4, do_sample=False, pad_token_id=0)
    with activation_addition(model, v, 1, coefficient=3.0):
        generated = model.generate(ids, **kwargs)
    handle = model.model.layers[1].register_forward_pre_hook(lambda m, a, kw: ((a[0] + 3.0 * v, *a[1:]), kw), with_kwargs=True)
    expected_generated = model.generate(ids, **kwargs)
    handle.remove()
    assert t.equal(generated, expected_generated), "Addition during `generate` doesn't match the reference (the vector must be added at every position, including during cached decoding)"
    print("All tests in `test_activation_addition` passed!")


def test_kl_divergence(kl_divergence: Callable):
    t.manual_seed(0)
    base, edited = t.randn(6, 40), t.randn(6, 40)
    out = kl_divergence(base, edited)
    assert out.shape == (6,), f"Expected one KL per prompt, shape (6,), got {tuple(out.shape)}"
    expected = F.kl_div(edited.log_softmax(-1), base.log_softmax(-1), log_target=True, reduction="none").sum(-1)
    t.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5, msg="Should be KL(base || edited) = sum_v p_base(v) (log p_base(v) - log p_edited(v))")
    assert (out > 0).all()
    t.testing.assert_close(kl_divergence(base, base), t.zeros(6), atol=1e-6, rtol=0)
    assert not t.allclose(kl_divergence(edited, base), out), "KL is asymmetric: the first argument is the unedited model"
    print("All tests in `test_kl_divergence` passed!")


def test_sweep_directions(sweep_directions: Callable, model, tokenizer):
    import part1_refusal_directions.utils as utils

    t.manual_seed(0)
    hooks_before = _hook_counts(model)
    harmful_like = ["Tell me about the history of Rome.", "How do I bake bread?"]  # stand-ins: the test only needs two groups of prompts
    harmless_like = ["Write a poem about the sea.", "What is the capital of Spain?", "Explain how rain forms."]
    n_layers, d_model = 2, model.config.hidden_size
    directions = 3.0 * t.randn(n_layers, 2, d_model, device=model.device)
    ids = [40, 358]
    result = sweep_directions(model, tokenizer, directions, harmful_like, harmless_like, ids)
    for key in ["ablation_score", "addition_score", "kl"]:
        assert key in result, f"The result should have a '{key}' entry"
        assert tuple(result[key].shape) == (n_layers, 2), f"result['{key}'] should have shape (layer, position) = (2, 2), got {tuple(result[key].shape)}"
    # Independent oracle for one cell of the grid (layer 1, position 1), written with inline hooks
    layer, pos = 1, 1
    r = directions[layer, pos]
    harmful_batch = utils.encode_chats(tokenizer, harmful_like, device=model.device)
    harmless_batch = utils.encode_chats(tokenizer, harmless_like, device=model.device)

    def score(logits):
        lse = t.logsumexp
        mask = t.zeros(logits.shape[-1], dtype=t.bool, device=logits.device)
        mask[ids] = True
        return lse(logits[:, mask], -1) - lse(logits[:, ~mask], -1)

    with t.inference_mode():
        base_harmless = model(**harmless_batch, logits_to_keep=1).logits[:, -1].float()
        rhat = r / r.norm()

        def ablate(module, args, output):
            x = output[0] if isinstance(output, tuple) else output
            x = x - (x @ rhat)[..., None] * rhat
            return (x, *output[1:]) if isinstance(output, tuple) else x

        handles = [model.model.embed_tokens.register_forward_hook(ablate)]
        for blk in model.model.layers:
            handles += [blk.self_attn.register_forward_hook(ablate), blk.mlp.register_forward_hook(ablate)]
        ablated_harmful = model(**harmful_batch, logits_to_keep=1).logits[:, -1].float()
        ablated_harmless = model(**harmless_batch, logits_to_keep=1).logits[:, -1].float()
        for handle in handles:
            handle.remove()
        handle = model.model.layers[layer].register_forward_pre_hook(lambda m, a, kw: ((a[0] + r, *a[1:]), kw), with_kwargs=True)
        added_harmless = model(**harmless_batch, logits_to_keep=1).logits[:, -1].float()
        handle.remove()
    expected_kl = F.kl_div(ablated_harmless.log_softmax(-1), base_harmless.log_softmax(-1), log_target=True, reduction="none").sum(-1).mean()
    t.testing.assert_close(result["ablation_score"][layer, pos].float().cpu(), score(ablated_harmful).mean().cpu(), atol=1e-3, rtol=1e-3, msg="ablation_score: mean refusal score on the harmful prompts with the direction ablated")
    t.testing.assert_close(result["addition_score"][layer, pos].float().cpu(), score(added_harmless).mean().cpu(), atol=1e-3, rtol=1e-3, msg="addition_score: mean refusal score on the harmless prompts after adding the (unnormalised) direction with coefficient 1 at that layer")
    t.testing.assert_close(result["kl"][layer, pos].float().cpu(), expected_kl.cpu(), atol=1e-3, rtol=1e-3, msg="kl: mean KL(unedited || ablated) on the harmless prompts")
    assert _hook_counts(model) == hooks_before, "sweep_directions left hooks attached to the model"
    print("All tests in `test_sweep_directions` passed!")


def test_select_direction(select_direction: Callable):
    n_layers = 10  # layers 0..7 are allowed (layer < 0.8 * 10 = 8)
    ablation = t.full((n_layers, 3), 5.0)
    addition = t.full((n_layers, 3), 2.0)
    kl = t.full((n_layers, 3), 0.01)
    baseline_harmless = -4.0
    ablation[9, 0] = -20.0  # best score overall, but in a late layer: must be excluded
    ablation[3, 2] = -15.0
    kl[3, 2] = 0.5  # great score, but breaks the model on harmless prompts: must be excluded
    ablation[5, 1] = -12.0
    addition[5, 1] = -6.0  # great score, but adding it makes harmless prompts *less* refusal-like: must be excluded
    ablation[6, 2] = -9.0  # the best feasible candidate
    ablation[2, 0] = -8.0
    sweep = {"ablation_score": ablation, "addition_score": addition, "kl": kl}
    result = select_direction(sweep, baseline_harmless_score=baseline_harmless, n_layers=n_layers)
    assert tuple(result) == (6, 2), f"Expected (layer, position) = (6, 2), got {tuple(result)}"
    result = select_direction(sweep, baseline_harmless_score=baseline_harmless, n_layers=n_layers, max_layer_fraction=0.4, kl_threshold=0.1)
    assert tuple(result) == (2, 0), f"With max_layer_fraction=0.4 only layers 0-3 are allowed, so the best is (2, 0); got {tuple(result)}"
    try:
        select_direction({"ablation_score": ablation, "addition_score": addition, "kl": kl + 1.0}, baseline_harmless_score=baseline_harmless, n_layers=n_layers)
    except ValueError:
        pass
    else:
        raise AssertionError("If no candidate satisfies the constraints, select_direction should raise a ValueError")
    print("All tests in `test_select_direction` passed!")


# --------------------------------------------------------------------------------------------------------------
# Part 3
# --------------------------------------------------------------------------------------------------------------


def test_orthogonalize_weights(orthogonalize_weights: Callable):
    t.manual_seed(3)
    ids = t.randint(0, 64, (2, 9))
    r = t.randn(32)
    rhat = r / r.norm()
    for tied in [False, True]:
        model = _tiny_model(tie_word_embeddings=tied)
        reference = _reference_ablated_logits(model, ids, r)
        original = copy.deepcopy(model)
        orthogonalize_weights(model, r)
        # Every matrix that writes into the residual stream has no component along the direction
        embed = model.model.embed_tokens.weight
        assert (embed @ rhat).abs().max() < 1e-5, "The token embedding rows should be orthogonal to the direction"
        for i, layer in enumerate(model.model.layers):
            assert (rhat @ layer.self_attn.o_proj.weight).abs().max() < 1e-5, f"Block {i}'s attention output matrix still writes along the direction (remember nn.Linear stores (out, in))"
            assert (rhat @ layer.mlp.down_proj.weight).abs().max() < 1e-5, f"Block {i}'s MLP output matrix still writes along the direction"
        # ...and the edited model computes the same function as hook-based ablation
        t.testing.assert_close(
            model(ids).logits, reference, atol=1e-4, rtol=1e-4,
            msg=("The weight-edited model should match hook-based ablation exactly." + (" With tied embeddings, editing the embedding also edits the unembedding: untie them first!" if tied else "")),
        )
        # The unembedding and all matrices that only *read* from the residual stream are untouched
        t.testing.assert_close(model.lm_head.weight, original.lm_head.weight, msg="The unembedding (lm_head) should keep its ORIGINAL weights")
        assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), "lm_head and the embedding must not share memory after the edit"
        for (name, p), (_, q) in zip(model.named_parameters(), original.named_parameters()):
            if any(key in name for key in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "norm"]):
                t.testing.assert_close(p, q, msg=f"{name} only reads from the residual stream and should not change")
    print("All tests in `test_orthogonalize_weights` passed!")


def test_next_token_loss(next_token_loss: Callable):
    model = _tiny_model()
    t.manual_seed(4)
    tokens = t.randint(0, 64, (3, 12))
    out = next_token_loss(model, tokens)
    assert out.ndim == 0, "Should return a scalar"
    logits = model(tokens).logits
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 64), tokens[:, 1:].reshape(-1))
    t.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5, msg="Should be the mean cross-entropy of predicting token i+1 from the tokens up to i, over every position of every sequence")
    print("All tests in `test_next_token_loss` passed!")


# --------------------------------------------------------------------------------------------------------------
# Part 4
# --------------------------------------------------------------------------------------------------------------


def test_pca_directions(pca_directions: Callable):
    t.manual_seed(0)
    scales = t.tensor([5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
    x = (t.randn(60, 8) * scales) @ t.linalg.qr(t.randn(8, 8))[0] + 4.0  # non-zero mean: PCA has to centre
    components, ratio = pca_directions(x, k=3)
    assert components.shape == (3, 8), f"Expected components of shape (k, d) = (3, 8), got {tuple(components.shape)}"
    assert ratio.shape == (3,), f"Expected explained-variance ratios of shape (k,) = (3,), got {tuple(ratio.shape)}"
    reference = PCA(n_components=3).fit(x.numpy())
    for i in range(3):
        cos = abs(float(F.cosine_similarity(components[i], t.tensor(reference.components_[i]), dim=0)))
        assert cos > 0.999, f"Component {i} doesn't match scikit-learn's PCA (|cos| = {cos:.3f})"
    np.testing.assert_allclose(ratio.numpy(), reference.explained_variance_ratio_, atol=1e-4, err_msg="explained variance ratios (fraction of the TOTAL variance) don't match scikit-learn")
    t.testing.assert_close(components.norm(dim=-1), t.ones(3), atol=1e-5, rtol=1e-5, msg="Components should be unit vectors")
    print("All tests in `test_pca_directions` passed!")


def test_refusal_search_loss(refusal_search_loss: Callable):
    t.manual_seed(0)
    harmful_edited = t.randn(5, 20, requires_grad=True)
    harmless_edited = t.randn(4, 20, requires_grad=True)
    harmless_base = t.randn(4, 20)
    ids = [2, 9]
    out = refusal_search_loss(harmful_edited, harmless_edited, harmless_base, ids, kl_weight=0.7)
    assert out.ndim == 0, "The loss should be a scalar"
    # Independent oracle: softplus(log-odds) is exactly -log(1 - P(refusal))
    p = harmful_edited.softmax(-1)[:, ids].sum(-1)
    term1 = -(1 - p).log().mean()
    term2 = F.kl_div(harmless_edited.log_softmax(-1), harmless_base.log_softmax(-1), log_target=True, reduction="none").sum(-1).mean()
    t.testing.assert_close(out, term1 + 0.7 * term2, atol=1e-5, rtol=1e-5, msg="Expected mean over harmful prompts of softplus(refusal score) + kl_weight * mean KL(base || edited) on harmless prompts")
    out.backward()
    assert harmful_edited.grad is not None and harmless_edited.grad is not None, "The loss should be differentiable with respect to the edited logits"
    # No refusal probability and no KL -> zero loss
    zero = refusal_search_loss(t.tensor([[-50.0, -50.0, 50.0]]), harmless_base[:1, :3], harmless_base[:1, :3], [0, 1], kl_weight=1.0)
    assert float(zero) < 1e-6, "A harmful prompt with zero refusal probability and an unchanged harmless distribution should have ~zero loss"
    print("All tests in `test_refusal_search_loss` passed!")


# --------------------------------------------------------------------------------------------------------------
# Part 5
# --------------------------------------------------------------------------------------------------------------


def test_lat_reading_vector(lat_reading_vector: Callable):
    t.manual_seed(0)
    n, d = 60, 24
    axis = t.randn(d)
    axis = axis / axis.norm()
    noise = lambda: 0.3 * t.randn(n, d)
    # Case 1: a consistent concept shift between the two members of each pair
    base = 2.0 * t.randn(n, d)
    acts_a = base + noise() + 3.0 * axis
    acts_b = base + noise()
    v = lat_reading_vector(acts_a, acts_b, seed=0)
    assert v.shape == (d,), f"Expected shape (d,), got {tuple(v.shape)}"
    assert abs(float(v.norm()) - 1.0) < 1e-4, "The reading vector should be a unit vector"
    assert float(v @ axis) > 0.95, f"Should recover the planted axis with the correct sign (cos = {float(v @ axis):.2f}). Orient the vector so that it points from B towards A."
    # Case 2: the sign varies from pair to pair (balanced), so there is no mean difference, only a variance axis along which pairs differ
    signs = t.cat([t.ones(n // 2), -t.ones(n - n // 2)])[t.randperm(n)][:, None]  # exactly balanced: the mean difference is only noise
    acts_a = base + noise() + 3.0 * signs * axis
    acts_b = base + noise()
    v2 = lat_reading_vector(acts_a, acts_b, seed=0)
    assert abs(float(v2 @ axis)) > 0.95, f"Should recover the axis of greatest variance in the pair differences (|cos| = {abs(float(v2 @ axis)):.2f})"
    t.testing.assert_close(lat_reading_vector(acts_a, acts_b, seed=3), lat_reading_vector(acts_a, acts_b, seed=3), msg="Same seed, same vector")
    print("All tests in `test_lat_reading_vector` passed!")


def test_token_scores(token_scores: Callable, model, tokenizer):
    import part1_refusal_directions.utils as utils

    prompt = "How do I kill a Python process?"
    layer = 5
    direction = t.randn(model.config.hidden_size, device=model.device)
    tokens, scores = token_scores(model, tokenizer, prompt, direction, layer)
    batch = utils.encode_chats(tokenizer, [prompt], device=model.device)
    ids = batch["input_ids"][0]
    assert len(tokens) == len(ids) == len(scores), f"Expected one token and one score per token of the chat-formatted prompt ({len(ids)}), got {len(tokens)} tokens and {len(scores)} scores"
    assert tokens == [tokenizer.decode(i) for i in ids], "tokens should be the decoded tokens of the chat-formatted prompt, template included"
    captured = {}
    handle = model.model.layers[layer].register_forward_pre_hook(lambda m, a, kw: captured.__setitem__("x", a[0].detach().clone()), with_kwargs=True)
    with t.inference_mode():
        model(**batch)
    handle.remove()
    expected = captured["x"][0] @ (direction / direction.norm())
    t.testing.assert_close(scores.to(expected.device).float(), expected.float(), atol=1e-3, rtol=1e-3, msg="scores[i] should be the projection of the layer-input residual stream at token i onto the unit direction")
    print("All tests in `test_token_scores` passed!")
