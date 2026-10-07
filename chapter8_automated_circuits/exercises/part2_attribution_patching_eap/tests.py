"""Tests for [8.2] Attribution Patching and EAP.

Wherever possible these use independent oracles (TransformerLens's own patching functions, autograd identities,
hand-worked examples) rather than comparing against the solutions.
"""

import contextlib
import io
from typing import Callable

import numpy as np
import torch as t
import transformer_lens.patching as tl_patching
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer


def _solutions():
    import part2_attribution_patching_eap.solutions as solutions

    return solutions


def _retain_grads(model: HookedTransformer, tokens: t.Tensor, metric_fn: Callable, names: list[str]):
    """Oracle forward/backward: retain_grad() on hooked activations (a different mechanism from backward hooks)."""
    stored = {}

    def make_differentiable(embed, hook):
        return embed.requires_grad_(True)

    def keep(act, hook):
        act.retain_grad()
        stored[hook.name] = act

    with model.hooks(fwd_hooks=[("hook_embed", make_differentiable), (lambda n: n in names, keep)]):
        value = metric_fn(model(tokens))
        value.backward()
    return value.item(), {n: a.detach() for n, a in stored.items()}, {n: a.grad.detach() for n, a in stored.items()}


def _ioi_metric(logits, answers):
    final = logits[:, -1]
    return final.gather(-1, answers[:, :1])[:, 0] - final.gather(-1, answers[:, 1:])[:, 0]


def _small_ioi(model: HookedTransformer):
    clean = model.to_tokens(["When Mary and John went to the store, John gave a drink to",
                             "When Tom and Anna went to the park, Anna gave a book to"])
    corrupt = model.to_tokens(["When Laura and Steve went to the store, Peter gave a drink to",
                               "When Kate and Paul went to the park, Henry gave a book to"])
    answers = model.to_tokens([" Mary", " John", " Tom", " Anna"], prepend_bos=False)[:, 0].reshape(2, 2)
    return clean, corrupt, answers


def test_ioi_metric(ioi_metric: Callable):
    logits = t.zeros(2, 3, 10)
    logits[0, -1, 4], logits[0, -1, 7] = 2.0, -1.0
    logits[1, -1, 1], logits[1, -1, 2] = 0.5, 1.5
    logits[0, 0, 4] = 100.0  # not the final position, should be ignored
    answers = t.tensor([[4, 7], [1, 2]])
    result = ioi_metric(logits, answers)
    assert result.shape == (2,), f"Expected one value per prompt (shape (2,)), got {tuple(result.shape)}"
    t.testing.assert_close(result, t.tensor([3.0, -1.0]), msg="Expected logit(IO) - logit(S) at the final position")
    print("All tests in `test_ioi_metric` passed!")


def test_get_cache_fwd_and_bwd(get_cache_fwd_and_bwd: Callable, model: HookedTransformer):
    clean, _, answers = _small_ioi(model)
    metric_fn = lambda logits: _ioi_metric(logits, answers).mean()
    final = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    names = ["blocks.3.attn.hook_z", final]
    value, acts, grads = get_cache_fwd_and_bwd(model, clean, metric_fn, names)
    assert isinstance(value, float), "The metric value should be returned as a float (use .item())"
    assert set(acts) == set(names) and set(grads) == set(names), f"Expected dicts with keys {names}"
    assert not acts[final].requires_grad and not grads[final].requires_grad, "Store detached tensors"
    # Oracle: differentiate ln_final + unembed applied to the cached final residual stream
    resid = acts[final].clone().requires_grad_(True)
    oracle_value = metric_fn(model.unembed(model.ln_final(resid)))
    (oracle_grad,) = t.autograd.grad(oracle_value, resid)
    assert abs(oracle_value.item() - value) < 1e-3, "The returned metric value doesn't match the clean run"
    t.testing.assert_close(grads[final], oracle_grad, atol=1e-5, rtol=1e-3)
    # And the z gradient should match an independent retain_grad() computation
    _, _, oracle_grads = _retain_grads(model, clean, metric_fn, ["blocks.3.attn.hook_z"])
    t.testing.assert_close(grads["blocks.3.attn.hook_z"], oracle_grads["blocks.3.attn.hook_z"], atol=1e-5, rtol=1e-3)
    assert all(p.grad is None for p in model.parameters()), "No parameter gradients should be accumulated"
    print("All tests in `test_get_cache_fwd_and_bwd` passed!")


def test_attribution_patch_heads(attribution_patch_heads: Callable, model: HookedTransformer):
    # Hand-made example: 1 layer, batch 1, pos 2, 2 heads, d_head 2
    name = "blocks.0.attn.hook_z"
    clean = {name: t.zeros(1, 2, 2, 2)}
    corrupt = {name: t.tensor([[[[1.0, 0.0], [0.0, 2.0]], [[1.0, 1.0], [0.0, 0.0]]]])}
    grads = {name: t.tensor([[[[3.0, 5.0], [7.0, 1.0]], [[1.0, -1.0], [4.0, 4.0]]]])}
    result = attribution_patch_heads(clean, grads, corrupt, 1)
    assert result.shape == (1, 2), f"Expected shape (n_layers, n_heads) = (1, 2), got {tuple(result.shape)}"
    t.testing.assert_close(result, t.tensor([[3.0 + 0.0, 2.0]]))
    # Identity: summed over heads, AP on hook_z equals AP on hook_attn_out (attn_out = sum_h z_h W_O^h + b_O)
    clean_tokens, corrupt_tokens, answers = _small_ioi(model)
    metric_fn = lambda logits: _ioi_metric(logits, answers).mean()
    names = [f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)] + [f"blocks.{l}.hook_attn_out" for l in range(model.cfg.n_layers)]
    _, clean_acts, clean_grads = _retain_grads(model, clean_tokens, metric_fn, names)
    _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=lambda n: n in names)
    result = attribution_patch_heads(clean_acts, clean_grads, {n: corrupt_cache[n] for n in names}, model.cfg.n_layers)
    assert result.shape == (model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        n = f"blocks.{layer}.hook_attn_out"
        layer_ap = ((corrupt_cache[n] - clean_acts[n]) * clean_grads[n]).sum()
        t.testing.assert_close(result[layer].sum(), layer_ap, atol=1e-4, rtol=1e-3,
                               msg=f"Head scores in layer {layer} should add up to the AP score of hook_attn_out")
    print("All tests in `test_attribution_patch_heads` passed!")


def test_exact_patch_heads(exact_patch_heads: Callable, model: HookedTransformer, task):
    names = [f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)]
    _, corrupt_cache = model.run_with_cache(task.corrupt_tokens)
    corrupt_acts = {n: corrupt_cache[n] for n in names}
    result = exact_patch_heads(model, task, corrupt_acts)
    assert result.shape == (model.cfg.n_layers, model.cfg.n_heads), f"Expected shape (layer, head), got {tuple(result.shape)}"
    with t.no_grad():
        clean_value = task.metric(model(task.clean_tokens), task.answers).mean()
    # Oracle: TransformerLens's own patching (it patches cache values into the run on the given tokens)
    with contextlib.redirect_stderr(io.StringIO()):
        expected = tl_patching.get_act_patch_attn_head_out_all_pos(
            model, task.clean_tokens, corrupt_cache, lambda logits: task.metric(logits, task.answers).mean() - clean_value
        )
    t.testing.assert_close(result.cpu().float(), expected.cpu().float(), atol=1e-3, rtol=1e-3)
    print("All tests in `test_exact_patch_heads` passed!")


def test_spearman_correlation(spearman_correlation: Callable):
    g = t.Generator().manual_seed(0)
    x, y = t.randn(50, generator=g), t.randn(50, generator=g)
    y = 0.6 * x + y
    assert abs(spearman_correlation(x, y) - spearmanr(x.numpy(), y.numpy())[0]) < 1e-5, "Doesn't match scipy's spearmanr"
    assert abs(spearman_correlation(x, x.exp()) - 1.0) < 1e-6, "A monotonic transformation should have rank correlation 1"
    assert abs(spearman_correlation(x, -x**3) + 1.0) < 1e-6, "A decreasing transformation should have rank correlation -1"
    m = t.randn(4, 5, generator=g)
    assert abs(spearman_correlation(m, m * 2) - 1.0) < 1e-6, "Should work on tensors of any shape (flatten first)"
    print("All tests in `test_spearman_correlation` passed!")


def test_baseline_head_scores(baseline_head_scores: Callable):
    g = t.Generator().manual_seed(0)
    n_layers, batch, pos, heads, d = 2, 3, 4, 3, 5
    names = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]
    clean = {n: t.randn(batch, pos, heads, d, generator=g) for n in names}
    corrupt = {n: t.randn(batch, pos, heads, d, generator=g) for n in names}
    grads = {n: t.randn(batch, pos, heads, d, generator=g) for n in names}
    result = baseline_head_scores(clean, grads, corrupt, n_layers, seed=0)
    expected_keys = {"activation change only", "gradient only", "AP, shuffled pairing"}
    assert set(result) == expected_keys, f"Expected keys {expected_keys}, got {set(result)}"
    for key, value in result.items():
        assert value.shape == (n_layers, heads), f"{key!r} should have shape (n_layers, n_heads), got {tuple(value.shape)}"
    delta = t.stack([corrupt[n] - clean[n] for n in names])
    grad = t.stack([grads[n] for n in names])
    t.testing.assert_close(result["activation change only"], delta.norm(dim=-1).sum((1, 2)))
    t.testing.assert_close(result["gradient only"], grad.norm(dim=-1).sum((1, 2)))
    # The shuffled pairing must pair each head's delta with exactly one head's gradient, as a permutation
    d_flat = delta.permute(1, 2, 0, 3, 4).reshape(batch, pos, n_layers * heads, d)
    g_flat = grad.permute(1, 2, 0, 3, 4).reshape(batch, pos, n_layers * heads, d)
    pairings = t.einsum("bpid,bpjd->ij", d_flat, g_flat)  # pairings[i, j] = delta_i . grad_j
    shuffled = result["AP, shuffled pairing"].flatten()
    matches = (pairings - shuffled[:, None]).abs() < 1e-4
    assert matches.any(dim=1).all(), "Each shuffled score should be delta_i . grad_j for some head j"
    partners = matches.float().argmax(dim=1)
    assert len(set(partners.tolist())) == n_layers * heads, "The shuffled pairing should be a permutation of the heads"
    assert (partners != t.arange(n_layers * heads)).any(), "The pairing should actually be shuffled"
    print("All tests in `test_baseline_head_scores` passed!")


def test_attribution_patch_resid(attribution_patch_resid: Callable):
    g = t.Generator().manual_seed(1)
    names = [f"blocks.{l}.hook_resid_pre" for l in range(3)]
    clean = {n: t.randn(2, 4, 6, generator=g) for n in names}
    corrupt = {n: t.randn(2, 4, 6, generator=g) for n in names}
    grads = {n: t.randn(2, 4, 6, generator=g) for n in names}
    result = attribution_patch_resid(clean, grads, corrupt, 3)
    assert result.shape == (3, 4), f"Expected shape (n_layers, seq) = (3, 4), got {tuple(result.shape)}"
    for l, n in enumerate(names):
        for p in range(4):
            expected = sum((corrupt[n][b, p] - clean[n][b, p]) @ grads[n][b, p] for b in range(2))
            t.testing.assert_close(result[l, p], expected)
    print("All tests in `test_attribution_patch_resid` passed!")


def test_exact_patch_resid(exact_patch_resid: Callable, model: HookedTransformer, task):
    _, corrupt_cache = model.run_with_cache(task.corrupt_tokens)
    names = [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)]
    result = exact_patch_resid(model, task, {n: corrupt_cache[n] for n in names})
    seq = task.clean_tokens.shape[1]
    assert result.shape == (model.cfg.n_layers, seq), f"Expected shape (n_layers, seq), got {tuple(result.shape)}"
    with t.no_grad():
        clean_value = task.metric(model(task.clean_tokens), task.answers).mean()
    with contextlib.redirect_stderr(io.StringIO()):
        expected = tl_patching.get_act_patch_resid_pre(
            model, task.clean_tokens, corrupt_cache, lambda logits: task.metric(logits, task.answers).mean() - clean_value
        )
    t.testing.assert_close(result.cpu().float(), expected.cpu().float(), atol=1e-3, rtol=1e-3)
    print("All tests in `test_exact_patch_resid` passed!")


def test_metric_along_path(metric_along_path: Callable, model: HookedTransformer, task):
    hook_name, head = "blocks.9.attn.hook_z", 9
    index = np.s_[:, :, head]
    with t.no_grad():
        _, clean_cache = model.run_with_cache(task.clean_tokens, names_filter=hook_name)
        _, corrupt_cache = model.run_with_cache(task.corrupt_tokens, names_filter=hook_name)
    clean_acts, corrupt_acts = {hook_name: clean_cache[hook_name]}, {hook_name: corrupt_cache[hook_name]}
    values, slopes = metric_along_path(model, task, hook_name, index, [0.0, 0.5, 1.0], clean_acts, corrupt_acts)
    assert len(values) == 3 and len(slopes) == 3, "Expected one value and one slope per alpha"
    assert abs(values[0]) < 1e-4, f"f(0) should be 0 (the clean run), got {values[0]}"

    # f(1) should equal exact patching of this head
    def patch(z, hook):
        z[:, :, head] = corrupt_acts[hook_name][:, :, head]
        return z

    with t.no_grad():
        clean_value = task.metric(model(task.clean_tokens), task.answers).mean().item()
        patched = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=[(hook_name, patch)]), task.answers).mean().item()
    assert abs(values[2] - (patched - clean_value)) < 1e-3, f"f(1) = {values[2]:.4f} should equal the exact patching effect {patched - clean_value:.4f}"
    # f'(0) should equal the AP score of this head
    metric_fn = lambda logits: task.metric(logits, task.answers).mean()
    _, acts, grads = _retain_grads(model, task.clean_tokens, metric_fn, [hook_name])
    ap = ((corrupt_acts[hook_name][:, :, head] - acts[hook_name][:, :, head]) * grads[hook_name][:, :, head]).sum().item()
    assert abs(slopes[0] - ap) < 1e-3 * max(1, abs(ap)), f"f'(0) = {slopes[0]:.4f} should equal the AP score {ap:.4f}"
    # The slope at 0.5 should match a finite difference of f around 0.5
    v_lo, _ = metric_along_path(model, task, hook_name, index, [0.49], clean_acts, corrupt_acts)
    v_hi, _ = metric_along_path(model, task, hook_name, index, [0.51], clean_acts, corrupt_acts)
    fd = (v_hi[0] - v_lo[0]) / 0.02
    assert abs(slopes[1] - fd) < 0.05 * max(1, abs(fd)), f"f'(0.5) = {slopes[1]:.4f} doesn't match the finite difference {fd:.4f}"
    print("All tests in `test_metric_along_path` passed!")


def test_integrated_gradients_component(integrated_gradients_component: Callable, metric_along_path: Callable, model: HookedTransformer, task):
    hook_name = "blocks.0.hook_mlp_out"
    index = np.s_[:, 10]
    with t.no_grad():
        _, clean_cache = model.run_with_cache(task.clean_tokens, names_filter=hook_name)
        _, corrupt_cache = model.run_with_cache(task.corrupt_tokens, names_filter=hook_name)
    clean_acts, corrupt_acts = {hook_name: clean_cache[hook_name]}, {hook_name: corrupt_cache[hook_name]}
    _, slopes = metric_along_path(model, task, hook_name, index, [0.0, 0.25, 0.5, 0.75], clean_acts, corrupt_acts)
    ig1 = integrated_gradients_component(model, task, hook_name, index, 1, clean_acts, corrupt_acts)
    assert abs(ig1 - slopes[0]) < 1e-4, "With one step, integrated gradients should equal attribution patching (the slope at alpha=0)"
    ig4 = integrated_gradients_component(model, task, hook_name, index, 4, clean_acts, corrupt_acts)
    assert abs(ig4 - sum(slopes) / 4) < 1e-4, "With 4 steps, use the left Riemann sum over alpha = 0, 0.25, 0.5, 0.75"
    print("All tests in `test_integrated_gradients_component` passed!")


def test_build_edge_mask(build_edge_mask: Callable, sender_names: list[str], receiver_names: list[str]):
    mask = build_edge_mask()
    assert mask.dtype == t.bool, "The mask should be boolean"
    assert mask.shape == (len(sender_names), len(receiver_names)), f"Expected shape (senders, receivers), got {tuple(mask.shape)}"
    assert mask.sum().item() == 32491, f"Expected 32,491 edges, got {mask.sum().item():,}"
    s, r = sender_names.index, receiver_names.index
    checks = [
        ("input", "a0.0<q>", True), ("input", "logits", True), ("a0.0", "a0.1<q>", False), ("a0.0", "m0", True),
        ("m0", "a0.3<k>", False), ("m0", "a1.0<v>", True), ("a11.7", "logits", True), ("m11", "logits", True),
        ("a11.0", "m10", False), ("a5.5", "a6.9<k>", True), ("m3", "m3", False), ("a3.0", "a3.0<v>", False),
    ]
    for sender, receiver, expected in checks:
        assert mask[s(sender), r(receiver)].item() == expected, f"Edge {sender} -> {receiver} should be {'present' if expected else 'absent'}"
    print("All tests in `test_build_edge_mask` passed!")


def test_sender_activations(sender_activations: Callable, model: HookedTransformer):
    tokens, _, _ = _small_ioi(model)
    result = sender_activations(model, tokens)
    n_senders = 1 + model.cfg.n_layers * (model.cfg.n_heads + 1)
    assert result.shape == (*tokens.shape, n_senders, model.cfg.d_model), f"Expected shape (batch, pos, {n_senders}, d_model), got {tuple(result.shape)}"
    _, cache = model.run_with_cache(tokens)
    final = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    t.testing.assert_close(result.sum(dim=2) + model.b_O.sum(0), final, atol=1e-3, rtol=1e-3,
                           msg="Senders plus the attention output biases should add up to the final residual stream")
    t.testing.assert_close(result[:, :, 0], cache["blocks.0.hook_resid_pre"], msg="Sender 0 should be the input embeddings")
    t.testing.assert_close(result[:, :, 1 + model.cfg.n_heads], cache["blocks.0.hook_mlp_out"], msg="Sender 13 should be MLP0")
    print("All tests in `test_sender_activations` passed!")


def test_receiver_gradients(receiver_gradients: Callable, model: HookedTransformer):
    clean, corrupt, answers = _small_ioi(model)
    metric_fn = lambda logits: _ioi_metric(logits, answers).mean()
    grads = receiver_gradients(model, clean, metric_fn)
    n_receivers = model.cfg.n_layers * (3 * model.cfg.n_heads + 1) + 1
    assert grads.shape == (*clean.shape, n_receivers, model.cfg.d_model), f"Expected shape (batch, pos, {n_receivers}, d_model), got {tuple(grads.shape)}"
    final = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    _, _, oracle = _retain_grads(model, clean, metric_fn, ["blocks.0.hook_resid_pre", final, "blocks.4.hook_k_input", "blocks.7.hook_mlp_in"])
    t.testing.assert_close(grads[:, :, -1], oracle[final], atol=1e-5, rtol=1e-3, msg="The last receiver should be the logits (final resid_post)")
    t.testing.assert_close(grads.sum(dim=2), oracle["blocks.0.hook_resid_pre"], atol=1e-4, rtol=1e-3,
                           msg="Summed over all receivers, the gradients should equal the gradient at the input embedding")
    k_start = 4 * (3 * model.cfg.n_heads + 1) + model.cfg.n_heads
    t.testing.assert_close(grads[:, :, k_start : k_start + model.cfg.n_heads], oracle["blocks.4.hook_k_input"], atol=1e-5, rtol=1e-3,
                           msg="Layer 4's key-input gradients are in the wrong place")
    t.testing.assert_close(grads[:, :, 7 * (3 * model.cfg.n_heads + 1) + 3 * model.cfg.n_heads], oracle["blocks.7.hook_mlp_in"], atol=1e-5, rtol=1e-3,
                           msg="MLP7's input gradient is in the wrong place")
    # input_embed should replace the token embeddings
    with t.no_grad():
        corrupt_embed = model.embed(corrupt)
    swapped = receiver_gradients(model, clean, metric_fn, input_embed=corrupt_embed)
    direct = receiver_gradients(model, corrupt, metric_fn)
    t.testing.assert_close(swapped, direct, atol=1e-5, rtol=1e-3, msg="With input_embed, the model should run on those embeddings")
    print("All tests in `test_receiver_gradients` passed!")


def test_eap_scores(eap_scores: Callable, sender_activations: Callable, receiver_gradients: Callable, model: HookedTransformer, task):
    solutions = _solutions()
    scores = eap_scores(model, task, batch_size=len(task))
    assert scores.shape == solutions.EDGE_MASK.shape, f"Expected shape (senders, receivers), got {tuple(scores.shape)}"
    assert (scores.cpu()[~solutions.EDGE_MASK] == 0).all(), "Scores for non-edges should be zero (multiply by EDGE_MASK)"
    chunked = eap_scores(model, task, batch_size=2)
    t.testing.assert_close(chunked, scores, atol=1e-4, rtol=1e-3, msg="Scores should not depend on the batch size (check the 1/len(task) scaling)")
    # Oracle for edges into the logits: delta . gradient at the final residual stream, from an independent backward pass
    final = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    metric_fn = lambda logits: task.metric(logits, task.answers).mean()
    _, _, oracle = _retain_grads(model, task.clean_tokens, metric_fn, [final])
    delta = sender_activations(model, task.corrupt_tokens) - sender_activations(model, task.clean_tokens)
    expected = t.einsum("bpud,bpd->u", delta, oracle[final])
    t.testing.assert_close(scores[:, -1], expected, atol=1e-4, rtol=1e-3, msg="Scores of edges into the logits are wrong")
    print("All tests in `test_eap_scores` passed!")


def test_patch_single_edges(patch_single_edges: Callable, sender_activations: Callable, model: HookedTransformer, task):
    solutions = _solutions()
    logits_receiver = solutions.LOGITS_RECEIVER
    edges = [(0, logits_receiver), (solutions.head_sender(9, 9), logits_receiver), (solutions.mlp_sender(0), logits_receiver)]
    result = patch_single_edges(model, task, edges, batch_size=4)
    assert result.shape == (3,), f"Expected one effect per edge, got shape {tuple(result.shape)}"
    # Oracle: corrupting u -> logits adds delta_u to the final residual stream
    _, cache = model.run_with_cache(task.clean_tokens)
    final = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    delta = sender_activations(model, task.corrupt_tokens) - sender_activations(model, task.clean_tokens)
    with t.no_grad():
        clean_value = task.metric(model.unembed(model.ln_final(final)), task.answers).mean()
        for i, (u, _) in enumerate(edges):
            expected = task.metric(model.unembed(model.ln_final(final + delta[:, :, u])), task.answers).mean() - clean_value
            assert abs(result[i].item() - expected.item()) < 1e-3, f"Edge {i} into the logits: expected {expected.item():.4f}, got {result[i].item():.4f}"
    # Head and MLP receivers, compared against the solution
    edges = [(0, solutions.head_receiver(0, "k", 10)), (solutions.head_sender(5, 5), solutions.head_receiver(6, "q", 9)),
             (solutions.head_sender(2, 2), solutions.mlp_receiver(4))]
    t.testing.assert_close(patch_single_edges(model, task, edges, batch_size=4), solutions.patch_single_edges(model, task, edges), atol=1e-4, rtol=1e-3)
    print("All tests in `test_patch_single_edges` passed!")


def test_run_with_circuit(run_with_circuit: Callable, patch_single_edges: Callable, model: HookedTransformer, task):
    solutions = _solutions()
    mask = solutions.EDGE_MASK
    with t.no_grad():
        clean_value = task.metric(model(task.clean_tokens), task.answers).mean().item()
        corrupt_value = task.metric(model(task.corrupt_tokens), task.answers).mean().item()
    full = run_with_circuit(model, task, mask.clone(), batch_size=4)
    assert abs(full - clean_value) < 1e-3, f"The full graph should give the clean metric {clean_value:.4f}, got {full:.4f}"
    empty = run_with_circuit(model, task, t.zeros_like(mask), batch_size=4)
    assert abs(empty - corrupt_value) < 1e-3, f"The empty circuit should give the corrupted metric {corrupt_value:.4f}, got {empty:.4f}"
    edges = [(0, solutions.head_receiver(0, "q", 1)), (solutions.mlp_sender(0), solutions.mlp_receiver(3)), (solutions.head_sender(9, 9), solutions.LOGITS_RECEIVER)]
    single = patch_single_edges(model, task, edges)
    for (u, v), effect in zip(edges, single.tolist()):
        circuit = mask.clone()
        circuit[u, v] = False
        value = run_with_circuit(model, task, circuit, batch_size=4)
        assert abs(value - clean_value - effect) < 1e-3, (
            f"Removing only edge {solutions.SENDER_NAMES[u]} -> {solutions.RECEIVER_NAMES[v]} should match single-edge patching "
            f"({effect:+.4f}), got {value - clean_value:+.4f}"
        )
    print("All tests in `test_run_with_circuit` passed!")


def test_faithfulness_curve(faithfulness_curve: Callable, model: HookedTransformer, task):
    solutions = _solutions()
    n_edges = int(solutions.EDGE_MASK.sum())
    scores = t.rand(*solutions.EDGE_MASK.shape, generator=t.Generator().manual_seed(0))
    result = faithfulness_curve(model, task, scores, [n_edges, 50])
    assert len(result) == 2, "Expected one faithfulness value per k"
    assert abs(result[0] - 1.0) < 1e-3, f"Keeping every edge should give faithfulness 1, got {result[0]:.4f}"
    expected = solutions.faithfulness_curve(model, task, scores, [50])[0]
    assert abs(result[1] - expected) < 1e-3, f"Expected faithfulness {expected:.4f} for the top-50 circuit, got {result[1]:.4f}"
    print("All tests in `test_faithfulness_curve` passed!")


def test_eap_ig_scores(eap_ig_scores: Callable, eap_scores: Callable, model: HookedTransformer, task):
    solutions = _solutions()
    one_step = eap_ig_scores(model, task, n_steps=1, batch_size=len(task))
    t.testing.assert_close(one_step, eap_scores(model, task, batch_size=len(task)), atol=1e-4, rtol=1e-3,
                           msg="With one step (alpha = 0 only), EAP-IG should equal EAP")
    three_steps = eap_ig_scores(model, task, n_steps=3, batch_size=3)
    expected = solutions.eap_ig_scores(model, task, n_steps=3, batch_size=len(task))
    t.testing.assert_close(three_steps, expected, atol=1e-4, rtol=1e-3, msg="Use alpha = k / n_steps for k = 0, ..., n_steps - 1")
    print("All tests in `test_eap_ig_scores` passed!")


def test_joint_vs_individual(joint_vs_individual: Callable, model: HookedTransformer, task):
    solutions = _solutions()
    edge = (solutions.head_sender(9, 9), solutions.LOGITS_RECEIVER)
    individual, joint = joint_vs_individual(model, task, [edge])
    assert individual.shape == (1,), "Expected one individual effect per edge"
    assert abs(individual.item() - joint) < 1e-3, "For a single edge, the joint and individual effects should be equal"
    edges = [(0, solutions.head_receiver(0, qkv, 1)) for qkv in "qkv"]
    individual, joint = joint_vs_individual(model, task, edges)
    expected_individual, expected_joint = solutions.joint_vs_individual(model, task, edges)
    t.testing.assert_close(individual, expected_individual, atol=1e-4, rtol=1e-3)
    assert abs(joint - expected_joint) < 1e-3, f"Expected joint effect {expected_joint:.4f}, got {joint:.4f}"
    print("All tests in `test_joint_vs_individual` passed!")
