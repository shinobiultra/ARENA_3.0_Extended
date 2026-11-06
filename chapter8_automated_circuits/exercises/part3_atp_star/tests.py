from typing import Callable

import torch as t
from transformer_lens import HookedTransformer

KINDS = ["q", "k", "v", "z"]


def _small_task(task, n: int = 6):
    return task[:n]


def test_exact_node_effects(exact_node_effects: Callable, cache_nodes: Callable, model: HookedTransformer, task):
    task = _small_task(task)
    clean_cache = cache_nodes(model, task.clean_tokens)
    corrupt_cache = cache_nodes(model, task.corrupt_tokens)
    per_prompt = exact_node_effects(model, task, clean_cache, corrupt_cache, per_prompt=True)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    assert per_prompt.shape == (4, L, H, len(task)), f"Expected shape (4, n_layers, n_heads, batch), got {tuple(per_prompt.shape)}"
    mean = exact_node_effects(model, task, clean_cache, corrupt_cache, per_prompt=False)
    assert mean.shape == (4, L, H), f"Expected shape (4, n_layers, n_heads), got {tuple(mean.shape)}"
    t.testing.assert_close(mean.cpu(), per_prompt.mean(-1).cpu(), atol=1e-4, rtol=1e-4)
    # Independent check of two nodes by patching directly
    with t.no_grad():
        clean_metric = task.metric(model(task.clean_tokens), task.answers)
        for kind, layer, head in [("k", 6, 9), ("z", 9, 9)]:
            name = f"blocks.{layer}.attn.hook_{kind}"
            corrupt_act = model.run_with_cache(task.corrupt_tokens, names_filter=name)[1][name]

            def hook(act, hook):
                act[:, :, head] = corrupt_act[:, :, head]
                return act

            patched = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, hook)]), task.answers)
            t.testing.assert_close(
                per_prompt[KINDS.index(kind), layer, head].cpu(), (patched - clean_metric).cpu(), atol=1e-3, rtol=1e-3,
                msg=f"Wrong per-prompt effect for {kind}{layer}.{head}",
            )
    print("All tests in `test_exact_node_effects` passed!")


def test_node_gradients(node_gradients: Callable, model: HookedTransformer, task, check_drop: bool = False):
    task = _small_task(task, 4)
    grads = node_gradients(model, task)
    name = "blocks.5.attn.hook_v"
    assert name in grads, f"Expected a gradient for {name}"
    assert "blocks.5.attn.hook_pattern" in grads, "Expected a gradient for the attention pattern too"
    # Finite-difference check: nudge one head's value vectors in a random direction
    t.manual_seed(0)
    _, cache = model.run_with_cache(task.clean_tokens, names_filter=name)
    direction = t.randn_like(cache[name][:, :, 3])
    eps = 1e-2

    def nudge(act, hook):
        act[:, :, 3] = act[:, :, 3] + eps * direction
        return act

    with t.no_grad():
        base = task.metric(model(task.clean_tokens), task.answers)
        nudged = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, nudge)]), task.answers)
    finite_diff = (nudged - base) / eps
    analytic = (grads[name][:, :, 3] * direction).sum((1, 2))
    t.testing.assert_close(analytic.cpu(), finite_diff.cpu(), atol=0.05, rtol=0.05,
                           msg="Gradients don't match finite differences: are you differentiating the *sum* of the per-prompt metric?")
    if not check_drop:
        print("All tests in `test_node_gradients` passed! (drop_layer not checked yet)")
        return
    # GradDrop: dropping layer 5 must zero the gradient of layer 5's heads, but not change the forward pass
    dropped = node_gradients(model, task, drop_layer=5)
    assert dropped[name].abs().max() < 1e-6, "Dropping layer 5 should zero the gradient reaching layer 5's value vectors"
    assert dropped["blocks.2.attn.hook_v"].abs().max() > 0, "Dropping layer 5 shouldn't zero gradients at earlier layers"
    print("All tests in `test_node_gradients` passed!")


def test_atp_estimates(atp_estimates: Callable, cache_nodes: Callable, node_gradients: Callable, model: HookedTransformer, task):
    task = _small_task(task, 4)
    clean_cache = cache_nodes(model, task.clean_tokens)
    corrupt_cache = cache_nodes(model, task.corrupt_tokens)
    grads = node_gradients(model, task)
    est = atp_estimates(clean_cache, corrupt_cache, grads)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    assert est.shape == (4, L, H, len(task)), f"Expected shape (4, n_layers, n_heads, batch), got {tuple(est.shape)}"
    name = "blocks.7.attn.hook_k"
    expected = ((corrupt_cache[name][:, :, 9] - clean_cache[name][:, :, 9]) * grads[name][:, :, 9]).sum((1, 2))
    t.testing.assert_close(est[1, 7, 9].cpu(), expected.cpu(), atol=1e-4, rtol=1e-4)
    print("All tests in `test_atp_estimates` passed!")


def test_recall_curve(recall_curve: Callable):
    exact = t.tensor([0.0, 0.5, -0.3, 0.05, 0.2])
    scores = t.tensor([0.9, 0.1, 0.8, 0.7, 0.0])
    curve = recall_curve(scores, exact, threshold=0.1)
    # important nodes: 1, 2, 4. Order by |score|: 0, 2, 3, 1, 4
    expected = t.tensor([0, 1, 1, 2, 3]) / 3
    assert curve.shape == (5,), f"Expected one value per number of verified nodes, got shape {tuple(curve.shape)}"
    t.testing.assert_close(curve.float(), expected.float())
    neg_scores = -scores
    t.testing.assert_close(recall_curve(neg_scores, exact, threshold=0.1).float(), expected.float(),
                           msg="Rank by the absolute value of the scores")
    print("All tests in `test_recall_curve` passed!")


def test_pattern_from_qk(pattern_from_qk: Callable, model: HookedTransformer, task):
    task = _small_task(task, 3)
    names = ["blocks.4.attn.hook_q", "blocks.4.attn.hook_k", "blocks.4.attn.hook_pattern"]
    _, cache = model.run_with_cache(task.clean_tokens, names_filter=lambda n: n in names)
    pattern = pattern_from_qk(cache[names[0]], cache[names[1]])
    assert pattern.shape == cache[names[2]].shape, f"Expected shape {tuple(cache[names[2]].shape)}, got {tuple(pattern.shape)}"
    t.testing.assert_close(pattern, cache[names[2]], atol=1e-4, rtol=1e-4,
                           msg="Your pattern doesn't match the model's hook_pattern: check the scale and the causal mask")
    print("All tests in `test_pattern_from_qk` passed!")


def test_qk_fix_estimates(qk_fix_estimates: Callable, cache_nodes: Callable, node_gradients: Callable, model: HookedTransformer, task):
    task = _small_task(task, 4)
    clean_cache = cache_nodes(model, task.clean_tokens)
    corrupt_cache = cache_nodes(model, task.corrupt_tokens)
    grads = node_gradients(model, task)
    est = qk_fix_estimates(clean_cache, corrupt_cache, grads)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    assert est.shape == (2, L, H, len(task)), f"Expected shape (2, n_layers, n_heads, batch), got {tuple(est.shape)}"
    # Oracle: run the model with the patched key and read off the resulting attention pattern
    layer, head = 10, 7
    k_name, p_name = f"blocks.{layer}.attn.hook_k", f"blocks.{layer}.attn.hook_pattern"
    store = {}

    def patch_k(act, hook):
        act[:, :, head] = corrupt_cache[k_name][:, :, head]
        return act

    def grab(act, hook):
        store["p"] = act.detach().clone()

    with t.no_grad():
        model.run_with_hooks(task.clean_tokens, fwd_hooks=[(k_name, patch_k), (p_name, grab)])
    expected = ((store["p"][:, head] - clean_cache[p_name][:, head]) * grads[p_name][:, head]).sum((1, 2))
    t.testing.assert_close(est[1, layer, head].cpu(), expected.cpu(), atol=1e-4, rtol=1e-4)
    print("All tests in `test_qk_fix_estimates` passed!")


def test_toy(toy_forward: Callable, toy_estimates: Callable):
    x = t.tensor(1.0)
    assert abs(toy_forward(x).item() - 2.0) < 1e-6, "Clean output of the toy should be 2"
    assert abs(toy_forward(x, node_override=t.tensor(0.0)).item() - 1.0) < 1e-6, "Patched output should be 1"
    est = toy_estimates(x_clean=t.tensor(1.0), x_corrupt=t.tensor(0.0))
    assert abs(est["exact"] - (-1.0)) < 1e-6, f"Exact effect should be -1, got {est['exact']}"
    assert abs(est["atp"]) < 1e-6, f"AtP should be exactly 0 (the two gradient paths cancel), got {est['atp']}"
    assert abs(est["graddrop"] - 2.0) < 1e-6, f"GradDrop score should be 2, got {est['graddrop']}"
    print("All tests in `test_toy` passed!")


def test_graddrop_scores(graddrop_scores: Callable, cache_nodes: Callable, model: HookedTransformer, task):
    import part3_atp_star.solutions as solutions

    task = _small_task(task, 4)
    clean_cache = cache_nodes(model, task.clean_tokens)
    corrupt_cache = cache_nodes(model, task.corrupt_tokens)
    scores = graddrop_scores(model, task, clean_cache, corrupt_cache)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    assert scores.shape == (4, L, H), f"Expected shape (4, n_layers, n_heads), got {tuple(scores.shape)}"
    assert (scores >= 0).all(), "GradDrop scores are averages of absolute values, so they should be non-negative"
    expected = solutions.graddrop_scores(model, task, clean_cache, corrupt_cache)
    t.testing.assert_close(scores.cpu(), expected.cpu(), atol=1e-4, rtol=1e-4)
    print("All tests in `test_graddrop_scores` passed!")


def test_neuron_atp(neuron_atp: Callable, model: HookedTransformer, task):
    task = _small_task(task, 4)
    scores = neuron_atp(model, task)
    assert scores.shape == (model.cfg.n_layers, model.cfg.d_mlp), f"Expected shape (n_layers, d_mlp), got {tuple(scores.shape)}"
    # Finite-difference oracle for one neuron: interpolate a small step towards the corrupted activation
    layer, idx = 9, 100
    name = f"blocks.{layer}.mlp.hook_post"
    with t.no_grad():
        corrupt = model.run_with_cache(task.corrupt_tokens, names_filter=name)[1][name][:, :, idx]
        clean_metric = task.metric(model(task.clean_tokens), task.answers)
        eps = 1e-2

        def step(act, hook):
            act[:, :, idx] = act[:, :, idx] + eps * (corrupt - act[:, :, idx])
            return act

        stepped = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, step)]), task.answers)
    expected = ((stepped - clean_metric) / eps).mean()
    t.testing.assert_close(scores[layer, idx].cpu(), expected.cpu(), atol=2e-3, rtol=0.05,
                           msg="AtP for a neuron should be the per-prompt directional derivative, averaged over prompts")
    per_prompt = neuron_atp(model, task, per_prompt=True)
    assert per_prompt.shape == (model.cfg.n_layers, model.cfg.d_mlp, len(task.answers)), (
        f"With per_prompt=True, expected shape (n_layers, d_mlp, batch), got {tuple(per_prompt.shape)}"
    )
    t.testing.assert_close(per_prompt.mean(-1), scores, atol=1e-5, rtol=1e-4,
                           msg="The per-prompt estimates should average to the per_prompt=False scores")
    print("All tests in `test_neuron_atp` passed!")


def test_exact_neuron_effects(exact_neuron_effects: Callable, model: HookedTransformer, task):
    task = _small_task(task, 4)
    neurons = [(9, 100), (3, 5)]
    effects = exact_neuron_effects(model, task, neurons)
    assert effects.shape == (2,), f"Expected one effect per neuron, got shape {tuple(effects.shape)}"
    layer, idx = neurons[0]
    name = f"blocks.{layer}.mlp.hook_post"
    with t.no_grad():
        corrupt = model.run_with_cache(task.corrupt_tokens, names_filter=name)[1][name][:, :, idx]

        def patch(act, hook):
            act[:, :, idx] = corrupt
            return act

        patched = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, patch)]), task.answers)
        clean = task.metric(model(task.clean_tokens), task.answers)
    t.testing.assert_close(effects[0].cpu(), (patched - clean).mean().cpu(), atol=1e-4, rtol=1e-4)
    print("All tests in `test_exact_neuron_effects` passed!")


def test_group_effects(group_effects: Callable, exact_neuron_effects: Callable, model: HookedTransformer, task):
    task = _small_task(task, 4)
    groups = [[(9, 100)], [(9, 100), (3, 5)]]
    effects = group_effects(model, task, groups)
    assert effects.shape == (2,), f"Expected one effect per group, got shape {tuple(effects.shape)}"
    single = exact_neuron_effects(model, task, [(9, 100)])
    t.testing.assert_close(effects[0].cpu(), single[0].cpu(), atol=1e-4, rtol=1e-4,
                           msg="A group containing one neuron should have the same effect as patching that neuron")
    print("All tests in `test_group_effects` passed!")


def test_bisect_group(bisect_group: Callable):
    # A synthetic additive problem with known answers: 1000 neurons with tiny effects, plus two planted ones
    rng = t.Generator().manual_seed(0)
    group = [(i // 100, i % 100) for i in range(1000)]
    effects = {neuron: 0.0002 * float(t.randn(1, generator=rng)) for neuron in group}
    effects[(3, 17)], effects[(8, 90)] = -0.5, 0.3
    calls = []

    def effect_fn(subgroup):
        calls.append(len(subgroup))
        return sum(effects[n] for n in subgroup)

    found, n_calls = bisect_group(group, effect_fn, threshold=0.1)
    found_neurons = sorted(n for n, _ in found)
    assert found_neurons == [(3, 17), (8, 90)], f"Expected to find the planted neurons [(3, 17), (8, 90)], got {found_neurons}"
    assert all(abs(e - effects[n]) < 1e-9 for n, e in found), "Each found neuron should be reported with its own effect"
    assert n_calls == len(calls), f"You reported {n_calls} calls, but effect_fn was called {len(calls)} times"
    assert n_calls <= 45, f"Bisection should need about 2 log2(1000) calls per planted neuron, but used {n_calls}"

    # A group with nothing big in it should be rejected with a single call
    calls.clear()
    small = [n for n in group if n not in ((3, 17), (8, 90))]
    found, n_calls = bisect_group(small, effect_fn, threshold=0.1)
    assert found == [] and n_calls == 1, f"A group whose effect is below threshold shouldn't be split (got {found}, {n_calls} calls)"
    print("All tests in `test_bisect_group` passed!")
