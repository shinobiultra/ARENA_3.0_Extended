from typing import Callable

import torch as t
import torch.nn.functional as F
from transformer_lens import HookedTransformer

N_LAYERS, N_HEADS = 2, 12


def _names():
    senders = ["embed"] + [f"{layer}.{head}" for layer in range(N_LAYERS) for head in range(N_HEADS)]
    receivers = [f"{layer}.{head}<{c}>" for layer in range(N_LAYERS) for c in "qkv" for head in range(N_HEADS)] + ["logits"]
    return senders, receivers


def test_build_edge_mask(build_edge_mask: Callable):
    senders, receivers = _names()
    mask = build_edge_mask()
    assert mask.shape == (len(senders), len(receivers)), f"Expected shape {(len(senders), len(receivers))}, got {tuple(mask.shape)}"
    assert mask.dtype == t.bool, "The edge mask should be a boolean tensor"
    s, r = senders.index, receivers.index
    assert mask[s("embed")].all(), "The embedding is read by every receiver"
    assert mask[s("0.7"), r("1.4<k>")] and mask[s("0.7"), r("logits")], "Layer-0 heads write to layer-1 inputs and the logits"
    assert not mask[s("0.7"), r("0.3<q>")], "A layer-0 head can't send to another layer-0 head (they run in parallel)"
    assert not mask[s("1.2"), r("1.5<v>")], "A layer-1 head can't send to a layer-1 head"
    assert not mask[s("1.2"), r("0.5<v>")], "Later heads can't send to earlier ones"
    assert mask[s("1.2"), r("logits")]
    assert mask.sum() == 529, f"Expected 529 edges, got {int(mask.sum())}"
    print("All tests in `test_build_edge_mask` passed!")


def test_sender_outputs(sender_outputs: Callable, model: HookedTransformer, tokens: t.Tensor):
    out = sender_outputs(model, tokens)
    batch, seq = tokens.shape
    assert out.shape == (batch, seq, 25, model.cfg.d_model), f"Expected (batch, seq, 25, d_model), got {tuple(out.shape)}"
    # Independent check: the residual stream is the sum of all senders plus the output biases
    _, cache = model.run_with_cache(tokens, names_filter="blocks.1.hook_resid_post")
    resid = out.sum(dim=2) + model.b_O.sum(0)
    t.testing.assert_close(resid, cache["blocks.1.hook_resid_post"], atol=1e-3, rtol=1e-4,
                           msg="Sender outputs plus the output biases should add up to the final residual stream")
    t.testing.assert_close(out[:, :, 0], model.W_E[tokens], msg="Sender 0 should be the token embedding")
    print("All tests in `test_sender_outputs` passed!")


def _hook_patched_logits(model, clean, corrupt_out, clean_out, layer, kind, head, sender):
    """Oracle for removing one edge (sender -> head's `kind` input), using TransformerLens hooks."""
    model.set_use_split_qkv_input(True)

    def patch(x, hook):
        x = x.clone()
        x[:, :, head] += corrupt_out[:, :, sender] - clean_out[:, :, sender]
        return x

    try:
        return model.run_with_hooks(clean, fwd_hooks=[(f"blocks.{layer}.hook_{kind}_input", patch)])
    finally:
        model.set_use_split_qkv_input(False)


@t.no_grad()
def test_run_graph(run_graph: Callable, sender_outputs: Callable, edge_mask: t.Tensor, model: HookedTransformer,
                   clean: t.Tensor, corrupt: t.Tensor):
    senders, receivers = _names()
    clean, corrupt = clean[:4], corrupt[:4]
    corrupt_out, clean_out = sender_outputs(model, corrupt), sender_outputs(model, clean)
    full = run_graph(model, clean, corrupt_out, edge_mask)
    t.testing.assert_close(full, model(clean), atol=1e-3, rtol=1e-4, msg="With every edge in the circuit, you should get the model's logits")
    empty = run_graph(model, clean, corrupt_out, t.zeros_like(edge_mask))
    t.testing.assert_close(empty, model(corrupt), atol=1e-3, rtol=1e-4, msg="With no edges, you should get the logits on the corrupted prompts")

    # Removing a single edge should match patching that one input with TransformerLens hooks
    for sender, receiver, (layer, kind, head) in [("0.7", "1.4<k>", (1, "k", 4)), ("embed", "0.7<v>", (0, "v", 7))]:
        circuit = edge_mask.clone()
        circuit[senders.index(sender), receivers.index(receiver)] = False
        ours = run_graph(model, clean, corrupt_out, circuit)
        oracle = _hook_patched_logits(model, clean, corrupt_out, clean_out, layer, kind, head, senders.index(sender))
        t.testing.assert_close(ours, oracle, atol=1e-3, rtol=1e-4,
                               msg=f"Removing the edge {sender} -> {receiver} should match patching that input with a hook")
    # Edges that aren't in the graph must have no effect, whatever the circuit says about them
    junk = edge_mask.clone() | ~edge_mask
    t.testing.assert_close(run_graph(model, clean, corrupt_out, junk), full, atol=1e-3, rtol=1e-4,
                           msg="Entries of in_circuit outside the edge mask should be ignored")
    print("All tests in `test_run_graph` passed!")


def test_kl_divergence(kl_divergence: Callable):
    t.manual_seed(0)
    full, circuit = t.randn(3, 5, 11), t.randn(3, 5, 11)
    ours = kl_divergence(circuit, full.log_softmax(-1))
    expected = F.kl_div(circuit.log_softmax(-1), full.log_softmax(-1), log_target=True, reduction="none").sum(-1).mean()
    assert abs(float(ours) - expected.item()) < 1e-5, f"Expected {expected.item():.5f}, got {float(ours):.5f}"
    assert abs(float(kl_divergence(full, full.log_softmax(-1)))) < 1e-6, "KL of a distribution with itself is zero"
    print("All tests in `test_kl_divergence` passed!")


def test_acdc(acdc: Callable):
    # A synthetic graph with known answers: senders [in, A, B], receivers [A_in, B_in, out].
    edge_mask = t.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=t.bool)
    importance = t.tensor([[0.8, 0.9, 0.5], [0.0, 0.9, 1.0], [0.0, 0.0, 0.01]])
    calls = []

    def metric(circuit):  # additive: each removed edge costs its importance
        calls.append(circuit.clone())
        return float((importance * (edge_mask & ~circuit)).sum())

    circuit, n_tests = acdc(metric, edge_mask, receiver_order=[2, 1, 0], receiver_owner=[1, 2, None], threshold=0.1)
    expected = t.tensor([[1, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=t.bool)
    assert t.equal(circuit, expected), (
        f"Expected the circuit\n{expected.int()}\ngot\n{circuit.int()}\n(B's only output edge is unimportant, so B leaves "
        "the circuit, and its input edges should be removed without being tested even though they look important)"
    )
    assert n_tests == 4, f"Expected 4 edge tests (3 into `out`, none into B_in, 1 into A_in), got {n_tests}"
    assert len(calls) == n_tests + 1, "Call the metric once at the start, then once per tested edge"
    # Within a receiver, senders are tried latest first: the first test into `out` removes B -> out
    first = calls[1]
    assert not first[2, 2] and first[1, 2] and first[0, 2], "Try removing the latest sender's edge first (B -> out)"

    # Recomputation: removing an edge changes the effect of the others
    def nonadditive(circuit):
        removed = edge_mask & ~circuit
        return float(0.06 * removed[:, 2].sum() + 0.5 * (removed[0, 2] & removed[1, 2]))

    circuit, _ = acdc(nonadditive, edge_mask, receiver_order=[2, 1, 0], receiver_owner=[1, 2, None], threshold=0.1)
    assert not circuit[2, 2] and not circuit[1, 2] and circuit[0, 2], (
        "Each decision should use the metric of the *current* circuit: once A -> out is gone, removing in -> out "
        "costs 0.56, so it must stay"
    )
    print("All tests in `test_acdc` passed!")


def test_faithfulness(faithfulness: Callable):
    assert abs(faithfulness(-3.0, full=-3.0, empty=-13.0) - 1.0) < 1e-9
    assert abs(faithfulness(-13.0, full=-3.0, empty=-13.0)) < 1e-9
    assert abs(faithfulness(-8.0, full=-3.0, empty=-13.0) - 0.5) < 1e-9
    assert faithfulness(-2.0, full=-3.0, empty=-13.0) > 1, "A circuit can do better than the full model"
    print("All tests in `test_faithfulness` passed!")


def test_random_circuit(random_circuit: Callable, edge_mask: t.Tensor):
    g = t.Generator().manual_seed(0)
    c = random_circuit(edge_mask, 17, g)
    assert c.dtype == t.bool and c.shape == edge_mask.shape
    assert c.sum() == 17, f"Expected exactly 17 edges, got {int(c.sum())}"
    assert not (c & ~edge_mask).any(), "Random circuits must only contain valid edges"
    counts = sum(random_circuit(edge_mask, 100, g).float() for _ in range(200))
    valid = counts[edge_mask]
    assert valid.min() > 0, "Every valid edge should be sampled sometimes"
    assert abs(valid.mean() - 200 * 100 / int(edge_mask.sum())) < 1e-6
    print("All tests in `test_random_circuit` passed!")


def test_roc_point(roc_point: Callable):
    edge_mask = t.ones(2, 5, dtype=t.bool)
    edge_mask[1, 0] = False  # 9 valid edges
    reference = t.zeros(2, 5, dtype=t.bool)
    reference[0, :3] = True  # 3 positives, 6 negatives
    circuit = t.zeros(2, 5, dtype=t.bool)
    circuit[0, :2] = True
    circuit[1, 4] = True
    fpr, tpr = roc_point(circuit, reference, edge_mask)
    assert abs(tpr - 2 / 3) < 1e-6, f"TPR should be 2/3, got {tpr}"
    assert abs(fpr - 1 / 6) < 1e-6, f"FPR should be 1/6, got {fpr}"
    print("All tests in `test_roc_point` passed!")


def test_gpt2_order(gpt2_receiver_order: Callable, gpt2_receiver_owner: Callable, graph):
    order, owner = gpt2_receiver_order(), gpt2_receiver_owner()
    assert sorted(order) == list(range(graph.N_RECEIVERS)), "The order should contain every receiver exactly once"
    assert order[0] == graph.LOGITS_RECEIVER, "Start from the logits"
    pos = {v: i for i, v in enumerate(order)}
    mask = graph.EDGE_MASK
    for v in range(graph.N_RECEIVERS):
        if owner[v] is None:
            continue
        # every receiver of the node that owns v must already have been processed
        later_receivers = mask[owner[v]].nonzero().flatten().tolist()
        assert all(pos[w] < pos[v] for w in later_receivers), (
            f"{graph.RECEIVER_NAMES[v]} is processed before some receiver that its node {graph.SENDER_NAMES[owner[v]]} sends to"
        )
    assert owner[graph.LOGITS_RECEIVER] is None
    assert owner[graph.head_receiver(5, "k", 5)] == graph.head_sender(5, 5)
    assert owner[graph.mlp_receiver(3)] == graph.mlp_sender(3)
    print("All tests in `test_gpt2_order` passed!")


def test_single_edge_effects(single_edge_effects: Callable):
    mask = t.tensor([[True, False, True], [False, True, False]])
    before, calls = mask.clone(), []
    costs = t.tensor([[2., 99., -3.], [99., 5., 99.]])

    def metric(circuit):
        calls.append(circuit.clone())
        assert t.equal(circuit & ~mask, t.zeros_like(mask)), 'Only valid graph edges may be active'
        assert int((mask & ~circuit).sum()) == 1, 'Start each intervention from the full graph and cut exactly one edge'
        return 10 + float(costs[mask & ~circuit].sum())

    actual = single_edge_effects(metric, mask)
    t.testing.assert_close(actual, t.tensor([[12., 0., 7.], [0., 15., 0.]]))
    assert len(calls) == 3 and len({tuple(c.flatten().tolist()) for c in calls}) == 3
    t.testing.assert_close(mask, before, msg='Do not change the supplied full graph')
    print('All tests in `test_single_edge_effects` passed!')


def test_attention_by_offset(attention_by_offset: Callable):
    # Explicit probability rows; head 1 looks one token back, head 0 is diffuse.
    diffuse = t.tensor([[1., 0., 0., 0., 0.], [.2, .8, 0., 0., 0.], [.1, .3, .6, 0., 0.],
                        [.05, .15, .3, .5, 0.], [.1, .1, .2, .2, .4]])
    previous = t.tensor([[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],
                         [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.]])
    pattern = t.stack([t.stack([diffuse, previous]), t.stack([t.eye(5), diffuse])])
    before = pattern.clone()
    tokens = t.zeros(2, 5, dtype=t.long)

    class Model:
        def run_with_cache(self, supplied, names_filter):
            t.testing.assert_close(supplied, tokens)
            assert names_filter == 'blocks.2.attn.hook_pattern'
            return None, {names_filter: pattern}

    for start_pos, max_offset in ((1, 4), (0, 2), (3, 2)):
        actual = attention_by_offset(Model(), tokens, layer=2, max_offset=max_offset, start_pos=start_pos)
        # Independent query/key indexing, including all valid queries when d > start_pos.
        expected = t.tensor([[t.stack([pattern[b, h, q, q - d]
                                      for b in range(2) for q in range(max(start_pos, d), 5)]).mean()
                              for d in range(max_offset + 1)] for h in range(2)])
        t.testing.assert_close(actual, expected, msg='Average attention at key=query-offset over every valid query')
    t.testing.assert_close(pattern, before)
    print('All tests in `test_attention_by_offset` passed!')


def test_top_k_circuit(top_k_circuit: Callable):
    scores = t.tensor([[2., 100., -1.], [4., 3., 90.]])
    mask = t.tensor([[True, False, True], [True, True, False]])
    saved_scores, saved_mask = scores.clone(), mask.clone()
    actual = top_k_circuit(scores, mask, 2)
    assert actual.dtype == t.bool
    assert t.equal(actual, t.tensor([[False, False, False], [True, True, False]])), 'Rank by score among valid edges only'
    assert not top_k_circuit(scores, mask, 0).any()
    assert t.equal(top_k_circuit(scores, mask, 4), mask), 'Keeping all valid edges must recover the graph'
    t.testing.assert_close(scores, saved_scores)
    t.testing.assert_close(mask, saved_mask)
    print('All tests in `test_top_k_circuit` passed!')


def test_completeness_scores(completeness_scores: Callable):
    mask = t.tensor([[True, True], [True, False]])
    circuit = t.tensor([[True, False], [True, False]])
    weights = t.tensor([[1., 2.], [4., 8.]])
    snapshots = [circuit.clone(), mask.clone()]

    for fraction in (0., 1., .5):
        calls = []

        def faith(graph):
            calls.append(graph.clone())
            return float(weights[graph].sum())

        c_scores, m_scores = completeness_scores(circuit, mask, faith, 6, fraction, t.Generator().manual_seed(11))
        assert c_scores.shape == m_scores.shape == (6,) and len(calls) == 12
        for i, (smaller, full) in enumerate(zip(calls[::2], calls[1::2])):
            knockout = circuit & ~smaller
            assert not (smaller & ~circuit).any(), 'Only remove edges from the circuit'
            assert t.equal(mask & ~full, knockout), 'Use the identical knockout K in C\\K and M\\K'
            assert abs(c_scores[i] - weights[smaller].sum()) < 1e-6
            assert abs(m_scores[i] - weights[full].sum()) < 1e-6
            if fraction == 0:
                assert c_scores[i] == 5 and m_scores[i] == 7
            if fraction == 1:
                assert c_scores[i] == 0 and m_scores[i] == 2
        if fraction == .5:
            assert any(not t.equal(calls[0], c) for c in calls[2::2]), 'Draw fresh subsets from the supplied generator'
    for actual, saved in zip((circuit, mask), snapshots):
        t.testing.assert_close(actual, saved)
    print('All tests in `test_completeness_scores` passed!')


def test_edge_drops(edge_drops: Callable):
    circuit = t.tensor([[True, False], [True, True]])
    before, calls = circuit.clone(), []
    weights = t.tensor([[2., 99.], [-3., 5.]])

    def faith(graph):
        calls.append(graph.clone())
        return float(weights[graph].sum() + 7 * (graph[0, 0] & graph[1, 1]))

    actual = edge_drops(circuit, faith)
    t.testing.assert_close(actual, t.tensor([9., -3., 12.]), msg='Return baseline minus knockout faithfulness in nonzero order, retaining negative effects')
    assert len(calls) == 4 and t.equal(calls[0], circuit)
    for graph, index in zip(calls[1:], [(0, 0), (1, 0), (1, 1)]):
        expected = circuit.clone()
        expected[index] = False
        assert t.equal(graph, expected), 'Remove one edge at a time from the original circuit'
    t.testing.assert_close(circuit, before)
    assert edge_drops(t.zeros_like(circuit), faith).numel() == 0
    print('All tests in `test_edge_drops` passed!')
