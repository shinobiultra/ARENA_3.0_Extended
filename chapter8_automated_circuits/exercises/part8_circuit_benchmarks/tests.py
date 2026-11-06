from typing import Callable

import torch as t


def test_select_aggregate(compile_select_aggregate: Callable, make_model: Callable, subspaces: dict):
    model = make_model()
    tok, pos = subspaces["tok"], subspaces["pos"]
    n_tok, n_pos = tok.stop - tok.start, pos.stop - pos.start
    # predicate: query position i attends to key position j iff j == i - 1
    predicate = t.zeros(n_pos, n_pos)
    for i in range(1, n_pos):
        predicate[i, i - 1] = 1
    compile_select_aggregate(model, layer=0, head=0, query_space=pos, key_space=pos, predicate=predicate,
                             value_space=tok, out_space=subspaces["prev"], sharpness=20.0)
    W_Q, W_K = model.W_Q[0, 0], model.W_K[0, 0]
    scores = (t.eye(model.cfg.d_model)[pos] @ W_Q) @ (t.eye(model.cfg.d_model)[pos] @ W_K).T / model.cfg.d_head**0.5
    t.testing.assert_close(scores.cpu(), 20.0 * predicate, atol=1e-4, rtol=1e-4,
                           msg="Attention scores between one-hot query and key values should be sharpness * predicate[query, key]")
    ov = model.W_V[0, 0] @ model.W_O[0, 0]
    expected = t.zeros(model.cfg.d_model, model.cfg.d_model)
    expected[tok, subspaces["prev"]] = t.eye(n_tok)
    t.testing.assert_close(ov.cpu(), expected, atol=1e-5, rtol=1e-5,
                           msg="The OV circuit should copy value_space to out_space (identity) and nothing else")
    for other in range(1, model.cfg.n_heads):
        assert model.W_Q[0, other].abs().sum() == 0, "Only the given head should be written"
    print("All tests in `test_select_aggregate` passed!")


def test_roc_auc(roc_auc: Callable):
    edge_mask = t.ones(1, 6, dtype=t.bool)
    gt = t.tensor([[1, 1, 0, 0, 0, 0]], dtype=t.bool)
    assert abs(roc_auc(t.tensor([[6.0, 5, 4, 3, 2, 1]]), gt, edge_mask) - 1.0) < 1e-6, "A perfect ranking has AUC 1"
    assert abs(roc_auc(t.tensor([[1.0, 2, 3, 4, 5, 6]]), gt, edge_mask) - 0.0) < 1e-6, "A reversed ranking has AUC 0"
    # one positive ranked first, one ranked third: of the 8 (positive, negative) pairs, 7 are ordered correctly
    assert abs(roc_auc(t.tensor([[6.0, 4, 5, 3, 2, 1]]), gt, edge_mask) - 7 / 8) < 1e-6
    assert abs(roc_auc(t.tensor([[-6.0, 5, 4, 3, 2, 1]]), gt, edge_mask) - 1.0) < 1e-6, "Rank by absolute value"
    # ties count half (here every score is equal)
    assert abs(roc_auc(t.zeros(1, 6), gt, edge_mask) - 0.5) < 1e-6, "Ties between a positive and a negative count as half"
    print("All tests in `test_roc_auc` passed!")


class _AnalyticEdgeGraph:
    """CPU-only scoring oracle, independent of compiled models and learner methods.

    Four causal edges have unequal signed effects and two distractor edges have
    zero effects. Invalid entries must also stay zero. Quadratics and a two-edge
    interaction distinguish EAP, integrated gradients and full-minus-one effects.
    """

    edge_mask = t.tensor([[True, True, True], [True, True, True], [False, False, False]])
    linear = t.tensor([[0.2, -0.4, 0.0], [0.0, 1.1, -0.3], [0.0, 0.0, 0.0]])
    quadratic = t.tensor([[0.3, 0.1, 0.0], [0.0, 0.2, -0.1], [0.0, 0.0, 0.0]])

    def sender_outputs(self, tokens):
        return t.zeros(1)

    def run(self, tokens, corrupt_out, keep):
        keep = keep.float() * self.edge_mask
        independent = (self.linear * keep + self.quadratic * keep.square()).sum()
        interaction = 0.6 * keep[0, 0] * keep[0, 1]
        return (independent + interaction).reshape(1)


def _test_analytic_edge_methods(patching_scores, eap_scores, eap_ig_scores):
    """Check each edge, including distractors, without calling another learner scorer."""
    graph = _AnalyticEdgeGraph()
    tokens = t.zeros((1, 1), dtype=t.long)
    corrupt_out = graph.sender_outputs(tokens)
    metric = lambda logits: logits.sum()
    # f(k)=sum(a*k+b*k^2)+.6*k00*k01. Each interacting edge's
    # removal/endpoint gradient gets .6, whereas its path integral gets .3.
    interaction = t.zeros_like(graph.linear)
    interaction[0, 0] = interaction[0, 1] = 0.6
    exact_expected = graph.linear + graph.quadratic + interaction
    exact = patching_scores(graph, tokens, corrupt_out, metric)
    eap = eap_scores(graph, tokens, corrupt_out, metric)
    ig = eap_ig_scores(graph, tokens, corrupt_out, metric, steps=8)
    for name, actual, expected in (
        ("exact full-minus-one", exact, exact_expected),
        ("endpoint EAP", eap, graph.linear + 2 * graph.quadratic + interaction),
        ("edgewise IG", ig, graph.linear + graph.quadratic + interaction / 2),
    ):
        assert actual.shape == graph.edge_mask.shape, f"{name}: wrong score shape"
        assert t.isfinite(actual).all(), f"{name}: non-finite scores"
        t.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5,
                               msg=f"{name} must match every causal/distractor edge of the analytic graph")


def test_edge_methods(patching_scores: Callable, eap_scores: Callable, eap_ig_scores: Callable, graph, clean, corrupt, metric):
    """Check independent causal effects, endpoint derivatives and path integrals."""
    _test_analytic_edge_methods(patching_scores, eap_scores, eap_ig_scores)
    corrupt_out = graph.sender_outputs(corrupt)
    exact = patching_scores(graph, clean, corrupt_out, metric)
    assert exact.shape == graph.edge_mask.shape
    assert t.isfinite(exact).all(), "Exact effects must be finite"
    assert (exact[~graph.edge_mask] == 0).all(), "Invalid edges should score 0"
    # Recompute full-minus-one forwards independently of every scoring method.
    # Sample across the whole supplied graph; the analytic oracle above always
    # exercises all four causal and both zero-effect distractor edges.
    valid = graph.edge_mask.nonzero()
    assert len(valid) > 0, "The scoring fixture needs valid edges"
    selected = t.linspace(0, len(valid) - 1, steps=min(6, len(valid))).long().unique()
    with t.no_grad():
        full_keep = graph.edge_mask.float()
        full_metric = metric(graph.run(clean, corrupt_out, full_keep)).item()
        for index in selected.tolist():
            source, target = valid[index].tolist()
            removed = full_keep.clone()
            removed[source, target] = 0
            expected = full_metric - metric(graph.run(clean, corrupt_out, removed)).item()
            t.testing.assert_close(exact[source, target].cpu(), t.tensor(expected), atol=1e-5, rtol=1e-4,
                                   msg=f"Exact score at {(source, target)} must be full minus that edge removed")
    eap = eap_scores(graph, clean, corrupt_out, metric)
    # EAP is the derivative of the metric with respect to the edge's mask: compare with a finite difference
    u, v = valid[min(5, len(valid) - 1)].tolist()
    with t.no_grad():
        keep = graph.edge_mask.float()
        base = metric(graph.run(clean, corrupt_out, keep)).item()
        keep[u, v] = 1 - 1e-3
        stepped = metric(graph.run(clean, corrupt_out, keep)).item()
    fd = (base - stepped) / 1e-3
    assert abs(eap[u, v].item() - fd) < 1e-2 * max(1.0, abs(fd)), f"EAP for edge {(u, v)} should be d metric / d mask = {fd:.4f}, got {eap[u, v]:.4f}"
    ig = eap_ig_scores(graph, clean, corrupt_out, metric, steps=8)
    # completeness of integrated gradients: the scores sum to metric(full) - metric(empty)
    with t.no_grad():
        total = metric(graph.run(clean, corrupt_out, graph.edge_mask.float())).item() - metric(graph.run(clean, corrupt_out, t.zeros_like(graph.edge_mask).float())).item()
    assert abs(ig.sum().item() - total) < 0.1 * max(1.0, abs(total)), (
        f"Integrated-gradient scores should approximately sum to metric(full) - metric(empty) = {total:.3f}; got {ig.sum().item():.3f}"
    )
    print("All tests in `test_edge_methods` passed!")
