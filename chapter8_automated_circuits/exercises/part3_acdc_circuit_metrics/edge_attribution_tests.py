"""Gradient oracles and state-cleanup checks for student EAP/EAP-IG functions."""
from __future__ import annotations

import pytest
import torch as t

from . import edge_attribution as reference
from .edge_tests import example_inputs, small_model
from .real_edges import EdgeIntervention, answer_logit_differences, build_residual_graph, output_reachable_sources, clean_kl_divergence


def test_gradient_contraction(fn=None):
    fn = fn or reference.contract_edge_gradients
    delta = t.arange(1, 25, dtype=t.float64).reshape(2, 2, 3, 2)
    gradient = t.tensor([[[[1., -2.], [3., 0.]], [[-1., 2.], [1., 4.]]],
                         [[[0., 1.], [-2., 3.]], [[4., -3.], [2., 1.]]]], dtype=t.float64)
    expected = t.zeros(3, 2, dtype=t.float64)
    for sender in range(3):
        for head in range(2):
            expected[sender, head] = sum(
                float(delta[b, p, sender, d]) * float(gradient[b, p, head, d])
                for b in range(2) for p in range(2) for d in range(2)
            )
    t.testing.assert_close(fn(delta, gradient), expected)
    t.testing.assert_close(fn(delta, gradient[:, :, 0]), expected[:, :1])
    with pytest.raises(ValueError):
        fn(delta, gradient[:1])


def test_clean_kl_has_zero_first_derivative(fn=None):
    fn = fn or clean_kl_divergence
    clean = t.tensor([[1., 2., -1.], [-2., .5, 3.]], dtype=t.float64)
    logits = clean.clone().requires_grad_(True)
    loss = fn(logits, clean)
    gradient = t.autograd.grad(loss, logits)[0]
    assert abs(float(loss)) < 1e-12
    assert float(gradient.abs().max()) < 1e-12, "Vanilla EAP with KL at clean has no first-order signal."
    assert float(fn(clean.roll(1, 1), clean)) > .1, "A finite corruption can still have substantial KL."


def test_eap_matches_infinitesimal_edge_deletion(fn=None):
    fn = fn or reference.edge_attribution_scores
    model = small_model().to(t.float64)
    clean, corrupt, ends = example_inputs()
    answers = t.tensor([[0, 1], [2, 3], [4, 5]])
    def loss(logits):
        return -answer_logit_differences(logits, answers).mean()
    scores = fn(model, clean, corrupt, ends, loss, method="eap")
    engine = EdgeIntervention(model, clean, corrupt, ends)
    full = engine.graph.legal_mask.double()
    baseline = float(loss(engine(full)))
    epsilon = 1e-5
    lookup = {receiver.name: i for i, receiver in enumerate(engine.graph.receivers)}
    for source, receiver_name in ((0, "L0H0.q"), (0, "L0H1.v"), (1, "L1H1.k"),
                                  (2, "L1H0.v"), (3, "L1MLP.in"), (5, "output")):
        receiver = lookup[receiver_name]
        mask = full.clone()
        mask[source, receiver] -= epsilon
        finite_difference = (float(loss(engine(mask))) - baseline) / epsilon
        assert float(scores[source, receiver]) == pytest.approx(finite_difference, abs=3e-5, rel=2e-3)
    assert not bool(scores[~engine.graph.legal_mask].any())


def test_attribution_batch_mean_and_cleanup(fn=None):
    fn = fn or reference.edge_attribution_scores
    model = small_model()
    clean, corrupt, ends = example_inputs()
    answers = t.tensor([[0, 1], [2, 3], [4, 5]])
    def loss(logits):
        return -answer_logit_differences(logits, answers).mean()
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    flags = [p.requires_grad for p in parameters]
    parameters[1].grad = t.ones_like(parameters[1])
    sentinel = parameters[1].grad.clone()
    model.train()
    batch = fn(model, clean, corrupt, ends, loss, steps=3)
    individual = []
    for row in range(3):
        def one_loss(logits, row=row):
            return -answer_logit_differences(logits, answers[row:row + 1]).mean()
        individual.append(fn(model, clean[row:row + 1], corrupt[row:row + 1], ends[row:row + 1], one_loss, steps=3))
    t.testing.assert_close(batch, t.stack(individual).mean(0), atol=3e-6, rtol=3e-5)
    assert model.training
    assert flags == [p.requires_grad for p in parameters]
    t.testing.assert_close(parameters[1].grad, sentinel)
    assert not any(hook.fwd_hooks or hook.bwd_hooks for hook in model.hook_dict.values())

    def failing_loss(logits):
        raise RuntimeError("intentional loss failure")
    with pytest.raises(RuntimeError, match="intentional loss failure"):
        fn(model, clean, corrupt, ends, failing_loss, steps=2)
    assert flags == [p.requires_grad for p in parameters] and model.training
    assert not any(hook.fwd_hooks or hook.bwd_hooks for hook in model.hook_dict.values())
    t.testing.assert_close(parameters[1].grad, sentinel)


def test_top_edges_use_absolute_legal_scores(fn=None):
    fn = fn or reference.top_edge_mask
    graph = build_residual_graph(1, 1)
    scores = t.zeros(graph.legal_mask.shape)
    scores[~graph.legal_mask] = 1000.
    scores[0, 0], scores[1, 4], scores[2, 4] = -10., 3., 3.
    mask = fn(graph, scores, 2)
    assert mask.nonzero().tolist() == [[0, 0], [1, 4]], "Keep inhibitory scores by magnitude; break ties in graph order."
    assert not bool(fn(graph, scores, 0).any())
    t.testing.assert_close(fn(graph, scores, int(graph.legal_mask.sum())), graph.legal_mask)
    with pytest.raises(ValueError):
        fn(graph, scores, 9)


def test_greedy_scores_follow_the_output_frontier(fn=None):
    fn = fn or reference.greedy_score_mask
    graph = build_residual_graph(1, 1)
    scores = t.zeros(graph.legal_mask.shape)
    scores[~graph.legal_mask] = 10000.
    scores[0, 0], scores[1, 4], scores[2, 4], scores[0, 3] = -100., 10., 9., 1000.
    selected = fn(graph, scores, 2)
    assert selected.nonzero().tolist() == [[0, 0], [1, 4]], "The highest global edge is unavailable until its receiver joins the output frontier."
    assert fn(graph, scores, 4).nonzero().tolist() == [[0, 0], [0, 3], [1, 4], [2, 4]]
    for count in range(9):
        mask = fn(graph, scores, count)
        assert int(mask.sum()) == count
        assert not bool((mask & ~graph.legal_mask).any())
        assert set(mask.nonzero()[:, 0].tolist()) <= output_reachable_sources(graph, mask)
    t.testing.assert_close(fn(graph, scores, 8), graph.legal_mask)
    with pytest.raises(AssertionError):
        test_greedy_scores_follow_the_output_frontier(reference.top_edge_mask)
