"""Numerical task, random-circuit and backup-mechanism tests for the learner."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch as t

from . import edge_metrics as reference
from .real_edges import build_residual_graph


def test_task_recovery(fn=None):
    fn = fn or reference.circuit_behavior
    answers = t.tensor([[0, 1], [1, 0]])
    clean = t.tensor([[4., 0.], [0., 2.]])
    corrupt = t.tensor([[0., 0.], [0., 1.]])
    candidate = t.tensor([[2., 0.], [0., 0.]])
    result = fn(candidate, clean, corrupt, answers)
    assert result["mean_logit_difference"] == pytest.approx(1.)
    assert result["recovery"] == pytest.approx(.2), "Use the ratio of means; averaging per-example ratios gives -.25."
    assert result["pairwise_accuracy"] == pytest.approx(.5), "A tied pair is not a correct answer."
    assert result["per_example_logit_difference"] == [2., 0.]
    assert fn(clean, clean, corrupt, answers)["recovery"] == pytest.approx(1.)
    assert fn(corrupt, clean, corrupt, answers)["recovery"] == pytest.approx(0.)
    with pytest.raises(ValueError):
        fn(candidate, clean, clean, answers)


def test_same_size_controls(fn=None):
    fn = fn or reference.same_size_random_masks
    graph = build_residual_graph(2, 2)
    selected = t.zeros_like(graph.legal_mask)
    selected[0, 0] = selected[0, 4] = selected[1, 6] = selected[3, 14] = True
    before = selected.clone()
    for match_receivers in (False, True):
        first = fn(graph, selected, count=12, seed=39, match_receivers=match_receivers)
        second = fn(graph, selected, count=12, seed=39, match_receivers=match_receivers)
        assert len(first) == 12
        for a, b in zip(first, second, strict=True):
            t.testing.assert_close(a, b)
            assert int(a.sum()) == 4, "Random circuits must have exactly the selected edge count."
            assert not bool((a & ~graph.legal_mask).any())
            if match_receivers:
                t.testing.assert_close(a.sum(0), selected.sum(0))
        assert len({str(mask.nonzero().tolist()) for mask in first}) > 1
    t.testing.assert_close(before, selected)
    fractional = selected.float() * .5
    with pytest.raises(ValueError, match="binary"):
        fn(graph, fractional)
    # The null should accept fully disconnected selections, not resample them.
    for mask in fn(graph, t.zeros_like(selected), count=2):
        assert not bool(mask.any())


class BackupOracle:
    """Two nonlinear paths: either suffices, but the backup is weaker."""
    def __init__(self):
        self.graph = build_residual_graph(1, 1)
        self.calls = []
        self.clean_logits = t.tensor([[2., 0.]], dtype=t.float64)

    def __call__(self, mask):
        self.calls.append(mask.clone())
        primary = bool(mask[0, 0] and mask[1, 4])
        backup = bool(mask[0, 3] and mask[2, 4])
        score = 2. if primary else (1. if backup else 0.)
        return t.tensor([[score, 0.]], dtype=t.float64)


def primary_path(graph):
    circuit = t.zeros_like(graph.legal_mask)
    circuit[0, 0] = circuit[1, 4] = True
    return circuit


def test_edge_minimality_recomputes(fn=None):
    fn = fn or reference.minimality_diagnostics
    engine = BackupOracle()
    circuit = primary_path(engine.graph)
    before = circuit.clone()
    rows = fn(engine, circuit, t.tensor([[0, 1]]))
    assert len(rows) == 2 and len(engine.calls) == 3
    probability = math.exp(2) / (1 + math.exp(2))
    exact_kl = probability * math.log(probability / .5) + (1 - probability) * math.log((1 - probability) / .5)
    for row in rows:
        assert row["logit_difference_damage"] == pytest.approx(2.)
        assert row["kl_damage"] == pytest.approx(exact_kl)
        assert circuit[row["source"], row["receiver"]]
    t.testing.assert_close(circuit, before)


def test_completeness_exposes_backup(fn=None):
    fn = fn or reference.sampled_completeness_diagnostics
    engine = BackupOracle()
    circuit = primary_path(engine.graph)
    before = circuit.clone()
    rows = fn(engine, circuit, t.tensor([[0, 1]]), subset_sizes=(1, 2, 4), samples_per_size=5, seed=18)
    assert len(rows) == 10, "Skip subsets larger than the circuit."
    for index, row in enumerate(rows):
        assert row["circuit_minus_subset"] == 0.
        assert row["full_minus_subset"] == 1.
        assert row["omitted_backup_gap"] == 1., "The omitted path is invisible in an intact-circuit sufficiency test."
        sparse, full = engine.calls[index * 2:index * 2 + 2]
        removed = t.tensor(row["removed_edges"])
        assert len(removed) == row["subset_size"]
        assert bool(circuit[removed[:, 0], removed[:, 1]].all()), "K must be a subset of retained edges."
        assert not bool(sparse[removed[:, 0], removed[:, 1]].any())
        assert not bool(full[removed[:, 0], removed[:, 1]].any()), "Ablate exactly the same K in the full graph."
        assert bool(full[0, 3] and full[2, 4]), "Keep omitted backup edges in the full comparison."
    t.testing.assert_close(circuit, before)


def test_metrics_reject_plausible_wrong_answers():
    with pytest.raises(AssertionError):
        test_task_recovery(lambda *args: {"mean_logit_difference": 1., "recovery": -.25})
    with pytest.raises(AssertionError):
        test_completeness_exposes_backup(lambda *args, **kwargs: [])


def test_diagnostics_accept_explicit_task_units(minimality_fn=None, completeness_fn=None):
    minimality_fn = minimality_fn or reference.minimality_diagnostics
    completeness_fn = completeness_fn or reference.sampled_completeness_diagnostics
    engine = BackupOracle()
    circuit = primary_path(engine.graph)
    def metric(logits):
        return logits[:, 0] * .1
    rows = minimality_fn(engine, circuit, metric)
    assert all(row["task_damage"] == pytest.approx(.2) for row in rows)
    assert all("logit_difference_damage" not in row for row in rows), "Do not label another task's units as IOI logits."
    completeness = completeness_fn(engine, circuit, metric, subset_sizes=(1,), samples_per_size=3)
    assert all(row["omitted_backup_gap"] == pytest.approx(.1) for row in completeness)
