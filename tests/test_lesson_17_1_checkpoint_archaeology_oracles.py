"""17.1 Checkpoint archaeology: tiny finite-table, Fourier and emergence oracles on the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = 'chapter17_training_dynamics.exercises.part1_checkpoint_archaeology'


@pytest.fixture(scope='module')
def methods():
    return importlib.import_module(MODULE + '.solutions')


def test_tiny_finite_table_and_exact_fourier_oracles(methods):
    import torch as t

    p = 5
    pairs, labels = methods.make_modular_addition_dataset(modulus=p)
    assert pairs.tolist() == [[a, b] for a in range(p) for b in range(p)]
    assert labels.tolist() == [(a + b) % p for a in range(p) for b in range(p)]
    logits = methods.exact_addition_logits(modulus=p)
    legal = methods.addition_fourier_mask(modulus=p)
    expected = t.zeros((p, p, p), dtype=t.bool)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                expected[a, b, c] = c != 0 and a == b == (-c) % p
    assert t.equal(legal, expected)
    assert methods.fourier_power_fraction(logits, legal, modulus=p) == pytest.approx(1.0, abs=1e-6)
    random = methods.random_fourier_mask(modulus=p, seed=7)
    assert int(random.sum()) == int(legal.sum())
    assert not bool((random & legal).any())
    for a, b, c in random.nonzero().tolist():
        assert random[-a % p, -b % p, -c % p]
    assert methods.fourier_power_fraction(logits, random, modulus=p) < 1e-10
    centered = logits - logits.mean(-1, keepdim=True)
    kept = methods.project_logits_to_fourier_mask(logits, legal, modulus=p, keep=True)
    ablated = methods.project_logits_to_fourier_mask(logits, legal, modulus=p, keep=False)
    assert t.allclose(kept, logits, atol=2e-6)
    assert t.allclose(ablated, logits.mean(-1, keepdim=True).expand_as(logits), atol=2e-6)
    assert t.allclose(kept + ablated - logits.mean(-1, keepdim=True), logits, atol=2e-6)
    assert methods.fourier_power_fraction(centered * 3, legal, modulus=p) == pytest.approx(1.0, abs=1e-6)
    assert methods.fourier_power_fraction(t.ones_like(logits), legal, modulus=p) == 0
    with pytest.raises(ValueError, match='shape'):
        methods.fourier_power_fraction(t.zeros((3, 3)), legal, modulus=p)


def test_stable_crossing_and_matched_control_oracles(methods):
    import torch as t

    steps = t.tensor([0, 2, 4, 8, 16])
    values = t.tensor([0.1, 0.6, 0.2, 0.7, 0.8])
    assert methods.first_threshold_crossing(steps, values, threshold=0.5) == 2
    assert methods.stable_threshold_step(steps, values, threshold=0.5, min_consecutive=2) == 8
    assert methods.monotonicity_violations(values) == 1
    report = methods.mechanism_emergence_report(steps, values, metric_name='fixture', threshold=0.5, min_consecutive=2)
    assert report.first_crossing_step == 2 and report.stable_from_step == 8
    assert report.emerged
    assert methods.random_control_report(t.tensor([0.01, 0.02]), metric_name='fixture', max_allowed_value=0.05).control_passed
