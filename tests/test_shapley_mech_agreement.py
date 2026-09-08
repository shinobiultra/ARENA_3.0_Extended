import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        additive_game,
        attribution_agreement_report,
        interaction_agreement_report,
        spearman_rank_correlation,
        topk_overlap_fraction,
        xor_game,
    )


def test_attribution_agreement_report_matches_additive_mechanistic_ground_truth():
    values = additive_game(t.tensor([1.0, 2.0, 0.5]))

    report = attribution_agreement_report(
        values,
        mechanistic_scores=t.tensor([1.0, 2.0, 0.5]),
        num_players=3,
    )

    t.testing.assert_close(report.shapley_values, t.tensor([1.0, 2.0, 0.5], dtype=t.float64))
    t.testing.assert_close(report.patching_effects, report.shapley_values)
    assert report.spearman_correlation == pytest.approx(1.0)
    assert report.topk_overlap == pytest.approx(1.0)
    assert report.deletion_drop > report.random_baseline_drop
    assert report.agrees_with_mechanistic


def test_interaction_agreement_report_recovers_xor_pair_when_shapley_misses():
    report = interaction_agreement_report(xor_game())

    t.testing.assert_close(report.shapley_values, t.zeros(2, dtype=t.float64))
    assert report.recovered_pair_interaction == pytest.approx(2.0)
    assert report.ordinary_shapley_misses
    assert report.interaction_recovers_pair


def test_rank_and_topk_metrics_are_deterministic():
    first = t.tensor([1.0, 3.0, 2.0])
    second = t.tensor([0.1, 0.3, 0.2])

    assert spearman_rank_correlation(first, second) == pytest.approx(1.0)
    assert topk_overlap_fraction(first, second, k=2) == pytest.approx(1.0)
