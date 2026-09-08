import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SHAPIQ_AVAILABLE = importlib.util.find_spec("shapiq") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        additive_game,
        interaction_game,
        pairwise_interaction_report,
        pairwise_shapley_interactions,
        shapiq_interaction_parity_report,
    )


def test_pairwise_interactions_vanish_for_additive_games():
    values = additive_game(t.tensor([1.0, -2.0, 0.5]))

    interactions = pairwise_shapley_interactions(values, num_players=3)

    t.testing.assert_close(interactions, t.zeros((3, 3), dtype=t.float64))


def test_pairwise_interaction_report_recovers_target_pair_only():
    values = interaction_game(
        3,
        pair=(0, 1),
        pair_weight=1.0,
        additive_weights=t.tensor([0.5, -1.0, 2.0]),
    )

    report = pairwise_interaction_report(values, num_players=3, target_pair=(0, 1))

    assert report.recovers_interaction
    assert report.target_interaction == pytest.approx(1.0)
    assert report.max_spurious_interaction < 1e-9
    t.testing.assert_close(report.pair_interactions, report.pair_interactions.T)
    t.testing.assert_close(t.diag(report.pair_interactions), t.zeros(3, dtype=t.float64))


@pytest.mark.skipif(not SHAPIQ_AVAILABLE, reason="shapiq is not installed")
def test_shapiq_pairwise_interactions_match_exact_sii_on_complete_table():
    values = interaction_game(
        3,
        pair=(0, 1),
        pair_weight=1.0,
        additive_weights=t.tensor([0.5, -1.0, 2.0]),
    )

    report = shapiq_interaction_parity_report(values, num_players=3, index="SII")

    assert report.shapiq_available
    assert report.matches_shapiq
    assert report.max_abs_error < 1e-5
    assert report.shapiq_pair_interactions[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_pairwise_interactions_require_at_least_two_players():
    with pytest.raises(ValueError, match="at least two"):
        pairwise_shapley_interactions({frozenset(): 0.0}, num_players=1)
