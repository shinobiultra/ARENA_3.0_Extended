import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        additive_game,
        conjunction_game,
        exact_shapley_values,
        interaction_gap_report,
        leave_one_out_values,
        permutation_parity_report,
        permutation_shapley_values,
        shapley_efficiency_report,
    )


def test_exact_shapley_values_match_additive_ground_truth():
    weights = t.tensor([1.0, 2.0, -0.5])
    values = additive_game(weights)

    shapley = exact_shapley_values(values, num_players=3)

    t.testing.assert_close(shapley, weights.double())


def test_conjunction_game_splits_credit_equally_and_satisfies_efficiency():
    values = conjunction_game(3)

    shapley = exact_shapley_values(values, num_players=3)
    report = shapley_efficiency_report(values, num_players=3)

    t.testing.assert_close(shapley, t.full((3,), 1 / 3, dtype=t.float64))
    assert report.total_value_delta == 1.0
    assert report.satisfies_efficiency


def test_permutation_shapley_matches_closed_form_exact_values():
    values = conjunction_game(3)

    exact = exact_shapley_values(values, num_players=3)
    permutation = permutation_shapley_values(values, num_players=3)
    report = permutation_parity_report(values, num_players=3)

    t.testing.assert_close(permutation, exact)
    assert report.matches_exact


def test_leave_one_out_overcounts_interaction_heavy_games():
    values = conjunction_game(2)

    leave_one_out = leave_one_out_values(values, num_players=2)
    report = interaction_gap_report(values, num_players=2, min_overcount=0.5)

    t.testing.assert_close(leave_one_out, t.ones(2, dtype=t.float64))
    assert report.shapley_total == pytest.approx(1.0)
    assert report.leave_one_out_total == pytest.approx(2.0)
    assert report.detects_interaction_overcount


def test_exact_shapley_requires_complete_coalition_table():
    with pytest.raises(ValueError, match="missing"):
        exact_shapley_values({frozenset(): 0.0, frozenset({0}): 1.0}, num_players=2)
