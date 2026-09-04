import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        activation_patching_effects,
        additive_game,
        conjunction_game,
        interaction_patching_failure_report,
        shapley_patching_comparison_report,
    )


def test_activation_patching_effects_match_shapley_on_additive_games():
    values = additive_game(t.tensor([1.0, 2.0, 0.5]))

    patching = activation_patching_effects(values, num_players=3)
    report = shapley_patching_comparison_report(values, num_players=3)

    t.testing.assert_close(patching, t.tensor([1.0, 2.0, 0.5], dtype=t.float64))
    assert report.agrees_with_shapley
    assert report.top_feature_agrees
    assert report.shapley_top_feature == 1


def test_activation_patching_overcounts_pure_interactions():
    values = conjunction_game(2)

    report = interaction_patching_failure_report(values, num_players=2)

    t.testing.assert_close(report.shapley_values, t.full((2,), 0.5, dtype=t.float64))
    t.testing.assert_close(report.patching_effects, t.ones(2, dtype=t.float64))
    assert report.shapley_total == pytest.approx(1.0)
    assert report.patching_total == pytest.approx(2.0)
    assert report.overcount == pytest.approx(1.0)
    assert report.documents_overcount
