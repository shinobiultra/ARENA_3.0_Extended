import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        additive_game,
        coalition_values_from_function,
        conjunction_game,
        exact_shapley_values,
        kernelshap_approximation_report,
        kernelshap_kernel_weight,
        kernelshap_values,
        partition_shap_report,
        partition_shap_values,
    )


def test_kernelshap_full_table_matches_exact_shapley_for_additive_game():
    values = additive_game(t.tensor([1.0, -2.0, 0.5]))

    kernel_values = kernelshap_values(values, num_players=3)
    exact = exact_shapley_values(values, num_players=3)
    report = kernelshap_approximation_report(values, num_players=3)

    t.testing.assert_close(kernel_values, exact)
    assert report.approximates_exact


def test_kernelshap_full_table_matches_exact_shapley_for_interaction_game():
    values = conjunction_game(3)

    report = kernelshap_approximation_report(values, num_players=3)

    t.testing.assert_close(report.shapley_values, t.full((3,), 1 / 3, dtype=t.float64))
    assert report.max_abs_error < 1e-9


def test_kernel_weight_rejects_empty_and_full_coalitions():
    assert kernelshap_kernel_weight(1, 3) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="finite"):
        kernelshap_kernel_weight(0, 3)
    with pytest.raises(ValueError, match="finite"):
        kernelshap_kernel_weight(3, 3)


def test_partition_shap_recovers_exact_values_for_additive_grouped_game():
    values = additive_game(t.tensor([1.0, 2.0, 3.0, 4.0]))
    groups = ((0, 1), (2, 3))

    partition_values = partition_shap_values(values, groups=groups)
    report = partition_shap_report(values, groups=groups)

    t.testing.assert_close(partition_values, t.tensor([1.0, 2.0, 3.0, 4.0], dtype=t.float64))
    t.testing.assert_close(report.group_values, t.tensor([3.0, 7.0], dtype=t.float64))
    assert report.recovers_exact


def test_partition_shap_splits_credit_inside_pure_interaction_group():
    values = conjunction_game(2)
    groups = ((0, 1),)

    report = partition_shap_report(values, groups=groups)

    t.testing.assert_close(report.player_values, t.full((2,), 0.5, dtype=t.float64))
    assert report.recovers_exact


def test_partition_shap_handles_cross_group_interaction_without_irrelevant_credit():
    values = coalition_values_from_function(
        3,
        lambda coalition: {0, 2}.issubset(coalition),
    )
    groups = ((0, 1), (2,))

    report = partition_shap_report(values, groups=groups)

    t.testing.assert_close(
        report.player_values,
        t.tensor([0.5, 0.0, 0.5], dtype=t.float64),
    )
    assert report.recovers_exact
