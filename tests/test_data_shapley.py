import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        data_shapley_report,
        exact_data_shapley_values,
        in_run_data_shapley_report,
        monte_carlo_data_shapley_report,
        one_step_linear_utility,
        toy_data_shapley_problem,
    )
    from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run import (
        solutions as data_shapley_solutions,
    )


def test_exact_data_shapley_identifies_helpful_and_harmful_examples():
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()

    exact = exact_data_shapley_values(train_x, train_y, val_x, val_y)
    report = data_shapley_report(train_x, train_y, val_x, val_y)

    t.testing.assert_close(
        exact,
        t.tensor(
            [0.6412037037037037, 0.6412037037037037, 0.6412037037037037, -1.173611111111111],
            dtype=t.float64,
        ),
    )
    assert report.harmful_index == 3
    assert report.harmful_value < 0.0
    assert report.deletion_test_passes
    assert report.addition_test_passes


def test_monte_carlo_data_shapley_converges_toward_exact_values():
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()

    report = monte_carlo_data_shapley_report(
        train_x,
        train_y,
        val_x,
        val_y,
        num_samples=512,
        seed=0,
    )

    assert report.approximates_exact
    assert report.harmful_example_matches
    assert report.top_example_matches
    assert report.max_abs_error < 0.08


def test_in_run_first_order_scores_correlate_with_exact_data_shapley():
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()

    report = in_run_data_shapley_report(train_x, train_y, val_x, val_y)

    assert report.correlates_with_exact
    assert report.identifies_harmful
    assert report.identifies_helpful
    assert report.pearson_correlation == pytest.approx(1.0)


def test_one_step_utility_requires_selected_data_for_improvement():
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()

    assert one_step_linear_utility(train_x, train_y, val_x, val_y, frozenset()) == 0.0
    assert one_step_linear_utility(train_x, train_y, val_x, val_y, frozenset({0})) == 1.0
    assert one_step_linear_utility(train_x, train_y, val_x, val_y, frozenset({3})) < 0.0


def test_random_data_attribution_failure_control_breaks_planted_signal():
    report = data_shapley_solutions.random_data_attribution_failure_smoke_test()

    assert report["random_data_attribution_fails"]
    assert report["harmful_index"] != report["original_harmful_index"]
    assert report["helpful_index"] != report["original_helpful_index"]
    assert report["max_abs_signal_correlation"] <= 0.25


def test_label_shuffled_attribution_failure_control_moves_harmful_example():
    report = data_shapley_solutions.label_shuffled_attribution_failure_smoke_test()

    assert report["label_shuffled_attribution_fails"]
    assert report["harmful_index"] != report["original_harmful_index"]
    assert report["in_run_harmful_index"] != report["original_harmful_index"]
    assert report["signal_correlation"] <= 0.0
    assert report["in_run_signal_correlation"] <= 0.0


def test_runtime_overhead_metrics_are_reported():
    report = data_shapley_solutions.runtime_overhead_smoke_test(repeats=8)

    assert report["runtime_overhead_reported"]
    assert report["runtime_measurement_repeats"] == 8
    assert report["runtime_full_update_seconds"] > 0.0
    assert report["runtime_exact_enumeration_seconds"] > 0.0
    assert report["runtime_in_run_scores_seconds"] > 0.0
