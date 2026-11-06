import importlib.util
import time

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        data_shapley_report,
        exact_data_shapley_values,
        in_run_data_shapley_report,
        in_run_first_order_data_scores,
        monte_carlo_data_shapley_report,
        one_step_linear_utility,
        toy_data_shapley_problem,
    )
    # The notebook setup intentionally selects float64 and one thread. Importing
    # it during pytest collection must not impose those settings on other tests.
    _caller_dtype, _caller_threads = t.get_default_dtype(), t.get_num_threads()
    try:
        from chapter16_shapley_attribution_baselines.exercises.part9_data_shapley import (
            solutions as data_shapley_solutions,
        )
    finally:
        t.set_default_dtype(_caller_dtype)
        t.set_num_threads(_caller_threads)


@pytest.fixture(autouse=True)
def _lesson_precision():
    if not TORCH_AVAILABLE:
        yield
        return
    dtype, threads = t.get_default_dtype(), t.get_num_threads()
    try:
        t.set_default_dtype(t.float64)
        t.set_num_threads(1)
        yield
    finally:
        t.set_default_dtype(dtype)
        t.set_num_threads(threads)


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
    # Keep the legacy seed and four-point linear game; only the estimator API changes.
    gen = t.Generator().manual_seed(13)
    problem = (
        t.randn(4, 3, generator=gen, dtype=t.float64),
        t.randn(4, generator=gen, dtype=t.float64),
        t.randn(2, 3, generator=gen, dtype=t.float64),
        t.randn(2, generator=gen, dtype=t.float64),
    )
    signal = exact_data_shapley_values(*toy_data_shapley_problem())
    exact, _ = data_shapley_solutions.exact_data_shapley(
        lambda coalition: one_step_linear_utility(*problem, coalition), 4
    )
    first = data_shapley_solutions.first_order_credit(
        in_run_first_order_data_scores(*problem)[:, None], lr=1.0, denominator=1
    )
    assert exact.argmin() != signal.argmin()
    assert exact.argmax() != signal.argmax()
    assert max(abs(t.corrcoef(t.stack([signal, value]))[0, 1]) for value in [exact, first]) <= 0.25


def test_label_shuffled_attribution_failure_control_moves_harmful_example():
    x, y, vx, vy = toy_data_shapley_problem()
    problem = x, y[t.tensor([0, 3, 2, 1])], vx, vy
    signal = exact_data_shapley_values(x, y, vx, vy)
    exact, _ = data_shapley_solutions.exact_data_shapley(
        lambda coalition: one_step_linear_utility(*problem, coalition), 4
    )
    first = data_shapley_solutions.first_order_credit(
        in_run_first_order_data_scores(*problem)[:, None], lr=1.0, denominator=1
    )
    for values in [exact, first]:
        assert values.argmin() != signal.argmin()
        assert t.corrcoef(t.stack([signal, values]))[0, 1] <= 0.0


def test_runtime_overhead_metrics_are_reported():
    problem = toy_data_shapley_problem()
    utility = lambda coalition: one_step_linear_utility(*problem, coalition)
    # Positive elapsed times remain a smoke check, never a speedup assertion.
    routes = [
        lambda: utility(frozenset(range(4))),
        lambda: data_shapley_solutions.exact_data_shapley(utility, 4),
        lambda: data_shapley_solutions.first_order_credit(
            in_run_first_order_data_scores(*problem)[:, None], lr=1.0, denominator=1
        ),
    ]
    for route in routes:
        route()  # untimed warmup
        start = time.perf_counter()
        for _ in range(8):
            route()
        assert (time.perf_counter() - start) / 8 > 0.0
