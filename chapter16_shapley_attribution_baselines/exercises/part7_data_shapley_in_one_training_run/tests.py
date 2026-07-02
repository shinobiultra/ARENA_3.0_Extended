import json
from collections.abc import Callable
from pathlib import Path

import torch as t


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run import (
        solutions,
    )

    return solutions


def _toy_problem():
    return _solutions().toy_data_shapley_problem()


def test_one_step_linear_utility_toy_oracle(
    one_step_linear_utility: Callable | None = None,
):
    one_step_linear_utility = (
        one_step_linear_utility or _solutions().one_step_linear_utility
    )
    train_x, train_y, val_x, val_y = _toy_problem()
    assert one_step_linear_utility(train_x, train_y, val_x, val_y, frozenset()) == 0.0, (
        "The empty coalition should have zero utility by definition."
    )
    assert one_step_linear_utility(train_x, train_y, val_x, val_y, frozenset({0})) == 1.0, (
        "A single helpful example should improve validation utility by 1.0."
    )
    assert one_step_linear_utility(train_x, train_y, val_x, val_y, frozenset({3})) == -3.0, (
        "The flipped-label example alone should damage validation utility."
    )
    assert one_step_linear_utility(
        train_x,
        train_y,
        val_x,
        val_y,
        frozenset({0, 1, 2, 3}),
    ) == 0.75, "The full coalition should match the one-step full-batch update."
    print("All tests in `test_one_step_linear_utility_toy_oracle` passed!")


def test_data_coalition_values_complete_table(
    data_coalition_values: Callable | None = None,
):
    data_coalition_values = data_coalition_values or _solutions().data_coalition_values
    train_x, train_y, val_x, val_y = _toy_problem()
    values = data_coalition_values(train_x, train_y, val_x, val_y)
    assert len(values) == 16, "Four training examples should produce 2**4 coalitions."
    assert values[frozenset()] == 0.0, "The empty coalition utility should be explicit."
    assert values[frozenset({0, 1, 2})] == 1.0, (
        "The three helpful examples should produce the maximum validation utility."
    )
    assert values[frozenset({0, 1, 2, 3})] == 0.75, (
        "The full coalition should be worse than the helpful-only coalition."
    )
    print("All tests in `test_data_coalition_values_complete_table` passed!")


def test_exact_data_shapley_values_matches_hand_checked_result(
    exact_data_shapley_values: Callable | None = None,
):
    exact_data_shapley_values = (
        exact_data_shapley_values or _solutions().exact_data_shapley_values
    )
    values = exact_data_shapley_values(*_toy_problem())
    expected = t.tensor(
        [0.6412037037037037, 0.6412037037037037, 0.6412037037037037, -1.173611111111111],
        dtype=t.float64,
    )
    assert isinstance(values, t.Tensor), "Exact Data Shapley should return a tensor."
    assert values.dtype == t.float64, "Exact finite-game arithmetic should use float64."
    assert t.allclose(values, expected), (
        "Exact Data Shapley should match the hand-checked four-example oracle."
    )
    print("All tests in `test_exact_data_shapley_values_matches_hand_checked_result` passed!")


def test_sampled_permutation_data_shapley_approximates_exact(
    sampled_permutation_shapley_values: Callable | None = None,
):
    sampled_permutation_shapley_values = (
        sampled_permutation_shapley_values
        or _solutions().sampled_permutation_shapley_values
    )
    train_x, train_y, val_x, val_y = _toy_problem()
    values = _solutions().data_coalition_values(train_x, train_y, val_x, val_y)
    exact = _solutions().exact_data_shapley_values(train_x, train_y, val_x, val_y)
    sampled = sampled_permutation_shapley_values(
        values,
        num_players=4,
        num_samples=512,
        seed=0,
    )
    assert float((sampled - exact).abs().max().item()) < 0.08, (
        "Seeded sampled permutation Data Shapley should approximate exact values."
    )
    assert int(sampled.argmin().item()) == 3, (
        "The sampled estimate should still identify the flipped-label harmful example."
    )
    print("All tests in `test_sampled_permutation_data_shapley_approximates_exact` passed!")


def test_in_run_first_order_scores_toy_oracle(
    in_run_first_order_data_scores: Callable | None = None,
):
    in_run_first_order_data_scores = (
        in_run_first_order_data_scores or _solutions().in_run_first_order_data_scores
    )
    scores = in_run_first_order_data_scores(*_toy_problem())
    assert t.allclose(scores, t.tensor([4.0, 4.0, 4.0, -4.0], dtype=t.float64)), (
        "The gradient-dot proxy should assign positive scores to helpful examples "
        "and a negative score to the flipped-label example."
    )
    print("All tests in `test_in_run_first_order_scores_toy_oracle` passed!")


def test_exact_data_shapley_smoke_test(
    exact_data_shapley_smoke_test: Callable | None = None,
):
    exact_data_shapley_smoke_test = (
        exact_data_shapley_smoke_test or _solutions().exact_data_shapley_smoke_test
    )
    result = exact_data_shapley_smoke_test()
    assert result["harmful_index"] == 3, (
        "Exact Data Shapley should identify the flipped-label fourth example as harmful."
    )
    assert result["harmful_value"] < 0.0, (
        "The harmful example should have negative exact Data Shapley value."
    )
    assert result["deletion_test_passes"], (
        "Removing the harmful example should improve validation utility."
    )
    assert result["addition_test_passes"], (
        "Adding helpful examples should improve validation utility in the exact report."
    )
    print("All tests in `test_exact_data_shapley_smoke_test` passed!")


def test_monte_carlo_data_shapley_smoke_test(
    monte_carlo_data_shapley_smoke_test: Callable | None = None,
):
    monte_carlo_data_shapley_smoke_test = (
        monte_carlo_data_shapley_smoke_test
        or _solutions().monte_carlo_data_shapley_smoke_test
    )
    result = monte_carlo_data_shapley_smoke_test()
    assert result["approximates_exact"], (
        "The sampled permutation estimate should be close to exact Data Shapley."
    )
    assert result["harmful_example_matches"], (
        "Monte Carlo Data Shapley should identify the same harmful example as exact Shapley."
    )
    assert result["top_example_matches"], (
        "Monte Carlo Data Shapley should identify one of the tied top helpful examples."
    )
    assert result["max_abs_error"] < 0.08, (
        f"Sampled Data Shapley max error should stay below 0.08, got {result['max_abs_error']}."
    )
    print("All tests in `test_monte_carlo_data_shapley_smoke_test` passed!")


def test_in_run_data_shapley_smoke_test(
    in_run_data_shapley_smoke_test: Callable | None = None,
):
    in_run_data_shapley_smoke_test = (
        in_run_data_shapley_smoke_test or _solutions().in_run_data_shapley_smoke_test
    )
    result = in_run_data_shapley_smoke_test()
    assert result["correlates_with_exact"], (
        "The in-run gradient-dot proxy should strongly correlate with exact Data Shapley."
    )
    assert result["identifies_harmful"], (
        "The in-run proxy should identify the flipped-label harmful example."
    )
    assert result["identifies_helpful"], (
        "The in-run proxy should identify a top helpful example."
    )
    assert result["pearson_correlation"] > 0.99, (
        "The toy in-run proxy should have Pearson correlation above 0.99 with exact values."
    )
    print("All tests in `test_in_run_data_shapley_smoke_test` passed!")


def test_random_data_attribution_failure_smoke_test(
    random_data_attribution_failure_smoke_test: Callable | None = None,
):
    random_data_attribution_failure_smoke_test = (
        random_data_attribution_failure_smoke_test
        or _solutions().random_data_attribution_failure_smoke_test
    )
    result = random_data_attribution_failure_smoke_test()
    assert result["random_data_attribution_fails"], (
        "Attribution on deterministic random data should fail to recover the planted "
        "helpful/harmful ordering."
    )
    assert result["harmful_index"] != result["original_harmful_index"], (
        "The random-data control should not identify the planted flipped-label index."
    )
    assert result["max_abs_signal_correlation"] <= 0.25, (
        "Random-data attributions should have low correlation with the planted signal."
    )
    print("All tests in `test_random_data_attribution_failure_smoke_test` passed!")


def test_label_shuffled_attribution_failure_smoke_test(
    label_shuffled_attribution_failure_smoke_test: Callable | None = None,
):
    label_shuffled_attribution_failure_smoke_test = (
        label_shuffled_attribution_failure_smoke_test
        or _solutions().label_shuffled_attribution_failure_smoke_test
    )
    result = label_shuffled_attribution_failure_smoke_test()
    assert result["label_shuffled_attribution_fails"], (
        "The label-shuffled control should move the harmful attribution away from "
        "the planted flipped-label index."
    )
    assert result["harmful_index"] != result["original_harmful_index"], (
        "The shuffled-label exact values should not identify the original harmful index."
    )
    assert result["signal_correlation"] <= 0.0, (
        "The shuffled-label exact values should not preserve the planted signal."
    )
    print("All tests in `test_label_shuffled_attribution_failure_smoke_test` passed!")


def test_runtime_overhead_smoke_test(
    runtime_overhead_smoke_test: Callable | None = None,
):
    runtime_overhead_smoke_test = (
        runtime_overhead_smoke_test or _solutions().runtime_overhead_smoke_test
    )
    result = runtime_overhead_smoke_test()
    assert result["runtime_overhead_reported"], (
        "The Data Shapley contract should report measured runtime overhead metrics."
    )
    assert result["runtime_measurement_repeats"] > 0, (
        "Runtime metrics should record the number of repeated measurements."
    )
    assert result["runtime_full_update_seconds"] > 0.0, (
        "Runtime metrics should include a positive measured full-update time."
    )
    assert result["runtime_exact_enumeration_seconds"] > 0.0, (
        "Runtime metrics should include a positive exact-enumeration time."
    )
    assert result["runtime_in_run_scores_seconds"] > 0.0, (
        "Runtime metrics should include a positive in-run score time."
    )
    print("All tests in `test_runtime_overhead_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["exact"]["deletion_test_passes"], (
        "The notebook contract should include the exact harmful-example deletion test."
    )
    assert result["monte_carlo"]["approximates_exact"], (
        "The notebook contract should include the Monte Carlo approximation check."
    )
    assert result["in_run"]["correlates_with_exact"], (
        "The notebook contract should include the in-run proxy correlation check."
    )
    assert result["random_data_control"]["random_data_attribution_fails"], (
        "The notebook contract should include the random-data failure control."
    )
    assert result["label_shuffle_control"]["label_shuffled_attribution_fails"], (
        "The notebook contract should include the label-shuffled failure control."
    )
    assert result["runtime_overhead"]["runtime_overhead_reported"], (
        "The notebook contract should include runtime-overhead measurements."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_records_exact_proxy_and_controls():
    report_path = Path(__file__).with_name("verification_report.json")
    report = json.loads(report_path.read_text())
    assert report["accepted"], "The committed 16.7 report should be accepted."
    assert report["gt_tier"] == "GT-0", "This section is exact finite-game ground truth."

    gpu = report["metrics"]["gpu_test"]
    assert gpu["preflight_passed"], "The CUDA one-step Data Shapley preflight should pass."
    assert gpu["cuda_version"] == "13.2", "The report should use the CUDA 13.2 torch wheel."
    assert gpu["model_family"] == "cuda_one_step_linear_regression_data_shapley", (
        "The GPU report should use the generated one-step regression model organism."
    )
    assert gpu["training_example_count"] == 4, "The toy problem should have four examples."
    assert gpu["coalition_count"] == 16, "The GPU path should enumerate all 2**4 coalitions."
    assert gpu["harmful_index"] == 3, "The flipped-label example should remain harmful."
    assert gpu["harmful_value"] < 0.0, "The harmful example should have negative value."
    assert gpu["harmful_removal_delta"] > 0.0, (
        "Deleting the harmful example should improve validation utility."
    )
    assert gpu["sampled_approximates_exact"], "Monte Carlo estimates should match exact values."
    assert gpu["sampled_max_abs_error"] <= 0.08, (
        "The sampled estimate should stay within the section tolerance."
    )
    assert gpu["pearson_correlation"] >= 0.99, (
        "The in-run gradient-dot proxy should correlate strongly with exact values."
    )
    assert gpu["identifies_harmful"], "The in-run proxy should identify the harmful example."
    assert gpu["identifies_helpful"], "The in-run proxy should identify a helpful example."
    assert gpu["random_data_attribution_fails"], "The random-data control should fail."
    assert gpu["label_shuffled_attribution_fails"], "The label-shuffled control should fail."
    assert gpu["runtime_overhead_reported"], "Runtime overhead metrics should be present."
    assert gpu["runtime_in_run_faster_than_exact"], (
        "The in-run score path should be faster than exact enumeration in the GPU report."
    )
    assert gpu["peak_vram_gb"] < 1.0, "The preflight should stay far below the 24GB budget."
    assert gpu["within_vram_budget"], "The committed report should satisfy the VRAM budget."
    print("All tests in `test_committed_gpu_report_records_exact_proxy_and_controls` passed!")
