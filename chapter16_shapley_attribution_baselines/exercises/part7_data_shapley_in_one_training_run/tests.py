from collections.abc import Callable


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run import (
        solutions,
    )

    return solutions


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
        "Monte Carlo Data Shapley should identify the same top helpful example as exact Shapley."
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
