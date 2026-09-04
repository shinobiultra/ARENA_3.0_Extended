import json
from collections.abc import Callable
from pathlib import Path

import torch as t


SECTION_DIR = Path(__file__).resolve().parent


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
    expected = {
        frozenset(): 0.0,
        frozenset({0}): 1.0,
        frozenset({3}): -3.0,
        frozenset({0, 1, 2}): 1.0,
        frozenset({0, 1, 2, 3}): 0.75,
    }
    for coalition, target in expected.items():
        actual = one_step_linear_utility(train_x, train_y, val_x, val_y, coalition)
        assert abs(actual - target) < 1e-12, (
            f"v({sorted(coalition)}) should be {target}, got {actual}."
        )
    print("All tests in `test_one_step_linear_utility_toy_oracle` passed!")


def test_all_coalitions_enumerates_power_set(all_coalitions: Callable | None = None):
    all_coalitions = all_coalitions or _solutions().all_coalitions
    coalitions = all_coalitions(4)
    assert len(coalitions) == 16, "Four examples require all 2**4 coalitions."
    assert len(set(coalitions)) == 16, "Every coalition should appear exactly once."
    assert coalitions[0] == frozenset(), "The empty coalition should be explicit."
    assert coalitions[-1] == frozenset(range(4)), "The full coalition should be explicit."
    print("All tests in `test_all_coalitions_enumerates_power_set` passed!")


def test_data_coalition_values_complete_table(
    data_coalition_values: Callable | None = None,
):
    data_coalition_values = data_coalition_values or _solutions().data_coalition_values
    values = data_coalition_values(*_toy_problem())
    assert len(values) == 16, "The utility table must contain all 16 coalitions."
    assert values[frozenset()] == 0.0, "The empty coalition must define zero utility."
    assert values[frozenset({0, 1, 2})] == 1.0, (
        "The three clean examples should achieve the exact maximum utility."
    )
    assert values[frozenset({0, 1, 2, 3})] == 0.75, (
        "Adding the planted harmful example should reduce full-coalition utility to 0.75."
    )
    print("All tests in `test_data_coalition_values_complete_table` passed!")


def test_exact_data_shapley_matches_hand_checked_oracle(
    exact_data_shapley_values: Callable | None = None,
):
    exact_data_shapley_values = (
        exact_data_shapley_values or _solutions().exact_data_shapley_values
    )
    actual = exact_data_shapley_values(*_toy_problem())
    expected = t.tensor(
        [0.6412037037037037, 0.6412037037037037, 0.6412037037037037, -1.173611111111111],
        dtype=t.float64,
    )
    assert actual.dtype == t.float64, "Exact Shapley should preserve float64 oracle precision."
    assert t.allclose(actual, expected, atol=1e-12, rtol=0.0), (
        "Exact values should match independent enumeration of all 24 player orderings."
    )
    assert abs(float(actual.sum().item()) - 0.75) < 1e-12, (
        "Efficiency requires the values to sum to v(all) - v(empty) = 0.75."
    )
    print("All tests in `test_exact_data_shapley_matches_hand_checked_oracle` passed!")


def test_sampled_permutations_recover_harmful_example(
    sampled_permutation_shapley_values: Callable | None = None,
):
    sampled_permutation_shapley_values = (
        sampled_permutation_shapley_values
        or _solutions().sampled_permutation_shapley_values
    )
    solution = _solutions()
    values = solution.data_coalition_values(*_toy_problem())
    exact = solution.exact_data_shapley_values(*_toy_problem())
    sampled = sampled_permutation_shapley_values(
        values,
        num_players=4,
        num_samples=512,
        seed=0,
    )
    assert int(sampled.argmin().item()) == 3, (
        "The sampled estimator should still identify the planted harmful example."
    )
    assert float((sampled - exact).abs().max().item()) < 0.08, (
        "The pinned 512-order estimate should approximate the exact toy values."
    )
    print("All tests in `test_sampled_permutations_recover_harmful_example` passed!")


def test_in_run_scores_match_gradient_oracle(
    in_run_first_order_data_scores: Callable | None = None,
    pearson_correlation: Callable | None = None,
):
    in_run_first_order_data_scores = (
        in_run_first_order_data_scores or _solutions().in_run_first_order_data_scores
    )
    pearson_correlation = pearson_correlation or _solutions().pearson_correlation
    scores = in_run_first_order_data_scores(*_toy_problem())
    expected = t.tensor([4.0, 4.0, 4.0, -4.0], dtype=t.float64)
    exact = _solutions().exact_data_shapley_values(*_toy_problem())
    assert t.equal(scores.cpu(), expected), (
        "The first-order score should match the hand-derived gradient-dot oracle."
    )
    assert pearson_correlation(exact, scores) > 0.999999, (
        "The toy in-run score should preserve the exact helpful-versus-harmful ranking."
    )
    print("All tests in `test_in_run_scores_match_gradient_oracle` passed!")


def test_leave_one_out_and_deletion_semantics(
    leave_one_out_values: Callable | None = None,
):
    leave_one_out_values = leave_one_out_values or _solutions().leave_one_out_values
    solution = _solutions()
    values = solution.data_coalition_values(*_toy_problem())
    actual = leave_one_out_values(values, num_players=4)
    expected = t.tensor([7 / 36, 7 / 36, 7 / 36, -0.25], dtype=t.float64)
    assert t.allclose(actual, expected, atol=1e-12, rtol=0.0), (
        "Leave-one-out values should match the exact full-coalition deletion effects."
    )
    full = frozenset(range(4))
    assert values[full - {3}] - values[full] == 0.25, (
        "Deleting the exact-bottom example should improve utility from 0.75 to 1.0."
    )
    print("All tests in `test_leave_one_out_and_deletion_semantics` passed!")


def test_semantic_controls_remove_and_relocate_negative_credit():
    solution = _solutions()
    clean = solution.clean_label_control()
    relocated = solution.relocated_corruption_control()
    assert t.allclose(
        clean["exact_values"],
        t.full((4,), 0.25, dtype=t.float64),
        atol=1e-12,
        rtol=0.0,
    ), "Clean symmetric examples should each receive one quarter of the utility."
    assert not clean["has_negative_value"], (
        "Removing the corrupted label should remove negative exact attribution."
    )
    assert relocated["harmful_index"] == 1, (
        "The exact negative credit should follow the relocated corruption to index 1."
    )
    assert relocated["proxy_harmful_index"] == 1, (
        "The in-run proxy should also follow the relocated corruption."
    )
    assert float(relocated["exact_values"][1].item()) < 0.0, (
        "The relocated corrupted example should receive negative exact value."
    )
    print("All tests in `test_semantic_controls_remove_and_relocate_negative_credit` passed!")


def test_proxy_stress_sweep_exposes_rank_reversal(
    proxy_alignment_sweep: Callable | None = None,
    first_proxy_rank_failure: Callable | None = None,
):
    proxy_alignment_sweep = proxy_alignment_sweep or _solutions().proxy_alignment_sweep
    first_proxy_rank_failure = (
        first_proxy_rank_failure or _solutions().first_proxy_rank_failure
    )
    learning_rates = [step / 20 for step in range(1, 51)]
    sweep = proxy_alignment_sweep(learning_rates)
    failure = first_proxy_rank_failure(learning_rates)
    lr_index = learning_rates.index(0.5)
    high_lr_index = learning_rates.index(2.0)
    assert float(sweep["correlations"][lr_index].item()) > 0.999999, (
        "At the calibrated step size, the proxy should agree with exact Shapley."
    )
    assert int(sweep["harmful_indices"][lr_index].item()) == 3, (
        "At the calibrated step size, the proxy should find the planted error."
    )
    assert float(sweep["correlations"][high_lr_index].item()) < -0.999999, (
        "At high learning rate, the proxy should visibly reverse the exact ranking."
    )
    assert int(sweep["harmful_indices"][high_lr_index].item()) != 3, (
        "The stress test should expose a concrete harmful-rank failure."
    )
    assert failure is not None and 1.90 <= failure <= 2.00, (
        "The first seeded rank failure should occur in the pinned learning-rate interval."
    )
    print("All tests in `test_proxy_stress_sweep_exposes_rank_reversal` passed!")


def test_training_run_subset_training_oracle(
    train_logistic_subset: Callable | None = None,
    training_run_utility: Callable | None = None,
):
    solution = _solutions()
    train_logistic_subset = train_logistic_subset or solution.train_logistic_subset
    training_run_utility = training_run_utility or solution.training_run_utility
    problem = solution.training_run_data_shapley_problem()
    train_x, train_y, _, _ = problem
    zero_parameters = train_logistic_subset(train_x, train_y, frozenset())
    full_parameters = train_logistic_subset(
        train_x, train_y, frozenset(range(len(train_y)))
    )
    assert t.equal(zero_parameters, t.zeros(3, dtype=t.float64)), (
        "The empty coalition must leave the fixed zero initialization unchanged."
    )
    assert full_parameters.shape == (3,), (
        "Two features plus a bias require exactly three learned parameters."
    )
    singleton = training_run_utility(*problem, frozenset({0}))
    duplicate_singleton = training_run_utility(*problem, frozenset({1}))
    harmful_singleton = training_run_utility(*problem, frozenset({6}))
    full = training_run_utility(*problem, frozenset(range(8)))
    assert abs(singleton - 0.6102960840586487) < 1e-10, (
        "Example 0 should achieve the pinned positive singleton utility."
    )
    assert singleton == duplicate_singleton, (
        "The exact duplicate pair must be indistinguishable under training."
    )
    assert abs(harmful_singleton - (-2.070779030062206)) < 1e-10, (
        "The mislabeled singleton should strongly damage held-out loss."
    )
    assert abs(full - 0.5082118812234312) < 1e-10, (
        "The 40-step full-data run should match the pinned held-out utility."
    )
    print("All tests in `test_training_run_subset_training_oracle` passed!")


def test_training_run_exact_ground_truth(
    training_run_coalition_values: Callable | None = None,
    exact_shapley_values: Callable | None = None,
):
    solution = _solutions()
    training_run_coalition_values = (
        training_run_coalition_values or solution.training_run_coalition_values
    )
    exact_shapley_values = exact_shapley_values or solution.exact_shapley_values
    values = training_run_coalition_values(*solution.training_run_data_shapley_problem())
    exact = exact_shapley_values(values, num_players=8)
    expected = t.tensor(
        [
            0.18830704371122473,
            0.18830704371122473,
            0.1617386027321338,
            0.15184833086204583,
            0.19281294634969415,
            0.17838310257782203,
            -0.6180514150010005,
            0.06486622628028535,
        ],
        dtype=t.float64,
    )
    full = frozenset(range(8))
    assert len(values) == 256, "Eight examples require all 2**8 retraining runs."
    assert t.allclose(exact, expected, atol=1e-10, rtol=0.0), (
        "Exact Shapley must match the pinned all-subsets training oracle."
    )
    assert abs(float(exact.sum().item()) - values[full]) < 1e-12, (
        "Efficiency must recover the full-set utility relative to the empty set."
    )
    assert exact[0] == exact[1], "Exact duplicates must receive equal Shapley value."
    assert int(exact.argmin().item()) == 6, (
        "The known mislabeled point must be the exact-bottom example."
    )
    print("All tests in `test_training_run_exact_ground_truth` passed!")


def test_random_order_and_budget_controls(
    sampled_permutation_shapley_values: Callable | None = None,
    fixed_order_marginal_values: Callable | None = None,
    permutation_budget_sweep: Callable | None = None,
):
    solution = _solutions()
    sampled_permutation_shapley_values = (
        sampled_permutation_shapley_values
        or solution.sampled_permutation_shapley_values
    )
    fixed_order_marginal_values = (
        fixed_order_marginal_values or solution.fixed_order_marginal_values
    )
    permutation_budget_sweep = (
        permutation_budget_sweep or solution.permutation_budget_sweep
    )
    values = solution.training_run_coalition_values(
        *solution.training_run_data_shapley_problem()
    )
    exact = solution.exact_shapley_values(values, num_players=8)
    sampled = sampled_permutation_shapley_values(
        values, num_players=8, num_samples=256, seed=0
    )
    ascending = fixed_order_marginal_values(values, tuple(range(8)))
    descending = fixed_order_marginal_values(values, tuple(reversed(range(8))))
    sweep = permutation_budget_sweep(
        values, exact, budgets=(4, 16, 64, 256), num_seeds=16
    )
    assert float((sampled - exact).abs().max().item()) < 0.025, (
        "The pinned 256-order estimate should be close to exact ground truth."
    )
    assert float((ascending - descending).abs().max().item()) > 0.9, (
        "Two fixed orders should visibly disagree, exposing ordering bias."
    )
    expected_means = t.tensor(
        [0.27275320168059536, 0.15616368873151773, 0.09894371634661243, 0.04370903321104829],
        dtype=t.float64,
    )
    assert t.allclose(sweep["mean_max_error"], expected_means, atol=1e-10, rtol=0.0), (
        "The seeded budget curve should match the Monte Carlo error oracle."
    )
    assert bool((sweep["harmful_hit_rate"] == 1.0).all().item()), (
        "Every tested budget should preserve the bottom-ranked harmful example."
    )
    print("All tests in `test_random_order_and_budget_controls` passed!")


def test_checkpoint_one_run_estimator(
    checkpoint_gradient_data_scores: Callable | None = None,
    pearson_correlation: Callable | None = None,
):
    solution = _solutions()
    checkpoint_gradient_data_scores = (
        checkpoint_gradient_data_scores or solution.checkpoint_gradient_data_scores
    )
    pearson_correlation = pearson_correlation or solution.pearson_correlation
    problem = solution.training_run_data_shapley_problem()
    values = solution.training_run_coalition_values(*problem)
    exact = solution.exact_shapley_values(values, num_players=8)
    scores = checkpoint_gradient_data_scores(*problem)
    expected = t.tensor(
        [
            0.1875495908099044,
            0.1875495908099044,
            0.17059583812151005,
            0.20802161745976464,
            0.1921222091273457,
            0.17508668294788773,
            -0.7084487583811443,
            0.12089843479743552,
        ],
        dtype=t.float64,
    )
    assert t.allclose(scores, expected, atol=1e-10, rtol=0.0), (
        "Checkpoint gradient accumulation should match the hand-pinned run."
    )
    assert pearson_correlation(exact, scores) > 0.995, (
        "The claimed one-run correlation must be visible on the exact organism."
    )
    assert int(scores.argmin().item()) == 6, (
        "The one-run estimator must identify the known mislabeled example."
    )
    assert scores[0] == scores[1], (
        "The estimator must preserve the duplicate pair's symmetry."
    )
    print("All tests in `test_checkpoint_one_run_estimator` passed!")


def test_influence_loo_and_matched_deletion_controls(
    influence_function_scores: Callable | None = None,
    leave_one_out_values: Callable | None = None,
):
    solution = _solutions()
    influence_function_scores = (
        influence_function_scores or solution.influence_function_scores
    )
    leave_one_out_values = leave_one_out_values or solution.leave_one_out_values
    problem = solution.training_run_data_shapley_problem()
    values = solution.training_run_coalition_values(*problem)
    exact = solution.exact_shapley_values(values, num_players=8)
    influence = influence_function_scores(*problem)
    leave_one_out = leave_one_out_values(values, num_players=8)
    full = frozenset(range(8))
    harmful_removal = values[full - {6}] - values[full]
    matched_removal = values[full - {0}] - values[full]
    assert int(influence.argmin().item()) == 6, (
        "The damped influence baseline should find the harmful example."
    )
    assert int(leave_one_out.argmin().item()) == 6, (
        "Leave-one-out should find the harmful example in the full-set context."
    )
    assert solution.pearson_correlation(exact, influence) > 0.97, (
        "Influence should be useful but not identical to coalition averaging."
    )
    assert solution.pearson_correlation(exact, leave_one_out) < 0.95, (
        "The pinned LOO baseline should expose its weaker global ordering."
    )
    assert harmful_removal > 0.13, (
        "Removing the harmful point should improve held-out utility by over 0.13."
    )
    assert matched_removal < -0.02, (
        "Removing one matched helpful duplicate should reduce held-out utility."
    )
    print("All tests in `test_influence_loo_and_matched_deletion_controls` passed!")


def test_shuffled_label_control_relocates_harm(
    shuffled_label_exact_values: Callable | None = None,
):
    solution = _solutions()
    shuffled_label_exact_values = (
        shuffled_label_exact_values or solution.shuffled_label_exact_values
    )
    problem = solution.training_run_data_shapley_problem()
    labels, exact = shuffled_label_exact_values(
        *problem, solution.TRAINING_RUN_LABEL_PERMUTATION
    )
    original_values = solution.training_run_coalition_values(*problem)
    original = solution.exact_shapley_values(original_values, num_players=8)
    assert int(original.argmin().item()) == 6, (
        "The original mislabeled point should be index 6."
    )
    assert int(exact.argmin().item()) == 0, (
        "Moving the zero label to duplicate index 0 should move negative credit there."
    )
    assert labels[0] == 0.0 and labels[6] == 1.0, (
        "The control must swap labels while holding all feature vectors fixed."
    )
    assert solution.pearson_correlation(original, exact) < 0.0, (
        "The attribution pattern should not survive relocation of the bad label."
    )
    print("All tests in `test_shuffled_label_control_relocates_harm` passed!")


def test_notebook_contract():
    exercise_path = SECTION_DIR / "16.7_Data_Shapley_in_One_Training_Run_exercises.ipynb"
    solution_path = SECTION_DIR / "16.7_Data_Shapley_in_One_Training_Run_solutions.ipynb"
    exercise = json.loads(exercise_path.read_text())
    solution = json.loads(solution_path.read_text())
    exercise_text = "\n".join("".join(cell["source"]) for cell in exercise["cells"])
    solution_text = "\n".join("".join(cell["source"]) for cell in solution["cells"])

    required_phrases = (
        "one precise claim",
        "Toy exact ground truth first",
        "Exact retraining ground truth",
        "Signature result",
        "Try It Yourself",
        "Anomaly hunt",
        "Real paper connection",
        "Limitations",
        "<details><summary>Expected output</summary>",
        "<details><summary>Help - implementation hint</summary>",
        "<details><summary>Interpretation</summary>",
        "<details><summary>Solution</summary>",
    )
    for phrase in required_phrases:
        assert phrase in exercise_text, f"Exercise notebook is missing `{phrase}`."
    assert exercise_text.count("raise NotImplementedError()") >= 10, (
        "The learner notebook must expose the core training and attribution methods."
    )
    assert "raise NotImplementedError()" not in solution_text, (
        "The solution notebook must contain executable implementations."
    )
    assert "verification_report.json" in exercise_text + solution_text, (
        "The notebook should identify the supporting release-evidence artifact."
    )
    for signature in (
        "def run_gpu_test(max_vram_gb: float = 24.0)",
        "def run_full_experiment(max_vram_gb: float = 24.0)",
    ):
        assert signature in exercise_text, (
            f"The exercise notebook is missing the callable release contract `{signature}`."
        )
    assert "supporting evidence, not the lesson" in exercise_text, (
        "The notebook must keep the verification report subordinate to live pedagogy."
    )
    assert "data_shapley_training_run_signature.png" in solution_text, (
        "The solution must save the new bitmap signature result."
    )
    assert "![Four-panel Data Shapley signature result]" in solution_text, (
        "The executed solution must display the signature result in the notebook."
    )
    solution_code = [cell for cell in solution["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is not None for cell in solution_code), (
        "Every solution code cell must be executed in place."
    )
    assert not any(
        output.get("output_type") == "error"
        for cell in solution_code
        for output in cell.get("outputs", [])
    ), "The executed solution must contain no error outputs."
    assert "256 real subset-training runs" in exercise_text, (
        "The exact ground-truth cost must be explicit to learners."
    )
    assert exercise_text.index("Toy exact ground truth first") < exercise_text.index(
        "The training organism"
    ), "The inspectable toy oracle must precede the real training game."
    print("All tests in `test_notebook_contract` passed!")


def test_run_smoke_test_packages_live_semantic_evidence():
    result = _solutions().run_smoke_test(cpu=True)
    assert result["exact"]["harmful_index"] == 3, (
        "The exact smoke test should identify the planted harmful toy example."
    )
    assert result["exact"]["deletion_test_passes"], (
        "Deleting the exact-bottom toy example should improve utility."
    )
    assert result["monte_carlo"]["approximates_exact"], (
        "The sampled toy estimator should satisfy its pinned approximation tolerance."
    )
    assert result["in_run"]["correlates_with_exact"], (
        "The toy in-run score should clear the exact-correlation threshold."
    )
    assert not result["clean_label_control"]["has_negative_value"], (
        "The clean-label control should eliminate negative attribution."
    )
    assert result["relocated_corruption_control"]["harmful_index"] == 1, (
        "The relocation control should move exact harm to index 1."
    )
    assert 1.90 <= result["first_proxy_rank_failure_lr"] <= 2.00, (
        "The smoke test should retain the calibrated proxy failure boundary."
    )
    lab = result["training_run_lab"]
    assert lab["harmful_index"] == 6, (
        "Exact retraining should locate the known mislabeled training row."
    )
    assert lab["one_run_correlation"] > 0.995, (
        "The one-run estimator should clear the notebook's stated correlation claim."
    )
    assert lab["harmful_removal_delta"] > 0.13, (
        "Deleting the harmful row should causally improve held-out utility."
    )
    assert lab["matched_removal_delta"] < -0.02, (
        "Deleting a matched helpful duplicate should reduce held-out utility."
    )
    assert lab["shuffled_label_control"]["shuffled_harmful_index"] == 0, (
        "Negative exact attribution should follow the shuffled bad label to index 0."
    )
    print("All tests in `test_run_smoke_test_packages_live_semantic_evidence` passed!")
