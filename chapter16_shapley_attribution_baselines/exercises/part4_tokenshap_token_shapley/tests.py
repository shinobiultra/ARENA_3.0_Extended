from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path

import torch as t

LAB_TOKENS = ("Please", "approve", "allow", "urgent", "transfer")
RANDOM_CONTROL_TOKENS = ("Please", "approve", "allow", "banana", "transfer")
SPLIT_TOKENS = ("Please", "approve", "allow", "urgent", "trans", "##fer")
MASK_TOKEN = "[MASK]"


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part4_tokenshap_token_shapley import (
        solutions,
    )
    return solutions


def _assert_close(actual, expected, *, message: str, atol: float = 1e-9):
    actual_tensor = t.as_tensor(actual, dtype=t.float64)
    expected_tensor = t.as_tensor(expected, dtype=t.float64)
    assert actual_tensor.shape == expected_tensor.shape, (
        f"{message}: expected shape {tuple(expected_tensor.shape)}, got {tuple(actual_tensor.shape)}."
    )
    error = float((actual_tensor - expected_tensor).abs().max().item())
    assert error <= atol, f"{message}: maximum absolute error {error:.3g} exceeds {atol:.3g}."


def _independent_exact(values: dict[frozenset[int], float], n: int) -> t.Tensor:
    result = t.zeros(n, dtype=t.float64)
    for permutation in itertools.permutations(range(n)):
        coalition = frozenset()
        for player in permutation:
            with_player = coalition | {player}
            result[player] += values[with_player] - values[coalition]
            coalition = with_player
    return result / math.factorial(n)


def test_mask_tokens_and_coalition_values_preserve_positions(
    mask_tokens: Callable | None = None,
    token_coalition_values: Callable | None = None,
    structured_token_score: Callable | None = None,
):
    sol = _solutions()
    mask_tokens = mask_tokens or sol.mask_tokens
    token_coalition_values = token_coalition_values or sol.token_coalition_values
    structured_token_score = structured_token_score or sol.structured_token_score

    masked = mask_tokens(LAB_TOKENS, frozenset({1, 4}))
    assert masked == (MASK_TOKEN, "approve", MASK_TOKEN, MASK_TOKEN, "transfer")
    values = token_coalition_values(LAB_TOKENS, structured_token_score)
    assert len(values) == 32 and values[frozenset()] == 0.0
    assert values[frozenset({4})] == 4.0, "The necessary transfer token supplies the base score."
    assert values[frozenset({1, 2, 3})] == 0.0, "Nothing should score without transfer."
    assert values[frozenset(range(5))] == 9.0
    try:
        mask_tokens(LAB_TOKENS, {5})
    except ValueError as exc:
        assert "out-of-range" in str(exc)
    else:
        raise AssertionError("An invalid token position should raise ValueError.")
    print("All tests in `test_mask_tokens_and_coalition_values_preserve_positions` passed!")


def test_exact_shapley_recovers_known_token_roles(
    exact_shapley_values: Callable | None = None,
    token_coalition_values: Callable | None = None,
    structured_token_score: Callable | None = None,
):
    sol = _solutions()
    exact_shapley_values = exact_shapley_values or sol.exact_shapley_values
    token_coalition_values = token_coalition_values or sol.token_coalition_values
    structured_token_score = structured_token_score or sol.structured_token_score
    values = token_coalition_values(LAB_TOKENS, structured_token_score)
    observed = exact_shapley_values(values, num_players=5)
    expected = t.tensor([0.0, 1 / 3, 1 / 3, 1.5, 41 / 6], dtype=t.float64)
    _assert_close(observed, expected, message="Exact token Shapley oracle")
    _assert_close(observed, _independent_exact(values, 5), message="Permutation oracle")
    assert abs(float(observed.sum()) - 9.0) < 1e-9
    print("All tests in `test_exact_shapley_recovers_known_token_roles` passed!")


def test_sampled_shapley_matches_exact_and_is_seeded(
    sampled_permutation_shapley_values: Callable | None = None,
    token_coalition_values: Callable | None = None,
    structured_token_score: Callable | None = None,
):
    sol = _solutions()
    sampled_permutation_shapley_values = sampled_permutation_shapley_values or sol.sampled_permutation_shapley_values
    token_coalition_values = token_coalition_values or sol.token_coalition_values
    structured_token_score = structured_token_score or sol.structured_token_score
    values = token_coalition_values(LAB_TOKENS, structured_token_score)
    first = sampled_permutation_shapley_values(values, num_players=5, num_samples=2048, seed=7)
    second = sampled_permutation_shapley_values(values, num_players=5, num_samples=2048, seed=7)
    exact = _independent_exact(values, 5)
    _assert_close(first, second, message="Seeded Monte Carlo determinism")
    assert float((first - exact).abs().max()) < 0.08
    assert abs(float(first.sum()) - 9.0) < 1e-9, "Each sampled ordering must telescope to efficiency."
    print("All tests in `test_sampled_shapley_matches_exact_and_is_seeded` passed!")


def test_local_and_position_controls_fail_semantically(
    leave_one_out_values: Callable | None = None,
    recency_position_control: Callable | None = None,
    structured_token_score: Callable | None = None,
):
    sol = _solutions()
    leave_one_out_values = leave_one_out_values or sol.leave_one_out_values
    recency_position_control = recency_position_control or sol.recency_position_control
    structured_token_score = structured_token_score or sol.structured_token_score
    loo = leave_one_out_values(LAB_TOKENS, structured_token_score)
    recency = recency_position_control(9.0, len(LAB_TOKENS))
    _assert_close(loo, [0.0, 0.0, 0.0, 3.0, 9.0], message="Leave-one-out oracle")
    _assert_close(recency, [0.6, 1.2, 1.8, 2.4, 3.0], message="Recency control")
    assert float(loo.sum()) == 12.0, "Local deletion double-counts interactions and violates efficiency."
    assert float(recency[0]) > 0.0, "The position control should falsely credit the distractor."
    print("All tests in `test_local_and_position_controls_fail_semantically` passed!")


def test_pair_differences_reveal_redundancy_and_synergy(
    discrete_pair_interaction: Callable | None = None,
    token_coalition_values: Callable | None = None,
    structured_token_score: Callable | None = None,
):
    sol = _solutions()
    discrete_pair_interaction = discrete_pair_interaction or sol.discrete_pair_interaction
    token_coalition_values = token_coalition_values or sol.token_coalition_values
    structured_token_score = structured_token_score or sol.structured_token_score
    values = token_coalition_values(LAB_TOKENS, structured_token_score)
    redundancy = discrete_pair_interaction(values, (1, 2), context={3, 4}, num_players=5)
    synergy = discrete_pair_interaction(values, (3, 4), context=frozenset(), num_players=5)
    distractor = discrete_pair_interaction(values, (0, 4), context=frozenset(), num_players=5)
    assert redundancy == -2.0 and synergy == 3.0 and distractor == 0.0
    print("All tests in `test_pair_differences_reveal_redundancy_and_synergy` passed!")


def test_grouped_players_expose_tokenization_dependence(
    grouped_coalition_values: Callable | None = None,
    exact_shapley_values: Callable | None = None,
    token_coalition_values: Callable | None = None,
    split_token_score: Callable | None = None,
):
    sol = _solutions()
    grouped_coalition_values = grouped_coalition_values or sol.grouped_coalition_values
    exact_shapley_values = exact_shapley_values or sol.exact_shapley_values
    token_coalition_values = token_coalition_values or sol.token_coalition_values
    split_token_score = split_token_score or sol.split_token_score
    token_values = token_coalition_values(SPLIT_TOKENS, split_token_score)
    token_shapley = exact_shapley_values(token_values, num_players=6)
    _assert_close(
        token_shapley,
        [0.0, 1 / 6, 1 / 6, 1.0, 23 / 6, 23 / 6],
        message="Split-token Shapley oracle",
    )
    groups = ((0,), (1,), (2,), (3,), (4, 5))
    group_values = grouped_coalition_values(SPLIT_TOKENS, groups, split_token_score)
    group_shapley = exact_shapley_values(group_values, num_players=5)
    _assert_close(
        group_shapley,
        [0.0, 1 / 3, 1 / 3, 1.5, 41 / 6],
        message="Grouped-concept Shapley oracle",
    )
    assert abs(float(token_shapley[4:6].sum()) - 46 / 6) < 1e-9
    assert abs(float(group_shapley[4]) - 41 / 6) < 1e-9
    print("All tests in `test_grouped_players_expose_tokenization_dependence` passed!")


def test_correlated_support_does_not_identify_attribution(
    correlated_pair_audit: Callable | None = None,
    structured_token_score: Callable | None = None,
):
    sol = _solutions()
    correlated_pair_audit = correlated_pair_audit or sol.correlated_pair_audit
    structured_token_score = structured_token_score or sol.structured_token_score
    audit = correlated_pair_audit(
        LAB_TOKENS,
        lambda tokens: structured_token_score(tokens, redundant_mode="or"),
        lambda tokens: structured_token_score(tokens, redundant_mode="and"),
        pair=(1, 2),
    )
    assert audit.observed_max_abs_difference == 0.0
    assert audit.off_manifold_max_abs_difference == 2.0
    assert abs(audit.attribution_max_abs_difference - 2 / 3) < 1e-9
    assert not audit.identified_from_observed_support
    _assert_close(
        audit.synergy_shapley,
        [0.0, 2 / 3, 2 / 3, 1.5, 37 / 6],
        message="Correlated synergy extension oracle",
    )
    print("All tests in `test_correlated_support_does_not_identify_attribution` passed!")


def test_sampling_convergence_reports_seed_distribution(
    sampling_convergence: Callable | None = None,
    token_coalition_values: Callable | None = None,
    structured_token_score: Callable | None = None,
):
    sol = _solutions()
    sampling_convergence = sampling_convergence or sol.sampling_convergence
    token_coalition_values = token_coalition_values or sol.token_coalition_values
    structured_token_score = structured_token_score or sol.structured_token_score
    values = token_coalition_values(LAB_TOKENS, structured_token_score)
    rows = sampling_convergence(
        values,
        num_players=5,
        budgets=(4, 16, 64, 1024),
        seeds=tuple(range(40)),
    )
    assert [row["budget"] for row in rows] == [4, 16, 64, 1024]
    assert rows[-1]["mean_max_abs_error"] < rows[0]["mean_max_abs_error"]
    assert rows[-1]["p90_max_abs_error"] < 0.2
    print("All tests in `test_sampling_convergence_reports_seed_distribution` passed!")


def test_random_token_and_shuffled_value_controls_fail(
    shuffle_coalition_values_within_sizes: Callable | None = None,
    exact_shapley_values: Callable | None = None,
    exact_token_shapley_values: Callable | None = None,
    sampling_convergence: Callable | None = None,
    token_coalition_values: Callable | None = None,
    structured_token_score: Callable | None = None,
):
    sol = _solutions()
    shuffle_coalition_values_within_sizes = (
        shuffle_coalition_values_within_sizes or sol.shuffle_coalition_values_within_sizes
    )
    exact_shapley_values = exact_shapley_values or sol.exact_shapley_values
    exact_token_shapley_values = exact_token_shapley_values or sol.exact_token_shapley_values
    sampling_convergence = sampling_convergence or sol.sampling_convergence
    token_coalition_values = token_coalition_values or sol.token_coalition_values
    structured_token_score = structured_token_score or sol.structured_token_score

    random_exact = exact_token_shapley_values(RANDOM_CONTROL_TOKENS, structured_token_score)
    _assert_close(
        random_exact,
        [0.0, 1 / 3, 1 / 3, 0.0, 16 / 3],
        message="Random-token control oracle",
    )

    values = token_coalition_values(LAB_TOKENS, structured_token_score)
    shuffled = shuffle_coalition_values_within_sizes(values, num_players=5, seed=11)
    for size in range(6):
        before = sorted(value for coalition, value in values.items() if len(coalition) == size)
        after = sorted(value for coalition, value in shuffled.items() if len(coalition) == size)
        assert before == after, f"The size-{size} score distribution must be preserved."
    assert shuffled[frozenset()] == 0.0 and shuffled[frozenset(range(5))] == 9.0

    exact = exact_shapley_values(values, num_players=5)
    shuffled_exact = exact_shapley_values(shuffled, num_players=5)
    semantic_error = float((shuffled_exact - exact).abs().max().item())
    assert semantic_error >= 1.0, "Shuffling should destroy the token semantics despite preserving endpoints."
    control_rows = sampling_convergence(
        shuffled,
        num_players=5,
        budgets=(16, 256, 4096),
        seeds=tuple(range(20)),
        reference_values=exact,
    )
    assert control_rows[-1]["mean_max_abs_error"] >= 1.0
    print("All tests in `test_random_token_and_shuffled_value_controls_fail` passed!")


def test_release_smoke_contract_remains_compatible(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["coalitions"] == {"empty": 0.0, "target_only": 1.0, "context_and_target": 3.0}
    assert result["exact"]["exact_values"] == [0.0, 1.0, 0.0, 2.0]
    assert result["exact"]["baseline"]["satisfies_efficiency"]
    assert result["sampled"]["approximates_exact"]
    print("All tests in `test_release_smoke_contract_remains_compatible` passed!")


def test_committed_gpu_report_matches_token_shapley_contract(result: dict | None = None):
    if result is None:
        report = json.loads((Path(__file__).resolve().parent / "verification_report.json").read_text())
        result = report["metrics"]["gpu_test"]
    assert result["cuda_available"] and result["preflight_passed"]
    assert result["model_family"] == "cuda_trained_tiny_token_scorer_mlp"
    assert result["token_count"] == 4 and result["coalition_count"] == 16
    assert result["exact_shapley_max_abs_error"] <= 1e-5
    assert result["sampled_max_abs_error"] <= 0.1 and result["sampled_rank_matches"]
    assert result["shuffled_control_error"] >= 1.0 and result["shuffled_control_rejected"]
    print("All tests in `test_committed_gpu_report_matches_token_shapley_contract` passed!")
