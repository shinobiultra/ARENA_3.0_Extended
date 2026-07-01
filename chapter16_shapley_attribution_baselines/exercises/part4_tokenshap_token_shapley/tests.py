from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import torch as t

from arena_ext import shapley_attribution as reference


TOKENS = ("The", "capital", "is", "Paris")
MASK_TOKEN = "[MASK]"


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part4_tokenshap_token_shapley import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _gpu_report() -> dict:
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    return report["metrics"]["gpu_test"]


def _assert_close(actual: float, expected: float, *, msg: str, tol: float = 1e-9) -> None:
    assert abs(float(actual) - float(expected)) <= tol, (
        f"{msg}: expected {expected}, got {actual}."
    )


def _assert_tensor_close(
    actual: t.Tensor,
    expected: t.Tensor,
    *,
    msg: str,
    tol: float = 1e-9,
) -> None:
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    assert actual.shape == expected.shape, (
        f"{msg}: expected shape {tuple(expected.shape)}, got {tuple(actual.shape)}."
    )
    max_error = float((actual - expected).abs().max().item())
    assert max_error <= tol, f"{msg}: max absolute error {max_error} exceeds {tol}."


def _report_to_plain(report: object) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def test_all_coalitions_enumerates_complete_powerset(
    all_coalitions: Callable | None = None,
):
    all_coalitions = all_coalitions or _solutions().all_coalitions
    coalitions = all_coalitions(3)
    expected = reference.all_coalitions(3)
    assert coalitions == expected, (
        "Coalitions should be ordered by size and match the exact powerset reference."
    )
    assert len(coalitions) == 8 and len(set(coalitions)) == 8, (
        "A three-player game should expose each of its 2**3 coalitions exactly once."
    )
    try:
        all_coalitions(0)
    except ValueError as exc:
        assert "positive" in str(exc), (
            "Invalid player counts should raise a clear positive-count error."
        )
    else:
        raise AssertionError("num_players=0 should raise ValueError.")
    print("All tests in `test_all_coalitions_enumerates_complete_powerset` passed!")


def test_token_coalition_values_masks_absent_positions(
    token_coalition_values: Callable | None = None,
):
    token_coalition_values = token_coalition_values or _solutions().token_coalition_values
    values = token_coalition_values(TOKENS, reference.keyword_interaction_token_score)
    expected = reference.token_coalition_values(
        TOKENS,
        reference.keyword_interaction_token_score,
    )
    assert values == expected, (
        "Token coalition values should match the independent masked-token reference."
    )
    assert len(values) == 16, "Four token positions require a complete 2**4 table."
    assert values[frozenset()] == 0.0, (
        "The empty coalition should score the fully masked baseline."
    )
    assert values[frozenset({3})] == 1.0, (
        "The Paris-only coalition should get only the target-token contribution."
    )
    assert values[frozenset({1, 3})] == 3.0, (
        "The capital/Paris coalition should include target plus interaction credit."
    )
    try:
        token_coalition_values((), reference.keyword_interaction_token_score)
    except ValueError as exc:
        assert "nonempty" in str(exc), "Empty token sequences should raise a clear error."
    else:
        raise AssertionError("Empty token sequences should raise ValueError.")
    print("All tests in `test_token_coalition_values_masks_absent_positions` passed!")


def test_keyword_interaction_token_score_is_target_context_game(
    keyword_interaction_token_score: Callable | None = None,
):
    keyword_interaction_token_score = (
        keyword_interaction_token_score or _solutions().keyword_interaction_token_score
    )
    _assert_close(
        keyword_interaction_token_score((MASK_TOKEN, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN)),
        0.0,
        msg="Fully masked prompt score",
    )
    _assert_close(
        keyword_interaction_token_score(("The", MASK_TOKEN, "is", "Paris")),
        1.0,
        msg="Target-only prompt score",
    )
    _assert_close(
        keyword_interaction_token_score(("The", "capital", "is", "Paris")),
        3.0,
        msg="Full context-target prompt score",
    )
    print("All tests in `test_keyword_interaction_token_score_is_target_context_game` passed!")


def test_exact_token_shapley_values_splits_context_target_interaction(
    exact_token_shapley_values: Callable | None = None,
):
    exact_token_shapley_values = (
        exact_token_shapley_values or _solutions().exact_token_shapley_values
    )
    exact = exact_token_shapley_values(TOKENS, reference.keyword_interaction_token_score)
    expected = reference.exact_token_shapley_values(
        TOKENS,
        reference.keyword_interaction_token_score,
    )
    _assert_tensor_close(
        exact,
        expected,
        msg="Exact token Shapley values should match the analytic token game",
    )
    assert exact.tolist() == [0.0, 1.0, 0.0, 2.0], (
        "The context token should receive one interaction point and Paris should receive two points."
    )
    print(
        "All tests in `test_exact_token_shapley_values_splits_context_target_interaction` passed!"
    )


def test_token_baseline_report_checks_efficiency(
    token_baseline_report: Callable | None = None,
):
    token_baseline_report = token_baseline_report or _solutions().token_baseline_report
    report = token_baseline_report(TOKENS, reference.keyword_interaction_token_score)
    expected = reference.token_baseline_report(
        TOKENS,
        reference.keyword_interaction_token_score,
    )
    assert report.__dict__ == expected.__dict__, (
        "Token baseline report should expose the same fields and values as the reference."
    )
    assert report.full_score == 3.0 and report.baseline_score == 0.0, (
        "The report should keep full and masked-baseline scores separately."
    )
    assert report.total_delta == 3.0 and report.satisfies_efficiency, (
        "Exact token Shapley values should sum to full_score - baseline_score."
    )
    print("All tests in `test_token_baseline_report_checks_efficiency` passed!")


def test_token_shapley_sampling_report_matches_exact_ranking(
    token_shapley_sampling_report: Callable | None = None,
):
    token_shapley_sampling_report = (
        token_shapley_sampling_report or _solutions().token_shapley_sampling_report
    )
    report = token_shapley_sampling_report(
        TOKENS,
        reference.keyword_interaction_token_score,
        num_samples=512,
        seed=0,
        tolerance=0.1,
    )
    expected = reference.token_shapley_sampling_report(
        TOKENS,
        reference.keyword_interaction_token_score,
        num_samples=512,
        seed=0,
        tolerance=0.1,
    )
    assert _report_to_plain(report) == _report_to_plain(expected), (
        "Sampled TokenSHAP should be deterministic under seed=0 and match the reference."
    )
    assert report.approximates_exact and report.max_abs_error < 0.1, (
        "The 512-sample estimate should be within the declared exact-value tolerance."
    )
    assert report.rank_matches and report.sampled_top_token == "Paris", (
        "Sampling noise should preserve Paris as the top-attributed token."
    )
    print("All tests in `test_token_shapley_sampling_report_matches_exact_ranking` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["coalitions"]["empty"] == 0.0, (
        "The notebook contract should include the masked-baseline coalition."
    )
    assert result["coalitions"]["context_and_target"] == 3.0, (
        "The notebook contract should include the context-target coalition score."
    )
    assert result["exact"]["exact_values"] == [0.0, 1.0, 0.0, 2.0], (
        "The notebook contract should include exact analytic token Shapley values."
    )
    assert result["exact"]["baseline"]["satisfies_efficiency"], (
        "The notebook contract should include the token-efficiency check."
    )
    assert result["sampled"]["approximates_exact"], (
        "The notebook contract should include sampled TokenSHAP parity."
    )
    assert result["sampled"]["sampled_top_token"] == "Paris", (
        "The notebook contract should preserve the sampled top-token check."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_matches_token_shapley_contract(result: dict | None = None):
    result = result or _gpu_report()
    assert result["cuda_available"] and result["preflight_passed"], (
        "The committed report should come from the CUDA token-scorer preflight."
    )
    assert result["model_family"] == "cuda_trained_tiny_token_scorer_mlp", (
        "The report should name the trained CUDA token scorer used for evidence."
    )
    assert result["token_count"] == 4 and result["coalition_count"] == 16, (
        "The report should cover the complete four-token coalition table."
    )
    assert result["training_example_count"] == 16 and result["training_steps"] == 1200, (
        "The report should train on every masked coalition for the declared step count."
    )
    assert result["fit_mse"] <= 1e-10, (
        "The CUDA token scorer should fit the finite coalition table nearly exactly."
    )
    assert result["exact_shapley_max_abs_error"] <= 1e-5, (
        "Model-output exact TokenSHAP should match the analytic token game."
    )
    assert result["sampled_max_abs_error"] <= 0.1 and result["sampled_rank_matches"], (
        "Sampled TokenSHAP should stay within tolerance and preserve the ranking."
    )
    assert result["top_token"] == "Paris" and result["sampled_top_token"] == "Paris", (
        "Exact and sampled TokenSHAP should agree on Paris as the top token."
    )
    assert result["satisfies_efficiency"] and result["baseline_efficiency_error"] <= 1e-8, (
        "The trained-model token attributions should satisfy Shapley efficiency."
    )
    assert result["shuffled_control_error"] >= 1.0 and result["shuffled_control_rejected"], (
        "The shuffled-label trained scorer should be rejected as a negative control."
    )
    assert result["peak_vram_gb"] < 1.0 and result["within_vram_budget"], (
        "The finite CUDA preflight should stay under the declared 1 GB evidence budget."
    )
    print("All tests in `test_committed_gpu_report_matches_token_shapley_contract` passed!")
