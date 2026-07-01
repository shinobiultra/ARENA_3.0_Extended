from collections.abc import Callable

import torch as t

from arena_ext import attribution_patching as reference


def _solutions():
    from chapter8_automated_circuits.exercises.part2_attribution_patching_eap import (
        solutions,
    )

    return solutions


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} fields should match the independent reference implementation."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, float):
            assert abs(actual_value - expected_value) < 1e-6, (
                f"{msg} field {key!r} should be {expected_value}, got {actual_value}."
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_attribution_patch_scores_sums_non_component_dims(
    attribution_patch_scores: Callable | None = None,
):
    attribution_patch_scores = attribution_patch_scores or _solutions().attribution_patch_scores
    clean = t.tensor([[2.0, 0.0], [0.0, 3.0]])
    corrupt = t.zeros_like(clean)
    gradients = t.tensor([[0.5, 0.5], [1.0, 1.0]])
    scores = attribution_patch_scores(clean, corrupt, gradients)
    expected = reference.attribution_patch_scores(clean, corrupt, gradients)
    assert scores.tolist() == expected.tolist() == [1.0, 3.0], (
        "Attribution scores should sum (clean-corrupt) * gradient over non-component dims."
    )
    column_scores = attribution_patch_scores(
        clean,
        corrupt,
        gradients,
        component_dim=1,
    )
    assert column_scores.tolist() == [1.0, 3.0], (
        "Changing component_dim should preserve that axis and reduce all other axes."
    )
    try:
        attribution_patch_scores(clean, corrupt[:1], gradients)
    except ValueError as exc:
        assert "matching shape" in str(exc), (
            "Mismatched clean/corrupt activations should fail with a shape message."
        )
    else:
        raise AssertionError("Shape-mismatched activations should raise ValueError.")
    print("All tests in `test_attribution_patch_scores_sums_non_component_dims` passed!")


def test_integrated_gradient_patch_scores_average_path_gradients(
    integrated_gradient_patch_scores: Callable | None = None,
):
    integrated_gradient_patch_scores = (
        integrated_gradient_patch_scores or _solutions().integrated_gradient_patch_scores
    )
    clean = t.tensor([[2.0, 0.0], [0.0, 3.0]])
    corrupt = t.zeros_like(clean)
    path_gradients = t.tensor(
        [
            [[0.25, 0.25], [0.5, 0.5]],
            [[0.75, 0.75], [1.5, 1.5]],
        ]
    )
    scores = integrated_gradient_patch_scores(clean, corrupt, path_gradients)
    expected = reference.integrated_gradient_patch_scores(clean, corrupt, path_gradients)
    assert scores.tolist() == expected.tolist() == [1.0, 3.0], (
        "Integrated-gradient scores should use the mean gradient along the interpolation path."
    )
    try:
        integrated_gradient_patch_scores(clean, corrupt, path_gradients[0])
    except ValueError as exc:
        assert "path_gradients" in str(exc), (
            "Path-gradient rank errors should name the required path_gradients shape."
        )
    else:
        raise AssertionError("Rank-mismatched path gradients should raise ValueError.")
    print(
        "All tests in `test_integrated_gradient_patch_scores_average_path_gradients` passed!"
    )


def test_edge_attribution_scores_forms_upstream_downstream_matrix(
    edge_attribution_scores: Callable | None = None,
):
    edge_attribution_scores = edge_attribution_scores or _solutions().edge_attribution_scores
    upstream_delta = t.tensor([[1.0, 0.0], [0.0, 2.0]])
    downstream_gradients = t.tensor([[3.0, 0.0], [0.0, 4.0]])
    scores = edge_attribution_scores(upstream_delta, downstream_gradients)
    expected = reference.edge_attribution_scores(upstream_delta, downstream_gradients)
    assert scores.tolist() == expected.tolist() == [[3.0, 0.0], [0.0, 8.0]], (
        "EAP edge scores should be upstream deltas times downstream gradients."
    )
    try:
        edge_attribution_scores(upstream_delta, downstream_gradients[:, :1])
    except ValueError as exc:
        assert "hidden dimensions" in str(exc), (
            "Hidden-width mismatches should raise a helpful dimension error."
        )
    else:
        raise AssertionError("Mismatched hidden dimensions should raise ValueError.")
    print("All tests in `test_edge_attribution_scores_forms_upstream_downstream_matrix` passed!")


def test_exact_vs_approx_reports_measure_correlation_and_topk_overlap(
    score_correlation_report: Callable | None = None,
    topk_overlap_report: Callable | None = None,
):
    solutions = _solutions()
    score_correlation_report = score_correlation_report or solutions.score_correlation_report
    topk_overlap_report = topk_overlap_report or solutions.topk_overlap_report
    exact = t.tensor([0.1, 0.9, 0.8, 0.0])
    approx = t.tensor([0.2, 0.85, 0.7, 0.1])
    correlation = score_correlation_report(exact, approx, min_correlation=0.95)
    expected_correlation = reference.score_correlation_report(
        exact,
        approx,
        min_correlation=0.95,
    )
    _assert_report_close(correlation, expected_correlation, msg="Score correlation report")
    assert correlation.correlation > 0.95 and correlation.passes_threshold, (
        "Approximate scores should pass when Pearson correlation with exact scores is high."
    )
    bad_correlation = score_correlation_report(
        exact,
        t.tensor([0.9, 0.1, 0.0, 0.8]),
        min_correlation=0.95,
    )
    assert not bad_correlation.passes_threshold, (
        "A badly ordered approximation should fail the correlation threshold."
    )

    overlap = topk_overlap_report(exact, approx, top_k=2, min_overlap=1.0)
    expected_overlap = reference.topk_overlap_report(
        exact,
        approx,
        top_k=2,
        min_overlap=1.0,
    )
    _assert_report_close(overlap, expected_overlap, msg="Top-k overlap report")
    assert overlap.exact_top_indices == (1, 2) and overlap.approx_top_indices == (1, 2), (
        "Top-k report should expose exact and approximate priority sets."
    )
    assert overlap.topk_overlap == 1.0 and overlap.passes_threshold, (
        "Exact and approximate top-2 sets should fully overlap in this fixture."
    )
    missed_topk = topk_overlap_report(
        exact,
        t.tensor([0.9, 0.1, 0.0, 0.8]),
        top_k=2,
        min_overlap=1.0,
    )
    assert not missed_topk.passes_threshold, (
        "Top-k overlap should fail when approximate scores prioritize different components."
    )
    print(
        "All tests in `test_exact_vs_approx_reports_measure_correlation_and_topk_overlap` passed!"
    )


def test_runtime_and_false_negative_reports_enforce_accountability(
    runtime_improvement_report: Callable | None = None,
    false_negative_report: Callable | None = None,
):
    solutions = _solutions()
    runtime_improvement_report = runtime_improvement_report or solutions.runtime_improvement_report
    false_negative_report = false_negative_report or solutions.false_negative_report
    runtime = runtime_improvement_report(
        exact_runtime_s=10.0,
        approx_runtime_s=2.0,
        min_speedup=4.0,
    )
    expected_runtime = reference.runtime_improvement_report(
        exact_runtime_s=10.0,
        approx_runtime_s=2.0,
        min_speedup=4.0,
    )
    _assert_report_close(runtime, expected_runtime, msg="Runtime improvement report")
    assert runtime.speedup == 5.0 and runtime.passes_speedup, (
        "Approximate patching should report measured speedup over exact patching."
    )
    try:
        runtime_improvement_report(
            exact_runtime_s=10.0,
            approx_runtime_s=0.0,
            min_speedup=4.0,
        )
    except ValueError as exc:
        assert "positive" in str(exc), (
            "Nonpositive runtimes should raise a helpful ValueError."
        )
    else:
        raise AssertionError("Zero approximate runtime should raise ValueError.")

    exact = t.tensor([0.1, 0.9, 0.8])
    approx = t.tensor([0.1, 0.2, 0.7])
    false_negative = false_negative_report(
        exact,
        approx,
        exact_threshold=0.75,
        approx_threshold=0.5,
        documentation={1: "Approximation misses a nonlinear interaction."},
    )
    expected_false_negative = reference.false_negative_report(
        exact,
        approx,
        exact_threshold=0.75,
        approx_threshold=0.5,
        documentation={1: "Approximation misses a nonlinear interaction."},
    )
    _assert_report_close(false_negative, expected_false_negative, msg="False-negative report")
    assert false_negative.false_negative_indices == (1,), (
        "False-negative report should identify exact-important components missed by the approximation."
    )
    assert false_negative.documented, (
        "False negatives should pass only when every missed component has a nonempty note."
    )
    undocumented = false_negative_report(
        exact,
        approx,
        exact_threshold=0.75,
        approx_threshold=0.5,
        documentation={},
    )
    assert not undocumented.documented, (
        "Missing documentation for false negatives should fail the accountability check."
    )
    print(
        "All tests in `test_runtime_and_false_negative_reports_enforce_accountability` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["attribution_scores"] == [1.0, 3.0], (
        "Notebook contract should include first-order attribution patch scores."
    )
    assert result["integrated_gradients"] == [1.0, 3.0], (
        "Notebook contract should include integrated-gradient patch scores."
    )
    assert result["edge_scores"] == [[3.0, 0.0], [0.0, 8.0]], (
        "Notebook contract should include the EAP-style edge matrix."
    )
    assert result["correlation"]["passes_threshold"], (
        "Notebook contract should compare approximate scores against exact scores by correlation."
    )
    assert result["topk_overlap"]["passes_threshold"], (
        "Notebook contract should compare exact and approximate top-k components."
    )
    assert result["runtime"]["passes_speedup"], (
        "Notebook contract should include measured runtime improvement."
    )
    assert result["false_negative"]["documented"], (
        "Notebook contract should document any exact-important missed components."
    )
    print("All tests in `test_notebook_contract` passed!")
