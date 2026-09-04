"""Visible sub-function tests for [8.3] ACDC and Circuit Metrics."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as t


def _reference():
    from chapter8_automated_circuits.exercises.part3_acdc_circuit_metrics import solutions

    return solutions


def _fn(candidate: Callable | None, name: str) -> Callable:
    return candidate if candidate is not None else getattr(_reference(), name)


def test_toy_graph_forward_values(run_toy_graph: Callable | None = None):
    sol = _reference()
    run = _fn(run_toy_graph, "run_toy_graph")
    graph = sol.build_toy_acdc_graph()
    clean = run(graph, sol.TOY_CLEAN_INPUTS)
    corrupt = run(graph, sol.TOY_CORRUPT_INPUTS)
    assert clean.metric == pytest.approx(2.5), (
        "The clean graph should contain primary=2.0, backup=0.4, and background=0.1."
    )
    assert corrupt.metric == pytest.approx(0.1), (
        "Only the matched background path should survive in the corrupt graph."
    )
    assert clean.activations["L0M0.binding"] == pytest.approx(1.0), (
        "The product gate should be one only when both upstream signals arrive."
    )
    assert corrupt.activations["L0M0.binding"] == pytest.approx(0.0), (
        "The corrupt product gate should be zero when both task signals are absent."
    )
    print("All tests in `test_toy_graph_forward_values` passed!")


def test_toy_graph_rejects_bad_inputs(run_toy_graph: Callable | None = None):
    sol = _reference()
    run = _fn(run_toy_graph, "run_toy_graph")
    graph = sol.build_toy_acdc_graph()
    missing = dict(sol.TOY_CLEAN_INPUTS)
    missing.pop("tokens.position")
    with pytest.raises(ValueError, match="exactly"):
        run(graph, missing)
    nonfinite = dict(sol.TOY_CLEAN_INPUTS)
    nonfinite["tokens.position"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run(graph, nonfinite)
    print("All tests in `test_toy_graph_rejects_bad_inputs` passed!")


def test_edge_intervention_endpoints(run_edge_intervention: Callable | None = None):
    sol = _reference()
    intervene = _fn(run_edge_intervention, "run_edge_intervention")
    graph = sol.build_toy_acdc_graph()
    edge_names = [edge.name for edge in graph.edges]
    all_clean = intervene(
        graph, sol.TOY_CLEAN_INPUTS, sol.TOY_CORRUPT_INPUTS, edge_names
    )
    all_corrupt = intervene(
        graph, sol.TOY_CLEAN_INPUTS, sol.TOY_CORRUPT_INPUTS, []
    )
    assert all_clean.metric == pytest.approx(2.5), (
        "Keeping every edge should reproduce the fully clean graph."
    )
    assert all_corrupt.metric == pytest.approx(0.1), (
        "Removing every edge should reproduce the corrupt graph, not a zero graph."
    )
    print("All tests in `test_edge_intervention_endpoints` passed!")


def test_edge_intervention_recomputes_downstream(
    run_edge_intervention: Callable | None = None,
):
    sol = _reference()
    intervene = _fn(run_edge_intervention, "run_edge_intervention")
    graph = sol.build_toy_acdc_graph()
    edge_names = {edge.name for edge in graph.edges}
    without_name = edge_names - {"tokens.io_name -> L0H0.name_copy"}
    missing_core = intervene(
        graph, sol.TOY_CLEAN_INPUTS, sol.TOY_CORRUPT_INPUTS, without_name
    )
    without_decoy = edge_names - {"L0H2.background -> logits.io"}
    missing_decoy = intervene(
        graph, sol.TOY_CLEAN_INPUTS, sol.TOY_CORRUPT_INPUTS, without_decoy
    )
    assert missing_core.metric == pytest.approx(0.5), (
        "Removing one input to the product gate should collapse the full primary path, "
        "leaving only backup=0.4 and background=0.1."
    )
    assert missing_decoy.metric == pytest.approx(2.5), (
        "The background edge is a matched decoy, so replacing it with corrupt should do nothing."
    )
    assert missing_core.activations["L0M0.binding"] == pytest.approx(0.0), (
        "Downstream nodes must be recomputed after the edge intervention."
    )
    print("All tests in `test_edge_intervention_recomputes_downstream` passed!")


def test_normalized_recovery(normalized_recovery: Callable | None = None):
    recovery = _fn(normalized_recovery, "normalized_recovery")
    assert recovery(clean_metric=2.5, corrupt_metric=0.1, metric=2.5) == pytest.approx(1.0), (
        "The clean endpoint must normalize to one."
    )
    assert recovery(clean_metric=2.5, corrupt_metric=0.1, metric=0.1) == pytest.approx(0.0), (
        "The corrupt endpoint must normalize to zero."
    )
    assert recovery(clean_metric=2.5, corrupt_metric=0.1, metric=0.5) == pytest.approx(1 / 6), (
        "The backup-only circuit should recover one sixth of the clean-corrupt gap."
    )
    with pytest.raises(ValueError, match="differ"):
        recovery(clean_metric=1.0, corrupt_metric=1.0, metric=1.0)
    print("All tests in `test_normalized_recovery` passed!")


def test_one_shot_scores_expose_interaction_failure(
    one_shot_insertion_scores: Callable | None = None,
):
    sol = _reference()
    score = _fn(one_shot_insertion_scores, "one_shot_insertion_scores")
    graph = sol.build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    scores = score(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    assert set(scores) == set(edge_names), "Every named edge should receive a score."
    assert max(abs(value) for value in scores.values()) < 1e-9, (
        "Every one-shot insertion should fail because no single edge completes either path."
    )
    print("All tests in `test_one_shot_scores_expose_interaction_failure` passed!")


def test_initial_order_puts_decoys_first(initial_deletion_order: Callable | None = None):
    sol = _reference()
    rank = _fn(initial_deletion_order, "initial_deletion_order")
    graph = sol.build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    order = rank(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    assert set(order[:2]) == set(graph.decoy_edges), (
        "Matched decoy edges should be the least damaging full-circuit deletions."
    )
    assert set(order) == set(edge_names), "The order must retain every candidate exactly once."
    print("All tests in `test_initial_order_puts_decoys_first` passed!")


def test_greedy_acdc_recovers_ground_truth(greedy_acdc: Callable | None = None):
    sol = _reference()
    search = _fn(greedy_acdc, "greedy_acdc")
    graph = sol.build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    order = sol.initial_deletion_order(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    result = search(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        threshold=0.1,
        order=order,
    )
    assert set(result.kept_edges) == set(graph.ground_truth_edges), (
        "Greedy recomputation should remove both decoys and retain all eight causal edges."
    )
    assert result.recovery == pytest.approx(1.0), "The exact circuit should be fully faithful."
    decisions = {step.edge: step.decision for step in result.steps}
    assert all(decisions[edge] == "remove" for edge in graph.decoy_edges), (
        "Both matched background edges should be deleted."
    )
    assert all(decisions[edge] == "keep" for edge in graph.ground_truth_edges), (
        "Every ground-truth edge should be necessary at threshold 0.1."
    )
    print("All tests in `test_greedy_acdc_recovers_ground_truth` passed!")


def test_greedy_acdc_recomputes_after_each_removal(greedy_acdc: Callable | None = None):
    sol = _reference()
    search = _fn(greedy_acdc, "greedy_acdc")
    graph = sol.build_toy_acdc_graph()
    base_evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    order = sol.initial_deletion_order(
        edge_names,
        base_evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    seen: list[frozenset[str]] = []

    def recording_evaluator(active: frozenset[str]) -> float:
        seen.append(active)
        return base_evaluator(active)

    search(
        edge_names,
        recording_evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        threshold=0.1,
        order=order,
    )
    first_decoy, second_decoy = order[:2]
    later_trials = seen[2:]
    assert any(first_decoy not in active for active in later_trials), (
        "Later candidates must be evaluated after the first accepted deletion."
    )
    assert any(
        first_decoy not in active and second_decoy not in active for active in later_trials
    ), "The evaluator should see the cumulatively pruned graph, not repeated full-graph trials."
    print("All tests in `test_greedy_acdc_recomputes_after_each_removal` passed!")


def test_greedy_acdc_rejects_invalid_search(greedy_acdc: Callable | None = None):
    sol = _reference()
    search = _fn(greedy_acdc, "greedy_acdc")
    graph = sol.build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    with pytest.raises(ValueError, match="nonnegative"):
        search(
            edge_names,
            evaluator,
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            threshold=-0.1,
        )
    with pytest.raises(ValueError, match="permutation"):
        search(
            edge_names,
            evaluator,
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            threshold=0.1,
            order=edge_names[:-1],
        )
    print("All tests in `test_greedy_acdc_rejects_invalid_search` passed!")


def test_threshold_sweep_shows_two_failure_cliffs(threshold_sweep: Callable | None = None):
    sol = _reference()
    sweep_fn = _fn(threshold_sweep, "threshold_sweep")
    graph = sol.build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    order = sol.initial_deletion_order(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    points = sweep_fn(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        thresholds=(0.1, 0.17, 0.84),
        ground_truth_edges=graph.ground_truth_edges,
        order=order,
    )
    assert [point.circuit_size for point in points] == [8, 6, 0], (
        "The backup path should disappear near 1/6 and the primary path near 5/6."
    )
    assert [point.recovery for point in points] == pytest.approx([1.0, 5 / 6, 0.0]), (
        "Threshold curves should reveal the backup and primary contribution scales."
    )
    print("All tests in `test_threshold_sweep_shows_two_failure_cliffs` passed!")


def test_circuit_metrics_are_distinct(evaluate_circuit_metrics: Callable | None = None):
    sol = _reference()
    metric_fn = _fn(evaluate_circuit_metrics, "evaluate_circuit_metrics")
    graph = sol.build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    report = metric_fn(
        edge_names,
        graph.ground_truth_edges,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    assert report.circuit_recovery == pytest.approx(1.0), "Faithfulness should be exact."
    assert report.min_edge_damage == pytest.approx(1 / 6), (
        "The weakest necessary edge should be one of the backup-path edges."
    )
    assert report.max_omitted_gain == pytest.approx(0.0), (
        "Adding either matched decoy should provide no gain."
    )
    assert report.passes_faithfulness, "The exact circuit should pass faithfulness."
    assert report.passes_minimality, "Every retained edge should pass edgewise minimality."
    assert report.passes_completeness, "No omitted decoy should improve the circuit."
    print("All tests in `test_circuit_metrics_are_distinct` passed!")


def test_circuit_metrics_reject_incomplete_circuit(
    evaluate_circuit_metrics: Callable | None = None,
):
    sol = _reference()
    metric_fn = _fn(evaluate_circuit_metrics, "evaluate_circuit_metrics")
    graph = sol.build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    primary_only = graph.ground_truth_edges[:6]
    report = metric_fn(
        edge_names,
        primary_only,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        min_faithfulness=0.95,
        max_omitted_gain=0.05,
    )
    assert report.circuit_recovery == pytest.approx(5 / 6), (
        "The primary-only graph should expose the missing backup contribution."
    )
    assert not report.passes_faithfulness, "A high faithfulness threshold should reject it."
    assert not report.passes_completeness, "Adding a missing backup edge should reveal incompleteness."
    print("All tests in `test_circuit_metrics_reject_incomplete_circuit` passed!")


def test_same_size_controls_are_exact(same_size_circuit_report: Callable | None = None):
    sol = _reference()
    compare = _fn(same_size_circuit_report, "same_size_circuit_report")
    graph = sol.build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = sol.make_toy_evaluator(graph)
    edge_names = [edge.name for edge in graph.edges]
    report = compare(
        edge_names,
        graph.ground_truth_edges,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    assert report.num_circuits == 45, "Ten choose eight should enumerate 45 same-size circuits."
    assert report.discovered_rank == 1, "The ground-truth circuit should be uniquely best."
    assert report.exact_empirical_pvalue == pytest.approx(1 / 45), (
        "Exactly one of the 45 circuits should preserve the complete behavior."
    )
    assert report.control_best_recovery == pytest.approx(5 / 6), (
        "The strongest wrong circuit should miss only the weak backup path."
    )
    assert report.control_mean_recovery < 0.2, (
        "Most same-size wrong circuits should break the nonlinear primary path."
    )
    print("All tests in `test_same_size_controls_are_exact` passed!")


def test_toy_ood_preserves_exact_circuit(evaluate_toy_ood: Callable | None = None):
    sol = _reference()
    evaluate = _fn(evaluate_toy_ood, "evaluate_toy_ood")
    graph = sol.build_toy_acdc_graph()
    report = evaluate(graph, graph.ground_truth_edges)
    assert len(report.recoveries) == 3, "All held-out signal regimes should be evaluated."
    assert report.worst_recovery == pytest.approx(1.0), (
        "The exact circuit should remain exact under held-out signal scales."
    )
    assert report.passes_ood, "The known circuit should pass the held-out toy regimes."
    print("All tests in `test_toy_ood_preserves_exact_circuit` passed!")


def test_answer_logit_diff(answer_logit_diff: Callable | None = None):
    metric = _fn(answer_logit_diff, "answer_logit_diff")
    logits = t.tensor([[[0.0, 3.0, -1.0], [0.0, 5.0, 1.0]]])
    assert metric(logits, positive_token_id=1, negative_token_id=2) == pytest.approx(4.0), (
        "The metric should use the final position and return positive minus negative."
    )
    with pytest.raises(ValueError, match="differ"):
        metric(logits, positive_token_id=1, negative_token_id=1)
    print("All tests in `test_answer_logit_diff` passed!")


def test_patch_selected_head_results(patch_selected_head_results: Callable | None = None):
    patch = _fn(patch_selected_head_results, "patch_selected_head_results")
    corrupt = t.zeros(1, 2, 3, 4)
    clean = t.arange(24, dtype=t.float32).reshape(1, 2, 3, 4)
    result = patch(corrupt, clean, [0, 2])
    t.testing.assert_close(result[:, :, [0, 2]], clean[:, :, [0, 2]])
    t.testing.assert_close(result[:, :, 1], corrupt[:, :, 1])
    assert corrupt.count_nonzero().item() == 0, "The hook helper must not mutate its input tensor."
    print("All tests in `test_patch_selected_head_results` passed!")


def test_patch_selected_head_results_rejects_bad_heads(
    patch_selected_head_results: Callable | None = None,
):
    patch = _fn(patch_selected_head_results, "patch_selected_head_results")
    corrupt = t.zeros(1, 2, 3, 4)
    clean = t.ones_like(corrupt)
    with pytest.raises(ValueError, match="duplicates"):
        patch(corrupt, clean, [1, 1])
    with pytest.raises(ValueError, match="out of range"):
        patch(corrupt, clean, [3])
    with pytest.raises(ValueError, match="matching"):
        patch(corrupt, clean[:, :, :2], [1])
    print("All tests in `test_patch_selected_head_results_rejects_bad_heads` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run = _fn(run_smoke_test, "run_smoke_test")
    report = run(cpu=True)
    assert report["accepted"] is True, "The exact toy ACDC study should satisfy every gate."
    assert set(report["acdc"]["kept_edges"]) == set(report["ground_truth_edges"]), (
        "The notebook contract must expose exact circuit recovery, not only passing metrics."
    )
    assert report["same_size_random"]["discovered_rank"] == 1, (
        "The discovered circuit should beat every same-size alternative."
    )
    assert report["ood"]["passes_ood"] is True, (
        "The notebook contract should include held-out toy regimes."
    )
    print("All tests in `test_notebook_contract` passed!")
