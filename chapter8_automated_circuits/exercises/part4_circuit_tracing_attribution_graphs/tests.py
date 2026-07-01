from collections.abc import Callable

import torch as t

from arena_ext import circuit_tracing as reference


def _solutions():
    from chapter8_automated_circuits.exercises.part4_circuit_tracing_attribution_graphs import (
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
            assert _normalize(actual_value) == _normalize(expected_value), (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def _normalize(value: object) -> object:
    if hasattr(value, "__dict__"):
        return {key: _normalize(item) for key, item in value.__dict__.items()}
    if isinstance(value, tuple):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def test_edge_attribution_scores_forms_position_edge_matrix(
    edge_attribution_scores: Callable | None = None,
):
    edge_attribution_scores = edge_attribution_scores or _solutions().edge_attribution_scores
    upstream_delta = t.tensor([[1.0, 0.0], [0.0, 2.0]])
    downstream_gradients = t.tensor([[3.0, 0.0], [0.0, 4.0]])
    scores = edge_attribution_scores(upstream_delta, downstream_gradients)
    assert scores.tolist() == [[3.0, 0.0], [0.0, 8.0]], (
        "EAP edge scores should multiply upstream deltas by downstream gradients."
    )
    try:
        edge_attribution_scores(upstream_delta, downstream_gradients[:, :1])
    except ValueError as exc:
        assert "hidden dimensions" in str(exc), (
            "Hidden-width mismatches should raise a helpful dimension error."
        )
    else:
        raise AssertionError("Mismatched hidden dimensions should raise ValueError.")
    print("All tests in `test_edge_attribution_scores_forms_position_edge_matrix` passed!")


def test_build_local_attribution_graph_keeps_top_directed_edges(
    build_local_attribution_graph: Callable | None = None,
):
    build_local_attribution_graph = (
        build_local_attribution_graph or _solutions().build_local_attribution_graph
    )
    edge_scores = t.tensor(
        [
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0],
        ]
    )
    node_names = ["feature:fact", "transcoder:mlp", "logit:Paris"]
    graph = build_local_attribution_graph(edge_scores, node_names, top_k=2)
    expected = reference.build_local_attribution_graph(edge_scores, node_names, top_k=2)
    _assert_report_close(graph, expected, msg="Local attribution graph")
    assert list(graph.nodes) == node_names, (
        "Graph construction should preserve the learner-visible node names."
    )
    assert graph.edges[0].source == "transcoder:mlp", (
        "The largest directed edge should be sorted first."
    )
    assert graph.edges[0].target == "logit:Paris" and graph.edges[0].score == 0.9, (
        "The top edge should point from the transcoder node to the logit node."
    )
    assert graph.edges[1].source == "feature:fact", (
        "The second edge should keep the feature-to-transcoder direction."
    )
    zero_graph = build_local_attribution_graph(t.zeros(2, 2), ["a", "b"], top_k=4)
    assert zero_graph.edges == (), "Zero-score edges should be omitted from the graph."
    try:
        build_local_attribution_graph(edge_scores[:2], node_names, top_k=2)
    except ValueError as exc:
        assert "square matrix" in str(exc), (
            "Non-square edge scores should raise a square-matrix error."
        )
    else:
        raise AssertionError("Non-square edge scores should raise ValueError.")
    print("All tests in `test_build_local_attribution_graph_keeps_top_directed_edges` passed!")


def test_graph_metric_report_measures_explained_fraction(
    graph_metric_report: Callable | None = None,
):
    graph_metric_report = graph_metric_report or _solutions().graph_metric_report
    report = graph_metric_report(
        full_metric=3.0,
        corrupt_metric=0.0,
        graph_metric=2.4,
        min_explained_fraction=0.75,
    )
    expected = reference.graph_metric_report(
        full_metric=3.0,
        corrupt_metric=0.0,
        graph_metric=2.4,
        min_explained_fraction=0.75,
    )
    _assert_report_close(report, expected, msg="Graph metric report")
    assert abs(report.explained_fraction - 0.8) < 1e-6, (
        "Graph metric explanation should be normalized by the clean-corrupt gap."
    )
    assert report.explains_target_metric, (
        "A graph explaining 80% should pass a 75% threshold."
    )
    weak = graph_metric_report(
        full_metric=3.0,
        corrupt_metric=0.0,
        graph_metric=1.0,
        min_explained_fraction=0.75,
    )
    assert not weak.explains_target_metric, (
        "A low-explanation graph should fail the target-metric threshold."
    )
    try:
        graph_metric_report(full_metric=1.0, corrupt_metric=1.0, graph_metric=1.0)
    except ValueError as exc:
        assert "must differ" in str(exc), (
            "Degenerate full/corrupt metrics should raise a denominator error."
        )
    else:
        raise AssertionError("Equal full/corrupt metrics should raise ValueError.")
    print("All tests in `test_graph_metric_report_measures_explained_fraction` passed!")


def test_path_perturbation_and_alternative_baseline_reports(
    path_perturbation_report: Callable | None = None,
    alternative_graph_baseline_report: Callable | None = None,
):
    solutions = _solutions()
    path_perturbation_report = (
        path_perturbation_report or solutions.path_perturbation_report
    )
    alternative_graph_baseline_report = (
        alternative_graph_baseline_report or solutions.alternative_graph_baseline_report
    )
    perturbation = path_perturbation_report(
        original_metric=2.4,
        perturbed_metric=0.7,
        min_metric_drop=1.0,
    )
    expected_perturbation = reference.path_perturbation_report(
        original_metric=2.4,
        perturbed_metric=0.7,
        min_metric_drop=1.0,
    )
    _assert_report_close(perturbation, expected_perturbation, msg="Path perturbation")
    assert abs(perturbation.metric_drop - 1.7) < 1e-6, (
        "Top-path perturbation damage should be original_metric - perturbed_metric."
    )
    assert perturbation.top_path_survives_test, (
        "A large metric drop should pass the top-path perturbation check."
    )
    weak_perturbation = path_perturbation_report(
        original_metric=2.4,
        perturbed_metric=2.1,
        min_metric_drop=1.0,
    )
    assert not weak_perturbation.top_path_survives_test, (
        "A graph path should fail if perturbing it barely changes the metric."
    )

    alternative = alternative_graph_baseline_report(
        graph_metric=2.4,
        alternative_metric=1.0,
        min_margin=1.0,
    )
    expected_alternative = reference.alternative_graph_baseline_report(
        graph_metric=2.4,
        alternative_metric=1.0,
        min_margin=1.0,
    )
    _assert_report_close(alternative, expected_alternative, msg="Alternative graph")
    assert abs(alternative.margin - 1.4) < 1e-6 and alternative.alternative_baseline_fails, (
        "The selected graph should beat a plausible alternative graph by the margin."
    )
    too_close = alternative_graph_baseline_report(
        graph_metric=2.4,
        alternative_metric=2.0,
        min_margin=1.0,
    )
    assert not too_close.alternative_baseline_fails, (
        "An alternative graph should not count as failed when its metric is too close."
    )
    print("All tests in `test_path_perturbation_and_alternative_baseline_reports` passed!")


def test_counterfactual_summary_report_checks_direction(
    graph_summary_counterfactual_report: Callable | None = None,
):
    graph_summary_counterfactual_report = (
        graph_summary_counterfactual_report
        or _solutions().graph_summary_counterfactual_report
    )
    report = graph_summary_counterfactual_report(
        predicted_direction="decrease",
        baseline_metric=2.4,
        counterfactual_metric=0.8,
    )
    expected = reference.graph_summary_counterfactual_report(
        predicted_direction="decrease",
        baseline_metric=2.4,
        counterfactual_metric=0.8,
    )
    _assert_report_close(report, expected, msg="Counterfactual summary")
    assert abs(report.observed_delta + 1.6) < 1e-6 and report.predicts_counterfactual, (
        "A predicted decrease should pass when the counterfactual metric goes down."
    )
    wrong_direction = graph_summary_counterfactual_report(
        predicted_direction="increase",
        baseline_metric=2.4,
        counterfactual_metric=0.8,
    )
    assert not wrong_direction.predicts_counterfactual, (
        "A summary should fail when it predicts the wrong counterfactual direction."
    )
    try:
        graph_summary_counterfactual_report(
            predicted_direction="flat",
            baseline_metric=2.4,
            counterfactual_metric=0.8,
        )
    except ValueError as exc:
        assert "increase" in str(exc) and "decrease" in str(exc), (
            "Invalid counterfactual directions should name the allowed directions."
        )
    else:
        raise AssertionError("Invalid counterfactual direction should raise ValueError.")
    print("All tests in `test_counterfactual_summary_report_checks_direction` passed!")


def test_top_attribution_path_recovers_multi_hop_chain(
    top_attribution_path: Callable | None = None,
):
    top_attribution_path = top_attribution_path or _solutions().top_attribution_path
    edge_scores = t.tensor(
        [
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    node_names = ["feature:subject", "transcoder:mlp", "feature:object", "logit:Paris"]
    graph = reference.build_local_attribution_graph(edge_scores, node_names, top_k=5)

    report = top_attribution_path(
        graph,
        source="feature:subject",
        target="logit:Paris",
        max_depth=3,
    )
    expected = reference.top_attribution_path(
        graph,
        source="feature:subject",
        target="logit:Paris",
        max_depth=3,
    )
    _assert_report_close(report, expected, msg="Attribution path")
    assert report.reaches_target, "A valid local attribution path should reach the logit node."
    assert report.path == (
        "feature:subject",
        "transcoder:mlp",
        "feature:object",
        "logit:Paris",
    ), "The report should keep the whole multi-hop path, not only the top edge."
    assert report.path_score > 0.5, (
        "Path scores should combine the edge strengths along the causal chain."
    )

    missing = top_attribution_path(
        graph,
        source="logit:Paris",
        target="feature:subject",
        max_depth=3,
    )
    assert not missing.reaches_target and missing.path == (), (
        "The report should explicitly mark missing directed paths."
    )
    print("All tests in `test_top_attribution_path_recovers_multi_hop_chain` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["graph"]["edges"][0][1] == "logit:Paris", (
        "The notebook smoke contract should include the top graph edge."
    )
    assert result["metric"]["explains_target_metric"], (
        "The notebook smoke contract should include target-metric explanation."
    )
    assert result["path_perturbation"]["top_path_survives_test"], (
        "The notebook smoke contract should include a passing path perturbation."
    )
    assert result["alternative"]["alternative_baseline_fails"], (
        "The notebook smoke contract should include a failing alternative baseline."
    )
    assert result["counterfactual"]["predicts_counterfactual"], (
        "The notebook smoke contract should include a correct counterfactual prediction."
    )
    assert result["path"]["reaches_target"], (
        "The notebook smoke contract should include a multi-hop graph path."
    )
    print("All tests in `test_notebook_contract` passed!")
