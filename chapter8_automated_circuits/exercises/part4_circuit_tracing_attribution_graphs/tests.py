import json
from collections.abc import Callable
from pathlib import Path

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


def test_edge_attribution_scores_rejects_degenerate_inputs(
    edge_attribution_scores: Callable | None = None,
):
    edge_attribution_scores = edge_attribution_scores or _solutions().edge_attribution_scores
    try:
        edge_attribution_scores(t.empty(0, 2), t.ones(1, 2))
    except ValueError as exc:
        assert "non-empty" in str(exc) or "component" in str(exc), (
            "Empty EAP component tensors should raise a helpful non-empty/component error."
        )
    else:
        raise AssertionError("Empty upstream components should raise ValueError.")
    try:
        edge_attribution_scores(t.tensor([[float("nan"), 0.0]]), t.ones(1, 2))
    except ValueError as exc:
        assert "finite" in str(exc), (
            "Non-finite EAP activations should raise a helpful finite-value error."
        )
    else:
        raise AssertionError("Non-finite upstream activations should raise ValueError.")
    print("All tests in `test_edge_attribution_scores_rejects_degenerate_inputs` passed!")


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


def test_build_local_attribution_graph_rejects_bad_nodes_or_scores(
    build_local_attribution_graph: Callable | None = None,
):
    build_local_attribution_graph = (
        build_local_attribution_graph or _solutions().build_local_attribution_graph
    )
    bad_scores = t.tensor([[0.0, float("inf")], [0.0, 0.0]])
    for scores, names, expected in [
        (bad_scores, ["a", "b"], "finite"),
        (t.zeros(2, 2), ["a", ""], "blank"),
        (t.zeros(2, 2), ["a", "a"], "unique"),
    ]:
        try:
            build_local_attribution_graph(scores, names, top_k=1)
        except ValueError as exc:
            assert expected in str(exc), (
                f"Bad graph inputs should raise a helpful error containing {expected!r}."
            )
        else:
            raise AssertionError(f"Expected ValueError containing {expected!r}.")
    print("All tests in `test_build_local_attribution_graph_rejects_bad_nodes_or_scores` passed!")


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


def test_metric_reports_reject_nonfinite_or_negative_thresholds(
    graph_metric_report: Callable | None = None,
    path_perturbation_report: Callable | None = None,
    alternative_graph_baseline_report: Callable | None = None,
):
    solutions = _solutions()
    graph_metric_report = graph_metric_report or solutions.graph_metric_report
    path_perturbation_report = (
        path_perturbation_report or solutions.path_perturbation_report
    )
    alternative_graph_baseline_report = (
        alternative_graph_baseline_report or solutions.alternative_graph_baseline_report
    )
    for call, expected in [
        (
            lambda: graph_metric_report(
                full_metric=float("nan"),
                corrupt_metric=0.0,
                graph_metric=1.0,
            ),
            "finite",
        ),
        (
            lambda: graph_metric_report(
                full_metric=2.0,
                corrupt_metric=0.0,
                graph_metric=1.0,
                min_explained_fraction=-0.1,
            ),
            "non-negative",
        ),
        (
            lambda: path_perturbation_report(
                original_metric=2.0,
                perturbed_metric=1.0,
                min_metric_drop=-1.0,
            ),
            "non-negative",
        ),
        (
            lambda: alternative_graph_baseline_report(
                graph_metric=2.0,
                alternative_metric=1.0,
                min_margin=-1.0,
            ),
            "non-negative",
        ),
    ]:
        try:
            call()
        except ValueError as exc:
            assert expected in str(exc), (
                f"Metric report validators should raise an error containing {expected!r}."
            )
        else:
            raise AssertionError(f"Expected ValueError containing {expected!r}.")
    print("All tests in `test_metric_reports_reject_nonfinite_or_negative_thresholds` passed!")


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


def test_top_attribution_path_rejects_invalid_graphs(
    top_attribution_path: Callable | None = None,
):
    top_attribution_path = top_attribution_path or _solutions().top_attribution_path
    solutions = _solutions()
    cases = [
        (
            solutions.LocalAttributionGraph(
                nodes=("a", "b"),
                edges=(solutions.CircuitTraceEdge("a", "missing", 1.0),),
            ),
            "connect",
        ),
        (
            solutions.LocalAttributionGraph(
                nodes=("a", "b"),
                edges=(solutions.CircuitTraceEdge("a", "b", float("nan")),),
            ),
            "finite",
        ),
        (
            solutions.LocalAttributionGraph(nodes=("a", "a"), edges=()),
            "unique",
        ),
    ]
    for graph, expected in cases:
        try:
            top_attribution_path(graph, source="a", target="b")
        except ValueError as exc:
            assert expected in str(exc), (
                f"Invalid attribution graphs should raise an error containing {expected!r}."
            )
        else:
            raise AssertionError(f"Expected ValueError containing {expected!r}.")
    print("All tests in `test_top_attribution_path_rejects_invalid_graphs` passed!")


def test_planted_circuit_has_meaningful_component_nodes(
    make_planted_attribution_circuit: Callable | None = None,
    planted_circuit_metric: Callable | None = None,
):
    solutions = _solutions()
    make_planted_attribution_circuit = (
        make_planted_attribution_circuit or solutions.make_planted_attribution_circuit
    )
    planted_circuit_metric = planted_circuit_metric or solutions.planted_circuit_metric
    circuit = make_planted_attribution_circuit()
    node_names = circuit.node_names
    assert any(name.startswith("feature:") for name in node_names), (
        "The planted graph should expose feature nodes, not only positions."
    )
    assert any(name.startswith("head:") for name in node_names), (
        "The planted graph should expose head-like routing nodes."
    )
    assert any(name.startswith("mlp:") for name in node_names), (
        "The planted graph should expose MLP/transcoder-like nodes."
    )
    assert any(name.startswith("logit:") for name in node_names), (
        "The planted graph should expose the final logit-difference node."
    )
    assert circuit.ground_truth_path == (
        "feature:country=France",
        "head:subject-router",
        "mlp:relation-lookup",
        "feature:capital=Paris",
        "logit:Paris-vs-Rome",
    ), "The exact theorem should name the intended top path."
    full_metric = planted_circuit_metric(circuit)
    corrupt_metric = planted_circuit_metric(circuit, inputs=circuit.corrupt_inputs)
    assert full_metric > 1.9 and corrupt_metric == 0.0, (
        "The clean graph should produce a real target metric gap."
    )
    print("All tests in `test_planted_circuit_has_meaningful_component_nodes` passed!")


def test_exact_edge_ablation_effects_identify_ground_truth_path(
    make_planted_attribution_circuit: Callable | None = None,
    exact_edge_ablation_effects: Callable | None = None,
):
    solutions = _solutions()
    make_planted_attribution_circuit = (
        make_planted_attribution_circuit or solutions.make_planted_attribution_circuit
    )
    exact_edge_ablation_effects = (
        exact_edge_ablation_effects or solutions.exact_edge_ablation_effects
    )
    circuit = make_planted_attribution_circuit()
    effects = exact_edge_ablation_effects(circuit)
    path_edges = list(zip(circuit.ground_truth_path[:-1], circuit.ground_truth_path[1:]))
    node_to_index = {node: index for index, node in enumerate(circuit.node_names)}
    for source, target in path_edges:
        assert effects[node_to_index[source], node_to_index[target]] > 1.7, (
            f"The exact ablation effect for {source} -> {target} should be large."
        )
    shortcut_effect = effects[
        node_to_index["mlp:relation-lookup"],
        node_to_index["logit:Paris-vs-Rome"],
    ]
    assert 0.15 < shortcut_effect < 0.35, (
        "The direct shortcut should be visible but much weaker than the planted path."
    )
    distractor_effects = [
        effects[node_to_index[source], node_to_index[target]].item()
        for source, target in circuit.distractor_edges
        if source != "mlp:relation-lookup"
    ]
    assert max(distractor_effects) < 0.05, (
        "Syntax/style distractor edges should not explain the target metric."
    )
    print("All tests in `test_exact_edge_ablation_effects_identify_ground_truth_path` passed!")


def test_integrated_edge_scores_recover_top_path_edges(
    make_planted_attribution_circuit: Callable | None = None,
    integrated_edge_attribution_scores: Callable | None = None,
    threshold_attribution_graph: Callable | None = None,
    top_attribution_path: Callable | None = None,
):
    solutions = _solutions()
    make_planted_attribution_circuit = (
        make_planted_attribution_circuit or solutions.make_planted_attribution_circuit
    )
    integrated_edge_attribution_scores = (
        integrated_edge_attribution_scores or solutions.integrated_edge_attribution_scores
    )
    threshold_attribution_graph = threshold_attribution_graph or solutions.threshold_attribution_graph
    top_attribution_path = top_attribution_path or solutions.top_attribution_path
    circuit = make_planted_attribution_circuit()
    scores = integrated_edge_attribution_scores(circuit, ig_steps=16)
    graph = threshold_attribution_graph(scores, circuit.node_names, min_abs_score=0.25)
    path = top_attribution_path(
        graph,
        source=circuit.source_node,
        target=circuit.target_node,
        max_depth=4,
    )
    assert path.reaches_target, "The thresholded attribution graph should reach the logit node."
    assert path.path == circuit.ground_truth_path, (
        "IG edge scores should recover the planted country -> head -> MLP -> feature -> logit path."
    )
    assert len(graph.edges) == 4, (
        "The default threshold should select the four planted path edges and reject distractors."
    )
    print("All tests in `test_integrated_edge_scores_recover_top_path_edges` passed!")


def test_exact_path_patch_report_measures_causal_survival(
    make_planted_attribution_circuit: Callable | None = None,
    exact_path_patch_report: Callable | None = None,
):
    solutions = _solutions()
    make_planted_attribution_circuit = (
        make_planted_attribution_circuit or solutions.make_planted_attribution_circuit
    )
    exact_path_patch_report = exact_path_patch_report or solutions.exact_path_patch_report
    circuit = make_planted_attribution_circuit()
    report = exact_path_patch_report(circuit, circuit.ground_truth_path, min_fraction=0.8)
    assert report.top_path_survives_test, (
        "The planted path should pass faithfulness, completeness, and minimality checks."
    )
    assert report.faithfulness > 0.85, "The path-only graph should recover most of the metric."
    assert report.completeness > 0.85, "Removing the path should remove most of the metric."
    assert report.minimality > 0.85, "Every edge on the path should matter."
    assert report.path_removed_metric < 0.3, (
        "After removing the top path, only weak shortcuts/distractors should remain."
    )
    print("All tests in `test_exact_path_patch_report_measures_causal_survival` passed!")


def test_same_size_random_and_reversed_graph_controls_fail(
    make_planted_attribution_circuit: Callable | None = None,
    integrated_edge_attribution_scores: Callable | None = None,
    threshold_attribution_graph: Callable | None = None,
    graph_control_report: Callable | None = None,
):
    solutions = _solutions()
    make_planted_attribution_circuit = (
        make_planted_attribution_circuit or solutions.make_planted_attribution_circuit
    )
    integrated_edge_attribution_scores = (
        integrated_edge_attribution_scores or solutions.integrated_edge_attribution_scores
    )
    threshold_attribution_graph = threshold_attribution_graph or solutions.threshold_attribution_graph
    graph_control_report = graph_control_report or solutions.graph_control_report
    circuit = make_planted_attribution_circuit()
    scores = integrated_edge_attribution_scores(circuit, ig_steps=16)
    graph = threshold_attribution_graph(scores, circuit.node_names, min_abs_score=0.25)
    report = graph_control_report(circuit, graph, random_seed=0, min_control_margin=0.5)
    assert report.same_size_random_fails, (
        "A same-size random non-path graph should not recover the target metric."
    )
    assert report.reversed_edges_fail, (
        "A reversed-edge graph should fail because direction is part of the claim."
    )
    assert report.graph_metric > 1.7, "The selected graph should carry the planted signal."
    assert report.same_size_random_metric < 0.3, (
        "The same-size random graph should be visibly worse than the top path."
    )
    assert report.reversed_edge_metric == 0.0, (
        "Reversed DAG edges should not propagate country information to the logit."
    )
    print("All tests in `test_same_size_random_and_reversed_graph_controls_fail` passed!")


def test_planted_graph_signature_result_is_visible_and_bounded(
    run_planted_graph_signature_result: Callable | None = None,
):
    run_planted_graph_signature_result = (
        run_planted_graph_signature_result
        or _solutions().run_planted_graph_signature_result
    )
    result = run_planted_graph_signature_result(threshold=0.25, ig_steps=16, random_seed=0)
    metrics = result["metrics"]
    assert result["accepted"], "The CPU signature theorem should pass."
    assert result["top_path"] == result["ground_truth_path"], (
        "The visible signature result should name the recovered ground-truth path."
    )
    assert len(result["exact_edge_effects"]) == len(result["node_names"]), (
        "The signature result should include a notebook-plottable exact-effect heatmap."
    )
    assert len(result["attribution_scores"]) == len(result["node_names"]), (
        "The signature result should include a notebook-plottable attribution heatmap."
    )
    assert metrics["top_path_survives_test"], (
        "The signature metrics should include the exact causal path-patching check."
    )
    assert metrics["same_size_random_fails"] and metrics["reversed_edges_fail"], (
        "The signature metrics should include failing control graphs."
    )
    assert "position_5" not in " ".join(result["top_path"]), (
        "The learner-facing flagship should not be the old position_5 self-edge preflight."
    )
    assert len(result["threshold_sweep"]) >= 5, (
        "The notebook should have data for a threshold/IG play plot."
    )
    print("All tests in `test_planted_graph_signature_result_is_visible_and_bounded` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["accepted"], "The notebook smoke contract should pass the CPU theorem."
    assert result["top_path"] == [
        "feature:country=France",
        "head:subject-router",
        "mlp:relation-lookup",
        "feature:capital=Paris",
        "logit:Paris-vs-Rome",
    ], "The notebook smoke contract should expose the meaningful recovered path."
    assert result["metrics"]["top_path_survives_test"], (
        "The notebook smoke contract should include exact path-patching survival."
    )
    assert result["metrics"]["same_size_random_fails"], (
        "The notebook smoke contract should include a failing same-size random graph."
    )
    assert result["metrics"]["reversed_edges_fail"], (
        "The notebook smoke contract should include a failing reversed-edge graph."
    )
    assert len(result["threshold_sweep"]) >= 5, (
        "The notebook smoke contract should include threshold sweep data for play cells."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_notebook_learner_surface_contract():
    notebook_path = (
        Path(__file__).parent
        / "8.4_Circuit_Tracing_with_Attribution_Graphs_exercises.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") in {"markdown", "code"}
    )
    assert text.count("## Exercise ") >= 7, (
        "8.4 should expose at least seven learner exercises, not a report wrapper."
    )
    for required in [
        "By the end of this notebook",
        "Expected output",
        "Help -",
        "Solution",
        "Try It Yourself",
        "Bonus - Anomaly Hunt",
        "same-size random",
        "reversed-edge",
        "circuit_tracing_attribution_graphs_signature_result.png",
        "circuit_tracing_attribution_graphs_exact_vs_approx_heatmap.png",
        "circuit_tracing_attribution_graphs_metrics_controls.png",
    ]:
        assert required in text, f"Missing required ARENA learner-surface marker: {required}"
    assert "position_5 -> position_5" in text and "not the flagship claim" in text, (
        "The notebook must explicitly demote the old position-level preflight."
    )
    print("All tests in `test_notebook_learner_surface_contract` passed!")


def test_solution_notebook_exposes_taught_implementations():
    notebook_path = (
        Path(__file__).parent
        / "8.4_Circuit_Tracing_with_Attribution_Graphs_solutions.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for function_name in [
        "make_planted_attribution_circuit",
        "planted_circuit_activations",
        "exact_edge_ablation_effects",
        "integrated_edge_attribution_scores",
        "threshold_attribution_graph",
        "top_attribution_path",
        "exact_path_patch_report",
        "graph_control_report",
        "run_planted_graph_signature_result",
    ]:
        assert f"def {function_name}(" in code, (
            f"The solution notebook must expose `{function_name}` rather than hide it in solutions.py."
        )
    assert "solutions.run_planted_graph_signature_result" not in code, (
        "The solution notebook must generate its flagship result from its visible implementation."
    )
    print("All tests in `test_solution_notebook_exposes_taught_implementations` passed!")
