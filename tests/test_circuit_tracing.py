import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.circuit_tracing import (
        CircuitTraceEdge,
        LocalAttributionGraph,
        alternative_graph_baseline_report,
        build_local_attribution_graph,
        graph_metric_report,
        graph_summary_counterfactual_report,
        path_perturbation_report,
        top_attribution_path,
    )


def test_build_local_attribution_graph_keeps_top_edges():
    edge_scores = t.tensor(
        [
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0],
        ]
    )
    node_names = ["feature:fact", "transcoder:mlp", "logit:Paris"]

    graph = build_local_attribution_graph(edge_scores, node_names, top_k=2)

    assert graph.nodes == tuple(node_names)
    assert [(edge.source, edge.target) for edge in graph.edges] == [
        ("transcoder:mlp", "logit:Paris"),
        ("feature:fact", "transcoder:mlp"),
    ]
    assert [edge.score for edge in graph.edges] == [0.9, 0.8]


def test_build_local_attribution_graph_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="finite"):
        build_local_attribution_graph(
            t.tensor([[0.0, float("inf")], [0.0, 0.0]]),
            ["a", "b"],
            top_k=1,
        )
    with pytest.raises(ValueError, match="blank"):
        build_local_attribution_graph(t.zeros(2, 2), ["a", ""], top_k=1)
    with pytest.raises(ValueError, match="unique"):
        build_local_attribution_graph(t.zeros(2, 2), ["a", "a"], top_k=1)


def test_graph_metric_report_checks_explained_fraction():
    report = graph_metric_report(
        full_metric=3.0,
        corrupt_metric=0.0,
        graph_metric=2.4,
        min_explained_fraction=0.75,
    )

    assert report.explained_fraction == pytest.approx(0.8)
    assert report.explains_target_metric


def test_metric_reports_reject_bad_scalars_or_thresholds():
    with pytest.raises(ValueError, match="finite"):
        graph_metric_report(
            full_metric=float("nan"),
            corrupt_metric=0.0,
            graph_metric=1.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        graph_metric_report(
            full_metric=2.0,
            corrupt_metric=0.0,
            graph_metric=1.0,
            min_explained_fraction=-0.1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        path_perturbation_report(
            original_metric=2.0,
            perturbed_metric=1.0,
            min_metric_drop=-1.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        alternative_graph_baseline_report(
            graph_metric=2.0,
            alternative_metric=1.0,
            min_margin=-1.0,
        )


def test_path_perturbation_report_requires_metric_drop():
    report = path_perturbation_report(
        original_metric=2.4,
        perturbed_metric=0.7,
        min_metric_drop=1.0,
    )

    assert report.metric_drop == pytest.approx(1.7)
    assert report.top_path_survives_test


def test_alternative_graph_baseline_report_requires_margin():
    report = alternative_graph_baseline_report(
        graph_metric=2.4,
        alternative_metric=1.0,
        min_margin=1.0,
    )

    assert report.margin == pytest.approx(1.4)
    assert report.alternative_baseline_fails


def test_graph_summary_counterfactual_report_matches_direction():
    report = graph_summary_counterfactual_report(
        predicted_direction="decrease",
        baseline_metric=2.4,
        counterfactual_metric=0.8,
    )

    assert report.observed_delta == pytest.approx(-1.6)
    assert report.predicts_counterfactual


def test_top_attribution_path_recovers_multi_hop_causal_chain():
    edge_scores = t.tensor(
        [
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    node_names = ["feature:subject", "transcoder:mlp", "feature:object", "logit:Paris"]
    graph = build_local_attribution_graph(edge_scores, node_names, top_k=5)

    report = top_attribution_path(
        graph,
        source="feature:subject",
        target="logit:Paris",
        max_depth=3,
    )

    assert report.reaches_target
    assert report.path == (
        "feature:subject",
        "transcoder:mlp",
        "feature:object",
        "logit:Paris",
    )
    assert report.edge_scores == (0.8, 0.9, 0.7)
    assert report.path_score == pytest.approx(0.504)


def test_top_attribution_path_rejects_invalid_graphs():
    with pytest.raises(ValueError, match="connect"):
        top_attribution_path(
            LocalAttributionGraph(
                nodes=("a", "b"),
                edges=(CircuitTraceEdge("a", "missing", 1.0),),
            ),
            source="a",
            target="b",
        )
    with pytest.raises(ValueError, match="finite"):
        top_attribution_path(
            LocalAttributionGraph(
                nodes=("a", "b"),
                edges=(CircuitTraceEdge("a", "b", float("nan")),),
            ),
            source="a",
            target="b",
        )
    with pytest.raises(ValueError, match="unique"):
        top_attribution_path(
            LocalAttributionGraph(nodes=("a", "a"), edges=()),
            source="a",
            target="a",
        )


def test_top_attribution_path_reports_missing_target_path():
    edge_scores = t.tensor([[0.0, 0.8], [0.0, 0.0]])
    graph = build_local_attribution_graph(edge_scores, ["feature:a", "feature:b"], top_k=1)

    report = top_attribution_path(
        graph,
        source="feature:b",
        target="feature:a",
    )

    assert not report.reaches_target
    assert report.path == ()
    assert report.path_score == 0.0
