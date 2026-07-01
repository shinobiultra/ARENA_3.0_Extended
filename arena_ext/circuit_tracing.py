"""Circuit tracing utilities for attribution-graph notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t


CounterfactualDirection = Literal["increase", "decrease"]


@dataclass(frozen=True)
class CircuitTraceEdge:
    source: str
    target: str
    score: float


@dataclass(frozen=True)
class LocalAttributionGraph:
    nodes: tuple[str, ...]
    edges: tuple[CircuitTraceEdge, ...]


@dataclass(frozen=True)
class GraphMetricReport:
    full_metric: float
    corrupt_metric: float
    graph_metric: float
    explained_fraction: float
    explains_target_metric: bool


@dataclass(frozen=True)
class PathPerturbationReport:
    original_metric: float
    perturbed_metric: float
    metric_drop: float
    top_path_survives_test: bool


@dataclass(frozen=True)
class AlternativeGraphBaselineReport:
    graph_metric: float
    alternative_metric: float
    margin: float
    alternative_baseline_fails: bool


@dataclass(frozen=True)
class GraphSummaryCounterfactualReport:
    predicted_direction: CounterfactualDirection
    observed_delta: float
    predicts_counterfactual: bool


@dataclass(frozen=True)
class AttributionPathReport:
    source: str
    target: str
    path: tuple[str, ...]
    edge_scores: tuple[float, ...]
    path_score: float
    reaches_target: bool


def build_local_attribution_graph(
    edge_scores: t.Tensor,
    node_names: list[str],
    *,
    top_k: int = 3,
) -> LocalAttributionGraph:
    """Build a top-k directed attribution graph from an edge-score matrix."""

    if edge_scores.ndim != 2 or edge_scores.shape[0] != edge_scores.shape[1]:
        raise ValueError("edge_scores must be a square matrix.")
    if edge_scores.shape[0] != len(node_names):
        raise ValueError("edge_scores and node_names must align.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    flat_scores = edge_scores.flatten().float()
    k = min(top_k, flat_scores.numel())
    top_values, top_indices = flat_scores.topk(k=k)
    edges = []
    num_nodes = edge_scores.shape[0]
    for value, flat_index in zip(top_values.tolist(), top_indices.tolist()):
        source_index = int(flat_index // num_nodes)
        target_index = int(flat_index % num_nodes)
        if value == 0:
            continue
        edges.append(
            CircuitTraceEdge(
                source=node_names[source_index],
                target=node_names[target_index],
                score=round(float(value), 6),
            )
        )
    return LocalAttributionGraph(
        nodes=tuple(node_names),
        edges=tuple(edges),
    )


def graph_metric_report(
    *,
    full_metric: float,
    corrupt_metric: float,
    graph_metric: float,
    min_explained_fraction: float = 0.75,
) -> GraphMetricReport:
    """Check how much of a target metric an attribution graph explains."""

    denominator = full_metric - corrupt_metric
    if denominator == 0:
        raise ValueError("full_metric and corrupt_metric must differ.")
    explained_fraction = (graph_metric - corrupt_metric) / denominator
    return GraphMetricReport(
        full_metric=full_metric,
        corrupt_metric=corrupt_metric,
        graph_metric=graph_metric,
        explained_fraction=explained_fraction,
        explains_target_metric=explained_fraction >= min_explained_fraction,
    )


def path_perturbation_report(
    *,
    original_metric: float,
    perturbed_metric: float,
    min_metric_drop: float = 0.5,
) -> PathPerturbationReport:
    """Check whether perturbing the top graph path damages the metric."""

    metric_drop = original_metric - perturbed_metric
    return PathPerturbationReport(
        original_metric=original_metric,
        perturbed_metric=perturbed_metric,
        metric_drop=metric_drop,
        top_path_survives_test=metric_drop >= min_metric_drop,
    )


def alternative_graph_baseline_report(
    *,
    graph_metric: float,
    alternative_metric: float,
    min_margin: float = 0.5,
) -> AlternativeGraphBaselineReport:
    """Check that an alternative graph baseline performs worse."""

    margin = graph_metric - alternative_metric
    return AlternativeGraphBaselineReport(
        graph_metric=graph_metric,
        alternative_metric=alternative_metric,
        margin=margin,
        alternative_baseline_fails=margin >= min_margin,
    )


def graph_summary_counterfactual_report(
    *,
    predicted_direction: CounterfactualDirection,
    baseline_metric: float,
    counterfactual_metric: float,
) -> GraphSummaryCounterfactualReport:
    """Check whether a written graph summary predicts a counterfactual."""

    observed_delta = counterfactual_metric - baseline_metric
    if predicted_direction == "increase":
        predicts_counterfactual = observed_delta > 0
    elif predicted_direction == "decrease":
        predicts_counterfactual = observed_delta < 0
    else:
        raise ValueError("predicted_direction must be 'increase' or 'decrease'.")
    return GraphSummaryCounterfactualReport(
        predicted_direction=predicted_direction,
        observed_delta=observed_delta,
        predicts_counterfactual=predicts_counterfactual,
    )


def top_attribution_path(
    graph: LocalAttributionGraph,
    *,
    source: str,
    target: str,
    max_depth: int = 4,
) -> AttributionPathReport:
    """Find the highest-scoring directed path from source to target.

    Path score is the product of absolute edge scores, which favors paths whose
    whole chain has strong attribution rather than one isolated large edge.
    """

    if max_depth <= 0:
        raise ValueError("max_depth must be positive.")
    if source not in graph.nodes:
        raise ValueError("source must be a graph node.")
    if target not in graph.nodes:
        raise ValueError("target must be a graph node.")

    adjacency: dict[str, list[CircuitTraceEdge]] = {node: [] for node in graph.nodes}
    for edge in graph.edges:
        adjacency.setdefault(edge.source, []).append(edge)
    for edges in adjacency.values():
        edges.sort(key=lambda edge: abs(edge.score), reverse=True)

    best_path: tuple[str, ...] = ()
    best_scores: tuple[float, ...] = ()
    best_score = float("-inf")

    def search(
        current: str,
        path: tuple[str, ...],
        scores: tuple[float, ...],
        score_product: float,
    ) -> None:
        nonlocal best_path, best_scores, best_score
        if current == target and scores:
            if score_product > best_score:
                best_path = path
                best_scores = scores
                best_score = score_product
            return
        if len(scores) >= max_depth:
            return
        for edge in adjacency.get(current, []):
            if edge.target in path:
                continue
            search(
                edge.target,
                (*path, edge.target),
                (*scores, round(float(edge.score), 6)),
                score_product * abs(float(edge.score)),
            )

    search(source, (source,), (), 1.0)
    if not best_path:
        return AttributionPathReport(
            source=source,
            target=target,
            path=(),
            edge_scores=(),
            path_score=0.0,
            reaches_target=False,
        )
    return AttributionPathReport(
        source=source,
        target=target,
        path=best_path,
        edge_scores=best_scores,
        path_score=round(best_score, 6),
        reaches_target=True,
    )
