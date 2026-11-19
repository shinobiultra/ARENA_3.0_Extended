# %%
"""Reference solutions for [8.4] Circuit Tracing with Attribution Graphs."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t

from arena_ext import circuit_tracing as reference_impl

chapter = "chapter8_automated_circuits"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"


# %%
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


def _require_finite_tensor(name: str, value: t.Tensor) -> None:
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not t.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")


def _require_finite_scalar(name: str, value: float) -> float:
    numeric = float(value)
    if not t.isfinite(t.tensor(numeric)):
        raise ValueError(f"{name} must be finite.")
    return numeric


def edge_attribution_scores(
    upstream_activation_delta: t.Tensor,
    downstream_gradients: t.Tensor,
) -> t.Tensor:
    """Return EAP-style upstream-by-downstream edge attribution scores."""

    if upstream_activation_delta.ndim != 2 or downstream_gradients.ndim != 2:
        raise ValueError("inputs must have shape (components, d_model).")
    _require_finite_tensor("upstream_activation_delta", upstream_activation_delta)
    _require_finite_tensor("downstream_gradients", downstream_gradients)
    if upstream_activation_delta.shape[0] == 0 or downstream_gradients.shape[0] == 0:
        raise ValueError("inputs must include at least one component.")
    if upstream_activation_delta.shape[-1] != downstream_gradients.shape[-1]:
        raise ValueError("upstream and downstream hidden dimensions must match.")
    return upstream_activation_delta.float() @ downstream_gradients.float().T


def build_local_attribution_graph(
    edge_scores: t.Tensor,
    node_names: list[str],
    *,
    top_k: int = 3,
) -> LocalAttributionGraph:
    """Build a top-k directed attribution graph from an edge-score matrix."""

    if edge_scores.ndim != 2 or edge_scores.shape[0] != edge_scores.shape[1]:
        raise ValueError("edge_scores must be a square matrix.")
    _require_finite_tensor("edge_scores", edge_scores)
    if edge_scores.shape[0] != len(node_names):
        raise ValueError("edge_scores and node_names must align.")
    if not node_names:
        raise ValueError("node_names must be non-empty.")
    if any(not name.strip() for name in node_names):
        raise ValueError("node_names must not contain blank names.")
    if len(set(node_names)) != len(node_names):
        raise ValueError("node_names must be unique.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    flat_scores = edge_scores.flatten().float()
    k = min(top_k, flat_scores.numel())
    top_values, top_indices = flat_scores.topk(k=k)
    edges = []
    num_nodes = edge_scores.shape[0]
    for value, flat_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
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

    full_metric = _require_finite_scalar("full_metric", full_metric)
    corrupt_metric = _require_finite_scalar("corrupt_metric", corrupt_metric)
    graph_metric = _require_finite_scalar("graph_metric", graph_metric)
    min_explained_fraction = _require_finite_scalar(
        "min_explained_fraction",
        min_explained_fraction,
    )
    if min_explained_fraction < 0:
        raise ValueError("min_explained_fraction must be non-negative.")
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

    original_metric = _require_finite_scalar("original_metric", original_metric)
    perturbed_metric = _require_finite_scalar("perturbed_metric", perturbed_metric)
    min_metric_drop = _require_finite_scalar("min_metric_drop", min_metric_drop)
    if min_metric_drop < 0:
        raise ValueError("min_metric_drop must be non-negative.")
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

    graph_metric = _require_finite_scalar("graph_metric", graph_metric)
    alternative_metric = _require_finite_scalar("alternative_metric", alternative_metric)
    min_margin = _require_finite_scalar("min_margin", min_margin)
    if min_margin < 0:
        raise ValueError("min_margin must be non-negative.")
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

    baseline_metric = _require_finite_scalar("baseline_metric", baseline_metric)
    counterfactual_metric = _require_finite_scalar(
        "counterfactual_metric",
        counterfactual_metric,
    )
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
    """Find the highest-scoring directed path from source to target."""

    if max_depth <= 0:
        raise ValueError("max_depth must be positive.")
    if not graph.nodes:
        raise ValueError("graph must contain at least one node.")
    if any(not node.strip() for node in graph.nodes):
        raise ValueError("graph node names must not be blank.")
    if len(set(graph.nodes)) != len(graph.nodes):
        raise ValueError("graph node names must be unique.")
    if source not in graph.nodes:
        raise ValueError("source must be a graph node.")
    if target not in graph.nodes:
        raise ValueError("target must be a graph node.")

    adjacency: dict[str, list[CircuitTraceEdge]] = {node: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in graph.nodes or edge.target not in graph.nodes:
            raise ValueError("graph edges must connect declared graph nodes.")
        _require_finite_scalar("edge.score", edge.score)
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


# %%
PlantedAttributionCircuit = reference_impl.PlantedAttributionCircuit
ExactPathPatchReport = reference_impl.ExactPathPatchReport
GraphControlReport = reference_impl.GraphControlReport

make_planted_attribution_circuit = reference_impl.make_planted_attribution_circuit
planted_circuit_activations = reference_impl.planted_circuit_activations
planted_circuit_metric = reference_impl.planted_circuit_metric
exact_edge_ablation_effects = reference_impl.exact_edge_ablation_effects
integrated_edge_attribution_scores = reference_impl.integrated_edge_attribution_scores
threshold_attribution_graph = reference_impl.threshold_attribution_graph
graph_edge_mask = reference_impl.graph_edge_mask
graph_metric_from_edges = reference_impl.graph_metric_from_edges
exact_path_patch_report = reference_impl.exact_path_patch_report
same_size_random_graph = reference_impl.same_size_random_graph
reversed_edge_graph = reference_impl.reversed_edge_graph
graph_control_report = reference_impl.graph_control_report
run_planted_graph_signature_result = reference_impl.run_planted_graph_signature_result


def graph_smoke_test() -> dict:
    edge_scores = t.tensor(
        [
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0],
        ]
    )
    node_names = ["feature:fact", "transcoder:mlp", "logit:Paris"]
    graph = build_local_attribution_graph(edge_scores, node_names, top_k=2)
    return {
        "nodes": list(graph.nodes),
        "edges": [
            (edge.source, edge.target, round(edge.score, 6))
            for edge in graph.edges
        ],
    }


def graph_metric_smoke_test() -> dict:
    return graph_metric_report(
        full_metric=3.0,
        corrupt_metric=0.0,
        graph_metric=2.4,
        min_explained_fraction=0.75,
    ).__dict__


def path_perturbation_smoke_test() -> dict:
    return path_perturbation_report(
        original_metric=2.4,
        perturbed_metric=0.7,
        min_metric_drop=1.0,
    ).__dict__


def alternative_graph_smoke_test() -> dict:
    return alternative_graph_baseline_report(
        graph_metric=2.4,
        alternative_metric=1.0,
        min_margin=1.0,
    ).__dict__


def counterfactual_smoke_test() -> dict:
    return graph_summary_counterfactual_report(
        predicted_direction="decrease",
        baseline_metric=2.4,
        counterfactual_metric=0.8,
    ).__dict__


def path_smoke_test() -> dict:
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
    return top_attribution_path(
        graph,
        source="feature:subject",
        target="logit:Paris",
        max_depth=3,
    ).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    signature = run_planted_graph_signature_result()
    return {
        "claim": signature["claim"],
        "accepted": signature["accepted"],
        "top_path": signature["top_path"],
        "graph_edges": signature["graph_edges"],
        "metrics": signature["metrics"],
        "threshold_sweep": signature["threshold_sweep"],
        "legacy_preflight_contract": {
            "graph": graph_smoke_test(),
            "metric": graph_metric_smoke_test(),
            "path_perturbation": path_perturbation_smoke_test(),
            "alternative": alternative_graph_smoke_test(),
            "counterfactual": counterfactual_smoke_test(),
            "path": path_smoke_test(),
        },
    }


if MAIN:
    print(run_smoke_test())
