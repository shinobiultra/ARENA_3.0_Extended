# %%
"""Reference solutions for [8.3] ACDC and Circuit Metrics."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import torch as t

chapter = "chapter8_automated_circuits"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"


# %%
@dataclass(frozen=True)
class Edge:
    name: str
    sender: str
    receiver: str
    weight: float = 1.0


@dataclass(frozen=True)
class ToyCircuitGraph:
    node_order: tuple[str, ...]
    input_nodes: tuple[str, ...]
    output_node: str
    operations: Mapping[str, str]
    edges: tuple[Edge, ...]
    ground_truth_edges: tuple[str, ...]
    decoy_edges: tuple[str, ...]


@dataclass(frozen=True)
class GraphRun:
    activations: Mapping[str, float]
    metric: float
    active_edges: tuple[str, ...]


@dataclass(frozen=True)
class ACDCStep:
    edge: str
    before_recovery: float
    trial_recovery: float
    normalized_damage: float
    decision: str


@dataclass(frozen=True)
class ACDCResult:
    kept_edges: tuple[str, ...]
    removed_edges: tuple[str, ...]
    threshold: float
    recovery: float
    steps: tuple[ACDCStep, ...]


@dataclass(frozen=True)
class ThresholdSweepPoint:
    threshold: float
    circuit_size: int
    recovery: float
    exact_ground_truth: bool


@dataclass(frozen=True)
class CircuitMetricsReport:
    circuit_recovery: float
    minimality_by_edge: Mapping[str, float]
    min_edge_damage: float
    completeness_by_edge: Mapping[str, float]
    completeness_by_subset: Mapping[str, float]
    strongest_omitted_subset: tuple[str, ...]
    max_omitted_gain: float
    passes_faithfulness: bool
    passes_minimality: bool
    passes_completeness: bool


@dataclass(frozen=True)
class SameSizeCircuitReport:
    circuit_size: int
    num_circuits: int
    discovered_recovery: float
    discovered_rank: int
    exact_empirical_pvalue: float
    control_mean_recovery: float
    control_best_recovery: float
    control_recoveries: tuple[float, ...]


@dataclass(frozen=True)
class OODCircuitReport:
    recoveries: Mapping[str, float]
    worst_recovery: float
    passes_ood: bool


MetricEvaluator = Callable[[frozenset[str]], float]


# %%
def build_toy_acdc_graph() -> ToyCircuitGraph:
    """Create a nonlinear graph with a known eight-edge causal circuit.

    The primary path contains a two-input product gate. Consequently, inserting
    any one primary edge into the corrupt graph has zero effect even though
    every primary edge is necessary in the full circuit. The weaker backup path
    creates a visible threshold tradeoff. The background path is an exact decoy
    because its source is identical in clean and corrupt inputs.
    """

    primary = (
        "tokens.io_name -> L0H0.name_copy",
        "tokens.position -> L0H1.position",
        "L0H0.name_copy -> L0M0.binding",
        "L0H1.position -> L0M0.binding",
        "L0M0.binding -> L1H0.answer",
        "L1H0.answer -> logits.io",
    )
    backup = (
        "tokens.backup -> L1M0.backup",
        "L1M0.backup -> logits.io",
    )
    decoys = (
        "tokens.background -> L0H2.background",
        "L0H2.background -> logits.io",
    )
    return ToyCircuitGraph(
        node_order=(
            "tokens.io_name",
            "tokens.position",
            "tokens.backup",
            "tokens.background",
            "L0H0.name_copy",
            "L0H1.position",
            "L0H2.background",
            "L0M0.binding",
            "L1M0.backup",
            "L1H0.answer",
            "logits.io",
        ),
        input_nodes=(
            "tokens.io_name",
            "tokens.position",
            "tokens.backup",
            "tokens.background",
        ),
        output_node="logits.io",
        operations={
            "L0H0.name_copy": "sum",
            "L0H1.position": "sum",
            "L0H2.background": "sum",
            "L0M0.binding": "product",
            "L1M0.backup": "sum",
            "L1H0.answer": "sum",
            "logits.io": "sum",
        },
        edges=(
            Edge(primary[0], "tokens.io_name", "L0H0.name_copy"),
            Edge(primary[1], "tokens.position", "L0H1.position"),
            Edge(primary[2], "L0H0.name_copy", "L0M0.binding"),
            Edge(primary[3], "L0H1.position", "L0M0.binding"),
            Edge(primary[4], "L0M0.binding", "L1H0.answer"),
            Edge(primary[5], "L1H0.answer", "logits.io", 2.0),
            Edge(backup[0], "tokens.backup", "L1M0.backup"),
            Edge(backup[1], "L1M0.backup", "logits.io", 0.4),
            Edge(decoys[0], "tokens.background", "L0H2.background"),
            Edge(decoys[1], "L0H2.background", "logits.io", 0.2),
        ),
        ground_truth_edges=(*primary, *backup),
        decoy_edges=decoys,
    )


TOY_CLEAN_INPUTS = {
    "tokens.io_name": 1.0,
    "tokens.position": 1.0,
    "tokens.backup": 1.0,
    "tokens.background": 0.5,
}
TOY_CORRUPT_INPUTS = {
    "tokens.io_name": 0.0,
    "tokens.position": 0.0,
    "tokens.backup": 0.0,
    "tokens.background": 0.5,
}
TOY_OOD_INPUTS = (
    (
        "strong_name_weak_position",
        {
            "tokens.io_name": 1.2,
            "tokens.position": 0.8,
            "tokens.backup": 0.5,
            "tokens.background": 0.9,
        },
        {
            "tokens.io_name": 0.0,
            "tokens.position": 0.0,
            "tokens.backup": 0.0,
            "tokens.background": 0.9,
        },
    ),
    (
        "weak_name_strong_position",
        {
            "tokens.io_name": 0.75,
            "tokens.position": 1.25,
            "tokens.backup": 1.5,
            "tokens.background": -0.4,
        },
        {
            "tokens.io_name": 0.0,
            "tokens.position": 0.0,
            "tokens.backup": 0.0,
            "tokens.background": -0.4,
        },
    ),
    (
        "small_signal",
        {
            "tokens.io_name": 0.6,
            "tokens.position": 0.7,
            "tokens.backup": 0.4,
            "tokens.background": 0.2,
        },
        {
            "tokens.io_name": 0.0,
            "tokens.position": 0.0,
            "tokens.backup": 0.0,
            "tokens.background": 0.2,
        },
    ),
)


def _validate_graph(graph: ToyCircuitGraph) -> None:
    node_set = set(graph.node_order)
    if len(node_set) != len(graph.node_order):
        raise ValueError("node_order must contain unique names.")
    if not set(graph.input_nodes).issubset(node_set):
        raise ValueError("every input node must appear in node_order.")
    if graph.output_node not in node_set:
        raise ValueError("output_node must appear in node_order.")
    edge_names = [edge.name for edge in graph.edges]
    if len(set(edge_names)) != len(edge_names):
        raise ValueError("edge names must be unique.")
    index = {node: position for position, node in enumerate(graph.node_order)}
    for edge in graph.edges:
        if edge.sender not in node_set or edge.receiver not in node_set:
            raise ValueError(f"edge {edge.name!r} references an unknown node.")
        if index[edge.sender] >= index[edge.receiver]:
            raise ValueError(f"edge {edge.name!r} violates topological order.")
        if not t.isfinite(t.tensor(edge.weight)).item():
            raise ValueError(f"edge {edge.name!r} has a non-finite weight.")
    known = set(edge_names)
    if set(graph.ground_truth_edges) | set(graph.decoy_edges) != known:
        raise ValueError("ground-truth and decoy edges must partition the graph.")


def _validate_inputs(graph: ToyCircuitGraph, inputs: Mapping[str, float]) -> None:
    if set(inputs) != set(graph.input_nodes):
        raise ValueError("inputs must contain exactly the graph input nodes.")
    if not all(t.isfinite(t.tensor(value)).item() for value in inputs.values()):
        raise ValueError("all input values must be finite.")


def _apply_operation(operation: str, messages: Sequence[float]) -> float:
    if not messages:
        raise ValueError("non-input nodes must have at least one incoming edge.")
    if operation == "sum":
        return float(sum(messages))
    if operation == "product":
        value = 1.0
        for message in messages:
            value *= message
        return float(value)
    raise ValueError(f"unknown node operation: {operation!r}")


def run_toy_graph(graph: ToyCircuitGraph, inputs: Mapping[str, float]) -> GraphRun:
    """Run every edge of the toy graph on one set of inputs."""

    _validate_graph(graph)
    _validate_inputs(graph, inputs)
    incoming = {
        node: tuple(edge for edge in graph.edges if edge.receiver == node)
        for node in graph.node_order
    }
    activations: dict[str, float] = {name: float(inputs[name]) for name in graph.input_nodes}
    for node in graph.node_order:
        if node in graph.input_nodes:
            continue
        messages = [activations[edge.sender] * edge.weight for edge in incoming[node]]
        activations[node] = _apply_operation(graph.operations[node], messages)
    return GraphRun(
        activations=activations,
        metric=activations[graph.output_node],
        active_edges=tuple(edge.name for edge in graph.edges),
    )


def run_edge_intervention(
    graph: ToyCircuitGraph,
    clean_inputs: Mapping[str, float],
    corrupt_inputs: Mapping[str, float],
    active_edges: Sequence[str] | set[str] | frozenset[str],
) -> GraphRun:
    """Run clean inputs while replacing missing edge messages with corrupt ones."""

    _validate_graph(graph)
    _validate_inputs(graph, clean_inputs)
    _validate_inputs(graph, corrupt_inputs)
    known_edges = {edge.name for edge in graph.edges}
    active = frozenset(active_edges)
    unknown = active - known_edges
    if unknown:
        raise ValueError(f"unknown active edges: {sorted(unknown)}")
    corrupt = run_toy_graph(graph, corrupt_inputs)
    incoming = {
        node: tuple(edge for edge in graph.edges if edge.receiver == node)
        for node in graph.node_order
    }
    activations: dict[str, float] = {
        name: float(clean_inputs[name]) for name in graph.input_nodes
    }
    for node in graph.node_order:
        if node in graph.input_nodes:
            continue
        messages = []
        for edge in incoming[node]:
            sender_value = (
                activations[edge.sender]
                if edge.name in active
                else corrupt.activations[edge.sender]
            )
            messages.append(sender_value * edge.weight)
        activations[node] = _apply_operation(graph.operations[node], messages)
    ordered_active = tuple(edge.name for edge in graph.edges if edge.name in active)
    return GraphRun(activations, activations[graph.output_node], ordered_active)


def normalized_recovery(*, clean_metric: float, corrupt_metric: float, metric: float) -> float:
    """Normalize a circuit metric so corrupt is zero and clean is one."""

    values = t.tensor([clean_metric, corrupt_metric, metric], dtype=t.float64)
    if not t.isfinite(values).all():
        raise ValueError("clean, corrupt, and circuit metrics must be finite.")
    denominator = clean_metric - corrupt_metric
    if abs(denominator) < 1e-12:
        raise ValueError("clean_metric and corrupt_metric must differ.")
    return float((metric - corrupt_metric) / denominator)


def make_toy_evaluator(
    graph: ToyCircuitGraph,
    clean_inputs: Mapping[str, float] = TOY_CLEAN_INPUTS,
    corrupt_inputs: Mapping[str, float] = TOY_CORRUPT_INPUTS,
) -> tuple[MetricEvaluator, float, float]:
    clean_metric = run_toy_graph(graph, clean_inputs).metric
    corrupt_metric = run_toy_graph(graph, corrupt_inputs).metric

    def evaluate(active_edges: frozenset[str]) -> float:
        return run_edge_intervention(
            graph,
            clean_inputs,
            corrupt_inputs,
            active_edges,
        ).metric

    return evaluate, clean_metric, corrupt_metric


def _validate_edge_names(edge_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(edge_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("edge_names must contain nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError("edge_names must be unique.")
    return names


def one_shot_insertion_scores(
    edge_names: Sequence[str],
    evaluate: MetricEvaluator,
    *,
    clean_metric: float,
    corrupt_metric: float,
) -> Mapping[str, float]:
    """Score each edge alone in the otherwise corrupt graph."""

    names = _validate_edge_names(edge_names)
    return {
        edge: normalized_recovery(
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            metric=evaluate(frozenset({edge})),
        )
        for edge in names
    }


def initial_deletion_order(
    edge_names: Sequence[str],
    evaluate: MetricEvaluator,
    *,
    clean_metric: float,
    corrupt_metric: float,
) -> tuple[str, ...]:
    """Rank least damaging full-circuit deletions first."""

    names = _validate_edge_names(edge_names)
    full = frozenset(names)
    full_recovery = normalized_recovery(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        metric=evaluate(full),
    )
    damages = {}
    for edge in names:
        trial_recovery = normalized_recovery(
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            metric=evaluate(full - {edge}),
        )
        damages[edge] = full_recovery - trial_recovery
    return tuple(sorted(names, key=lambda edge: (damages[edge], names.index(edge))))


def greedy_acdc(
    edge_names: Sequence[str],
    evaluate: MetricEvaluator,
    *,
    clean_metric: float,
    corrupt_metric: float,
    threshold: float,
    order: Sequence[str] | None = None,
) -> ACDCResult:
    """Delete edges greedily, recomputing the metric after every decision."""

    names = _validate_edge_names(edge_names)
    if not t.isfinite(t.tensor(threshold)).item() or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative.")
    deletion_order = tuple(order) if order is not None else names
    if len(deletion_order) != len(names) or set(deletion_order) != set(names):
        raise ValueError("order must be a permutation of edge_names.")
    active = frozenset(names)
    current_metric = evaluate(active)
    current_recovery = normalized_recovery(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        metric=current_metric,
    )
    steps: list[ACDCStep] = []
    removed: list[str] = []
    for edge in deletion_order:
        before_recovery = current_recovery
        trial_active = active - {edge}
        trial_metric = evaluate(trial_active)
        trial_recovery = normalized_recovery(
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            metric=trial_metric,
        )
        damage = before_recovery - trial_recovery
        if damage <= threshold:
            decision = "remove"
            active = trial_active
            current_metric = trial_metric
            current_recovery = trial_recovery
            removed.append(edge)
        else:
            decision = "keep"
        steps.append(ACDCStep(edge, before_recovery, trial_recovery, damage, decision))
    kept = tuple(edge for edge in names if edge in active)
    return ACDCResult(kept, tuple(removed), threshold, current_recovery, tuple(steps))


def threshold_sweep(
    edge_names: Sequence[str],
    evaluate: MetricEvaluator,
    *,
    clean_metric: float,
    corrupt_metric: float,
    thresholds: Sequence[float],
    ground_truth_edges: Sequence[str],
    order: Sequence[str] | None = None,
) -> tuple[ThresholdSweepPoint, ...]:
    """Run the full deletion search independently at every threshold."""

    if not thresholds:
        raise ValueError("thresholds must be nonempty.")
    truth = set(ground_truth_edges)
    points = []
    for threshold in thresholds:
        result = greedy_acdc(
            edge_names,
            evaluate,
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            threshold=float(threshold),
            order=order,
        )
        points.append(
            ThresholdSweepPoint(
                float(threshold),
                len(result.kept_edges),
                result.recovery,
                set(result.kept_edges) == truth,
            )
        )
    return tuple(points)


def evaluate_circuit_metrics(
    edge_names: Sequence[str],
    circuit_edges: Sequence[str],
    evaluate: MetricEvaluator,
    *,
    clean_metric: float,
    corrupt_metric: float,
    min_faithfulness: float = 0.95,
    min_edge_damage: float = 0.05,
    max_omitted_gain: float = 0.05,
    max_completeness_subset_size: int = 2,
) -> CircuitMetricsReport:
    """Measure faithfulness, edgewise minimality, and subset completeness."""

    names = _validate_edge_names(edge_names)
    circuit = frozenset(circuit_edges)
    if not circuit or not circuit.issubset(names):
        raise ValueError("circuit_edges must be a nonempty subset of edge_names.")
    for label, value in {
        "min_faithfulness": min_faithfulness,
        "min_edge_damage": min_edge_damage,
        "max_omitted_gain": max_omitted_gain,
    }.items():
        if not t.isfinite(t.tensor(value)).item() or value < 0:
            raise ValueError(f"{label} must be finite and nonnegative.")
    if max_completeness_subset_size <= 0:
        raise ValueError("max_completeness_subset_size must be positive.")

    def recovery(active: frozenset[str]) -> float:
        return normalized_recovery(
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            metric=evaluate(active),
        )

    circuit_recovery = recovery(circuit)
    minimality = {
        edge: circuit_recovery - recovery(circuit - {edge})
        for edge in names
        if edge in circuit
    }
    omitted = tuple(edge for edge in names if edge not in circuit)
    completeness = {edge: recovery(circuit | {edge}) - circuit_recovery for edge in omitted}
    subset_rows: dict[tuple[str, ...], float] = {}
    for subset_size in range(1, min(max_completeness_subset_size, len(omitted)) + 1):
        for subset in combinations(omitted, subset_size):
            subset_rows[subset] = recovery(circuit | set(subset)) - circuit_recovery
    min_damage = min(minimality.values())
    strongest_subset = max(subset_rows, key=subset_rows.get) if subset_rows else ()
    max_gain = subset_rows.get(strongest_subset, 0.0)
    return CircuitMetricsReport(
        circuit_recovery,
        minimality,
        min_damage,
        completeness,
        {" + ".join(subset): gain for subset, gain in subset_rows.items()},
        strongest_subset,
        max_gain,
        circuit_recovery >= min_faithfulness,
        min_damage >= min_edge_damage,
        max_gain <= max_omitted_gain,
    )


def same_size_circuit_report(
    edge_names: Sequence[str],
    circuit_edges: Sequence[str],
    evaluate: MetricEvaluator,
    *,
    clean_metric: float,
    corrupt_metric: float,
) -> SameSizeCircuitReport:
    """Enumerate every same-size circuit, an exact random-circuit null."""

    names = _validate_edge_names(edge_names)
    circuit = frozenset(circuit_edges)
    if not circuit or not circuit.issubset(names):
        raise ValueError("circuit_edges must be a nonempty subset of edge_names.")
    all_rows: list[tuple[frozenset[str], float]] = []
    for candidate in combinations(names, len(circuit)):
        candidate_set = frozenset(candidate)
        recovery = normalized_recovery(
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            metric=evaluate(candidate_set),
        )
        all_rows.append((candidate_set, recovery))
    discovered = next(value for candidate, value in all_rows if candidate == circuit)
    controls = [value for candidate, value in all_rows if candidate != circuit]
    rank = 1 + sum(value > discovered + 1e-12 for _, value in all_rows)
    pvalue = sum(value >= discovered - 1e-12 for _, value in all_rows) / len(all_rows)
    return SameSizeCircuitReport(
        len(circuit),
        len(all_rows),
        discovered,
        rank,
        pvalue,
        float(sum(controls) / len(controls)) if controls else 0.0,
        float(max(controls)) if controls else discovered,
        tuple(float(value) for value in controls),
    )


def evaluate_toy_ood(
    graph: ToyCircuitGraph,
    circuit_edges: Sequence[str],
    templates: Sequence[tuple[str, Mapping[str, float], Mapping[str, float]]] = TOY_OOD_INPUTS,
    *,
    min_recovery: float = 0.95,
) -> OODCircuitReport:
    """Evaluate the exact circuit on held-out signal scales and nuisance values."""

    if not templates:
        raise ValueError("templates must be nonempty.")
    recoveries = {}
    for name, clean_inputs, corrupt_inputs in templates:
        evaluator, clean_metric, corrupt_metric = make_toy_evaluator(
            graph, clean_inputs, corrupt_inputs
        )
        recoveries[name] = normalized_recovery(
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            metric=evaluator(frozenset(circuit_edges)),
        )
    worst = min(recoveries.values())
    return OODCircuitReport(recoveries, worst, worst >= min_recovery)


def run_toy_study(threshold: float = 0.1) -> dict[str, object]:
    """Run the complete exact-ground-truth ACDC study."""

    graph = build_toy_acdc_graph()
    evaluator, clean_metric, corrupt_metric = make_toy_evaluator(graph)
    edge_names = tuple(edge.name for edge in graph.edges)
    order = initial_deletion_order(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    one_shot = one_shot_insertion_scores(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    acdc = greedy_acdc(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        threshold=threshold,
        order=order,
    )
    metrics = evaluate_circuit_metrics(
        edge_names,
        acdc.kept_edges,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    random_report = same_size_circuit_report(
        edge_names,
        acdc.kept_edges,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    sweep = threshold_sweep(
        edge_names,
        evaluator,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        thresholds=(0.0, 0.05, 0.1, 0.16, 0.17, 0.4, 0.83, 0.84),
        ground_truth_edges=graph.ground_truth_edges,
        order=order,
    )
    ood = evaluate_toy_ood(graph, acdc.kept_edges)
    exact_match = set(acdc.kept_edges) == set(graph.ground_truth_edges)
    accepted = (
        exact_match
        and max(abs(value) for value in one_shot.values()) < 1e-9
        and metrics.passes_faithfulness
        and metrics.passes_minimality
        and metrics.passes_completeness
        and random_report.discovered_rank == 1
        and ood.passes_ood
    )
    return {
        "accepted": accepted,
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
        "edge_names": edge_names,
        "ground_truth_edges": graph.ground_truth_edges,
        "decoy_edges": graph.decoy_edges,
        "one_shot_scores": dict(one_shot),
        "deletion_order": order,
        "acdc": asdict(acdc),
        "metrics": asdict(metrics),
        "same_size_random": asdict(random_report),
        "threshold_sweep": [asdict(point) for point in sweep],
        "ood": asdict(ood),
    }


# %%
TL_MODEL_NAME = "gelu-1l"
TL_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_HEAD_HOOK = "blocks.0.attn.hook_result"
TL_MLP_HOOK = "blocks.0.hook_mlp_out"
TL_PRIMARY_PAIR = ("The cat sat on the", "The bird flew over the")
TL_HELDOUT_PAIRS = (
    ("dog_vs_plane", "The dog slept on the", "The plane flew over the"),
    ("child_vs_plane", "The child sat on the", "The plane flew over the"),
    ("cup_vs_plane", "The cup sat on the", "The plane flew over the"),
)


def load_pinned_gelu1l(device: str = "cpu"):
    """Load the pinned one-layer TransformerLens model used in the real path."""

    os.environ.setdefault("BNB_CUDA_VERSION", TL_BNB_CUDA_OVERRIDE)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TL_TOKENIZER_ID,
        revision=TL_TOKENIZER_REVISION,
    )
    model = HookedTransformer.from_pretrained(
        TL_MODEL_NAME,
        device=device,
        dtype="float32",
        revision=TL_REVISION,
        tokenizer=tokenizer,
    )
    model.cfg.use_attn_result = True
    model.eval()
    return model


def answer_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    """Return positive-minus-negative logit difference at the final token."""

    if logits.ndim not in (1, 2, 3):
        raise ValueError("logits must have shape [vocab], [pos, vocab], or [batch, pos, vocab].")
    vocab_size = logits.shape[-1]
    if positive_token_id == negative_token_id:
        raise ValueError("positive and negative token ids must differ.")
    if not 0 <= positive_token_id < vocab_size or not 0 <= negative_token_id < vocab_size:
        raise ValueError("token ids must lie inside the vocabulary dimension.")
    if not t.isfinite(logits).all():
        raise ValueError("logits must be finite.")
    final_logits = logits if logits.ndim == 1 else logits[..., -1, :]
    return float(
        (final_logits[..., positive_token_id] - final_logits[..., negative_token_id])
        .float()
        .mean()
        .item()
    )


def patch_selected_head_results(
    activation: t.Tensor,
    clean_head_results: t.Tensor,
    selected_heads: Sequence[int],
) -> t.Tensor:
    """Patch selected clean head results into a corrupt hook activation."""

    if activation.ndim != 4:
        raise ValueError("head result must have shape [batch, position, head, d_model].")
    if activation.shape != clean_head_results.shape:
        raise ValueError("clean and corrupt head-result tensors must have matching shapes.")
    if not t.isfinite(activation).all() or not t.isfinite(clean_head_results).all():
        raise ValueError("head-result tensors must be finite.")
    heads = tuple(int(head) for head in selected_heads)
    if len(set(heads)) != len(heads):
        raise ValueError("selected_heads must not contain duplicates.")
    if any(head < 0 or head >= activation.shape[2] for head in heads):
        raise ValueError("selected head index is out of range.")
    patched = activation.clone()
    if heads:
        patched[:, :, list(heads), :] = clean_head_results[:, :, list(heads), :]
    return patched


def _prepare_transformerlens_task(model, clean_prompt: str, corrupt_prompt: str) -> dict[str, object]:
    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(
            f"clean and corrupt token shapes differ: {clean_tokens.shape} vs {corrupt_tokens.shape}"
        )
    with t.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda name: name in {TL_HEAD_HOOK, TL_MLP_HOOK},
        )
        corrupt_logits = model(corrupt_tokens)
    positive_token_id = int(clean_logits[0, -1].argmax().item())
    corrupt_order = corrupt_logits[0, -1].argsort(descending=True).tolist()
    negative_token_id = next(
        int(token_id) for token_id in corrupt_order if int(token_id) != positive_token_id
    )
    clean_metric = answer_logit_diff(
        clean_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    corrupt_metric = answer_logit_diff(
        corrupt_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    return {
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "clean_head_results": clean_cache[TL_HEAD_HOOK],
        "clean_mlp_output": clean_cache[TL_MLP_HOOK],
        "positive_token_id": positive_token_id,
        "negative_token_id": negative_token_id,
        "positive_token": model.to_string(positive_token_id),
        "negative_token": model.to_string(negative_token_id),
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
    }


def _make_head_evaluator(model, task: Mapping[str, object]) -> MetricEvaluator:
    names = tuple(f"L0H{head}" for head in range(model.cfg.n_heads))

    def evaluate(active_heads: frozenset[str]) -> float:
        unknown = active_heads - set(names)
        if unknown:
            raise ValueError(f"unknown head names: {sorted(unknown)}")
        head_indices = tuple(int(name.removeprefix("L0H")) for name in active_heads)

        def patch_hook(activation: t.Tensor, hook) -> t.Tensor:
            del hook
            return patch_selected_head_results(
                activation,
                task["clean_head_results"],
                head_indices,
            )

        with t.inference_mode():
            logits = model.run_with_hooks(
                task["corrupt_tokens"],
                fwd_hooks=[(TL_HEAD_HOOK, patch_hook)],
            )
        return answer_logit_diff(
            logits,
            positive_token_id=task["positive_token_id"],
            negative_token_id=task["negative_token_id"],
        )

    return evaluate


def _mlp_bottleneck_recovery(model, task: Mapping[str, object]) -> float:
    def patch_mlp(activation: t.Tensor, hook) -> t.Tensor:
        del activation, hook
        return task["clean_mlp_output"].clone()

    with t.inference_mode():
        logits = model.run_with_hooks(
            task["corrupt_tokens"],
            fwd_hooks=[(TL_MLP_HOOK, patch_mlp)],
        )
    metric = answer_logit_diff(
        logits,
        positive_token_id=task["positive_token_id"],
        negative_token_id=task["negative_token_id"],
    )
    return normalized_recovery(
        clean_metric=task["clean_metric"],
        corrupt_metric=task["corrupt_metric"],
        metric=metric,
    )


def run_transformerlens_component_study(
    device: str = "cpu",
    *,
    threshold: float = 0.05,
    max_vram_gb: float = 24.0,
) -> dict[str, object]:
    """Run honest component-level ACDC over all heads in pinned GELU-1L."""

    if device.startswith("cuda") and not t.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.startswith("cuda"):
        t.cuda.reset_peak_memory_stats()
    model = load_pinned_gelu1l(device=device)
    primary = _prepare_transformerlens_task(model, *TL_PRIMARY_PAIR)
    names = tuple(f"L0H{head}" for head in range(model.cfg.n_heads))
    evaluator = _make_head_evaluator(model, primary)
    order = initial_deletion_order(
        names,
        evaluator,
        clean_metric=primary["clean_metric"],
        corrupt_metric=primary["corrupt_metric"],
    )
    acdc = greedy_acdc(
        names,
        evaluator,
        clean_metric=primary["clean_metric"],
        corrupt_metric=primary["corrupt_metric"],
        threshold=threshold,
        order=order,
    )
    all_heads = frozenset(names)
    all_recovery = normalized_recovery(
        clean_metric=primary["clean_metric"],
        corrupt_metric=primary["corrupt_metric"],
        metric=evaluator(all_heads),
    )
    deletion_damages = {}
    for name in names:
        without = normalized_recovery(
            clean_metric=primary["clean_metric"],
            corrupt_metric=primary["corrupt_metric"],
            metric=evaluator(all_heads - {name}),
        )
        deletion_damages[name] = all_recovery - without
    random_report = same_size_circuit_report(
        names,
        acdc.kept_edges,
        evaluator,
        clean_metric=primary["clean_metric"],
        corrupt_metric=primary["corrupt_metric"],
    )
    heldout = {}
    heldout_tokens = {}
    for label, clean_prompt, corrupt_prompt in TL_HELDOUT_PAIRS:
        task = _prepare_transformerlens_task(model, clean_prompt, corrupt_prompt)
        task_evaluator = _make_head_evaluator(model, task)
        heldout[label] = normalized_recovery(
            clean_metric=task["clean_metric"],
            corrupt_metric=task["corrupt_metric"],
            metric=task_evaluator(frozenset(acdc.kept_edges)),
        )
        heldout_tokens[label] = {
            "positive": task["positive_token"],
            "negative": task["negative_token"],
        }
    mlp_recovery = _mlp_bottleneck_recovery(model, primary)
    peak_vram_gb = 0.0
    if device.startswith("cuda"):
        t.cuda.synchronize()
        peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
        if peak_vram_gb > max_vram_gb:
            raise RuntimeError(
                f"Peak VRAM {peak_vram_gb:.3f} GB exceeds budget {max_vram_gb:.3f} GB."
            )
    accepted = (
        acdc.kept_edges == ("L0H0", "L0H4", "L0H6")
        and acdc.recovery >= 0.9
        and random_report.discovered_rank == 1
        and min(heldout.values()) >= 0.75
        and mlp_recovery >= 0.95
    )
    return {
        "accepted": accepted,
        "device": device,
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "model_name": TL_MODEL_NAME,
        "hf_model_id": TL_HF_ID,
        "hf_revision": TL_REVISION,
        "tokenizer_id": TL_TOKENIZER_ID,
        "tokenizer_revision": TL_TOKENIZER_REVISION,
        "head_hook": TL_HEAD_HOOK,
        "mlp_hook": TL_MLP_HOOK,
        "clean_prompt": primary["clean_prompt"],
        "corrupt_prompt": primary["corrupt_prompt"],
        "positive_token": primary["positive_token"],
        "negative_token": primary["negative_token"],
        "clean_metric": primary["clean_metric"],
        "corrupt_metric": primary["corrupt_metric"],
        "clean_corrupt_gap": primary["clean_metric"] - primary["corrupt_metric"],
        "threshold": threshold,
        "initial_deletion_damage": deletion_damages,
        "deletion_order": order,
        "acdc": asdict(acdc),
        "same_size_random": asdict(random_report),
        "heldout_recoveries": heldout,
        "heldout_tokens": heldout_tokens,
        "worst_heldout_recovery": min(heldout.values()),
        "mlp_bottleneck_recovery": mlp_recovery,
        "peak_vram_gb": peak_vram_gb,
        "claim_scope": (
            "Component-level head deletion in one layer with the real MLP recomputed; "
            "not a full edge-level IOI or greater-than ACDC replication."
        ),
    }


# %%
def run_smoke_test(cpu: bool = True) -> dict[str, object]:
    """Run the exact toy contract without loading a model."""

    _ = cpu
    return run_toy_study()


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, object]:
    """Run the exact toy study and the pinned real-model study on CUDA."""

    if not t.cuda.is_available():
        raise RuntimeError("Section 8.3 CUDA verification requires an available GPU.")

    toy = run_toy_study()
    real = run_transformerlens_component_study(
        device="cuda",
        max_vram_gb=max_vram_gb,
    )
    toy_kept = tuple(toy["acdc"]["kept_edges"])
    result = {
        "accepted": bool(toy["accepted"] and real["accepted"]),
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "peak_vram_gb": real["peak_vram_gb"],
        "within_vram_budget": real["peak_vram_gb"] <= max_vram_gb,
        "full_path": (
            "Exact nonlinear ACDC ground truth plus pinned gelu-1l attention-head "
            "component deletion, exhaustive same-size controls, and held-out transfer."
        ),
        "toy_one_shot_max_abs": max(abs(value) for value in toy["one_shot_scores"].values()),
        "toy_exact_ground_truth_recovered": (
            set(toy_kept) == set(toy["ground_truth_edges"])
        ),
        "toy_kept_edge_count": len(toy_kept),
        "toy_recovery": toy["acdc"]["recovery"],
        "toy_min_edge_damage": toy["metrics"]["min_edge_damage"],
        "toy_max_omitted_subset_gain": toy["metrics"]["max_omitted_gain"],
        "toy_same_size_circuit_count": toy["same_size_random"]["num_circuits"],
        "toy_same_size_rank": toy["same_size_random"]["discovered_rank"],
        "toy_same_size_exact_pvalue": toy["same_size_random"]["exact_empirical_pvalue"],
        "toy_same_size_best_wrong_recovery": toy["same_size_random"]["control_best_recovery"],
        "toy_worst_heldout_recovery": toy["ood"]["worst_recovery"],
        "threshold_backup_cliff": 0.17,
        "threshold_primary_cliff": 0.84,
        "real_model_component_result_passed": real["accepted"],
        "model_name": real["model_name"],
        "hf_revision": real["hf_revision"],
        "tokenizer_revision": real["tokenizer_revision"],
        "head_hook": real["head_hook"],
        "mlp_hook": real["mlp_hook"],
        "target_token": real["positive_token"],
        "distractor_token": real["negative_token"],
        "clean_corrupt_gap": real["clean_corrupt_gap"],
        "real_kept_heads": list(real["acdc"]["kept_edges"]),
        "real_primary_recovery": real["acdc"]["recovery"],
        "real_same_size_set_count": real["same_size_random"]["num_circuits"],
        "real_same_size_rank": real["same_size_random"]["discovered_rank"],
        "real_same_size_exact_pvalue": real["same_size_random"]["exact_empirical_pvalue"],
        "real_same_size_best_other_recovery": real["same_size_random"]["control_best_recovery"],
        "real_worst_heldout_recovery": real["worst_heldout_recovery"],
        "real_mlp_bottleneck_recovery": real["mlp_bottleneck_recovery"],
        "real_model": real,
    }
    if not result["accepted"]:
        raise RuntimeError("Section 8.3 CUDA evidence did not satisfy its declared gates.")
    return result


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, object]:
    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
