# %%
"""Reference solutions for [8.4] Circuit Tracing with Attribution Graphs."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t

chapter = "chapter8_automated_circuits"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_PATCH_HOOK_NAME = "blocks.0.hook_resid_post"
TL_CLEAN_PROMPT = "The cat sat on the"
TL_CORRUPT_PROMPT = "The bird flew over the"
TL_BNB_CUDA_OVERRIDE = "130"


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


def answer_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    """Return mean positive-minus-negative logit difference."""

    vocab_size = logits.shape[-1]
    if not 0 <= positive_token_id < vocab_size:
        raise ValueError("positive_token_id is out of range.")
    if not 0 <= negative_token_id < vocab_size:
        raise ValueError("negative_token_id is out of range.")
    diff = logits[..., positive_token_id] - logits[..., negative_token_id]
    return diff.float().mean().item()


def edge_attribution_scores(
    upstream_activation_delta: t.Tensor,
    downstream_gradients: t.Tensor,
) -> t.Tensor:
    """Return EAP-style upstream-by-downstream edge attribution scores."""

    if upstream_activation_delta.ndim != 2 or downstream_gradients.ndim != 2:
        raise ValueError("inputs must have shape (components, d_model).")
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
    if edge_scores.shape[0] != len(node_names):
        raise ValueError("edge_scores and node_names must align.")
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
    """Find the highest-scoring directed path from source to target."""

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


# %%
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
    return {
        "graph": graph_smoke_test(),
        "metric": graph_metric_smoke_test(),
        "path_perturbation": path_perturbation_smoke_test(),
        "alternative": alternative_graph_smoke_test(),
        "counterfactual": counterfactual_smoke_test(),
        "path": path_smoke_test(),
    }


def _load_gelu1l_model_on_cuda():
    os.environ.setdefault("BNB_CUDA_VERSION", TL_BNB_CUDA_OVERRIDE)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        TL_GELU1L_MODEL_NAME,
        device="cuda",
        dtype="float32",
        revision=TL_GELU1L_REVISION,
    )


def _patched_metric(
    model,
    corrupt_tokens: t.Tensor,
    clean_cache: dict[str, t.Tensor],
    positions: list[int],
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    def patch_hook(activation: t.Tensor, hook) -> t.Tensor:
        patched = activation.clone()
        for position in positions:
            patched[:, position, :] = clean_cache[TL_PATCH_HOOK_NAME][:, position, :]
        return patched

    with t.inference_mode():
        patched_logits = model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[(TL_PATCH_HOOK_NAME, patch_hook)],
        )
    return answer_logit_diff(
        patched_logits[0, -1],
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )


def run_transformerlens_attribution_graph_preflight(max_vram_gb: float = 24.0) -> dict:
    """Build and perturb a real EAP-derived residual-position attribution graph."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l position-edge attribution graph preflight.",
        }

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()

    clean_tokens = model.to_tokens(TL_CLEAN_PROMPT)
    corrupt_tokens = model.to_tokens(TL_CORRUPT_PROMPT)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise RuntimeError(
            f"clean/corrupt token shapes differ: {clean_tokens.shape} vs {corrupt_tokens.shape}"
        )

    with t.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda name: name == TL_PATCH_HOOK_NAME,
        )
        corrupt_logits, corrupt_cache = model.run_with_cache(
            corrupt_tokens,
            names_filter=lambda name: name == TL_PATCH_HOOK_NAME,
        )

    clean_final_logits = clean_logits[0, -1]
    corrupt_final_logits = corrupt_logits[0, -1]
    clean_top_tokens = clean_final_logits.topk(5).indices.tolist()
    corrupt_top_tokens = corrupt_final_logits.topk(5).indices.tolist()
    target_token_id = clean_top_tokens[0]
    distractor_token_id = next(
        token_id
        for token_id in [*corrupt_top_tokens, *clean_top_tokens[1:]]
        if token_id != target_token_id
    )
    clean_metric = answer_logit_diff(
        clean_final_logits,
        positive_token_id=target_token_id,
        negative_token_id=distractor_token_id,
    )
    corrupt_metric = answer_logit_diff(
        corrupt_final_logits,
        positive_token_id=target_token_id,
        negative_token_id=distractor_token_id,
    )

    saved_activation: dict[str, t.Tensor] = {}

    def save_corrupt_activation(activation: t.Tensor, hook) -> t.Tensor:
        activation.retain_grad()
        saved_activation["value"] = activation
        return activation

    logits_for_grad = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(TL_PATCH_HOOK_NAME, save_corrupt_activation)],
    )
    metric = logits_for_grad[0, -1, target_token_id] - logits_for_grad[
        0, -1, distractor_token_id
    ]
    model.zero_grad(set_to_none=True)
    metric.backward()
    corrupt_activation = saved_activation["value"].detach()
    corrupt_gradient = saved_activation["value"].grad.detach()

    sequence_length = int(clean_tokens.shape[1])
    target_position = sequence_length - 1
    node_names = [f"position_{index}" for index in range(sequence_length)]
    edge_scores = edge_attribution_scores(
        (clean_cache[TL_PATCH_HOOK_NAME] - corrupt_activation).squeeze(0),
        corrupt_gradient.squeeze(0),
    )
    edge_abs = edge_scores.abs()
    graph = build_local_attribution_graph(edge_abs, node_names, top_k=1)
    top_edge = graph.edges[0]

    graph_metric = _patched_metric(
        model,
        corrupt_tokens,
        clean_cache,
        [target_position],
        positive_token_id=target_token_id,
        negative_token_id=distractor_token_id,
    )
    alternative_metric = _patched_metric(
        model,
        corrupt_tokens,
        clean_cache,
        [0],
        positive_token_id=target_token_id,
        negative_token_id=distractor_token_id,
    )
    graph_report = graph_metric_report(
        full_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        graph_metric=graph_metric,
        min_explained_fraction=0.99,
    )
    perturbation = path_perturbation_report(
        original_metric=graph_metric,
        perturbed_metric=corrupt_metric,
        min_metric_drop=1.0,
    )
    alternative = alternative_graph_baseline_report(
        graph_metric=graph_metric,
        alternative_metric=alternative_metric,
        min_margin=1.0,
    )
    counterfactual = graph_summary_counterfactual_report(
        predicted_direction="decrease",
        baseline_metric=graph_metric,
        counterfactual_metric=corrupt_metric,
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        len(graph.edges) == 1
        and top_edge.source == f"position_{target_position}"
        and top_edge.target == f"position_{target_position}"
        and top_edge.score >= 6.0
        and graph_report.explains_target_metric
        and perturbation.top_path_survives_test
        and alternative.alternative_baseline_fails
        and counterfactual.predicts_counterfactual
        and within_vram_budget
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "preflight_passed": preflight_passed,
        "model_name": TL_GELU1L_MODEL_NAME,
        "hf_model_id": TL_GELU1L_HF_ID,
        "hf_revision": TL_GELU1L_REVISION,
        "tokenizer_id": TL_GELU1L_TOKENIZER_ID,
        "tokenizer_revision": TL_GELU1L_TOKENIZER_REVISION,
        "bnb_cuda_override": TL_BNB_CUDA_OVERRIDE,
        "hook_name": TL_PATCH_HOOK_NAME,
        "clean_prompt": TL_CLEAN_PROMPT,
        "corrupt_prompt": TL_CORRUPT_PROMPT,
        "sequence_length": sequence_length,
        "target_position": target_position,
        "target_token": model.to_string(target_token_id),
        "distractor_token": model.to_string(distractor_token_id),
        "clean_corrupt_gap": clean_metric - corrupt_metric,
        "graph_node_count": len(graph.nodes),
        "num_graph_edges": len(graph.edges),
        "graph_edge_source": top_edge.source,
        "graph_edge_target": top_edge.target,
        "graph_edge_score": top_edge.score,
        "top_edge_score_abs": float(edge_abs.max().item()),
        "explained_fraction": graph_report.explained_fraction,
        "explains_target_metric": graph_report.explains_target_metric,
        "path_metric_drop": perturbation.metric_drop,
        "top_path_survives_test": perturbation.top_path_survives_test,
        "alternative_baseline_margin": alternative.margin,
        "alternative_baseline_fails": alternative.alternative_baseline_fails,
        "counterfactual_predicted_direction": counterfactual.predicted_direction,
        "counterfactual_observed_delta": counterfactual.observed_delta,
        "predicts_counterfactual": counterfactual.predicts_counterfactual,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l position-edge attribution graph preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_attribution_graph_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
