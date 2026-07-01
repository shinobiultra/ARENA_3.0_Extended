# %%
"""Reference solutions for [8.3] ACDC and Circuit Metrics."""

import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
TL_BNB_CUDA_OVERRIDE = "130"
TL_PRIMARY_CLEAN_PROMPT = "The cat sat on the"
TL_PRIMARY_CORRUPT_PROMPT = "The bird flew over the"
TL_TEMPLATE_PAIRS = [
    ("The cat sat on the", "The bird flew over the"),
    ("To make tea, boil the", "To make bread, bake the"),
    ("The recipe calls for sugar and", "The recipe calls for salt and"),
    ("The chef cooked a", "The teacher taught a"),
]


# %%
@dataclass(frozen=True)
class ActivationPatchingSweep:
    patch_scores: t.Tensor
    best_index: int
    best_score: float


@dataclass(frozen=True)
class ACDCPruningReport:
    kept_edges: tuple[str, ...]
    removed_edges: tuple[str, ...]
    threshold: float
    num_kept: int


@dataclass(frozen=True)
class CircuitFaithfulnessReport:
    full_metric: float
    corrupt_metric: float
    circuit_metric: float
    preserved_fraction: float
    passes_faithfulness: bool


@dataclass(frozen=True)
class CircuitMinimalityReport:
    circuit_metric: float
    ablated_metric: float
    metric_damage: float
    passes_minimality: bool


@dataclass(frozen=True)
class CircuitCompletenessReport:
    circuit_metric: float
    expanded_metric: float
    omitted_node_gain: float
    passes_completeness: bool


@dataclass(frozen=True)
class RandomCircuitBaselineReport:
    circuit_metric: float
    random_metric: float
    margin: float
    circuit_beats_random: bool


@dataclass(frozen=True)
class OODTemplateReport:
    per_template_accuracy: dict[int, float]
    worst_template_accuracy: float
    passes_ood: bool


@dataclass(frozen=True)
class CircuitMethodComparisonReport:
    exact_top_edges: tuple[str, ...]
    method_top_edges: dict[str, tuple[str, ...]]
    topk_overlap: dict[str, float]
    score_correlations: dict[str, float]
    circuit_sizes: dict[str, int]
    best_matching_method: str
    passes_comparison: bool


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


def activation_patching_sweep(
    *,
    clean_metric: float,
    corrupt_metric: float,
    patched_metrics: t.Tensor,
) -> ActivationPatchingSweep:
    """Convert per-component patched metrics into normalized recovery scores."""

    denominator = clean_metric - corrupt_metric
    if denominator == 0:
        raise ValueError("clean_metric and corrupt_metric must differ.")
    patch_scores = (patched_metrics.float() - corrupt_metric) / denominator
    best_index = int(patch_scores.argmax().item())
    return ActivationPatchingSweep(
        patch_scores=patch_scores,
        best_index=best_index,
        best_score=float(patch_scores[best_index].item()),
    )


def acdc_pruning_report(
    edge_scores: t.Tensor,
    edge_names: list[str],
    *,
    threshold: float,
) -> ACDCPruningReport:
    """Keep edges whose score survives an ACDC-style threshold."""

    scores = edge_scores.flatten().float()
    if scores.numel() != len(edge_names):
        raise ValueError("edge_scores and edge_names must align.")
    kept = []
    removed = []
    for score, name in zip(scores.tolist(), edge_names, strict=True):
        if score >= threshold:
            kept.append(name)
        else:
            removed.append(name)
    return ACDCPruningReport(
        kept_edges=tuple(kept),
        removed_edges=tuple(removed),
        threshold=threshold,
        num_kept=len(kept),
    )


def circuit_faithfulness_report(
    *,
    full_metric: float,
    corrupt_metric: float,
    circuit_metric: float,
    min_preserved_fraction: float = 0.75,
) -> CircuitFaithfulnessReport:
    """Check how much clean-vs-corrupt behavior the circuit preserves."""

    denominator = full_metric - corrupt_metric
    if denominator == 0:
        raise ValueError("full_metric and corrupt_metric must differ.")
    preserved_fraction = (circuit_metric - corrupt_metric) / denominator
    return CircuitFaithfulnessReport(
        full_metric=full_metric,
        corrupt_metric=corrupt_metric,
        circuit_metric=circuit_metric,
        preserved_fraction=preserved_fraction,
        passes_faithfulness=preserved_fraction >= min_preserved_fraction,
    )


def circuit_minimality_report(
    *,
    circuit_metric: float,
    ablated_metric: float,
    min_metric_damage: float = 0.5,
) -> CircuitMinimalityReport:
    """Check whether removing circuit nodes damages behavior."""

    metric_damage = circuit_metric - ablated_metric
    return CircuitMinimalityReport(
        circuit_metric=circuit_metric,
        ablated_metric=ablated_metric,
        metric_damage=metric_damage,
        passes_minimality=metric_damage >= min_metric_damage,
    )


def circuit_completeness_report(
    *,
    circuit_metric: float,
    expanded_metric: float,
    max_omitted_node_gain: float = 0.2,
) -> CircuitCompletenessReport:
    """Check whether adding top omitted nodes improves little."""

    omitted_node_gain = expanded_metric - circuit_metric
    return CircuitCompletenessReport(
        circuit_metric=circuit_metric,
        expanded_metric=expanded_metric,
        omitted_node_gain=omitted_node_gain,
        passes_completeness=omitted_node_gain <= max_omitted_node_gain,
    )


def random_circuit_baseline_report(
    *,
    circuit_metric: float,
    random_metric: float,
    min_margin: float = 0.5,
) -> RandomCircuitBaselineReport:
    """Check that a discovered circuit beats a same-size random circuit."""

    margin = circuit_metric - random_metric
    return RandomCircuitBaselineReport(
        circuit_metric=circuit_metric,
        random_metric=random_metric,
        margin=margin,
        circuit_beats_random=margin >= min_margin,
    )


def ood_template_report(
    logits: t.Tensor,
    answer_ids: t.Tensor,
    template_ids: t.Tensor,
    *,
    min_accuracy: float = 0.75,
) -> OODTemplateReport:
    """Report circuit answer accuracy on held-out prompt templates."""

    if logits.shape[:-1] != answer_ids.shape:
        raise ValueError("answer_ids must match logits leading dimensions.")
    if answer_ids.shape != template_ids.shape:
        raise ValueError("answer_ids and template_ids must match.")

    predictions = logits.argmax(dim=-1)
    per_template: dict[int, float] = {}
    for template_id in template_ids.unique(sorted=True):
        mask = template_ids.eq(template_id)
        accuracy = predictions[mask].eq(answer_ids[mask]).float().mean().item()
        per_template[int(template_id.item())] = accuracy
    worst_accuracy = min(per_template.values()) if per_template else 0.0
    return OODTemplateReport(
        per_template_accuracy=per_template,
        worst_template_accuracy=worst_accuracy,
        passes_ood=worst_accuracy >= min_accuracy,
    )


def _top_edge_names(scores: t.Tensor, edge_names: list[str], *, top_k: int) -> tuple[str, ...]:
    flat_scores = scores.flatten().float()
    if flat_scores.numel() != len(edge_names):
        raise ValueError("scores and edge_names must align.")
    k = min(top_k, flat_scores.numel())
    top_indices = flat_scores.topk(k=k).indices.tolist()
    return tuple(edge_names[int(index)] for index in top_indices)


def _pearson_correlation(left: t.Tensor, right: t.Tensor) -> float:
    left = left.flatten().float()
    right = right.flatten().float()
    if left.shape != right.shape:
        raise ValueError("score tensors must have matching shapes.")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((left_centered @ right_centered / denominator).item())


def circuit_method_comparison_report(
    exact_scores: t.Tensor,
    method_scores: Mapping[str, t.Tensor],
    edge_names: list[str],
    *,
    top_k: int,
    min_topk_overlap: float = 0.5,
    min_score_correlation: float = 0.5,
) -> CircuitMethodComparisonReport:
    """Compare approximate circuit-discovery methods against exact patching."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not method_scores:
        raise ValueError("method_scores must contain at least one method.")
    exact_flat = exact_scores.flatten().float()
    if exact_flat.numel() != len(edge_names):
        raise ValueError("exact_scores and edge_names must align.")

    exact_top_edges = _top_edge_names(exact_flat, edge_names, top_k=top_k)
    exact_top_set = set(exact_top_edges)
    method_top_edges: dict[str, tuple[str, ...]] = {}
    topk_overlap: dict[str, float] = {}
    score_correlations: dict[str, float] = {}
    circuit_sizes: dict[str, int] = {"exact": len(exact_top_edges)}

    for method_name, scores in method_scores.items():
        method_flat = scores.flatten().float()
        if method_flat.shape != exact_flat.shape:
            raise ValueError(f"{method_name} scores must match exact_scores shape.")
        top_edges = _top_edge_names(method_flat, edge_names, top_k=top_k)
        overlap = len(exact_top_set.intersection(top_edges)) / len(exact_top_edges)
        method_top_edges[method_name] = top_edges
        topk_overlap[method_name] = overlap
        score_correlations[method_name] = _pearson_correlation(exact_flat, method_flat)
        circuit_sizes[method_name] = len(top_edges)

    best_matching_method = max(
        method_scores,
        key=lambda name: (topk_overlap[name], score_correlations[name]),
    )
    passes_comparison = all(
        topk_overlap[name] >= min_topk_overlap
        and score_correlations[name] >= min_score_correlation
        for name in method_scores
    )
    return CircuitMethodComparisonReport(
        exact_top_edges=exact_top_edges,
        method_top_edges=method_top_edges,
        topk_overlap=topk_overlap,
        score_correlations=score_correlations,
        circuit_sizes=circuit_sizes,
        best_matching_method=best_matching_method,
        passes_comparison=passes_comparison,
    )


# %%
def acdc_pruning_smoke_test() -> dict:
    scores = t.tensor([0.9, 0.2, 0.7])
    names = ["name-mover", "backup", "negative"]
    return acdc_pruning_report(scores, names, threshold=0.5).__dict__


def faithfulness_smoke_test() -> dict:
    return circuit_faithfulness_report(
        full_metric=3.0,
        corrupt_metric=-1.0,
        circuit_metric=2.2,
        min_preserved_fraction=0.75,
    ).__dict__


def minimality_smoke_test() -> dict:
    return circuit_minimality_report(
        circuit_metric=2.2,
        ablated_metric=0.5,
        min_metric_damage=1.0,
    ).__dict__


def completeness_smoke_test() -> dict:
    return circuit_completeness_report(
        circuit_metric=2.2,
        expanded_metric=2.35,
        max_omitted_node_gain=0.2,
    ).__dict__


def random_baseline_smoke_test() -> dict:
    return random_circuit_baseline_report(
        circuit_metric=2.2,
        random_metric=0.8,
        min_margin=1.0,
    ).__dict__


def ood_smoke_test() -> dict:
    logits = t.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    answer_ids = t.tensor([0, 1, 0, 1])
    template_ids = t.tensor([0, 0, 1, 1])
    return ood_template_report(logits, answer_ids, template_ids, min_accuracy=1.0).__dict__


def method_comparison_smoke_test() -> dict:
    exact = t.tensor([0.95, 0.8, 0.15, 0.05])
    eap_ig = t.tensor([0.9, 0.7, 0.2, 0.1])
    names = ["name-mover", "backup-name-mover", "mlp-noise", "wrong-position"]
    return circuit_method_comparison_report(
        exact,
        {"eap_ig": eap_ig},
        names,
        top_k=2,
        min_topk_overlap=1.0,
        min_score_correlation=0.9,
    ).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "acdc": acdc_pruning_smoke_test(),
        "faithfulness": faithfulness_smoke_test(),
        "minimality": minimality_smoke_test(),
        "completeness": completeness_smoke_test(),
        "random_baseline": random_baseline_smoke_test(),
        "ood": ood_smoke_test(),
        "method_comparison": method_comparison_smoke_test(),
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


def _real_patch_sweep(model, clean_prompt: str, corrupt_prompt: str) -> dict:
    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise RuntimeError(
            f"clean/corrupt token shapes differ: {clean_tokens.shape} vs {corrupt_tokens.shape}"
        )

    with t.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda name: name == TL_PATCH_HOOK_NAME,
        )
        corrupt_logits = model(corrupt_tokens)

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

    sequence_length = int(clean_tokens.shape[1])
    patched_metrics = [
        _patched_metric(
            model,
            corrupt_tokens,
            clean_cache,
            [position],
            positive_token_id=target_token_id,
            negative_token_id=distractor_token_id,
        )
        for position in range(sequence_length)
    ]
    sweep = activation_patching_sweep(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        patched_metrics=t.tensor(patched_metrics, device=clean_logits.device),
    )
    target_position = sequence_length - 1
    random_position = 0
    expanded_positions = [target_position, random_position]
    return {
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "sequence_length": sequence_length,
        "target_position": target_position,
        "random_position": random_position,
        "target_token_id": int(target_token_id),
        "distractor_token_id": int(distractor_token_id),
        "target_token": model.to_string(target_token_id),
        "distractor_token": model.to_string(distractor_token_id),
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
        "clean_corrupt_gap": clean_metric - corrupt_metric,
        "patch_scores": sweep.patch_scores,
        "best_position": sweep.best_index,
        "best_score": sweep.best_score,
        "circuit_metric": patched_metrics[target_position],
        "random_metric": patched_metrics[random_position],
        "expanded_metric": _patched_metric(
            model,
            corrupt_tokens,
            clean_cache,
            expanded_positions,
            positive_token_id=target_token_id,
            negative_token_id=distractor_token_id,
        ),
    }


def run_transformerlens_acdc_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run real residual-position circuit pruning and metric checks on CUDA."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l ACDC-style position-circuit preflight.",
        }

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()

    primary = _real_patch_sweep(model, TL_PRIMARY_CLEAN_PROMPT, TL_PRIMARY_CORRUPT_PROMPT)
    edge_names = [f"position_{index}" for index in range(primary["sequence_length"])]
    pruning = acdc_pruning_report(
        primary["patch_scores"],
        edge_names,
        threshold=0.5,
    )
    faithfulness = circuit_faithfulness_report(
        full_metric=primary["clean_metric"],
        corrupt_metric=primary["corrupt_metric"],
        circuit_metric=primary["circuit_metric"],
        min_preserved_fraction=0.99,
    )
    minimality = circuit_minimality_report(
        circuit_metric=primary["circuit_metric"],
        ablated_metric=primary["corrupt_metric"],
        min_metric_damage=1.0,
    )
    completeness = circuit_completeness_report(
        circuit_metric=primary["circuit_metric"],
        expanded_metric=primary["expanded_metric"],
        max_omitted_node_gain=1e-5,
    )
    random_baseline = random_circuit_baseline_report(
        circuit_metric=primary["circuit_metric"],
        random_metric=primary["random_metric"],
        min_margin=1.0,
    )

    template_reports = [
        _real_patch_sweep(model, clean_prompt, corrupt_prompt)
        for clean_prompt, corrupt_prompt in TL_TEMPLATE_PAIRS
    ]
    template_recoveries = [
        (template["circuit_metric"] - template["corrupt_metric"])
        / template["clean_corrupt_gap"]
        for template in template_reports
    ]
    template_random_recoveries = [
        (template["random_metric"] - template["corrupt_metric"])
        / template["clean_corrupt_gap"]
        for template in template_reports
    ]
    template_best_positions = [template["best_position"] for template in template_reports]
    template_target_positions = [
        template["target_position"] for template in template_reports
    ]
    passes_ood = all(
        recovery >= 0.99
        and random_recovery <= 1e-4
        and best_position == target_position
        for recovery, random_recovery, best_position, target_position in zip(
            template_recoveries,
            template_random_recoveries,
            template_best_positions,
            template_target_positions,
            strict=True,
        )
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        pruning.num_kept == 1
        and pruning.kept_edges == (f"position_{primary['target_position']}",)
        and primary["best_position"] == primary["target_position"]
        and primary["best_score"] >= 0.99
        and primary["clean_corrupt_gap"] >= 6.0
        and faithfulness.passes_faithfulness
        and minimality.passes_minimality
        and completeness.passes_completeness
        and random_baseline.circuit_beats_random
        and passes_ood
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
        "clean_prompt": TL_PRIMARY_CLEAN_PROMPT,
        "corrupt_prompt": TL_PRIMARY_CORRUPT_PROMPT,
        "sequence_length": primary["sequence_length"],
        "target_position": primary["target_position"],
        "target_token": primary["target_token"],
        "distractor_token": primary["distractor_token"],
        "patch_scores_by_position": [
            float(score) for score in primary["patch_scores"].tolist()
        ],
        "best_position": primary["best_position"],
        "best_score": primary["best_score"],
        "num_kept_edges": pruning.num_kept,
        "kept_edges": list(pruning.kept_edges),
        "clean_corrupt_gap": primary["clean_corrupt_gap"],
        "preserved_fraction": faithfulness.preserved_fraction,
        "passes_faithfulness": faithfulness.passes_faithfulness,
        "minimality_metric_damage": minimality.metric_damage,
        "passes_minimality": minimality.passes_minimality,
        "omitted_node_gain": completeness.omitted_node_gain,
        "passes_completeness": completeness.passes_completeness,
        "random_baseline_margin": random_baseline.margin,
        "circuit_beats_random": random_baseline.circuit_beats_random,
        "template_count": len(template_reports),
        "template_recoveries": [float(recovery) for recovery in template_recoveries],
        "template_random_recoveries": [
            float(recovery) for recovery in template_random_recoveries
        ],
        "template_best_positions": template_best_positions,
        "template_target_positions": template_target_positions,
        "passes_ood": passes_ood,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l ACDC-style position-circuit preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_acdc_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
