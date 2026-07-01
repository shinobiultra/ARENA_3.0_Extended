# %%
"""Reference solutions for [8.2] Attribution Patching and EAP."""

import logging
import os
import sys
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
TL_CLEAN_PROMPT = "The cat sat on the"
TL_CORRUPT_PROMPT = "The bird flew over the"
TL_BNB_CUDA_OVERRIDE = "130"
TL_IG_STEPS = 5


# %%
@dataclass(frozen=True)
class ActivationPatchingSweep:
    patch_scores: t.Tensor
    best_index: int
    best_score: float


@dataclass(frozen=True)
class ScoreCorrelationReport:
    correlation: float
    passes_threshold: bool


@dataclass(frozen=True)
class TopKOverlapReport:
    exact_top_indices: tuple[int, ...]
    approx_top_indices: tuple[int, ...]
    topk_overlap: float
    passes_threshold: bool


@dataclass(frozen=True)
class RuntimeImprovementReport:
    exact_runtime_s: float
    approx_runtime_s: float
    speedup: float
    passes_speedup: bool


@dataclass(frozen=True)
class FalseNegativeReport:
    false_negative_indices: tuple[int, ...]
    num_false_negatives: int
    documented: bool


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
    """Convert per-component patched metrics into recovery scores."""

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


def attribution_patch_scores(
    clean_activations: t.Tensor,
    corrupt_activations: t.Tensor,
    corrupt_gradients: t.Tensor,
    *,
    component_dim: int = 0,
) -> t.Tensor:
    """Approximate patching effects with (clean - corrupt) dot gradient."""

    if clean_activations.shape != corrupt_activations.shape:
        raise ValueError("clean and corrupt activations must have matching shape.")
    if clean_activations.shape != corrupt_gradients.shape:
        raise ValueError("gradients must match activation shape.")
    if not 0 <= component_dim < clean_activations.ndim:
        raise ValueError("component_dim is out of range.")
    contribution = clean_activations.float() - corrupt_activations.float()
    contribution = contribution * corrupt_gradients.float()
    reduce_dims = tuple(dim for dim in range(contribution.ndim) if dim != component_dim)
    return contribution.sum(dim=reduce_dims)


def integrated_gradient_patch_scores(
    clean_activations: t.Tensor,
    corrupt_activations: t.Tensor,
    path_gradients: t.Tensor,
    *,
    component_dim: int = 0,
) -> t.Tensor:
    """Approximate patching effects using averaged path gradients."""

    if path_gradients.ndim != clean_activations.ndim + 1:
        raise ValueError("path_gradients must have shape (steps, *activation_shape).")
    if path_gradients.shape[1:] != clean_activations.shape:
        raise ValueError("path gradient activation dimensions must match activations.")
    mean_gradient = path_gradients.float().mean(dim=0)
    return attribution_patch_scores(
        clean_activations,
        corrupt_activations,
        mean_gradient,
        component_dim=component_dim,
    )


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


def score_correlation_report(
    exact_scores: t.Tensor,
    approx_scores: t.Tensor,
    *,
    min_correlation: float = 0.8,
) -> ScoreCorrelationReport:
    """Report Pearson correlation between exact and approximate patch scores."""

    exact = exact_scores.flatten().float()
    approx = approx_scores.flatten().float()
    if exact.shape != approx.shape:
        raise ValueError("exact_scores and approx_scores must have matching shape.")
    if exact.numel() < 2:
        raise ValueError("at least two scores are required for correlation.")
    exact_centered = exact - exact.mean()
    approx_centered = approx - approx.mean()
    denominator = exact_centered.norm() * approx_centered.norm()
    if denominator.item() == 0:
        correlation = 0.0
    else:
        correlation = float((exact_centered @ approx_centered / denominator).item())
    return ScoreCorrelationReport(
        correlation=correlation,
        passes_threshold=correlation >= min_correlation,
    )


def topk_overlap_report(
    exact_scores: t.Tensor,
    approx_scores: t.Tensor,
    *,
    top_k: int = 3,
    min_overlap: float = 0.5,
) -> TopKOverlapReport:
    """Report top-k overlap between exact and approximate patch scores."""

    exact = exact_scores.flatten().float()
    approx = approx_scores.flatten().float()
    if exact.shape != approx.shape:
        raise ValueError("exact_scores and approx_scores must have matching shape.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    k = min(top_k, exact.numel())
    exact_top = tuple(int(index) for index in exact.topk(k=k).indices.tolist())
    approx_top = tuple(int(index) for index in approx.topk(k=k).indices.tolist())
    overlap = len(set(exact_top) & set(approx_top)) / k
    return TopKOverlapReport(
        exact_top_indices=exact_top,
        approx_top_indices=approx_top,
        topk_overlap=overlap,
        passes_threshold=overlap >= min_overlap,
    )


def runtime_improvement_report(
    *,
    exact_runtime_s: float,
    approx_runtime_s: float,
    min_speedup: float = 2.0,
) -> RuntimeImprovementReport:
    """Report exact-patching runtime divided by approximate-patching runtime."""

    if exact_runtime_s <= 0 or approx_runtime_s <= 0:
        raise ValueError("runtimes must be positive.")
    speedup = exact_runtime_s / approx_runtime_s
    return RuntimeImprovementReport(
        exact_runtime_s=exact_runtime_s,
        approx_runtime_s=approx_runtime_s,
        speedup=speedup,
        passes_speedup=speedup >= min_speedup,
    )


def false_negative_report(
    exact_scores: t.Tensor,
    approx_scores: t.Tensor,
    *,
    exact_threshold: float,
    approx_threshold: float,
    documentation: dict[int, str] | None = None,
) -> FalseNegativeReport:
    """Find exact-important components missed by approximate scores."""

    exact = exact_scores.flatten().float()
    approx = approx_scores.flatten().float()
    if exact.shape != approx.shape:
        raise ValueError("exact_scores and approx_scores must have matching shape.")
    important = exact >= exact_threshold
    missed = approx < approx_threshold
    false_negative_indices = (important & missed).nonzero(as_tuple=False).flatten()
    indices = tuple(int(index.item()) for index in false_negative_indices)
    documentation = documentation or {}
    documented = all(bool(documentation.get(index, "").strip()) for index in indices)
    return FalseNegativeReport(
        false_negative_indices=indices,
        num_false_negatives=len(indices),
        documented=documented,
    )


def attribution_scores_smoke_test() -> list[float]:
    clean = t.tensor([[2.0, 0.0], [0.0, 3.0]])
    corrupt = t.zeros_like(clean)
    gradients = t.tensor([[0.5, 0.5], [1.0, 1.0]])
    return attribution_patch_scores(clean, corrupt, gradients).tolist()


def integrated_gradients_smoke_test() -> list[float]:
    clean = t.tensor([[2.0, 0.0], [0.0, 3.0]])
    corrupt = t.zeros_like(clean)
    path_gradients = t.tensor(
        [
            [[0.25, 0.25], [0.5, 0.5]],
            [[0.75, 0.75], [1.5, 1.5]],
        ]
    )
    return integrated_gradient_patch_scores(clean, corrupt, path_gradients).tolist()


def edge_scores_smoke_test() -> list[list[float]]:
    upstream_delta = t.tensor([[1.0, 0.0], [0.0, 2.0]])
    downstream_gradients = t.tensor([[3.0, 0.0], [0.0, 4.0]])
    return edge_attribution_scores(upstream_delta, downstream_gradients).tolist()


def correlation_smoke_test() -> dict:
    exact = t.tensor([0.1, 0.9, 0.8, 0.0])
    approx = t.tensor([0.2, 0.85, 0.7, 0.1])
    return score_correlation_report(exact, approx, min_correlation=0.95).__dict__


def topk_overlap_smoke_test() -> dict:
    exact = t.tensor([0.1, 0.9, 0.8, 0.0])
    approx = t.tensor([0.2, 0.85, 0.7, 0.1])
    return topk_overlap_report(exact, approx, top_k=2, min_overlap=1.0).__dict__


def runtime_smoke_test() -> dict:
    return runtime_improvement_report(
        exact_runtime_s=10.0,
        approx_runtime_s=2.0,
        min_speedup=4.0,
    ).__dict__


def false_negative_smoke_test() -> dict:
    exact = t.tensor([0.1, 0.9, 0.8])
    approx = t.tensor([0.1, 0.2, 0.7])
    return false_negative_report(
        exact,
        approx,
        exact_threshold=0.75,
        approx_threshold=0.5,
        documentation={1: "Approximation misses a nonlinear interaction."},
    ).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "attribution_scores": attribution_scores_smoke_test(),
        "integrated_gradients": integrated_gradients_smoke_test(),
        "edge_scores": edge_scores_smoke_test(),
        "correlation": correlation_smoke_test(),
        "topk_overlap": topk_overlap_smoke_test(),
        "runtime": runtime_smoke_test(),
        "false_negative": false_negative_smoke_test(),
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


def run_transformerlens_attribution_patching_preflight(max_vram_gb: float = 24.0) -> dict:
    """Compare exact patching, attribution patching, EAP, and IG on real activations."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l exact-vs-attribution patching preflight.",
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
    clean_corrupt_gap = clean_metric - corrupt_metric

    patched_metrics = []
    with t.inference_mode():
        for position in range(clean_tokens.shape[1]):
            def patch_hook(activation: t.Tensor, hook, patch_position: int = position) -> t.Tensor:
                patched = activation.clone()
                patched[:, patch_position, :] = clean_cache[TL_PATCH_HOOK_NAME][
                    :, patch_position, :
                ]
                return patched

            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(TL_PATCH_HOOK_NAME, patch_hook)],
            )
            patched_metrics.append(
                answer_logit_diff(
                    patched_logits[0, -1],
                    positive_token_id=target_token_id,
                    negative_token_id=distractor_token_id,
                )
            )

    exact_sweep = activation_patching_sweep(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        patched_metrics=t.tensor(patched_metrics, device=clean_logits.device),
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

    attribution_scores = attribution_patch_scores(
        clean_cache[TL_PATCH_HOOK_NAME].detach(),
        corrupt_activation,
        corrupt_gradient,
        component_dim=1,
    )
    attribution_recovery_scores = attribution_scores / clean_corrupt_gap

    path_gradients = []
    interpolation_points = t.linspace(0.0, 1.0, TL_IG_STEPS, device=clean_logits.device)
    for alpha in interpolation_points:
        interpolated_activation = (
            corrupt_cache[TL_PATCH_HOOK_NAME]
            + alpha * (clean_cache[TL_PATCH_HOOK_NAME] - corrupt_cache[TL_PATCH_HOOK_NAME])
        ).detach()
        saved_path_activation: dict[str, t.Tensor] = {}

        def replace_with_path_activation(
            activation: t.Tensor,
            hook,
            replacement: t.Tensor = interpolated_activation,
        ) -> t.Tensor:
            path_activation = replacement.clone().requires_grad_(True)
            path_activation.retain_grad()
            saved_path_activation["value"] = path_activation
            return path_activation

        path_logits = model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[(TL_PATCH_HOOK_NAME, replace_with_path_activation)],
        )
        path_metric = path_logits[0, -1, target_token_id] - path_logits[
            0, -1, distractor_token_id
        ]
        model.zero_grad(set_to_none=True)
        path_metric.backward()
        path_gradients.append(saved_path_activation["value"].grad.detach())

    ig_scores = integrated_gradient_patch_scores(
        clean_cache[TL_PATCH_HOOK_NAME].detach(),
        corrupt_cache[TL_PATCH_HOOK_NAME].detach(),
        t.stack(path_gradients, dim=0),
        component_dim=1,
    )
    ig_recovery_scores = ig_scores / clean_corrupt_gap

    edge_scores = edge_attribution_scores(
        (clean_cache[TL_PATCH_HOOK_NAME] - corrupt_cache[TL_PATCH_HOOK_NAME]).squeeze(0),
        corrupt_gradient.squeeze(0),
    )
    edge_abs = edge_scores.abs()
    top_edge_flat = int(edge_abs.argmax().item())
    sequence_length = int(clean_tokens.shape[1])
    top_edge_upstream = top_edge_flat // sequence_length
    top_edge_downstream = top_edge_flat % sequence_length
    target_position = sequence_length - 1

    correlation = score_correlation_report(
        exact_sweep.patch_scores,
        attribution_recovery_scores,
        min_correlation=0.99,
    )
    topk_overlap = topk_overlap_report(
        exact_sweep.patch_scores,
        attribution_recovery_scores,
        top_k=1,
        min_overlap=1.0,
    )
    ig_correlation = score_correlation_report(
        exact_sweep.patch_scores,
        ig_recovery_scores,
        min_correlation=0.99,
    )
    ig_topk_overlap = topk_overlap_report(
        exact_sweep.patch_scores,
        ig_recovery_scores,
        top_k=1,
        min_overlap=1.0,
    )

    grad_norms = corrupt_gradient.float().norm(dim=-1).squeeze(0)
    nonfinal_grad_norm_max = float(grad_norms[:target_position].max().item())
    final_grad_norm = float(grad_norms[target_position].item())
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        clean_corrupt_gap >= 6.0
        and float(exact_sweep.patch_scores[target_position].item()) >= 0.99
        and float(attribution_recovery_scores[target_position].item()) >= 0.9
        and float(ig_recovery_scores[target_position].item()) >= 0.95
        and correlation.passes_threshold
        and topk_overlap.passes_threshold
        and ig_correlation.passes_threshold
        and ig_topk_overlap.passes_threshold
        and top_edge_upstream == target_position
        and top_edge_downstream == target_position
        and nonfinal_grad_norm_max <= 1e-7
        and final_grad_norm >= 1.0
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
        "clean_corrupt_gap": clean_corrupt_gap,
        "exact_patch_scores_by_position": [
            float(score) for score in exact_sweep.patch_scores.tolist()
        ],
        "attribution_scores_by_position": [
            float(score) for score in attribution_recovery_scores.tolist()
        ],
        "ig_scores_by_position": [float(score) for score in ig_recovery_scores.tolist()],
        "exact_best_position": exact_sweep.best_index,
        "attribution_best_position": int(attribution_recovery_scores.argmax().item()),
        "ig_best_position": int(ig_recovery_scores.argmax().item()),
        "exact_final_recovery": float(exact_sweep.patch_scores[target_position].item()),
        "attribution_final_recovery": float(
            attribution_recovery_scores[target_position].item()
        ),
        "ig_final_recovery": float(ig_recovery_scores[target_position].item()),
        "exact_attribution_correlation": correlation.correlation,
        "exact_attribution_top1_overlap": topk_overlap.topk_overlap,
        "exact_ig_correlation": ig_correlation.correlation,
        "exact_ig_top1_overlap": ig_topk_overlap.topk_overlap,
        "eap_edge_score_shape": list(edge_scores.shape),
        "eap_top_edge_upstream_position": top_edge_upstream,
        "eap_top_edge_downstream_position": top_edge_downstream,
        "eap_top_edge_score_abs": float(edge_abs.flatten()[top_edge_flat].item()),
        "nonfinal_gradient_norm_max": nonfinal_grad_norm_max,
        "final_gradient_norm": final_grad_norm,
        "integrated_gradient_steps": TL_IG_STEPS,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l exact-vs-attribution patching preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_attribution_patching_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
