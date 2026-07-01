# %%
"""Reference solutions for [6.2] Gemma Scope Deep Dive."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t

chapter = "chapter6_sparse_feature_methods"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.gemma_scope import (
    GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL,
    gemma_base_model_access_report,
    gemma_scope_artifact_preflight,
    gemma_scope_real_activation_preflight,
)

MAIN = __name__ == "__main__"
GEMMA_SCOPE_REPO_ID = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL.repo_id
GEMMA_SCOPE_REVISION = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL.revision
GEMMA_SCOPE_ARTIFACT_PATH = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL.artifact_path


@dataclass(frozen=True)
class FeatureArtifactMetadata:
    model_name: str
    artifact_name: str
    artifact_type: Literal["sae", "transcoder"]
    layer: int
    hook_name: str
    d_model: int
    n_features: int


@dataclass(frozen=True)
class TaggedFeatureSpec:
    feature_id: int
    layer: int
    tags: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class FeatureValidationSuiteReport:
    feature_auc: float
    baseline_auc: float
    auc_margin: float
    threshold_accuracy: float
    positive_mean: float
    negative_mean: float
    passes_baseline: bool


@dataclass(frozen=True)
class BaseInstructionFeatureDelta:
    base_mean: float
    instruction_mean: float
    delta: float
    abs_delta: float


@dataclass(frozen=True)
class AblationControlReport:
    baseline_mean: float
    ablated_mean: float
    random_ablated_mean: float
    ablation_delta: float
    random_delta: float
    passes_control: bool


@dataclass(frozen=True)
class SteeringSafetyReport:
    baseline_mean: float
    steered_mean: float
    random_mean: float
    steered_delta: float
    random_delta: float
    perplexity_ratio: float
    passes_control: bool
    passes_perplexity_guard: bool


# %%
def metadata_is_complete(metadata: FeatureArtifactMetadata) -> bool:
    """Check minimum artifact metadata needed for reproducible feature work."""

    return (
        bool(metadata.model_name)
        and bool(metadata.artifact_name)
        and metadata.artifact_type in {"sae", "transcoder"}
        and metadata.layer >= 0
        and bool(metadata.hook_name)
        and metadata.d_model > 0
        and metadata.n_features > 0
    )


def features_with_tag(
    features: list[TaggedFeatureSpec],
    tag: str,
) -> list[TaggedFeatureSpec]:
    """Return feature specs carrying a tag such as refusal, code, or sentiment."""

    return [feature for feature in features if tag in feature.tags]


def feature_score_vector(
    feature_acts: t.Tensor,
    feature_id: int,
    *,
    reduction: Literal["max", "mean", "last"] = "max",
) -> t.Tensor:
    """Reduce per-token feature activations to one score per example."""

    if feature_acts.ndim == 2:
        if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
            raise IndexError("feature_id is out of range.")
        return feature_acts[:, feature_id].float()
    if feature_acts.ndim != 3:
        raise ValueError(
            "feature_acts must have shape (examples, features) or (examples, seq, features)."
        )
    if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")

    per_token = feature_acts[..., feature_id].float()
    if reduction == "max":
        return per_token.max(dim=-1).values
    if reduction == "mean":
        return per_token.mean(dim=-1)
    if reduction == "last":
        return per_token[:, -1]
    raise ValueError("reduction must be 'max', 'mean', or 'last'.")


def roc_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Compute binary ROC AUC using rank statistics, including ties."""

    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same number of elements.")
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires at least one positive and one negative example.")

    order = scores.argsort()
    sorted_scores = scores[order]
    ranks_sorted = t.empty_like(sorted_scores)
    _, counts = t.unique_consecutive(sorted_scores, return_counts=True)
    start = 0
    for count in counts.tolist():
        end = start + count
        average_rank = (start + 1 + end) / 2
        ranks_sorted[start:end] = average_rank
        start = end
    ranks = t.empty_like(scores)
    ranks[order] = ranks_sorted
    pos_rank_sum = ranks[labels].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc.item())


def _feature_detection_stats(
    scores: t.Tensor,
    labels: t.Tensor,
    threshold: float | None = None,
) -> tuple[float, float, float, float]:
    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("Need at least one positive and one negative example.")
    if threshold is None:
        threshold = 0.5 * (positives.mean().item() + negatives.mean().item())
    predictions = scores >= threshold
    threshold_accuracy = predictions.eq(labels).float().mean().item()
    return (
        roc_auc_binary(scores, labels),
        positives.mean().item(),
        negatives.mean().item(),
        threshold_accuracy,
    )


def validate_feature_scores(
    feature_scores: t.Tensor,
    labels: t.Tensor,
    baseline_scores: t.Tensor,
    *,
    min_auc_margin: float = 0.1,
) -> FeatureValidationSuiteReport:
    """Validate a feature against matched negatives and a baseline score vector."""

    feature_auc, positive_mean, negative_mean, threshold_accuracy = _feature_detection_stats(
        feature_scores,
        labels,
    )
    baseline_auc = roc_auc_binary(baseline_scores.flatten().float(), labels.flatten().bool())
    auc_margin = feature_auc - baseline_auc
    return FeatureValidationSuiteReport(
        feature_auc=feature_auc,
        baseline_auc=baseline_auc,
        auc_margin=auc_margin,
        threshold_accuracy=threshold_accuracy,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        passes_baseline=auc_margin >= min_auc_margin,
    )


def base_instruction_feature_delta(
    base_scores: t.Tensor,
    instruction_scores: t.Tensor,
) -> BaseInstructionFeatureDelta:
    """Compare mean feature activation between base and instruction-tuned models."""

    if base_scores.shape != instruction_scores.shape:
        raise ValueError("base_scores and instruction_scores must have matching shapes.")
    base_mean = base_scores.float().mean().item()
    instruction_mean = instruction_scores.float().mean().item()
    delta = instruction_mean - base_mean
    return BaseInstructionFeatureDelta(
        base_mean=base_mean,
        instruction_mean=instruction_mean,
        delta=delta,
        abs_delta=abs(delta),
    )


def ablation_control_report(
    baseline_scores: t.Tensor,
    ablated_scores: t.Tensor,
    random_ablated_scores: t.Tensor,
) -> AblationControlReport:
    """Check whether ablating a feature reduces a target behavior more than control."""

    baseline_mean = baseline_scores.float().mean().item()
    ablated_mean = ablated_scores.float().mean().item()
    random_mean = random_ablated_scores.float().mean().item()
    ablation_delta = baseline_mean - ablated_mean
    random_delta = baseline_mean - random_mean
    return AblationControlReport(
        baseline_mean=baseline_mean,
        ablated_mean=ablated_mean,
        random_ablated_mean=random_mean,
        ablation_delta=ablation_delta,
        random_delta=random_delta,
        passes_control=ablation_delta > random_delta,
    )


def steering_safety_report(
    baseline_scores: t.Tensor,
    steered_scores: t.Tensor,
    random_control_scores: t.Tensor,
    *,
    baseline_perplexity: float,
    steered_perplexity: float,
    max_perplexity_ratio: float = 1.2,
) -> SteeringSafetyReport:
    """Check steering target increase while bounding perplexity degradation."""

    if baseline_perplexity <= 0:
        raise ValueError("baseline_perplexity must be positive.")
    baseline_mean = baseline_scores.float().mean().item()
    steered_mean = steered_scores.float().mean().item()
    random_mean = random_control_scores.float().mean().item()
    steered_delta = steered_mean - baseline_mean
    random_delta = random_mean - baseline_mean
    perplexity_ratio = steered_perplexity / baseline_perplexity
    return SteeringSafetyReport(
        baseline_mean=baseline_mean,
        steered_mean=steered_mean,
        random_mean=random_mean,
        steered_delta=steered_delta,
        random_delta=random_delta,
        perplexity_ratio=perplexity_ratio,
        passes_control=steered_delta > random_delta,
        passes_perplexity_guard=perplexity_ratio <= max_perplexity_ratio,
    )


def direct_logit_attribution(
    decoder_vectors: t.Tensor,
    unembedding: t.Tensor,
    token_ids: t.Tensor | list[int] | None = None,
) -> t.Tensor:
    """Project decoder vectors into vocabulary-logit space."""

    if decoder_vectors.ndim != 2 or unembedding.ndim != 2:
        raise ValueError("decoder_vectors and unembedding must both be rank-2 tensors.")
    if decoder_vectors.shape[-1] != unembedding.shape[0]:
        raise ValueError("decoder_vectors last dimension must match unembedding first dimension.")
    logit_effects = decoder_vectors.float() @ unembedding.float()
    if token_ids is None:
        return logit_effects
    return logit_effects[:, t.as_tensor(token_ids, device=logit_effects.device)]


def metadata_and_tag_smoke_test() -> dict:
    metadata = FeatureArtifactMetadata(
        model_name="gemma-test",
        artifact_name="layer_0_resid_sae",
        artifact_type="sae",
        layer=0,
        hook_name="resid_post",
        d_model=4,
        n_features=8,
    )
    features = [
        TaggedFeatureSpec(0, 0, ("refusal", "safety"), "Refusal phrase feature"),
        TaggedFeatureSpec(1, 0, ("code",), "Code formatting feature"),
        TaggedFeatureSpec(2, 0, ("sentiment",), "Positive sentiment feature"),
    ]
    return {
        "metadata_complete": metadata_is_complete(metadata),
        "refusal_feature_ids": [
            feature.feature_id for feature in features_with_tag(features, "refusal")
        ],
        "code_feature_ids": [feature.feature_id for feature in features_with_tag(features, "code")],
    }


def score_reduction_smoke_test() -> dict:
    feature_acts = t.tensor(
        [
            [[0.0, 0.1], [0.0, 0.5], [0.0, 0.2]],
            [[0.0, 0.4], [0.0, 0.3], [0.0, 0.9]],
        ]
    )
    as_display_list = lambda values: [round(float(value), 6) for value in values]
    return {
        "max": as_display_list(feature_score_vector(feature_acts, 1, reduction="max")),
        "mean": as_display_list(feature_score_vector(feature_acts, 1, reduction="mean")),
        "last": as_display_list(feature_score_vector(feature_acts, 1, reduction="last")),
    }


def feature_validation_smoke_test() -> dict:
    scores = t.tensor([0.1, 0.2, 0.9, 1.0])
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)
    baseline_scores = t.tensor([1.0, 0.9, 0.2, 0.1])
    return validate_feature_scores(scores, labels, baseline_scores).__dict__


def base_instruction_delta_smoke_test() -> dict:
    base = t.tensor([0.1, 0.2, 0.3])
    instruction = t.tensor([0.3, 0.4, 0.5])
    return base_instruction_feature_delta(base, instruction).__dict__


def ablation_control_smoke_test() -> dict:
    baseline = t.tensor([1.0, 1.0])
    ablated = t.tensor([0.2, 0.3])
    random_ablated = t.tensor([0.8, 0.9])
    return ablation_control_report(baseline, ablated, random_ablated).__dict__


def steering_safety_smoke_test() -> dict:
    baseline = t.tensor([0.1, 0.2])
    steered = t.tensor([0.6, 0.7])
    random = t.tensor([0.2, 0.3])
    return steering_safety_report(
        baseline,
        steered,
        random,
        baseline_perplexity=10.0,
        steered_perplexity=11.0,
    ).__dict__


def logit_attribution_smoke_test() -> list[list[float]]:
    decoder_vectors = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return direct_logit_attribution(decoder_vectors, unembedding, token_ids=[0, 2]).tolist()


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "metadata": metadata_and_tag_smoke_test(),
        "score_reduction": score_reduction_smoke_test(),
        "validation": feature_validation_smoke_test(),
        "base_instruction_delta": base_instruction_delta_smoke_test(),
        "ablation": ablation_control_smoke_test(),
        "steering": steering_safety_smoke_test(),
        "logit_attribution": logit_attribution_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("Gemma Scope Deep Dive GPU verification requires CUDA.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    validation = validate_feature_scores(
        t.tensor([0.1, 0.2, 0.9, 1.0], device=device),
        t.tensor([0, 0, 1, 1], dtype=t.bool, device=device),
        t.tensor([1.0, 0.9, 0.2, 0.1], device=device),
    )
    ablation = ablation_control_report(
        t.tensor([1.0, 1.0], device=device),
        t.tensor([0.2, 0.3], device=device),
        t.tensor([0.8, 0.9], device=device),
    )
    steering = steering_safety_report(
        t.tensor([0.1, 0.2], device=device),
        t.tensor([0.6, 0.7], device=device),
        t.tensor([0.2, 0.3], device=device),
        baseline_perplexity=10.0,
        steered_perplexity=11.0,
    )
    attribution = direct_logit_attribution(
        t.eye(2, device=device),
        t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device),
        token_ids=[0, 2],
    )
    t.cuda.synchronize()
    toy_peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    artifact_preflight = gemma_scope_artifact_preflight(max_vram_gb=max_vram_gb)
    real_activation_preflight = gemma_scope_real_activation_preflight(
        max_vram_gb=max_vram_gb
    )
    base_access = gemma_base_model_access_report()
    peak_vram_gb = max(
        toy_peak_vram_gb,
        artifact_preflight["peak_vram_gb"],
        real_activation_preflight["peak_vram_gb"],
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "feature_auc": validation.feature_auc,
        "baseline_auc": validation.baseline_auc,
        "passes_baseline": validation.passes_baseline,
        "ablation_delta": ablation.ablation_delta,
        "steering_passes_control": steering.passes_control,
        "logit_attribution_sum": float(attribution.sum().item()),
        "gemma_scope_artifact_preflight_passed": artifact_preflight["preflight_passed"],
        "gemma_scope_repo_id": artifact_preflight["repo_id"],
        "gemma_scope_revision": artifact_preflight["revision"],
        "gemma_scope_artifact_path": artifact_preflight["artifact_path"],
        "gemma_scope_model_name": artifact_preflight["model_name"],
        "gemma_scope_architecture": artifact_preflight["architecture"],
        "gemma_scope_hook_point": artifact_preflight["hook_point_in"],
        "gemma_scope_layer": artifact_preflight["layer"],
        "gemma_scope_width": artifact_preflight["width"],
        "gemma_scope_d_model": artifact_preflight["d_model"],
        "gemma_scope_tensor_shapes": artifact_preflight["validation"]["tensor_shapes"],
        "gemma_scope_w_enc_shape": artifact_preflight["validation"]["tensor_shapes"][
            "w_enc"
        ],
        "gemma_scope_w_dec_shape": artifact_preflight["validation"]["tensor_shapes"][
            "w_dec"
        ],
        "gemma_scope_selected_feature_ids": artifact_preflight["selected_feature_ids"],
        "gemma_scope_selected_feature_margins": artifact_preflight["selected_feature_margins"],
        "gemma_scope_forward_passed": artifact_preflight["forward_passed"],
        "gemma_scope_constructed_probe_active_features_mean": artifact_preflight[
            "constructed_probe_active_features_mean"
        ],
        "gemma_scope_peak_vram_gb": artifact_preflight["peak_vram_gb"],
        "gemma_scope_artifact_semantic_feature_claimed": artifact_preflight[
            "semantic_feature_claimed"
        ],
        "gemma_scope_real_activation_preflight": real_activation_preflight,
        "gemma_scope_real_activation_preflight_passed": real_activation_preflight[
            "preflight_passed"
        ],
        "gemma_scope_real_activation_model_id": real_activation_preflight["model_id"],
        "gemma_scope_real_activation_layer": real_activation_preflight["model_layer"],
        "gemma_scope_real_activation_residual_shape": real_activation_preflight[
            "residual_shape"
        ],
        "gemma_scope_real_activation_feature_score_shape": real_activation_preflight[
            "feature_score_shape"
        ],
        "gemma_scope_real_activation_selected_feature_id": real_activation_preflight[
            "selected_feature_id"
        ],
        "gemma_scope_real_activation_random_control_feature_id": real_activation_preflight[
            "random_control_feature_id"
        ],
        "gemma_scope_real_activation_feature_auc": real_activation_preflight[
            "feature_auc"
        ],
        "gemma_scope_real_activation_baseline_auc": real_activation_preflight[
            "baseline_auc"
        ],
        "gemma_scope_real_activation_auc_margin": real_activation_preflight[
            "auc_margin"
        ],
        "gemma_scope_real_activation_label_shuffle_auc": real_activation_preflight[
            "label_shuffle_auc"
        ],
        "gemma_scope_real_activation_label_shuffle_control_passed": real_activation_preflight[
            "label_shuffle_control_passed"
        ],
        "gemma_scope_real_activation_forward_passed": real_activation_preflight[
            "real_activation_forward_passed"
        ],
        "gemma_scope_real_activation_peak_vram_gb": real_activation_preflight[
            "peak_vram_gb"
        ],
        "gemma_scope_semantic_feature_claimed": real_activation_preflight[
            "semantic_feature_claimed"
        ],
        "gemma3_base_access_report": base_access,
        "gemma3_base_authenticated": base_access["authenticated"],
        "gemma3_base_repo_listed": base_access["repo_listed"],
        "gemma3_base_local_non_ref_file_count": base_access["local_non_ref_file_count"],
        "gemma3_base_local_config_available": base_access["local_config_available"],
        "gemma3_base_local_tokenizer_available": base_access["local_tokenizer_available"],
        "gemma3_base_local_weight_available": base_access["local_weight_available"],
        "gemma3_base_missing_local_patterns": base_access["missing_local_patterns"],
        "gemma3_base_remote_config_available": base_access["remote_config_available"],
        "gemma3_base_remote_tokenizer_available": base_access["remote_tokenizer_available"],
        "gemma3_base_remote_weight_available": base_access["remote_weight_available"],
        "gemma3_base_missing_remote_patterns": base_access["missing_remote_patterns"],
        "gemma3_base_remote_ready_for_real_activations": base_access[
            "remote_ready_for_real_activations"
        ],
        "gemma3_base_local_ready_for_real_activations": base_access[
            "local_ready_for_real_activations"
        ],
        "gemma3_base_ready_for_real_activations": base_access[
            "ready_for_real_activations"
        ],
        "gemma3_base_access_error_type": base_access["access_error_type"],
        "gemma3_base_auth_error_type": base_access["auth_error_type"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": (
            "Pinned Gemma Scope 2 1B-IT residual SAE artifact preflight passes on CUDA, "
            "and authenticated Gemma 3 1B IT layer-13 activations are encoded through "
            "the SAE with held-out benign semantic feature validation and random-feature "
            "plus label-shuffle controls."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
