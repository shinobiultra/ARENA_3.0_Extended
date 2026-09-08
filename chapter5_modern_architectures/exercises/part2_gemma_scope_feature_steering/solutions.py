# %%
"""Reference solutions for [5.2] Gemma Scope and Feature Steering."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t
import torch.nn.functional as F

chapter = "chapter5_modern_architectures"
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
class SAEReconstructionMetrics:
    l0: float
    feature_density_mean: float
    dead_feature_fraction: float
    reconstruction_mse: float
    reconstruction_kl: float | None = None
    loss_recovered: float | None = None


@dataclass(frozen=True)
class FeatureDetectionReport:
    auc: float
    positive_mean: float
    negative_mean: float
    separation: float
    threshold_accuracy: float


@dataclass(frozen=True)
class SteeringComparisonReport:
    baseline_mean: float
    steered_mean: float
    random_mean: float
    steered_delta: float
    random_delta: float
    passes_control: bool


# %%
def synthetic_feature_batch() -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Return activations, reconstructions, and sparse feature activations."""

    t.manual_seed(0)
    activations = t.randn(4, 5, 6)
    reconstructed = activations + 0.05 * t.randn_like(activations)
    feature_acts = t.zeros(4, 5, 8)
    feature_acts[..., 0] = t.rand(4, 5)
    feature_acts[0:2, :, 3] = 2.0
    feature_acts[2:, :, 5] = 1.5
    return activations, reconstructed, feature_acts


def feature_density(feature_acts: t.Tensor, threshold: float = 0.0) -> t.Tensor:
    """Return the firing rate for each feature over all non-feature dimensions."""

    if feature_acts.ndim < 2:
        raise ValueError("feature_acts must have at least batch and feature dimensions.")
    fired = feature_acts > threshold
    reduce_dims = tuple(range(feature_acts.ndim - 1))
    return fired.float().mean(dim=reduce_dims)


def l0(feature_acts: t.Tensor, threshold: float = 0.0) -> float:
    """Return the average number of active features per activation vector."""

    return (feature_acts > threshold).float().sum(dim=-1).mean().item()


def dead_feature_fraction(feature_acts: t.Tensor, threshold: float = 0.0) -> float:
    """Return the fraction of features that never fire."""

    return feature_density(feature_acts, threshold=threshold).eq(0).float().mean().item()


def mean_kl_divergence(reference_logits: t.Tensor, reconstructed_logits: t.Tensor) -> float:
    """Compute mean KL(reference || reconstructed) across all non-vocab axes."""

    if reference_logits.shape != reconstructed_logits.shape:
        raise ValueError("reference_logits and reconstructed_logits must match.")
    ref_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    recon_log_probs = F.log_softmax(reconstructed_logits.float(), dim=-1)
    ref_probs = ref_log_probs.exp()
    return (ref_probs * (ref_log_probs - recon_log_probs)).sum(dim=-1).mean().item()


def loss_recovered(
    *,
    clean_loss: float,
    reconstructed_loss: float,
    zero_ablation_loss: float,
) -> float:
    """Return the usual SAE loss-recovered score."""

    denominator = zero_ablation_loss - clean_loss
    if denominator == 0:
        raise ValueError("zero_ablation_loss and clean_loss must differ.")
    return (zero_ablation_loss - reconstructed_loss) / denominator


def compute_sae_reconstruction_metrics(
    *,
    activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    feature_acts: t.Tensor,
    threshold: float = 0.0,
    reference_logits: t.Tensor | None = None,
    reconstructed_logits: t.Tensor | None = None,
    clean_loss: float | None = None,
    reconstructed_loss: float | None = None,
    zero_ablation_loss: float | None = None,
) -> SAEReconstructionMetrics:
    """Compute reconstruction, sparsity, KL, and loss-recovered metrics."""

    if activations.shape != reconstructed_activations.shape:
        raise ValueError("activations and reconstructed_activations must match.")

    reconstruction_kl = None
    if reference_logits is not None or reconstructed_logits is not None:
        if reference_logits is None or reconstructed_logits is None:
            raise ValueError("Provide both logits tensors, or neither.")
        reconstruction_kl = mean_kl_divergence(reference_logits, reconstructed_logits)

    recovered = None
    if clean_loss is not None or reconstructed_loss is not None or zero_ablation_loss is not None:
        if clean_loss is None or reconstructed_loss is None or zero_ablation_loss is None:
            raise ValueError("Provide clean, reconstructed, and zero-ablation losses together.")
        recovered = loss_recovered(
            clean_loss=clean_loss,
            reconstructed_loss=reconstructed_loss,
            zero_ablation_loss=zero_ablation_loss,
        )

    return SAEReconstructionMetrics(
        l0=l0(feature_acts, threshold=threshold),
        feature_density_mean=feature_density(feature_acts, threshold=threshold).mean().item(),
        dead_feature_fraction=dead_feature_fraction(feature_acts, threshold=threshold),
        reconstruction_mse=F.mse_loss(
            reconstructed_activations.float(),
            activations.float(),
        ).item(),
        reconstruction_kl=reconstruction_kl,
        loss_recovered=recovered,
    )


def direct_logit_attribution(
    decoder_vectors: t.Tensor,
    unembedding: t.Tensor,
    token_ids: t.Tensor | list[int] | None = None,
) -> t.Tensor:
    """Project feature decoder vectors through the unembedding matrix."""

    if decoder_vectors.ndim != 2 or unembedding.ndim != 2:
        raise ValueError("decoder_vectors and unembedding must both be rank-2.")
    if decoder_vectors.shape[-1] != unembedding.shape[0]:
        raise ValueError("decoder width must match unembedding residual dimension.")
    effects = decoder_vectors.float() @ unembedding.float()
    if token_ids is None:
        return effects
    return effects[:, t.as_tensor(token_ids, device=effects.device)]


def roc_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Compute binary ROC AUC using average ranks, including ties."""

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


def feature_detection_report(
    scores: t.Tensor,
    labels: t.Tensor,
    threshold: float | None = None,
) -> FeatureDetectionReport:
    """Evaluate a feature against held-out positives and matched negatives."""

    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same number of elements.")
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("Need at least one positive and one negative example.")

    if threshold is None:
        threshold = 0.5 * (positives.mean().item() + negatives.mean().item())
    predictions = scores >= threshold
    positive_mean = positives.mean().item()
    negative_mean = negatives.mean().item()
    return FeatureDetectionReport(
        auc=roc_auc_binary(scores, labels),
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        separation=positive_mean - negative_mean,
        threshold_accuracy=predictions.eq(labels).float().mean().item(),
    )


def top_activating_examples(
    feature_acts: t.Tensor,
    feature_id: int,
    k: int = 10,
) -> tuple[t.Tensor, t.Tensor]:
    """Return flattened positions and values for the top activations of one feature."""

    if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")
    scores = feature_acts[..., feature_id].flatten()
    values, indices = scores.topk(min(k, scores.numel()))
    return indices, values


def ablate_features(
    feature_acts: t.Tensor,
    feature_ids: t.Tensor | list[int],
    replacement: Literal["zero", "mean"] = "zero",
) -> t.Tensor:
    """Ablate selected sparse feature activations."""

    feature_ids_tensor = t.as_tensor(feature_ids, device=feature_acts.device, dtype=t.long)
    ablated = feature_acts.clone()
    if replacement == "zero":
        ablated[..., feature_ids_tensor] = 0
    elif replacement == "mean":
        means = feature_acts[..., feature_ids_tensor].mean(
            dim=tuple(range(feature_acts.ndim - 1)),
            keepdim=True,
        )
        ablated[..., feature_ids_tensor] = means
    else:
        raise ValueError("replacement must be 'zero' or 'mean'.")
    return ablated


def apply_decoder_steering(
    activations: t.Tensor,
    decoder_vectors: t.Tensor,
    feature_ids: t.Tensor | list[int],
    coefficients: t.Tensor | list[float] | float,
    *,
    positions: Literal["all", "last"] = "last",
) -> t.Tensor:
    """Add selected feature decoder directions to the residual stream."""

    feature_ids_tensor = t.as_tensor(feature_ids, device=activations.device, dtype=t.long)
    selected = decoder_vectors.to(device=activations.device, dtype=activations.dtype)[
        feature_ids_tensor
    ]
    coeffs = t.as_tensor(coefficients, device=activations.device, dtype=activations.dtype)
    if coeffs.ndim == 0:
        coeffs = coeffs.expand(selected.shape[0])
    if coeffs.numel() != selected.shape[0]:
        raise ValueError("coefficients must be scalar or match feature_ids.")
    steering_vector = (coeffs[:, None] * selected).sum(dim=0)

    steered = activations.clone()
    if positions == "all":
        steered = steered + steering_vector
    elif positions == "last":
        steered[..., -1, :] = steered[..., -1, :] + steering_vector
    else:
        raise ValueError("positions must be 'all' or 'last'.")
    return steered


def steering_comparison_report(
    baseline_scores: t.Tensor,
    steered_scores: t.Tensor,
    random_control_scores: t.Tensor,
) -> SteeringComparisonReport:
    """Compare feature steering against a random-feature control."""

    baseline_mean = baseline_scores.float().mean().item()
    steered_mean = steered_scores.float().mean().item()
    random_mean = random_control_scores.float().mean().item()
    steered_delta = steered_mean - baseline_mean
    random_delta = random_mean - baseline_mean
    return SteeringComparisonReport(
        baseline_mean=baseline_mean,
        steered_mean=steered_mean,
        random_mean=random_mean,
        steered_delta=steered_delta,
        random_delta=random_delta,
        passes_control=abs(steered_delta) > abs(random_delta),
    )


def compute_metrics_smoke_test() -> dict:
    activations, reconstructed, feature_acts = synthetic_feature_batch()
    metrics = compute_sae_reconstruction_metrics(
        activations=activations,
        reconstructed_activations=reconstructed,
        feature_acts=feature_acts,
        clean_loss=1.0,
        reconstructed_loss=1.2,
        zero_ablation_loss=2.0,
    )
    return metrics.__dict__


def feature_validation_smoke_test() -> dict:
    scores = t.tensor([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    labels = t.tensor([1, 1, 1, 0, 0, 0], dtype=t.bool)
    return feature_detection_report(scores, labels).__dict__


def logit_attribution_smoke_test() -> t.Tensor:
    decoder_vectors = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return direct_logit_attribution(decoder_vectors, unembedding, token_ids=[0, 2])


def steering_smoke_test() -> dict:
    activations = t.zeros(2, 3, 4)
    decoder_vectors = t.eye(4)
    steered = apply_decoder_steering(activations, decoder_vectors, [1], 2.0)
    assert steered[:, -1, 1].eq(2.0).all()

    report = steering_comparison_report(
        baseline_scores=t.tensor([0.1, 0.2, 0.3]),
        steered_scores=t.tensor([0.5, 0.6, 0.7]),
        random_control_scores=t.tensor([0.2, 0.2, 0.3]),
    )
    return report.__dict__


def ablation_smoke_test() -> dict:
    _, _, feature_acts = synthetic_feature_batch()
    ablated = ablate_features(feature_acts, [3], replacement="zero")
    indices, values = top_activating_examples(feature_acts, feature_id=3, k=3)
    return {
        "feature_3_zeroed": bool(ablated[..., 3].eq(0).all()),
        "top_indices": indices.tolist(),
        "top_values": values.tolist(),
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "metrics": compute_metrics_smoke_test(),
        "feature_validation": feature_validation_smoke_test(),
        "logit_attribution": logit_attribution_smoke_test().tolist(),
        "steering": steering_smoke_test(),
        "ablation": ablation_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("Gemma Scope and Feature Steering GPU verification requires CUDA.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    activations = t.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, -1.0]]],
        device=device,
    )
    reconstructed = activations + 0.01
    feature_acts = t.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 2.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, 3.0, 1.0]]],
        device=device,
    )
    metrics = compute_sae_reconstruction_metrics(
        activations=activations,
        reconstructed_activations=reconstructed,
        feature_acts=feature_acts,
        clean_loss=1.0,
        reconstructed_loss=1.1,
        zero_ablation_loss=2.0,
    )
    feature_report = feature_detection_report(
        t.tensor([0.9, 0.8, 0.1, 0.0], device=device),
        t.tensor([1, 1, 0, 0], dtype=t.bool, device=device),
    )
    steering_report = steering_comparison_report(
        baseline_scores=t.tensor([0.1, 0.2, 0.3], device=device),
        steered_scores=t.tensor([0.7, 0.8, 0.9], device=device),
        random_control_scores=t.tensor([0.2, 0.25, 0.2], device=device),
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
        "reconstruction_mse": metrics.reconstruction_mse,
        "l0": metrics.l0,
        "feature_auc": feature_report.auc,
        "steering_passes_control": steering_report.passes_control,
        "gemma_scope_artifact_preflight_passed": artifact_preflight["preflight_passed"],
        "gemma_scope_repo_id": artifact_preflight["repo_id"],
        "gemma_scope_revision": artifact_preflight["revision"],
        "gemma_scope_artifact_path": artifact_preflight["artifact_path"],
        "gemma_scope_model_name": artifact_preflight["model_name"],
        "gemma_scope_architecture": artifact_preflight["architecture"],
        "gemma_scope_hook_point": artifact_preflight["hook_point_in"],
        "gemma_scope_width": artifact_preflight["width"],
        "gemma_scope_d_model": artifact_preflight["d_model"],
        "gemma_scope_w_enc_shape": artifact_preflight["validation"]["tensor_shapes"][
            "w_enc"
        ],
        "gemma_scope_w_dec_shape": artifact_preflight["validation"]["tensor_shapes"][
            "w_dec"
        ],
        "gemma_scope_forward_passed": artifact_preflight["forward_passed"],
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
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)
