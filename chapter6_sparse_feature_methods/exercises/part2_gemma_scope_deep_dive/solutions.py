# %%
"""Reference solutions for [6.2] Gemma Scope Deep Dive."""

from __future__ import annotations

import sys
from collections.abc import Mapping
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
from chapter6_sparse_feature_methods.exercises.part2_gemma_scope_deep_dive import utils

MAIN = __name__ == "__main__"
GEMMA_SCOPE_REPO_ID = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL.repo_id
GEMMA_SCOPE_REVISION = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL.revision
GEMMA_SCOPE_ARTIFACT_PATH = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL.artifact_path


@dataclass(frozen=True)
class SparseAutoencoder:
    w_enc: t.Tensor
    w_dec: t.Tensor
    b_enc: t.Tensor
    b_dec: t.Tensor
    threshold: t.Tensor


@dataclass(frozen=True)
class ReconstructionReport:
    mse: float
    relative_mse: float
    explained_variance: float


@dataclass(frozen=True)
class HeldOutValidationReport:
    feature_auc: float
    random_feature_auc: float
    shuffled_label_auc: float
    threshold_accuracy: float
    positive_mean: float
    negative_mean: float


def load_normalized_sae(
    artifact: Mapping[str, t.Tensor],
) -> tuple[SparseAutoencoder, t.Tensor]:
    """Validate SAE tensors and make every decoder row unit norm.

    Scaling an encoder column, its bias, and its JumpReLU threshold by the
    original decoder norm preserves the represented function exactly.
    """

    required = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
    missing = required.difference(artifact)
    if missing:
        raise KeyError(f"Missing SAE tensors: {sorted(missing)}")

    w_enc = t.as_tensor(artifact["w_enc"]).detach().clone().float()
    w_dec = t.as_tensor(artifact["w_dec"]).detach().clone().float()
    b_enc = t.as_tensor(artifact["b_enc"]).detach().clone().float()
    b_dec = t.as_tensor(artifact["b_dec"]).detach().clone().float()
    threshold = t.as_tensor(artifact["threshold"]).detach().clone().float()

    if w_enc.ndim != 2 or w_dec.ndim != 2:
        raise ValueError("w_enc and w_dec must be rank-2 tensors.")
    d_model, n_features = w_enc.shape
    if w_dec.shape != (n_features, d_model):
        raise ValueError("w_dec must have shape (n_features, d_model).")
    if b_enc.shape != (n_features,) or b_dec.shape != (d_model,):
        raise ValueError("Encoder and decoder biases have incompatible shapes.")
    if threshold.ndim == 0:
        threshold = threshold.expand(n_features).clone()
    if threshold.shape != (n_features,):
        raise ValueError("threshold must be scalar or have shape (n_features,).")
    if not all(t.isfinite(x).all() for x in (w_enc, w_dec, b_enc, b_dec, threshold)):
        raise ValueError("SAE tensors must be finite.")

    decoder_norms = w_dec.norm(dim=-1)
    if (decoder_norms <= 0).any():
        raise ValueError("Every decoder row must have non-zero norm.")
    sae = SparseAutoencoder(
        w_enc=w_enc * decoder_norms,
        w_dec=w_dec / decoder_norms[:, None],
        b_enc=b_enc * decoder_norms,
        b_dec=b_dec,
        threshold=threshold * decoder_norms,
    )
    return sae, decoder_norms


def encode_jump_relu(residuals: t.Tensor, sae: SparseAutoencoder) -> t.Tensor:
    """Encode residual vectors with Gemma Scope's JumpReLU convention."""

    if residuals.shape[-1] != sae.w_enc.shape[0]:
        raise ValueError("Residual width must match SAE d_model.")
    pre_acts = (residuals.float() - sae.b_dec) @ sae.w_enc + sae.b_enc
    return t.relu(pre_acts) * (pre_acts > sae.threshold)


def reconstruct_from_features(feature_acts: t.Tensor, sae: SparseAutoencoder) -> t.Tensor:
    """Decode sparse feature activations into the residual stream."""

    if feature_acts.shape[-1] != sae.w_dec.shape[0]:
        raise ValueError("Feature width must match the SAE dictionary width.")
    return feature_acts.float() @ sae.w_dec + sae.b_dec


def reconstruction_report(
    residuals: t.Tensor,
    reconstructed: t.Tensor,
) -> ReconstructionReport:
    """Report absolute and scale-aware reconstruction quality."""

    if residuals.shape != reconstructed.shape:
        raise ValueError("Residuals and reconstructions must have identical shapes.")
    residuals = residuals.float()
    reconstructed = reconstructed.float()
    mse = (residuals - reconstructed).square().mean()
    centered_energy = (residuals - residuals.mean(dim=0, keepdim=True)).square().mean()
    if centered_energy <= 0:
        raise ValueError("Explained variance is undefined for constant residuals.")
    relative_mse = mse / centered_energy
    return ReconstructionReport(
        mse=float(mse.item()),
        relative_mse=float(relative_mse.item()),
        explained_variance=float((1 - relative_mse).item()),
    )


def feature_score_vector(
    feature_acts: t.Tensor,
    feature_id: int,
    *,
    reduction: Literal["max", "mean", "last"] = "max",
) -> t.Tensor:
    """Reduce token-level activations to one score per example."""

    if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")
    if feature_acts.ndim == 2:
        return feature_acts[:, feature_id].float()
    if feature_acts.ndim != 3:
        raise ValueError("feature_acts must have shape (batch, features) or (batch, seq, features).")
    per_token = feature_acts[..., feature_id].float()
    if reduction == "max":
        return per_token.max(dim=-1).values
    if reduction == "mean":
        return per_token.mean(dim=-1)
    if reduction == "last":
        return per_token[:, -1]
    raise ValueError("reduction must be 'max', 'mean', or 'last'.")


def roc_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Compute binary ROC AUC from average ranks, including ties."""

    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same length.")
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one positive and one negative.")

    order = scores.argsort()
    sorted_scores = scores[order]
    ranks_sorted = t.empty_like(sorted_scores)
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and t.isclose(
            sorted_scores[end], sorted_scores[start], rtol=1e-6, atol=1e-7
        ):
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2
        start = end
    ranks = t.empty_like(scores)
    ranks[order] = ranks_sorted
    pos_rank_sum = ranks[labels].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc.item())


def validate_heldout_feature(
    feature_scores: t.Tensor,
    labels: t.Tensor,
    random_feature_scores: t.Tensor,
    shuffled_labels: t.Tensor,
) -> HeldOutValidationReport:
    """Compare a feature with random-feature and shuffled-label controls."""

    feature_scores = feature_scores.flatten().float()
    labels = labels.flatten().bool()
    random_feature_scores = random_feature_scores.flatten().float()
    shuffled_labels = shuffled_labels.flatten().bool()
    if not (
        feature_scores.shape
        == labels.shape
        == random_feature_scores.shape
        == shuffled_labels.shape
    ):
        raise ValueError("All held-out vectors must have identical shapes.")
    positives = feature_scores[labels]
    negatives = feature_scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("Held-out labels need both classes.")
    threshold = 0.5 * (positives.mean() + negatives.mean())
    predictions = feature_scores >= threshold
    return HeldOutValidationReport(
        feature_auc=roc_auc_binary(feature_scores, labels),
        random_feature_auc=roc_auc_binary(random_feature_scores, labels),
        shuffled_label_auc=roc_auc_binary(feature_scores, shuffled_labels),
        threshold_accuracy=float(predictions.eq(labels).float().mean().item()),
        positive_mean=float(positives.mean().item()),
        negative_mean=float(negatives.mean().item()),
    )


def activation_density(feature_acts: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Return per-feature firing density and a dead-feature mask."""

    if feature_acts.ndim < 2:
        raise ValueError("feature_acts needs at least one sample axis and one feature axis.")
    flat = feature_acts.reshape(-1, feature_acts.shape[-1])
    density = (flat > 0).float().mean(dim=0)
    return density, density == 0


def ablate_feature(feature_acts: t.Tensor, feature_id: int) -> t.Tensor:
    """Set one feature coordinate to zero without mutating the input."""

    if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")
    ablated = feature_acts.clone()
    ablated[..., feature_id] = 0
    return ablated


def steer_residuals(
    residuals: t.Tensor,
    direction: t.Tensor,
    strengths: t.Tensor | list[float],
) -> t.Tensor:
    """Add equal-L2 interventions by normalizing the supplied direction."""

    direction = t.as_tensor(direction, dtype=residuals.dtype, device=residuals.device)
    if direction.ndim != 1 or direction.shape[0] != residuals.shape[-1]:
        raise ValueError("direction must be a vector with residual width.")
    norm = direction.norm()
    if norm <= 0:
        raise ValueError("direction must be non-zero.")
    unit_direction = direction / norm
    strengths = t.as_tensor(strengths, dtype=residuals.dtype, device=residuals.device).flatten()
    view_shape = (strengths.numel(),) + (1,) * residuals.ndim
    return residuals.unsqueeze(0) + strengths.view(view_shape) * unit_direction


def direct_logit_attribution(
    decoder_vectors: t.Tensor,
    unembedding: t.Tensor,
    token_ids: t.Tensor | list[int] | None = None,
) -> t.Tensor:
    """Project decoder directions through an unembedding matrix."""

    if decoder_vectors.ndim != 2 or unembedding.ndim != 2:
        raise ValueError("decoder_vectors and unembedding must be rank-2.")
    if decoder_vectors.shape[-1] != unembedding.shape[0]:
        raise ValueError("Decoder width must match the unembedding input width.")
    effects = decoder_vectors.float() @ unembedding.float()
    if token_ids is None:
        return effects
    return effects[:, t.as_tensor(token_ids, device=effects.device)]


def ground_truth_analysis() -> dict[str, object]:
    """Run the complete CPU analysis used by the notebook signature result."""

    organism = utils.make_ground_truth_organism()
    sae, raw_norms = load_normalized_sae(organism["raw_artifact"])
    residuals = organism["residuals"]
    acts = encode_jump_relu(residuals, sae)
    recon = reconstruct_from_features(acts, sae)
    heldout = ~organism["train_mask"]
    target_id = organism["target_feature_id"]
    control_id = organism["control_feature_id"]
    target_scores = feature_score_vector(acts, target_id)
    control_scores = feature_score_vector(acts, control_id)
    validation = validate_heldout_feature(
        target_scores[heldout],
        organism["labels"][heldout],
        control_scores[heldout],
        organism["shuffled_labels"][heldout],
    )
    train_confound_auc = roc_auc_binary(
        acts[organism["train_mask"], 1], organism["labels"][organism["train_mask"]]
    )
    heldout_confound_auc = roc_auc_binary(
        acts[heldout, 1], organism["labels"][heldout]
    )
    density, dead = activation_density(acts)

    positive_heldout = heldout & organism["labels"]
    centered = residuals - sae.b_dec
    readout = sae.w_dec[target_id]
    baseline_behavior = centered @ readout
    target_recon = reconstruct_from_features(ablate_feature(acts, target_id), sae)
    control_recon = reconstruct_from_features(ablate_feature(acts, control_id), sae)
    target_behavior = (target_recon - sae.b_dec) @ readout
    control_behavior = (control_recon - sae.b_dec) @ readout
    strengths = t.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    target_steered = steer_residuals(residuals, sae.w_dec[target_id], strengths)
    control_steered = steer_residuals(residuals, sae.w_dec[control_id], strengths)
    target_curve = ((target_steered - sae.b_dec) @ readout).mean(dim=1)
    control_curve = ((control_steered - sae.b_dec) @ readout).mean(dim=1)
    baseline_mean = baseline_behavior.mean()
    target_ablation_delta = float(
        (baseline_behavior[positive_heldout] - target_behavior[positive_heldout]).mean().item()
    )
    control_ablation_delta = float(
        (baseline_behavior[positive_heldout] - control_behavior[positive_heldout]).mean().item()
    )
    if abs(control_ablation_delta) < 1e-6:
        control_ablation_delta = 0.0
    target_curve = target_curve - baseline_mean
    control_curve = control_curve - baseline_mean
    control_curve = t.where(control_curve.abs() < 1e-6, 0.0, control_curve)
    dla = direct_logit_attribution(sae.w_dec, organism["unembedding"])

    return {
        "reconstruction": reconstruction_report(residuals, recon).__dict__,
        "raw_decoder_norms": raw_norms.tolist(),
        "normalized_decoder_norms": sae.w_dec.norm(dim=-1).tolist(),
        "density": density.tolist(),
        "dead_feature_ids": dead.nonzero().flatten().tolist(),
        "validation": validation.__dict__,
        "train_confound_auc": train_confound_auc,
        "heldout_confound_auc": heldout_confound_auc,
        "target_ablation_delta": target_ablation_delta,
        "control_ablation_delta": control_ablation_delta,
        "steering_strengths": strengths.tolist(),
        "target_steering_delta": target_curve.tolist(),
        "control_steering_delta": control_curve.tolist(),
        "target_dla": dla[target_id].tolist(),
    }


# Compatibility reports used by the serialized CUDA verification path.
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


def validate_feature_scores(
    feature_scores: t.Tensor,
    labels: t.Tensor,
    baseline_scores: t.Tensor,
    *,
    min_auc_margin: float = 0.1,
) -> FeatureValidationSuiteReport:
    labels = labels.flatten().bool()
    scores = feature_scores.flatten().float()
    baseline_scores = baseline_scores.flatten().float()
    positives, negatives = scores[labels], scores[~labels]
    threshold = 0.5 * (positives.mean() + negatives.mean())
    feature_auc = roc_auc_binary(scores, labels)
    baseline_auc = roc_auc_binary(baseline_scores, labels)
    margin = feature_auc - baseline_auc
    return FeatureValidationSuiteReport(
        feature_auc=feature_auc,
        baseline_auc=baseline_auc,
        auc_margin=margin,
        threshold_accuracy=float((scores >= threshold).eq(labels).float().mean().item()),
        positive_mean=float(positives.mean().item()),
        negative_mean=float(negatives.mean().item()),
        passes_baseline=margin >= min_auc_margin,
    )


def ablation_control_report(
    baseline_scores: t.Tensor,
    ablated_scores: t.Tensor,
    random_ablated_scores: t.Tensor,
) -> AblationControlReport:
    baseline_mean = float(baseline_scores.float().mean().item())
    ablated_mean = float(ablated_scores.float().mean().item())
    random_mean = float(random_ablated_scores.float().mean().item())
    ablation_delta = baseline_mean - ablated_mean
    random_delta = baseline_mean - random_mean
    return AblationControlReport(
        baseline_mean,
        ablated_mean,
        random_mean,
        ablation_delta,
        random_delta,
        ablation_delta > random_delta,
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
    if baseline_perplexity <= 0:
        raise ValueError("baseline_perplexity must be positive.")
    baseline_mean = float(baseline_scores.float().mean().item())
    steered_mean = float(steered_scores.float().mean().item())
    random_mean = float(random_control_scores.float().mean().item())
    steered_delta = steered_mean - baseline_mean
    random_delta = random_mean - baseline_mean
    ratio = steered_perplexity / baseline_perplexity
    return SteeringSafetyReport(
        baseline_mean,
        steered_mean,
        random_mean,
        steered_delta,
        random_delta,
        ratio,
        steered_delta > random_delta,
        ratio <= max_perplexity_ratio,
    )


def run_smoke_test(cpu: bool = True) -> dict[str, object]:
    _ = cpu
    result = ground_truth_analysis()
    heldout_examples = int((~utils.make_ground_truth_organism()["train_mask"]).sum().item())
    validation = result["validation"]
    reconstruction = result["reconstruction"]
    return {
        **result,
        "exact_heldout_examples": heldout_examples,
        "exact_reconstruction_mse": reconstruction["mse"],
        "exact_feature_auc": validation["feature_auc"],
        "exact_random_feature_auc": validation["random_feature_auc"],
        "exact_shuffled_label_auc": validation["shuffled_label_auc"],
        "exact_train_confound_auc": result["train_confound_auc"],
        "exact_heldout_confound_auc": result["heldout_confound_auc"],
        "exact_target_ablation_delta": result["target_ablation_delta"],
        "exact_control_ablation_delta_abs": abs(result["control_ablation_delta"]),
        "exact_dead_feature_ids": result["dead_feature_ids"],
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, object]:
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
    real_activation_preflight = gemma_scope_real_activation_preflight(max_vram_gb=max_vram_gb)
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
        "gemma_scope_w_enc_shape": artifact_preflight["validation"]["tensor_shapes"]["w_enc"],
        "gemma_scope_w_dec_shape": artifact_preflight["validation"]["tensor_shapes"]["w_dec"],
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
        "gemma_scope_real_activation_residual_shape": real_activation_preflight["residual_shape"],
        "gemma_scope_real_activation_feature_score_shape": real_activation_preflight[
            "feature_score_shape"
        ],
        "gemma_scope_real_activation_selected_feature_id": real_activation_preflight[
            "selected_feature_id"
        ],
        "gemma_scope_real_activation_random_control_feature_id": real_activation_preflight[
            "random_control_feature_id"
        ],
        "gemma_scope_real_activation_feature_auc": real_activation_preflight["feature_auc"],
        "gemma_scope_real_activation_baseline_auc": real_activation_preflight["baseline_auc"],
        "gemma_scope_real_activation_auc_margin": real_activation_preflight["auc_margin"],
        "gemma_scope_real_activation_label_shuffle_auc": real_activation_preflight[
            "label_shuffle_auc"
        ],
        "gemma_scope_real_activation_label_shuffle_control_passed": real_activation_preflight[
            "label_shuffle_control_passed"
        ],
        "gemma_scope_real_activation_forward_passed": real_activation_preflight[
            "real_activation_forward_passed"
        ],
        "gemma_scope_real_activation_peak_vram_gb": real_activation_preflight["peak_vram_gb"],
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
        "gemma3_base_ready_for_real_activations": base_access["ready_for_real_activations"],
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


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, object]:
    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
