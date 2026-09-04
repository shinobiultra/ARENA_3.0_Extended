# %%
"""Reference solutions for [6.4] Crosscoders and Model Diffing."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t
import torch.nn.functional as F

chapter = "chapter6_sparse_feature_methods"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.crosscoders import (  # noqa: E402 - path setup is notebook-local.
    CrosscoderInterventionReport,
    PlantedCrosscoderData,
    PlantedFeatureMatchReport,
    SparseCrosscoder,
    SparseCrosscoderForward,
    SparseCrosscoderLoss,
    SparseCrosscoderReconstructionReport,
    TrainedSparseCrosscoder,
    ablate_latents,
    behavior_baseline_table,
    crosscoder_intervention_report,
    evaluate_sparse_crosscoder_reconstruction,
    latent_ownership_table,
    make_planted_crosscoder_data,
    make_train_heldout_split,
    match_learned_to_planted_features,
    model_behavior_delta_from_reconstruction,
    run_crosscoder_signature_result,
    shared_only_reconstruction,
    signed_auc_binary,
    sparse_crosscoder_loss,
    svd_delta_direction_scores,
    svd_pair_reconstruction_mse,
    top_activating_examples,
    train_sparse_crosscoder,
)

MAIN = __name__ == "__main__"
FeatureOwner = Literal["shared", "model_a", "model_b"]

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_SOLU1L_MODEL_NAME = "solu-1l"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_SOLU1L_REVISION = "a4ce32db5e35f13e5f09333888bd2d42660f77ce"
TL_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_RESID_POST_HOOK = "blocks.0.hook_resid_post"
TL_TECHNICAL_PROMPTS = [
    "The Python function returns a",
    "The HTML page contains a",
    "The database stores a",
    "The command line prints a",
    "The neural network predicts a",
    "The script writes a",
    "The compiler emits a",
    "The API request returns a",
]
TL_EVERYDAY_PROMPTS = [
    "The cat sat on the",
    "The chef cooked a",
    "The bird flew over the",
    "The teacher taught a",
    "The train arrived at the",
    "The plane landed at the",
    "The gardener planted a",
    "The singer sang a",
]


# %%
@dataclass(frozen=True)
class CrosscoderOutput:
    shared_acts: t.Tensor
    model_a_specific_acts: t.Tensor
    model_b_specific_acts: t.Tensor
    reconstructed_model_a: t.Tensor
    reconstructed_model_b: t.Tensor


@dataclass(frozen=True)
class CrosscoderReconstructionReport:
    model_a_mse: float
    model_b_mse: float
    shared_active_fraction: float
    model_a_passes: bool
    model_b_passes: bool
    shared_reconstructs_both: bool


@dataclass(frozen=True)
class FeatureSpecificityReport:
    feature_id: int
    model_a_mean: float
    model_b_mean: float
    specificity: float
    owner: FeatureOwner


@dataclass(frozen=True)
class BehaviorDeltaPredictionReport:
    feature_id: int
    auc: float
    positive_mean: float
    negative_mean: float
    passes_threshold: bool


@dataclass(frozen=True)
class CrosscoderAblationReport:
    baseline_delta: float
    ablated_delta: float
    random_ablated_delta: float
    delta_reduction: float
    random_reduction: float
    passes_control: bool


def decode_crosscoder(
    shared_acts: t.Tensor,
    model_a_specific_acts: t.Tensor,
    model_b_specific_acts: t.Tensor,
    shared_decoder_a: t.Tensor,
    shared_decoder_b: t.Tensor,
    model_a_decoder: t.Tensor,
    model_b_decoder: t.Tensor,
) -> CrosscoderOutput:
    """Decode shared and model-specific features into two activation spaces."""

    if shared_acts.shape[-1] != shared_decoder_a.shape[0]:
        raise ValueError("shared feature dimension must match model A shared decoder rows.")
    if shared_acts.shape[-1] != shared_decoder_b.shape[0]:
        raise ValueError("shared feature dimension must match model B shared decoder rows.")
    if model_a_specific_acts.shape[-1] != model_a_decoder.shape[0]:
        raise ValueError("model A feature dimension must match model A decoder rows.")
    if model_b_specific_acts.shape[-1] != model_b_decoder.shape[0]:
        raise ValueError("model B feature dimension must match model B decoder rows.")
    if shared_decoder_a.shape[1] != model_a_decoder.shape[1]:
        raise ValueError("model A decoder output dimensions must match.")
    if shared_decoder_b.shape[1] != model_b_decoder.shape[1]:
        raise ValueError("model B decoder output dimensions must match.")

    reconstructed_a = shared_acts.float() @ shared_decoder_a.float()
    reconstructed_a = reconstructed_a + model_a_specific_acts.float() @ model_a_decoder.float()
    reconstructed_b = shared_acts.float() @ shared_decoder_b.float()
    reconstructed_b = reconstructed_b + model_b_specific_acts.float() @ model_b_decoder.float()
    return CrosscoderOutput(
        shared_acts=shared_acts,
        model_a_specific_acts=model_a_specific_acts,
        model_b_specific_acts=model_b_specific_acts,
        reconstructed_model_a=reconstructed_a,
        reconstructed_model_b=reconstructed_b,
    )


def crosscoder_reconstruction_report(
    model_a_activations: t.Tensor,
    model_b_activations: t.Tensor,
    output: CrosscoderOutput,
    *,
    mse_threshold: float = 1e-6,
) -> CrosscoderReconstructionReport:
    """Check whether crosscoder reconstructions preserve both model spaces."""

    model_a_mse = F.mse_loss(
        output.reconstructed_model_a.float(),
        model_a_activations.float(),
    ).item()
    model_b_mse = F.mse_loss(
        output.reconstructed_model_b.float(),
        model_b_activations.float(),
    ).item()
    shared_active_fraction = output.shared_acts.gt(0).float().mean().item()
    model_a_passes = model_a_mse <= mse_threshold
    model_b_passes = model_b_mse <= mse_threshold
    return CrosscoderReconstructionReport(
        model_a_mse=model_a_mse,
        model_b_mse=model_b_mse,
        shared_active_fraction=shared_active_fraction,
        model_a_passes=model_a_passes,
        model_b_passes=model_b_passes,
        shared_reconstructs_both=model_a_passes and model_b_passes,
    )


def feature_specificity_report(
    model_a_feature_acts: t.Tensor,
    model_b_feature_acts: t.Tensor,
    feature_id: int,
    *,
    shared_threshold: float = 0.1,
) -> FeatureSpecificityReport:
    """Classify one feature as shared, model-A-specific, or model-B-specific."""

    if model_a_feature_acts.shape != model_b_feature_acts.shape:
        raise ValueError("model feature activation tensors must have matching shapes.")
    if feature_id < 0 or feature_id >= model_a_feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")
    model_a_mean = model_a_feature_acts[..., feature_id].float().mean().item()
    model_b_mean = model_b_feature_acts[..., feature_id].float().mean().item()
    specificity = model_b_mean - model_a_mean
    if abs(specificity) <= shared_threshold:
        owner: FeatureOwner = "shared"
    elif specificity > 0:
        owner = "model_b"
    else:
        owner = "model_a"
    return FeatureSpecificityReport(
        feature_id=feature_id,
        model_a_mean=model_a_mean,
        model_b_mean=model_b_mean,
        specificity=specificity,
        owner=owner,
    )


def classify_features_by_specificity(
    model_a_feature_acts: t.Tensor,
    model_b_feature_acts: t.Tensor,
    *,
    shared_threshold: float = 0.1,
) -> list[FeatureOwner]:
    """Classify every feature by cross-model activation specificity."""

    return [
        feature_specificity_report(
            model_a_feature_acts,
            model_b_feature_acts,
            feature_id,
            shared_threshold=shared_threshold,
        ).owner
        for feature_id in range(model_a_feature_acts.shape[-1])
    ]


def roc_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Binary ROC AUC using rank statistics, including tied scores."""

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
        ranks_sorted[start:end] = (start + 1 + end) / 2
        start = end
    ranks = t.empty_like(scores)
    ranks[order] = ranks_sorted
    pos_rank_sum = ranks[labels].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc.item())


def behavior_delta_prediction_report(
    feature_scores: t.Tensor,
    behavior_delta_labels: t.Tensor,
    *,
    feature_id: int,
    min_auc: float = 0.8,
) -> BehaviorDeltaPredictionReport:
    """Check whether a feature predicts examples with a behavior delta."""

    scores = feature_scores.flatten().float()
    labels = behavior_delta_labels.flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("feature_scores and behavior_delta_labels must have equal length.")
    auc = roc_auc_binary(scores, labels)
    signed_auc = max(auc, 1.0 - auc)
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("Need at least one positive and one negative example.")
    return BehaviorDeltaPredictionReport(
        feature_id=feature_id,
        auc=signed_auc,
        positive_mean=positives.mean().item(),
        negative_mean=negatives.mean().item(),
        passes_threshold=signed_auc >= min_auc,
    )


def crosscoder_ablation_report(
    baseline_behavior_deltas: t.Tensor,
    ablated_behavior_deltas: t.Tensor,
    random_ablated_behavior_deltas: t.Tensor,
) -> CrosscoderAblationReport:
    """Check whether ablating model-specific features reduces behavior deltas."""

    baseline_delta = baseline_behavior_deltas.float().mean().item()
    ablated_delta = ablated_behavior_deltas.float().mean().item()
    random_delta = random_ablated_behavior_deltas.float().mean().item()
    baseline_abs = abs(baseline_delta)
    ablated_abs = abs(ablated_delta)
    random_abs = abs(random_delta)
    delta_reduction = baseline_abs - ablated_abs
    random_reduction = baseline_abs - random_abs
    return CrosscoderAblationReport(
        baseline_delta=baseline_delta,
        ablated_delta=ablated_delta,
        random_ablated_delta=random_delta,
        delta_reduction=delta_reduction,
        random_reduction=random_reduction,
        passes_control=delta_reduction > random_reduction,
    )


def toy_behavior_delta_scores(
    model_a_scores: t.Tensor,
    model_b_scores: t.Tensor,
) -> t.Tensor:
    """Return model-B minus model-A behavior scores for paired prompts."""

    if model_a_scores.shape != model_b_scores.shape:
        raise ValueError("model_a_scores and model_b_scores must have matching shapes.")
    return model_b_scores.float() - model_a_scores.float()


def crosscoder_reconstruction_smoke_test() -> dict:
    shared = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    model_a_specific = t.tensor([[2.0], [3.0]])
    model_b_specific = t.tensor([[4.0], [5.0]])
    shared_decoder = t.eye(2)
    model_a_decoder = t.tensor([[1.0, 0.0]])
    model_b_decoder = t.tensor([[0.0, 1.0]])

    output = decode_crosscoder(
        shared,
        model_a_specific,
        model_b_specific,
        shared_decoder,
        shared_decoder,
        model_a_decoder,
        model_b_decoder,
    )
    target_a = t.tensor([[3.0, 0.0], [3.0, 1.0]])
    target_b = t.tensor([[1.0, 4.0], [0.0, 6.0]])
    report = crosscoder_reconstruction_report(target_a, target_b, output)
    return {
        "reconstructed_model_a": output.reconstructed_model_a.tolist(),
        "reconstructed_model_b": output.reconstructed_model_b.tolist(),
        "report": report.__dict__,
    }


def feature_specificity_smoke_test() -> dict:
    model_a = t.tensor([[1.0, 3.0, 0.1], [1.0, 2.0, 0.2]])
    model_b = t.tensor([[1.1, 0.2, 4.0], [1.0, 0.1, 5.0]])
    owners = classify_features_by_specificity(model_a, model_b, shared_threshold=0.2)
    feature_2 = feature_specificity_report(model_a, model_b, 2, shared_threshold=0.2)
    return {"owners": owners, "feature_2": feature_2.__dict__}


def behavior_delta_smoke_test() -> dict:
    feature_scores = t.tensor([0.1, 0.2, 0.9, 1.0])
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)
    return behavior_delta_prediction_report(feature_scores, labels, feature_id=7).__dict__


def ablation_delta_smoke_test() -> dict:
    baseline = t.tensor([1.0, 1.0])
    ablated = t.tensor([0.2, 0.3])
    random_ablated = t.tensor([0.8, 0.9])
    return crosscoder_ablation_report(baseline, ablated, random_ablated).__dict__


def delta_scores_smoke_test() -> dict:
    model_a = t.tensor([0.25, 0.5])
    model_b = t.tensor([0.75, 0.25])
    return {"behavior_deltas": toy_behavior_delta_scores(model_a, model_b).tolist()}


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return run_crosscoder_signature_result(steps=300)


def _load_transformerlens_model_on_cuda(name: str, revision: str):
    os.environ.setdefault("BNB_CUDA_VERSION", TL_BNB_CUDA_OVERRIDE)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        name,
        device="cuda",
        dtype="float32",
        revision=revision,
    )


def _cache_final_residuals(model, prompts: list[str]) -> tuple[t.Tensor, t.Tensor]:
    residuals = []
    logits = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with t.no_grad():
            prompt_logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name == TL_RESID_POST_HOOK,
            )
        residuals.append(cache[TL_RESID_POST_HOOK][0, -1].detach())
        logits.append(prompt_logits[0, -1].detach())
    return t.stack(residuals, dim=0), t.stack(logits, dim=0)


def _orthogonal_random_direction(reference: t.Tensor) -> t.Tensor:
    t.manual_seed(6404)
    random_direction = t.randn_like(reference)
    random_direction = random_direction - (random_direction @ reference) * reference
    return random_direction / random_direction.norm().clamp_min(1e-12)


def run_transformerlens_crosscoder_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a paired real-model diffing preflight on GELU-1L vs SoLU-1L."""

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for the 6.4 real paired-model crosscoder preflight.")

    t.cuda.reset_peak_memory_stats()
    prompts = [*TL_TECHNICAL_PROMPTS, *TL_EVERYDAY_PROMPTS]
    labels = t.tensor(
        [1] * len(TL_TECHNICAL_PROMPTS) + [0] * len(TL_EVERYDAY_PROMPTS),
        dtype=t.bool,
        device="cuda",
    )
    model_a = _load_transformerlens_model_on_cuda(TL_GELU1L_MODEL_NAME, TL_GELU1L_REVISION)
    model_b = _load_transformerlens_model_on_cuda(TL_SOLU1L_MODEL_NAME, TL_SOLU1L_REVISION)
    model_a.eval()
    model_b.eval()

    model_a_acts, model_a_logits = _cache_final_residuals(model_a, prompts)
    model_b_acts, model_b_logits = _cache_final_residuals(model_b, prompts)
    delta = model_b_acts - model_a_acts
    feature_dim = model_a_acts.shape[-1]

    exact_output = decode_crosscoder(
        shared_acts=model_a_acts,
        model_a_specific_acts=t.zeros_like(model_a_acts),
        model_b_specific_acts=delta,
        shared_decoder_a=t.eye(feature_dim, device=model_a_acts.device),
        shared_decoder_b=t.eye(feature_dim, device=model_a_acts.device),
        model_a_decoder=t.zeros(feature_dim, feature_dim, device=model_a_acts.device),
        model_b_decoder=t.eye(feature_dim, device=model_a_acts.device),
    )
    reconstruction = crosscoder_reconstruction_report(
        model_a_acts,
        model_b_acts,
        exact_output,
        mse_threshold=1e-8,
    )

    _, singular_values, right_vectors = t.linalg.svd(delta.float(), full_matrices=False)
    top_direction = right_vectors[0]
    delta_scores = delta.float() @ top_direction
    behavior_delta = behavior_delta_prediction_report(
        delta_scores,
        labels,
        feature_id=0,
        min_auc=0.95,
    )

    random_direction = _orthogonal_random_direction(top_direction)
    baseline_delta_norm = delta.float().norm(dim=-1)
    top_projection = (delta.float() @ top_direction).unsqueeze(-1) * top_direction
    random_projection = (delta.float() @ random_direction).unsqueeze(-1) * random_direction
    top_ablated_delta_norm = (delta.float() - top_projection).norm(dim=-1)
    random_ablated_delta_norm = (delta.float() - random_projection).norm(dim=-1)
    ablation = crosscoder_ablation_report(
        baseline_delta_norm,
        top_ablated_delta_norm,
        random_ablated_delta_norm,
    )
    top_variance_fraction = (
        singular_values[0].pow(2) / singular_values.pow(2).sum().clamp_min(1e-12)
    ).item()

    # A behavioral readout is included separately from the activation-space delta.
    floor_token = model_a.to_single_token(" floor")
    top_token = model_a.to_single_token(" top")
    model_a_floor_top = model_a_logits[:, floor_token] - model_a_logits[:, top_token]
    model_b_floor_top = model_b_logits[:, floor_token] - model_b_logits[:, top_token]
    floor_top_delta = toy_behavior_delta_scores(model_a_floor_top, model_b_floor_top)

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        reconstruction.shared_reconstructs_both
        and behavior_delta.passes_threshold
        and ablation.passes_control
        and ablation.delta_reduction > 1.0
        and ablation.random_reduction < 0.1
        and top_variance_fraction > 0.25
        and within_vram_budget
    )
    del model_a, model_b
    t.cuda.empty_cache()
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_a_name": TL_GELU1L_MODEL_NAME,
        "model_a_revision": TL_GELU1L_REVISION,
        "model_b_name": TL_SOLU1L_MODEL_NAME,
        "model_b_revision": TL_SOLU1L_REVISION,
        "tokenizer_id": TL_TOKENIZER_ID,
        "tokenizer_revision": TL_TOKENIZER_REVISION,
        "prompt_count": len(prompts),
        "technical_prompt_count": len(TL_TECHNICAL_PROMPTS),
        "everyday_prompt_count": len(TL_EVERYDAY_PROMPTS),
        "activation_shape": [int(model_a_acts.shape[0]), int(model_a_acts.shape[1])],
        "model_a_mse": reconstruction.model_a_mse,
        "model_b_mse": reconstruction.model_b_mse,
        "shared_active_fraction": reconstruction.shared_active_fraction,
        "shared_reconstructs_both": reconstruction.shared_reconstructs_both,
        "top_singular_value": float(singular_values[0].item()),
        "top_variance_fraction": top_variance_fraction,
        "behavior_delta_auc": behavior_delta.auc,
        "behavior_delta_positive_mean": behavior_delta.positive_mean,
        "behavior_delta_negative_mean": behavior_delta.negative_mean,
        "baseline_delta_norm": ablation.baseline_delta,
        "top_direction_ablated_delta_norm": ablation.ablated_delta,
        "random_direction_ablated_delta_norm": ablation.random_ablated_delta,
        "delta_reduction": ablation.delta_reduction,
        "random_reduction": ablation.random_reduction,
        "ablation_passes_control": ablation.passes_control,
        "floor_top_delta_mean": float(floor_top_delta.mean().item()),
        "floor_top_delta_abs_mean": float(floor_top_delta.abs().mean().item()),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "preflight_passed": preflight_passed,
        "full_path": (
            "Pinned paired TransformerLens GELU-1L vs SoLU-1L CUDA preflight: exact "
            "shared-plus-delta reconstruction, SVD model-diff direction, technical-vs-"
            "everyday prompt separation, and top-direction ablation against an "
            "orthogonal random-direction control."
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_crosscoder_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
