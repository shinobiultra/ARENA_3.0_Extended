"""Toy crosscoder and model-diffing utilities for sparse feature notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from arena_ext.features import roc_auc_binary


FeatureOwner = Literal["shared", "model_a", "model_b"]


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
    """Decode shared and model-specific features into two model activation spaces."""

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
    """Check whether crosscoder reconstructions preserve both model activation spaces."""

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


@dataclass(frozen=True)
class PlantedCrosscoderData:
    """A paired activation dataset with known shared and model-specific features."""

    model_a_activations: t.Tensor
    model_b_activations: t.Tensor
    true_latents: t.Tensor
    behavior_labels: t.Tensor
    behavior_deltas: t.Tensor
    true_decoder_a: t.Tensor
    true_decoder_b: t.Tensor
    feature_names: tuple[str, ...]
    feature_owners: tuple[FeatureOwner, ...]
    prompts: tuple[str, ...]
    train_idx: t.Tensor
    heldout_idx: t.Tensor


@dataclass(frozen=True)
class SparseCrosscoderForward:
    model_a_feature_acts: t.Tensor
    model_b_feature_acts: t.Tensor
    reconstructed_model_a: t.Tensor
    reconstructed_model_b: t.Tensor


@dataclass(frozen=True)
class SparseCrosscoderLoss:
    reconstruction_loss: t.Tensor
    l1_loss: t.Tensor
    total_loss: t.Tensor


@dataclass
class TrainedSparseCrosscoder:
    model: "SparseCrosscoder"
    history: list[dict[str, float]]
    train_idx: t.Tensor
    heldout_idx: t.Tensor


@dataclass(frozen=True)
class SparseCrosscoderReconstructionReport:
    model_a_mse: float
    model_b_mse: float
    heldout_reconstruction_mse: float
    zero_reconstruction_mse: float
    model_a_l0: float
    model_b_l0: float
    beats_zero_baseline: bool


@dataclass(frozen=True)
class PlantedFeatureMatchReport:
    feature_names: tuple[str, ...]
    true_owners: tuple[FeatureOwner, ...]
    matched_latents: tuple[int, ...]
    predicted_owners: tuple[FeatureOwner, ...]
    correlations: tuple[float, ...]
    ownership_accuracy: float
    min_correlation: float
    target_model_b_feature_id: int
    target_model_b_latent: int


@dataclass(frozen=True)
class CrosscoderInterventionReport:
    target_latent: int
    baseline_abs_delta: float
    target_ablated_abs_delta: float
    same_norm_random_abs_delta: float
    orthogonal_abs_delta: float
    target_reduction: float
    same_norm_random_reduction: float
    orthogonal_reduction: float
    passes_controls: bool


def _orthonormal_rows(n_rows: int, n_cols: int, generator: t.Generator) -> t.Tensor:
    if n_rows > n_cols:
        raise ValueError("n_rows must be <= n_cols for this toy orthogonal dictionary.")
    q, _ = t.linalg.qr(t.randn(n_cols, n_rows, generator=generator), mode="reduced")
    return q.T.contiguous()


def make_train_heldout_split(
    n_examples: int,
    *,
    train_fraction: float = 0.75,
    seed: int = 99,
) -> tuple[t.Tensor, t.Tensor]:
    """Return deterministic train and held-out indices."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    generator = t.Generator().manual_seed(seed)
    permutation = t.randperm(n_examples, generator=generator)
    n_train = int(train_fraction * n_examples)
    return permutation[:n_train], permutation[n_train:]


def make_planted_crosscoder_data(
    *,
    n_examples: int = 1024,
    d_model: int = 12,
    seed: int = 6404,
    noise_std: float = 0.005,
    train_fraction: float = 0.75,
) -> PlantedCrosscoderData:
    """Generate paired activations from exact shared-plus-specific ground truth.

    The ground-truth dictionary has four features: two shared features, one
    model-A-specific feature, and one model-B-specific feature. The B-specific
    feature is also the behavior-delta feature used later for held-out AUC and
    intervention tests.
    """

    if d_model < 4:
        raise ValueError("d_model must be at least 4.")
    if n_examples < 64:
        raise ValueError("n_examples must be at least 64.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    generator = t.Generator().manual_seed(seed)
    n_features = 4
    base = _orthonormal_rows(n_features, d_model, generator)
    jitter = 0.03 * _orthonormal_rows(n_features, d_model, generator)

    true_decoder_a = t.zeros(n_features, d_model)
    true_decoder_b = t.zeros(n_features, d_model)
    for feature_id in range(2):
        true_decoder_a[feature_id] = base[feature_id]
        true_decoder_b[feature_id] = F.normalize(
            0.98 * base[feature_id] + jitter[feature_id],
            dim=0,
        )
    true_decoder_a[2] = base[2]
    true_decoder_b[3] = base[3]

    feature_probabilities = t.tensor([0.45, 0.35, 0.12, 0.18])
    present = t.rand(n_examples, n_features, generator=generator) < feature_probabilities
    magnitudes = 0.5 + t.rand(n_examples, n_features, generator=generator)
    true_latents = present.float() * magnitudes

    behavior_labels = t.rand(n_examples, generator=generator) < 0.5
    n_positive = int(behavior_labels.sum().item())
    n_negative = n_examples - n_positive
    true_latents[behavior_labels, 3] += (
        1.1 + 0.4 * t.rand(n_positive, generator=generator)
    )
    true_latents[~behavior_labels, 2] += (
        0.8 + 0.3 * t.rand(n_negative, generator=generator)
    )

    clean_a = true_latents @ true_decoder_a
    clean_b = true_latents @ true_decoder_b
    model_a_activations = clean_a + noise_std * t.randn(n_examples, d_model, generator=generator)
    model_b_activations = clean_b + noise_std * t.randn(n_examples, d_model, generator=generator)
    behavior_deltas = (
        1.6 * true_latents[:, 3]
        - 0.75 * true_latents[:, 2]
        + 0.12 * true_latents[:, 1]
        + 0.02 * t.randn(n_examples, generator=generator)
    )

    technical_terms = ("debugger", "compiler", "tensor", "kernel", "API", "cache")
    everyday_terms = ("garden", "recipe", "train", "letter", "market", "piano")
    prompts = []
    for index, is_positive in enumerate(behavior_labels.tolist()):
        terms = technical_terms if is_positive else everyday_terms
        prompts.append(f"The {terms[index % len(terms)]} example number {index:04d}")

    train_idx, heldout_idx = make_train_heldout_split(
        n_examples,
        train_fraction=train_fraction,
        seed=99,
    )
    return PlantedCrosscoderData(
        model_a_activations=model_a_activations,
        model_b_activations=model_b_activations,
        true_latents=true_latents,
        behavior_labels=behavior_labels,
        behavior_deltas=behavior_deltas,
        true_decoder_a=true_decoder_a,
        true_decoder_b=true_decoder_b,
        feature_names=(
            "shared topic feature",
            "shared syntax feature",
            "model A everyday feature",
            "model B technical feature",
        ),
        feature_owners=("shared", "shared", "model_a", "model_b"),
        prompts=tuple(prompts),
        train_idx=train_idx,
        heldout_idx=heldout_idx,
    )


class SparseCrosscoder(nn.Module):
    """A small tied-encoder sparse crosscoder for paired model activations."""

    def __init__(self, d_model: int, n_latents: int = 6, *, seed: int = 2):
        super().__init__()
        if d_model <= 0 or n_latents <= 0:
            raise ValueError("d_model and n_latents must be positive.")

        rng_state = t.random.get_rng_state()
        t.manual_seed(seed)
        self.encoder = nn.Linear(d_model, n_latents)
        self.decoder_a = nn.Parameter(0.08 * t.randn(n_latents, d_model))
        self.decoder_b = nn.Parameter(0.08 * t.randn(n_latents, d_model))
        self.bias_a = nn.Parameter(t.zeros(d_model))
        self.bias_b = nn.Parameter(t.zeros(d_model))
        t.random.set_rng_state(rng_state)

    def encode_model_a(self, model_a_activations: t.Tensor) -> t.Tensor:
        return F.relu(self.encoder(model_a_activations.float()))

    def encode_model_b(self, model_b_activations: t.Tensor) -> t.Tensor:
        return F.relu(self.encoder(model_b_activations.float()))

    def forward(self, model_a_activations: t.Tensor, model_b_activations: t.Tensor):
        if model_a_activations.shape != model_b_activations.shape:
            raise ValueError("paired activations must have matching shapes.")
        model_a_feature_acts = self.encode_model_a(model_a_activations)
        model_b_feature_acts = self.encode_model_b(model_b_activations)
        reconstructed_model_a = model_a_feature_acts @ self.decoder_a + self.bias_a
        reconstructed_model_b = model_b_feature_acts @ self.decoder_b + self.bias_b
        return SparseCrosscoderForward(
            model_a_feature_acts=model_a_feature_acts,
            model_b_feature_acts=model_b_feature_acts,
            reconstructed_model_a=reconstructed_model_a,
            reconstructed_model_b=reconstructed_model_b,
        )


def sparse_crosscoder_loss(
    output: SparseCrosscoderForward,
    model_a_activations: t.Tensor,
    model_b_activations: t.Tensor,
    *,
    l1_coefficient: float = 0.008,
) -> SparseCrosscoderLoss:
    """Compute reconstruction plus sparse-activation loss."""

    reconstruction_loss = F.mse_loss(
        output.reconstructed_model_a,
        model_a_activations.float(),
    ) + F.mse_loss(
        output.reconstructed_model_b,
        model_b_activations.float(),
    )
    l1_loss = output.model_a_feature_acts.mean() + output.model_b_feature_acts.mean()
    total_loss = reconstruction_loss + l1_coefficient * l1_loss
    return SparseCrosscoderLoss(
        reconstruction_loss=reconstruction_loss,
        l1_loss=l1_loss,
        total_loss=total_loss,
    )


def evaluate_sparse_crosscoder_reconstruction(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    indices: t.Tensor | None = None,
) -> SparseCrosscoderReconstructionReport:
    """Evaluate held-out reconstruction and sparsity against a zero baseline."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    model_a_mse = F.mse_loss(
        output.reconstructed_model_a,
        data.model_a_activations[indices],
    ).item()
    model_b_mse = F.mse_loss(
        output.reconstructed_model_b,
        data.model_b_activations[indices],
    ).item()
    heldout_mse = 0.5 * (model_a_mse + model_b_mse)
    zero_mse = 0.5 * (
        data.model_a_activations[indices].pow(2).mean().item()
        + data.model_b_activations[indices].pow(2).mean().item()
    )
    model_a_l0 = output.model_a_feature_acts.gt(1e-4).float().sum(dim=-1).mean().item()
    model_b_l0 = output.model_b_feature_acts.gt(1e-4).float().sum(dim=-1).mean().item()
    return SparseCrosscoderReconstructionReport(
        model_a_mse=model_a_mse,
        model_b_mse=model_b_mse,
        heldout_reconstruction_mse=heldout_mse,
        zero_reconstruction_mse=zero_mse,
        model_a_l0=model_a_l0,
        model_b_l0=model_b_l0,
        beats_zero_baseline=heldout_mse < 0.01 * zero_mse,
    )


def train_sparse_crosscoder(
    data: PlantedCrosscoderData,
    *,
    n_latents: int = 6,
    steps: int = 500,
    learning_rate: float = 4e-3,
    l1_coefficient: float = 0.008,
    log_every: int = 50,
    seed: int = 2,
) -> TrainedSparseCrosscoder:
    """Train the toy sparse crosscoder on CPU-sized paired activations."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    model = SparseCrosscoder(
        d_model=data.model_a_activations.shape[-1],
        n_latents=n_latents,
        seed=seed,
    )
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_idx = data.train_idx
    heldout_idx = data.heldout_idx
    history: list[dict[str, float]] = []

    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            with t.no_grad():
                train_output = model(
                    data.model_a_activations[train_idx],
                    data.model_b_activations[train_idx],
                )
                heldout_output = model(
                    data.model_a_activations[heldout_idx],
                    data.model_b_activations[heldout_idx],
                )
                train_loss = sparse_crosscoder_loss(
                    train_output,
                    data.model_a_activations[train_idx],
                    data.model_b_activations[train_idx],
                    l1_coefficient=l1_coefficient,
                )
                heldout_loss = sparse_crosscoder_loss(
                    heldout_output,
                    data.model_a_activations[heldout_idx],
                    data.model_b_activations[heldout_idx],
                    l1_coefficient=l1_coefficient,
                )
                heldout_mse = 0.5 * (
                    F.mse_loss(
                        heldout_output.reconstructed_model_a,
                        data.model_a_activations[heldout_idx],
                    ).item()
                    + F.mse_loss(
                        heldout_output.reconstructed_model_b,
                        data.model_b_activations[heldout_idx],
                    ).item()
                )
                history.append(
                    {
                        "step": float(step),
                        "train_total_loss": float(train_loss.total_loss.item()),
                        "heldout_total_loss": float(heldout_loss.total_loss.item()),
                        "heldout_reconstruction_mse": heldout_mse,
                        "model_a_l0": float(
                            heldout_output.model_a_feature_acts.gt(1e-4)
                            .float()
                            .sum(dim=-1)
                            .mean()
                            .item()
                        ),
                        "model_b_l0": float(
                            heldout_output.model_b_feature_acts.gt(1e-4)
                            .float()
                            .sum(dim=-1)
                            .mean()
                            .item()
                        ),
                    }
                )
        if step == steps:
            break

        optimizer.zero_grad()
        output = model(
            data.model_a_activations[train_idx],
            data.model_b_activations[train_idx],
        )
        loss = sparse_crosscoder_loss(
            output,
            data.model_a_activations[train_idx],
            data.model_b_activations[train_idx],
            l1_coefficient=l1_coefficient,
        ).total_loss
        loss.backward()
        optimizer.step()

    return TrainedSparseCrosscoder(
        model=model,
        history=history,
        train_idx=train_idx,
        heldout_idx=heldout_idx,
    )


def latent_ownership_table(
    output: SparseCrosscoderForward,
    *,
    shared_threshold: float = 0.08,
) -> list[dict[str, float | str | int]]:
    """Classify learned latents by model-A vs model-B activation means."""

    model_a_means = output.model_a_feature_acts.mean(dim=0)
    model_b_means = output.model_b_feature_acts.mean(dim=0)
    rows: list[dict[str, float | str | int]] = []
    for latent_id, (model_a_mean, model_b_mean) in enumerate(zip(model_a_means, model_b_means)):
        specificity = float((model_b_mean - model_a_mean).item())
        if abs(specificity) <= shared_threshold:
            owner: FeatureOwner = "shared"
        elif specificity > 0:
            owner = "model_b"
        else:
            owner = "model_a"
        rows.append(
            {
                "latent": latent_id,
                "model_a_mean": float(model_a_mean.item()),
                "model_b_mean": float(model_b_mean.item()),
                "specificity": specificity,
                "predicted_owner": owner,
            }
        )
    return rows


def _columnwise_abs_correlation(reference: t.Tensor, candidates: t.Tensor) -> t.Tensor:
    reference = reference.float()
    candidates = candidates.float()
    reference = (reference - reference.mean()) / reference.std().clamp_min(1e-8)
    candidates = (candidates - candidates.mean(dim=0)) / candidates.std(dim=0).clamp_min(1e-8)
    return (reference[:, None] * candidates).mean(dim=0).abs()


def match_learned_to_planted_features(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    indices: t.Tensor | None = None,
    *,
    shared_threshold: float = 0.08,
    target_model_b_feature_id: int = 3,
) -> PlantedFeatureMatchReport:
    """Match learned latents to planted features by held-out activation correlation."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    owner_by_latent = {
        int(row["latent"]): str(row["predicted_owner"])
        for row in latent_ownership_table(output, shared_threshold=shared_threshold)
    }
    matched_latents: list[int] = []
    predicted_owners: list[FeatureOwner] = []
    correlations: list[float] = []
    for feature_id, true_owner in enumerate(data.feature_owners):
        true_activation = data.true_latents[indices, feature_id]
        if true_owner == "shared":
            learned_activations = 0.5 * (
                output.model_a_feature_acts + output.model_b_feature_acts
            )
        elif true_owner == "model_a":
            learned_activations = output.model_a_feature_acts
        else:
            learned_activations = output.model_b_feature_acts
        correlation = _columnwise_abs_correlation(true_activation, learned_activations)
        matched_latent = int(correlation.argmax().item())
        matched_latents.append(matched_latent)
        predicted_owners.append(owner_by_latent[matched_latent])  # type: ignore[arg-type]
        correlations.append(float(correlation[matched_latent].item()))

    correct = [
        predicted == true
        for predicted, true in zip(predicted_owners, data.feature_owners, strict=True)
    ]
    if target_model_b_feature_id < 0 or target_model_b_feature_id >= len(data.feature_owners):
        raise IndexError("target_model_b_feature_id is out of range.")
    return PlantedFeatureMatchReport(
        feature_names=data.feature_names,
        true_owners=data.feature_owners,
        matched_latents=tuple(matched_latents),
        predicted_owners=tuple(predicted_owners),
        correlations=tuple(correlations),
        ownership_accuracy=sum(correct) / len(correct),
        min_correlation=min(correlations),
        target_model_b_feature_id=target_model_b_feature_id,
        target_model_b_latent=matched_latents[target_model_b_feature_id],
    )


def signed_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Return max(AUC, 1-AUC) so opposite signed features still count."""

    auc = roc_auc_binary(scores, labels)
    return max(auc, 1.0 - auc)


def model_behavior_delta_from_reconstruction(
    reconstructed_model_a: t.Tensor,
    reconstructed_model_b: t.Tensor,
    data: PlantedCrosscoderData,
    *,
    target_feature_id: int = 3,
) -> t.Tensor:
    """Read out the planted behavior difference from reconstructed activations."""

    return (
        reconstructed_model_b.float() @ data.true_decoder_b[target_feature_id].float()
        - reconstructed_model_a.float() @ data.true_decoder_a[target_feature_id].float()
    )


def ablate_latents(
    output: SparseCrosscoderForward,
    model: SparseCrosscoder,
    latent_ids: list[int] | tuple[int, ...],
) -> tuple[t.Tensor, t.Tensor]:
    """Reconstruct both models after zeroing selected learned latents."""

    model_a_acts = output.model_a_feature_acts.clone()
    model_b_acts = output.model_b_feature_acts.clone()
    if latent_ids:
        model_a_acts[:, list(latent_ids)] = 0
        model_b_acts[:, list(latent_ids)] = 0
    reconstructed_a = model_a_acts @ model.decoder_a + model.bias_a
    reconstructed_b = model_b_acts @ model.decoder_b + model.bias_b
    return reconstructed_a, reconstructed_b


def shared_only_reconstruction(
    output: SparseCrosscoderForward,
    model: SparseCrosscoder,
    owner_rows: list[dict[str, float | str | int]],
) -> tuple[t.Tensor, t.Tensor]:
    """Keep only latents classified as shared and reconstruct both models."""

    non_shared = [
        int(row["latent"])
        for row in owner_rows
        if row["predicted_owner"] != "shared"
    ]
    return ablate_latents(output, model, non_shared)


def svd_delta_direction_scores(
    data: PlantedCrosscoderData,
    *,
    train_idx: t.Tensor | None = None,
    heldout_idx: t.Tensor | None = None,
) -> t.Tensor:
    """Fit a top SVD direction on train deltas, then score held-out deltas."""

    if train_idx is None:
        train_idx = data.train_idx
    if heldout_idx is None:
        heldout_idx = data.heldout_idx
    train_delta = (
        data.model_b_activations[train_idx] - data.model_a_activations[train_idx]
    ).float()
    heldout_delta = (
        data.model_b_activations[heldout_idx] - data.model_a_activations[heldout_idx]
    ).float()
    _, _, right_vectors = t.linalg.svd(train_delta, full_matrices=False)
    return heldout_delta @ right_vectors[0]


def svd_pair_reconstruction_mse(
    data: PlantedCrosscoderData,
    *,
    rank: int = 4,
    train_idx: t.Tensor | None = None,
    heldout_idx: t.Tensor | None = None,
) -> float:
    """Evaluate a train-fitted low-rank SVD baseline on concatenated held-out pairs."""

    if train_idx is None:
        train_idx = data.train_idx
    if heldout_idx is None:
        heldout_idx = data.heldout_idx
    train_pair = t.cat(
        [data.model_a_activations[train_idx], data.model_b_activations[train_idx]],
        dim=-1,
    ).float()
    heldout_pair = t.cat(
        [data.model_a_activations[heldout_idx], data.model_b_activations[heldout_idx]],
        dim=-1,
    ).float()
    center = train_pair.mean(dim=0, keepdim=True)
    _, _, right_vectors = t.linalg.svd(train_pair - center, full_matrices=False)
    basis = right_vectors[:rank]
    reconstructed = (heldout_pair - center) @ basis.T @ basis + center
    return F.mse_loss(reconstructed, heldout_pair).item()


def behavior_baseline_table(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    shuffle_seed: int = 12,
) -> list[dict[str, float | str]]:
    """Compare learned target-latent behavior against simple baselines."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    labels = data.behavior_labels[indices]
    reconstruction = evaluate_sparse_crosscoder_reconstruction(model, data, indices)
    target_scores = output.model_b_feature_acts[:, feature_match.target_model_b_latent]
    target_auc = signed_auc_binary(target_scores, labels)

    owner_rows = latent_ownership_table(output)
    shared_a, shared_b = shared_only_reconstruction(output, model, owner_rows)
    shared_delta = model_behavior_delta_from_reconstruction(shared_a, shared_b, data)
    shared_auc = signed_auc_binary(shared_delta, labels)
    shared_mse = 0.5 * (
        F.mse_loss(shared_a, data.model_a_activations[indices]).item()
        + F.mse_loss(shared_b, data.model_b_activations[indices]).item()
    )

    svd_scores = svd_delta_direction_scores(data, heldout_idx=indices)
    svd_auc = signed_auc_binary(svd_scores, labels)
    svd_mse = svd_pair_reconstruction_mse(data, heldout_idx=indices)

    generator = t.Generator().manual_seed(shuffle_seed)
    shuffled_labels = labels[t.randperm(labels.numel(), generator=generator)]
    shuffled_auc = signed_auc_binary(target_scores, shuffled_labels)

    return [
        {
            "method": "learned sparse crosscoder target latent",
            "heldout_reconstruction_mse": reconstruction.heldout_reconstruction_mse,
            "behavior_auc": target_auc,
            "role": "main result",
        },
        {
            "method": "zero reconstruction",
            "heldout_reconstruction_mse": reconstruction.zero_reconstruction_mse,
            "behavior_auc": 0.5,
            "role": "reconstruction floor",
        },
        {
            "method": "shared-only learned latents",
            "heldout_reconstruction_mse": shared_mse,
            "behavior_auc": shared_auc,
            "role": "removes model-specific latents",
        },
        {
            "method": "SVD top delta direction",
            "heldout_reconstruction_mse": svd_mse,
            "behavior_auc": svd_auc,
            "role": "non-sparse direction baseline",
        },
        {
            "method": "label-shuffled target latent",
            "heldout_reconstruction_mse": reconstruction.heldout_reconstruction_mse,
            "behavior_auc": shuffled_auc,
            "role": "negative control",
        },
    ]


def crosscoder_intervention_report(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    random_seed: int = 111,
    control_margin: float = 0.25,
) -> CrosscoderInterventionReport:
    """Compare targeted model-specific ablation against random/orthogonal controls."""

    if indices is None:
        indices = data.heldout_idx
    target_latent = feature_match.target_model_b_latent
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
        baseline_delta = model_behavior_delta_from_reconstruction(
            output.reconstructed_model_a,
            output.reconstructed_model_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )
        target_a, target_b = ablate_latents(output, model, [target_latent])
        target_delta = model_behavior_delta_from_reconstruction(
            target_a,
            target_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

        target_pair = t.cat([model.decoder_a[target_latent], model.decoder_b[target_latent]])
        generator = t.Generator().manual_seed(random_seed)
        random_pair = t.randn(target_pair.shape, generator=generator)
        random_pair = random_pair / random_pair.norm().clamp_min(1e-12)
        random_pair = random_pair * target_pair.norm().clamp_min(1e-12)
        random_a = random_pair[: model.decoder_a.shape[-1]]
        random_b = random_pair[model.decoder_a.shape[-1] :]
        random_reconstructed_a = (
            output.reconstructed_model_a
            - output.model_a_feature_acts[:, target_latent : target_latent + 1] * random_a
        )
        random_reconstructed_b = (
            output.reconstructed_model_b
            - output.model_b_feature_acts[:, target_latent : target_latent + 1] * random_b
        )
        random_delta = model_behavior_delta_from_reconstruction(
            random_reconstructed_a,
            random_reconstructed_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

        orthogonal_pair = random_pair - (
            (random_pair @ target_pair) / target_pair.pow(2).sum().clamp_min(1e-12)
        ) * target_pair
        orthogonal_pair = orthogonal_pair / orthogonal_pair.norm().clamp_min(1e-12)
        orthogonal_pair = orthogonal_pair * target_pair.norm().clamp_min(1e-12)
        orthogonal_a = orthogonal_pair[: model.decoder_a.shape[-1]]
        orthogonal_b = orthogonal_pair[model.decoder_a.shape[-1] :]
        orthogonal_reconstructed_a = (
            output.reconstructed_model_a
            - output.model_a_feature_acts[:, target_latent : target_latent + 1] * orthogonal_a
        )
        orthogonal_reconstructed_b = (
            output.reconstructed_model_b
            - output.model_b_feature_acts[:, target_latent : target_latent + 1] * orthogonal_b
        )
        orthogonal_delta = model_behavior_delta_from_reconstruction(
            orthogonal_reconstructed_a,
            orthogonal_reconstructed_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

    positive = data.behavior_labels[indices]
    baseline_abs = baseline_delta[positive].abs().mean().item()
    target_abs = target_delta[positive].abs().mean().item()
    random_abs = random_delta[positive].abs().mean().item()
    orthogonal_abs = orthogonal_delta[positive].abs().mean().item()
    target_reduction = baseline_abs - target_abs
    random_reduction = baseline_abs - random_abs
    orthogonal_reduction = baseline_abs - orthogonal_abs
    return CrosscoderInterventionReport(
        target_latent=target_latent,
        baseline_abs_delta=baseline_abs,
        target_ablated_abs_delta=target_abs,
        same_norm_random_abs_delta=random_abs,
        orthogonal_abs_delta=orthogonal_abs,
        target_reduction=target_reduction,
        same_norm_random_reduction=random_reduction,
        orthogonal_reduction=orthogonal_reduction,
        passes_controls=(
            target_reduction > random_reduction + control_margin
            and target_reduction > orthogonal_reduction + control_margin
        ),
    )


def top_activating_examples(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    k: int = 6,
) -> list[dict[str, float | str | int | bool]]:
    """Return the top held-out examples for the matched model-B-specific latent."""

    if indices is None:
        indices = data.heldout_idx
    target_latent = feature_match.target_model_b_latent
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    scores = output.model_b_feature_acts[:, target_latent]
    top_local = scores.argsort(descending=True)[:k]
    rows: list[dict[str, float | str | int | bool]] = []
    for rank, local_index in enumerate(top_local.tolist(), start=1):
        global_index = int(indices[local_index].item())
        rows.append(
            {
                "rank": rank,
                "example_index": global_index,
                "prompt": data.prompts[global_index],
                "is_behavior_delta_positive": bool(data.behavior_labels[global_index].item()),
                "learned_activation": float(scores[local_index].item()),
                "true_model_b_feature": float(
                    data.true_latents[global_index, feature_match.target_model_b_feature_id].item()
                ),
                "behavior_delta": float(data.behavior_deltas[global_index].item()),
            }
        )
    return rows


def run_crosscoder_signature_result(
    *,
    steps: int = 500,
    n_latents: int = 6,
) -> dict[str, object]:
    """Train and evaluate the CPU learned-crosscoder signature result."""

    data = make_planted_crosscoder_data()
    trained = train_sparse_crosscoder(data, n_latents=n_latents, steps=steps)
    reconstruction = evaluate_sparse_crosscoder_reconstruction(
        trained.model,
        data,
        data.heldout_idx,
    )
    feature_match = match_learned_to_planted_features(
        trained.model,
        data,
        data.heldout_idx,
    )
    baselines = behavior_baseline_table(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    intervention = crosscoder_intervention_report(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    top_examples = top_activating_examples(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    learned_auc = next(
        float(row["behavior_auc"])
        for row in baselines
        if row["method"] == "learned sparse crosscoder target latent"
    )
    label_shuffle_auc = next(
        float(row["behavior_auc"])
        for row in baselines
        if row["method"] == "label-shuffled target latent"
    )
    contract_passed = (
        reconstruction.beats_zero_baseline
        and reconstruction.heldout_reconstruction_mse < 3e-4
        and feature_match.ownership_accuracy >= 0.99
        and feature_match.min_correlation > 0.9
        and learned_auc > 0.93
        and label_shuffle_auc < learned_auc - 0.2
        and intervention.passes_controls
    )
    return {
        "contract_passed": contract_passed,
        "accepted": contract_passed,
        "tests_passed": contract_passed,
        "n_heldout": int(data.heldout_idx.numel()),
        "heldout_reconstruction_mse": reconstruction.heldout_reconstruction_mse,
        "zero_reconstruction_mse": reconstruction.zero_reconstruction_mse,
        "ownership_accuracy": feature_match.ownership_accuracy,
        "min_correlation": feature_match.min_correlation,
        "learned_behavior_auc": learned_auc,
        "label_shuffle_behavior_auc": label_shuffle_auc,
        "target_reduction": intervention.target_reduction,
        "same_norm_random_reduction": intervention.same_norm_random_reduction,
        "orthogonal_reduction": intervention.orthogonal_reduction,
        "dataset": {
            "n_examples": int(data.model_a_activations.shape[0]),
            "d_model": int(data.model_a_activations.shape[1]),
            "n_train": int(data.train_idx.numel()),
            "n_heldout": int(data.heldout_idx.numel()),
            "feature_names": list(data.feature_names),
            "feature_owners": list(data.feature_owners),
        },
        "training_history": trained.history,
        "reconstruction": reconstruction.__dict__,
        "feature_match": feature_match.__dict__,
        "baselines": baselines,
        "intervention": intervention.__dict__,
        "top_examples": top_examples,
    }
