# %%
"""Reference solutions for [6.1] SAE Variants.

The CPU lesson trains all four variants on exact planted sparse ground truth.
The optional CUDA entry point at the bottom trains a small TopK SAE on pinned
Pythia-70M hidden states; it is supporting real-model evidence, not the lesson's
signature result.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NamedTuple

import torch as t
import torch.nn.functional as F

chapter = "chapter6_sparse_feature_methods"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"
Variant = Literal["relu_l1", "topk", "gated", "jumprelu"]
VARIANTS: tuple[Variant, ...] = ("relu_l1", "topk", "gated", "jumprelu")

PYTHIA_SAE_MODEL_ID = "EleutherAI/pythia-70m-deduped"
PYTHIA_SAE_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
PYTHIA_SAE_HIDDEN_LAYER = -1
PYTHIA_SAE_WIDTH = 256
PYTHIA_SAE_TOPK = 16
PYTHIA_SAE_TRAINING_STEPS = 800
PYTHIA_SAE_STEERING_ALPHA = 5.0

PYTHIA_SAE_TECHNICAL_TOPICS = [
    "python debugging",
    "matrix algebra",
    "neural network",
    "data pipeline",
    "compiler design",
    "probability theorem",
]
PYTHIA_SAE_EVERYDAY_TOPICS = [
    "sourdough recipe",
    "garden planting",
    "travel itinerary",
    "music practice",
    "painting class",
    "meal planning",
]
PYTHIA_SAE_TRAIN_TEMPLATES = [
    "Course note. Topic: {topic}. Key idea:",
    "Brief tutorial about {topic}. Main concept:",
    "Research memo on {topic}. Important detail:",
    "Student question about {topic}. Answer outline:",
    "Workshop summary for {topic}. Takeaway:",
    "Reference card: {topic}. Definition:",
    "Practice exercise involving {topic}. Hint:",
    "Comparison paragraph about {topic}. Focus:",
]
PYTHIA_SAE_HELDOUT_TEMPLATES = [
    "Evaluation note. Subject: {topic}. Main point:",
    "Short classroom example about {topic}. Response:",
    "Checklist item for {topic}. Next step:",
    "One sentence about {topic}. Conclusion:",
]
PYTHIA_SAE_SAFE_LOGIT_TOKENS = [
    " code",
    " Python",
    " data",
    " theorem",
    " network",
    " matrix",
    " compiler",
    " algorithm",
    " recipe",
    " garden",
    " travel",
    " music",
    " painting",
    " meal",
    " class",
]


@dataclass(frozen=True)
class PlantedSparseBatch:
    feature_acts: t.Tensor
    activations: t.Tensor
    labels: t.Tensor
    dictionary: t.Tensor


ToySuperpositionBatch = PlantedSparseBatch


@dataclass(frozen=True)
class SAEForward:
    reconstruction: t.Tensor
    feature_acts: t.Tensor
    pre_acts: t.Tensor
    gate_pre_acts: t.Tensor | None = None
    auxiliary_reconstruction: t.Tensor | None = None
    jump_threshold: t.Tensor | None = None


@dataclass(frozen=True)
class SAELoss:
    total: t.Tensor
    reconstruction: t.Tensor
    sparsity: t.Tensor
    auxiliary: t.Tensor


@dataclass(frozen=True)
class SAEVariantMetrics:
    name: str
    l0: float
    feature_density_mean: float
    dead_feature_fraction: float
    reconstruction_mse: float


@dataclass(frozen=True)
class DictionaryRecoveryReport:
    mean_best_cosine: float
    recovered_fraction: float
    duplicate_fraction: float
    best_learned_for_true: t.Tensor
    cosine_matrix: t.Tensor


@dataclass(frozen=True)
class FeatureAUCReport:
    feature_id: int
    auc: float
    polarity: int


@dataclass(frozen=True)
class SteeringComparisonReport:
    baseline_mean: float
    steered_mean: float
    random_mean: float
    steered_delta: float
    random_delta: float
    passes_control: bool


@dataclass(frozen=True)
class CausalInterventionReport:
    target_feature_id: int
    decoder_cosine: float
    steering_delta: float
    random_steering_delta: float
    ablation_drop: float
    random_ablation_drop: float


@dataclass(frozen=True)
class VariantResult:
    name: Variant
    model: "SparseAutoencoder"
    metrics: SAEVariantMetrics
    recovery: DictionaryRecoveryReport
    heldout_auc: float
    shuffled_auc: float
    intervention: CausalInterventionReport
    train_losses: tuple[float, ...]


@dataclass(frozen=True)
class SAEComparison:
    train: PlantedSparseBatch
    heldout: PlantedSparseBatch
    results: dict[str, VariantResult]
    zero_baseline_mse: float
    true_l0: float
    random_decoder_recovery: float


class EncoderFunctions(NamedTuple):
    relu_l1: Callable[[t.Tensor], t.Tensor]
    topk: Callable[[t.Tensor, int], t.Tensor]
    gated: Callable[[t.Tensor, t.Tensor], t.Tensor]
    jumprelu: Callable[[t.Tensor, t.Tensor, float], t.Tensor]


@dataclass(frozen=True)
class SAETrainConfig:
    variant: Variant
    d_sae: int = 8
    k: int = 2
    sparsity_coefficient: float = 0.02
    gated_aux_coefficient: float = 1.0
    jump_bandwidth: float = 0.1
    learning_rate: float = 0.01
    steps: int = 900
    batch_size: int = 256
    seed: int = 3


# %%
def make_planted_dictionary(*, n_features: int, d_model: int, seed: int = 0) -> t.Tensor:
    """Return a deterministic overcomplete unit-norm dictionary."""

    if n_features <= 0 or d_model <= 0:
        raise ValueError("n_features and d_model must be positive")
    generator = t.Generator(device="cpu").manual_seed(seed)
    return F.normalize(t.randn(n_features, d_model, generator=generator), dim=-1)


def sample_planted_batch(
    dictionary: t.Tensor,
    *,
    n_examples: int,
    feature_probability: float = 0.15,
    noise_std: float = 0.01,
    target_feature: int = 0,
    seed: int = 1,
) -> PlantedSparseBatch:
    """Sample sparse nonnegative latents and mix them through a known dictionary."""

    if dictionary.ndim != 2:
        raise ValueError("dictionary must have shape (features, d_model)")
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    if not 0.0 < feature_probability < 1.0:
        raise ValueError("feature_probability must be strictly between zero and one")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    if not 0 <= target_feature < dictionary.shape[0]:
        raise ValueError("target_feature is outside the dictionary")

    generator = t.Generator(device="cpu").manual_seed(seed)
    is_active = t.rand(n_examples, dictionary.shape[0], generator=generator) < feature_probability
    magnitudes = 0.5 + t.rand(n_examples, dictionary.shape[0], generator=generator)
    feature_acts = is_active.float() * magnitudes
    activations = feature_acts @ dictionary
    if noise_std:
        activations = activations + noise_std * t.randn(
            activations.shape,
            generator=generator,
        )
    return PlantedSparseBatch(
        feature_acts=feature_acts,
        activations=activations,
        labels=is_active[:, target_feature],
        dictionary=dictionary,
    )


def make_planted_splits(
    *,
    train_examples: int = 4096,
    heldout_examples: int = 2048,
    n_features: int = 8,
    d_model: int = 6,
    feature_probability: float = 0.15,
    noise_std: float = 0.01,
    seed: int = 1,
) -> tuple[PlantedSparseBatch, PlantedSparseBatch]:
    """Create independent train and held-out samples from one exact dictionary."""

    dictionary = make_planted_dictionary(n_features=n_features, d_model=d_model, seed=seed)
    train = sample_planted_batch(
        dictionary,
        n_examples=train_examples,
        feature_probability=feature_probability,
        noise_std=noise_std,
        seed=seed + 1,
    )
    heldout = sample_planted_batch(
        dictionary,
        n_examples=heldout_examples,
        feature_probability=feature_probability,
        noise_std=noise_std,
        seed=seed + 2,
    )
    return train, heldout


def make_toy_superposition_batch(
    *,
    batch: int = 256,
    n_features: int = 8,
    d_model: int = 4,
    feature_probability: float = 0.2,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> PlantedSparseBatch:
    """Compatibility wrapper for the earlier section-local API."""

    dictionary = make_planted_dictionary(n_features=n_features, d_model=d_model, seed=seed)
    return sample_planted_batch(
        dictionary,
        n_examples=batch,
        feature_probability=feature_probability,
        noise_std=noise_scale,
        seed=seed + 1,
    )


# %%
def relu_l1_encode(pre_acts: t.Tensor, *, l1_coefficient: float | None = None) -> t.Tensor:
    """ReLU encoder; L1 pressure belongs in the loss, not in this forward rule."""

    if l1_coefficient is not None and l1_coefficient < 0:
        raise ValueError("l1_coefficient must be nonnegative")
    return F.relu(pre_acts)


def topk_encode(pre_acts: t.Tensor, *, k: int) -> t.Tensor:
    """Keep the k largest nonnegative activations in each example."""

    if pre_acts.ndim < 1:
        raise ValueError("pre_acts must have at least one dimension")
    if not 0 <= k <= pre_acts.shape[-1]:
        raise ValueError("k must be between zero and the number of features")
    positive = F.relu(pre_acts)
    if k == 0:
        return t.zeros_like(positive)
    values, indices = positive.topk(k, dim=-1)
    return t.zeros_like(positive).scatter(-1, indices, values)


def gated_encode(magnitude_pre_acts: t.Tensor, gate_pre_acts: t.Tensor) -> t.Tensor:
    """Separate binary detection from the nonnegative activation magnitude."""

    if magnitude_pre_acts.shape != gate_pre_acts.shape:
        raise ValueError("magnitude and gate pre-activations must have matching shapes")
    return F.relu(magnitude_pre_acts) * (gate_pre_acts > 0).to(magnitude_pre_acts.dtype)


def rectangle_kernel(x: t.Tensor) -> t.Tensor:
    """Unit-width rectangle used by the JumpReLU threshold pseudo-gradient."""

    return (x.abs() < 0.5).to(x.dtype)


class HeavisideSTE(t.autograd.Function):
    """Hard firing indicator with a local pseudo-gradient for its threshold."""

    @staticmethod
    def forward(ctx, values: t.Tensor, threshold: t.Tensor, bandwidth: float) -> t.Tensor:
        ctx.save_for_backward(values, threshold)
        ctx.bandwidth = bandwidth
        return (values > threshold).to(values.dtype)

    @staticmethod
    def backward(ctx, grad_output: t.Tensor):
        values, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        local = rectangle_kernel((values - threshold) / bandwidth)
        grad_threshold = (-(local / bandwidth) * grad_output).sum(dim=0)
        return t.zeros_like(values), grad_threshold, None


class JumpReLU(t.autograd.Function):
    """Hard JumpReLU forward pass and the published threshold pseudo-gradient."""

    @staticmethod
    def forward(ctx, values: t.Tensor, threshold: t.Tensor, bandwidth: float) -> t.Tensor:
        ctx.save_for_backward(values, threshold)
        ctx.bandwidth = bandwidth
        return values * (values > threshold).to(values.dtype)

    @staticmethod
    def backward(ctx, grad_output: t.Tensor):
        values, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        is_active = (values > threshold).to(values.dtype)
        local = rectangle_kernel((values - threshold) / bandwidth)
        grad_values = is_active * grad_output
        grad_threshold = (-(threshold / bandwidth) * local * grad_output).sum(dim=0)
        return grad_values, grad_threshold, None


def jumprelu_encode(pre_acts: t.Tensor, threshold: t.Tensor | float, bandwidth: float = 0.1) -> t.Tensor:
    """Apply JumpReLU to nonnegative pre-activations."""

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    threshold_tensor = t.as_tensor(threshold, dtype=pre_acts.dtype, device=pre_acts.device)
    return JumpReLU.apply(F.relu(pre_acts), threshold_tensor, bandwidth)


REFERENCE_ENCODERS = EncoderFunctions(
    relu_l1=lambda pre: relu_l1_encode(pre),
    topk=lambda pre, k: topk_encode(pre, k=k),
    gated=gated_encode,
    jumprelu=jumprelu_encode,
)


def decode_features(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor | None = None,
) -> t.Tensor:
    """Decode sparse feature activations back into activation space."""

    if feature_acts.shape[-1] != decoder_weight.shape[0]:
        raise ValueError("feature dimension must match decoder rows")
    reconstruction = feature_acts @ decoder_weight
    return reconstruction if decoder_bias is None else reconstruction + decoder_bias


def sparse_autoencoder_loss(
    output: SAEForward,
    activations: t.Tensor,
    *,
    variant: Variant,
    sparsity_coefficient: float,
    gated_aux_coefficient: float = 1.0,
    jump_bandwidth: float = 0.1,
) -> SAELoss:
    """Compute the variant-specific objective while keeping terms visible."""

    reconstruction = F.mse_loss(output.reconstruction, activations)
    zero = reconstruction.new_zeros(())
    auxiliary = zero
    if variant == "relu_l1":
        sparsity = output.feature_acts.abs().sum(dim=-1).mean()
    elif variant == "topk":
        sparsity = (output.feature_acts > 0).float().sum(dim=-1).mean()
    elif variant == "gated":
        if output.gate_pre_acts is None or output.auxiliary_reconstruction is None:
            raise ValueError("gated loss requires gate logits and an auxiliary reconstruction")
        sparsity = F.relu(output.gate_pre_acts).sum(dim=-1).mean()
        auxiliary = gated_aux_coefficient * F.mse_loss(output.auxiliary_reconstruction, activations)
    elif variant == "jumprelu":
        if output.jump_threshold is None:
            raise ValueError("JumpReLU loss requires learned thresholds")
        sparsity = HeavisideSTE.apply(
            F.relu(output.pre_acts),
            output.jump_threshold,
            jump_bandwidth,
        ).sum(dim=-1).mean()
    else:
        raise ValueError(f"unknown SAE variant: {variant}")

    sparsity_term = zero if variant == "topk" else sparsity_coefficient * sparsity
    return SAELoss(
        total=reconstruction + sparsity_term + auxiliary,
        reconstruction=reconstruction,
        sparsity=sparsity,
        auxiliary=auxiliary,
    )


# %%
class SparseAutoencoder(t.nn.Module):
    """Small trainable SAE whose encoder and loss rules come from learner code."""

    def __init__(self, train_activations: t.Tensor, config: SAETrainConfig):
        super().__init__()
        if train_activations.device.type != "cpu":
            raise ValueError("the planted comparison is intentionally CPU-only")
        self.config = config
        generator = t.Generator(device="cpu").manual_seed(config.seed)
        indices = t.randperm(train_activations.shape[0], generator=generator)[: config.d_sae]
        center = train_activations.mean(dim=0)
        decoder = F.normalize(train_activations[indices] - center, dim=-1)
        self.w_dec = t.nn.Parameter(decoder)
        self.w_enc = t.nn.Parameter(decoder.T.clone())
        self.b_enc = t.nn.Parameter(t.zeros(config.d_sae))
        self.b_dec = t.nn.Parameter(center.clone())
        if config.variant == "gated":
            self.w_gate = t.nn.Parameter(decoder.T.clone())
            self.b_gate = t.nn.Parameter(t.zeros(config.d_sae))
        if config.variant == "jumprelu":
            self.log_threshold = t.nn.Parameter(t.full((config.d_sae,), -2.3))

    def forward(self, activations: t.Tensor, encoders: EncoderFunctions = REFERENCE_ENCODERS) -> SAEForward:
        centered = activations - self.b_dec
        pre_acts = centered @ self.w_enc + self.b_enc
        gate_pre_acts = None
        auxiliary_reconstruction = None
        jump_threshold = None
        if self.config.variant == "relu_l1":
            feature_acts = encoders.relu_l1(pre_acts)
        elif self.config.variant == "topk":
            feature_acts = encoders.topk(pre_acts, self.config.k)
        elif self.config.variant == "gated":
            gate_pre_acts = centered @ self.w_gate + self.b_gate
            feature_acts = encoders.gated(pre_acts, gate_pre_acts)
            auxiliary_reconstruction = F.relu(gate_pre_acts) @ self.w_dec.detach() + self.b_dec.detach()
        else:
            jump_threshold = self.log_threshold.exp()
            feature_acts = encoders.jumprelu(pre_acts, jump_threshold, self.config.jump_bandwidth)
        return SAEForward(
            reconstruction=decode_features(feature_acts, self.w_dec, self.b_dec),
            feature_acts=feature_acts,
            pre_acts=pre_acts,
            gate_pre_acts=gate_pre_acts,
            auxiliary_reconstruction=auxiliary_reconstruction,
            jump_threshold=jump_threshold,
        )

    @t.no_grad()
    def normalize_decoder(self) -> None:
        self.w_dec.div_(self.w_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def train_sae_variant(
    train_activations: t.Tensor,
    config: SAETrainConfig,
    *,
    encoders: EncoderFunctions = REFERENCE_ENCODERS,
    loss_fn: Callable[..., SAELoss] = sparse_autoencoder_loss,
) -> tuple[SparseAutoencoder, tuple[float, ...]]:
    """Train one variant with deterministic CPU minibatches."""

    model = SparseAutoencoder(train_activations, config)
    optimizer = t.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = t.Generator(device="cpu").manual_seed(config.seed + 100)
    loss_log: list[float] = []
    for step in range(config.steps):
        indices = t.randint(train_activations.shape[0], (config.batch_size,), generator=generator)
        batch = train_activations[indices]
        output = model(batch, encoders)
        losses = loss_fn(
            output,
            batch,
            variant=config.variant,
            sparsity_coefficient=config.sparsity_coefficient,
            gated_aux_coefficient=config.gated_aux_coefficient,
            jump_bandwidth=config.jump_bandwidth,
        )
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.normalize_decoder()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == config.steps:
            loss_log.append(float(losses.total.detach()))
    return model, tuple(loss_log)


# %%
def feature_density(feature_acts: t.Tensor, threshold: float = 0.0) -> t.Tensor:
    if feature_acts.ndim < 2:
        raise ValueError("feature_acts must include example and feature dimensions")
    reduce_dims = tuple(range(feature_acts.ndim - 1))
    return (feature_acts > threshold).float().mean(dim=reduce_dims)


def l0(feature_acts: t.Tensor, threshold: float = 0.0) -> float:
    return float((feature_acts > threshold).float().sum(dim=-1).mean())


def dead_feature_fraction(feature_acts: t.Tensor, threshold: float = 0.0) -> float:
    return float(feature_density(feature_acts, threshold).eq(0).float().mean())


def density_is_nondegenerate(
    feature_acts: t.Tensor,
    *,
    min_active_fraction: float = 0.001,
    max_active_fraction: float = 0.95,
) -> bool:
    densities = feature_density(feature_acts)
    return bool(densities.gt(min_active_fraction).any() and densities.lt(max_active_fraction).any())


def sae_variant_metrics(
    name: str,
    *,
    activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    feature_acts: t.Tensor,
) -> SAEVariantMetrics:
    if activations.shape != reconstructed_activations.shape:
        raise ValueError("activations and reconstructions must have matching shapes")
    return SAEVariantMetrics(
        name=name,
        l0=l0(feature_acts),
        feature_density_mean=float(feature_density(feature_acts).mean()),
        dead_feature_fraction=dead_feature_fraction(feature_acts),
        reconstruction_mse=float(F.mse_loss(reconstructed_activations, activations)),
    )


def dictionary_recovery_report(
    learned_decoder: t.Tensor,
    true_dictionary: t.Tensor,
    *,
    threshold: float = 0.8,
) -> DictionaryRecoveryReport:
    if learned_decoder.ndim != 2 or true_dictionary.ndim != 2:
        raise ValueError("both dictionaries must be rank-two tensors")
    if learned_decoder.shape[1] != true_dictionary.shape[1]:
        raise ValueError("dictionary dimensions must match")
    cosine = F.normalize(learned_decoder.float(), dim=-1) @ F.normalize(true_dictionary.float(), dim=-1).T
    best_cosine, best_learned = cosine.max(dim=0)
    best_true_for_learned = cosine.argmax(dim=-1)
    duplicate_fraction = 1.0 - best_true_for_learned.unique().numel() / learned_decoder.shape[0]
    return DictionaryRecoveryReport(
        mean_best_cosine=float(best_cosine.mean()),
        recovered_fraction=float((best_cosine >= threshold).float().mean()),
        duplicate_fraction=duplicate_fraction,
        best_learned_for_true=best_learned,
        cosine_matrix=cosine,
    )


def roc_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Compute tie-aware binary ROC AUC from ranks."""

    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have equal length")
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one example from each class")
    order = scores.argsort()
    sorted_scores = scores[order]
    sorted_ranks = t.empty_like(sorted_scores)
    _, counts = t.unique_consecutive(sorted_scores, return_counts=True)
    start = 0
    for count in counts.tolist():
        end = start + count
        sorted_ranks[start:end] = (start + 1 + end) / 2
        start = end
    ranks = t.empty_like(scores)
    ranks[order] = sorted_ranks
    positive_rank_sum = ranks[labels].sum()
    auc = (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def best_feature_auc(feature_acts: t.Tensor, labels: t.Tensor) -> FeatureAUCReport:
    """Select a feature and polarity on one split only."""

    if feature_acts.ndim != 2 or labels.shape != (feature_acts.shape[0],):
        raise ValueError("expected feature_acts (examples, features) and labels (examples,)")
    best = FeatureAUCReport(feature_id=0, auc=-1.0, polarity=1)
    for feature_id in range(feature_acts.shape[1]):
        raw_auc = roc_auc_binary(feature_acts[:, feature_id], labels)
        polarity = 1 if raw_auc >= 0.5 else -1
        oriented_auc = raw_auc if polarity == 1 else 1.0 - raw_auc
        if oriented_auc > best.auc:
            best = FeatureAUCReport(feature_id, oriented_auc, polarity)
    return best


def evaluate_selected_auc(feature_acts: t.Tensor, labels: t.Tensor, selection: FeatureAUCReport) -> float:
    raw_auc = roc_auc_binary(feature_acts[:, selection.feature_id], labels)
    return raw_auc if selection.polarity == 1 else 1.0 - raw_auc


def apply_decoder_steering(
    activations: t.Tensor,
    decoder_vectors: t.Tensor,
    feature_ids: t.Tensor | list[int],
    coefficients: t.Tensor | list[float] | float,
    *,
    positions: Literal["all", "last"] = "last",
) -> t.Tensor:
    feature_ids_tensor = t.as_tensor(feature_ids, device=activations.device, dtype=t.long)
    selected = decoder_vectors.to(device=activations.device, dtype=activations.dtype)[feature_ids_tensor]
    coeffs = t.as_tensor(coefficients, device=activations.device, dtype=activations.dtype)
    if coeffs.ndim == 0:
        coeffs = coeffs.expand(selected.shape[0])
    if coeffs.numel() != selected.shape[0]:
        raise ValueError("coefficients must be scalar or match feature_ids")
    steering_vector = (coeffs[:, None] * selected).sum(dim=0)
    steered = activations.clone()
    if positions == "all":
        return steered + steering_vector
    if positions == "last":
        steered[..., -1, :] += steering_vector
        return steered
    raise ValueError("positions must be 'all' or 'last'")


def steering_comparison_report(
    baseline_scores: t.Tensor,
    steered_scores: t.Tensor,
    random_control_scores: t.Tensor,
) -> SteeringComparisonReport:
    baseline_mean = float(baseline_scores.float().mean())
    steered_mean = float(steered_scores.float().mean())
    random_mean = float(random_control_scores.float().mean())
    steered_delta = steered_mean - baseline_mean
    random_delta = random_mean - baseline_mean
    return SteeringComparisonReport(
        baseline_mean,
        steered_mean,
        random_mean,
        steered_delta,
        random_delta,
        abs(steered_delta) > abs(random_delta),
    )


@t.no_grad()
def causal_intervention_report(
    model: SparseAutoencoder,
    heldout: PlantedSparseBatch,
    *,
    target_feature: int = 0,
    steering_coefficient: float = 0.75,
    random_seed: int = 17,
) -> CausalInterventionReport:
    """Compare matched decoder interventions with equal-strength random directions."""

    target_direction = F.normalize(heldout.dictionary[target_feature], dim=0)
    decoder = F.normalize(model.w_dec.detach(), dim=-1)
    target_feature_id = int((decoder @ target_direction).argmax())
    matched_direction = decoder[target_feature_id]
    generator = t.Generator(device="cpu").manual_seed(random_seed)
    random_direction = t.randn(target_direction.shape, generator=generator)
    random_direction -= (random_direction @ target_direction) * target_direction
    random_direction = F.normalize(random_direction, dim=0)

    negative_examples = heldout.activations[~heldout.labels]
    baseline_scores = negative_examples @ target_direction
    steered_scores = (negative_examples + steering_coefficient * matched_direction) @ target_direction
    random_scores = (negative_examples + steering_coefficient * random_direction) @ target_direction

    positive_output = model(heldout.activations[heldout.labels])
    coefficients = positive_output.feature_acts[:, target_feature_id]
    baseline_reconstruction = positive_output.reconstruction
    target_ablated = baseline_reconstruction - coefficients[:, None] * matched_direction
    random_ablated = baseline_reconstruction - coefficients[:, None] * random_direction
    baseline_projection = baseline_reconstruction @ target_direction
    return CausalInterventionReport(
        target_feature_id=target_feature_id,
        decoder_cosine=float(matched_direction @ target_direction),
        steering_delta=float((steered_scores - baseline_scores).mean()),
        random_steering_delta=float((random_scores - baseline_scores).mean()),
        ablation_drop=float((baseline_projection - target_ablated @ target_direction).mean()),
        random_ablation_drop=float((baseline_projection - random_ablated @ target_direction).mean()),
    )


@t.no_grad()
def evaluate_variant(
    name: Variant,
    model: SparseAutoencoder,
    train: PlantedSparseBatch,
    heldout: PlantedSparseBatch,
    train_losses: tuple[float, ...],
    *,
    encoders: EncoderFunctions = REFERENCE_ENCODERS,
) -> VariantResult:
    train_output = model(train.activations, encoders)
    heldout_output = model(heldout.activations, encoders)
    metrics = sae_variant_metrics(
        name,
        activations=heldout.activations,
        reconstructed_activations=heldout_output.reconstruction,
        feature_acts=heldout_output.feature_acts,
    )
    recovery = dictionary_recovery_report(model.w_dec, heldout.dictionary)
    selection = best_feature_auc(train_output.feature_acts, train.labels)
    heldout_auc = evaluate_selected_auc(heldout_output.feature_acts, heldout.labels, selection)
    generator = t.Generator(device="cpu").manual_seed(1000 + VARIANTS.index(name))
    shuffled_train_labels = train.labels[t.randperm(train.labels.numel(), generator=generator)]
    shuffled_selection = best_feature_auc(train_output.feature_acts, shuffled_train_labels)
    shuffled_heldout_labels = heldout.labels[t.randperm(heldout.labels.numel(), generator=generator)]
    shuffled_auc = evaluate_selected_auc(heldout_output.feature_acts, shuffled_heldout_labels, shuffled_selection)
    intervention = causal_intervention_report(model, heldout)
    return VariantResult(
        name,
        model,
        metrics,
        recovery,
        heldout_auc,
        shuffled_auc,
        intervention,
        train_losses,
    )


def run_variant_comparison(
    *,
    steps: int = 900,
    train_examples: int = 4096,
    heldout_examples: int = 2048,
    n_features: int = 8,
    d_model: int = 6,
    feature_probability: float = 0.15,
    noise_std: float = 0.01,
    sparsity_coefficient: float = 0.02,
    k: int = 2,
    seed: int = 1,
    encoders: EncoderFunctions = REFERENCE_ENCODERS,
    loss_fn: Callable[..., SAELoss] = sparse_autoencoder_loss,
) -> SAEComparison:
    """Train and evaluate all four variants on identical planted splits."""

    train, heldout = make_planted_splits(
        train_examples=train_examples,
        heldout_examples=heldout_examples,
        n_features=n_features,
        d_model=d_model,
        feature_probability=feature_probability,
        noise_std=noise_std,
        seed=seed,
    )
    results: dict[str, VariantResult] = {}
    for variant in VARIANTS:
        config = SAETrainConfig(
            variant=variant,
            d_sae=n_features,
            k=k,
            sparsity_coefficient=sparsity_coefficient,
            steps=steps,
            seed=seed + 2,
        )
        model, losses = train_sae_variant(
            train.activations,
            config,
            encoders=encoders,
            loss_fn=loss_fn,
        )
        results[variant] = evaluate_variant(
            variant,
            model,
            train,
            heldout,
            losses,
            encoders=encoders,
        )
    zero_baseline = train.activations.mean(0).expand_as(heldout.activations)
    zero_baseline_mse = float(F.mse_loss(zero_baseline, heldout.activations))
    generator = t.Generator(device="cpu").manual_seed(seed + 500)
    random_decoder = F.normalize(t.randn(n_features, d_model, generator=generator), dim=-1)
    random_decoder_recovery = dictionary_recovery_report(
        random_decoder,
        heldout.dictionary,
    ).recovered_fraction
    return SAEComparison(
        train=train,
        heldout=heldout,
        results=results,
        zero_baseline_mse=zero_baseline_mse,
        true_l0=l0(heldout.feature_acts),
        random_decoder_recovery=random_decoder_recovery,
    )


def comparison_rows(comparison: SAEComparison) -> list[dict[str, float | str]]:
    rows = []
    for name in VARIANTS:
        result = comparison.results[name]
        rows.append(
            {
                "variant": name,
                "heldout_mse": result.metrics.reconstruction_mse,
                "l0": result.metrics.l0,
                "mean_density": result.metrics.feature_density_mean,
                "dead_fraction": result.metrics.dead_feature_fraction,
                "mean_best_cosine": result.recovery.mean_best_cosine,
                "recovered_fraction": result.recovery.recovered_fraction,
                "heldout_auc": result.heldout_auc,
                "shuffled_auc": result.shuffled_auc,
                "steering_delta": result.intervention.steering_delta,
                "random_steering_delta": result.intervention.random_steering_delta,
                "ablation_drop": result.intervention.ablation_drop,
                "random_ablation_drop": result.intervention.random_ablation_drop,
            }
        )
    return rows


def plot_signature_result(comparison: SAEComparison, *, save_path: str | Path | None = None):
    """Plot all headline metrics with their appropriate negative controls."""

    import matplotlib.pyplot as plt

    names = list(VARIANTS)
    labels = ["ReLU-L1", "TopK", "Gated", "JumpReLU"]
    colors = ["#2563eb", "#d97706", "#059669", "#7c3aed"]
    x = t.arange(len(names)).numpy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))

    axes[0, 0].bar(labels, [comparison.results[n].metrics.reconstruction_mse for n in names], color=colors)
    axes[0, 0].axhline(comparison.zero_baseline_mse, color="#374151", linestyle="--", label="train-mean baseline")
    axes[0, 0].set(title="Held-out reconstruction", ylabel="MSE (lower is better)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(labels, [comparison.results[n].metrics.l0 for n in names], color=colors)
    axes[0, 1].axhline(comparison.true_l0, color="#374151", linestyle="--", label="true latent L0")
    axes[0, 1].set(title="Sparsity budget", ylabel="active features / example")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].bar(labels, [comparison.results[n].recovery.recovered_fraction for n in names], color=colors)
    axes[0, 2].axhline(comparison.random_decoder_recovery, color="#374151", linestyle="--", label="random decoder")
    axes[0, 2].set(title="Planted dictionary recovery", ylabel="fraction cosine >= 0.8", ylim=(0, 1.08))
    axes[0, 2].legend(fontsize=8)

    width = 0.36
    axes[1, 0].bar(x - width / 2, [comparison.results[n].heldout_auc for n in names], width, color=colors, label="held-out label")
    axes[1, 0].bar(x + width / 2, [comparison.results[n].shuffled_auc for n in names], width, color="#9ca3af", label="shuffled label")
    axes[1, 0].axhline(0.5, color="#111827", linewidth=0.8)
    axes[1, 0].set(title="Feature predicts target presence", ylabel="ROC AUC", ylim=(0.35, 1.03), xticks=x, xticklabels=labels)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(x - width / 2, [comparison.results[n].intervention.steering_delta for n in names], width, color=colors, label="matched decoder")
    random_steering = [comparison.results[n].intervention.random_steering_delta for n in names]
    axes[1, 1].bar(x + width / 2, random_steering, width, color="#9ca3af", label="random orthogonal")
    axes[1, 1].scatter(x + width / 2, random_steering, color="#374151", marker="D", s=24, zorder=3)
    axes[1, 1].set(title="Causal steering", ylabel="target projection delta", xticks=x, xticklabels=labels)
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].bar(x - width / 2, [comparison.results[n].intervention.ablation_drop for n in names], width, color=colors, label="remove decoder contribution")
    random_ablation = [comparison.results[n].intervention.random_ablation_drop for n in names]
    axes[1, 2].bar(x + width / 2, random_ablation, width, color="#9ca3af", label="same coefficient, random dir")
    axes[1, 2].scatter(x + width / 2, random_ablation, color="#374151", marker="D", s=24, zorder=3)
    axes[1, 2].set(title="Causal ablation", ylabel="target projection drop", xticks=x, xticklabels=labels)
    axes[1, 2].legend(fontsize=8)

    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("SAE variants on one exact planted sparse dictionary", fontsize=15)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu  # Kept for the repository-wide report runner's historical API.
    comparison = run_variant_comparison(steps=250, train_examples=1024, heldout_examples=512)
    return {
        "variants": comparison_rows(comparison),
        "zero_baseline_mse": comparison.zero_baseline_mse,
        "true_l0": comparison.true_l0,
        "random_decoder_recovery": comparison.random_decoder_recovery,
    }


# %%
# Optional real-model path. The parent verification runner owns CUDA execution.
class _TinyTopKSAE(t.nn.Module):
    def __init__(self, *, d_model: int, d_sae: int, k: int, device: t.device, seed: int = 0):
        super().__init__()
        self.k = k
        generator = t.Generator(device=device).manual_seed(seed)
        decoder = F.normalize(t.randn(d_sae, d_model, generator=generator, device=device), dim=-1)
        self.w_dec = t.nn.Parameter(decoder)
        self.w_enc = t.nn.Parameter(decoder.T.clone())
        self.b_enc = t.nn.Parameter(t.zeros(d_sae, device=device))
        self.b_dec = t.nn.Parameter(t.zeros(d_model, device=device))

    def forward(self, activations: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        feature_acts = topk_encode(activations @ self.w_enc + self.b_enc, k=self.k)
        return feature_acts @ self.w_dec + self.b_dec, feature_acts

    @t.no_grad()
    def normalize_decoder(self) -> None:
        self.w_dec.div_(self.w_dec.norm(dim=-1, keepdim=True).clamp_min(1e-6))


def _load_pythia_sae_model_on_cuda():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PYTHIA_SAE_MODEL_ID, revision=PYTHIA_SAE_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        PYTHIA_SAE_MODEL_ID,
        revision=PYTHIA_SAE_REVISION,
        dtype=t.float32,
    ).to("cuda")
    model.eval()
    return tokenizer, model


def _build_pythia_sae_examples(
    *,
    technical_topics: list[str],
    everyday_topics: list[str],
    templates: list[str],
) -> list[tuple[str, int]]:
    examples: list[tuple[str, int]] = []
    for topic in technical_topics:
        examples.extend((template.format(topic=topic), 1) for template in templates)
    for topic in everyday_topics:
        examples.extend((template.format(topic=topic), 0) for template in templates)
    return examples


def _final_token_hidden_states(tokenizer, model, prompts: list[str], *, batch_size: int = 16) -> t.Tensor:
    hidden_states = []
    with t.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[start : start + batch_size], padding=True, return_tensors="pt").to("cuda")
            output = model(**encoded, output_hidden_states=True)
            final_token_indices = encoded.attention_mask.sum(dim=1) - 1
            batch_indices = t.arange(encoded.input_ids.shape[0], device="cuda")
            hidden_states.append(
                output.hidden_states[PYTHIA_SAE_HIDDEN_LAYER][batch_indices, final_token_indices].float().detach()
            )
    return t.cat(hidden_states, dim=0)


def _train_tiny_topk_sae(
    train_activations: t.Tensor,
    *,
    d_sae: int = PYTHIA_SAE_WIDTH,
    k: int = PYTHIA_SAE_TOPK,
    steps: int = PYTHIA_SAE_TRAINING_STEPS,
) -> _TinyTopKSAE:
    sae = _TinyTopKSAE(
        d_model=train_activations.shape[-1],
        d_sae=d_sae,
        k=k,
        device=train_activations.device,
    )
    optimizer = t.optim.AdamW(sae.parameters(), lr=3e-3, weight_decay=1e-4)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, feature_acts = sae(train_activations)
        loss = F.mse_loss(reconstruction, train_activations) + 1e-5 * feature_acts.abs().mean()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()
    return sae


def _safe_single_token_ids(tokenizer) -> list[int]:
    token_ids = []
    for token in PYTHIA_SAE_SAFE_LOGIT_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded:
            token_ids.append(int(encoded[0]))
    return sorted(set(token_ids))


def run_pythia_sae_variants_preflight(max_vram_gb: float = 24.0) -> dict:
    """Train a TopK SAE on pinned Pythia hidden states and run bounded controls."""

    if not t.cuda.is_available():
        raise RuntimeError("the optional Pythia SAE preflight requires CUDA")
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    tokenizer, model = _load_pythia_sae_model_on_cuda()
    train_examples = _build_pythia_sae_examples(
        technical_topics=PYTHIA_SAE_TECHNICAL_TOPICS[:4],
        everyday_topics=PYTHIA_SAE_EVERYDAY_TOPICS[:4],
        templates=PYTHIA_SAE_TRAIN_TEMPLATES,
    )
    heldout_examples = _build_pythia_sae_examples(
        technical_topics=PYTHIA_SAE_TECHNICAL_TOPICS[4:],
        everyday_topics=PYTHIA_SAE_EVERYDAY_TOPICS[4:],
        templates=PYTHIA_SAE_HELDOUT_TEMPLATES,
    )
    train_hidden = _final_token_hidden_states(tokenizer, model, [prompt for prompt, _ in train_examples])
    heldout_hidden = _final_token_hidden_states(tokenizer, model, [prompt for prompt, _ in heldout_examples])
    train_labels = t.tensor([label for _, label in train_examples], device=device).bool()
    heldout_labels = t.tensor([label for _, label in heldout_examples], device=device).bool()
    train_mean = train_hidden.mean(0)
    train_std = train_hidden.std(0, unbiased=False).clamp_min(1e-3)
    train_normalized = (train_hidden - train_mean) / train_std
    heldout_normalized = (heldout_hidden - train_mean) / train_std
    sae = _train_tiny_topk_sae(train_normalized)

    with t.no_grad():
        train_reconstruction, train_features = sae(train_normalized)
        heldout_reconstruction, heldout_features = sae(heldout_normalized)
        train_metrics = sae_variant_metrics(
            "pythia_train",
            activations=train_normalized,
            reconstructed_activations=train_reconstruction,
            feature_acts=train_features,
        )
        heldout_metrics = sae_variant_metrics(
            "pythia_heldout",
            activations=heldout_normalized,
            reconstructed_activations=heldout_reconstruction,
            feature_acts=heldout_features,
        )
        zero_baseline_mse = float(heldout_normalized.square().mean())
        generator = t.Generator(device=device).manual_seed(2)
        permutation = t.randperm(sae.w_dec.shape[0], generator=generator, device=device)
        permuted_reconstruction = heldout_features @ sae.w_dec[permutation] + sae.b_dec
        permuted_decoder_mse = float(F.mse_loss(permuted_reconstruction, heldout_normalized))
        selection = best_feature_auc(train_features.cpu(), train_labels.cpu())
        feature_auc = evaluate_selected_auc(heldout_features.cpu(), heldout_labels.cpu(), selection)
        semantic_feature_negative_result = feature_auc < 0.65
        selected_scores = heldout_features[:, selection.feature_id]
        feature_positive_mean = float(selected_scores[heldout_labels].mean())
        feature_negative_mean = float(selected_scores[~heldout_labels].mean())

        decoder_direction = sae.w_dec[selection.feature_id]
        projection_baseline = heldout_normalized @ decoder_direction
        projection_steered = (heldout_normalized + PYTHIA_SAE_STEERING_ALPHA * decoder_direction) @ decoder_direction
        random_direction = t.randn(
            decoder_direction.shape,
            generator=t.Generator(device=device).manual_seed(1),
            device=device,
        )
        random_direction -= (random_direction @ decoder_direction) * decoder_direction / decoder_direction.square().sum().clamp_min(1e-8)
        random_direction = F.normalize(random_direction, dim=0) * decoder_direction.norm()
        projection_random = (heldout_normalized + PYTHIA_SAE_STEERING_ALPHA * random_direction) @ decoder_direction

        decoder_direction_hidden = decoder_direction * train_std
        candidate_ids = t.tensor(_safe_single_token_ids(tokenizer), device=device)
        candidate_effects = model.embed_out(decoder_direction_hidden)[candidate_ids]
        target_index = int(candidate_effects.abs().argmax())
        target_token_id = int(candidate_ids[target_index])
        target_effect = candidate_effects[target_index]
        signed_direction = (1.0 if float(target_effect) >= 0 else -1.0) * decoder_direction_hidden
        baseline_logits = model.embed_out(heldout_hidden)[:, target_token_id]
        steered_logits = model.embed_out(
            heldout_hidden + PYTHIA_SAE_STEERING_ALPHA * signed_direction
        )[:, target_token_id]
        safe_logit_delta = float(steered_logits.mean() - baseline_logits.mean())

    steering = steering_comparison_report(projection_baseline, projection_steered, projection_random)
    reconstruction_improvement = (zero_baseline_mse - heldout_metrics.reconstruction_mse) / zero_baseline_mse
    decoder_ratio = permuted_decoder_mse / heldout_metrics.reconstruction_mse
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        reconstruction_improvement >= 0.15
        and decoder_ratio >= 1.2
        and density_is_nondegenerate(heldout_features)
        and semantic_feature_negative_result
        and safe_logit_delta >= 0.5
        and steering.passes_control
        and within_budget
    )
    safe_logit_token = tokenizer.decode([target_token_id])
    del model
    t.cuda.empty_cache()
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_id": PYTHIA_SAE_MODEL_ID,
        "revision": PYTHIA_SAE_REVISION,
        "hidden_layer": PYTHIA_SAE_HIDDEN_LAYER,
        "generation_used": False,
        "train_prompt_count": len(train_examples),
        "heldout_prompt_count": len(heldout_examples),
        "d_model": int(train_hidden.shape[-1]),
        "sae_width": PYTHIA_SAE_WIDTH,
        "sae_k": PYTHIA_SAE_TOPK,
        "training_steps": PYTHIA_SAE_TRAINING_STEPS,
        "train_reconstruction_mse": train_metrics.reconstruction_mse,
        "heldout_reconstruction_mse": heldout_metrics.reconstruction_mse,
        "zero_baseline_mse": zero_baseline_mse,
        "reconstruction_improvement_vs_zero": reconstruction_improvement,
        "permuted_decoder_mse": permuted_decoder_mse,
        "random_decoder_mse_ratio": decoder_ratio,
        "random_decoder_control_passed": decoder_ratio >= 1.2,
        "heldout_l0": heldout_metrics.l0,
        "heldout_feature_density_mean": heldout_metrics.feature_density_mean,
        "heldout_dead_feature_fraction": heldout_metrics.dead_feature_fraction,
        "density_nondegenerate": density_is_nondegenerate(heldout_features),
        "best_feature_id": selection.feature_id,
        "train_selected_feature_auc": selection.auc,
        "best_feature_auc": feature_auc,
        "best_feature_positive_mean": feature_positive_mean,
        "best_feature_negative_mean": feature_negative_mean,
        "best_feature_polarity": "technical" if selection.polarity == 1 else "everyday",
        "semantic_feature_claimed": False,
        "semantic_feature_negative_result": semantic_feature_negative_result,
        "safe_logit_token_id": target_token_id,
        "safe_logit_token": safe_logit_token,
        "safe_logit_target_effect_abs": abs(float(target_effect)),
        "safe_logit_delta": safe_logit_delta,
        "decoder_projection_steered_delta": steering.steered_delta,
        "decoder_projection_random_delta": steering.random_delta,
        "passes_decoder_steering_control": steering.passes_control,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_budget,
        "preflight_passed": preflight_passed,
        "full_path": (
            "Pinned Pythia-70M final-token hidden states, train-split normalization, "
            "a width-256 TopK-16 SAE, held-out reconstruction, permuted-decoder, "
            "held-out topic AUC, and orthogonal-direction intervention controls."
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_pythia_sae_variants_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
