# %%
"""Reference solutions for [6.1] SAE Variants."""

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

MAIN = __name__ == "__main__"

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
class ToySuperpositionBatch:
    feature_acts: t.Tensor
    activations: t.Tensor
    dictionary: t.Tensor


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


@dataclass(frozen=True)
class FeatureAUCReport:
    feature_id: int
    auc: float


@dataclass(frozen=True)
class SteeringComparisonReport:
    baseline_mean: float
    steered_mean: float
    random_mean: float
    steered_delta: float
    random_delta: float
    passes_control: bool


# %%
def make_toy_superposition_batch(
    *,
    batch: int = 256,
    n_features: int = 8,
    d_model: int = 4,
    feature_probability: float = 0.2,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> ToySuperpositionBatch:
    """Create sparse planted features mixed through a random dictionary."""

    if batch <= 0 or n_features <= 0 or d_model <= 0:
        raise ValueError("batch, n_features, and d_model must be positive.")
    if not 0 <= feature_probability <= 1:
        raise ValueError("feature_probability must be in [0, 1].")
    if noise_scale < 0:
        raise ValueError("noise_scale must be nonnegative.")

    generator = t.Generator().manual_seed(seed)
    dictionary = F.normalize(t.randn(n_features, d_model, generator=generator), dim=-1)
    active = t.rand(batch, n_features, generator=generator) < feature_probability
    magnitudes = 0.5 + t.rand(batch, n_features, generator=generator)
    feature_acts = active.float() * magnitudes
    activations = feature_acts @ dictionary
    if noise_scale:
        activations = activations + noise_scale * t.randn(
            activations.shape,
            generator=generator,
        )
    return ToySuperpositionBatch(
        feature_acts=feature_acts,
        activations=activations,
        dictionary=dictionary,
    )


def relu_l1_encode(pre_acts: t.Tensor, *, l1_coefficient: float = 0.0) -> t.Tensor:
    """ReLU encoder with soft-thresholding standing in for L1 pressure."""

    if l1_coefficient < 0:
        raise ValueError("l1_coefficient must be nonnegative.")
    return (pre_acts - l1_coefficient).clamp_min(0)


def topk_encode(pre_acts: t.Tensor, *, k: int) -> t.Tensor:
    """Keep the top-k nonnegative feature activations per example."""

    if pre_acts.ndim < 1:
        raise ValueError("pre_acts must have at least one dimension.")
    if k < 0 or k > pre_acts.shape[-1]:
        raise ValueError("k must be between 0 and the number of features.")
    relu_acts = pre_acts.clamp_min(0)
    if k == 0:
        return t.zeros_like(relu_acts)
    values, indices = relu_acts.topk(k=k, dim=-1)
    encoded = t.zeros_like(relu_acts)
    return encoded.scatter(-1, indices, values)


def gated_encode(
    pre_acts: t.Tensor,
    gate_logits: t.Tensor,
    *,
    gate_threshold: float = 0.0,
) -> t.Tensor:
    """Gated SAE-style encoder decoupling detection from magnitude."""

    if pre_acts.shape != gate_logits.shape:
        raise ValueError("pre_acts and gate_logits must have matching shapes.")
    return pre_acts.clamp_min(0) * gate_logits.gt(gate_threshold)


def jumprelu_encode(pre_acts: t.Tensor, *, threshold: float) -> t.Tensor:
    """JumpReLU encoder: activations fire only after crossing a threshold."""

    return t.where(pre_acts > threshold, pre_acts, t.zeros_like(pre_acts))


def decode_features(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor | None = None,
) -> t.Tensor:
    """Decode sparse feature activations back into model activations."""

    if feature_acts.shape[-1] != decoder_weight.shape[0]:
        raise ValueError("feature dimension must match decoder rows.")
    reconstructed = feature_acts.float() @ decoder_weight.float()
    if decoder_bias is not None:
        reconstructed = reconstructed + decoder_bias.to(reconstructed.device)
    return reconstructed


def feature_density(feature_acts: t.Tensor, threshold: float = 0.0) -> t.Tensor:
    """Return per-feature firing rate over all non-feature dimensions."""

    if feature_acts.ndim < 2:
        raise ValueError("feature_acts must have at least batch and feature dimensions.")
    fired = feature_acts > threshold
    reduce_dims = tuple(range(feature_acts.ndim - 1))
    return fired.float().mean(dim=reduce_dims)


def l0(feature_acts: t.Tensor, threshold: float = 0.0) -> float:
    """Return average number of active features per activation vector."""

    return (feature_acts > threshold).float().sum(dim=-1).mean().item()


def dead_feature_fraction(feature_acts: t.Tensor, threshold: float = 0.0) -> float:
    """Return fraction of features that never fire above threshold."""

    densities = feature_density(feature_acts, threshold=threshold)
    return densities.eq(0).float().mean().item()


def sae_variant_metrics(
    name: str,
    *,
    activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    feature_acts: t.Tensor,
) -> SAEVariantMetrics:
    """Compact metrics for comparing SAE variants."""

    if activations.shape != reconstructed_activations.shape:
        raise ValueError("activations and reconstructed_activations must have matching shapes.")
    return SAEVariantMetrics(
        name=name,
        l0=l0(feature_acts),
        feature_density_mean=feature_density(feature_acts).mean().item(),
        dead_feature_fraction=dead_feature_fraction(feature_acts),
        reconstruction_mse=F.mse_loss(
            reconstructed_activations.float(),
            activations.float(),
        ).item(),
    )


def dictionary_recovery_report(
    learned_decoder: t.Tensor,
    true_dictionary: t.Tensor,
    *,
    threshold: float = 0.8,
) -> DictionaryRecoveryReport:
    """Check whether learned decoder directions recover planted feature directions."""

    if learned_decoder.ndim != 2 or true_dictionary.ndim != 2:
        raise ValueError("learned_decoder and true_dictionary must be rank-2 tensors.")
    if learned_decoder.shape[1] != true_dictionary.shape[1]:
        raise ValueError("decoder and dictionary dimensions must match.")
    learned = F.normalize(learned_decoder.float(), dim=-1)
    true = F.normalize(true_dictionary.float(), dim=-1)
    cosine = learned @ true.T
    best_cosine, best_learned = cosine.max(dim=0)
    recovered = best_cosine >= threshold

    best_true_for_learned = cosine.argmax(dim=-1)
    unique_best = best_true_for_learned.unique().numel()
    duplicate_fraction = 1.0 - unique_best / learned_decoder.shape[0]

    return DictionaryRecoveryReport(
        mean_best_cosine=best_cosine.mean().item(),
        recovered_fraction=recovered.float().mean().item(),
        duplicate_fraction=duplicate_fraction,
        best_learned_for_true=best_learned,
    )


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


def best_feature_auc(feature_acts: t.Tensor, labels: t.Tensor) -> FeatureAUCReport:
    """Return the feature with the strongest binary-label separation."""

    if feature_acts.ndim != 2:
        raise ValueError("feature_acts must have shape (examples, features).")
    if labels.shape != (feature_acts.shape[0],):
        raise ValueError("labels must have shape (examples,).")

    best_id = 0
    best_auc = -1.0
    for feature_id in range(feature_acts.shape[1]):
        auc = roc_auc_binary(feature_acts[:, feature_id], labels)
        score = max(auc, 1.0 - auc)
        if score > best_auc:
            best_id = feature_id
            best_auc = score
    return FeatureAUCReport(feature_id=best_id, auc=best_auc)


def density_is_nondegenerate(
    feature_acts: t.Tensor,
    *,
    min_active_fraction: float = 0.05,
    max_active_fraction: float = 0.95,
) -> bool:
    """Check that some features fire, but not all features fire everywhere."""

    densities = feature_density(feature_acts)
    return bool(
        densities.gt(min_active_fraction).any()
        and densities.lt(max_active_fraction).any()
        and l0(feature_acts) > 0
    )


def apply_decoder_steering(
    activations: t.Tensor,
    decoder_vectors: t.Tensor,
    feature_ids: t.Tensor | list[int],
    coefficients: t.Tensor | list[float] | float,
    *,
    positions: Literal["all", "last"] = "last",
) -> t.Tensor:
    """Add decoder-vector steering directions to activations."""

    feature_ids_tensor = t.as_tensor(feature_ids, device=activations.device, dtype=t.long)
    selected = decoder_vectors.to(
        device=activations.device,
        dtype=activations.dtype,
    )[feature_ids_tensor]
    coeffs = t.as_tensor(coefficients, device=activations.device, dtype=activations.dtype)
    if coeffs.ndim == 0:
        coeffs = coeffs.expand(selected.shape[0])
    if coeffs.numel() != selected.shape[0]:
        raise ValueError("coefficients must be scalar or match number of feature_ids.")
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
    """Compare feature steering against random-feature steering."""

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


def encoder_variants_smoke_test() -> dict:
    pre_acts = t.tensor([[1.0, -1.0, 0.2, 3.0]])
    gate_logits = t.tensor([[1.0, 1.0, -1.0, 1.0]])
    return {
        "relu_l1": relu_l1_encode(pre_acts, l1_coefficient=0.5).tolist(),
        "topk": topk_encode(pre_acts, k=2).tolist(),
        "gated": gated_encode(pre_acts, gate_logits).tolist(),
        "jumprelu": jumprelu_encode(pre_acts, threshold=0.5).tolist(),
    }


def reconstruction_metrics_smoke_test() -> dict:
    feature_acts = t.tensor([[1.0, 0.0], [0.0, 2.0]])
    decoder = t.eye(2)
    activations = decode_features(feature_acts, decoder)
    metrics = sae_variant_metrics(
        "identity",
        activations=activations,
        reconstructed_activations=activations,
        feature_acts=feature_acts,
    )
    return metrics.__dict__


def toy_superposition_smoke_test() -> dict:
    batch = make_toy_superposition_batch(
        batch=32,
        n_features=6,
        d_model=3,
        feature_probability=0.25,
        seed=0,
    )
    return {
        "feature_shape": list(batch.feature_acts.shape),
        "activation_shape": list(batch.activations.shape),
        "dictionary_shape": list(batch.dictionary.shape),
        "density_nondegenerate": density_is_nondegenerate(batch.feature_acts),
    }


def dictionary_recovery_smoke_test() -> dict:
    true_dictionary = t.eye(3)
    learned_decoder = t.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    report = dictionary_recovery_report(learned_decoder, true_dictionary, threshold=0.9)
    return {
        **report.__dict__,
        "best_learned_for_true": report.best_learned_for_true.tolist(),
    }


def feature_auc_smoke_test() -> dict:
    feature_acts = t.tensor(
        [
            [0.1, 0.0],
            [0.2, 0.0],
            [0.1, 1.0],
            [0.2, 2.0],
        ]
    )
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)
    return best_feature_auc(feature_acts, labels).__dict__


def steering_control_smoke_test() -> dict:
    activations = t.zeros(2, 3, 3)
    decoder_vectors = t.eye(3)
    steered = apply_decoder_steering(activations, decoder_vectors, [1], 2.0)
    random_control = apply_decoder_steering(activations, decoder_vectors, [2], 0.1)
    report = steering_comparison_report(
        baseline_scores=activations[:, -1, 1],
        steered_scores=steered[:, -1, 1],
        random_control_scores=random_control[:, -1, 1],
    )
    return report.__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "encoders": encoder_variants_smoke_test(),
        "reconstruction": reconstruction_metrics_smoke_test(),
        "toy_superposition": toy_superposition_smoke_test(),
        "dictionary_recovery": dictionary_recovery_smoke_test(),
        "feature_auc": feature_auc_smoke_test(),
        "steering": steering_control_smoke_test(),
    }


class _TinyTopKSAE(t.nn.Module):
    """Small TopK SAE used for the real-LM activation preflight."""

    def __init__(
        self,
        *,
        d_model: int,
        d_sae: int,
        k: int,
        device: t.device,
        seed: int = 0,
    ):
        super().__init__()
        self.k = k
        generator = t.Generator(device=device).manual_seed(seed)
        decoder = t.randn(d_sae, d_model, generator=generator, device=device)
        decoder = t.nn.functional.normalize(decoder, dim=-1)
        self.w_dec = t.nn.Parameter(decoder)
        self.w_enc = t.nn.Parameter(decoder.T.clone())
        self.b_enc = t.nn.Parameter(t.zeros(d_sae, device=device))
        self.b_dec = t.nn.Parameter(t.zeros(d_model, device=device))

    def forward(self, activations: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        pre_acts = activations @ self.w_enc + self.b_enc
        feature_acts = topk_encode(pre_acts, k=self.k)
        reconstructed = feature_acts @ self.w_dec + self.b_dec
        return reconstructed, feature_acts

    def normalize_decoder(self) -> None:
        with t.no_grad():
            self.w_dec.div_(self.w_dec.norm(dim=-1, keepdim=True).clamp_min(1e-6))


def _load_pythia_sae_model_on_cuda():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        PYTHIA_SAE_MODEL_ID,
        revision=PYTHIA_SAE_REVISION,
    )
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


def _final_token_hidden_states(
    tokenizer,
    model,
    prompts: list[str],
    *,
    batch_size: int = 16,
) -> t.Tensor:
    hidden_states = []
    with t.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start : start + batch_size],
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            output = model(**encoded, output_hidden_states=True)
            final_token_indices = encoded.attention_mask.sum(dim=1) - 1
            batch_indices = t.arange(encoded.input_ids.shape[0], device="cuda")
            hidden_states.append(
                output.hidden_states[PYTHIA_SAE_HIDDEN_LAYER][
                    batch_indices,
                    final_token_indices,
                ]
                .float()
                .detach()
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
        reconstructed, feature_acts = sae(train_activations)
        loss = t.nn.functional.mse_loss(reconstructed, train_activations)
        loss = loss + 1e-5 * feature_acts.abs().mean()
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
    """Train a tiny TopK SAE on pinned Pythia hidden states and verify controls."""

    if not t.cuda.is_available():
        raise RuntimeError("SAE Variants GPU verification requires CUDA.")

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
    train_hidden_states = _final_token_hidden_states(
        tokenizer,
        model,
        [prompt for prompt, _label in train_examples],
    )
    heldout_hidden_states = _final_token_hidden_states(
        tokenizer,
        model,
        [prompt for prompt, _label in heldout_examples],
    )
    train_labels = t.tensor([label for _prompt, label in train_examples], device=device).bool()
    heldout_labels = t.tensor([label for _prompt, label in heldout_examples], device=device).bool()

    train_mean = train_hidden_states.mean(dim=0)
    train_std = train_hidden_states.std(dim=0, unbiased=False).clamp_min(1e-3)
    train_normalized = (train_hidden_states - train_mean) / train_std
    heldout_normalized = (heldout_hidden_states - train_mean) / train_std

    sae = _train_tiny_topk_sae(train_normalized)
    with t.no_grad():
        train_reconstructed, train_feature_acts = sae(train_normalized)
        heldout_reconstructed, heldout_feature_acts = sae(heldout_normalized)
        heldout_metrics = sae_variant_metrics(
            "pythia_topk_sae_heldout",
            activations=heldout_normalized,
            reconstructed_activations=heldout_reconstructed,
            feature_acts=heldout_feature_acts,
        )
        train_metrics = sae_variant_metrics(
            "pythia_topk_sae_train",
            activations=train_normalized,
            reconstructed_activations=train_reconstructed,
            feature_acts=train_feature_acts,
        )
        zero_baseline_mse = heldout_normalized.square().mean().item()
        permutation_generator = t.Generator(device=device).manual_seed(2)
        permuted_decoder = sae.w_dec[
            t.randperm(sae.w_dec.shape[0], generator=permutation_generator, device=device)
        ]
        permuted_reconstructed = heldout_feature_acts @ permuted_decoder + sae.b_dec
        permuted_decoder_mse = t.nn.functional.mse_loss(
            permuted_reconstructed,
            heldout_normalized,
        ).item()

        feature_auc = best_feature_auc(heldout_feature_acts, heldout_labels)
        selected_feature_scores = heldout_feature_acts[:, feature_auc.feature_id]
        feature_positive_mean = selected_feature_scores[heldout_labels].mean().item()
        feature_negative_mean = selected_feature_scores[~heldout_labels].mean().item()
        selected_feature_polarity = (
            "technical"
            if feature_positive_mean >= feature_negative_mean
            else "everyday"
        )

        decoder_direction = sae.w_dec[feature_auc.feature_id]
        projection_baseline_scores = heldout_normalized @ decoder_direction
        projection_steered_scores = (
            heldout_normalized + PYTHIA_SAE_STEERING_ALPHA * decoder_direction
        ) @ decoder_direction
        generator = t.Generator(device=device).manual_seed(1)
        random_direction = t.randn(decoder_direction.shape, generator=generator, device=device)
        random_direction = random_direction - (random_direction @ decoder_direction) * (
            decoder_direction / decoder_direction.norm().clamp_min(1e-8)
        )
        random_direction = random_direction / random_direction.norm().clamp_min(1e-8)
        random_direction = random_direction * decoder_direction.norm().clamp_min(1e-8)
        projection_random_scores = (
            heldout_normalized + PYTHIA_SAE_STEERING_ALPHA * random_direction
        ) @ decoder_direction

        decoder_direction_hidden = decoder_direction * train_std
        candidate_token_ids = t.tensor(_safe_single_token_ids(tokenizer), device=device)
        candidate_effects = model.embed_out(decoder_direction_hidden)[candidate_token_ids]
        target_index = int(candidate_effects.abs().argmax().item())
        target_token_id = int(candidate_token_ids[target_index].item())
        target_effect = candidate_effects[target_index]
        steering_sign = 1.0 if target_effect.item() >= 0 else -1.0
        steering_direction_hidden = steering_sign * decoder_direction_hidden
        baseline_scores = model.embed_out(heldout_hidden_states)[:, target_token_id]
        logit_steered_scores = model.embed_out(
            heldout_hidden_states + PYTHIA_SAE_STEERING_ALPHA * steering_direction_hidden
        )[:, target_token_id]
        safe_logit_delta = (
            logit_steered_scores.float().mean() - baseline_scores.float().mean()
        ).item()

    steering = steering_comparison_report(
        baseline_scores=projection_baseline_scores,
        steered_scores=projection_steered_scores,
        random_control_scores=projection_random_scores,
    )
    reconstruction_improvement_vs_zero = (
        zero_baseline_mse - heldout_metrics.reconstruction_mse
    ) / zero_baseline_mse
    random_decoder_ratio = permuted_decoder_mse / heldout_metrics.reconstruction_mse
    random_decoder_control_passed = random_decoder_ratio >= 1.2
    density_nondegenerate = density_is_nondegenerate(heldout_feature_acts)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        reconstruction_improvement_vs_zero >= 0.15
        and random_decoder_control_passed
        and density_nondegenerate
        and feature_auc.auc >= 0.95
        and safe_logit_delta >= 0.5
        and steering.passes_control
        and within_vram_budget
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
        "d_model": int(train_hidden_states.shape[-1]),
        "sae_width": PYTHIA_SAE_WIDTH,
        "sae_k": PYTHIA_SAE_TOPK,
        "training_steps": PYTHIA_SAE_TRAINING_STEPS,
        "train_reconstruction_mse": train_metrics.reconstruction_mse,
        "heldout_reconstruction_mse": heldout_metrics.reconstruction_mse,
        "zero_baseline_mse": zero_baseline_mse,
        "reconstruction_improvement_vs_zero": reconstruction_improvement_vs_zero,
        "permuted_decoder_mse": permuted_decoder_mse,
        "random_decoder_mse_ratio": random_decoder_ratio,
        "random_decoder_control_passed": random_decoder_control_passed,
        "heldout_l0": heldout_metrics.l0,
        "heldout_feature_density_mean": heldout_metrics.feature_density_mean,
        "heldout_dead_feature_fraction": heldout_metrics.dead_feature_fraction,
        "density_nondegenerate": density_nondegenerate,
        "best_feature_id": feature_auc.feature_id,
        "best_feature_auc": feature_auc.auc,
        "best_feature_positive_mean": feature_positive_mean,
        "best_feature_negative_mean": feature_negative_mean,
        "best_feature_polarity": selected_feature_polarity,
        "safe_logit_token_id": target_token_id,
        "safe_logit_token": safe_logit_token,
        "safe_logit_target_effect_abs": abs(float(target_effect.item())),
        "safe_logit_delta": safe_logit_delta,
        "decoder_projection_steered_delta": steering.steered_delta,
        "decoder_projection_random_delta": steering.random_delta,
        "passes_decoder_steering_control": steering.passes_control,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "preflight_passed": preflight_passed,
        "full_path": (
            "Pinned Pythia-70M hidden-state TopK SAE training preflight on CUDA, "
            "with held-out reconstruction improvement, permuted-decoder control, "
            "nondegenerate feature density, feature-label AUC, decoder-projection "
            "steering control, and safe vocabulary-logit movement."
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_pythia_sae_variants_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
