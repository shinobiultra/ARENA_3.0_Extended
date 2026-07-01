"""Mini Natural Language Autoencoder utilities for activation notebooks."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch as t
import torch.nn.functional as F

from arena_ext.activation_language import prediction_accuracy


@dataclass(frozen=True)
class NLATrainingBatch:
    activations: t.Tensor
    original_text_spans: tuple[str, ...]
    synthetic_latent_labels: tuple[str, ...]
    generated_explanations: tuple[str, ...]


@dataclass(frozen=True)
class NLAReconstructionReport:
    activation_mse: float
    text_only_mse: float
    mean_cosine_similarity: float
    beats_text_only: bool


@dataclass(frozen=True)
class LogitDiffPreservationReport:
    original_logit_diff: float
    reconstructed_logit_diff: float
    mean_abs_error: float
    preserves_target_logit_diff: bool


@dataclass(frozen=True)
class LatentPreservationReport:
    original_probe_accuracy: float
    reconstructed_probe_accuracy: float
    prediction_agreement: float
    preserves_latents: bool


@dataclass(frozen=True)
class GeneratedTextBrevityReport:
    generated_word_count: int
    original_word_count: int
    compression_ratio: float
    shorter_than_original: bool


@dataclass(frozen=True)
class CounterfactualExplanationReport:
    original_explanation: str
    counterfactual_explanation: str
    activation_delta: float
    explanation_changed: bool


@dataclass(frozen=True)
class TrainableNLABottleneckReport:
    encoder_final_loss: float
    decoder_final_mse: float
    encoder_train_accuracy: float
    eval_phrase_accuracy: float
    reconstruction_mse: float
    blank_text_mse: float
    beats_blank_text: bool
    generated_explanations: tuple[str, ...]
    phrase_count: int
    training_steps: int
    seed: int


def numeric_literal_count(texts: list[str]) -> int:
    """Count numeric literals in generated explanations."""

    return sum(len(re.findall(r"[+-]?\d+(?:\.\d+)?", text)) for text in texts)


def build_nla_training_batch(
    activations: t.Tensor,
    original_text_spans: list[str],
    synthetic_latent_labels: list[str],
    generated_explanations: list[str],
) -> NLATrainingBatch:
    """Bundle activations with text spans, latent labels, and explanations."""

    if activations.ndim < 2:
        raise ValueError("activations must have at least shape (examples, d_model).")
    expected_examples = activations.shape[0]
    lengths = {
        len(original_text_spans),
        len(synthetic_latent_labels),
        len(generated_explanations),
    }
    if lengths != {expected_examples}:
        raise ValueError("all text fields must have one entry per activation.")
    return NLATrainingBatch(
        activations=activations,
        original_text_spans=tuple(original_text_spans),
        synthetic_latent_labels=tuple(synthetic_latent_labels),
        generated_explanations=tuple(generated_explanations),
    )


def activation_reconstruction_report(
    original_activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    text_only_reconstructions: t.Tensor,
) -> NLAReconstructionReport:
    """Compare activation-to-text-to-activation reconstruction to a baseline."""

    matching_shapes = (
        original_activations.shape
        == reconstructed_activations.shape
        == text_only_reconstructions.shape
    )
    if not matching_shapes:
        raise ValueError("all activation tensors must have matching shape.")
    activation_mse = F.mse_loss(
        reconstructed_activations.float(),
        original_activations.float(),
    ).item()
    text_only_mse = F.mse_loss(
        text_only_reconstructions.float(),
        original_activations.float(),
    ).item()

    original_flat = original_activations.float().reshape(original_activations.shape[0], -1)
    reconstructed_flat = reconstructed_activations.float().reshape(
        reconstructed_activations.shape[0],
        -1,
    )
    mean_cosine = F.cosine_similarity(original_flat, reconstructed_flat, dim=-1)
    return NLAReconstructionReport(
        activation_mse=activation_mse,
        text_only_mse=text_only_mse,
        mean_cosine_similarity=mean_cosine.mean().item(),
        beats_text_only=activation_mse < text_only_mse,
    )


def batch_target_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> t.Tensor:
    """Return positive-minus-negative logit differences over a batch."""

    vocab_size = logits.shape[-1]
    if not 0 <= positive_token_id < vocab_size:
        raise ValueError("positive_token_id is out of range.")
    if not 0 <= negative_token_id < vocab_size:
        raise ValueError("negative_token_id is out of range.")
    return logits[..., positive_token_id] - logits[..., negative_token_id]


def logit_diff_preservation_report(
    original_logits: t.Tensor,
    reconstructed_logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
    max_mean_abs_error: float = 0.25,
) -> LogitDiffPreservationReport:
    """Check whether reconstructed activations preserve a target logit diff."""

    if original_logits.shape != reconstructed_logits.shape:
        raise ValueError("original_logits and reconstructed_logits must match.")
    original_diff = batch_target_logit_diff(
        original_logits.float(),
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    reconstructed_diff = batch_target_logit_diff(
        reconstructed_logits.float(),
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    mean_abs_error = (original_diff - reconstructed_diff).abs().mean().item()
    return LogitDiffPreservationReport(
        original_logit_diff=original_diff.mean().item(),
        reconstructed_logit_diff=reconstructed_diff.mean().item(),
        mean_abs_error=mean_abs_error,
        preserves_target_logit_diff=mean_abs_error <= max_mean_abs_error,
    )


def latent_preservation_report(
    original_probe_logits: t.Tensor,
    reconstructed_probe_logits: t.Tensor,
    latent_ids: t.Tensor,
    *,
    min_accuracy: float = 0.75,
    min_agreement: float = 0.75,
) -> LatentPreservationReport:
    """Check whether reconstructed activations preserve probe-decoded latents."""

    if original_probe_logits.shape != reconstructed_probe_logits.shape:
        raise ValueError("probe logits must have matching shape.")
    original_accuracy = prediction_accuracy(original_probe_logits, latent_ids)
    reconstructed_accuracy = prediction_accuracy(reconstructed_probe_logits, latent_ids)
    original_predictions = original_probe_logits.argmax(dim=-1)
    reconstructed_predictions = reconstructed_probe_logits.argmax(dim=-1)
    prediction_agreement = original_predictions.eq(reconstructed_predictions)
    prediction_agreement = prediction_agreement.float().mean().item()
    return LatentPreservationReport(
        original_probe_accuracy=original_accuracy,
        reconstructed_probe_accuracy=reconstructed_accuracy,
        prediction_agreement=prediction_agreement,
        preserves_latents=(
            reconstructed_accuracy >= min_accuracy and prediction_agreement >= min_agreement
        ),
    )


def generated_text_brevity_report(
    generated_explanations: list[str],
    original_prompts: list[str],
) -> GeneratedTextBrevityReport:
    """Check that generated explanations compress the original prompt text."""

    if len(generated_explanations) != len(original_prompts):
        raise ValueError("generated_explanations and original_prompts must align.")
    generated_word_count = sum(len(text.split()) for text in generated_explanations)
    original_word_count = sum(len(text.split()) for text in original_prompts)
    if original_word_count == 0:
        raise ValueError("original_prompts must contain at least one word.")
    compression_ratio = generated_word_count / original_word_count
    return GeneratedTextBrevityReport(
        generated_word_count=generated_word_count,
        original_word_count=original_word_count,
        compression_ratio=compression_ratio,
        shorter_than_original=generated_word_count < original_word_count,
    )


def counterfactual_explanation_report(
    original_activation: t.Tensor,
    counterfactual_activation: t.Tensor,
    original_explanation: str,
    counterfactual_explanation: str,
    *,
    min_activation_delta: float = 0.0,
) -> CounterfactualExplanationReport:
    """Check whether a counterfactual activation changes generated text."""

    if original_activation.shape != counterfactual_activation.shape:
        raise ValueError("activation tensors must have matching shape.")
    activation_delta = (
        counterfactual_activation.float() - original_activation.float()
    ).norm().item()
    explanation_changed = (
        original_explanation.strip().lower()
        != counterfactual_explanation.strip().lower()
    )
    return CounterfactualExplanationReport(
        original_explanation=original_explanation,
        counterfactual_explanation=counterfactual_explanation,
        activation_delta=activation_delta,
        explanation_changed=explanation_changed and activation_delta > min_activation_delta,
    )


def train_discrete_nla_bottleneck(
    train_activations: t.Tensor,
    train_phrase_ids: t.Tensor,
    eval_activations: t.Tensor,
    eval_phrase_ids: t.Tensor,
    phrase_texts: tuple[str, ...],
    *,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, TrainableNLABottleneckReport]:
    """Train a discrete activation->phrase->activation bottleneck.

    The encoder is a linear classifier from activations to phrase ids. The decoder
    is a trainable table mapping phrase ids back into activation space. This is a
    small local NLA, not a numeric coordinate payload: the only transmitted code is
    a discrete natural-language phrase id.
    """

    if train_activations.ndim != 2 or eval_activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    if train_activations.shape[1] != eval_activations.shape[1]:
        raise ValueError("train and eval activations must share d_model.")
    if train_phrase_ids.shape != (train_activations.shape[0],):
        raise ValueError("train_phrase_ids must have shape (train_examples,).")
    if eval_phrase_ids.shape != (eval_activations.shape[0],):
        raise ValueError("eval_phrase_ids must have shape (eval_examples,).")
    if not phrase_texts:
        raise ValueError("phrase_texts must be nonempty.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")

    n_phrases = len(phrase_texts)
    train_phrase_ids = train_phrase_ids.long()
    eval_phrase_ids = eval_phrase_ids.long()
    all_ids = t.cat([train_phrase_ids, eval_phrase_ids])
    if int(all_ids.min().item()) < 0 or int(all_ids.max().item()) >= n_phrases:
        raise ValueError("phrase ids must be in [0, len(phrase_texts)).")

    t.manual_seed(seed)
    if train_activations.device.type == "cuda":
        t.cuda.manual_seed_all(seed)

    train_x = train_activations.float()
    eval_x = eval_activations.float()
    mean = train_x.mean(dim=0, keepdim=True)
    train_features = F.normalize(train_x - mean, dim=-1)
    eval_features = F.normalize(eval_x - mean, dim=-1)

    d_model = train_x.shape[1]
    encoder_weight = (0.01 * t.randn(d_model, n_phrases, device=train_x.device)).requires_grad_()
    encoder_bias = t.zeros(n_phrases, device=train_x.device, requires_grad=True)
    global_mean = train_x.mean(dim=0)
    decoder_init = []
    for phrase_id in range(n_phrases):
        mask = train_phrase_ids == phrase_id
        decoder_init.append(train_x[mask].mean(dim=0) if bool(mask.any()) else global_mean)
    decoder_table = t.stack(decoder_init).detach().clone().requires_grad_()

    optimizer = t.optim.Adam(
        [encoder_weight, encoder_bias, decoder_table],
        lr=lr,
    )
    encoder_final_loss = float("nan")
    decoder_final_mse = float("nan")
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = train_features @ encoder_weight + encoder_bias
        encoder_loss = F.cross_entropy(logits, train_phrase_ids)
        decoder_reconstruction = decoder_table[train_phrase_ids]
        decoder_loss = F.mse_loss(decoder_reconstruction, train_x)
        loss = encoder_loss + decoder_loss
        loss.backward()
        optimizer.step()
        encoder_final_loss = float(encoder_loss.detach().item())
        decoder_final_mse = float(decoder_loss.detach().item())

    with t.no_grad():
        train_logits = train_features @ encoder_weight + encoder_bias
        eval_logits = eval_features @ encoder_weight + encoder_bias
        train_accuracy = prediction_accuracy(train_logits, train_phrase_ids)
        eval_predictions = eval_logits.argmax(dim=-1)
        eval_accuracy = eval_predictions.eq(eval_phrase_ids).float().mean().item()
        reconstructed_eval = decoder_table[eval_predictions].detach()
        blank_text = train_x.mean(dim=0, keepdim=True).expand_as(eval_x)
        reconstruction_mse = F.mse_loss(reconstructed_eval, eval_x).item()
        blank_text_mse = F.mse_loss(blank_text, eval_x).item()
        generated = tuple(phrase_texts[int(index)] for index in eval_predictions.tolist())

    report = TrainableNLABottleneckReport(
        encoder_final_loss=encoder_final_loss,
        decoder_final_mse=decoder_final_mse,
        encoder_train_accuracy=train_accuracy,
        eval_phrase_accuracy=eval_accuracy,
        reconstruction_mse=reconstruction_mse,
        blank_text_mse=blank_text_mse,
        beats_blank_text=reconstruction_mse < blank_text_mse,
        generated_explanations=generated,
        phrase_count=n_phrases,
        training_steps=steps,
        seed=seed,
    )
    return (
        encoder_weight.detach(),
        encoder_bias.detach(),
        decoder_table.detach(),
        eval_predictions.detach(),
        reconstructed_eval.detach(),
        report,
    )
