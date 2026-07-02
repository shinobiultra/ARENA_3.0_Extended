"""Activation-to-language utilities for lens and Patchscope notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t
import torch.nn.functional as F


PatchscopeTemplate = Literal["entity", "next_token", "fact"]


@dataclass(frozen=True)
class LensAccuracyReport:
    logit_lens_accuracy: float
    tuned_lens_accuracy: float
    improvement: float
    tuned_lens_improves: bool


@dataclass(frozen=True)
class PatchscopeAccuracyReport:
    patchscope_accuracy: float
    text_only_accuracy: float
    improvement: float
    beats_text_only: bool


@dataclass(frozen=True)
class CounterfactualActivationReport:
    original_answer: int
    patched_answer: int
    changed: bool


@dataclass(frozen=True)
class RandomActivationConfidenceReport:
    mean_confidence: float
    max_confidence: float
    passes_low_confidence: bool


def logit_lens(residual_stream: t.Tensor, unembedding: t.Tensor) -> t.Tensor:
    """Project residual activations directly into vocabulary logits."""

    if residual_stream.shape[-1] != unembedding.shape[0]:
        raise ValueError("residual_stream last dimension must match unembedding rows.")
    return residual_stream.float() @ unembedding.float()


def tuned_lens(
    residual_stream: t.Tensor,
    lens_weight: t.Tensor,
    lens_bias: t.Tensor | None,
    unembedding: t.Tensor,
) -> t.Tensor:
    """Apply a learned affine correction before projecting to vocabulary logits."""

    if residual_stream.shape[-1] != lens_weight.shape[0]:
        raise ValueError("residual_stream last dimension must match lens_weight rows.")
    transformed = residual_stream.float() @ lens_weight.float()
    if lens_bias is not None:
        transformed = transformed + lens_bias.to(transformed.device)
    return logit_lens(transformed, unembedding)


def attention_lens(
    attention_pattern: t.Tensor,
    value_vectors: t.Tensor,
    unembedding: t.Tensor,
) -> t.Tensor:
    """Decode attention-weighted value vectors through the unembedding."""

    if attention_pattern.ndim != 3 or value_vectors.ndim != 3:
        raise ValueError("attention_pattern and value_vectors must be rank-3 tensors.")
    if attention_pattern.shape[-1] != value_vectors.shape[-2]:
        raise ValueError("attention key dimension must match value sequence length.")
    attended = attention_pattern.float() @ value_vectors.float()
    return logit_lens(attended, unembedding)


def top_tokens(logits: t.Tensor, *, k: int = 5) -> tuple[t.Tensor, t.Tensor]:
    """Return top token ids and probabilities."""

    if k <= 0 or k > logits.shape[-1]:
        raise ValueError("k must be between 1 and vocab size.")
    probs = F.softmax(logits.float(), dim=-1)
    values, indices = probs.topk(k=k, dim=-1)
    return indices, values


def top_token_table(
    logits: t.Tensor,
    id_to_token,
    *,
    k: int = 5,
    target_token_ids: t.Tensor | None = None,
    row_labels: list[str] | None = None,
) -> list[dict[str, object]]:
    """Build a learner-readable table of top decoded tokens for rank-2 logits."""

    if logits.ndim != 2:
        raise ValueError("top_token_table expects logits with shape [row, vocab].")
    if target_token_ids is not None and target_token_ids.shape != logits.shape[:1]:
        raise ValueError("target_token_ids must have one id per logits row.")
    if row_labels is not None and len(row_labels) != logits.shape[0]:
        raise ValueError("row_labels must have one label per logits row.")

    top_ids, top_probs = top_tokens(logits, k=k)

    def decode(token_id: int) -> str:
        if callable(id_to_token):
            return str(id_to_token(token_id))
        return str(id_to_token[token_id])

    rows: list[dict[str, object]] = []
    sorted_ids = logits.float().argsort(dim=-1, descending=True)
    for row in range(logits.shape[0]):
        target_id = None if target_token_ids is None else int(target_token_ids[row].item())
        target_rank = None
        target_token = None
        if target_id is not None:
            target_token = decode(target_id)
            target_rank = int((sorted_ids[row] == target_id).nonzero(as_tuple=False)[0].item()) + 1
        rows.append(
            {
                "row": row if row_labels is None else row_labels[row],
                "top_ids": [int(x) for x in top_ids[row].tolist()],
                "top_tokens": [decode(int(x)) for x in top_ids[row].tolist()],
                "top_probs": [float(x) for x in top_probs[row].tolist()],
                "target_token": target_token,
                "target_rank": target_rank,
            }
        )
    return rows


def prediction_accuracy(logits: t.Tensor, target_token_ids: t.Tensor) -> float:
    """Top-1 accuracy for decoded logits."""

    if logits.shape[:-1] != target_token_ids.shape:
        raise ValueError("target_token_ids must match logits leading dimensions.")
    predictions = logits.argmax(dim=-1)
    return predictions.eq(target_token_ids).float().mean().item()


def fit_ridge_tuned_lens(
    residual_stream: t.Tensor,
    target_residual_stream: t.Tensor,
    *,
    ridge: float = 1e-2,
) -> tuple[t.Tensor, t.Tensor]:
    """Fit an affine map from early residuals to target residuals by ridge regression."""

    if residual_stream.ndim != 2 or target_residual_stream.ndim != 2:
        raise ValueError("ridge fitting expects rank-2 [example, d_model] tensors.")
    if residual_stream.shape[0] != target_residual_stream.shape[0]:
        raise ValueError("source and target tensors must have the same number of examples.")
    design = t.cat(
        [
            residual_stream.float(),
            t.ones(residual_stream.shape[0], 1, device=residual_stream.device),
        ],
        dim=1,
    )
    penalty = t.eye(design.shape[1], device=design.device)
    penalty[-1, -1] = 0.0
    solution = t.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ target_residual_stream.float(),
    )
    return solution[:-1], solution[-1]


def evaluate_lens_on_heldout(
    residual_stream: t.Tensor,
    lens_weight: t.Tensor,
    lens_bias: t.Tensor | None,
    unembedding: t.Tensor,
    target_token_ids: t.Tensor,
) -> LensAccuracyReport:
    """Evaluate ordinary logit lens and tuned lens on held-out target ids."""

    logit_logits = logit_lens(residual_stream, unembedding)
    tuned_logits = tuned_lens(residual_stream, lens_weight, lens_bias, unembedding)
    return lens_accuracy_report(logit_logits, tuned_logits, target_token_ids)


def lens_accuracy_report(
    logit_lens_logits: t.Tensor,
    tuned_lens_logits: t.Tensor,
    target_token_ids: t.Tensor,
) -> LensAccuracyReport:
    """Compare tuned-lens decoding against ordinary logit lens decoding."""

    logit_acc = prediction_accuracy(logit_lens_logits, target_token_ids)
    tuned_acc = prediction_accuracy(tuned_lens_logits, target_token_ids)
    improvement = tuned_acc - logit_acc
    return LensAccuracyReport(
        logit_lens_accuracy=logit_acc,
        tuned_lens_accuracy=tuned_acc,
        improvement=improvement,
        tuned_lens_improves=improvement > 0,
    )


def patchscope_prompt(template: PatchscopeTemplate, placeholder: str = "<ACT>") -> str:
    """Return a minimal prompt template for a Patchscope-style decode."""

    if template == "entity":
        return f"What entity is represented by {placeholder}?"
    if template == "next_token":
        return f"What token will {placeholder} become next?"
    if template == "fact":
        return f"What fact is stored in {placeholder}?"
    raise ValueError("unknown Patchscope template.")


def patchscope_accuracy_report(
    patchscope_logits: t.Tensor,
    text_only_logits: t.Tensor,
    target_answer_ids: t.Tensor,
) -> PatchscopeAccuracyReport:
    """Compare Patchscope answers against a text-only baseline."""

    patchscope_acc = prediction_accuracy(patchscope_logits, target_answer_ids)
    text_only_acc = prediction_accuracy(text_only_logits, target_answer_ids)
    improvement = patchscope_acc - text_only_acc
    return PatchscopeAccuracyReport(
        patchscope_accuracy=patchscope_acc,
        text_only_accuracy=text_only_acc,
        improvement=improvement,
        beats_text_only=improvement > 0,
    )


def replace_final_position_activation(
    activations: t.Tensor,
    source_activation: t.Tensor,
) -> t.Tensor:
    """Return activations with the final sequence position replaced by source_activation."""

    if activations.ndim != 3:
        raise ValueError("activations must have shape [batch, seq, d_model].")
    if source_activation.ndim != 1:
        raise ValueError("source_activation must have shape [d_model].")
    if activations.shape[0] != 1:
        raise ValueError("this teaching helper expects batch size 1.")
    if activations.shape[-1] != source_activation.shape[0]:
        raise ValueError("source_activation dimension must match activations d_model.")
    patched = activations.clone()
    patched[0, -1] = source_activation.to(device=activations.device, dtype=activations.dtype)
    return patched


def counterfactual_activation_report(
    original_logits: t.Tensor,
    patched_logits: t.Tensor,
) -> CounterfactualActivationReport:
    """Check whether a counterfactual activation changes the decoded answer."""

    if original_logits.ndim != 1 or patched_logits.ndim != 1:
        raise ValueError("original_logits and patched_logits must be rank-1 tensors.")
    original_answer = int(original_logits.argmax().item())
    patched_answer = int(patched_logits.argmax().item())
    return CounterfactualActivationReport(
        original_answer=original_answer,
        patched_answer=patched_answer,
        changed=original_answer != patched_answer,
    )


def random_activation_confidence_report(
    random_logits: t.Tensor,
    *,
    max_allowed_confidence: float = 0.6,
) -> RandomActivationConfidenceReport:
    """Check that random activations decode to low-confidence answers."""

    confidence = F.softmax(random_logits.float(), dim=-1).max(dim=-1).values
    mean_confidence = confidence.mean().item()
    max_confidence = confidence.max().item()
    return RandomActivationConfidenceReport(
        mean_confidence=mean_confidence,
        max_confidence=max_confidence,
        passes_low_confidence=max_confidence <= max_allowed_confidence,
    )


def patchscope_eval(
    patchscope_logits: t.Tensor,
    text_only_logits: t.Tensor,
    random_logits: t.Tensor,
    target_answer_ids: t.Tensor,
    *,
    max_allowed_random_confidence: float = 0.6,
) -> dict[str, object]:
    """Bundle Patchscope, text-only, and random-activation controls."""

    patchscope = patchscope_accuracy_report(
        patchscope_logits,
        text_only_logits,
        target_answer_ids,
    )
    random_control = random_activation_confidence_report(
        random_logits,
        max_allowed_confidence=max_allowed_random_confidence,
    )
    return {
        "patchscope_accuracy": patchscope.patchscope_accuracy,
        "text_only_accuracy": patchscope.text_only_accuracy,
        "improvement": patchscope.improvement,
        "beats_text_only": patchscope.beats_text_only,
        "random_mean_confidence": random_control.mean_confidence,
        "random_max_confidence": random_control.max_confidence,
        "random_passes_low_confidence": random_control.passes_low_confidence,
        "passes": patchscope.beats_text_only and random_control.passes_low_confidence,
    }
