"""Mini Activation Oracle utilities for activation-to-language notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t
import torch.nn.functional as F

from arena_ext.activation_language import prediction_accuracy


QuestionKind = Literal[
    "token",
    "code",
    "question",
    "ioi",
    "refusal",
    "truth",
    "latent_state",
]


@dataclass(frozen=True)
class ActivationQuestionBatch:
    activations: t.Tensor
    question_ids: t.Tensor
    answer_ids: t.Tensor
    template_ids: t.Tensor
    questions: tuple[str, ...]


@dataclass(frozen=True)
class OracleComparisonReport:
    oracle_accuracy: float
    text_only_accuracy: float
    linear_probe_accuracy: float
    mlp_probe_accuracy: float
    sae_classifier_accuracy: float
    beats_text_only: bool
    beats_or_matches_probe: bool


@dataclass(frozen=True)
class OODGeneralizationReport:
    heldout_template_accuracy: float
    new_name_accuracy: float
    long_context_accuracy: float
    adversarial_accuracy: float
    passes_ood: bool


@dataclass(frozen=True)
class RandomActivationOracleReport:
    mean_confidence: float
    abstention_rate: float
    passes_graceful_failure: bool


@dataclass(frozen=True)
class ActivationPatchingOracleReport:
    original_answer: int
    patched_answer: int
    changed: bool


def default_activation_questions() -> tuple[str, ...]:
    """Return the standard mini Activation Oracle question bank."""

    return (
        "What token is represented?",
        "Is this activation from code?",
        "Is the prompt asking a question?",
        "Which of two names is the indirect object?",
        "Is the model about to refuse?",
        "Is this a truthful or false factual completion?",
        "Which synthetic latent variable is active?",
    )


def build_activation_question_batch(
    activations: t.Tensor,
    question_ids: t.Tensor,
    answer_ids: t.Tensor,
    template_ids: t.Tensor,
    questions: tuple[str, ...] | None = None,
) -> ActivationQuestionBatch:
    """Bundle activations with natural-language question and answer ids."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    expected_shape = (activations.shape[0],)
    if question_ids.shape != expected_shape:
        raise ValueError("question_ids must have shape (examples,).")
    if answer_ids.shape != expected_shape:
        raise ValueError("answer_ids must have shape (examples,).")
    if template_ids.shape != expected_shape:
        raise ValueError("template_ids must have shape (examples,).")
    if questions is None:
        questions = default_activation_questions()
    if len(questions) == 0:
        raise ValueError("questions must include at least one question.")
    question_ids = question_ids.long()
    answer_ids = answer_ids.long()
    template_ids = template_ids.long()
    if question_ids.numel() and (
        int(question_ids.min().item()) < 0
        or int(question_ids.max().item()) >= len(questions)
    ):
        raise ValueError("question_ids must index into the questions tuple.")
    return ActivationQuestionBatch(
        activations=activations,
        question_ids=question_ids,
        answer_ids=answer_ids,
        template_ids=template_ids,
        questions=questions,
    )


def oracle_comparison_report(
    oracle_logits: t.Tensor,
    text_only_logits: t.Tensor,
    linear_probe_logits: t.Tensor,
    mlp_probe_logits: t.Tensor,
    sae_classifier_logits: t.Tensor,
    answer_ids: t.Tensor,
) -> OracleComparisonReport:
    """Compare an Activation Oracle against text-only and probe baselines."""

    oracle_accuracy = prediction_accuracy(oracle_logits, answer_ids)
    text_only_accuracy = prediction_accuracy(text_only_logits, answer_ids)
    linear_accuracy = prediction_accuracy(linear_probe_logits, answer_ids)
    mlp_accuracy = prediction_accuracy(mlp_probe_logits, answer_ids)
    sae_accuracy = prediction_accuracy(sae_classifier_logits, answer_ids)
    best_probe = max(linear_accuracy, mlp_accuracy, sae_accuracy)
    return OracleComparisonReport(
        oracle_accuracy=oracle_accuracy,
        text_only_accuracy=text_only_accuracy,
        linear_probe_accuracy=linear_accuracy,
        mlp_probe_accuracy=mlp_accuracy,
        sae_classifier_accuracy=sae_accuracy,
        beats_text_only=oracle_accuracy > text_only_accuracy,
        beats_or_matches_probe=oracle_accuracy >= best_probe,
    )


def split_accuracy_by_template(
    logits: t.Tensor,
    answer_ids: t.Tensor,
    template_ids: t.Tensor,
) -> dict[int, float]:
    """Return answer accuracy for each template id."""

    if logits.shape[:-1] != answer_ids.shape or answer_ids.shape != template_ids.shape:
        raise ValueError("logits, answer_ids, and template_ids shapes are incompatible.")
    accuracies: dict[int, float] = {}
    for template_id in template_ids.unique(sorted=True):
        mask = template_ids.eq(template_id)
        accuracies[int(template_id.item())] = prediction_accuracy(logits[mask], answer_ids[mask])
    return accuracies


def ood_generalization_report(
    *,
    heldout_template_logits: t.Tensor,
    heldout_template_answers: t.Tensor,
    new_name_logits: t.Tensor,
    new_name_answers: t.Tensor,
    long_context_logits: t.Tensor,
    long_context_answers: t.Tensor,
    adversarial_logits: t.Tensor,
    adversarial_answers: t.Tensor,
    min_accuracy: float = 0.75,
) -> OODGeneralizationReport:
    """Evaluate Activation Oracle accuracy on OOD-style splits."""

    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must lie between 0 and 1.")
    heldout_template_accuracy = prediction_accuracy(
        heldout_template_logits,
        heldout_template_answers,
    )
    new_name_accuracy = prediction_accuracy(new_name_logits, new_name_answers)
    long_context_accuracy = prediction_accuracy(long_context_logits, long_context_answers)
    adversarial_accuracy = prediction_accuracy(adversarial_logits, adversarial_answers)
    passes = all(
        accuracy >= min_accuracy
        for accuracy in [
            heldout_template_accuracy,
            new_name_accuracy,
            long_context_accuracy,
            adversarial_accuracy,
        ]
    )
    return OODGeneralizationReport(
        heldout_template_accuracy=heldout_template_accuracy,
        new_name_accuracy=new_name_accuracy,
        long_context_accuracy=long_context_accuracy,
        adversarial_accuracy=adversarial_accuracy,
        passes_ood=passes,
    )


def random_activation_oracle_report(
    random_logits: t.Tensor,
    *,
    abstain_answer_id: int,
    min_abstention_rate: float = 0.5,
    max_mean_confidence: float = 0.6,
) -> RandomActivationOracleReport:
    """Check that the oracle fails gracefully on random activations."""

    if random_logits.ndim < 2:
        raise ValueError("random_logits must have shape (..., answer_classes).")
    if not 0 <= min_abstention_rate <= 1:
        raise ValueError("min_abstention_rate must lie between 0 and 1.")
    if not 0 <= max_mean_confidence <= 1:
        raise ValueError("max_mean_confidence must lie between 0 and 1.")
    if not 0 <= abstain_answer_id < random_logits.shape[-1]:
        raise ValueError("abstain_answer_id is out of range.")
    probs = F.softmax(random_logits.float(), dim=-1)
    confidence, predictions = probs.max(dim=-1)
    abstention_rate = predictions.eq(abstain_answer_id).float().mean().item()
    mean_confidence = confidence.mean().item()
    return RandomActivationOracleReport(
        mean_confidence=mean_confidence,
        abstention_rate=abstention_rate,
        passes_graceful_failure=(
            abstention_rate >= min_abstention_rate
            and mean_confidence <= max_mean_confidence
        ),
    )


def activation_patching_oracle_report(
    original_logits: t.Tensor,
    patched_logits: t.Tensor,
) -> ActivationPatchingOracleReport:
    """Check whether activation patching changes the oracle answer."""

    if original_logits.ndim != 1 or patched_logits.ndim != 1:
        raise ValueError("original_logits and patched_logits must be rank-1 tensors.")
    if original_logits.shape != patched_logits.shape:
        raise ValueError("original_logits and patched_logits must have matching shape.")
    original_answer = int(original_logits.argmax().item())
    patched_answer = int(patched_logits.argmax().item())
    return ActivationPatchingOracleReport(
        original_answer=original_answer,
        patched_answer=patched_answer,
        changed=original_answer != patched_answer,
    )
