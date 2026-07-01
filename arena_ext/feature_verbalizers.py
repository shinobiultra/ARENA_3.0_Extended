"""Feature verbalizer utilities for activation-to-language notebooks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

import torch as t


ExampleKind = Literal["top", "bottom", "random", "contrastive"]
TOKEN_RE = re.compile(r"[a-zA-Z]+")
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "beside",
    "in",
    "near",
    "of",
    "on",
    "over",
    "the",
    "to",
}


@dataclass(frozen=True)
class VerbalizerExample:
    text: str
    score: float
    label: bool
    kind: ExampleKind


@dataclass(frozen=True)
class VerbalizerExampleSet:
    top: tuple[VerbalizerExample, ...]
    bottom: tuple[VerbalizerExample, ...]
    random: tuple[VerbalizerExample, ...]
    contrastive: tuple[VerbalizerExample, ...]


@dataclass(frozen=True)
class ExplanationPredictionReport:
    accuracy: float
    baseline_accuracy: float
    contrastive_accuracy: float
    passes_baseline: bool
    survives_contrastive: bool


@dataclass(frozen=True)
class InterventionPredictionReport:
    predicted_direction: Literal["increase", "decrease"]
    observed_delta: float
    matches_prediction: bool


@dataclass(frozen=True)
class ExplanationBrevityReport:
    explanation_word_count: int
    examples_word_count: int
    shorter_than_examples: bool


@dataclass(frozen=True)
class CounterexampleReport:
    num_counterexamples: int
    counterexamples: tuple[str, ...]


def _make_examples(
    texts: list[str],
    scores: t.Tensor,
    labels: t.Tensor,
    indices: t.Tensor,
    kind: ExampleKind,
) -> tuple[VerbalizerExample, ...]:
    return tuple(
        VerbalizerExample(
            text=texts[int(index.item())],
            score=float(scores[int(index.item())].item()),
            label=bool(labels[int(index.item())].item()),
            kind=kind,
        )
        for index in indices
    )


def gather_verbalizer_examples(
    texts: list[str],
    scores: t.Tensor,
    labels: t.Tensor,
    *,
    k: int = 3,
    threshold: float | None = None,
    seed: int = 0,
) -> VerbalizerExampleSet:
    """Collect top, bottom, random, and contrastive near-miss examples."""

    scores = scores.flatten().float()
    labels = labels.flatten().bool()
    if len(texts) != scores.numel() or labels.shape != scores.shape:
        raise ValueError("texts, scores, and labels must describe the same examples.")
    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, scores.numel())
    top_indices = scores.topk(k=k).indices
    bottom_indices = (-scores).topk(k=k).indices

    generator = t.Generator().manual_seed(seed)
    random_indices = t.randperm(scores.numel(), generator=generator)[:k]

    if threshold is None:
        threshold = scores.mean().item()
    near_miss_order = (scores - threshold).abs().argsort()
    contrastive_indices = near_miss_order[:k]

    return VerbalizerExampleSet(
        top=_make_examples(texts, scores, labels, top_indices, "top"),
        bottom=_make_examples(texts, scores, labels, bottom_indices, "bottom"),
        random=_make_examples(texts, scores, labels, random_indices, "random"),
        contrastive=_make_examples(texts, scores, labels, contrastive_indices, "contrastive"),
    )


def keyword_explanation_predictions(
    texts: list[str],
    explanation_terms: list[str],
) -> t.Tensor:
    """Turn a simple keyword-style explanation into activation predictions."""

    normalized_terms = [term.lower() for term in explanation_terms if term]
    if not normalized_terms:
        raise ValueError("explanation_terms must include at least one nonempty term.")
    predictions = []
    for text in texts:
        tokens = set(TOKEN_RE.findall(text.lower()))
        predictions.append(any(term in tokens for term in normalized_terms))
    return t.tensor(predictions, dtype=t.bool)


def learn_verbalizer_terms(
    texts: list[str],
    labels: t.Tensor,
    *,
    top_k: int = 5,
    stopwords: set[str] | None = None,
) -> list[str]:
    """Learn keyword explanation terms from labeled examples only."""

    labels = labels.flatten().bool()
    if len(texts) != labels.numel():
        raise ValueError("texts and labels must describe the same examples.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
    counts: dict[str, list[int]] = {}
    for text, label in zip(texts, labels.tolist(), strict=True):
        for token in set(TOKEN_RE.findall(text.lower())):
            if token in stopwords:
                continue
            counts.setdefault(token, [0, 0])[int(bool(label))] += 1
    scored_terms = [
        (pos - neg, pos, -neg, token)
        for token, (neg, pos) in counts.items()
        if pos > 0 and pos > neg
    ]
    scored_terms.sort(reverse=True)
    return [token for _, _, _, token in scored_terms[:top_k]]


def explanation_prediction_report(
    predictions: t.Tensor,
    labels: t.Tensor,
    baseline_predictions: t.Tensor,
    contrastive_mask: t.Tensor,
) -> ExplanationPredictionReport:
    """Score explanation-derived predictions against baseline and contrastives."""

    predictions = predictions.flatten().bool()
    labels = labels.flatten().bool()
    baseline_predictions = baseline_predictions.flatten().bool()
    contrastive_mask = contrastive_mask.flatten().bool()
    matching_shapes = (
        predictions.shape == labels.shape == baseline_predictions.shape == contrastive_mask.shape
    )
    if not matching_shapes:
        raise ValueError("all prediction, label, and mask tensors must have matching shape.")

    accuracy = predictions.eq(labels).float().mean().item()
    baseline_accuracy = baseline_predictions.eq(labels).float().mean().item()
    if contrastive_mask.any():
        contrastive_accuracy = predictions[contrastive_mask].eq(labels[contrastive_mask]).float()
        contrastive_accuracy = contrastive_accuracy.mean().item()
    else:
        contrastive_accuracy = float("nan")

    return ExplanationPredictionReport(
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        contrastive_accuracy=contrastive_accuracy,
        passes_baseline=accuracy > baseline_accuracy,
        survives_contrastive=bool(
            contrastive_mask.any() and contrastive_accuracy > baseline_accuracy
        ),
    )


def find_counterexamples(
    texts: list[str],
    predictions: t.Tensor,
    labels: t.Tensor,
    *,
    max_examples: int = 3,
) -> CounterexampleReport:
    """Return examples where the explanation prediction is wrong."""

    predictions = predictions.flatten().bool()
    labels = labels.flatten().bool()
    if len(texts) != predictions.numel() or labels.shape != predictions.shape:
        raise ValueError("texts, predictions, and labels must describe the same examples.")
    if max_examples <= 0:
        raise ValueError("max_examples must be positive.")

    wrong_indices = predictions.ne(labels).nonzero(as_tuple=False).flatten()
    examples = tuple(texts[int(index.item())] for index in wrong_indices[:max_examples])
    return CounterexampleReport(
        num_counterexamples=int(wrong_indices.numel()),
        counterexamples=examples,
    )


def revise_explanation(
    explanation: str,
    counterexamples: tuple[str, ...],
    *,
    revision_note: str,
) -> str:
    """Append a short revision note grounded in counterexamples."""

    if not counterexamples:
        return explanation
    return f"{explanation} Revision: {revision_note}"


def intervention_prediction_report(
    baseline_scores: t.Tensor,
    intervened_scores: t.Tensor,
    *,
    predicted_direction: Literal["increase", "decrease"],
) -> InterventionPredictionReport:
    """Check whether an explanation predicts intervention direction."""

    baseline_mean = baseline_scores.float().mean().item()
    intervened_mean = intervened_scores.float().mean().item()
    observed_delta = intervened_mean - baseline_mean
    if predicted_direction == "increase":
        matches_prediction = observed_delta > 0
    elif predicted_direction == "decrease":
        matches_prediction = observed_delta < 0
    else:
        raise ValueError("predicted_direction must be 'increase' or 'decrease'.")
    return InterventionPredictionReport(
        predicted_direction=predicted_direction,
        observed_delta=observed_delta,
        matches_prediction=matches_prediction,
    )


def explanation_brevity_report(
    explanation: str,
    examples: list[str] | tuple[str, ...],
) -> ExplanationBrevityReport:
    """Check whether an explanation is shorter than an examples-only baseline."""

    explanation_word_count = len(explanation.split())
    examples_word_count = sum(len(example.split()) for example in examples)
    return ExplanationBrevityReport(
        explanation_word_count=explanation_word_count,
        examples_word_count=examples_word_count,
        shorter_than_examples=explanation_word_count < examples_word_count,
    )
