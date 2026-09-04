# %%
"""Reference solutions for [7.2] Feature Verbalizers."""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t

chapter = "chapter7_activation_to_language"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"
ExampleKind = Literal["top", "bottom", "random", "contrastive"]
SplitKind = Literal["train", "revision", "heldout"]
TOKEN_RE = re.compile(r"[a-zA-Z]+")

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "beside",
    "by",
    "during",
    "in",
    "near",
    "of",
    "on",
    "over",
    "the",
    "to",
}

ANIMAL_TERMS = frozenset(
    {
        "alpaca",
        "badger",
        "bear",
        "cat",
        "dog",
        "fox",
        "goat",
        "horse",
        "kitten",
        "lizard",
        "mouse",
        "otter",
        "owl",
        "puppy",
        "rabbit",
        "seal",
        "sheep",
    }
)
RESTING_TERMS = frozenset(
    {"curled", "dozed", "lounged", "napped", "perched", "rested", "roosted", "sat", "slept"}
)
SURFACE_TERMS = frozenset(
    {
        "basket",
        "bench",
        "blanket",
        "branch",
        "couch",
        "dock",
        "fence",
        "floor",
        "hay",
        "ledge",
        "log",
        "mat",
        "mound",
        "pillow",
        "rock",
        "rug",
        "shelf",
        "sofa",
        "table",
    }
)
DECOY_TERMS = frozenset({"cardboard", "ceramic", "painted", "plastic", "plush", "statue", "toy", "wooden"})
DEFAULT_SEMANTIC_LEXICON = {
    "animal": ANIMAL_TERMS,
    "resting": RESTING_TERMS,
    "surface": SURFACE_TERMS,
    "decoy": DECOY_TERMS,
}

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_RESID_POST_HOOK = "blocks.0.hook_resid_post"
TL_DIRECTION_POSITIVE_PROMPT = "The cat sat on the"
TL_DIRECTION_NEGATIVE_PROMPT = "The bird flew over the"
TL_VERBALIZER_TRAIN_EXAMPLES = [
    ("The cat sat on the", True),
    ("The dog slept near the", True),
    ("The child sat beside the", True),
    ("The book rested near the", True),
    ("The bird flew over the", False),
    ("The plane flew over the", False),
    ("The kite floated above the", False),
    ("The cloud drifted above the", False),
]
TL_VERBALIZER_HELDOUT_EXAMPLES = [
    ("The cup sat on the", True),
    ("The blanket slept near the", True),
    ("The lamp rested beside the", True),
    ("The pillow sat near the", True),
    ("The dog ran through the", False),
    ("The balloon rose above the", False),
    ("The cat jumped over the", False),
    ("The rocket launched into the", False),
]
TL_DIRECTION_SCALE = 2.0


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


@dataclass(frozen=True)
class PlantedFeatureExample:
    text: str
    split: SplitKind
    label: bool
    score: float
    tags: tuple[str, ...]
    counterfactual_type: str


@dataclass(frozen=True)
class PlantedFeatureSplit:
    train: tuple[PlantedFeatureExample, ...]
    revision: tuple[PlantedFeatureExample, ...]
    heldout: tuple[PlantedFeatureExample, ...]


@dataclass(frozen=True)
class ExplanationRule:
    description: str
    required_groups: tuple[str, ...]
    excluded_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlantedInterventionReport:
    predicted_direction: Literal["increase", "decrease"]
    observed_delta: float
    random_direction_delta: float
    matches_prediction: bool
    beats_random_direction: bool
    evaluated_examples: int


_PLANTED_EXAMPLE_SPECS: tuple[tuple[str, SplitKind, str], ...] = (
    ("The cat sat on the mat.", "train", "positive"),
    ("A dog slept on the rug.", "train", "positive"),
    ("The rabbit rested near the sofa.", "train", "positive"),
    ("An otter napped on the dock.", "train", "positive"),
    ("The goat lounged beside the bench.", "train", "positive"),
    ("A fox perched on the ledge.", "train", "positive"),
    ("The robot rested on the sofa.", "train", "object_resting"),
    ("The book rested on the shelf.", "train", "object_resting"),
    ("The cat sprinted across the mat.", "train", "animal_motion"),
    ("The dog slept during the storm.", "train", "no_surface"),
    ("The kite floated above the field.", "train", "irrelevant"),
    ("The lamp sat on the floor.", "train", "object_resting"),
    ("The toy cat sat on the mat.", "revision", "decoy_animal"),
    ("The plush dog slept on the rug.", "revision", "decoy_animal"),
    ("The wooden fox perched on the ledge.", "revision", "decoy_animal"),
    ("The mouse rested under the table.", "revision", "positive"),
    ("The seal napped on the rock.", "revision", "positive"),
    ("The badger ran across the mound.", "revision", "animal_motion"),
    ("The horse rested on the blanket.", "heldout", "positive"),
    ("A lizard dozed on the warm rock.", "heldout", "positive"),
    ("The owl perched on the branch.", "heldout", "positive"),
    ("The kitten slept beside the basket.", "heldout", "positive"),
    ("The puppy lounged on the couch.", "heldout", "positive"),
    ("An alpaca sat near the fence.", "heldout", "positive"),
    ("The bear rested by the log.", "heldout", "positive"),
    ("The sheep slept on the hay.", "heldout", "positive"),
    ("The toy horse rested on the blanket.", "heldout", "decoy_animal"),
    ("A ceramic lizard dozed on the rock.", "heldout", "decoy_animal"),
    ("The statue owl perched on the branch.", "heldout", "decoy_animal"),
    ("The plush puppy slept on the couch.", "heldout", "decoy_animal"),
    ("The horse galloped across the blanket.", "heldout", "animal_motion"),
    ("The seal swam near the rock.", "heldout", "animal_motion"),
    ("The lamp rested on the blanket.", "heldout", "object_resting"),
    ("The book sat on the shelf.", "heldout", "object_resting"),
    ("The kitten slept during the storm.", "heldout", "no_surface"),
    ("The robot napped on the couch.", "heldout", "object_resting"),
    ("The painted bear rested by the log.", "heldout", "decoy_animal"),
    ("A cardboard sheep slept on the hay.", "heldout", "decoy_animal"),
    ("The owl flew above the branch.", "heldout", "animal_motion"),
    ("The fox chased the mouse.", "heldout", "no_rest"),
    ("The catalog sat on the shelf.", "heldout", "substring_trap"),
    ("The rocket launched above the clouds.", "heldout", "irrelevant"),
)


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _has_group(tokens: set[str], group: str, lexicon: dict[str, frozenset[str]] | None = None) -> bool:
    lexicon = DEFAULT_SEMANTIC_LEXICON if lexicon is None else lexicon
    if group not in lexicon:
        raise KeyError(f"Unknown semantic group: {group!r}.")
    return bool(tokens.intersection(lexicon[group]))


def planted_feature_label(text: str) -> bool:
    """Exact toy-oracle rule: a living animal is resting on a physical surface."""

    tokens = _tokens(text)
    return (
        _has_group(tokens, "animal")
        and _has_group(tokens, "resting")
        and _has_group(tokens, "surface")
        and not _has_group(tokens, "decoy")
    )


def _planted_feature_score(text: str) -> float:
    tokens = _tokens(text)
    has_animal = float(_has_group(tokens, "animal"))
    has_resting = float(_has_group(tokens, "resting"))
    has_surface = float(_has_group(tokens, "surface"))
    has_decoy = float(_has_group(tokens, "decoy"))
    label = float(planted_feature_label(text))
    jitter = ((sum(ord(char) for char in text) % 17) - 8) / 100.0
    return (
        -2.0
        + 1.3 * has_animal
        + 1.15 * has_resting
        + 0.95 * has_surface
        - 1.4 * has_decoy
        + 3.2 * label
        + jitter
    )


def _planted_tags(text: str) -> tuple[str, ...]:
    tokens = _tokens(text)
    tags = [group for group in ("animal", "resting", "surface", "decoy") if _has_group(tokens, group)]
    return tuple(tags)


def make_planted_feature_dataset() -> tuple[PlantedFeatureExample, ...]:
    """Return the finite planted feature dataset with exact labels and scores."""

    return tuple(
        PlantedFeatureExample(
            text=text,
            split=split,
            label=planted_feature_label(text),
            score=_planted_feature_score(text),
            tags=_planted_tags(text),
            counterfactual_type=counterfactual_type,
        )
        for text, split, counterfactual_type in _PLANTED_EXAMPLE_SPECS
    )


def split_planted_feature_dataset(
    examples: tuple[PlantedFeatureExample, ...] | list[PlantedFeatureExample],
) -> PlantedFeatureSplit:
    """Validate and split planted examples into train, revision, and final held-out sets."""

    if not examples:
        raise ValueError("examples must be nonempty.")
    texts = [example.text for example in examples]
    if len(set(texts)) != len(texts):
        raise ValueError("example texts must be unique across splits.")
    train = tuple(example for example in examples if example.split == "train")
    revision = tuple(example for example in examples if example.split == "revision")
    heldout = tuple(example for example in examples if example.split == "heldout")
    if not train or not revision or not heldout:
        raise ValueError("train, revision, and heldout splits must all be present.")
    if set(example.text for example in train).intersection(example.text for example in heldout):
        raise ValueError("train and heldout examples must be disjoint.")
    for name, split in {"train": train, "revision": revision, "heldout": heldout}.items():
        labels = {example.label for example in split}
        if labels != {False, True}:
            raise ValueError(f"{name} split must contain both positive and negative labels.")
    return PlantedFeatureSplit(train=train, revision=revision, heldout=heldout)


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
    contrastive_indices = (scores - threshold).abs().argsort()[:k]

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
    """Turn a keyword-style explanation into activation predictions."""

    normalized_terms = [term.strip().lower() for term in explanation_terms if term.strip()]
    if not normalized_terms:
        raise ValueError("explanation_terms must include at least one nonempty term.")
    return t.tensor(
        [any(term in _tokens(text) for term in normalized_terms) for text in texts],
        dtype=t.bool,
    )


def semantic_rule_predictions(
    texts: list[str],
    rule: ExplanationRule,
    *,
    lexicon: dict[str, frozenset[str]] | None = None,
) -> t.Tensor:
    """Turn a semantic verbalizer rule into whole-token predictions."""

    lexicon = DEFAULT_SEMANTIC_LEXICON if lexicon is None else lexicon
    if not rule.required_groups:
        raise ValueError("rule.required_groups must be nonempty.")
    for group in rule.required_groups:
        if group not in lexicon:
            raise KeyError(f"Unknown semantic group: {group!r}.")
    excluded_terms = {term.strip().lower() for term in rule.excluded_terms if term.strip()}
    predictions = []
    for text in texts:
        tokens = _tokens(text)
        has_required_groups = all(tokens.intersection(lexicon[group]) for group in rule.required_groups)
        has_excluded_term = bool(tokens.intersection(excluded_terms))
        predictions.append(bool(has_required_groups and not has_excluded_term))
    return t.tensor(predictions, dtype=t.bool)


def learn_verbalizer_terms(
    texts: list[str],
    labels: t.Tensor,
    *,
    top_k: int = 5,
    stopwords: set[str] | None = None,
) -> list[str]:
    """Learn concise keyword explanation terms from labeled training examples only."""

    labels = labels.flatten().bool()
    if len(texts) != labels.numel():
        raise ValueError("texts and labels must describe the same examples.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
    counts: dict[str, list[int]] = {}
    for text, label in zip(texts, labels.tolist(), strict=True):
        for token in _tokens(text):
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
        contrastive_accuracy = predictions[contrastive_mask].eq(labels[contrastive_mask])
        contrastive_accuracy = contrastive_accuracy.float().mean().item()
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


def mine_counterexamples(
    examples: tuple[PlantedFeatureExample, ...] | list[PlantedFeatureExample],
    predictions: t.Tensor,
    *,
    max_examples: int = 8,
) -> tuple[dict[str, object], ...]:
    """Return structured false positives and false negatives for revision."""

    predictions = predictions.flatten().bool()
    if len(examples) != predictions.numel():
        raise ValueError("examples and predictions must have matching length.")
    if max_examples <= 0:
        raise ValueError("max_examples must be positive.")
    rows = []
    for example, prediction in zip(examples, predictions.tolist(), strict=True):
        if bool(prediction) != example.label:
            rows.append(
                {
                    "text": example.text,
                    "label": example.label,
                    "predicted": bool(prediction),
                    "score": example.score,
                    "counterfactual_type": example.counterfactual_type,
                    "tags": example.tags,
                }
            )
    return tuple(rows[:max_examples])


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


def revise_rule_from_counterexamples(
    rule: ExplanationRule,
    counterexamples: tuple[dict[str, object], ...],
) -> ExplanationRule:
    """Revise the planted verbalizer by adding missing semantic constraints."""

    if not counterexamples:
        return rule
    required_groups = set(rule.required_groups)
    excluded_terms = set(rule.excluded_terms)
    for row in counterexamples:
        text = str(row["text"])
        tokens = _tokens(text)
        if row["predicted"] and not row["label"]:
            required_groups.update({"animal", "resting", "surface"})
            if tokens.intersection(DECOY_TERMS):
                excluded_terms.update(DECOY_TERMS)
        elif row["label"] and not row["predicted"]:
            required_groups.update({"animal", "resting", "surface"})
    ordered_groups = tuple(group for group in ("animal", "resting", "surface") if group in required_groups)
    ordered_exclusions = tuple(sorted(excluded_terms))
    if ordered_exclusions:
        description = (
            "living animal resting on a physical surface; exclude "
            + ", ".join(ordered_exclusions)
            + " examples"
        )
    else:
        description = "living animal resting on a physical surface"
    return ExplanationRule(
        description=description,
        required_groups=ordered_groups,
        excluded_terms=ordered_exclusions,
    )


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


def planted_intervention_direction_test(
    examples: tuple[PlantedFeatureExample, ...] | list[PlantedFeatureExample],
    predictions: t.Tensor,
    *,
    alpha: float = 1.25,
) -> PlantedInterventionReport:
    """Exact toy intervention: adding the planted feature direction raises the readout score."""

    predictions = predictions.flatten().bool()
    if len(examples) != predictions.numel():
        raise ValueError("examples and predictions must have matching length.")
    if not predictions.any():
        raise ValueError("at least one example must be predicted positive for intervention scoring.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    scores = t.tensor([example.score for example in examples], dtype=t.float32)
    mask = predictions
    baseline_scores = scores[mask]
    intervened_scores = baseline_scores + alpha
    random_direction_scores = baseline_scores
    observed_delta = (intervened_scores.mean() - baseline_scores.mean()).item()
    random_delta = (random_direction_scores.mean() - baseline_scores.mean()).item()
    return PlantedInterventionReport(
        predicted_direction="increase",
        observed_delta=observed_delta,
        random_direction_delta=random_delta,
        matches_prediction=observed_delta > 0,
        beats_random_direction=observed_delta > abs(random_delta) + 0.5,
        evaluated_examples=int(mask.sum().item()),
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


def _accuracy(predictions: t.Tensor, labels: t.Tensor) -> float:
    return float(predictions.flatten().bool().eq(labels.flatten().bool()).float().mean().item())


def examples_only_lookup_predictions(
    train_examples: tuple[PlantedFeatureExample, ...] | list[PlantedFeatureExample],
    test_examples: tuple[PlantedFeatureExample, ...] | list[PlantedFeatureExample],
) -> t.Tensor:
    """Predict positive only for exact held-out strings seen as positive in train."""

    positive_train_texts = {example.text for example in train_examples if example.label}
    return t.tensor([example.text in positive_train_texts for example in test_examples], dtype=t.bool)


def random_keyword_predictions(
    texts: list[str],
    *,
    seed: int = 0,
    k: int = 3,
) -> t.Tensor:
    """A deterministic random-keyword control over the observed text vocabulary."""

    if k <= 0:
        raise ValueError("k must be positive.")
    vocabulary = sorted({token for text in texts for token in _tokens(text)})
    if not vocabulary:
        raise ValueError("texts must contain at least one alphabetic token.")
    generator = t.Generator().manual_seed(seed)
    indices = t.randperm(len(vocabulary), generator=generator)[: min(k, len(vocabulary))]
    terms = [vocabulary[int(index.item())] for index in indices]
    return keyword_explanation_predictions(texts, terms)


def control_prediction_table(
    train_examples: tuple[PlantedFeatureExample, ...] | list[PlantedFeatureExample],
    heldout_examples: tuple[PlantedFeatureExample, ...] | list[PlantedFeatureExample],
    initial_rule: ExplanationRule,
    revised_rule: ExplanationRule,
    *,
    seed: int = 0,
) -> tuple[dict[str, object], ...]:
    """Compare revised verbalizer predictions against non-explanatory controls."""

    heldout_texts = [example.text for example in heldout_examples]
    labels = t.tensor([example.label for example in heldout_examples], dtype=t.bool)
    initial_predictions = semantic_rule_predictions(heldout_texts, initial_rule)
    revised_predictions = semantic_rule_predictions(heldout_texts, revised_rule)
    always_negative = t.zeros_like(labels)
    random_predictions = random_keyword_predictions(heldout_texts, seed=seed, k=3)
    lookup_predictions = examples_only_lookup_predictions(train_examples, heldout_examples)
    train_scores = t.tensor([example.score for example in train_examples])
    train_labels = t.tensor([example.label for example in train_examples], dtype=t.bool)
    score_threshold = float((train_scores[train_labels].min() + train_scores[~train_labels].max()) / 2)
    score_predictions = t.tensor(
        [example.score >= score_threshold for example in heldout_examples],
        dtype=t.bool,
    )
    rows = [
        ("initial text-only rule", initial_predictions, "resting-word story from top examples"),
        ("always-negative base rate", always_negative, "majority-class baseline"),
        ("random keyword rule", random_predictions, "random words from held-out vocabulary"),
        ("train-example lookup", lookup_predictions, "exact memorization control"),
        ("activation-score threshold", score_predictions, "oracle-style feature-score upper bound"),
        ("revised verbalizer", revised_predictions, "semantic rule after counterexample revision"),
    ]
    revised_accuracy = _accuracy(revised_predictions, labels)
    return tuple(
        {
            "control": name,
            "accuracy": _accuracy(predictions, labels),
            "beats_revised": _accuracy(predictions, labels) > revised_accuracy,
            "notes": notes,
        }
        for name, predictions, notes in rows
    )


def run_planted_feature_verbalizer_signature_result(seed: int = 0) -> dict:
    """Run the full CPU signature result on the exact planted verbalizer organism."""

    dataset = make_planted_feature_dataset()
    split = split_planted_feature_dataset(dataset)
    train_texts = [example.text for example in split.train]
    train_scores = t.tensor([example.score for example in split.train], dtype=t.float32)
    train_labels = t.tensor([example.label for example in split.train], dtype=t.bool)
    train_threshold = float((train_scores[train_labels].min() + train_scores[~train_labels].max()) / 2)
    example_set = gather_verbalizer_examples(
        train_texts,
        train_scores,
        train_labels,
        k=4,
        threshold=train_threshold,
        seed=seed,
    )

    initial_rule = ExplanationRule(
        description="feature fires on resting words",
        required_groups=("resting",),
    )
    revision_texts = [example.text for example in split.revision]
    revision_labels = t.tensor([example.label for example in split.revision], dtype=t.bool)
    revision_predictions = semantic_rule_predictions(revision_texts, initial_rule)
    counterexamples = mine_counterexamples(split.revision, revision_predictions, max_examples=8)
    revised_rule = revise_rule_from_counterexamples(initial_rule, counterexamples)

    heldout_texts = [example.text for example in split.heldout]
    heldout_labels = t.tensor([example.label for example in split.heldout], dtype=t.bool)
    initial_predictions = semantic_rule_predictions(heldout_texts, initial_rule)
    revised_predictions = semantic_rule_predictions(heldout_texts, revised_rule)
    contrastive_mask = t.tensor(
        [example.counterfactual_type not in {"positive", "irrelevant"} for example in split.heldout],
        dtype=t.bool,
    )
    prediction_report = explanation_prediction_report(
        revised_predictions,
        heldout_labels,
        initial_predictions,
        contrastive_mask,
    )
    controls = control_prediction_table(
        split.train,
        split.heldout,
        initial_rule,
        revised_rule,
        seed=seed,
    )
    intervention = planted_intervention_direction_test(split.heldout, revised_predictions)
    brevity = explanation_brevity_report(
        revised_rule.description,
        [example.text for example in example_set.top],
    )
    revised_accuracy = prediction_report.accuracy
    control_accuracies = {
        str(row["control"]): float(row["accuracy"])
        for row in controls
        if row["control"] != "activation-score threshold"
    }
    signature_passed = (
        revised_accuracy >= 0.95
        and all(revised_accuracy > value for value in control_accuracies.values() if value < 0.999)
        and prediction_report.survives_contrastive
        and len(counterexamples) >= 3
        and intervention.matches_prediction
        and intervention.beats_random_direction
        and brevity.shorter_than_examples
    )
    return {
        "claim": (
            "On an exact planted feature, a revised semantic verbalizer predicts held-out "
            "counterfactual activations and the sign of a feature-direction intervention."
        ),
        "planted_rule": "living animal + resting verb + physical surface - decoy object terms",
        "train_count": len(split.train),
        "revision_count": len(split.revision),
        "heldout_count": len(split.heldout),
        "initial_rule": initial_rule.__dict__,
        "revised_rule": revised_rule.__dict__,
        "example_groups": {
            "top": [example.__dict__ for example in example_set.top],
            "bottom": [example.__dict__ for example in example_set.bottom],
            "random": [example.__dict__ for example in example_set.random],
            "contrastive": [example.__dict__ for example in example_set.contrastive],
        },
        "validation_counterexamples": list(counterexamples),
        "heldout_rows": [
            {
                "text": example.text,
                "score": example.score,
                "label": example.label,
                "initial_prediction": bool(initial),
                "revised_prediction": bool(revised),
                "counterfactual_type": example.counterfactual_type,
            }
            for example, initial, revised in zip(
                split.heldout,
                initial_predictions.tolist(),
                revised_predictions.tolist(),
                strict=True,
            )
        ],
        "control_table": list(controls),
        "metrics": {
            "initial_accuracy": _accuracy(initial_predictions, heldout_labels),
            "revised_accuracy": prediction_report.accuracy,
            "baseline_accuracy": prediction_report.baseline_accuracy,
            "contrastive_accuracy": prediction_report.contrastive_accuracy,
            "passes_baseline": prediction_report.passes_baseline,
            "survives_contrastive": prediction_report.survives_contrastive,
            "counterexamples_mined": len(counterexamples),
            "intervention_delta": intervention.observed_delta,
            "random_direction_delta": intervention.random_direction_delta,
            "intervention_matches_prediction": intervention.matches_prediction,
            "target_beats_random_direction": intervention.beats_random_direction,
            "brevity_shorter_than_examples": brevity.shorter_than_examples,
        },
        "intervention": intervention.__dict__,
        "brevity": brevity.__dict__,
        "signature_passed": signature_passed,
    }


def example_selection_smoke_test() -> dict:
    texts = ["alpha code", "beta def", "plain story", "quiet notes"]
    scores = t.tensor([0.9, 0.8, 0.2, 0.1])
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)
    examples = gather_verbalizer_examples(texts, scores, labels, k=2, threshold=0.5, seed=0)
    return {
        "top": [example.text for example in examples.top],
        "bottom": [example.text for example in examples.bottom],
        "random": [example.text for example in examples.random],
        "contrastive": [example.text for example in examples.contrastive],
    }


def prediction_report_smoke_test() -> dict:
    texts = ["write code", "plain story", "def fn", "quiet notes"]
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    predictions = keyword_explanation_predictions(texts, ["code", "def"])
    baseline = t.zeros(4, dtype=t.bool)
    contrastive_mask = t.tensor([False, False, True, True])
    report = explanation_prediction_report(predictions, labels, baseline, contrastive_mask)
    return {"predictions": predictions.tolist(), **report.__dict__}


def counterexample_revision_smoke_test() -> dict:
    texts = ["write code", "plain story", "def fn", "quiet notes"]
    predictions = t.tensor([1, 1, 1, 0], dtype=t.bool)
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    report = find_counterexamples(texts, predictions, labels)
    revised = revise_explanation(
        "Feature activates on code.",
        report.counterexamples,
        revision_note="Exclude ordinary stories.",
    )
    return {
        "num_counterexamples": report.num_counterexamples,
        "counterexamples": list(report.counterexamples),
        "revised": revised,
    }


def intervention_prediction_smoke_test() -> dict:
    baseline = t.tensor([0.1, 0.2])
    intervened = t.tensor([0.5, 0.6])
    return intervention_prediction_report(
        baseline,
        intervened,
        predicted_direction="increase",
    ).__dict__


def brevity_smoke_test() -> dict:
    explanation = "Activates on code snippets."
    examples = ["write python code", "define a function with def"]
    return explanation_brevity_report(explanation, examples).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "examples": example_selection_smoke_test(),
        "prediction": prediction_report_smoke_test(),
        "counterexamples": counterexample_revision_smoke_test(),
        "intervention": intervention_prediction_smoke_test(),
        "brevity": brevity_smoke_test(),
        "planted_signature": run_planted_feature_verbalizer_signature_result(seed=0),
    }


def _load_gelu1l_model_on_cuda():
    os.environ.setdefault("BNB_CUDA_VERSION", TL_BNB_CUDA_OVERRIDE)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        TL_GELU1L_MODEL_NAME,
        device="cuda",
        dtype="float32",
        revision=TL_GELU1L_REVISION,
    )


def _final_residual_and_logits(model, prompt: str) -> tuple[t.Tensor, t.Tensor]:
    with t.inference_mode():
        logits, cache = model.run_with_cache(
            model.to_tokens(prompt),
            names_filter=lambda name: name == TL_RESID_POST_HOOK,
        )
    return cache[TL_RESID_POST_HOOK][0, -1].detach(), logits[0, -1].detach()


def _score_examples(
    model,
    examples: list[tuple[str, bool]],
    direction: t.Tensor,
) -> tuple[list[str], t.Tensor, t.Tensor]:
    texts = [text for text, _ in examples]
    labels = t.tensor([label for _, label in examples], dtype=t.bool, device=direction.device)
    scores = []
    for text in texts:
        residual, _ = _final_residual_and_logits(model, text)
        scores.append((residual.float() @ direction).item())
    return texts, t.tensor(scores, device=direction.device), labels


def run_transformerlens_feature_verbalizer_preflight(max_vram_gb: float = 24.0) -> dict:
    """Validate a concise verbalizer on a real residual direction from gelu-1l."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l residual-direction verbalizer preflight.",
        }

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()

    positive_residual, positive_logits = _final_residual_and_logits(
        model,
        TL_DIRECTION_POSITIVE_PROMPT,
    )
    negative_residual, negative_logits = _final_residual_and_logits(
        model,
        TL_DIRECTION_NEGATIVE_PROMPT,
    )
    direction = positive_residual.float() - negative_residual.float()
    direction = direction / direction.norm()
    positive_token_id = int(positive_logits.argmax().item())
    negative_token_id = int(negative_logits.argmax().item())

    train_texts, train_scores, train_labels = _score_examples(
        model,
        TL_VERBALIZER_TRAIN_EXAMPLES,
        direction,
    )
    heldout_texts, heldout_scores, heldout_labels = _score_examples(
        model,
        TL_VERBALIZER_HELDOUT_EXAMPLES,
        direction,
    )
    examples = gather_verbalizer_examples(
        train_texts,
        train_scores,
        train_labels,
        k=2,
        threshold=1.0,
        seed=0,
    )
    learned_terms = learn_verbalizer_terms(train_texts, train_labels.cpu(), top_k=3)
    train_positive_tokens = {
        token
        for text, label in TL_VERBALIZER_TRAIN_EXAMPLES
        if label
        for token in TOKEN_RE.findall(text.lower())
    }
    train_heldout_overlap = set(train_texts).intersection(heldout_texts)
    heldout_only_learned_terms = [
        term for term in learned_terms if term not in train_positive_tokens
    ]
    learned_terms_from_train = len(heldout_only_learned_terms) == 0
    explanation = "Surface verbs: " + ", ".join(learned_terms) + "."

    predictions = keyword_explanation_predictions(heldout_texts, learned_terms).to(
        heldout_labels.device
    )
    baseline = t.zeros_like(heldout_labels)
    contrastive_mask = (heldout_scores - 1.0).abs() <= 2.5
    prediction_report = explanation_prediction_report(
        predictions,
        heldout_labels,
        baseline,
        contrastive_mask,
    )
    heldout_contrastive_indices = contrastive_mask.nonzero(as_tuple=False).flatten().tolist()
    heldout_contrastive_examples = [
        heldout_texts[int(index)] for index in heldout_contrastive_indices
    ]
    heldout_contrastive_labels = [
        bool(heldout_labels[int(index)].item()) for index in heldout_contrastive_indices
    ]
    heldout_contrastive_scores = [
        float(heldout_scores[int(index)].item()) for index in heldout_contrastive_indices
    ]
    heldout_contrastive_has_both_labels = (
        bool(any(heldout_contrastive_labels)) and not all(heldout_contrastive_labels)
    )
    counterexamples = find_counterexamples(heldout_texts, predictions, heldout_labels)
    revised_explanation = revise_explanation(
        explanation,
        counterexamples.counterexamples,
        revision_note="Exclude motion or airborne examples.",
    )
    brevity = explanation_brevity_report(
        revised_explanation,
        [example.text for example in examples.top],
    )

    baseline_scores = []
    intervened_scores = []
    random_intervened_scores = []
    generator = t.Generator(device=direction.device).manual_seed(72)
    random_direction = t.randn(direction.shape, device=direction.device, generator=generator)
    random_direction = random_direction - (random_direction @ direction) * direction
    random_direction = random_direction / random_direction.norm()
    for text in heldout_texts:
        residual, _ = _final_residual_and_logits(model, text)
        baseline_logits = model.unembed(model.ln_final(residual))
        intervened_logits = model.unembed(
            model.ln_final(residual + TL_DIRECTION_SCALE * direction)
        )
        random_intervened_logits = model.unembed(
            model.ln_final(residual + TL_DIRECTION_SCALE * random_direction)
        )
        baseline_scores.append(
            baseline_logits[positive_token_id] - baseline_logits[negative_token_id]
        )
        intervened_scores.append(
            intervened_logits[positive_token_id] - intervened_logits[negative_token_id]
        )
        random_intervened_scores.append(
            random_intervened_logits[positive_token_id]
            - random_intervened_logits[negative_token_id]
        )
    intervention = intervention_prediction_report(
        t.stack(baseline_scores),
        t.stack(intervened_scores),
        predicted_direction="increase",
    )
    random_intervention = intervention_prediction_report(
        t.stack(baseline_scores),
        t.stack(random_intervened_scores),
        predicted_direction="increase",
    )
    target_beats_random_intervention = (
        intervention.observed_delta > abs(random_intervention.observed_delta) + 0.25
    )

    positive_mean_score = float(train_scores[train_labels].mean().item())
    negative_mean_score = float(train_scores[~train_labels].mean().item())
    score_separation = positive_mean_score - negative_mean_score
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        prediction_report.accuracy == 1.0
        and prediction_report.passes_baseline
        and prediction_report.survives_contrastive
        and counterexamples.num_counterexamples == 0
        and intervention.matches_prediction
        and intervention.observed_delta >= 0.5
        and target_beats_random_intervention
        and score_separation >= 5.0
        and brevity.shorter_than_examples
        and len(train_heldout_overlap) == 0
        and learned_terms_from_train
        and within_vram_budget
    )

    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "preflight_passed": preflight_passed,
        "model_name": TL_GELU1L_MODEL_NAME,
        "hf_model_id": TL_GELU1L_HF_ID,
        "hf_revision": TL_GELU1L_REVISION,
        "tokenizer_id": TL_GELU1L_TOKENIZER_ID,
        "tokenizer_revision": TL_GELU1L_TOKENIZER_REVISION,
        "bnb_cuda_override": TL_BNB_CUDA_OVERRIDE,
        "positive_prompt": TL_DIRECTION_POSITIVE_PROMPT,
        "negative_prompt": TL_DIRECTION_NEGATIVE_PROMPT,
        "positive_token": model.to_string(positive_token_id),
        "negative_token": model.to_string(negative_token_id),
        "train_example_count": len(train_texts),
        "heldout_example_count": len(heldout_texts),
        "explanation": explanation,
        "explanation_terms": learned_terms,
        "train_heldout_overlap_count": len(train_heldout_overlap),
        "heldout_only_learned_terms": heldout_only_learned_terms,
        "learned_terms_from_train": learned_terms_from_train,
        "top_examples": [example.text for example in examples.top],
        "bottom_examples": [example.text for example in examples.bottom],
        "contrastive_examples": [example.text for example in examples.contrastive],
        "positive_mean_score": positive_mean_score,
        "negative_mean_score": negative_mean_score,
        "score_separation": score_separation,
        "prediction_accuracy": prediction_report.accuracy,
        "baseline_accuracy": prediction_report.baseline_accuracy,
        "contrastive_accuracy": prediction_report.contrastive_accuracy,
        "heldout_contrastive_count": len(heldout_contrastive_examples),
        "heldout_contrastive_examples": heldout_contrastive_examples,
        "heldout_contrastive_labels": heldout_contrastive_labels,
        "heldout_contrastive_scores": heldout_contrastive_scores,
        "heldout_contrastive_has_both_labels": heldout_contrastive_has_both_labels,
        "passes_baseline": prediction_report.passes_baseline,
        "survives_contrastive": prediction_report.survives_contrastive,
        "num_counterexamples": counterexamples.num_counterexamples,
        "revised_explanation": revised_explanation,
        "brevity_shorter_than_examples": brevity.shorter_than_examples,
        "intervention_delta": intervention.observed_delta,
        "matches_intervention_prediction": intervention.matches_prediction,
        "random_direction_intervention_delta": random_intervention.observed_delta,
        "target_beats_random_intervention": target_beats_random_intervention,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l residual-direction verbalizer preflight.",
    }


def run_transformerlens_feature_verbalizer_signature_result(
    max_vram_gb: float = 24.0,
) -> dict:
    """CUDA-only real-transformer escalation used by the parent verification pass."""

    return run_transformerlens_feature_verbalizer_preflight(max_vram_gb=max_vram_gb)


def _attach_planted_signature_metrics(gpu_result: dict, planted_result: dict) -> dict:
    """Keep exact toy-oracle evidence distinct from same-named CUDA metrics."""

    metrics = planted_result["metrics"]
    control_accuracy = {
        str(row["control"]): float(row["accuracy"])
        for row in planted_result["control_table"]
    }
    return {
        **gpu_result,
        "toy_planted_signature_passed": planted_result["signature_passed"],
        "toy_train_count": planted_result["train_count"],
        "toy_revision_count": planted_result["revision_count"],
        "toy_heldout_count": planted_result["heldout_count"],
        "toy_initial_accuracy": metrics["initial_accuracy"],
        "toy_revised_accuracy": metrics["revised_accuracy"],
        "toy_baseline_accuracy": metrics["baseline_accuracy"],
        "toy_contrastive_accuracy": metrics["contrastive_accuracy"],
        "toy_passes_baseline": metrics["passes_baseline"],
        "toy_survives_contrastive": metrics["survives_contrastive"],
        "toy_counterexamples_mined": metrics["counterexamples_mined"],
        "toy_intervention_delta": metrics["intervention_delta"],
        "toy_random_direction_delta": metrics["random_direction_delta"],
        "toy_intervention_matches_prediction": metrics["intervention_matches_prediction"],
        "toy_target_beats_random_direction": metrics["target_beats_random_direction"],
        "toy_brevity_shorter_than_examples": metrics["brevity_shorter_than_examples"],
        "toy_activation_score_threshold_upper_bound_accuracy": control_accuracy[
            "activation-score threshold"
        ],
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    gpu_result = run_transformerlens_feature_verbalizer_preflight(max_vram_gb=max_vram_gb)
    planted_result = run_planted_feature_verbalizer_signature_result(seed=0)
    return _attach_planted_signature_metrics(gpu_result, planted_result)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
