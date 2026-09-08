# %%
"""Reference solutions for [7.2] Feature Verbalizers.

The learner-facing notebooks implement the same core functions inline. This module
exists so the focused tests and release checks have an importable reference.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch as t

MODEL_ID = "EleutherAI/pythia-70m"
MODEL_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
LAYER = 3
IS_TOKEN = " is"
ARE_TOKEN = " are"
PAIR_CAUSAL_NULL_SEED = 1729
PAIR_CAUSAL_NULL_COUNT = 256
Number = Literal["singular", "plural"]
Split = Literal["train", "validation", "heldout"]
Morphology = Literal["regular", "irregular", "nonce"]


@dataclass(frozen=True)
class DirectionDecoder:
    direction: t.Tensor
    threshold: float
    negative_mean: float
    positive_mean: float


@dataclass(frozen=True)
class ActivationCache:
    prompts: tuple[str, ...]
    activations: t.Tensor
    next_token_margins: t.Tensor


@dataclass(frozen=True)
class NumberExample:
    prompt: str
    noun: str
    number: Number
    split: Split
    morphology: Morphology
    semantic_class: str


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    description: str


@dataclass(frozen=True)
class BinaryEvaluation:
    accuracy: float
    baseline_accuracy: float
    subset_accuracy: float
    subset_baseline_accuracy: float
    beats_baseline: bool
    survives_subset: bool


@dataclass(frozen=True)
class HookIntervention:
    next_token_margins: t.Tensor
    hook_calls: int


@dataclass(frozen=True)
class LayerSweepCache:
    layer_names: tuple[str, ...]
    activations: tuple[t.Tensor, ...]


HYPOTHESES = (
    Hypothesis(
        "subject_number_plural",
        "The final subject noun is grammatically plural.",
    ),
    Hypothesis(
        "surface_suffix_s",
        "The final subject token ends with the letter s.",
    ),
    Hypothesis(
        "subject_is_person",
        "The subject denotes a person, regardless of number.",
    ),
    Hypothesis(
        "noun_has_six_letters",
        "The displayed noun token has at least six letters.",
    ),
)

TRAIN_PAIRS = (
    ("pilot", "pilots", "person"),
    ("doctor", "doctors", "person"),
    ("artist", "artists", "person"),
    ("farmer", "farmers", "person"),
    ("chef", "chefs", "person"),
    ("lawyer", "lawyers", "person"),
    ("student", "students", "person"),
    ("teacher", "teachers", "person"),
    ("driver", "drivers", "person"),
    ("writer", "writers", "person"),
    ("singer", "singers", "person"),
    ("guard", "guards", "person"),
)

VALIDATION_PAIRS = (
    ("baker", "bakers", "regular", "person"),
    ("dancer", "dancers", "regular", "person"),
    ("mouse", "mice", "irregular", "animal"),
    ("goose", "geese", "irregular", "animal"),
    ("wug", "wugs", "nonce", "nonce"),
    ("dax", "daxes", "nonce", "nonce"),
)

HELDOUT_PAIRS = (
    ("engineer", "engineers", "regular", "person"),
    ("musician", "musicians", "regular", "person"),
    ("scientist", "scientists", "regular", "person"),
    ("gardener", "gardeners", "regular", "person"),
    ("man", "men", "irregular", "person"),
    ("woman", "women", "irregular", "person"),
    ("child", "children", "irregular", "person"),
    ("person", "people", "irregular", "person"),
    ("blicket", "blickets", "nonce", "nonce"),
    ("zorp", "zorps", "nonce", "nonce"),
    ("plin", "plins", "nonce", "nonce"),
    ("nup", "nups", "nonce", "nonce"),
)


def _pair_examples(
    pairs: Sequence[tuple[str, str, str]],
    *,
    split: Split,
    morphology: Morphology,
) -> list[NumberExample]:
    examples: list[NumberExample] = []
    for singular, plural, semantic_class in pairs:
        examples.extend(
            [
                NumberExample(
                    f"The {singular}", singular, "singular", split, morphology, semantic_class
                ),
                NumberExample(
                    f"The {plural}", plural, "plural", split, morphology, semantic_class
                ),
            ]
        )
    return examples


def make_number_dataset() -> tuple[NumberExample, ...]:
    """Build paired train, validation, and held-out prompt families."""

    train = _pair_examples(TRAIN_PAIRS, split="train", morphology="regular")
    validation: list[NumberExample] = []
    for singular, plural, morphology, semantic_class in VALIDATION_PAIRS:
        validation.extend(
            _pair_examples(
                [(singular, plural, semantic_class)],
                split="validation",
                morphology=morphology,
            )
        )
    heldout: list[NumberExample] = []
    for singular, plural, morphology, semantic_class in HELDOUT_PAIRS:
        heldout.extend(
            _pair_examples(
                [(singular, plural, semantic_class)],
                split="heldout",
                morphology=morphology,
            )
        )
    examples = tuple(train + validation + heldout)
    validate_number_dataset(examples)
    return examples


def split_number_dataset(
    examples: Sequence[NumberExample],
) -> tuple[tuple[NumberExample, ...], tuple[NumberExample, ...], tuple[NumberExample, ...]]:
    return tuple(
        tuple(example for example in examples if example.split == split)
        for split in ("train", "validation", "heldout")
    )


def validate_number_dataset(examples: Sequence[NumberExample]) -> None:
    """Reject overlap and malformed number pairs before any activation fitting."""

    if not examples:
        raise ValueError("examples must be nonempty.")
    prompts = [example.prompt for example in examples]
    if len(set(prompts)) != len(prompts):
        raise ValueError("prompts must be unique.")
    train, validation, heldout = split_number_dataset(examples)
    if not train or not validation or not heldout:
        raise ValueError("train, validation, and heldout splits must all be nonempty.")
    split_nouns = [{example.noun for example in split} for split in (train, validation, heldout)]
    if (
        split_nouns[0] & split_nouns[1]
        or split_nouns[0] & split_nouns[2]
        or split_nouns[1] & split_nouns[2]
    ):
        raise ValueError("noun vocabulary must be disjoint across splits.")
    for name, split in zip(
        ("train", "validation", "heldout"),
        (train, validation, heldout),
        strict=True,
    ):
        counts = {
            number: sum(example.number == number for example in split)
            for number in ("singular", "plural")
        }
        if counts["singular"] != counts["plural"]:
            raise ValueError(f"{name} must contain balanced singular/plural pairs.")


def number_labels(examples: Sequence[NumberExample]) -> t.Tensor:
    return t.tensor([example.number == "plural" for example in examples], dtype=t.bool)


def fit_mean_difference_decoder(activations: t.Tensor, labels: t.Tensor) -> DirectionDecoder:
    """Fit a unit mean-difference direction using labeled training activations only."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape [examples, d_model].")
    labels = labels.flatten().bool()
    if activations.shape[0] != labels.numel():
        raise ValueError("activations and labels must contain the same number of examples.")
    if not labels.any() or labels.all():
        raise ValueError("labels must contain both classes.")
    activations = activations.float()
    negative_mean_vector = activations[~labels].mean(dim=0)
    positive_mean_vector = activations[labels].mean(dim=0)
    difference = positive_mean_vector - negative_mean_vector
    norm = difference.norm()
    if not t.isfinite(norm) or norm <= 1e-8:
        raise ValueError("class means do not define a finite nonzero direction.")
    direction = difference / norm
    negative_mean = float(negative_mean_vector @ direction)
    positive_mean = float(positive_mean_vector @ direction)
    return DirectionDecoder(
        direction=direction,
        threshold=(negative_mean + positive_mean) / 2,
        negative_mean=negative_mean,
        positive_mean=positive_mean,
    )


def direction_scores(activations: t.Tensor, decoder: DirectionDecoder) -> t.Tensor:
    if activations.ndim != 2 or activations.shape[1] != decoder.direction.numel():
        raise ValueError("activations must be [examples, d_model] and match the decoder.")
    return activations.float() @ decoder.direction.float()


def direction_predictions(activations: t.Tensor, decoder: DirectionDecoder) -> t.Tensor:
    return direction_scores(activations, decoder) >= decoder.threshold


def replace_direction_component(
    activations: t.Tensor,
    direction: t.Tensor,
    target_scores: t.Tensor | float,
) -> t.Tensor:
    """Replace only the component parallel to a unit direction."""

    if (
        activations.ndim != 2
        or direction.ndim != 1
        or activations.shape[1] != direction.numel()
    ):
        raise ValueError("expected activations [batch, d_model] and matching direction [d_model].")
    direction = direction.to(device=activations.device, dtype=activations.dtype)
    expected_norm = t.tensor(1.0, device=direction.device, dtype=direction.dtype)
    if not t.allclose(direction.norm(), expected_norm, atol=1e-5):
        raise ValueError("direction must have unit norm.")
    targets = t.as_tensor(target_scores, device=activations.device, dtype=activations.dtype)
    targets = targets.expand(activations.shape[0])
    current_scores = activations @ direction
    return activations + (targets - current_scores).unsqueeze(-1) * direction


def _single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{text!r} must tokenize to exactly one token, got {token_ids}.")
    return int(token_ids[0])


def load_pinned_model(device: str | t.device = "cpu"):
    """Load the pinned checkpoint from the local HF cache onto one explicit device."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=t.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def cache_final_token_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    layer: int = LAYER,
) -> ActivationCache:
    """Cache real final-token residual activations and the downstream are-is margin."""

    if not prompts:
        raise ValueError("prompts must be nonempty.")
    if layer < 0 or layer >= len(model.gpt_neox.layers):
        raise ValueError("layer is outside the model's transformer blocks.")
    device = next(model.parameters()).device
    encoded = tokenizer(list(prompts), padding=True, return_tensors="pt").to(device)
    with t.inference_mode():
        output = model(**encoded, output_hidden_states=True)
    activations = output.hidden_states[layer + 1][:, -1, :].detach().cpu().float()
    logits = output.logits[:, -1, :].detach().cpu().float()
    is_id = _single_token_id(tokenizer, IS_TOKEN)
    are_id = _single_token_id(tokenizer, ARE_TOKEN)
    margins = logits[:, are_id] - logits[:, is_id]
    return ActivationCache(tuple(prompts), activations, margins)


def cache_layer_sweep(
    model,
    tokenizer,
    prompts: Sequence[str],
) -> LayerSweepCache:
    """Cache the embedding residual and every post-block final-token residual."""

    if not prompts:
        raise ValueError("prompts must be nonempty.")
    device = next(model.parameters()).device
    encoded = tokenizer(list(prompts), padding=True, return_tensors="pt").to(device)
    with t.inference_mode():
        output = model(**encoded, output_hidden_states=True)
    activations = tuple(
        hidden[:, -1, :].detach().cpu().float() for hidden in output.hidden_states
    )
    expected_count = len(model.gpt_neox.layers) + 1
    if len(activations) != expected_count:
        raise AssertionError("hidden-state count must equal embedding plus transformer blocks")
    names = ("embedding",) + tuple(f"block {index}" for index in range(len(activations) - 1))
    return LayerSweepCache(names, activations)


def layer_sweep_accuracies(
    train_cache: LayerSweepCache,
    train_labels: t.Tensor,
    test_cache: LayerSweepCache,
    test_labels: t.Tensor,
) -> tuple[float, ...]:
    """Fit on training activations independently at every depth and test held out."""

    if train_cache.layer_names != test_cache.layer_names:
        raise ValueError("train and test sweeps must contain the same layers")
    return tuple(
        float(direction_predictions(
            test_activations, fit_mean_difference_decoder(train_activations, train_labels)
        ).eq(test_labels.bool()).float().mean())
        for train_activations, test_activations in zip(
            train_cache.activations, test_cache.activations, strict=True
        )
    )


def hypothesis_predictions(
    examples: Sequence[NumberExample],
    hypothesis: Hypothesis,
) -> t.Tensor:
    """Execute a verbalizer as a semantic predicate, never as an answer token."""

    if hypothesis.hypothesis_id == "subject_number_plural":
        values = [example.number == "plural" for example in examples]
    elif hypothesis.hypothesis_id == "surface_suffix_s":
        values = [example.noun.lower().endswith("s") for example in examples]
    elif hypothesis.hypothesis_id == "subject_is_person":
        values = [example.semantic_class == "person" for example in examples]
    elif hypothesis.hypothesis_id == "noun_has_six_letters":
        values = [len(example.noun) >= 6 for example in examples]
    else:
        raise KeyError(f"Unknown hypothesis id: {hypothesis.hypothesis_id!r}.")
    return t.tensor(values, dtype=t.bool)


def score_hypotheses(
    examples: Sequence[NumberExample],
    activation_predictions: t.Tensor,
    candidates: Sequence[Hypothesis] = HYPOTHESES,
    *,
    seed: int = 0,
) -> tuple[dict[str, object], ...]:
    """Score candidate meaning against activations after randomizing display order."""

    activation_predictions = activation_predictions.flatten().bool()
    if len(examples) != activation_predictions.numel():
        raise ValueError("examples and activation_predictions must align.")
    if not candidates or len({candidate.hypothesis_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate ids must be nonempty and unique.")
    randomized = list(candidates)
    random.Random(seed).shuffle(randomized)
    rows = []
    for display_index, candidate in enumerate(randomized, start=1):
        predictions = hypothesis_predictions(examples, candidate)
        rows.append(
            {
                "display_index": display_index,
                "hypothesis_id": candidate.hypothesis_id,
                "description": candidate.description,
                "agreement": float(predictions.eq(activation_predictions).float().mean()),
                "word_count": len(candidate.description.rstrip(".").split()),
                "counterexample_count": int(predictions.ne(activation_predictions).sum()),
            }
        )
    return tuple(rows)


def select_hypothesis(rows: Sequence[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("rows must be nonempty.")
    selected = max(
        rows,
        key=lambda row: (
            float(row["agreement"]),
            -int(row["word_count"]),
            str(row["hypothesis_id"]),
        ),
    )
    return str(selected["hypothesis_id"])


def rank_feature_examples(
    examples: Sequence[NumberExample],
    scores: t.Tensor,
    threshold: float,
    *,
    n: int = 2,
    seed: int = 0,
) -> tuple[dict[str, object], ...]:
    """Return top, bottom, random, and near-threshold evidence rows."""

    scores = scores.flatten().float()
    if len(examples) != scores.numel() or not 0 < n <= len(examples):
        raise ValueError("scores must align with examples and n must select available rows")
    order = t.argsort(scores)
    groups = {
        "bottom": order[:n].tolist(),
        "top": order[-n:].flip(0).tolist(),
        "near miss": t.argsort((scores - threshold).abs())[:n].tolist(),
        "random": random.Random(seed).sample(range(len(examples)), n),
    }
    rows = []
    for group, indices in groups.items():
        for index in indices:
            example = examples[index]
            rows.append(
                {
                    "group": group,
                    "prompt": example.prompt,
                    "score": float(scores[index]),
                    "predicted_side": "positive" if scores[index] >= threshold else "negative",
                    "morphology": example.morphology,
                }
            )
    return tuple(rows)


def mine_counterexamples(
    examples: Sequence[NumberExample],
    activation_predictions: t.Tensor,
    hypothesis: Hypothesis,
) -> tuple[dict[str, object], ...]:
    """Find rows where an executable explanation disagrees with the feature side."""

    activation_predictions = activation_predictions.flatten().bool()
    if len(examples) != activation_predictions.numel():
        raise ValueError("examples and activation_predictions must align")
    explanation_predictions = hypothesis_predictions(examples, hypothesis)
    return tuple(
        {
            "prompt": example.prompt,
            "feature_side": "positive" if feature_value else "negative",
            "explanation_prediction": "positive" if explanation_value else "negative",
            "morphology": example.morphology,
        }
        for example, feature_value, explanation_value in zip(
            examples,
            activation_predictions,
            explanation_predictions,
            strict=True,
        )
        if bool(feature_value) != bool(explanation_value)
    )


def brevity_frontier(rows: Sequence[dict[str, object]]) -> tuple[str, ...]:
    """Return explanations not dominated in both agreement and word count."""

    frontier = []
    for row in rows:
        dominated = any(
            float(other["agreement"]) >= float(row["agreement"])
            and int(other["word_count"]) <= int(row["word_count"])
            and (
                float(other["agreement"]) > float(row["agreement"])
                or int(other["word_count"]) < int(row["word_count"])
            )
            for other in rows
        )
        if not dominated:
            frontier.append(str(row["hypothesis_id"]))
    return tuple(sorted(frontier))


def evaluate_binary_predictions(
    predictions: t.Tensor,
    labels: t.Tensor,
    baseline_predictions: t.Tensor,
    subset_mask: t.Tensor,
) -> BinaryEvaluation:
    """Compare method and baseline on both the full set and the exact same subset."""

    predictions = predictions.flatten().bool()
    labels = labels.flatten().bool()
    baseline_predictions = baseline_predictions.flatten().bool()
    subset_mask = subset_mask.flatten().bool()
    if not (predictions.shape == labels.shape == baseline_predictions.shape == subset_mask.shape):
        raise ValueError("predictions, labels, baseline, and subset mask must have matching shape.")
    if not subset_mask.any():
        raise ValueError("subset_mask must select at least one example.")
    accuracy = float(predictions.eq(labels).float().mean())
    baseline_accuracy = float(baseline_predictions.eq(labels).float().mean())
    subset_accuracy = float(predictions[subset_mask].eq(labels[subset_mask]).float().mean())
    subset_baseline_accuracy = float(
        baseline_predictions[subset_mask].eq(labels[subset_mask]).float().mean()
    )
    return BinaryEvaluation(
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        subset_accuracy=subset_accuracy,
        subset_baseline_accuracy=subset_baseline_accuracy,
        beats_baseline=accuracy > baseline_accuracy,
        survives_subset=subset_accuracy > subset_baseline_accuracy,
    )


def final_noun_token_ids(tokenizer, examples: Sequence[NumberExample]) -> tuple[int, ...]:
    """Return each noun's final tokenizer id without pretending all nouns are one token."""

    token_ids: list[int] = []
    for example in examples:
        encoded = tokenizer.encode(" " + example.noun, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"noun {example.noun!r} produced no tokens")
        token_ids.append(int(encoded[-1]))
    return tuple(token_ids)


def seen_plural_lookup_predictions(
    tokenizer,
    train_examples: Sequence[NumberExample],
    test_examples: Sequence[NumberExample],
) -> t.Tensor:
    """Text-only control that memorizes final plural tokenizer ids from training."""

    train_ids = final_noun_token_ids(tokenizer, train_examples)
    seen_plural_ids = {
        token_id
        for token_id, example in zip(train_ids, train_examples, strict=True)
        if example.number == "plural"
    }
    return t.tensor(
        [token_id in seen_plural_ids for token_id in final_noun_token_ids(tokenizer, test_examples)],
        dtype=t.bool,
    )


def orthographic_feature_matrix(examples: Sequence[NumberExample]) -> t.Tensor:
    """Encode a deliberately simple text-only spelling baseline."""

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    rows = []
    for example in examples:
        noun = example.noun.lower()
        rows.append(
            [
                1.0,
                len(noun) / 10.0,
                float(noun.endswith("s")),
                float(noun.endswith("en")),
                float(noun.endswith("ice")),
                float(noun.endswith("ese")),
                *[noun.count(letter) / max(len(noun), 1) for letter in alphabet],
            ]
        )
    return t.tensor(rows, dtype=t.float32)


def orthographic_ridge_predictions(
    train_examples: Sequence[NumberExample],
    train_labels: t.Tensor,
    test_examples: Sequence[NumberExample],
    *,
    ridge: float = 1e-2,
) -> t.Tensor:
    """Fit a train-only ridge classifier on spelling features, then threshold at 0.5."""

    train_features = orthographic_feature_matrix(train_examples)
    test_features = orthographic_feature_matrix(test_examples)
    labels = train_labels.flatten().float()
    if train_features.shape[0] != labels.numel() or ridge <= 0:
        raise ValueError("training labels must align and ridge must be positive")
    identity = t.eye(train_features.shape[1], dtype=train_features.dtype)
    weights = t.linalg.solve(
        train_features.T @ train_features + ridge * identity,
        train_features.T @ labels,
    )
    return test_features @ weights >= 0.5


def paired_swap_decoder_accuracies(
    train_activations: t.Tensor,
    train_labels: t.Tensor,
    test_activations: t.Tensor,
    test_labels: t.Tensor,
) -> tuple[float, ...]:
    """Enumerate the exact null that swaps labels only within matched noun pairs."""

    if train_activations.shape[0] != train_labels.numel() or train_labels.numel() % 2:
        raise ValueError("training rows must form consecutive singular/plural pairs")
    paired_labels = train_labels.bool().view(-1, 2)
    if not t.all(paired_labels.sum(dim=1) == 1):
        raise ValueError("each training pair must contain one singular and one plural label")
    pair_count = paired_labels.shape[0]
    if pair_count > 20:
        raise ValueError("exact enumeration is limited to at most 20 pairs")

    accuracies = []
    for pattern in range(2**pair_count):
        swapped = paired_labels.clone()
        swap_mask = t.tensor(
            [(pattern >> pair_index) & 1 for pair_index in range(pair_count)],
            dtype=t.bool,
            device=train_labels.device,
        )
        swapped[swap_mask] = swapped[swap_mask].flip(dims=(1,))
        try:
            decoder = fit_mean_difference_decoder(train_activations, swapped.flatten())
            predictions = direction_predictions(test_activations, decoder)
        except ValueError as error:
            if "nonzero direction" not in str(error):
                raise
            predictions = t.zeros_like(test_labels, dtype=t.bool)
        accuracies.append(float(predictions.eq(test_labels.bool()).float().mean()))
    return tuple(accuracies)


def paired_swap_decoder_directions(
    train_activations: t.Tensor,
    train_labels: t.Tensor,
    *,
    count: int = PAIR_CAUSAL_NULL_COUNT,
    seed: int = PAIR_CAUSAL_NULL_SEED,
) -> tuple[tuple[int, ...], t.Tensor]:
    """Precommit decoder directions induced by within-pair label swaps."""

    if train_activations.shape[0] != train_labels.numel() or train_labels.numel() % 2:
        raise ValueError("training rows must form consecutive singular/plural pairs")
    paired_labels = train_labels.bool().view(-1, 2)
    if not t.all(paired_labels.sum(dim=1) == 1):
        raise ValueError("each training pair must contain one singular and one plural label")
    pair_count = paired_labels.shape[0]
    available_patterns = 2**pair_count - 2
    if not 0 < count <= available_patterns:
        raise ValueError("count must fit the non-target, non-inverted swap patterns")
    patterns = tuple(random.Random(seed).sample(range(1, 2**pair_count - 1), count))
    directions = []
    for pattern in patterns:
        swapped = paired_labels.clone()
        swap_mask = t.tensor(
            [(pattern >> pair_index) & 1 for pair_index in range(pair_count)],
            dtype=t.bool,
            device=train_labels.device,
        )
        swapped[swap_mask] = swapped[swap_mask].flip(dims=(1,))
        directions.append(
            fit_mean_difference_decoder(train_activations, swapped.flatten()).direction
        )
    bank = t.stack(directions)
    if not t.allclose(bank.norm(dim=-1), bank.new_ones(count), rtol=1e-5, atol=1e-5):
        raise AssertionError("every paired-null decoder direction must have unit norm")
    return patterns, bank


def orthogonal_random_direction(direction: t.Tensor, *, seed: int = 0) -> t.Tensor:
    generator = t.Generator(device=direction.device).manual_seed(seed)
    control = t.randn(direction.shape, generator=generator, device=direction.device)
    control = control - (control @ direction) * direction
    return control / control.norm()


def orthogonal_random_directions(
    direction: t.Tensor, *, seeds: Sequence[int] = tuple(range(64))
) -> t.Tensor:
    """Construct a distribution of unit directions orthogonal to the target."""

    if not seeds:
        raise ValueError("seeds must be nonempty")
    controls = t.stack(
        [orthogonal_random_direction(direction, seed=int(seed)) for seed in seeds]
    )
    if (controls @ direction).abs().max() > 1e-5:
        raise AssertionError("orthogonal controls are not numerically orthogonal")
    if (controls.norm(dim=-1) - 1).abs().max() > 1e-5:
        raise AssertionError("orthogonal controls must have unit norm")
    return controls


def run_residual_edit(
    model,
    tokenizer,
    prompts: Sequence[str],
    edit_vectors: t.Tensor,
    *,
    layer: int = LAYER,
) -> HookIntervention:
    """Add per-example edits inside a real transformer block forward hook."""

    if edit_vectors.ndim != 2 or edit_vectors.shape[0] != len(prompts):
        raise ValueError("edit_vectors must have shape [len(prompts), d_model].")
    device = next(model.parameters()).device
    encoded = tokenizer(list(prompts), padding=True, return_tensors="pt").to(device)
    edits = edit_vectors.to(device=device, dtype=next(model.parameters()).dtype)
    hook_calls = 0

    def edit_hook(_module, _inputs, output):
        nonlocal hook_calls
        hook_calls += 1
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != edits.shape[0] or hidden.shape[-1] != edits.shape[-1]:
            raise ValueError("hook output and edit vectors do not align.")
        edited = hidden.clone()
        edited[:, -1, :] = edited[:, -1, :] + edits
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    handle = model.gpt_neox.layers[layer].register_forward_hook(edit_hook)
    try:
        with t.inference_mode():
            logits = model(**encoded).logits[:, -1, :].detach().cpu().float()
    finally:
        handle.remove()
    is_id = _single_token_id(tokenizer, IS_TOKEN)
    are_id = _single_token_id(tokenizer, ARE_TOKEN)
    return HookIntervention(logits[:, are_id] - logits[:, is_id], hook_calls)


def run_orthogonal_control_distribution(
    model,
    tokenizer,
    prompts: Sequence[str],
    replacement_coefficients: t.Tensor,
    target_direction: t.Tensor,
    baseline_margin: float,
    *,
    seeds: Sequence[int] = tuple(range(64)),
) -> dict[str, object]:
    """Apply many matched-coefficient orthogonal edits in one real model forward."""

    if replacement_coefficients.shape != (len(prompts),):
        raise ValueError("replacement_coefficients must have one value per prompt")
    controls = orthogonal_random_directions(target_direction, seeds=seeds)
    edits = controls[:, None, :] * replacement_coefficients[None, :, None]
    expanded_prompts = tuple(prompt for _ in seeds for prompt in prompts)
    intervention = run_residual_edit(
        model,
        tokenizer,
        expanded_prompts,
        edits.reshape(-1, edits.shape[-1]),
    )
    mean_margins = intervention.next_token_margins.view(len(seeds), len(prompts)).mean(dim=1)
    deltas = mean_margins - baseline_margin
    return {
        "deltas": tuple(float(value) for value in deltas),
        "absolute_p95": float(t.quantile(deltas.abs(), 0.95)),
        "absolute_max": float(deltas.abs().max()),
        "orthogonality_max_abs_dot": float((controls @ target_direction).abs().max()),
        "norm_error_max": float((controls.norm(dim=-1) - 1).abs().max()),
        "hook_calls": intervention.hook_calls,
    }


def run_direction_control_distribution(
    model,
    tokenizer,
    prompts: Sequence[str],
    replacement_coefficients: t.Tensor,
    control_directions: t.Tensor,
    baseline_margin: float,
    *,
    effect_sign: float,
    chunk_size: int = 32,
    layer: int = LAYER,
) -> dict[str, object]:
    """Run a bank of same-coefficient directions through the real block hook."""

    coefficients = replacement_coefficients.flatten().float()
    if coefficients.shape != (len(prompts),):
        raise ValueError("replacement_coefficients must have one value per prompt")
    expected_dim = int(model.config.hidden_size)
    if control_directions.ndim != 2 or control_directions.shape[1] != expected_dim:
        raise ValueError("control_directions must have shape [controls, d_model]")
    if not len(control_directions) or chunk_size <= 0 or effect_sign not in (-1.0, 1.0):
        raise ValueError("controls, positive chunk_size, and effect_sign +/-1 are required")
    if not t.allclose(
        control_directions.float().norm(dim=-1),
        control_directions.new_ones(len(control_directions)),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError("control directions must be unit norm")

    effects: list[float] = []
    hook_calls: list[int] = []
    for start in range(0, len(control_directions), chunk_size):
        directions = control_directions[start : start + chunk_size]
        edits = directions[:, None, :] * coefficients[None, :, None]
        expanded_prompts = tuple(prompt for _ in range(len(directions)) for prompt in prompts)
        intervention = run_residual_edit(
            model,
            tokenizer,
            expanded_prompts,
            edits.reshape(-1, edits.shape[-1]),
            layer=layer,
        )
        margins = intervention.next_token_margins.view(len(directions), len(prompts)).mean(dim=1)
        effects.extend(float(value) for value in effect_sign * (margins - baseline_margin))
        hook_calls.append(intervention.hook_calls)
    effect_tensor = t.tensor(effects)
    return {
        "effects": tuple(effects),
        "mean": float(effect_tensor.mean()),
        "p95": float(t.quantile(effect_tensor, 0.95)),
        "max": float(effect_tensor.max()),
        "hook_calls": tuple(hook_calls),
        "coefficient_checksum": float(coefficients.sum()),
    }


def _accuracy(predictions: t.Tensor, labels: t.Tensor) -> float:
    return float(predictions.bool().eq(labels.bool()).float().mean())


def run_real_model_signature(model=None, tokenizer=None, *, seed: int = 0) -> dict[str, object]:
    """Run the complete offline signature experiment on real model activations."""

    if model is None or tokenizer is None:
        model, tokenizer = load_pinned_model()
    dataset = make_number_dataset()
    train, validation, heldout = split_number_dataset(dataset)
    train_labels = number_labels(train)
    validation_labels = number_labels(validation)
    heldout_labels = number_labels(heldout)

    train_cache = cache_final_token_activations(model, tokenizer, [x.prompt for x in train])
    validation_cache = cache_final_token_activations(model, tokenizer, [x.prompt for x in validation])
    heldout_cache = cache_final_token_activations(model, tokenizer, [x.prompt for x in heldout])
    train_sweep = cache_layer_sweep(model, tokenizer, [x.prompt for x in train])
    heldout_sweep = cache_layer_sweep(model, tokenizer, [x.prompt for x in heldout])
    sweep_accuracies = layer_sweep_accuracies(
        train_sweep,
        train_labels,
        heldout_sweep,
        heldout_labels,
    )
    decoder = fit_mean_difference_decoder(train_cache.activations, train_labels)
    validation_predictions = direction_predictions(validation_cache.activations, decoder)
    heldout_predictions = direction_predictions(heldout_cache.activations, decoder)

    hypothesis_rows = score_hypotheses(
        validation,
        validation_predictions,
        HYPOTHESES,
        seed=seed,
    )
    selected_hypothesis = select_hypothesis(hypothesis_rows)
    feature_examples = rank_feature_examples(
        validation,
        direction_scores(validation_cache.activations, decoder),
        decoder.threshold,
        n=2,
        seed=seed,
    )
    suffix_hypothesis = next(
        hypothesis for hypothesis in HYPOTHESES if hypothesis.hypothesis_id == "surface_suffix_s"
    )
    suffix_counterexamples = mine_counterexamples(
        validation,
        validation_predictions,
        suffix_hypothesis,
    )
    explanation_frontier = brevity_frontier(hypothesis_rows)
    suffix_predictions = hypothesis_predictions(
        heldout,
        suffix_hypothesis,
    )
    lookup_predictions = seen_plural_lookup_predictions(tokenizer, train, heldout)
    orthographic_predictions = orthographic_ridge_predictions(train, train_labels, heldout)
    contrastive_mask = t.tensor(
        [example.morphology != "regular" for example in heldout],
        dtype=t.bool,
    )
    decoder_evaluation = evaluate_binary_predictions(
        heldout_predictions,
        heldout_labels,
        suffix_predictions,
        contrastive_mask,
    )
    paired_null_accuracies = paired_swap_decoder_accuracies(
        train_cache.activations,
        train_labels,
        heldout_cache.activations,
        heldout_labels,
    )
    causal_null_patterns, causal_null_directions = paired_swap_decoder_directions(
        train_cache.activations,
        train_labels,
    )
    singular_indices = t.tensor(
        [index for index, example in enumerate(heldout) if example.number == "singular"],
        dtype=t.long,
    )
    singular_examples = tuple(heldout[int(index)] for index in singular_indices)
    singular_activations = heldout_cache.activations[singular_indices]
    full_replacements = (
        replace_direction_component(
            singular_activations,
            decoder.direction,
            decoder.positive_mean,
        )
        - singular_activations
    )
    strengths = (0.0, 0.25, 0.5, 0.75, 1.0)
    dose_margins = []
    hook_calls = []
    for strength in strengths:
        intervention = run_residual_edit(
            model,
            tokenizer,
            [example.prompt for example in singular_examples],
            strength * full_replacements,
        )
        dose_margins.append(float(intervention.next_token_margins.mean()))
        hook_calls.append(intervention.hook_calls)

    replacement_coefficients = full_replacements @ decoder.direction
    baseline_margin = dose_margins[0]
    target_delta = dose_margins[-1] - baseline_margin
    target_result = run_residual_edit(
        model,
        tokenizer,
        [example.prompt for example in singular_examples],
        full_replacements,
    )

    plural_indices = t.tensor(
        [index for index, example in enumerate(heldout) if example.number == "plural"],
        dtype=t.long,
    )
    plural_examples = tuple(heldout[int(index)] for index in plural_indices)
    plural_activations = heldout_cache.activations[plural_indices]
    reverse_replacements = (
        replace_direction_component(
            plural_activations,
            decoder.direction,
            decoder.negative_mean,
        )
        - plural_activations
    )
    reverse_baseline = float(heldout_cache.next_token_margins[plural_indices].mean())
    reverse_result = run_residual_edit(
        model,
        tokenizer,
        [example.prompt for example in plural_examples],
        reverse_replacements,
    )
    reverse_final_margin = float(reverse_result.next_token_margins.mean())

    orthogonal_distribution = run_orthogonal_control_distribution(
        model,
        tokenizer,
        [example.prompt for example in singular_examples],
        replacement_coefficients,
        decoder.direction,
        baseline_margin,
        seeds=tuple(range(seed + 101, seed + 165)),
    )
    reverse_coefficients = reverse_replacements @ decoder.direction
    reverse_orthogonal_distribution = run_orthogonal_control_distribution(
        model,
        tokenizer,
        [example.prompt for example in plural_examples],
        reverse_coefficients,
        decoder.direction,
        reverse_baseline,
        seeds=tuple(range(seed + 201, seed + 265)),
    )
    paired_causal_distribution = run_direction_control_distribution(
        model,
        tokenizer,
        [example.prompt for example in singular_examples],
        replacement_coefficients,
        causal_null_directions,
        baseline_margin,
        effect_sign=1.0,
    )
    reverse_paired_causal_distribution = run_direction_control_distribution(
        model,
        tokenizer,
        [example.prompt for example in plural_examples],
        reverse_coefficients,
        causal_null_directions,
        reverse_baseline,
        effect_sign=-1.0,
    )

    activation_accuracy = _accuracy(heldout_predictions, heldout_labels)
    suffix_accuracy = _accuracy(suffix_predictions, heldout_labels)
    lookup_accuracy = _accuracy(lookup_predictions, heldout_labels)
    orthographic_accuracy = _accuracy(orthographic_predictions, heldout_labels)
    behavior_accuracy = _accuracy(heldout_cache.next_token_margins >= 0, heldout_labels)
    paired_null_tensor = t.tensor(paired_null_accuracies)
    paired_null_mean = sum(paired_null_accuracies) / len(paired_null_accuracies)
    paired_null_p95 = float(t.quantile(paired_null_tensor, 0.95))
    paired_null_exact_p = float((paired_null_tensor >= activation_accuracy).float().mean())
    forward_causal_p = (
        1
        + sum(
            effect >= target_delta
            for effect in paired_causal_distribution["effects"]
        )
    ) / (len(causal_null_patterns) + 1)
    reverse_target_delta = reverse_baseline - reverse_final_margin
    reverse_causal_p = (
        1
        + sum(
            effect >= reverse_target_delta
            for effect in reverse_paired_causal_distribution["effects"]
        )
    ) / (len(causal_null_patterns) + 1)
    morphology_breakdown = {
        morphology: {
            "count": int(sum(example.morphology == morphology for example in heldout)),
            "activation_accuracy": _accuracy(
                heldout_predictions[t.tensor([x.morphology == morphology for x in heldout])],
                heldout_labels[t.tensor([x.morphology == morphology for x in heldout])],
            ),
            "suffix_accuracy": _accuracy(
                suffix_predictions[t.tensor([x.morphology == morphology for x in heldout])],
                heldout_labels[t.tensor([x.morphology == morphology for x in heldout])],
            ),
            "token_lookup_accuracy": _accuracy(
                lookup_predictions[t.tensor([x.morphology == morphology for x in heldout])],
                heldout_labels[t.tensor([x.morphology == morphology for x in heldout])],
            ),
        }
        for morphology in ("regular", "irregular", "nonce")
    }
    train_final_ids = set(final_noun_token_ids(tokenizer, train))
    heldout_final_ids = set(final_noun_token_ids(tokenizer, heldout))
    multi_token_noun_count = sum(
        len(tokenizer.encode(" " + example.noun, add_special_tokens=False)) > 1
        for example in heldout
    )
    heldout_rows = tuple(
        {
            "prompt": example.prompt,
            "morphology": example.morphology,
            "number": example.number,
            "activation_score": float(score),
            "activation_prediction": "plural" if prediction else "singular",
            "are_minus_is": float(margin),
        }
        for example, score, prediction, margin in zip(
            heldout,
            direction_scores(heldout_cache.activations, decoder),
            heldout_predictions,
            heldout_cache.next_token_margins,
            strict=True,
        )
    )
    return {
        "claim": (
            "Pythia-70m's embedding already linearly separates this prompt family, so "
            "held-out decoding is a lexical negative result. A train-only direction still "
            "has bidirectional causal effects that exceed precommitted within-pair decoder "
            "nulls, while an operationalized explanation loop revises an s-suffix story "
            "using validation counterexamples."
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer": LAYER,
        "train_count": len(train),
        "validation_count": len(validation),
        "heldout_count": len(heldout),
        "decoder": {
            "threshold": decoder.threshold,
            "negative_mean": decoder.negative_mean,
            "positive_mean": decoder.positive_mean,
        },
        "hypothesis_rows": hypothesis_rows,
        "selected_hypothesis": selected_hypothesis,
        "feature_examples": feature_examples,
        "suffix_counterexamples": suffix_counterexamples,
        "explanation_frontier": explanation_frontier,
        "layer_sweep": {
            "layer_names": train_sweep.layer_names,
            "heldout_accuracies": sweep_accuracies,
        },
        "heldout_rows": heldout_rows,
        "controls": (
            {"method": "activation direction", "accuracy": activation_accuracy},
            {"method": "surface suffix s", "accuracy": suffix_accuracy},
            {"method": "train final-token-id lookup", "accuracy": lookup_accuracy},
            {"method": "orthographic ridge", "accuracy": orthographic_accuracy},
            {"method": "embedding residual direction", "accuracy": sweep_accuracies[0]},
            {"method": "exact within-pair label-swap null (mean)", "accuracy": paired_null_mean},
        ),
        "metrics": {
            "validation_activation_accuracy": _accuracy(validation_predictions, validation_labels),
            "heldout_activation_accuracy": activation_accuracy,
            "heldout_behavior_accuracy": behavior_accuracy,
            "suffix_accuracy": suffix_accuracy,
            "lookup_accuracy": lookup_accuracy,
            "orthographic_accuracy": orthographic_accuracy,
            "embedding_accuracy": sweep_accuracies[0],
            "paired_null_accuracy_mean": paired_null_mean,
            "paired_null_accuracy_max": max(paired_null_accuracies),
            "paired_null_accuracy_p95": paired_null_p95,
            "paired_null_exact_p": paired_null_exact_p,
            "paired_null_count": len(paired_null_accuracies),
            "paired_null_perfect_count": sum(
                accuracy >= activation_accuracy for accuracy in paired_null_accuracies
            ),
            "forward_orthogonal_absolute_p95": orthogonal_distribution["absolute_p95"],
            "reverse_orthogonal_absolute_p95": reverse_orthogonal_distribution["absolute_p95"],
            "paired_causal_count": len(causal_null_patterns),
            "forward_paired_causal_p95": paired_causal_distribution["p95"],
            "reverse_paired_causal_p95": reverse_paired_causal_distribution["p95"],
            "forward_paired_causal_p": forward_causal_p,
            "reverse_paired_causal_p": reverse_causal_p,
            "train_heldout_final_token_overlap": len(train_final_ids & heldout_final_ids),
            "heldout_multi_token_noun_count": multi_token_noun_count,
            **asdict(decoder_evaluation),
        },
        "morphology_breakdown": morphology_breakdown,
        "paired_null_accuracies": paired_null_accuracies,
        "intervention": {
            "strengths": strengths,
            "target_mean_margins": tuple(dose_margins),
            "baseline_margin": baseline_margin,
            "target_delta": target_delta,
            "orthogonal_distribution": orthogonal_distribution,
            "reverse_orthogonal_distribution": reverse_orthogonal_distribution,
            "paired_causal_patterns": causal_null_patterns,
            "paired_causal_distribution": paired_causal_distribution,
            "reverse_paired_causal_distribution": reverse_paired_causal_distribution,
            "target_plural_preference_rate": float(
                target_result.next_token_margins.gt(0).float().mean()
            ),
            "reverse_baseline_margin": reverse_baseline,
            "reverse_final_margin": reverse_final_margin,
            "reverse_target_delta": reverse_target_delta,
            "reverse_singular_preference_rate": float(
                reverse_result.next_token_margins.lt(0).float().mean()
            ),
            "hook_calls": tuple(hook_calls),
            "mean_edit_norm": float(full_replacements.norm(dim=1).mean()),
        },
        "signature_passed": bool(
            selected_hypothesis == "subject_number_plural"
            and activation_accuracy >= 0.95
            and decoder_evaluation.survives_subset
            and activation_accuracy > suffix_accuracy
            and len(paired_null_accuracies) == 4096
            and paired_null_p95 >= activation_accuracy
            and paired_null_exact_p >= 0.05
            and target_delta > 2.0
            and sweep_accuracies[0] >= activation_accuracy
            and len(causal_null_patterns) == PAIR_CAUSAL_NULL_COUNT
            and target_delta > float(paired_causal_distribution["p95"])
            and reverse_target_delta > float(reverse_paired_causal_distribution["p95"])
            and forward_causal_p < 0.05
            and reverse_causal_p < 0.05
            and len(train_final_ids & heldout_final_ids) == 0
            and all(call_count == 1 for call_count in hook_calls)
            and int(orthogonal_distribution["hook_calls"]) == 1
            and int(reverse_orthogonal_distribution["hook_calls"]) == 1
            and all(call_count == 1 for call_count in paired_causal_distribution["hook_calls"])
            and all(
                call_count == 1
                for call_count in reverse_paired_causal_distribution["hook_calls"]
            )
        ),
    }


def run_smoke_test(cpu: bool = True) -> dict[str, object]:
    """Run the pinned-model contract on the explicitly requested device."""

    if not cpu and not t.cuda.is_available():
        raise RuntimeError("CUDA is required when cpu=False")
    model, tokenizer = load_pinned_model("cpu" if cpu else "cuda")
    return run_real_model_signature(model, tokenizer)


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, object]:
    """Run the complete real-model signature on CUDA with no CPU fallback."""

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for 7.2 GPU verification")
    t.cuda.reset_peak_memory_stats()
    model, tokenizer = load_pinned_model("cuda")
    result = run_real_model_signature(model, tokenizer)
    peak_vram_gb = t.cuda.max_memory_allocated() / 2**30
    if peak_vram_gb > max_vram_gb:
        raise AssertionError(f"peak VRAM {peak_vram_gb:.3f} GB exceeds {max_vram_gb:.1f} GB")
    if not result["signature_passed"]:
        raise AssertionError("7.2 CUDA signature did not satisfy its scientific contract")
    result.update(
        {
            "cuda_available": True,
            "gpu_name": t.cuda.get_device_name(),
            "torch_version": t.__version__,
            "cuda_version": t.version.cuda,
            "peak_vram_gb": peak_vram_gb,
            "within_vram_budget": True,
            "accepted": True,
        }
    )
    return result


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, object]:
    return run_gpu_test(max_vram_gb=max_vram_gb)


if __name__ == "__main__":
    result = run_real_model_signature()
    print(result["claim"])
    print(result["metrics"])
    print(result["intervention"])
