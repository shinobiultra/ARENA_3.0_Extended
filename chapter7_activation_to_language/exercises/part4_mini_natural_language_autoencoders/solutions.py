"""Reference implementation for [7.4] Mini Natural Language Autoencoders.

This lesson is a deliberately small supervised warm-start, not a reproduction
of paper-scale NLA reinforcement learning. Its activation verbalizer predicts
GPT-NeoX token sequences. Its activation reconstructor sends those tokens
through a frozen language model before applying the affine output head used by
the NLA paper. No phrase ID, factor parser, or activation prototype appears in
the learned round trip.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch as t
import torch.nn.functional as F


chapter = "chapter7_activation_to_language"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))


PYTHIA_MODEL_ID = "EleutherAI/pythia-70m-deduped"
PYTHIA_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
PYTHIA_HIDDEN_STATE_INDEX = 4
PYTHIA_PATCH_BLOCK_INDEX = 3
AGREEMENT_CANDIDATES = (" is", " are", " was", " were")
FACTOR_STATES = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))
AR_PROMPT = "Summary of the following text: <text>{}</text> <summary>"

NOUN_PAIRS = (
    ("key", "keys"),
    ("book", "books"),
    ("car", "cars"),
    ("tree", "trees"),
    ("house", "houses"),
    ("chair", "chairs"),
    ("table", "tables"),
    ("door", "doors"),
    ("apple", "apples"),
    ("river", "rivers"),
    ("city", "cities"),
    ("student", "students"),
    ("teacher", "teachers"),
    ("doctor", "doctors"),
    ("artist", "artists"),
    ("pilot", "pilots"),
)

TEMPLATE_PAIRS = (
    ("Today, the {}", "Yesterday, the {}", "plain"),
    ("Currently, the {}", "Earlier, the {}", "formal"),
)


@dataclass(frozen=True)
class AgreementExample:
    prompt: str
    explanation: str
    factors: tuple[float, float]
    context_id: int
    noun_lemma: str
    style: str
    held_out_state: tuple[float, float]
    split: str


@dataclass(frozen=True)
class PythiaNLADataset:
    train_activations: t.Tensor
    eval_activations: t.Tensor
    train_factors: t.Tensor
    eval_factors: t.Tensor
    train_examples: tuple[AgreementExample, ...]
    eval_examples: tuple[AgreementExample, ...]
    eval_input_ids: t.Tensor
    eval_attention_mask: t.Tensor
    eval_positions: t.Tensor
    candidate_token_ids: t.Tensor
    original_candidate_logits: t.Tensor


@dataclass(frozen=True)
class ExplanationTokens:
    token_ids: t.Tensor
    local_token_ids: t.Tensor
    attention_mask: t.Tensor
    vocabulary_token_ids: t.Tensor
    eos_token_id: int


@dataclass(frozen=True)
class TokenVerbalizer:
    weight: t.Tensor
    vocabulary_token_ids: t.Tensor
    eos_token_id: int
    sequence_length: int


@dataclass(frozen=True)
class GeneratedExplanations:
    texts: tuple[str, ...]
    token_ids: t.Tensor
    local_token_ids: t.Tensor
    logits: t.Tensor


@dataclass(frozen=True)
class TokenReconstructor:
    feature_mean: t.Tensor
    target_mean: t.Tensor
    weight: t.Tensor
    output_norm: float


@dataclass(frozen=True)
class ReconstructionMetrics:
    fve: float
    within_factor_fve: float
    normalized_mse: float
    mean_cosine: float
    distinct_rows: int


@dataclass(frozen=True)
class InterventionResult:
    base_text: str
    edited_text: str
    activation_delta_norm: float
    selected_logit_change: float
    other_rows_max_logit_change: float
    original_winner: str
    intervened_winner: str
    candidate_logits: t.Tensor


@dataclass(frozen=True)
class MiniNLAReport:
    train_count: int
    eval_count: int
    token_accuracy: float
    exact_text_accuracy: float
    reconstruction: ReconstructionMetrics
    bag_of_tokens: ReconstructionMetrics
    prompt_only_factor: ReconstructionMetrics
    direct_factor: ReconstructionMetrics
    class_mean: ReconstructionMetrics
    mean_conditioned: ReconstructionMetrics
    lexical_relabel: ReconstructionMetrics
    original_task_accuracy: float
    reconstructed_task_accuracy: float
    behavior_parity_accuracy: float
    bag_of_tokens_behavior_parity: float
    oracle_text_behavior_parity: float
    omitted_composition_end_to_end_parity: float
    prompt_only_factor_behavior_parity: float
    direct_factor_behavior_parity: float
    class_mean_behavior_parity: float
    lexical_relabel_behavior_parity: float
    same_context_token_multiset_fraction: float
    shuffled_fve_by_seed: tuple[tuple[int, float], ...]
    shuffled_behavior_parity_by_seed: tuple[tuple[int, float], ...]
    identity_max_logit_error: float
    intervention: InterventionResult
    generated_explanations: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MiniNLAExperiment:
    report: MiniNLAReport
    verbalizer: TokenVerbalizer
    reconstructor: TokenReconstructor
    reconstructed_activations: t.Tensor


def natural_language_explanation(
    noun_lemma: str,
    style: str,
    factors: tuple[float, float],
) -> str:
    """Write a readable explanation whose factor values are encoded by word order.

    Within one noun/style context, all four factor states have exactly the same
    token multiset. This makes a bag-of-tokens reconstructor a genuine order-free
    control for the contextual Pythia reconstructor.
    """

    if not noun_lemma.isalpha() or not style.isalpha():
        raise ValueError("noun_lemma and style must be alphabetic words.")
    if factors not in FACTOR_STATES:
        raise ValueError("factors must be one of the four precommitted states.")
    quantity = "one not many" if factors[0] > 0 else "many not one"
    time = "current not earlier" if factors[1] > 0 else "earlier not current"
    return (
        f"The topic is {noun_lemma}. The wording is {style}. "
        f"Quantity is {quantity}. Time is {time}."
    )


def build_agreement_examples() -> tuple[tuple[AgreementExample, ...], tuple[AgreementExample, ...]]:
    """Build 32 lexical contexts with one composition withheld in each."""

    train: list[AgreementExample] = []
    evaluation: list[AgreementExample] = []
    for noun_index, (singular, plural) in enumerate(NOUN_PAIRS):
        for template_index, (current_template, earlier_template, style) in enumerate(TEMPLATE_PAIRS):
            context_id = 2 * noun_index + template_index
            held_out_state = FACTOR_STATES[context_id % len(FACTOR_STATES)]
            for number_factor, noun in ((1.0, singular), (-1.0, plural)):
                for time_factor, template in ((1.0, current_template), (-1.0, earlier_template)):
                    factors = (number_factor, time_factor)
                    split = "eval" if factors == held_out_state else "train"
                    example = AgreementExample(
                        prompt=template.format(noun),
                        explanation=natural_language_explanation(singular, style, factors),
                        factors=factors,
                        context_id=context_id,
                        noun_lemma=singular,
                        style=style,
                        held_out_state=held_out_state,
                        split=split,
                    )
                    (evaluation if split == "eval" else train).append(example)
    return tuple(train), tuple(evaluation)


def load_pythia(device: str | t.device = "cpu"):
    """Load the pinned checkpoint from the local cache on an explicit device."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        PYTHIA_MODEL_ID,
        revision=PYTHIA_REVISION,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        PYTHIA_MODEL_ID,
        revision=PYTHIA_REVISION,
        local_files_only=True,
        dtype=t.float32,
    ).to(device)
    model.eval()
    return tokenizer, model


def load_pythia_cpu():
    """Compatibility wrapper used by the fast CPU exercise tests."""

    return load_pythia("cpu")


def _require_informative_matrix(values: t.Tensor, name: str) -> t.Tensor:
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"{name} must be a matrix with at least two rows.")
    values = values.detach().float()
    if not bool(t.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values.")
    if float(values.norm(dim=-1).mean()) <= 1e-8:
        raise ValueError(f"{name} cannot be zero-filled.")
    if float((values - values.mean(dim=0)).square().sum()) <= 1e-8:
        raise ValueError(f"{name} must contain row-level variation.")
    return values


def _batched_final_hidden(
    tokenizer,
    model,
    texts: Sequence[str],
    *,
    hidden_state_index: int,
    batch_size: int = 64,
) -> t.Tensor:
    if not texts:
        raise ValueError("texts must be nonempty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    device = next(model.parameters()).device
    chunks: list[t.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        positions = encoded.attention_mask.sum(dim=1) - 1
        rows = t.arange(len(positions), device=device)
        with t.inference_mode():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
        chunks.append(
            output.hidden_states[hidden_state_index][rows, positions].detach().cpu().float()
        )
    return t.cat(chunks)


def _candidate_indices(factors: t.Tensor) -> t.Tensor:
    if factors.ndim != 2 or factors.shape[1] != 2:
        raise ValueError("factors must have shape [batch, 2].")
    plural_offset = factors[:, 0].lt(0).long()
    earlier_offset = 2 * factors[:, 1].lt(0).long()
    return plural_offset + earlier_offset


def cache_pythia_nla_dataset(tokenizer, model) -> PythiaNLADataset:
    """Cache real layer-4 states for the compositional agreement split."""

    train_examples, eval_examples = build_agreement_examples()
    candidate_ids: list[int] = []
    for candidate in AGREEMENT_CANDIDATES:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"agreement candidate {candidate!r} is not one token.")
        candidate_ids.append(encoded[0])
    candidate_token_ids = t.tensor(candidate_ids, dtype=t.long)

    def cache(examples: Sequence[AgreementExample]):
        texts = [example.prompt for example in examples]
        device = next(model.parameters()).device
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        positions = encoded.attention_mask.sum(dim=1) - 1
        rows = t.arange(len(examples), device=device)
        with t.inference_mode():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
        activations = (
            output.hidden_states[PYTHIA_HIDDEN_STATE_INDEX][rows, positions]
            .detach()
            .cpu()
            .float()
        )
        logits = (
            output.logits[rows, positions][:, candidate_token_ids.to(device)]
            .detach()
            .cpu()
            .float()
        )
        return activations, logits, encoded.to("cpu"), positions.to("cpu")

    train_activations, _, _, _ = cache(train_examples)
    eval_activations, original_logits, eval_encoded, eval_positions = cache(eval_examples)
    _require_informative_matrix(train_activations, "train_activations")
    _require_informative_matrix(eval_activations, "eval_activations")
    return PythiaNLADataset(
        train_activations=train_activations,
        eval_activations=eval_activations,
        train_factors=t.tensor([example.factors for example in train_examples]),
        eval_factors=t.tensor([example.factors for example in eval_examples]),
        train_examples=train_examples,
        eval_examples=eval_examples,
        eval_input_ids=eval_encoded.input_ids,
        eval_attention_mask=eval_encoded.attention_mask,
        eval_positions=eval_positions,
        candidate_token_ids=candidate_token_ids,
        original_candidate_logits=original_logits,
    )


def tokenize_explanations(
    tokenizer,
    explanations: Sequence[str],
    *,
    vocabulary_token_ids: t.Tensor | None = None,
) -> ExplanationTokens:
    """Tokenize complete explanations into a small GPT-NeoX token vocabulary."""

    if not explanations or any(not text.strip() for text in explanations):
        raise ValueError("explanations must be nonempty strings.")
    eos_token_id = int(tokenizer.eos_token_id)
    sequences = [tokenizer.encode(text, add_special_tokens=False) + [eos_token_id] for text in explanations]
    if vocabulary_token_ids is None:
        vocabulary_token_ids = t.tensor(sorted({token for row in sequences for token in row}), dtype=t.long)
    else:
        vocabulary_token_ids = vocabulary_token_ids.detach().long().cpu()
    if vocabulary_token_ids.ndim != 1 or len(vocabulary_token_ids) < 2:
        raise ValueError("vocabulary_token_ids must contain at least two token IDs.")
    if len(t.unique(vocabulary_token_ids)) != len(vocabulary_token_ids):
        raise ValueError("vocabulary_token_ids cannot contain duplicates.")

    sequence_length = max(len(row) for row in sequences)
    token_ids = t.full((len(sequences), sequence_length), eos_token_id, dtype=t.long)
    attention_mask = t.zeros_like(token_ids, dtype=t.bool)
    for row_index, row in enumerate(sequences):
        token_ids[row_index, : len(row)] = t.tensor(row)
        attention_mask[row_index, : len(row)] = True
    matches = token_ids[..., None].eq(vocabulary_token_ids[None, None, :])
    if not bool(matches.any(dim=-1).all()):
        missing = t.unique(token_ids[~matches.any(dim=-1)]).tolist()
        raise ValueError(f"explanation contains token IDs outside the training vocabulary: {missing}")
    local_token_ids = matches.float().argmax(dim=-1)
    return ExplanationTokens(
        token_ids=token_ids,
        local_token_ids=local_token_ids,
        attention_mask=attention_mask,
        vocabulary_token_ids=vocabulary_token_ids,
        eos_token_id=eos_token_id,
    )


def permute_explanation_tokens(tokens: ExplanationTokens, permutation: t.Tensor) -> ExplanationTokens:
    """Permute text targets while retaining the same token vocabulary."""

    if permutation.shape != (len(tokens.token_ids),) or not t.equal(
        t.sort(permutation).values, t.arange(len(permutation))
    ):
        raise ValueError("permutation must contain every row index exactly once.")
    return ExplanationTokens(
        token_ids=tokens.token_ids[permutation],
        local_token_ids=tokens.local_token_ids[permutation],
        attention_mask=tokens.attention_mask[permutation],
        vocabulary_token_ids=tokens.vocabulary_token_ids,
        eos_token_id=tokens.eos_token_id,
    )


def fit_token_verbalizer(
    train_activations: t.Tensor,
    train_tokens: ExplanationTokens,
    *,
    ridge: float = 0.1,
) -> TokenVerbalizer:
    """Fit activation-to-token logits at every sequence position."""

    activations = _require_informative_matrix(train_activations, "train_activations")
    if len(activations) != len(train_tokens.token_ids):
        raise ValueError("activation and token rows must align.")
    if ridge <= 0:
        raise ValueError("ridge must be positive.")
    vocabulary_size = len(train_tokens.vocabulary_token_ids)
    targets = F.one_hot(train_tokens.local_token_ids, vocabulary_size).float()
    features = F.normalize(activations, dim=-1)
    design = t.cat([t.ones(len(features), 1), features], dim=1)
    dual = design @ design.T + ridge * t.eye(len(design))
    weight = design.T @ t.linalg.solve(dual, targets.reshape(len(targets), -1))
    return TokenVerbalizer(
        weight=weight,
        vocabulary_token_ids=train_tokens.vocabulary_token_ids,
        eos_token_id=train_tokens.eos_token_id,
        sequence_length=train_tokens.token_ids.shape[1],
    )


def generate_token_explanations(
    tokenizer,
    activations: t.Tensor,
    verbalizer: TokenVerbalizer,
) -> GeneratedExplanations:
    """Generate natural-language token sequences without phrase lookup."""

    if activations.ndim != 2 or activations.shape[1] + 1 != verbalizer.weight.shape[0]:
        raise ValueError("activations and verbalizer must share d_model.")
    design = t.cat([t.ones(len(activations), 1), F.normalize(activations.float(), dim=-1)], dim=1)
    vocabulary_size = len(verbalizer.vocabulary_token_ids)
    logits = (design @ verbalizer.weight).reshape(
        len(activations), verbalizer.sequence_length, vocabulary_size
    )
    local_token_ids = logits.argmax(dim=-1)
    token_ids = verbalizer.vocabulary_token_ids[local_token_ids]
    texts: list[str] = []
    for row in token_ids.tolist():
        end = row.index(verbalizer.eos_token_id) if verbalizer.eos_token_id in row else len(row)
        texts.append(tokenizer.decode(row[:end], clean_up_tokenization_spaces=False))
    return GeneratedExplanations(
        texts=tuple(texts),
        token_ids=token_ids,
        local_token_ids=local_token_ids,
        logits=logits,
    )


def token_generation_accuracy(
    generated: GeneratedExplanations,
    targets: ExplanationTokens,
) -> tuple[float, float]:
    """Return masked token accuracy and whole-sequence exact match."""

    if generated.token_ids.shape != targets.token_ids.shape:
        raise ValueError("generated and target token tensors must have the same shape.")
    correct = generated.token_ids.eq(targets.token_ids)
    token_accuracy = correct[targets.attention_mask].float().mean().item()
    exact_match = (correct | ~targets.attention_mask).all(dim=1).float().mean().item()
    return token_accuracy, exact_match


def encode_reconstructor_text(
    tokenizer,
    model,
    explanations: Sequence[str],
    *,
    batch_size: int = 64,
) -> t.Tensor:
    """Run explanation tokens through the LM and read its layer-4 final token."""

    wrapped = [AR_PROMPT.format(text) for text in explanations]
    return _batched_final_hidden(
        tokenizer,
        model,
        wrapped,
        hidden_state_index=PYTHIA_HIDDEN_STATE_INDEX,
        batch_size=batch_size,
    )


def bag_of_tokens_features(
    tokenizer,
    explanations: Sequence[str],
    *,
    vocabulary_token_ids: t.Tensor | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    """Count explanation tokens while deliberately discarding their order."""

    if not explanations or any(not text.strip() for text in explanations):
        raise ValueError("explanations must be nonempty strings.")
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in explanations]
    if vocabulary_token_ids is None:
        vocabulary_token_ids = t.tensor(
            sorted({token_id for sequence in sequences for token_id in sequence}),
            dtype=t.long,
        )
    else:
        vocabulary_token_ids = vocabulary_token_ids.detach().cpu().long()
    if vocabulary_token_ids.ndim != 1 or len(vocabulary_token_ids) < 2:
        raise ValueError("vocabulary_token_ids must be a one-dimensional vocabulary.")
    if len(t.unique(vocabulary_token_ids)) != len(vocabulary_token_ids):
        raise ValueError("vocabulary_token_ids cannot contain duplicates.")

    token_to_column = {
        int(token_id): column for column, token_id in enumerate(vocabulary_token_ids)
    }
    counts = t.zeros(len(sequences), len(vocabulary_token_ids))
    for row, sequence in enumerate(sequences):
        for token_id in sequence:
            column = token_to_column.get(int(token_id))
            if column is not None:
                counts[row, column] += 1
    return counts, vocabulary_token_ids


def fit_token_reconstructor(
    train_text_features: t.Tensor,
    train_activations: t.Tensor,
    *,
    ridge: float = 1e-3,
) -> TokenReconstructor:
    """Fit the affine AR head after contextual token processing."""

    text_features = _require_informative_matrix(train_text_features, "train_text_features")
    activations = _require_informative_matrix(train_activations, "train_activations")
    if len(text_features) != len(activations):
        raise ValueError("text features and activations must have the same batch size.")
    if ridge <= 0:
        raise ValueError("ridge must be positive.")
    features = F.normalize(text_features, dim=-1)
    targets = F.normalize(activations, dim=-1)
    feature_mean = features.mean(dim=0)
    target_mean = targets.mean(dim=0)
    centered_features = features - feature_mean
    centered_targets = targets - target_mean
    dual = centered_features @ centered_features.T + ridge * t.eye(
        len(features), device=features.device, dtype=features.dtype
    )
    weight = centered_features.T @ t.linalg.solve(dual, centered_targets)
    return TokenReconstructor(
        feature_mean=feature_mean,
        target_mean=target_mean,
        weight=weight,
        output_norm=float(activations.norm(dim=-1).mean()),
    )


def decode_text_features(text_features: t.Tensor, reconstructor: TokenReconstructor) -> t.Tensor:
    """Apply the AR head and restore the training-set mean activation norm."""

    if text_features.ndim != 2 or text_features.shape[1] != reconstructor.weight.shape[0]:
        raise ValueError("text features and reconstructor must share d_model.")
    features = F.normalize(text_features.float(), dim=-1)
    reconstructed = (features - reconstructor.feature_mean) @ reconstructor.weight
    reconstructed = reconstructed + reconstructor.target_mean
    return F.normalize(reconstructed, dim=-1) * reconstructor.output_norm


def reconstruct_from_explanations(
    tokenizer,
    model,
    explanations: Sequence[str],
    reconstructor: TokenReconstructor,
) -> t.Tensor:
    """Run the complete natural-language-token-to-activation path."""

    features = encode_reconstructor_text(tokenizer, model, explanations)
    return decode_text_features(features, reconstructor)


def fraction_variance_explained(original: t.Tensor, reconstructed: t.Tensor) -> float:
    """Compute the NLA paper's row-normalized held-out FVE."""

    if original.shape != reconstructed.shape or original.ndim != 2:
        raise ValueError("original and reconstructed must be aligned matrices.")
    original = F.normalize(original.float(), dim=-1)
    reconstructed = F.normalize(reconstructed.float(), dim=-1)
    denominator = (original - original.mean(dim=0)).square().sum()
    if float(denominator) <= 1e-8:
        raise ValueError("original activations need nonzero held-out variance.")
    return float(1 - (reconstructed - original).square().sum() / denominator)


def within_factor_fve(original: t.Tensor, reconstructed: t.Tensor, factors: t.Tensor) -> float:
    """Measure reconstruction after removing each factor combination's mean."""

    if original.shape != reconstructed.shape or factors.shape != (len(original), 2):
        raise ValueError("original, reconstructed, and factor rows must align.")
    original = F.normalize(original.float(), dim=-1)
    reconstructed = F.normalize(reconstructed.float(), dim=-1)
    factors = factors.to(original.device)
    original_residuals: list[t.Tensor] = []
    reconstructed_residuals: list[t.Tensor] = []
    for state in FACTOR_STATES:
        mask = factors.eq(
            t.tensor(state, device=factors.device, dtype=factors.dtype)
        ).all(dim=1)
        if int(mask.sum()) < 2:
            raise ValueError("each factor state needs at least two rows.")
        original_group = original[mask]
        reconstructed_group = reconstructed[mask]
        original_residuals.append(original_group - original_group.mean(dim=0))
        reconstructed_residuals.append(reconstructed_group - reconstructed_group.mean(dim=0))
    original_residual = t.cat(original_residuals)
    reconstructed_residual = t.cat(reconstructed_residuals)
    denominator = original_residual.square().sum()
    if float(denominator) <= 1e-8:
        raise ValueError("within-factor activations need nonzero variance.")
    return float(1 - (reconstructed_residual - original_residual).square().sum() / denominator)


def count_distinct_rows(values: t.Tensor, *, atol: float = 1e-2) -> int:
    """Count rows up to an explicit Euclidean tolerance."""

    if values.ndim != 2 or atol <= 0:
        raise ValueError("values must be a matrix and atol must be positive.")
    representatives: list[t.Tensor] = []
    for row in values:
        if not any(float((row - representative).norm()) <= atol for representative in representatives):
            representatives.append(row)
    return len(representatives)


def reconstruction_metrics(
    original: t.Tensor,
    reconstructed: t.Tensor,
    factors: t.Tensor,
) -> ReconstructionMetrics:
    """Score full-vector and within-factor reconstruction on aligned rows."""

    normalized_error = (
        F.normalize(original, dim=-1) - F.normalize(reconstructed, dim=-1)
    ).square().sum(dim=-1).mean()
    return ReconstructionMetrics(
        fve=fraction_variance_explained(original, reconstructed),
        within_factor_fve=within_factor_fve(original, reconstructed, factors),
        normalized_mse=float(normalized_error),
        mean_cosine=float(F.cosine_similarity(original, reconstructed, dim=-1).mean()),
        distinct_rows=count_distinct_rows(reconstructed),
    )


def assert_nonprototype_reconstruction(
    original: t.Tensor,
    reconstructed: t.Tensor,
    factors: t.Tensor,
    *,
    minimum_within_factor_fve: float = 0.2,
) -> None:
    """Reject four-vector prototypes even if they preserve agreement labels."""

    factor_state_count = len(t.unique(factors, dim=0))
    distinct_rows = count_distinct_rows(reconstructed)
    if distinct_rows <= factor_state_count:
        raise AssertionError(
            f"prototype collapse: found {distinct_rows} rows for {factor_state_count} factor states."
        )
    score = within_factor_fve(original, reconstructed, factors)
    if score <= minimum_within_factor_fve:
        raise AssertionError(
            f"within-factor activation variation was not reconstructed: FVE={score:.3f}."
        )


def prompt_only_factor_explanations(examples: Sequence[AgreementExample]) -> tuple[str, ...]:
    """Hold context fixed while preserving the exact explanation template."""

    return tuple(
        natural_language_explanation("key", "plain", example.factors)
        for example in examples
    )


def lexically_relabel_explanations(examples: Sequence[AgreementExample]) -> tuple[str, ...]:
    """Cycle noun meanings while preserving factor words and sentence shape."""

    if len(examples) < 4:
        raise ValueError("lexical relabeling needs at least four examples.")
    offset = 4
    relabeled = tuple(
        natural_language_explanation(
            examples[(index + offset) % len(examples)].noun_lemma,
            example.style,
            example.factors,
        )
        for index, example in enumerate(examples)
    )
    if any(text == example.explanation for text, example in zip(relabeled, examples)):
        raise AssertionError("lexical relabeling must change every explanation.")
    return relabeled


def factor_prototype_reconstruction(
    train_activations: t.Tensor,
    train_factors: t.Tensor,
    eval_factors: t.Tensor,
) -> t.Tensor:
    """Explicit class-mean baseline used only to expose prototype shortcuts."""

    if train_factors.shape != (len(train_activations), 2) or eval_factors.ndim != 2:
        raise ValueError("activation and factor rows must align.")
    train_factors = train_factors.to(train_activations.device)
    eval_factors = eval_factors.to(train_activations.device)
    rows: list[t.Tensor] = []
    for factors in eval_factors:
        mask = train_factors.eq(factors).all(dim=1)
        if not bool(mask.any()):
            raise ValueError("every evaluation state needs training rows for the prototype baseline.")
        rows.append(train_activations[mask].mean(dim=0))
    return t.stack(rows)


def replace_with_factor_prototypes(activations: t.Tensor, factors: t.Tensor) -> t.Tensor:
    """Replace every row by its factor mean for an adversarial training control."""

    return factor_prototype_reconstruction(activations, factors, factors)


def fit_direct_factor_decoder(
    train_factors: t.Tensor,
    train_activations: t.Tensor,
    *,
    ridge: float = 1e-3,
) -> t.Tensor:
    """Fit the forbidden two-scalar shortcut as an explicit negative baseline."""

    activations = _require_informative_matrix(train_activations, "train_activations")
    if train_factors.shape != (len(activations), 2):
        raise ValueError("train_factors must align with activations.")
    train_factors = train_factors.to(activations.device)
    design = t.cat(
        [
            t.ones(
                len(train_factors),
                1,
                device=activations.device,
                dtype=activations.dtype,
            ),
            train_factors.to(dtype=activations.dtype),
        ],
        dim=1,
    )
    regularizer = ridge * t.eye(
        design.shape[1], device=design.device, dtype=design.dtype
    )
    regularizer[0, 0] = 0
    return t.linalg.solve(design.T @ design + regularizer, design.T @ activations)


def decode_direct_factors(factors: t.Tensor, parameters: t.Tensor) -> t.Tensor:
    """Apply the two-scalar baseline; never use this in the NLA path."""

    if factors.ndim != 2 or factors.shape[1] != 2 or parameters.shape[0] != 3:
        raise ValueError("expected factors [batch, 2] and parameters [3, d_model].")
    factors = factors.to(device=parameters.device, dtype=parameters.dtype)
    return t.cat(
        [
            t.ones(
                len(factors),
                1,
                device=parameters.device,
                dtype=parameters.dtype,
            ),
            factors,
        ],
        dim=1,
    ) @ parameters


def patch_pythia_candidate_logits(
    model,
    dataset: PythiaNLADataset,
    replacements: t.Tensor,
    *,
    row_indices: Sequence[int] | None = None,
) -> t.Tensor:
    """Patch all rows or a declared subset and return every row's candidate logits."""

    device = next(model.parameters()).device
    if row_indices is None:
        row_indices = tuple(range(len(dataset.eval_activations)))
    rows = t.tensor(tuple(row_indices), dtype=t.long)
    if len(rows) == 0 or len(t.unique(rows)) != len(rows):
        raise ValueError("row_indices must be nonempty and unique.")
    if int(rows.min()) < 0 or int(rows.max()) >= len(dataset.eval_activations):
        raise ValueError("row_indices are outside the evaluation batch.")
    if replacements.shape != (len(rows), dataset.eval_activations.shape[1]):
        raise ValueError("replacements must have shape [selected_rows, d_model].")
    if not bool(t.isfinite(replacements).all()):
        raise ValueError("replacements must be finite.")

    def replace_selected_rows(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        device_rows = rows.to(hidden.device)
        positions = dataset.eval_positions[rows].to(hidden.device)
        hidden[device_rows, positions] = replacements.to(hidden)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = model.gpt_neox.layers[PYTHIA_PATCH_BLOCK_INDEX].register_forward_hook(
        replace_selected_rows
    )
    all_rows = t.arange(len(dataset.eval_activations), device=device)
    try:
        with t.inference_mode():
            output = model(
                input_ids=dataset.eval_input_ids.to(device),
                attention_mask=dataset.eval_attention_mask.to(device),
                use_cache=False,
            )
        logits = output.logits[all_rows, dataset.eval_positions.to(device)]
        return logits[:, dataset.candidate_token_ids.to(device)].detach().cpu().float()
    finally:
        handle.remove()


def candidate_accuracy(candidate_logits: t.Tensor, factors: t.Tensor) -> float:
    """Score the four-way grammatical target without a label lookup table."""

    targets = _candidate_indices(factors).to(candidate_logits.device)
    return float(candidate_logits.argmax(dim=1).eq(targets).float().mean())


def behavior_parity(candidate_logits: t.Tensor, original_candidate_logits: t.Tensor) -> float:
    """Measure whether a patch preserves the target model's own choices."""

    if candidate_logits.shape != original_candidate_logits.shape:
        raise ValueError("candidate logit tensors must align.")
    return float(
        candidate_logits.argmax(dim=1).eq(original_candidate_logits.argmax(dim=1)).float().mean()
    )


def run_text_edit_intervention(
    tokenizer,
    model,
    dataset: PythiaNLADataset,
    reconstructor: TokenReconstructor,
    *,
    row_index: int,
    base_text: str,
    edited_text: str,
    strength: float = 1.0,
) -> InterventionResult:
    """Apply an AR(edit)-AR(base) steering vector to exactly one real row."""

    if not 0 <= row_index < len(dataset.eval_activations):
        raise ValueError("row_index is outside the evaluation batch.")
    if not t.isfinite(t.tensor(strength)) or strength == 0:
        raise ValueError("strength must be finite and nonzero.")
    reconstructions = reconstruct_from_explanations(
        tokenizer, model, (base_text, edited_text), reconstructor
    )
    delta = strength * (reconstructions[1] - reconstructions[0])
    replacement = dataset.eval_activations[row_index] + delta
    patched_logits = patch_pythia_candidate_logits(
        model,
        dataset,
        replacement[None, :],
        row_indices=(row_index,),
    )
    logit_change = (patched_logits - dataset.original_candidate_logits).abs()
    other_rows = t.arange(len(patched_logits)) != row_index
    original_winner = AGREEMENT_CANDIDATES[
        int(dataset.original_candidate_logits[row_index].argmax())
    ].strip()
    intervened_winner = AGREEMENT_CANDIDATES[int(patched_logits[row_index].argmax())].strip()
    return InterventionResult(
        base_text=base_text,
        edited_text=edited_text,
        activation_delta_norm=float(delta.norm()),
        selected_logit_change=float(logit_change[row_index].max()),
        other_rows_max_logit_change=float(logit_change[other_rows].max()),
        original_winner=original_winner,
        intervened_winner=intervened_winner,
        candidate_logits=patched_logits[row_index],
    )


def _counterfactual_explanation(example: AgreementExample) -> str:
    flipped = (-example.factors[0], example.factors[1])
    return natural_language_explanation(example.noun_lemma, example.style, flipped)


def _metrics_for(
    original: t.Tensor,
    candidates: dict[str, t.Tensor],
    factors: t.Tensor,
) -> dict[str, ReconstructionMetrics]:
    return {
        name: reconstruction_metrics(original, reconstructed, factors)
        for name, reconstructed in candidates.items()
    }


def evaluate_mini_nla(
    tokenizer,
    model,
    dataset: PythiaNLADataset,
    *,
    shuffle_seeds: Sequence[int] = tuple(range(8)),
) -> MiniNLAExperiment:
    """Run the held-out token NLA and all adversarial controls."""

    train_texts = tuple(example.explanation for example in dataset.train_examples)
    eval_texts = tuple(example.explanation for example in dataset.eval_examples)
    train_tokens = tokenize_explanations(tokenizer, train_texts)
    eval_tokens = tokenize_explanations(
        tokenizer,
        eval_texts,
        vocabulary_token_ids=train_tokens.vocabulary_token_ids,
    )
    verbalizer = fit_token_verbalizer(dataset.train_activations, train_tokens)
    generated = generate_token_explanations(tokenizer, dataset.eval_activations, verbalizer)
    token_accuracy, exact_text_accuracy = token_generation_accuracy(generated, eval_tokens)

    train_text_features = encode_reconstructor_text(tokenizer, model, train_texts)
    reconstructor = fit_token_reconstructor(train_text_features, dataset.train_activations)
    generated_features = encode_reconstructor_text(tokenizer, model, generated.texts)
    learned = decode_text_features(generated_features, reconstructor)
    oracle = reconstruct_from_explanations(tokenizer, model, eval_texts, reconstructor)

    train_bag_features, bag_vocabulary = bag_of_tokens_features(tokenizer, train_texts)
    eval_bag_features, _ = bag_of_tokens_features(
        tokenizer,
        eval_texts,
        vocabulary_token_ids=bag_vocabulary,
    )
    train_bag_features = train_bag_features.to(dataset.train_activations.device)
    eval_bag_features = eval_bag_features.to(dataset.eval_activations.device)
    bag_reconstructor = fit_token_reconstructor(
        train_bag_features,
        dataset.train_activations,
    )
    bag_reconstruction = decode_text_features(eval_bag_features, bag_reconstructor)

    factor_texts = prompt_only_factor_explanations(dataset.eval_examples)
    factor_text_reconstruction = reconstruct_from_explanations(
        tokenizer, model, factor_texts, reconstructor
    )
    lexical_texts = lexically_relabel_explanations(dataset.eval_examples)
    lexical_reconstruction = reconstruct_from_explanations(
        tokenizer, model, lexical_texts, reconstructor
    )
    class_mean_reconstruction = factor_prototype_reconstruction(
        dataset.train_activations, dataset.train_factors, dataset.eval_factors
    )
    direct_parameters = fit_direct_factor_decoder(
        dataset.train_factors, dataset.train_activations
    )
    direct_reconstruction = decode_direct_factors(dataset.eval_factors, direct_parameters)
    mean_conditioned_targets = replace_with_factor_prototypes(
        dataset.train_activations, dataset.train_factors
    )
    mean_conditioned_head = fit_token_reconstructor(
        train_text_features, mean_conditioned_targets
    )
    mean_conditioned_reconstruction = decode_text_features(
        encode_reconstructor_text(tokenizer, model, eval_texts), mean_conditioned_head
    )

    candidates = {
        "reconstruction": learned,
        "bag_of_tokens": bag_reconstruction,
        "prompt_only_factor": factor_text_reconstruction,
        "direct_factor": direct_reconstruction,
        "class_mean": class_mean_reconstruction,
        "mean_conditioned": mean_conditioned_reconstruction,
        "lexical_relabel": lexical_reconstruction,
    }
    metrics = _metrics_for(dataset.eval_activations, candidates, dataset.eval_factors)
    assert_nonprototype_reconstruction(
        dataset.eval_activations, learned, dataset.eval_factors
    )

    identity_logits = patch_pythia_candidate_logits(
        model, dataset, dataset.eval_activations
    )
    learned_logits = patch_pythia_candidate_logits(model, dataset, learned)
    bag_logits = patch_pythia_candidate_logits(model, dataset, bag_reconstruction)
    oracle_logits = patch_pythia_candidate_logits(model, dataset, oracle)
    factor_logits = patch_pythia_candidate_logits(model, dataset, factor_text_reconstruction)
    direct_logits = patch_pythia_candidate_logits(model, dataset, direct_reconstruction)
    class_mean_logits = patch_pythia_candidate_logits(model, dataset, class_mean_reconstruction)
    lexical_logits = patch_pythia_candidate_logits(model, dataset, lexical_reconstruction)

    shuffled_fve: list[tuple[int, float]] = []
    shuffled_parity: list[tuple[int, float]] = []
    for seed in shuffle_seeds:
        generator = t.Generator().manual_seed(int(seed))
        permutation = t.randperm(len(train_tokens.token_ids), generator=generator)
        shuffled_tokens = permute_explanation_tokens(train_tokens, permutation)
        shuffled_verbalizer = fit_token_verbalizer(
            dataset.train_activations, shuffled_tokens
        )
        shuffled_text = generate_token_explanations(
            tokenizer, dataset.eval_activations, shuffled_verbalizer
        )
        shuffled_reconstruction = reconstruct_from_explanations(
            tokenizer, model, shuffled_text.texts, reconstructor
        )
        shuffled_logits = patch_pythia_candidate_logits(
            model, dataset, shuffled_reconstruction
        )
        shuffled_fve.append(
            (int(seed), fraction_variance_explained(dataset.eval_activations, shuffled_reconstruction))
        )
        shuffled_parity.append(
            (int(seed), behavior_parity(shuffled_logits, dataset.original_candidate_logits))
        )

    first_example = dataset.eval_examples[0]
    intervention = run_text_edit_intervention(
        tokenizer,
        model,
        dataset,
        reconstructor,
        row_index=0,
        base_text=generated.texts[0],
        edited_text=_counterfactual_explanation(first_example),
    )
    if intervention.activation_delta_norm <= 1e-3 or intervention.selected_logit_change <= 1e-3:
        raise AssertionError("the text edit must have a nonzero activation and causal logit effect.")

    targets = _candidate_indices(dataset.eval_factors)
    original_winners = dataset.original_candidate_logits.argmax(dim=1)
    reconstructed_winners = learned_logits.argmax(dim=1)
    all_examples = (*dataset.train_examples, *dataset.eval_examples)
    all_bag_features, _ = bag_of_tokens_features(
        tokenizer,
        tuple(example.explanation for example in all_examples),
    )
    multiset_matches = []
    for context_id in sorted({example.context_id for example in all_examples}):
        rows_for_context = [
            index
            for index, example in enumerate(all_examples)
            if example.context_id == context_id
        ]
        reference = all_bag_features[rows_for_context[0]]
        multiset_matches.append(
            all(t.equal(reference, all_bag_features[index]) for index in rows_for_context[1:])
        )
    rows = tuple(
        {
            "prompt": example.prompt,
            "generated explanation": explanation,
            "target": AGREEMENT_CANDIDATES[int(target)].strip(),
            "original": AGREEMENT_CANDIDATES[int(original)].strip(),
            "reconstructed": AGREEMENT_CANDIDATES[int(reconstructed)].strip(),
            "behavior preserved": bool(original == reconstructed),
            "row cosine": float(
                F.cosine_similarity(
                    dataset.eval_activations[index : index + 1],
                    learned[index : index + 1],
                )
            ),
        }
        for index, (example, explanation, target, original, reconstructed) in enumerate(
            zip(
                dataset.eval_examples,
                generated.texts,
                targets.tolist(),
                original_winners.tolist(),
                reconstructed_winners.tolist(),
            )
        )
    )

    report = MiniNLAReport(
        train_count=len(dataset.train_examples),
        eval_count=len(dataset.eval_examples),
        token_accuracy=token_accuracy,
        exact_text_accuracy=exact_text_accuracy,
        reconstruction=metrics["reconstruction"],
        bag_of_tokens=metrics["bag_of_tokens"],
        prompt_only_factor=metrics["prompt_only_factor"],
        direct_factor=metrics["direct_factor"],
        class_mean=metrics["class_mean"],
        mean_conditioned=metrics["mean_conditioned"],
        lexical_relabel=metrics["lexical_relabel"],
        original_task_accuracy=candidate_accuracy(
            dataset.original_candidate_logits, dataset.eval_factors
        ),
        reconstructed_task_accuracy=candidate_accuracy(learned_logits, dataset.eval_factors),
        behavior_parity_accuracy=behavior_parity(
            learned_logits, dataset.original_candidate_logits
        ),
        bag_of_tokens_behavior_parity=behavior_parity(
            bag_logits, dataset.original_candidate_logits
        ),
        oracle_text_behavior_parity=behavior_parity(
            oracle_logits, dataset.original_candidate_logits
        ),
        omitted_composition_end_to_end_parity=behavior_parity(
            learned_logits, dataset.original_candidate_logits
        ),
        prompt_only_factor_behavior_parity=behavior_parity(
            factor_logits, dataset.original_candidate_logits
        ),
        direct_factor_behavior_parity=behavior_parity(
            direct_logits, dataset.original_candidate_logits
        ),
        class_mean_behavior_parity=behavior_parity(
            class_mean_logits, dataset.original_candidate_logits
        ),
        lexical_relabel_behavior_parity=behavior_parity(
            lexical_logits, dataset.original_candidate_logits
        ),
        same_context_token_multiset_fraction=sum(multiset_matches) / len(multiset_matches),
        shuffled_fve_by_seed=tuple(shuffled_fve),
        shuffled_behavior_parity_by_seed=tuple(shuffled_parity),
        identity_max_logit_error=float(
            (identity_logits - dataset.original_candidate_logits).abs().max()
        ),
        intervention=intervention,
        generated_explanations=generated.texts,
        rows=rows,
    )
    return MiniNLAExperiment(
        report=report,
        verbalizer=verbalizer,
        reconstructor=reconstructor,
        reconstructed_activations=learned,
    )


def save_signature_figure(report: MiniNLAReport, path: str | Path) -> Path:
    """Create the learner-facing result from live report values."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    labels = ["token NLA", "bag of tokens", "factor text", "direct factors", "class mean", "relabel"]
    fves = [
        report.reconstruction.fve,
        report.bag_of_tokens.fve,
        report.prompt_only_factor.fve,
        report.direct_factor.fve,
        report.class_mean.fve,
        report.lexical_relabel.fve,
    ]
    colors = ["#16837a", "#8172b3", "#6f6f6f", "#b07c26", "#c44e52", "#4c72b0"]
    axes[0].bar(labels, fves, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Held-out full-vector FVE", ylabel="fraction variance explained")
    axes[0].tick_params(axis="x", rotation=32)

    within = [
        report.reconstruction.within_factor_fve,
        report.bag_of_tokens.within_factor_fve,
        report.prompt_only_factor.within_factor_fve,
        report.direct_factor.within_factor_fve,
        report.class_mean.within_factor_fve,
        report.lexical_relabel.within_factor_fve,
    ]
    axes[1].bar(labels, within, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Variation within the same factors", ylabel="within-factor FVE")
    axes[1].tick_params(axis="x", rotation=32)

    behavior_labels = [
        "token NLA",
        "bag of tokens",
        "factor text",
        "direct factors",
        "class mean",
        "relabel",
        "shuffled AV",
    ]
    shuffle_values = [value for _, value in report.shuffled_behavior_parity_by_seed]
    behavior = [
        report.behavior_parity_accuracy,
        report.bag_of_tokens_behavior_parity,
        report.prompt_only_factor_behavior_parity,
        report.direct_factor_behavior_parity,
        report.class_mean_behavior_parity,
        report.lexical_relabel_behavior_parity,
        sum(shuffle_values) / len(shuffle_values),
    ]
    axes[2].bar(behavior_labels, behavior, color=[*colors, "#9b9b9b"])
    axes[2].scatter(
        [5.82 + 0.05 * index for index in range(len(shuffle_values))],
        shuffle_values,
        color="black",
        s=18,
        label="shuffled AV seeds",
        zorder=3,
    )
    axes[2].set(
        ylim=(0, 1.08),
        title="Causal behavior parity",
        ylabel="fraction matching original model",
    )
    axes[2].tick_params(axis="x", rotation=32)
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Same words, different order: "
        f"Pythia AR parity {report.behavior_parity_accuracy:.2f} vs "
        f"bag-of-tokens {report.bag_of_tokens_behavior_parity:.2f}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    """Return structural facts only; scientific acceptance is never report-driven."""

    del cpu
    train, evaluation = build_agreement_examples()
    return {
        "notebook_id": "7_4_mini_natural_language_autoencoders",
        "train_count": len(train),
        "eval_count": len(evaluation),
        "contexts": len({example.context_id for example in (*train, *evaluation)}),
        "eval_factor_counts": {
            str(state): sum(example.factors == state for example in evaluation)
            for state in FACTOR_STATES
        },
    }


def run_cpu_test() -> dict[str, Any]:
    """Run the complete pinned experiment on CPU and return measured evidence."""

    tokenizer, model = load_pythia_cpu()
    dataset = cache_pythia_nla_dataset(tokenizer, model)
    report = evaluate_mini_nla(tokenizer, model, dataset).report
    return {
        "model_id": PYTHIA_MODEL_ID,
        "revision": PYTHIA_REVISION,
        "device": "cpu",
        "train_count": report.train_count,
        "eval_count": report.eval_count,
        "token_accuracy": report.token_accuracy,
        "exact_text_accuracy": report.exact_text_accuracy,
        "fve": report.reconstruction.fve,
        "bag_of_tokens_fve": report.bag_of_tokens.fve,
        "within_factor_fve": report.reconstruction.within_factor_fve,
        "bag_of_tokens_within_factor_fve": report.bag_of_tokens.within_factor_fve,
        "distinct_rows": report.reconstruction.distinct_rows,
        "normalized_mse": report.reconstruction.normalized_mse,
        "mean_cosine": report.reconstruction.mean_cosine,
        "original_task_accuracy": report.original_task_accuracy,
        "reconstructed_task_accuracy": report.reconstructed_task_accuracy,
        "behavior_parity_accuracy": report.behavior_parity_accuracy,
        "bag_of_tokens_behavior_parity": report.bag_of_tokens_behavior_parity,
        "same_context_token_multiset_fraction": report.same_context_token_multiset_fraction,
        "omitted_composition_end_to_end_parity": report.omitted_composition_end_to_end_parity,
        "prompt_only_factor_fve": report.prompt_only_factor.fve,
        "direct_factor_fve": report.direct_factor.fve,
        "class_mean_fve": report.class_mean.fve,
        "mean_conditioned_fve": report.mean_conditioned.fve,
        "lexical_relabel_fve": report.lexical_relabel.fve,
        "lexical_relabel_within_factor_fve": report.lexical_relabel.within_factor_fve,
        "prompt_only_factor_behavior_parity": report.prompt_only_factor_behavior_parity,
        "direct_factor_behavior_parity": report.direct_factor_behavior_parity,
        "class_mean_behavior_parity": report.class_mean_behavior_parity,
        "lexical_relabel_behavior_parity": report.lexical_relabel_behavior_parity,
        "identity_max_logit_error": report.identity_max_logit_error,
        "single_row_other_max_logit_change": report.intervention.other_rows_max_logit_change,
        "intervention_selected_logit_change": report.intervention.selected_logit_change,
        "shuffled_fve_by_seed": report.shuffled_fve_by_seed,
        "shuffled_behavior_parity_by_seed": report.shuffled_behavior_parity_by_seed,
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Run the full text bottleneck on CUDA with no fallback."""

    if max_vram_gb <= 0:
        raise ValueError("max_vram_gb must be positive.")
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for 7.4 GPU verification.")
    t.cuda.reset_peak_memory_stats()
    tokenizer, model = load_pythia("cuda")
    dataset = cache_pythia_nla_dataset(tokenizer, model)
    report = evaluate_mini_nla(tokenizer, model, dataset).report
    peak_vram_gb = t.cuda.max_memory_allocated() / 2**30
    if peak_vram_gb > max_vram_gb:
        raise AssertionError(
            f"peak VRAM {peak_vram_gb:.3f} GB exceeds {max_vram_gb:.1f} GB"
        )
    checks = {
        "exact_text": report.exact_text_accuracy == 1.0,
        "normalized_fve": report.reconstruction.fve >= 0.75,
        "beats_bag_of_tokens_fve": (
            report.reconstruction.fve > report.bag_of_tokens.fve + 0.1
        ),
        "behavior_parity": report.behavior_parity_accuracy == 1.0,
        "beats_bag_of_tokens_behavior": (
            report.behavior_parity_accuracy
            > report.bag_of_tokens_behavior_parity + 0.5
        ),
        "same_token_multiset": report.same_context_token_multiset_fraction == 1.0,
        "identity_patch": report.identity_max_logit_error == 0.0,
        "single_row_locality": report.intervention.other_rows_max_logit_change == 0.0,
        "causal_text_edit": report.intervention.selected_logit_change > 1.0,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"7.4 scientific contract failed: {failed}")
    return {
        "accepted": True,
        "cuda_available": True,
        "device": "cuda",
        "gpu_name": t.cuda.get_device_name(),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": True,
        "model_id": PYTHIA_MODEL_ID,
        "revision": PYTHIA_REVISION,
        "train_count": report.train_count,
        "eval_count": report.eval_count,
        "token_accuracy": report.token_accuracy,
        "exact_text_accuracy": report.exact_text_accuracy,
        "held_out_fve": report.reconstruction.fve,
        "within_factor_fve": report.reconstruction.within_factor_fve,
        "bag_of_tokens_fve": report.bag_of_tokens.fve,
        "bag_of_tokens_within_factor_fve": report.bag_of_tokens.within_factor_fve,
        "behavior_parity": report.behavior_parity_accuracy,
        "bag_of_tokens_behavior_parity": report.bag_of_tokens_behavior_parity,
        "same_context_token_multiset_fraction": report.same_context_token_multiset_fraction,
        "distinct_reconstruction_rows": report.reconstruction.distinct_rows,
        "identity_max_logit_error": report.identity_max_logit_error,
        "intervention_selected_logit_change": report.intervention.selected_logit_change,
        "single_row_other_max_logit_change": report.intervention.other_rows_max_logit_change,
        "checks": checks,
    }


def run_full_experiment(max_vram_gb: float = 24.0):
    """Run the complete pinned CUDA experiment."""

    return run_gpu_test(max_vram_gb=max_vram_gb)
