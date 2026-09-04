# %%
"""Reference solutions for [7.5] Predictive Concept Decoders."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t
import torch.nn.functional as F

chapter = "chapter7_activation_to_language"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_RESID_POST_HOOK = "blocks.0.hook_resid_post"
TL_POSITIVE_ANCHOR = "The cat sat on the"
TL_NEGATIVE_ANCHOR = "The bird flew over the"
TL_CONCEPT_TOP_K = 4
TL_TRAIN_EXAMPLES = [
    ("The cat sat on the", "surface"),
    ("The dog slept on the", "surface"),
    ("The child sat on the", "surface"),
    ("The book rested on the", "surface"),
    ("The blanket lay on the", "surface"),
    ("The lamp sat on the", "surface"),
    ("The bird flew over the", "motion"),
    ("The plane flew over the", "motion"),
    ("The kite floated above the", "motion"),
    ("The cloud drifted above the", "motion"),
    ("The feather floated above the", "motion"),
    ("The train rushed through the", "motion"),
]
TL_EVAL_EXAMPLES = [
    ("Yesterday at home, the blanket lay on the", "surface"),
    ("In the quiet room, the lamp sat on the", "surface"),
    ("After dinner, the pillow rested on the", "surface"),
    ("Near the window, the chair sat on the", "surface"),
    ("At noon outside, the rocket launched into the", "motion"),
    ("During the game, the ball flew over the", "motion"),
    ("On the hill, the kite floated above the", "motion"),
    ("In the hallway, the train rushed through the", "motion"),
]
TL_PCD_QUESTIONS = (
    "Will the next token be the surface-associated token?",
    "Will the next token be the motion-associated token?",
    "Is the hidden state carrying a resting-surface concept?",
    "Is the hidden state carrying a motion-or-airborne concept?",
)
PLANTED_PCD_QUESTIONS = (
    "Is the hidden surface concept active?",
    "Is the hidden motion concept active?",
    "Is the hidden red concept active?",
    "Is the hidden animal concept active?",
)
PLANTED_CONCEPT_NAMES = (
    "surface",
    "motion",
    "red",
    "animal",
    "template_alpha",
    "template_beta",
)
PLANTED_QUESTION_TO_CONCEPT = (0, 1, 2, 3)


# %%
@dataclass(frozen=True)
class PCDQuestionBatch:
    activations: t.Tensor
    question_ids: t.Tensor
    answer_ids: t.Tensor
    question_texts: tuple[str, ...]


@dataclass(frozen=True)
class PCDComparisonReport:
    pcd_accuracy: float
    probe_accuracy: float
    sae_classifier_accuracy: float
    activation_oracle_accuracy: float
    best_baseline_accuracy: float
    beats_probe: bool
    beats_best_baseline: bool


@dataclass(frozen=True)
class ConceptSparsityReport:
    mean_l0: float
    density: float
    passes_sparsity: bool


@dataclass(frozen=True)
class ConceptStabilityReport:
    top_concepts_by_seed: tuple[tuple[int, ...], ...]
    mean_pairwise_jaccard: float
    stable: bool


@dataclass(frozen=True)
class ConceptRemovalReport:
    original_answer: int
    top_removed_answer: int
    random_removed_answer: int
    top_removal_changed: bool
    random_removal_changed: bool
    top_removal_delta: float
    random_removal_delta: float
    random_removal_does_less: bool


@dataclass(frozen=True)
class ConceptAuditReport:
    selected_concept_ids: tuple[int, ...]
    selected_concept_names: tuple[str, ...]
    explanation: str
    names_expected_cluster: bool


@dataclass(frozen=True)
class DecoderTrainingReport:
    final_loss: float
    train_accuracy: float
    steps: int
    seed: int


@dataclass(frozen=True)
class PlantedPCDWorld:
    train_activations: t.Tensor
    heldout_activations: t.Tensor
    train_true_concepts: t.Tensor
    heldout_true_concepts: t.Tensor
    concept_directions: t.Tensor
    train_template_ids: t.Tensor
    heldout_template_ids: t.Tensor
    question_texts: tuple[str, ...]
    concept_names: tuple[str, ...]
    train_prompts: tuple[str, ...]
    heldout_prompts: tuple[str, ...]


@dataclass(frozen=True)
class PCDBaselineSweepReport:
    pcd_accuracy: float
    question_agnostic_probe_accuracy: float
    text_template_shortcut_accuracy: float
    shuffled_question_accuracy: float
    random_label_accuracy: float
    single_concept_accuracy: float
    dense_noninteraction_accuracy: float
    best_control_accuracy: float
    pcd_margin_over_best_control: float
    passes_controls: bool


@dataclass(frozen=True)
class TargetedConceptRemovalReport:
    row_index: int
    prompt: str
    question: str
    target_concept: str
    active_control_concept: str
    original_answer: int
    target_removed_answer: int
    active_control_removed_answer: int
    target_logit_delta: float
    active_control_logit_delta: float
    target_removal_changed: bool
    active_control_changed: bool
    active_control_does_less: bool


def _prediction_accuracy(logits: t.Tensor, labels: t.Tensor) -> float:
    """Return exact top-1 accuracy for answer logits."""

    if logits.ndim < 2:
        raise ValueError("logits must have shape (..., classes).")
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits prefix shape must match labels.")
    return logits.argmax(dim=-1).eq(labels).float().mean().item()


def default_pcd_questions() -> tuple[str, ...]:
    """Return the standard mini PCD behavioral question bank."""

    return (
        "Will the model answer Paris?",
        "Will the model refuse?",
        "Will the next token be a number?",
        "Is the hidden variable even?",
    )


def build_pcd_question_batch(
    activations: t.Tensor,
    question_ids: t.Tensor,
    answer_ids: t.Tensor,
    question_texts: tuple[str, ...] | None = None,
) -> PCDQuestionBatch:
    """Bundle activations with question ids, binary answers, and question text."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    if activations.shape[0] == 0:
        raise ValueError("activations must contain at least one example.")
    expected_shape = (activations.shape[0],)
    if question_ids.shape != expected_shape:
        raise ValueError("question_ids must have shape (examples,).")
    if answer_ids.shape != expected_shape:
        raise ValueError("answer_ids must have shape (examples,).")
    if question_texts is None:
        question_texts = default_pcd_questions()
    if not question_texts:
        raise ValueError("question_texts must be nonempty.")
    if int(question_ids.min().item()) < 0 or int(question_ids.max().item()) >= len(question_texts):
        raise ValueError("question_ids must index question_texts.")
    return PCDQuestionBatch(
        activations=activations,
        question_ids=question_ids.long(),
        answer_ids=answer_ids.long(),
        question_texts=question_texts,
    )


def sparse_concept_encode(
    activations: t.Tensor,
    concept_directions: t.Tensor,
    *,
    bias: t.Tensor | None = None,
    top_k: int | None = None,
    threshold: float = 0.0,
) -> t.Tensor:
    """Encode activations into sparse nonnegative concept activations."""

    if activations.ndim != 2 or concept_directions.ndim != 2:
        raise ValueError("activations and concept_directions must be rank-2 tensors.")
    if activations.shape[0] == 0 or concept_directions.shape[1] == 0:
        raise ValueError("activations and concept_directions must be nonempty.")
    if activations.shape[-1] != concept_directions.shape[0]:
        raise ValueError("activation dimension must match concept directions.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    raw_scores = activations.float() @ concept_directions.float()
    if bias is not None:
        if bias.shape != (concept_directions.shape[1],):
            raise ValueError("bias must have shape (n_concepts,).")
        raw_scores = raw_scores + bias.to(raw_scores.device)
    concepts = F.relu(raw_scores)
    if threshold > 0:
        concepts = concepts.where(concepts >= threshold, t.zeros_like(concepts))
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        top_k = min(top_k, concepts.shape[-1])
        values, indices = concepts.topk(k=top_k, dim=-1)
        concepts = t.zeros_like(concepts).scatter(dim=-1, index=indices, src=values)
    return concepts


def question_conditioned_concept_features(
    concepts: t.Tensor,
    question_ids: t.Tensor,
    question_count: int,
) -> t.Tensor:
    """Create question-specific concept slots for a sparse PCD decoder."""

    if concepts.ndim != 2:
        raise ValueError("concepts must have shape (examples, n_concepts).")
    if concepts.shape[0] == 0 or concepts.shape[1] == 0:
        raise ValueError("concepts must be nonempty.")
    if question_ids.shape != (concepts.shape[0],):
        raise ValueError("question_ids must have shape (examples,).")
    if question_count <= 0:
        raise ValueError("question_count must be positive.")
    if int(question_ids.min().item()) < 0 or int(question_ids.max().item()) >= question_count:
        raise ValueError("question_ids must be in [0, question_count).")

    n_examples, n_concepts = concepts.shape
    features = concepts.new_zeros(n_examples, question_count * n_concepts)
    offsets = question_ids.long() * n_concepts
    concept_offsets = t.arange(n_concepts, device=concepts.device)
    feature_indices = offsets[:, None] + concept_offsets[None, :]
    return features.scatter(dim=1, index=feature_indices, src=concepts.float())


def concept_sparsity_report(
    concepts: t.Tensor,
    *,
    active_threshold: float = 0.0,
    max_density: float = 0.3,
) -> ConceptSparsityReport:
    """Report concept density and mean active concept count."""

    if concepts.ndim != 2:
        raise ValueError("concepts must have shape (examples, n_concepts).")
    if concepts.shape[0] == 0 or concepts.shape[1] == 0:
        raise ValueError("concepts must be nonempty.")
    if active_threshold < 0:
        raise ValueError("active_threshold must be non-negative.")
    if not 0.0 <= max_density <= 1.0:
        raise ValueError("max_density must be between 0 and 1.")
    active = concepts.abs() > active_threshold
    mean_l0 = active.sum(dim=-1).float().mean().item()
    density = active.float().mean().item()
    return ConceptSparsityReport(
        mean_l0=mean_l0,
        density=density,
        passes_sparsity=density <= max_density,
    )


def question_conditioned_decoder_logits(
    concepts: t.Tensor,
    question_embeddings: t.Tensor,
    decoder_weight: t.Tensor,
    *,
    decoder_bias: t.Tensor | None = None,
) -> t.Tensor:
    """Decode sparse concepts and question embeddings into answer logits."""

    if concepts.ndim != 2 or question_embeddings.ndim != 2:
        raise ValueError("concepts and question_embeddings must be rank-2 tensors.")
    if concepts.shape[0] != question_embeddings.shape[0]:
        raise ValueError("concept and question batches must have the same size.")
    if decoder_weight.ndim != 2:
        raise ValueError("decoder_weight must be rank-2.")
    decoder_input = t.cat([concepts.float(), question_embeddings.float()], dim=-1)
    if decoder_input.shape[-1] != decoder_weight.shape[0]:
        raise ValueError("decoder_weight rows must match concatenated input width.")
    logits = decoder_input @ decoder_weight.float()
    if decoder_bias is not None:
        if decoder_bias.shape != (decoder_weight.shape[1],):
            raise ValueError("decoder_bias must have shape (n_answers,).")
        logits = logits + decoder_bias.to(logits.device)
    return logits


def train_question_conditioned_decoder(
    concepts: t.Tensor,
    question_embeddings: t.Tensor,
    answer_ids: t.Tensor,
    *,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
    weight_decay: float = 0.0,
) -> tuple[t.Tensor, t.Tensor, DecoderTrainingReport]:
    """Fit a small question-conditioned linear decoder with cross-entropy."""

    if concepts.ndim != 2 or question_embeddings.ndim != 2:
        raise ValueError("concepts and question_embeddings must be rank-2 tensors.")
    if concepts.shape[0] == 0:
        raise ValueError("training batches must be nonempty.")
    if concepts.shape[0] != question_embeddings.shape[0]:
        raise ValueError("concepts and question_embeddings must have the same batch size.")
    if answer_ids.shape != (concepts.shape[0],):
        raise ValueError("answer_ids must have shape (examples,).")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")

    t.manual_seed(seed)
    if concepts.device.type == "cuda":
        t.cuda.manual_seed_all(seed)
    decoder_width = concepts.shape[1] + question_embeddings.shape[1]
    decoder_weight = (0.01 * t.randn(decoder_width, 2, device=concepts.device)).requires_grad_()
    decoder_bias = t.zeros(2, device=concepts.device, requires_grad=True)
    optimizer = t.optim.Adam(
        [decoder_weight, decoder_bias],
        lr=lr,
        weight_decay=weight_decay,
    )
    answer_ids = answer_ids.long()
    final_loss = float("nan")
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = question_conditioned_decoder_logits(
            concepts,
            question_embeddings,
            decoder_weight,
            decoder_bias=decoder_bias,
        )
        loss = F.cross_entropy(logits, answer_ids)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    with t.no_grad():
        train_logits = question_conditioned_decoder_logits(
            concepts,
            question_embeddings,
            decoder_weight,
            decoder_bias=decoder_bias,
        )
        train_accuracy = _prediction_accuracy(train_logits, answer_ids)
    report = DecoderTrainingReport(
        final_loss=final_loss,
        train_accuracy=train_accuracy,
        steps=steps,
        seed=seed,
    )
    return decoder_weight.detach(), decoder_bias.detach(), report


def pcd_comparison_report(
    pcd_logits: t.Tensor,
    probe_logits: t.Tensor,
    sae_classifier_logits: t.Tensor,
    activation_oracle_logits: t.Tensor,
    answer_ids: t.Tensor,
) -> PCDComparisonReport:
    """Compare a PCD against probe, SAE-classifier, and oracle-style baselines."""

    pcd_accuracy = _prediction_accuracy(pcd_logits, answer_ids)
    probe_accuracy = _prediction_accuracy(probe_logits, answer_ids)
    sae_accuracy = _prediction_accuracy(sae_classifier_logits, answer_ids)
    oracle_accuracy = _prediction_accuracy(activation_oracle_logits, answer_ids)
    best_baseline = max(probe_accuracy, sae_accuracy, oracle_accuracy)
    return PCDComparisonReport(
        pcd_accuracy=pcd_accuracy,
        probe_accuracy=probe_accuracy,
        sae_classifier_accuracy=sae_accuracy,
        activation_oracle_accuracy=oracle_accuracy,
        best_baseline_accuracy=best_baseline,
        beats_probe=pcd_accuracy > probe_accuracy,
        beats_best_baseline=pcd_accuracy > best_baseline,
    )


def concept_stability_report(
    concept_scores_by_seed: list[t.Tensor],
    *,
    top_k: int = 3,
    min_jaccard: float = 0.5,
) -> ConceptStabilityReport:
    """Check whether top concepts are stable across random seeds."""

    if len(concept_scores_by_seed) == 0:
        raise ValueError("concept_scores_by_seed must be nonempty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0.0 <= min_jaccard <= 1.0:
        raise ValueError("min_jaccard must be between 0 and 1.")

    top_concepts: list[tuple[int, ...]] = []
    for scores in concept_scores_by_seed:
        flat_scores = scores.flatten().float()
        k = min(top_k, flat_scores.numel())
        indices = flat_scores.topk(k=k).indices.tolist()
        top_concepts.append(tuple(int(index) for index in indices))

    overlaps = []
    for i, left in enumerate(top_concepts):
        for right in top_concepts[i + 1 :]:
            left_set = set(left)
            right_set = set(right)
            overlaps.append(len(left_set & right_set) / len(left_set | right_set))
    mean_jaccard = sum(overlaps) / len(overlaps) if overlaps else 1.0
    return ConceptStabilityReport(
        top_concepts_by_seed=tuple(top_concepts),
        mean_pairwise_jaccard=mean_jaccard,
        stable=mean_jaccard >= min_jaccard,
    )


def concept_removal_report(
    original_logits: t.Tensor,
    top_removed_logits: t.Tensor,
    random_removed_logits: t.Tensor,
    *,
    target_answer_id: int | None = None,
) -> ConceptRemovalReport:
    """Check that top removal beats a lower-impact comparison removal."""

    if original_logits.ndim != 1:
        raise ValueError("original_logits must be rank-1.")
    if top_removed_logits.shape != original_logits.shape:
        raise ValueError("top_removed_logits must match original_logits.")
    if random_removed_logits.shape != original_logits.shape:
        raise ValueError("random_removed_logits must match original_logits.")
    if target_answer_id is None:
        target_answer_id = int(original_logits.argmax().item())
    if not 0 <= target_answer_id < original_logits.shape[0]:
        raise ValueError("target_answer_id is out of range.")

    original_answer = int(original_logits.argmax().item())
    top_removed_answer = int(top_removed_logits.argmax().item())
    random_removed_answer = int(random_removed_logits.argmax().item())
    top_delta = original_logits[target_answer_id] - top_removed_logits[target_answer_id]
    random_delta = original_logits[target_answer_id] - random_removed_logits[target_answer_id]
    return ConceptRemovalReport(
        original_answer=original_answer,
        top_removed_answer=top_removed_answer,
        random_removed_answer=random_removed_answer,
        top_removal_changed=original_answer != top_removed_answer,
        random_removal_changed=original_answer != random_removed_answer,
        top_removal_delta=float(top_delta.item()),
        random_removal_delta=float(random_delta.item()),
        random_removal_does_less=abs(random_delta.item()) < abs(top_delta.item()),
    )


def concept_audit_report(
    concept_scores: t.Tensor,
    concept_names: list[str],
    expected_cluster_terms: list[str],
    *,
    top_k: int = 2,
) -> ConceptAuditReport:
    """Select top concepts and check whether their names match an expected cluster."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if concept_scores.numel() == 0:
        raise ValueError("concept_scores must be nonempty.")
    if concept_scores.numel() != len(concept_names):
        raise ValueError("concept_scores and concept_names must align.")
    if len(expected_cluster_terms) == 0:
        raise ValueError("expected_cluster_terms must be nonempty.")
    flat_scores = concept_scores.flatten().float()
    k = min(top_k, flat_scores.numel())
    selected_ids = tuple(int(index) for index in flat_scores.topk(k=k).indices.tolist())
    selected_names = tuple(concept_names[index] for index in selected_ids)
    normalized_terms = [term.lower() for term in expected_cluster_terms]
    matched = any(
        term in name.lower()
        for name in selected_names
        for term in normalized_terms
    )
    explanation = "Top concepts: " + ", ".join(selected_names)
    return ConceptAuditReport(
        selected_concept_ids=selected_ids,
        selected_concept_names=selected_names,
        explanation=explanation,
        names_expected_cluster=matched,
    )


def _dense_orthonormal_columns(
    *,
    d_model: int,
    columns: int,
    seed: int,
) -> t.Tensor:
    if d_model < columns:
        raise ValueError("d_model must be at least as large as the requested column count.")
    generator = t.Generator(device="cpu").manual_seed(seed)
    matrix = t.randn(d_model, d_model, generator=generator)
    q, _r = t.linalg.qr(matrix)
    return q[:, :columns].float()


def make_planted_pcd_world(
    *,
    seed: int = 0,
    d_model: int = 12,
) -> PlantedPCDWorld:
    """Create an exact sparse-concept world with dense-looking activations."""

    question_count = len(PLANTED_PCD_QUESTIONS)
    target_concept_count = len(PLANTED_QUESTION_TO_CONCEPT)
    concept_count = len(PLANTED_CONCEPT_NAMES)
    if d_model < concept_count + 2:
        raise ValueError("d_model must leave room for planted concepts and null directions.")

    rows = []
    for index in range(2**target_concept_count):
        rows.append([(index >> bit) & 1 for bit in range(target_concept_count)])
    base_targets = t.tensor(rows, dtype=t.float32).repeat_interleave(2, dim=0)
    replicate_ids = t.arange(base_targets.shape[0]) % 2

    concept_basis = _dense_orthonormal_columns(
        d_model=d_model,
        columns=concept_count + 2,
        seed=seed,
    )
    concept_directions = concept_basis[:, :concept_count]
    null_directions = concept_basis[:, concept_count : concept_count + 2]

    def build_split(split_name: str, offset: int) -> tuple[t.Tensor, t.Tensor, t.Tensor, tuple[str, ...]]:
        template_ids = (replicate_ids + offset) % question_count
        nuisance = t.stack(
            [
                replicate_ids.float(),
                1.0 - replicate_ids.float(),
            ],
            dim=1,
        )
        concepts = t.cat([base_targets, nuisance], dim=1)
        null_features = t.stack(
            [
                ((t.arange(base_targets.shape[0]) + offset) % 5).float() / 10.0,
                ((t.arange(base_targets.shape[0]) * 2 + offset) % 7).float() / 10.0,
            ],
            dim=1,
        )
        activations = concepts @ concept_directions.T + 0.1 * null_features @ null_directions.T
        prompts = tuple(
            f"{split_name} latent card {i:02d} / template {int(template_id.item())}"
            for i, template_id in enumerate(template_ids)
        )
        return activations.float(), concepts.float(), template_ids.long(), prompts

    train_activations, train_concepts, train_templates, train_prompts = build_split(
        "train",
        offset=0,
    )
    heldout_activations, heldout_concepts, heldout_templates, heldout_prompts = build_split(
        "heldout",
        offset=1,
    )

    recovered_train = train_activations @ concept_directions
    recovered_heldout = heldout_activations @ concept_directions
    if not t.allclose(recovered_train, train_concepts, atol=1e-5, rtol=1e-5):
        raise RuntimeError("planted train concepts are not exactly recoverable.")
    if not t.allclose(recovered_heldout, heldout_concepts, atol=1e-5, rtol=1e-5):
        raise RuntimeError("planted held-out concepts are not exactly recoverable.")

    return PlantedPCDWorld(
        train_activations=train_activations,
        heldout_activations=heldout_activations,
        train_true_concepts=train_concepts,
        heldout_true_concepts=heldout_concepts,
        concept_directions=concept_directions,
        train_template_ids=train_templates,
        heldout_template_ids=heldout_templates,
        question_texts=PLANTED_PCD_QUESTIONS,
        concept_names=PLANTED_CONCEPT_NAMES,
        train_prompts=train_prompts,
        heldout_prompts=heldout_prompts,
    )


def build_planted_question_rows(
    concepts: t.Tensor,
    *,
    question_to_concept: tuple[int, ...] = PLANTED_QUESTION_TO_CONCEPT,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Repeat concept rows under each question and label with the planted answer."""

    if concepts.ndim != 2:
        raise ValueError("concepts must have shape (examples, n_concepts).")
    if concepts.shape[0] == 0 or concepts.shape[1] == 0:
        raise ValueError("concepts must be nonempty.")
    if len(question_to_concept) == 0:
        raise ValueError("question_to_concept must be nonempty.")
    if min(question_to_concept) < 0 or max(question_to_concept) >= concepts.shape[1]:
        raise ValueError("question_to_concept must index concept columns.")

    question_count = len(question_to_concept)
    row_concepts = concepts.repeat_interleave(question_count, dim=0).float()
    question_ids = t.arange(question_count, device=concepts.device).repeat(concepts.shape[0])
    question_embeddings = t.eye(question_count, device=concepts.device).repeat(concepts.shape[0], 1)
    answer_matrix = concepts[:, list(question_to_concept)].round().long().clamp(0, 1)
    answer_ids = answer_matrix.reshape(-1)
    return row_concepts, question_embeddings, question_ids.long(), answer_ids.long()


def _fit_eval_decoder(
    train_features: t.Tensor,
    train_questions: t.Tensor,
    train_answers: t.Tensor,
    heldout_features: t.Tensor,
    heldout_questions: t.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
    weight_decay: float = 0.0,
) -> tuple[t.Tensor, t.Tensor, DecoderTrainingReport, t.Tensor]:
    decoder_weight, decoder_bias, report = train_question_conditioned_decoder(
        train_features,
        train_questions,
        train_answers,
        steps=steps,
        lr=lr,
        seed=seed,
        weight_decay=weight_decay,
    )
    heldout_logits = question_conditioned_decoder_logits(
        heldout_features,
        heldout_questions,
        decoder_weight,
        decoder_bias=decoder_bias,
    )
    return decoder_weight, decoder_bias, report, heldout_logits


def _template_features(template_ids: t.Tensor, *, question_count: int) -> t.Tensor:
    if template_ids.ndim != 1:
        raise ValueError("template_ids must have shape (examples,).")
    repeated_templates = template_ids.long().repeat_interleave(question_count)
    return F.one_hot(repeated_templates, num_classes=question_count).float()


def concept_question_weight_heatmap(
    decoder_weight: t.Tensor,
    *,
    question_count: int,
    concept_count: int,
) -> t.Tensor:
    """Return class-1 minus class-0 interaction weights as [question, concept]."""

    if decoder_weight.ndim != 2 or decoder_weight.shape[1] < 2:
        raise ValueError("decoder_weight must have shape (features, at least two classes).")
    interaction_width = question_count * concept_count
    if decoder_weight.shape[0] < interaction_width:
        raise ValueError("decoder_weight is too narrow for the requested interaction map.")
    weights = decoder_weight[:interaction_width, 1] - decoder_weight[:interaction_width, 0]
    return weights.reshape(question_count, concept_count)


def pcd_baseline_sweep_report(
    *,
    pcd_logits: t.Tensor,
    question_agnostic_probe_logits: t.Tensor,
    text_template_shortcut_logits: t.Tensor,
    shuffled_question_logits: t.Tensor,
    random_label_logits: t.Tensor,
    single_concept_logits: t.Tensor,
    dense_noninteraction_logits: t.Tensor,
    answer_ids: t.Tensor,
    min_margin: float = 0.20,
) -> PCDBaselineSweepReport:
    """Score the PCD and every control from independent logits."""

    pcd_accuracy = _prediction_accuracy(pcd_logits, answer_ids)
    probe_accuracy = _prediction_accuracy(question_agnostic_probe_logits, answer_ids)
    text_accuracy = _prediction_accuracy(text_template_shortcut_logits, answer_ids)
    shuffled_accuracy = _prediction_accuracy(shuffled_question_logits, answer_ids)
    random_label_accuracy = _prediction_accuracy(random_label_logits, answer_ids)
    single_concept_accuracy = _prediction_accuracy(single_concept_logits, answer_ids)
    dense_accuracy = _prediction_accuracy(dense_noninteraction_logits, answer_ids)
    best_control = max(
        probe_accuracy,
        text_accuracy,
        shuffled_accuracy,
        random_label_accuracy,
        single_concept_accuracy,
        dense_accuracy,
    )
    margin = pcd_accuracy - best_control
    return PCDBaselineSweepReport(
        pcd_accuracy=pcd_accuracy,
        question_agnostic_probe_accuracy=probe_accuracy,
        text_template_shortcut_accuracy=text_accuracy,
        shuffled_question_accuracy=shuffled_accuracy,
        random_label_accuracy=random_label_accuracy,
        single_concept_accuracy=single_concept_accuracy,
        dense_noninteraction_accuracy=dense_accuracy,
        best_control_accuracy=best_control,
        pcd_margin_over_best_control=margin,
        passes_controls=pcd_accuracy == 1.0 and margin >= min_margin,
    )


def targeted_concept_removal_report(
    row_concepts: t.Tensor,
    question_ids: t.Tensor,
    question_embeddings: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor,
    *,
    prompts: tuple[str, ...],
    question_texts: tuple[str, ...],
    concept_names: tuple[str, ...],
    example_index: int,
    question_id: int,
    target_concept_id: int,
    active_control_concept_id: int,
) -> TargetedConceptRemovalReport:
    """Remove the named concept and compare against a matched active control concept."""

    question_count = len(question_texts)
    n_concepts = len(concept_names)
    row_index = example_index * question_count + question_id
    if row_concepts.ndim != 2:
        raise ValueError("row_concepts must have shape (rows, concepts).")
    if row_index < 0 or row_index >= row_concepts.shape[0]:
        raise ValueError("requested removal row is out of range.")
    if row_concepts.shape[1] != n_concepts:
        raise ValueError("row_concepts width must match concept_names.")
    if not 0 <= target_concept_id < n_concepts:
        raise ValueError("target_concept_id is out of range.")
    if not 0 <= active_control_concept_id < n_concepts:
        raise ValueError("active_control_concept_id is out of range.")
    if row_concepts[row_index, target_concept_id].item() <= 0:
        raise ValueError("target concept must be active in the selected row.")
    if row_concepts[row_index, active_control_concept_id].item() <= 0:
        raise ValueError("active control concept must be active in the selected row.")

    def logits_after_removal(concept_id: int | None) -> t.Tensor:
        patched = row_concepts[row_index : row_index + 1].clone()
        if concept_id is not None:
            patched[0, concept_id] = 0.0
        features = question_conditioned_concept_features(
            patched,
            question_ids[row_index : row_index + 1],
            question_count,
        )
        return question_conditioned_decoder_logits(
            features,
            question_embeddings[row_index : row_index + 1],
            decoder_weight,
            decoder_bias=decoder_bias,
        )[0]

    original_logits = logits_after_removal(None)
    target_removed_logits = logits_after_removal(target_concept_id)
    active_removed_logits = logits_after_removal(active_control_concept_id)
    original_answer = int(original_logits.argmax().item())
    target_removed_answer = int(target_removed_logits.argmax().item())
    active_removed_answer = int(active_removed_logits.argmax().item())
    target_delta = original_logits[original_answer] - target_removed_logits[original_answer]
    active_delta = original_logits[original_answer] - active_removed_logits[original_answer]
    return TargetedConceptRemovalReport(
        row_index=row_index,
        prompt=prompts[example_index],
        question=question_texts[question_id],
        target_concept=concept_names[target_concept_id],
        active_control_concept=concept_names[active_control_concept_id],
        original_answer=original_answer,
        target_removed_answer=target_removed_answer,
        active_control_removed_answer=active_removed_answer,
        target_logit_delta=float(target_delta.item()),
        active_control_logit_delta=float(active_delta.item()),
        target_removal_changed=original_answer != target_removed_answer,
        active_control_changed=original_answer != active_removed_answer,
        active_control_does_less=abs(float(active_delta.item())) < abs(float(target_delta.item())),
    )


def run_planted_pcd_experiment(
    *,
    seed: int = 0,
    steps: int = 450,
    lr: float = 0.08,
) -> dict:
    """Train/evaluate PCD and controls on the exact planted sparse-concept world."""

    world = make_planted_pcd_world(seed=seed)
    train_encoded = sparse_concept_encode(
        world.train_activations,
        world.concept_directions,
        threshold=0.5,
    )
    heldout_encoded = sparse_concept_encode(
        world.heldout_activations,
        world.concept_directions,
        threshold=0.5,
    )
    train_recovery_error = float((train_encoded - world.train_true_concepts).abs().max().item())
    heldout_recovery_error = float((heldout_encoded - world.heldout_true_concepts).abs().max().item())

    train_rows, train_questions, train_question_ids, train_answers = build_planted_question_rows(
        train_encoded,
    )
    heldout_rows, heldout_questions, heldout_question_ids, heldout_answers = (
        build_planted_question_rows(heldout_encoded)
    )
    question_count = len(world.question_texts)
    concept_count = len(world.concept_names)

    train_features = question_conditioned_concept_features(
        train_rows,
        train_question_ids,
        question_count,
    )
    heldout_features = question_conditioned_concept_features(
        heldout_rows,
        heldout_question_ids,
        question_count,
    )
    decoder_weight, decoder_bias, pcd_training, pcd_logits = _fit_eval_decoder(
        train_features,
        train_questions,
        train_answers,
        heldout_features,
        heldout_questions,
        steps=steps,
        lr=lr,
        seed=seed,
    )

    zero_train_questions = t.zeros(train_rows.shape[0], 0)
    zero_heldout_questions = t.zeros(heldout_rows.shape[0], 0)
    _probe_weight, _probe_bias, probe_training, probe_logits = _fit_eval_decoder(
        train_rows,
        zero_train_questions,
        train_answers,
        heldout_rows,
        zero_heldout_questions,
        steps=steps,
        lr=lr,
        seed=seed + 1,
    )

    train_template_features = _template_features(world.train_template_ids, question_count=question_count)
    heldout_template_features = _template_features(world.heldout_template_ids, question_count=question_count)
    _text_weight, _text_bias, text_training, text_logits = _fit_eval_decoder(
        train_template_features,
        train_questions,
        train_answers,
        heldout_template_features,
        heldout_questions,
        steps=steps,
        lr=lr,
        seed=seed + 2,
    )

    shuffled_question_ids = heldout_question_ids.roll(shifts=1, dims=0)
    shuffled_questions = heldout_questions.roll(shifts=1, dims=0)
    shuffled_features = question_conditioned_concept_features(
        heldout_rows,
        shuffled_question_ids,
        question_count,
    )
    shuffled_logits = question_conditioned_decoder_logits(
        shuffled_features,
        shuffled_questions,
        decoder_weight,
        decoder_bias=decoder_bias,
    )

    generator = t.Generator(device="cpu").manual_seed(seed + 3)
    random_train_answers = train_answers[t.randperm(train_answers.numel(), generator=generator)]
    _rand_weight, _rand_bias, random_training, random_label_logits = _fit_eval_decoder(
        train_features,
        train_questions,
        random_train_answers,
        heldout_features,
        heldout_questions,
        steps=steps,
        lr=lr,
        seed=seed + 3,
    )

    single_train_features = question_conditioned_concept_features(
        train_rows[:, :1],
        train_question_ids,
        question_count,
    )
    single_heldout_features = question_conditioned_concept_features(
        heldout_rows[:, :1],
        heldout_question_ids,
        question_count,
    )
    _single_weight, _single_bias, single_training, single_logits = _fit_eval_decoder(
        single_train_features,
        train_questions,
        train_answers,
        single_heldout_features,
        heldout_questions,
        steps=steps,
        lr=lr,
        seed=seed + 4,
    )

    train_dense_rows = world.train_activations.repeat_interleave(question_count, dim=0)
    heldout_dense_rows = world.heldout_activations.repeat_interleave(question_count, dim=0)
    _dense_weight, _dense_bias, dense_training, dense_logits = _fit_eval_decoder(
        train_dense_rows,
        train_questions,
        train_answers,
        heldout_dense_rows,
        heldout_questions,
        steps=steps,
        lr=lr,
        seed=seed + 5,
    )

    baseline_report = pcd_baseline_sweep_report(
        pcd_logits=pcd_logits,
        question_agnostic_probe_logits=probe_logits,
        text_template_shortcut_logits=text_logits,
        shuffled_question_logits=shuffled_logits,
        random_label_logits=random_label_logits,
        single_concept_logits=single_logits,
        dense_noninteraction_logits=dense_logits,
        answer_ids=heldout_answers,
    )
    heatmap = concept_question_weight_heatmap(
        decoder_weight,
        question_count=question_count,
        concept_count=concept_count,
    )
    removal_example_index = int(
        (heldout_encoded[:, : len(PLANTED_QUESTION_TO_CONCEPT)] > 0).all(dim=1).nonzero()[0].item()
    )
    removal = targeted_concept_removal_report(
        heldout_rows,
        heldout_question_ids,
        heldout_questions,
        decoder_weight,
        decoder_bias,
        prompts=world.heldout_prompts,
        question_texts=world.question_texts,
        concept_names=world.concept_names,
        example_index=removal_example_index,
        question_id=0,
        target_concept_id=0,
        active_control_concept_id=1,
    )
    baseline_names = (
        "PCD",
        "question-agnostic probe",
        "text/template shortcut",
        "shuffled question",
        "random label",
        "single concept",
        "dense non-interaction",
    )
    baseline_accuracies = (
        baseline_report.pcd_accuracy,
        baseline_report.question_agnostic_probe_accuracy,
        baseline_report.text_template_shortcut_accuracy,
        baseline_report.shuffled_question_accuracy,
        baseline_report.random_label_accuracy,
        baseline_report.single_concept_accuracy,
        baseline_report.dense_noninteraction_accuracy,
    )
    prediction_rows = planted_prediction_rows(
        world,
        heldout_answers=heldout_answers,
        pcd_logits=pcd_logits,
        probe_logits=probe_logits,
        text_template_logits=text_logits,
        dense_noninteraction_logits=dense_logits,
    )
    return {
        "world": world,
        "train_encoded": train_encoded,
        "heldout_encoded": heldout_encoded,
        "train_recovery_error": train_recovery_error,
        "heldout_recovery_error": heldout_recovery_error,
        "train_rows": train_rows,
        "heldout_rows": heldout_rows,
        "train_question_ids": train_question_ids,
        "heldout_question_ids": heldout_question_ids,
        "train_questions": train_questions,
        "heldout_questions": heldout_questions,
        "train_answers": train_answers,
        "heldout_answers": heldout_answers,
        "decoder_weight": decoder_weight,
        "decoder_bias": decoder_bias,
        "pcd_training": pcd_training,
        "probe_training": probe_training,
        "text_training": text_training,
        "random_training": random_training,
        "single_training": single_training,
        "dense_training": dense_training,
        "pcd_logits": pcd_logits,
        "probe_logits": probe_logits,
        "text_template_logits": text_logits,
        "shuffled_question_logits": shuffled_logits,
        "random_label_logits": random_label_logits,
        "single_concept_logits": single_logits,
        "dense_noninteraction_logits": dense_logits,
        "baseline_report": baseline_report,
        "baseline_names": baseline_names,
        "baseline_accuracies": baseline_accuracies,
        "heatmap": heatmap,
        "removal_report": removal,
        "prediction_rows": prediction_rows,
    }


def planted_prediction_rows(
    world: PlantedPCDWorld,
    *,
    heldout_answers: t.Tensor,
    pcd_logits: t.Tensor,
    probe_logits: t.Tensor,
    text_template_logits: t.Tensor,
    dense_noninteraction_logits: t.Tensor,
) -> list[dict[str, object]]:
    """Create a learner-facing held-out table for the planted world."""

    question_count = len(world.question_texts)
    rows: list[dict[str, object]] = []
    pcd_preds = pcd_logits.argmax(dim=-1)
    probe_preds = probe_logits.argmax(dim=-1)
    text_preds = text_template_logits.argmax(dim=-1)
    dense_preds = dense_noninteraction_logits.argmax(dim=-1)
    for row_index, answer in enumerate(heldout_answers.tolist()):
        example_index = row_index // question_count
        question_id = row_index % question_count
        rows.append(
            {
                "prompt": world.heldout_prompts[example_index],
                "question": world.question_texts[question_id],
                "answer": int(answer),
                "pcd_pred": int(pcd_preds[row_index].item()),
                "probe_pred": int(probe_preds[row_index].item()),
                "text_template_pred": int(text_preds[row_index].item()),
                "dense_noninteraction_pred": int(dense_preds[row_index].item()),
            }
        )
    return rows


def gelu1l_prompt_prediction_table(report: dict) -> list[dict[str, object]]:
    """Extract direct gelu-1l rows when present; otherwise make an honest aggregate table."""

    metrics = report.get("metrics", {}).get("gpu_test", report)
    direct_rows = metrics.get("eval_prediction_rows")
    if direct_rows is not None:
        return list(direct_rows)

    required = {
        "pcd_accuracy",
        "pcd_row_count",
        "eval_example_count",
        "question_count",
        "preflight_passed",
    }
    missing = sorted(key for key in required if key not in metrics)
    if missing:
        raise ValueError(f"report is missing required aggregate fields: {missing}")
    if metrics["eval_example_count"] != len(TL_EVAL_EXAMPLES):
        raise ValueError("aggregate report eval count does not match the pinned prompt list.")
    if metrics["question_count"] != len(TL_PCD_QUESTIONS):
        raise ValueError("aggregate report question count does not match the pinned question bank.")
    if metrics["pcd_row_count"] != len(TL_EVAL_EXAMPLES) * len(TL_PCD_QUESTIONS):
        raise ValueError("aggregate report row count does not match prompts times questions.")
    if metrics["pcd_accuracy"] != 1.0:
        raise ValueError("cannot infer per-row PCD correctness unless aggregate PCD accuracy is 1.0.")

    rows = []
    for prompt_index, (prompt, label) in enumerate(TL_EVAL_EXAMPLES):
        for question_id, question in enumerate(TL_PCD_QUESTIONS):
            answer = int(
                (question_id in (0, 2) and label == "surface")
                or (question_id in (1, 3) and label == "motion")
            )
            rows.append(
                {
                    "prompt": prompt,
                    "latent_label": label,
                    "question": question,
                    "answer": answer,
                    "pcd_pred": answer,
                    "source": "aggregate_report_pcd_accuracy_1.0_no_baseline_row_logits",
                }
            )
    return rows


def batch_smoke_test() -> dict:
    activations = t.eye(4)
    question_ids = t.tensor([0, 1, 2, 3])
    answer_ids = t.tensor([1, 0, 1, 0])
    batch = build_pcd_question_batch(activations, question_ids, answer_ids)
    return {
        "activation_shape": list(batch.activations.shape),
        "num_questions": len(default_pcd_questions()),
        "answer_ids": batch.answer_ids.tolist(),
    }


def sparse_encoding_smoke_test() -> dict:
    activations = t.eye(3)
    concept_directions = t.eye(3)
    concepts = sparse_concept_encode(activations, concept_directions, top_k=1)
    sparsity = concept_sparsity_report(concepts, max_density=0.34)
    return {
        "concepts": concepts.tolist(),
        "sparsity": sparsity.__dict__,
    }


def decoder_smoke_test() -> list[list[float]]:
    concepts = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    question_embeddings = t.tensor([[0.0, 1.0], [1.0, 0.0]])
    decoder_weight = t.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    logits = question_conditioned_decoder_logits(
        concepts,
        question_embeddings,
        decoder_weight,
    )
    return logits.tolist()


def decoder_training_smoke_test() -> dict:
    concepts = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    repeated_concepts = concepts.repeat_interleave(2, dim=0)
    question_ids = t.tensor([0, 1, 0, 1])
    question_embeddings = t.eye(2).repeat(2, 1)
    answer_ids = t.tensor([1, 0, 0, 1])
    conditioned = question_conditioned_concept_features(
        repeated_concepts,
        question_ids,
        question_count=2,
    )
    decoder_weight, decoder_bias, report = train_question_conditioned_decoder(
        conditioned,
        question_embeddings,
        answer_ids,
        steps=250,
        lr=0.1,
        seed=0,
    )
    logits = question_conditioned_decoder_logits(
        conditioned,
        question_embeddings,
        decoder_weight,
        decoder_bias=decoder_bias,
    )
    return {
        "conditioned_shape": list(conditioned.shape),
        "train_accuracy": report.train_accuracy,
        "final_loss": report.final_loss,
        "predictions": logits.argmax(dim=-1).tolist(),
        "answer_ids": answer_ids.tolist(),
    }


def comparison_smoke_test() -> dict:
    answer_ids = t.tensor([0, 1, 0, 1])
    pcd_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    probe_logits = t.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    sae_logits = probe_logits.clone()
    oracle_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    return pcd_comparison_report(
        pcd_logits,
        probe_logits,
        sae_logits,
        oracle_logits,
        answer_ids,
    ).__dict__


def stability_smoke_test() -> dict:
    scores_by_seed = [
        t.tensor([0.9, 0.8, 0.1, 0.0]),
        t.tensor([0.8, 0.7, 0.2, 0.0]),
        t.tensor([0.95, 0.85, 0.05, 0.0]),
    ]
    return concept_stability_report(scores_by_seed, top_k=2, min_jaccard=0.75).__dict__


def removal_smoke_test() -> dict:
    original_logits = t.tensor([3.0, 1.0])
    top_removed_logits = t.tensor([0.0, 2.0])
    random_removed_logits = t.tensor([2.5, 1.0])
    return concept_removal_report(
        original_logits,
        top_removed_logits,
        random_removed_logits,
    ).__dict__


def audit_smoke_test() -> dict:
    concept_scores = t.tensor([0.1, 0.9, 0.8])
    concept_names = ["syntax feature", "refusal feature", "safety refusal"]
    return concept_audit_report(
        concept_scores,
        concept_names,
        ["refusal"],
        top_k=2,
    ).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    planted = run_planted_pcd_experiment(seed=0, steps=350, lr=0.08)
    baseline_report = planted["baseline_report"]
    removal_report = planted["removal_report"]
    return {
        "batch": batch_smoke_test(),
        "sparse_encoding": sparse_encoding_smoke_test(),
        "decoder": decoder_smoke_test(),
        "decoder_training": decoder_training_smoke_test(),
        "comparison": comparison_smoke_test(),
        "stability": stability_smoke_test(),
        "removal": removal_smoke_test(),
        "audit": audit_smoke_test(),
        "planted": {
            "train_examples": len(planted["world"].train_prompts),
            "heldout_examples": len(planted["world"].heldout_prompts),
            "question_count": len(planted["world"].question_texts),
            "concept_names": list(planted["world"].concept_names),
            "train_recovery_error": planted["train_recovery_error"],
            "heldout_recovery_error": planted["heldout_recovery_error"],
            "heatmap_shape": list(planted["heatmap"].shape),
            "heatmap": planted["heatmap"].tolist(),
            "baselines": baseline_report.__dict__,
            "baseline_names": list(planted["baseline_names"]),
            "baseline_accuracies": list(planted["baseline_accuracies"]),
            "removal": removal_report.__dict__,
            "prediction_rows_preview": planted["prediction_rows"][:8],
        },
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
    return cache[TL_RESID_POST_HOOK][0, -1].detach().float(), logits[0, -1].detach()


def _residual_direction(model) -> tuple[t.Tensor, int, int]:
    positive_residual, positive_logits = _final_residual_and_logits(model, TL_POSITIVE_ANCHOR)
    negative_residual, negative_logits = _final_residual_and_logits(model, TL_NEGATIVE_ANCHOR)
    direction = positive_residual - negative_residual
    direction = direction / direction.norm()
    return direction, int(positive_logits.argmax().item()), int(negative_logits.argmax().item())


def _collect_residuals(model, examples: list[tuple[str, str]]) -> tuple[list[str], list[str], t.Tensor]:
    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]
    residuals = [_final_residual_and_logits(model, text)[0] for text in texts]
    return texts, labels, t.stack(residuals)


def _concept_directions(
    train_residuals: t.Tensor,
    direction: t.Tensor,
) -> tuple[t.Tensor, list[str], t.Tensor]:
    mean_residual = train_residuals.mean(dim=0)
    _, _, vh = t.linalg.svd(train_residuals - mean_residual, full_matrices=False)
    pcs: list[t.Tensor] = []
    for vector in vh:
        orthogonal = vector.clone()
        for component in [direction, *pcs]:
            orthogonal = orthogonal - (orthogonal @ component) * component
        norm = orthogonal.norm()
        if norm > 1e-5:
            pcs.append(orthogonal / norm)
        if len(pcs) >= 3:
            break
    if len(pcs) < 3:
        raise RuntimeError("could not build the required concept directions.")
    directions = [direction, -direction]
    for pc in pcs:
        directions.extend([pc, -pc])
    names = [
        "surface_direction",
        "motion_direction",
        "pc1_positive",
        "pc1_negative",
        "pc2_positive",
        "pc2_negative",
        "pc3_positive",
        "pc3_negative",
    ]
    return t.stack(directions, dim=1), names, mean_residual


def _build_question_rows(
    concepts: t.Tensor,
    labels: list[str],
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    row_concepts = []
    question_embeddings = []
    question_ids = []
    answer_ids = []
    eye = t.eye(len(TL_PCD_QUESTIONS), device=concepts.device)
    for row_index, label in enumerate(labels):
        for question_id in range(len(TL_PCD_QUESTIONS)):
            row_concepts.append(concepts[row_index])
            question_embeddings.append(eye[question_id])
            question_ids.append(question_id)
            answer_ids.append(
                1
                if (
                    (question_id in (0, 2) and label == "surface")
                    or (question_id in (1, 3) and label == "motion")
                )
                else 0
            )
    question_ids_tensor = t.tensor(question_ids, device=concepts.device)
    repeated_concepts = t.stack(row_concepts)
    return (
        repeated_concepts,
        t.stack(question_embeddings),
        question_ids_tensor,
        t.tensor(answer_ids, device=concepts.device),
    )


def _fit_linear_decoder(
    train_features: t.Tensor,
    train_questions: t.Tensor,
    train_answers: t.Tensor,
    eval_features: t.Tensor,
    eval_questions: t.Tensor,
    *,
    steps: int = 500,
    lr: float = 0.08,
    seed: int = 0,
) -> tuple[t.Tensor, t.Tensor, DecoderTrainingReport, t.Tensor]:
    decoder_weight, decoder_bias, report = train_question_conditioned_decoder(
        train_features,
        train_questions,
        train_answers,
        steps=steps,
        lr=lr,
        seed=seed,
    )
    eval_logits = question_conditioned_decoder_logits(
        eval_features,
        eval_questions,
        decoder_weight,
        decoder_bias=decoder_bias,
    )
    return decoder_weight, decoder_bias, report, eval_logits


def _question_agnostic_probe_logits(concepts: t.Tensor, labels: list[str]) -> t.Tensor:
    _ = labels
    surface_score = concepts[:, 0] - concepts[:, 1]
    rows = []
    for score in surface_score:
        for _question_id in range(len(TL_PCD_QUESTIONS)):
            rows.append(t.stack([-score, score]))
    return t.stack(rows)


def _single_concept_classifier_logits(concepts: t.Tensor) -> t.Tensor:
    rows = []
    for concept_row in concepts:
        for _question_id in range(len(TL_PCD_QUESTIONS)):
            rows.append(t.stack([t.zeros((), device=concepts.device), concept_row[0]]))
    return t.stack(rows)


def _non_interaction_question_baseline_logits(
    concepts: t.Tensor,
    question_embeddings: t.Tensor,
) -> t.Tensor:
    repeated = concepts.repeat_interleave(len(TL_PCD_QUESTIONS), dim=0)
    decoder_weight = t.zeros(
        repeated.shape[1] + question_embeddings.shape[1],
        2,
        device=concepts.device,
    )
    decoder_weight[0, 1] = 1.0
    decoder_weight[1, 0] = 1.0
    return question_conditioned_decoder_logits(repeated, question_embeddings, decoder_weight)


def _concept_scores_from_decoder(
    decoder_weight: t.Tensor,
    conditioned_rows: t.Tensor,
    *,
    n_concepts: int,
    question_count: int,
) -> t.Tensor:
    interaction_weights = decoder_weight[: question_count * n_concepts].abs().sum(dim=1)
    interaction_weights = interaction_weights.reshape(question_count, n_concepts)
    mean_activation = conditioned_rows.abs().mean(dim=0).reshape(question_count, n_concepts)
    return (interaction_weights * mean_activation).sum(dim=0)


def _active_concept_removal_case(
    row_concepts: t.Tensor,
    question_ids: t.Tensor,
    question_embeddings: t.Tensor,
    answer_ids: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor,
) -> tuple[ConceptRemovalReport, int, int, int, bool]:
    question_count = len(TL_PCD_QUESTIONS)
    n_concepts = row_concepts.shape[1]
    original_features = question_conditioned_concept_features(
        row_concepts,
        question_ids,
        question_count,
    )
    original_logits = question_conditioned_decoder_logits(
        original_features,
        question_embeddings,
        decoder_weight,
        decoder_bias=decoder_bias,
    )

    for row_index, concept_row in enumerate(row_concepts):
        active_ids = concept_row.nonzero().flatten().tolist()
        if len(active_ids) < 2:
            continue
        target_answer_id = int(answer_ids[row_index].item())
        other_answer_id = 1 - target_answer_id
        question_id = int(question_ids[row_index].item())
        slot_offset = question_id * n_concepts
        margins = {}
        for concept_id in active_ids:
            feature_value = concept_row[concept_id]
            target_weight = decoder_weight[slot_offset + concept_id, target_answer_id]
            other_weight = decoder_weight[slot_offset + concept_id, other_answer_id]
            margins[int(concept_id)] = abs(float((feature_value * (target_weight - other_weight)).item()))
        ranked = sorted(margins, key=margins.get, reverse=True)
        top_id = ranked[0]
        control_id = ranked[-1]
        if control_id == top_id:
            continue

        top_row = row_concepts[row_index : row_index + 1].clone()
        top_row[0, top_id] = 0.0
        random_row = row_concepts[row_index : row_index + 1].clone()
        random_row[0, control_id] = 0.0
        top_features = question_conditioned_concept_features(
            top_row,
            question_ids[row_index : row_index + 1],
            question_count,
        )
        random_features = question_conditioned_concept_features(
            random_row,
            question_ids[row_index : row_index + 1],
            question_count,
        )
        top_removed_logits = question_conditioned_decoder_logits(
            top_features,
            question_embeddings[row_index : row_index + 1],
            decoder_weight,
            decoder_bias=decoder_bias,
        )[0]
        random_removed_logits = question_conditioned_decoder_logits(
            random_features,
            question_embeddings[row_index : row_index + 1],
            decoder_weight,
            decoder_bias=decoder_bias,
        )[0]
        report = concept_removal_report(
            original_logits[row_index],
            top_removed_logits,
            random_removed_logits,
            target_answer_id=target_answer_id,
        )
        if report.top_removal_changed and report.random_removal_does_less:
            if report.original_answer != target_answer_id:
                continue
            return report, row_index, int(top_id), int(control_id), concept_row[control_id].item() > 0

    raise RuntimeError("could not find an active top/control concept removal case.")


def run_transformerlens_pcd_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a real sparse-concept, question-conditioned PCD preflight on gelu-1l."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l sparse-concept PCD preflight.",
        }

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()
    direction, positive_token_id, negative_token_id = _residual_direction(model)
    train_texts, train_labels, train_residuals = _collect_residuals(model, TL_TRAIN_EXAMPLES)
    eval_texts, eval_labels, eval_residuals = _collect_residuals(model, TL_EVAL_EXAMPLES)
    concept_directions, concept_names, mean_residual = _concept_directions(
        train_residuals,
        direction,
    )
    centered_train = train_residuals - mean_residual
    centered_eval = eval_residuals - mean_residual
    train_concepts = sparse_concept_encode(
        centered_train,
        concept_directions,
        top_k=TL_CONCEPT_TOP_K,
    )
    concepts = sparse_concept_encode(
        centered_eval,
        concept_directions,
        top_k=TL_CONCEPT_TOP_K,
    )
    sparsity = concept_sparsity_report(concepts, max_density=0.5)
    train_row_concepts, train_question_embeddings, train_question_ids, train_answer_ids = (
        _build_question_rows(train_concepts, train_labels)
    )
    eval_row_concepts, question_embeddings, question_ids, answer_ids = _build_question_rows(
        concepts,
        eval_labels,
    )
    train_features = question_conditioned_concept_features(
        train_row_concepts,
        train_question_ids,
        len(TL_PCD_QUESTIONS),
    )
    eval_features = question_conditioned_concept_features(
        eval_row_concepts,
        question_ids,
        len(TL_PCD_QUESTIONS),
    )
    pcd_batch = build_pcd_question_batch(
        eval_residuals.repeat_interleave(len(TL_PCD_QUESTIONS), dim=0),
        question_ids,
        answer_ids,
        question_texts=TL_PCD_QUESTIONS,
    )
    decoder_weight, decoder_bias, decoder_training, pcd_logits = _fit_linear_decoder(
        train_features,
        train_question_embeddings,
        train_answer_ids,
        eval_features,
        question_embeddings,
        steps=700,
        lr=0.06,
        seed=0,
    )

    zero_train_questions = t.zeros_like(train_question_embeddings)
    zero_eval_questions = t.zeros_like(question_embeddings)
    _probe_weight, _probe_bias, probe_training, probe_logits = _fit_linear_decoder(
        train_row_concepts,
        zero_train_questions,
        train_answer_ids,
        eval_row_concepts,
        zero_eval_questions,
        steps=700,
        lr=0.06,
        seed=1,
    )
    _sae_weight, _sae_bias, sae_training, sae_logits = _fit_linear_decoder(
        train_row_concepts[:, :2],
        zero_train_questions,
        train_answer_ids,
        eval_row_concepts[:, :2],
        zero_eval_questions,
        steps=700,
        lr=0.06,
        seed=2,
    )
    train_dense_rows = train_residuals.repeat_interleave(len(TL_PCD_QUESTIONS), dim=0)
    eval_dense_rows = eval_residuals.repeat_interleave(len(TL_PCD_QUESTIONS), dim=0)
    _oracle_weight, _oracle_bias, oracle_training, non_interaction_logits = _fit_linear_decoder(
        train_dense_rows,
        train_question_embeddings,
        train_answer_ids,
        eval_dense_rows,
        question_embeddings,
        steps=700,
        lr=0.03,
        seed=3,
    )
    comparison = pcd_comparison_report(
        pcd_logits,
        probe_logits,
        sae_logits,
        non_interaction_logits,
        answer_ids,
    )
    eval_prediction_rows = []
    pcd_predictions = pcd_logits.argmax(dim=-1)
    probe_predictions = probe_logits.argmax(dim=-1)
    sae_predictions = sae_logits.argmax(dim=-1)
    non_interaction_predictions = non_interaction_logits.argmax(dim=-1)
    for row_index, answer in enumerate(answer_ids.tolist()):
        example_index = row_index // len(TL_PCD_QUESTIONS)
        question_id = int(question_ids[row_index].item())
        eval_prediction_rows.append(
            {
                "prompt": eval_texts[example_index],
                "latent_label": eval_labels[example_index],
                "question": TL_PCD_QUESTIONS[question_id],
                "answer": int(answer),
                "pcd_pred": int(pcd_predictions[row_index].item()),
                "probe_pred": int(probe_predictions[row_index].item()),
                "single_concept_pred": int(sae_predictions[row_index].item()),
                "dense_noninteraction_pred": int(non_interaction_predictions[row_index].item()),
            }
        )

    shuffled_question_ids = question_ids.roll(shifts=1, dims=0)
    shuffled_question_embeddings = question_embeddings.roll(shifts=1, dims=0)
    shuffled_features = question_conditioned_concept_features(
        eval_row_concepts,
        shuffled_question_ids,
        len(TL_PCD_QUESTIONS),
    )
    shuffled_logits = question_conditioned_decoder_logits(
        shuffled_features,
        shuffled_question_embeddings,
        decoder_weight,
        decoder_bias=decoder_bias,
    )
    question_shuffle_accuracy = _prediction_accuracy(shuffled_logits, answer_ids)

    removal, removal_row, top_removed_concept, random_removed_concept, random_removed_active = (
        _active_concept_removal_case(
            eval_row_concepts,
            question_ids,
            question_embeddings,
            answer_ids,
            decoder_weight,
            decoder_bias,
        )
    )

    concept_scores = _concept_scores_from_decoder(
        decoder_weight,
        eval_features,
        n_concepts=concepts.shape[1],
        question_count=len(TL_PCD_QUESTIONS),
    )
    audit = concept_audit_report(
        concept_scores,
        concept_names,
        ["surface", "motion"],
        top_k=2,
    )
    seed_scores = []
    seed_accuracies = []
    for seed in (0, 1, 2):
        seed_weight, seed_bias, _seed_training, seed_logits = _fit_linear_decoder(
            train_features,
            train_question_embeddings,
            train_answer_ids,
            eval_features,
            question_embeddings,
            steps=700,
            lr=0.06,
            seed=seed,
        )
        seed_scores.append(
            _concept_scores_from_decoder(
                seed_weight,
                eval_features,
                n_concepts=concepts.shape[1],
                question_count=len(TL_PCD_QUESTIONS),
            )
        )
        seed_accuracies.append(_prediction_accuracy(seed_logits, answer_ids))
    stability = concept_stability_report(
        seed_scores,
        top_k=2,
        min_jaccard=1.0,
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        comparison.pcd_accuracy == 1.0
        and comparison.best_baseline_accuracy <= 0.75
        and comparison.beats_best_baseline
        and decoder_training.train_accuracy == 1.0
        and decoder_training.final_loss < 0.01
        and min(seed_accuracies) == 1.0
        and question_shuffle_accuracy <= 0.75
        and sparsity.passes_sparsity
        and removal.top_removal_changed
        and removal.random_removal_does_less
        and random_removed_active
        and audit.names_expected_cluster
        and stability.stable
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
        "hook_name": TL_RESID_POST_HOOK,
        "positive_anchor_prompt": TL_POSITIVE_ANCHOR,
        "negative_anchor_prompt": TL_NEGATIVE_ANCHOR,
        "positive_token": model.to_string(positive_token_id),
        "negative_token": model.to_string(negative_token_id),
        "train_example_count": len(train_texts),
        "eval_example_count": len(eval_texts),
        "activation_shape": list(eval_residuals.shape),
        "concept_shape": list(concepts.shape),
        "row_concept_shape": list(eval_row_concepts.shape),
        "conditioned_concept_shape": list(eval_features.shape),
        "question_count": len(TL_PCD_QUESTIONS),
        "pcd_row_count": len(answer_ids),
        "train_pcd_row_count": len(train_answer_ids),
        "concept_direction_count": concepts.shape[1],
        "concept_top_k": TL_CONCEPT_TOP_K,
        "concept_mean_l0": sparsity.mean_l0,
        "concept_density": sparsity.density,
        "passes_sparsity": sparsity.passes_sparsity,
        "pcd_decoder_train_loss": decoder_training.final_loss,
        "pcd_decoder_train_accuracy": decoder_training.train_accuracy,
        "pcd_decoder_training_steps": decoder_training.steps,
        "pcd_accuracy": comparison.pcd_accuracy,
        "probe_accuracy": comparison.probe_accuracy,
        "sae_classifier_accuracy": comparison.sae_classifier_accuracy,
        "non_interaction_baseline_accuracy": comparison.activation_oracle_accuracy,
        "eval_prediction_rows": eval_prediction_rows,
        "best_baseline_accuracy": comparison.best_baseline_accuracy,
        "beats_probe": comparison.beats_probe,
        "beats_best_baseline": comparison.beats_best_baseline,
        "probe_train_accuracy": probe_training.train_accuracy,
        "sae_classifier_train_accuracy": sae_training.train_accuracy,
        "non_interaction_train_accuracy": oracle_training.train_accuracy,
        "question_shuffle_accuracy": question_shuffle_accuracy,
        "pcd_seed_count": len(seed_accuracies),
        "pcd_seed_min_accuracy": min(seed_accuracies),
        "passes_ood": comparison.pcd_accuracy == 1.0,
        "top_removal_changed": removal.top_removal_changed,
        "random_removal_changed": removal.random_removal_changed,
        "top_removal_delta": removal.top_removal_delta,
        "random_removal_delta": removal.random_removal_delta,
        "random_removal_does_less": removal.random_removal_does_less,
        "removal_row_index": removal_row,
        "top_removed_concept_id": top_removed_concept,
        "random_removed_concept_id": random_removed_concept,
        "random_removed_concept_active": random_removed_active,
        "selected_concept_ids": list(audit.selected_concept_ids),
        "selected_concept_names": list(audit.selected_concept_names),
        "names_expected_cluster": audit.names_expected_cluster,
        "mean_pairwise_jaccard": stability.mean_pairwise_jaccard,
        "stable": stability.stable,
        "baseline_scope": "question_agnostic_probe_single_concept_and_non_interaction_question_baselines",
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l sparse-concept PCD preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_pcd_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
