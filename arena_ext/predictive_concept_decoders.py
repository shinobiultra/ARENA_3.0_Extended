"""Predictive Concept Decoder utilities for activation-to-language notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
import torch.nn.functional as F

from arena_ext.activation_language import prediction_accuracy

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
    return prediction_accuracy(logits, labels)


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
    """Bundle activations with question ids and binary answer ids."""

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
        masked = t.zeros_like(concepts)
        concepts = masked.scatter(dim=-1, index=indices, src=values)
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
        train_accuracy = prediction_accuracy(train_logits, answer_ids)
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
    """Compare a PCD against probe, SAE-classifier, and oracle baselines."""

    pcd_accuracy = prediction_accuracy(pcd_logits, answer_ids)
    probe_accuracy = prediction_accuracy(probe_logits, answer_ids)
    sae_accuracy = prediction_accuracy(sae_classifier_logits, answer_ids)
    oracle_accuracy = prediction_accuracy(activation_oracle_logits, answer_ids)
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
