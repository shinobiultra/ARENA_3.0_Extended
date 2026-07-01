"""Predictive Concept Decoder utilities for activation-to-language notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
import torch.nn.functional as F

from arena_ext.activation_language import prediction_accuracy


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
