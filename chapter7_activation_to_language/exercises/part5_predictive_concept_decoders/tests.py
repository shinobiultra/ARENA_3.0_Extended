import ast
import json
from collections.abc import Callable
from pathlib import Path

import torch as t

from arena_ext import predictive_concept_decoders as reference


def _solutions():
    from chapter7_activation_to_language.exercises.part5_predictive_concept_decoders import (
        solutions,
    )

    return solutions


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} fields should match the independent reference implementation."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, float):
            assert abs(actual_value - expected_value) < 1e-6, (
                f"{msg} field {key!r} should be {expected_value}, got {actual_value}."
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_build_pcd_question_batch_validates_shapes_and_questions(
    build_pcd_question_batch: Callable | None = None,
    default_pcd_questions: Callable | None = None,
):
    solutions = _solutions()
    build_pcd_question_batch = build_pcd_question_batch or solutions.build_pcd_question_batch
    default_pcd_questions = default_pcd_questions or solutions.default_pcd_questions
    activations = t.eye(4)
    question_ids = t.tensor([0, 1, 2, 3])
    answer_ids = t.tensor([1, 0, 1, 0])
    batch = build_pcd_question_batch(activations, question_ids, answer_ids)
    expected = reference.build_pcd_question_batch(activations, question_ids, answer_ids)
    assert batch.activations.shape == (4, 4), (
        "PCD question batches should preserve [examples, d_model] activations."
    )
    assert batch.question_ids.dtype == t.long and batch.answer_ids.dtype == t.long, (
        "Question ids and answer ids should be integer class-index tensors."
    )
    assert batch.question_texts == default_pcd_questions() == expected.question_texts, (
        "The default PCD question bank should be stable and match the reference contract."
    )
    assert batch.answer_ids.tolist() == [1, 0, 1, 0], (
        "Answer ids should stay aligned with activation rows."
    )
    try:
        build_pcd_question_batch(t.ones(4), question_ids, answer_ids)
    except ValueError as exc:
        assert "activations" in str(exc), (
            "Rank-1 activations should fail with a message about activation shape."
        )
    else:
        raise AssertionError("Rank-1 activations should be rejected with ValueError.")
    print(
        "All tests in `test_build_pcd_question_batch_validates_shapes_and_questions` passed!"
    )


def test_build_pcd_question_batch_rejects_empty_and_bad_question_ids(
    build_pcd_question_batch: Callable | None = None,
):
    build_pcd_question_batch = build_pcd_question_batch or _solutions().build_pcd_question_batch
    try:
        build_pcd_question_batch(t.empty(0, 3), t.empty(0, dtype=t.long), t.empty(0, dtype=t.long))
    except ValueError as exc:
        assert "at least one example" in str(exc), (
            "Empty PCD batches should fail before question-id reductions produce invalid results."
        )
    else:
        raise AssertionError("Empty PCD batches should raise ValueError.")

    try:
        build_pcd_question_batch(t.eye(2), t.tensor([0, 99]), t.tensor([1, 0]))
    except ValueError as exc:
        assert "question_texts" in str(exc), (
            "Out-of-range question ids should mention the question bank they index."
        )
    else:
        raise AssertionError("Out-of-range question ids should raise ValueError.")
    print(
        "All tests in `test_build_pcd_question_batch_rejects_empty_and_bad_question_ids` passed!"
    )


def test_sparse_concept_encode_and_sparsity_controls(
    sparse_concept_encode: Callable | None = None,
    concept_sparsity_report: Callable | None = None,
):
    solutions = _solutions()
    sparse_concept_encode = sparse_concept_encode or solutions.sparse_concept_encode
    concept_sparsity_report = concept_sparsity_report or solutions.concept_sparsity_report
    activations = t.eye(3)
    concept_directions = t.eye(3)
    concepts = sparse_concept_encode(activations, concept_directions, top_k=1)
    expected_concepts = reference.sparse_concept_encode(
        activations,
        concept_directions,
        top_k=1,
    )
    assert concepts.tolist() == expected_concepts.tolist() == t.eye(3).tolist(), (
        "Top-k sparse concept encoding should keep exactly the active identity concept."
    )
    sparsity = concept_sparsity_report(concepts, max_density=0.34)
    expected_sparsity = reference.concept_sparsity_report(concepts, max_density=0.34)
    _assert_report_close(sparsity, expected_sparsity, msg="Concept sparsity report")
    assert sparsity.mean_l0 == 1.0 and sparsity.density < 0.34, (
        "The identity toy example should have one active concept per row and density 1/3."
    )
    assert sparsity.passes_sparsity, (
        "The report should pass when concept density is below the threshold."
    )
    thresholded = sparse_concept_encode(
        t.tensor([[0.2, 0.9, -1.0]]),
        t.eye(3),
        threshold=0.5,
    )
    assert t.allclose(thresholded, t.tensor([[0.0, 0.9, 0.0]])), (
        "Thresholding should zero weak positive scores and all negative scores after ReLU."
    )
    try:
        sparse_concept_encode(activations, concept_directions, top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc), (
            "Invalid top_k should fail with a message naming top_k."
        )
    else:
        raise AssertionError("top_k=0 should be rejected with ValueError.")
    print("All tests in `test_sparse_concept_encode_and_sparsity_controls` passed!")


def test_sparse_concept_encode_and_sparsity_reject_bad_controls(
    sparse_concept_encode: Callable | None = None,
    concept_sparsity_report: Callable | None = None,
):
    solutions = _solutions()
    sparse_concept_encode = sparse_concept_encode or solutions.sparse_concept_encode
    concept_sparsity_report = concept_sparsity_report or solutions.concept_sparsity_report
    try:
        sparse_concept_encode(t.empty(0, 3), t.eye(3))
    except ValueError as exc:
        assert "nonempty" in str(exc), "Empty activation batches should be rejected."
    else:
        raise AssertionError("Empty activation batches should raise ValueError.")
    try:
        sparse_concept_encode(t.eye(3), t.eye(3), threshold=-0.1)
    except ValueError as exc:
        assert "non-negative" in str(exc), "Negative thresholds should be rejected."
    else:
        raise AssertionError("Negative thresholds should raise ValueError.")
    try:
        concept_sparsity_report(t.eye(3), max_density=1.5)
    except ValueError as exc:
        assert "between 0 and 1" in str(exc), (
            "Density thresholds should be validated as probabilities."
        )
    else:
        raise AssertionError("Invalid max_density should raise ValueError.")
    print(
        "All tests in `test_sparse_concept_encode_and_sparsity_reject_bad_controls` passed!"
    )


def test_question_conditioned_decoder_uses_question_information(
    question_conditioned_decoder_logits: Callable | None = None,
):
    question_conditioned_decoder_logits = (
        question_conditioned_decoder_logits
        or _solutions().question_conditioned_decoder_logits
    )
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
    expected = reference.question_conditioned_decoder_logits(
        concepts,
        question_embeddings,
        decoder_weight,
    )
    assert logits.tolist() == expected.tolist() == [[3.0, 0.0], [0.0, 3.0]], (
        "Decoder logits should come from concatenated [concepts, question_embeddings]."
    )
    swapped_questions = question_conditioned_decoder_logits(
        concepts,
        question_embeddings.flip(0),
        decoder_weight,
    )
    assert swapped_questions.tolist() != logits.tolist(), (
        "Changing question embeddings should change logits when the decoder uses them."
    )
    try:
        question_conditioned_decoder_logits(concepts, question_embeddings[:1], decoder_weight)
    except ValueError as exc:
        assert "same size" in str(exc), (
            "Batch mismatch errors should explain that concepts and questions align by row."
        )
    else:
        raise AssertionError("Mismatched concept/question batches should raise ValueError.")
    print(
        "All tests in `test_question_conditioned_decoder_uses_question_information` passed!"
    )


def test_question_conditioned_decoder_respects_arbitrary_weight_and_bias(
    question_conditioned_decoder_logits: Callable | None = None,
):
    question_conditioned_decoder_logits = (
        question_conditioned_decoder_logits
        or _solutions().question_conditioned_decoder_logits
    )
    concepts = t.tensor([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
    question_embeddings = t.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    decoder_weight = t.tensor(
        [
            [1.0, -1.0, 0.5],
            [0.0, 2.0, -0.5],
            [3.0, 0.0, 1.0],
            [-2.0, 1.0, 0.0],
            [0.25, -0.75, 2.0],
        ]
    )
    decoder_bias = t.tensor([0.1, -0.2, 0.3])

    logits = question_conditioned_decoder_logits(
        concepts,
        question_embeddings,
        decoder_weight,
        decoder_bias=decoder_bias,
    )
    expected = reference.question_conditioned_decoder_logits(
        concepts,
        question_embeddings,
        decoder_weight,
        decoder_bias=decoder_bias,
    )

    assert t.allclose(logits, expected), (
        "Decoder helper should use the supplied arbitrary weights and bias, not a baked fixture."
    )
    assert t.allclose(
        logits,
        t.tensor([[-0.9, 3.8, -0.2], [3.1, -2.2, 1.8], [3.35, -1.95, 3.3]]),
    ), "The arbitrary decoder fixture should produce the manually computed logits."
    print(
        "All tests in `test_question_conditioned_decoder_respects_arbitrary_weight_and_bias` passed!"
    )


def test_question_conditioned_features_and_decoder_reject_bad_shapes(
    question_conditioned_concept_features: Callable | None = None,
    question_conditioned_decoder_logits: Callable | None = None,
):
    solutions = _solutions()
    question_conditioned_concept_features = (
        question_conditioned_concept_features
        or solutions.question_conditioned_concept_features
    )
    question_conditioned_decoder_logits = (
        question_conditioned_decoder_logits
        or solutions.question_conditioned_decoder_logits
    )
    try:
        question_conditioned_concept_features(t.eye(2), t.tensor([0, 2]), question_count=2)
    except ValueError as exc:
        assert "question_ids" in str(exc), "Out-of-range question ids should fail."
    else:
        raise AssertionError("Out-of-range question ids should raise ValueError.")

    try:
        question_conditioned_decoder_logits(t.eye(2), t.eye(2), t.ones(4))
    except ValueError as exc:
        assert "decoder_weight" in str(exc), "Rank-1 decoder weights should be rejected."
    else:
        raise AssertionError("Rank-1 decoder weights should raise ValueError.")
    print(
        "All tests in `test_question_conditioned_features_and_decoder_reject_bad_shapes` passed!"
    )


def test_trained_question_conditioned_decoder_learns_concept_question_interaction(
    question_conditioned_concept_features: Callable | None = None,
    train_question_conditioned_decoder: Callable | None = None,
    question_conditioned_decoder_logits: Callable | None = None,
):
    solutions = _solutions()
    question_conditioned_concept_features = (
        question_conditioned_concept_features
        or solutions.question_conditioned_concept_features
    )
    train_question_conditioned_decoder = (
        train_question_conditioned_decoder
        or solutions.train_question_conditioned_decoder
    )
    question_conditioned_decoder_logits = (
        question_conditioned_decoder_logits
        or solutions.question_conditioned_decoder_logits
    )
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

    assert conditioned.shape == (4, 4), (
        "Question-conditioned concept features should allocate one sparse slot per question."
    )
    assert report.train_accuracy == 1.0, (
        "The tiny trained decoder should solve the interaction fixture."
    )
    assert logits.argmax(dim=-1).tolist() == answer_ids.tolist(), (
        "The trained decoder should learn the concept-question interaction from labels."
    )
    print(
        "All tests in `test_trained_question_conditioned_decoder_learns_concept_question_interaction` passed!"
    )


def test_train_question_conditioned_decoder_rejects_empty_and_misaligned_batches(
    train_question_conditioned_decoder: Callable | None = None,
):
    train_question_conditioned_decoder = (
        train_question_conditioned_decoder
        or _solutions().train_question_conditioned_decoder
    )
    try:
        train_question_conditioned_decoder(
            t.empty(0, 2),
            t.empty(0, 2),
            t.empty(0, dtype=t.long),
        )
    except ValueError as exc:
        assert "nonempty" in str(exc), "Empty training batches should fail clearly."
    else:
        raise AssertionError("Empty training batches should raise ValueError.")

    try:
        train_question_conditioned_decoder(
            t.eye(2),
            t.eye(3),
            t.tensor([0, 1]),
        )
    except ValueError as exc:
        assert "same batch size" in str(exc), (
            "Concept/question batch mismatch should be rejected before training."
        )
    else:
        raise AssertionError("Misaligned concept/question batches should raise ValueError.")
    print(
        "All tests in `test_train_question_conditioned_decoder_rejects_empty_and_misaligned_batches` passed!"
    )


def test_pcd_comparison_report_beats_baselines(
    pcd_comparison_report: Callable | None = None,
):
    pcd_comparison_report = pcd_comparison_report or _solutions().pcd_comparison_report
    answer_ids = t.tensor([0, 1, 0, 1])
    pcd_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    probe_logits = t.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    sae_logits = probe_logits.clone()
    oracle_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    report = pcd_comparison_report(
        pcd_logits,
        probe_logits,
        sae_logits,
        oracle_logits,
        answer_ids,
    )
    expected = reference.pcd_comparison_report(
        pcd_logits,
        probe_logits,
        sae_logits,
        oracle_logits,
        answer_ids,
    )
    _assert_report_close(report, expected, msg="PCD comparison report")
    assert report.pcd_accuracy == 1.0 and report.probe_accuracy == 0.5, (
        "The toy PCD should solve all rows while the question-agnostic probe misses half."
    )
    assert report.activation_oracle_accuracy == 0.75, (
        "The oracle-style baseline in this fixture should remain below the PCD."
    )
    assert report.beats_probe and report.beats_best_baseline, (
        "PCD comparison should require beating both the probe and the best baseline."
    )
    tied = pcd_comparison_report(
        oracle_logits,
        probe_logits,
        sae_logits,
        oracle_logits,
        answer_ids,
    )
    assert not tied.beats_best_baseline, (
        "Matching the best baseline is not enough for the stricter PCD comparison."
    )
    print("All tests in `test_pcd_comparison_report_beats_baselines` passed!")


def test_pcd_comparison_report_scores_each_baseline_independently(
    pcd_comparison_report: Callable | None = None,
):
    pcd_comparison_report = pcd_comparison_report or _solutions().pcd_comparison_report
    answer_ids = t.tensor([0, 0, 0, 1, 1, 2])

    def logits_from_preds(preds: list[int]) -> t.Tensor:
        logits = t.zeros(len(preds), 3)
        logits[t.arange(len(preds)), t.tensor(preds)] = 3.0
        return logits

    report = pcd_comparison_report(
        logits_from_preds([0, 0, 0, 1, 1, 0]),
        logits_from_preds([0, 0, 1, 0, 2, 0]),
        logits_from_preds([0, 0, 0, 0, 2, 1]),
        logits_from_preds([0, 0, 0, 1, 2, 1]),
        answer_ids,
    )

    assert abs(report.pcd_accuracy - 5 / 6) < 1e-6, (
        "PCD accuracy should be scored from PCD logits only."
    )
    assert abs(report.probe_accuracy - 2 / 6) < 1e-6, (
        "Probe accuracy should be scored independently from probe logits."
    )
    assert abs(report.sae_classifier_accuracy - 3 / 6) < 1e-6, (
        "SAE baseline accuracy should not be copied from the probe."
    )
    assert abs(report.activation_oracle_accuracy - 4 / 6) < 1e-6, (
        "Oracle-style baseline accuracy should not be copied from the PCD."
    )
    assert abs(report.best_baseline_accuracy - 4 / 6) < 1e-6, (
        "Best baseline should be the max of independently scored baselines."
    )
    print(
        "All tests in `test_pcd_comparison_report_scores_each_baseline_independently` passed!"
    )


def test_concept_stability_removal_and_audit_controls(
    concept_stability_report: Callable | None = None,
    concept_removal_report: Callable | None = None,
    concept_audit_report: Callable | None = None,
):
    solutions = _solutions()
    concept_stability_report = concept_stability_report or solutions.concept_stability_report
    concept_removal_report = concept_removal_report or solutions.concept_removal_report
    concept_audit_report = concept_audit_report or solutions.concept_audit_report

    scores_by_seed = [
        t.tensor([0.9, 0.8, 0.1, 0.0]),
        t.tensor([0.8, 0.7, 0.2, 0.0]),
        t.tensor([0.95, 0.85, 0.05, 0.0]),
    ]
    stability = concept_stability_report(scores_by_seed, top_k=2, min_jaccard=0.75)
    expected_stability = reference.concept_stability_report(
        scores_by_seed,
        top_k=2,
        min_jaccard=0.75,
    )
    _assert_report_close(stability, expected_stability, msg="Concept stability report")
    assert stability.top_concepts_by_seed == ((0, 1), (0, 1), (0, 1)), (
        "Stable concept reports should keep the same top concept ids across seeds."
    )
    assert stability.mean_pairwise_jaccard == 1.0 and stability.stable, (
        "Identical top-k sets should have mean pairwise Jaccard 1.0 and pass."
    )
    unstable = concept_stability_report(
        [
            t.tensor([0.9, 0.8, 0.1, 0.0]),
            t.tensor([0.1, 0.0, 0.9, 0.8]),
        ],
        top_k=2,
        min_jaccard=0.5,
    )
    assert not unstable.stable, (
        "Disjoint top concepts across seeds should fail the stability control."
    )
    disjoint_three_seed = concept_stability_report(
        [
            t.tensor([0.91, 0.90, 0.05, 0.04, 0.03, 0.02]),
            t.tensor([0.05, 0.04, 0.91, 0.90, 0.03, 0.02]),
            t.tensor([0.05, 0.04, 0.03, 0.02, 0.91, 0.90]),
        ],
        top_k=2,
        min_jaccard=0.1,
    )
    assert disjoint_three_seed.top_concepts_by_seed == ((0, 1), (2, 3), (4, 5)), (
        "Stability should use seed-specific top-k ids, not just score shape."
    )
    assert disjoint_three_seed.mean_pairwise_jaccard == 0.0 and not disjoint_three_seed.stable, (
        "Disjoint top-k concepts across seeds should be reported as unstable."
    )

    original_logits = t.tensor([3.0, 1.0])
    top_removed_logits = t.tensor([0.0, 2.0])
    random_removed_logits = t.tensor([2.5, 1.0])
    removal = concept_removal_report(
        original_logits,
        top_removed_logits,
        random_removed_logits,
    )
    expected_removal = reference.concept_removal_report(
        original_logits,
        top_removed_logits,
        random_removed_logits,
    )
    _assert_report_close(removal, expected_removal, msg="Concept removal report")
    assert removal.top_removal_changed and not removal.random_removal_changed, (
        "Removing the top concept should flip the answer while random removal should not."
    )
    assert removal.top_removal_delta == 3.0 and removal.random_removal_delta == 0.5, (
        "Removal deltas should be measured on the original target-answer logit."
    )
    assert removal.random_removal_does_less, (
        "Random removal should damage the target logit less than top-concept removal."
    )
    equally_damaging = concept_removal_report(
        t.tensor([4.0, 1.0, 0.0]),
        t.tensor([0.5, 3.0, 0.0]),
        t.tensor([0.4, 3.1, 0.0]),
        target_answer_id=0,
    )
    assert equally_damaging.random_removal_changed, (
        "The random-removal report should expose when the random control also flips the answer."
    )
    assert not equally_damaging.random_removal_does_less, (
        "A random removal that is equally damaging should fail the negative control."
    )

    concept_scores = t.tensor([0.1, 0.9, 0.8])
    concept_names = ["syntax feature", "refusal feature", "safety refusal"]
    audit = concept_audit_report(concept_scores, concept_names, ["refusal"], top_k=2)
    expected_audit = reference.concept_audit_report(
        concept_scores,
        concept_names,
        ["refusal"],
        top_k=2,
    )
    _assert_report_close(audit, expected_audit, msg="Concept audit report")
    assert audit.selected_concept_ids == (1, 2), (
        "Concept audit should select the highest-scoring concepts."
    )
    assert "refusal feature" in audit.explanation and audit.names_expected_cluster, (
        "Selected concept names should make the expected semantic cluster legible."
    )
    off_cluster = concept_audit_report(
        concept_scores,
        ["syntax feature", "arithmetic feature", "calendar feature"],
        ["refusal"],
        top_k=2,
    )
    assert not off_cluster.names_expected_cluster, (
        "Top concepts whose names lack the expected terms should fail the name audit."
    )
    print(
        "All tests in `test_concept_stability_removal_and_audit_controls` passed!"
    )


def test_concept_audit_controls_reject_bad_inputs(
    concept_stability_report: Callable | None = None,
    concept_removal_report: Callable | None = None,
    concept_audit_report: Callable | None = None,
):
    solutions = _solutions()
    concept_stability_report = concept_stability_report or solutions.concept_stability_report
    concept_removal_report = concept_removal_report or solutions.concept_removal_report
    concept_audit_report = concept_audit_report or solutions.concept_audit_report

    try:
        concept_stability_report([t.tensor([1.0, 0.0])], min_jaccard=1.5)
    except ValueError as exc:
        assert "between 0 and 1" in str(exc), "Jaccard thresholds should be probabilities."
    else:
        raise AssertionError("Invalid min_jaccard should raise ValueError.")

    try:
        concept_removal_report(
            t.tensor([1.0, 0.0]),
            t.tensor([0.0, 1.0]),
            t.tensor([0.5, 0.5]),
            target_answer_id=5,
        )
    except ValueError as exc:
        assert "out of range" in str(exc), "Bad target answer ids should be rejected."
    else:
        raise AssertionError("Out-of-range target_answer_id should raise ValueError.")

    try:
        concept_audit_report(t.tensor([]), [], ["surface"], top_k=1)
    except ValueError as exc:
        assert "nonempty" in str(exc), "Empty concept scores should fail clearly."
    else:
        raise AssertionError("Empty concept scores should raise ValueError.")
    print(
        "All tests in `test_concept_audit_controls_reject_bad_inputs` passed!"
    )


def test_make_planted_pcd_world_has_exact_sparse_ground_truth(
    make_planted_pcd_world: Callable | None = None,
):
    make_planted_pcd_world = make_planted_pcd_world or _solutions().make_planted_pcd_world
    world = make_planted_pcd_world(seed=0)
    expected = reference.make_planted_pcd_world(seed=0)
    assert world.question_texts == expected.question_texts, (
        "The planted question bank should be stable and learner-readable."
    )
    assert world.concept_names == expected.concept_names, (
        "Concept names should expose the planted audit surface."
    )
    assert world.train_activations.shape == expected.train_activations.shape == (32, 12), (
        "The planted world should use 32 balanced train examples in dense activation space."
    )
    assert world.heldout_activations.shape == expected.heldout_activations.shape == (32, 12), (
        "The planted world should use a separate 32-example held-out split."
    )
    recovered = world.heldout_activations @ world.concept_directions
    assert t.allclose(recovered, world.heldout_true_concepts, atol=1e-5, rtol=1e-5), (
        "The planted concept directions should recover the hidden sparse concepts exactly."
    )
    assert not any(name in " ".join(world.heldout_prompts).lower() for name in ("surface", "motion", "red", "animal")), (
        "Toy prompt text should not leak the hidden concept labels."
    )
    print("All tests in `test_make_planted_pcd_world_has_exact_sparse_ground_truth` passed!")


def test_build_planted_question_rows_aligns_every_question(
    build_planted_question_rows: Callable | None = None,
):
    build_planted_question_rows = (
        build_planted_question_rows or _solutions().build_planted_question_rows
    )
    concepts = reference.make_planted_pcd_world(seed=0).heldout_true_concepts
    row_concepts, question_embeddings, question_ids, answer_ids = build_planted_question_rows(
        concepts,
    )
    expected_rows = concepts.shape[0] * len(reference.PLANTED_PCD_QUESTIONS)
    assert row_concepts.shape == (expected_rows, concepts.shape[1]), (
        "Every held-out activation should be repeated under every behavioral question."
    )
    assert question_embeddings.shape == (expected_rows, len(reference.PLANTED_PCD_QUESTIONS)), (
        "Question embeddings should be one-hot and row-aligned."
    )
    assert question_ids[:8].tolist() == [0, 1, 2, 3, 0, 1, 2, 3], (
        "Question ids should cycle within each example before moving to the next example."
    )
    assert answer_ids.float().mean().item() == 0.5, (
        "Each planted question should be balanced on held-out rows, so question-only shortcuts fail."
    )
    assert answer_ids[:4].tolist() == concepts[0, :4].long().tolist(), (
        "Answers for one example should be its planted concept bits under the four questions."
    )
    print("All tests in `test_build_planted_question_rows_aligns_every_question` passed!")


def test_planted_pcd_experiment_beats_requested_controls(
    run_planted_pcd_experiment: Callable | None = None,
):
    run_planted_pcd_experiment = (
        run_planted_pcd_experiment or _solutions().run_planted_pcd_experiment
    )
    result = run_planted_pcd_experiment(seed=0, steps=350, lr=0.08)
    report = result["baseline_report"]
    expected_keys = {
        "PCD",
        "question-agnostic probe",
        "text/template shortcut",
        "shuffled question",
        "random label",
        "single concept",
        "dense non-interaction",
    }
    assert set(result["baseline_names"]) == expected_keys, (
        "The planted signature result should include every requested control."
    )
    assert result["train_recovery_error"] < 1e-5 and result["heldout_recovery_error"] < 1e-5, (
        "The toy world should start from exact planted sparse-concept ground truth."
    )
    assert report.pcd_accuracy == 1.0 and report.passes_controls, (
        "The question-conditioned PCD should solve held-out questions and pass the control margin."
    )
    assert report.question_agnostic_probe_accuracy <= 0.75, (
        "A question-agnostic probe should fail on repeated concept rows with different questions."
    )
    assert report.text_template_shortcut_accuracy <= 0.75, (
        "Text/template shortcuts should fail because the toy prompts do not reveal hidden concepts."
    )
    assert report.shuffled_question_accuracy <= 0.75, (
        "Shuffling questions should break the learned concept-question alignment."
    )
    assert report.random_label_accuracy <= 0.75, (
        "Random-label training should not produce a convincing held-out result."
    )
    assert report.single_concept_accuracy <= 0.75, (
        "A single concept should not solve a four-question bottleneck."
    )
    assert report.dense_noninteraction_accuracy <= 0.75, (
        "Dense additive activation+question features should not replace interaction slots."
    )
    heatmap = result["heatmap"]
    assert heatmap.shape == (4, 6), (
        "The signature heatmap should be [question, concept], including nuisance concepts."
    )
    top_concepts = heatmap[:, :4].argmax(dim=1).tolist()
    assert top_concepts == [0, 1, 2, 3], (
        "The learned interaction weights should put each question on its planted concept."
    )
    assert len(result["prediction_rows"]) == 128, (
        "The held-out table should expose all 32 examples under all 4 questions."
    )
    print("All tests in `test_planted_pcd_experiment_beats_requested_controls` passed!")


def test_pcd_baseline_sweep_report_scores_all_controls(
    pcd_baseline_sweep_report: Callable | None = None,
):
    pcd_baseline_sweep_report = (
        pcd_baseline_sweep_report or _solutions().pcd_baseline_sweep_report
    )
    answer_ids = t.tensor([0, 1, 0, 1, 1, 0, 1, 0])

    def logits_from_preds(preds: list[int]) -> t.Tensor:
        logits = t.zeros(len(preds), 2)
        logits[t.arange(len(preds)), t.tensor(preds)] = 4.0
        return logits

    report = pcd_baseline_sweep_report(
        pcd_logits=logits_from_preds([0, 1, 0, 1, 1, 0, 1, 0]),
        question_agnostic_probe_logits=logits_from_preds([0, 0, 0, 0, 1, 1, 1, 1]),
        text_template_shortcut_logits=logits_from_preds([0, 0, 0, 0, 0, 0, 0, 0]),
        shuffled_question_logits=logits_from_preds([1, 0, 1, 0, 0, 1, 0, 1]),
        random_label_logits=logits_from_preds([0, 1, 1, 0, 0, 1, 1, 0]),
        single_concept_logits=logits_from_preds([0, 1, 0, 0, 1, 0, 0, 0]),
        dense_noninteraction_logits=logits_from_preds([0, 1, 1, 1, 1, 0, 0, 0]),
        answer_ids=answer_ids,
        min_margin=0.20,
    )
    expected = reference.pcd_baseline_sweep_report(
        pcd_logits=logits_from_preds([0, 1, 0, 1, 1, 0, 1, 0]),
        question_agnostic_probe_logits=logits_from_preds([0, 0, 0, 0, 1, 1, 1, 1]),
        text_template_shortcut_logits=logits_from_preds([0, 0, 0, 0, 0, 0, 0, 0]),
        shuffled_question_logits=logits_from_preds([1, 0, 1, 0, 0, 1, 0, 1]),
        random_label_logits=logits_from_preds([0, 1, 1, 0, 0, 1, 1, 0]),
        single_concept_logits=logits_from_preds([0, 1, 0, 0, 1, 0, 0, 0]),
        dense_noninteraction_logits=logits_from_preds([0, 1, 1, 1, 1, 0, 0, 0]),
        answer_ids=answer_ids,
        min_margin=0.20,
    )
    _assert_report_close(report, expected, msg="PCD baseline sweep report")
    assert report.pcd_accuracy == 1.0 and report.best_control_accuracy == 0.75, (
        "The report should score the PCD and independently take the best control accuracy."
    )
    assert report.passes_controls and abs(report.pcd_margin_over_best_control - 0.25) < 1e-6, (
        "The report should require a positive margin over the best non-PCD control."
    )
    tied = pcd_baseline_sweep_report(
        pcd_logits=logits_from_preds([0, 1, 1, 1, 1, 0, 0, 0]),
        question_agnostic_probe_logits=logits_from_preds([0, 1, 1, 1, 1, 0, 0, 0]),
        text_template_shortcut_logits=logits_from_preds([0, 0, 0, 0, 0, 0, 0, 0]),
        shuffled_question_logits=logits_from_preds([1, 0, 1, 0, 0, 1, 0, 1]),
        random_label_logits=logits_from_preds([0, 1, 1, 0, 0, 1, 1, 0]),
        single_concept_logits=logits_from_preds([0, 1, 0, 0, 1, 0, 0, 0]),
        dense_noninteraction_logits=logits_from_preds([0, 1, 1, 1, 1, 0, 0, 0]),
        answer_ids=answer_ids,
        min_margin=0.20,
    )
    assert not tied.passes_controls, (
        "A dense or probe control matching the PCD should fail the PCD-name claim."
    )
    print("All tests in `test_pcd_baseline_sweep_report_scores_all_controls` passed!")


def test_targeted_concept_removal_report_scores_active_control(
    targeted_concept_removal_report: Callable | None = None,
):
    targeted_concept_removal_report = (
        targeted_concept_removal_report or _solutions().targeted_concept_removal_report
    )
    solutions = _solutions()
    row_concepts = t.tensor([[1.0, 1.0]])
    question_ids = t.tensor([0])
    question_embeddings = t.tensor([[1.0, 0.0]])
    decoder_weight = t.zeros(6, 2)
    decoder_weight[0, 1] = 5.0
    decoder_weight[1, 1] = 0.25
    decoder_weight[4, 0] = 1.0
    decoder_bias = t.tensor([1.5, -1.5])
    report = targeted_concept_removal_report(
        row_concepts,
        question_ids,
        question_embeddings,
        decoder_weight,
        decoder_bias,
        prompts=("heldout latent card 00 / template 0",),
        question_texts=("Is the hidden surface concept active?", "Is the hidden motion concept active?"),
        concept_names=("surface", "motion"),
        example_index=0,
        question_id=0,
        target_concept_id=0,
        active_control_concept_id=1,
    )
    expected = solutions.TargetedConceptRemovalReport(
        row_index=0,
        prompt="heldout latent card 00 / template 0",
        question="Is the hidden surface concept active?",
        target_concept="surface",
        active_control_concept="motion",
        original_answer=1,
        target_removed_answer=0,
        active_control_removed_answer=1,
        target_logit_delta=5.0,
        active_control_logit_delta=0.25,
        target_removal_changed=True,
        active_control_changed=False,
        active_control_does_less=True,
    )
    _assert_report_close(report, expected, msg="Targeted concept removal report")
    try:
        targeted_concept_removal_report(
            row_concepts,
            question_ids,
            question_embeddings,
            decoder_weight,
            decoder_bias,
            prompts=("heldout latent card 00 / template 0",),
            question_texts=("Is the hidden surface concept active?", "Is the hidden motion concept active?"),
            concept_names=("surface", "motion"),
            example_index=0,
            question_id=0,
            target_concept_id=0,
            active_control_concept_id=5,
        )
    except ValueError as exc:
        assert "out of range" in str(exc), (
            "Bad active-control ids should fail before a misleading removal chart is made."
        )
    else:
        raise AssertionError("Out-of-range active-control concept ids should raise ValueError.")
    print("All tests in `test_targeted_concept_removal_report_scores_active_control` passed!")


def test_targeted_concept_removal_beats_active_control(
    run_planted_pcd_experiment: Callable | None = None,
):
    run_planted_pcd_experiment = (
        run_planted_pcd_experiment or _solutions().run_planted_pcd_experiment
    )
    result = run_planted_pcd_experiment(seed=0, steps=350, lr=0.08)
    removal = result["removal_report"]
    assert removal.target_concept == "surface" and removal.active_control_concept == "motion", (
        "The removal case should name the target and matched active control concepts."
    )
    assert removal.target_removal_changed, (
        "Removing the targeted concept should change the answer on the selected held-out row."
    )
    assert not removal.active_control_changed, (
        "Removing a matched active non-target concept should preserve the answer."
    )
    assert removal.active_control_does_less, (
        "The active control removal should damage the target logit less than targeted removal."
    )
    print("All tests in `test_targeted_concept_removal_beats_active_control` passed!")


def test_gelu1l_prompt_prediction_table_uses_direct_rows_or_honest_aggregate(
    gelu1l_prompt_prediction_table: Callable | None = None,
):
    gelu1l_prompt_prediction_table = (
        gelu1l_prompt_prediction_table or _solutions().gelu1l_prompt_prediction_table
    )
    direct = {
        "metrics": {
            "gpu_test": {
                "eval_prediction_rows": [
                    {
                        "prompt": "p",
                        "question": "q",
                        "answer": 1,
                        "pcd_pred": 1,
                    }
                ]
            }
        }
    }
    assert gelu1l_prompt_prediction_table(direct) == direct["metrics"]["gpu_test"]["eval_prediction_rows"], (
        "Fresh GPU reports with direct row logits should be returned unchanged."
    )
    aggregate = {
        "metrics": {
            "gpu_test": {
                "pcd_accuracy": 1.0,
                "pcd_row_count": 32,
                "eval_example_count": 8,
                "question_count": 4,
                "preflight_passed": True,
            }
        }
    }
    rows = gelu1l_prompt_prediction_table(aggregate)
    assert len(rows) == 32 and rows[0]["source"].startswith("aggregate_report"), (
        "Older aggregate-only reports should be converted honestly and marked as aggregate-derived."
    )
    try:
        bad_aggregate = {
            "metrics": {
                "gpu_test": {
                    "pcd_accuracy": 0.875,
                    "pcd_row_count": 32,
                    "eval_example_count": 8,
                    "question_count": 4,
                    "preflight_passed": True,
                }
            }
        }
        gelu1l_prompt_prediction_table(bad_aggregate)
    except ValueError as exc:
        assert "aggregate PCD accuracy is 1.0" in str(exc), (
            "The helper must not invent per-row correctness when aggregate accuracy is imperfect."
        )
    else:
        raise AssertionError("Imperfect aggregate accuracy should raise ValueError.")
    print(
        "All tests in `test_gelu1l_prompt_prediction_table_uses_direct_rows_or_honest_aggregate` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["batch"]["num_questions"] == 4, (
        "Notebook contract should include the complete four-question PCD bank."
    )
    assert result["sparse_encoding"]["sparsity"]["passes_sparsity"], (
        "Notebook contract should prove the concept bottleneck stays sparse."
    )
    assert result["decoder"] == [[3.0, 0.0], [0.0, 3.0]], (
        "Notebook contract should expose the question-conditioned decoder logits."
    )
    assert result["decoder_training"]["conditioned_shape"] == [4, 4], (
        "Notebook contract should train on explicit concept-question interaction features."
    )
    assert result["decoder_training"]["train_accuracy"] == 1.0, (
        "Notebook contract should prove the tiny question-conditioned decoder trains."
    )
    assert result["decoder_training"]["predictions"] == result["decoder_training"]["answer_ids"], (
        "Notebook contract should expose the trained decoder predictions."
    )
    assert result["comparison"]["beats_probe"], (
        "Notebook contract should compare the PCD against a probe baseline."
    )
    assert result["comparison"]["beats_best_baseline"], (
        "Notebook contract should require the PCD to beat the best baseline."
    )
    assert result["stability"]["stable"], (
        "Notebook contract should include top-concept stability across seeds."
    )
    assert result["removal"]["random_removal_does_less"], (
        "Notebook contract should include a random-removal causal control."
    )
    assert result["audit"]["names_expected_cluster"], (
        "Notebook contract should include a concept-name audit."
    )
    planted = result["planted"]
    assert planted["train_examples"] == 32 and planted["heldout_examples"] == 32, (
        "Notebook contract should start with a balanced planted train/held-out world."
    )
    assert planted["train_recovery_error"] < 1e-5 and planted["heldout_recovery_error"] < 1e-5, (
        "Notebook contract should prove the toy concept bottleneck has exact ground truth."
    )
    assert planted["heatmap_shape"] == [4, 6], (
        "Notebook contract should include a notebook-generated concept-question heatmap."
    )
    assert planted["baselines"]["pcd_accuracy"] == 1.0, (
        "Notebook contract should expose the held-out PCD accuracy."
    )
    assert planted["baselines"]["passes_controls"], (
        "Notebook contract should require the PCD to beat all listed controls."
    )
    assert planted["removal"]["target_removal_changed"], (
        "Notebook contract should include targeted concept removal."
    )
    assert planted["removal"]["active_control_does_less"], (
        "Notebook contract should include a matched active-control removal."
    )
    assert len(planted["prediction_rows_preview"]) > 0, (
        "Notebook contract should expose held-out prediction rows, not only aggregate metrics."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_solution_notebook_exposes_taught_implementations():
    notebook_path = Path(__file__).with_name(
        "7.5_Predictive_Concept_Decoders_solutions.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    markdown = "\n".join(
        "".join(cell.get("source", ""))
        if isinstance(cell.get("source", ""), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    function_names: set[str] = set()
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        tree = ast.parse(source)
        function_names.update(
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        )

    required_functions = {
        "make_planted_pcd_world",
        "build_planted_question_rows",
        "sparse_concept_encode",
        "question_conditioned_concept_features",
        "question_conditioned_decoder_logits",
        "train_question_conditioned_decoder",
        "pcd_baseline_sweep_report",
        "targeted_concept_removal_report",
        "run_planted_pcd_experiment",
        "gelu1l_prompt_prediction_table",
        "run_smoke_test",
    }
    missing = sorted(required_functions - function_names)
    assert not missing, (
        "The solution notebook must expose the taught implementations inline; "
        f"missing {missing}."
    )
    assert "## Learning Objectives" in markdown, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert "## Signature Result" in markdown, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert "## Try It Yourself" in markdown, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert "<summary>Expected output</summary>" in markdown, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert "<summary>Solution</summary>" in markdown, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert "<summary>Help" in markdown, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert "<summary>Interpretation</summary>" in markdown, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
