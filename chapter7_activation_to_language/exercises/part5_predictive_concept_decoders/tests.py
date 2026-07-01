from collections.abc import Callable

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
    print("All tests in `test_notebook_contract` passed!")
