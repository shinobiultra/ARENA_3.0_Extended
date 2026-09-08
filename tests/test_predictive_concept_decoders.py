import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.predictive_concept_decoders import (
        build_pcd_question_batch,
        concept_audit_report,
        concept_removal_report,
        concept_sparsity_report,
        concept_stability_report,
        default_pcd_questions,
        pcd_comparison_report,
        question_conditioned_concept_features,
        question_conditioned_decoder_logits,
        sparse_concept_encode,
        train_question_conditioned_decoder,
    )


def test_build_pcd_question_batch_uses_default_questions():
    activations = t.eye(4)
    question_ids = t.tensor([0, 1, 2, 3])
    answer_ids = t.tensor([1, 0, 1, 0])

    batch = build_pcd_question_batch(activations, question_ids, answer_ids)

    assert batch.activations.shape == (4, 4)
    assert len(batch.question_texts) == len(default_pcd_questions())
    assert batch.answer_ids.tolist() == [1, 0, 1, 0]


def test_sparse_concept_encode_and_sparsity_report():
    activations = t.eye(3)
    concept_directions = t.eye(3)

    concepts = sparse_concept_encode(activations, concept_directions, top_k=1)
    report = concept_sparsity_report(concepts, max_density=0.34)

    assert concepts.tolist() == t.eye(3).tolist()
    assert report.mean_l0 == 1.0
    assert report.density == pytest.approx(1 / 3)
    assert report.passes_sparsity


def test_question_conditioned_decoder_logits():
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

    assert logits.tolist() == [[3.0, 0.0], [0.0, 3.0]]


def test_question_conditioned_decoder_respects_arbitrary_weight_and_bias():
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

    expected = t.tensor([[-0.9, 3.8, -0.2], [3.1, -2.2, 1.8], [3.35, -1.95, 3.3]])
    assert t.allclose(logits, expected)


def test_question_conditioned_feature_map_and_trained_decoder_solve_interaction():
    concepts = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = t.tensor([1, 0, 0, 1])
    question_ids = t.tensor([0, 1, 0, 1])
    repeated_concepts = concepts.repeat_interleave(2, dim=0)
    question_embeddings = t.eye(2).repeat(2, 1)
    conditioned = question_conditioned_concept_features(
        repeated_concepts,
        question_ids,
        question_count=2,
    )

    weight, bias, report = train_question_conditioned_decoder(
        conditioned,
        question_embeddings,
        labels,
        steps=250,
        lr=0.1,
        seed=0,
    )
    logits = question_conditioned_decoder_logits(
        conditioned,
        question_embeddings,
        weight,
        decoder_bias=bias,
    )

    assert conditioned.shape == (4, 4)
    assert report.train_accuracy == 1.0
    assert logits.argmax(dim=-1).tolist() == labels.tolist()


def test_pcd_comparison_report_beats_probe_on_compositional_task():
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

    assert report.pcd_accuracy == 1.0
    assert report.probe_accuracy == 0.5
    assert report.activation_oracle_accuracy == 0.75
    assert report.beats_probe
    assert report.beats_best_baseline


def test_pcd_comparison_report_scores_each_baseline_independently():
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

    assert report.pcd_accuracy == pytest.approx(5 / 6)
    assert report.probe_accuracy == pytest.approx(2 / 6)
    assert report.sae_classifier_accuracy == pytest.approx(3 / 6)
    assert report.activation_oracle_accuracy == pytest.approx(4 / 6)
    assert report.best_baseline_accuracy == pytest.approx(4 / 6)


def test_concept_stability_report_uses_pairwise_jaccard():
    scores_by_seed = [
        t.tensor([0.9, 0.8, 0.1, 0.0]),
        t.tensor([0.8, 0.7, 0.2, 0.0]),
        t.tensor([0.95, 0.85, 0.05, 0.0]),
    ]

    report = concept_stability_report(scores_by_seed, top_k=2, min_jaccard=0.75)

    assert report.top_concepts_by_seed == ((0, 1), (0, 1), (0, 1))
    assert report.mean_pairwise_jaccard == 1.0
    assert report.stable


def test_concept_stability_report_rejects_disjoint_seed_topk():
    scores_by_seed = [
        t.tensor([0.91, 0.90, 0.05, 0.04, 0.03, 0.02]),
        t.tensor([0.05, 0.04, 0.91, 0.90, 0.03, 0.02]),
        t.tensor([0.05, 0.04, 0.03, 0.02, 0.91, 0.90]),
    ]

    report = concept_stability_report(scores_by_seed, top_k=2, min_jaccard=0.1)

    assert report.top_concepts_by_seed == ((0, 1), (2, 3), (4, 5))
    assert report.mean_pairwise_jaccard == 0.0
    assert not report.stable


def test_concept_removal_report_checks_random_removal_control():
    original_logits = t.tensor([3.0, 1.0])
    top_removed_logits = t.tensor([0.0, 2.0])
    random_removed_logits = t.tensor([2.5, 1.0])

    report = concept_removal_report(
        original_logits,
        top_removed_logits,
        random_removed_logits,
    )

    assert report.original_answer == 0
    assert report.top_removed_answer == 1
    assert report.random_removed_answer == 0
    assert report.top_removal_changed
    assert not report.random_removal_changed
    assert report.top_removal_delta == 3.0
    assert report.random_removal_delta == 0.5
    assert report.random_removal_does_less


def test_concept_removal_report_fails_when_random_removal_is_equally_damaging():
    report = concept_removal_report(
        t.tensor([4.0, 1.0, 0.0]),
        t.tensor([0.5, 3.0, 0.0]),
        t.tensor([0.4, 3.1, 0.0]),
        target_answer_id=0,
    )

    assert report.top_removal_changed
    assert report.random_removal_changed
    assert report.top_removal_delta == pytest.approx(3.5)
    assert report.random_removal_delta == pytest.approx(3.6)
    assert not report.random_removal_does_less


def test_concept_audit_report_names_expected_cluster():
    concept_scores = t.tensor([0.1, 0.9, 0.8])
    concept_names = ["syntax feature", "refusal feature", "safety refusal"]

    report = concept_audit_report(
        concept_scores,
        concept_names,
        ["refusal"],
        top_k=2,
    )

    assert report.selected_concept_ids == (1, 2)
    assert report.selected_concept_names == ("refusal feature", "safety refusal")
    assert "refusal feature" in report.explanation
    assert report.names_expected_cluster
