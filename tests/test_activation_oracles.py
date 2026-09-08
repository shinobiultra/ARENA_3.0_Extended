import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.activation_oracles import (
        activation_patching_oracle_report,
        build_activation_question_batch,
        default_activation_questions,
        ood_generalization_report,
        oracle_comparison_report,
        random_activation_oracle_report,
        split_accuracy_by_template,
    )


def test_build_activation_question_batch_uses_default_questions():
    activations = t.eye(3)
    question_ids = t.tensor([0, 1, 2])
    answer_ids = t.tensor([1, 0, 1])
    template_ids = t.tensor([0, 0, 1])

    batch = build_activation_question_batch(activations, question_ids, answer_ids, template_ids)

    assert batch.activations.shape == (3, 3)
    assert len(batch.questions) == len(default_activation_questions())
    assert batch.answer_ids.tolist() == [1, 0, 1]


def test_oracle_comparison_report_beats_text_and_matches_probe():
    answer_ids = t.tensor([0, 1, 0, 1])
    oracle_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    text_only_logits = t.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    linear_probe_logits = oracle_logits.clone()
    mlp_probe_logits = text_only_logits.clone()
    sae_logits = text_only_logits.clone()

    report = oracle_comparison_report(
        oracle_logits,
        text_only_logits,
        linear_probe_logits,
        mlp_probe_logits,
        sae_logits,
        answer_ids,
    )

    assert report.oracle_accuracy == 1.0
    assert report.text_only_accuracy == 0.5
    assert report.linear_probe_accuracy == 1.0
    assert report.beats_text_only
    assert report.beats_or_matches_probe


def test_split_accuracy_by_template_reports_each_template():
    logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    answers = t.tensor([0, 1, 1, 0])
    template_ids = t.tensor([0, 0, 1, 1])

    accuracies = split_accuracy_by_template(logits, answers, template_ids)

    assert accuracies == {0: 1.0, 1: 0.5}


def test_ood_generalization_report_requires_all_splits_to_pass():
    correct_logits = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    answers = t.tensor([0, 1])

    report = ood_generalization_report(
        heldout_template_logits=correct_logits,
        heldout_template_answers=answers,
        new_name_logits=correct_logits,
        new_name_answers=answers,
        long_context_logits=correct_logits,
        long_context_answers=answers,
        adversarial_logits=correct_logits,
        adversarial_answers=answers,
        min_accuracy=0.75,
    )

    assert report.heldout_template_accuracy == 1.0
    assert report.new_name_accuracy == 1.0
    assert report.long_context_accuracy == 1.0
    assert report.adversarial_accuracy == 1.0
    assert report.passes_ood


def test_random_activation_oracle_report_checks_abstention_and_confidence():
    random_logits = t.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])

    report = random_activation_oracle_report(
        random_logits,
        abstain_answer_id=2,
        min_abstention_rate=1.0,
        max_mean_confidence=0.4,
    )

    assert report.abstention_rate == 1.0
    assert report.mean_confidence < 0.4
    assert report.passes_graceful_failure


def test_activation_patching_oracle_report_detects_answer_change():
    original = t.tensor([2.0, 0.0])
    patched = t.tensor([0.0, 3.0])

    report = activation_patching_oracle_report(original, patched)

    assert report.original_answer == 0
    assert report.patched_answer == 1
    assert report.changed
