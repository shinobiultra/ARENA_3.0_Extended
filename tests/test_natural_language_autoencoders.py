import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.natural_language_autoencoders import (
        activation_reconstruction_report,
        build_nla_training_batch,
        counterfactual_explanation_report,
        generated_text_brevity_report,
        latent_preservation_report,
        logit_diff_preservation_report,
        numeric_literal_count,
        train_discrete_nla_bottleneck,
    )


def test_build_nla_training_batch_aligns_text_fields():
    activations = t.eye(3)
    batch = build_nla_training_batch(
        activations,
        ["Alice gave Bob the book.", "def add(x, y):", "The answer is Paris."],
        ["ioi", "code", "fact"],
        ["indirect object is Bob", "python function", "stored capital fact"],
    )

    assert batch.activations.shape == (3, 3)
    assert batch.original_text_spans[0] == "Alice gave Bob the book."
    assert batch.synthetic_latent_labels == ("ioi", "code", "fact")
    assert batch.generated_explanations[-1] == "stored capital fact"


def test_numeric_literal_count_rejects_coefficient_payloads():
    phrase_explanations = [
        "blanket lying on support",
        "rocket moving above path",
    ]
    coefficient_payloads = [
        "surface +3.761 -2.767 -1.806 +0.109",
        "motion -4.0",
    ]

    assert numeric_literal_count(phrase_explanations) == 0
    assert numeric_literal_count(coefficient_payloads) == 5


def test_activation_reconstruction_report_beats_text_only():
    original = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    reconstructed = t.tensor([[0.9, 0.1], [0.1, 0.9]])
    text_only = t.zeros_like(original)

    report = activation_reconstruction_report(original, reconstructed, text_only)

    assert report.activation_mse == pytest.approx(0.01)
    assert report.text_only_mse == pytest.approx(0.5)
    assert report.mean_cosine_similarity > 0.99
    assert report.beats_text_only


def test_logit_diff_preservation_report_tracks_target_diff():
    original_logits = t.tensor([[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    reconstructed_logits = t.tensor([[2.9, 1.1, 0.0], [2.1, 0.0, 1.1]])

    report = logit_diff_preservation_report(
        original_logits,
        reconstructed_logits,
        positive_token_id=0,
        negative_token_id=1,
        max_mean_abs_error=0.25,
    )

    assert report.original_logit_diff == pytest.approx(2.0)
    assert report.reconstructed_logit_diff == pytest.approx(1.95)
    assert report.mean_abs_error == pytest.approx(0.15)
    assert report.preserves_target_logit_diff


def test_latent_preservation_report_checks_probe_predictions():
    latent_ids = t.tensor([0, 1, 2])
    original_logits = t.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    reconstructed_logits = t.tensor(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.5, 2.0]]
    )

    report = latent_preservation_report(
        original_logits,
        reconstructed_logits,
        latent_ids,
        min_accuracy=0.75,
        min_agreement=0.75,
    )

    assert report.original_probe_accuracy == 1.0
    assert report.reconstructed_probe_accuracy == 1.0
    assert report.prediction_agreement == 1.0
    assert report.preserves_latents


def test_generated_text_brevity_report_requires_compression():
    generated = ["ioi target Bob", "python function"]
    prompts = [
        "Alice walked to the hall and gave Bob the book",
        "Please write a python function that adds two numbers",
    ]

    report = generated_text_brevity_report(generated, prompts)

    assert report.generated_word_count == 5
    assert report.original_word_count == 19
    assert report.compression_ratio == pytest.approx(5 / 19)
    assert report.shorter_than_original


def test_counterfactual_explanation_report_detects_changed_text():
    original_activation = t.tensor([1.0, 0.0])
    counterfactual_activation = t.tensor([0.0, 1.0])

    report = counterfactual_explanation_report(
        original_activation,
        counterfactual_activation,
        "indirect object is Bob",
        "indirect object is Alice",
        min_activation_delta=0.5,
    )

    assert report.activation_delta == pytest.approx(2**0.5)
    assert report.explanation_changed


def test_train_discrete_nla_bottleneck_learns_phrase_codes():
    train_activations = t.tensor(
        [
            [2.0, 0.0],
            [1.8, 0.1],
            [-2.0, 0.0],
            [-1.8, -0.1],
        ]
    )
    eval_activations = t.tensor([[1.9, 0.0], [-1.9, 0.0]])
    train_phrase_ids = t.tensor([0, 0, 1, 1])
    eval_phrase_ids = t.tensor([0, 1])

    *_, report = train_discrete_nla_bottleneck(
        train_activations,
        train_phrase_ids,
        eval_activations,
        eval_phrase_ids,
        ("positive direction", "negative direction"),
        steps=120,
        lr=0.08,
        seed=0,
    )

    assert report.encoder_train_accuracy == 1.0
    assert report.eval_phrase_accuracy == 1.0
    assert report.encoder_final_loss < 0.05
    assert report.reconstruction_mse < report.blank_text_mse
    assert report.beats_blank_text
    assert report.generated_explanations == ("positive direction", "negative direction")
