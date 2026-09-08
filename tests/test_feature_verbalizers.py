import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.feature_verbalizers import (
        explanation_brevity_report,
        explanation_prediction_report,
        find_counterexamples,
        gather_verbalizer_examples,
        intervention_prediction_report,
        keyword_explanation_predictions,
        learn_verbalizer_terms,
        revise_explanation,
    )


def test_gather_verbalizer_examples_returns_four_buckets():
    texts = ["alpha code", "beta def", "plain story", "quiet notes"]
    scores = t.tensor([0.9, 0.8, 0.2, 0.1])
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)

    examples = gather_verbalizer_examples(texts, scores, labels, k=2, threshold=0.5, seed=0)

    assert [example.text for example in examples.top] == ["alpha code", "beta def"]
    assert [example.text for example in examples.bottom] == ["quiet notes", "plain story"]
    assert len(examples.random) == 2
    assert [example.text for example in examples.contrastive] == ["beta def", "plain story"]


def test_keyword_explanation_predictions_and_report_beat_baseline():
    texts = ["write code", "plain story", "def fn", "quiet notes"]
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    predictions = keyword_explanation_predictions(texts, ["code", "def"])
    baseline = t.zeros(4, dtype=t.bool)
    contrastive_mask = t.tensor([False, False, True, True])

    report = explanation_prediction_report(predictions, labels, baseline, contrastive_mask)

    assert predictions.tolist() == [True, False, True, False]
    assert report.accuracy == 1.0
    assert report.baseline_accuracy == 0.5
    assert report.contrastive_accuracy == 1.0
    assert report.passes_baseline
    assert report.survives_contrastive


def test_keyword_predictions_use_tokens_not_substrings():
    predictions = keyword_explanation_predictions(
        ["the catalog is open", "the cat sat here"],
        ["cat"],
    )

    assert predictions.tolist() == [False, True]


def test_learn_verbalizer_terms_uses_training_labels_only():
    texts = [
        "The cat sat on the",
        "The dog slept near the",
        "The bird flew over the",
        "The kite floated above the",
    ]
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)

    terms = learn_verbalizer_terms(texts, labels, top_k=3)

    assert set(terms).issubset({"cat", "sat", "dog", "slept"})
    assert "flew" not in terms
    assert "floated" not in terms


def test_find_counterexamples_and_revision():
    texts = ["write code", "plain story", "def fn", "quiet notes"]
    predictions = t.tensor([1, 1, 1, 0], dtype=t.bool)
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)

    report = find_counterexamples(texts, predictions, labels)
    revised = revise_explanation(
        "Feature activates on code.",
        report.counterexamples,
        revision_note="Exclude ordinary stories.",
    )

    assert report.num_counterexamples == 1
    assert report.counterexamples == ("plain story",)
    assert revised.endswith("Revision: Exclude ordinary stories.")


def test_intervention_prediction_report_matches_increase():
    baseline = t.tensor([0.1, 0.2])
    intervened = t.tensor([0.5, 0.6])

    report = intervention_prediction_report(
        baseline,
        intervened,
        predicted_direction="increase",
    )

    assert report.observed_delta == pytest.approx(0.4)
    assert report.matches_prediction


def test_explanation_brevity_report_compares_to_examples():
    explanation = "Activates on code snippets."
    examples = ["write python code", "define a function with def"]

    report = explanation_brevity_report(explanation, examples)

    assert report.explanation_word_count == 4
    assert report.examples_word_count == 8
    assert report.shorter_than_examples
