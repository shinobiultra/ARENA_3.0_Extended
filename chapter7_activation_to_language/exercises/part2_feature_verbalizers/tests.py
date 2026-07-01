from collections.abc import Callable

import torch as t

from arena_ext import feature_verbalizers as reference


def _solutions():
    from chapter7_activation_to_language.exercises.part2_feature_verbalizers import solutions

    return solutions


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} fields should match the independent reference."
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


def test_gather_verbalizer_examples_covers_top_bottom_random_contrastive(
    gather_verbalizer_examples: Callable | None = None,
):
    solutions = _solutions()
    gather_verbalizer_examples = gather_verbalizer_examples or solutions.gather_verbalizer_examples
    texts = ["alpha code", "beta def", "plain story", "quiet notes"]
    scores = t.tensor([0.9, 0.8, 0.2, 0.1])
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)
    examples = gather_verbalizer_examples(texts, scores, labels, k=2, threshold=0.5, seed=0)
    expected = reference.gather_verbalizer_examples(
        texts,
        scores,
        labels,
        k=2,
        threshold=0.5,
        seed=0,
    )
    assert [example.__dict__ for example in examples.top] == [
        example.__dict__ for example in expected.top
    ], "Top examples should be the highest-scoring examples with labels attached."
    assert [example.text for example in examples.bottom] == ["quiet notes", "plain story"], (
        "Bottom examples should be the lowest-scoring examples."
    )
    assert [example.text for example in examples.contrastive] == ["beta def", "plain story"], (
        "Contrastive examples should be nearest the activation threshold."
    )
    groups = (examples.top, examples.bottom, examples.random, examples.contrastive)
    assert all(
        example.kind in {"top", "bottom", "random", "contrastive"}
        for group in groups
        for example in group
    ), "Every selected example should retain its source kind."
    print(
        "All tests in `test_gather_verbalizer_examples_covers_top_bottom_random_contrastive` passed!"
    )


def test_gather_verbalizer_examples_rejects_bad_shapes_and_k(
    gather_verbalizer_examples: Callable | None = None,
):
    solutions = _solutions()
    gather_verbalizer_examples = gather_verbalizer_examples or solutions.gather_verbalizer_examples
    texts = ["alpha code", "beta def"]
    scores = t.tensor([0.9, 0.8])
    labels = t.tensor([1, 1], dtype=t.bool)
    try:
        gather_verbalizer_examples(texts, scores, labels, k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("k <= 0 should be rejected.")
    try:
        gather_verbalizer_examples(texts[:1], scores, labels, k=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched texts, scores, and labels should be rejected.")
    print("All tests in `test_gather_verbalizer_examples_rejects_bad_shapes_and_k` passed!")


def test_keyword_predictions_and_explanation_report_use_baseline_and_contrastives(
    keyword_explanation_predictions: Callable | None = None,
    explanation_prediction_report: Callable | None = None,
):
    solutions = _solutions()
    keyword_explanation_predictions = (
        keyword_explanation_predictions or solutions.keyword_explanation_predictions
    )
    explanation_prediction_report = (
        explanation_prediction_report or solutions.explanation_prediction_report
    )
    texts = ["write code", "plain story", "def fn", "quiet notes"]
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    predictions = keyword_explanation_predictions(texts, ["code", "def"])
    expected_predictions = reference.keyword_explanation_predictions(texts, ["code", "def"])
    t.testing.assert_close(
        predictions,
        expected_predictions,
        msg="Keyword verbalizer predictions should match case-insensitive substring matches.",
    )
    baseline = t.zeros(4, dtype=t.bool)
    contrastive_mask = t.tensor([False, False, True, True])
    report = explanation_prediction_report(predictions, labels, baseline, contrastive_mask)
    expected = reference.explanation_prediction_report(
        predictions,
        labels,
        baseline,
        contrastive_mask,
    )
    _assert_report_close(report, expected, msg="Explanation prediction report")
    assert report.accuracy == 1.0 and report.baseline_accuracy == 0.5, (
        "Explanation predictions should beat the always-negative baseline."
    )
    assert report.contrastive_accuracy == 1.0 and report.survives_contrastive, (
        "Explanation predictions should survive the contrastive near-miss subset."
    )
    print(
        "All tests in `test_keyword_predictions_and_explanation_report_use_baseline_and_contrastives` passed!"
    )


def test_keyword_predictions_reject_empty_explanation_terms(
    keyword_explanation_predictions: Callable | None = None,
):
    solutions = _solutions()
    keyword_explanation_predictions = (
        keyword_explanation_predictions or solutions.keyword_explanation_predictions
    )
    try:
        keyword_explanation_predictions(["write code"], ["", " "])
    except ValueError:
        pass
    else:
        raise AssertionError("Empty explanation terms should be rejected.")
    print("All tests in `test_keyword_predictions_reject_empty_explanation_terms` passed!")


def test_keyword_predictions_do_not_match_substrings(
    keyword_explanation_predictions: Callable | None = None,
):
    solutions = _solutions()
    keyword_explanation_predictions = (
        keyword_explanation_predictions or solutions.keyword_explanation_predictions
    )
    predictions = keyword_explanation_predictions(
        ["the catalog is open", "the cat sat here"],
        ["cat"],
    )
    t.testing.assert_close(
        predictions,
        t.tensor([False, True]),
        msg="Keyword explanations should match whole tokens, not substrings.",
    )
    print("All tests in `test_keyword_predictions_do_not_match_substrings` passed!")


def test_explanation_prediction_report_rejects_shape_mismatch(
    explanation_prediction_report: Callable | None = None,
):
    solutions = _solutions()
    explanation_prediction_report = (
        explanation_prediction_report or solutions.explanation_prediction_report
    )
    try:
        explanation_prediction_report(
            t.tensor([True, False]),
            t.tensor([True]),
            t.tensor([False, False]),
            t.tensor([True, True]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched prediction, label, and mask shapes should fail.")
    print("All tests in `test_explanation_prediction_report_rejects_shape_mismatch` passed!")


def test_learned_verbalizer_terms_do_not_use_heldout_only_words(
    learn_verbalizer_terms: Callable | None = None,
):
    solutions = _solutions()
    learn_verbalizer_terms = learn_verbalizer_terms or solutions.learn_verbalizer_terms
    texts = [
        "The cat sat on the",
        "The dog slept near the",
        "The bird flew over the",
        "The kite floated above the",
    ]
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)
    terms = learn_verbalizer_terms(texts, labels, top_k=3)
    assert set(terms).issubset({"cat", "sat", "dog", "slept"}), (
        "Learned terms should come from positive training examples only."
    )
    assert not {"flew", "floated"}.intersection(terms), (
        "Negative-only terms should not appear in a positive verbalizer."
    )
    print(
        "All tests in `test_learned_verbalizer_terms_do_not_use_heldout_only_words` passed!"
    )


def test_learned_verbalizer_terms_reject_bad_inputs(
    learn_verbalizer_terms: Callable | None = None,
):
    solutions = _solutions()
    learn_verbalizer_terms = learn_verbalizer_terms or solutions.learn_verbalizer_terms
    try:
        learn_verbalizer_terms(["The cat sat"], t.tensor([True]), top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("top_k <= 0 should be rejected.")
    try:
        learn_verbalizer_terms(["The cat sat", "The bird flew"], t.tensor([True]), top_k=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched texts and labels should be rejected.")
    print("All tests in `test_learned_verbalizer_terms_reject_bad_inputs` passed!")


def test_counterexamples_and_revision_are_grounded_in_failures(
    find_counterexamples: Callable | None = None,
    revise_explanation: Callable | None = None,
):
    solutions = _solutions()
    find_counterexamples = find_counterexamples or solutions.find_counterexamples
    revise_explanation = revise_explanation or solutions.revise_explanation
    texts = ["write code", "plain story", "def fn", "quiet notes"]
    predictions = t.tensor([1, 1, 1, 0], dtype=t.bool)
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)
    report = find_counterexamples(texts, predictions, labels)
    expected = reference.find_counterexamples(texts, predictions, labels)
    _assert_report_close(report, expected, msg="Counterexample report")
    revised = revise_explanation(
        "Feature activates on code.",
        report.counterexamples,
        revision_note="Exclude ordinary stories.",
    )
    assert report.num_counterexamples == 1 and report.counterexamples == ("plain story",), (
        "Counterexample report should identify the single false positive."
    )
    assert revised.endswith("Revision: Exclude ordinary stories."), (
        "Revision should be appended only when counterexamples exist."
    )
    assert revise_explanation("Feature activates on code.", (), revision_note="No change.") == (
        "Feature activates on code."
    ), "Revision should leave the explanation unchanged when there are no counterexamples."
    print("All tests in `test_counterexamples_and_revision_are_grounded_in_failures` passed!")


def test_counterexamples_reject_bad_inputs(
    find_counterexamples: Callable | None = None,
):
    solutions = _solutions()
    find_counterexamples = find_counterexamples or solutions.find_counterexamples
    try:
        find_counterexamples(["write code"], t.tensor([True, False]), t.tensor([True, False]))
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched texts, predictions, and labels should be rejected.")
    try:
        find_counterexamples(["write code"], t.tensor([True]), t.tensor([False]), max_examples=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_examples <= 0 should be rejected.")
    print("All tests in `test_counterexamples_reject_bad_inputs` passed!")


def test_intervention_prediction_checks_signed_direction(
    intervention_prediction_report: Callable | None = None,
):
    solutions = _solutions()
    intervention_prediction_report = (
        intervention_prediction_report or solutions.intervention_prediction_report
    )
    baseline = t.tensor([0.1, 0.2])
    intervened = t.tensor([0.5, 0.6])
    report = intervention_prediction_report(
        baseline,
        intervened,
        predicted_direction="increase",
    )
    expected = reference.intervention_prediction_report(
        baseline,
        intervened,
        predicted_direction="increase",
    )
    _assert_report_close(report, expected, msg="Intervention prediction report")
    assert abs(report.observed_delta - 0.4) < 1e-6 and report.matches_prediction, (
        "Observed intervention delta should match the predicted increase."
    )
    wrong = intervention_prediction_report(
        baseline,
        intervened,
        predicted_direction="decrease",
    )
    assert not wrong.matches_prediction, (
        "Intervention report should fail when the observed sign contradicts the prediction."
    )
    print("All tests in `test_intervention_prediction_checks_signed_direction` passed!")


def test_intervention_prediction_rejects_invalid_direction(
    intervention_prediction_report: Callable | None = None,
):
    solutions = _solutions()
    intervention_prediction_report = (
        intervention_prediction_report or solutions.intervention_prediction_report
    )
    try:
        intervention_prediction_report(
            t.tensor([0.1, 0.2]),
            t.tensor([0.5, 0.6]),
            predicted_direction="sideways",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid predicted_direction should be rejected.")
    print("All tests in `test_intervention_prediction_rejects_invalid_direction` passed!")


def test_explanation_brevity_compares_against_examples_only_baseline(
    explanation_brevity_report: Callable | None = None,
):
    solutions = _solutions()
    explanation_brevity_report = (
        explanation_brevity_report or solutions.explanation_brevity_report
    )
    explanation = "Activates on code snippets."
    examples = ["write python code", "define a function with def"]
    report = explanation_brevity_report(explanation, examples)
    expected = reference.explanation_brevity_report(explanation, examples)
    _assert_report_close(report, expected, msg="Brevity report")
    assert report.explanation_word_count == 4 and report.examples_word_count == 8, (
        "Brevity report should count explanation words and examples-only words."
    )
    assert report.shorter_than_examples, (
        "The explanation should compress the examples-only baseline."
    )
    print(
        "All tests in `test_explanation_brevity_compares_against_examples_only_baseline` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["examples"]["top"] == ["alpha code", "beta def"], (
        "Notebook contract should include top activating examples."
    )
    assert result["prediction"]["passes_baseline"], (
        "Notebook contract should include explanation predictions beating a baseline."
    )
    assert result["prediction"]["survives_contrastive"], (
        "Notebook contract should include contrastive near-miss performance."
    )
    assert result["counterexamples"]["num_counterexamples"] == 1, (
        "Notebook contract should include counterexample discovery and revision."
    )
    assert result["intervention"]["matches_prediction"], (
        "Notebook contract should include intervention direction matching the explanation."
    )
    assert result["brevity"]["shorter_than_examples"], (
        "Notebook contract should include a brevity check against examples-only baseline."
    )
    print("All tests in `test_notebook_contract` passed!")
