import ast
import json
from collections.abc import Callable
from pathlib import Path

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
    assert result["planted_signature"]["signature_passed"], (
        "Notebook contract should include the planted feature-verbalizer signature result."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_gpu_report_payload_keeps_toy_and_real_metrics_distinct():
    solutions = _solutions()
    planted = solutions.run_planted_feature_verbalizer_signature_result(seed=0)
    combined = solutions._attach_planted_signature_metrics(
        {
            "preflight_passed": True,
            "baseline_accuracy": 0.5,
            "intervention_delta": 0.75,
        },
        planted,
    )
    assert combined["baseline_accuracy"] == 0.5
    assert combined["intervention_delta"] == 0.75
    assert combined["toy_baseline_accuracy"] == planted["metrics"]["baseline_accuracy"]
    assert combined["toy_intervention_delta"] == planted["metrics"]["intervention_delta"]
    assert combined["toy_planted_signature_passed"]
    assert combined["toy_activation_score_threshold_upper_bound_accuracy"] == 1.0
    print("All tests in `test_gpu_report_payload_keeps_toy_and_real_metrics_distinct` passed!")


def test_planted_feature_dataset_exact_rule_and_disjoint_splits(
    planted_feature_label: Callable | None = None,
    make_planted_feature_dataset: Callable | None = None,
    split_planted_feature_dataset: Callable | None = None,
):
    solutions = _solutions()
    planted_feature_label = planted_feature_label or solutions.planted_feature_label
    make_planted_feature_dataset = (
        make_planted_feature_dataset or solutions.make_planted_feature_dataset
    )
    split_planted_feature_dataset = (
        split_planted_feature_dataset or solutions.split_planted_feature_dataset
    )
    assert planted_feature_label("The cat sat on the mat.")
    assert not planted_feature_label("The toy cat sat on the mat.")
    assert not planted_feature_label("The catalog sat on the shelf."), (
        "The planted oracle should use whole-token semantic groups, not substrings."
    )
    split = split_planted_feature_dataset(make_planted_feature_dataset())
    assert len(split.train) >= 10 and len(split.heldout) >= 20
    assert not {example.text for example in split.train}.intersection(
        example.text for example in split.heldout
    ), "Train and held-out examples must be disjoint."
    assert {example.label for example in split.train} == {False, True}
    assert {example.label for example in split.revision} == {False, True}
    assert {example.label for example in split.heldout} == {False, True}
    assert all(not example.label for example in split.heldout if "decoy" in example.tags), (
        "Decoy toy/plush/statue/painted animals should be exact negative counterexamples."
    )
    print("All tests in `test_planted_feature_dataset_exact_rule_and_disjoint_splits` passed!")


def test_semantic_rule_predictions_require_all_groups_and_exclusions(
    semantic_rule_predictions: Callable | None = None,
):
    solutions = _solutions()
    semantic_rule_predictions = semantic_rule_predictions or solutions.semantic_rule_predictions
    rule = solutions.ExplanationRule(
        description="living animal resting on a surface",
        required_groups=("animal", "resting", "surface"),
        excluded_terms=("toy", "plush"),
    )
    texts = [
        "The cat sat on the mat.",
        "The toy cat sat on the mat.",
        "The cat sprinted across the mat.",
        "The catalog sat on the shelf.",
    ]
    predictions = semantic_rule_predictions(texts, rule)
    t.testing.assert_close(
        predictions,
        t.tensor([True, False, False, False]),
        msg="Semantic rules should require all named groups, honor exclusions, and use whole tokens.",
    )
    try:
        semantic_rule_predictions(["The cat sat on the mat."], solutions.ExplanationRule("", ()))
    except ValueError:
        pass
    else:
        raise AssertionError("Rules with no required semantic groups should be rejected.")
    print("All tests in `test_semantic_rule_predictions_require_all_groups_and_exclusions` passed!")


def test_counterexample_revision_improves_heldout_counterfactuals(
    semantic_rule_predictions: Callable | None = None,
    mine_counterexamples: Callable | None = None,
    revise_rule_from_counterexamples: Callable | None = None,
):
    solutions = _solutions()
    semantic_rule_predictions = semantic_rule_predictions or solutions.semantic_rule_predictions
    mine_counterexamples = mine_counterexamples or solutions.mine_counterexamples
    revise_rule_from_counterexamples = (
        revise_rule_from_counterexamples or solutions.revise_rule_from_counterexamples
    )
    split = solutions.split_planted_feature_dataset(solutions.make_planted_feature_dataset())
    initial_rule = solutions.ExplanationRule(
        description="feature fires on resting words",
        required_groups=("resting",),
    )
    revision_texts = [example.text for example in split.revision]
    revision_predictions = semantic_rule_predictions(revision_texts, initial_rule)
    counterexamples = mine_counterexamples(split.revision, revision_predictions, max_examples=8)
    revised_rule = revise_rule_from_counterexamples(initial_rule, counterexamples)
    assert len(counterexamples) >= 3
    assert set(revised_rule.required_groups) == {"animal", "resting", "surface"}
    assert {"toy", "plush", "statue", "painted", "cardboard"}.issubset(
        set(revised_rule.excluded_terms)
    ), "Revision should generalize from seen decoys to the full decoy category."
    heldout_texts = [example.text for example in split.heldout]
    labels = t.tensor([example.label for example in split.heldout], dtype=t.bool)
    initial_accuracy = semantic_rule_predictions(heldout_texts, initial_rule).eq(labels).float().mean()
    revised_accuracy = semantic_rule_predictions(heldout_texts, revised_rule).eq(labels).float().mean()
    assert initial_accuracy < 0.75
    assert revised_accuracy == 1.0
    print("All tests in `test_counterexample_revision_improves_heldout_counterfactuals` passed!")


def test_control_table_revised_beats_text_random_and_lookup_controls(
    control_prediction_table: Callable | None = None,
):
    solutions = _solutions()
    control_prediction_table = control_prediction_table or solutions.control_prediction_table
    split = solutions.split_planted_feature_dataset(solutions.make_planted_feature_dataset())
    initial_rule = solutions.ExplanationRule(
        description="feature fires on resting words",
        required_groups=("resting",),
    )
    counterexamples = solutions.mine_counterexamples(
        split.revision,
        solutions.semantic_rule_predictions(
            [example.text for example in split.revision],
            initial_rule,
        ),
    )
    revised_rule = solutions.revise_rule_from_counterexamples(initial_rule, counterexamples)
    rows = control_prediction_table(split.train, split.heldout, initial_rule, revised_rule)
    by_name = {str(row["control"]): row for row in rows}
    assert by_name["revised verbalizer"]["accuracy"] == 1.0
    for control_name in [
        "initial text-only rule",
        "always-negative base rate",
        "random keyword rule",
        "train-example lookup",
    ]:
        assert by_name[control_name]["accuracy"] < by_name["revised verbalizer"]["accuracy"], (
            f"Revised verbalizer should beat {control_name} on final held-out examples."
        )
    assert by_name["activation-score threshold"]["accuracy"] == 1.0, (
        "The score threshold is an oracle-style upper bound, not a defeated text baseline."
    )
    print(
        "All tests in `test_control_table_revised_beats_text_random_and_lookup_controls` passed!"
    )


def test_planted_intervention_direction_beats_random_direction(
    planted_intervention_direction_test: Callable | None = None,
):
    solutions = _solutions()
    planted_intervention_direction_test = (
        planted_intervention_direction_test or solutions.planted_intervention_direction_test
    )
    split = solutions.split_planted_feature_dataset(solutions.make_planted_feature_dataset())
    signature = solutions.run_planted_feature_verbalizer_signature_result()
    predictions = t.tensor(
        [row["revised_prediction"] for row in signature["heldout_rows"]],
        dtype=t.bool,
    )
    report = planted_intervention_direction_test(split.heldout, predictions, alpha=1.25)
    assert report.matches_prediction
    assert report.beats_random_direction
    assert abs(report.observed_delta - 1.25) < 1e-6
    assert abs(report.random_direction_delta) < 1e-6
    try:
        planted_intervention_direction_test(split.heldout, t.zeros(len(split.heldout), dtype=t.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("Intervention scoring should reject an all-negative verbalizer.")
    print("All tests in `test_planted_intervention_direction_beats_random_direction` passed!")


def test_planted_signature_result_contains_visible_evidence(
    run_planted_feature_verbalizer_signature_result: Callable | None = None,
):
    solutions = _solutions()
    run_planted_feature_verbalizer_signature_result = (
        run_planted_feature_verbalizer_signature_result
        or solutions.run_planted_feature_verbalizer_signature_result
    )
    result = run_planted_feature_verbalizer_signature_result(seed=0)
    assert result["signature_passed"]
    assert result["heldout_count"] >= 20
    assert len(result["heldout_rows"]) == result["heldout_count"]
    assert len(result["validation_counterexamples"]) >= 3
    assert result["metrics"]["revised_accuracy"] == 1.0
    assert result["metrics"]["initial_accuracy"] < 0.75
    assert result["metrics"]["target_beats_random_direction"]
    assert "verification_report" not in json.dumps(result), (
        "The signature should be generated from notebook-visible data, not loaded from a report."
    )
    print("All tests in `test_planted_signature_result_contains_visible_evidence` passed!")


def test_exercise_notebook_exposes_arena_learner_surface():
    notebook_path = Path(__file__).with_name("7.2_Feature_Verbalizers_exercises.ipynb")
    text = notebook_path.read_text()
    required_strings = [
        "By the end of this notebook",
        "### Exercise - implement the exact planted feature oracle",
        "### Exercise - gather top, bottom, random, and contrastive examples",
        "### Exercise - turn a semantic explanation into predictions",
        "### Exercise - mine counterexamples and revise the explanation",
        "### Exercise - compare the revised verbalizer against controls",
        "### Exercise - test the intervention direction",
        "### Exercise - build the signature result table",
        "Expected output",
        "Help -",
        "Interpreting the result",
        "<summary>Solution</summary>",
        "## Try It Yourself",
        "## Bonus: Hunt an Anomaly",
        "## Reading Links",
        "Bills et al.",
        "run_planted_feature_verbalizer_signature_result",
        "fig, axes = plt.subplots(1, 3",
        "The static image is only a preview",
        "feature_verbalizers_planted_signature.png",
    ]
    missing = [needle for needle in required_strings if needle not in text]
    assert not missing, f"Exercise notebook is missing ARENA learner-surface pieces: {missing}"
    assert text.count("### Exercise -") >= 7
    print("All tests in `test_exercise_notebook_exposes_arena_learner_surface` passed!")


def _notebook_sources(path: Path) -> tuple[list[str], list[str]]:
    notebook = json.loads(path.read_text())
    markdown_sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    code_sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    return markdown_sources, code_sources


def test_notebooks_expose_toy_ground_truth_without_reference_leakage():
    section_dir = Path(__file__).parent
    for filename in [
        "7.2_Feature_Verbalizers_exercises.ipynb",
        "7.2_Feature_Verbalizers_solutions.ipynb",
    ]:
        _, code_sources = _notebook_sources(section_dir / filename)
        code = "\n".join(code_sources)
        for required in [
            "ANIMAL_TERMS = frozenset",
            "RESTING_TERMS = frozenset",
            "SURFACE_TERMS = frozenset",
            "DECOY_TERMS = frozenset",
            "_PLANTED_EXAMPLE_SPECS:",
            '"The toy cat sat on the mat."',
            '"The catalog sat on the shelf."',
        ]:
            assert required in code, f"{filename} must expose toy ground truth: {required}"
        for forbidden in [
            "TOKEN_RE = reference.TOKEN_RE",
            "ANIMAL_TERMS = reference.ANIMAL_TERMS",
            "_PLANTED_EXAMPLE_SPECS = reference._PLANTED_EXAMPLE_SPECS",
        ]:
            assert forbidden not in code, (
                f"{filename} hides learner-visible ground truth behind solutions.py: {forbidden}"
            )
    print("All tests in `test_notebooks_expose_toy_ground_truth_without_reference_leakage` passed!")


def test_solution_notebook_mirrors_progression_and_inlines_taught_implementations():
    section_dir = Path(__file__).parent
    exercise_markdown, _ = _notebook_sources(
        section_dir / "7.2_Feature_Verbalizers_exercises.ipynb"
    )
    solution_markdown, solution_code = _notebook_sources(
        section_dir / "7.2_Feature_Verbalizers_solutions.ipynb"
    )

    exercise_headings = [
        line
        for source in exercise_markdown
        for line in source.splitlines()
        if line.startswith("### Exercise -")
    ]
    solution_headings = [
        line
        for source in solution_markdown
        for line in source.splitlines()
        if line.startswith("### Exercise -")
    ]
    assert solution_headings == exercise_headings, (
        "The solved notebook must mirror the complete seven-exercise learner progression."
    )

    required_functions = {
        "planted_feature_label",
        "_planted_feature_score",
        "_planted_tags",
        "make_planted_feature_dataset",
        "split_planted_feature_dataset",
        "_make_examples",
        "gather_verbalizer_examples",
        "keyword_explanation_predictions",
        "semantic_rule_predictions",
        "explanation_prediction_report",
        "find_counterexamples",
        "mine_counterexamples",
        "revise_explanation",
        "revise_rule_from_counterexamples",
        "examples_only_lookup_predictions",
        "random_keyword_predictions",
        "control_prediction_table",
        "intervention_prediction_report",
        "planted_intervention_direction_test",
        "explanation_brevity_report",
        "run_planted_feature_verbalizer_signature_result",
        "run_smoke_test",
    }
    defined_functions: set[str] = set()
    for index, source in enumerate(solution_code):
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"Solution code cell {index} does not parse: {error}") from error
        defined_functions.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )

    missing = sorted(required_functions - defined_functions)
    assert not missing, f"Solved notebook hides taught implementations: {missing}"
    assert all("raise NotImplementedError" not in source for source in solution_code), (
        "Solved notebook must contain no learner stubs."
    )
    print(
        "All tests in `test_solution_notebook_mirrors_progression_and_inlines_taught_implementations` passed!"
    )
