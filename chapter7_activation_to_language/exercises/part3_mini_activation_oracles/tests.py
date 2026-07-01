from collections.abc import Callable

import torch as t

from arena_ext import activation_oracles as reference


def _solutions():
    from chapter7_activation_to_language.exercises.part3_mini_activation_oracles import (
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


def test_build_activation_question_batch_validates_shapes_and_questions(
    build_activation_question_batch: Callable | None = None,
    default_activation_questions: Callable | None = None,
):
    solutions = _solutions()
    build_activation_question_batch = (
        build_activation_question_batch or solutions.build_activation_question_batch
    )
    default_activation_questions = default_activation_questions or solutions.default_activation_questions
    activations = t.eye(3)
    question_ids = t.tensor([0, 1, 2])
    answer_ids = t.tensor([1, 0, 1])
    template_ids = t.tensor([0, 0, 1])
    batch = build_activation_question_batch(
        activations,
        question_ids,
        answer_ids,
        template_ids,
    )
    expected = reference.build_activation_question_batch(
        activations,
        question_ids,
        answer_ids,
        template_ids,
    )
    assert batch.activations.shape == (3, 3), (
        "Activation-question batches should preserve [examples, d_model] activations."
    )
    assert batch.question_ids.dtype == t.long and batch.answer_ids.dtype == t.long, (
        "Question and answer ids should be converted to integer class-index tensors."
    )
    assert batch.questions == default_activation_questions() == expected.questions, (
        "The default question bank should be stable and match the reference contract."
    )
    try:
        build_activation_question_batch(t.ones(3), question_ids, answer_ids, template_ids)
    except ValueError as exc:
        assert "activations" in str(exc), (
            "Shape errors should explain that activations need rank-2 shape."
        )
    else:
        raise AssertionError("Rank-1 activations should be rejected with a ValueError.")
    print(
        "All tests in `test_build_activation_question_batch_validates_shapes_and_questions` passed!"
    )


def test_question_conditioned_oracle_uses_question_ids_not_copied_probe_logits(
    make_question_conditioned_rows: Callable | None = None,
    train_question_conditioned_oracle: Callable | None = None,
    oracle_logits_for_batch: Callable | None = None,
    train_activation_only_baseline: Callable | None = None,
):
    solutions = _solutions()
    make_question_conditioned_rows = (
        make_question_conditioned_rows or solutions.make_question_conditioned_rows
    )
    train_question_conditioned_oracle = (
        train_question_conditioned_oracle or solutions.train_question_conditioned_oracle
    )
    oracle_logits_for_batch = oracle_logits_for_batch or solutions.oracle_logits_for_batch
    train_activation_only_baseline = (
        train_activation_only_baseline or solutions._train_activation_only_baseline
    )
    residuals = t.tensor([[1.0, 0.0], [-1.0, 0.0], [0.8, 0.0], [-0.8, 0.0]])
    labels = t.tensor([1, 0, 1, 0])
    direction = t.tensor([1.0, 0.0])
    batch = make_question_conditioned_rows(residuals, labels)

    model, score_mean, score_std, train_loss = train_question_conditioned_oracle(
        batch,
        direction,
        steps=250,
        lr=0.05,
    )
    oracle_logits = oracle_logits_for_batch(
        model,
        batch,
        direction,
        score_mean,
        score_std,
    )
    probe_logits = train_activation_only_baseline(
        batch,
        batch,
        direction,
        score_mean,
        score_std,
        steps=150,
    )

    assert train_loss < 0.05, "The tiny question-conditioned oracle should fit the fixture."
    assert oracle_logits[0::2].argmax(dim=-1).ne(oracle_logits[1::2].argmax(dim=-1)).all(), (
        "The same activation should receive different answers for opposite questions."
    )
    assert not t.allclose(oracle_logits, probe_logits), (
        "Activation-only probe logits must be independently trained, not copied from oracle logits."
    )
    assert solutions._prediction_accuracy(probe_logits, batch.answer_ids) <= 0.75, (
        "A probe without question ids should not solve contradictory question rows."
    )
    print(
        "All tests in `test_question_conditioned_oracle_uses_question_ids_not_copied_probe_logits` passed!"
    )


def test_oracle_comparison_report_beats_text_and_probe_baselines(
    oracle_comparison_report: Callable | None = None,
):
    oracle_comparison_report = oracle_comparison_report or _solutions().oracle_comparison_report
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
    expected = reference.oracle_comparison_report(
        oracle_logits,
        text_only_logits,
        linear_probe_logits,
        mlp_probe_logits,
        sae_logits,
        answer_ids,
    )
    _assert_report_close(report, expected, msg="Oracle comparison report")
    assert report.oracle_accuracy == 1.0 and report.text_only_accuracy == 0.5, (
        "Oracle logits should solve the task and beat the text-only majority baseline."
    )
    assert report.beats_or_matches_probe, (
        "The report should record that the oracle matches the best probe baseline."
    )
    print(
        "All tests in `test_oracle_comparison_report_beats_text_and_probe_baselines` passed!"
    )


def test_template_split_and_ood_reports_expose_generalization_failures(
    split_accuracy_by_template: Callable | None = None,
    ood_generalization_report: Callable | None = None,
):
    solutions = _solutions()
    split_accuracy_by_template = split_accuracy_by_template or solutions.split_accuracy_by_template
    ood_generalization_report = ood_generalization_report or solutions.ood_generalization_report
    logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    answers = t.tensor([0, 1, 1, 0])
    template_ids = t.tensor([0, 0, 1, 1])
    template_scores = split_accuracy_by_template(logits, answers, template_ids)
    assert template_scores == {0: 1.0, 1: 0.5}, (
        "Template split accuracy should expose the weak held-out template separately."
    )

    correct_logits = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    correct_answers = t.tensor([0, 1])
    weak_logits = t.tensor([[2.0, 0.0], [2.0, 0.0]])
    report = ood_generalization_report(
        heldout_template_logits=correct_logits,
        heldout_template_answers=correct_answers,
        new_name_logits=correct_logits,
        new_name_answers=correct_answers,
        long_context_logits=correct_logits,
        long_context_answers=correct_answers,
        adversarial_logits=weak_logits,
        adversarial_answers=correct_answers,
        min_accuracy=0.75,
    )
    assert report.adversarial_accuracy == 0.5 and not report.passes_ood, (
        "OOD report should fail if any required split falls below the threshold."
    )
    print(
        "All tests in `test_template_split_and_ood_reports_expose_generalization_failures` passed!"
    )


def test_random_activation_report_requires_abstention_or_low_confidence(
    random_activation_oracle_report: Callable | None = None,
):
    random_activation_oracle_report = (
        random_activation_oracle_report or _solutions().random_activation_oracle_report
    )
    random_logits = t.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])
    report = random_activation_oracle_report(
        random_logits,
        abstain_answer_id=2,
        min_abstention_rate=1.0,
        max_mean_confidence=0.4,
    )
    expected = reference.random_activation_oracle_report(
        random_logits,
        abstain_answer_id=2,
        min_abstention_rate=1.0,
        max_mean_confidence=0.4,
    )
    _assert_report_close(report, expected, msg="Random activation report")
    assert report.abstention_rate == 1.0 and report.passes_graceful_failure, (
        "Random activations should route to the abstain class under this toy contract."
    )
    overconfident = random_activation_oracle_report(
        t.tensor([[8.0, 0.0, 0.1], [0.0, 8.0, 0.1]]),
        abstain_answer_id=2,
        min_abstention_rate=1.0,
        max_mean_confidence=0.4,
    )
    assert not overconfident.passes_graceful_failure, (
        "Overconfident non-abstain answers on random activations should fail the control."
    )
    print(
        "All tests in `test_random_activation_report_requires_abstention_or_low_confidence` passed!"
    )


def test_activation_patching_report_checks_answer_change(
    activation_patching_oracle_report: Callable | None = None,
):
    activation_patching_oracle_report = (
        activation_patching_oracle_report or _solutions().activation_patching_oracle_report
    )
    original = t.tensor([2.0, 0.0])
    patched = t.tensor([0.0, 3.0])
    report = activation_patching_oracle_report(original, patched)
    expected = reference.activation_patching_oracle_report(original, patched)
    _assert_report_close(report, expected, msg="Activation patching report")
    assert report.original_answer == 0 and report.patched_answer == 1 and report.changed, (
        "Patching report should record the clean answer, patched answer, and flip."
    )
    unchanged = activation_patching_oracle_report(original, t.tensor([4.0, 0.0]))
    assert not unchanged.changed, (
        "Patching report should not claim causal evidence when the answer is unchanged."
    )
    print("All tests in `test_activation_patching_report_checks_answer_change` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["batch"]["num_questions"] == 7, (
        "Notebook contract should include the complete activation-question bank."
    )
    assert result["comparison"]["beats_text_only"], (
        "Notebook contract should prove the oracle beats a text-only baseline."
    )
    assert result["comparison"]["beats_or_matches_probe"], (
        "Notebook contract should compare the oracle against probe-style baselines."
    )
    assert result["template_split"] == {0: 1.0, 1: 0.5}, (
        "Notebook contract should expose template-split accuracy."
    )
    assert result["ood"]["passes_ood"], (
        "Notebook contract should include an OOD report that passes on the toy all-correct splits."
    )
    assert result["random_activation"]["passes_graceful_failure"], (
        "Notebook contract should include a random-activation graceful-failure control."
    )
    assert result["patching"]["changed"], (
        "Notebook contract should include an activation-patching answer flip."
    )
    print("All tests in `test_notebook_contract` passed!")
