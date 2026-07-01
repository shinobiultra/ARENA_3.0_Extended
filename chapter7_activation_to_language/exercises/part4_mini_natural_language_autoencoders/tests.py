from collections.abc import Callable

import torch as t

from arena_ext import natural_language_autoencoders as reference


def _solutions():
    from chapter7_activation_to_language.exercises.part4_mini_natural_language_autoencoders import (
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


def test_build_nla_training_batch_validates_alignment(
    build_nla_training_batch: Callable | None = None,
):
    build_nla_training_batch = build_nla_training_batch or _solutions().build_nla_training_batch
    activations = t.eye(3)
    batch = build_nla_training_batch(
        activations,
        ["Alice gave Bob the book.", "def add(x, y):", "The answer is Paris."],
        ["ioi", "code", "fact"],
        ["indirect object is Bob", "python function", "stored capital fact"],
    )
    expected = reference.build_nla_training_batch(
        activations,
        ["Alice gave Bob the book.", "def add(x, y):", "The answer is Paris."],
        ["ioi", "code", "fact"],
        ["indirect object is Bob", "python function", "stored capital fact"],
    )
    assert batch.activations.shape == (3, 3), (
        "NLA batches should preserve the [examples, d_model] activation tensor."
    )
    assert batch.original_text_spans == expected.original_text_spans, (
        "Original text spans should stay aligned with activation rows."
    )
    assert batch.synthetic_latent_labels == ("ioi", "code", "fact"), (
        "Synthetic latent labels should be stored as an immutable tuple."
    )
    assert batch.generated_explanations[-1] == "stored capital fact", (
        "Generated explanations should stay in the same row order as activations."
    )
    try:
        build_nla_training_batch(
            activations,
            ["only one span"],
            ["ioi", "code", "fact"],
            ["indirect object is Bob", "python function", "stored capital fact"],
        )
    except ValueError as exc:
        assert "one entry per activation" in str(exc), (
            "Alignment errors should tell the learner that every text field needs one row per activation."
        )
    else:
        raise AssertionError("Mismatched text-field lengths should raise ValueError.")
    print("All tests in `test_build_nla_training_batch_validates_alignment` passed!")


def test_build_nla_training_batch_rejects_empty_batches(
    build_nla_training_batch: Callable | None = None,
):
    build_nla_training_batch = build_nla_training_batch or _solutions().build_nla_training_batch
    try:
        build_nla_training_batch(
            t.empty(0, 3),
            [],
            [],
            [],
        )
    except ValueError as exc:
        assert "at least one example" in str(exc), (
            "Empty NLA batches should be rejected before downstream metrics produce NaNs."
        )
    else:
        raise AssertionError("Empty NLA batches should raise ValueError.")
    print("All tests in `test_build_nla_training_batch_rejects_empty_batches` passed!")


def test_generated_explanations_do_not_hide_numeric_coefficients(
    numeric_literal_count: Callable | None = None,
):
    if numeric_literal_count is None:
        numeric_literal_count = getattr(_solutions(), "_numeric_literal_count")
    phrase_explanations = [
        "blanket lying on support",
        "rocket moving above path",
    ]
    coefficient_payloads = [
        "surface +3.761 -2.767 -1.806 +0.109",
        "motion -4.0",
    ]

    assert numeric_literal_count(phrase_explanations) == 0, (
        "Natural-language phrase explanations should not carry numeric coordinates."
    )
    assert numeric_literal_count(coefficient_payloads) == 5, (
        "Signed residual-coefficient payloads should be detectable and rejected by the full report."
    )
    print(
        "All tests in `test_generated_explanations_do_not_hide_numeric_coefficients` passed!"
    )


def test_activation_reconstruction_report_beats_text_only_baseline(
    activation_reconstruction_report: Callable | None = None,
):
    activation_reconstruction_report = (
        activation_reconstruction_report or _solutions().activation_reconstruction_report
    )
    original = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    reconstructed = t.tensor([[0.9, 0.1], [0.1, 0.9]])
    text_only = t.zeros_like(original)
    report = activation_reconstruction_report(original, reconstructed, text_only)
    expected = reference.activation_reconstruction_report(original, reconstructed, text_only)
    _assert_report_close(report, expected, msg="Reconstruction report")
    assert abs(report.activation_mse - 0.01) < 1e-6, (
        "The toy NLA reconstruction MSE should average the squared residual error."
    )
    assert abs(report.text_only_mse - 0.5) < 1e-6, (
        "The text-only baseline should be scored against the same original activations."
    )
    assert report.mean_cosine_similarity > 0.99 and report.beats_text_only, (
        "A useful mini NLA should be close in direction and beat the text-only baseline."
    )
    try:
        activation_reconstruction_report(original, reconstructed[:1], text_only)
    except ValueError as exc:
        assert "matching shape" in str(exc), (
            "Shape errors should explain that all reconstruction tensors must align."
        )
    else:
        raise AssertionError("Shape-mismatched reconstructions should raise ValueError.")
    print(
        "All tests in `test_activation_reconstruction_report_beats_text_only_baseline` passed!"
    )


def test_activation_reconstruction_report_rejects_empty_and_rank1_inputs(
    activation_reconstruction_report: Callable | None = None,
):
    activation_reconstruction_report = (
        activation_reconstruction_report or _solutions().activation_reconstruction_report
    )
    for bad in [t.empty(0, 2), t.ones(2)]:
        try:
            activation_reconstruction_report(bad, bad.clone(), bad.clone())
        except ValueError as exc:
            assert "activation" in str(exc), (
                "Invalid activation tensors should fail before MSE/cosine summaries are computed."
            )
        else:
            raise AssertionError("Invalid activation tensors should raise ValueError.")
    print(
        "All tests in `test_activation_reconstruction_report_rejects_empty_and_rank1_inputs` passed!"
    )


def test_logit_diff_preservation_report_checks_actual_logit_diff(
    logit_diff_preservation_report: Callable | None = None,
):
    logit_diff_preservation_report = (
        logit_diff_preservation_report or _solutions().logit_diff_preservation_report
    )
    original_logits = t.tensor([[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    reconstructed_logits = t.tensor([[2.9, 1.1, 0.0], [2.1, 0.0, 1.1]])
    report = logit_diff_preservation_report(
        original_logits,
        reconstructed_logits,
        positive_token_id=0,
        negative_token_id=1,
        max_mean_abs_error=0.25,
    )
    assert abs(report.original_logit_diff - 2.0) < 1e-6, (
        "Original logit diff should average positive-minus-negative logits over the batch."
    )
    assert abs(report.reconstructed_logit_diff - 1.95) < 1e-6, (
        "Reconstructed logit diff should use the same positive-minus-negative readout."
    )
    assert 0.149 < report.mean_abs_error < 0.151, (
        "Mean absolute error should compare the per-example logit differences, not just one token."
    )
    assert report.preserves_target_logit_diff, (
        "The report should pass when mean logit-diff error is below the tolerance."
    )
    strict_report = logit_diff_preservation_report(
        original_logits,
        reconstructed_logits,
        positive_token_id=0,
        negative_token_id=1,
        max_mean_abs_error=0.05,
    )
    assert not strict_report.preserves_target_logit_diff, (
        "The same reconstruction should fail when the logit-diff tolerance is too strict."
    )
    print(
        "All tests in `test_logit_diff_preservation_report_checks_actual_logit_diff` passed!"
    )


def test_logit_diff_preservation_report_rejects_bad_inputs(
    logit_diff_preservation_report: Callable | None = None,
):
    logit_diff_preservation_report = (
        logit_diff_preservation_report or _solutions().logit_diff_preservation_report
    )
    logits = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    try:
        logit_diff_preservation_report(
            t.tensor([1.0, 0.0]),
            t.tensor([0.9, 0.1]),
            positive_token_id=0,
            negative_token_id=1,
        )
    except ValueError as exc:
        assert "logits" in str(exc), "Rank-1 logits should be rejected as non-batched."
    else:
        raise AssertionError("Rank-1 logits should raise ValueError.")
    try:
        logit_diff_preservation_report(
            logits,
            logits,
            positive_token_id=0,
            negative_token_id=1,
            max_mean_abs_error=-0.1,
        )
    except ValueError as exc:
        assert "non-negative" in str(exc), "Negative tolerances should be rejected."
    else:
        raise AssertionError("Negative max_mean_abs_error should raise ValueError.")
    print(
        "All tests in `test_logit_diff_preservation_report_rejects_bad_inputs` passed!"
    )


def test_latent_preservation_report_requires_accuracy_and_agreement(
    latent_preservation_report: Callable | None = None,
):
    latent_preservation_report = latent_preservation_report or _solutions().latent_preservation_report
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
    expected = reference.latent_preservation_report(
        original_logits,
        reconstructed_logits,
        latent_ids,
        min_accuracy=0.75,
        min_agreement=0.75,
    )
    _assert_report_close(report, expected, msg="Latent preservation report")
    assert report.preserves_latents, (
        "A reconstruction should pass when probe accuracy and prediction agreement stay high."
    )
    bad_logits = t.tensor([[0.0, 4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 4.0, 0.0]])
    bad_report = latent_preservation_report(
        original_logits,
        bad_logits,
        latent_ids,
        min_accuracy=0.75,
        min_agreement=0.75,
    )
    assert bad_report.reconstructed_probe_accuracy < 0.75, (
        "A collapsed reconstructed probe should expose low reconstructed accuracy."
    )
    assert not bad_report.preserves_latents, (
        "Low probe accuracy or agreement should fail latent preservation."
    )
    print(
        "All tests in `test_latent_preservation_report_requires_accuracy_and_agreement` passed!"
    )


def test_latent_preservation_report_rejects_invalid_thresholds(
    latent_preservation_report: Callable | None = None,
):
    latent_preservation_report = latent_preservation_report or _solutions().latent_preservation_report
    logits = t.eye(2)
    labels = t.tensor([0, 1])
    for kwargs in [{"min_accuracy": 1.1}, {"min_agreement": -0.1}]:
        try:
            latent_preservation_report(logits, logits, labels, **kwargs)
        except ValueError as exc:
            assert "between 0 and 1" in str(exc), (
                "Threshold validators should explain the valid probability range."
            )
        else:
            raise AssertionError("Invalid latent preservation thresholds should raise ValueError.")
    print(
        "All tests in `test_latent_preservation_report_rejects_invalid_thresholds` passed!"
    )


def test_brevity_and_counterfactual_reports_reject_prompt_copying(
    generated_text_brevity_report: Callable | None = None,
    counterfactual_explanation_report: Callable | None = None,
):
    solutions = _solutions()
    generated_text_brevity_report = (
        generated_text_brevity_report or solutions.generated_text_brevity_report
    )
    counterfactual_explanation_report = (
        counterfactual_explanation_report or solutions.counterfactual_explanation_report
    )
    generated = ["ioi target Bob", "python function"]
    prompts = [
        "Alice walked to the hall and gave Bob the book",
        "Please write a python function that adds two numbers",
    ]
    brevity = generated_text_brevity_report(generated, prompts)
    expected_brevity = reference.generated_text_brevity_report(generated, prompts)
    _assert_report_close(brevity, expected_brevity, msg="Brevity report")
    assert brevity.generated_word_count == 5 and brevity.original_word_count == 19, (
        "Brevity should count all generated explanation words and all original prompt words."
    )
    assert brevity.compression_ratio < 0.5 and brevity.shorter_than_original, (
        "The toy generated explanations should form a real text bottleneck."
    )
    copied = generated_text_brevity_report(prompts, prompts)
    assert not copied.shorter_than_original, (
        "Copying the prompt into the explanation should fail the compression check."
    )

    counterfactual = counterfactual_explanation_report(
        t.tensor([1.0, 0.0]),
        t.tensor([0.0, 1.0]),
        "indirect object is Bob",
        "indirect object is Alice",
        min_activation_delta=0.5,
    )
    expected_counterfactual = reference.counterfactual_explanation_report(
        t.tensor([1.0, 0.0]),
        t.tensor([0.0, 1.0]),
        "indirect object is Bob",
        "indirect object is Alice",
        min_activation_delta=0.5,
    )
    _assert_report_close(
        counterfactual,
        expected_counterfactual,
        msg="Counterfactual explanation report",
    )
    assert counterfactual.explanation_changed and counterfactual.activation_delta > 1.0, (
        "Counterfactual activations should change the generated explanation by a meaningful delta."
    )
    unchanged = counterfactual_explanation_report(
        t.tensor([1.0, 0.0]),
        t.tensor([0.0, 1.0]),
        "indirect object is Bob",
        "  Indirect Object Is Bob  ",
        min_activation_delta=0.5,
    )
    assert not unchanged.explanation_changed, (
        "Case and whitespace changes alone should not count as a counterfactual explanation change."
    )
    tiny_delta = counterfactual_explanation_report(
        t.tensor([1.0, 0.0]),
        t.tensor([1.1, 0.0]),
        "indirect object is Bob",
        "indirect object is Alice",
        min_activation_delta=0.5,
    )
    assert not tiny_delta.explanation_changed, (
        "Text changes should fail if the activation delta is below the requested threshold."
    )
    print(
        "All tests in `test_brevity_and_counterfactual_reports_reject_prompt_copying` passed!"
    )


def test_brevity_and_counterfactual_reports_reject_bad_controls(
    generated_text_brevity_report: Callable | None = None,
    counterfactual_explanation_report: Callable | None = None,
):
    solutions = _solutions()
    generated_text_brevity_report = (
        generated_text_brevity_report or solutions.generated_text_brevity_report
    )
    counterfactual_explanation_report = (
        counterfactual_explanation_report or solutions.counterfactual_explanation_report
    )
    try:
        generated_text_brevity_report([""], ["the original prompt has words"])
    except ValueError as exc:
        assert "contain text" in str(exc), (
            "Empty explanations should not pass just because they are short."
        )
    else:
        raise AssertionError("Empty generated explanations should raise ValueError.")
    try:
        counterfactual_explanation_report(
            t.tensor([1.0, 0.0]),
            t.tensor([0.0, 1.0]),
            "surface phrase",
            "motion phrase",
            min_activation_delta=-1.0,
        )
    except ValueError as exc:
        assert "non-negative" in str(exc), "Negative counterfactual thresholds should fail."
    else:
        raise AssertionError("Negative min_activation_delta should raise ValueError.")
    print(
        "All tests in `test_brevity_and_counterfactual_reports_reject_bad_controls` passed!"
    )


def test_trainable_discrete_bottleneck_learns_phrase_ids(
    train_discrete_nla_bottleneck: Callable | None = None,
):
    train_discrete_nla_bottleneck = (
        train_discrete_nla_bottleneck or _solutions().train_discrete_nla_bottleneck
    )
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
    expected = reference.train_discrete_nla_bottleneck(
        train_activations,
        train_phrase_ids,
        eval_activations,
        eval_phrase_ids,
        ("positive direction", "negative direction"),
        steps=120,
        lr=0.08,
        seed=0,
    )[-1]
    _assert_report_close(report, expected, msg="Trainable NLA bottleneck report")
    assert report.encoder_train_accuracy == 1.0, (
        "The activation-to-phrase encoder should learn the training phrase ids."
    )
    assert report.eval_phrase_accuracy == 1.0, (
        "The trained encoder should generalize to the held-out phrase-code fixture."
    )
    assert report.encoder_final_loss < 0.05, (
        "The encoder loss should decrease enough to show a real trained bottleneck."
    )
    assert report.beats_blank_text and report.reconstruction_mse < report.blank_text_mse, (
        "The phrase-to-activation decoder should beat a blank-text mean reconstruction."
    )
    assert report.generated_explanations == ("positive direction", "negative direction"), (
        "The transmitted bottleneck should be phrase text, not numeric coordinates."
    )
    print("All tests in `test_trainable_discrete_bottleneck_learns_phrase_ids` passed!")


def test_trainable_discrete_bottleneck_rejects_empty_splits(
    train_discrete_nla_bottleneck: Callable | None = None,
):
    train_discrete_nla_bottleneck = (
        train_discrete_nla_bottleneck or _solutions().train_discrete_nla_bottleneck
    )
    try:
        train_discrete_nla_bottleneck(
            t.empty(0, 2),
            t.empty(0, dtype=t.long),
            t.tensor([[1.0, 0.0]]),
            t.tensor([0]),
            ("positive direction",),
        )
    except ValueError as exc:
        assert "nonempty" in str(exc), (
            "Empty train/eval splits should be rejected before training starts."
        )
    else:
        raise AssertionError("Empty train/eval splits should raise ValueError.")
    print(
        "All tests in `test_trainable_discrete_bottleneck_rejects_empty_splits` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["batch"]["latent_labels"] == ["ioi", "code", "fact"], (
        "Notebook contract should include the aligned mini NLA batch."
    )
    assert result["reconstruction"]["beats_text_only"], (
        "Notebook contract should show reconstruction beating a text-only baseline."
    )
    assert result["logit_diff"]["preserves_target_logit_diff"], (
        "Notebook contract should check a preserved target logit difference."
    )
    assert 0.149 < result["logit_diff"]["mean_abs_error"] < 0.151, (
        "Notebook contract should report the corrected per-example logit-diff error."
    )
    assert result["latent_preservation"]["preserves_latents"], (
        "Notebook contract should include probe-latent preservation."
    )
    assert result["brevity"]["shorter_than_original"], (
        "Notebook contract should verify the generated text is a bottleneck."
    )
    assert result["counterfactual"]["explanation_changed"], (
        "Notebook contract should include a counterfactual explanation change."
    )
    assert result["trainable_bottleneck"]["eval_phrase_accuracy"] == 1.0, (
        "Notebook contract should include a trainable activation-to-phrase bottleneck."
    )
    assert result["trainable_bottleneck"]["beats_blank_text"], (
        "The trained phrase decoder should beat the blank-text reconstruction baseline."
    )
    print("All tests in `test_notebook_contract` passed!")
