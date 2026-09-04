import ast
import json
from collections.abc import Callable
from pathlib import Path

import torch as t

from arena_ext import activation_oracles as reference


SECTION_DIR = Path(__file__).resolve().parent


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


def test_build_activation_question_batch_rejects_out_of_range_question_ids(
    build_activation_question_batch: Callable | None = None,
):
    solutions = _solutions()
    build_activation_question_batch = (
        build_activation_question_batch or solutions.build_activation_question_batch
    )
    activations = t.eye(2)
    answer_ids = t.tensor([1, 0])
    template_ids = t.tensor([0, 1])
    try:
        build_activation_question_batch(
            activations,
            t.tensor([0, 3]),
            answer_ids,
            template_ids,
            questions=("question zero", "question one"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("question_ids outside the question bank should be rejected.")
    try:
        build_activation_question_batch(
            activations,
            t.tensor([0, 1]),
            answer_ids,
            template_ids,
            questions=(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Empty question banks should be rejected.")
    print(
        "All tests in `test_build_activation_question_batch_rejects_out_of_range_question_ids` passed!"
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


def test_ood_generalization_report_rejects_invalid_threshold(
    ood_generalization_report: Callable | None = None,
):
    solutions = _solutions()
    ood_generalization_report = ood_generalization_report or solutions.ood_generalization_report
    logits = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    answers = t.tensor([0, 1])
    try:
        ood_generalization_report(
            heldout_template_logits=logits,
            heldout_template_answers=answers,
            new_name_logits=logits,
            new_name_answers=answers,
            long_context_logits=logits,
            long_context_answers=answers,
            adversarial_logits=logits,
            adversarial_answers=answers,
            min_accuracy=1.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("min_accuracy outside [0, 1] should be rejected.")
    print(
        "All tests in `test_ood_generalization_report_rejects_invalid_threshold` passed!"
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


def test_random_activation_report_rejects_bad_rank_and_thresholds(
    random_activation_oracle_report: Callable | None = None,
):
    random_activation_oracle_report = (
        random_activation_oracle_report or _solutions().random_activation_oracle_report
    )
    try:
        random_activation_oracle_report(t.tensor([0.0, 0.1]), abstain_answer_id=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Rank-1 random logits should be rejected.")
    try:
        random_activation_oracle_report(
            t.tensor([[0.0, 0.1]]),
            abstain_answer_id=1,
            min_abstention_rate=-0.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid abstention threshold should be rejected.")
    print(
        "All tests in `test_random_activation_report_rejects_bad_rank_and_thresholds` passed!"
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


def test_activation_patching_report_rejects_incompatible_logits(
    activation_patching_oracle_report: Callable | None = None,
):
    activation_patching_oracle_report = (
        activation_patching_oracle_report or _solutions().activation_patching_oracle_report
    )
    try:
        activation_patching_oracle_report(t.tensor([[2.0, 0.0]]), t.tensor([0.0, 3.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Rank-2 logits should be rejected for a single patching row.")
    try:
        activation_patching_oracle_report(t.tensor([2.0, 0.0]), t.tensor([0.0, 3.0, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Original and patched logits must have matching answer classes.")
    print(
        "All tests in `test_activation_patching_report_rejects_incompatible_logits` passed!"
    )


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
    assert result["model_organism"]["baseline_accuracies"]["LoRA oracle"] == 1.0, (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    assert result["model_organism"]["baseline_accuracies"]["text only"] == 0.5, (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    assert result["model_organism"]["patch_changed_questions"] == [0, 2], (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    print("All tests in `test_notebook_contract` passed!")


def test_factor_world_has_exact_ground_truth(
    make_factor_world: Callable | None = None,
):
    make_factor_world = make_factor_world or _solutions().make_factor_world
    world = make_factor_world("train", repeats=3)
    decoded = world.activations @ world.mixing
    assert world.activations.shape == (12, 8), (
        'The factor world must preserve its exact truth table and orthogonal coordinate contract before the oracle is evaluated.'
    )
    assert t.allclose(decoded[:, :3], world.latent_factors, atol=1e-6), (
        "The first three decoded coordinates must exactly recover color, shape, and their interaction."
    )
    assert t.allclose(decoded[:, 2], decoded[:, 0] * decoded[:, 1], atol=1e-6), (
        "The interaction coordinate must equal color times shape."
    )
    assert t.allclose(decoded[:, -1], t.zeros(12), atol=1e-6), (
        "The reserved off-manifold coordinate should be exactly zero."
    )
    assert world.latent_factors.unique(dim=0).shape[0] == 4, (
        "Every binary factor combination must appear."
    )
    for bad_call in (
        lambda: make_factor_world("unknown"),
        lambda: make_factor_world("train", repeats=0),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid model-organism configurations must fail loudly.")
    print("All tests in `test_factor_world_has_exact_ground_truth` passed!")


def test_factor_question_rows_require_question_conditioning(
    make_factor_world: Callable | None = None,
    make_factor_question_rows: Callable | None = None,
):
    solutions = _solutions()
    make_factor_world = make_factor_world or solutions.make_factor_world
    make_factor_question_rows = make_factor_question_rows or solutions.make_factor_question_rows
    world = make_factor_world("train", repeats=1)
    batch = make_factor_question_rows(world)
    expected_answers = t.tensor(
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1]
    )
    assert batch.activations.shape == (12, 8), (
        'Activation-question rows must preserve activation-major alignment and contradictory labels that force question conditioning.'
    )
    assert batch.questions == solutions.FACTOR_QUESTIONS, (
        'Activation-question rows must preserve activation-major alignment and contradictory labels that force question conditioning.'
    )
    assert t.equal(batch.question_ids, t.arange(3).repeat(4)), (
        'Activation-question rows must preserve activation-major alignment and contradictory labels that force question conditioning.'
    )
    assert t.equal(batch.answer_ids.cpu(), expected_answers), (
        "Question answers must follow the exact color, shape, interaction truth table."
    )
    assert t.allclose(batch.activations[0], batch.activations[1]), (
        'Activation-question rows must preserve activation-major alignment and contradictory labels that force question conditioning.'
    )
    assert batch.answer_ids[0] != batch.answer_ids[2], (
        "The same activation must support different answers to different questions."
    )
    print("All tests in `test_factor_question_rows_require_question_conditioning` passed!")


def test_low_rank_oracle_freezes_base_weights(
    low_rank_cls=None,
    oracle_cls=None,
):
    solutions = _solutions()
    low_rank_cls = low_rank_cls or solutions.LowRankLinear
    oracle_cls = oracle_cls or solutions.MiniActivationOracle
    layer = low_rank_cls(5, 4, rank=2)
    assert not layer.base.weight.requires_grad and not layer.base.bias.requires_grad, (
        'The mini oracle must train only genuine low-rank adapters while preserving the declared input and output shapes.'
    )
    assert layer.lora_A.requires_grad and layer.lora_B.requires_grad, (
        'The mini oracle must train only genuine low-rank adapters while preserving the declared input and output shapes.'
    )
    assert layer(t.ones(3, 5)).shape == (3, 4), (
        'The mini oracle must train only genuine low-rank adapters while preserving the declared input and output shapes.'
    )
    oracle = oracle_cls(activation_dim=8)
    assert all(
        ("lora_" in name) == parameter.requires_grad
        for name, parameter in oracle.named_parameters()
    ), "Only LoRA matrices should be trainable in the mini oracle."
    assert oracle(t.zeros(6, 8), t.arange(3).repeat(2)).shape == (6, 2), (
        'The mini oracle must train only genuine low-rank adapters while preserving the declared input and output shapes.'
    )
    print("All tests in `test_low_rank_oracle_freezes_base_weights` passed!")


def test_mini_oracle_learns_three_question_truth_table(
    make_factor_world: Callable | None = None,
    make_factor_question_rows: Callable | None = None,
    train_mini_activation_oracle: Callable | None = None,
):
    solutions = _solutions()
    make_factor_world = make_factor_world or solutions.make_factor_world
    make_factor_question_rows = make_factor_question_rows or solutions.make_factor_question_rows
    train_mini_activation_oracle = (
        train_mini_activation_oracle or solutions.train_mini_activation_oracle
    )
    batch = make_factor_question_rows(make_factor_world("train"))
    model, losses = train_mini_activation_oracle(batch, steps=160)
    with t.inference_mode():
        predictions = model(batch.activations, batch.question_ids).argmax(dim=-1)
    assert float(losses[-1]) < 1e-3, (
        'The trained oracle must fit all three exact questions; a higher loss or missed row means the conditional routing was not learned.'
    )
    assert predictions.eq(batch.answer_ids).all(), (
        "The oracle should recover all three exact question functions."
    )
    predictions_by_question = predictions.reshape(-1, 3)
    assert predictions_by_question.unique(dim=0).shape[0] == 4, (
        "The oracle must route the same activation through distinct question semantics."
    )
    print("All tests in `test_mini_oracle_learns_three_question_truth_table` passed!")


def test_shortcut_baselines_fail_for_the_expected_reason(
    model_organism_baseline_accuracies: Callable | None = None,
):
    solutions = _solutions()
    model_organism_baseline_accuracies = (
        model_organism_baseline_accuracies
        or solutions.model_organism_baseline_accuracies
    )
    world = solutions.make_factor_world("train")
    batch = solutions.make_factor_question_rows(world)
    model, _ = solutions.train_mini_activation_oracle(batch, steps=160)
    scores = model_organism_baseline_accuracies(model, batch, batch, world.mixing)
    assert scores["LoRA oracle"] == 1.0, (
        'Each baseline must exhibit its predicted information limit so the oracle result cannot be explained by a shortcut.'
    )
    assert scores["text only"] == 0.5, (
        'Each baseline must exhibit its predicted information limit so the oracle result cannot be explained by a shortcut.'
    )
    assert scores["activation-only linear"] <= 0.75, (
        'Each baseline must exhibit its predicted information limit so the oracle result cannot be explained by a shortcut.'
    )
    assert scores["activation-only MLP"] <= 0.75, (
        'Each baseline must exhibit its predicted information limit so the oracle result cannot be explained by a shortcut.'
    )
    assert scores["linear probe bank"] == 1.0, (
        'Each baseline must exhibit its predicted information limit so the oracle result cannot be explained by a shortcut.'
    )
    assert scores["exact feature classifier"] == 1.0, (
        'Each baseline must exhibit its predicted information limit so the oracle result cannot be explained by a shortcut.'
    )
    print("All tests in `test_shortcut_baselines_fail_for_the_expected_reason` passed!")


def test_ood_splits_and_random_activations_are_visible_controls(
    evaluate_factor_ood_splits: Callable | None = None,
    factor_manifold_distance: Callable | None = None,
    add_off_manifold_abstention: Callable | None = None,
):
    solutions = _solutions()
    evaluate_factor_ood_splits = evaluate_factor_ood_splits or solutions.evaluate_factor_ood_splits
    factor_manifold_distance = factor_manifold_distance or solutions.factor_manifold_distance
    add_off_manifold_abstention = (
        add_off_manifold_abstention or solutions.add_off_manifold_abstention
    )
    world = solutions.make_factor_world("train")
    batch = solutions.make_factor_question_rows(world)
    model, _ = solutions.train_mini_activation_oracle(batch, steps=160)
    ood = evaluate_factor_ood_splits(model)
    assert set(ood) == {
        "heldout_template",
        "new_names",
        "long_context",
        "adversarial_distractor",
    }, (
        'Named nuisance shifts and off-manifold random activations must remain separate visible controls with their declared outcomes.'
    )
    assert min(ood.values()) == 1.0, "Every named nuisance shift should be reported separately."

    generator = t.Generator().manual_seed(777)
    random_activations = t.randn(256, 8, generator=generator) * 1.4
    random_questions = t.arange(3).repeat(86)[:256]
    with t.inference_mode():
        binary_logits = model(random_activations, random_questions)
    guarded = add_off_manifold_abstention(
        binary_logits, random_activations, world.mixing
    )
    assert factor_manifold_distance(world.activations, world.mixing).max() < 1e-5, (
        'Named nuisance shifts and off-manifold random activations must remain separate visible controls with their declared outcomes.'
    )
    assert guarded.argmax(dim=-1).eq(2).float().mean() > 0.95, (
        "Most random activations should visibly trigger abstention."
    )
    print("All tests in `test_ood_splits_and_random_activations_are_visible_controls` passed!")


def test_factor_patching_is_selective_not_just_any_answer_flip(
    patch_factor_activation: Callable | None = None,
):
    solutions = _solutions()
    patch_factor_activation = patch_factor_activation or solutions.patch_factor_activation
    world = solutions.make_factor_world("train", repeats=1)
    source = world.activations[0]
    color_donor = world.activations[2]
    patched = patch_factor_activation(
        source, color_donor, world.mixing, factor="color"
    )
    decoded_source = source @ world.mixing
    decoded_patched = patched @ world.mixing
    assert t.allclose(decoded_patched[:3], t.tensor([1.0, -1.0, -1.0]), atol=1e-6), (
        'A structural factor patch must alter only the decoded factor and the answers causally downstream of it.'
    )
    assert t.allclose(decoded_patched[3:], decoded_source[3:], atol=1e-6), (
        "Patching color must preserve every nuisance coordinate."
    )
    source_answers = decoded_source[:3].gt(0)
    patched_answers = decoded_patched[:3].gt(0)
    assert t.equal(source_answers.ne(patched_answers), t.tensor([True, False, True])), (
        'A structural factor patch must alter only the decoded factor and the answers causally downstream of it.'
    )
    print("All tests in `test_factor_patching_is_selective_not_just_any_answer_flip` passed!")


def test_model_organism_signature_metrics_are_not_white_noise(
    run_model_organism_signature: Callable | None = None,
):
    run_model_organism_signature = (
        run_model_organism_signature or _solutions().run_model_organism_signature
    )
    result = run_model_organism_signature()
    assert result["baseline_accuracies"] == {
        "LoRA oracle": 1.0,
        "text only": 0.5,
        "activation-only linear": 0.75,
        "activation-only MLP": 0.75,
        "linear probe bank": 1.0,
        "exact feature classifier": 1.0,
    }, (
        'The signature metrics must reproduce the exact model-organism result and its falsifying controls.'
    )
    assert min(result["ood_accuracies"].values()) == 1.0, (
        'The signature metrics must reproduce the exact model-organism result and its falsifying controls.'
    )
    assert result["random_abstention_rate"] > 0.95, (
        'The signature metrics must reproduce the exact model-organism result and its falsifying controls.'
    )
    assert result["patch_before"] == [0, 0, 1], (
        'The signature metrics must reproduce the exact model-organism result and its falsifying controls.'
    )
    assert result["patch_after"] == [1, 0, 0], (
        'The signature metrics must reproduce the exact model-organism result and its falsifying controls.'
    )
    assert result["patch_changed_questions"] == [0, 2], (
        'The signature metrics must reproduce the exact model-organism result and its falsifying controls.'
    )
    print("All tests in `test_model_organism_signature_metrics_are_not_white_noise` passed!")


def test_solution_notebook_exposes_taught_implementations():
    path = SECTION_DIR / "7.3_Mini_Activation_Oracles_solutions.ipynb"
    notebook = json.loads(path.read_text())
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    code = "\n\n".join(code_cells)
    for cell_code in code_cells:
        ast.parse(cell_code)

    required_definitions = {
        "make_factor_world",
        "make_factor_question_rows",
        "LowRankLinear",
        "MiniActivationOracle",
        "train_mini_activation_oracle",
        "question_only_logits",
        "train_activation_only_classifier",
        "train_question_probe_bank",
        "exact_feature_classifier_logits",
        "evaluate_factor_ood_splits",
        "factor_manifold_distance",
        "add_off_manifold_abstention",
        "patch_factor_activation",
    }
    defined = {
        node.name
        for tree in (ast.parse(cell_code) for cell_code in code_cells)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert required_definitions <= defined, (
        "The solved notebook must contain every taught implementation inline."
    )
    assert "import solutions" not in code and "solutions." not in code, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert code.count("verification_report.json") == 1, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert "raise NotImplementedError" not in code, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert markdown.count("### Exercise") >= 7, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert markdown.count("<summary>Expected output") >= 7, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert markdown.count("<summary>Help") >= 7, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert markdown.count("<summary>Solution") >= 7, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    for marker in (
        "By the end of this notebook",
        "## Signature Result",
        "## Try It Yourself",
        "## Bonus Anomaly Hunt",
        "## Limitations",
    ):
        assert marker in markdown, (
            'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
        )
    assert "plt.subplots" in code and "savefig" in code, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    print("All tests in `test_solution_notebook_exposes_taught_implementations` passed!")
