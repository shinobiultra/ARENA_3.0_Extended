"""Scientific and learner-surface tests for section 7.4."""

from __future__ import annotations

import ast
import inspect
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import torch as t

SECTION_DIR = Path(__file__).resolve().parent
EXERCISE_NOTEBOOK = SECTION_DIR / "7.4_Mini_Natural_Language_Autoencoders_exercises.ipynb"
SOLUTION_NOTEBOOK = SECTION_DIR / "7.4_Mini_Natural_Language_Autoencoders_solutions.ipynb"
PAGE = (
    SECTION_DIR.parents[1]
    / "instructions"
    / "pages"
    / "04_[7.4]_Mini_Natural_Language_Autoencoders.md"
)
REPORT = SECTION_DIR / "verification_report.json"


def _solutions():
    from chapter7_activation_to_language.exercises.part4_mini_natural_language_autoencoders import (
        solutions,
    )

    return solutions


@pytest.fixture(scope="module")
def cpu_bundle():
    solutions = _solutions()
    tokenizer, model = solutions.load_pythia_cpu()
    dataset = solutions.cache_pythia_nla_dataset(tokenizer, model)
    return tokenizer, model, dataset


@pytest.fixture(scope="module")
def experiment(cpu_bundle):
    tokenizer, model, dataset = cpu_bundle
    return _solutions().evaluate_mini_nla(tokenizer, model, dataset)


def check_compositional_split(build_agreement_examples):
    train, evaluation = build_agreement_examples()
    assert (
        len(train) == 96 and len(evaluation) == 32
    ), "check_compositional_split: expected len(train) == 96 and len(evaluation) == 32."
    all_examples = (*train, *evaluation)
    context_counts = Counter(example.context_id for example in all_examples)
    train_counts = Counter(example.context_id for example in train)
    eval_counts = Counter(example.context_id for example in evaluation)
    assert context_counts == Counter(
        {context_id: 4 for context_id in range(32)}
    ), "check_compositional_split: expected context_counts == Counter({context_id: 4 for context_id in range(32)})."
    assert train_counts == Counter(
        {context_id: 3 for context_id in range(32)}
    ), "check_compositional_split: expected train_counts == Counter({context_id: 3 for context_id in range(32)})."
    assert eval_counts == Counter(
        {context_id: 1 for context_id in range(32)}
    ), "check_compositional_split: expected eval_counts == Counter({context_id: 1 for context_id in range(32)})."
    assert Counter(example.factors for example in evaluation) == Counter(
        {state: 8 for state in _solutions().FACTOR_STATES}
    ), "check_compositional_split: expected Counter((example.factors for example in evaluation)) == Counter({state: 8 for state in _solutions().FACTOR_...."
    assert all(
        example.factors == example.held_out_state for example in evaluation
    ), "check_compositional_split: expected all((example.factors == example.held_out_state for example in evaluation))."
    assert not (
        {example.explanation for example in train} & {example.explanation for example in evaluation}
    ), "check_compositional_split: expected not {example.explanation for example in train} & {example.explanation for example in evaluation}."
    print("Compositional split: 32 contexts, 3 train + 1 held-out row each.")
    print("All tests in `check_compositional_split` passed!")


def check_tokenization(tokenizer, train_examples, tokenize_explanations):
    texts = tuple(example.explanation for example in train_examples)
    tokens = tokenize_explanations(tokenizer, texts)
    assert tokens.token_ids.shape == (96, 24), "Expected 96 fixed-length 24-token explanations."
    assert tokens.local_token_ids.shape == tokens.token_ids.shape, "Local token IDs must align."
    assert (
        len(tokens.vocabulary_token_ids) == 33
    ), "The frozen training vocabulary should have 33 tokens."
    assert tokens.attention_mask.all(), "Every fixed-length explanation position should be live."
    assert (
        tokens.token_ids[:, -1].eq(tokenizer.eos_token_id).all()
    ), "Every explanation must end in EOS."
    print("Tokenization: 24-token explanations over a 33-token GPT-NeoX vocabulary.")
    print("All tests in `check_tokenization` passed!")


def check_token_verbalizer(
    tokenizer,
    dataset,
    tokenize_explanations,
    fit_token_verbalizer,
    generate_token_explanations,
    token_generation_accuracy,
):
    train_text = tuple(example.explanation for example in dataset.train_examples)
    eval_text = tuple(example.explanation for example in dataset.eval_examples)
    train_tokens = tokenize_explanations(tokenizer, train_text)
    eval_tokens = tokenize_explanations(
        tokenizer,
        eval_text,
        vocabulary_token_ids=train_tokens.vocabulary_token_ids,
    )
    verbalizer = fit_token_verbalizer(dataset.train_activations, train_tokens)
    generated = generate_token_explanations(tokenizer, dataset.eval_activations, verbalizer)
    token_accuracy, exact_accuracy = token_generation_accuracy(generated, eval_tokens)
    assert (
        token_accuracy == 1.0 and exact_accuracy == 1.0
    ), "Every held-out token sequence should be exact."
    assert (
        len(set(generated.texts)) == 32
    ), "The AV must generate one distinct explanation per context."
    assert verbalizer.weight.shape[1] == 24 * 33, "The AV must score every position-token pair."
    print("Token AV: 792 token logits per row; 32/32 held-out sequences exact.")
    print("All tests in `check_token_verbalizer` passed!")


def check_reconstructor_language_features(
    tokenizer,
    model,
    encode_reconstructor_text,
):
    texts = (
        "The topic is key. The wording is plain. Quantity is one. Time is current.",
        "Current is Time. one is Quantity. plain is wording. key is topic The.",
    )
    features = encode_reconstructor_text(tokenizer, model, texts)
    assert features.shape == (
        2,
        512,
    ), "check_reconstructor_language_features: expected features.shape == (2, 512)."
    assert not t.allclose(
        features[0], features[1]
    ), "check_reconstructor_language_features: expected not t.allclose(features[0], features[1])."
    assert (
        float((features[0] - features[1]).norm()) > 1.0
    ), "check_reconstructor_language_features: expected float((features[0] - features[1]).norm()) > 1.0."
    print("AR language trunk: same words in different order produce different features.")
    print("All tests in `check_reconstructor_language_features` passed!")


def check_nonprototype_metrics(
    original,
    reconstructed,
    factors,
    reconstruction_metrics,
    assert_nonprototype_reconstruction,
):
    metrics = reconstruction_metrics(original, reconstructed, factors)
    assert metrics.fve > 0.75, "check_nonprototype_metrics: expected metrics.fve > 0.75."
    assert (
        metrics.within_factor_fve > 0.75
    ), "check_nonprototype_metrics: expected metrics.within_factor_fve > 0.75."
    assert (
        metrics.distinct_rows == 32
    ), "check_nonprototype_metrics: expected metrics.distinct_rows == 32."
    assert_nonprototype_reconstruction(original, reconstructed, factors)
    print(
        f"Reconstruction: FVE={metrics.fve:.3f}, within-factor FVE="
        f"{metrics.within_factor_fve:.3f}, distinct rows={metrics.distinct_rows}."
    )
    print("All tests in `check_nonprototype_metrics` passed!")


def check_bag_of_tokens_control(
    tokenizer,
    dataset,
    bag_of_tokens_features,
    fit_token_reconstructor,
    decode_text_features,
    fraction_variance_explained,
):
    train_texts = tuple(example.explanation for example in dataset.train_examples)
    eval_texts = tuple(example.explanation for example in dataset.eval_examples)
    train_features, vocabulary = bag_of_tokens_features(tokenizer, train_texts)
    eval_features, returned_vocabulary = bag_of_tokens_features(
        tokenizer,
        eval_texts,
        vocabulary_token_ids=vocabulary,
    )
    assert t.equal(
        vocabulary, returned_vocabulary
    ), "Evaluation must reuse the training vocabulary."
    decoder = fit_token_reconstructor(train_features, dataset.train_activations)
    reconstructed = decode_text_features(eval_features, decoder)
    score = fraction_variance_explained(dataset.eval_activations, reconstructed)
    assert (
        0.55 < score < 0.75
    ), "The lexical bypass should recover context but miss ordered factors."
    print(f"Bag-of-tokens control: normalized held-out FVE={score:.3f}.")
    print("All tests in `check_bag_of_tokens_control` passed!")


def check_identity_and_locality(model, dataset, patch_pythia_candidate_logits):
    identity = patch_pythia_candidate_logits(model, dataset, dataset.eval_activations)
    assert t.equal(
        identity, dataset.original_candidate_logits
    ), "check_identity_and_locality: expected t.equal(identity, dataset.original_candidate_logits)."
    replacement = dataset.eval_activations[1:2].clone()
    changed = patch_pythia_candidate_logits(
        model,
        dataset,
        replacement,
        row_indices=(0,),
    )
    assert (
        float((changed[0] - dataset.original_candidate_logits[0]).abs().max()) > 1e-3
    ), "check_identity_and_locality: expected float((changed[0] - dataset.original_candidate_logits[0]).abs().max()) > 0.001."
    assert t.equal(
        changed[1:], dataset.original_candidate_logits[1:]
    ), "check_identity_and_locality: expected t.equal(changed[1:], dataset.original_candidate_logits[1:])."
    print("Hook parity/locality: identity exact; a cross-example patch changes only row 0.")
    print("All tests in `check_identity_and_locality` passed!")


def test_compositional_split_withholds_one_state_per_lexical_context():
    check_compositional_split(_solutions().build_agreement_examples)
    print("All tests in `test_compositional_split_withholds_one_state_per_lexical_context` passed!")


def test_real_cache_is_nonzero_and_aligned(cpu_bundle):
    _, _, dataset = cpu_bundle
    assert dataset.train_activations.shape == (
        96,
        512,
    ), "test_real_cache_is_nonzero_and_aligned: expected dataset.train_activations.shape == (96, 512)."
    assert dataset.eval_activations.shape == (
        32,
        512,
    ), "test_real_cache_is_nonzero_and_aligned: expected dataset.eval_activations.shape == (32, 512)."
    assert (
        dataset.eval_input_ids.shape == dataset.eval_attention_mask.shape
    ), "test_real_cache_is_nonzero_and_aligned: expected dataset.eval_input_ids.shape == dataset.eval_attention_mask.shape."
    assert dataset.original_candidate_logits.shape == (
        32,
        4,
    ), "test_real_cache_is_nonzero_and_aligned: expected dataset.original_candidate_logits.shape == (32, 4)."
    assert (
        dataset.train_activations.device.type == "cpu"
    ), "test_real_cache_is_nonzero_and_aligned: expected dataset.train_activations.device.type == 'cpu'."
    assert (
        float(dataset.train_activations.std()) > 0.1
    ), "test_real_cache_is_nonzero_and_aligned: expected float(dataset.train_activations.std()) > 0.1."
    assert (
        int(t.unique(dataset.eval_activations, dim=0).shape[0]) == 32
    ), "test_real_cache_is_nonzero_and_aligned: expected int(t.unique(dataset.eval_activations, dim=0).shape[0]) == 32."
    print("All tests in `test_real_cache_is_nonzero_and_aligned` passed!")


def test_explanations_are_real_token_sequences(cpu_bundle):
    tokenizer, _, dataset = cpu_bundle
    check_tokenization(tokenizer, dataset.train_examples, _solutions().tokenize_explanations)
    train_tokens = _solutions().tokenize_explanations(
        tokenizer,
        tuple(example.explanation for example in dataset.train_examples),
    )
    assert (
        train_tokens.local_token_ids.ndim == 2
    ), "test_explanations_are_real_token_sequences: expected train_tokens.local_token_ids.ndim == 2."
    assert (
        train_tokens.local_token_ids.shape[1] > 8
    ), "test_explanations_are_real_token_sequences: expected train_tokens.local_token_ids.shape[1] > 8."
    assert set(dataset.train_factors.flatten().tolist()) == {
        -1.0,
        1.0,
    }, "test_explanations_are_real_token_sequences: expected set(dataset.train_factors.flatten().tolist()) == {-1.0, 1.0}."
    assert (
        train_tokens.local_token_ids.shape[1] != dataset.train_factors.shape[1]
    ), "test_explanations_are_real_token_sequences: expected train_tokens.local_token_ids.shape[1] != dataset.train_factors.shape[1]."
    print("All tests in `test_explanations_are_real_token_sequences` passed!")


def test_token_verbalizer_generalizes_the_omitted_compositions(cpu_bundle):
    tokenizer, _, dataset = cpu_bundle
    solutions = _solutions()
    check_token_verbalizer(
        tokenizer,
        dataset,
        solutions.tokenize_explanations,
        solutions.fit_token_verbalizer,
        solutions.generate_token_explanations,
        solutions.token_generation_accuracy,
    )
    print("All tests in `test_token_verbalizer_generalizes_the_omitted_compositions` passed!")


def test_reconstructor_uses_contextual_token_order(cpu_bundle):
    tokenizer, model, _ = cpu_bundle
    check_reconstructor_language_features(tokenizer, model, _solutions().encode_reconstructor_text)
    source = inspect.getsource(_solutions().reconstruct_from_explanations)
    assert (
        "encode_reconstructor_text" in source
    ), "test_reconstructor_uses_contextual_token_order: expected 'encode_reconstructor_text' in source."
    assert (
        "decode_text_features" in source
    ), "test_reconstructor_uses_contextual_token_order: expected 'decode_text_features' in source."
    print("All tests in `test_reconstructor_uses_contextual_token_order` passed!")


def test_end_to_end_reconstruction_preserves_within_factor_variation(experiment, cpu_bundle):
    _, _, dataset = cpu_bundle
    report = experiment.report
    check_nonprototype_metrics(
        dataset.eval_activations,
        experiment.reconstructed_activations,
        dataset.eval_factors,
        _solutions().reconstruction_metrics,
        _solutions().assert_nonprototype_reconstruction,
    )
    assert (
        report.token_accuracy == report.exact_text_accuracy == 1.0
    ), "test_end_to_end_reconstruction_preserves_within_factor_variation: expected report.token_accuracy == report.exact_text_accuracy == 1.0."
    assert (
        report.reconstruction.mean_cosine > 0.98
    ), "test_end_to_end_reconstruction_preserves_within_factor_variation: expected report.reconstruction.mean_cosine > 0.98."
    print("All tests in `test_end_to_end_reconstruction_preserves_within_factor_variation` passed!")


def test_order_sensitive_contextual_path_beats_bag_of_tokens(experiment, cpu_bundle):
    tokenizer, _, dataset = cpu_bundle
    solutions = _solutions()
    report = experiment.report
    assert (
        report.same_context_token_multiset_fraction == 1.0
    ), "All four factor states in each lexical context must use the same token multiset."
    assert (
        report.reconstruction.fve > report.bag_of_tokens.fve + 0.1
    ), "Contextual Pythia features must beat the order-blind lexical bypass on normalized FVE."
    assert (
        report.behavior_parity_accuracy > report.bag_of_tokens_behavior_parity + 0.5
    ), "The order-sensitive path must preserve behavior far better than bag-of-tokens."
    all_examples = (*dataset.train_examples, *dataset.eval_examples)
    features, _ = solutions.bag_of_tokens_features(
        tokenizer,
        tuple(example.explanation for example in all_examples),
    )
    for context_id in range(32):
        rows = [
            index for index, example in enumerate(all_examples) if example.context_id == context_id
        ]
        assert all(
            t.equal(features[rows[0]], features[row]) for row in rows[1:]
        ), f"Context {context_id} changes token counts across factor states."
    print("All tests in `test_order_sensitive_contextual_path_beats_bag_of_tokens` passed!")


def test_factor_and_prototype_shortcuts_fail_adversarial_checks(experiment, cpu_bundle):
    _, _, dataset = cpu_bundle
    solutions = _solutions()
    report = experiment.report
    assert (
        report.prompt_only_factor.fve < 0.0
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected report.prompt_only_factor.fve < 0.0."
    assert (
        report.direct_factor.fve < 0.1
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected report.direct_factor.fve < 0.1."
    assert (
        report.class_mean.fve < 0.1
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected report.class_mean.fve < 0.1."
    assert (
        report.prompt_only_factor.within_factor_fve <= 1e-3
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected report.prompt_only_factor.within_factor_fve <= 0.001."
    assert (
        len(set(solutions.prompt_only_factor_explanations(dataset.eval_examples))) == 4
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected len(set(solutions.prompt_only_factor_explanations(dataset.eval_examples))) == 4."
    assert (
        report.direct_factor.distinct_rows <= 4
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected report.direct_factor.distinct_rows <= 4."
    assert (
        report.class_mean.distinct_rows == 4
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected report.class_mean.distinct_rows == 4."
    assert (
        report.reconstruction.fve > report.class_mean.fve + 0.7
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected report.reconstruction.fve > report.class_mean.fve + 0.7."
    class_means = solutions.factor_prototype_reconstruction(
        dataset.train_activations,
        dataset.train_factors,
        dataset.eval_factors,
    )
    with pytest.raises(AssertionError, match="prototype collapse"):
        solutions.assert_nonprototype_reconstruction(
            dataset.eval_activations,
            class_means,
            dataset.eval_factors,
        )
    assert not t.allclose(
        experiment.reconstructed_activations, class_means, atol=1e-3
    ), "test_factor_and_prototype_shortcuts_fail_adversarial_checks: expected not t.allclose(experiment.reconstructed_activations, class_means, atol=0.001)."
    print("All tests in `test_factor_and_prototype_shortcuts_fail_adversarial_checks` passed!")


def test_class_mean_conditioning_does_not_reproduce_the_reconstructor(experiment):
    report = experiment.report
    assert (
        report.mean_conditioned.fve < 0.1
    ), "test_class_mean_conditioning_does_not_reproduce_the_reconstructor: expected report.mean_conditioned.fve < 0.1."
    assert (
        report.mean_conditioned.within_factor_fve <= 1e-3
    ), "test_class_mean_conditioning_does_not_reproduce_the_reconstructor: expected report.mean_conditioned.within_factor_fve <= 0.001."
    assert (
        report.reconstruction.fve > report.mean_conditioned.fve + 0.7
    ), "test_class_mean_conditioning_does_not_reproduce_the_reconstructor: expected report.reconstruction.fve > report.mean_conditioned.fve + 0.7."
    print(
        "All tests in `test_class_mean_conditioning_does_not_reproduce_the_reconstructor` passed!"
    )


def test_lexical_relabeling_destroys_row_alignment_not_just_factor_labels(experiment):
    report = experiment.report
    assert (
        report.lexical_relabel.fve < 0.0
    ), "test_lexical_relabeling_destroys_row_alignment_not_just_factor_labels: expected report.lexical_relabel.fve < 0.0."
    assert (
        report.lexical_relabel.within_factor_fve < 0.0
    ), "test_lexical_relabeling_destroys_row_alignment_not_just_factor_labels: expected report.lexical_relabel.within_factor_fve < 0.0."
    assert (
        report.reconstruction.fve > report.lexical_relabel.fve + 1.0
    ), "test_lexical_relabeling_destroys_row_alignment_not_just_factor_labels: expected report.reconstruction.fve > report.lexical_relabel.fve + 1.0."
    assert (
        report.lexical_relabel.distinct_rows == 32
    ), "test_lexical_relabeling_destroys_row_alignment_not_just_factor_labels: expected report.lexical_relabel.distinct_rows == 32."
    print(
        "All tests in `test_lexical_relabeling_destroys_row_alignment_not_just_factor_labels` passed!"
    )


def test_zero_filled_training_data_is_rejected(cpu_bundle):
    tokenizer, _, dataset = cpu_bundle
    solutions = _solutions()
    tokens = solutions.tokenize_explanations(
        tokenizer,
        tuple(example.explanation for example in dataset.train_examples),
    )
    with pytest.raises(ValueError, match="zero-filled"):
        solutions.fit_token_verbalizer(t.zeros_like(dataset.train_activations), tokens)
    with pytest.raises(ValueError, match="zero-filled"):
        solutions.fit_token_reconstructor(
            dataset.train_activations, t.zeros_like(dataset.train_activations)
        )
    print("All tests in `test_zero_filled_training_data_is_rejected` passed!")


def test_hook_has_identity_parity_and_single_row_locality(cpu_bundle):
    _, model, dataset = cpu_bundle
    check_identity_and_locality(model, dataset, _solutions().patch_pythia_candidate_logits)
    print("All tests in `test_hook_has_identity_parity_and_single_row_locality` passed!")


def test_real_text_edit_has_nonzero_row_local_causal_effect(experiment):
    intervention = experiment.report.intervention
    assert (
        intervention.activation_delta_norm > 1.0
    ), "test_real_text_edit_has_nonzero_row_local_causal_effect: expected intervention.activation_delta_norm > 1.0."
    assert (
        intervention.selected_logit_change > 1.0
    ), "test_real_text_edit_has_nonzero_row_local_causal_effect: expected intervention.selected_logit_change > 1.0."
    assert (
        intervention.other_rows_max_logit_change == 0.0
    ), "test_real_text_edit_has_nonzero_row_local_causal_effect: expected intervention.other_rows_max_logit_change == 0.0."
    assert (
        intervention.base_text != intervention.edited_text
    ), "test_real_text_edit_has_nonzero_row_local_causal_effect: expected intervention.base_text != intervention.edited_text."
    print("All tests in `test_real_text_edit_has_nonzero_row_local_causal_effect` passed!")


def test_omitted_composition_metric_is_end_to_end_and_honest(experiment):
    report = experiment.report
    assert (
        report.omitted_composition_end_to_end_parity == 1.0
    ), "test_omitted_composition_metric_is_end_to_end_and_honest: expected report.omitted_composition_end_to_end_parity == 1.0."
    assert (
        report.behavior_parity_accuracy == 1.0
    ), "test_omitted_composition_metric_is_end_to_end_and_honest: expected report.behavior_parity_accuracy == 1.0."
    assert report.original_task_accuracy == pytest.approx(
        30 / 32
    ), "test_omitted_composition_metric_is_end_to_end_and_honest: expected report.original_task_accuracy == pytest.approx(30 / 32)."
    assert report.reconstructed_task_accuracy == pytest.approx(
        30 / 32
    ), "test_omitted_composition_metric_is_end_to_end_and_honest: expected report.reconstructed_task_accuracy == pytest.approx(30 / 32)."
    assert (
        report.original_task_accuracy != report.omitted_composition_end_to_end_parity
    ), "test_omitted_composition_metric_is_end_to_end_and_honest: expected report.original_task_accuracy != report.omitted_composition_end_to_end_parity."
    print("All tests in `test_omitted_composition_metric_is_end_to_end_and_honest` passed!")


def test_each_shuffled_seed_is_reported_and_fails(experiment):
    report = experiment.report
    assert tuple(seed for seed, _ in report.shuffled_fve_by_seed) == tuple(
        range(8)
    ), "test_each_shuffled_seed_is_reported_and_fails: expected tuple((seed for seed, _ in report.shuffled_fve_by_seed)) == tuple(range(8))."
    assert tuple(seed for seed, _ in report.shuffled_behavior_parity_by_seed) == tuple(
        range(8)
    ), "test_each_shuffled_seed_is_reported_and_fails: expected tuple((seed for seed, _ in report.shuffled_behavior_parity_by_seed)) == tuple(range(8))."
    shuffled_fves = tuple(value for _, value in report.shuffled_fve_by_seed)
    shuffled_parities = tuple(value for _, value in report.shuffled_behavior_parity_by_seed)
    assert (
        len(set(shuffled_fves)) == 8
    ), "test_each_shuffled_seed_is_reported_and_fails: expected len(set(shuffled_fves)) == 8."
    assert (
        max(shuffled_fves) < 0.0
    ), "test_each_shuffled_seed_is_reported_and_fails: expected max(shuffled_fves) < 0.0."
    assert (
        max(shuffled_parities) <= 0.5
    ), "test_each_shuffled_seed_is_reported_and_fails: expected max(shuffled_parities) <= 0.5."
    assert (
        min(shuffled_parities) >= 0.0
    ), "test_each_shuffled_seed_is_reported_and_fails: expected min(shuffled_parities) >= 0.0."
    print("All tests in `test_each_shuffled_seed_is_reported_and_fails` passed!")


def test_fabricated_report_cannot_grant_acceptance():
    solutions = _solutions()
    fake_report = {"accepted": True, "fve": 999.0, "tests_passed": True}
    with pytest.raises(TypeError):
        solutions.run_smoke_test(report=fake_report)
    smoke = solutions.run_smoke_test()
    assert (
        "accepted" not in smoke
    ), "test_fabricated_report_cannot_grant_acceptance: expected 'accepted' not in smoke."
    assert (
        "tests_passed" not in smoke
    ), "test_fabricated_report_cannot_grant_acceptance: expected 'tests_passed' not in smoke."
    assert (
        "contract_passed" not in smoke
    ), "test_fabricated_report_cannot_grant_acceptance: expected 'contract_passed' not in smoke."
    source = inspect.getsource(solutions)
    assert (
        "verification_report.json" not in source
    ), "test_fabricated_report_cannot_grant_acceptance: expected 'verification_report.json' not in source."
    assert (
        "json.load" not in source
    ), "test_fabricated_report_cannot_grant_acceptance: expected 'json.load' not in source."
    stored_report = json.loads(REPORT.read_text())
    assert (
        stored_report["notebook_id"] == "7_4_mini_natural_language_autoencoders"
    ), "The release report must identify this exact learner section."
    assert (
        stored_report.get("report_inputs", {}).get("algorithm") == "sha256"
    ), "The release report must bind evidence to the current source files."
    print("All tests in `test_fabricated_report_cannot_grant_acceptance` passed!")


def _notebook_text(path: Path) -> tuple[str, str]:
    notebook = json.loads(path.read_text())
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return markdown, code


def _unfinished_functions(code: str) -> list[str]:
    result = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [
            statement
            for statement in node.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        if not body or all(isinstance(statement, (ast.Raise, ast.Pass)) for statement in body):
            result.append(node.name)
    return result


def test_notebooks_expose_progressive_core_functions_and_immediate_tests():
    required_functions = {
        "build_agreement_examples",
        "tokenize_explanations",
        "fit_token_verbalizer",
        "generate_token_explanations",
        "encode_reconstructor_text",
        "bag_of_tokens_features",
        "fit_token_reconstructor",
        "decode_text_features",
        "fraction_variance_explained",
        "within_factor_fve",
        "assert_nonprototype_reconstruction",
        "patch_pythia_candidate_logits",
        "run_text_edit_intervention",
    }
    immediate_checks = (
        "check_compositional_split",
        "check_tokenization",
        "check_token_verbalizer",
        "check_reconstructor_language_features",
        "check_nonprototype_metrics",
        "check_bag_of_tokens_control",
        "check_identity_and_locality",
    )
    for path in (EXERCISE_NOTEBOOK, SOLUTION_NOTEBOOK):
        markdown, code = _notebook_text(path)
        parsed = ast.parse(code)
        defined = {
            node.name
            for node in ast.walk(parsed)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert (
            required_functions <= defined
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected required_functions <= defined."
        assert all(
            name in code for name in immediate_checks
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected all((name in code for name in immediate_checks))."
        assert (
            "## Core Question" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected '## Core Question' in markdown."
        assert (
            "## Signature Result" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected '## Signature Result' in markdown."
        assert (
            "## Try It Yourself" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected '## Try It Yourself' in markdown."
        assert (
            "## Anomaly Hunt" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected '## Anomaly Hunt' in markdown."
        assert (
            "## Limitations" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected '## Limitations' in markdown."
        assert (
            "Expected output" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected 'Expected output' in markdown."
        assert (
            "<summary>Help" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected '<summary>Help' in markdown."
        assert (
            "<summary>Interpretation" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected '<summary>Interpretation' in markdown."
        assert (
            "<summary>Solution" in markdown
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected '<summary>Solution' in markdown."
        assert (
            "verification_report.json" not in code
        ), "test_notebooks_expose_progressive_core_functions_and_immediate_tests: expected 'verification_report.json' not in code."
    _, exercise_code = _notebook_text(EXERCISE_NOTEBOOK)
    _, solution_code = _notebook_text(SOLUTION_NOTEBOOK)
    assert required_functions <= set(
        _unfinished_functions(exercise_code)
    ), "The learner must implement every core operation."
    assert not _unfinished_functions(
        solution_code
    ), "The completed notebook must supply every operation."
    print(
        "All tests in `test_notebooks_expose_progressive_core_functions_and_immediate_tests` passed!"
    )


def test_solution_dropdowns_supply_every_exercise_dependency():
    notebook = json.loads(EXERCISE_NOTEBOOK.read_text())
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] != "markdown" or "<summary>Solution</summary>" not in source:
            continue
        count += 1
        dropdown = re.search(r"<summary>Solution</summary>(.*?)</details>", source, re.S)[1]
        answer = "\n".join(re.findall(r"```python\n(.*?)```", dropdown, re.S))
        names = {node.name for node in ast.parse(answer).body if isinstance(node, ast.FunctionDef)}
        required = set(_unfinished_functions("".join(notebook["cells"][index + 1]["source"])))
        assert required <= names, f"Answer {count} omits {required - names}"
        assert not _unfinished_functions(
            answer
        ), "A solution must contain implemented function bodies."
    assert count == 9, "All nine progressive exercises need a copyable solution."
    print("All tests in `test_solution_dropdowns_supply_every_exercise_dependency` passed!")


def test_learned_path_has_no_factor_parser_or_phrase_lookup():
    source = inspect.getsource(_solutions())
    forbidden = (
        "phrase_feature_matrix",
        "agreement_phrase_features",
        "train_phrase_ids",
        "eval_phrase_ids",
        "phrase_to_activation",
    )
    assert not any(
        term in source for term in forbidden
    ), "test_learned_path_has_no_factor_parser_or_phrase_lookup: expected not any((term in source for term in forbidden))."
    learned = inspect.getsource(_solutions().evaluate_mini_nla)
    assert (
        "fit_direct_factor_decoder" in learned
    ), "test_learned_path_has_no_factor_parser_or_phrase_lookup: expected 'fit_direct_factor_decoder' in learned."
    assert (
        "direct_reconstruction" in learned
    ), "test_learned_path_has_no_factor_parser_or_phrase_lookup: expected 'direct_reconstruction' in learned."
    assert (
        "generated.texts" in learned
    ), "test_learned_path_has_no_factor_parser_or_phrase_lookup: expected 'generated.texts' in learned."
    assert (
        "encode_reconstructor_text" in learned
    ), "test_learned_path_has_no_factor_parser_or_phrase_lookup: expected 'encode_reconstructor_text' in learned."
    print("All tests in `test_learned_path_has_no_factor_parser_or_phrase_lookup` passed!")


def test_instruction_page_matches_the_repaired_claim():
    text = PAGE.read_text()
    assert (
        "96" in text and "32" in text
    ), "test_instruction_page_matches_the_repaired_claim: expected '96' in text and '32' in text."
    assert (
        "token sequence" in text.lower()
    ), "test_instruction_page_matches_the_repaired_claim: expected 'token sequence' in text.lower()."
    assert (
        "within-factor FVE" in text
    ), "test_instruction_page_matches_the_repaired_claim: expected 'within-factor FVE' in text."
    assert (
        "identity" in text.lower() and "single-row" in text.lower()
    ), "test_instruction_page_matches_the_repaired_claim: expected 'identity' in text.lower() and 'single-row' in text.lower()."
    assert (
        "lexical relabel" in text.lower()
    ), "test_instruction_page_matches_the_repaired_claim: expected 'lexical relabel' in text.lower()."
    assert (
        "eight" in text.lower() and "shuffle" in text.lower()
    ), "test_instruction_page_matches_the_repaired_claim: expected 'eight' in text.lower() and 'shuffle' in text.lower()."
    assert (
        "This is not" in text
    ), "test_instruction_page_matches_the_repaired_claim: expected 'This is not' in text."
    assert (
        "unseen nouns" not in text
    ), "test_instruction_page_matches_the_repaired_claim: expected 'unseen nouns' not in text."
    assert (
        "four-phrase" not in text.lower()
    ), "test_instruction_page_matches_the_repaired_claim: expected 'four-phrase' not in text.lower()."
    assert (
        "supporting provenance" in text
    ), "The page must explain that the verification report supports provenance only."
    print("All tests in `test_instruction_page_matches_the_repaired_claim` passed!")


def test_structural_smoke_contract_is_not_an_acceptance_report():
    smoke = _solutions().run_smoke_test(cpu=True)
    assert (
        smoke["train_count"] == 96
    ), "test_structural_smoke_contract_is_not_an_acceptance_report: expected smoke['train_count'] == 96."
    assert (
        smoke["eval_count"] == 32
    ), "test_structural_smoke_contract_is_not_an_acceptance_report: expected smoke['eval_count'] == 32."
    assert (
        smoke["contexts"] == 32
    ), "test_structural_smoke_contract_is_not_an_acceptance_report: expected smoke['contexts'] == 32."
    assert set(smoke["eval_factor_counts"].values()) == {
        8
    }, "test_structural_smoke_contract_is_not_an_acceptance_report: expected set(smoke['eval_factor_counts'].values()) == {8}."
    assert (
        not {"accepted", "tests_passed", "contract_passed"} & smoke.keys()
    ), "test_structural_smoke_contract_is_not_an_acceptance_report: expected not {'accepted', 'tests_passed', 'contract_passed'} & smoke.keys()."
    print("All tests in `test_structural_smoke_contract_is_not_an_acceptance_report` passed!")


def test_gpu_entrypoint_is_strict_and_has_no_cpu_fallback():
    source = inspect.getsource(_solutions().run_gpu_test)
    assert "t.cuda.is_available" in source, "The release entrypoint must require CUDA explicitly."
    assert 'load_pythia("cuda")' in source, "The scientific run must load Pythia on CUDA."
    assert "run_cpu_test" not in source, "The CUDA entrypoint must not delegate to a CPU fallback."
    assert "bag_of_tokens_fve" in source, "CUDA acceptance must include the lexical bypass metric."
    print("All tests in `test_gpu_entrypoint_is_strict_and_has_no_cpu_fallback` passed!")


def test_executed_solution_notebook_contains_visible_signature():
    notebook = json.loads(SOLUTION_NOTEBOOK.read_text())
    rendered = json.dumps(notebook)
    assert (
        '"image/png"' in rendered
    ), "The executed solution notebook must embed the live signature figure."
    assert (
        '"output_type": "error"' not in rendered
    ), "The executed solution notebook must not contain stored error outputs."
    print("All tests in `test_executed_solution_notebook_contains_visible_signature` passed!")
