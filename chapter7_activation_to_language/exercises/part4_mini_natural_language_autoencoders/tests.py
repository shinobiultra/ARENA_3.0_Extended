"""Focused semantic and learner-surface tests for section 7.4."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from pathlib import Path

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


def _solutions():
    from chapter7_activation_to_language.exercises.part4_mini_natural_language_autoencoders import (
        solutions,
    )

    return solutions


def test_planted_dataset_has_exact_semantic_ground_truth(
    make_planted_nla_dataset: Callable | None = None,
    latent_phrase: Callable | None = None,
):
    solutions = _solutions()
    make_planted_nla_dataset = make_planted_nla_dataset or solutions.make_planted_nla_dataset
    latent_phrase = latent_phrase or solutions.latent_phrase
    dataset = make_planted_nla_dataset()

    assert dataset.activations.shape == (40, 8), (
        'The NLA organism must keep balanced semantic states in orthogonal directions so prompt identity cannot reveal the answer.'
    )
    assert dataset.latent_bits.shape == (40, 2), (
        'The NLA organism must keep balanced semantic states in orthogonal directions so prompt identity cannot reveal the answer.'
    )
    gram = dataset.semantic_directions @ dataset.semantic_directions.T
    assert t.allclose(gram, t.eye(2), atol=1e-6), "The planted semantic axes must be orthonormal."
    cross = dataset.semantic_directions @ dataset.nuisance_directions.T
    assert t.allclose(cross, t.zeros(2, 2), atol=1e-6), "Nuisance must be orthogonal to semantics."

    semantic_coordinates = dataset.activations @ dataset.semantic_directions.T
    expected_coordinates = dataset.latent_bits * t.tensor([2.5, 2.0])
    assert t.allclose(semantic_coordinates, expected_coordinates, atol=1e-5), (
        'The NLA organism must keep balanced semantic states in orthogonal directions so prompt identity cannot reveal the answer.'
    )
    assert tuple(latent_phrase(bits) for bits in dataset.latent_bits) == dataset.phrases, (
        'The NLA organism must keep balanced semantic states in orthogonal directions so prompt identity cannot reveal the answer.'
    )

    for prompt in sorted(set(dataset.prompts)):
        rows = [i for i, value in enumerate(dataset.prompts) if value == prompt]
        assert len(rows) == 4, "Every visible prompt must be paired with all four hidden states."
        assert {tuple(row.tolist()) for row in dataset.latent_bits[rows]} == {
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        }, (
            'The NLA organism must keep balanced semantic states in orthogonal directions so prompt identity cannot reveal the answer.'
        )
    print("All tests in `test_planted_dataset_has_exact_semantic_ground_truth` passed!")


def test_phrase_features_are_compositional(
    phrase_feature_matrix: Callable | None = None,
):
    phrase_feature_matrix = phrase_feature_matrix or _solutions().phrase_feature_matrix
    phrases = (
        "route north; cargo fragile",
        "route north; cargo standard",
        "route south; cargo fragile",
        "route south; cargo standard",
    )
    expected = t.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    assert t.equal(phrase_feature_matrix(phrases), expected), (
        'Phrase features must represent route and cargo words compositionally and reject malformed explanations.'
    )
    assert t.equal(phrase_feature_matrix(("cargo fragile; route north",)), expected[:1]), (
        'Phrase features must represent route and cargo words compositionally and reject malformed explanations.'
    )
    try:
        phrase_feature_matrix(("route north; cargo mysterious",))
    except ValueError as exc:
        assert "one route word and one cargo word" in str(exc), (
            'Phrase features must represent route and cargo words compositionally and reject malformed explanations.'
        )
    else:
        raise AssertionError("Unknown semantic words must not silently become a valid code.")
    print("All tests in `test_phrase_features_are_compositional` passed!")


def test_activation_encoder_recovers_held_out_phrases(
    make_planted_nla_dataset: Callable | None = None,
    fit_activation_encoder: Callable | None = None,
    encode_activations_to_phrases: Callable | None = None,
):
    solutions = _solutions()
    make_planted_nla_dataset = make_planted_nla_dataset or solutions.make_planted_nla_dataset
    fit_activation_encoder = fit_activation_encoder or solutions.fit_activation_encoder
    encode_activations_to_phrases = encode_activations_to_phrases or solutions.encode_activations_to_phrases
    dataset = make_planted_nla_dataset()
    train = dataset.split_ids == 0
    evaluation = dataset.split_ids == 1
    weight, bias = fit_activation_encoder(dataset.activations[train], dataset.latent_bits[train])
    phrases, bits = encode_activations_to_phrases(dataset.activations[evaluation], weight, bias)
    assert t.equal(bits, dataset.latent_bits[evaluation]), (
        'The activation encoder must recover both planted semantic bits and their exact held-out phrases.'
    )
    expected = tuple(phrase for phrase, keep in zip(dataset.phrases, evaluation.tolist()) if keep)
    assert phrases == expected, (
        'The activation encoder must recover both planted semantic bits and their exact held-out phrases.'
    )
    print("All tests in `test_activation_encoder_recovers_held_out_phrases` passed!")


def test_phrase_decoder_recovers_semantic_coordinates(
    make_planted_nla_dataset: Callable | None = None,
    fit_phrase_decoder: Callable | None = None,
    decode_phrases: Callable | None = None,
):
    solutions = _solutions()
    make_planted_nla_dataset = make_planted_nla_dataset or solutions.make_planted_nla_dataset
    fit_phrase_decoder = fit_phrase_decoder or solutions.fit_phrase_decoder
    decode_phrases = decode_phrases or solutions.decode_phrases
    dataset = make_planted_nla_dataset()
    train = dataset.split_ids == 0
    train_phrases = tuple(phrase for phrase, keep in zip(dataset.phrases, train.tolist()) if keep)
    weight, bias = fit_phrase_decoder(train_phrases, dataset.activations[train])
    canonical_phrases = (
        "route north; cargo fragile",
        "route north; cargo standard",
        "route south; cargo fragile",
        "route south; cargo standard",
    )
    decoded = decode_phrases(canonical_phrases, weight, bias)
    coordinates = decoded @ dataset.semantic_directions.T
    expected = t.tensor([[2.5, 2.0], [2.5, -2.0], [-2.5, 2.0], [-2.5, -2.0]])
    assert t.allclose(coordinates, expected, atol=1e-4), (
        'The phrase decoder must reconstruct semantic coordinates while discarding orthogonal nuisance information.'
    )
    assert (decoded @ dataset.nuisance_directions.T).abs().max() < 1e-5, (
        'The phrase decoder must reconstruct semantic coordinates while discarding orthogonal nuisance information.'
    )
    print("All tests in `test_phrase_decoder_recovers_semantic_coordinates` passed!")


def test_reconstruction_beats_prompt_only_and_opposite_phrases(
    build_signature_payload: Callable | None = None,
):
    build_signature_payload = build_signature_payload or _solutions().build_signature_payload
    report = build_signature_payload()["reconstruction"]
    assert report.nla_mse < 0.02, (
        'NLA reconstruction must beat aligned prompt-only and opposite-meaning controls on the same held-out rows.'
    )
    assert report.prompt_only_mse > 1.0, (
        'NLA reconstruction must beat aligned prompt-only and opposite-meaning controls on the same held-out rows.'
    )
    assert report.shuffled_phrase_mse > 5.0, (
        'NLA reconstruction must beat aligned prompt-only and opposite-meaning controls on the same held-out rows.'
    )
    assert report.mean_cosine > 0.99, (
        'NLA reconstruction must beat aligned prompt-only and opposite-meaning controls on the same held-out rows.'
    )
    assert report.nla_beats_prompt_only and report.shuffled_control_fails, (
        'NLA reconstruction must beat aligned prompt-only and opposite-meaning controls on the same held-out rows.'
    )
    print("All tests in `test_reconstruction_beats_prompt_only_and_opposite_phrases` passed!")


def test_reconstruction_metrics_reject_misaligned_rows(
    reconstruction_comparison: Callable | None = None,
):
    reconstruction_comparison = reconstruction_comparison or _solutions().reconstruction_comparison
    original = t.zeros(4, 3)
    try:
        reconstruction_comparison(original, original[:3], original, original)
    except ValueError as exc:
        assert "same shape" in str(exc), (
            'Reconstruction metrics must reject misaligned tensors before comparing methods.'
        )
    else:
        raise AssertionError("A row-misaligned comparison must fail before scoring.")
    print("All tests in `test_reconstruction_metrics_reject_misaligned_rows` passed!")


def test_reconstruction_preserves_planted_behavior(
    build_signature_payload: Callable | None = None,
):
    build_signature_payload = build_signature_payload or _solutions().build_signature_payload
    report = build_signature_payload()["behavior"]
    assert report.nla_mae < 1e-4, (
        'The decoded phrase must preserve both semantic labels and the planted downstream behavior better than controls.'
    )
    assert report.prompt_only_mae > 3.0, (
        'The decoded phrase must preserve both semantic labels and the planted downstream behavior better than controls.'
    )
    assert report.shuffled_phrase_mae > 6.0, (
        'The decoded phrase must preserve both semantic labels and the planted downstream behavior better than controls.'
    )
    assert report.route_accuracy == 1.0, (
        'The decoded phrase must preserve both semantic labels and the planted downstream behavior better than controls.'
    )
    assert report.cargo_accuracy == 1.0, (
        'The decoded phrase must preserve both semantic labels and the planted downstream behavior better than controls.'
    )
    assert report.behavior_sign_accuracy == 1.0, (
        'The decoded phrase must preserve both semantic labels and the planted downstream behavior better than controls.'
    )
    assert report.nla_beats_controls, (
        'The decoded phrase must preserve both semantic labels and the planted downstream behavior better than controls.'
    )
    print("All tests in `test_reconstruction_preserves_planted_behavior` passed!")


def test_controls_are_semantic_and_explanations_are_short(
    antipodal_phrase_control: Callable | None = None,
    word_compression_ratio: Callable | None = None,
    make_planted_nla_dataset: Callable | None = None,
):
    solutions = _solutions()
    antipodal_phrase_control = antipodal_phrase_control or solutions.antipodal_phrase_control
    word_compression_ratio = word_compression_ratio or solutions.word_compression_ratio
    make_planted_nla_dataset = make_planted_nla_dataset or solutions.make_planted_nla_dataset
    dataset = make_planted_nla_dataset()
    evaluation = dataset.split_ids == 1
    phrases = tuple(phrase for phrase, keep in zip(dataset.phrases, evaluation.tolist()) if keep)
    prompts = tuple(prompt for prompt, keep in zip(dataset.prompts, evaluation.tolist()) if keep)
    opposite = antipodal_phrase_control(phrases)
    assert all(a != b for a, b in zip(phrases, opposite)), (
        'The wrong-phrase and compression controls must test semantic content without leaking numerical coordinates.'
    )
    assert antipodal_phrase_control(opposite) == phrases, (
        'The wrong-phrase and compression controls must test semantic content without leaking numerical coordinates.'
    )
    assert word_compression_ratio(phrases, prompts) == 0.4, (
        'The wrong-phrase and compression controls must test semantic content without leaking numerical coordinates.'
    )
    assert word_compression_ratio(prompts, prompts) == 1.0, (
        'The wrong-phrase and compression controls must test semantic content without leaking numerical coordinates.'
    )
    assert not any(re.search(r"[+-]?\d+(?:\.\d+)?", phrase) for phrase in phrases), (
        'The wrong-phrase and compression controls must test semantic content without leaking numerical coordinates.'
    )
    print("All tests in `test_controls_are_semantic_and_explanations_are_short` passed!")


def test_counterfactual_route_flip_changes_only_route_semantics(
    counterfactual_route_flip: Callable | None = None,
    make_planted_nla_dataset: Callable | None = None,
    fit_activation_encoder: Callable | None = None,
    encode_activations_to_phrases: Callable | None = None,
):
    solutions = _solutions()
    counterfactual_route_flip = counterfactual_route_flip or solutions.counterfactual_route_flip
    make_planted_nla_dataset = make_planted_nla_dataset or solutions.make_planted_nla_dataset
    fit_activation_encoder = fit_activation_encoder or solutions.fit_activation_encoder
    encode_activations_to_phrases = encode_activations_to_phrases or solutions.encode_activations_to_phrases
    dataset = make_planted_nla_dataset()
    train = dataset.split_ids == 0
    evaluation = dataset.split_ids == 1
    weight, bias = fit_activation_encoder(dataset.activations[train], dataset.latent_bits[train])
    original = dataset.activations[evaluation][0:1]
    changed = counterfactual_route_flip(original, dataset.semantic_directions[0])
    original_phrase, original_bits = encode_activations_to_phrases(original, weight, bias)
    changed_phrase, changed_bits = encode_activations_to_phrases(changed, weight, bias)
    assert original_phrase == ("route north; cargo fragile",), (
        'The route intervention must flip route text and behavior while preserving cargo and nuisance coordinates.'
    )
    assert changed_phrase == ("route south; cargo fragile",), (
        'The route intervention must flip route text and behavior while preserving cargo and nuisance coordinates.'
    )
    assert changed_bits[0, 0] == -original_bits[0, 0], (
        'The route intervention must flip route text and behavior while preserving cargo and nuisance coordinates.'
    )
    assert changed_bits[0, 1] == original_bits[0, 1], (
        'The route intervention must flip route text and behavior while preserving cargo and nuisance coordinates.'
    )
    delta = changed - original
    assert (delta @ dataset.nuisance_directions.T).abs().max() < 1e-5, (
        'The route intervention must flip route text and behavior while preserving cargo and nuisance coordinates.'
    )
    behavior_change = (delta @ dataset.behavior_direction).item()
    assert behavior_change < -6.9, (
        'The route intervention must flip route text and behavior while preserving cargo and nuisance coordinates.'
    )
    print("All tests in `test_counterfactual_route_flip_changes_only_route_semantics` passed!")


def test_signature_payload_is_actual_computation(
    build_signature_payload: Callable | None = None,
):
    build_signature_payload = build_signature_payload or _solutions().build_signature_payload
    low_noise = build_signature_payload(nuisance_scale=0.1)
    high_noise = build_signature_payload(nuisance_scale=0.7)
    assert low_noise["phrase_accuracy"] == high_noise["phrase_accuracy"] == 1.0, (
        'The signature payload must respond to nuisance changes while preserving the exact semantic result.'
    )
    assert low_noise["reconstruction"].nla_mse < high_noise["reconstruction"].nla_mse, (
        'The signature payload must respond to nuisance changes while preserving the exact semantic result.'
    )
    assert high_noise["reconstruction"].nla_mse < high_noise["reconstruction"].prompt_only_mse, (
        'The signature payload must respond to nuisance changes while preserving the exact semantic result.'
    )
    print("All tests in `test_signature_payload_is_actual_computation` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    contract = run_smoke_test(cpu=True)
    assert contract["accepted"] and contract["tests_passed"] and contract["contract_passed"], (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    assert contract["toy_phrase_accuracy"] == 1.0, (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    assert contract["toy_nla_mse"] < contract["toy_prompt_only_mse"], (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    assert contract["toy_prompt_only_mse"] < contract["toy_shuffled_phrase_mse"], (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    assert contract["toy_behavior_mae"] < 1e-4, (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    assert contract["toy_compression_ratio"] < 0.5, (
        'The composed CPU contract must retain every exact result and control required by the visible notebook conclusion.'
    )
    print("All tests in `test_notebook_contract` passed!")


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


def test_solution_notebook_exposes_taught_implementations():
    markdown, code = _notebook_text(SOLUTION_NOTEBOOK)
    required = {
        "latent_phrase",
        "make_planted_nla_dataset",
        "phrase_feature_matrix",
        "fit_activation_encoder",
        "encode_activations_to_phrases",
        "fit_phrase_decoder",
        "decode_phrases",
        "prompt_only_reconstruction",
        "reconstruction_comparison",
        "behavior_preservation",
        "antipodal_phrase_control",
        "word_compression_ratio",
        "counterfactual_route_flip",
        "build_signature_payload",
        "cache_pythia_text_bottleneck_dataset",
        "fit_real_residual_phrase_encoder",
        "predict_residual_phrases",
        "pythia_phrase_text_features",
        "fit_phrase_text_decoder",
        "decode_phrase_text_features",
        "evaluate_pythia_text_bottleneck",
    }
    tree = ast.parse(code)
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert required <= defined, f"Solved notebook is missing inline implementations: {required - defined}"
    assert "raise NotImplementedError" not in code, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert "solutions." not in code and "import solutions" not in code, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    assert markdown.count("### Exercise -") == 12, (
        'The solution notebook must keep taught implementations, learner aids, and the visible signature result inline.'
    )
    print("All tests in `test_solution_notebook_exposes_taught_implementations` passed!")


def test_learner_surfaces_have_complete_progression():
    required_markers = (
        "By the end of this notebook",
        "Learning Objectives",
        "Cold Open",
        "## Signature Result",
        "## Try It Yourself",
        "## Bonus Anomaly Hunt",
        "## Limitations",
    )
    for path in (EXERCISE_NOTEBOOK, SOLUTION_NOTEBOOK):
        markdown, code = _notebook_text(path)
        assert all(marker in markdown for marker in required_markers), f"Missing progression marker in {path.name}"
        assert markdown.count("### Exercise -") == 12, (
            'Each NLA learner surface must contain all twelve exercises and their expected, help, interpretation, and solution aids.'
        )
        assert markdown.count("<summary>Expected output</summary>") == 12, (
            'Each NLA learner surface must contain all twelve exercises and their expected, help, interpretation, and solution aids.'
        )
        assert markdown.count("<summary>Help") == 12, (
            'Each NLA learner surface must contain all twelve exercises and their expected, help, interpretation, and solution aids.'
        )
        assert markdown.count("<summary>Interpretation</summary>") == 12, (
            'Each NLA learner surface must contain all twelve exercises and their expected, help, interpretation, and solution aids.'
        )
        assert markdown.count("<summary>Solution</summary>") == 12, (
            'Each NLA learner surface must contain all twelve exercises and their expected, help, interpretation, and solution aids.'
        )
        for cell in json.loads(path.read_text())["cells"]:
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])))
        assert code.count("verification_report.json") == 1, (
            'The report may appear once as optional release evidence, never as the learner result.'
        )

    page = PAGE.read_text()
    assert all(marker in page for marker in required_markers), (
        'Each NLA learner surface must contain all eight exercises and their expected, help, interpretation, and solution aids.'
    )
    assert page.count("### Exercise -") == 12, (
        'Each NLA learner surface must contain all twelve exercises and their expected, help, interpretation, and solution aids.'
    )
    assert "mini_nla_signature_result.png" in page, (
        'Each NLA learner surface must contain all eight exercises and their expected, help, interpretation, and solution aids.'
    )
    assert "pythia_real_text_bottleneck.png" in page, (
        "The page must display the real Pythia text-bottleneck signature figure."
    )
    print("All tests in `test_learner_surfaces_have_complete_progression` passed!")


def test_real_phrase_features_use_pythia_token_embeddings():
    solutions = _solutions()
    tokenizer, model = solutions.load_pythia_text_bottleneck_model()
    features = solutions.pythia_phrase_text_features(
        tokenizer,
        model,
        solutions.PYTHIA_NLA_PHRASES,
    )
    assert features.shape == (6, model.config.hidden_size), (
        "Every phrase must become one full-width Pythia embedding feature."
    )
    assert t.linalg.matrix_rank(features).item() == 6, (
        "The six phrase embeddings must remain distinguishable to the decoder."
    )
    token_ids = tokenizer.encode(
        solutions.PYTHIA_NLA_PHRASES[0],
        add_special_tokens=False,
        return_tensors="pt",
    )
    with t.inference_mode():
        expected = model.get_input_embeddings()(token_ids).mean(dim=1).squeeze(0).float()
    t.testing.assert_close(features[0], expected)
    print("All tests in `test_real_phrase_features_use_pythia_token_embeddings` passed!")


def test_real_pythia_residual_text_bottleneck_beats_controls():
    solutions = _solutions()
    tokenizer, model = solutions.load_pythia_text_bottleneck_model()
    dataset = solutions.cache_pythia_text_bottleneck_dataset(tokenizer, model)
    report = solutions.evaluate_pythia_text_bottleneck(tokenizer, model, dataset)

    assert dataset.train_residuals.shape == (36, model.config.hidden_size), (
        "The real encoder must train on all 36 precommitted Pythia residuals."
    )
    assert dataset.eval_residuals.shape == (18, model.config.hidden_size), (
        "The real result must retain the complete 18-prompt blind split."
    )
    assert report.phrase_accuracy >= 0.75, (
        "Real residuals must support useful held-out phrase recovery."
    )
    assert report.shuffled_label_accuracy <= 0.25, (
        "Shuffled labels must not be recoverable by the same encoder capacity."
    )
    assert report.reconstruction_mse < report.no_text_mse < report.shuffled_phrase_mse, (
        "Reconstruction error must order learned text, no text, then wrong text."
    )
    assert report.oracle_phrase_mse < report.no_text_mse, (
        "Correct natural-language phrases must beat the global-mean residual."
    )
    assert report.reconstructed_target_accuracy >= 0.75, (
        "The end-to-end text bottleneck must preserve most next-token behavior."
    )
    assert report.oracle_phrase_target_accuracy == 1.0, (
        "The text decoder must preserve every target when supplied the correct phrase."
    )
    assert report.no_text_target_accuracy <= 0.20, (
        "The no-text mean must fail most class-specific next-token targets."
    )
    assert report.shuffled_phrase_target_accuracy == 0.0, (
        "Cyclic wrong phrases must recover none of the intended token classes."
    )
    assert len(report.rows) == 18, (
        "The notebook must expose one auditable row per held-out prompt."
    )
    release_metrics = solutions.summarize_pythia_text_bottleneck_lab(
        {"dataset": dataset, "report": report}
    )
    assert release_metrics == {
        "learner_cpu_train_count": 36,
        "learner_cpu_eval_count": 18,
        "learner_cpu_phrase_accuracy": report.phrase_accuracy,
        "learner_cpu_shuffled_label_accuracy": report.shuffled_label_accuracy,
        "learner_cpu_reconstructed_target_accuracy": report.reconstructed_target_accuracy,
        "learner_cpu_oracle_phrase_target_accuracy": report.oracle_phrase_target_accuracy,
        "learner_cpu_no_text_target_accuracy": report.no_text_target_accuracy,
        "learner_cpu_wrong_phrase_target_accuracy": report.shuffled_phrase_target_accuracy,
        "learner_cpu_reconstruction_mse": report.reconstruction_mse,
        "learner_cpu_no_text_mse": report.no_text_mse,
        "learner_cpu_wrong_phrase_mse": report.shuffled_phrase_mse,
    }, "Release metrics must be derived from the exact learner-visible experiment."
    print("All tests in `test_real_pythia_residual_text_bottleneck_beats_controls` passed!")
