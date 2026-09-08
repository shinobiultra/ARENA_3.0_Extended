"""Semantic tests for [16.6] SHAP vs Activation Patching."""

from collections.abc import Callable

import pytest
import torch as t

TOKEN_LABELS = ("RED", "SQUARE", "BRIGHT", "FILLER")
HIDDEN_LABELS = (
    "red_direct",
    "square_direct",
    "bright_direct",
    "filler_direct",
    "red_x_square_gate",
)


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part6_shap_vs_activation_patching import (
        solutions,
    )

    return solutions


def test_exact_model_oracle(
    encode_tokens: Callable | None = None,
    score_from_hidden: Callable | None = None,
    exact_model: Callable | None = None,
):
    solutions = _solutions()
    encode_tokens = encode_tokens or solutions.encode_tokens
    score_from_hidden = score_from_hidden or solutions.score_from_hidden
    exact_model = exact_model or solutions.exact_model
    clean = t.ones(4, dtype=t.float64)
    corrupt = t.zeros_like(clean)
    hidden = encode_tokens(clean)
    assert hidden.shape == (5,), "The encoder must append one interaction unit to four token units."
    assert t.equal(hidden, t.ones(5, dtype=t.float64)), (
        "Clean RED+SQUARE must turn on the post-ReLU interaction unit exactly."
    )
    assert score_from_hidden(hidden).item() == pytest.approx(4.25, abs=1e-12), (
        "The hidden readout must sum the direct terms and the 2.4 gate contribution."
    )
    clean_score, cache = exact_model(clean, return_cache=True)
    assert clean_score.item() == pytest.approx(4.25, abs=1e-12), (
        "The complete clean token input must produce the declared 4.25 score."
    )
    assert exact_model(corrupt).item() == 0.0, "The all-zero corrupt baseline must score zero."
    assert set(cache) == {"token_activations", "post_relu_hidden"}, (
        "The model cache must expose the named token and post-ReLU activation levels."
    )
    assert cache["post_relu_hidden"][-1].item() == 1.0, (
        "RED and SQUARE together must activate the ReLU interaction unit."
    )
    without_square = t.tensor([1.0, 0.0, 1.0, 1.0], dtype=t.float64)
    assert exact_model(without_square).item() == pytest.approx(1.25, abs=1e-12), (
        "Removing SQUARE must switch off the interaction gate."
    )
    print("All tests in `test_exact_model_oracle` passed!")


def test_coalition_value_table_oracle(
    all_coalitions: Callable | None = None,
    coalition_value_table: Callable | None = None,
    exact_model: Callable | None = None,
):
    solutions = _solutions()
    all_coalitions = all_coalitions or solutions.all_coalitions
    coalition_value_table = coalition_value_table or solutions.coalition_value_table
    exact_model = exact_model or solutions.exact_model
    coalitions = all_coalitions(4)
    assert len(coalitions) == 16 and len(set(coalitions)) == 16, (
        "Four players require all 16 distinct coalitions."
    )
    assert coalitions[0] == frozenset(), "Coalition enumeration must include the empty baseline first."
    clean = t.ones(4, dtype=t.float64)
    corrupt = t.zeros_like(clean)
    values = coalition_value_table(clean, corrupt, exact_model)
    assert set(values) == set(coalitions), "The value table must evaluate every coalition exactly once."
    assert values[frozenset()] == 0.0, "The empty clean coalition must reproduce the corrupt score."
    assert values[frozenset({0, 1})] == pytest.approx(4.0, abs=1e-12), (
        "RED and SQUARE must include both direct terms and the interaction gate."
    )
    assert values[frozenset({0, 2})] == pytest.approx(1.25, abs=1e-12), (
        "RED and BRIGHT must not activate the RED x SQUARE gate."
    )
    assert values[frozenset(range(4))] == pytest.approx(4.25, abs=1e-12), (
        "The full coalition must reproduce the complete clean score."
    )
    print("All tests in `test_coalition_value_table_oracle` passed!")


def test_exact_shapley_token_ground_truth(
    coalition_value_table: Callable | None = None,
    exact_shapley_values: Callable | None = None,
    exact_model: Callable | None = None,
):
    solutions = _solutions()
    coalition_value_table = coalition_value_table or solutions.coalition_value_table
    exact_shapley_values = exact_shapley_values or solutions.exact_shapley_values
    exact_model = exact_model or solutions.exact_model
    clean = t.ones(4, dtype=t.float64)
    corrupt = t.zeros_like(clean)
    values = coalition_value_table(clean, corrupt, exact_model)
    shapley = exact_shapley_values(values, num_players=4)
    oracle = t.tensor([2.2, 1.8, 0.25, 0.0], dtype=t.float64)
    assert shapley.dtype == t.float64, "Exact Shapley should retain float64 oracle precision."
    assert t.allclose(shapley, oracle, atol=1e-12, rtol=0), (
        "Token Shapley must split the symmetric 2.4 interaction equally between RED and SQUARE."
    )
    assert shapley.sum().item() == pytest.approx(4.25, abs=1e-12), (
        "Shapley must allocate the complete clean-minus-corrupt score."
    )
    incomplete = dict(values)
    incomplete.pop(frozenset({0, 1}))
    with pytest.raises(ValueError, match="complete"):
        exact_shapley_values(incomplete, num_players=4)
    print("All tests in `test_exact_shapley_token_ground_truth` passed!")


def test_token_activation_patching_ground_truth(
    activation_patching_effects: Callable | None = None,
    exact_model: Callable | None = None,
):
    solutions = _solutions()
    activation_patching_effects = (
        activation_patching_effects or solutions.activation_patching_effects
    )
    exact_model = exact_model or solutions.exact_model
    clean = t.ones(4, dtype=t.float64)
    corrupt = t.zeros_like(clean)
    effects = activation_patching_effects(clean, corrupt, exact_model)
    oracle = t.tensor([3.4, 3.0, 0.25, 0.0], dtype=t.float64)
    assert t.allclose(effects, oracle, atol=1e-12, rtol=0), (
        "Single-site noising must remove each direct term and any full-context gate contribution."
    )
    assert effects.sum().item() == pytest.approx(6.65, abs=1e-12), (
        "Token patching must visibly overcount the shared gate contribution."
    )
    assert effects[-1].item() == 0.0, "FILLER is the exact wrong-location control."
    assert t.equal(clean, t.ones_like(clean)), "Patching must not mutate the clean cache."
    print("All tests in `test_token_activation_patching_ground_truth` passed!")


def test_alignment_reorders_and_rejects_mismatched_units(
    align_named_attributions: Callable | None = None,
):
    align_named_attributions = align_named_attributions or _solutions().align_named_attributions
    reference = t.tensor([2.2, 1.8, 0.25, 0.0])
    candidate_labels = ("FILLER", "BRIGHT", "SQUARE", "RED")
    candidate = t.tensor([0.0, 0.25, 1.8, 2.2])
    labels, aligned_reference, aligned_candidate = align_named_attributions(
        TOKEN_LABELS, reference, candidate_labels, candidate
    )
    assert labels == TOKEN_LABELS, "Alignment must preserve reference-label order."
    assert t.allclose(aligned_reference, aligned_candidate), (
        "Name-based reordering must recover equal values from a permuted candidate vector."
    )
    with pytest.raises(ValueError, match="different player sets"):
        align_named_attributions(TOKEN_LABELS, reference, HIDDEN_LABELS, t.ones(5))
    print("All tests in `test_alignment_reorders_and_rejects_mismatched_units` passed!")


def test_aligned_token_disagreement_and_hidden_agreement(
    coalition_value_table: Callable | None = None,
    exact_shapley_values: Callable | None = None,
    activation_patching_effects: Callable | None = None,
    attribution_comparison: Callable | None = None,
    encode_tokens: Callable | None = None,
    score_from_hidden: Callable | None = None,
    exact_model: Callable | None = None,
):
    solutions = _solutions()
    coalition_value_table = coalition_value_table or solutions.coalition_value_table
    exact_shapley_values = exact_shapley_values or solutions.exact_shapley_values
    activation_patching_effects = (
        activation_patching_effects or solutions.activation_patching_effects
    )
    attribution_comparison = attribution_comparison or solutions.attribution_comparison
    encode_tokens = encode_tokens or solutions.encode_tokens
    score_from_hidden = score_from_hidden or solutions.score_from_hidden
    exact_model = exact_model or solutions.exact_model

    clean_tokens = t.ones(4, dtype=t.float64)
    corrupt_tokens = t.zeros_like(clean_tokens)
    token_values = coalition_value_table(clean_tokens, corrupt_tokens, exact_model)
    token_shapley = exact_shapley_values(token_values, num_players=4)
    token_patching = activation_patching_effects(clean_tokens, corrupt_tokens, exact_model)
    token_result = attribution_comparison(
        TOKEN_LABELS,
        token_shapley,
        TOKEN_LABELS,
        token_patching,
        output_delta=4.25,
    )
    assert token_result["max_abs_error"] == pytest.approx(1.2, abs=1e-12), (
        "Each interaction parent must differ by half the 2.4 gate contribution."
    )
    assert token_result["candidate_efficiency_gap"] == pytest.approx(2.4, abs=1e-12), (
        "Token patching must count the shared interaction one extra time."
    )
    assert token_result["top_agrees"], (
        "Top-1 agreement is deliberately insufficient to establish method agreement."
    )

    clean_hidden = encode_tokens(clean_tokens)
    corrupt_hidden = encode_tokens(corrupt_tokens)
    hidden_values = coalition_value_table(clean_hidden, corrupt_hidden, score_from_hidden)
    hidden_shapley = exact_shapley_values(hidden_values, num_players=5)
    hidden_patching = activation_patching_effects(
        clean_hidden, corrupt_hidden, score_from_hidden
    )
    hidden_result = attribution_comparison(
        HIDDEN_LABELS,
        hidden_shapley,
        HIDDEN_LABELS,
        hidden_patching,
        output_delta=4.25,
    )
    oracle = t.tensor([1.0, 0.6, 0.25, 0.0, 2.4], dtype=t.float64)
    assert t.allclose(hidden_shapley, oracle, atol=1e-12, rtol=0), (
        "Hidden Shapley must recover the additive direct-unit and explicit-gate readout weights."
    )
    assert t.allclose(hidden_patching, oracle, atol=1e-12, rtol=0), (
        "Hidden-unit patching must match the additive readout oracle."
    )
    assert hidden_result["max_abs_error"] < 1e-12, (
        "Aligned hidden-unit Shapley and patching must agree to numerical precision."
    )
    assert abs(hidden_result["candidate_efficiency_gap"]) < 1e-12, (
        "Hidden patching must conserve the complete output delta."
    )
    print("All tests in `test_aligned_token_disagreement_and_hidden_agreement` passed!")


def test_shuffled_and_random_controls(
    shuffled_label_control: Callable | None = None,
    random_direction_patching_effects: Callable | None = None,
):
    solutions = _solutions()
    shuffled_label_control = shuffled_label_control or solutions.shuffled_label_control
    random_direction_patching_effects = (
        random_direction_patching_effects or solutions.random_direction_patching_effects
    )
    shapley = t.tensor([2.2, 1.8, 0.25, 0.0], dtype=t.float64)
    patching = t.tensor([3.4, 3.0, 0.25, 0.0], dtype=t.float64)
    shuffled = shuffled_label_control(patching, (2, 0, 3, 1))
    shuffled_cosine = t.nn.functional.cosine_similarity(shapley, shuffled, dim=0)
    assert shuffled.tolist() == [0.25, 3.4, 0.0, 3.0], (
        "The fixed permutation must shuffle values while leaving token labels in place."
    )
    assert shuffled_cosine.item() == pytest.approx(0.5147262229810324, abs=1e-12), (
        "The shuffled-value control must visibly reduce alignment with token Shapley."
    )

    clean_hidden = t.ones(5, dtype=t.float64)
    weights = t.tensor([1.0, 0.6, 0.25, 0.0, 2.4], dtype=t.float64)
    effects = random_direction_patching_effects(
        clean_hidden, weights, target_delta_norm=1.0, num_samples=512, seed=1666
    ).abs()
    assert effects.shape == (512,), "The random-direction control must return one effect per draw."
    assert effects.mean().item() == pytest.approx(0.9943479120912706, abs=1e-12), (
        "The seeded matched-norm random control mean has an exact regression oracle."
    )
    assert t.quantile(effects, 0.95).item() == pytest.approx(
        2.165827819479382, abs=1e-12
    ), "The seeded random-direction p95 must remain below the true 2.4 gate effect."
    assert float((effects < 2.4).double().mean().item()) == pytest.approx(
        0.986328125, abs=1e-12
    ), "The matched gate must exceed 98.6 percent of seeded random directions."
    print("All tests in `test_shuffled_and_random_controls` passed!")


def test_interaction_sweep_exact_oracle(
    interaction_sweep: Callable | None = None,
):
    interaction_sweep = interaction_sweep or _solutions().interaction_sweep
    strengths = t.tensor([0.0, 0.4, 1.2, 2.4, 3.2], dtype=t.float64)
    result = interaction_sweep(strengths)
    assert t.allclose(result["interaction_strength"], strengths), (
        "The sweep must preserve the requested interaction-strength grid."
    )
    assert t.allclose(result["token_credit_overcount"], strengths, atol=1e-12, rtol=0), (
        "Token patching overcount must equal the planted interaction strength."
    )
    assert t.allclose(
        result["token_max_abs_error"], strengths / 2, atol=1e-12, rtol=0
    ), "Each symmetric interaction parent must differ by half the interaction strength."
    assert float(result["hidden_max_abs_error"].max().item()) < 1e-12, (
        "Hidden-level agreement must remain exact across the full interaction sweep."
    )
    print("All tests in `test_interaction_sweep_exact_oracle` passed!")


def test_notebook_claim_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["random_control_count"] == 512, (
        "The learner contract should retain its exact 512-direction regression oracle."
    )
    assert result["claim_passed"], "The exact organism and all three controls must support the claim."
    assert result["clean_score"] == pytest.approx(4.25, abs=1e-12), (
        "The smoke result must retain the exact clean-score oracle."
    )
    assert result["token_shapley"] == pytest.approx([2.2, 1.8, 0.25, 0.0], abs=1e-12), (
        "The smoke result must expose exact token Shapley values."
    )
    assert result["token_patching"] == pytest.approx([3.4, 3.0, 0.25, 0.0], abs=1e-12), (
        "The smoke result must expose exact token patching effects."
    )
    assert result["token_credit_overcount"] == pytest.approx(2.4, abs=1e-12), (
        "The smoke result must expose the full interaction overcount."
    )
    assert result["hidden_max_abs_error"] < 1e-12, (
        "The hidden additive control must agree to numerical precision."
    )
    assert result["wrong_location_effect"] == 0.0, (
        "The FILLER wrong-location patch must remain exactly inert."
    )
    assert result["shuffled_label_cosine"] < 0.7, (
        "The shuffled-value control must fail alignment with token Shapley."
    )
    assert result["target_gate_effect"] > result["random_direction_p95_abs_effect"], (
        "The true gate patch must beat the p95 matched-random effect."
    )
    print("All tests in `test_notebook_claim_contract` passed!")
