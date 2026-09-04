"""Visible, section-local tests for [6.2] Gemma Scope Deep Dive."""

from __future__ import annotations

from collections.abc import Callable

import torch as t

from chapter6_sparse_feature_methods.exercises.part2_gemma_scope_deep_dive import utils


def _solutions():
    from chapter6_sparse_feature_methods.exercises.part2_gemma_scope_deep_dive import solutions

    return solutions


def _raises(error_type: type[BaseException], fn: Callable[[], object]) -> None:
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


def test_load_normalized_sae_preserves_function(
    SparseAutoencoder: type | None = None,
    load_normalized_sae: Callable | None = None,
) -> None:
    solutions = _solutions()
    SparseAutoencoder = SparseAutoencoder or solutions.SparseAutoencoder
    load_normalized_sae = load_normalized_sae or solutions.load_normalized_sae
    organism = utils.make_ground_truth_organism()
    raw = organism["raw_artifact"]
    raw_before = {name: value.clone() for name, value in raw.items()}

    sae, raw_norms = load_normalized_sae(raw)
    assert isinstance(sae, SparseAutoencoder), "The loader should return the notebook's SAE type."
    t.testing.assert_close(raw_norms, organism["raw_decoder_norms"])
    t.testing.assert_close(sae.w_dec.norm(dim=-1), t.ones(6), atol=1e-6, rtol=1e-6)
    t.testing.assert_close(sae.w_dec, organism["canonical_decoder"], atol=1e-6, rtol=1e-6)
    for name in raw:
        t.testing.assert_close(raw[name], raw_before[name], msg="Loading must not mutate the artifact.")

    residuals = organism["residuals"]
    raw_pre = (residuals - raw["b_dec"]) @ raw["w_enc"] + raw["b_enc"]
    raw_acts = t.relu(raw_pre) * (raw_pre > raw["threshold"])
    raw_recon = raw_acts @ raw["w_dec"] + raw["b_dec"]
    normalized_pre = (residuals - sae.b_dec) @ sae.w_enc + sae.b_enc
    normalized_acts = t.relu(normalized_pre) * (normalized_pre > sae.threshold)
    normalized_recon = normalized_acts @ sae.w_dec + sae.b_dec
    t.testing.assert_close(normalized_recon, raw_recon, atol=1e-6, rtol=1e-6)

    _raises(KeyError, lambda: load_normalized_sae({"w_enc": t.eye(2)}))
    bad = {name: value.clone() for name, value in raw.items()}
    bad["w_dec"][0] = 0
    _raises(ValueError, lambda: load_normalized_sae(bad))
    print("All tests in `test_load_normalized_sae_preserves_function` passed!")


def test_jump_relu_recovers_exact_latents(
    load_normalized_sae: Callable | None = None,
    encode_jump_relu: Callable | None = None,
) -> None:
    solutions = _solutions()
    load_normalized_sae = load_normalized_sae or solutions.load_normalized_sae
    encode_jump_relu = encode_jump_relu or solutions.encode_jump_relu
    organism = utils.make_ground_truth_organism()
    sae, _ = load_normalized_sae(organism["raw_artifact"])
    acts = encode_jump_relu(organism["residuals"], sae)
    t.testing.assert_close(acts, organism["latent_codes"], atol=1e-5, rtol=1e-5)

    identity_sae = solutions.SparseAutoencoder(
        w_enc=t.eye(2),
        w_dec=t.eye(2),
        b_enc=t.zeros(2),
        b_dec=t.zeros(2),
        threshold=t.tensor([0.25, 0.25]),
    )
    boundary_acts = encode_jump_relu(t.tensor([0.25, 0.25]), identity_sae)
    assert boundary_acts.count_nonzero() == 0, "JumpReLU should use a strict threshold gate."
    _raises(ValueError, lambda: encode_jump_relu(t.zeros(2, 7), sae))
    print("All tests in `test_jump_relu_recovers_exact_latents` passed!")


def test_reconstruction_is_exact(
    load_normalized_sae: Callable | None = None,
    encode_jump_relu: Callable | None = None,
    reconstruct_from_features: Callable | None = None,
    reconstruction_report: Callable | None = None,
) -> None:
    solutions = _solutions()
    load_normalized_sae = load_normalized_sae or solutions.load_normalized_sae
    encode_jump_relu = encode_jump_relu or solutions.encode_jump_relu
    reconstruct_from_features = reconstruct_from_features or solutions.reconstruct_from_features
    reconstruction_report = reconstruction_report or solutions.reconstruction_report
    organism = utils.make_ground_truth_organism()
    sae, _ = load_normalized_sae(organism["raw_artifact"])
    acts = encode_jump_relu(organism["residuals"], sae)
    recon = reconstruct_from_features(acts, sae)
    report = reconstruction_report(organism["residuals"], recon)
    t.testing.assert_close(recon, organism["residuals"], atol=1e-5, rtol=1e-5)
    assert report.mse < 1e-12, "The exact organism should reconstruct to floating-point precision."
    assert report.relative_mse < 1e-11, "Relative MSE should also expose exact reconstruction."
    assert abs(report.explained_variance - 1.0) < 1e-6, (
        "Exact reconstruction should have explained variance one."
    )
    print("All tests in `test_reconstruction_is_exact` passed!")


def test_feature_score_reductions(
    feature_score_vector: Callable | None = None,
) -> None:
    feature_score_vector = feature_score_vector or _solutions().feature_score_vector
    acts = t.tensor(
        [
            [[0.0, 0.1], [0.0, 0.5], [0.0, 0.2]],
            [[0.0, 0.4], [0.0, 0.3], [0.0, 0.9]],
        ]
    )
    t.testing.assert_close(feature_score_vector(acts, 1, reduction="max"), t.tensor([0.5, 0.9]))
    t.testing.assert_close(
        feature_score_vector(acts, 1, reduction="mean"), t.tensor([0.8 / 3, 1.6 / 3])
    )
    t.testing.assert_close(feature_score_vector(acts, 1, reduction="last"), t.tensor([0.2, 0.9]))
    t.testing.assert_close(feature_score_vector(acts[:, -1], 1), t.tensor([0.2, 0.9]))
    _raises(ValueError, lambda: feature_score_vector(acts, 1, reduction="sum"))
    print("All tests in `test_feature_score_reductions` passed!")


def test_auc_and_heldout_controls(
    roc_auc_binary: Callable | None = None,
    validate_heldout_feature: Callable | None = None,
    load_normalized_sae: Callable | None = None,
    encode_jump_relu: Callable | None = None,
) -> None:
    solutions = _solutions()
    roc_auc_binary = roc_auc_binary or solutions.roc_auc_binary
    validate_heldout_feature = validate_heldout_feature or solutions.validate_heldout_feature
    load_normalized_sae = load_normalized_sae or solutions.load_normalized_sae
    encode_jump_relu = encode_jump_relu or solutions.encode_jump_relu
    assert roc_auc_binary(t.tensor([0.0, 0.0, 1.0, 1.0]), t.tensor([0, 1, 0, 1])) == 0.5, (
        "AUC should count tied positive-negative pairs as half wins."
    )

    organism = utils.make_ground_truth_organism()
    sae, _ = load_normalized_sae(organism["raw_artifact"])
    acts = encode_jump_relu(organism["residuals"], sae)
    heldout = ~organism["train_mask"]
    report = validate_heldout_feature(
        acts[heldout, organism["target_feature_id"]],
        organism["labels"][heldout],
        acts[heldout, organism["control_feature_id"]],
        organism["shuffled_labels"][heldout],
    )
    assert report.feature_auc == 1.0, "The planted semantic feature should rank every positive first."
    assert report.random_feature_auc == 0.5, "The matched random feature should be at chance."
    assert report.shuffled_label_auc == 0.5, "Rank-balanced shuffled labels should be at chance."
    assert report.threshold_accuracy == 1.0, "The midpoint threshold should separate the exact classes."
    assert abs(report.positive_mean - 1.15) < 1e-6, "Held-out positive activation mean should be 1.15."
    assert report.negative_mean == 0.0, "Held-out negatives should not activate the target feature."
    _raises(ValueError, lambda: roc_auc_binary(t.ones(3), t.ones(3, dtype=t.bool)))
    print("All tests in `test_auc_and_heldout_controls` passed!")


def test_density_finds_rare_and_dead_features(
    activation_density: Callable | None = None,
) -> None:
    activation_density = activation_density or _solutions().activation_density
    acts = utils.make_ground_truth_organism()["latent_codes"]
    density, dead = activation_density(acts)
    t.testing.assert_close(
        density,
        t.tensor([0.5, 0.5, 0.65, 0.075, 0.5, 0.0]),
        atol=1e-7,
        rtol=0,
    )
    assert dead.nonzero().flatten().tolist() == [5], "Only planted feature 5 should be dead."
    print("All tests in `test_density_finds_rare_and_dead_features` passed!")


def test_ablation_has_matched_feature_control(
    ablate_feature: Callable | None = None,
    load_normalized_sae: Callable | None = None,
    encode_jump_relu: Callable | None = None,
    reconstruct_from_features: Callable | None = None,
) -> None:
    solutions = _solutions()
    ablate_feature = ablate_feature or solutions.ablate_feature
    load_normalized_sae = load_normalized_sae or solutions.load_normalized_sae
    encode_jump_relu = encode_jump_relu or solutions.encode_jump_relu
    reconstruct_from_features = reconstruct_from_features or solutions.reconstruct_from_features
    organism = utils.make_ground_truth_organism()
    sae, _ = load_normalized_sae(organism["raw_artifact"])
    acts = encode_jump_relu(organism["residuals"], sae)
    original = acts.clone()
    target_ablated = ablate_feature(acts, 0)
    control_ablated = ablate_feature(acts, 4)
    t.testing.assert_close(acts, original, msg="Ablation must not mutate the cached activations.")
    assert target_ablated[:, 0].count_nonzero() == 0, "Target ablation should zero feature 0."
    assert control_ablated[:, 4].count_nonzero() == 0, "Control ablation should zero feature 4."

    positive_heldout = (~organism["train_mask"]) & organism["labels"]
    readout = sae.w_dec[0]
    baseline = (organism["residuals"] - sae.b_dec) @ readout
    target = (reconstruct_from_features(target_ablated, sae) - sae.b_dec) @ readout
    control = (reconstruct_from_features(control_ablated, sae) - sae.b_dec) @ readout
    target_delta = (baseline[positive_heldout] - target[positive_heldout]).mean()
    control_delta = (baseline[positive_heldout] - control[positive_heldout]).mean()
    t.testing.assert_close(target_delta, t.tensor(1.15), atol=1e-6, rtol=1e-6)
    t.testing.assert_close(control_delta, t.tensor(0.0), atol=1e-6, rtol=0)
    print("All tests in `test_ablation_has_matched_feature_control` passed!")


def test_steering_normalizes_direction(
    steer_residuals: Callable | None = None,
) -> None:
    steer_residuals = steer_residuals or _solutions().steer_residuals
    residuals = t.tensor([[1.0, 2.0], [3.0, 4.0]])
    strengths = t.tensor([-2.0, 0.0, 1.5])
    steered = steer_residuals(residuals, t.tensor([30.0, 40.0]), strengths)
    assert steered.shape == (3, 2, 2), "Steering should prepend one axis for intervention strength."
    step_norms = (steered - residuals).norm(dim=-1)
    t.testing.assert_close(step_norms, strengths.abs()[:, None].expand_as(step_norms))
    t.testing.assert_close(steered[1], residuals)
    _raises(ValueError, lambda: steer_residuals(residuals, t.zeros(2), strengths))
    print("All tests in `test_steering_normalizes_direction` passed!")


def test_direct_logit_attribution_has_known_tokens(
    direct_logit_attribution: Callable | None = None,
    load_normalized_sae: Callable | None = None,
) -> None:
    solutions = _solutions()
    direct_logit_attribution = direct_logit_attribution or solutions.direct_logit_attribution
    load_normalized_sae = load_normalized_sae or solutions.load_normalized_sae
    organism = utils.make_ground_truth_organism()
    sae, _ = load_normalized_sae(organism["raw_artifact"])
    dla = direct_logit_attribution(sae.w_dec, organism["unembedding"])
    t.testing.assert_close(
        dla[0], t.tensor([1.2, -1.0, 0.9, -0.7, 0.0, 0.0]), atol=1e-6, rtol=1e-6
    )
    assert dla[0].topk(2).indices.tolist() == [0, 2], (
        "The target feature should directly promote the known code tokens."
    )
    assert organism["token_names"][dla[0].argmin().item()] == " story", (
        "The target feature should most strongly suppress the known story token."
    )
    print("All tests in `test_direct_logit_attribution_has_known_tokens` passed!")


def test_ground_truth_contract(run_smoke_test: Callable | None = None) -> None:
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["reconstruction"]["mse"] < 1e-12, "Notebook analysis should reconstruct exactly."
    assert result["validation"]["feature_auc"] == 1.0, "Target held-out AUC should be one."
    assert result["validation"]["random_feature_auc"] == 0.5, (
        "Random-feature held-out AUC should be chance."
    )
    assert result["validation"]["shuffled_label_auc"] == 0.5, (
        "Shuffled-label held-out AUC should be chance."
    )
    assert result["dead_feature_ids"] == [5], "Notebook analysis should expose the dead feature."
    assert abs(result["train_confound_auc"] - 1.0) < 1e-6, (
        "The planted confound should look perfect on training examples."
    )
    assert abs(result["heldout_confound_auc"] - 0.5) < 1e-6, (
        "The planted confound should collapse to chance on held-out examples."
    )
    assert abs(result["target_ablation_delta"] - 1.15) < 1e-6, (
        "Target ablation should remove the positive held-out feature contribution."
    )
    assert abs(result["control_ablation_delta"]) < 1e-6, (
        "Matched control ablation should leave the target readout unchanged."
    )
    t.testing.assert_close(
        t.tensor(result["target_steering_delta"]),
        t.tensor(result["steering_strengths"]),
        atol=1e-6,
        rtol=1e-6,
    )
    t.testing.assert_close(
        t.tensor(result["control_steering_delta"]), t.zeros(5), atol=1e-6, rtol=0
    )
    print("All tests in `test_ground_truth_contract` passed!")
