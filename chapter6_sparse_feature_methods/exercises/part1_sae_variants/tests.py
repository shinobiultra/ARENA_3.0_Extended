from collections.abc import Callable

import torch as t

from arena_ext import features as feature_reference
from arena_ext import sae_variants as reference


def _solutions():
    from chapter6_sparse_feature_methods.exercises.part1_sae_variants import solutions

    return solutions


def test_encoder_variants_match_reference_and_sparsity_rules(
    relu_l1_encode: Callable | None = None,
    topk_encode: Callable | None = None,
    gated_encode: Callable | None = None,
    jumprelu_encode: Callable | None = None,
):
    solutions = _solutions()
    relu_l1_encode = relu_l1_encode or solutions.relu_l1_encode
    topk_encode = topk_encode or solutions.topk_encode
    gated_encode = gated_encode or solutions.gated_encode
    jumprelu_encode = jumprelu_encode or solutions.jumprelu_encode
    pre_acts = t.tensor([[1.0, -1.0, 0.2, 3.0], [0.1, 2.0, -0.5, 0.4]])
    gate_logits = t.tensor([[1.0, 1.0, -1.0, 1.0], [-1.0, 2.0, 1.0, 0.0]])

    relu_l1 = relu_l1_encode(pre_acts, l1_coefficient=0.5)
    topk = topk_encode(pre_acts, k=2)
    gated = gated_encode(pre_acts, gate_logits)
    jumprelu = jumprelu_encode(pre_acts, threshold=0.5)

    t.testing.assert_close(
        relu_l1,
        reference.relu_l1_encode(pre_acts, l1_coefficient=0.5),
        msg="ReLU/L1 encoder should subtract the L1 threshold before clamping at zero.",
    )
    t.testing.assert_close(
        topk,
        reference.topk_encode(pre_acts, k=2),
        msg="TopK encoder should keep only the k largest nonnegative features per row.",
    )
    t.testing.assert_close(
        gated,
        reference.gated_encode(pre_acts, gate_logits),
        msg="Gated encoder should require both positive magnitude and an open gate.",
    )
    t.testing.assert_close(
        jumprelu,
        reference.jumprelu_encode(pre_acts, threshold=0.5),
        msg="JumpReLU should preserve values only after the threshold is crossed.",
    )
    assert topk.gt(0).sum(dim=-1).tolist() == [2, 2], (
        "TopK should keep exactly two active positive entries in each controlled row."
    )
    assert gated[1, 3].item() == 0.0, (
        "Gate logits exactly at the threshold should not open the gated feature."
    )
    print("All tests in `test_encoder_variants_match_reference_and_sparsity_rules` passed!")


def test_decode_and_metrics_match_identity_contract(
    decode_features: Callable | None = None,
    sae_variant_metrics: Callable | None = None,
    feature_density: Callable | None = None,
    l0: Callable | None = None,
    dead_feature_fraction: Callable | None = None,
):
    solutions = _solutions()
    decode_features = decode_features or solutions.decode_features
    sae_variant_metrics = sae_variant_metrics or solutions.sae_variant_metrics
    feature_density = feature_density or solutions.feature_density
    l0 = l0 or solutions.l0
    dead_feature_fraction = dead_feature_fraction or solutions.dead_feature_fraction
    feature_acts = t.tensor([[1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    decoder = t.eye(3)
    activations = decode_features(feature_acts, decoder)
    t.testing.assert_close(
        activations,
        reference.decode_features(feature_acts, decoder),
        msg="Decoding should multiply feature activations by decoder rows.",
    )
    metrics = sae_variant_metrics(
        "identity",
        activations=activations,
        reconstructed_activations=activations,
        feature_acts=feature_acts,
    )
    reference_metrics = reference.sae_variant_metrics(
        "identity",
        activations=activations,
        reconstructed_activations=activations,
        feature_acts=feature_acts,
    )
    assert metrics.__dict__ == reference_metrics.__dict__, (
        "SAE metrics should match reconstruction MSE, L0, density, and dead-feature reference."
    )
    t.testing.assert_close(
        feature_density(feature_acts),
        t.tensor([0.5, 0.5, 0.5]),
        msg="Feature density should be per-feature firing rate over examples.",
    )
    assert l0(feature_acts) == 1.5, "L0 should average the number of active features per row."
    assert dead_feature_fraction(feature_acts) == 0.0, (
        "No feature should be dead when every column fires at least once."
    )
    assert metrics.reconstruction_mse == 0.0, (
        "Identity decoder should have zero reconstruction error."
    )
    print("All tests in `test_decode_and_metrics_match_identity_contract` passed!")


def test_toy_superposition_batch_has_planted_sparse_structure(
    make_toy_superposition_batch: Callable | None = None,
    density_is_nondegenerate: Callable | None = None,
):
    solutions = _solutions()
    make_toy_superposition_batch = (
        make_toy_superposition_batch or solutions.make_toy_superposition_batch
    )
    density_is_nondegenerate = density_is_nondegenerate or solutions.density_is_nondegenerate
    batch = make_toy_superposition_batch(
        batch=32,
        n_features=6,
        d_model=3,
        feature_probability=0.25,
        seed=0,
    )
    reference_batch = reference.make_toy_superposition_batch(
        batch=32,
        n_features=6,
        d_model=3,
        feature_probability=0.25,
        seed=0,
    )
    t.testing.assert_close(
        batch.feature_acts,
        reference_batch.feature_acts,
        msg="Toy superposition feature activations should be deterministic under the seed.",
    )
    t.testing.assert_close(
        batch.activations,
        batch.feature_acts @ batch.dictionary,
        msg="Toy activations should be sparse features mixed through the planted dictionary.",
    )
    t.testing.assert_close(
        batch.dictionary.norm(dim=-1),
        t.ones(batch.dictionary.shape[0]),
        msg="Planted dictionary rows should be unit norm.",
    )
    assert list(batch.feature_acts.shape) == [32, 6], (
        "Toy feature activations should have shape (batch, n_features)."
    )
    assert list(batch.activations.shape) == [32, 3], (
        "Toy activations should have shape (batch, d_model)."
    )
    assert density_is_nondegenerate(batch.feature_acts), (
        "Toy sparse features should contain some active entries without firing everywhere."
    )
    print("All tests in `test_toy_superposition_batch_has_planted_sparse_structure` passed!")


def test_dictionary_recovery_detects_duplicates_and_missing_features(
    dictionary_recovery_report: Callable | None = None,
):
    solutions = _solutions()
    dictionary_recovery_report = dictionary_recovery_report or solutions.dictionary_recovery_report
    true_dictionary = t.eye(3)
    learned_decoder = t.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    report = dictionary_recovery_report(learned_decoder, true_dictionary, threshold=0.9)
    reference_report = reference.dictionary_recovery_report(
        learned_decoder,
        true_dictionary,
        threshold=0.9,
    )
    assert abs(report.recovered_fraction - reference_report.recovered_fraction) < 1e-6, (
        "Recovered fraction should match the cosine-threshold reference."
    )
    assert abs(report.recovered_fraction - (2 / 3)) < 1e-6, (
        "The duplicated decoder should recover two of three true feature directions."
    )
    assert abs(report.duplicate_fraction - (1 / 3)) < 1e-6, (
        "Duplicate fraction should detect that two learned rows map to the same true feature."
    )
    t.testing.assert_close(
        report.best_learned_for_true,
        t.tensor([0, 2, 0]),
        msg="The missing third true feature should fall back to the first tied learned row.",
    )
    print("All tests in `test_dictionary_recovery_detects_duplicates_and_missing_features` passed!")


def test_best_feature_auc_handles_predictive_and_antipredictive_features(
    roc_auc_binary: Callable | None = None,
    best_feature_auc: Callable | None = None,
):
    solutions = _solutions()
    roc_auc_binary = roc_auc_binary or solutions.roc_auc_binary
    best_feature_auc = best_feature_auc or solutions.best_feature_auc
    feature_acts = t.tensor(
        [
            [0.1, 0.0, 2.0],
            [0.2, 0.0, 1.8],
            [0.1, 1.0, 0.2],
            [0.2, 2.0, 0.1],
        ]
    )
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)
    auc_report = best_feature_auc(feature_acts, labels)
    reference_report = reference.best_feature_auc(feature_acts, labels)
    assert auc_report.__dict__ == reference_report.__dict__, (
        "Best feature AUC should match the independent rank-statistic implementation."
    )
    assert auc_report.feature_id in {1, 2}, (
        "The best feature should be one of the perfectly separating directions."
    )
    assert auc_report.auc == 1.0, (
        "Both a predictive and an antipredictive feature should count as perfect separation."
    )
    assert roc_auc_binary(feature_acts[:, 1], labels) == 1.0, (
        "Feature 1 should rank positives above negatives."
    )
    assert roc_auc_binary(feature_acts[:, 2], labels) == 0.0, (
        "Feature 2 should rank positives below negatives before polarity correction."
    )
    print("All tests in `test_best_feature_auc_handles_predictive_and_antipredictive_features` passed!")


def test_decoder_steering_changes_last_position_and_reports_control(
    apply_decoder_steering: Callable | None = None,
    steering_comparison_report: Callable | None = None,
):
    solutions = _solutions()
    apply_decoder_steering = apply_decoder_steering or solutions.apply_decoder_steering
    steering_comparison_report = (
        steering_comparison_report or solutions.steering_comparison_report
    )
    activations = t.zeros(2, 3, 3)
    decoder_vectors = t.eye(3)
    steered = apply_decoder_steering(activations, decoder_vectors, [1], 2.0)
    all_position_steered = apply_decoder_steering(
        activations,
        decoder_vectors,
        [1],
        2.0,
        positions="all",
    )
    random_control = apply_decoder_steering(activations, decoder_vectors, [2], 0.1)

    t.testing.assert_close(
        steered,
        feature_reference.apply_decoder_steering(activations, decoder_vectors, [1], 2.0),
        msg="Default decoder steering should add the selected decoder vector at the last position.",
    )
    assert steered[:, :-1, :].abs().sum().item() == 0.0, (
        "Default steering should leave non-final positions unchanged."
    )
    assert all_position_steered[:, :, 1].eq(2.0).all().item(), (
        "All-position steering should add the steering direction at every sequence position."
    )
    report = steering_comparison_report(
        baseline_scores=activations[:, -1, 1],
        steered_scores=steered[:, -1, 1],
        random_control_scores=random_control[:, -1, 1],
    )
    reference_report = feature_reference.steering_comparison_report(
        baseline_scores=activations[:, -1, 1],
        steered_scores=steered[:, -1, 1],
        random_control_scores=random_control[:, -1, 1],
    )
    assert report.__dict__ == reference_report.__dict__, (
        "Steering report should match the reference baseline/steered/random deltas."
    )
    assert report.steered_delta > 0.0 and report.random_delta == 0.0, (
        "Target decoder steering should move the target score while random control should not."
    )
    assert report.passes_control, (
        "Steered delta magnitude should exceed the random-control delta magnitude."
    )
    print("All tests in `test_decoder_steering_changes_last_position_and_reports_control` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["reconstruction"]["reconstruction_mse"] == 0.0, (
        "Notebook contract should include an exact identity-decoder reconstruction."
    )
    assert result["toy_superposition"]["density_nondegenerate"], (
        "Notebook contract should include a nondegenerate toy sparse-feature batch."
    )
    assert result["dictionary_recovery"]["recovered_fraction"] > 0.6, (
        "Notebook contract should include planted dictionary recovery diagnostics."
    )
    assert result["feature_auc"]["auc"] == 1.0, (
        "Notebook contract should include held-out feature-label AUC."
    )
    assert result["steering"]["passes_control"], (
        "Notebook contract should include decoder-vector steering against a random control."
    )
    print("All tests in `test_notebook_contract` passed!")
