from collections.abc import Callable

import torch as t

from arena_ext import features as reference


def _solutions():
    from chapter5_modern_architectures.exercises.part2_gemma_scope_feature_steering import (
        solutions,
    )

    return solutions


def test_feature_density_l0_and_dead_fraction_match_reference(
    feature_density: Callable | None = None,
    l0: Callable | None = None,
    dead_feature_fraction: Callable | None = None,
):
    solutions = _solutions()
    feature_density = feature_density or solutions.feature_density
    l0 = l0 or solutions.l0
    dead_feature_fraction = dead_feature_fraction or solutions.dead_feature_fraction
    feature_acts = t.tensor(
        [
            [[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            [[4.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        ]
    )
    t.testing.assert_close(
        feature_density(feature_acts),
        reference.feature_density(feature_acts),
        msg="Feature density should be the per-feature firing rate over batch/position axes.",
    )
    assert l0(feature_acts) == reference.l0(feature_acts), (
        "L0 should be the average number of active sparse features per activation vector."
    )
    assert dead_feature_fraction(feature_acts) == reference.dead_feature_fraction(feature_acts), (
        "Dead-feature fraction should count features that never fire above threshold."
    )
    print("All tests in `test_feature_density_l0_and_dead_fraction_match_reference` passed!")


def test_compute_sae_reconstruction_metrics_matches_manual_and_reference(
    compute_sae_reconstruction_metrics: Callable | None = None,
):
    solutions = _solutions()
    compute_sae_reconstruction_metrics = (
        compute_sae_reconstruction_metrics or solutions.compute_sae_reconstruction_metrics
    )
    activations = t.tensor([[[1.0, -1.0], [2.0, 0.0]]])
    reconstructed = activations + t.tensor([[[0.5, -0.5], [0.0, 1.0]]])
    feature_acts = t.tensor([[[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]]])
    reference_logits = t.tensor([[[2.0, 0.0, -1.0]]])
    reconstructed_logits = t.tensor([[[1.5, 0.1, -0.7]]])
    actual = compute_sae_reconstruction_metrics(
        activations=activations,
        reconstructed_activations=reconstructed,
        feature_acts=feature_acts,
        reference_logits=reference_logits,
        reconstructed_logits=reconstructed_logits,
        clean_loss=1.0,
        reconstructed_loss=1.25,
        zero_ablation_loss=2.0,
    )
    expected = reference.compute_sae_reconstruction_metrics(
        activations=activations,
        reconstructed_activations=reconstructed,
        feature_acts=feature_acts,
        reference_logits=reference_logits,
        reconstructed_logits=reconstructed_logits,
        clean_loss=1.0,
        reconstructed_loss=1.25,
        zero_ablation_loss=2.0,
    )
    manual_mse = ((reconstructed - activations) ** 2).mean().item()
    assert actual.reconstruction_mse == manual_mse, (
        "Reconstruction MSE should be the mean squared error over activation elements."
    )
    assert actual.loss_recovered == 0.75, (
        "Loss recovered should be (zero_ablation_loss - reconstructed_loss) / "
        "(zero_ablation_loss - clean_loss)."
    )
    t.testing.assert_close(
        t.tensor(list(actual.__dict__.values())),
        t.tensor(list(expected.__dict__.values())),
        msg="All SAE reconstruction metrics should match the independent reference.",
    )
    print("All tests in `test_compute_sae_reconstruction_metrics_matches_manual_and_reference` passed!")


def test_direct_logit_attribution_projects_decoder_vectors(
    direct_logit_attribution: Callable | None = None,
):
    solutions = _solutions()
    direct_logit_attribution = direct_logit_attribution or solutions.direct_logit_attribution
    decoder_vectors = t.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    unembedding = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = direct_logit_attribution(decoder_vectors, unembedding, token_ids=[0, 2])
    expected = t.tensor([[1.0, 3.0], [4.0, 6.0], [-2.0, 0.0]])
    t.testing.assert_close(
        actual,
        expected,
        msg="DLA should compute decoder_vector @ W_U and then select requested tokens.",
    )
    print("All tests in `test_direct_logit_attribution_projects_decoder_vectors` passed!")


def test_feature_detection_report_handles_heldout_controls(
    feature_detection_report: Callable | None = None,
    roc_auc_binary: Callable | None = None,
):
    solutions = _solutions()
    feature_detection_report = feature_detection_report or solutions.feature_detection_report
    roc_auc_binary = roc_auc_binary or solutions.roc_auc_binary
    scores = t.tensor([0.9, 0.8, 0.2, 0.2])
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)
    report = feature_detection_report(scores, labels)
    assert report.auc == 1.0, (
        f"Perfectly separated positives and negatives should have AUC 1.0, got {report.auc}."
    )
    assert report.threshold_accuracy == 1.0, (
        "Default midpoint threshold should classify this held-out control perfectly."
    )
    tied_auc = roc_auc_binary(
        t.tensor([0.5, 0.5, 0.0, 1.0]),
        t.tensor([1, 0, 0, 1], dtype=t.bool),
    )
    assert tied_auc == reference.roc_auc_binary(
        t.tensor([0.5, 0.5, 0.0, 1.0]),
        t.tensor([1, 0, 0, 1], dtype=t.bool),
    ), "AUC should use average ranks for tied scores."
    print("All tests in `test_feature_detection_report_handles_heldout_controls` passed!")


def test_top_activating_examples_and_ablation_match_reference(
    top_activating_examples: Callable | None = None,
    ablate_features: Callable | None = None,
):
    solutions = _solutions()
    top_activating_examples = top_activating_examples or solutions.top_activating_examples
    ablate_features = ablate_features or solutions.ablate_features
    _, _, feature_acts = solutions.synthetic_feature_batch()
    indices, values = top_activating_examples(feature_acts, feature_id=3, k=3)
    reference_indices, reference_values = reference.top_activating_examples(
        feature_acts,
        feature_id=3,
        k=3,
    )
    t.testing.assert_close(
        indices,
        reference_indices,
        msg="Top activating example indices should match flattened top-k positions.",
    )
    t.testing.assert_close(
        values,
        reference_values,
        msg="Top activating example values should match flattened top-k scores.",
    )
    zeroed = ablate_features(feature_acts, [3], replacement="zero")
    meaned = ablate_features(feature_acts, [3], replacement="mean")
    assert bool(zeroed[..., 3].eq(0).all().item()), (
        "Zero ablation should set every selected feature activation to zero."
    )
    t.testing.assert_close(
        meaned[..., 3],
        reference.ablate_features(feature_acts, [3], replacement="mean")[..., 3],
        msg="Mean ablation should replace selected features with their global mean.",
    )
    print("All tests in `test_top_activating_examples_and_ablation_match_reference` passed!")


def test_decoder_steering_and_random_control_report(
    apply_decoder_steering: Callable | None = None,
    steering_comparison_report: Callable | None = None,
):
    solutions = _solutions()
    apply_decoder_steering = apply_decoder_steering or solutions.apply_decoder_steering
    steering_comparison_report = (
        steering_comparison_report or solutions.steering_comparison_report
    )
    activations = t.zeros(2, 3, 4)
    decoder_vectors = t.eye(4)
    steered_last = apply_decoder_steering(activations, decoder_vectors, [1], 2.0)
    steered_all = apply_decoder_steering(
        activations,
        decoder_vectors,
        [0, 2],
        [1.0, -1.0],
        positions="all",
    )
    assert bool(steered_last[:, -1, 1].eq(2.0).all().item()), (
        "Last-position steering should add the selected decoder direction only at the last token."
    )
    assert bool(steered_last[:, :-1, :].eq(0).all().item()), (
        "Last-position steering should leave earlier positions unchanged."
    )
    assert bool(steered_all[..., 0].eq(1.0).all().item()), (
        "All-position steering should broadcast the first selected direction to every token."
    )
    assert bool(steered_all[..., 2].eq(-1.0).all().item()), (
        "All-position steering should apply the signed coefficient for each feature."
    )
    report = steering_comparison_report(
        baseline_scores=t.tensor([0.1, 0.2, 0.3]),
        steered_scores=t.tensor([0.5, 0.6, 0.7]),
        random_control_scores=t.tensor([0.2, 0.2, 0.3]),
    )
    assert report.passes_control, (
        "Feature steering should pass only when its score delta beats the random-control delta."
    )
    print("All tests in `test_decoder_steering_and_random_control_report` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["metrics"]["l0"] > 0, "Smoke metrics should report nonzero sparse activity."
    assert result["feature_validation"]["auc"] == 1.0, (
        "Smoke validation should include a perfectly separated held-out feature control."
    )
    assert result["logit_attribution"] == [[1.0, 3.0], [4.0, 6.0]], (
        "Smoke DLA should return the expected selected-token effects."
    )
    assert result["steering"]["passes_control"], (
        "Smoke steering should beat the random-feature control."
    )
    assert result["ablation"]["feature_3_zeroed"], (
        "Smoke ablation should zero the selected feature."
    )
    print("All tests in `test_notebook_contract` passed!")
