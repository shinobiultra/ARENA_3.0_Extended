"""Immediate semantic tests for the [6.1] learner exercises."""

from collections.abc import Callable

import torch as t


def _solutions():
    from chapter6_sparse_feature_methods.exercises.part1_sae_variants import solutions

    return solutions


def test_planted_sparse_ground_truth(
    make_planted_dictionary: Callable | None = None,
    sample_planted_batch: Callable | None = None,
):
    solutions = _solutions()
    make_planted_dictionary = make_planted_dictionary or solutions.make_planted_dictionary
    sample_planted_batch = sample_planted_batch or solutions.sample_planted_batch
    dictionary = make_planted_dictionary(n_features=5, d_model=3, seed=4)
    batch = sample_planted_batch(
        dictionary,
        n_examples=128,
        feature_probability=0.25,
        noise_std=0.0,
        seed=5,
    )
    assert batch.activations.shape == (128, 3), "Activations must have shape (examples, d_model)."
    assert batch.feature_acts.shape == (128, 5), "Feature acts must have shape (examples, features)."
    t.testing.assert_close(dictionary.norm(dim=-1), t.ones(5))
    t.testing.assert_close(batch.activations, batch.feature_acts @ dictionary)
    t.testing.assert_close(batch.labels, batch.feature_acts[:, 0] > 0)
    assert 0 < int(batch.labels.sum()) < batch.labels.numel(), (
        "The controlled batch must contain both target-positive and target-negative examples."
    )
    print("All tests in `test_planted_sparse_ground_truth` passed!")


def test_encoder_rules_and_jumprelu_gradient(
    relu_l1_encode: Callable | None = None,
    topk_encode: Callable | None = None,
    gated_encode: Callable | None = None,
    jumprelu_encode: Callable | None = None,
    JumpReLU=None,
):
    solutions = _solutions()
    relu_l1_encode = relu_l1_encode or solutions.relu_l1_encode
    topk_encode = topk_encode or solutions.topk_encode
    gated_encode = gated_encode or solutions.gated_encode
    jumprelu_encode = jumprelu_encode or solutions.jumprelu_encode
    JumpReLU = JumpReLU or solutions.JumpReLU
    pre = t.tensor([[1.0, -1.0, 0.2, 3.0], [0.1, 2.0, -0.5, 0.4]])
    gate = t.tensor([[1.0, 1.0, -1.0, 1.0], [-1.0, 2.0, 1.0, 0.0]])
    t.testing.assert_close(relu_l1_encode(pre), t.tensor([[1.0, 0.0, 0.2, 3.0], [0.1, 2.0, 0.0, 0.4]]))
    t.testing.assert_close(topk_encode(pre, k=2), t.tensor([[1.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, 0.4]]))
    t.testing.assert_close(gated_encode(pre, gate), t.tensor([[1.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, 0.0]]))
    t.testing.assert_close(jumprelu_encode(pre, 0.5, 0.1), t.tensor([[1.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, 0.0]]))

    values = t.tensor([[0.8, 1.0, 1.2]], requires_grad=True)
    threshold = t.ones(3, requires_grad=True)
    JumpReLU.apply(values, threshold, 0.5).sum().backward()
    t.testing.assert_close(values.grad, t.tensor([[0.0, 0.0, 1.0]]))
    t.testing.assert_close(threshold.grad, t.tensor([-2.0, -2.0, -2.0]))
    print("All tests in `test_encoder_rules_and_jumprelu_gradient` passed!")


def test_variant_loss_terms(sparse_autoencoder_loss: Callable | None = None):
    solutions = _solutions()
    sparse_autoencoder_loss = sparse_autoencoder_loss or solutions.sparse_autoencoder_loss
    activations = t.zeros(2, 2)
    reconstruction = t.ones(2, 2)
    feature_acts = t.tensor([[1.0, 0.0], [0.0, 2.0]])
    base = dict(reconstruction=reconstruction, feature_acts=feature_acts, pre_acts=feature_acts)
    relu_output = solutions.SAEForward(**base)
    relu = sparse_autoencoder_loss(
        relu_output,
        activations,
        variant="relu_l1",
        sparsity_coefficient=0.1,
    )
    topk = sparse_autoencoder_loss(
        relu_output,
        activations,
        variant="topk",
        sparsity_coefficient=0.1,
    )
    gated = sparse_autoencoder_loss(
        solutions.SAEForward(
            **base,
            gate_pre_acts=t.tensor([[1.0, -1.0], [0.5, 0.0]]),
            auxiliary_reconstruction=t.zeros(2, 2),
        ),
        activations,
        variant="gated",
        sparsity_coefficient=0.1,
    )
    jump = sparse_autoencoder_loss(
        solutions.SAEForward(**base, jump_threshold=t.tensor([0.5, 0.5])),
        activations,
        variant="jumprelu",
        sparsity_coefficient=0.1,
        jump_bandwidth=0.1,
    )
    t.testing.assert_close(relu.reconstruction, t.tensor(1.0))
    t.testing.assert_close(relu.sparsity, t.tensor(1.5))
    t.testing.assert_close(relu.total, t.tensor(1.15))
    t.testing.assert_close(topk.total, t.tensor(1.0))
    t.testing.assert_close(gated.sparsity, t.tensor(0.75))
    t.testing.assert_close(gated.total, t.tensor(1.075))
    t.testing.assert_close(jump.sparsity, t.tensor(1.0))
    t.testing.assert_close(jump.total, t.tensor(1.1))
    print("All tests in `test_variant_loss_terms` passed!")


def test_metrics_recovery_and_heldout_auc(
    decode_features: Callable | None = None,
    dictionary_recovery_report: Callable | None = None,
    best_feature_auc: Callable | None = None,
    evaluate_selected_auc: Callable | None = None,
):
    solutions = _solutions()
    decode_features = decode_features or solutions.decode_features
    dictionary_recovery_report = dictionary_recovery_report or solutions.dictionary_recovery_report
    best_feature_auc = best_feature_auc or solutions.best_feature_auc
    evaluate_selected_auc = evaluate_selected_auc or solutions.evaluate_selected_auc
    feature_acts = t.tensor([[1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    t.testing.assert_close(decode_features(feature_acts, t.eye(3)), feature_acts)
    recovery = dictionary_recovery_report(
        t.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        t.eye(3),
        threshold=0.9,
    )
    assert abs(recovery.recovered_fraction - 2 / 3) < 1e-6, (
        "Recovery must count planted directions, so the missing e2 direction leaves 2/3 recovered."
    )
    assert abs(recovery.duplicate_fraction - 1 / 3) < 1e-6, (
        "Two learned rows map to e0, so one of three decoder rows is a duplicate."
    )
    train_features = t.tensor([[0.0, 0.1], [0.0, 0.2], [2.0, 0.1], [3.0, 0.2]])
    heldout_features = t.tensor([[0.1, 4.0], [0.2, 3.0], [2.1, 2.0], [3.1, 1.0]])
    labels = t.tensor([False, False, True, True])
    selection = best_feature_auc(train_features, labels)
    assert selection.feature_id == 0, "Feature selection must use the perfectly predictive train feature."
    assert selection.auc == 1.0, "The controlled train feature should separate both classes exactly."
    assert evaluate_selected_auc(heldout_features, labels, selection) == 1.0, (
        "The frozen train-selected feature and polarity must transfer to held-out data."
    )
    print("All tests in `test_metrics_recovery_and_heldout_auc` passed!")


def test_causal_interventions_beat_equal_strength_random_controls(
    causal_intervention_report: Callable | None = None,
):
    solutions = _solutions()
    causal_intervention_report = causal_intervention_report or solutions.causal_intervention_report
    dictionary = t.eye(3)
    heldout = solutions.sample_planted_batch(
        dictionary,
        n_examples=256,
        feature_probability=0.4,
        noise_std=0.0,
        seed=8,
    )
    model = solutions.SparseAutoencoder(
        heldout.activations,
        solutions.SAETrainConfig(variant="relu_l1", d_sae=3, steps=1),
    )
    with t.no_grad():
        model.w_dec.copy_(t.eye(3))
        model.w_enc.copy_(t.eye(3))
        model.b_enc.zero_()
        model.b_dec.zero_()
    report = causal_intervention_report(model, heldout)
    assert report.target_feature_id == 0, "Identity decoder row zero must match planted feature zero."
    assert abs(report.steering_delta - 0.75) < 1e-6, (
        "Unit-norm matched steering at coefficient 0.75 must move the target projection by 0.75."
    )
    assert abs(report.random_steering_delta) < 1e-6, (
        "The orthogonal random steering direction must not move the target projection."
    )
    assert report.ablation_drop > 0.8, (
        "Removing the matched decoded contribution must substantially reduce the target projection."
    )
    assert abs(report.random_ablation_drop) < 1e-6, (
        "The equal-coefficient orthogonal ablation control must have negligible target effect."
    )
    print("All tests in `test_causal_interventions_beat_equal_strength_random_controls` passed!")


def test_end_to_end_variant_comparison(run_variant_comparison: Callable | None = None):
    solutions = _solutions()
    run_variant_comparison = run_variant_comparison or solutions.run_variant_comparison
    comparison = run_variant_comparison(steps=450, train_examples=2048, heldout_examples=1024)
    assert set(comparison.results) == set(solutions.VARIANTS), (
        "The comparison must evaluate ReLU-L1, TopK, gated, and JumpReLU."
    )
    assert comparison.true_l0 > 0, "The planted dataset must contain active latent features."
    for result in comparison.results.values():
        assert result.metrics.reconstruction_mse < comparison.zero_baseline_mse, (
            f"{result.name} must reconstruct held-out activations better than the train-mean baseline."
        )
        assert 0.35 < result.shuffled_auc < 0.65, (
            f"{result.name} shuffled-label AUC should remain near chance."
        )
        assert abs(result.intervention.steering_delta) > abs(result.intervention.random_steering_delta) + 0.2, (
            f"{result.name} matched steering must beat its equal-strength orthogonal control."
        )
        assert abs(result.intervention.ablation_drop) > abs(result.intervention.random_ablation_drop) + 0.1, (
            f"{result.name} matched ablation must beat its equal-coefficient random-direction control."
        )
    assert max(r.recovery.recovered_fraction for r in comparison.results.values()) >= 0.75, (
        "At least one variant must recover three quarters of the planted dictionary."
    )
    assert max(r.heldout_auc for r in comparison.results.values()) >= 0.9, (
        "At least one train-selected feature must exceed 0.9 held-out AUC."
    )
    print("All tests in `test_end_to_end_variant_comparison` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    solutions = _solutions()
    run_smoke_test = run_smoke_test or solutions.run_smoke_test
    report = run_smoke_test(cpu=True)
    assert len(report["variants"]) == 4, "The smoke report must include all four SAE variants."
    assert report["zero_baseline_mse"] > 0, "The reconstruction baseline must be nondegenerate."
    assert report["true_l0"] > 0, "The smoke dataset must contain active planted features."
    print("All tests in `test_notebook_contract` passed!")
