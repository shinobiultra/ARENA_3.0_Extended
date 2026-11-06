from collections.abc import Callable

import torch as t


def _solutions():
    from chapter6_sparse_feature_methods.exercises.part4_crosscoders_model_diffing import (
        solutions,
    )

    return solutions


def test_make_planted_crosscoder_data_has_exact_ground_truth(
    make_planted_crosscoder_data: Callable | None = None,
):
    solutions = _solutions()
    make_planted_crosscoder_data = (
        make_planted_crosscoder_data or solutions.make_planted_crosscoder_data
    )
    data = make_planted_crosscoder_data(n_examples=128, d_model=12, noise_std=0.0)

    expected_a = data.true_latents @ data.true_decoder_a
    expected_b = data.true_latents @ data.true_decoder_b
    t.testing.assert_close(data.model_a_activations, expected_a)
    t.testing.assert_close(data.model_b_activations, expected_b)
    assert data.feature_owners == ("shared", "shared", "model_a", "model_b"), (
        "The planted dictionary should contain two shared features followed by one "
        "feature specific to each model."
    )
    assert t.count_nonzero(data.true_decoder_b[2]).item() == 0, (
        "The model-A-specific feature must have no decoder contribution in model B."
    )
    assert t.count_nonzero(data.true_decoder_a[3]).item() == 0, (
        "The model-B-specific feature must have no decoder contribution in model A."
    )
    assert 0.3 <= data.behavior_labels.float().mean().item() <= 0.7, (
        "The planted behavior labels should be balanced enough for held-out AUC to be meaningful."
    )
    assert data.train_idx.numel() + data.heldout_idx.numel() == 128, (
        "The deterministic split must account for every generated example exactly once."
    )
    print("All tests in `test_make_planted_crosscoder_data_has_exact_ground_truth` passed!")


def test_sparse_crosscoder_forward_and_loss_shapes(
    SparseCrosscoder: type | None = None,
    sparse_crosscoder_loss: Callable | None = None,
    make_planted_crosscoder_data: Callable | None = None,
):
    solutions = _solutions()
    SparseCrosscoder = SparseCrosscoder or solutions.SparseCrosscoder
    sparse_crosscoder_loss = sparse_crosscoder_loss or solutions.sparse_crosscoder_loss
    make_planted_crosscoder_data = (
        make_planted_crosscoder_data or solutions.make_planted_crosscoder_data
    )

    data = make_planted_crosscoder_data(n_examples=64, d_model=12)
    model = SparseCrosscoder(d_model=12, n_latents=6, seed=2)
    output = model(data.model_a_activations[:16], data.model_b_activations[:16])
    assert output.model_a_feature_acts.shape == (16, 6), (
        "Model A should receive one nonnegative activation per learned latent and example."
    )
    assert output.model_b_feature_acts.shape == (16, 6), (
        "Model B should receive one nonnegative activation per learned latent and example."
    )
    assert output.reconstructed_model_a.shape == (16, 12), (
        "The model-A reconstruction must return to the original activation shape."
    )
    assert output.reconstructed_model_b.shape == (16, 12), (
        "The model-B reconstruction must return to the original activation shape."
    )
    assert output.model_a_feature_acts.min().item() >= 0, (
        "ReLU crosscoder activations for model A must be nonnegative."
    )
    assert output.model_b_feature_acts.min().item() >= 0, (
        "ReLU crosscoder activations for model B must be nonnegative."
    )

    loss = sparse_crosscoder_loss(
        output,
        data.model_a_activations[:16],
        data.model_b_activations[:16],
        l1_coefficient=0.008,
    )
    t.testing.assert_close(
        loss.total_loss,
        loss.reconstruction_loss + 0.008 * loss.l1_loss,
    )
    assert loss.reconstruction_loss.item() > 0, (
        "An untrained crosscoder should have a positive reconstruction error on planted activations."
    )
    print("All tests in `test_sparse_crosscoder_forward_and_loss_shapes` passed!")


def test_train_sparse_crosscoder_reduces_heldout_reconstruction(
    train_sparse_crosscoder: Callable | None = None,
    evaluate_sparse_crosscoder_reconstruction: Callable | None = None,
    make_planted_crosscoder_data: Callable | None = None,
):
    solutions = _solutions()
    train_sparse_crosscoder = train_sparse_crosscoder or solutions.train_sparse_crosscoder
    evaluate_sparse_crosscoder_reconstruction = (
        evaluate_sparse_crosscoder_reconstruction
        or solutions.evaluate_sparse_crosscoder_reconstruction
    )
    make_planted_crosscoder_data = (
        make_planted_crosscoder_data or solutions.make_planted_crosscoder_data
    )

    data = make_planted_crosscoder_data(n_examples=512, d_model=12)
    trained = train_sparse_crosscoder(data, steps=120, log_every=40)
    first_mse = trained.history[0]["heldout_reconstruction_mse"]
    final_mse = trained.history[-1]["heldout_reconstruction_mse"]
    report = evaluate_sparse_crosscoder_reconstruction(trained.model, data)
    assert final_mse < 0.02 * first_mse, (
        "Training should reduce held-out reconstruction error by at least fifty-fold."
    )
    assert report.beats_zero_baseline, (
        "The learned crosscoder must reconstruct held-out activations better than predicting zero."
    )
    assert report.heldout_reconstruction_mse < 0.002, (
        "Held-out reconstruction should recover the low-noise planted activation pair."
    )
    print("All tests in `test_train_sparse_crosscoder_reduces_heldout_reconstruction` passed!")


def test_latent_ownership_table_classifies_specificity(
    latent_ownership_table: Callable | None = None,
):
    solutions = _solutions()
    latent_ownership_table = latent_ownership_table or solutions.latent_ownership_table
    forward = solutions.SparseCrosscoderForward(
        model_a_feature_acts=t.tensor(
            [
                [1.0, 2.0, 0.1],
                [1.2, 2.5, 0.0],
                [0.8, 2.2, 0.2],
            ]
        ),
        model_b_feature_acts=t.tensor(
            [
                [1.1, 0.2, 3.0],
                [1.0, 0.1, 3.5],
                [0.9, 0.0, 4.0],
            ]
        ),
        reconstructed_model_a=t.zeros(3, 2),
        reconstructed_model_b=t.zeros(3, 2),
    )
    rows = latent_ownership_table(forward, shared_threshold=0.15)
    assert [row["predicted_owner"] for row in rows] == ["shared", "model_a", "model_b"], (
        "Ownership classification should distinguish shared, model-A, and model-B activation patterns."
    )
    print("All tests in `test_latent_ownership_table_classifies_specificity` passed!")


def test_feature_matching_recovers_planted_owners(
    train_sparse_crosscoder: Callable | None = None,
    match_learned_to_planted_features: Callable | None = None,
    make_planted_crosscoder_data: Callable | None = None,
):
    solutions = _solutions()
    train_sparse_crosscoder = train_sparse_crosscoder or solutions.train_sparse_crosscoder
    match_learned_to_planted_features = (
        match_learned_to_planted_features or solutions.match_learned_to_planted_features
    )
    make_planted_crosscoder_data = (
        make_planted_crosscoder_data or solutions.make_planted_crosscoder_data
    )

    data = make_planted_crosscoder_data(n_examples=512, d_model=12)
    trained = train_sparse_crosscoder(data, steps=120, log_every=60)
    match = match_learned_to_planted_features(trained.model, data)
    assert match.ownership_accuracy == 1.0, (
        "Every planted feature should be assigned to its correct shared or model-specific owner."
    )
    assert match.min_correlation > 0.9, (
        "Each matched learned latent should track its planted feature with correlation above 0.9."
    )
    assert match.predicted_owners == match.true_owners, (
        "Feature matching should preserve the complete planted ownership pattern."
    )
    print("All tests in `test_feature_matching_recovers_planted_owners` passed!")


def test_behavior_baselines_include_real_controls(
    train_sparse_crosscoder: Callable | None = None,
    match_learned_to_planted_features: Callable | None = None,
    behavior_baseline_table: Callable | None = None,
    make_planted_crosscoder_data: Callable | None = None,
):
    solutions = _solutions()
    train_sparse_crosscoder = train_sparse_crosscoder or solutions.train_sparse_crosscoder
    match_learned_to_planted_features = (
        match_learned_to_planted_features or solutions.match_learned_to_planted_features
    )
    behavior_baseline_table = behavior_baseline_table or solutions.behavior_baseline_table
    make_planted_crosscoder_data = (
        make_planted_crosscoder_data or solutions.make_planted_crosscoder_data
    )

    data = make_planted_crosscoder_data(n_examples=512, d_model=12)
    trained = train_sparse_crosscoder(data, steps=120, log_every=60)
    match = match_learned_to_planted_features(trained.model, data)
    rows = behavior_baseline_table(trained.model, data, match)
    by_method = {row["method"]: row for row in rows}
    learned_auc = by_method["learned sparse crosscoder target latent"]["behavior_auc"]
    shuffled_auc = by_method["label-shuffled target latent"]["behavior_auc"]
    shared_only_auc = by_method["shared-only learned latents"]["behavior_auc"]
    assert learned_auc > 0.95, (
        "The learned model-B-specific latent should predict the planted behavior on held-out examples."
    )
    assert shuffled_auc < learned_auc - 0.25, (
        "Shuffling behavior labels should visibly destroy the learned latent's predictive advantage."
    )
    assert shared_only_auc < learned_auc - 0.15, (
        "Shared latents alone should not explain the model-B-specific planted behavior."
    )
    assert (
        by_method["zero reconstruction"]["heldout_reconstruction_mse"]
        > by_method["learned sparse crosscoder target latent"]["heldout_reconstruction_mse"]
    ), "The learned crosscoder should beat the zero-reconstruction baseline on held-out activations."
    print("All tests in `test_behavior_baselines_include_real_controls` passed!")


def test_targeted_ablation_beats_same_norm_and_orthogonal_controls(
    train_sparse_crosscoder: Callable | None = None,
    match_learned_to_planted_features: Callable | None = None,
    crosscoder_intervention_report: Callable | None = None,
    make_planted_crosscoder_data: Callable | None = None,
):
    solutions = _solutions()
    train_sparse_crosscoder = train_sparse_crosscoder or solutions.train_sparse_crosscoder
    match_learned_to_planted_features = (
        match_learned_to_planted_features or solutions.match_learned_to_planted_features
    )
    crosscoder_intervention_report = (
        crosscoder_intervention_report or solutions.crosscoder_intervention_report
    )
    make_planted_crosscoder_data = (
        make_planted_crosscoder_data or solutions.make_planted_crosscoder_data
    )

    data = make_planted_crosscoder_data(n_examples=512, d_model=12)
    trained = train_sparse_crosscoder(data, steps=120, log_every=60)
    match = match_learned_to_planted_features(trained.model, data)
    report = crosscoder_intervention_report(trained.model, data, match)
    assert report.passes_controls, (
        "The target latent ablation should pass both same-norm random and orthogonal controls."
    )
    assert report.target_reduction > report.same_norm_random_reduction + 0.25, (
        "Targeted ablation should reduce the planted behavior substantially more than a same-norm random direction."
    )
    assert report.target_reduction > report.orthogonal_reduction + 0.25, (
        "Targeted ablation should reduce the planted behavior substantially more than an orthogonal direction."
    )
    print(
        "All tests in `test_targeted_ablation_beats_same_norm_and_orthogonal_controls` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["contract_passed"], "The learned crosscoder signature result should pass."
    assert result["dataset"]["n_heldout"] >= 128, (
        "The signature result needs at least 128 held-out examples."
    )
    assert result["reconstruction"]["heldout_reconstruction_mse"] < 3e-4, (
        "The signature crosscoder should reconstruct the planted held-out pair nearly exactly."
    )
    assert result["feature_match"]["ownership_accuracy"] == 1.0, (
        "The signature result should recover every planted feature owner."
    )
    assert result["feature_match"]["min_correlation"] > 0.9, (
        "The weakest matched signature latent should still correlate strongly with ground truth."
    )
    assert result["intervention"]["passes_controls"], (
        "The signature intervention must beat its matched random and orthogonal controls."
    )
    learned_auc = next(
        row["behavior_auc"]
        for row in result["baselines"]
        if row["method"] == "learned sparse crosscoder target latent"
    )
    shuffled_auc = next(
        row["behavior_auc"]
        for row in result["baselines"]
        if row["method"] == "label-shuffled target latent"
    )
    assert learned_auc > 0.93, (
        "The signature target latent should predict the held-out behavior with high AUC."
    )
    assert shuffled_auc < learned_auc - 0.2, (
        "The label-shuffled baseline should lose at least 0.2 AUC versus the learned target latent."
    )
    print("All tests in `test_notebook_contract` passed!")
