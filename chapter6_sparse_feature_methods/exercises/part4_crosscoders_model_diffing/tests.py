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
    assert data.feature_owners == ("shared", "shared", "model_a", "model_b")
    assert t.count_nonzero(data.true_decoder_b[2]).item() == 0
    assert t.count_nonzero(data.true_decoder_a[3]).item() == 0
    assert 0.3 <= data.behavior_labels.float().mean().item() <= 0.7
    assert data.train_idx.numel() + data.heldout_idx.numel() == 128
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
    assert output.model_a_feature_acts.shape == (16, 6)
    assert output.model_b_feature_acts.shape == (16, 6)
    assert output.reconstructed_model_a.shape == (16, 12)
    assert output.reconstructed_model_b.shape == (16, 12)
    assert output.model_a_feature_acts.min().item() >= 0
    assert output.model_b_feature_acts.min().item() >= 0

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
    assert loss.reconstruction_loss.item() > 0
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
    assert final_mse < 0.02 * first_mse
    assert report.beats_zero_baseline
    assert report.heldout_reconstruction_mse < 0.002
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
    assert [row["predicted_owner"] for row in rows] == ["shared", "model_a", "model_b"]
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
    assert match.ownership_accuracy == 1.0
    assert match.min_correlation > 0.9
    assert match.predicted_owners == match.true_owners
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
    assert learned_auc > 0.95
    assert shuffled_auc < learned_auc - 0.25
    assert shared_only_auc < learned_auc - 0.15
    assert (
        by_method["zero reconstruction"]["heldout_reconstruction_mse"]
        > by_method["learned sparse crosscoder target latent"]["heldout_reconstruction_mse"]
    )
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
    assert report.passes_controls
    assert report.target_reduction > report.same_norm_random_reduction + 0.25
    assert report.target_reduction > report.orthogonal_reduction + 0.25
    print(
        "All tests in `test_targeted_ablation_beats_same_norm_and_orthogonal_controls` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["contract_passed"], "The learned crosscoder signature result should pass."
    assert result["dataset"]["n_heldout"] >= 128
    assert result["reconstruction"]["heldout_reconstruction_mse"] < 3e-4
    assert result["feature_match"]["ownership_accuracy"] == 1.0
    assert result["feature_match"]["min_correlation"] > 0.9
    assert result["intervention"]["passes_controls"]
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
    assert learned_auc > 0.93
    assert shuffled_auc < learned_auc - 0.2
    print("All tests in `test_notebook_contract` passed!")
