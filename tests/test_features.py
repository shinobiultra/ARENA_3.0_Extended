import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.features import (
        ablate_features,
        apply_decoder_steering,
        compute_sae_reconstruction_metrics,
        direct_logit_attribution,
        feature_detection_report,
        feature_density,
        l0,
        loss_recovered,
        roc_auc_binary,
        steering_comparison_report,
        top_activating_examples,
    )


def test_feature_density_and_l0():
    feature_acts = t.tensor(
        [
            [[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        ]
    )

    densities = feature_density(feature_acts)

    assert t.allclose(densities, t.tensor([0.5, 0.25, 0.5]))
    assert l0(feature_acts) == pytest.approx(1.25)


def test_sae_reconstruction_metrics():
    activations = t.ones(2, 3, 4)
    reconstructed = activations + 0.5
    feature_acts = t.tensor([[[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]])
    reference_logits = t.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    reconstructed_logits = reference_logits.clone()

    metrics = compute_sae_reconstruction_metrics(
        activations=activations,
        reconstructed_activations=reconstructed,
        feature_acts=feature_acts,
        reference_logits=reference_logits,
        reconstructed_logits=reconstructed_logits,
        clean_loss=1.0,
        reconstructed_loss=1.5,
        zero_ablation_loss=3.0,
    )

    assert metrics.l0 == pytest.approx(2 / 3)
    assert metrics.reconstruction_mse == pytest.approx(0.25)
    assert metrics.reconstruction_kl == pytest.approx(0.0)
    assert metrics.loss_recovered == pytest.approx(0.75)


def test_loss_recovered_rejects_degenerate_denominator():
    with pytest.raises(ValueError):
        loss_recovered(clean_loss=1.0, reconstructed_loss=1.0, zero_ablation_loss=1.0)


def test_direct_logit_attribution_selects_tokens():
    decoder_vectors = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    effects = direct_logit_attribution(decoder_vectors, unembedding, token_ids=[0, 2])

    assert t.equal(effects, t.tensor([[1.0, 3.0], [4.0, 6.0]]))


def test_feature_detection_report_separates_positives():
    scores = t.tensor([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    labels = t.tensor([1, 1, 1, 0, 0, 0], dtype=t.bool)

    report = feature_detection_report(scores, labels)

    assert report.auc == pytest.approx(1.0)
    assert report.separation > 0
    assert report.threshold_accuracy == pytest.approx(1.0)


def test_roc_auc_binary_averages_tied_ranks():
    labels = t.tensor([1, 0, 1, 0], dtype=t.bool)

    assert roc_auc_binary(t.zeros(4), labels) == pytest.approx(0.5)
    assert roc_auc_binary(t.tensor([0.0, 0.0, 1.0, 0.0]), labels) == pytest.approx(0.75)


def test_top_activating_examples_returns_flattened_indices():
    feature_acts = t.tensor([[[1.0, 0.0], [3.0, 0.0]], [[2.0, 0.0], [4.0, 0.0]]])

    indices, values = top_activating_examples(feature_acts, feature_id=0, k=2)

    assert t.equal(indices, t.tensor([3, 1]))
    assert t.equal(values, t.tensor([4.0, 3.0]))


def test_ablate_features_zero_and_mean():
    feature_acts = t.tensor([[[1.0, 10.0], [3.0, 20.0]]])

    zeroed = ablate_features(feature_acts, [0], replacement="zero")
    meaned = ablate_features(feature_acts, [0], replacement="mean")

    assert t.equal(zeroed[..., 0], t.zeros_like(zeroed[..., 0]))
    assert t.equal(meaned[..., 0], t.tensor([[2.0, 2.0]]))
    assert t.equal(meaned[..., 1], feature_acts[..., 1])


def test_apply_decoder_steering_last_position():
    activations = t.zeros(2, 3, 4)
    decoder_vectors = t.eye(4)

    steered = apply_decoder_steering(activations, decoder_vectors, [1, 3], [2.0, -1.0])

    assert t.equal(steered[:, :-1, :], t.zeros_like(steered[:, :-1, :]))
    assert t.equal(steered[:, -1, :], t.tensor([[0.0, 2.0, 0.0, -1.0]]).expand(2, -1))


def test_steering_comparison_report_requires_random_control():
    baseline = t.tensor([0.1, 0.2, 0.3])
    steered = t.tensor([0.5, 0.6, 0.7])
    random = t.tensor([0.2, 0.2, 0.3])

    report = steering_comparison_report(baseline, steered, random)

    assert report.steered_delta > report.random_delta
    assert report.passes_control
