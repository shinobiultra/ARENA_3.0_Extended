import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.sae_variants import (
        best_feature_auc,
        decode_features,
        density_is_nondegenerate,
        dictionary_recovery_report,
        gated_encode,
        jumprelu_encode,
        make_toy_superposition_batch,
        relu_l1_encode,
        sae_variant_metrics,
        topk_encode,
    )


def test_sae_encoders_have_expected_sparse_patterns():
    pre_acts = t.tensor([[1.0, -1.0, 0.2, 3.0]])
    gate_logits = t.tensor([[1.0, 1.0, -1.0, 1.0]])

    relu_l1 = relu_l1_encode(pre_acts, l1_coefficient=0.5)
    topk = topk_encode(pre_acts, k=2)
    gated = gated_encode(pre_acts, gate_logits)
    jumprelu = jumprelu_encode(pre_acts, threshold=0.5)

    assert t.equal(relu_l1, t.tensor([[0.5, 0.0, 0.0, 2.5]]))
    assert t.equal(topk, t.tensor([[1.0, 0.0, 0.0, 3.0]]))
    assert t.equal(gated, t.tensor([[1.0, 0.0, 0.0, 3.0]]))
    assert t.equal(jumprelu, t.tensor([[1.0, 0.0, 0.0, 3.0]]))


def test_decode_features_and_variant_metrics_exact_reconstruction():
    feature_acts = t.tensor([[1.0, 0.0], [0.0, 2.0]])
    decoder = t.eye(2)
    activations = decode_features(feature_acts, decoder)

    metrics = sae_variant_metrics(
        "identity",
        activations=activations,
        reconstructed_activations=activations,
        feature_acts=feature_acts,
    )

    assert metrics.reconstruction_mse == 0.0
    assert metrics.l0 == 1.0
    assert metrics.dead_feature_fraction == 0.0


def test_toy_superposition_batch_has_sparse_features():
    batch = make_toy_superposition_batch(
        batch=32,
        n_features=6,
        d_model=3,
        feature_probability=0.25,
        seed=0,
    )

    assert batch.feature_acts.shape == (32, 6)
    assert batch.activations.shape == (32, 3)
    assert batch.dictionary.shape == (6, 3)
    assert density_is_nondegenerate(batch.feature_acts)


def test_dictionary_recovery_detects_missing_and_duplicate_features():
    true_dictionary = t.eye(3)
    learned_decoder = t.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    report = dictionary_recovery_report(learned_decoder, true_dictionary, threshold=0.9)

    assert report.recovered_fraction == pytest.approx(2 / 3)
    assert report.duplicate_fraction == pytest.approx(1 / 3)
    assert report.best_learned_for_true.tolist() == [0, 2, 0]


def test_best_feature_auc_finds_predictive_feature():
    feature_acts = t.tensor(
        [
            [0.1, 0.0],
            [0.2, 0.0],
            [0.1, 1.0],
            [0.2, 2.0],
        ]
    )
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)

    report = best_feature_auc(feature_acts, labels)

    assert report.feature_id == 1
    assert report.auc == 1.0
