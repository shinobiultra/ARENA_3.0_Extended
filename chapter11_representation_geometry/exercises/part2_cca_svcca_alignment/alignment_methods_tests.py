"""Independent mathematical checks; pass the learner's own function namespace."""

import pytest
import torch as t
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment import (
    alignment_methods as m,
)


def test_whitening_covariance_rank_and_training_only_fit(methods=m):
    gen = t.Generator().manual_seed(801)
    x = t.randn(80, 4, generator=gen, dtype=t.float64)
    x[:, 3] = x[:, 0] + x[:, 1]
    fitted = methods.svd_whitener(x)
    q = (x - fitted["mean"]) @ fitted["whitening"]
    assert fitted["retained_rank"] == 3, (
        "A duplicated linear direction must not create another canonical coordinate."
    )
    t.testing.assert_close(q.T @ q / (len(x) - 1), t.eye(3, dtype=x.dtype), atol=1e-10, rtol=1e-10)
    original = fitted["mean"].clone()
    test = x + 100
    _ = (test - fitted["mean"]) @ fitted["whitening"]
    t.testing.assert_close(fitted["mean"], original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="constant"):
        methods.svd_whitener(t.ones(8, 3))
    print(
        "Whitening checks passed: identity covariance, numerical rank and training-only centering agree."
    )


def known_paired_views():
    gen = t.Generator().manual_seed(802)
    raw = t.randn(96, 6, generator=gen, dtype=t.float64)
    raw = raw - raw.mean(0)
    q = t.linalg.qr(raw, mode="reduced").Q * (len(raw) - 1) ** 0.5
    rho = t.tensor([0.95, 0.6, 0.1], dtype=t.float64)
    a = t.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]], dtype=t.float64)
    b = t.tensor([[1.0, 0.0, 2.0], [1.0, 2.0, 0.0], [0.0, 1.0, 3.0]], dtype=t.float64)
    x = q[:, :3] @ a + 7
    y = (q[:, :3] * rho + q[:, 3:] * t.sqrt(1 - rho.square())) @ b - 13
    return x, y, rho


def test_canonical_correlations_match_known_oracle_under_affine_maps(methods=m):
    x, y, rho = known_paired_views()
    fit = methods.fit_cca(x, y)
    t.testing.assert_close(fit["correlations"], rho, atol=1e-10, rtol=1e-10)
    actual = methods.heldout_correlations(x, y, fit)
    t.testing.assert_close(actual, rho, atol=1e-10, rtol=1e-10)
    left = (x - fit["left"]["mean"]) @ fit["x_map"]
    right = (y - fit["right"]["mean"]) @ fit["y_map"]
    t.testing.assert_close(left.T @ right / (len(x) - 1), t.diag(rho), atol=1e-10, rtol=1e-10)
    print("CCA checks passed: known correlations survive invertible affine coordinate changes.")


def test_heldout_direction_is_fixed_and_negative_correlation_is_visible(methods=m):
    x, y, _ = known_paired_views()
    fit = methods.fit_cca(x, y)
    expected = -fit["correlations"]
    inverted = 2 * fit["right"]["mean"] - y
    t.testing.assert_close(
        methods.heldout_correlations(x, inverted, fit), expected, atol=1e-10, rtol=1e-10
    )
    undefined = methods.paired_column_correlations(t.ones(8, 2), t.ones(8, 2))
    assert t.isnan(undefined).all(), (
        "Constant coordinates have undefined correlation, not a perfect match."
    )
    print(
        "Held-out checks passed: fitted directions stay fixed and negative or undefined correlations remain visible."
    )


def test_independent_high_width_views_overfit_training_and_fail_heldout(methods=m):
    gen = t.Generator().manual_seed(803)
    x = t.randn(24, 40, generator=gen, dtype=t.float64)
    y = t.randn(24, 45, generator=gen, dtype=t.float64)
    fit = methods.fit_cca(x, y, components=10)
    t.testing.assert_close(fit["correlations"], t.ones(10, dtype=x.dtype), atol=1e-10, rtol=1e-10)
    test_x = t.randn(1000, 40, generator=gen, dtype=t.float64)
    test_y = t.randn(1000, 45, generator=gen, dtype=t.float64)
    assert methods.heldout_correlations(test_x, test_y, fit).abs().max() < 0.12, (
        "Independent new rows must expose the perfect-training-correlation illusion."
    )
    print(
        "Overfitting control passed: independent views look perfect on training rows and fail on new rows."
    )


def test_svcca_variance_cutoff_can_remove_a_low_variance_feature(methods=m):
    x, y, _ = known_paired_views()
    x = x - x.mean(0)
    orth = t.linalg.qr(x, mode="reduced").Q
    scaled = orth * t.tensor([10.0, 2.0, 0.01], dtype=x.dtype)
    fitted = methods.svd_whitener(scaled, variance_fraction=0.99)
    assert fitted["retained_rank"] == 2, (
        "The 99% variance cutoff should remove the tiny third direction."
    )
    t.testing.assert_close(
        fitted["basis"].T @ t.tensor([0.0, 0.0, 1.0], dtype=x.dtype),
        t.zeros(2, dtype=x.dtype),
        atol=1e-10,
        rtol=0,
    )
    assert methods.svd_whitener(scaled, max_rank=1)["retained_rank"] == 1, (
        "An explicit rank cap must apply after variance selection."
    )
    print(
        "SVCCA cutoff checks passed: a low-variance feature can be discarded even if it matters to a task."
    )


def test_procrustes_preserves_lengths_and_cannot_undo_anisotropic_scaling(methods=m):
    gen = t.Generator().manual_seed(804)
    x = t.randn(80, 3, generator=gen, dtype=t.float64)
    rotation = t.linalg.qr(t.randn(3, 3, generator=gen, dtype=t.float64)).Q
    y = x @ rotation + 11
    fit = methods.fit_procrustes(x, y)
    predicted = (x - fit["mean_x"]) @ fit["rotation"] + fit["mean_y"]
    t.testing.assert_close(predicted, y, atol=1e-10, rtol=1e-10)
    stretched = x * t.tensor([1.0, 3.0, 8.0], dtype=x.dtype)
    wrong = methods.fit_procrustes(x, stretched)
    error = (
        ((x - wrong["mean_x"]) @ wrong["rotation"] + wrong["mean_y"] - stretched).square().mean()
    )
    assert error > 1, "An orthogonal alignment cannot silently rescale one axis."
    print(
        "Procrustes checks passed: rigid alignment works, anisotropic scaling remains a visible mismatch."
    )


def test_canonical_transport_equals_exact_linear_map_and_zero_rank_noop(methods=m):
    gen = t.Generator().manual_seed(805)
    x = t.randn(100, 3, generator=gen, dtype=t.float64)
    mapping = t.tensor(
        [[2.0, 0.0, 1.0, 0.0], [0.0, 3.0, 0.0, 0.0], [1.0, 0.0, 2.0, 0.0]], dtype=x.dtype
    )
    y = x @ mapping + t.tensor([1.0, -2.0, 3.0, 19.0], dtype=x.dtype)
    fit = methods.fit_cca(x, y)
    delta = t.randn(8, 3, generator=gen, dtype=x.dtype)
    transported = methods.cca_difference_transport(delta, fit)
    t.testing.assert_close(transported, delta @ mapping, atol=1e-10, rtol=1e-10)
    t.testing.assert_close(transported[:, -1], t.zeros(8, dtype=x.dtype), atol=1e-10, rtol=0)
    t.testing.assert_close(
        methods.cca_difference_transport(delta, fit, 0),
        t.zeros(8, 4, dtype=x.dtype),
        rtol=0,
        atol=0,
    )
    print(
        "Transport checks passed: exact differences transfer and an unrelated constant coordinate is preserved."
    )


def test_ridge_transport_matches_primal_oracle_and_unpenalized_intercept(methods=m):
    x, y, _ = known_paired_views()
    ridge = 0.2
    fit = methods.fit_ridge_transport(x, y, ridge)
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    expected = t.linalg.solve(xc.T @ xc + ridge * t.eye(x.shape[1], dtype=x.dtype), xc.T @ yc)
    t.testing.assert_close(fit["weights"], expected, rtol=1e-10, atol=1e-10)
    t.testing.assert_close(fit["mean_y"], y.mean(0), rtol=0, atol=0)
    print(
        "Ridge baseline checks passed: dual and primal predictions agree and the intercept remains unpenalized."
    )


def test_distribution_kl_uses_full_vocabulary_and_correct_orientation(methods=m):
    p = t.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]], dtype=t.float64)
    q = t.tensor([[0.5, 0.2, 0.3], [0.2, 0.2, 0.6]], dtype=t.float64)
    expected = (p * t.log(p / q)).sum(-1)
    t.testing.assert_close(
        methods.distribution_kl(p.log(), q.log()), expected, rtol=1e-12, atol=1e-12
    )
    t.testing.assert_close(
        methods.distribution_kl(p.log(), p.log()), t.zeros(2, dtype=p.dtype), rtol=0, atol=0
    )
    assert not t.allclose(methods.distribution_kl(q.log(), p.log()), expected), (
        "KL orientation must follow the chosen donor target distribution."
    )
    print(
        "Distribution checks passed: KL uses all tokens and the intended target-to-actual direction."
    )
