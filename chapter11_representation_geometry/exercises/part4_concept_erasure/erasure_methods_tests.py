"""Immediate numerical checks with independent structural and optimization oracles."""
import itertools
import math
from types import SimpleNamespace

import numpy as np
import torch as t
import torch.nn.functional as F
from chapter11_representation_geometry.exercises.part4_concept_erasure import erasure_methods as reference


def four_points():
    u, z = t.tensor(list(itertools.product((-2., 2.), (-1., 1.))), dtype=t.float64).T
    return t.stack((u, u + z), 1), z[:, None]


def test_centered_moments(m=reference):
    x, z = four_points()
    stats = m.centered_statistics(x + t.tensor([3., -7.]), z)
    t.testing.assert_close(stats['mean_x'], t.tensor([3., -7.], dtype=x.dtype))
    t.testing.assert_close(stats['covariance'], t.tensor([[4., 4.], [4., 5.]], dtype=x.dtype))
    t.testing.assert_close(stats['cross_covariance'], t.tensor([[0.], [1.]], dtype=x.dtype))
    assert stats['mean_z'].item() == 0, 'Center the paired concept labels too.'
    print('Centered mean, covariance and cross-covariance checks passed!')


def test_psd_support_and_rejection(m=reference):
    covariance = t.diag(t.tensor([9., 1., 0.], dtype=t.float64))
    fitted = m.psd_whitener(covariance)
    t.testing.assert_close(fitted['whiten'], t.diag(t.tensor([1 / 3, 1., 0.], dtype=t.float64)))
    t.testing.assert_close(fitted['unwhiten'], t.diag(t.tensor([3., 1., 0.], dtype=t.float64)))
    assert fitted['rank'] == 2, 'A constant feature has no empirical variance to whiten.'
    t.testing.assert_close(fitted['whiten'] @ covariance @ fitted['whiten'], t.diag(t.tensor([1., 1., 0.], dtype=t.float64)))
    for invalid in (t.tensor([[1., 1.], [0., 1.]], dtype=t.float64),
                    t.diag(t.tensor([1., -0.1], dtype=t.float64))):
        try:
            m.psd_whitener(invalid)
        except ValueError:
            continue
        raise AssertionError('Reject nonsymmetric and materially negative covariance matrices.')
    print('Whitening, numerical support and invalid covariance checks passed!')


def test_four_point_optimal_edit(m=reference):
    x, z = four_points()
    leace, orth = m.fit_leace(x, z), m.fit_orthogonal_eraser(x, z)
    p = t.eye(2, dtype=x.dtype) - leace['left'] @ leace['right']
    t.testing.assert_close(p, t.tensor([[1., 0.], [1., 0.]], dtype=x.dtype), atol=1e-12, rtol=1e-12)
    edited = m.apply_eraser(x, leace)
    t.testing.assert_close(edited, t.stack((x[:, 0], x[:, 0]), 1))
    assert math.isclose(float((edited - x).square().sum(1).mean()), 1.), 'Only the independent concept contribution must move.'
    assert math.isclose(float((m.apply_eraser(x, orth) - x).square().sum(1).mean()), 5.), 'Orthogonal removal discards the predictable side contribution too.'
    assert not t.allclose(p, p.T), 'The least-distorting eraser is oblique in these coordinates.'
    t.testing.assert_close(p @ p, p, atol=1e-12, rtol=1e-12)
    print('Four-point oblique erasure and independent distortion checks passed!')


def test_affine_mean_preservation_and_identity(m=reference):
    x, z = four_points()
    shift = t.tensor([7., -3.], dtype=x.dtype)
    centered = m.apply_eraser(x, m.fit_leace(x, z))
    eraser = m.fit_leace(x + shift, z)
    result = m.apply_eraser(x + shift, eraser)
    t.testing.assert_close(result, centered + shift)
    t.testing.assert_close(result.mean(0), (x + shift).mean(0))
    t.testing.assert_close(m.apply_eraser(result, eraser), result, atol=1e-12, rtol=1e-12)
    assert t.equal(x, four_points()[0]), 'Do not modify the caller’s examples while applying an eraser.'
    zero = m.fit_leace(x, t.zeros_like(z))
    assert zero['rank'] == 0, 'Zero cross-covariance needs a rank-zero edit.'
    t.testing.assert_close(m.apply_eraser(x, zero), x, atol=0, rtol=0)
    constant = t.ones(8, 3, dtype=t.float64)
    labels = t.arange(8, dtype=t.float64)[:, None]
    t.testing.assert_close(m.apply_eraser(constant, m.fit_leace(constant, labels)), constant, atol=0, rtol=0)
    print('Affine mean, repeated erasure, unchanged input and zero-rank checks passed!')


def test_independent_kkt_optimality(m=reference):
    rng = np.random.default_rng(412)
    z = np.tile([-1., 1.], 64)[:, None]
    x = rng.normal(size=(128, 5)) @ rng.normal(size=(5, 5)) + z @ rng.normal(size=(1, 5))
    x -= x.mean(0)
    covariance, cross = x.T @ x / len(x), x.T @ z / len(x)
    # Independent constrained quadratic solve for each output coordinate.
    kkt = np.block([[covariance, cross], [cross.T, np.zeros((1, 1))]])
    rhs = np.vstack([covariance, np.zeros((1, 5))])
    oracle = np.linalg.solve(kkt, rhs)[:5].T
    xt, zt = t.tensor(x), t.tensor(z)
    fitted = m.fit_leace(xt, zt)
    actual = t.eye(5, dtype=xt.dtype) - fitted['left'] @ fitted['right']
    np.testing.assert_allclose(actual.numpy(), oracle, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(oracle @ cross, 0, atol=1e-12)
    assert np.linalg.norm(oracle - np.eye(5)) > 0.1, 'The constraint must require an actual edit.'
    print('Independent KKT constrained-least-squares oracle checks passed!')


def test_singular_covariance_and_randomized_invariants(m=reference):
    for seed in range(6):
        generator = t.Generator().manual_seed(seed)
        latent = t.randn(90, 5, generator=generator, dtype=t.float64)
        labels = t.arange(90) % 3
        z = F.one_hot(labels, 3).double()
        x = (latent + z @ t.randn(3, 5, generator=generator, dtype=t.float64)) @ t.randn(5, 9, generator=generator, dtype=t.float64)
        x += t.randn(9, generator=generator, dtype=t.float64)
        fit = m.fit_leace(x, z)
        edited = m.apply_eraser(x, fit)
        t.testing.assert_close(m.centered_statistics(edited, z)['cross_covariance'], t.zeros(9, 3, dtype=x.dtype), atol=2e-11, rtol=0)
        t.testing.assert_close(edited.mean(0), x.mean(0), atol=2e-11, rtol=0)
        t.testing.assert_close(m.apply_eraser(edited, fit), edited, atol=2e-11, rtol=0)
        assert fit['covariance_rank'] == 5, 'Track the empirical support rather than inverting null directions.'
        assert fit['rank'] <= 2, 'Centered three-class labels have at most two independent contrasts.'
    print('Six singular-covariance affine and erasure-invariant cases passed!')


def test_probe_removal_requires_refitting(m=reference):
    x, z = four_points()
    labels = (z[:, 0] > 0).long()
    coefficient = t.tensor([-1., 1.], dtype=x.dtype)
    q = coefficient / coefficient.norm()
    removed = x - (x @ q)[:, None] * q
    old = dict(coefficient=coefficient, intercept=t.tensor(0., dtype=x.dtype))
    old_loss = m.evaluate_binary_probe(removed, labels, old)['log_loss']
    new = m.fit_binary_probe(removed, labels, l2=1e-4)
    new_loss = m.evaluate_binary_probe(removed, labels, new)['log_loss']
    assert math.isclose(old_loss, math.log(2)), 'Removing this normal must disable the original linear readout.'
    assert new_loss < math.log(2) - 0.01, 'Refit a new probe: one disabled classifier does not establish erasure.'
    erased = m.apply_eraser(x, m.fit_leace(x, z))
    fresh = m.fit_binary_probe(erased, labels, l2=1e-4)
    assert abs(m.evaluate_binary_probe(erased, labels, fresh)['log_loss'] - math.log(2)) < 1e-10, 'Equal centroids should make the constant logistic predictor optimal here.'
    print('Disabled-old-probe versus independently refitted probe checks passed!')


def test_nonlinear_information_and_loss_boundary(m=reference):
    x = t.tensor([[-2.], [2.], [-1.], [1.]], dtype=t.float64)
    labels = t.tensor([1, 1, 0, 0])
    z = F.one_hot(labels, 2).double()
    erased = m.apply_eraser(x, m.fit_leace(x, z))
    t.testing.assert_close(erased, x, atol=0, rtol=0)
    probe = m.fit_binary_probe(erased, labels)
    assert abs(m.evaluate_binary_probe(erased, labels, probe)['log_loss'] - math.log(2)) < 1e-10, 'The convex linear loss is minimized by a constant here.'
    nonlinear = (erased[:, 0].square() > 2.5)
    assert t.equal(nonlinear, labels.bool()), 'The label is still perfectly available nonlinearly.'
    threshold = (erased[:, 0] > 1.5)
    assert float((threshold == labels.bool()).double().mean()) == 0.75, 'A 0-1 threshold can beat chance despite convex-loss guardedness.'
    print('Nonlinear information and convex-loss-versus-accuracy counterexample passed!')


def test_predeclared_data_splits():
    from chapter11_representation_geometry.exercises.part4_concept_erasure import erasure_data as data
    splits = {split: data.prompt_rows(split) for split in ('train', 'probe_train', 'development', 'final')}
    assert [len(rows) for rows in splits.values()] == [512, 320, 160, 288], 'Every named pair, template and role must be retained.'
    sets = [set(row['prompt'] for row in rows) for rows in splits.values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:]), 'Prompt families must not overlap across fitting and evaluation.'
    for task in ('role', 'object'):
        for split in splits:
            rows = data.prompt_rows(split, task)
            assert len({row['id'] for row in rows}) == len(rows), 'Keep a unique identity for every declared example.'
            assert sum(row['role'] for row in rows) * 2 == len(rows), 'Both roles must be balanced without result-based filtering.'
            for row in rows:
                assert row['prompt'][slice(*row['span'])] == (row['a'], row['b'])[row['role']], 'The intervention position must identify the repeated subject.'
                assert row['answer'] == ((row['b'], row['a'])[row['role']] if task == 'role' else row['object']), 'Derive answers from task structure.'
    print('Disjoint prompts, all declared rows, balanced roles and exact task labels passed!')


def test_random_control_matches_each_edit_norm(m=reference):
    x, z = four_points()
    eraser = m.fit_leace(x, z)
    q = t.tensor([3., 4.], dtype=x.dtype) / 5
    result = m.matched_random_edit(x, x.mean(0), q, eraser)
    t.testing.assert_close((result - x).norm(dim=-1), (m.apply_eraser(x, eraser) - x).norm(dim=-1), atol=1e-12, rtol=1e-12)
    perpendicular = t.tensor([-4., 3.], dtype=x.dtype) / 5
    t.testing.assert_close((result - x) @ perpendicular, t.zeros(len(x), dtype=x.dtype), atol=1e-12, rtol=0)
    identity = m.fit_leace(x, t.zeros_like(z))
    t.testing.assert_close(m.matched_random_edit(x, x.mean(0), q, identity), x, atol=0, rtol=0)
    assert not t.allclose(result, m.apply_eraser(x, eraser)), 'Equal magnitudes must not silently reuse the target direction.'
    print('Per-example norm, random direction and zero-edit control checks passed!')


def test_actual_vocabulary_outcome_metrics(m=reference):
    p = t.tensor([[.6, .3, .1], [.1, .6, .3]], dtype=t.float64)
    q = t.tensor([[.7, .2, .1], [.1, .2, .7]], dtype=t.float64)
    metrics = m.compare_model_outputs(p.log(), q.log(), t.tensor([0, 2]))
    expected_kl = sum(float(qi) * math.log(float(qi / pi)) for row_p, row_q in zip(p, q) for pi, qi in zip(row_p, row_q)) / 2
    assert metrics['rows'] == 2 and metrics['correct'] == 1, 'Evaluate every row against its actual requested answer.'
    assert metrics['native_correct_count'] == 2 and metrics['preserved_native_correct'] == 1, 'Keep the native-correct denominator explicit.'
    assert math.isclose(metrics['answer_loss'], -(math.log(.6) + math.log(.3)) / 2), 'Target loss needs the complete vocabulary normalizer.'
    assert math.isclose(metrics['answer_probability'], .45), 'Average actual target probabilities.'
    assert math.isclose(metrics['kl_from_native'], expected_kl), 'KL is measured from the original distribution to the intervened distribution.'
    assert metrics['changed_winner_fraction'] == .5, 'Changed output tokens need not all be incorrect.'
    identity = m.compare_model_outputs(q.log(), q.log(), t.tensor([0, 2]))
    assert abs(identity['kl_from_native']) < 1e-12, 'No intervention must have zero native-distribution KL.'
    print('Actual target, native-error denominator and full-vocabulary KL checks passed!')
