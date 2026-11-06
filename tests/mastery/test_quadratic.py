"""Analytic toy checks; 1e-12 tolerates float64 rounding, not a changed recurrence."""
from dataclasses import FrozenInstanceError, fields
import json

import numpy as np
import pytest


def api():
    from arena_ext.mastery import quadratic
    return quadratic


def case(**changes):
    return api().QuadraticConfig(**{
        'lambda_min': 1., 'lambda_max': 10., 'angle_deg': 0.,
        'x0': (1., 1.), 'eta': .1, 'beta': 0., 'noise_std': 0.,
        'seed': 0, 'steps': 2, **changes})


def test_frozen_config_exact_fields():
    q = api()
    assert [f.name for f in fields(q.QuadraticConfig)] == [
        'lambda_min', 'lambda_max', 'angle_deg', 'x0', 'eta', 'beta', 'noise_std', 'seed', 'steps']
    with pytest.raises(FrozenInstanceError):
        q.QuadraticConfig().eta = .2
    assert q.SIMULATOR_VERSION == 'quadratic-heavy-ball-v1'


def test_gd_one_step():
    trace = api().simulate(case(steps=1))
    np.testing.assert_allclose(trace.states, [(1, 1), (.9, 0)], atol=1e-12, rtol=0)
    np.testing.assert_allclose(trace.losses, [5.5, .405], atol=1e-12, rtol=0)
    np.testing.assert_allclose(trace.update_norms, [0, np.hypot(.1, 1)], atol=1e-12, rtol=0)
    assert trace.states.dtype == np.float64
    assert trace.stop_reason == 'completed'


def test_heavy_ball_convention():
    trace = api().simulate(case(beta=.9))
    np.testing.assert_allclose(trace.states[2], [.72, -.9], atol=1e-12, rtol=0)


def test_stability_boundary():
    q = api()
    assert q.spectral_radius(case(eta=.19)) < 1
    assert q.spectral_radius(case(eta=.20)) == pytest.approx(1, abs=1e-12)
    assert q.spectral_radius(case(eta=.21)) > 1
    assert q.spectral_radius(case(eta=0)) == pytest.approx(1)


def test_stability_oracle_agrees_with_independent_polynomial_roots():
    for beta in (0., .2, .9):
        for eta in (.001, .1, .25, 1.):
            config = case(beta=beta, eta=eta)
            expected = max(abs(root) for lam in (1., 10.)
                           for root in np.roots([1, -(1 + beta - eta * lam), beta]))
            assert api().spectral_radius(config) == pytest.approx(expected, abs=1e-12)


def test_zero_learning_rate():
    trace = api().simulate(case(eta=0, noise_std=4, steps=12))
    np.testing.assert_array_equal(trace.states, np.tile([1., 1.], (13, 1)))
    np.testing.assert_array_equal(trace.update_norms, np.zeros(13))


def test_stationary_noiseless_optimum():
    trace = api().simulate(case(x0=(0, 0), beta=.9))
    np.testing.assert_array_equal(trace.losses, np.zeros(3))
    np.testing.assert_array_equal(trace.states, np.zeros((3, 2)))


def test_same_seed_replay():
    config = case(noise_std=.2, steps=20)
    np.testing.assert_array_equal(api().simulate(config).states, api().simulate(config).states)


def test_different_seed_noisy_paths():
    assert not np.array_equal(api().simulate(case(noise_std=.2, seed=1)).states,
                              api().simulate(case(noise_std=.2, seed=2)).states)


def test_rotation_invariance_with_rotated_initial_state():
    angle = np.deg2rad(43.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    original = api().simulate(case(steps=20, beta=.2))
    rotated = api().simulate(case(angle_deg=43, x0=tuple(rotation @ [1., 1.]), steps=20, beta=.2))
    np.testing.assert_allclose(rotated.losses, original.losses, atol=1e-12, rtol=0)
    np.testing.assert_allclose(api().hessian(case(angle_deg=43)), rotation @ np.diag([1., 10.]) @ rotation.T,
                               atol=1e-12, rtol=0)


@pytest.mark.parametrize('changes', [
    {'lambda_min': 0}, {'lambda_min': 11}, {'lambda_max': float('inf')},
    {'angle_deg': -1}, {'angle_deg': 180}, {'x0': (1,)}, {'x0': (1, float('nan'))},
    {'eta': -1}, {'eta': True}, {'beta': 1}, {'noise_std': -1},
    {'seed': -1}, {'seed': 2**32}, {'seed': 1.0}, {'seed': True},
    {'steps': 0}, {'steps': 1001}, {'steps': 2.0}, {'steps': False},
    {'x0': (False, 1)}, {'eta': '0.1'}, {'noise_std': float('nan')},
])
def test_invalid_parameters(changes):
    with pytest.raises(ValueError):
        case(**changes)


def test_finite_json_compatible_divergence_trace():
    trace = api().simulate(case(eta=3, steps=1000))
    assert trace.stop_reason == 'loss_cap'
    assert len(trace.states) < 1001
    assert np.max(trace.losses) <= 1e12
    np.testing.assert_allclose(trace.losses, .5 * np.sum(trace.states**2 * [1, 10], axis=1))
    json.dumps({'states': trace.states.tolist(), 'losses': trace.losses.tolist(),
                'updates': trace.update_norms.tolist()}, allow_nan=False)


def test_nonfinite_proposal_preserves_accepted_state():
    trace = api().simulate(case(eta=1e308))
    assert trace.stop_reason == 'nonfinite'
    np.testing.assert_array_equal(trace.states, [[1, 1]])
    np.testing.assert_allclose(trace.losses, [5.5], atol=1e-12, rtol=0)


def test_unrepresentable_initial_loss_has_explicit_empty_trace():
    trace = api().simulate(case(x0=(1e308, 1e308)))
    assert trace.stop_reason == 'nonfinite'
    assert trace.states.shape == (0, 2)
    assert trace.losses.shape == trace.update_norms.shape == (0,)
    json.dumps(trace.losses.tolist(), allow_nan=False)


def test_initial_loss_cap_is_visible():
    trace = api().simulate(case(x0=(1e7, 0)))
    assert trace.stop_reason == 'loss_cap'
    np.testing.assert_array_equal(trace.states, [[1e7, 0]])
    assert trace.losses[0] == pytest.approx(5e13)


def test_finite_extreme_oracle_never_claims_stability():
    assert api().spectral_radius(case(lambda_min=1e308, lambda_max=1e308, eta=1e308)) > 1
    np.testing.assert_allclose(api().hessian(case(lambda_min=1e308, lambda_max=1e308, angle_deg=45)),
                               np.eye(2) * 1e308, rtol=1e-15, atol=0)


def test_no_global_rng_mutation():
    before = np.random.get_state()
    api().simulate(case(noise_std=.2))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_momentum_neutral_boundaries_never_round_to_strict_stability():
    for beta in (.1, .5, .9, .99):
        assert api().spectral_radius(case(eta=0, beta=beta)) >= 1
        assert api().spectral_radius(case(eta=2 * (1 + beta) / 10, beta=beta)) >= 1


def test_extreme_coordinates_keep_representable_initial_loss():
    trace = api().simulate(case(lambda_min=1e-310, lambda_max=1e-310,
                               angle_deg=45, x0=(1.7e308, 1.7e308)))
    assert trace.stop_reason == 'loss_cap'
    np.testing.assert_array_equal(trace.states, [[1.7e308, 1.7e308]])
    assert trace.losses[0] == pytest.approx(2.89e306, rel=1e-12)
