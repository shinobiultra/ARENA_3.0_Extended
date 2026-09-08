"""Semantic exercise checks and independent analytic estimator regressions."""

from __future__ import annotations

import itertools
import math

import pytest
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods as m


def _complete(n=4):
    edges = m.boundary_masks(n)
    inside = t.tensor([mask for mask in range(2**n) if 2 <= mask.bit_count() <= n - 2], dtype=t.int64)
    probability = t.full((len(inside),), 1 / len(inside), dtype=t.float64)
    return edges, inside, probability


def _polynomial(n=4):
    masks = t.arange(2**n, dtype=t.int64)
    z = m.coalition_design(masks, n_players=n)
    additive = 1.75 + z @ t.arange(1, n + 1, dtype=t.float64)
    game = additive + 3 * z[:, 0] * z[:, 1] - 2 * z[:, 0] * z[:, 2] * z[:, 3]
    return t.stack((t.ones_like(game) * 7, additive, game, -2 * game))


def test_coalition_design(fn=m.coalition_design):
    masks = t.tensor([0, 1, 6, 15, 6], dtype=t.int64)
    expected = t.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=t.float64)
    t.testing.assert_close(fn(masks, n_players=4), expected, rtol=0, atol=0)
    print("All tests in test_coalition_design passed!")


def test_sii_query_weights(fn=m.sii_query_weights):
    # T={0,1}; pair order is 01,02,03,12,13,23. Signs reflect 2,1,1,1,1,0 overlaps.
    row = fn(t.tensor([3], dtype=t.int64), n_players=4, order=2)
    t.testing.assert_close(row, t.tensor([[1/3, -1/6, -1/6, -1/6, -1/6, 1/3]], dtype=t.float64), rtol=0, atol=1e-15)
    values = _polynomial()
    expected = t.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [3, -1, -1, 0, 0, -1], [-6, 2, 2, 0, 0, 2]], dtype=t.float64)
    actual = (values - values[:, :1]) @ fn(t.arange(16), n_players=4, order=2)
    t.testing.assert_close(actual, expected, rtol=0, atol=1e-13)
    print("All tests in test_sii_query_weights passed!")


def test_sample_coalitions(fn=m.sample_coalitions):
    before = t.random.get_rng_state().clone()
    a, p = fn(n_players=8, draws=400, seed=161700, proposal="kernel")
    b, q = fn(n_players=8, draws=400, seed=161700, proposal="kernel")
    t.testing.assert_close(a, b, rtol=0, atol=0)
    t.testing.assert_close(p, q, rtol=0, atol=0)
    t.testing.assert_close(t.random.get_rng_state(), before, rtol=0, atol=0)
    assert a.dtype == t.int64 and p.dtype == t.float64 and len(a.unique()) < len(a)
    normalization = sum(1 / (k * (8-k)) for k in range(2, 7))
    expected = t.tensor([1 / (mask.bit_count() * (8-mask.bit_count()) * normalization * math.comb(8, mask.bit_count())) for mask in a.tolist()], dtype=t.float64)
    t.testing.assert_close(p, expected, rtol=1e-14, atol=1e-15)
    uniform, probabilities = fn(n_players=8, draws=400, seed=161700, proposal="uniform")
    assert all(2 <= x.bit_count() <= 6 for x in uniform.tolist())
    t.testing.assert_close(probabilities, t.full((400,), 1 / 238, dtype=t.float64), rtol=0, atol=1e-15)
    # Larger player count must still sample without constructing a 2**n table.
    wide, _ = fn(n_players=20, draws=17, seed=0)
    assert wide.shape == (17,)
    print("All tests in test_sample_coalitions passed!")


def test_shapiq_estimate(fn=m.shapiq_estimate):
    edges, interior, p = _complete()
    values = _polynomial().reshape(2, 2, 16)
    estimate, se = fn(values[..., edges], values[..., interior], interior, p, n_players=4)
    t.testing.assert_close(estimate, m.exact_sii(values, n_players=4), rtol=0, atol=1e-13)
    assert se.shape == estimate.shape and bool((se >= 0).all())
    repeated = t.tensor([3, 3, 5, 10, 12], dtype=t.int64)
    p = t.full((5,), 1/6, dtype=t.float64)
    mean, observed_se = fn(values[..., edges], values[..., repeated], repeated, p, n_players=4)
    weights = m.sii_query_weights(repeated, n_players=4) / p[:, None]
    contributions = (values[..., repeated] - values[..., :1])[..., :, None] * weights
    exact = (values[..., edges] - values[..., :1]) @ m.sii_query_weights(edges, n_players=4)
    expected_se = ((contributions - contributions.mean(-2, keepdim=True)).square().sum(-2) / (5 * 4)).sqrt()
    t.testing.assert_close(mean, exact + contributions.mean(-2))
    t.testing.assert_close(observed_se, expected_se)
    shifted, shifted_se = fn(values[..., edges] + 13, values[..., repeated] + 13, repeated, p, n_players=4)
    t.testing.assert_close(shifted, mean, atol=1e-13, rtol=0)
    t.testing.assert_close(shifted_se, observed_se, atol=1e-13, rtol=0)
    # A nonzero additive point estimate at finite budget is sampling noise, not a failing oracle.
    assert float(mean[0, 1].abs().max()) > 0.1
    assert float(mean[0, 0].abs().max()) == 0
    print("All tests in test_shapiq_estimate passed!")


def test_constrained_wls(fn=m.constrained_wls):
    x = t.eye(3, dtype=t.float64)
    y = t.tensor([[1., 2., 3.], [-1., 0., 4.]], dtype=t.float64)
    weights = t.ones(3, dtype=t.float64)
    constraint = t.ones((1, 3), dtype=t.float64)
    target = t.tensor([[0.], [2.]], dtype=t.float64)
    solved = fn(x, y, weights, constraint, target)
    expected = y + (target - y.sum(-1, keepdim=True)) / 3
    t.testing.assert_close(solved['coefficients'], expected, rtol=0, atol=1e-12)
    assert solved['identified'] and solved['free_rank'] == 2
    deficient = fn(t.ones((2, 3), dtype=t.float64), t.zeros(2, dtype=t.float64), t.ones(2, dtype=t.float64), constraint, t.ones(1, dtype=t.float64))
    assert not deficient['identified']
    assert deficient['free_rank'] == 0
    t.testing.assert_close(deficient['coefficients'].sum(), t.tensor(1., dtype=t.float64), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match='inconsistent'):
        fn(x, y, weights, t.cat((constraint, constraint)), t.tensor([[0., 1.], [0., 1.]], dtype=t.float64))
    print("All tests in test_constrained_wls passed!")


def test_pair_residual_design(fn=m.pair_residual_design):
    masks = t.arange(16, dtype=t.int64)
    design = fn(masks, n_players=4)
    assert set(design.unique().tolist()) == {-0.5, 0.0}
    t.testing.assert_close(design, design.flip(0), rtol=0, atol=0)
    expected = t.tensor([[-.5, -.5, -.5, 0, 0, 0]], dtype=t.float64)
    t.testing.assert_close(design[1:2], expected, rtol=0, atol=0)
    assert not bool(design[[0, -1]].any())
    print("All tests in test_pair_residual_design passed!")


def test_kernel_weights(fn=m.kernel_weights):
    masks=t.tensor([0,1,3,7,15,31,63],dtype=t.int64)
    t.testing.assert_close(fn(masks,n_players=6,order=1),t.tensor([0,1,.25,1/6,.25,1,0],dtype=t.float64),rtol=0,atol=1e-15)
    t.testing.assert_close(fn(masks,n_players=6,order=2),t.tensor([0,0,1,.5,1,0,0],dtype=t.float64),rtol=0,atol=1e-15)
    print("All tests in test_kernel_weights passed!")


def test_kernelshap_iq(fn=m.kernelshap_iq):
    n = 4
    edges, interior, p = _complete(n)
    values = _polynomial(n).reshape(2, 2, 16)
    result = fn(values[..., edges], values[..., interior], interior, p, n_players=n)
    t.testing.assert_close(result['shapley'], m.exact_sii(values, n_players=n, order=1), rtol=0, atol=1e-12)
    t.testing.assert_close(result['pairs'], m.exact_sii(values, n_players=n, order=2), rtol=0, atol=1e-12)
    assert result['pair_stage']['identified']
    # A pure four-way term cannot fit every boundary residual exactly.
    pure_four = t.zeros(16, dtype=t.float64); pure_four[-1] = 1
    result = fn(pure_four[edges], pure_four[interior], interior, p, n_players=n)
    t.testing.assert_close(result['pairs'], t.full((6,), 1/3, dtype=t.float64), rtol=0, atol=1e-12)
    residual = pure_four[edges] - m.coalition_design(edges, n_players=n) @ result['shapley']
    boundary_error = residual - m.pair_residual_design(edges, n_players=n) @ result['pairs']
    assert float(boundary_error.abs().max()) > .2
    t.testing.assert_close(result['pair_stage']['constraint_residual'], t.zeros(4, dtype=t.float64), rtol=0, atol=1e-12)
    print("All tests in test_kernelshap_iq passed!")


def test_query_sampled_game(fn=m.query_sampled_game):
    requested = t.tensor([3, 3, 5, 12], dtype=t.int64)
    seen = []
    def oracle(masks):
        seen.append(masks.clone())
        assert len(masks.unique()) == len(masks)
        return t.stack((masks.double(), -masks.double()))
    record = fn(oracle, requested, n_players=4)
    assert len(seen) == 1 and record['unique_queries'] == 13
    assert record['total_requests'] == 14 and record['draw_requests'] == 4
    t.testing.assert_close(record['sampled_values'][0], requested.double(), rtol=0, atol=0)
    assert set(seen[0].tolist()) == set(m.boundary_masks(4).tolist()) | set(requested.tolist())
    print("All tests in test_query_sampled_game passed!")


@pytest.mark.parametrize('n', [4, 5, 6, 7, 8])
def test_complete_basis_operators(n):
    """All games follow by linearity: test every basis column, not selected nice games."""
    basis = t.eye(2**n, dtype=t.float64)
    edges, inside, p = _complete(n)
    observed = m.kernelshap_iq(basis[:, edges], basis[:, inside], inside, p, n_players=n)
    for order, key in [(1, 'shapley'), (2, 'pairs')]:
        derivative_reference = m.exact_sii(basis, n_players=n, order=order)
        t.testing.assert_close(observed[key], derivative_reference, rtol=0, atol=2e-12)
        gamma_reference = m.sii_query_weights(t.arange(2**n), n_players=n, order=order)
        t.testing.assert_close(gamma_reference, derivative_reference, rtol=0, atol=2e-14)
    print(f"All tests in test_complete_basis_operators passed! ({n} players; all {2**n} basis games)")


def test_exact_shapiq_expectation_and_variance():
    """Exhaustive two-iid-draw law checks unbiasedness and the sample-SE formula."""
    n=4; edges, inside, p = _complete(n)
    v = _polynomial(n)[2]
    means=[]; estimated_variances=[]
    for a,b in itertools.product(inside.tolist(),repeat=2):
        draws=t.tensor([a,b],dtype=t.int64)
        mean,se=m.shapiq_estimate(v[edges],v[draws],draws,p[:2],n_players=n)
        means.append(mean); estimated_variances.append(se.square())
    means=t.stack(means); variances=t.stack(estimated_variances)
    exact=m.exact_sii(v,n_players=n)
    t.testing.assert_close(means.mean(0),exact,rtol=0,atol=2e-13)
    true_variance=(means-exact).square().mean(0)
    t.testing.assert_close(variances.mean(0),true_variance,rtol=1e-13,atol=1e-13)
    print("All tests in test_exact_shapiq_expectation_and_variance passed! (complete two-draw law)")


def test_kernel_additive_control_at_sampled_budgets():
    n=8; masks,p=m.sample_coalitions(n_players=n,draws=64,seed=42)
    v=_polynomial(n)[1]; edges=m.boundary_masks(n)
    result=m.kernelshap_iq(v[edges],v[masks],masks,p,n_players=n)
    t.testing.assert_close(result['shapley'],t.arange(1,n+1,dtype=t.float64),rtol=0,atol=1e-11)
    t.testing.assert_close(result['pairs'],t.zeros(math.comb(n,2),dtype=t.float64),rtol=0,atol=1e-11)
    print("All tests in test_kernel_additive_control_at_sampled_budgets passed! (zero pair control)")


@pytest.mark.parametrize('function,kwargs',[
    (m.sample_coalitions,{'n_players':4,'draws':1,'seed':0}),
    (m.sample_coalitions,{'n_players':3,'draws':10,'seed':0}),
    (m.sample_coalitions,{'n_players':4,'draws':10,'seed':0,'proposal':'unknown'}),
    (m.coalition_design,{'masks':t.tensor([16]),'n_players':4}),
    (m.coalition_design,{'masks':t.tensor([0.]),'n_players':4}),
])
def test_invalid_inputs(function,kwargs):
    with pytest.raises((TypeError,ValueError)):
        function(**kwargs)
    print(f"All tests in test_invalid_inputs passed! ({function.__name__} rejected {kwargs})")
