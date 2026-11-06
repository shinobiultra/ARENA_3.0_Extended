"""Independent gradient/game checks and strict numeric evidence rejection checks.

The maintained lesson is 16.9. The retired 16.7 publication report is not imported.
"""
import copy
import itertools
import math

import pytest
import torch
import torch.nn.functional as F

_caller_dtype, _caller_threads = torch.get_default_dtype(), torch.get_num_threads()
try:
    from chapter16_shapley_attribution_baselines.exercises.part9_data_shapley import solutions as methods
finally:
    # Preserve the notebook's float64 setup without leaking it during collection.
    torch.set_default_dtype(_caller_dtype)
    torch.set_num_threads(_caller_threads)
from chapter16_shapley_attribution_baselines.exercises.part9_data_shapley import tests as checks
from arena_ext import legacy_shapley_evidence as evidence


@pytest.fixture(autouse=True)
def _lesson_precision():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)


def _strict_case():
    params = methods.init_mlp([3, 2, 1], seed=41)
    x = torch.tensor([[1., 0., 1.], [1., 0., 1.], [-1., 1., 0.], [0., 2., 1.]], dtype=torch.float64)
    y = torch.tensor([1., 1., 0., 1.], dtype=torch.float64)
    return params, x, y


def _strict_autograd_rows(params, x, y):
    # nn.Linear's transposed storage is independent of the lesson's factor code.
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh(), torch.nn.Linear(2, 1)).double()
    with torch.no_grad():
        for layer, (weight, bias) in zip((model[0], model[2]), params, strict=True):
            layer.weight.copy_(weight.T)
            layer.bias.copy_(bias)
    rows = []
    for example, label in zip(x, y, strict=True):
        loss = F.binary_cross_entropy_with_logits(model(example).squeeze(), label)
        grads = torch.autograd.grad(loss, tuple(model.parameters()))
        rows.append(torch.cat([g.T.flatten() if g.ndim == 2 else g.flatten() for g in grads]))
    return torch.stack(rows)


def _assert_strict_ghost_contract(ghost):
    params, x, y = _strict_case()
    factors = methods.backprop_factors(params, x, y)
    other = methods.backprop_factors(params, x[:2] * .7, 1 - y[:2])
    expected = _strict_autograd_rows(params, x, y) @ _strict_autograd_rows(params, x[:2] * .7, 1 - y[:2]).T
    torch.testing.assert_close(ghost(factors, other), expected, atol=1e-12, rtol=1e-12)


def test_historical_strict_gradient_and_ghost_tolerances():
    params, x, y = _strict_case()
    actual = methods.per_example_grads(methods.backprop_factors(params, x, y))
    torch.testing.assert_close(actual, _strict_autograd_rows(params, x, y), atol=1e-12, rtol=1e-12)
    _assert_strict_ghost_contract(methods.ghost_dot_products)


def test_strict_ghost_oracle_rejects_one_e_minus_eight_mutation():
    def perturbed(left, right):
        return methods.ghost_dot_products(left, right) + 1e-8
    with pytest.raises(AssertionError):
        _assert_strict_ghost_contract(perturbed)


def test_historical_strict_sgd_and_local_game_tolerances():
    params, x, y = _strict_case()
    factors = methods.backprop_factors(params, x, y)
    rows = _strict_autograd_rows(params, x, y)
    flat = checks._flat(params)
    keep = torch.tensor([True, False, True, False])
    actual = checks._flat(methods.sgd_step(params, factors, .2, 4, keep))
    torch.testing.assert_close(actual, flat - .2 / 4 * rows[keep].sum(0), atol=1e-12, rtol=1e-12)
    def loss(vector):
        logits = (torch.tanh(x @ vector[:6].reshape(3, 2) + vector[6:8]) @ vector[8:10] + vector[10])
        return F.binary_cross_entropy_with_logits(logits, y)
    expected = []
    for mask in range(16):
        ids = [i for i in range(4) if mask & (1 << i)]
        gradient = rows[ids].sum(0) if ids else torch.zeros_like(flat)
        expected.append(loss(flat) - loss(flat - .1 / 4 * gradient))
    values = methods.local_game_values(params, factors, x, y, .1, 4)
    torch.testing.assert_close(values, torch.stack(expected), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(checks._flat(params), flat, atol=0, rtol=0)


def test_historical_strict_taylor_and_trajectory_tolerances():
    # The original three-player quadratic includes both self and cross terms.
    gradients = torch.tensor([[1., 2.], [-1., 0.], [0., 1.]], dtype=torch.float64)
    gradient = torch.tensor([2., -1.], dtype=torch.float64)
    hessian = torch.tensor([[3., .5], [.5, 2.]], dtype=torch.float64)
    scale = .3 / 3
    # Encode explicit vectors as two weight coordinates; pad the bias coordinate
    # with zero curvature so it cannot change this original quadratic game.
    factors = [(gradients, torch.ones(3, 1))]
    padded_hessian = F.pad(hessian, (0, 1, 0, 1))
    first = methods.first_order_credit((gradients @ gradient)[:, None].repeat(1, 7), .3, 3)
    torch.testing.assert_close(first, scale * gradients @ gradient, atol=1e-12, rtol=1e-12)
    def hvp_fn(params, vx, vy, vector):
        result = padded_hessian @ checks._flat(vector)
        return [(result[:2, None], result[2:])]
    observed = methods.second_order_credit(
        [(torch.zeros(2, 1), torch.zeros(1))], factors, None, None, .3, 3, first, hvp_fn=hvp_fn
    )
    values = []
    for mask in range(8):
        ids = [i for i in range(3) if mask & (1 << i)]
        delta = -scale * (gradients[ids].sum(0) if ids else torch.zeros(2))
        values.append(-gradient @ delta - .5 * delta @ hessian @ delta)
    torch.testing.assert_close(observed, methods.exact_local_shapley(torch.stack(values)), atol=1e-12, rtol=1e-12)
    params, x, y = _strict_case()
    schedule = methods.make_schedule(4, 2, 3, 8)
    result = methods.run_in_run_shapley(params, x, y, x, y, schedule, .1, exact_local=True)
    torch.testing.assert_close(result['exact_per_step'].sum(1), result['losses'][:-1] - result['losses'][1:], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result['exact'].sum(), result['losses'][0] - result['losses'][-1], atol=1e-12, rtol=1e-12)


def test_independent_gradient_oracle():
    checks.test_mlp_forward(methods.mlp_forward, methods.init_mlp)
    checks.test_backprop_factors(
        methods.backprop_factors, methods.per_example_grads, methods.mlp_forward, methods.init_mlp
    )
    checks.test_ghost_dot_products(methods.ghost_dot_products, methods.backprop_factors, methods.init_mlp)


def test_ghost_products_reject_mismatched_layer_counts():
    params = methods.init_mlp([3, 2, 1], seed=41)
    x = torch.ones(4, 3)
    factors = methods.backprop_factors(params, x, torch.ones(4))
    with pytest.raises(ValueError):
        methods.ghost_dot_products(factors, factors[:1])


def test_fixed_denominator_exact_local_game():
    checks.test_sgd_step(methods.sgd_step, methods.backprop_factors, methods.init_mlp)
    checks.test_sgd_train(methods.sgd_train, methods.init_mlp)
    checks.test_local_game_values(methods.local_game_values, methods.backprop_factors, methods.init_mlp)
    # Keep the legacy nonzero-empty random game and all 24 independent permutations.
    values = torch.randn(16, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    expected = torch.zeros(4, dtype=torch.float64)
    for order in itertools.permutations(range(4)):
        mask = 0
        for player in order:
            updated = mask | (1 << player)
            expected[player] += (values[updated] - values[mask]) / math.factorial(4)
            mask = updated
    actual = methods.exact_local_shapley(values)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual.sum(), values[-1] - values[0], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('values', [torch.zeros(0), torch.zeros(1), torch.zeros(7), torch.zeros(2, 2)])
def test_exact_local_game_rejects_incomplete_bitmask_table(values):
    with pytest.raises(ValueError):
        methods.exact_local_shapley(values)


def test_taylor_self_cross_and_complete_trajectory():
    checks.test_first_order_credit(methods.first_order_credit)
    checks.test_ghost_vector_products(methods.ghost_vector_products, methods.backprop_factors, methods.init_mlp)
    checks.test_second_order_credit(methods.second_order_credit, methods.ghost_vector_products)
    checks.test_run_in_run_shapley(methods.run_in_run_shapley, methods.sgd_train, methods.init_mlp)
    # Repeat an ID within a minibatch, not only across steps: credits must accumulate.
    x, y = checks._small_problem(seed=41, n=4, d=3)
    params = methods.init_mlp([3, 2, 1], seed=41)
    schedule = [torch.tensor([0, 0, 1]), torch.tensor([1, 2, 3])]
    result = methods.run_in_run_shapley(params, x, y, x, y, schedule, 0.1, exact_local=True)
    factors = methods.backprop_factors(params, x[schedule[0]], y[schedule[0]])
    local = methods.first_order_credit(
        methods.ghost_dot_products(factors, methods.backprop_factors(params, x, y)), 0.1, 3
    )
    torch.testing.assert_close(result['first_per_step'][0, 0], local[:2].sum())
    torch.testing.assert_close(
        result['exact'].sum(), result['losses'][0] - result['losses'][-1], atol=1e-12, rtol=1e-12
    )
    # All-masked training is still inert via the maintained sgd_train interface.
    unchanged = methods.sgd_train(params, x, y, schedule, 0.1, keep=torch.zeros(4))
    torch.testing.assert_close(checks._flat(unchanged), checks._flat(params), atol=0, rtol=0)


def _numeric_payload():
    tensor = {'tensor': True, 'dtype': 'torch.float64', 'shape': [2], 'values': [1., 2.]}
    return {
        'digit_runs': {seed: {'run': {'wall_seconds': 1.}} for seed in ('16701', '16702', '16703')},
        'timing_rows': [{'seconds': 1.} for _ in range(14)],
        'training_times': [{'seconds': 1.} for _ in range(6)],
        'actual': tensor,
    }


def _timing_paths():
    return frozenset(
        [('digit_runs', seed, 'run', 'wall_seconds') for seed in ('16701', '16702', '16703')]
        + [('timing_rows', i, 'seconds') for i in range(14)]
        + [('training_times', i, 'seconds') for i in range(6)]
    )


def test_numeric_comparison_rejects_changes_and_unknown_timing_fields():
    payload = _numeric_payload()
    fresh = copy.deepcopy(payload)
    fresh['timing_rows'][0]['seconds'] = 2.
    counts, excluded = evidence.compare(payload, fresh, timing_paths=_timing_paths())
    assert counts['tensor_elements'] == 2 and len(excluded) == 23
    fresh['actual']['values'][0] += 1e-12
    with pytest.raises(ValueError, match='Tensor changed'):
        evidence.compare(payload, fresh, timing_paths=_timing_paths())
    fresh = copy.deepcopy(payload)
    fresh['other_wall_seconds'] = 2.
    with pytest.raises(ValueError, match='Keys/order changed'):
        evidence.compare(payload, fresh, timing_paths=_timing_paths())
    fresh = copy.deepcopy(payload)
    fresh['actual']['dtype'] = 'torch.float32'
    with pytest.raises(ValueError, match='Tensor changed'):
        evidence.compare(payload, fresh, timing_paths=_timing_paths())


@pytest.mark.parametrize('value, message', [
    ({'tensor': True, 'dtype': 'torch.float64', 'shape': [3], 'values': [1., 2.]}, 'shape'),
    ({'tensor': True, 'dtype': 'torch.float64', 'shape': [1], 'values': [float('nan')]}, 'Nonfinite'),
])
def test_invalid_tensor_shape_and_nonfinite_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        evidence.array(value)


@pytest.mark.parametrize('mutation', ['unknown_timing_change', 'zero_timing', 'missing_timing', 'dtype_missing', 'nonfinite_scalar', 'scalar_type'])
def test_evidence_rejects_unlisted_drift(mutation):
    payload = _numeric_payload()
    payload['wall_seconds'] = 1.0  # deliberately outside the timing allowlist
    payload['scalar'] = 1
    fresh = copy.deepcopy(payload)
    if mutation == 'unknown_timing_change':
        fresh['wall_seconds'] = 2.0
    elif mutation == 'zero_timing':
        fresh['timing_rows'][0]['seconds'] = 0.0
    elif mutation == 'missing_timing':
        fresh['timing_rows'].pop()
    elif mutation == 'dtype_missing':
        del fresh['actual']['dtype']
    elif mutation == 'nonfinite_scalar':
        fresh['scalar'] = float('inf')
    else:
        fresh['scalar'] = True
    with pytest.raises(ValueError):
        evidence.compare(payload, fresh, timing_paths=_timing_paths())
