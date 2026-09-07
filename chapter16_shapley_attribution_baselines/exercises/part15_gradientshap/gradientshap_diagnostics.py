"""Staged, learner-visible diagnostics for Expected Gradients.

Every ``diagnose_*`` function calls only methods already taught at that point.
The fixtures use literal float64 models and independent analytic answers; plots
render these returned arrays but do not implement the attribution method.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import torch as t
from torch import Tensor, nn


DTYPE = t.float64


def _method(methods, name):
    if methods is None:
        from . import gradientshap_methods

        candidate = getattr(gradientshap_methods, name)
    elif isinstance(methods, Mapping):
        candidate = methods.get(name)
    else:
        candidate = getattr(methods, name, None)
    if not callable(candidate):
        raise TypeError(
            f"Learner method namespace must provide a callable {name!r}; "
            "pass globals(), a mapping, or a SimpleNamespace built from the taught methods."
        )
    return candidate


def _linear(weight, bias):
    weight = t.as_tensor(weight, dtype=DTYPE)
    bias = t.as_tensor(bias, dtype=DTYPE).reshape(-1)
    layer = nn.Linear(weight.shape[1], weight.shape[0], dtype=DTYPE)
    with t.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer


def _prediction(model, rows):
    with t.no_grad():
        return model(rows).reshape(-1)


def _max_abs(left, right):
    left = t.as_tensor(left, dtype=DTYPE)
    right = t.as_tensor(right, dtype=DTYPE)
    return float((left - right).abs().max())


def _relu_model():
    return nn.Sequential(_linear([[1.0]], [0.0]), nn.ReLU()).eval()


def _threshold_model(weights=(2.0, 1.0), bias=-2.0):
    weights = t.as_tensor(weights, dtype=DTYPE).reshape(1, -1)
    return nn.Sequential(_linear(weights, [bias]), nn.ReLU()).eval()


def _quadratic_model():
    class Quadratic(nn.Module):
        def forward(self, rows):
            return rows[:, 0].square()

    return Quadratic().eval()


def _two_feature_shapley(values):
    values = t.as_tensor(values, dtype=DTYPE).reshape(4)
    phi_0 = ((values[1] - values[0]) + (values[3] - values[2])) / 2
    phi_1 = ((values[2] - values[0]) + (values[3] - values[1])) / 2
    return t.stack((phi_0, phi_1))


def schedule_fixture():
    return {
        "source_ids": t.tensor([101, 305, 909], dtype=t.int64),
        "background_rows": 5,
        "short_samples": 8,
        "long_samples": 32,
        "seed": 161510,
    }


def diagnose_schedule(sample_path_schedule_fn):
    fixture = schedule_fixture()
    source_ids = fixture["source_ids"]
    args = (fixture["background_rows"], fixture["seed"])
    indices_8, alpha_8 = sample_path_schedule_fn(
        source_ids, args[0], fixture["short_samples"], args[1]
    )
    indices_32, alpha_32 = sample_path_schedule_fn(
        source_ids, args[0], fixture["long_samples"], args[1]
    )
    reordered = source_ids[t.tensor([2, 0], dtype=t.int64)]
    reorder_indices, reorder_alpha = sample_path_schedule_fn(
        reordered, args[0], fixture["short_samples"], args[1]
    )
    single_indices, single_alpha = sample_path_schedule_fn(
        source_ids[:1], args[0], fixture["short_samples"], args[1]
    )
    return {
        **fixture,
        "indices_8": indices_8,
        "alpha_8": alpha_8,
        "indices_32": indices_32,
        "alpha_32": alpha_32,
        "prefix_match": bool(
            t.equal(indices_8, indices_32[:, :8]) and t.equal(alpha_8, alpha_32[:, :8])
        ),
        "reorder_match": bool(
            t.equal(reorder_indices, indices_8[[2, 0]])
            and t.equal(reorder_alpha, alpha_8[[2, 0]])
        ),
        "single_row_match": bool(
            t.equal(single_indices, indices_8[:1]) and t.equal(single_alpha, alpha_8[:1])
        ),
    }


def path_fixture():
    return {
        "query": t.tensor([[2.0, 1.0]], dtype=DTYPE),
        "background": t.tensor([[-1.0, 0.0], [0.0, 2.0]], dtype=DTYPE),
        "indices": t.tensor([[0, 1, 0, 1]], dtype=t.int64),
        "alpha": t.tensor([[0.20, 0.35, 0.75, 0.90]], dtype=DTYPE),
    }


def diagnose_interpolation(interpolate_paths_fn):
    fixture = path_fixture()
    points, deltas = interpolate_paths_fn(
        fixture["query"],
        fixture["background"],
        fixture["indices"],
        fixture["alpha"],
    )
    donors = fixture["background"][fixture["indices"]]
    expected = donors + fixture["alpha"][..., None] * (fixture["query"][:, None, :] - donors)
    return {
        **fixture,
        "donors": donors,
        "points": points,
        "deltas": deltas,
        "expected_points": expected,
        "max_abs_error": _max_abs(points, expected),
    }


def gradient_fixture():
    affine = nn.Sequential(_linear([[2.0, -3.0]], [0.5])).eval()
    affine_points = t.tensor(
        [[[-1.0, 0.0], [0.5, 2.0], [2.0, -1.0], [3.0, 3.0]]],
        dtype=DTYPE,
    )
    z = t.linspace(-1.0, 1.5, 21, dtype=DTYPE)
    return {
        "affine_model": affine,
        "affine_points": affine_points,
        "affine_expected": t.tensor([2.0, -3.0], dtype=DTYPE),
        "quadratic_model": _quadratic_model(),
        "quadratic_points": z.reshape(1, -1, 1),
        "quadratic_expected": (2 * z).reshape(1, -1, 1),
    }


def diagnose_gradients(input_gradients_fn):
    fixture = gradient_fixture()
    affine = input_gradients_fn(fixture["affine_model"], fixture["affine_points"])
    quadratic = input_gradients_fn(fixture["quadratic_model"], fixture["quadratic_points"])
    return {
        "affine_points": fixture["affine_points"],
        "affine_gradients": affine,
        "affine_expected": fixture["affine_expected"],
        "affine_max_abs_error": _max_abs(affine, fixture["affine_expected"]),
        "quadratic_points": fixture["quadratic_points"],
        "quadratic_gradients": quadratic,
        "quadratic_expected": fixture["quadratic_expected"],
        "quadratic_max_abs_error": _max_abs(quadratic, fixture["quadratic_expected"]),
    }


def sample_fixture():
    return {
        "relu_model": _relu_model(),
        "relu_query": t.tensor([[1.0]], dtype=DTYPE),
        "relu_background": t.tensor([[-1.0]], dtype=DTYPE),
        "source_ids": t.tensor([17], dtype=t.int64),
        "samples": 128,
        "seed": 161510,
        "quadratic_model": _quadratic_model(),
        "quadratic_query": t.tensor([[1.0]], dtype=DTYPE),
        "quadratic_background": t.tensor([[0.0]], dtype=DTYPE),
        "quadratic_indices": t.tensor([[0, 0]], dtype=t.int64),
        "quadratic_alpha": t.tensor([[0.25, 0.75]], dtype=DTYPE),
    }


def diagnose_expected_gradient_samples(
    expected_gradient_samples_fn,
    sample_path_schedule_fn,
):
    fixture = sample_fixture()
    indices, alpha = sample_path_schedule_fn(
        fixture["source_ids"], 1, fixture["samples"], fixture["seed"]
    )
    relu_samples = expected_gradient_samples_fn(
        fixture["relu_model"],
        fixture["relu_query"],
        fixture["relu_background"],
        indices,
        alpha,
    )
    quadratic_samples = expected_gradient_samples_fn(
        fixture["quadratic_model"],
        fixture["quadratic_query"],
        fixture["quadratic_background"],
        fixture["quadratic_indices"],
        fixture["quadratic_alpha"],
    )
    relu_expected = (alpha > 0.5).to(DTYPE)[..., None] * 2.0
    quadratic_expected = (2 * fixture["quadratic_alpha"])[..., None]
    return {
        "alpha": alpha,
        "relu_samples": relu_samples,
        "relu_expected": relu_expected,
        "relu_max_abs_error": _max_abs(relu_samples, relu_expected),
        "quadratic_alpha": fixture["quadratic_alpha"],
        "quadratic_samples": quadratic_samples,
        "quadratic_expected": quadratic_expected,
        "quadratic_max_abs_error": _max_abs(quadratic_samples, quadratic_expected),
        "population_mean": t.tensor(1.0, dtype=DTYPE),
        "population_variance": t.tensor(1.0, dtype=DTYPE),
    }


def diagnose_convergence(
    sample_path_schedule_fn,
    expected_gradient_samples_fn,
    summarize_expected_gradients_fn,
):
    fixture = sample_fixture()
    indices, alpha = sample_path_schedule_fn(
        fixture["source_ids"], 1, fixture["samples"], fixture["seed"]
    )
    samples = expected_gradient_samples_fn(
        fixture["relu_model"],
        fixture["relu_query"],
        fixture["relu_background"],
        indices,
        alpha,
    )
    budgets = (2, 4, 8, 16, 32, 64, 128)
    means, standard_errors = summarize_expected_gradients_fn(samples, budgets)
    budget_tensor = t.tensor(budgets, dtype=DTYPE)
    return {
        "budgets": t.tensor(budgets, dtype=t.int64),
        "means": means[:, 0, 0],
        "standard_errors": standard_errors[:, 0, 0],
        "population_mean": t.tensor(1.0, dtype=DTYPE),
        "analytic_standard_error": budget_tensor.rsqrt(),
        "absolute_error": (means[:, 0, 0] - 1.0).abs(),
        "samples": samples,
    }


def estimator_fixture():
    affine = nn.Sequential(_linear([[2.0, -3.0]], [0.5])).eval()
    dummy = nn.Sequential(_linear([[2.0, -3.0, 0.0]], [0.5])).eval()
    constant = nn.Sequential(_linear([[0.0, 0.0]], [2.0])).eval()
    return {
        "affine_model": affine,
        "affine_query": t.tensor([[2.0, 1.0]], dtype=DTYPE),
        "affine_background": t.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=DTYPE),
        "balanced_indices": t.tensor([[0, 1, 0, 1]], dtype=t.int64),
        "balanced_alpha": t.tensor([[0.15, 0.35, 0.65, 0.85]], dtype=DTYPE),
        "affine_expected_phi": t.tensor([[3.0, 0.0]], dtype=DTYPE),
        "dummy_model": dummy,
        "dummy_query": t.tensor([[2.0, 1.0, 9.0]], dtype=DTYPE),
        "dummy_background": t.tensor([[0.0, 0.0, -5.0], [1.0, 2.0, 7.0]], dtype=DTYPE),
        "constant_model": constant,
        "relu_model": _relu_model(),
        "relu_query": t.tensor([[1.0]], dtype=DTYPE),
        "relu_background": t.tensor([[-1.0]], dtype=DTYPE),
        "relu_indices": t.zeros((1, 5), dtype=t.int64),
        "relu_alpha": t.tensor([[0.10, 0.20, 0.30, 0.40, 0.60]], dtype=DTYPE),
    }


def diagnose_expected_gradients(expected_gradients_fn, sample_path_schedule_fn):
    fixture = estimator_fixture()
    affine_base, affine_phi, affine_se = expected_gradients_fn(
        fixture["affine_model"],
        fixture["affine_query"],
        fixture["affine_background"],
        fixture["balanced_indices"],
        fixture["balanced_alpha"],
    )
    scheduled_indices, scheduled_alpha = sample_path_schedule_fn(
        t.tensor([41], dtype=t.int64), 2, 32, 161510
    )
    dummy_base, dummy_phi, dummy_se = expected_gradients_fn(
        fixture["dummy_model"],
        fixture["dummy_query"],
        fixture["dummy_background"],
        scheduled_indices,
        scheduled_alpha,
    )
    constant_base, constant_phi, constant_se = expected_gradients_fn(
        fixture["constant_model"],
        fixture["affine_query"],
        fixture["affine_background"],
        scheduled_indices,
        scheduled_alpha,
    )
    relu_base, relu_phi, relu_se = expected_gradients_fn(
        fixture["relu_model"],
        fixture["relu_query"],
        fixture["relu_background"],
        fixture["relu_indices"],
        fixture["relu_alpha"],
    )

    def result(model, query, base, phi, se):
        prediction = _prediction(model, query)[0]
        return {
            "base": base,
            "phi": phi,
            "standard_error": se,
            "prediction": prediction,
            "reconstruction": base + phi.sum(),
            "residual": base + phi.sum() - prediction,
        }

    return {
        "affine": result(
            fixture["affine_model"],
            fixture["affine_query"],
            affine_base,
            affine_phi,
            affine_se,
        ),
        "dummy": result(
            fixture["dummy_model"],
            fixture["dummy_query"],
            dummy_base,
            dummy_phi,
            dummy_se,
        ),
        "constant": result(
            fixture["constant_model"],
            fixture["affine_query"],
            constant_base,
            constant_phi,
            constant_se,
        ),
        "finite_relu": result(
            fixture["relu_model"],
            fixture["relu_query"],
            relu_base,
            relu_phi,
            relu_se,
        ),
        "affine_expected_phi": fixture["affine_expected_phi"],
        "affine_max_abs_error": _max_abs(affine_phi, fixture["affine_expected_phi"]),
        "dummy_max_abs": float(dummy_phi[:, 2].abs().max()),
        "constant_max_abs": float(constant_phi.abs().max()),
    }


def quadrature_fixture():
    return {
        "relu_model": _relu_model(),
        "query": t.tensor([[1.0]], dtype=DTYPE),
        "background": t.tensor([[-2.0]], dtype=DTYPE),
        "nodes": (4, 8, 16, 32, 64, 128, 256, 512),
        "analytic": t.tensor(1.0, dtype=DTYPE),
        "quadratic_model": _quadratic_model(),
        "quadratic_query": t.tensor([[1.0]], dtype=DTYPE),
        "quadratic_background": t.tensor([[0.0]], dtype=DTYPE),
    }


def diagnose_quadrature(gauss_legendre_pairs_fn, input_gradients_fn):
    fixture = quadrature_fixture()
    estimates = []
    for nodes in fixture["nodes"]:
        pair = gauss_legendre_pairs_fn(
            fixture["relu_model"], fixture["query"], fixture["background"], nodes
        )
        estimates.append(pair[0, 0, 0])
    estimates = t.stack(estimates)
    quadratic = gauss_legendre_pairs_fn(
        fixture["quadratic_model"],
        fixture["quadratic_query"],
        fixture["quadratic_background"],
        2,
    )[0, 0, 0]
    probe_points = t.tensor([[[-0.5], [0.25], [1.0]]], dtype=DTYPE)
    probe_gradients = input_gradients_fn(fixture["quadratic_model"], probe_points)
    return {
        "nodes": t.tensor(fixture["nodes"], dtype=t.int64),
        "relu_estimates": estimates,
        "analytic": fixture["analytic"],
        "absolute_error": (estimates - fixture["analytic"]).abs(),
        "successive_drift": t.cat((t.zeros(1, dtype=DTYPE), estimates.diff().abs())),
        "quadratic_n2": quadratic,
        "quadratic_error": float(abs(quadratic - 1.0)),
        "quadratic_probe_gradients": probe_gradients,
        "quadratic_probe_expected": 2 * probe_points,
    }


def diagnose_background_average(gauss_legendre_pairs_fn):
    model = _relu_model()
    query = t.tensor([[2.0]], dtype=DTYPE)
    background = t.tensor([[-2.0], [1.0]], dtype=DTYPE)
    mean_background = background.mean(dim=0, keepdim=True)
    pair_256 = gauss_legendre_pairs_fn(model, query, background, 256)[0, :, 0]
    pair_512 = gauss_legendre_pairs_fn(model, query, background, 512)[0, :, 0]
    mean_reference_512 = gauss_legendre_pairs_fn(model, query, mean_background, 512)[0, 0, 0]
    return {
        "background": background[:, 0],
        "pair_256": pair_256,
        "pair_512": pair_512,
        "pair_expected": t.tensor([2.0, 1.0], dtype=DTYPE),
        "mean_pair": pair_512.mean(),
        "mean_reference": mean_reference_512,
        "expected_mean_pair": t.tensor(1.5, dtype=DTYPE),
        "expected_mean_reference": t.tensor(2.0, dtype=DTYPE),
        "base": _prediction(model, background).mean(),
        "mean_reference_base": _prediction(model, mean_background)[0],
        "quadrature_drift": (pair_512 - pair_256).abs(),
    }


def noise_fixture(samples=65_536):
    rng = np.random.default_rng(161512)
    return {
        "model": _relu_model(),
        "query": t.tensor([[0.0]], dtype=DTYPE),
        "background": t.tensor([[-1.0]], dtype=DTYPE),
        "indices": t.zeros((1, samples), dtype=t.int64),
        "alpha": t.tensor(rng.random((1, samples)), dtype=DTYPE),
        "standard_normal": t.tensor(rng.standard_normal((1, samples, 1)), dtype=DTYPE),
        "sigmas": t.tensor([0.0, 0.25, 0.50, 1.0], dtype=DTYPE),
    }


def diagnose_noise_target(
    noisy_expected_gradient_samples_fn,
    expected_gradient_samples_fn,
    summarize_expected_gradients_fn,
):
    fixture = noise_fixture()
    clean = expected_gradient_samples_fn(
        fixture["model"],
        fixture["query"],
        fixture["background"],
        fixture["indices"],
        fixture["alpha"],
    )
    estimates, standard_errors = [], []
    zero_samples = None
    for sigma in fixture["sigmas"]:
        noise = sigma * fixture["standard_normal"]
        samples = noisy_expected_gradient_samples_fn(
            fixture["model"],
            fixture["query"],
            fixture["background"],
            fixture["indices"],
            fixture["alpha"],
            noise,
        )
        mean, standard_error = summarize_expected_gradients_fn(samples, (samples.shape[1],))
        estimates.append(mean[0, 0, 0])
        standard_errors.append(standard_error[0, 0, 0])
        if float(sigma) == 0.0:
            zero_samples = samples
    estimates = t.stack(estimates)
    standard_errors = t.stack(standard_errors)
    analytic = fixture["sigmas"] / math.sqrt(2 * math.pi)
    return {
        "sigmas": fixture["sigmas"],
        "estimates": estimates,
        "standard_errors": standard_errors,
        "analytic_smoothed_target": analytic,
        "clean_target": t.zeros_like(analytic),
        "clean_residual": estimates,
        "smoothed_residual": estimates - analytic,
        "zero_noise_equal": bool(t.equal(clean, zero_samples)),
        "samples": int(fixture["alpha"].shape[1]),
    }


def threshold_fixture():
    return {
        "model": _threshold_model(),
        "query": t.tensor([[1.0, 1.0]], dtype=DTYPE),
        "background": t.tensor([[0.0, 0.0]], dtype=DTYPE),
        "coalition_values": t.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DTYPE),
        "exact_shapley": t.tensor([[0.5, 0.5]], dtype=DTYPE),
        "analytic_path": t.tensor([[2 / 3, 1 / 3]], dtype=DTYPE),
        "necessity_drop": t.tensor([1.0, 1.0], dtype=DTYPE),
    }


def kink_fixture():
    first = nn.Sequential(
        _linear([[0.0, 1.0], [-1.0, 1.0]], [0.0, 0.0]),
        nn.ReLU(),
        _linear([[1.0, -1.0]], [0.0]),
    ).eval()
    mirrored = nn.Sequential(
        _linear([[1.0, 0.0], [1.0, -1.0]], [0.0, 0.0]),
        nn.ReLU(),
        _linear([[1.0, -1.0]], [0.0]),
    ).eval()
    return {
        "models": {"x1 route": first, "x0 route": mirrored},
        "query": t.tensor([[1.0, 1.0]], dtype=DTYPE),
        "background": t.tensor([[0.0, 0.0]], dtype=DTYPE),
        "binary_inputs": t.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=DTYPE
        ),
        "expected_binary": t.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE),
        "exact_shapley": t.tensor([[0.5, 0.5]], dtype=DTYPE),
    }


def replacement_fixture():
    contributions = t.tensor([3.0, -2.0, 0.0], dtype=DTYPE)
    masks = t.arange(8, dtype=t.int64)
    present = (masks[:, None].bitwise_and(1 << t.arange(3)) != 0).to(DTYPE)
    return {
        "values": (10.0 + present @ contributions).reshape(1, -1),
        "phi": contributions.reshape(1, -1),
        "all_orders": t.tensor(
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
            dtype=t.int64,
        ),
    }


def diagnose_path_vs_coalition(gauss_legendre_pairs_fn, ranked_coalition_curves_fn):
    threshold = threshold_fixture()
    path_256 = gauss_legendre_pairs_fn(
        threshold["model"], threshold["query"], threshold["background"], 256
    ).mean(dim=1)
    path_512 = gauss_legendre_pairs_fn(
        threshold["model"], threshold["query"], threshold["background"], 512
    ).mean(dim=1)
    threshold_orders = t.tensor([[0, 1], [1, 0]], dtype=t.int64)
    threshold_curves = [
        ranked_coalition_curves_fn(
            threshold["coalition_values"], path_512, order.reshape(1, -1)
        )
        for order in threshold_orders
    ]

    kink = kink_fixture()
    kink_phi, kink_outputs = {}, {}
    for name, model in kink["models"].items():
        kink_phi[name] = gauss_legendre_pairs_fn(
            model, kink["query"], kink["background"], 32
        ).mean(dim=1)
        kink_outputs[name] = _prediction(model, kink["binary_inputs"])

    replacement = replacement_fixture()
    local_curve = ranked_coalition_curves_fn(replacement["values"], replacement["phi"])
    random_curves = [
        ranked_coalition_curves_fn(
            replacement["values"], replacement["phi"], order.reshape(1, -1)
        )
        for order in replacement["all_orders"]
    ]
    return {
        "threshold": {
            "path_256": path_256,
            "path_512": path_512,
            "quadrature_drift": (path_512 - path_256).abs(),
            "analytic_path": threshold["analytic_path"],
            "exact_shapley": threshold["exact_shapley"],
            "necessity_drop": threshold["necessity_drop"],
            "orders": threshold_orders,
            "curves": threshold_curves,
        },
        "kink": {
            "binary_inputs": kink["binary_inputs"],
            "expected_binary": kink["expected_binary"],
            "outputs": kink_outputs,
            "path_phi": kink_phi,
            "exact_shapley": kink["exact_shapley"],
        },
        "replacement": {
            "values": replacement["values"],
            "phi": replacement["phi"],
            "local_curve": local_curve,
            "random_curves": random_curves,
        },
    }


def threshold_play(
    methods,
    weights: Sequence[float] = (2.0, 1.0),
    bias: float = -2.0,
    query: Sequence[float] = (1.0, 1.0),
    reference: Sequence[float] = (0.0, 0.0),
    samples: int = 512,
    seed: int = 161510,
):
    schedule_fn = _method(methods, "sample_path_schedule")
    sample_fn = _method(methods, "expected_gradient_samples")
    summarize_fn = _method(methods, "summarize_expected_gradients")
    quadrature_fn = _method(methods, "gauss_legendre_pairs")
    curves_fn = _method(methods, "ranked_coalition_curves")
    model = _threshold_model(weights, bias)
    query_tensor = t.as_tensor(query, dtype=DTYPE).reshape(1, -1)
    reference_tensor = t.as_tensor(reference, dtype=DTYPE).reshape(1, -1)
    if query_tensor.shape != reference_tensor.shape:
        raise ValueError("query and reference must have the same feature count.")
    if query_tensor.shape[1] != len(weights):
        raise ValueError("weights must contain one value per query feature.")
    indices, alpha = schedule_fn(t.tensor([0], dtype=t.int64), 1, samples, seed)
    draws = sample_fn(model, query_tensor, reference_tensor, indices, alpha)
    mean, standard_error = summarize_fn(draws, (samples,))
    quadrature = quadrature_fn(model, query_tensor, reference_tensor, 512).mean(dim=1)

    feature_count = query_tensor.shape[1]
    if feature_count != 2:
        raise ValueError("The editable teaching play currently requires exactly two features.")
    masks = t.arange(4, dtype=t.int64)
    present = masks[:, None].bitwise_and(1 << t.arange(2)) != 0
    hybrids = t.where(present, query_tensor, reference_tensor)
    values = _prediction(model, hybrids).reshape(1, -1)
    exact = _two_feature_shapley(values).reshape(1, -1)
    curves = {
        "expected_gradients": curves_fn(values, mean[0]),
        "exact_shapley": curves_fn(values, exact),
    }
    return {
        "weights": t.as_tensor(weights, dtype=DTYPE),
        "bias": float(bias),
        "query": query_tensor,
        "reference": reference_tensor,
        "alpha": alpha,
        "draws": draws,
        "mean": mean[0],
        "standard_error": standard_error[0],
        "quadrature": quadrature,
        "coalition_values": values,
        "exact_shapley": exact,
        "prediction": _prediction(model, query_tensor)[0],
        "base": _prediction(model, reference_tensor)[0],
        "curves": curves,
        "samples": samples,
        "seed": seed,
    }


def run_diagnostics(methods=None):
    schedule = diagnose_schedule(_method(methods, "sample_path_schedule"))
    interpolation = diagnose_interpolation(_method(methods, "interpolate_paths"))
    gradients = diagnose_gradients(_method(methods, "input_gradients"))
    samples = diagnose_expected_gradient_samples(
        _method(methods, "expected_gradient_samples"),
        _method(methods, "sample_path_schedule"),
    )
    convergence = diagnose_convergence(
        _method(methods, "sample_path_schedule"),
        _method(methods, "expected_gradient_samples"),
        _method(methods, "summarize_expected_gradients"),
    )
    estimator = diagnose_expected_gradients(
        _method(methods, "expected_gradients"),
        _method(methods, "sample_path_schedule"),
    )
    quadrature = diagnose_quadrature(
        _method(methods, "gauss_legendre_pairs"),
        _method(methods, "input_gradients"),
    )
    background = diagnose_background_average(_method(methods, "gauss_legendre_pairs"))
    noise = diagnose_noise_target(
        _method(methods, "noisy_expected_gradient_samples"),
        _method(methods, "expected_gradient_samples"),
        _method(methods, "summarize_expected_gradients"),
    )
    game = diagnose_path_vs_coalition(
        _method(methods, "gauss_legendre_pairs"),
        _method(methods, "ranked_coalition_curves"),
    )
    namespace = methods
    if namespace is None:
        from . import gradientshap_methods

        namespace = gradientshap_methods
    play = threshold_play(namespace)
    return {
        "schedule": schedule,
        "interpolation": interpolation,
        "gradients": gradients,
        "samples": samples,
        "convergence": convergence,
        "estimator": estimator,
        "quadrature": quadrature,
        "background": background,
        "noise": noise,
        "game": game,
        "threshold_play": play,
    }
