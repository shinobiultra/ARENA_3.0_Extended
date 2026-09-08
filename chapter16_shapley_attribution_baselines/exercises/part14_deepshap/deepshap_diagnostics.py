"""Learner-visible DeepSHAP diagnostics with independent numerical oracles.

Each ``diagnose_*`` function resolves only methods already taught at that point
in the lesson.  There is deliberately no aggregate runner: an early notebook
cell never depends on a student function from a later exercise.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

DTYPE = torch.float64


def _method(methods, name):
    """Resolve a taught function from a module, namespace, or mapping."""
    if methods is None:
        local = globals().get(name)
        if callable(local):
            return local
        from . import deepshap_methods

        return getattr(deepshap_methods, name)
    if hasattr(methods, name):
        return getattr(methods, name)
    return methods[name]


def _linear(weight, bias):
    """Construct one deterministic float64 Linear layer from literal arrays."""
    weight = torch.as_tensor(weight, dtype=DTYPE)
    bias = torch.as_tensor(bias, dtype=DTYPE).reshape(-1)
    layer = nn.Linear(weight.shape[1], weight.shape[0], dtype=DTYPE)
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer


def _prediction(model, rows):
    with torch.no_grad():
        return model(rows).reshape(-1)


def _max_abs(left, right):
    difference = torch.as_tensor(left).detach() - torch.as_tensor(right).detach()
    return float(difference.abs().max())


def affine_fixture():
    """Return affine, dummy-input and constant controls with exact answers."""
    query = torch.tensor([[4.0, 2.0]], dtype=DTYPE)
    reference = torch.tensor([1.0, -1.0], dtype=DTYPE)
    affine = nn.Sequential(_linear([[2.0, -3.0]], [5.0])).eval()

    dummy_query = torch.tensor([[4.0, 2.0, 9.0]], dtype=DTYPE)
    dummy_reference = torch.tensor([1.0, -1.0, -4.0], dtype=DTYPE)
    dummy = nn.Sequential(_linear([[2.0, -3.0, 0.0]], [5.0])).eval()
    constant = nn.Sequential(_linear([[0.0, 0.0]], [7.0])).eval()
    return {
        "affine": {
            "model": affine,
            "query": query,
            "reference": reference,
            "expected_phi": torch.tensor([[6.0, -9.0]], dtype=DTYPE),
            "expected_base": torch.tensor(10.0, dtype=DTYPE),
            "expected_prediction": torch.tensor(7.0, dtype=DTYPE),
        },
        "dummy": {
            "model": dummy,
            "query": dummy_query,
            "reference": dummy_reference,
            "expected_phi": torch.tensor([[6.0, -9.0, 0.0]], dtype=DTYPE),
            "expected_base": torch.tensor(10.0, dtype=DTYPE),
            "expected_prediction": torch.tensor(7.0, dtype=DTYPE),
        },
        "constant": {
            "model": constant,
            "query": query,
            "reference": reference,
            "expected_phi": torch.zeros((1, 2), dtype=DTYPE),
            "expected_base": torch.tensor(7.0, dtype=DTYPE),
            "expected_prediction": torch.tensor(7.0, dtype=DTYPE),
        },
    }


def threshold_fixture(
    weights: Sequence[float] = (2.0, 1.0),
    bias: float = -2.0,
    query: Sequence[float] = (1.0, 1.0),
    reference: Sequence[float] = (0.0, 0.0),
):
    """Build ``ReLU(w @ x + bias)`` as a supported explicit Sequential."""
    weights = torch.as_tensor(weights, dtype=DTYPE).reshape(1, -1)
    query = torch.as_tensor(query, dtype=DTYPE).reshape(1, -1)
    reference = torch.as_tensor(reference, dtype=DTYPE).reshape(-1)
    if query.shape[1] != weights.shape[1] or reference.shape != query.shape[1:]:
        raise ValueError("weights, query and reference must have one common feature size.")
    model = nn.Sequential(
        _linear(weights, [bias]),
        nn.ReLU(),
        _linear([[1.0]], [0.0]),
    ).eval()
    return {"model": model, "query": query, "reference": reference}


def near_zero_fixture():
    """Return the named ReLU crossing inside the 1e-6 fallback branch."""
    model = nn.Sequential(
        _linear([[1.0]], [0.0]),
        nn.ReLU(),
        _linear([[1.0]], [0.0]),
    ).eval()
    return {
        "model": model,
        "query": torch.tensor([[4e-7]], dtype=DTYPE),
        "reference": torch.tensor([-4e-7], dtype=DTYPE),
        "epsilon": 1e-6,
        "expected_primary_phi": torch.tensor([[0.0]], dtype=DTYPE),
        "expected_exact_secant_phi": torch.tensor([[4e-7]], dtype=DTYPE),
    }


def background_average_fixture():
    """Return a nonlinear case where averaging references and inputs differ."""
    fixture = near_zero_fixture()
    fixture.update(
        {
            "query": torch.tensor([[2.0]], dtype=DTYPE),
            "background": torch.tensor([[-2.0], [1.0]], dtype=DTYPE),
            "expected_pair_phi": torch.tensor([[2.0], [1.0]], dtype=DTYPE),
            "expected_mean_phi": torch.tensor([[1.5]], dtype=DTYPE),
            "expected_base": torch.tensor(0.5, dtype=DTYPE),
            "expected_mean_reference_phi": torch.tensor([[2.0]], dtype=DTYPE),
            "expected_mean_reference_base": torch.tensor(0.0, dtype=DTYPE),
        }
    )
    fixture.pop("reference")
    fixture.pop("epsilon")
    fixture.pop("expected_primary_phi")
    fixture.pop("expected_exact_secant_phi")
    return fixture


def min_fixture():
    """Return two graph forms of min(x0, x1) on nonnegative inputs."""
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
        "query": torch.tensor([[1.0, 1.0]], dtype=DTYPE),
        "reference": torch.tensor([0.0, 0.0], dtype=DTYPE),
        "binary_inputs": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=DTYPE
        ),
        "expected_binary": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE),
        "expected_exact": torch.tensor([[0.5, 0.5]], dtype=DTYPE),
        "expected_deep": {
            "x1 route": torch.tensor([[0.0, 1.0]], dtype=DTYPE),
            "x0 route": torch.tensor([[1.0, 0.0]], dtype=DTYPE),
        },
    }


def replacement_fixture():
    """Return an additive three-player game with one exact dummy feature."""
    contributions = torch.tensor([3.0, -2.0, 0.0], dtype=DTYPE)
    masks = torch.tensor(
        [[bool(mask & (1 << feature)) for feature in range(3)] for mask in range(8)]
    )
    values = 10.0 + (masks.to(DTYPE) * contributions).sum(dim=1)
    return {
        "masks": masks,
        "values": values[None],
        "phi": contributions[None],
        "expected_order": torch.tensor([[0, 2, 1]], dtype=torch.long),
        "expected_insertion": torch.tensor([[10.0, 13.0, 13.0, 11.0]], dtype=DTYPE),
        "expected_deletion": torch.tensor([[11.0, 8.0, 8.0, 10.0]], dtype=DTYPE),
    }


def diagnose_rescale_limits(methods):
    """Exercise the secant, exact-zero and named small-delta branches."""
    rescale_slope = _method(methods, "rescale_slope")
    reference = torch.tensor([-1.0, 0.5, -4e-7], dtype=DTYPE)
    actual = torch.tensor([1.0, 0.5, 4e-7], dtype=DTYPE)
    primary = rescale_slope(actual, reference, epsilon=1e-6)
    exact_nonzero = rescale_slope(actual, reference, epsilon=0.0)
    expected_primary = torch.tensor([0.5, 1.0, 0.0], dtype=DTYPE)
    expected_exact = torch.tensor([0.5, 1.0, 0.5], dtype=DTYPE)
    grid = torch.linspace(-2.0, 2.0, 161, dtype=DTYPE)
    grid_reference = torch.full_like(grid, -1.0)
    return {
        "reference": reference,
        "actual": actual,
        "epsilon": 1e-6,
        "primary": primary,
        "exact_nonzero": exact_nonzero,
        "expected_primary": expected_primary,
        "expected_exact_nonzero": expected_exact,
        "curve": {
            "actual": grid,
            "reference": grid_reference,
            "slope": rescale_slope(grid, grid_reference, epsilon=1e-6),
        },
        "primary_max_abs_error": _max_abs(primary, expected_primary),
        "exact_nonzero_max_abs_error": _max_abs(exact_nonzero, expected_exact),
    }


def diagnose_near_zero(methods):
    """Expose the expected local-accuracy loss from the named epsilon branch."""
    single_reference = _method(methods, "single_reference")
    fixture = near_zero_fixture()
    model, query, reference = (
        fixture["model"],
        fixture["query"],
        fixture["reference"],
    )
    primary = single_reference(model, query, reference, epsilon=fixture["epsilon"])
    exact_secant = single_reference(model, query, reference, epsilon=0.0)
    prediction = _prediction(model, query)[0]
    base = _prediction(model, reference[None])[0]
    delta = prediction - base
    primary_residual = delta - primary.sum()
    exact_residual = delta - exact_secant.sum()
    return {
        "query": query,
        "reference": reference,
        "epsilon": fixture["epsilon"],
        "prediction": prediction,
        "base": base,
        "prediction_delta": delta,
        "primary_phi": primary,
        "exact_secant_phi": exact_secant,
        "expected_primary_phi": fixture["expected_primary_phi"],
        "expected_exact_secant_phi": fixture["expected_exact_secant_phi"],
        "primary_completeness_residual": primary_residual,
        "exact_secant_completeness_residual": exact_residual,
        "expected_primary_residual": torch.tensor(4e-7, dtype=DTYPE),
        "primary_phi_max_abs_error": _max_abs(primary, fixture["expected_primary_phi"]),
        "exact_secant_phi_max_abs_error": _max_abs(
            exact_secant, fixture["expected_exact_secant_phi"]
        ),
    }


def diagnose_affine_controls(methods):
    """Check exact affine credit, an unused input, and a constant network."""
    single_reference = _method(methods, "single_reference")
    answer = {}
    for name, fixture in affine_fixture().items():
        phi = single_reference(fixture["model"], fixture["query"], fixture["reference"])
        prediction = _prediction(fixture["model"], fixture["query"])[0]
        base = _prediction(fixture["model"], fixture["reference"][None])[0]
        answer[name] = {
            "query": fixture["query"],
            "reference": fixture["reference"],
            "phi": phi,
            "expected_phi": fixture["expected_phi"],
            "prediction": prediction,
            "expected_prediction": fixture["expected_prediction"],
            "base": base,
            "expected_base": fixture["expected_base"],
            "completeness_residual": prediction - base - phi.sum(),
            "phi_max_abs_error": _max_abs(phi, fixture["expected_phi"]),
        }
    answer["dummy_feature_max_abs"] = float(answer["dummy"]["phi"][0, 2].detach().abs())
    return answer


def diagnose_background_average(methods):
    """Compare averaging attributions with explaining the mean reference."""
    single_reference = _method(methods, "single_reference")
    background_deepshap = _method(methods, "background_deepshap")
    fixture = background_average_fixture()
    model, query, background = fixture["model"], fixture["query"], fixture["background"]
    base, phi = background_deepshap(model, query, background)
    pair_phi = torch.stack(
        [single_reference(model, query, reference) for reference in background], dim=1
    ).squeeze(0)
    mean_reference = background.mean(dim=0)
    mean_reference_phi = single_reference(model, query, mean_reference)
    mean_reference_base = _prediction(model, mean_reference[None])[0]
    prediction = _prediction(model, query)[0]
    return {
        "query": query,
        "background": background,
        "mean_reference": mean_reference,
        "pair_phi": pair_phi,
        "expected_pair_phi": fixture["expected_pair_phi"],
        "phi": phi,
        "expected_phi": fixture["expected_mean_phi"],
        "base": base,
        "expected_base": fixture["expected_base"],
        "mean_reference_phi": mean_reference_phi,
        "expected_mean_reference_phi": fixture["expected_mean_reference_phi"],
        "mean_reference_base": mean_reference_base,
        "expected_mean_reference_base": fixture["expected_mean_reference_base"],
        "prediction": prediction,
        "averaging_gap": mean_reference_phi - phi,
        "phi_max_abs_error": _max_abs(phi, fixture["expected_mean_phi"]),
        "pair_phi_max_abs_error": _max_abs(pair_phi, fixture["expected_pair_phi"]),
    }


def diagnose_threshold(methods):
    """Show that Rescale can be complete while differing from exact Shapley."""
    single_reference = _method(methods, "single_reference")
    coalition_values = _method(methods, "coalition_values")
    exact_shapley = _method(methods, "exact_shapley")
    gradient_delta = _method(methods, "gradient_delta")
    fixture = threshold_fixture()
    model, query, reference = (
        fixture["model"],
        fixture["query"],
        fixture["reference"],
    )
    masks = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.bool)
    game = coalition_values(model, query, reference[None], masks)
    deep = single_reference(model, query, reference)
    exact = exact_shapley(game)
    gradient = gradient_delta(model, query, reference[None])
    prediction, base = game[0, -1], game[0, 0]
    expected_game = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DTYPE)
    expected_exact = torch.tensor([[0.5, 0.5]], dtype=DTYPE)
    expected_deep = torch.tensor([[2.0 / 3.0, 1.0 / 3.0]], dtype=DTYPE)
    expected_gradient = torch.tensor([[2.0, 1.0]], dtype=DTYPE)
    full_minus_feature = torch.stack([prediction - game[0, 2], prediction - game[0, 1]])
    return {
        "query": query,
        "reference": reference,
        "masks": masks,
        "game": game,
        "expected_game": expected_game,
        "deep": deep,
        "expected_deep": expected_deep,
        "exact": exact,
        "expected_exact": expected_exact,
        "gradient": gradient,
        "expected_gradient": expected_gradient,
        "full_minus_feature": full_minus_feature,
        "expected_full_minus_feature": torch.ones(2, dtype=DTYPE),
        "prediction": prediction,
        "base": base,
        "prediction_delta": prediction - base,
        "deep_completeness_residual": prediction - base - deep.sum(),
        "exact_completeness_residual": prediction - base - exact.sum(),
        "gradient_completeness_residual": prediction - base - gradient.sum(),
        "deep_exact_max_abs_error": _max_abs(deep, exact),
        "fixture_max_abs_error": max(
            _max_abs(game, expected_game),
            _max_abs(deep, expected_deep),
            _max_abs(exact, expected_exact),
            _max_abs(gradient, expected_gradient),
        ),
    }


def diagnose_cancellation(methods):
    """Compare two equivalent min graphs and the bounded RevealCancel repair."""
    single_reference = _method(methods, "single_reference")
    coalition_values = _method(methods, "coalition_values")
    exact_shapley = _method(methods, "exact_shapley")
    reveal_cancel = _method(methods, "reveal_cancel_groups")
    fixture = min_fixture()
    query, reference = fixture["query"], fixture["reference"]
    masks = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.bool)
    positive, negative = reveal_cancel(
        torch.relu,
        torch.tensor(0.0, dtype=DTYPE),
        torch.tensor(1.0, dtype=DTYPE),
        torch.tensor(-1.0, dtype=DTYPE),
    )
    reveal_phi = {
        "x1 route": torch.stack([-negative, 1.0 - positive])[None],
        "x0 route": torch.stack([1.0 - positive, -negative])[None],
    }
    models = {}
    for name, model in fixture["models"].items():
        game = coalition_values(model, query, reference[None], masks)
        deep = single_reference(model, query, reference)
        exact = exact_shapley(game)
        binary_prediction = _prediction(model, fixture["binary_inputs"])
        models[name] = {
            "binary_prediction": binary_prediction,
            "expected_binary": fixture["expected_binary"],
            "game": game,
            "deep": deep,
            "expected_deep": fixture["expected_deep"][name],
            "exact": exact,
            "expected_exact": fixture["expected_exact"],
            "reveal_cancel": reveal_phi[name],
            "binary_max_abs_error": _max_abs(binary_prediction, fixture["expected_binary"]),
            "deep_max_abs_error": _max_abs(deep, fixture["expected_deep"][name]),
            "exact_max_abs_error": _max_abs(exact, fixture["expected_exact"]),
            "reveal_cancel_max_abs_error": _max_abs(
                reveal_phi[name], fixture["expected_exact"]
            ),
        }
    return {
        "query": query,
        "reference": reference,
        "masks": masks,
        "models": models,
        "group_allocation": {"positive": positive, "negative": negative},
        "computational_domain": "nonnegative inputs",
        "deep_implementation_gap": models["x0 route"]["deep"] - models["x1 route"]["deep"],
    }


def diagnose_replacement_control(methods):
    """Check ranked insertion/deletion endpoints and a zero-credit feature."""
    ranked_curves = _method(methods, "ranked_replacement_curves")
    fixture = replacement_fixture()
    result = ranked_curves(fixture["values"], fixture["phi"])
    return {
        **fixture,
        "result": result,
        "order_max_abs_error": _max_abs(result["orders"], fixture["expected_order"]),
        "insertion_max_abs_error": _max_abs(result["insertion"], fixture["expected_insertion"]),
        "deletion_max_abs_error": _max_abs(result["deletion"], fixture["expected_deletion"]),
        "dummy_insertion_change": result["insertion"][0, 2] - result["insertion"][0, 1],
        "dummy_deletion_change": result["deletion"][0, 2] - result["deletion"][0, 1],
    }


def threshold_play(
    methods,
    weights: Sequence[float] = (2.0, 1.0),
    bias: float = -2.0,
    query: Sequence[float] = (1.0, 1.0),
    reference: Sequence[float] = (0.0, 0.0),
    epsilon: float = 1e-6,
):
    """Recompute the editable gate with the learner's implemented methods."""
    single_reference = _method(methods, "single_reference")
    coalition_values = _method(methods, "coalition_values")
    exact_shapley = _method(methods, "exact_shapley")
    gradient_delta = _method(methods, "gradient_delta")
    fixture = threshold_fixture(weights, bias, query, reference)
    model, query_tensor, reference_tensor = (
        fixture["model"],
        fixture["query"],
        fixture["reference"],
    )
    dimension = query_tensor.shape[1]
    masks = torch.tensor(
        [
            [bool(mask & (1 << feature)) for feature in range(dimension)]
            for mask in range(2**dimension)
        ]
    )
    game = coalition_values(model, query_tensor, reference_tensor[None], masks)
    deep = single_reference(model, query_tensor, reference_tensor, epsilon=epsilon)
    exact = exact_shapley(game)
    gradient = gradient_delta(model, query_tensor, reference_tensor[None])
    return {
        "weights": torch.as_tensor(weights, dtype=DTYPE),
        "bias": float(bias),
        "query": query_tensor,
        "reference": reference_tensor,
        "epsilon": float(epsilon),
        "masks": masks,
        "game": game,
        "prediction": game[0, -1],
        "base": game[0, 0],
        "deep": deep,
        "exact": exact,
        "gradient": gradient,
        "deep_completeness_residual": game[0, -1] - game[0, 0] - deep.sum(),
        "deep_exact_max_abs_error": _max_abs(deep, exact),
    }


def run_diagnostics(methods=None, include_timing=True):
    """Collect every checkpoint after all nine methods have been defined.

    The lesson calls each diagnostic separately at its own checkpoint.  This
    convenience runner is reserved for the final native/reproduction harness,
    where the complete student-method namespace is already available.  There
    is no timing study in this section, so the separately persisted timing
    branch is intentionally empty.
    """

    def detach_tree(value):
        if isinstance(value, Tensor):
            return value.detach()
        if isinstance(value, dict):
            return {key: detach_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [detach_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(detach_tree(item) for item in value)
        return value

    mechanism = {
        "rescale_limits": diagnose_rescale_limits(methods),
        "near_zero": diagnose_near_zero(methods),
        "affine_controls": diagnose_affine_controls(methods),
        "background_average": diagnose_background_average(methods),
        "threshold": diagnose_threshold(methods),
        "cancellation": diagnose_cancellation(methods),
        "replacement_control": diagnose_replacement_control(methods),
        "threshold_play": threshold_play(methods),
    }
    return {"mechanism": detach_tree(mechanism), "timing": {} if include_timing else {}}
