"""Analytic and staged-namespace tests for GradientSHAP diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

from . import gradientshap_diagnostics as diagnostics
from . import gradientshap_methods as methods


METHOD_NAMES = (
    "sample_path_schedule",
    "interpolate_paths",
    "input_gradients",
    "expected_gradient_samples",
    "summarize_expected_gradients",
    "expected_gradients",
    "gauss_legendre_pairs",
    "noisy_expected_gradient_samples",
    "ranked_coalition_curves",
)


@pytest.fixture(scope="module")
def analytic_results():
    namespace = SimpleNamespace(**{name: getattr(methods, name) for name in METHOD_NAMES})
    return diagnostics.run_diagnostics(namespace)


def test_analytic_diagnostics_retain_independent_controls(analytic_results):
    result = analytic_results
    assert result["schedule"]["prefix_match"], (
        "The K=8 schedule must be a literal prefix of the K=32 schedule."
    )
    assert result["schedule"]["reorder_match"], (
        "Reordering source rows must preserve each source-owned random stream."
    )
    assert result["schedule"]["single_row_match"], (
        "A source schedule must not depend on which other rows share its batch."
    )
    assert result["interpolation"]["max_abs_error"] < 1e-12, (
        "Every sampled point must stay on its selected donor-to-query path."
    )
    assert result["gradients"]["affine_max_abs_error"] < 1e-12, (
        "Input gradients must match the literal affine oracle."
    )
    assert result["gradients"]["quadratic_max_abs_error"] < 1e-12, (
        "Input gradients must follow the changing 2z quadratic oracle."
    )
    assert result["samples"]["relu_max_abs_error"] == 0, (
        "The ReLU path fixture must retain its exact zero-or-two draw distribution."
    )
    assert result["samples"]["quadratic_max_abs_error"] < 1e-12, (
        "Quadratic per-draw contributions must equal 2*alpha."
    )
    convergence = result["convergence"]
    assert bool((convergence["standard_errors"] >= 0).all()), (
        "Every ordinary Monte Carlo standard error must be nonnegative."
    )
    assert result["estimator"]["affine_max_abs_error"] < 1e-12, (
        "Balanced affine Expected Gradients must match its exact attribution."
    )
    assert result["estimator"]["dummy_max_abs"] == 0, (
        "A feature ignored by the model must receive zero attribution."
    )
    assert result["estimator"]["constant_max_abs"] == 0, (
        "A constant model must receive an all-zero attribution vector."
    )
    assert abs(float(result["estimator"]["finite_relu"]["residual"])) > 0, (
        "The finite ReLU fixture must keep its deliberately nonzero sampling residual."
    )
    assert result["quadrature"]["quadratic_error"] < 1e-12, (
        "Two-node quadrature must integrate the quadratic path control."
    )
    assert bool((result["quadrature"]["successive_drift"][1:] > 0).any()), (
        "ReLU quadrature doubling must expose finite numerical drift."
    )
    t.testing.assert_close(
        result["background"]["pair_512"],
        result["background"]["pair_expected"],
        atol=1e-3,
        rtol=0,
        msg="Donor-wise quadrature must approach both analytic path attributions.",
    )
    assert abs(
        float(result["background"]["mean_pair"])
        - float(result["background"]["mean_reference"])
    ) > 0.25, "Averaging nonlinear explanations must remain distinct from using the mean donor."
    assert result["noise"]["zero_noise_equal"], (
        "The supplied zero-noise schedule must exactly equal the clean estimator."
    )
    noise_error = result["noise"]["smoothed_residual"].abs()
    assert bool((noise_error <= 4 * result["noise"]["standard_errors"] + 1e-12).all()), (
        "The fixed large-sample noisy ReLU run should resolve its analytic smoothed target."
    )
    threshold = result["game"]["threshold"]
    assert not t.allclose(threshold["analytic_path"], threshold["exact_shapley"]), (
        "The threshold fixture must preserve complete but different path and coalition credit."
    )
    kink = result["game"]["kink"]
    for output in kink["outputs"].values():
        t.testing.assert_close(
            output,
            kink["expected_binary"],
            msg="Both exact-kink graphs must agree on every binary coalition input.",
        )
    path_values = list(kink["path_phi"].values())
    assert not t.allclose(path_values[0], path_values[1]), (
        "Equivalent binary functions may follow different graph-form derivatives at an exact kink."
    )
    replacement = result["game"]["replacement"]
    for curve in replacement["random_curves"]:
        assert curve["insertion"][0, 0] == replacement["values"][0, 0], (
            "Every replacement order must retain the empty-game endpoint."
        )
        assert curve["insertion"][0, -1] == replacement["values"][0, -1], (
            "Every replacement order must retain the full-game endpoint."
        )


def test_staged_diagnostics_accept_the_visible_learner_namespaces():
    mapping = {name: getattr(methods, name) for name in METHOD_NAMES}
    play_from_mapping = diagnostics.threshold_play(mapping, samples=32)
    play_from_namespace = diagnostics.threshold_play(SimpleNamespace(**mapping), samples=32)
    t.testing.assert_close(
        play_from_mapping["mean"],
        play_from_namespace["mean"],
        rtol=0,
        atol=0,
        msg="globals()-style mappings and the lesson SimpleNamespace must call the same methods.",
    )
    with pytest.raises(TypeError, match="must provide a callable"):
        diagnostics.threshold_play({"sample_path_schedule": methods.sample_path_schedule})


def test_every_aggregate_diagnostic_uses_the_supplied_learner_method():
    calls = {name: 0 for name in METHOD_NAMES}

    def counted(name):
        function = getattr(methods, name)

        def wrapper(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return wrapper

    namespace = SimpleNamespace(**{name: counted(name) for name in METHOD_NAMES})
    diagnostics.run_diagnostics(namespace)
    assert all(counts > 0 for counts in calls.values()), (
        "The final aggregate diagnostic must use every function from the learner namespace."
    )
