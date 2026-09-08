"""No-fit tests for the pinned GradientSHAP package-reference adapters."""

from __future__ import annotations

from collections import OrderedDict
import copy
import random

import numpy as np
import pytest
import torch as t
from torch import nn

from . import gradientshap_methods as methods
from . import gradientshap_reference as reference


@pytest.fixture(scope="module", autouse=True)
def _benchmark_cpu_threads():
    """Use the retained benchmark's declared four-thread execution setting."""
    previous = t.get_num_threads()
    t.set_num_threads(4)
    try:
        yield
    finally:
        t.set_num_threads(previous)


def _affine(features: int = 2) -> nn.Sequential:
    model = nn.Sequential(nn.Linear(features, 1, dtype=t.float64))
    with t.no_grad():
        if features == 2:
            weight = t.tensor([[2.0, -3.0]], dtype=t.float64)
        else:
            weight = t.arange(1, features + 1, dtype=t.float64)[None] / features
        model[0].weight.copy_(weight)
        model[0].bias.fill_(5.0)
    return model


def _probe_inputs() -> tuple[t.Tensor, t.Tensor]:
    queries = t.tensor([[4.0, 2.0], [1.0, 1.0]], dtype=t.float64)
    background = t.tensor(
        [[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]], dtype=t.float64
    )
    return queries, background


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _state_snapshot(model: nn.Module) -> dict:
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": t.random.get_rng_state().clone(),
        "dtype": t.get_default_dtype(),
        "modes": tuple(module.training for module in model.modules()),
        "state": {name: value.detach().clone() for name, value in model.state_dict().items()},
        "grad_objects": tuple(parameter.grad for parameter in model.parameters()),
        "grad_values": tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ),
    }


def _assert_state_restored(model: nn.Module, before: dict) -> None:
    assert random.getstate() == before["python"], "Python RNG state was not restored."
    assert _numpy_state_equal(
        np.random.get_state(), before["numpy"]
    ), "NumPy RNG state was not restored."
    assert t.equal(
        t.random.get_rng_state(), before["torch"]
    ), "Torch RNG state was not restored."
    assert t.get_default_dtype() == before["dtype"], "Torch default dtype was not restored."
    assert tuple(module.training for module in model.modules()) == before[
        "modes"
    ], "Per-module training modes were not restored."
    for name, expected in before["state"].items():
        assert t.equal(model.state_dict()[name], expected), f"Model state changed at {name}."
    for parameter, original, expected in zip(
        model.parameters(), before["grad_objects"], before["grad_values"], strict=True
    ):
        assert parameter.grad is original, "The original gradient object identity was not restored."
        if expected is not None:
            assert t.equal(parameter.grad, expected), "A parameter gradient value changed."


def _seeded_untrained_benchmark() -> tuple[nn.Sequential, t.Tensor, t.Tensor]:
    state = t.random.get_rng_state().clone()
    try:
        t.manual_seed(161600)
        model = nn.Sequential(
            nn.Linear(8, 32, dtype=t.float64),
            nn.ReLU(),
            nn.Linear(32, 32, dtype=t.float64),
            nn.ReLU(),
            nn.Linear(32, 1, dtype=t.float64),
        )
    finally:
        t.random.set_rng_state(state)
    generator = t.Generator(device="cpu").manual_seed(161601)
    background = t.randn((64, 8), dtype=t.float64, generator=generator)
    queries = t.randn((272, 8), dtype=t.float64, generator=generator)
    return model.eval(), queries, background


def test_reference_environment_is_exactly_source_pinned():
    environment = reference.reference_environment()
    assert environment["versions"] == reference.EXPECTED_VERSIONS
    assert environment["installed_source_sha256"] == reference.EXPECTED_SOURCE_SHA256
    assert environment["cached_source_sha256"] == reference.EXPECTED_SOURCE_SHA256
    assert len(environment["source_evidence_sha256"]) == 64


def test_package_draws_use_separate_numpy_and_torch_streams_without_global_rng():
    random.seed(700)
    np.random.seed(701)
    t.manual_seed(702)
    before = (random.getstate(), copy.deepcopy(np.random.get_state()), t.random.get_rng_state())

    shap_a = reference.shap_draw_table(
        3, 2, 5, 19, numpy_seed=11, torch_seed=12, local_smoothing=0.2
    )
    shap_other_torch = reference.shap_draw_table(
        3, 2, 5, 19, numpy_seed=11, torch_seed=13, local_smoothing=0.2
    )
    shap_other_numpy = reference.shap_draw_table(
        3, 2, 5, 19, numpy_seed=14, torch_seed=12, local_smoothing=0.2
    )
    captum = reference.captum_draw_table(
        3, 2, 5, 19, numpy_seed=11, torch_seed=12, stdev=0.2
    )

    np.testing.assert_array_equal(shap_a["donor_indices"], shap_other_torch["donor_indices"])
    np.testing.assert_array_equal(shap_a["alphas"], shap_other_torch["alphas"])
    assert not np.array_equal(shap_a["noise"], shap_other_torch["noise"])
    np.testing.assert_array_equal(shap_a["noise"], shap_other_numpy["noise"])
    assert not np.array_equal(shap_a["donor_indices"], shap_other_numpy["donor_indices"])
    assert not np.array_equal(shap_a["alphas"], shap_other_numpy["alphas"])
    assert not np.array_equal(shap_a["donor_indices"], captum["donor_indices"])
    assert not np.array_equal(shap_a["alphas"], captum["alphas"])

    assert random.getstate() == before[0]
    assert _numpy_state_equal(np.random.get_state(), before[1])
    assert t.equal(t.random.get_rng_state(), before[2])


def test_shap_probe_regression_normalizes_scalar_shape_and_names_variance_proxy():
    model = _affine().train()
    queries, background = _probe_inputs()
    old_dtype = t.get_default_dtype()
    try:
        t.set_default_dtype(t.float32)
        result = reference.shap_same_draw_reference(
            model,
            queries,
            background,
            methods,
            samples=37,
            numpy_seed=161590,
            torch_seed=161591,
            batch_size=11,
        )
        assert t.get_default_dtype() == t.float32
    finally:
        t.set_default_dtype(old_dtype)

    expected_phi = np.array(
        [[6.0, -6.162162162162162], [-0.16216216216216217, -3.608108108108108]]
    )
    np.testing.assert_array_equal(result["author_phi"], expected_phi)
    np.testing.assert_array_equal(result["student_phi"], expected_phi)
    assert result["author_phi_raw"].shape == (2, 2, 1)
    assert result["parity_max_abs_error"] == 0.0
    assert result["parity_passed"] is True
    assert result["variance_proxy_max_abs_error"] < 1e-14
    assert any("neither ordinary" in warning for warning in result["warnings"])


def test_captum_probe_regression_retains_all_repeated_deltas():
    model = _affine()
    queries, background = _probe_inputs()
    result = reference.captum_same_draw_reference(
        model,
        queries,
        background,
        methods,
        samples=37,
        numpy_seed=161592,
        torch_seed=161593,
        stdev=0.1,
    )
    assert result["author_phi"].shape == (2, 2)
    assert result["author_convergence_delta"].shape == (74,)
    assert result["student_repeated_convergence_delta"].shape == (74,)
    assert result["parity_max_abs_error"] == 0.0
    assert result["convergence_delta_max_abs_error"] < 1e-14
    assert result["parity_passed"] is True
    assert any("neither estimator uncertainty" in warning for warning in result["warnings"])


def test_positive_noise_is_checked_separately_against_a_finite_draw_relu_oracle():
    result = reference.positive_noise_reference(methods)
    assert result["status"] == "PASS"
    assert result["samples"] == 257
    assert "not cross-package equality targets" in result["warning"]
    for backend in ("shap", "captum"):
        package = result["packages"][backend]
        assert package["positive_noise_confirmed"] is True
        assert package["passed"] is True
        assert package["finite_draw_analytic_max_abs_error"] <= 1e-10
        assert package["same_draw"]["parity_passed"] is True
        assert package["finite_draw_minus_population"] != 0.0
    assert not np.array_equal(
        result["packages"]["shap"]["same_draw"]["draws"]["noise"],
        result["packages"]["captum"]["same_draw"]["draws"]["noise"],
    )


@pytest.mark.parametrize("backend", ("shap", "captum"))
def test_adapters_restore_rng_dtype_modes_state_and_gradient_identity_on_success(backend):
    model = _affine().train()
    model[0].eval()  # Preserve a deliberately mixed per-module mode state.
    parameters = list(model.parameters())
    parameters[0].grad = t.full_like(parameters[0], 7.0)
    parameters[1].grad = t.full_like(parameters[1], -4.0)
    queries, background = _probe_inputs()
    random.seed(710)
    np.random.seed(711)
    t.manual_seed(712)
    old_dtype = t.get_default_dtype()
    try:
        t.set_default_dtype(t.float32)
        before = _state_snapshot(model)
        if backend == "shap":
            reference.shap_same_draw_reference(
                model,
                queries,
                background,
                methods,
                samples=13,
                numpy_seed=21,
                torch_seed=22,
                batch_size=7,
                local_smoothing=0.1,
            )
        else:
            reference.captum_same_draw_reference(
                model,
                queries,
                background,
                methods,
                samples=13,
                numpy_seed=23,
                torch_seed=24,
                stdev=0.1,
            )
        _assert_state_restored(model, before)
    finally:
        t.set_default_dtype(old_dtype)


def _mutate_everything(model: nn.Module) -> None:
    random.seed(991)
    np.random.seed(992)
    t.manual_seed(993)
    t.set_default_dtype(t.float16)
    model.eval()
    with t.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
            parameter.grad = None
        for buffer in model.buffers():
            if buffer.is_floating_point():
                buffer.add_(200.0)


def test_shap_restores_state_when_the_author_constructor_raises(monkeypatch):
    import shap

    model = _affine().train()
    model.register_buffer("sentinel", t.tensor([9.0], dtype=t.float64))
    model[0].eval()
    for parameter in model.parameters():
        parameter.grad = t.full_like(parameter, 3.0)
    queries, background = _probe_inputs()
    old_dtype = t.get_default_dtype()

    class FailingExplainer:
        def __init__(self, supplied_model, *_args, **_kwargs):
            _mutate_everything(supplied_model)
            raise RuntimeError("intentional SHAP failure")

    try:
        t.set_default_dtype(t.float32)
        random.seed(720)
        np.random.seed(721)
        t.manual_seed(722)
        before = _state_snapshot(model)
        monkeypatch.setattr(shap, "GradientExplainer", FailingExplainer)
        with pytest.raises(RuntimeError, match="intentional SHAP failure"):
            reference.shap_same_draw_reference(
                model,
                queries,
                background,
                methods,
                samples=5,
                numpy_seed=31,
                torch_seed=32,
            )
        _assert_state_restored(model, before)
    finally:
        t.set_default_dtype(old_dtype)


def test_captum_restores_state_when_the_author_call_raises(monkeypatch):
    import captum.attr

    model = _affine().train()
    model.register_buffer("sentinel", t.tensor([-8.0], dtype=t.float64))
    model[0].eval()
    for parameter in model.parameters():
        parameter.grad = t.full_like(parameter, -6.0)
    queries, background = _probe_inputs()
    old_dtype = t.get_default_dtype()

    class FailingGradientShap:
        def __init__(self, supplied_model):
            self.model = supplied_model

        def attribute(self, *_args, **_kwargs):
            _mutate_everything(self.model)
            raise RuntimeError("intentional Captum failure")

    try:
        t.set_default_dtype(t.float64)
        random.seed(730)
        np.random.seed(731)
        t.manual_seed(732)
        before = _state_snapshot(model)
        monkeypatch.setattr(captum.attr, "GradientShap", FailingGradientShap)
        with pytest.raises(RuntimeError, match="intentional Captum failure"):
            reference.captum_same_draw_reference(
                model,
                queries,
                background,
                methods,
                samples=5,
                numpy_seed=33,
                torch_seed=34,
            )
        _assert_state_restored(model, before)
    finally:
        t.set_default_dtype(old_dtype)


def test_one_dimensional_model_output_is_rejected_without_leaking_state():
    class SqueezeLast(nn.Module):
        def forward(self, value):
            return value.squeeze(-1)

    model = nn.Sequential(nn.Linear(2, 1, dtype=t.float64), SqueezeLast()).train()
    queries, background = _probe_inputs()
    before = _state_snapshot(model)
    with pytest.raises(ValueError, match=r"shape \(rows, 1\)"):
        reference.shap_same_draw_reference(
            model,
            queries,
            background,
            methods,
            samples=5,
            numpy_seed=35,
            torch_seed=36,
        )
    _assert_state_restored(model, before)


def test_full_272_untrained_regression_matches_retained_benchmark_identities():
    model, queries, background = _seeded_untrained_benchmark()
    shap_result = reference.shap_same_draw_reference(
        model,
        queries,
        background,
        methods,
        samples=128,
        numpy_seed=161602,
        torch_seed=161604,
        batch_size=128,
    )
    assert shap_result["identity_sha256"] == {
        **shap_result["identity_sha256"],
        "model_state": "a9ce569032181616606f1be816645f3f8041095e2ff6ca7d275d03d43ba32628",
        "queries": "26c4964f5eda8e72f4fb98b742d1f8aed1fdcbb68f3f8d3c9e8c1bc0096ecadf",
        "background": "22a6d6bcb0ff42381e78f53675265d289806d761ed16a7f67bc129a5c706d022",
        "donor_indices": "77059c51f96c934f376ce2bcc83c0b540f50cdadd797ccb0c0565abc05e788d5",
        "alphas": "d5b57260ee8022f7cbafadc160c4a636c074863a3581024df519a932495472c1",
        "noise": "e64d32ff6f9150e0244fe04b65c97d52f57a65e82902d750d635296bf8e4acf7",
        "author_phi": "4e777da9222f5750eb402fcd9fc516b2b05a5f86266ee83ff6be276c81916796",
        "student_phi": "b01ccc8826e2d961f41a860a5987568b64a849d3264f55874271c32f13f34e14",
        "student_contributions": "69b3c11b3518411e10b914afe671f9015c0695734c6e7d451e77d374edc60daf",
    }
    assert shap_result["parity_max_abs_error"] < 1e-15

    captum_result = reference.captum_same_draw_reference(
        model,
        queries,
        background,
        methods,
        samples=128,
        numpy_seed=161603,
        torch_seed=161604,
    )
    expected = {
        "model_state": "a9ce569032181616606f1be816645f3f8041095e2ff6ca7d275d03d43ba32628",
        "queries": "26c4964f5eda8e72f4fb98b742d1f8aed1fdcbb68f3f8d3c9e8c1bc0096ecadf",
        "background": "22a6d6bcb0ff42381e78f53675265d289806d761ed16a7f67bc129a5c706d022",
        "donor_indices": "9a61b03cf0ee8015386dc03f2d20b3c716a0d04da508ba5fe165533cc20119eb",
        "alphas": "2c96f14f98612687054eb7fc6f334f62c394a7abd46c20793c0e129bb6f5ead4",
        "noise": "e64d32ff6f9150e0244fe04b65c97d52f57a65e82902d750d635296bf8e4acf7",
        "author_phi": "5044530f4d78471b5bcbd91b4443ddaa45e84e76c208bc8addb5cc109e84d02a",
        "student_phi": "5044530f4d78471b5bcbd91b4443ddaa45e84e76c208bc8addb5cc109e84d02a",
        "student_contributions": "47759519498177197d5443992fb0041f08b50619c481dbc3458a94f561396ba5",
    }
    for name, digest in expected.items():
        assert captum_result["identity_sha256"][name] == digest, name
    assert captum_result["parity_max_abs_error"] == 0.0
    assert captum_result["author_convergence_delta"].shape == (272 * 128,)


def test_validate_reference_runs_all_seven_models_and_all_272_rows(monkeypatch):
    models = OrderedDict()
    for index, key in enumerate(reference.MODEL_KEYS, start=1):
        model = _affine(8)
        with t.no_grad():
            model[0].weight.mul_(index / 7)
            model[0].bias.fill_(index)
        models[key] = model.eval()
    generator = t.Generator(device="cpu").manual_seed(161699)
    background = t.randn((64, 8), dtype=t.float64, generator=generator)
    queries = t.randn((272, 8), dtype=t.float64, generator=generator)
    source_ids = t.arange(10_000, 10_272, dtype=t.int64)

    # The separate positive-noise path has its own direct test above.  Keep this
    # complete 14-call coverage test focused on the all-model/all-row clean run.
    monkeypatch.setattr(
        reference,
        "positive_noise_reference",
        lambda _methods: {"status": "PASS", "separately_tested": True},
    )
    result = reference.validate_reference(
        models, queries, background, source_ids, methods, samples=128
    )
    assert result["status"] == "PASS"
    assert result["counts"] == {
        "models": 7,
        "model_rows": 1904,
        "background_rows": 64,
        "samples_per_package": 128,
        "shap_vectors": 1904,
        "shap_coordinates": 15232,
        "captum_vectors": 1904,
        "captum_coordinates": 15232,
        "package_vectors": 3808,
        "package_coordinates": 30464,
    }
    assert result["versions"] == {"shap": "0.49.1", "captum": "0.9.0"}
    assert result["tolerances"] == {
        "same_draw_atol": 1e-10,
        "same_draw_rtol": 1e-10,
    }
    assert result["passes"] and all(
        type(value) is bool and value for value in result["passes"].values()
    )
    assert tuple(result["packages"]["shap"]["models"]) == reference.MODEL_KEYS
    assert tuple(result["packages"]["captum"]["models"]) == reference.MODEL_KEYS
    assert result["packages"]["shap"]["draw_identity_sha256"] != result[
        "packages"
    ]["captum"]["draw_identity_sha256"]
    for backend in ("shap", "captum"):
        package = result["packages"][backend]
        assert package["all_models_passed"] is True
        assert package["draws"]["donor_indices"].shape == (272, 128)
        for item in package["models"].values():
            assert item["author_phi"].shape == (272, 8)
            assert item["student_phi"].shape == (272, 8)
            assert item["student_contributions"].shape == (272, 128, 8)
            assert item["parity_passed"] is True
