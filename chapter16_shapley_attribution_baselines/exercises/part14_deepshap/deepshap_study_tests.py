"""Independent unit tests for the visible DeepSHAP study orchestration."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part14_deepshap import (
    deepshap_methods as methods,
)
from chapter16_shapley_attribution_baselines.exercises.part14_deepshap import (
    deepshap_study as study,
)


def _affine_domain_mlp(scale: float) -> t.nn.Sequential:
    """An exact 8-32-32-1 graph whose ReLUs stay active on the test domain."""
    model = t.nn.Sequential(
        t.nn.Linear(8, 32, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(32, 32, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(32, 1, dtype=t.float64),
    )
    with t.no_grad():
        model[0].weight.copy_(
            t.arange(32 * 8, dtype=t.float64).reshape(32, 8) * (scale / 10_000)
        )
        model[0].bias.fill_(0.5)
        model[2].weight.copy_(t.eye(32, dtype=t.float64) * (0.4 + scale / 20))
        model[2].bias.fill_(0.25)
        model[4].weight.copy_(
            t.linspace(-0.08, 0.12, 32, dtype=t.float64)[None, :] * scale
        )
        model[4].bias.fill_(12.0 + scale)
    return model


def _item(model: t.nn.Sequential, key: str) -> dict:
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    control = "ridge" if key == "ridge" else key.rsplit("_", 1)[0]
    return {
        "kind": "ridge" if key == "ridge" else "mlp",
        "control": control,
        "seed": 0 if key == "ridge" else study.MODEL_SEEDS[int(key[-1])],
        "initial_state": {},
        "standardized_state": {name: value.clone() for name, value in state.items()},
        "state": state,
        "loss_trace": t.zeros(1, dtype=t.float64),
        "target_permutation": t.arange(140, dtype=t.int64),
        "training_targets_raw": t.zeros(140, dtype=t.float64),
        "training_prediction": t.zeros(140, dtype=t.float64),
        "parameters": {},
    }


def _fixture():
    grid = t.arange(140 * 8, dtype=t.float64).reshape(140, 8)
    train_x = (grid.remainder(37) / 37).contiguous()
    train_y = 20 + train_x @ t.linspace(0.1, 0.8, 8, dtype=t.float64)
    train = {
        "x": train_x.numpy().copy(),
        "y": train_y.numpy().copy(),
        "ids": np.arange(140, dtype=np.int64),
    }
    explain_x = train_x[[3, 27]]
    audit_x = train_x[[51, 89]]
    explain = {
        "x": explain_x.numpy().copy(),
        "y": (20 + explain_x.sum(1)).numpy().copy(),
        "ids": np.array([203, 227], dtype=np.int64),
    }
    audit = {
        "x": audit_x.numpy().copy(),
        "y": (20 + audit_x.sum(1)).numpy().copy(),
        "ids": np.array([351, 389], dtype=np.int64),
    }
    model_items = {
        key: _item(_affine_domain_mlp(1.0 + index / 10), key)
        for index, key in enumerate(study.MODEL_KEYS[:-1])
    }
    ridge = t.nn.Sequential(t.nn.Linear(8, 1, dtype=t.float64))
    with t.no_grad():
        ridge[0].weight.copy_(t.linspace(-1, 1, 8, dtype=t.float64)[None, :])
        ridge[0].bias.fill_(4.0)
    model_items["ridge"] = _item(ridge, "ridge")
    fitted = {
        "models": model_items,
        "normalization": {
            "x_mean": t.zeros(8, dtype=t.float64),
            "x_std": t.ones(8, dtype=t.float64),
            "y_mean": t.tensor(0.0, dtype=t.float64),
            "y_std": t.tensor(1.0, dtype=t.float64),
        },
        "train_ids": t.from_numpy(train["ids"]).clone(),
        "train_x_raw": t.from_numpy(train["x"]).clone(),
        "train_y_raw": t.from_numpy(train["y"]).clone(),
        "units": dict(study.UNITS),
        "phase": "models_fitted_on_train_only",
    }
    return fitted, train, explain, audit


def test_native_fit_recipe_is_fixed_and_rejects_bad_training_rows_without_fitting():
    assert tuple(inspect.signature(study.fit_models).parameters) == ("train",), (
        "fit_models must not accept evaluation data or a training-step override."
    )
    assert (
        study.TRAINING_STEPS,
        study.LEARNING_RATE,
        study.WEIGHT_DECAY,
        study.MODEL_SEEDS,
        study.SHUFFLE_SEEDS,
    ) == (800, 0.003, 0.0001, (161410, 161411, 161412), (161420, 161421, 161422)), (
        "The native optimization recipe or seed namespace changed."
    )
    invalid = {
        "x": np.zeros((4, 7), dtype=np.float64),
        "y": np.zeros(4, dtype=np.float64),
        "ids": np.arange(4, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="eight-feature"):
        study.fit_models(invalid)


def test_mlp_and_ridge_folding_preserve_raw_predictions_and_states():
    t.manual_seed(161490)
    standard_mlp = t.nn.Sequential(
        t.nn.Linear(8, 32, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(32, 32, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(32, 1, dtype=t.float64),
    )
    standard_state = {
        name: value.detach().clone() for name, value in standard_mlp.state_dict().items()
    }
    normalization = {
        "x_mean": t.linspace(-2, 2, 8, dtype=t.float64),
        "x_std": t.linspace(0.5, 1.5, 8, dtype=t.float64),
        "y_mean": t.tensor(31.0, dtype=t.float64),
        "y_std": t.tensor(7.0, dtype=t.float64),
    }
    raw_x = t.randn(11, 8, dtype=t.float64)
    raw_state = study._fold_mlp_state(standard_state, normalization)
    raw_model = study.make_model({"kind": "mlp", "state": raw_state})
    expected = standard_mlp(
        (raw_x - normalization["x_mean"]) / normalization["x_std"]
    ).squeeze(-1) * normalization["y_std"] + normalization["y_mean"]
    assert t.allclose(raw_model(raw_x).squeeze(-1), expected, atol=1e-12, rtol=1e-12), (
        "Folding the input/output normalization changed MLP predictions."
    )
    assert all(
        t.equal(standard_state[name], standard_mlp.state_dict()[name])
        for name in standard_state
    ), (
        "Folding mutated the standardized MLP state."
    )

    ridge_state = {
        "0.weight": t.linspace(-1, 1, 8, dtype=t.float64)[None, :],
        "0.bias": t.tensor([0.7], dtype=t.float64),
    }
    raw_ridge_state = study._fold_ridge_state(ridge_state, normalization)
    raw_ridge = study.make_model({"kind": "ridge", "state": raw_ridge_state})
    standardized = (raw_x - normalization["x_mean"]) / normalization["x_std"]
    expected_ridge = (
        standardized @ ridge_state["0.weight"].T + ridge_state["0.bias"]
    ).squeeze(-1) * normalization["y_std"] + normalization["y_mean"]
    assert t.allclose(
        raw_ridge(raw_x).squeeze(-1), expected_ridge, atol=1e-12, rtol=1e-12
    ), "Folding the ridge control changed its predictions."


def test_make_model_rejects_wrong_kind_shape_dtype_and_nonfinite_state():
    good = _affine_domain_mlp(1.0).state_dict()
    with pytest.raises(ValueError, match="Unsupported"):
        study.make_model({"kind": "forest", "state": good})
    bad_shape = {name: value.detach().clone() for name, value in good.items()}
    bad_shape["0.weight"] = bad_shape["0.weight"][:, :7]
    with pytest.raises(ValueError, match="shape"):
        study.make_model({"kind": "mlp", "state": bad_shape})
    bad_dtype = {name: value.detach().clone() for name, value in good.items()}
    bad_dtype["0.bias"] = bad_dtype["0.bias"].float()
    with pytest.raises(TypeError, match="float64"):
        study.make_model({"kind": "mlp", "state": bad_dtype})
    bad_value = {name: value.detach().clone() for name, value in good.items()}
    bad_value["4.bias"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        study.make_model({"kind": "mlp", "state": bad_value})


def test_explain_block_uses_student_namespace_for_all_seven_raw_models():
    fitted, train, explain, _ = _fixture()
    background = t.from_numpy(train["x"][:4]).clone()
    calls: dict[str, int] = {}

    def counted(name):
        original = getattr(methods, name)

        def wrapped(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return original(*args, **kwargs)

        return wrapped

    namespace = SimpleNamespace(
        **{name: counted(name) for name in study.METHOD_NAMES}
    )
    result = study.explain_block(fitted, explain, background, namespace)
    assert tuple(result) == study.MODEL_KEYS, "Every retained model must be explained."
    for key, item in result.items():
        assert item["prediction"].shape == (2,), f"{key} predictions lost row identity."
        assert item["base"].shape == (), f"{key} base must be one empirical expectation."
        assert item["pair_deep"].shape == (2, 4, 8), f"{key} lost per-donor values."
        assert item["game"].shape == (2, 256), f"{key} lacks the complete game."
        assert t.allclose(item["phi_deep"], item["phi_exact"], atol=1e-10, rtol=1e-10), (
            f"{key} should be affine on the controlled fixture domain."
        )
        assert t.allclose(
            item["phi_gradient"], item["phi_exact"], atol=1e-10, rtol=1e-10
        ), f"{key} gradient control should match the affine fixture."
    for name in study.METHOD_NAMES[:-1]:
        expected_calls = 7
        assert calls.get(name) == expected_calls, f"Study bypassed student method {name}."
    assert calls.get("ranked_replacement_curves") is None, (
        "Replacement curves belong to evaluation after rankings are frozen."
    )


def test_prepare_study_freezes_primary_background_and_mean_absolute_rankings():
    fitted, train, explain, _ = _fixture()
    prepared = study.prepare_study(fitted, train, explain, methods)
    assert prepared["phase"] == "rankings_frozen_before_audit", (
        "Preparation must end at the explicit pre-audit boundary."
    )
    assert prepared["audit_used_for_selection"] is False, (
        "Audit rows cannot select a DeepSHAP model or ranking."
    )
    assert prepared["background"].shape == (64, 8), "Primary64 must retain full donors."
    assert set(prepared["background_indices"]) == {
        "primary64",
        "nested16",
        "nested128",
        "independent64_a",
        "independent64_b",
    }, "All frozen background designs must be retained."
    for key in study.MODEL_KEYS:
        assert set(prepared["rankings"][key]) == {"exact", "deep", "gradient"}, (
            f"{key} is missing a prespecified attribution ranking."
        )
        for order in prepared["rankings"][key].values():
            assert t.equal(order.sort().values, t.arange(8)), (
                f"{key} ranking must be an exact feature permutation."
            )
        for name in ("exact", "deep", "gradient"):
            expected = t.argsort(
                prepared["models"][key][f"phi_{name}"].abs().mean(0),
                descending=True,
                stable=True,
            )
            assert t.equal(prepared["rankings"][key][name], expected), (
                f"{key}/{name} ranking was not frozen from mean absolute explanation values."
            )
    primary = study._background_indices(fitted["train_ids"])["primary64"]
    assert t.equal(prepared["background_ids"], fitted["train_ids"][primary]), (
        "Primary background identities do not follow the frozen source-ID hash."
    )
    overlapping = {**explain, "ids": np.array([0, 227], dtype=np.int64)}
    with pytest.raises(ValueError, match="must be disjoint"):
        study.prepare_study(fitted, train, overlapping, methods)


def test_evaluate_study_retains_audit_curves_random_orders_and_new_background_games():
    fitted, train, explain, audit = _fixture()
    prepared = study.prepare_study(fitted, train, explain, methods)
    evaluated = study.evaluate_study(fitted, prepared, train, explain, audit, methods)
    assert evaluated["phase"] == "audit_evaluated_after_frozen_rankings", (
        "Evaluation did not retain its post-freeze boundary."
    )
    assert tuple(evaluated["performance"]) == (*study.MODEL_KEYS, "training_mean"), (
        "Performance must retain seven models and the training-mean baseline."
    )
    assert evaluated["random_orders"].shape == (100, 8), (
        "Exactly 100 random-order controls must be retained."
    )
    assert len(t.unique(evaluated["random_orders"], dim=0)) > 1, (
        "Random controls must use one RNG stream rather than repeat one order."
    )
    rng = np.random.default_rng(study.RANDOM_ORDER_SEED)
    expected_random = t.from_numpy(np.stack([rng.permutation(8) for _ in range(100)]))
    assert t.equal(evaluated["random_orders"], expected_random), (
        "Random replacement controls changed seed, stream or ordering."
    )
    for key in study.MODEL_KEYS:
        curves = evaluated["replacement_curves"][key]
        for name in ("exact", "deep", "gradient"):
            assert curves[name]["insertion"].shape == (2, 9), (
                f"{key}/{name} lacks every insertion prefix."
            )
            assert curves[name]["deletion"].shape == (2, 9), (
                f"{key}/{name} lacks every deletion prefix."
            )
            expected = prepared["rankings"][key][name][None, :].expand(2, 8)
            assert t.equal(curves[name]["orders"], expected), (
                f"{key}/{name} used an audit-selected order."
            )
        assert curves["random"]["insertion"].shape == (100, 2, 9), (
            f"{key} lost row-level random insertion controls."
        )
        assert curves["random"]["deletion_masks"].shape == (100, 2, 9), (
            f"{key} lost exact random deletion masks."
        )

    variants = evaluated["background_variants"]
    assert tuple(variants) == evaluated["background_variant_order"], (
        "Background variants changed order or identity."
    )
    assert [len(variants[name]["background"]) for name in variants] == [64, 16, 128, 64, 64], (
        "Background variants have the wrong frozen donor counts."
    )
    for name, item in variants.items():
        assert item["game"].shape == (4, 256), f"{name} lacks all-row exact games."
        assert item["phi_deep"].shape == (4, 8), f"{name} lacks all-row DeepSHAP."
        assert item["pair_deep"].shape == (4, len(item["background"]), 8), (
            f"{name} lacks complete per-reference explanations."
        )
        assert t.equal(item["row_split"], t.tensor([0, 0, 1, 1])), (
            f"{name} changed explanation-then-audit concatenation."
        )
    expected_primary = t.cat(
        [prepared["models"]["clean_0"]["game"], evaluated["models"]["clean_0"]["game"]]
    )
    assert t.equal(variants["primary64"]["game"], expected_primary), (
        "Primary64 should reuse the identical prepared and audit games."
    )


def test_evaluate_rejects_audit_selection_and_changed_explanation_identity():
    fitted, train, explain, audit = _fixture()
    prepared = study.prepare_study(fitted, train, explain, methods)
    selected = {**prepared, "audit_used_for_selection": True}
    with pytest.raises(ValueError, match="cannot select"):
        study.evaluate_study(fitted, selected, train, explain, audit, methods)
    changed = {**prepared, "explain_ids": prepared["explain_ids"].flip(0)}
    with pytest.raises(ValueError, match="identities"):
        study.evaluate_study(fitted, changed, train, explain, audit, methods)
    changed_values = {**explain, "x": explain["x"].copy()}
    changed_values["x"][0, 0] += 1.0
    with pytest.raises(ValueError, match="values changed"):
        study.evaluate_study(fitted, prepared, train, changed_values, audit, methods)
    changed_background = {**prepared, "background": prepared["background"].clone()}
    changed_background["background"][0, 0] += 1.0
    with pytest.raises(ValueError, match="background identities"):
        study.evaluate_study(fitted, changed_background, train, explain, audit, methods)
    overlapping_audit = {**audit, "ids": np.array([0, 389], dtype=np.int64)}
    with pytest.raises(ValueError, match="must be disjoint"):
        study.evaluate_study(fitted, prepared, train, explain, overlapping_audit, methods)


def test_native_play_recomputes_one_row_and_validates_editable_choices():
    fitted, train, _, audit = _fixture()
    play = study.native_play(
        fitted, train, audit, methods, model_key="ridge", row=1, size=16
    )
    assert play["source_id"] == 389 and play["model_key"] == "ridge", (
        "The play lost the selected audit/model identity."
    )
    assert play["query"].shape == (8,) and play["game"].shape == (256,), (
        "The play must retain the query and its complete game."
    )
    assert play["pair_deep"].shape == (16, 8), (
        "The play must expose every selected background contribution."
    )
    assert set(play["curves"]) == {"exact", "deep", "gradient"}, (
        "The play must compare all three fixed attribution orders."
    )
    with pytest.raises(ValueError, match="seven retained"):
        study.native_play(fitted, train, audit, methods, model_key="best_model")
    with pytest.raises(ValueError, match="16, 64 or 128"):
        study.native_play(fitted, train, audit, methods, size=32)


def test_student_namespace_cannot_omit_full_games_pair_values_or_curves():
    fitted, train, explain, audit = _fixture()
    incomplete = SimpleNamespace(
        **{name: getattr(methods, name) for name in study.METHOD_NAMES if name != "exact_shapley"}
    )
    with pytest.raises(TypeError, match="exact_shapley"):
        study.explain_block(fitted, explain, t.from_numpy(train["x"][:4]), incomplete)

    wrong = SimpleNamespace(**{name: getattr(methods, name) for name in study.METHOD_NAMES})
    wrong.ranked_replacement_curves = lambda *args, **kwargs: {"insertion": t.zeros(1)}
    prepared = study.prepare_study(fitted, train, explain, methods)
    with pytest.raises(ValueError, match="unexpected payload"):
        study.evaluate_study(fitted, prepared, train, explain, audit, wrong)
