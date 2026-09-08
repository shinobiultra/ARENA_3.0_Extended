"""Complete CPU DeepSHAP study; these functions are learner-visible."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.deepshap_data import (
    background_order,
)


FEATURES = 8
HIDDEN = 32
TRAINING_STEPS = 800
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
RIDGE_PENALTY = 0.01
MODEL_SEEDS = (161410, 161411, 161412)
SHUFFLE_SEEDS = (161420, 161421, 161422)
PRIMARY_BACKGROUND_SEED = 161460
INDEPENDENT_BACKGROUND_SEEDS = (161461, 161462)
RANDOM_ORDER_SEED = 161480
MODEL_KEYS = tuple(
    [f"clean_{index}" for index in range(3)]
    + [f"shuffled_{index}" for index in range(3)]
    + ["ridge"]
)
FEATURE_NAMES = (
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "Superplasticizer",
    "Coarse aggregate",
    "Fine aggregate",
    "Age",
)
MODEL_LABELS = {
    **{f"clean_{index}": f"clean MLP seed {seed}" for index, seed in enumerate(MODEL_SEEDS)},
    **{
        f"shuffled_{index}": f"shuffled MLP seed {seed}"
        for index, seed in enumerate(MODEL_SEEDS)
    },
    "ridge": "ridge control",
}
METHOD_NAMES = (
    "rescale_multipliers",
    "background_deepshap",
    "coalition_values",
    "exact_shapley",
    "gradient_delta",
    "coalition_masks",
    "ranked_replacement_curves",
)
UNITS = {
    "input": "source feature units",
    "target": "MPa",
    "prediction": "MPa",
    "base": "MPa",
    "attribution": "MPa",
    "game": "MPa",
    "squared_error": "MPa squared",
}


def _float_tensor(value: Any, name: str, dimensions: int) -> t.Tensor:
    tensor = value if isinstance(value, t.Tensor) else t.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor or array.")
    if tensor.ndim != dimensions or not bool(t.isfinite(tensor).all()):
        raise ValueError(f"{name} has the wrong shape or contains nonfinite values.")
    return tensor.detach().clone()


def _ids_tensor(value: Any, rows: int, name: str) -> t.Tensor:
    tensor = value if isinstance(value, t.Tensor) else t.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.dtype != t.int64 or tensor.shape != (rows,):
        raise TypeError(f"{name} must be a one-dimensional CPU int64 array.")
    if len(t.unique(tensor)) != rows:
        raise ValueError(f"{name} must contain unique source-row identities.")
    return tensor.detach().clone()


def _rows(rows: Mapping[str, Any], name: str) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    if not isinstance(rows, Mapping) or not {"x", "y", "ids"} <= set(rows):
        raise TypeError(f"{name} must contain x, y and ids.")
    x = _float_tensor(rows["x"], f"{name}.x", 2)
    y = _float_tensor(rows["y"], f"{name}.y", 1)
    if x.shape != (len(y), FEATURES) or len(y) == 0:
        raise ValueError(f"{name} must contain nonempty eight-feature rows.")
    ids = _ids_tensor(rows["ids"], len(y), f"{name}.ids")
    return x, y, ids


def _require_disjoint_ids(**splits: t.Tensor) -> None:
    """Reject source-row reuse across named train, explanation and audit splits."""
    names = tuple(splits)
    for left_index, left_name in enumerate(names):
        left = splits[left_name]
        for right_name in names[left_index + 1 :]:
            right = splits[right_name]
            if bool(t.isin(left, right).any()):
                raise ValueError(
                    f"{left_name} and {right_name} source-row identities must be disjoint."
                )


def _clone_state(state: Mapping[str, t.Tensor]) -> dict[str, t.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError("A model state must be a tensor mapping.")
    cloned: dict[str, t.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, t.Tensor):
            raise TypeError("State keys must be strings and values must be tensors.")
        if value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError("Every model-state tensor must be CPU float64.")
        if not bool(t.isfinite(value).all()):
            raise ValueError("Model-state tensors must be finite.")
        cloned[name] = value.detach().clone()
    return cloned


def _state_shapes(kind: str) -> dict[str, tuple[int, ...]]:
    if kind == "mlp":
        return {
            "0.weight": (HIDDEN, FEATURES),
            "0.bias": (HIDDEN,),
            "2.weight": (HIDDEN, HIDDEN),
            "2.bias": (HIDDEN,),
            "4.weight": (1, HIDDEN),
            "4.bias": (1,),
        }
    if kind == "ridge":
        return {"0.weight": (1, FEATURES), "0.bias": (1,)}
    raise ValueError(f"Unsupported fitted-model kind: {kind!r}.")


def _model_from_state(kind: str, state: Mapping[str, t.Tensor]) -> t.nn.Sequential:
    copied = _clone_state(state)
    shapes = _state_shapes(kind)
    if set(copied) != set(shapes):
        raise ValueError(f"{kind} state has unexpected parameter names.")
    if any(tuple(copied[name].shape) != shape for name, shape in shapes.items()):
        raise ValueError(f"{kind} state has an unexpected parameter shape.")
    if kind == "mlp":
        model = t.nn.Sequential(
            t.nn.Linear(FEATURES, HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(HIDDEN, HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(HIDDEN, 1, dtype=t.float64),
        )
    else:
        model = t.nn.Sequential(t.nn.Linear(FEATURES, 1, dtype=t.float64))
    model.load_state_dict(copied, strict=True)
    model.eval()
    return model


def make_model(item: Mapping[str, Any]) -> t.nn.Sequential:
    """Reconstruct a raw-input, raw-output Sequential solely from saved state."""
    if not isinstance(item, Mapping) or "kind" not in item or "state" not in item:
        raise TypeError("A fitted model item must contain kind and raw state.")
    return _model_from_state(str(item["kind"]), item["state"])


def _normalization_tensors(normalization: Mapping[str, Any]) -> tuple[t.Tensor, ...]:
    if not isinstance(normalization, Mapping):
        raise TypeError("normalization must be a mapping.")
    x_mean = _float_tensor(normalization.get("x_mean"), "x_mean", 1)
    x_std = _float_tensor(normalization.get("x_std"), "x_std", 1)
    y_mean = _float_tensor(normalization.get("y_mean"), "y_mean", 0)
    y_std = _float_tensor(normalization.get("y_std"), "y_std", 0)
    if x_mean.shape != (FEATURES,) or x_std.shape != (FEATURES,):
        raise ValueError("Normalization must describe exactly eight input features.")
    if bool((x_std <= 0).any()) or not bool(y_std > 0):
        raise ValueError("Training standard deviations must be positive.")
    return x_mean, x_std, y_mean, y_std


def _fold_mlp_state(
    standardized_state: Mapping[str, t.Tensor], normalization: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Fold input and output standardization into the first and last affine layers."""
    state = _clone_state(standardized_state)
    if set(state) != set(_state_shapes("mlp")):
        raise ValueError("Cannot fold an unexpected MLP state.")
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    first_weight = state["0.weight"].clone()
    state["0.weight"] = first_weight / x_std[None, :]
    state["0.bias"] = state["0.bias"] - first_weight @ (x_mean / x_std)
    state["4.weight"] = y_std * state["4.weight"]
    state["4.bias"] = y_std * state["4.bias"] + y_mean
    return state


def _fold_ridge_state(
    standardized_state: Mapping[str, t.Tensor], normalization: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Fold standardized linear regression back into raw feature and MPa units."""
    state = _clone_state(standardized_state)
    if set(state) != set(_state_shapes("ridge")):
        raise ValueError("Cannot fold an unexpected ridge state.")
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    weight = state["0.weight"].clone()
    state["0.weight"] = y_std * weight / x_std[None, :]
    state["0.bias"] = y_std * (
        state["0.bias"] - (weight * (x_mean / x_std)[None, :]).sum(1)
    ) + y_mean
    return state


def _folding_error(
    kind: str,
    standardized_state: Mapping[str, t.Tensor],
    raw_state: Mapping[str, t.Tensor],
    normalization: Mapping[str, Any],
    x_raw: t.Tensor,
) -> float:
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    standard_model = _model_from_state(kind, standardized_state)
    raw_model = _model_from_state(kind, raw_state)
    with t.no_grad():
        expected = standard_model((x_raw - x_mean) / x_std).squeeze(-1) * y_std + y_mean
        actual = raw_model(x_raw).squeeze(-1)
    return float((actual - expected).abs().max())


def _training_normalization(
    x_raw: t.Tensor, y_raw: t.Tensor
) -> tuple[dict[str, t.Tensor], t.Tensor, t.Tensor]:
    """Compute train-only moments and return standardized training coordinates."""
    normalization = {
        "x_mean": x_raw.mean(0),
        "x_std": x_raw.std(0, correction=0),
        "y_mean": y_raw.mean(),
        "y_std": y_raw.std(correction=0),
    }
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std
    return normalization, x, y


def _train_mlp(network: t.nn.Sequential, x: t.Tensor, labels: t.Tensor) -> t.Tensor:
    """Run the fixed full-batch Adam loop and retain every pre-update loss."""
    optimizer = t.optim.Adam(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    losses: list[t.Tensor] = []
    network.train()
    with t.enable_grad():
        for _ in range(TRAINING_STEPS):
            optimizer.zero_grad()
            loss = (network(x).squeeze(-1) - labels).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().clone())
    return t.stack(losses)


def _fold_mlp_result(
    network: t.nn.Sequential,
    normalization: Mapping[str, Any],
    x_raw: t.Tensor,
) -> tuple[dict[str, t.Tensor], dict[str, t.Tensor], t.Tensor, float]:
    """Freeze both MLP coordinate systems and verify raw-unit prediction parity."""
    standardized_state = _clone_state(network.state_dict())
    raw_state = _fold_mlp_state(standardized_state, normalization)
    folding_error = _folding_error(
        "mlp", standardized_state, raw_state, normalization, x_raw
    )
    if folding_error > 1e-10:
        raise AssertionError("Raw-unit MLP folding changed its training predictions.")
    with t.no_grad():
        prediction = _model_from_state("mlp", raw_state)(x_raw).squeeze(-1)
    return standardized_state, raw_state, prediction, folding_error


def fit_one_mlp_control(
    control: str,
    index: int,
    x: t.Tensor,
    y: t.Tensor,
    x_raw: t.Tensor,
    y_raw: t.Tensor,
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Train one predeclared clean or label-shuffled MLP and fold it to raw units."""
    if control not in {"clean", "shuffled"} or index not in range(3):
        raise ValueError("Choose a clean or shuffled control and seed index 0, 1 or 2.")
    seed = MODEL_SEEDS[index]
    t.manual_seed(seed)
    network = t.nn.Sequential(
        t.nn.Linear(FEATURES, HIDDEN, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(HIDDEN, HIDDEN, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(HIDDEN, 1, dtype=t.float64),
    )
    initial_state = _clone_state(network.state_dict())
    permutation = t.arange(len(y), dtype=t.int64)
    shuffle_seed = None
    if control == "shuffled":
        shuffle_seed = SHUFFLE_SEEDS[index]
        permutation = t.from_numpy(
            np.random.default_rng(shuffle_seed).permutation(len(y))
        )
    labels = y[permutation]
    loss_trace = _train_mlp(network, x, labels)
    standardized_state, raw_state, prediction, folding_error = _fold_mlp_result(
        network, normalization, x_raw
    )
    return {
        "kind": "mlp",
        "control": control,
        "seed": seed,
        "initial_state": initial_state,
        "standardized_state": standardized_state,
        "state": raw_state,
        "loss_trace": loss_trace,
        "target_permutation": permutation.clone(),
        "training_targets_raw": y_raw[permutation].clone(),
        "training_prediction": prediction.clone(),
        "folding_max_abs_error": folding_error,
        "parameters": {
            "architecture": [FEATURES, HIDDEN, HIDDEN, 1],
            "steps": TRAINING_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "Adam",
            "risk": "mean_squared_error",
            "batch": "full_training_split",
            "shuffle_seed": shuffle_seed,
            "dtype": "float64",
            "parameter_count": sum(value.numel() for value in raw_state.values()),
        },
    }


def fit_ridge_control(
    x: t.Tensor,
    y: t.Tensor,
    x_raw: t.Tensor,
    y_raw: t.Tensor,
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve the fixed standardized ridge control once, excluding its intercept."""
    design = t.cat([x, t.ones((len(x), 1), dtype=t.float64)], dim=1)
    penalty = t.eye(FEATURES + 1, dtype=t.float64) * RIDGE_PENALTY
    penalty[-1, -1] = 0.0
    coefficients = t.linalg.solve(
        design.T @ design / len(x) + penalty,
        design.T @ y / len(x),
    )
    ridge_standardized_state = {
        "0.weight": coefficients[:FEATURES][None, :].clone(),
        "0.bias": coefficients[FEATURES:].clone(),
    }
    ridge_state = _fold_ridge_state(ridge_standardized_state, normalization)
    ridge_error = _folding_error(
        "ridge", ridge_standardized_state, ridge_state, normalization, x_raw
    )
    if ridge_error > 1e-10:
        raise AssertionError("Raw-unit ridge folding changed its training predictions.")
    with t.no_grad():
        ridge_prediction = _model_from_state("ridge", ridge_state)(x_raw).squeeze(-1)
    ridge_loss = (
        (_model_from_state("ridge", ridge_standardized_state)(x).squeeze(-1) - y)
        .square()
        .mean()
        .detach()
    )
    return {
        "kind": "ridge",
        "control": "ridge",
        "seed": 0,
        "initial_state": {},
        "standardized_state": _clone_state(ridge_standardized_state),
        "state": ridge_state,
        "loss_trace": ridge_loss[None].clone(),
        "target_permutation": t.arange(len(y_raw), dtype=t.int64),
        "training_targets_raw": y_raw.clone(),
        "training_prediction": ridge_prediction.clone(),
        "folding_max_abs_error": ridge_error,
        "parameters": {
            "penalty": RIDGE_PENALTY,
            "penalize_intercept": False,
            "risk": "mean_squared_error",
            "dtype": "float64",
            "parameter_count": FEATURES + 1,
        },
    }


def fit_models(train: Mapping[str, Any]) -> dict[str, Any]:
    """Fit six fixed MLPs and one ridge using training rows only."""
    x_raw, y_raw, train_ids = _rows(train, "train")
    t.set_num_threads(4)
    t.use_deterministic_algorithms(True)
    normalization, x, y = _training_normalization(x_raw, y_raw)
    models = {
        f"{control}_{index}": fit_one_mlp_control(
            control, index, x, y, x_raw, y_raw, normalization
        )
        for control in ("clean", "shuffled")
        for index in range(3)
    }
    models["ridge"] = fit_ridge_control(x, y, x_raw, y_raw, normalization)
    return {
        "models": models,
        "normalization": {name: value.detach().clone() for name, value in normalization.items()},
        "train_ids": train_ids,
        "train_x_raw": x_raw,
        "train_y_raw": y_raw,
        "units": dict(UNITS),
        "feature_names": FEATURE_NAMES,
        "model_labels": dict(MODEL_LABELS),
        "phase": "models_fitted_on_train_only",
    }


def _require_methods(methods: Any) -> None:
    missing = [name for name in METHOD_NAMES if not callable(getattr(methods, name, None))]
    if missing:
        raise TypeError(f"Student methods namespace is missing {missing}.")


def _validate_fitted(fitted: Mapping[str, Any]) -> None:
    if not isinstance(fitted, Mapping) or fitted.get("phase") != "models_fitted_on_train_only":
        raise ValueError("Use the complete train-only fitted payload.")
    if not isinstance(fitted.get("models"), Mapping) or tuple(fitted["models"]) != MODEL_KEYS:
        raise ValueError("The fitted payload must retain all seven models in frozen order.")
    _normalization_tensors(fitted.get("normalization"))
    x = _float_tensor(fitted.get("train_x_raw"), "train_x_raw", 2)
    y = _float_tensor(fitted.get("train_y_raw"), "train_y_raw", 1)
    if x.shape != (len(y), FEATURES):
        raise ValueError("Fitted training arrays have incompatible shapes.")
    _ids_tensor(fitted.get("train_ids"), len(y), "train_ids")
    for key in MODEL_KEYS:
        model = fitted["models"][key]
        expected_kind = "ridge" if key == "ridge" else "mlp"
        if model.get("kind") != expected_kind:
            raise ValueError(f"{key} has the wrong model kind.")
        make_model(model)


def _output_tensor(value: Any, shape: tuple[int, ...], name: str) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        raise TypeError(f"{name} must be a Torch tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64 or tuple(value.shape) != shape:
        raise ValueError(f"{name} has the wrong dtype, device or shape.")
    if not bool(t.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values.")
    return value.detach().clone()


def _explain_model(
    item: Mapping[str, Any],
    x: t.Tensor,
    background: t.Tensor,
    methods: Any,
) -> dict[str, t.Tensor]:
    model = make_model(item)
    rows, donors = len(x), len(background)
    with t.no_grad():
        prediction = model(x).squeeze(-1)
        expected_base = model(background).squeeze(-1).mean()
    base, phi_deep = methods.background_deepshap(model, x, background)
    multipliers = methods.rescale_multipliers(model, x, background)
    pair_deep = multipliers * (x[:, None, :] - background[None, :, :])
    masks = methods.coalition_masks(FEATURES)
    game = methods.coalition_values(model, x, background, masks)
    phi_exact = methods.exact_shapley(game)
    phi_gradient = methods.gradient_delta(model, x, background)

    prediction = _output_tensor(prediction, (rows,), "prediction")
    base = _output_tensor(base, (), "base")
    phi_deep = _output_tensor(phi_deep, (rows, FEATURES), "phi_deep")
    pair_deep = _output_tensor(pair_deep, (rows, donors, FEATURES), "pair_deep")
    game = _output_tensor(game, (rows, 2**FEATURES), "game")
    phi_exact = _output_tensor(phi_exact, (rows, FEATURES), "phi_exact")
    phi_gradient = _output_tensor(phi_gradient, (rows, FEATURES), "phi_gradient")
    if not t.allclose(base, expected_base, atol=1e-10, rtol=1e-10):
        raise AssertionError("DeepSHAP base is not the mean background prediction.")
    if not t.allclose(phi_deep, pair_deep.mean(1), atol=1e-10, rtol=1e-10):
        raise AssertionError("DeepSHAP must average complete per-reference explanations.")
    if not t.allclose(game[:, 0], base.expand(rows), atol=1e-10, rtol=1e-10):
        raise AssertionError("The empty coalition does not equal the empirical background base.")
    if not t.allclose(game[:, -1], prediction, atol=1e-10, rtol=1e-10):
        raise AssertionError("The full coalition does not equal the raw model prediction.")
    return {
        "prediction": prediction,
        "base": base,
        "phi_deep": phi_deep,
        "phi_exact": phi_exact,
        "phi_gradient": phi_gradient,
        "game": game,
        "pair_deep": pair_deep,
    }


def explain_block(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    background: Any,
    methods: Any,
) -> dict[str, dict[str, t.Tensor]]:
    """Explain every retained model on one named row block in raw units."""
    _validate_fitted(fitted)
    _require_methods(methods)
    x, _, _ = _rows(rows, "rows")
    background_tensor = _float_tensor(background, "background", 2)
    if background_tensor.shape[1:] != (FEATURES,) or len(background_tensor) == 0:
        raise ValueError("background must contain nonempty eight-feature donor rows.")
    return {
        key: _explain_model(fitted["models"][key], x, background_tensor, methods)
        for key in MODEL_KEYS
    }


def _background_indices(train_ids: t.Tensor) -> dict[str, t.Tensor]:
    primary = t.from_numpy(background_order(train_ids.numpy(), PRIMARY_BACKGROUND_SEED))
    independent_a = t.from_numpy(
        background_order(train_ids.numpy(), INDEPENDENT_BACKGROUND_SEEDS[0])
    )
    independent_b = t.from_numpy(
        background_order(train_ids.numpy(), INDEPENDENT_BACKGROUND_SEEDS[1])
    )
    if len(primary) < 128:
        raise ValueError("The frozen study requires at least 128 training donors.")
    return {
        "primary64": primary[:64].clone(),
        "nested16": primary[:16].clone(),
        "nested128": primary[:128].clone(),
        "independent64_a": independent_a[:64].clone(),
        "independent64_b": independent_b[:64].clone(),
    }


def prepare_study(
    fitted: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Explain and rank on the explanation split without receiving audit rows."""
    _validate_fitted(fitted)
    train_x, train_y, train_ids = _rows(train, "train")
    explain_x, explain_y, explain_ids = _rows(explain, "explain")
    _require_disjoint_ids(train=train_ids, explanation=explain_ids)
    if not t.equal(train_ids, fitted["train_ids"]):
        raise ValueError("Training identities changed after fitting.")
    if not t.equal(train_x, fitted["train_x_raw"]) or not t.equal(
        train_y, fitted["train_y_raw"]
    ):
        raise ValueError("Training values changed after fitting.")
    indices = _background_indices(train_ids)
    primary = train_x[indices["primary64"]]
    models = explain_block(fitted, explain, primary, methods)
    rankings = {
        key: {
            "exact": t.argsort(
                models[key]["phi_exact"].abs().mean(0), descending=True, stable=True
            ),
            "deep": t.argsort(models[key]["phi_deep"].abs().mean(0), descending=True, stable=True),
            "gradient": t.argsort(
                models[key]["phi_gradient"].abs().mean(0), descending=True, stable=True
            ),
        }
        for key in MODEL_KEYS
    }
    return {
        "models": models,
        "explain_ids": explain_ids,
        "explain_x_raw": explain_x,
        "explain_targets": explain_y,
        "background_ids": train_ids[indices["primary64"]],
        "background": primary,
        "background_indices": indices,
        "rankings": rankings,
        "audit_used_for_selection": False,
        "units": dict(UNITS),
        "feature_names": FEATURE_NAMES,
        "model_labels": dict(MODEL_LABELS),
        "phase": "rankings_frozen_before_audit",
    }


def _clone_tree(value: Any) -> Any:
    if isinstance(value, t.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {name: _clone_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _ranked_curves(
    game: t.Tensor, phi: t.Tensor, order: t.Tensor, methods: Any
) -> dict[str, t.Tensor]:
    expanded = order[None, :].expand(len(game), FEATURES).clone()
    curve = methods.ranked_replacement_curves(game, phi, expanded)
    expected = {
        "orders": (len(game), FEATURES),
        "insertion_masks": (len(game), FEATURES + 1),
        "deletion_masks": (len(game), FEATURES + 1),
        "insertion": (len(game), FEATURES + 1),
        "deletion": (len(game), FEATURES + 1),
    }
    if not isinstance(curve, Mapping) or set(curve) != set(expected):
        raise ValueError("Replacement curves returned an unexpected payload.")
    result: dict[str, t.Tensor] = {}
    for name, shape in expected.items():
        value = curve[name]
        if name.endswith("masks") or name == "orders":
            if not isinstance(value, t.Tensor) or value.dtype != t.int64:
                raise TypeError(f"{name} must be a long tensor.")
            if value.device.type != "cpu" or tuple(value.shape) != shape:
                raise ValueError(f"{name} has the wrong shape or device.")
            result[name] = value.detach().clone()
        else:
            result[name] = _output_tensor(value, shape, name)
    return result


def _validate_rankings(rankings: Any) -> None:
    if not isinstance(rankings, Mapping) or tuple(rankings) != MODEL_KEYS:
        raise ValueError("Prepared rankings must retain all seven models in frozen order.")
    expected = t.arange(FEATURES, dtype=t.int64)
    for key in MODEL_KEYS:
        item = rankings[key]
        if not isinstance(item, Mapping) or set(item) != {"exact", "deep", "gradient"}:
            raise ValueError(f"{key} must retain exact, DeepSHAP and gradient rankings.")
        for name, order in item.items():
            if not isinstance(order, t.Tensor) or order.device.type != "cpu":
                raise TypeError(f"{key}/{name} ranking must be a CPU tensor.")
            if order.dtype != t.int64 or order.shape != (FEATURES,):
                raise ValueError(f"{key}/{name} ranking must be one int64 feature order.")
            if not t.equal(order.sort().values, expected):
                raise ValueError(f"{key}/{name} ranking must contain each feature once.")


def _random_curves(game: t.Tensor, orders: t.Tensor, methods: Any) -> dict[str, t.Tensor]:
    curves = [
        _ranked_curves(game, t.zeros((len(game), FEATURES), dtype=t.float64), order, methods)
        for order in orders
    ]
    return {
        "orders": orders.clone(),
        **{
            name: t.stack([curve[name] for curve in curves])
            for name in ("insertion_masks", "deletion_masks", "insertion", "deletion")
        },
    }


def _combine_explanations(
    explanation: Mapping[str, t.Tensor], audit: Mapping[str, t.Tensor]
) -> dict[str, t.Tensor]:
    if not t.allclose(explanation["base"], audit["base"], atol=0.0, rtol=0.0):
        raise ValueError("Primary background base changed between row blocks.")
    return {
        "base": explanation["base"].clone(),
        **{
            name: t.cat([explanation[name], audit[name]], dim=0)
            for name in (
                "prediction",
                "phi_deep",
                "phi_exact",
                "phi_gradient",
                "game",
                "pair_deep",
            )
        },
    }


def _validate_evaluation_inputs(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, t.Tensor]:
    """Recheck the persisted preparation boundary against all named row blocks."""
    _validate_fitted(fitted)
    if prepared.get("phase") != "rankings_frozen_before_audit":
        raise ValueError("Persist explanation rankings before audit evaluation.")
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Audit rows cannot select models, backgrounds or rankings.")
    train_x, train_y, train_ids = _rows(train, "train")
    explain_x, explain_y, explain_ids = _rows(explain, "explain")
    audit_x, audit_y, audit_ids = _rows(audit, "audit")
    _require_disjoint_ids(train=train_ids, explanation=explain_ids, audit=audit_ids)
    if not t.equal(train_ids, fitted["train_ids"]) or not t.equal(
        train_x, fitted["train_x_raw"]
    ) or not t.equal(train_y, fitted["train_y_raw"]):
        raise ValueError("Training rows changed after fitting.")
    if not t.equal(prepared["explain_ids"], explain_ids):
        raise ValueError("Explanation identities changed after rankings were frozen.")
    if not t.equal(prepared["explain_x_raw"], explain_x) or not t.equal(
        prepared["explain_targets"], explain_y
    ):
        raise ValueError("Explanation values changed after rankings were frozen.")
    expected_indices = _background_indices(train_ids)
    indices = prepared.get("background_indices")
    if not isinstance(indices, Mapping) or tuple(indices) != tuple(expected_indices):
        raise ValueError("Prepared background designs changed before audit evaluation.")
    if any(not t.equal(indices[name], expected_indices[name]) for name in expected_indices):
        raise ValueError("Prepared background indices changed before audit evaluation.")
    primary = expected_indices["primary64"]
    if not t.equal(prepared["background_ids"], train_ids[primary]) or not t.equal(
        prepared["background"], train_x[primary]
    ):
        raise ValueError("Primary background identities changed before audit evaluation.")
    _validate_rankings(prepared.get("rankings"))
    return {
        "train_x": train_x,
        "train_ids": train_ids,
        "explain_x": explain_x,
        "explain_y": explain_y,
        "explain_ids": explain_ids,
        "audit_x": audit_x,
        "audit_y": audit_y,
        "audit_ids": audit_ids,
    }


def _performance_table(
    fitted: Mapping[str, Any],
    models: Mapping[str, Mapping[str, t.Tensor]],
    audit_y: t.Tensor,
) -> dict[str, dict[str, Any]]:
    """Retain audit MSE for every model and the train-only mean baseline."""
    performance = {
        key: {
            "kind": fitted["models"][key]["kind"],
            "control": fitted["models"][key]["control"],
            "mse": (models[key]["prediction"] - audit_y).square().mean(),
        }
        for key in MODEL_KEYS
    }
    performance["training_mean"] = {
        "kind": "constant",
        "control": "training_mean",
        "mse": (audit_y - fitted["train_y_raw"].mean()).square().mean(),
    }
    return performance


def _fixed_random_orders() -> t.Tensor:
    """Draw the one frozen stream of 100 random feature-order controls."""
    rng = np.random.default_rng(RANDOM_ORDER_SEED)
    return t.from_numpy(np.stack([rng.permutation(FEATURES) for _ in range(100)]))


def _evaluate_replacements(
    models: Mapping[str, Mapping[str, t.Tensor]],
    rankings: Mapping[str, Mapping[str, t.Tensor]],
    random_orders: t.Tensor,
    methods: Any,
) -> dict[str, Any]:
    """Evaluate fixed global and random orders on every audit model and row."""
    replacement: dict[str, Any] = {}
    for key in MODEL_KEYS:
        replacement[key] = {
            name: _ranked_curves(
                models[key]["game"], models[key][f"phi_{name}"], rankings[key][name], methods
            )
            for name in ("exact", "deep", "gradient")
        }
        replacement[key]["random"] = _random_curves(
            models[key]["game"], random_orders, methods
        )
    return replacement


def _combine_row_blocks(rows: Mapping[str, t.Tensor]) -> dict[str, t.Tensor]:
    """Concatenate explanation then audit rows and retain a numeric split label."""
    return {
        "x": t.cat([rows["explain_x"], rows["audit_x"]], dim=0),
        "y": t.cat([rows["explain_y"], rows["audit_y"]], dim=0),
        "ids": t.cat([rows["explain_ids"], rows["audit_ids"]], dim=0),
        "split": t.cat(
            [
                t.zeros(len(rows["explain_ids"]), dtype=t.int64),
                t.ones(len(rows["audit_ids"]), dtype=t.int64),
            ]
        ),
    }


def _evaluate_background_variants(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    audit_models: Mapping[str, Mapping[str, t.Tensor]],
    rows: Mapping[str, t.Tensor],
    combined: Mapping[str, t.Tensor],
    methods: Any,
) -> dict[str, Any]:
    """Compare clean_0 on all held-out rows under five prespecified backgrounds."""
    primary = _combine_explanations(
        prepared["models"]["clean_0"], audit_models["clean_0"]
    )
    variants: dict[str, Any] = {
        "primary64": {
            **primary,
            "background": prepared["background"].clone(),
            "background_ids": prepared["background_ids"].clone(),
        }
    }
    for name in ("nested16", "nested128", "independent64_a", "independent64_b"):
        indices = prepared["background_indices"][name]
        background = rows["train_x"][indices]
        variants[name] = {
            **_explain_model(fitted["models"]["clean_0"], combined["x"], background, methods),
            "background": background.clone(),
            "background_ids": rows["train_ids"][indices].clone(),
        }
    for result in variants.values():
        result.update(
            row_ids=combined["ids"].clone(),
            row_split=combined["split"].clone(),
            targets=combined["y"].clone(),
        )
    return variants


def evaluate_study(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Evaluate frozen rankings, controls and backgrounds on the audit split."""
    _require_methods(methods)
    rows = _validate_evaluation_inputs(fitted, prepared, train, explain, audit)
    models = explain_block(fitted, audit, prepared["background"], methods)
    performance = _performance_table(fitted, models, rows["audit_y"])
    random_orders = _fixed_random_orders()
    replacement = _evaluate_replacements(
        models, prepared["rankings"], random_orders, methods
    )
    combined = _combine_row_blocks(rows)
    background_variants = _evaluate_background_variants(
        fitted, prepared, models, rows, combined, methods
    )

    return {
        "models": models,
        "audit_ids": rows["audit_ids"],
        "audit_targets": rows["audit_y"],
        "performance": performance,
        "replacement_curves": replacement,
        "random_orders": random_orders,
        "background_variants": background_variants,
        "background_variant_order": (
            "primary64",
            "nested16",
            "nested128",
            "independent64_a",
            "independent64_b",
        ),
        "combined_explanation_audit_ids": combined["ids"],
        "combined_split": combined["split"],
        "combined_split_labels": ("explanation", "audit"),
        "model_order": MODEL_KEYS,
        "model_labels": dict(MODEL_LABELS),
        "feature_names": FEATURE_NAMES,
        "units": dict(UNITS),
        "audit_used_for_selection": False,
        "phase": "audit_evaluated_after_frozen_rankings",
    }


def native_play(
    fitted: Mapping[str, Any],
    train: Mapping[str, Any],
    audit: Mapping[str, Any],
    methods: Any,
    model_key: str = "clean_0",
    row: int = 0,
    size: int = 64,
) -> dict[str, Any]:
    """Recompute one audit recipe under an editable model and nested background."""
    _validate_fitted(fitted)
    _require_methods(methods)
    train_x, _, train_ids = _rows(train, "train")
    audit_x, audit_y, audit_ids = _rows(audit, "audit")
    if model_key not in MODEL_KEYS:
        raise ValueError("Choose one of the seven retained model keys.")
    if not 0 <= row < len(audit_x):
        raise ValueError("Choose an existing audit row.")
    if size not in (16, 64, 128):
        raise ValueError("Choose a nested background size of 16, 64 or 128.")
    order = t.from_numpy(background_order(train_ids.numpy(), PRIMARY_BACKGROUND_SEED))
    indices = order[:size]
    background = train_x[indices]
    x = audit_x[row : row + 1]
    result = _explain_model(fitted["models"][model_key], x, background, methods)
    curves = {
        name: _clone_tree(
            methods.ranked_replacement_curves(
                result["game"], result[f"phi_{name}"]
            )
        )
        for name in ("exact", "deep", "gradient")
    }
    return {
        "model_key": model_key,
        "source_id": int(audit_ids[row]),
        "query": x[0].clone(),
        "target": audit_y[row].clone(),
        "prediction": result["prediction"][0].clone(),
        "base": result["base"].clone(),
        "deep": result["phi_deep"][0].clone(),
        "exact": result["phi_exact"][0].clone(),
        "gradient": result["phi_gradient"][0].clone(),
        "game": result["game"][0].clone(),
        "pair_deep": result["pair_deep"][0].clone(),
        "curves": curves,
        "background_size": size,
        "background_ids": train_ids[indices].clone(),
        "background": background.clone(),
        "units": dict(UNITS),
        "feature_names": FEATURE_NAMES,
        "model_label": MODEL_LABELS[model_key],
        "audit_reuse": "exploratory",
    }
