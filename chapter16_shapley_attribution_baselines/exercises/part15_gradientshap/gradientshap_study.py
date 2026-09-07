"""Complete CPU Expected Gradients study; scientific functions stay learner-visible.

The module owns the numerical study rather than loading the 16.14 result.  It
recreates the seven known predictors, checks their 21 state hashes, freezes all
explanation-side choices before audit evaluation, and retains every draw used by
the declared controls.  Serialization and package-reference plumbing live in
separate modules.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.gradientshap_data import (
    background_order,
    load_manifest,
    tensor_mapping_sha256,
)


FEATURES = 8
HIDDEN = 32
TRAINING_STEPS = 800
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
RIDGE_PENALTY = 0.01
MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
MODEL_SEEDS = (161410, 161411, 161412)
SHUFFLE_SEEDS = (161420, 161421, 161422)
BACKGROUND_SPECS = {
    "primary64": (161460, 64),
    "nested16": (161460, 16),
    "nested128": (161460, 128),
    "independent64_a": (161461, 64),
    "independent64_b": (161462, 64),
}
BUDGETS = (8, 32, 128, 512, 2048)
QUADRATURE_NODES = (256, 512)
NOISE_SIGMAS = (0.0, 0.02, 0.10)
PRIMARY_SCHEDULE_SEED = 161510
REPLICATE_SCHEDULE_SEED = 161511
NOISE_SCHEDULE_SEED = 161512
REFERENCE_SEEDS = {"shap": 161513, "captum_numpy": 161514, "captum_torch": 161515}
RANDOM_ORDER_SEED = 161480
REPLICATES = 64
REPLICATE_MODELS = ("clean_0", "shuffled_0", "ridge")
PATH_DISTANCE_SAMPLES = 128
COALITION_DISTANCE_DONORS = 8
FORWARD_BATCH = 32_768
DISTANCE_BATCH = 1_024

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
    **{f"clean_{i}": f"clean MLP seed {seed}" for i, seed in enumerate(MODEL_SEEDS)},
    **{
        f"shuffled_{i}": f"shuffled MLP seed {seed}"
        for i, seed in enumerate(MODEL_SEEDS)
    },
    "ridge": "ridge control",
}
UNITS = {
    "input": "source feature units",
    "target": "MPa",
    "prediction": "MPa",
    "base": "MPa",
    "attribution": "MPa",
    "game": "MPa",
    "squared_error": "MPa squared",
    "distance": "train-standardized Euclidean distance",
}
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
DEEP_METHOD_NAMES = (
    "rescale_multipliers",
    "background_deepshap",
    "coalition_masks",
    "coalition_values",
    "exact_shapley",
    "gradient_delta",
)


def _float_tensor(value: Any, name: str, dimensions: int) -> t.Tensor:
    tensor = value if isinstance(value, t.Tensor) else t.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor or array.")
    if tensor.ndim != dimensions or not bool(t.isfinite(tensor).all()):
        raise ValueError(f"{name} has the wrong shape or contains nonfinite values.")
    return tensor.detach().clone()


def _ids_tensor(value: Any, rows: int, name: str) -> t.Tensor:
    tensor = value if isinstance(value, t.Tensor) else t.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.dtype != t.int64:
        raise TypeError(f"{name} must be a CPU int64 tensor or array.")
    if tensor.shape != (rows,) or len(t.unique(tensor)) != rows:
        raise ValueError(f"{name} must contain one unique identity per row.")
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
    names = tuple(splits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            if bool(t.isin(splits[left_name], splits[right_name]).any()):
                raise ValueError(
                    f"{left_name} and {right_name} source-row identities must be disjoint."
                )


def _clone_state(state: Mapping[str, t.Tensor]) -> dict[str, t.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError("A model state must be a tensor mapping.")
    result: dict[str, t.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, t.Tensor):
            raise TypeError("State keys must be strings and values must be tensors.")
        if value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError("Every model-state tensor must be CPU float64.")
        if not bool(t.isfinite(value).all()):
            raise ValueError("Model-state tensors must be finite.")
        result[name] = value.detach().clone()
    return result


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


def _model_from_state(record: Mapping[str, Any]) -> t.nn.Sequential:
    """Rebuild one supported raw-input/raw-MPa model from its retained state."""
    if not isinstance(record, Mapping) or not {"kind", "state"} <= set(record):
        raise TypeError("A fitted model record must contain kind and state.")
    kind = str(record["kind"])
    state = _clone_state(record["state"])
    shapes = _state_shapes(kind)
    if set(state) != set(shapes):
        raise ValueError(f"{kind} state has unexpected parameter names.")
    if any(tuple(state[name].shape) != shape for name, shape in shapes.items()):
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
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


make_model = _model_from_state


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


def _training_normalization(train: Mapping[str, Any]) -> dict[str, t.Tensor]:
    """Compute the four train-only population moments used by every fit."""
    x_raw, y_raw, _ = _rows(train, "train")
    normalization = {
        "x_mean": x_raw.mean(0),
        "x_std": x_raw.std(0, correction=0),
        "y_mean": y_raw.mean(),
        "y_std": y_raw.std(correction=0),
    }
    _normalization_tensors(normalization)
    return normalization


def _train_mlp(
    model: t.nn.Sequential,
    x_standardized: t.Tensor,
    target_standardized: t.Tensor,
) -> tuple[dict[str, t.Tensor], t.Tensor]:
    """Run exactly 800 full-batch float64 Adam updates and retain every loss."""
    optimizer = t.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    losses: list[t.Tensor] = []
    model.train()
    with t.enable_grad():
        for _ in range(TRAINING_STEPS):
            optimizer.zero_grad()
            loss = (model(x_standardized).squeeze(-1) - target_standardized).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().clone())
    return _clone_state(model.state_dict()), t.stack(losses)


def _fold_mlp_state(
    state: Mapping[str, t.Tensor], normalization: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Fold train normalization into the first and last MLP affine layers."""
    folded = _clone_state(state)
    if set(folded) != set(_state_shapes("mlp")):
        raise ValueError("Cannot fold an unexpected MLP state.")
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    first_weight = folded["0.weight"].clone()
    folded["0.weight"] = first_weight / x_std[None, :]
    folded["0.bias"] = folded["0.bias"] - first_weight @ (x_mean / x_std)
    folded["4.weight"] = y_std * folded["4.weight"]
    folded["4.bias"] = y_std * folded["4.bias"] + y_mean
    return folded


def _fold_ridge_state(
    state: Mapping[str, t.Tensor], normalization: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Fold standardized ridge coefficients into raw feature and MPa units."""
    folded = _clone_state(state)
    if set(folded) != set(_state_shapes("ridge")):
        raise ValueError("Cannot fold an unexpected ridge state.")
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    weight = folded["0.weight"].clone()
    folded["0.weight"] = y_std * weight / x_std[None, :]
    folded["0.bias"] = y_std * (
        folded["0.bias"] - (weight * (x_mean / x_std)[None, :]).sum(1)
    ) + y_mean
    return folded


def _folding_error(
    kind: str,
    standardized_state: Mapping[str, t.Tensor],
    raw_state: Mapping[str, t.Tensor],
    normalization: Mapping[str, Any],
    x_raw: t.Tensor,
) -> tuple[t.Tensor, float]:
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    standard = _model_from_state({"kind": kind, "state": standardized_state})
    raw = _model_from_state({"kind": kind, "state": raw_state})
    with t.no_grad():
        expected = standard((x_raw - x_mean) / x_std).squeeze(-1) * y_std + y_mean
        prediction = raw(x_raw).squeeze(-1)
    return prediction, float((prediction - expected).abs().max())


def fit_one_mlp_control(
    train: Mapping[str, Any],
    normalization: Mapping[str, Any],
    seed: int,
    shuffle_seed: int | None,
) -> dict[str, Any]:
    """Fit and fully retain one clean or shuffled 8-32-32-1 control."""
    if seed not in MODEL_SEEDS:
        raise ValueError("seed must be one of the three frozen model seeds.")
    if shuffle_seed is not None and shuffle_seed not in SHUFFLE_SEEDS:
        raise ValueError("shuffle_seed must be a frozen paired shuffle seed or None.")
    x_raw, y_raw, _ = _rows(train, "train")
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std
    t.manual_seed(seed)
    model = t.nn.Sequential(
        t.nn.Linear(FEATURES, HIDDEN, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(HIDDEN, HIDDEN, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(HIDDEN, 1, dtype=t.float64),
    )
    initial_state = _clone_state(model.state_dict())
    permutation = t.arange(len(y), dtype=t.int64)
    if shuffle_seed is not None:
        permutation = t.from_numpy(np.random.default_rng(shuffle_seed).permutation(len(y)))
    standardized_state, loss_trace = _train_mlp(model, x, y[permutation])
    raw_state = _fold_mlp_state(standardized_state, normalization)
    prediction, folding_error = _folding_error(
        "mlp", standardized_state, raw_state, normalization, x_raw
    )
    if folding_error > 1e-10:
        raise AssertionError("Raw-unit MLP folding changed its training predictions.")
    return {
        "kind": "mlp",
        "control": "clean" if shuffle_seed is None else "shuffled",
        "seed": seed,
        "initial_state": initial_state,
        "standardized_state": standardized_state,
        "state": raw_state,
        "loss_trace": loss_trace,
        "target_permutation": permutation,
        "training_targets_raw": y_raw[permutation].clone(),
        "training_prediction": prediction,
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
    train: Mapping[str, Any], normalization: Mapping[str, Any]
) -> dict[str, Any]:
    """Solve the fixed penalty-0.01 standardized ridge normal equations once."""
    x_raw, y_raw, _ = _rows(train, "train")
    x_mean, x_std, y_mean, y_std = _normalization_tensors(normalization)
    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std
    design = t.cat([x, t.ones((len(x), 1), dtype=t.float64)], dim=1)
    penalty = t.eye(FEATURES + 1, dtype=t.float64) * RIDGE_PENALTY
    penalty[-1, -1] = 0.0
    coefficient = t.linalg.solve(
        design.T @ design / len(x) + penalty,
        design.T @ y / len(x),
    )
    standardized_state = {
        "0.weight": coefficient[:FEATURES][None, :].clone(),
        "0.bias": coefficient[FEATURES:].clone(),
    }
    raw_state = _fold_ridge_state(standardized_state, normalization)
    prediction, folding_error = _folding_error(
        "ridge", standardized_state, raw_state, normalization, x_raw
    )
    if folding_error > 1e-10:
        raise AssertionError("Raw-unit ridge folding changed its training predictions.")
    standard_model = _model_from_state({"kind": "ridge", "state": standardized_state})
    with t.no_grad():
        ridge_loss = (standard_model(x).squeeze(-1) - y).square().mean()
    return {
        "kind": "ridge",
        "control": "ridge",
        "seed": 0,
        "initial_state": {},
        "standardized_state": standardized_state,
        "state": raw_state,
        "loss_trace": ridge_loss[None].clone(),
        "target_permutation": t.arange(len(y), dtype=t.int64),
        "training_targets_raw": y_raw.clone(),
        "training_prediction": prediction,
        "folding_max_abs_error": folding_error,
        "parameters": {
            "penalty": RIDGE_PENALTY,
            "penalize_intercept": False,
            "risk": "mean_squared_error",
            "dtype": "float64",
            "parameter_count": FEATURES + 1,
        },
    }


def fit_models(train: Mapping[str, Any]) -> dict[str, Any]:
    """Refit all seven known predictors from the pinned training rows."""
    x_raw, y_raw, train_ids = _rows(train, "train")
    t.set_num_threads(4)
    t.use_deterministic_algorithms(True)
    normalization = _training_normalization(train)
    models: dict[str, Any] = {}
    for index, seed in enumerate(MODEL_SEEDS):
        models[f"clean_{index}"] = fit_one_mlp_control(
            train, normalization, seed, None
        )
    for index, seed in enumerate(MODEL_SEEDS):
        models[f"shuffled_{index}"] = fit_one_mlp_control(
            train, normalization, seed, SHUFFLE_SEEDS[index]
        )
    models["ridge"] = fit_ridge_control(train, normalization)
    for record in models.values():
        record["state_hashes"] = {
            "initial_state": tensor_mapping_sha256(record["initial_state"]),
            "standardized_state": tensor_mapping_sha256(record["standardized_state"]),
            "raw_folded_state": tensor_mapping_sha256(record["state"]),
        }
    return {
        "models": models,
        "normalization": normalization,
        "train_ids": train_ids,
        "train_x_raw": x_raw,
        "train_y_raw": y_raw,
        "units": dict(UNITS),
        "feature_names": FEATURE_NAMES,
        "model_labels": dict(MODEL_LABELS),
        "phase": "fitted_before_attribution",
    }


def _validate_fitted(fitted: Mapping[str, Any]) -> None:
    if not isinstance(fitted, Mapping) or fitted.get("phase") != "fitted_before_attribution":
        raise ValueError("Use the complete train-only fitted payload.")
    if not isinstance(fitted.get("models"), Mapping):
        raise ValueError("The fitted payload must contain all model records.")
    if tuple(fitted["models"]) != MODEL_KEYS:
        raise ValueError("The fitted payload must retain all seven models in frozen order.")
    _normalization_tensors(fitted.get("normalization"))
    x = _float_tensor(fitted.get("train_x_raw"), "train_x_raw", 2)
    y = _float_tensor(fitted.get("train_y_raw"), "train_y_raw", 1)
    ids = _ids_tensor(fitted.get("train_ids"), len(y), "train_ids")
    if x.shape != (len(y), FEATURES) or len(ids) != len(y):
        raise ValueError("Fitted training arrays have incompatible shapes.")
    for key in MODEL_KEYS:
        expected_kind = "ridge" if key == "ridge" else "mlp"
        if fitted["models"][key].get("kind") != expected_kind:
            raise ValueError(f"{key} has the wrong model kind.")
        _model_from_state(fitted["models"][key])


def verify_16_14_reproduction(
    fitted: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Hard-fail unless all 21 states and every deterministic fit identity agree."""
    _validate_fitted(fitted)
    evidence = manifest.get("known_16_14_evidence", {})
    expected_hashes = evidence.get("model_state_hashes", {})
    if tuple(expected_hashes) != MODEL_KEYS:
        raise ValueError("Manifest must contain all seven known 16.14 state records.")
    observed: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    state_fields = {
        "initial_state": "initial_state",
        "standardized_state": "standardized_state",
        "raw_folded_state": "state",
    }
    for key in MODEL_KEYS:
        record = fitted["models"][key]
        observed[key] = {
            label: tensor_mapping_sha256(record[field])
            for label, field in state_fields.items()
        }
        for label in state_fields:
            if observed[key][label] != expected_hashes[key][label]:
                mismatches.append(f"{key}.{label}")
        if record.get("state_hashes") != observed[key]:
            mismatches.append(f"{key}.retained_state_hashes")

    expected_train_ids = t.as_tensor(manifest.get("splits", {}).get("train"), dtype=t.int64)
    train_ids_match = t.equal(fitted["train_ids"], expected_train_ids)
    if not train_ids_match:
        mismatches.append("train_ids")
    normalization = _training_normalization(
        {
            "x": fitted["train_x_raw"],
            "y": fitted["train_y_raw"],
            "ids": fitted["train_ids"],
        }
    )
    if any(
        not t.equal(normalization[name], fitted["normalization"][name])
        for name in normalization
    ):
        mismatches.append("normalization")

    recipe = manifest.get("model_recipe", {})
    training_recipe_match = (
        recipe.get("model_keys") == list(MODEL_KEYS)
        and recipe.get("model_seeds") == list(MODEL_SEEDS)
        and recipe.get("shuffle_seeds") == list(SHUFFLE_SEEDS)
        and recipe.get("training_steps") == TRAINING_STEPS
        and recipe.get("learning_rate") == LEARNING_RATE
        and recipe.get("weight_decay") == WEIGHT_DECAY
        and recipe.get("ridge_penalty") == RIDGE_PENALTY
    )
    for index, key in enumerate(MODEL_KEYS):
        record = fitted["models"][key]
        expected_permutation = t.arange(len(fitted["train_ids"]), dtype=t.int64)
        if key.startswith("shuffled_"):
            expected_permutation = t.from_numpy(
                np.random.default_rng(SHUFFLE_SEEDS[index - 3]).permutation(
                    len(fitted["train_ids"])
                )
            )
        if not t.equal(record.get("target_permutation"), expected_permutation):
            mismatches.append(f"{key}.target_permutation")
        expected_targets = fitted["train_y_raw"][expected_permutation]
        if not t.equal(record.get("training_targets_raw"), expected_targets):
            mismatches.append(f"{key}.training_targets_raw")
        expected_steps = 1 if key == "ridge" else TRAINING_STEPS
        trace = record.get("loss_trace")
        if not isinstance(trace, t.Tensor) or trace.shape != (expected_steps,):
            mismatches.append(f"{key}.loss_trace")
        with t.no_grad():
            prediction = _model_from_state(record)(fitted["train_x_raw"]).squeeze(-1)
        if not t.equal(record.get("training_prediction"), prediction):
            mismatches.append(f"{key}.training_prediction")
        if float(record.get("folding_max_abs_error", float("inf"))) > 1e-10:
            mismatches.append(f"{key}.folding_max_abs_error")
    if not training_recipe_match:
        mismatches.append("training_recipe")
    if mismatches:
        raise AssertionError(
            "16.14 reproduction mismatch before attribution: " + ", ".join(mismatches)
        )
    return {
        "status": "PASS",
        "known_section": "16.14",
        "fitted_artifact_sha256": evidence.get("fitted_artifact_sha256"),
        "model_state_hashes": observed,
        "expected_model_state_hashes": expected_hashes,
        "mismatches": (),
        "train_ids_match": train_ids_match,
        "training_recipe_match": training_recipe_match,
    }


def _require_namespace(namespace: Any, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not callable(getattr(namespace, name, None))]
    if missing:
        raise TypeError(f"{label} namespace is missing {missing}.")


def _output_tensor(value: Any, shape: tuple[int, ...], name: str) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        raise TypeError(f"{name} must be a Torch tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if tuple(value.shape) != shape or not bool(t.isfinite(value).all()):
        raise ValueError(f"{name} has the wrong shape or contains nonfinite values.")
    return value


def _background_indices(
    train_ids: Any, manifest: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Recreate all five background index arrays and verify every donor identity."""
    ids = _ids_tensor(train_ids, len(train_ids), "train_ids")
    declared = manifest.get("backgrounds", {})
    if tuple(declared) != tuple(BACKGROUND_SPECS):
        raise ValueError("Manifest background names or order changed.")
    result: dict[str, t.Tensor] = {}
    for name, (seed, size) in BACKGROUND_SPECS.items():
        order = t.from_numpy(background_order(ids.numpy(), seed))
        indices = order[:size].clone()
        item = declared[name]
        if item.get("seed") != seed or item.get("rows") != size:
            raise ValueError(f"{name} changed its seed or donor count.")
        expected_indices = t.as_tensor(item.get("train_indices"), dtype=t.int64)
        expected_ids = t.as_tensor(item.get("source_ids"), dtype=t.int64)
        if not t.equal(indices, expected_indices) or not t.equal(ids[indices], expected_ids):
            raise ValueError(f"{name} donor indices or source identities changed.")
        result[name] = indices
    return result


def _primary_schedule(
    rows: Mapping[str, Any], background_rows: int, methods: Any
) -> dict[str, t.Tensor]:
    """Create the one shared K=2048 primary schedule for a named row block."""
    _require_namespace(methods, METHOD_NAMES, "Student methods")
    _, _, source_ids = _rows(rows, "rows")
    indices, alpha = methods.sample_path_schedule(
        source_ids, background_rows, max(BUDGETS), PRIMARY_SCHEDULE_SEED
    )
    return {"source_ids": source_ids, "baseline_index": indices, "alpha": alpha}


def _monte_carlo_block(
    model: t.nn.Module,
    rows: Mapping[str, Any],
    background: t.Tensor,
    schedule: Mapping[str, t.Tensor],
    methods: Any,
) -> dict[str, Any]:
    """Retain all per-draw contributions and every declared nested prefix."""
    x, _, source_ids = _rows(rows, "rows")
    if not t.equal(source_ids, schedule["source_ids"]):
        raise ValueError("Schedule source identities do not match the explained rows.")
    samples = methods.expected_gradient_samples(
        model, x, background, schedule["baseline_index"], schedule["alpha"], FORWARD_BATCH
    )
    samples = _output_tensor(samples, (len(x), max(BUDGETS), FEATURES), "samples")
    phi_prefix, se_prefix = methods.summarize_expected_gradients(samples, BUDGETS)
    phi_prefix = _output_tensor(phi_prefix, (len(BUDGETS), len(x), FEATURES), "phi_prefix")
    se_prefix = _output_tensor(se_prefix, (len(BUDGETS), len(x), FEATURES), "se_prefix")
    with t.no_grad():
        base = model(background).squeeze(-1).mean()
        prediction = model(x).squeeze(-1)
    residual = phi_prefix.sum(-1) - (prediction[None, :] - base)
    return {
        "source_ids": source_ids,
        "baseline_index": schedule["baseline_index"],
        "alpha": schedule["alpha"],
        "samples": samples,
        "budgets": t.tensor(BUDGETS, dtype=t.int64),
        "phi_prefix": phi_prefix,
        "se_prefix": se_prefix,
        "base_full": base.detach(),
        "prediction": prediction.detach(),
        "completeness_residual_prefix": residual.detach(),
    }


def _quadrature_block(
    model: t.nn.Module,
    rows: Mapping[str, Any],
    background: t.Tensor,
    methods: Any,
) -> dict[str, t.Tensor]:
    """Retain donor-wise Q=256/Q=512 numerical path controls and drift."""
    x, _, _ = _rows(rows, "rows")
    low, high = QUADRATURE_NODES
    pair_low = methods.gauss_legendre_pairs(
        model, x, background, low, row_batch=4, forward_batch=FORWARD_BATCH
    )
    pair_high = methods.gauss_legendre_pairs(
        model, x, background, high, row_batch=4, forward_batch=FORWARD_BATCH
    )
    shape = (len(x), len(background), FEATURES)
    pair_low = _output_tensor(pair_low, shape, f"pair_{low}")
    pair_high = _output_tensor(pair_high, shape, f"pair_{high}")
    mean_low, mean_high = pair_low.mean(1), pair_high.mean(1)
    return {
        f"pair_{low}": pair_low,
        f"pair_{high}": pair_high,
        f"mean_{low}": mean_low,
        f"mean_{high}": mean_high,
        "drift": mean_high - mean_low,
    }


def _coalition_hybrid_predictions(
    model: t.nn.Module,
    x: t.Tensor,
    background: t.Tensor,
    masks: t.Tensor,
    *,
    row_batch: int = 4,
    forward_batch: int = FORWARD_BATCH,
) -> t.Tensor:
    """Retain every query/coalition/donor prediction in bounded forward batches."""
    result = t.empty((len(x), len(masks), len(background)), dtype=t.float64)
    with t.no_grad():
        for start in range(0, len(x), row_batch):
            query = x[start : start + row_batch]
            hybrids = t.where(
                masks[None, :, None, :],
                query[:, None, None, :],
                background[None, None, :, :],
            )
            flat = hybrids.reshape(-1, FEATURES)
            output = t.cat(
                [
                    model(flat[index : index + forward_batch]).squeeze(-1)
                    for index in range(0, len(flat), forward_batch)
                ]
            )
            result[start : start + len(query)] = output.reshape(
                len(query), len(masks), len(background)
            )
    return result


def _comparison_baselines(
    model: t.nn.Module,
    rows: Mapping[str, Any],
    background: t.Tensor,
    deep_methods: Any,
) -> dict[str, t.Tensor]:
    """Retain exact-game hybrids, DeepSHAP pairs, and endpoint gradients."""
    _require_namespace(deep_methods, DEEP_METHOD_NAMES, "DeepSHAP control")
    x, _, _ = _rows(rows, "rows")
    with t.no_grad():
        prediction = model(x).squeeze(-1)
    masks = deep_methods.coalition_masks(FEATURES)
    hybrid_prediction = _coalition_hybrid_predictions(model, x, background, masks)
    coalition_values = hybrid_prediction.mean(-1)
    phi_exact = deep_methods.exact_shapley(coalition_values)
    base_deep, phi_deep = deep_methods.background_deepshap(model, x, background)
    multipliers = deep_methods.rescale_multipliers(model, x, background)
    deep_pair = multipliers * (x[:, None, :] - background[None, :, :])
    phi_gradient = deep_methods.gradient_delta(model, x, background)
    return {
        "prediction": _output_tensor(prediction, (len(x),), "prediction"),
        "hybrid_prediction": _output_tensor(
            hybrid_prediction,
            (len(x), 2**FEATURES, len(background)),
            "hybrid_prediction",
        ),
        "coalition_values": _output_tensor(
            coalition_values, (len(x), 2**FEATURES), "coalition_values"
        ),
        "phi_exact": _output_tensor(phi_exact, (len(x), FEATURES), "phi_exact"),
        "base_deep": _output_tensor(base_deep, (), "base_deep"),
        "phi_deep": _output_tensor(phi_deep, (len(x), FEATURES), "phi_deep"),
        "deep_pair": _output_tensor(
            deep_pair, (len(x), len(background), FEATURES), "deep_pair"
        ),
        "phi_gradient": _output_tensor(
            phi_gradient, (len(x), FEATURES), "phi_gradient"
        ),
    }


def _global_rankings(primary: Mapping[str, Any]) -> dict[str, dict[str, t.Tensor]]:
    """Freeze stable mean-absolute feature orders from explanation rows only."""
    fields = {
        "exact": lambda item: item["baselines"]["phi_exact"],
        "deep": lambda item: item["baselines"]["phi_deep"],
        "gradient": lambda item: item["baselines"]["phi_gradient"],
        "eg_k2048": lambda item: item["mc"]["phi_prefix"][-1],
        "quadrature_512": lambda item: item["quadrature"][f"mean_{QUADRATURE_NODES[1]}"],
    }
    return {
        method: {
            key: t.argsort(getter(primary[key]).abs().mean(0), descending=True, stable=True)
            for key in MODEL_KEYS
        }
        for method, getter in fields.items()
    }


def _fixed_random_orders() -> t.Tensor:
    rng = np.random.default_rng(RANDOM_ORDER_SEED)
    return t.from_numpy(np.stack([rng.permutation(FEATURES) for _ in range(100)]))


def _identity_hash(**arrays: Any) -> str:
    return tensor_mapping_sha256(arrays)


def prepare_study(
    fitted: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    *,
    methods: Any,
    deep_methods: Any,
) -> dict[str, Any]:
    """Prepare and seal all explanation-side choices before audit is supplied."""
    _require_namespace(methods, METHOD_NAMES, "Student methods")
    _require_namespace(deep_methods, DEEP_METHOD_NAMES, "DeepSHAP control")
    manifest = load_manifest()
    reproduction = verify_16_14_reproduction(fitted, manifest)
    train_x, train_y, train_ids = _rows(train, "train")
    explain_x, explain_y, explain_ids = _rows(explain, "explain")
    _require_disjoint_ids(train=train_ids, explanation=explain_ids)
    if not t.equal(train_ids, fitted["train_ids"]):
        raise ValueError("Training identities changed after fitting.")
    if not t.equal(train_x, fitted["train_x_raw"]) or not t.equal(
        train_y, fitted["train_y_raw"]
    ):
        raise ValueError("Training values changed after fitting.")
    expected_explain = t.as_tensor(manifest["splits"]["explain"], dtype=t.int64)
    if not t.equal(explain_ids, expected_explain):
        raise ValueError("Explanation identities differ from the pinned split.")

    indices = _background_indices(train_ids, manifest)
    backgrounds = {
        name: {
            "seed": BACKGROUND_SPECS[name][0],
            "indices": index,
            "source_ids": train_ids[index],
            "x_raw": train_x[index],
        }
        for name, index in indices.items()
    }
    primary_background = backgrounds["primary64"]["x_raw"]
    schedule = _primary_schedule(explain, len(primary_background), methods)
    primary: dict[str, Any] = {}
    for key in MODEL_KEYS:
        model = _model_from_state(fitted["models"][key])
        primary[key] = {
            "mc": _monte_carlo_block(model, explain, primary_background, schedule, methods),
            "quadrature": _quadrature_block(model, explain, primary_background, methods),
            "baselines": _comparison_baselines(
                model, explain, primary_background, deep_methods
            ),
        }
    random_orders = _fixed_random_orders()
    schedule_manifest = {
        "namespace": "arena16.15/gradientshap/v1",
        "seed": PRIMARY_SCHEDULE_SEED,
        "budgets": t.tensor(BUDGETS, dtype=t.int64),
        "source_ids": explain_ids,
        "baseline_index_sha256": _identity_hash(value=schedule["baseline_index"]),
        "alpha_sha256": _identity_hash(value=schedule["alpha"]),
    }
    payload_identities = {
        "train": _identity_hash(ids=train_ids, x=train_x, y=train_y),
        "explain": _identity_hash(ids=explain_ids, x=explain_x, y=explain_y),
        "backgrounds": {
            name: _identity_hash(
                indices=item["indices"], ids=item["source_ids"], x=item["x_raw"]
            )
            for name, item in backgrounds.items()
        },
    }
    return {
        "phase": "prepared_before_audit",
        "audit_used_for_selection": False,
        "reproduction": reproduction,
        "explain_ids": explain_ids,
        "explain_x_raw": explain_x,
        "explain_targets": explain_y,
        "backgrounds": backgrounds,
        "primary": primary,
        "rankings": _global_rankings(primary),
        "random_orders": random_orders,
        "schedule_manifest": schedule_manifest,
        "payload_identities": payload_identities,
        "units": dict(UNITS),
        "feature_names": FEATURE_NAMES,
        "model_labels": dict(MODEL_LABELS),
    }


def _performance_table(
    fitted: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Return audit MSE for every model and the train-mean predictor."""
    audit_x, audit_y, _ = _rows(audit, "audit")
    result: dict[str, Any] = {}
    with t.no_grad():
        for key in MODEL_KEYS:
            prediction = _model_from_state(fitted["models"][key])(audit_x).squeeze(-1)
            result[key] = {
                "kind": fitted["models"][key]["kind"],
                "control": fitted["models"][key]["control"],
                "prediction": prediction,
                "mse": (prediction - audit_y).square().mean(),
            }
    result["training_mean_mse"] = (
        audit_y - fitted["train_y_raw"].mean()
    ).square().mean()
    return result


def _combine_mc(
    explanation: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    if not t.equal(explanation["budgets"], audit["budgets"]):
        raise ValueError("Nested budgets changed between explanation and audit blocks.")
    if not t.equal(explanation["base_full"], audit["base_full"]):
        raise ValueError("Primary background base changed across the audit boundary.")
    return {
        "source_ids": t.cat([explanation["source_ids"], audit["source_ids"]]),
        "baseline_index": t.cat(
            [explanation["baseline_index"], audit["baseline_index"]]
        ),
        "alpha": t.cat([explanation["alpha"], audit["alpha"]]),
        "samples": t.cat([explanation["samples"], audit["samples"]]),
        "budgets": explanation["budgets"],
        "phi_prefix": t.cat(
            [explanation["phi_prefix"], audit["phi_prefix"]], dim=1
        ),
        "se_prefix": t.cat(
            [explanation["se_prefix"], audit["se_prefix"]], dim=1
        ),
        "base_full": explanation["base_full"],
        "prediction": t.cat([explanation["prediction"], audit["prediction"]]),
        "completeness_residual_prefix": t.cat(
            [
                explanation["completeness_residual_prefix"],
                audit["completeness_residual_prefix"],
            ],
            dim=1,
        ),
    }


def _combine_quadrature(
    explanation: Mapping[str, t.Tensor], audit: Mapping[str, t.Tensor]
) -> dict[str, t.Tensor]:
    result: dict[str, t.Tensor] = {}
    for name in explanation:
        result[name] = t.cat([explanation[name], audit[name]], dim=0)
    return result


def _combine_baselines(
    explanation: Mapping[str, t.Tensor], audit: Mapping[str, t.Tensor]
) -> dict[str, t.Tensor]:
    result: dict[str, t.Tensor] = {}
    for name in explanation:
        if name == "base_deep":
            if not t.equal(explanation[name], audit[name]):
                raise ValueError("Primary DeepSHAP base changed across row blocks.")
            result[name] = explanation[name]
        else:
            result[name] = t.cat([explanation[name], audit[name]], dim=0)
    return result


def _combine_primary(
    prepared: Mapping[str, Any], audit_primary: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: {
            "mc": _combine_mc(prepared["primary"][key]["mc"], audit_primary[key]["mc"]),
            "quadrature": _combine_quadrature(
                prepared["primary"][key]["quadrature"],
                audit_primary[key]["quadrature"],
            ),
            "baselines": _combine_baselines(
                prepared["primary"][key]["baselines"],
                audit_primary[key]["baselines"],
            ),
        }
        for key in MODEL_KEYS
    }


def _background_block(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    background: Mapping[str, Any],
    methods: Any,
    deep_methods: Any,
) -> dict[str, Any]:
    model = _model_from_state(fitted["models"]["clean_0"])
    schedule = _primary_schedule(rows, len(background["x_raw"]), methods)
    return {
        "source_ids": _rows(rows, "rows")[2],
        "targets": _rows(rows, "rows")[1],
        "background_seed": background["seed"],
        "background_indices": background["indices"],
        "background_source_ids": background["source_ids"],
        "background_x_raw": background["x_raw"],
        "mc": _monte_carlo_block(
            model, rows, background["x_raw"], schedule, methods
        ),
        "quadrature": _quadrature_block(
            model, rows, background["x_raw"], methods
        ),
        "baselines": _comparison_baselines(
            model, rows, background["x_raw"], deep_methods
        ),
    }


def _background_comparison(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    backgrounds: Mapping[str, Any],
    methods: Any,
    deep_methods: Any,
) -> dict[str, Any]:
    """Compute complete clean_0 blocks for every supplied finite background."""
    return {
        name: _background_block(
            fitted, rows, background, methods, deep_methods
        )
        for name, background in backgrounds.items()
    }


def _seed_words(seed: int, source_id: int, stream_tag: str) -> int:
    name = f"arena16.15/gradientshap/v1/{seed}/{source_id}/{stream_tag}"
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:16], "little")


def _standard_normal_schedule(source_ids: t.Tensor, samples: int) -> t.Tensor:
    arrays = []
    for source_id in source_ids.tolist():
        rng = np.random.Generator(
            np.random.PCG64(
                _seed_words(NOISE_SCHEDULE_SEED, int(source_id), "standard_normal")
            )
        )
        arrays.append(rng.standard_normal((samples, FEATURES)))
    return t.tensor(np.stack(arrays), dtype=t.float64)


def _batched_prediction(model: t.nn.Module, values: t.Tensor) -> t.Tensor:
    flat = values.reshape(-1, FEATURES)
    with t.no_grad():
        prediction = t.cat(
            [
                model(flat[start : start + FORWARD_BATCH]).squeeze(-1)
                for start in range(0, len(flat), FORWARD_BATCH)
            ]
        )
    return prediction.reshape(values.shape[:-1])


def _noise_comparison(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    primary: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Retain clean and positive-noise estimators with both completeness targets."""
    x, _, source_ids = _rows(rows, "rows")
    model = _model_from_state(fitted["models"]["clean_0"])
    clean = primary["clean_0"]["mc"]
    samples = clean["samples"].shape[1]
    standard_normal = _standard_normal_schedule(source_ids, samples)
    _, x_std, _, _ = _normalization_tensors(fitted["normalization"])
    base = clean["base_full"]
    clean_prediction = clean["prediction"]
    result: dict[str, Any] = {
        "standard_normal": standard_normal,
        "source_ids": source_ids,
    }
    for sigma in NOISE_SIGMAS:
        name = "sigma_0" if sigma == 0 else f"sigma_{sigma:.2f}".replace(".", "_")
        if sigma == 0:
            noise_raw = standard_normal * 0.0
            noisy_prediction = clean_prediction[:, None].expand(-1, samples)
            contribution = clean["samples"]
        else:
            noise_raw = standard_normal * (sigma * x_std)[None, None, :]
            contribution = methods.noisy_expected_gradient_samples(
                model,
                x,
                primary["_background_x_raw"],
                clean["baseline_index"],
                clean["alpha"],
                noise_raw,
                FORWARD_BATCH,
            )
            noisy_prediction = _batched_prediction(model, x[:, None, :] + noise_raw)
        phi, se = methods.summarize_expected_gradients(contribution, (samples,))
        smoothed_prediction = noisy_prediction.mean(1)
        smoothed_prediction_se = noisy_prediction.std(1, correction=1) / samples**0.5
        result[name] = {
            "sigma_standardized": sigma,
            "noise_raw": noise_raw,
            "samples": contribution,
            "noisy_prediction_samples": noisy_prediction,
            "phi": phi[0],
            "se": se[0],
            "clean_target_residual": phi[0].sum(-1)
            - (clean_prediction - base),
            "smoothed_target_residual": phi[0].sum(-1)
            - (smoothed_prediction - base),
            "smoothed_target_se": smoothed_prediction_se,
        }
    return result


def _replicate_seed(replicate: int) -> int:
    name = (
        "arena16.15/gradientshap/v1/"
        f"{REPLICATE_SCHEDULE_SEED}/replicate/{replicate}"
    )
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def _replicate_comparison(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    primary: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Retain 64 shared schedules and all three-model per-draw contributions."""
    x, _, source_ids = _rows(rows, "replicate rows")
    background = primary["_background_x_raw"]
    design = load_manifest()["replicate_diagnostic_candidate"]
    explain_ids = t.as_tensor(design["explanation_source_ids"], dtype=t.int64)
    audit_ids = t.as_tensor(design["audit_source_ids"], dtype=t.int64)
    if not t.equal(source_ids, t.cat([explain_ids, audit_ids])):
        raise ValueError("Replicate rows changed their frozen split boundary or order.")
    row_split = t.cat(
        [
            t.zeros(len(explain_ids), dtype=t.int64),
            t.ones(len(audit_ids), dtype=t.int64),
        ]
    )
    samples_per_replicate = max(BUDGETS)
    baseline_index = t.empty(
        (REPLICATES, len(x), samples_per_replicate), dtype=t.int64
    )
    alpha = t.empty(
        (REPLICATES, len(x), samples_per_replicate), dtype=t.float64
    )
    samples = t.empty(
        (
            REPLICATES,
            len(REPLICATE_MODELS),
            len(x),
            samples_per_replicate,
            FEATURES,
        ),
        dtype=t.float64,
    )
    seed_values = [_replicate_seed(index) for index in range(REPLICATES)]
    declared = load_manifest()["sampling_design_candidate"].get("replicate_seeds")
    if declared is not None and seed_values != declared[:REPLICATES]:
        raise ValueError("Derived replicate seeds differ from the frozen manifest.")
    seeds = t.tensor(seed_values, dtype=t.uint64)
    for replicate in range(REPLICATES):
        indices, draws = methods.sample_path_schedule(
            source_ids,
            len(background),
            samples_per_replicate,
            int(seeds[replicate].item()),
        )
        baseline_index[replicate] = indices
        alpha[replicate] = draws
        for model_index, key in enumerate(REPLICATE_MODELS):
            model = _model_from_state(fitted["models"][key])
            samples[replicate, model_index] = methods.expected_gradient_samples(
                model, x, background, indices, draws, FORWARD_BATCH
            )
    phi_prefix = t.empty(
        (REPLICATES, len(REPLICATE_MODELS), len(BUDGETS), len(x), FEATURES),
        dtype=t.float64,
    )
    se_prefix = t.empty_like(phi_prefix)
    for replicate in range(REPLICATES):
        for model_index in range(len(REPLICATE_MODELS)):
            phi, se = methods.summarize_expected_gradients(
                samples[replicate, model_index], BUDGETS
            )
            phi_prefix[replicate, model_index] = phi
            se_prefix[replicate, model_index] = se
    q512 = t.stack(
        [
            primary[key]["quadrature"][f"mean_{QUADRATURE_NODES[1]}"]
            for key in REPLICATE_MODELS
        ]
    )
    error = phi_prefix - q512[None, :, None, :, :]
    coverage = error.abs() <= 1.96 * se_prefix
    return {
        "source_ids": source_ids,
        "row_split": row_split,
        "split_labels": ("explanation", "audit"),
        "models": REPLICATE_MODELS,
        "model_labels": tuple(MODEL_LABELS[key] for key in REPLICATE_MODELS),
        "budgets": t.tensor(BUDGETS, dtype=t.int64),
        "replicate_seeds": seeds,
        "baseline_index": baseline_index,
        "alpha": alpha,
        "samples": samples,
        "phi_prefix": phi_prefix,
        "se_prefix": se_prefix,
        "q512": q512,
        "q512_error": error,
        "coverage_95": coverage,
        "coverage_rate": coverage.to(t.float64).mean(dim=(0, 3, 4)),
    }


def _nearest_train_distance(
    points: t.Tensor, train_standardized: t.Tensor
) -> t.Tensor:
    result = t.empty(len(points), dtype=t.float64)
    for start in range(0, len(points), DISTANCE_BATCH):
        block = points[start : start + DISTANCE_BATCH]
        result[start : start + len(block)] = t.cdist(block, train_standardized).min(1).values
    return result


def _distance_comparison(
    train: Mapping[str, Any],
    rows: Mapping[str, Any],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain every declared nearest-training distance and its exact identity."""
    train_x, _, train_ids = _rows(train, "train")
    x, _, source_ids = _rows(rows, "rows")
    x_mean = train_x.mean(0)
    x_std = train_x.std(0, correction=0)
    train_z = (train_x - x_mean) / x_std
    query_z = (x - x_mean) / x_std
    background = primary["_background_x_raw"]
    background_ids = primary["_background_source_ids"]
    schedule = primary["clean_0"]["mc"]

    path_indices = schedule["baseline_index"][:, :PATH_DISTANCE_SAMPLES]
    path_alpha = schedule["alpha"][:, :PATH_DISTANCE_SAMPLES]
    donors = background[path_indices]
    path_points = donors + path_alpha[:, :, None] * (x[:, None, :] - donors)
    path_distance = _nearest_train_distance(
        ((path_points - x_mean) / x_std).reshape(-1, FEATURES), train_z
    )

    masks = (
        t.arange(2**FEATURES, dtype=t.int64)[:, None]
        .bitwise_and(1 << t.arange(FEATURES, dtype=t.int64)[None, :])
        .ne(0)
    )
    donor_count = min(COALITION_DISTANCE_DONORS, len(background))
    coalition_distances: list[t.Tensor] = []
    for query in x:
        donor = background[:donor_count]
        hybrid = t.where(
            masks[None, :, :], query[None, None, :], donor[:, None, :]
        )
        coalition_distances.append(
            _nearest_train_distance(
                ((hybrid - x_mean) / x_std).reshape(-1, FEATURES), train_z
            )
        )
    coalition_distance = t.cat(coalition_distances)

    path_source = source_ids[:, None].expand(-1, PATH_DISTANCE_SAMPLES).reshape(-1)
    path_donor = background_ids[path_indices].reshape(-1)
    coalition_source = source_ids[:, None, None].expand(
        -1, donor_count, 2**FEATURES
    ).reshape(-1)
    coalition_donor = background_ids[:donor_count][None, :, None].expand(
        len(x), -1, 2**FEATURES
    ).reshape(-1)
    coalition_mask = t.arange(2**FEATURES, dtype=t.int64)[None, None, :].expand(
        len(x), donor_count, -1
    ).reshape(-1)
    return {
        "normalization": {"x_mean": x_mean, "x_std": x_std},
        "path": {
            "source_id": path_source,
            "donor_id": path_donor,
            "alpha": path_alpha.reshape(-1),
            "distance": path_distance,
        },
        "coalition": {
            "source_id": coalition_source,
            "donor_id": coalition_donor,
            "mask": coalition_mask,
            "distance": coalition_distance,
        },
        "query_distance": _nearest_train_distance(query_z, train_z),
        "donor_distance": _nearest_train_distance(
            (background[:donor_count] - x_mean) / x_std, train_z
        ),
        "donor_source_ids": background_ids[:donor_count],
        "scope": {
            "training_rows": len(train_ids),
            "query_rows": len(source_ids),
            "path_samples_per_row": PATH_DISTANCE_SAMPLES,
            "coalition_donors_per_row": donor_count,
            "coalition_masks": 2**FEATURES,
            "meaning": "geometry_not_density_or_ood_probability",
        },
    }


def _checked_curves(
    values: t.Tensor, phi: t.Tensor, order: t.Tensor, methods: Any
) -> dict[str, t.Tensor]:
    orders = order[None, :].expand(len(values), FEATURES).clone()
    result = methods.ranked_coalition_curves(values, phi, orders)
    expected = {
        "orders": (len(values), FEATURES),
        "insertion_masks": (len(values), FEATURES + 1),
        "deletion_masks": (len(values), FEATURES + 1),
        "insertion": (len(values), FEATURES + 1),
        "deletion": (len(values), FEATURES + 1),
    }
    if not isinstance(result, Mapping) or set(result) != set(expected):
        raise ValueError("Replacement curves returned an unexpected payload.")
    for name, shape in expected.items():
        value = result[name]
        if not isinstance(value, t.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Replacement curve {name} has the wrong shape.")
        if name in {"orders", "insertion_masks", "deletion_masks"}:
            if value.dtype != t.int64:
                raise TypeError(f"Replacement curve {name} must be int64.")
        else:
            _output_tensor(value, shape, name)
    return dict(result)


def _replacement_evaluation(
    prepared: Mapping[str, Any],
    audit_primary: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Apply frozen global and all random orders to every audit coalition table."""
    method_fields = {
        "exact": ("baselines", "phi_exact"),
        "deep": ("baselines", "phi_deep"),
        "gradient": ("baselines", "phi_gradient"),
        "eg_k2048": ("mc", "phi_prefix"),
        "quadrature_512": ("quadrature", f"mean_{QUADRATURE_NODES[1]}"),
    }
    result: dict[str, Any] = {}
    random_insertion = []
    random_deletion = []
    random_insertion_masks = None
    random_deletion_masks = None
    for key in MODEL_KEYS:
        values = audit_primary[key]["baselines"]["coalition_values"]
        result[key] = {}
        for method, (branch, field) in method_fields.items():
            phi = audit_primary[key][branch][field]
            if method == "eg_k2048":
                phi = phi[-1]
            result[key][method] = _checked_curves(
                values, phi, prepared["rankings"][method][key], methods
            )
        model_insertion = []
        model_deletion = []
        for order in prepared["random_orders"]:
            curve = _checked_curves(
                values, t.zeros((len(values), FEATURES), dtype=t.float64), order, methods
            )
            model_insertion.append(curve["insertion"])
            model_deletion.append(curve["deletion"])
            if random_insertion_masks is None:
                random_insertion_masks = []
                random_deletion_masks = []
            if key == MODEL_KEYS[0]:
                random_insertion_masks.append(curve["insertion_masks"])
                random_deletion_masks.append(curve["deletion_masks"])
        random_insertion.append(t.stack(model_insertion, dim=1))
        random_deletion.append(t.stack(model_deletion, dim=1))
    result["random"] = {
        "models": MODEL_KEYS,
        "orders": prepared["random_orders"],
        "insertion_masks": t.stack(random_insertion_masks, dim=1),
        "deletion_masks": t.stack(random_deletion_masks, dim=1),
        "insertion": t.stack(random_insertion),
        "deletion": t.stack(random_deletion),
    }
    return result


def _validate_evaluation_inputs(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, t.Tensor]:
    _validate_fitted(fitted)
    if prepared.get("phase") != "prepared_before_audit":
        raise ValueError("Persist the prepared payload before audit evaluation.")
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Audit rows cannot select models, schedules, backgrounds or rankings.")
    if prepared.get("reproduction", {}).get("status") != "PASS":
        raise ValueError("The 21-state reproduction gate must pass before evaluation.")
    train_x, train_y, train_ids = _rows(train, "train")
    explain_x, explain_y, explain_ids = _rows(explain, "explain")
    audit_x, audit_y, audit_ids = _rows(audit, "audit")
    _require_disjoint_ids(train=train_ids, explanation=explain_ids, audit=audit_ids)
    if not t.equal(train_ids, fitted["train_ids"]):
        raise ValueError("Training identities changed after fitting.")
    if not t.equal(train_x, fitted["train_x_raw"]) or not t.equal(
        train_y, fitted["train_y_raw"]
    ):
        raise ValueError("Training values changed after fitting.")
    if not t.equal(explain_ids, prepared["explain_ids"]):
        raise ValueError("Explanation identities changed after preparation.")
    if not t.equal(explain_x, prepared["explain_x_raw"]) or not t.equal(
        explain_y, prepared["explain_targets"]
    ):
        raise ValueError("Explanation values changed after preparation.")
    manifest = load_manifest()
    expected_audit = t.as_tensor(manifest["splits"]["audit"], dtype=t.int64)
    if not t.equal(audit_ids, expected_audit):
        raise ValueError("Audit identities differ from the literal pinned split.")
    identities = prepared.get("payload_identities", {})
    if identities.get("train") != _identity_hash(ids=train_ids, x=train_x, y=train_y):
        raise ValueError("Prepared training identity seal changed.")
    if identities.get("explain") != _identity_hash(
        ids=explain_ids, x=explain_x, y=explain_y
    ):
        raise ValueError("Prepared explanation identity seal changed.")
    expected_backgrounds = _background_indices(train_ids, manifest)
    for name, indices in expected_backgrounds.items():
        item = prepared.get("backgrounds", {}).get(name, {})
        if not t.equal(item.get("indices"), indices):
            raise ValueError(f"Prepared {name} background indices changed.")
        if not t.equal(item.get("source_ids"), train_ids[indices]):
            raise ValueError(f"Prepared {name} background identities changed.")
        if not t.equal(item.get("x_raw"), train_x[indices]):
            raise ValueError(f"Prepared {name} background values changed.")
        expected_hash = _identity_hash(
            indices=indices, ids=train_ids[indices], x=train_x[indices]
        )
        if identities.get("backgrounds", {}).get(name) != expected_hash:
            raise ValueError(f"Prepared {name} background seal changed.")
    primary_mc = prepared.get("primary", {}).get(MODEL_KEYS[0], {}).get("mc", {})
    schedule_manifest = prepared.get("schedule_manifest", {})
    if schedule_manifest.get("baseline_index_sha256") != _identity_hash(
        value=primary_mc.get("baseline_index")
    ):
        raise ValueError("Prepared donor schedule seal changed.")
    if schedule_manifest.get("alpha_sha256") != _identity_hash(
        value=primary_mc.get("alpha")
    ):
        raise ValueError("Prepared alpha schedule seal changed.")
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_ids": train_ids,
        "explain_x": explain_x,
        "explain_y": explain_y,
        "explain_ids": explain_ids,
        "audit_x": audit_x,
        "audit_y": audit_y,
        "audit_ids": audit_ids,
    }


def evaluate_study(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    methods: Any,
    deep_methods: Any,
) -> dict[str, Any]:
    """Evaluate every frozen control after receiving the literal audit split."""
    _require_namespace(methods, METHOD_NAMES, "Student methods")
    _require_namespace(deep_methods, DEEP_METHOD_NAMES, "DeepSHAP control")
    row_blocks = _validate_evaluation_inputs(fitted, prepared, train, explain, audit)
    primary_background = prepared["backgrounds"]["primary64"]["x_raw"]
    audit_schedule = _primary_schedule(audit, len(primary_background), methods)
    audit_primary: dict[str, Any] = {}
    for key in MODEL_KEYS:
        model = _model_from_state(fitted["models"][key])
        audit_primary[key] = {
            "mc": _monte_carlo_block(
                model, audit, primary_background, audit_schedule, methods
            ),
            "quadrature": _quadrature_block(model, audit, primary_background, methods),
            "baselines": _comparison_baselines(
                model, audit, primary_background, deep_methods
            ),
        }
    all_rows = _combine_primary(prepared, audit_primary)
    all_row_data = {
        "x": t.cat([row_blocks["explain_x"], row_blocks["audit_x"]]),
        "y": t.cat([row_blocks["explain_y"], row_blocks["audit_y"]]),
        "ids": t.cat([row_blocks["explain_ids"], row_blocks["audit_ids"]]),
    }
    row_split = t.cat(
        [
            t.zeros(len(row_blocks["explain_ids"]), dtype=t.int64),
            t.ones(len(row_blocks["audit_ids"]), dtype=t.int64),
        ]
    )

    primary_variant = {
        "source_ids": all_row_data["ids"],
        "targets": all_row_data["y"],
        "row_split": row_split,
        "background_seed": prepared["backgrounds"]["primary64"]["seed"],
        "background_indices": prepared["backgrounds"]["primary64"]["indices"],
        "background_source_ids": prepared["backgrounds"]["primary64"]["source_ids"],
        "background_x_raw": primary_background,
        **all_rows["clean_0"],
    }
    alternatives = {
        name: value
        for name, value in prepared["backgrounds"].items()
        if name != "primary64"
    }
    background_variants = {"primary64": primary_variant}
    background_variants.update(
        _background_comparison(
            fitted, all_row_data, alternatives, methods, deep_methods
        )
    )
    for value in background_variants.values():
        value.setdefault("row_split", row_split)

    primary_context = {
        **all_rows,
        "_background_x_raw": primary_background,
        "_background_source_ids": prepared["backgrounds"]["primary64"]["source_ids"],
    }
    manifest = load_manifest()
    replicate_ids = t.as_tensor(
        manifest["replicate_diagnostic_candidate"]["explanation_source_ids"]
        + manifest["replicate_diagnostic_candidate"]["audit_source_ids"],
        dtype=t.int64,
    )
    locations = {int(source): index for index, source in enumerate(all_row_data["ids"].tolist())}
    if any(int(source) not in locations for source in replicate_ids):
        raise ValueError("Replicate source IDs are absent from the held-out row order.")
    replicate_positions = t.tensor(
        [locations[int(source)] for source in replicate_ids], dtype=t.int64
    )
    replicate_rows = {
        "x": all_row_data["x"][replicate_positions],
        "y": all_row_data["y"][replicate_positions],
        "ids": replicate_ids,
    }
    replicate_primary = {
        key: {
            "quadrature": {
                name: value[replicate_positions]
                for name, value in all_rows[key]["quadrature"].items()
            }
        }
        for key in REPLICATE_MODELS
    }
    replicate_primary["_background_x_raw"] = primary_background

    return {
        "phase": "evaluated_after_audit",
        "audit_used_for_selection": False,
        "reproduction": prepared["reproduction"],
        "performance": _performance_table(fitted, audit),
        "audit_ids": row_blocks["audit_ids"],
        "audit_targets": row_blocks["audit_y"],
        "audit_primary": audit_primary,
        "all_rows": all_rows,
        "all_row_x_raw": all_row_data["x"],
        "all_row_targets": all_row_data["y"],
        "all_row_split": row_split,
        "background_variants": background_variants,
        "noise_variants": _noise_comparison(
            fitted, all_row_data, primary_context, methods
        ),
        "sampling_replicates": _replicate_comparison(
            fitted, replicate_rows, replicate_primary, methods
        ),
        "distance_diagnostic": _distance_comparison(
            train, all_row_data, primary_context
        ),
        "replacement_curves": _replacement_evaluation(
            prepared, audit_primary, methods
        ),
        "random_orders": prepared["random_orders"],
        "units": dict(UNITS),
        "feature_names": FEATURE_NAMES,
        "model_labels": dict(MODEL_LABELS),
        "query_source_ids": all_row_data["ids"],
    }


def native_play(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    methods: Any,
    deep_methods: Any,
    model_key: str = "clean_0",
    split: str = "audit",
    row_index: int = 0,
    background_key: str = "primary64",
    samples: int = 2048,
    sigma: float = 0.0,
) -> dict[str, Any]:
    """Recompute one editable real recipe without changing aggregate evidence."""
    _require_namespace(methods, METHOD_NAMES, "Student methods")
    _require_namespace(deep_methods, DEEP_METHOD_NAMES, "DeepSHAP control")
    _validate_evaluation_inputs(fitted, prepared, train, explain, audit)
    if evaluated.get("phase") != "evaluated_after_audit":
        raise ValueError("Run the complete frozen evaluation before the real play.")
    if model_key not in MODEL_KEYS:
        raise ValueError("Choose one of the seven retained model keys.")
    if split not in {"explain", "audit"}:
        raise ValueError("split must be 'explain' or 'audit'.")
    selected_rows = explain if split == "explain" else audit
    x_all, y_all, ids_all = _rows(selected_rows, split)
    if not 0 <= row_index < len(x_all):
        raise ValueError("row_index must select an existing held-out row.")
    if background_key not in prepared["backgrounds"]:
        raise ValueError("Choose one of the five frozen background keys.")
    if type(samples) is not int or samples < 2 or samples > max(BUDGETS):
        raise ValueError("samples must be an integer from 2 through 2048.")
    if sigma not in NOISE_SIGMAS:
        raise ValueError("sigma must be one of the three predeclared scales.")
    background_record = prepared["backgrounds"][background_key]
    background = background_record["x_raw"]
    x = x_all[row_index : row_index + 1]
    selected = {"x": x, "y": y_all[row_index : row_index + 1], "ids": ids_all[row_index : row_index + 1]}
    indices, alpha = methods.sample_path_schedule(
        selected["ids"], len(background), samples, PRIMARY_SCHEDULE_SEED
    )
    model = _model_from_state(fitted["models"][model_key])
    if sigma == 0:
        mc_samples = methods.expected_gradient_samples(
            model, x, background, indices, alpha, FORWARD_BATCH
        )
    else:
        standard_normal = _standard_normal_schedule(selected["ids"], samples)
        _, x_std, _, _ = _normalization_tensors(fitted["normalization"])
        noise = standard_normal * (sigma * x_std)[None, None, :]
        mc_samples = methods.noisy_expected_gradient_samples(
            model, x, background, indices, alpha, noise, FORWARD_BATCH
        )
    phi, se = methods.summarize_expected_gradients(mc_samples, (samples,))
    quadrature = _quadrature_block(model, selected, background, methods)
    baselines = _comparison_baselines(model, selected, background, deep_methods)
    local_values = baselines["coalition_values"]
    local_phi = {
        "eg": phi[0],
        "quadrature_512": quadrature[f"mean_{QUADRATURE_NODES[1]}"],
        "exact": baselines["phi_exact"],
        "deep": baselines["phi_deep"],
        "gradient": baselines["phi_gradient"],
    }
    curves = {
        name: methods.ranked_coalition_curves(local_values, value)
        for name, value in local_phi.items()
    }
    with t.no_grad():
        prediction = model(x).squeeze(-1)[0]
        base = model(background).squeeze(-1).mean()
    return {
        "selection": {
            "model_key": model_key,
            "model_label": MODEL_LABELS[model_key],
            "split": split,
            "row_index": row_index,
            "source_id": int(ids_all[row_index]),
            "query": x[0],
            "target": y_all[row_index],
            "background_key": background_key,
            "background_source_ids": background_record["source_ids"],
            "background_x_raw": background,
            "samples": samples,
            "sigma_standardized": sigma,
            "audit_reuse": "exploratory" if split == "audit" else "not_audit",
        },
        "prediction": prediction,
        "base": base,
        "mc_samples": mc_samples[0],
        "phi_eg": phi[0, 0],
        "se_eg": se[0, 0],
        "phi_q512": quadrature[f"mean_{QUADRATURE_NODES[1]}"][0],
        "phi_exact": baselines["phi_exact"][0],
        "phi_deep": baselines["phi_deep"][0],
        "phi_gradient": baselines["phi_gradient"][0],
        "coalition_values": local_values[0],
        "local_replacement_curves": {
            name: {field: item[0] for field, item in curve.items()}
            for name, curve in curves.items()
        },
        "units": dict(UNITS),
        "feature_names": FEATURE_NAMES,
    }
