"""Complete held-out SII, Shapley-Taylor, and Faith-Shap study.

Scientific operations are kept as short public functions so the learner
notebook can show and modify them. Data loading, serialization, and independent
reference adapters remain separate plumbing. No function runs at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any

import numpy as np
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_data import (
    background_order,
    load_manifest,
    tensor_mapping_sha256,
)


FEATURES = 8
HIDDEN = 32
EMPIRICAL_ORDER = 2
TRAINING_STEPS = 800
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
RIDGE_PENALTY = 0.01
FORWARD_BATCH = 32_768
DISTANCE_BATCH = 2_048
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
METHOD_KEYS = ("sii", "stii", "fsii")
METHOD_NAMES = (
    "mobius_transform",
    "shapley_taylor_interactions",
    "faith_design_matrix",
    "faith_kernel_weights",
    "faith_shap_interactions",
    "faith_shap_closed_form",
    "reconstruct_game",
    "faithfulness_error",
    "masked_model_game",
    "contextual_pair_synergy",
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
FEATURE_UNITS = ("kg/m3",) * 7 + ("days",)
TARGET_NAME = "Compressive strength"
TARGET_UNIT = "MPa"
RNG_NAMESPACE = "arena16.16/shapley-interactions/v1"
RANDOM_EXACT_SEED = 161610
WITHIN_SIZE_SEED = 161611
RANDOM_PAIR_SEED = 161612
WITHIN_SIZE_REPLICATES = 128
RANDOM_PAIR_ORDERS = 256
SUPPORT_DONORS = 8
DEFAULT_PLAY = {
    "model_key": "clean_0",
    "source_id": 0,
    "donor_source_id": 173,
    "background_key": "primary64",
    "pair": (0, 1),
    "context_mask": 252,
}


def _require_methods(methods: Any) -> None:
    missing = [name for name in METHOD_NAMES if not callable(getattr(methods, name, None))]
    if missing:
        raise TypeError(f"Student method namespace is missing {missing}.")


def _float_tensor(value: Any, name: str, dimensions: int | None = None) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        try:
            value = t.as_tensor(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a tensor or numeric array.") from error
    if value.device.type != "cpu" or value.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions.")
    if not bool(t.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values.")
    return value


def _int_tensor(value: Any, name: str, dimensions: int | None = None) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        try:
            value = t.as_tensor(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a tensor or integer array.") from error
    if value.device.type != "cpu" or value.dtype != t.int64:
        raise TypeError(f"{name} must be a CPU int64 tensor.")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions.")
    return value


def _rows(
    rows: Mapping[str, Any], name: str
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    if not isinstance(rows, Mapping) or not {"x", "y", "ids"} <= set(rows):
        raise TypeError(f"{name} must contain x, y, and ids.")
    x = _float_tensor(rows["x"], f"{name}.x", 2)
    y = _float_tensor(rows["y"], f"{name}.y", 1)
    ids = _int_tensor(rows["ids"], f"{name}.ids", 1)
    if x.shape != (len(y), FEATURES) or ids.shape != (len(y),) or len(y) == 0:
        raise ValueError(f"{name} must contain nonempty eight-feature rows.")
    if len(t.unique(ids)) != len(ids):
        raise ValueError(f"{name}.ids must be unique.")
    groups = rows.get("recipe_group_ids")
    if groups is None:
        groups = t.arange(len(ids), dtype=t.int64)
    groups = _int_tensor(groups, f"{name}.recipe_group_ids", 1)
    if groups.shape != ids.shape:
        raise ValueError(f"{name}.recipe_group_ids must have one value per row.")
    return x, y, ids, groups


def _require_disjoint_ids(**splits: t.Tensor) -> None:
    names = tuple(splits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            if bool(t.isin(splits[left_name], splits[right_name]).any()):
                raise ValueError(
                    f"{left_name} and {right_name} source IDs must be disjoint."
                )


def coalition_masks(n_players: int = FEATURES) -> t.Tensor:
    """Return ascending integer-mask rows with bit i representing player i."""

    if not isinstance(n_players, int) or not 1 <= n_players <= 20:
        raise ValueError("n_players must be an integer from 1 through 20.")
    mask_ids = t.arange(2**n_players, dtype=t.int64)
    bits = t.arange(n_players, dtype=t.int64)
    return ((mask_ids[:, None] >> bits[None, :]) & 1).bool()


def pair_indices(n_players: int = FEATURES) -> t.Tensor:
    """Return lexicographic unordered player pairs."""

    if not isinstance(n_players, int) or n_players < 2:
        raise ValueError("n_players must be at least two.")
    return t.tensor(
        [(left, right) for left in range(n_players) for right in range(left + 1, n_players)],
        dtype=t.int64,
    )


def _clone_state(state: Mapping[str, t.Tensor]) -> dict[str, t.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError("A model state must be a tensor mapping.")
    result: dict[str, t.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str):
            raise TypeError("Model-state keys must be strings.")
        result[name] = _float_tensor(value, f"state[{name}]").detach().clone()
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
    raise ValueError(f"Unsupported model kind {kind!r}.")


def model_from_state(record: Mapping[str, Any]) -> t.nn.Sequential:
    """Reconstruct one raw-input, raw-MPa model without serializing modules."""

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


def _training_normalization(train: Mapping[str, Any]) -> dict[str, t.Tensor]:
    x, y, _, _ = _rows(train, "train")
    result = {
        "x_mean": x.mean(0),
        "x_std": x.std(0, correction=0),
        "y_mean": y.mean(),
        "y_std": y.std(correction=0),
    }
    if bool((result["x_std"] <= 0).any()) or not bool(result["y_std"] > 0):
        raise ValueError("Training population standard deviations must be positive.")
    return result


def _train_mlp(
    model: t.nn.Sequential, x: t.Tensor, y: t.Tensor
) -> tuple[dict[str, t.Tensor], t.Tensor]:
    """Run the visible fixed 800-step full-batch Adam training loop."""

    optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    model.train()
    for _ in range(TRAINING_STEPS):
        optimizer.zero_grad()
        loss = (model(x).squeeze(-1) - y).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().clone())
    return _clone_state(model.state_dict()), t.stack(losses)


def _fold_mlp_state(
    state: Mapping[str, t.Tensor], normalization: Mapping[str, t.Tensor]
) -> dict[str, t.Tensor]:
    folded = _clone_state(state)
    x_mean, x_std = normalization["x_mean"], normalization["x_std"]
    y_mean, y_std = normalization["y_mean"], normalization["y_std"]
    first = folded["0.weight"].clone()
    folded["0.weight"] = first / x_std[None, :]
    folded["0.bias"] = folded["0.bias"] - first @ (x_mean / x_std)
    folded["4.weight"] = y_std * folded["4.weight"]
    folded["4.bias"] = y_std * folded["4.bias"] + y_mean
    return folded


def _fold_ridge_state(
    state: Mapping[str, t.Tensor], normalization: Mapping[str, t.Tensor]
) -> dict[str, t.Tensor]:
    folded = _clone_state(state)
    x_mean, x_std = normalization["x_mean"], normalization["x_std"]
    y_mean, y_std = normalization["y_mean"], normalization["y_std"]
    weight = folded["0.weight"].clone()
    folded["0.weight"] = y_std * weight / x_std[None, :]
    folded["0.bias"] = y_std * (
        folded["0.bias"] - (weight * (x_mean / x_std)[None, :]).sum(1)
    ) + y_mean
    return folded


def _folding_check(
    kind: str,
    standardized_state: Mapping[str, t.Tensor],
    raw_state: Mapping[str, t.Tensor],
    normalization: Mapping[str, t.Tensor],
    x_raw: t.Tensor,
) -> tuple[t.Tensor, float]:
    standard = model_from_state({"kind": kind, "state": standardized_state})
    raw = model_from_state({"kind": kind, "state": raw_state})
    x = (x_raw - normalization["x_mean"]) / normalization["x_std"]
    with t.no_grad():
        expected = standard(x).squeeze(-1) * normalization["y_std"] + normalization["y_mean"]
        observed = raw(x_raw).squeeze(-1)
    return observed, float((observed - expected).abs().max())


def _fit_one_mlp(
    train: Mapping[str, Any],
    normalization: Mapping[str, t.Tensor],
    seed: int,
    shuffle_seed: int | None,
) -> dict[str, Any]:
    x_raw, y_raw, _, _ = _rows(train, "train")
    x = (x_raw - normalization["x_mean"]) / normalization["x_std"]
    y = (y_raw - normalization["y_mean"]) / normalization["y_std"]
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
    prediction, folding_error = _folding_check(
        "mlp", standardized_state, raw_state, normalization, x_raw
    )
    if folding_error > 1e-10:
        raise AssertionError("Raw-unit MLP folding changed its training predictions.")
    return {
        "kind": "mlp",
        "control": "clean" if shuffle_seed is None else "shuffled",
        "seed": seed,
        "shuffle_seed": shuffle_seed,
        "initial_state": initial_state,
        "standardized_state": standardized_state,
        "state": raw_state,
        "loss_trace": loss_trace,
        "target_permutation": permutation,
        "training_targets_raw": y_raw[permutation].clone(),
        "training_prediction": prediction,
        "folding_max_abs_error": folding_error,
    }


def _fit_ridge(
    train: Mapping[str, Any], normalization: Mapping[str, t.Tensor]
) -> dict[str, Any]:
    x_raw, y_raw, _, _ = _rows(train, "train")
    x = (x_raw - normalization["x_mean"]) / normalization["x_std"]
    y = (y_raw - normalization["y_mean"]) / normalization["y_std"]
    design = t.cat([x, t.ones((len(x), 1), dtype=t.float64)], dim=1)
    penalty = t.eye(FEATURES + 1, dtype=t.float64) * RIDGE_PENALTY
    penalty[-1, -1] = 0.0
    coefficient = t.linalg.solve(
        design.T @ design / len(x) + penalty, design.T @ y / len(x)
    )
    standardized_state = {
        "0.weight": coefficient[:FEATURES][None, :].clone(),
        "0.bias": coefficient[FEATURES:].clone(),
    }
    raw_state = _fold_ridge_state(standardized_state, normalization)
    prediction, folding_error = _folding_check(
        "ridge", standardized_state, raw_state, normalization, x_raw
    )
    if folding_error > 1e-10:
        raise AssertionError("Raw-unit ridge folding changed its training predictions.")
    with t.no_grad():
        loss = (model_from_state({"kind": "ridge", "state": standardized_state})(x).squeeze(-1) - y).square().mean()
    return {
        "kind": "ridge",
        "control": "ridge",
        "seed": 0,
        "shuffle_seed": None,
        "initial_state": {},
        "standardized_state": standardized_state,
        "state": raw_state,
        "loss_trace": loss[None],
        "target_permutation": t.arange(len(y), dtype=t.int64),
        "training_targets_raw": y_raw.clone(),
        "training_prediction": prediction,
        "folding_max_abs_error": folding_error,
    }


def fit_known_models(train: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce all seven known predictors; callers enforce the freeze gate."""

    x, y, ids, groups = _rows(train, "train")
    t.set_num_threads(4)
    t.use_deterministic_algorithms(True)
    normalization = _training_normalization(train)
    models: dict[str, Any] = {}
    for index, seed in enumerate(MODEL_SEEDS):
        models[f"clean_{index}"] = _fit_one_mlp(train, normalization, seed, None)
    for index, seed in enumerate(MODEL_SEEDS):
        models[f"shuffled_{index}"] = _fit_one_mlp(
            train, normalization, seed, SHUFFLE_SEEDS[index]
        )
    models["ridge"] = _fit_ridge(train, normalization)
    for record in models.values():
        record["state_hashes"] = {
            "initial_state": tensor_mapping_sha256(record["initial_state"]),
            "standardized_state": tensor_mapping_sha256(record["standardized_state"]),
            "raw_folded_state": tensor_mapping_sha256(record["state"]),
        }
    return {
        "phase": "fitted_before_interactions",
        "models": models,
        "normalization": normalization,
        "train_x_raw": x.clone(),
        "train_y_raw": y.clone(),
        "train_ids": ids.clone(),
        "train_recipe_group_ids": groups.clone(),
        "feature_names": FEATURE_NAMES,
        "feature_units": FEATURE_UNITS,
        "target_name": TARGET_NAME,
        "target_unit": TARGET_UNIT,
    }


def _validate_fitted(fitted: Mapping[str, Any]) -> None:
    if not isinstance(fitted, Mapping) or fitted.get("phase") != "fitted_before_interactions":
        raise ValueError("Use the complete train-only fitted payload.")
    if tuple(fitted.get("models", {})) != MODEL_KEYS:
        raise ValueError("The fitted payload must retain all seven models in order.")
    x = _float_tensor(fitted.get("train_x_raw"), "train_x_raw", 2)
    y = _float_tensor(fitted.get("train_y_raw"), "train_y_raw", 1)
    ids = _int_tensor(fitted.get("train_ids"), "train_ids", 1)
    groups = _int_tensor(fitted.get("train_recipe_group_ids"), "train_recipe_group_ids", 1)
    if x.shape != (len(y), FEATURES) or ids.shape != y.shape or groups.shape != y.shape:
        raise ValueError("Fitted training arrays have incompatible shapes.")
    for key, record in fitted["models"].items():
        if record.get("kind") != ("ridge" if key == "ridge" else "mlp"):
            raise ValueError(f"{key} has the wrong model kind.")
        model_from_state(record)


def verify_known_reproduction(
    fitted: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Hard-fail unless all 21 known state hashes and fit identities agree."""

    _validate_fitted(fitted)
    expected = manifest.get("known_16_14_evidence", {}).get("model_state_hashes", {})
    if tuple(expected) != MODEL_KEYS:
        raise ValueError("Manifest must declare all seven known model hashes.")
    fields = {
        "initial_state": "initial_state",
        "standardized_state": "standardized_state",
        "raw_folded_state": "state",
    }
    observed: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    for key in MODEL_KEYS:
        record = fitted["models"][key]
        observed[key] = {
            label: tensor_mapping_sha256(record[field]) for label, field in fields.items()
        }
        for label in fields:
            if observed[key][label] != expected[key][label]:
                mismatches.append(f"{key}.{label}")
        if record.get("state_hashes") != observed[key]:
            mismatches.append(f"{key}.retained_state_hashes")
    train_ids = t.tensor(manifest.get("splits", {}).get("train", []), dtype=t.int64)
    if not t.equal(train_ids, fitted["train_ids"]):
        mismatches.append("train_ids")
    recipe = manifest.get("model_recipe", {})
    recipe_matches = (
        recipe.get("model_keys") == list(MODEL_KEYS)
        and recipe.get("model_seeds") == list(MODEL_SEEDS)
        and recipe.get("shuffle_seeds") == list(SHUFFLE_SEEDS)
        and recipe.get("training_steps") == TRAINING_STEPS
        and recipe.get("learning_rate") == LEARNING_RATE
        and recipe.get("weight_decay") == WEIGHT_DECAY
        and recipe.get("ridge_penalty") == RIDGE_PENALTY
    )
    if not recipe_matches:
        mismatches.append("training_recipe")
    if mismatches:
        raise AssertionError("Known-model reproduction mismatch: " + ", ".join(mismatches))
    return {
        "status": "PASS",
        "known_section": "16.14/16.15",
        "model_state_hashes": observed,
        "expected_model_state_hashes": expected,
        "mismatches": (),
        "state_comparisons": 21,
        "training_recipe_match": True,
        "train_ids_match": True,
        "non_independent_replication": True,
    }


def _zeta_transform(mobius: t.Tensor, n_players: int) -> t.Tensor:
    """Fixture plumbing: turn declared dividends into a complete game."""

    values = mobius.clone()
    for bit in range(n_players):
        step = 1 << bit
        for mask in range(2**n_players):
            if mask & step:
                values[..., mask] += values[..., mask ^ step]
    return values


def fixed_exact_games() -> dict[str, Any]:
    """Return the eight predeclared four-player games and exact dividend table."""

    keys = (
        "constant",
        "additive",
        "positive_pair",
        "negative_pair",
        "pure_triple",
        "pure_four",
        "mixed_degree2",
        "mixed_high",
    )
    a = t.zeros((len(keys), 16), dtype=t.float64)
    a[0, 0] = 2.0
    a[1, 0] = -1 / 2
    a[1, [1, 2, 4, 8]] = t.tensor(
        [1.0, -2.0, 3 / 2, 1 / 4], dtype=t.float64
    )
    a[2, 0b0011] = 1.0
    a[3, 0b1010] = -3 / 4
    a[4, 0b0111] = 1.0
    a[5, 0b1111] = 1.0
    a[6, 0] = 1 / 3
    a[6, [1, 2, 4, 8]] = t.tensor(
        [1 / 2, -1 / 4, 3 / 4, -1.0], dtype=t.float64
    )
    a[6, [0b0011, 0b1001, 0b0110, 0b1100]] = t.tensor(
        [2 / 3, -1 / 2, 1 / 5, 3 / 8], dtype=t.float64
    )
    a[7, 0] = -1 / 5
    a[7, 0b0001] = 1 / 3
    a[7, 0b0110] = -2 / 5
    a[7, 0b1101] = 3 / 4
    a[7, 0b1111] = -1 / 2
    return {
        "game_keys": keys,
        "n_players": 4,
        "mobius_oracle": a,
        "game_values": _zeta_transform(a, 4),
        "fraction_oracles": {
            "pure_triple": {
                "stii_singleton": (0, 1),
                "stii_contained_pair": (1, 3),
                "fsii_singleton": (-1, 6),
                "fsii_contained_pair": (1, 2),
                "shapley_singleton": (1, 3),
                "sii_contained_pair": (1, 2),
            },
            "pure_four": {
                "stii_singleton": (0, 1),
                "stii_pair": (1, 6),
                "fsii_singleton": (-1, 5),
                "fsii_pair": (3, 10),
                "shapley_singleton": (1, 4),
                "sii_pair": (1, 3),
            },
        },
    }


def _derived_seed(seed: int, *parts: Any) -> int:
    text = "/".join([RNG_NAMESPACE, str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "little")


def random_exact_games(
    *, seed: int = RANDOM_EXACT_SEED, games_per_size: int = 3
) -> dict[int, dict[str, t.Tensor]]:
    """Return all 15 frozen random complete games without touching global RNG."""

    if seed != RANDOM_EXACT_SEED or games_per_size != 3:
        raise ValueError("The native design fixes seed 161610 and three games per size.")
    result: dict[int, dict[str, t.Tensor]] = {}
    for n_players in range(3, 8):
        values = []
        seeds = []
        for replicate in range(games_per_size):
            derived = _derived_seed(seed, n_players, replicate)
            seeds.append(derived)
            values.append(np.random.Generator(np.random.PCG64(derived)).standard_normal(2**n_players))
        result[n_players] = {
            "game_values": t.tensor(np.stack(values), dtype=t.float64),
            "orders": t.arange(1, min(3, n_players) + 1, dtype=t.int64),
            "derived_seeds_hex": tuple(f"{value:032x}" for value in seeds),
        }
    declared = (
        load_manifest().get("rng_contract", {}).get("random_exact", {}).get("games", [])
    )
    flattened = [
        (
            n_players,
            replicate,
            result[n_players]["derived_seeds_hex"][replicate],
            result[n_players]["game_values"][replicate],
        )
        for n_players in range(3, 8)
        for replicate in range(games_per_size)
    ]
    if len(declared) != len(flattened):
        raise ValueError("Manifest must materialize all 15 random exact games.")
    for item, (n_players, replicate, seed_hex, values) in zip(
        declared, flattened, strict=True
    ):
        expected_values = t.tensor(item.get("values", []), dtype=t.float64)
        if (
            item.get("n_players") != n_players
            or item.get("replicate") != replicate
            or item.get("derived_seed_hex") != seed_hex
            or not t.equal(values, expected_values)
        ):
            raise ValueError("Random exact game differs from the manifest.")
    expected_hashes = (
        load_manifest()
        .get("rng_contract", {})
        .get("random_exact", {})
        .get("game_value_hashes", [])
    )
    observed_hashes = [tensor_mapping_sha256({"values": item[3]}) for item in flattened]
    if expected_hashes != observed_hashes:
        raise ValueError("Random exact game hashes differ from the manifest.")
    return result


def pairwise_sii(game_values: t.Tensor, *, n_players: int) -> t.Tensor:
    """Compute the exact pair SII comparator in lexicographic pair order."""

    values = _float_tensor(game_values, "game_values")
    if values.ndim < 1 or values.shape[-1] != 2**n_players or n_players < 2:
        raise ValueError("game_values must end in one complete game axis.")
    output = []
    for left, right in pair_indices(n_players).tolist():
        pair_mask = (1 << left) | (1 << right)
        contexts = [
            mask for mask in range(2**n_players) if mask & pair_mask == 0
        ]
        score = t.zeros(values.shape[:-1], dtype=t.float64)
        denominator = math.factorial(n_players - 1)
        for context in contexts:
            size = int(context).bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(n_players - size - 2)
                / denominator
            )
            delta = (
                values[..., context | pair_mask]
                - values[..., context | (1 << left)]
                - values[..., context | (1 << right)]
                + values[..., context]
            )
            score = score + weight * delta
        output.append(score)
    return t.stack(output, dim=-1)


def donor_replacement_games(
    model: t.nn.Module,
    queries: t.Tensor,
    donors: t.Tensor,
    *,
    methods: Any,
    donor_source_ids: t.Tensor | None = None,
    coalition_batch_size: int = 4096,
) -> dict[str, t.Tensor]:
    """Retain every single-donor game, then average the declared donor bank."""

    _require_methods(methods)
    x = _float_tensor(queries, "queries", 2)
    background = _float_tensor(donors, "donors", 2)
    if x.shape[1] != FEATURES or background.shape[1] != FEATURES or len(background) == 0:
        raise ValueError("queries and donors must contain eight-feature rows.")
    if not isinstance(coalition_batch_size, int) or coalition_batch_size <= 0:
        raise ValueError("coalition_batch_size must be positive.")
    if donor_source_ids is None:
        donor_source_ids = t.arange(len(background), dtype=t.int64)
    donor_source_ids = _int_tensor(donor_source_ids, "donor_source_ids", 1)
    if donor_source_ids.shape != (len(background),) or len(t.unique(donor_source_ids)) != len(background):
        raise ValueError("donor_source_ids must uniquely identify every donor.")
    hybrid = t.empty((len(x), 2**FEATURES, len(background)), dtype=t.float64)
    for donor_index, donor in enumerate(background):
        game = methods.masked_model_game(
            model, x, donor, coalition_batch_size=coalition_batch_size
        )
        game = _float_tensor(game, "single_donor_game", 2)
        if game.shape != (len(x), 2**FEATURES):
            raise ValueError("masked_model_game returned an unexpected shape.")
        hybrid[..., donor_index] = game
    return {
        "hybrid_prediction": hybrid,
        "coalition_values": hybrid.mean(-1),
        "donor_source_ids": donor_source_ids.clone(),
    }


def all_pair_context_synergies(
    game_values: t.Tensor, *, n_players: int, methods: Any
) -> dict[str, t.Tensor]:
    """Apply the learner's local derivative to all pairs and valid contexts."""

    _require_methods(methods)
    values = _float_tensor(game_values, "game_values")
    pairs = pair_indices(n_players)
    context_rows = []
    synergy_rows = []
    full_rows = []
    for left, right in pairs.tolist():
        contexts, synergy = methods.contextual_pair_synergy(
            values, (left, right), n_players=n_players
        )
        contexts = _int_tensor(contexts, "context_masks", 1)
        synergy = _float_tensor(synergy, "context_synergy")
        if synergy.shape != values.shape[:-1] + (len(contexts),):
            raise ValueError("contextual_pair_synergy returned an unexpected shape.")
        expected_full = (2**n_players - 1) ^ (1 << left) ^ (1 << right)
        positions = t.nonzero(contexts == expected_full).flatten()
        if len(positions) != 1:
            raise ValueError("The full pair context is absent or repeated.")
        context_rows.append(contexts)
        synergy_rows.append(synergy)
        full_rows.append(synergy[..., int(positions[0])])
    context_masks = t.stack(context_rows)
    context_synergy = t.stack(synergy_rows, dim=-2)
    return {
        "pair_indices": pairs,
        "context_masks": context_masks,
        "context_synergy": context_synergy,
        "full_context_synergy": t.stack(full_rows, dim=-1),
        "mean_abs_context_synergy": context_synergy.abs().mean(-1),
    }


def explain_complete_games(
    game_values: t.Tensor,
    *,
    methods: Any,
    n_players: int = FEATURES,
    order: int = EMPIRICAL_ORDER,
    include_contexts: bool = True,
) -> dict[str, t.Tensor]:
    """Apply every visible interaction method to a batch of complete games."""

    _require_methods(methods)
    values = _float_tensor(game_values, "game_values")
    if values.ndim < 1 or values.shape[-1] != 2**n_players:
        raise ValueError("game_values must end in one complete bitmask axis.")
    mobius = methods.mobius_transform(values, n_players=n_players)
    mobius = _float_tensor(mobius, "mobius")
    stii = methods.shapley_taylor_interactions(mobius, n_players=n_players, order=order)
    fsii = methods.faith_shap_interactions(values, n_players=n_players, max_order=order)
    closed = methods.faith_shap_closed_form(mobius, n_players=n_players, max_order=order)
    stii = _float_tensor(stii, "stii_coefficients")
    fsii = _float_tensor(fsii, "fsii_coefficients")
    closed = _float_tensor(closed, "fsii_closed_form")
    expected_shape = values.shape
    if stii.shape != expected_shape or fsii.shape != expected_shape or closed.shape != expected_shape:
        raise ValueError("Interaction coefficient tensors must preserve the game shape.")
    ordinary_full = methods.shapley_taylor_interactions(
        mobius, n_players=n_players, order=1
    )
    singleton_masks = 1 << t.arange(n_players, dtype=t.int64)
    pairs = pair_indices(n_players)
    pair_masks = (1 << pairs[:, 0]) | (1 << pairs[:, 1])
    sii = pairwise_sii(values, n_players=n_players)
    recon = t.stack(
        [
            methods.reconstruct_game(stii, n_players=n_players),
            methods.reconstruct_game(fsii, n_players=n_players),
        ]
    )
    recon = _float_tensor(recon, "reconstruction")
    residual = values.unsqueeze(0) - recon
    sizes = coalition_masks(n_players).sum(-1)
    rmse = t.stack(
        [
            t.sqrt(residual[..., sizes == size].square().mean(-1))
            for size in range(n_players + 1)
        ],
        dim=-1,
    )
    kernel_mse = t.stack(
        [
            methods.faithfulness_error(values, stii, n_players=n_players),
            methods.faithfulness_error(values, fsii, n_players=n_players),
        ]
    )
    result = {
        "coalition_values": values,
        "mobius": mobius,
        "ordinary_shapley": ordinary_full[..., singleton_masks],
        "sii_pairs": sii,
        "stii_coefficients": stii,
        "fsii_coefficients": fsii,
        "fsii_closed_form": closed,
        "fsii_closed_form_error": (fsii - closed).abs().amax(-1),
        "pair_values": t.stack([sii, stii[..., pair_masks], fsii[..., pair_masks]], dim=-2),
        "reconstruction": recon,
        "residual": residual,
        "kernel_mse": kernel_mse,
        "rmse_by_size": rmse,
        "efficiency_residual": t.stack(
            [stii.sum(-1) - values[..., -1], fsii.sum(-1) - values[..., -1]]
        ),
        "pair_indices": pairs,
        "pair_masks": pair_masks,
    }
    if include_contexts:
        result.update(all_pair_context_synergies(values, n_players=n_players, methods=methods))
    return result


def evaluate_model_games(
    model: t.nn.Module,
    queries: t.Tensor,
    donors: t.Tensor,
    *,
    methods: Any,
    donor_source_ids: t.Tensor,
    coalition_batch_size: int = 4096,
) -> dict[str, t.Tensor]:
    """Construct one retained donor game and compute every exact order-two output."""

    donor = donor_replacement_games(
        model,
        queries,
        donors,
        methods=methods,
        donor_source_ids=donor_source_ids,
        coalition_batch_size=coalition_batch_size,
    )
    explained = explain_complete_games(donor["coalition_values"], methods=methods)
    return {**donor, **explained}


def _background_records(
    train_x: t.Tensor, train_ids: t.Tensor, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    declared = manifest.get("backgrounds", {})
    if tuple(declared) != tuple(BACKGROUND_SPECS):
        raise ValueError("Manifest background names or order changed.")
    result: dict[str, dict[str, Any]] = {}
    for name, (seed, rows) in BACKGROUND_SPECS.items():
        order = t.from_numpy(background_order(train_ids.numpy(), seed))
        indices = order[:rows]
        item = declared[name]
        expected_indices = t.tensor(item.get("train_indices", []), dtype=t.int64)
        expected_ids = t.tensor(item.get("source_ids", []), dtype=t.int64)
        if item.get("seed") != seed or item.get("rows") != rows:
            raise ValueError(f"{name} changed its seed or donor count.")
        if not t.equal(indices, expected_indices) or not t.equal(train_ids[indices], expected_ids):
            raise ValueError(f"{name} donor indices or source IDs changed.")
        result[name] = {
            "seed": seed,
            "indices": indices,
            "source_ids": train_ids[indices],
            "x_raw": train_x[indices],
        }
    return result


def _storage_plan(manifest: Mapping[str, Any], stage: str) -> dict[str, Any]:
    contract = manifest.get("physical_hybrid_shards", {})
    shards = [item for item in contract.get("shards", []) if item.get("stage") == stage]
    aliases = [item for item in contract.get("aliases", []) if item.get("stage") == stage]
    if contract.get("count") != 10 or len(shards) != 5:
        raise ValueError("Manifest must declare five physical hybrid shards per stage.")
    if len(aliases) != 1:
        raise ValueError("Manifest must declare one primary clean_0 view per stage.")
    return {
        "stage": stage,
        "physical_hybrid_shards": shards,
        "descriptor_schema": tuple(contract.get("descriptor_schema", [])),
        "view_descriptor_schema": tuple(contract.get("view_descriptor_schema", [])),
        "descriptor_kind": contract.get("descriptor_kind"),
        "view_descriptor_kind": contract.get("view_descriptor_kind"),
        "shallow_replacement_only": contract.get("shallow_replacement_only"),
        "aliases": tuple(aliases),
        "logical_split_mappings": (
            tuple(contract.get("logical_split_mappings", []))
            if stage == "evaluated_after_audit"
            else ()
        ),
        "rehash_policy": contract.get("rehash_policy"),
    }


_SHARED_GAME_LEAVES = {"donor_source_ids", "pair_indices", "pair_masks", "context_masks"}
_METHOD_FIRST_LEAVES = {
    "reconstruction",
    "residual",
    "kernel_mse",
    "rmse_by_size",
    "efficiency_residual",
}


def _stack_model_blocks(blocks: Mapping[str, Mapping[str, t.Tensor]]) -> dict[str, Any]:
    if tuple(blocks) != MODEL_KEYS:
        raise ValueError("Model blocks must retain the seven-model order.")
    first = blocks[MODEL_KEYS[0]]
    result: dict[str, Any] = {"model_keys": MODEL_KEYS}
    for leaf in first:
        values = [blocks[key][leaf] for key in MODEL_KEYS]
        if leaf in _SHARED_GAME_LEAVES:
            if any(not t.equal(values[0], value) for value in values[1:]):
                raise ValueError(f"Shared game leaf {leaf} changed between models.")
            result[leaf] = values[0]
        elif leaf in _METHOD_FIRST_LEAVES:
            result[leaf] = t.stack(values, dim=1)
        else:
            result[leaf] = t.stack(values, dim=0)
    return result


def _select_model_block(primary: Mapping[str, Any], model_index: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for leaf, value in primary.items():
        if leaf == "model_keys":
            result["model_key"] = value[model_index]
        elif leaf in _SHARED_GAME_LEAVES:
            result[leaf] = value
        elif leaf in _METHOD_FIRST_LEAVES:
            result[leaf] = value[:, model_index]
        elif isinstance(value, t.Tensor):
            result[leaf] = value[model_index]
    return result


def _group_weighted_mean_abs(values: t.Tensor, group_ids: t.Tensor) -> t.Tensor:
    values = _float_tensor(values, "values")
    groups = _int_tensor(group_ids, "group_ids", 1)
    if values.shape[0] != len(groups):
        raise ValueError("The first values axis must match group_ids.")
    unique = t.unique(groups, sorted=True)
    return t.stack([values[groups == group].abs().mean(0) for group in unique]).mean(0)


def explanation_pair_rankings(
    pair_values: t.Tensor, recipe_group_ids: t.Tensor
) -> dict[str, t.Tensor]:
    """Rank all 28 pairs from explanation groups without dropping any pair."""

    values = _float_tensor(pair_values, "pair_values", 4)
    if values.shape[2:] != (len(METHOD_KEYS), math.comb(FEATURES, 2)):
        raise ValueError("pair_values must have shape [models, rows, 3, 28].")
    strengths = t.stack(
        [_group_weighted_mean_abs(model_values, recipe_group_ids) for model_values in values]
    )
    orders = t.argsort(strengths, dim=-1, descending=True, stable=True)
    return {"method_keys": METHOD_KEYS, "pair_strength": strengths, "pair_order": orders}


def fixed_random_pair_orders() -> dict[str, Any]:
    """Return all 256 content-independent random pair permutations."""

    orders = []
    seeds = []
    for replicate in range(RANDOM_PAIR_ORDERS):
        derived = _derived_seed(RANDOM_PAIR_SEED, replicate)
        seeds.append(derived)
        orders.append(np.random.Generator(np.random.PCG64(derived)).permutation(28))
    result = {
        "seed": RANDOM_PAIR_SEED,
        "derived_seeds_hex": tuple(f"{value:032x}" for value in seeds),
        "orders": t.tensor(np.stack(orders), dtype=t.int64),
    }
    declared = load_manifest().get("rng_contract", {}).get("random_pair_order", {})
    if declared.get("derived_seeds_hex") != list(result["derived_seeds_hex"]):
        raise ValueError("Random pair-order seeds differ from the manifest.")
    if declared.get("orders_value_sha256") != tensor_mapping_sha256(
        {"orders": result["orders"]}
    ):
        raise ValueError("Random pair orders differ from the manifest.")
    return result


def _reference_result(
    reference: Any | None,
    explained: Mapping[str, t.Tensor],
    *,
    n_players: int,
    order: int,
) -> dict[str, Any]:
    if reference is None:
        return {"status": "NOT_RUN", "reason": "reference adapter not supplied"}
    validate = getattr(reference, "reference_check", None)
    if not callable(validate):
        raise TypeError("Reference namespace must expose reference_check.")
    expected_game_count = int(explained["coalition_values"].numel() // (2**n_players))
    values = explained["coalition_values"].reshape(expected_game_count, 2**n_players)
    stii = explained["stii_coefficients"].reshape(expected_game_count, 2**n_players)
    fsii = explained["fsii_coefficients"].reshape(expected_game_count, 2**n_players)
    sii = explained["sii_pairs"].reshape(expected_game_count, math.comb(n_players, 2))
    return validate(
        values,
        stii,
        fsii,
        sii,
        n_players=n_players,
        order=order,
        expected_game_count=expected_game_count,
    )


def _exact_suite(methods: Any, reference: Any | None) -> dict[str, Any]:
    fixed = fixed_exact_games()
    fixed_explained = explain_complete_games(
        fixed["game_values"], methods=methods, n_players=4, order=2
    )
    fixed["explained"] = fixed_explained
    fixed["reference"] = _reference_result(
        reference, fixed_explained, n_players=4, order=2
    )
    random: dict[int, Any] = {}
    for n_players, item in random_exact_games().items():
        order_blocks = {}
        for order_tensor in item["orders"]:
            order = int(order_tensor)
            explained = explain_complete_games(
                item["game_values"],
                methods=methods,
                n_players=n_players,
                order=order,
                include_contexts=order == 2,
            )
            order_blocks[order] = {
                "explained": explained,
                "reference": _reference_result(
                    reference, explained, n_players=n_players, order=order
                ),
            }
        boundary_explained = explain_complete_games(
            item["game_values"][0:1],
            methods=methods,
            n_players=n_players,
            order=n_players,
            include_contexts=False,
        )
        random[n_players] = {
            **item,
            "order_blocks": order_blocks,
            "boundary_order_n": {
                "replicate": 0,
                "order": n_players,
                "reuses_grid_case": n_players == 3,
                "explained": boundary_explained,
                "reference": _reference_result(
                    reference,
                    boundary_explained,
                    n_players=n_players,
                    order=n_players,
                ),
            },
        }
    return {
        "fixed": fixed,
        "random": random,
        "counts": {
            "fixed_unique_games": 8,
            "random_unique_games": 15,
            "random_game_order_cases": 45,
            "random_stii_plus_fsii_cases": 90,
            "full_order_boundary_checks": 5,
            "boundary_checks_distinct_from_grid": 4,
        },
    }


def _empirical_reference_plan(stage: str) -> dict[str, Any]:
    """Declare the one post-audit empirical reference run without executing it."""

    if stage not in {"prepared_before_audit", "evaluated_after_audit"}:
        raise ValueError("Unknown empirical reference stage.")
    ready = stage == "evaluated_after_audit"
    return {
        "status": (
            "READY_FOR_SINGLE_EXHAUSTIVE_CHECK"
            if ready
            else "DEFERRED_UNTIL_SEALED_ALL_ROWS"
        ),
        "adapter": "interaction_reference.validate_study_reference",
        "adapter_calls": 1 if ready else 0,
        "unique_games": 2_992,
        "direct_shapiq_exact_computers": 2_992,
        "direct_shapiq_index_calls": 8_976,
        "requires_prepared_and_evaluated": True,
        "required_after_audit": True,
        "all_outcomes_retained": True,
    }


def _evaluate_primary(
    fitted: Mapping[str, Any],
    queries: t.Tensor,
    background: Mapping[str, Any],
    *,
    methods: Any,
) -> dict[str, Any]:
    blocks = {}
    for key in MODEL_KEYS:
        blocks[key] = evaluate_model_games(
            model_from_state(fitted["models"][key]),
            queries,
            background["x_raw"],
            methods=methods,
            donor_source_ids=background["source_ids"],
        )
    return _stack_model_blocks(blocks)


def _evaluate_backgrounds(
    fitted: Mapping[str, Any],
    queries: t.Tensor,
    backgrounds: Mapping[str, Mapping[str, Any]],
    primary: Mapping[str, Any],
    *,
    methods: Any,
) -> dict[str, Any]:
    result = {"primary64": _select_model_block(primary, 0)}
    model = model_from_state(fitted["models"]["clean_0"])
    for name in tuple(BACKGROUND_SPECS)[1:]:
        background = backgrounds[name]
        result[name] = evaluate_model_games(
            model,
            queries,
            background["x_raw"],
            methods=methods,
            donor_source_ids=background["source_ids"],
        )
    return result


def _direct_full_context_four(
    fitted: Mapping[str, Any], queries: t.Tensor, donors: t.Tensor
) -> dict[str, t.Tensor]:
    """Re-query four full-context inputs independently of coalition indexing."""

    x = _float_tensor(queries, "queries", 2)
    background = _float_tensor(donors, "donors", 2)
    pairs = pair_indices(FEATURES)
    outputs = t.empty((len(MODEL_KEYS), len(x), len(pairs), 4), dtype=t.float64)
    full = 2**FEATURES - 1
    for model_index, key in enumerate(MODEL_KEYS):
        model = model_from_state(fitted["models"][key])
        with t.no_grad():
            for pair_index, (left, right) in enumerate(pairs.tolist()):
                context = full ^ (1 << left) ^ (1 << right)
                mask_ids = t.tensor(
                    [context, context | (1 << left), context | (1 << right), full],
                    dtype=t.int64,
                )
                masks = coalition_masks(FEATURES)[mask_ids]
                for start in range(0, len(x), 4):
                    query = x[start : start + 4]
                    hybrid = t.where(
                        masks[None, :, None, :],
                        query[:, None, None, :],
                        background[None, None, :, :],
                    )
                    flat = hybrid.reshape(-1, FEATURES)
                    predicted = t.cat(
                        [
                            model(flat[index : index + FORWARD_BATCH]).squeeze(-1)
                            for index in range(0, len(flat), FORWARD_BATCH)
                        ]
                    ).reshape(len(query), 4, len(background))
                    outputs[model_index, start : start + len(query), pair_index] = predicted.mean(-1)
    synergy = outputs[..., 3] - outputs[..., 1] - outputs[..., 2] + outputs[..., 0]
    return {"four_values": outputs, "synergy": synergy}


def prepare_study(
    fitted: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    *,
    methods: Any,
    reference: Any | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute and seal every explanation-side decision before audit access."""

    _require_methods(methods)
    manifest = load_manifest() if manifest is None else manifest
    reproduction = verify_known_reproduction(fitted, manifest)
    train_x, train_y, train_ids, train_groups = _rows(train, "train")
    explain_x, explain_y, explain_ids, explain_groups = _rows(explain, "explain")
    _require_disjoint_ids(train=train_ids, explain=explain_ids)
    if not t.equal(train_x, fitted["train_x_raw"]) or not t.equal(train_y, fitted["train_y_raw"]):
        raise ValueError("Training values changed after fitting.")
    if not t.equal(train_ids, fitted["train_ids"]) or not t.equal(train_groups, fitted["train_recipe_group_ids"]):
        raise ValueError("Training identities changed after fitting.")
    expected_explain = t.tensor(manifest.get("splits", {}).get("explain", []), dtype=t.int64)
    if not t.equal(explain_ids, expected_explain):
        raise ValueError("Explanation source IDs differ from the pinned split.")
    backgrounds = _background_records(train_x, train_ids, manifest)
    primary = _evaluate_primary(fitted, explain_x, backgrounds["primary64"], methods=methods)
    variants = _evaluate_backgrounds(
        fitted, explain_x, backgrounds, primary, methods=methods
    )
    rankings = explanation_pair_rankings(primary["pair_values"], explain_groups)
    direct = _direct_full_context_four(fitted, explain_x, backgrounds["primary64"]["x_raw"])
    direct["table_synergy"] = primary["full_context_synergy"]
    direct["lookup_difference"] = direct["synergy"] - direct["table_synergy"]
    design_contract = {}
    for order in (1, 2, 3):
        design, term_masks = methods.faith_design_matrix(
            n_players=FEATURES, max_order=order
        )
        design_contract[order] = {
            "design": _float_tensor(design, f"faith_design_{order}", 2),
            "term_masks": _int_tensor(term_masks, f"term_masks_{order}", 1),
        }
    design_contract["kernel_weights"] = _float_tensor(
        methods.faith_kernel_weights(n_players=FEATURES), "kernel_weights", 1
    )
    return {
        "phase": "prepared_before_audit",
        "audit_used_for_selection": False,
        "reproduction": reproduction,
        "feature_names": FEATURE_NAMES,
        "feature_units": FEATURE_UNITS,
        "target_name": TARGET_NAME,
        "target_unit": TARGET_UNIT,
        "model_keys": MODEL_KEYS,
        "method_keys": METHOD_KEYS,
        "mask_ids": t.arange(2**FEATURES, dtype=t.int64),
        "coalition_masks": coalition_masks(FEATURES),
        "coalition_sizes": coalition_masks(FEATURES).sum(-1).to(t.int64),
        "pair_indices": pair_indices(FEATURES),
        "context_masks": primary["context_masks"],
        "design_contract": design_contract,
        "train_ids": train_ids,
        "train_recipe_group_ids": train_groups,
        "explain_source_ids": explain_ids,
        "explain_recipe_group_ids": explain_groups,
        "explain_x_raw": explain_x,
        "explain_targets": explain_y,
        "backgrounds": backgrounds,
        "exact_games": _exact_suite(methods, reference),
        "primary64": primary,
        "background_variants": variants,
        "direct_full_context": direct,
        "references": {
            "exact_suite_checked_in_preparation": reference is not None,
            "empirical": _empirical_reference_plan("prepared_before_audit"),
        },
        "rankings": rankings,
        "random_pair_orders": fixed_random_pair_orders(),
        "fixed_play_defaults": dict(DEFAULT_PLAY),
        "source_provenance": dict(manifest.get("source_provenance", {})),
        "storage_plan": _storage_plan(manifest, "prepared_before_audit"),
    }


def _combine_game_blocks(
    explanation: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    model_axis: bool,
) -> dict[str, Any]:
    """Join small result arrays while leaving donor tensors as physical shards."""

    result: dict[str, Any] = {}
    for leaf, explain_value in explanation.items():
        audit_value = audit.get(leaf)
        if leaf == "hybrid_prediction":
            logical_shape = list(explain_value.shape)
            logical_shape[1 if model_axis else 0] += audit_value.shape[1 if model_axis else 0]
            result[leaf] = {
                "storage": "split_external_shards",
                "logical_shape": tuple(logical_shape),
                "dtype": "float64",
                "explain": explain_value,
                "audit": audit_value,
            }
        elif leaf in _SHARED_GAME_LEAVES or leaf in {"model_keys", "model_key"}:
            if isinstance(explain_value, t.Tensor) and not t.equal(explain_value, audit_value):
                raise ValueError(f"Shared leaf {leaf} changed at the audit boundary.")
            elif not isinstance(explain_value, t.Tensor) and explain_value != audit_value:
                raise ValueError(f"Shared leaf {leaf} changed at the audit boundary.")
            result[leaf] = explain_value
        elif isinstance(explain_value, t.Tensor):
            if leaf in _METHOD_FIRST_LEAVES:
                row_axis = 2 if model_axis else 1
            else:
                row_axis = 1 if model_axis else 0
            result[leaf] = t.cat([explain_value, audit_value], dim=row_axis)
    return result


def _performance(
    fitted: Mapping[str, Any], x: t.Tensor, y: t.Tensor
) -> dict[str, t.Tensor]:
    prediction = []
    with t.no_grad():
        for key in MODEL_KEYS:
            prediction.append(model_from_state(fitted["models"][key])(x).squeeze(-1))
    prediction_tensor = t.stack(prediction)
    return {
        "prediction": prediction_tensor,
        "squared_error": (prediction_tensor - y[None, :]).square(),
        "mse": (prediction_tensor - y[None, :]).square().mean(-1),
        "training_mean_prediction": fitted["train_y_raw"].mean(),
        "training_mean_mse": (y - fitted["train_y_raw"].mean()).square().mean(),
    }


def pair_capture_curves(
    pair_orders: t.Tensor,
    audit_full_context_synergy: t.Tensor,
    audit_recipe_group_ids: t.Tensor,
    random_orders: t.Tensor,
) -> dict[str, t.Tensor]:
    """Evaluate explanation-derived and random pair orders on every audit group."""

    orders = _int_tensor(pair_orders, "pair_orders", 3)
    synergy = _float_tensor(audit_full_context_synergy, "audit_full_context_synergy", 3)
    random = _int_tensor(random_orders, "random_orders", 2)
    if orders.shape != (len(MODEL_KEYS), len(METHOD_KEYS), 28):
        raise ValueError("pair_orders must have shape [7,3,28].")
    if synergy.shape[0] != len(MODEL_KEYS) or synergy.shape[2] != 28:
        raise ValueError("audit_full_context_synergy must have shape [7,rows,28].")
    if random.shape != (RANDOM_PAIR_ORDERS, 28):
        raise ValueError("random_orders must have shape [256,28].")
    strength = t.stack(
        [_group_weighted_mean_abs(model_values, audit_recipe_group_ids) for model_values in synergy]
    )
    denominator = strength.sum(-1)
    zero = denominator == 0
    safe = t.where(zero, t.ones_like(denominator), denominator)
    observed = t.zeros((len(MODEL_KEYS), len(METHOD_KEYS), 29), dtype=t.float64)
    random_curves = t.zeros((len(MODEL_KEYS), RANDOM_PAIR_ORDERS, 29), dtype=t.float64)
    for model_index in range(len(MODEL_KEYS)):
        for method_index in range(len(METHOD_KEYS)):
            ordered = strength[model_index, orders[model_index, method_index]]
            observed[model_index, method_index, 1:] = ordered.cumsum(0) / safe[model_index]
        random_strength = strength[model_index][random]
        random_curves[model_index, :, 1:] = random_strength.cumsum(-1) / safe[model_index]
    observed[zero, :] = 0.0
    random_curves[zero, :] = 0.0
    return {
        "audit_full_context_strength": strength,
        "capture_curves": observed,
        "random_capture_curves": random_curves,
        "zero_denominator": zero,
    }


def _within_size_permutations(source_ids: t.Tensor) -> dict[str, Any]:
    ids = _int_tensor(source_ids, "source_ids", 1)
    permutations = t.arange(256, dtype=t.int16).expand(
        WITHIN_SIZE_REPLICATES, len(ids), 256
    ).clone()
    derived_hex: list[tuple[str, ...]] = []
    sizes = coalition_masks(FEATURES).sum(-1).numpy()
    for replicate in range(WITHIN_SIZE_REPLICATES):
        row_seeds = []
        for row_index, source_id in enumerate(ids.tolist()):
            derived = _derived_seed(WITHIN_SIZE_SEED, source_id, replicate)
            row_seeds.append(f"{derived:032x}")
            rng = np.random.Generator(np.random.PCG64(derived))
            for size in range(1, FEATURES):
                positions = np.flatnonzero(sizes == size)
                permutations[replicate, row_index, positions] = t.tensor(
                    rng.permutation(positions), dtype=t.int16
                )
        derived_hex.append(tuple(row_seeds))
    result = {
        "seed": WITHIN_SIZE_SEED,
        "derived_seeds_hex": tuple(derived_hex),
        "permutation": permutations,
    }
    manifest = load_manifest()
    expected_ids = t.tensor(manifest.get("splits", {}).get("audit", []), dtype=t.int64)
    if t.equal(ids, expected_ids):
        contract = manifest.get("rng_contract", {}).get("within_size_shuffle", {})
        expected_hash = contract.get("permutation_value_sha256")
        expected_seeds = contract.get("derived_seeds_hex")
        if expected_seeds != [list(row) for row in result["derived_seeds_hex"]]:
            raise ValueError("Within-size derived seeds differ from the manifest.")
        if expected_hash != tensor_mapping_sha256({"permutation": permutations}):
            raise ValueError("Within-size permutations differ from the manifest.")
    return result


def _localization_statistics(
    pair_values: t.Tensor, group_ids: t.Tensor
) -> dict[str, t.Tensor]:
    values = _float_tensor(pair_values, "pair_values")
    if values.ndim == 3:
        strengths = _group_weighted_mean_abs(values, group_ids)
    elif values.ndim == 4:
        strengths = t.stack(
            [_group_weighted_mean_abs(replicate, group_ids) for replicate in values]
        )
    else:
        raise ValueError("pair_values must have shape [rows,3,28] or [R,rows,3,28].")
    total = strengths.sum(-1)
    mean = strengths.mean(-1)
    zero_total = total == 0
    zero_mean = mean == 0
    top3 = t.topk(strengths, k=3, dim=-1).values.sum(-1)
    return {
        "pair_strength": strengths,
        "top3_share": t.where(zero_total, t.zeros_like(total), top3 / total.clamp_min(t.finfo(t.float64).tiny)),
        "peak_to_mean": t.where(zero_mean, t.zeros_like(mean), strengths.amax(-1) / mean.clamp_min(t.finfo(t.float64).tiny)),
        "zero_total": zero_total,
        "zero_mean": zero_mean,
    }


def within_size_shuffle_control(
    game_values: t.Tensor,
    observed_pair_values: t.Tensor,
    source_ids: t.Tensor,
    recipe_group_ids: t.Tensor,
    *,
    methods: Any,
) -> dict[str, Any]:
    """Build the fixed structural null while preserving endpoints and size marginals."""

    _require_methods(methods)
    values = _float_tensor(game_values, "game_values", 2)
    if values.shape != (len(source_ids), 256):
        raise ValueError("The within-size control requires all audit D=8 games.")
    schedule = _within_size_permutations(source_ids)
    index = schedule["permutation"].to(t.int64)
    shuffled = t.gather(values[None, :, :].expand(WITHIN_SIZE_REPLICATES, -1, -1), 2, index)
    flat = shuffled.reshape(-1, 256)
    mobius = methods.mobius_transform(flat, n_players=FEATURES)
    stii = methods.shapley_taylor_interactions(mobius, n_players=FEATURES, order=2)
    fsii = methods.faith_shap_interactions(flat, n_players=FEATURES, max_order=2)
    pairs = pair_indices(FEATURES)
    pair_masks = (1 << pairs[:, 0]) | (1 << pairs[:, 1])
    pair_values = t.stack(
        [pairwise_sii(flat, n_players=FEATURES), stii[..., pair_masks], fsii[..., pair_masks]],
        dim=-2,
    ).reshape(WITHIN_SIZE_REPLICATES, len(values), len(METHOD_KEYS), 28)
    return {
        **schedule,
        "method_keys": METHOD_KEYS,
        "pair_values": pair_values,
        "observed": _localization_statistics(observed_pair_values, recipe_group_ids),
        "null": _localization_statistics(pair_values, recipe_group_ids),
        "interpretation": "structural null; not expected to have zero pair interaction",
    }


def nearest_training_diagnostics(
    train_x: t.Tensor,
    train_ids: t.Tensor,
    queries: t.Tensor,
    donors: t.Tensor,
    normalization: Mapping[str, t.Tensor],
) -> dict[str, t.Tensor]:
    """Find exact standardized nearest training rows for the fixed support subset."""

    train = _float_tensor(train_x, "train_x", 2)
    ids = _int_tensor(train_ids, "train_ids", 1)
    x = _float_tensor(queries, "queries", 2)
    background = _float_tensor(donors, "donors", 2)
    if train.shape[1] != FEATURES or x.shape[1] != FEATURES:
        raise ValueError("Support diagnostics require eight-feature rows.")
    if ids.shape != (len(train),) or len(t.unique(ids)) != len(ids):
        raise ValueError("train_ids must uniquely identify every training row.")
    if background.shape != (SUPPORT_DONORS, FEATURES):
        raise ValueError("Support diagnostics require the first eight primary donors.")
    mean = _float_tensor(normalization["x_mean"], "x_mean", 1)
    std = _float_tensor(normalization["x_std"], "x_std", 1)
    train_standard = (train - mean) / std
    masks = coalition_masks(FEATURES)
    distance = t.empty((len(x), 256, SUPPORT_DONORS), dtype=t.float64)
    nearest = t.empty((len(x), 256, SUPPORT_DONORS), dtype=t.int64)
    for start in range(0, len(x), 4):
        query = x[start : start + 4]
        hybrid = t.where(
            masks[None, :, None, :],
            query[:, None, None, :],
            background[None, None, :, :],
        )
        flat = ((hybrid - mean) / std).reshape(-1, FEATURES)
        minima = []
        argmin = []
        for index in range(0, len(flat), DISTANCE_BATCH):
            query_block = flat[index : index + DISTANCE_BATCH]
            delta = query_block[:, None, :] - train_standard[None, :, :]
            block = delta.square().sum(-1).sqrt()
            block_min = block.min(-1).values
            tied_ids = t.where(
                block == block_min[:, None],
                ids[None, :],
                t.full_like(ids[None, :], t.iinfo(t.int64).max),
            )
            minima.append(block_min)
            argmin.append(tied_ids.min(-1).values)
        row_shape = (len(query), 256, SUPPORT_DONORS)
        distance[start : start + len(query)] = t.cat(minima).reshape(row_shape)
        nearest[start : start + len(query)] = t.cat(argmin).reshape(row_shape)
    return {
        "donor_rows": SUPPORT_DONORS,
        "nearest_train_distance": distance,
        "nearest_train_source_id": nearest,
    }


def _pair_ranks(pair_values: t.Tensor) -> t.Tensor:
    order = t.argsort(pair_values.abs(), dim=-1, descending=True, stable=True)
    ranks = t.empty_like(order)
    positions = t.arange(order.shape[-1], dtype=t.int64)
    ranks.scatter_(-1, order, positions.expand_as(order))
    return ranks


def background_sensitivity(
    variants: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare every clean_0 finite-background game with primary64."""

    if tuple(variants) != tuple(BACKGROUND_SPECS):
        raise ValueError("Background variants changed order or membership.")
    primary = variants["primary64"]
    result = {}
    for name, item in variants.items():
        pair_movement = item["pair_values"] - primary["pair_values"]
        result[name] = {
            "base_movement": item["coalition_values"][..., 0] - primary["coalition_values"][..., 0],
            "prediction_movement": item["coalition_values"][..., -1] - primary["coalition_values"][..., -1],
            "ordinary_shapley_movement": item["ordinary_shapley"] - primary["ordinary_shapley"],
            "pair_movement": pair_movement,
            "absolute_pair_movement": pair_movement.abs(),
            "pair_sign_change": t.signbit(item["pair_values"]) != t.signbit(primary["pair_values"]),
            "pair_rank_change": _pair_ranks(item["pair_values"]) - _pair_ranks(primary["pair_values"]),
            "full_context_movement": item["full_context_synergy"] - primary["full_context_synergy"],
        }
    return result


def evaluate_study(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    methods: Any,
    reference: Any | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate every prespecified audit outcome without changing preparation."""

    _require_methods(methods)
    if prepared.get("phase") != "prepared_before_audit" or prepared.get("audit_used_for_selection") is not False:
        raise ValueError("A sealed pre-audit preparation payload is required.")
    manifest = load_manifest() if manifest is None else manifest
    reproduction = verify_known_reproduction(fitted, manifest)
    audit_x, audit_y, audit_ids, audit_groups = _rows(audit, "audit")
    expected_audit = t.tensor(manifest.get("splits", {}).get("audit", []), dtype=t.int64)
    if not t.equal(audit_ids, expected_audit):
        raise ValueError("Audit source IDs differ from the pinned split.")
    _require_disjoint_ids(
        train=fitted["train_ids"],
        explain=prepared["explain_source_ids"],
        audit=audit_ids,
    )
    backgrounds = prepared["backgrounds"]
    primary_audit = _evaluate_primary(
        fitted, audit_x, backgrounds["primary64"], methods=methods
    )
    variants_audit = _evaluate_backgrounds(
        fitted, audit_x, backgrounds, primary_audit, methods=methods
    )
    primary_all = _combine_game_blocks(
        prepared["primary64"], primary_audit, model_axis=True
    )
    variants_all = {
        name: _combine_game_blocks(
            prepared["background_variants"][name], variants_audit[name], model_axis=False
        )
        for name in BACKGROUND_SPECS
    }
    direct_audit = _direct_full_context_four(
        fitted, audit_x, backgrounds["primary64"]["x_raw"]
    )
    direct_audit["table_synergy"] = primary_audit["full_context_synergy"]
    direct_audit["lookup_difference"] = direct_audit["synergy"] - direct_audit["table_synergy"]
    direct_all = {
        leaf: t.cat([prepared["direct_full_context"][leaf], direct_audit[leaf]], dim=1)
        for leaf in ("four_values", "synergy", "table_synergy", "lookup_difference")
    }
    random_orders = prepared["random_pair_orders"]["orders"]
    curves = pair_capture_curves(
        prepared["rankings"]["pair_order"],
        primary_audit["full_context_synergy"],
        audit_groups,
        random_orders,
    )
    null = within_size_shuffle_control(
        primary_audit["coalition_values"][0],
        primary_audit["pair_values"][0],
        audit_ids,
        audit_groups,
        methods=methods,
    )
    all_x = t.cat([prepared["explain_x_raw"], audit_x])
    all_ids = t.cat([prepared["explain_source_ids"], audit_ids])
    all_groups = t.cat([prepared["explain_recipe_group_ids"], audit_groups])
    support = nearest_training_diagnostics(
        fitted["train_x_raw"],
        fitted["train_ids"],
        all_x,
        backgrounds["primary64"]["x_raw"][:SUPPORT_DONORS],
        fitted["normalization"],
    )
    row_split = t.cat(
        [
            t.zeros(len(prepared["explain_source_ids"]), dtype=t.int64),
            t.ones(len(audit_ids), dtype=t.int64),
        ]
    )
    return {
        "phase": "evaluated_after_audit",
        "audit_used_for_selection": False,
        "reproduction": reproduction,
        "feature_names": FEATURE_NAMES,
        "feature_units": FEATURE_UNITS,
        "target_name": TARGET_NAME,
        "target_unit": TARGET_UNIT,
        "model_keys": MODEL_KEYS,
        "method_keys": METHOD_KEYS,
        "source_ids": all_ids,
        "row_split": row_split,
        "split_labels": ("explanation", "audit"),
        "recipe_group_ids": all_groups,
        "audit_source_ids": audit_ids,
        "audit_recipe_group_ids": audit_groups,
        "audit_x_raw": audit_x,
        "audit_targets": audit_y,
        "primary64_audit": primary_audit,
        "background_variants_audit": variants_audit,
        "all_rows": {
            "primary64": primary_all,
            "background_variants": variants_all,
        },
        "direct_full_context": direct_all,
        "performance": {
            "explanation": _performance(
                fitted, prepared["explain_x_raw"], prepared["explain_targets"]
            ),
            "audit": _performance(fitted, audit_x, audit_y),
        },
        "rankings": prepared["rankings"],
        "pair_capture": {
            **curves,
            "random_pair_orders": random_orders,
            "method_keys": METHOD_KEYS,
        },
        "random_controls": {"within_size": null},
        "background_sensitivity": background_sensitivity(variants_all),
        "support": support,
        "references": {
            "prepared": prepared["references"],
            "empirical": _empirical_reference_plan("evaluated_after_audit"),
            "adapter_supplied": reference is not None,
        },
        "fixed_play_defaults": dict(DEFAULT_PLAY),
        "source_provenance": dict(prepared.get("source_provenance", {})),
        "storage_plan": _storage_plan(manifest, "evaluated_after_audit"),
    }


def exact_dividend_play(
    *,
    methods: Any,
    n_players: int = 4,
    members: tuple[int, ...] = (0, 1, 2, 3),
    value: float = 1.0,
    order: int = 2,
) -> dict[str, Any]:
    """Rebuild one editable unanimity dividend and all three definitions."""

    _require_methods(methods)
    if not isinstance(n_players, int) or not 2 <= n_players <= 10:
        raise ValueError("n_players must be an integer from two through ten.")
    if not members or len(set(members)) != len(members):
        raise ValueError("members must contain distinct players.")
    if min(members) < 0 or max(members) >= n_players:
        raise ValueError("members contains an out-of-range player.")
    if not isinstance(order, int) or not 1 <= order <= n_players:
        raise ValueError("order must be between one and n_players.")
    if not math.isfinite(float(value)):
        raise ValueError("value must be finite.")
    dividend = t.zeros(2**n_players, dtype=t.float64)
    dividend[sum(1 << member for member in members)] = float(value)
    game = _zeta_transform(dividend, n_players)
    return {
        "kind": "exact_dividend_play",
        "n_players": n_players,
        "members": tuple(members),
        "value": float(value),
        "order": order,
        "mobius_oracle": dividend,
        "mobius": dividend,
        "game_values": game,
        "explained": explain_complete_games(
            game, methods=methods, n_players=n_players, order=order
        ),
    }


def _logical_hybrid_row(
    block: Mapping[str, Any], row_index: int, row_split: t.Tensor
) -> t.Tensor:
    hybrid = block["hybrid_prediction"]
    if not isinstance(hybrid, Mapping) or hybrid.get("storage") != "split_external_shards":
        raise ValueError("Expected split external hybrid shards.")
    explain_rows = int((row_split == 0).sum())
    if row_index < explain_rows:
        return hybrid["explain"][row_index]
    return hybrid["audit"][row_index - explain_rows]


def empirical_context_play(
    evaluated: Mapping[str, Any],
    methods: Any,
    *,
    model_key: str = "clean_0",
    source_id: int = 0,
    background_key: str = "primary64",
    pair: tuple[int, int] = (0, 1),
    context_mask: int = 252,
    donor_source_id: int | None = None,
) -> dict[str, Any]:
    """Inspect a real retained row, donor bank, pair, and coalition context."""

    _require_methods(methods)
    if evaluated.get("phase") != "evaluated_after_audit":
        raise ValueError("Use the complete evaluated payload.")
    if model_key not in MODEL_KEYS or background_key not in BACKGROUND_SPECS:
        raise ValueError("Unknown model or background key.")
    if background_key != "primary64" and model_key != "clean_0":
        raise ValueError("Alternate backgrounds are retained only for clean_0.")
    ids = _int_tensor(evaluated["source_ids"], "source_ids", 1)
    positions = t.nonzero(ids == int(source_id)).flatten()
    if len(positions) != 1:
        raise ValueError("source_id is absent or repeated.")
    row_index = int(positions[0])
    left, right = pair
    if left == right or min(pair) < 0 or max(pair) >= FEATURES:
        raise ValueError("pair must contain two distinct valid players.")
    pair_mask = (1 << left) | (1 << right)
    if context_mask < 0 or context_mask >= 256 or context_mask & pair_mask:
        raise ValueError("context_mask must be valid and disjoint from the pair.")
    if background_key == "primary64":
        primary = evaluated["all_rows"]["primary64"]
        model_index = MODEL_KEYS.index(model_key)
        block = {
            key: (value[:, model_index] if key in _METHOD_FIRST_LEAVES else value[model_index])
            if isinstance(value, t.Tensor) and key not in _SHARED_GAME_LEAVES
            else value
            for key, value in primary.items()
            if key != "model_keys"
        }
        hybrid_meta = primary["hybrid_prediction"]
        block["hybrid_prediction"] = {
            **hybrid_meta,
            "explain": hybrid_meta["explain"][model_index],
            "audit": hybrid_meta["audit"][model_index],
        }
    else:
        block = evaluated["all_rows"]["background_variants"][background_key]
    hybrid_row = _logical_hybrid_row(block, row_index, evaluated["row_split"])
    donor_ids = block["donor_source_ids"]
    if donor_source_id is None:
        game = block["coalition_values"][row_index]
        donor_values = hybrid_row
        selected_donor_ids = donor_ids
        game_label = "equal_weight_donor_bank"
    else:
        donor_positions = t.nonzero(donor_ids == int(donor_source_id)).flatten()
        if len(donor_positions) != 1:
            raise ValueError("donor_source_id is not in the selected background.")
        donor_index = int(donor_positions[0])
        game = hybrid_row[:, donor_index]
        donor_values = hybrid_row[:, donor_index : donor_index + 1]
        selected_donor_ids = donor_ids[donor_index : donor_index + 1]
        game_label = "single_donor_game"
    masks = t.tensor(
        [context_mask, context_mask | (1 << left), context_mask | (1 << right), context_mask | pair_mask],
        dtype=t.int64,
    )
    four_by_donor = donor_values[masks]
    explained = explain_complete_games(
        game, methods=methods, n_players=FEATURES, order=2
    )
    pair_row = t.nonzero((pair_indices(FEATURES) == t.tensor(pair)).all(-1)).flatten()
    if len(pair_row) != 1:
        reverse = t.tensor((right, left), dtype=t.int64)
        pair_row = t.nonzero((pair_indices(FEATURES) == reverse).all(-1)).flatten()
    pair_index = int(pair_row[0])
    return {
        "kind": "empirical_context_play",
        "model_key": model_key,
        "source_id": int(source_id),
        "split": "audit_reuse" if int(evaluated["row_split"][row_index]) else "explanation",
        "background_key": background_key,
        "game_label": game_label,
        "target_unit": TARGET_UNIT,
        "donor_source_ids": selected_donor_ids,
        "pair": tuple(pair),
        "context_mask": int(context_mask),
        "four_masks": masks,
        "four_values_by_donor": four_by_donor,
        "four_values": four_by_donor.mean(-1),
        "local_synergy_by_donor": four_by_donor[3] - four_by_donor[1] - four_by_donor[2] + four_by_donor[0],
        "local_synergy": game[masks[3]] - game[masks[1]] - game[masks[2]] + game[masks[0]],
        "pair_values": explained["pair_values"][..., pair_index],
        "method_keys": METHOD_KEYS,
        "game_values": game.clone(),  # Retain this small game without serializing its donor-bank storage.
        "interprets_model_input_intervention_only": True,
    }


def native_play(evaluated: Mapping[str, Any], *, methods: Any) -> dict[str, Any]:
    """Assemble the fixed content-blind native result without live model objects."""

    if evaluated.get("phase") != "evaluated_after_audit":
        raise ValueError("Run the complete frozen evaluation before native_play.")
    empirical = empirical_context_play(
        evaluated,
        methods,
        model_key=DEFAULT_PLAY["model_key"],
        source_id=DEFAULT_PLAY["source_id"],
        background_key=DEFAULT_PLAY["background_key"],
        pair=DEFAULT_PLAY["pair"],
        context_mask=DEFAULT_PLAY["context_mask"],
        donor_source_id=None,
    )
    donor = empirical_context_play(
        evaluated,
        methods,
        model_key=DEFAULT_PLAY["model_key"],
        source_id=DEFAULT_PLAY["source_id"],
        background_key=DEFAULT_PLAY["background_key"],
        pair=DEFAULT_PLAY["pair"],
        context_mask=DEFAULT_PLAY["context_mask"],
        donor_source_id=DEFAULT_PLAY["donor_source_id"],
    )
    ids = _int_tensor(evaluated["source_ids"], "source_ids", 1)
    row_positions = t.nonzero(ids == DEFAULT_PLAY["source_id"]).flatten()
    if len(row_positions) != 1:
        raise ValueError("The fixed source row is absent or repeated.")
    row_index = int(row_positions[0])
    primary = evaluated["all_rows"]["primary64"]
    clean_index = MODEL_KEYS.index("clean_0")
    ridge_index = MODEL_KEYS.index("ridge")
    fixed_row = {
        "model_key": "clean_0",
        "source_id": DEFAULT_PLAY["source_id"],
        "background_key": "primary64",
        "order": EMPIRICAL_ORDER,
        "donor_count": int(primary["donor_source_ids"].numel()),
        "target_unit": TARGET_UNIT,
        "pair_indices": primary["pair_indices"],
        "stii_coefficients": primary["stii_coefficients"][clean_index, row_index],
        "fsii_coefficients": primary["fsii_coefficients"][clean_index, row_index],
        "sii_pairs": primary["sii_pairs"][clean_index, row_index],
        "ridge_pair_max_abs": primary["pair_values"][ridge_index, row_index].abs().amax(),
    }
    return {
        "phase": "native_play_from_frozen_evaluation",
        "fixed_defaults": dict(DEFAULT_PLAY),
        "exact_dividend": exact_dividend_play(methods=methods),
        "empirical_background_average": empirical,
        "empirical_fixed_donor": donor,
        "fixed_row": fixed_row,
        "overview": {
            "feature_names": evaluated["feature_names"],
            "feature_units": evaluated["feature_units"],
            "target_name": evaluated["target_name"],
            "target_unit": evaluated["target_unit"],
            "model_keys": evaluated["model_keys"],
            "method_keys": evaluated["method_keys"],
            "source_ids": evaluated["source_ids"],
            "row_split": evaluated["row_split"],
            "performance": evaluated["performance"],
            "rankings": evaluated["rankings"],
            "pair_capture": evaluated["pair_capture"],
            "background_sensitivity": evaluated["background_sensitivity"],
            "direct_full_context": evaluated["direct_full_context"],
        },
    }


__all__ = [
    "BACKGROUND_SPECS",
    "DEFAULT_PLAY",
    "EMPIRICAL_ORDER",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "METHOD_KEYS",
    "MODEL_KEYS",
    "RANDOM_EXACT_SEED",
    "RANDOM_PAIR_ORDERS",
    "RANDOM_PAIR_SEED",
    "TARGET_NAME",
    "TARGET_UNIT",
    "WITHIN_SIZE_REPLICATES",
    "WITHIN_SIZE_SEED",
    "all_pair_context_synergies",
    "background_sensitivity",
    "coalition_masks",
    "donor_replacement_games",
    "empirical_context_play",
    "evaluate_model_games",
    "evaluate_study",
    "exact_dividend_play",
    "explain_complete_games",
    "explanation_pair_rankings",
    "fit_known_models",
    "fixed_exact_games",
    "fixed_random_pair_orders",
    "model_from_state",
    "native_play",
    "nearest_training_diagnostics",
    "pair_capture_curves",
    "pair_indices",
    "pairwise_sii",
    "prepare_study",
    "random_exact_games",
    "verify_known_reproduction",
    "within_size_shuffle_control",
]
