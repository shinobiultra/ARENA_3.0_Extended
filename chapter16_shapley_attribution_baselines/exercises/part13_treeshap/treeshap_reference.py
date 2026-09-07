"""Independent TreeSHAP checks against the pinned SHAP 0.49.1 C++ code.

The unmodified interventional reference is retained even when it fails.  SHAP's
``dense_independent`` code rounds sklearn's float64 split thresholds to float32;
``reference_boundary_addendum.md`` records why a separate, copied explainer
uses the greatest float32 threshold below each original threshold.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np

from .treeshap_data import source_data


HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "arena_extended" / "treeshap"
ADDENDUM_SHA256 = "2e639c56060d45131a093caeb838a8896399a2d6e89cba9fe477bcd9b816023e"
ATOL = RTOL = 1e-8
COMPILED_INTERVENTIONAL_ATOL = 2e-5
COMPILED_INTERVENTIONAL_RTOL = 2e-6
MODEL_KEYS = tuple(
    f"{family}_{target}_{seed_index}"
    for family in ("tree", "forest", "boost")
    for seed_index in range(3)
    for target in ("clean", "shuffled")
)
EXACT_KEYS = tuple(f"{family}_clean_0" for family in ("tree", "forest", "boost"))


def permutation_reference(game: np.ndarray) -> tuple[np.ndarray, int]:
    """Exact independent oracle: average marginal gains over every ordering.

    This intentionally does not call the lesson's subset contraction.  It is
    used on tiny fixtures because D! storage is unsuitable for the real study.
    """
    game = np.asarray(game, dtype=np.float64)
    coalition_count = game.shape[-1]
    d = coalition_count.bit_length() - 1
    if d < 1 or coalition_count != 2**d:
        raise ValueError("The final game axis must contain all 2**D coalitions.")
    orders = tuple(itertools.permutations(range(d)))
    contributions = np.zeros((*game.shape[:-1], len(orders), d), dtype=np.float64)
    for order_index, order in enumerate(orders):
        mask = 0
        for feature in order:
            added = mask | (1 << feature)
            contributions[..., order_index, feature] = game[..., added] - game[..., mask]
            mask = added
    return contributions.mean(axis=-2), len(orders)


def subset_reference(game: np.ndarray) -> np.ndarray:
    """Exact all-subset contraction, independently written for actual games."""
    game = np.asarray(game, dtype=np.float64)
    coalition_count = game.shape[-1]
    d = coalition_count.bit_length() - 1
    if d < 1 or coalition_count != 2**d:
        raise ValueError("The final game axis must contain all 2**D coalitions.")
    result = np.zeros((*game.shape[:-1], d), dtype=np.float64)
    for feature in range(d):
        for mask in range(coalition_count):
            if mask & (1 << feature):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(d - size - 1) / math.factorial(d)
            result[..., feature] += weight * (
                game[..., mask | (1 << feature)] - game[..., mask]
            )
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_hash(model) -> str:
    return hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    return float(np.max(np.abs(actual - expected), initial=0.0))


def _require_close(name, actual, expected, *, atol=ATOL, rtol=RTOL) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    try:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f"{name} failed full-value parity.\n{error}") from error
    return _max_abs(actual, expected)


def _isclose_rows(actual, expected, *, atol, rtol) -> np.ndarray:
    close = np.isclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol)
    return close if close.ndim == 1 else close.reshape(len(close), -1).all(axis=1)


def _interventional_game_reference(predict, x, background):
    """Enumerate all intact-donor games for the rounded-threshold proof."""
    masks = np.array(
        [[bool(mask & (1 << feature)) for feature in range(8)] for mask in range(256)]
    )
    values = np.empty((len(x), len(masks)), dtype=np.float64)
    for start in range(0, len(x), 8):
        query = x[start : start + 8]
        hybrids = np.where(
            masks[None, :, None, :],
            query[:, None, None, :],
            background[None, None, :, :],
        )
        prediction = predict(hybrids.reshape(-1, 8))
        values[start : start + len(query)] = prediction.reshape(
            len(query), 256, 64
        ).mean(axis=2)
    return values


def _reference_environment() -> dict:
    import shap
    import sklearn
    from shap.explainers import _tree

    manifest = json.loads((HERE / "author_reference.json").read_text())
    expected_versions = {"numpy": "1.26.4", "scikit_learn": "1.9.0", "shap": "0.49.1"}
    actual_versions = {
        "numpy": np.__version__, "scikit_learn": sklearn.__version__, "shap": shap.__version__
    }
    if actual_versions != expected_versions:
        raise ValueError(f"Reference version mismatch: {actual_versions} != {expected_versions}")
    if getattr(_tree, "_cext", None) is None or not hasattr(_tree._cext, "dense_tree_shap"):
        raise RuntimeError("Installed SHAP C++ dense_tree_shap is unavailable.")

    source_paths = {
        "explainers/pytree.py": CACHE / "explainers" / "pytree.py",
        "explainers/_tree.py": CACHE / "explainers" / "_tree.py",
        "cext/tree_shap.h": CACHE / "cext" / "tree_shap.h",
    }
    source_hashes = {}
    for name, path in source_paths.items():
        digest = _sha256(path)
        if digest != manifest["sources"][name]:
            raise ValueError(f"Cached reference checksum mismatch for {name}.")
        source_hashes[name] = digest
    installed_tree_hash = _sha256(Path(_tree.__file__))
    if installed_tree_hash != manifest["sources"]["explainers/_tree.py"]:
        raise ValueError("Installed SHAP TreeExplainer Python source is not the pinned source.")
    paper_hash = _sha256(CACHE / "treeshap_2018.pdf")
    if paper_hash != manifest["paper_pdf_sha256"]:
        raise ValueError("Cached TreeSHAP article checksum mismatch.")
    addendum_hash = _sha256(HERE / "reference_boundary_addendum.md")
    if addendum_hash != ADDENDUM_SHA256:
        raise ValueError("The post-freeze threshold addendum changed.")
    return {
        "versions": actual_versions,
        "cached_source_sha256": source_hashes,
        "installed_tree_source_sha256": installed_tree_hash,
        "paper_pdf_sha256": paper_hash,
        "boundary_addendum_sha256": addendum_hash,
        "cext_module": str(Path(_tree._cext.__file__).resolve()),
    }


def _frozen_identity():
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    data = source_data()
    ids = {
        name: np.asarray(manifest["splits"][name], dtype=np.int64)
        for name in ("train", "explain", "audit")
    }
    return data, ids


def _background_indices(train_ids: np.ndarray) -> np.ndarray:
    return np.array(
        sorted(
            range(len(train_ids)),
            key=lambda i: hashlib.sha256(
                f"arena16.13/background/161360/{int(train_ids[i])}".encode()
            ).digest(),
        ),
        dtype=np.int64,
    )[:64]


def _validate_identity(fitted, prepared, evaluated, explain, audit):
    data, frozen_ids = _frozen_identity()
    supplied = {
        "train": np.asarray(fitted["train_ids"]),
        "explain": np.asarray(explain["ids"]),
        "audit": np.asarray(audit["ids"]),
    }
    for name in ("train", "explain", "audit"):
        if not np.array_equal(supplied[name], frozen_ids[name]):
            raise ValueError(f"{name} row identity/order does not match the frozen manifest.")
    if not np.array_equal(np.asarray(prepared["explain_ids"]), frozen_ids["explain"]):
        raise ValueError("prepared explanation row order does not match the frozen manifest.")
    if not np.array_equal(np.asarray(evaluated["audit_ids"]), frozen_ids["audit"]):
        raise ValueError("evaluated audit row order does not match the frozen manifest.")

    for name, block in (("explain", explain), ("audit", audit)):
        expected = data[frozen_ids[name]]
        if not np.array_equal(np.asarray(block["x"]), expected[:, :8]):
            raise ValueError(f"{name} inputs do not match their frozen source-row order.")
        if not np.array_equal(np.asarray(block["y"]), expected[:, 8]):
            raise ValueError(f"{name} targets do not match their frozen source-row order.")
    if not np.array_equal(np.asarray(evaluated["audit_targets"]), data[frozen_ids["audit"], 8]):
        raise ValueError("evaluated audit targets do not match the frozen source rows.")

    order = _background_indices(frozen_ids["train"])
    expected_background_ids = frozen_ids["train"][order]
    if not np.array_equal(np.asarray(prepared["background_ids"]), expected_background_ids):
        raise ValueError("Primary background IDs/order are not the frozen 64 donors.")
    expected_background = data[expected_background_ids, :8].astype(np.float32)
    actual_background = np.asarray(prepared["background"], dtype=np.float32)
    if actual_background.shape != (64, 8) or not np.array_equal(
        actual_background, expected_background
    ):
        raise ValueError("Primary background values are not all 64 frozen float32 donors.")
    return (
        data[:, :8].astype(np.float32),
        np.asarray(explain["x"], dtype=np.float32),
        np.asarray(audit["x"], dtype=np.float32),
        expected_background,
        frozen_ids,
    )


def _validate_block(name, block, rows):
    if set(block) != set(MODEL_KEYS):
        raise ValueError(f"{name} must contain exactly the 18 frozen model keys.")
    ordinary = {
        "prediction": (rows,),
        "base_path": (rows,),
        "phi_path": (rows, 8),
        "base_interventional": (rows,),
        "phi_interventional": (rows, 8),
        "phi_saabas": (rows, 8),
    }
    for key in MODEL_KEYS:
        item = block[key]
        for field, shape in ordinary.items():
            if field not in item or np.asarray(item[field]).shape != shape:
                raise ValueError(f"{name}[{key!r}][{field!r}] must have shape {shape}.")
        for field in ("game_path", "game_interventional"):
            present = field in item
            if key in EXACT_KEYS and (not present or np.asarray(item[field]).shape != (rows, 256)):
                raise ValueError(f"{name}[{key!r}][{field!r}] must have shape {(rows, 256)}.")
            if key not in EXACT_KEYS and present:
                raise ValueError(f"{name}[{key!r}] unexpectedly contains {field}.")


def _floor_float32_thresholds(original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nearest = np.asarray(original, dtype=np.float32)
    rounded_up = nearest.astype(np.float64) > np.asarray(original, dtype=np.float64)
    corrected = nearest.copy()
    corrected[rounded_up] = np.nextafter(
        corrected[rounded_up], np.float32(-np.inf)
    )
    return nearest, corrected, rounded_up


def _threshold_copies(raw_explainer, sklearn_model, all_source_x):
    original = raw_explainer.model.thresholds.copy()
    internal = (raw_explainer.model.children_left >= 0) & (raw_explainer.model.features >= 0)
    nearest, corrected_values, rounded_up = _floor_float32_thresholds(original)
    change = internal & rounded_up

    corrected = copy.deepcopy(raw_explainer)
    corrected.model.thresholds[internal] = corrected_values[internal].astype(np.float64)
    corrected.expected_value = corrected.model.predict(corrected.data).mean(0)

    nearest_model = copy.deepcopy(raw_explainer)
    nearest_model.model.thresholds[internal] = nearest[internal].astype(np.float64)
    nearest_model.expected_value = nearest_model.model.predict(nearest_model.data).mean(0)

    positions = np.argwhere(change)
    trees = positions[:, 0] if len(positions) else np.empty(0, dtype=np.int64)
    nodes = positions[:, 1] if len(positions) else np.empty(0, dtype=np.int64)
    features = raw_explainer.model.features[change].astype(np.int64)
    changed_original = original[change]
    changed_nearest = nearest[change]
    changed_corrected = corrected_values[change]
    lower = np.nextafter(changed_corrected, np.float32(-np.inf))
    upper = np.nextafter(changed_corrected, np.float32(np.inf))
    neighbors = np.column_stack([lower, changed_corrected, upper]).astype(np.float32)
    original_routes = neighbors.astype(np.float64) <= changed_original[:, None]
    corrected_routes = neighbors <= changed_corrected[:, None]
    raw_routes = neighbors <= changed_nearest[:, None]
    expected_routes = np.tile(np.array([True, True, False]), (len(positions), 1))
    if not np.array_equal(original_routes, expected_routes):
        raise AssertionError("Float32 boundary neighbors do not bracket original thresholds.")
    if not np.array_equal(corrected_routes, expected_routes):
        raise AssertionError("Corrected threshold boundary proof failed.")

    source_mismatches_raw = np.zeros(len(positions), dtype=np.int64)
    for index, (feature, threshold, raw_threshold, fixed_threshold) in enumerate(
        zip(features, changed_original, changed_nearest, changed_corrected, strict=True)
    ):
        values = all_source_x[:, feature]
        original_route = values.astype(np.float64) <= threshold
        corrected_route = values <= fixed_threshold
        if not np.array_equal(original_route, corrected_route):
            raise AssertionError("Correction changed an original sklearn route on source data.")
        source_mismatches_raw[index] = np.count_nonzero(original_route != (values <= raw_threshold))

    corrected_source_route_disagreements = 0
    for tree_index, node_index in np.argwhere(internal):
        feature = raw_explainer.model.features[tree_index, node_index]
        values = all_source_x[:, feature]
        original_route = values.astype(np.float64) <= original[tree_index, node_index]
        corrected_route = values <= corrected_values[tree_index, node_index]
        corrected_source_route_disagreements += int(
            np.count_nonzero(original_route != corrected_route)
        )
    if corrected_source_route_disagreements:
        raise AssertionError("Floored thresholds changed original routes on source data.")

    corrected_prediction = corrected.model.predict(all_source_x)
    sklearn_prediction = sklearn_model.predict(all_source_x)
    corrected_prediction_error = _require_close(
        "corrected explainer routes on all 1,030 source rows",
        corrected_prediction,
        sklearn_prediction,
    )
    nearest_prediction = nearest_model.model.predict(all_source_x)
    return corrected, nearest_model, {
        "component_tree_count": int(len(original)),
        "internal_threshold_count": int(np.count_nonzero(internal)),
        "rounded_up_threshold_count": int(np.count_nonzero(change)),
        "tree_index": trees,
        "node_index": nodes,
        "feature": features,
        "original": changed_original,
        "raw_nearest_float32": changed_nearest,
        "corrected_floor_float32": changed_corrected,
        "neighbor_values": neighbors,
        "original_neighbor_routes": original_routes,
        "raw_neighbor_routes": raw_routes,
        "corrected_neighbor_routes": corrected_routes,
        "source_raw_route_mismatch_count": source_mismatches_raw,
        "all_source_corrected_route_disagreement_count": corrected_source_route_disagreements,
        "all_source_corrected_prediction_max_abs_error": corrected_prediction_error,
        "all_source_raw_rounded_prediction_max_abs_error": _max_abs(
            nearest_prediction, sklearn_prediction
        ),
        "all_source_raw_rounded_prediction_mismatch_count": int(
            np.count_nonzero(~np.isclose(nearest_prediction, sklearn_prediction, atol=ATOL, rtol=RTOL))
        ),
    }


def _compiled_reference(model, all_x, background, all_source_x, *, exact_raw_game=False):
    import shap

    model_hash_before = _model_hash(model)
    path_explainer = shap.TreeExplainer(
        model, model_output="raw", feature_perturbation="tree_path_dependent"
    )
    if path_explainer.feature_perturbation != "tree_path_dependent":
        raise AssertionError("Path reference did not retain the explicit perturbation mode.")
    phi_path = np.asarray(
        path_explainer.shap_values(all_x, approximate=False, check_additivity=True),
        dtype=np.float64,
    )
    base_path = np.full(len(all_x), float(path_explainer.expected_value))

    raw = shap.TreeExplainer(
        model,
        data=background,
        model_output="raw",
        feature_perturbation="interventional",
    )
    if raw.feature_perturbation != "interventional" or raw.model_output != "raw":
        raise AssertionError("Interventional reference changed the explicit mode/output.")
    if raw.data.shape != (64, 8) or raw.data.dtype != np.float32:
        raise AssertionError("TreeExplainer did not retain the complete float32 background.")
    if not np.array_equal(raw.data, background):
        raise AssertionError("TreeExplainer altered or subsampled the 64-row background.")
    phi_raw = np.asarray(
        # Preserve the unmodified library output even where SHAP's own loose
        # additivity guard raises; the strict residual is checked and returned below.
        raw.shap_values(all_x, approximate=False, check_additivity=False), dtype=np.float64
    )
    base_raw = np.full(len(all_x), float(raw.expected_value))

    corrected, nearest_model, threshold_audit = _threshold_copies(
        raw, model, all_source_x
    )
    if not np.array_equal(corrected.data, background) or corrected.data.shape != (64, 8):
        raise AssertionError("Corrected explainer must still use every frozen donor.")
    phi_corrected = np.asarray(
        corrected.shap_values(all_x, approximate=False, check_additivity=True),
        dtype=np.float64,
    )
    base_corrected = np.full(len(all_x), float(corrected.expected_value))
    nearest_prediction = nearest_model.model.predict(all_x)
    nearest_base = np.full(len(all_x), float(nearest_model.expected_value))
    raw_rounding_error = _require_close(
        "unmodified C++ totals against nearest-float32 threshold model",
        nearest_base + phi_raw.sum(axis=1),
        nearest_prediction,
        atol=COMPILED_INTERVENTIONAL_ATOL,
        rtol=COMPILED_INTERVENTIONAL_RTOL,
    )
    model_hash_after = _model_hash(model)
    if model_hash_before != model_hash_after:
        raise AssertionError("Reference construction mutated the fitted sklearn model.")
    result = {
        "prediction": np.asarray(model.predict(all_x), dtype=np.float64),
        "base_path": base_path,
        "phi_path": phi_path,
        "base_interventional_raw": base_raw,
        "phi_interventional_raw": phi_raw,
        "raw_nearest_threshold_prediction": nearest_prediction,
        "raw_nearest_threshold_base": nearest_base,
        "base_interventional_corrected": base_corrected,
        "phi_interventional_corrected": phi_corrected,
        "raw_nearest_threshold_additivity_max_abs_error": raw_rounding_error,
        "threshold_audit": threshold_audit,
        "sklearn_model_sha256_before": model_hash_before,
        "sklearn_model_sha256_after": model_hash_after,
    }
    if exact_raw_game:
        result["game_interventional_raw_nearest"] = _interventional_game_reference(
            nearest_model.model.predict, all_x, background
        )
    return result


def check_reference(fitted, prepared, evaluated, explain, audit):
    """Check every fitted model and row, returning all numeric reference arrays.

    The unmodified interventional arrays are evidence of a pinned library
    failure.  Only the separately copied threshold-corrected explainer is
    required to match the actual finite-background game.
    """
    environment = _reference_environment()
    if set(fitted["models"]) != set(MODEL_KEYS):
        raise ValueError("fitted['models'] must contain exactly the 18 frozen models.")
    _validate_block("prepared explanation", prepared["explanation"], 153)
    _validate_block("evaluated audit", evaluated["audit"], 119)
    all_source_x, explain_x, audit_x, background, frozen_ids = _validate_identity(
        fitted, prepared, evaluated, explain, audit
    )
    all_x = np.vstack([explain_x, audit_x])
    split_slices = {"explanation": slice(0, 153), "audit": slice(153, 272)}
    stored_blocks = {
        "explanation": prepared["explanation"],
        "audit": evaluated["audit"],
    }

    references = {"explanation": {}, "audit": {}}
    threshold_audit = {}
    model_hashes = {}
    per_model_max_errors = {}
    max_errors = {
        "prediction": 0.0,
        "path_base": 0.0,
        "path_phi": 0.0,
        "interventional_raw_base": 0.0,
        "interventional_raw_phi": 0.0,
        "interventional_raw_prediction_residual": 0.0,
        "interventional_raw_nearest_threshold_additivity": 0.0,
        "interventional_corrected_base": 0.0,
        "interventional_corrected_phi": 0.0,
        "interventional_corrected_prediction_residual": 0.0,
        "exact_path_phi": 0.0,
        "exact_interventional_phi": 0.0,
        "exact_interventional_raw_phi": 0.0,
        "exact_raw_rounded_interventional_phi": 0.0,
    }
    affected = {name: [] for name in ("split", "model_index", "row_index", "source_id")}
    affected.update(
        prediction_residual=[], phi_max_abs_error=[],
        prediction_failed=[], phi_failed=[], base_failed=[],
    )

    for model_index, key in enumerate(MODEL_KEYS):
        item = fitted["models"][key]
        if "sklearn" not in item:
            raise ValueError(f"Live fitted model {key} lacks its sklearn estimator.")
        compiled = _compiled_reference(
            item["sklearn"], all_x, background, all_source_x,
            exact_raw_game=key in EXACT_KEYS,
        )
        threshold_audit[key] = compiled.pop("threshold_audit")
        model_hashes[key] = {
            "before": compiled.pop("sklearn_model_sha256_before"),
            "after": compiled.pop("sklearn_model_sha256_after"),
        }
        max_errors["interventional_raw_nearest_threshold_additivity"] = max(
            max_errors["interventional_raw_nearest_threshold_additivity"],
            compiled.pop("raw_nearest_threshold_additivity_max_abs_error"),
        )
        per_model_max_errors[key] = {}

        for split, section in split_slices.items():
            stored = stored_blocks[split][key]
            ref = {
                field: np.asarray(values[section]).copy()
                for field, values in compiled.items()
            }
            ref["raw_prediction_residual"] = (
                ref["base_interventional_raw"]
                + ref["phi_interventional_raw"].sum(axis=1)
                - ref["prediction"]
            )
            ref["corrected_prediction_residual"] = (
                ref["base_interventional_corrected"]
                + ref["phi_interventional_corrected"].sum(axis=1)
                - ref["prediction"]
            )
            references[split][key] = ref

            errors = {
                "prediction": _require_close(
                    f"{split} {key} prediction", stored["prediction"], ref["prediction"]
                ),
                "path_base": _require_close(
                    f"{split} {key} path base", stored["base_path"], ref["base_path"]
                ),
                "path_phi": _require_close(
                    f"{split} {key} path local values", stored["phi_path"], ref["phi_path"]
                ),
                "interventional_corrected_base": _require_close(
                    f"{split} {key} corrected interventional base",
                    stored["base_interventional"],
                    ref["base_interventional_corrected"],
                    atol=COMPILED_INTERVENTIONAL_ATOL,
                    rtol=COMPILED_INTERVENTIONAL_RTOL,
                ),
                "interventional_corrected_phi": _require_close(
                    f"{split} {key} corrected interventional local values",
                    stored["phi_interventional"],
                    ref["phi_interventional_corrected"],
                    atol=COMPILED_INTERVENTIONAL_ATOL,
                    rtol=COMPILED_INTERVENTIONAL_RTOL,
                ),
                "interventional_raw_base": _max_abs(
                    stored["base_interventional"], ref["base_interventional_raw"]
                ),
                "interventional_raw_phi": _max_abs(
                    stored["phi_interventional"], ref["phi_interventional_raw"]
                ),
                "interventional_raw_prediction_residual": float(
                    np.max(np.abs(ref["raw_prediction_residual"]), initial=0.0)
                ),
                "interventional_corrected_prediction_residual": _require_close(
                    f"{split} {key} corrected interventional additivity",
                    ref["base_interventional_corrected"]
                    + ref["phi_interventional_corrected"].sum(axis=1),
                    ref["prediction"],
                    atol=COMPILED_INTERVENTIONAL_ATOL,
                    rtol=COMPILED_INTERVENTIONAL_RTOL,
                ),
            }
            _require_close(
                f"{split} {key} path additivity",
                ref["base_path"] + ref["phi_path"].sum(axis=1),
                ref["prediction"],
            )
            _require_close(
                f"{split} {key} Saabas additivity",
                np.asarray(stored["base_path"]) + np.asarray(stored["phi_saabas"]).sum(axis=1),
                ref["prediction"],
            )
            per_model_max_errors[key][split] = errors
            for name, value in errors.items():
                max_errors[name] = max(max_errors[name], value)

            raw_phi_close = _isclose_rows(
                ref["phi_interventional_raw"], stored["phi_interventional"],
                atol=COMPILED_INTERVENTIONAL_ATOL,
                rtol=COMPILED_INTERVENTIONAL_RTOL,
            )
            raw_base_close = _isclose_rows(
                ref["base_interventional_raw"], stored["base_interventional"],
                atol=COMPILED_INTERVENTIONAL_ATOL,
                rtol=COMPILED_INTERVENTIONAL_RTOL,
            )
            raw_prediction_close = _isclose_rows(
                ref["base_interventional_raw"] + ref["phi_interventional_raw"].sum(axis=1),
                ref["prediction"],
                atol=COMPILED_INTERVENTIONAL_ATOL,
                rtol=COMPILED_INTERVENTIONAL_RTOL,
            )
            raw_rows_close = raw_phi_close & raw_base_close & raw_prediction_close
            ids = frozen_ids["explain" if split == "explanation" else "audit"]
            for row in np.flatnonzero(~raw_rows_close):
                affected["split"].append(0 if split == "explanation" else 1)
                affected["model_index"].append(model_index)
                affected["row_index"].append(int(row))
                affected["source_id"].append(int(ids[row]))
                affected["prediction_residual"].append(float(ref["raw_prediction_residual"][row]))
                affected["phi_max_abs_error"].append(
                    float(np.max(np.abs(
                        ref["phi_interventional_raw"][row]
                        - np.asarray(stored["phi_interventional"])[row]
                    )))
                )
                affected["prediction_failed"].append(not raw_prediction_close[row])
                affected["phi_failed"].append(not raw_phi_close[row])
                affected["base_failed"].append(not raw_base_close[row])

    exact_max = {
        "path_phi": 0.0,
        "interventional_phi": 0.0,
        "interventional_raw_phi": 0.0,
        "raw_rounded_interventional_phi": 0.0,
        "path_base": 0.0,
        "interventional_base": 0.0,
        "prediction": 0.0,
    }
    for split in ("explanation", "audit"):
        for key in EXACT_KEYS:
            stored = stored_blocks[split][key]
            ref = references[split][key]
            exact_path = subset_reference(stored["game_path"])
            exact_interventional = subset_reference(stored["game_interventional"])
            exact_max["path_base"] = max(
                exact_max["path_base"],
                _require_close(
                    f"{split} {key} exact path empty coalition",
                    stored["game_path"][:, 0], stored["base_path"],
                ),
            )
            exact_max["interventional_base"] = max(
                exact_max["interventional_base"],
                _require_close(
                    f"{split} {key} exact interventional empty coalition",
                    stored["game_interventional"][:, 0], stored["base_interventional"],
                ),
            )
            for game_name in ("game_path", "game_interventional"):
                exact_max["prediction"] = max(
                    exact_max["prediction"],
                    _require_close(
                        f"{split} {key} {game_name} full coalition",
                        stored[game_name][:, -1], stored["prediction"],
                    ),
                )
            exact_max["path_phi"] = max(
                exact_max["path_phi"],
                _require_close(
                    f"{split} {key} exhaustive path values", exact_path, stored["phi_path"]
                ),
                _require_close(
                    f"{split} {key} compiled path versus exhaustive", ref["phi_path"], exact_path
                ),
            )
            exact_max["interventional_phi"] = max(
                exact_max["interventional_phi"],
                _require_close(
                    f"{split} {key} exhaustive interventional values",
                    exact_interventional, stored["phi_interventional"],
                ),
                _require_close(
                    f"{split} {key} corrected compiled versus exhaustive",
                    ref["phi_interventional_corrected"], exact_interventional,
                    atol=COMPILED_INTERVENTIONAL_ATOL,
                    rtol=COMPILED_INTERVENTIONAL_RTOL,
                ),
            )
            exact_max["interventional_raw_phi"] = max(
                exact_max["interventional_raw_phi"],
                _max_abs(ref["phi_interventional_raw"], exact_interventional),
            )
            raw_rounded_exact = subset_reference(ref["game_interventional_raw_nearest"])
            exact_max["raw_rounded_interventional_phi"] = max(
                exact_max["raw_rounded_interventional_phi"],
                _require_close(
                    f"{split} {key} raw C++ versus its rounded-threshold exact game",
                    ref["phi_interventional_raw"], raw_rounded_exact,
                    atol=COMPILED_INTERVENTIONAL_ATOL,
                    rtol=COMPILED_INTERVENTIONAL_RTOL,
                ),
            )
            _require_close(
                f"{split} {key} raw rounded game empty coalition",
                ref["game_interventional_raw_nearest"][:, 0],
                ref["raw_nearest_threshold_base"],
            )
            _require_close(
                f"{split} {key} raw rounded game full coalition",
                ref["game_interventional_raw_nearest"][:, -1],
                ref["raw_nearest_threshold_prediction"],
            )
    max_errors["exact_path_phi"] = exact_max["path_phi"]
    max_errors["exact_interventional_phi"] = exact_max["interventional_phi"]
    max_errors["exact_interventional_raw_phi"] = exact_max["interventional_raw_phi"]
    max_errors["exact_raw_rounded_interventional_phi"] = exact_max[
        "raw_rounded_interventional_phi"
    ]

    affected_arrays = {
        name: np.asarray(
            values,
            dtype=(
                np.float64
                if name in ("prediction_residual", "phi_max_abs_error")
                else np.bool_
                if name.endswith("_failed")
                else np.int64
            ),
        )
        for name, values in affected.items()
    }
    unmodified_passed = len(affected_arrays["row_index"]) == 0
    if unmodified_passed:
        raise AssertionError(
            "The pinned SHAP 0.49.1 raw interventional threshold failure was not reproduced."
        )
    threshold_counts = {
        "component_trees": int(
            sum(v["component_tree_count"] for v in threshold_audit.values())
        ),
        "internal_thresholds": int(
            sum(v["internal_threshold_count"] for v in threshold_audit.values())
        ),
        "corrected_thresholds": int(
            sum(v["rounded_up_threshold_count"] for v in threshold_audit.values())
        ),
        "raw_source_route_disagreements": int(
            sum(v["source_raw_route_mismatch_count"].sum() for v in threshold_audit.values())
        ),
        "corrected_source_route_disagreements": int(
            sum(
                v["all_source_corrected_route_disagreement_count"]
                for v in threshold_audit.values()
            )
        ),
    }
    if tuple(threshold_counts[name] for name in (
        "component_trees", "internal_thresholds", "corrected_thresholds",
        "raw_source_route_disagreements", "corrected_source_route_disagreements",
    )) != (486, 5480, 941, 217, 0):
        raise AssertionError(f"Frozen threshold audit counts changed: {threshold_counts}")
    return {
        "status": "PASS_WITH_EXPECTED_RAW_REFERENCE_FAILURE",
        "passes": {
            "row_identity": True,
            "background_identity_all_64": True,
            "prediction": True,
            "path_compiled": True,
            "interventional_unmodified": False,
            "interventional_threshold_corrected": True,
            "exhaustive_path": True,
            "exhaustive_interventional": True,
            "sklearn_models_unchanged": True,
        },
        "unmodified_interventional_passed": False,
        "threshold_corrected_interventional_passed": True,
        "model_keys": MODEL_KEYS,
        "counts": {
            "models": 18,
            "explanation_rows": 153,
            "audit_rows": 119,
            "model_rows": 18 * (153 + 119),
            "local_values_per_game": 18 * (153 + 119) * 8,
            "exact_models": 3,
            "exact_model_rows": 3 * (153 + 119),
            "background_rows": 64,
            "raw_affected_model_rows": len(affected_arrays["row_index"]),
            "raw_prediction_affected_model_rows": int(
                affected_arrays["prediction_failed"].sum()
            ),
            "raw_phi_affected_model_rows": int(affected_arrays["phi_failed"].sum()),
            **threshold_counts,
        },
        "explain_ids": frozen_ids["explain"].copy(),
        "audit_ids": frozen_ids["audit"].copy(),
        "background_ids": np.asarray(prepared["background_ids"]).copy(),
        "background_float32": background.copy(),
        "explanation": references["explanation"],
        "audit": references["audit"],
        "raw_affected_rows": affected_arrays,
        "threshold_audit": threshold_audit,
        "sklearn_model_hashes": model_hashes,
        "max_errors": max_errors,
        "exact_max_errors": exact_max,
        "per_model_max_errors": per_model_max_errors,
        "reference_environment": environment,
    }
