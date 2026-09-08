"""Independent SHAP 0.49.1 and algebraic references for the DeepSHAP study.

This module deliberately does not import the learner's attribution methods or
study implementation.  It reconstructs the seven raw-unit Sequential models
from their saved tensor states, checks every frozen row and donor identity, and
uses the installed, source-pinned ``shap.DeepExplainer`` as the author oracle.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.deepshap_data import (
    source_data,
)


HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache/arena_extended/deepshap"
FEATURES = 8
HIDDEN = 32
BACKGROUND_ROWS = 64
PRIMARY_BACKGROUND_SEED = 161460
MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
AUTHOR_ATOL = 1e-7
AUTHOR_RTOL = 1e-8
EXACT_ATOL = 1e-10
EXACT_RTOL = 1e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value: Any) -> np.ndarray:
    if isinstance(value, t.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _float64_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if isinstance(value, t.Tensor):
        if value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError(f"{name} must be a CPU float64 tensor or float64 array.")
    array = _array(value)
    if array.dtype != np.float64 or array.shape != shape:
        raise ValueError(f"{name} must have float64 shape {shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values.")
    return np.array(array, copy=True)


def _int64_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if isinstance(value, t.Tensor):
        if value.device.type != "cpu" or value.dtype != t.int64:
            raise TypeError(f"{name} must be a CPU int64 tensor or int64 array.")
    array = _array(value)
    if array.dtype != np.int64 or array.shape != shape:
        raise ValueError(f"{name} must have int64 shape {shape}.")
    return np.array(array, copy=True)


def _max_abs(actual: Any, expected: Any) -> float:
    difference = np.abs(_array(actual).astype(np.float64) - _array(expected).astype(np.float64))
    return float(difference.max(initial=0.0))


def _require_close(
    name: str,
    actual: Any,
    expected: Any,
    *,
    atol: float = EXACT_ATOL,
    rtol: float = EXACT_RTOL,
) -> float:
    actual_array = _array(actual)
    expected_array = _array(expected)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array, expected_array, atol=atol, rtol=rtol
    ):
        raise AssertionError(
            f"{name} failed full-value parity: shape {actual_array.shape} versus "
            f"{expected_array.shape}, max absolute error {_max_abs(actual_array, expected_array):.12g}."
        )
    return _max_abs(actual_array, expected_array)


def subset_reference(values: Any) -> np.ndarray:
    """Contract complete games with independently constructed Shapley weights."""
    game = np.asarray(values, dtype=np.float64)
    if game.ndim < 1 or not np.isfinite(game).all():
        raise ValueError("values must be a finite array with a coalition axis.")
    size = game.shape[-1]
    features = size.bit_length() - 1
    if features < 1 or size != 2**features:
        raise ValueError("The final axis must contain all 2**D coalitions for D >= 1.")
    result = np.zeros((*game.shape[:-1], features), dtype=np.float64)
    subset_ids = np.arange(size, dtype=np.int64)
    for feature in range(features):
        absent = subset_ids[(subset_ids & (1 << feature)) == 0]
        cardinality = np.fromiter((int(mask).bit_count() for mask in absent), dtype=np.int64)
        weights = np.array(
            [1.0 / (features * math.comb(features - 1, int(k))) for k in cardinality],
            dtype=np.float64,
        )
        result[..., feature] = (
            (game[..., absent | (1 << feature)] - game[..., absent]) * weights
        ).sum(axis=-1)
    return result


def permutation_reference(values: Any) -> tuple[np.ndarray, int]:
    """Average marginal contributions over every feature permutation.

    This deliberately slow definition is used only for small hand games.  It is
    algorithmically separate from the subset-weight contraction used for the
    real eight-feature games.
    """
    game = np.asarray(values, dtype=np.float64)
    if game.ndim < 1 or not np.isfinite(game).all():
        raise ValueError("values must be a finite array with a coalition axis.")
    size = game.shape[-1]
    features = size.bit_length() - 1
    if features < 1 or size != 2**features:
        raise ValueError("The final axis must contain all 2**D coalitions for D >= 1.")
    permutations = tuple(itertools.permutations(range(features)))
    result = np.zeros((*game.shape[:-1], features), dtype=np.float64)
    for order in permutations:
        before = 0
        for feature in order:
            after = before | (1 << feature)
            result[..., feature] += game[..., after] - game[..., before]
            before = after
    result /= len(permutations)
    return result, len(permutations)


def ridge_closed_form(
    weight: Any, bias: Any, x: Any, background: Any
) -> dict[str, np.ndarray | float]:
    """Analytic empirical-background SHAP values for one affine scalar model."""
    x_array = np.asarray(x, dtype=np.float64)
    background_array = np.asarray(background, dtype=np.float64)
    weight_array = np.asarray(weight, dtype=np.float64).reshape(-1)
    bias_value = float(np.asarray(bias, dtype=np.float64).reshape(-1)[0])
    if x_array.ndim != 2 or background_array.ndim != 2:
        raise ValueError("x and background must be two-dimensional.")
    if x_array.shape[1] != background_array.shape[1] or x_array.shape[1] != len(weight_array):
        raise ValueError("Affine inputs, donors and weights have incompatible shapes.")
    if not all(np.isfinite(value).all() for value in (x_array, background_array, weight_array)):
        raise ValueError("Affine reference inputs must be finite.")
    mean_background = background_array.mean(axis=0)
    base = float(mean_background @ weight_array + bias_value)
    prediction = x_array @ weight_array + bias_value
    phi = (x_array - mean_background) * weight_array
    return {"base": base, "prediction": prediction, "phi": phi}


def _state_hash(state: Mapping[str, Any]) -> str:
    if not isinstance(state, Mapping):
        raise TypeError("A model state must be a tensor mapping.")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, t.Tensor):
            raise TypeError("State keys must be strings and values must be tensors.")
        if value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError("Every raw model-state tensor must be CPU float64.")
        array = value.detach().numpy()
        if not np.isfinite(array).all():
            raise ValueError("Raw model-state tensors must be finite.")
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _validate_reference_model(model: t.nn.Module) -> list[t.nn.Module]:
    """Enforce the exact, deliberately narrow author-parity model boundary."""
    if type(model) is not t.nn.Sequential:
        raise TypeError("model must be an exact nn.Sequential instance.")
    layers = list(model)
    if not layers or type(layers[-1]) is not t.nn.Linear or layers[-1].out_features != 1:
        raise ValueError("model must end with an exact scalar-output nn.Linear layer.")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("Shared modules are unsupported by the reference boundary.")
    previous_width: int | None = None
    for index, layer in enumerate(layers):
        if type(layer) not in (t.nn.Linear, t.nn.ReLU):
            raise TypeError("Only exact nn.Linear and nn.ReLU modules are supported.")
        if type(layer) is t.nn.ReLU:
            if layer.inplace:
                raise ValueError("In-place ReLU modules are unsupported.")
            if index == 0 or index == len(layers) - 1:
                raise ValueError("A ReLU must occur between affine layers.")
        else:
            if previous_width is not None and layer.in_features != previous_width:
                raise ValueError("Adjacent layer dimensions are incompatible.")
            previous_width = layer.out_features
        for parameter in layer.parameters(recurse=False):
            if parameter.device.type != "cpu" or parameter.dtype != t.float64:
                raise TypeError("Every model parameter must be CPU float64.")
            if not bool(t.isfinite(parameter).all()):
                raise ValueError("Every model parameter must be finite.")
    return layers


def _model_from_item(item: Mapping[str, Any]) -> t.nn.Sequential:
    if not isinstance(item, Mapping) or item.get("kind") not in ("mlp", "ridge"):
        raise ValueError("Each fitted item must declare kind 'mlp' or 'ridge'.")
    state = item.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("Each fitted item must contain a raw tensor state.")
    if item["kind"] == "mlp":
        expected = {
            "0.weight": (HIDDEN, FEATURES),
            "0.bias": (HIDDEN,),
            "2.weight": (HIDDEN, HIDDEN),
            "2.bias": (HIDDEN,),
            "4.weight": (1, HIDDEN),
            "4.bias": (1,),
        }
        model = t.nn.Sequential(
            t.nn.Linear(FEATURES, HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(HIDDEN, HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(HIDDEN, 1, dtype=t.float64),
        )
    else:
        expected = {"0.weight": (1, FEATURES), "0.bias": (1,)}
        model = t.nn.Sequential(t.nn.Linear(FEATURES, 1, dtype=t.float64))
    if set(state) != set(expected):
        raise ValueError("Raw model state has unexpected parameter names.")
    copied: dict[str, t.Tensor] = {}
    for name, shape in expected.items():
        value = state[name]
        if not isinstance(value, t.Tensor) or value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError("Raw state values must be CPU float64 tensors.")
        if tuple(value.shape) != shape or not bool(t.isfinite(value).all()):
            raise ValueError(f"Raw state tensor {name!r} has the wrong shape or values.")
        copied[name] = value.detach().clone()
    model.load_state_dict(copied, strict=True)
    model.eval()
    _validate_reference_model(model)
    return model


def _model_hash(model: t.nn.Module) -> str:
    return _state_hash(model.state_dict())


def _reference_environment() -> dict[str, Any]:
    """Verify installed SHAP files and every cached primary reference by hash."""
    import shap
    import sklearn
    from shap.explainers import _deep

    manifest_path = HERE / "author_reference.json"
    manifest = json.loads(manifest_path.read_text())
    actual_versions = {
        "numpy": np.__version__,
        "torch": t.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
    }
    expected_versions = {
        "numpy": "1.26.4",
        "torch": "2.12.1+cu132",
        "scikit_learn": "1.9.0",
        "shap": "0.49.1",
    }
    if actual_versions != expected_versions or manifest.get("version") != "0.49.1":
        raise ValueError(f"Reference version mismatch: {actual_versions}.")
    expected_source_keys = {
        "shap/explainers/_deep/__init__.py",
        "shap/explainers/_deep/deep_pytorch.py",
        "shap/explainers/_deep/deep_utils.py",
        "LICENSE",
    }
    if set(manifest.get("sources", {})) != expected_source_keys:
        raise ValueError("The SHAP author-source manifest has unexpected entries.")
    package_root = Path(shap.__file__).resolve().parent
    source_hashes: dict[str, str] = {}
    installed_hashes: dict[str, str] = {}
    for name, record in manifest["sources"].items():
        cached = Path(record["cache_path"])
        digest = _sha256(cached)
        if digest != record["sha256"]:
            raise ValueError(f"Cached author source changed for {name}.")
        source_hashes[name] = digest
        if name != "LICENSE":
            installed = package_root / record["installed_relative_path"]
            installed_digest = _sha256(installed)
            if installed_digest != record["installed_sha256"] or installed_digest != digest:
                raise ValueError(f"Installed SHAP source is not the pinned v0.49.1 file: {name}.")
            installed_hashes[name] = installed_digest
    paper_hashes: dict[str, str] = {}
    for record in manifest.get("papers", []):
        digest = _sha256(Path(record["pdf_cache_path"]))
        if digest != record["pdf_sha256"]:
            raise ValueError(f"Cached paper changed: {record['title']}.")
        paper_hashes[record["title"]] = digest
    historical_hashes: dict[str, str] = {}
    historical = manifest.get("historical_deeplift", {})
    for name, record in historical.get("sources", {}).items():
        digest = _sha256(Path(record["cache_path"]))
        if digest != record["sha256"]:
            raise ValueError(f"Cached historical DeepLIFT source changed for {name}.")
        historical_hashes[name] = digest
    if not paper_hashes or not historical_hashes:
        raise ValueError("Primary-paper and historical DeepLIFT evidence must both be pinned.")
    return {
        "versions": actual_versions,
        "author_manifest_sha256": _sha256(manifest_path),
        "cached_source_sha256": source_hashes,
        "installed_source_sha256": installed_hashes,
        "paper_pdf_sha256": paper_hashes,
        "historical_deeplift_commit": historical.get("commit"),
        "historical_source_sha256": historical_hashes,
        "deep_module": str(Path(_deep.__file__).resolve()),
    }


def _frozen_identity() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    data = source_data()
    ids = {
        name: np.asarray(manifest["splits"][name], dtype=np.int64)
        for name in ("train", "explain", "audit")
    }
    if data.shape != (1030, 9) or {name: len(value) for name, value in ids.items()} != {
        "train": 758,
        "explain": 153,
        "audit": 119,
    }:
        raise ValueError("The frozen Concrete source or split sizes changed.")
    return data, ids


def _background_indices(train_ids: np.ndarray) -> np.ndarray:
    return np.array(
        sorted(
            range(len(train_ids)),
            key=lambda index: hashlib.sha256(
                f"arena16.14/background/{PRIMARY_BACKGROUND_SEED}/{int(train_ids[index])}".encode()
            ).digest(),
        ),
        dtype=np.int64,
    )[:BACKGROUND_ROWS]


def _validate_rows(name: str, supplied: Mapping[str, Any], data: np.ndarray, ids: np.ndarray) -> None:
    if not isinstance(supplied, Mapping):
        raise TypeError(f"{name} must be a row mapping.")
    actual_ids = _int64_array(supplied.get("ids"), (len(ids),), f"{name}.ids")
    x = _float64_array(supplied.get("x"), (len(ids), FEATURES), f"{name}.x")
    y = _float64_array(supplied.get("y"), (len(ids),), f"{name}.y")
    if not np.array_equal(actual_ids, ids):
        raise ValueError(f"{name} row identity/order does not match the frozen manifest.")
    if not np.array_equal(x, data[ids, :FEATURES]) or not np.array_equal(y, data[ids, FEATURES]):
        raise ValueError(f"{name} values do not match their frozen source rows.")


def _validate_identity(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    data, ids = _frozen_identity()
    for name, supplied in (("train", train), ("explain", explain), ("audit", audit)):
        _validate_rows(name, supplied, data, ids[name])
    fitted_ids = _int64_array(fitted.get("train_ids"), (758,), "fitted.train_ids")
    fitted_x = _float64_array(fitted.get("train_x_raw"), (758, FEATURES), "fitted.train_x_raw")
    fitted_y = _float64_array(fitted.get("train_y_raw"), (758,), "fitted.train_y_raw")
    if not np.array_equal(fitted_ids, ids["train"]):
        raise ValueError("Fitted training row identity/order changed.")
    if not np.array_equal(fitted_x, data[ids["train"], :FEATURES]) or not np.array_equal(
        fitted_y, data[ids["train"], FEATURES]
    ):
        raise ValueError("Fitted training values changed.")
    if not np.array_equal(
        _int64_array(prepared.get("explain_ids"), (153,), "prepared.explain_ids"), ids["explain"]
    ):
        raise ValueError("Prepared explanation row identity/order changed.")
    if not np.array_equal(
        _float64_array(prepared.get("explain_x_raw"), (153, FEATURES), "prepared.explain_x_raw"),
        data[ids["explain"], :FEATURES],
    ):
        raise ValueError("Prepared explanation values changed.")
    if not np.array_equal(
        _float64_array(prepared.get("explain_targets"), (153,), "prepared.explain_targets"),
        data[ids["explain"], FEATURES],
    ):
        raise ValueError("Prepared explanation targets changed.")
    if not np.array_equal(
        _int64_array(evaluated.get("audit_ids"), (119,), "evaluated.audit_ids"), ids["audit"]
    ):
        raise ValueError("Evaluated audit row identity/order changed.")
    if not np.array_equal(
        _float64_array(evaluated.get("audit_targets"), (119,), "evaluated.audit_targets"),
        data[ids["audit"], FEATURES],
    ):
        raise ValueError("Evaluated audit targets changed.")
    background_indices = _background_indices(ids["train"])
    background_ids = ids["train"][background_indices]
    background = data[background_ids, :FEATURES]
    actual_ids = _int64_array(prepared.get("background_ids"), (BACKGROUND_ROWS,), "background_ids")
    actual_background = _float64_array(
        prepared.get("background"), (BACKGROUND_ROWS, FEATURES), "background"
    )
    if not np.array_equal(actual_ids, background_ids):
        raise ValueError("Primary background IDs/order are not the frozen 64 donors.")
    if not np.array_equal(actual_background, background):
        raise ValueError("Primary background values are not all 64 frozen float64 donors.")
    supplied_indices = prepared.get("background_indices", {}).get("primary64")
    if not np.array_equal(
        _int64_array(supplied_indices, (BACKGROUND_ROWS,), "background_indices.primary64"),
        background_indices,
    ):
        raise ValueError("Primary background positions changed.")
    primary = evaluated.get("background_variants", {}).get("primary64", {})
    if not np.array_equal(
        _int64_array(primary.get("background_ids"), (BACKGROUND_ROWS,), "primary64.background_ids"),
        background_ids,
    ) or not np.array_equal(
        _float64_array(primary.get("background"), (BACKGROUND_ROWS, FEATURES), "primary64.background"),
        background,
    ):
        raise ValueError("Evaluated primary background changed.")
    combined_ids = np.concatenate([ids["explain"], ids["audit"]])
    combined_split = np.concatenate(
        [np.zeros(153, dtype=np.int64), np.ones(119, dtype=np.int64)]
    )
    for owner, supplied_ids, supplied_split in (
        (
            "evaluated",
            evaluated.get("combined_explanation_audit_ids"),
            evaluated.get("combined_split"),
        ),
        ("primary64", primary.get("row_ids"), primary.get("row_split")),
    ):
        if not np.array_equal(
            _int64_array(supplied_ids, (272,), f"{owner}.combined_ids"), combined_ids
        ) or not np.array_equal(
            _int64_array(supplied_split, (272,), f"{owner}.combined_split"), combined_split
        ):
            raise ValueError(f"{owner} combined row identity/order changed.")
    return {
        "train_ids": ids["train"].copy(),
        "explain_ids": ids["explain"].copy(),
        "audit_ids": ids["audit"].copy(),
        "background_indices": background_indices,
        "background_ids": background_ids,
        "background": background.copy(),
        "explain_x": data[ids["explain"], :FEATURES].copy(),
        "audit_x": data[ids["audit"], :FEATURES].copy(),
    }


def _validate_block(name: str, block: Any, rows: int) -> dict[str, dict[str, np.ndarray | float]]:
    if not isinstance(block, Mapping) or tuple(block) != MODEL_KEYS:
        raise ValueError(f"{name} must contain the seven models in frozen order.")
    shapes = {
        "prediction": (rows,),
        "base": (),
        "phi_deep": (rows, FEATURES),
        "phi_exact": (rows, FEATURES),
        "phi_gradient": (rows, FEATURES),
        "game": (rows, 2**FEATURES),
        "pair_deep": (rows, BACKGROUND_ROWS, FEATURES),
    }
    result: dict[str, dict[str, np.ndarray | float]] = {}
    for key in MODEL_KEYS:
        item = block[key]
        if not isinstance(item, Mapping):
            raise TypeError(f"{name}[{key!r}] must be a mapping.")
        converted: dict[str, np.ndarray | float] = {}
        for field, shape in shapes.items():
            array = _float64_array(item.get(field), shape, f"{name}[{key}].{field}")
            converted[field] = float(array) if shape == () else array
        result[key] = converted
    return result


def _author_deep(
    model: t.nn.Sequential, x: np.ndarray, background: np.ndarray
) -> dict[str, np.ndarray | float | str]:
    """Call source-pinned SHAP explicitly, retaining complete local vectors."""
    import shap

    _validate_reference_model(model)
    x_tensor = t.as_tensor(np.array(x, copy=True), dtype=t.float64)
    background_tensor = t.as_tensor(np.array(background, copy=True), dtype=t.float64)
    state_before = _model_hash(model)
    training_before = bool(model.training)
    explainer = shap.DeepExplainer(model, background_tensor)
    backend_data = getattr(explainer.explainer, "data", None)
    if not isinstance(backend_data, list) or len(backend_data) != 1:
        raise AssertionError("DeepExplainer did not retain one explicit background tensor.")
    if backend_data[0].shape != background_tensor.shape or not t.equal(
        backend_data[0], background_tensor
    ):
        raise AssertionError("DeepExplainer changed, summarized or reordered the background.")
    raw_phi = explainer.shap_values(x_tensor, check_additivity=False)
    if isinstance(raw_phi, list):
        raise AssertionError("The scalar-output reference unexpectedly returned multiple outputs.")
    phi = np.asarray(raw_phi, dtype=np.float64)
    if phi.shape == (len(x), x.shape[1], 1):
        phi = phi[..., 0]
    if phi.shape != x.shape or not np.isfinite(phi).all():
        raise AssertionError("DeepExplainer returned an unexpected local-value array.")
    expected = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    if expected.shape != (1,) or not np.isfinite(expected).all():
        raise AssertionError("DeepExplainer returned an unexpected expected value.")
    with t.no_grad():
        prediction = model(x_tensor).squeeze(-1).detach().numpy().copy()
        direct_base = float(model(background_tensor).squeeze(-1).mean())
    state_after = _model_hash(model)
    if state_after != state_before or bool(model.training) != training_before:
        raise AssertionError("SHAP DeepExplainer changed model parameters or training state.")
    base = float(expected[0])
    _require_close("DeepExplainer expected base", np.array(base), np.array(direct_base))
    return {
        "prediction": prediction,
        "base": base,
        "phi": phi.copy(),
        "local_accuracy_residual": prediction - base - phi.sum(axis=1),
        "model_state_sha256": state_after,
    }


def _bounded_pair_author(
    model: t.nn.Sequential, x: np.ndarray, background: np.ndarray
) -> dict[str, np.ndarray | float]:
    if x.shape != (1, background.shape[1]):
        raise ValueError("The bounded donor check requires exactly one query row.")
    pair_phi = np.empty((len(background), background.shape[1]), dtype=np.float64)
    pair_base = np.empty(len(background), dtype=np.float64)
    pair_residual = np.empty(len(background), dtype=np.float64)
    prediction: float | None = None
    for donor_index in range(len(background)):
        result = _author_deep(model, x, background[donor_index : donor_index + 1])
        pair_phi[donor_index] = np.asarray(result["phi"])[0]
        pair_base[donor_index] = float(result["base"])
        pair_residual[donor_index] = np.asarray(result["local_accuracy_residual"])[0]
        current_prediction = float(np.asarray(result["prediction"])[0])
        if prediction is None:
            prediction = current_prediction
        elif current_prediction != prediction:
            raise AssertionError("The query prediction changed across single-donor explainers.")
    return {
        "prediction": float(prediction),
        "base": pair_base,
        "phi": pair_phi,
        "local_accuracy_residual": pair_residual,
    }


def _linear_model(weight: list[list[float]], bias: list[float]) -> t.nn.Sequential:
    model = t.nn.Sequential(t.nn.Linear(len(weight[0]), len(weight), dtype=t.float64))
    with t.no_grad():
        model[0].weight.copy_(t.tensor(weight, dtype=t.float64))
        model[0].bias.copy_(t.tensor(bias, dtype=t.float64))
    model.eval()
    return model


def _relu_model(
    first_weight: list[list[float]], first_bias: list[float], final_weight: list[float]
) -> t.nn.Sequential:
    model = t.nn.Sequential(
        t.nn.Linear(len(first_weight[0]), len(first_weight), dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(len(first_weight), 1, dtype=t.float64),
    )
    with t.no_grad():
        model[0].weight.copy_(t.tensor(first_weight, dtype=t.float64))
        model[0].bias.copy_(t.tensor(first_bias, dtype=t.float64))
        model[2].weight.copy_(t.tensor([final_weight], dtype=t.float64))
        model[2].bias.zero_()
    model.eval()
    return model


def _coalition_game(model: t.nn.Sequential, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    features = x.shape[1]
    masks = np.array(
        [[bool(mask & (1 << feature)) for feature in range(features)] for mask in range(2**features)]
    )
    values = np.empty((len(x), len(masks)), dtype=np.float64)
    with t.no_grad():
        for row_index, row in enumerate(x):
            hybrids = np.where(masks[:, None, :], row[None, None, :], background[None, :, :])
            prediction = model(t.as_tensor(hybrids.reshape(-1, features), dtype=t.float64))
            values[row_index] = prediction.reshape(len(masks), len(background)).mean(1).numpy()
    return values


def known_diagnostics() -> dict[str, Any]:
    """Run the predeclared hand cases through independent algebra and SHAP."""
    threshold = _relu_model([[2.0, 1.0]], [-2.0], [1.0])
    one = np.array([[1.0, 1.0]], dtype=np.float64)
    zero = np.array([[0.0, 0.0]], dtype=np.float64)
    threshold_game = _coalition_game(threshold, one, zero)
    threshold_exact, threshold_permutations = permutation_reference(threshold_game)
    threshold_author = _author_deep(threshold, one, zero)
    _require_close("threshold coalition table", threshold_game, [[0.0, 0.0, 0.0, 1.0]])
    _require_close("threshold exact Shapley", threshold_exact, [[0.5, 0.5]])
    _require_close("threshold author DeepSHAP", threshold_author["phi"], [[2 / 3, 1 / 3]])

    minimum = _relu_model([[0.0, 1.0], [-1.0, 1.0]], [0.0, 0.0], [1.0, -1.0])
    mirrored = _relu_model([[1.0, 0.0], [1.0, -1.0]], [0.0, 0.0], [1.0, -1.0])
    min_game = _coalition_game(minimum, one, zero)
    mirrored_game = _coalition_game(mirrored, one, zero)
    min_exact, min_permutations = permutation_reference(min_game)
    mirrored_exact, mirrored_permutations = permutation_reference(mirrored_game)
    min_author = _author_deep(minimum, one, zero)
    mirrored_author = _author_deep(mirrored, one, zero)
    binary_inputs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    with t.no_grad():
        min_table = minimum(t.as_tensor(binary_inputs, dtype=t.float64)).squeeze(-1).numpy()
        mirrored_table = mirrored(t.as_tensor(binary_inputs, dtype=t.float64)).squeeze(-1).numpy()
    expected_min_table = binary_inputs.min(axis=1)
    _require_close("minimum binary domain", min_table, expected_min_table)
    _require_close("mirrored minimum binary domain", mirrored_table, expected_min_table)
    _require_close("minimum exact Shapley", min_exact, [[0.5, 0.5]])
    _require_close("mirrored exact Shapley", mirrored_exact, [[0.5, 0.5]])
    _require_close("minimum author DeepSHAP", min_author["phi"], [[0.0, 1.0]])
    _require_close("mirrored author DeepSHAP", mirrored_author["phi"], [[1.0, 0.0]])

    nearzero = _relu_model([[1.0]], [0.0], [1.0])
    nearzero_x = np.array([[4e-7]], dtype=np.float64)
    nearzero_background = np.array([[-4e-7]], dtype=np.float64)
    nearzero_author = _author_deep(nearzero, nearzero_x, nearzero_background)
    nearzero_delta = np.array([4e-7], dtype=np.float64)
    _require_close("near-zero guarded author value", nearzero_author["phi"], [[0.0]])
    nearzero_passed = bool(
        np.allclose(
            np.asarray(nearzero_author["base"]) + np.asarray(nearzero_author["phi"]).sum(1),
            np.asarray(nearzero_author["prediction"]),
            atol=AUTHOR_ATOL,
            rtol=AUTHOR_RTOL,
        )
    )
    if nearzero_passed:
        raise AssertionError("The predeclared near-zero conservation failure disappeared.")

    affine = _linear_model([[2.0, -3.0]], [5.0])
    affine_x = np.array([[4.0, 2.0]], dtype=np.float64)
    affine_background = np.array([[1.0, -1.0], [2.0, 0.0]], dtype=np.float64)
    affine_analytic = ridge_closed_form([[2.0, -3.0]], [5.0], affine_x, affine_background)
    affine_author = _author_deep(affine, affine_x, affine_background)
    affine_game = _coalition_game(affine, affine_x, affine_background)
    affine_exact, affine_permutations = permutation_reference(affine_game)
    _require_close("affine base", np.array(affine_analytic["base"]), np.array(9.5))
    _require_close("affine contribution", affine_analytic["phi"], [[5.0, -7.5]])
    _require_close("affine author", affine_author["phi"], affine_analytic["phi"])
    _require_close("affine exact", affine_exact, affine_analytic["phi"])

    background_relu = _relu_model([[1.0, 1.0]], [0.0], [1.0])
    background_x = np.array([[2.0, 1.0]], dtype=np.float64)
    background_two = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    background_author = _author_deep(background_relu, background_x, background_two)
    background_game = _coalition_game(background_relu, background_x, background_two)
    background_exact, background_permutations = permutation_reference(background_game)
    _require_close("two-background author", background_author["phi"], [[1.5, 1.5]])
    _require_close("two-background exact", background_exact, [[1.5, 1.5]])

    constant = _linear_model([[0.0, 0.0]], [3.0])
    constant_author = _author_deep(constant, affine_x, affine_background)
    _require_close("constant author", constant_author["phi"], [[0.0, 0.0]])
    dummy = _linear_model([[2.0, -3.0, 0.0]], [5.0])
    dummy_x = np.array([[4.0, 2.0, 99.0]], dtype=np.float64)
    dummy_background = np.array([[1.0, -1.0, -10.0], [2.0, 0.0, 30.0]], dtype=np.float64)
    dummy_author = _author_deep(dummy, dummy_x, dummy_background)
    dummy_game = _coalition_game(dummy, dummy_x, dummy_background)
    dummy_exact, dummy_permutations = permutation_reference(dummy_game)
    _require_close("dummy author", dummy_author["phi"][:, 2], [0.0])
    _require_close("dummy exact", dummy_exact[:, 2], [0.0])

    return {
        "threshold": {
            "game": threshold_game,
            "exact": threshold_exact,
            "deep_explainer": np.asarray(threshold_author["phi"]),
            "single_feature_necessity": np.array([[1.0, 1.0]]),
            "permutation_count": threshold_permutations,
        },
        "minimum": {
            "binary_inputs": binary_inputs,
            "minimum_outputs": min_table,
            "mirrored_outputs": mirrored_table,
            "game": min_game,
            "mirrored_game": mirrored_game,
            "exact": min_exact,
            "mirrored_exact": mirrored_exact,
            "deep_explainer": np.asarray(min_author["phi"]),
            "mirrored_deep_explainer": np.asarray(mirrored_author["phi"]),
            "representations_equal_on_binary_domain": True,
            "permutation_counts": np.array([min_permutations, mirrored_permutations]),
        },
        "nearzero": {
            "x": nearzero_x,
            "background": nearzero_background,
            "prediction_delta": nearzero_delta,
            "deep_explainer": np.asarray(nearzero_author["phi"]),
            "epsilon_zero_secant": nearzero_delta[:, None],
            "local_accuracy_residual": np.asarray(nearzero_author["local_accuracy_residual"]),
            "conservation_passed_at_primary_tolerance": nearzero_passed,
            "expected_failure_disclosed": not nearzero_passed,
        },
        "affine": {
            "base": float(affine_analytic["base"]),
            "prediction": np.asarray(affine_analytic["prediction"]),
            "analytic": np.asarray(affine_analytic["phi"]),
            "deep_explainer": np.asarray(affine_author["phi"]),
            "exact": affine_exact,
            "game": affine_game,
            "permutation_count": affine_permutations,
        },
        "two_background_relu": {
            "game": background_game,
            "deep_explainer": np.asarray(background_author["phi"]),
            "exact": background_exact,
            "permutation_count": background_permutations,
        },
        "constant": {
            "deep_explainer": np.asarray(constant_author["phi"]),
            "local_accuracy_residual": np.asarray(constant_author["local_accuracy_residual"]),
        },
        "dummy": {
            "game": dummy_game,
            "deep_explainer": np.asarray(dummy_author["phi"]),
            "exact": dummy_exact,
            "permutation_count": dummy_permutations,
        },
    }


def validate_reference(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return complete independent author/algebraic references.

    Author parity covers every averaged local vector.  Donor-level author
    parity is separately and accurately labelled as the two predeclared rows
    (first explanation and first audit row) for every model and all 64 donors.
    Neural DeepSHAP-versus-exact differences are returned as measurements and
    are never used as an acceptance threshold.
    """
    environment = _reference_environment()
    if not isinstance(fitted, Mapping) or fitted.get("phase") != "models_fitted_on_train_only":
        raise ValueError("Use the complete train-only fitted payload.")
    if not isinstance(fitted.get("models"), Mapping) or tuple(fitted["models"]) != MODEL_KEYS:
        raise ValueError("The fitted payload must retain all seven models in frozen order.")
    if prepared.get("phase") != "rankings_frozen_before_audit" or evaluated.get(
        "phase"
    ) != "audit_evaluated_after_frozen_rankings":
        raise ValueError("Use the prepared and evaluated payloads from the frozen study phases.")
    if prepared.get("audit_used_for_selection") is not False or evaluated.get(
        "audit_used_for_selection"
    ) is not False:
        raise ValueError("Audit rows must not select models or rankings.")
    identity = _validate_identity(fitted, prepared, evaluated, train, explain, audit)
    explanation = _validate_block("prepared.models", prepared.get("models"), 153)
    audit_block = _validate_block("evaluated.models", evaluated.get("models"), 119)
    background = identity["background"]
    combined_x = np.concatenate([identity["explain_x"], identity["audit_x"]], axis=0)
    split_slices = {"explanation": slice(0, 153), "audit": slice(153, 272)}
    stored_blocks = {"explanation": explanation, "audit": audit_block}
    row_ids = {"explanation": identity["explain_ids"], "audit": identity["audit_ids"]}

    references: dict[str, dict[str, Any]] = {"explanation": {}, "audit": {}}
    bounded_pairs: dict[str, dict[str, Any]] = {"explanation": {}, "audit": {}}
    descriptive: dict[str, dict[str, Any]] = {"explanation": {}, "audit": {}}
    per_model_errors: dict[str, dict[str, float]] = {}
    ridge_reference: dict[str, Any] = {}
    state_hashes: dict[str, str] = {}
    maxima: dict[str, float] = {
        "prediction": 0.0,
        "base": 0.0,
        "author_phi_deep": 0.0,
        "author_local_accuracy": 0.0,
        "own_local_accuracy": 0.0,
        "own_pair_mean": 0.0,
        "bounded_pair_phi": 0.0,
        "bounded_pair_base": 0.0,
        "bounded_pair_local_accuracy": 0.0,
        "bounded_pair_prediction": 0.0,
        "bounded_pair_average": 0.0,
        "exact_phi": 0.0,
        "exact_efficiency": 0.0,
        "ridge_analytic_prediction": 0.0,
        "ridge_analytic_base": 0.0,
        "ridge_analytic_phi": 0.0,
    }

    for key in MODEL_KEYS:
        item = fitted["models"][key]
        model = _model_from_item(item)
        item_state_before = _state_hash(item["state"])
        state_hashes[key] = item_state_before
        author = _author_deep(model, combined_x, background)
        if author["model_state_sha256"] != _model_hash(model):
            raise AssertionError("Author reference changed a reconstructed model state.")
        model_errors = {name: 0.0 for name in maxima}

        for split, row_slice in split_slices.items():
            stored = stored_blocks[split][key]
            author_prediction = np.asarray(author["prediction"])[row_slice].copy()
            author_phi = np.asarray(author["phi"])[row_slice].copy()
            author_residual = np.asarray(author["local_accuracy_residual"])[row_slice].copy()
            stored_prediction = np.asarray(stored["prediction"])
            stored_base = float(stored["base"])
            stored_phi = np.asarray(stored["phi_deep"])
            stored_pair = np.asarray(stored["pair_deep"])
            game = np.asarray(stored["game"])
            exact = subset_reference(game)
            exact_residual = game[:, -1] - game[:, 0] - exact.sum(axis=1)
            own_residual = stored_prediction - stored_base - stored_phi.sum(axis=1)

            errors = {
                "prediction": _require_close(
                    f"{split} {key} raw prediction", author_prediction, stored_prediction
                ),
                "base": _require_close(
                    f"{split} {key} expected base",
                    np.array(author["base"]),
                    np.array(stored_base),
                ),
                "author_phi_deep": _require_close(
                    f"{split} {key} author DeepSHAP",
                    author_phi,
                    stored_phi,
                    atol=AUTHOR_ATOL,
                    rtol=AUTHOR_RTOL,
                ),
                "author_local_accuracy": _require_close(
                    f"{split} {key} author local accuracy",
                    float(author["base"]) + author_phi.sum(axis=1),
                    author_prediction,
                    atol=AUTHOR_ATOL,
                    rtol=AUTHOR_RTOL,
                ),
                "own_local_accuracy": _require_close(
                    f"{split} {key} learner local accuracy",
                    stored_base + stored_phi.sum(axis=1),
                    stored_prediction,
                    atol=AUTHOR_ATOL,
                    rtol=AUTHOR_RTOL,
                ),
                "own_pair_mean": _require_close(
                    f"{split} {key} complete pair mean", stored_pair.mean(axis=1), stored_phi
                ),
                "exact_phi": _require_close(
                    f"{split} {key} exact coalition contraction", exact, stored["phi_exact"]
                ),
                "exact_efficiency": _require_close(
                    f"{split} {key} exact-game efficiency",
                    game[:, 0] + exact.sum(axis=1),
                    game[:, -1],
                ),
            }
            _require_close(f"{split} {key} empty coalition", game[:, 0], np.full(len(game), stored_base))
            _require_close(f"{split} {key} full coalition", game[:, -1], stored_prediction)

            reference_item = {
                "prediction": author_prediction,
                "base_deep_explainer": float(author["base"]),
                "phi_deep_explainer": author_phi,
                "local_accuracy_residual": author_residual,
                "phi_exact_independent": exact,
                "exact_efficiency_residual": exact_residual,
                "deep_minus_exact": author_phi - exact,
            }
            references[split][key] = reference_item
            difference = author_phi - exact
            descriptive[split][key] = {
                "deep_minus_exact": difference.copy(),
                "max_abs": float(np.abs(difference).max(initial=0.0)),
                "mean_abs": float(np.abs(difference).mean()),
                "rmse": float(np.sqrt(np.square(difference).mean())),
                "is_acceptance_gate": False,
            }

            pair = _bounded_pair_author(
                model,
                (identity["explain_x"] if split == "explanation" else identity["audit_x"])[0:1],
                background,
            )
            pair_errors = {
                "bounded_pair_phi": _require_close(
                    f"{split} {key} bounded donor-level DeepSHAP",
                    pair["phi"],
                    stored_pair[0],
                    atol=AUTHOR_ATOL,
                    rtol=AUTHOR_RTOL,
                ),
                "bounded_pair_base": _require_close(
                    f"{split} {key} bounded donor bases",
                    pair["base"],
                    model(t.as_tensor(background, dtype=t.float64)).detach().squeeze(-1).numpy(),
                ),
                "bounded_pair_local_accuracy": _require_close(
                    f"{split} {key} bounded donor local accuracy",
                    np.asarray(pair["base"]) + np.asarray(pair["phi"]).sum(axis=1),
                    np.full(BACKGROUND_ROWS, float(pair["prediction"])),
                    atol=AUTHOR_ATOL,
                    rtol=AUTHOR_RTOL,
                ),
                "bounded_pair_prediction": _require_close(
                    f"{split} {key} bounded query prediction",
                    np.array(pair["prediction"]),
                    np.array(stored_prediction[0]),
                ),
                "bounded_pair_average": max(
                    _require_close(
                        f"{split} {key} bounded-pair attribution average",
                        np.asarray(pair["phi"]).mean(axis=0),
                        author_phi[0],
                        atol=AUTHOR_ATOL,
                        rtol=AUTHOR_RTOL,
                    ),
                    _require_close(
                        f"{split} {key} bounded-pair base average",
                        np.array(np.asarray(pair["base"]).mean()),
                        np.array(author["base"]),
                    ),
                ),
            }
            bounded_pairs[split][key] = {
                "row_index": 0,
                "source_id": int(row_ids[split][0]),
                "background_ids": identity["background_ids"].copy(),
                "prediction": float(pair["prediction"]),
                "base_per_reference": np.asarray(pair["base"]).copy(),
                "phi_deep_explainer_per_reference": np.asarray(pair["phi"]).copy(),
                "local_accuracy_residual_per_reference": np.asarray(
                    pair["local_accuracy_residual"]
                ).copy(),
                "scope": "first row only; all 64 donors in frozen source-ID order",
            }
            errors.update(pair_errors)
            for name, error in errors.items():
                model_errors[name] = max(model_errors[name], error)
                maxima[name] = max(maxima[name], error)

            if key == "ridge":
                state = item["state"]
                analytic = ridge_closed_form(
                    _array(state["0.weight"]),
                    _array(state["0.bias"]),
                    identity["explain_x"] if split == "explanation" else identity["audit_x"],
                    background,
                )
                ridge_errors = {
                    "ridge_analytic_prediction": _require_close(
                        f"{split} ridge analytic prediction", analytic["prediction"], stored_prediction
                    ),
                    "ridge_analytic_base": _require_close(
                        f"{split} ridge analytic base",
                        np.array(analytic["base"]),
                        np.array(stored_base),
                    ),
                    "ridge_analytic_phi": max(
                        _require_close(
                            f"{split} ridge analytic learner DeepSHAP", analytic["phi"], stored_phi
                        ),
                        _require_close(
                            f"{split} ridge analytic exact Shapley", analytic["phi"], exact
                        ),
                        _require_close(
                            f"{split} ridge analytic author DeepSHAP", analytic["phi"], author_phi
                        ),
                    ),
                }
                ridge_reference[split] = {
                    "base": float(analytic["base"]),
                    "prediction": np.asarray(analytic["prediction"]).copy(),
                    "phi": np.asarray(analytic["phi"]).copy(),
                }
                for name, error in ridge_errors.items():
                    model_errors[name] = max(model_errors[name], error)
                    maxima[name] = max(maxima[name], error)

        if _state_hash(item["state"]) != item_state_before or _model_hash(model) != item_state_before:
            raise AssertionError(f"Reference evaluation changed the raw state for {key}.")
        per_model_errors[key] = model_errors

    diagnostics = known_diagnostics()
    passes = {
        "row_identity": True,
        "background_identity_all_64": True,
        "author_deep_explainer_all_model_rows": True,
        "expected_base": True,
        "bounded_pair_author": True,
        "own_full_pair_shape_finite_and_mean": True,
        "exact_game_contraction": True,
        "ridge_analytic": True,
        "model_states_unchanged": True,
        "tiny_threshold": True,
        "tiny_min_equivalent_representations": True,
        "nearzero_expected_failure_disclosed": bool(
            diagnostics["nearzero"]["expected_failure_disclosed"]
        ),
        "reference_model_boundaries": True,
    }
    if not all(passes.values()):
        raise AssertionError("An independent reference acceptance check failed.")
    return {
        "status": "PASS",
        "summary": {
            "author_averaged_vector_parity_passed": True,
            "author_bounded_pair_parity_passed": True,
            "exact_game_contraction_passed": True,
            "ridge_analytic_parity_passed": True,
            "neural_deep_vs_exact_is_descriptive": True,
            "nearzero_expected_failure_disclosed": True,
        },
        "passes": passes,
        "tolerances": {
            "author_deep_explainer": {"atol": AUTHOR_ATOL, "rtol": AUTHOR_RTOL},
            "exact_algebraic": {"atol": EXACT_ATOL, "rtol": EXACT_RTOL},
        },
        "counts": {
            "models": 7,
            "explanation_rows": 153,
            "audit_rows": 119,
            "model_rows": 1904,
            "background_rows": 64,
            "local_values": 15232,
            "author_attribution_coordinates": 15232,
            "own_pair_model_row_backgrounds": 121856,
            "own_pair_coordinates": 974848,
            "bounded_author_pair_checks": 896,
            "bounded_author_pair_coordinates": 7168,
            "pair_model_rows": 896,
            "pair_local_values": 7168,
            "exact_game_model_rows": 1904,
            "coalition_values": 487424,
        },
        "model_keys": MODEL_KEYS,
        "units": {"prediction": "MPa", "base": "MPa", "attribution": "MPa"},
        "identity": identity,
        "explanation": references["explanation"],
        "audit": references["audit"],
        "bounded_pair_reference": bounded_pairs,
        "max_errors": maxima,
        "per_model_max_errors": per_model_errors,
        "deep_vs_exact": {
            "role": "descriptive approximation comparison; not an author-parity gate",
            "is_acceptance_gate": False,
            **descriptive,
        },
        "ridge_analytic": ridge_reference,
        "diagnostics": diagnostics,
        "model_state_sha256": state_hashes,
        "reference_environment": environment,
    }
