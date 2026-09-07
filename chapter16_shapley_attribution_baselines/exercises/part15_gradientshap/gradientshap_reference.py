"""Pinned, state-restoring SHAP and Captum references for Expected Gradients.

Package parity in this module always means parity on the package's *actual*
donor, interpolation, and noise draws.  SHAP 0.49.1 and Captum 0.9.0 use
different random-call orders, so equal integer seeds do not define equal draw
tables.  The adapters reconstruct each table independently and replay it
through the visible student methods.

The native-study entry point, :func:`validate_reference`, is intentionally
narrow: seven already-reconstructed CPU-float64 scalar models, all 272 ordered
held-out rows, all 64 primary donors, and K=128.  It does not fit a model, load
Concrete data, or choose an interesting subset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import contextlib
import copy
import hashlib
import functools
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch as t
from torch import Tensor, nn


MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
FEATURES = 8
ALL_ROWS = 272
BACKGROUND_ROWS = 64
REFERENCE_SAMPLES = 128
SHAP_NUMPY_SEED = 161513
SHAP_TORCH_SEED = 161515
CAPTUM_NUMPY_SEED = 161514
CAPTUM_TORCH_SEED = 161515
AUTHOR_ATOL = 1e-10
AUTHOR_RTOL = 1e-10
FORWARD_BATCH = 32_768
SHAP_BATCH_SIZE = 128

REFERENCE_CACHE = Path.home() / ".cache/arena_extended/gradientshap/reference_recon"
EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "torch": "2.12.1+cu132",
    "shap": "0.49.1",
    "captum": "0.9.0",
}
EXPECTED_SOURCE_SHA256 = {
    "shap/explainers/_gradient.py": "ae1d0fe9405e62c9400c655e6d5af792ccc989a7632fd05171a731776551840a",
    "captum/attr/_core/gradient_shap.py": "b2333e8819b82eb18693ae5a8a78c7a9ff1e30986179de2ee97b2820dcbff00c",
    "captum/attr/_core/noise_tunnel.py": "d013c38cdbb4a10d4717a4e38ac063903cacc0b54389a6591a9bcbc4a9fa8606",
    "captum/_utils/common.py": "a1fc8b6752c48a76ffd3305983fed7972c7013ae6764ddb837c873c8eaa4198e",
}

SHAP_WARNINGS = (
    "SHAP 0.49.1 draws one donor then one alpha for each sample and query row; row order is part of the draw identity.",
    "SHAP rseed controls its legacy NumPy stream but does not seed Torch local-smoothing noise.",
    "The scalar-output raw attribution shape is (N,D,1); author_phi is the normalized (N,D) view.",
    "author_shap_variance_proxy is samples.var(ddof=0)/sqrt(K), neither ordinary variance-of-mean nor standard error, and is excluded from parity acceptance.",
    "The adapter temporarily uses the input dtype as Torch's default to repair SHAP's implicit sample-buffer dtype and restores it afterward.",
)
CAPTUM_WARNINGS = (
    "Captum 0.9.0 draws all donors, then Torch noise, then all alphas in one full-query call.",
    "Splitting the 272 rows into public Captum calls would change the draw table.",
    "author_convergence_delta contains one delta per query/sample pair; it is neither estimator uncertainty nor the averaged-vector completeness residual.",
    "Captum's positive-noise stream is generated from a float32 stdev buffer under the pinned ordinary default-dtype convention.",
)

__all__ = [
    "MODEL_KEYS",
    "REFERENCE_SAMPLES",
    "AUTHOR_ATOL",
    "AUTHOR_RTOL",
    "reference_environment",
    "shap_draw_table",
    "captum_draw_table",
    "replay_draw_table",
    "shap_same_draw_reference",
    "captum_same_draw_reference",
    "positive_noise_reference",
    "reference_check",
    "validate_reference",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _array_sha256(value: Any) -> str:
    """Hash array dtype, shape, and complete C-order bytes."""
    array = np.ascontiguousarray(_numpy(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(_array_sha256(value).encode())
    return digest.hexdigest()


def _model_structure_sha256(model: nn.Module) -> str:
    description: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        item: dict[str, Any] = {"name": name, "type": type(module).__qualname__}
        if isinstance(module, nn.Linear):
            item.update(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        if isinstance(module, nn.ReLU):
            item["inplace"] = module.inplace
        description.append(item)
    return hashlib.sha256(
        json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _max_abs(actual: Any, expected: Any) -> float:
    actual_array = _numpy(actual).astype(np.float64, copy=False)
    expected_array = _numpy(expected).astype(np.float64, copy=False)
    if actual_array.shape != expected_array.shape:
        return math.inf
    return float(np.abs(actual_array - expected_array).max(initial=0.0))


def _close(actual: Any, expected: Any) -> bool:
    actual_array, expected_array = _numpy(actual), _numpy(expected)
    return bool(
        actual_array.shape == expected_array.shape
        and np.isfinite(actual_array).all()
        and np.isfinite(expected_array).all()
        and np.allclose(
            actual_array,
            expected_array,
            atol=AUTHOR_ATOL,
            rtol=AUTHOR_RTOL,
        )
    )


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _seed(name: str, value: int) -> int:
    if type(value) is not int or value < 0 or value >= 2**32:
        raise ValueError(f"{name} must be an integer in [0, 2**32).")
    return value


def _float64_tensor(name: str, value: Tensor, ndim: int) -> Tensor:
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if value.ndim != ndim or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional tensor.")
    if not bool(t.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _source_ids(value: Tensor | np.ndarray | None, rows: int) -> np.ndarray:
    if value is None:
        return np.arange(rows, dtype=np.int64)
    array = _numpy(value)
    if array.dtype != np.int64 or array.shape != (rows,):
        raise ValueError(f"source_ids must have int64 shape ({rows},).")
    if np.any(array < 0):
        raise ValueError("source_ids must be nonnegative.")
    if len(np.unique(array)) != rows:
        raise ValueError("source_ids must be unique so row order has an unambiguous identity.")
    return np.array(array, copy=True)


def _validate_model(model: nn.Module, features: int) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != t.float64:
            raise TypeError("Every model parameter must be a CPU float64 tensor.")
        if not bool(t.isfinite(parameter).all()):
            raise ValueError("Every model parameter must be finite.")
    for buffer in model.buffers():
        if buffer.device.type != "cpu":
            raise TypeError("Every model buffer must be on CPU.")
        if buffer.is_floating_point() and buffer.dtype != t.float64:
            raise TypeError("Every floating model buffer must be float64.")
        if buffer.is_floating_point() and not bool(t.isfinite(buffer).all()):
            raise ValueError("Every floating model buffer must be finite.")
    if features < 1:
        raise ValueError("features must be positive.")
    return model


def _model_output(model: nn.Module, inputs: Tensor) -> Tensor:
    output = model(inputs)
    if type(output) is not Tensor or output.shape != (len(inputs), 1):
        raise ValueError("Package reference models must return float64 shape (rows, 1).")
    if output.device.type != "cpu" or output.dtype != t.float64:
        raise TypeError("Package reference models must return CPU float64 values.")
    if not bool(t.isfinite(output).all()):
        raise ValueError("The model returned a nonfinite value.")
    return output[:, 0]


@contextlib.contextmanager
def _preserve_execution_state(model: nn.Module) -> Iterator[None]:
    """Restore all caller-visible execution state on success and exceptions."""
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = t.random.get_rng_state().clone()
    default_dtype = t.get_default_dtype()
    modes = [(module, module.training) for module in model.modules()]
    parameters = [
        (
            parameter,
            parameter.detach().clone(),
            parameter.requires_grad,
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in model.parameters()
    ]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        t.random.set_rng_state(torch_state)
        t.set_default_dtype(default_dtype)
        for module, training in modes:
            module.training = training
        with t.no_grad():
            for parameter, value, requires_grad, original_grad, gradient in parameters:
                parameter.copy_(value)
                parameter.requires_grad_(requires_grad)
                if original_grad is None:
                    parameter.grad = None
                else:
                    parameter.grad = original_grad
                    original_grad.copy_(gradient)
            for buffer, value in buffers:
                buffer.copy_(value)


def _method(methods: Any, name: str) -> Callable[..., Tensor]:
    function = methods.get(name) if isinstance(methods, Mapping) else getattr(methods, name, None)
    if not callable(function):
        raise TypeError(f"methods must expose callable {name}.")
    return function


def _state_restoring_adapter(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Wrap a model-first adapter so even validation and import failures restore state."""

    @functools.wraps(function)
    def wrapped(model: nn.Module, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(model, nn.Module):
            return function(model, *args, **kwargs)
        with _preserve_execution_state(model):
            return function(model, *args, **kwargs)

    return wrapped


def reference_environment() -> dict[str, Any]:
    """Require the exact installed and cached package implementations."""
    import captum
    import shap

    versions = {
        "numpy": np.__version__,
        "torch": t.__version__,
        "shap": shap.__version__,
        "captum": captum.__version__,
    }
    if versions != EXPECTED_VERSIONS:
        raise ValueError(f"GradientSHAP reference version mismatch: {versions}.")

    shap_root = Path(shap.__file__).resolve().parent
    captum_root = Path(captum.__file__).resolve().parent
    installed = {
        "shap/explainers/_gradient.py": shap_root / "explainers/_gradient.py",
        "captum/attr/_core/gradient_shap.py": captum_root / "attr/_core/gradient_shap.py",
        "captum/attr/_core/noise_tunnel.py": captum_root / "attr/_core/noise_tunnel.py",
        "captum/_utils/common.py": captum_root / "_utils/common.py",
    }
    cached = {
        "shap/explainers/_gradient.py": REFERENCE_CACHE / "sources/shap_v0.49.1_gradient.py",
        "captum/attr/_core/gradient_shap.py": REFERENCE_CACHE / "sources/captum_v0.9.0_gradient_shap.py",
        "captum/attr/_core/noise_tunnel.py": REFERENCE_CACHE / "sources/captum_v0.9.0_noise_tunnel.py",
        "captum/_utils/common.py": REFERENCE_CACHE / "sources/captum_v0.9.0_common.py",
    }
    installed_hashes, cached_hashes = {}, {}
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        installed_hashes[name] = _sha256(installed[name])
        cached_hashes[name] = _sha256(cached[name])
        if installed_hashes[name] != expected or cached_hashes[name] != expected:
            raise ValueError(f"Pinned package source changed: {name}.")
    return {
        "versions": versions,
        "installed_source_sha256": installed_hashes,
        "cached_source_sha256": cached_hashes,
        "source_evidence_sha256": _sha256(REFERENCE_CACHE / "source_evidence.json"),
    }


def shap_draw_table(
    rows: int,
    features: int,
    background_rows: int,
    samples: int,
    *,
    numpy_seed: int,
    torch_seed: int,
    local_smoothing: float,
) -> dict[str, np.ndarray]:
    """Recreate SHAP 0.49.1's row-major donor/alpha/noise sequence locally."""
    rows = _positive_int("rows", rows)
    features = _positive_int("features", features)
    background_rows = _positive_int("background_rows", background_rows)
    samples = _positive_int("samples", samples)
    numpy_seed = _seed("numpy_seed", numpy_seed)
    torch_seed = _seed("torch_seed", torch_seed)
    if not isinstance(local_smoothing, (int, float)) or not math.isfinite(local_smoothing):
        raise ValueError("local_smoothing must be a finite nonnegative number.")
    local_smoothing = float(local_smoothing)
    if local_smoothing < 0:
        raise ValueError("local_smoothing must be nonnegative.")

    legacy = np.random.RandomState(numpy_seed)
    donor_indices = np.empty((rows, samples), dtype=np.int64)
    alphas = np.empty((rows, samples), dtype=np.float64)
    for row in range(rows):
        for sample in range(samples):
            donor_indices[row, sample] = legacy.choice(background_rows)
            alphas[row, sample] = legacy.uniform()

    generator = t.Generator(device="cpu").manual_seed(torch_seed)
    noise = np.zeros((rows, samples, features), dtype=np.float64)
    if local_smoothing > 0:
        for row in range(rows):
            for sample in range(samples):
                noise[row, sample] = (
                    t.empty(features, dtype=t.float64)
                    .normal_(generator=generator)
                    .mul(local_smoothing)
                    .numpy()
                )
    return {
        "donor_indices": donor_indices,
        "alphas": alphas,
        "noise": noise,
    }


def captum_draw_table(
    rows: int,
    features: int,
    background_rows: int,
    samples: int,
    *,
    numpy_seed: int,
    torch_seed: int,
    stdev: float,
) -> dict[str, np.ndarray]:
    """Recreate Captum 0.9.0's all-donor, noise, then all-alpha sequence."""
    rows = _positive_int("rows", rows)
    features = _positive_int("features", features)
    background_rows = _positive_int("background_rows", background_rows)
    samples = _positive_int("samples", samples)
    numpy_seed = _seed("numpy_seed", numpy_seed)
    torch_seed = _seed("torch_seed", torch_seed)
    if not isinstance(stdev, (int, float)) or not math.isfinite(stdev):
        raise ValueError("stdev must be a finite nonnegative number.")
    stdev = float(stdev)
    if stdev < 0:
        raise ValueError("stdev must be nonnegative.")

    legacy = np.random.RandomState(numpy_seed)
    donor_indices = legacy.choice(background_rows, rows * samples).reshape(rows, samples)
    generator = t.Generator(device="cpu").manual_seed(torch_seed)
    # Captum 0.9.0 NoiseTunnel constructs this buffer from Torch's default
    # dtype.  The canonical adapter below pins that default to float32.
    stdevs = t.tensor(stdev, dtype=t.float32).repeat(rows * samples, features)
    noise = t.normal(0.0, stdevs, generator=generator).reshape(rows, samples, features)
    alphas = legacy.uniform(0.0, 1.0, rows * samples).reshape(rows, samples)
    return {
        "donor_indices": donor_indices.astype(np.int64, copy=False),
        "alphas": alphas.astype(np.float64, copy=False),
        "noise": noise.numpy().astype(np.float64),
    }


def _validate_inputs(
    model: nn.Module,
    queries: Tensor,
    background: Tensor,
    source_ids: Tensor | np.ndarray | None,
) -> tuple[nn.Module, np.ndarray]:
    _float64_tensor("queries", queries, 2)
    _float64_tensor("background", background, 2)
    if queries.shape[1] != background.shape[1]:
        raise ValueError("queries and background must have the same feature dimension.")
    return _validate_model(model, queries.shape[1]), _source_ids(source_ids, len(queries))


def _validate_draw_table(
    draws: Mapping[str, Any], rows: int, samples: int, features: int, donors: int
) -> dict[str, np.ndarray]:
    if not isinstance(draws, Mapping) or set(draws) != {"donor_indices", "alphas", "noise"}:
        raise ValueError("draws must contain exactly donor_indices, alphas, and noise.")
    indices = _numpy(draws["donor_indices"])
    alphas = _numpy(draws["alphas"])
    noise = _numpy(draws["noise"])
    if indices.dtype != np.int64 or indices.shape != (rows, samples):
        raise ValueError("donor_indices has the wrong shape or dtype.")
    if alphas.dtype != np.float64 or alphas.shape != (rows, samples):
        raise ValueError("alphas has the wrong shape or dtype.")
    if noise.dtype != np.float64 or noise.shape != (rows, samples, features):
        raise ValueError("noise has the wrong shape or dtype.")
    if indices.min() < 0 or indices.max() >= donors:
        raise ValueError("donor_indices selects a row outside the supplied background.")
    if not np.isfinite(alphas).all() or np.any((alphas < 0) | (alphas > 1)):
        raise ValueError("alphas must be finite values in [0,1].")
    if not np.isfinite(noise).all():
        raise ValueError("noise must contain only finite values.")
    return {
        "donor_indices": np.array(indices, copy=True),
        "alphas": np.array(alphas, copy=True),
        "noise": np.array(noise, copy=True),
    }


def _model_values(model: nn.Module, inputs: Tensor, batch: int = FORWARD_BATCH) -> Tensor:
    values: list[Tensor] = []
    with t.no_grad():
        for start in range(0, len(inputs), batch):
            values.append(_model_output(model, inputs[start : start + batch]).detach())
    return t.cat(values)


@_state_restoring_adapter
def replay_draw_table(
    model: nn.Module,
    queries: Tensor,
    background: Tensor,
    draws: Mapping[str, Any],
    methods: Any,
    *,
    forward_batch: int = FORWARD_BATCH,
) -> dict[str, np.ndarray | float]:
    """Replay one explicit package draw table through the visible student methods."""
    model, _ = _validate_inputs(model, queries, background, None)
    forward_batch = _positive_int("forward_batch", forward_batch)
    if not isinstance(draws, Mapping) or "donor_indices" not in draws:
        raise ValueError("draws must be a mapping with a donor_indices table.")
    index_shape = _numpy(draws["donor_indices"]).shape
    if len(index_shape) != 2:
        raise ValueError("donor_indices must be a two-dimensional table.")
    samples = index_shape[1]
    table = _validate_draw_table(
        draws, len(queries), samples, queries.shape[1], len(background)
    )
    indices = t.from_numpy(table["donor_indices"].copy())
    alphas = t.from_numpy(table["alphas"].copy())
    noise = t.from_numpy(table["noise"].copy())
    query_copy, background_copy = queries.detach().clone(), background.detach().clone()

    with _preserve_execution_state(model):
        model.eval()
        if np.count_nonzero(table["noise"]) == 0:
            function = _method(methods, "expected_gradient_samples")
            contributions = function(
                model,
                query_copy,
                background_copy,
                indices,
                alphas,
                forward_batch=forward_batch,
            )
        else:
            function = _method(methods, "noisy_expected_gradient_samples")
            contributions = function(
                model,
                query_copy,
                background_copy,
                indices,
                alphas,
                noise,
                forward_batch=forward_batch,
            )
        _float64_tensor("student contributions", contributions, 3)
        expected_shape = (len(queries), samples, queries.shape[1])
        if tuple(contributions.shape) != expected_shape:
            raise ValueError(f"Student contributions must have shape {expected_shape}.")
        donors = background_copy[indices]
        noisy_queries = query_copy[:, None, :] + noise
        flat_donors = donors.reshape(-1, queries.shape[1])
        flat_noisy = noisy_queries.reshape(-1, queries.shape[1])
        donor_values = _model_values(model, flat_donors, forward_batch).reshape(
            len(queries), samples
        )
        noisy_values = _model_values(model, flat_noisy, forward_batch).reshape(
            len(queries), samples
        )
        prediction = _model_values(model, query_copy, forward_batch)
        background_values = _model_values(model, background_copy, forward_batch)

    phi = contributions.mean(dim=1)
    standard_error = (
        contributions.std(dim=1, correction=1) / math.sqrt(samples)
        if samples >= 2
        else t.full_like(phi, math.nan)
    )
    variance_proxy = contributions.var(dim=1, correction=0) / math.sqrt(samples)
    repeated_delta = contributions.sum(dim=2) - (noisy_values - donor_values)
    return {
        "student_phi": phi.detach().numpy().copy(),
        "student_contributions": contributions.detach().numpy().copy(),
        "ordinary_standard_error": standard_error.detach().numpy().copy(),
        "shap_variance_proxy": variance_proxy.detach().numpy().copy(),
        "repeated_convergence_delta": repeated_delta.detach().numpy().reshape(-1).copy(),
        "prediction": prediction.detach().numpy().copy(),
        "full_background_base": float(background_values.mean()),
        "sampled_background_base": donor_values.mean(dim=1).detach().numpy().copy(),
        "noisy_query_values": noisy_values.detach().numpy().copy(),
    }


def _identity(
    model: nn.Module,
    queries: Tensor,
    background: Tensor,
    ids: np.ndarray,
    draws: Mapping[str, np.ndarray],
    author_phi: np.ndarray,
    replay: Mapping[str, Any],
    **extra: Any,
) -> dict[str, str]:
    result = {
        "model_state": _model_state_sha256(model),
        "model_structure": _model_structure_sha256(model),
        "source_ids": _array_sha256(ids),
        "queries": _array_sha256(queries),
        "background": _array_sha256(background),
        "donor_indices": _array_sha256(draws["donor_indices"]),
        "alphas": _array_sha256(draws["alphas"]),
        "noise": _array_sha256(draws["noise"]),
        "author_phi": _array_sha256(author_phi),
        "student_phi": _array_sha256(replay["student_phi"]),
        "student_contributions": _array_sha256(replay["student_contributions"]),
    }
    result.update({name: _array_sha256(value) for name, value in extra.items()})
    return result


@_state_restoring_adapter
def shap_same_draw_reference(
    model: nn.Module,
    queries: Tensor,
    background: Tensor,
    methods: Any,
    *,
    samples: int,
    numpy_seed: int,
    torch_seed: int,
    batch_size: int = SHAP_BATCH_SIZE,
    local_smoothing: float = 0.0,
    source_ids: Tensor | np.ndarray | None = None,
    forward_batch: int = FORWARD_BATCH,
) -> dict[str, Any]:
    """Call SHAP 0.49.1 and replay its exact draws through student methods."""
    import shap

    model, ids = _validate_inputs(model, queries, background, source_ids)
    samples = _positive_int("samples", samples)
    batch_size = _positive_int("batch_size", batch_size)
    _seed("numpy_seed", numpy_seed)
    _seed("torch_seed", torch_seed)
    reference_environment()
    draws = shap_draw_table(
        len(queries),
        queries.shape[1],
        len(background),
        samples,
        numpy_seed=numpy_seed,
        torch_seed=torch_seed,
        local_smoothing=local_smoothing,
    )
    query_copy, background_copy = queries.detach().clone(), background.detach().clone()
    state_before = _model_state_sha256(model)
    with _preserve_execution_state(model):
        t.set_default_dtype(t.float64)
        np.random.seed(numpy_seed)
        t.manual_seed(torch_seed)
        model.eval()
        _model_output(model, query_copy[:1])
        explainer = shap.GradientExplainer(
            model,
            background_copy,
            batch_size=batch_size,
            local_smoothing=float(local_smoothing),
        )
        raw_phi, raw_variance = explainer.shap_values(
            query_copy,
            nsamples=samples,
            rseed=numpy_seed,
            return_variances=True,
        )
        raw_phi_array = np.asarray(raw_phi, dtype=np.float64)
        if raw_phi_array.shape != (*queries.shape, 1):
            raise AssertionError(
                f"SHAP scalar-output attribution had unexpected shape {raw_phi_array.shape}."
            )
        author_phi = raw_phi_array[..., 0].copy()
        if not isinstance(raw_variance, list) or len(raw_variance) != 1:
            raise AssertionError("SHAP scalar-output variance must be list[one ndarray].")
        variance_proxy = np.asarray(raw_variance[0], dtype=np.float64)
        if variance_proxy.shape != tuple(queries.shape) or not np.isfinite(variance_proxy).all():
            raise AssertionError("SHAP returned an invalid variance-proxy array.")
        prediction = _model_values(model, query_copy).numpy().copy()
        full_background_base = float(_model_values(model, background_copy).mean())

    replay = replay_draw_table(
        model,
        queries,
        background,
        draws,
        methods,
        forward_batch=forward_batch,
    )
    state_after = _model_state_sha256(model)
    if state_after != state_before:
        raise AssertionError("SHAP reference adapter did not restore the model state.")
    parity_error = _max_abs(author_phi, replay["student_phi"])
    variance_error = _max_abs(variance_proxy, replay["shap_variance_proxy"])
    prediction_error = _max_abs(prediction, replay["prediction"])
    base_error = abs(full_background_base - float(replay["full_background_base"]))
    return {
        "backend": "shap",
        "version": shap.__version__,
        "samples": samples,
        "numpy_seed": numpy_seed,
        "torch_seed": torch_seed,
        "batch_size": batch_size,
        "local_smoothing": float(local_smoothing),
        "source_ids": ids,
        "draws": draws,
        "author_phi_raw": raw_phi_array.copy(),
        "author_phi": author_phi,
        "student_phi": replay["student_phi"],
        "student_contributions": replay["student_contributions"],
        "ordinary_standard_error": replay["ordinary_standard_error"],
        "author_shap_variance_proxy": variance_proxy.copy(),
        "student_shap_variance_proxy": replay["shap_variance_proxy"],
        "prediction": prediction,
        "full_background_base": full_background_base,
        "sampled_background_base": replay["sampled_background_base"],
        "parity_max_abs_error": parity_error,
        "variance_proxy_max_abs_error": variance_error,
        "prediction_max_abs_error": prediction_error,
        "base_max_abs_error": base_error,
        "parity_passed": _close(author_phi, replay["student_phi"]),
        "warnings": SHAP_WARNINGS,
        "identity_sha256": _identity(
            model,
            queries,
            background,
            ids,
            draws,
            author_phi,
            replay,
            author_phi_raw=raw_phi_array,
            author_shap_variance_proxy=variance_proxy,
            student_shap_variance_proxy=replay["shap_variance_proxy"],
        ),
    }


@_state_restoring_adapter
def captum_same_draw_reference(
    model: nn.Module,
    queries: Tensor,
    background: Tensor,
    methods: Any,
    *,
    samples: int,
    numpy_seed: int,
    torch_seed: int,
    stdev: float = 0.0,
    source_ids: Tensor | np.ndarray | None = None,
    forward_batch: int = FORWARD_BATCH,
) -> dict[str, Any]:
    """Call Captum 0.9.0 once and replay its exact full-query draw table."""
    from captum.attr import GradientShap
    import captum

    model, ids = _validate_inputs(model, queries, background, source_ids)
    samples = _positive_int("samples", samples)
    _seed("numpy_seed", numpy_seed)
    _seed("torch_seed", torch_seed)
    reference_environment()
    draws = captum_draw_table(
        len(queries),
        queries.shape[1],
        len(background),
        samples,
        numpy_seed=numpy_seed,
        torch_seed=torch_seed,
        stdev=stdev,
    )
    query_copy, background_copy = queries.detach().clone(), background.detach().clone()
    state_before = _model_state_sha256(model)
    with _preserve_execution_state(model):
        # Pin the source-observed NoiseTunnel convention.  This is restored even
        # if construction, attribution, or result validation raises.
        t.set_default_dtype(t.float32)
        np.random.seed(numpy_seed)
        t.manual_seed(torch_seed)
        model.eval()
        _model_output(model, query_copy[:1])
        raw_phi, raw_delta = GradientShap(model).attribute(
            query_copy,
            background_copy,
            n_samples=samples,
            stdevs=float(stdev),
            return_convergence_delta=True,
        )
        if type(raw_phi) is not Tensor or raw_phi.shape != queries.shape:
            raise AssertionError("Captum returned an unexpected attribution tensor.")
        if raw_phi.device.type != "cpu" or raw_phi.dtype != t.float64:
            raise AssertionError("Captum attribution must remain CPU float64.")
        if type(raw_delta) is not Tensor or raw_delta.shape != (len(queries) * samples,):
            raise AssertionError("Captum returned an unexpected repeated-delta tensor.")
        author_phi = raw_phi.detach().numpy().copy()
        author_delta = raw_delta.detach().numpy().copy()

    replay = replay_draw_table(
        model,
        queries,
        background,
        draws,
        methods,
        forward_batch=forward_batch,
    )
    state_after = _model_state_sha256(model)
    if state_after != state_before:
        raise AssertionError("Captum reference adapter did not restore the model state.")
    parity_error = _max_abs(author_phi, replay["student_phi"])
    delta_error = _max_abs(author_delta, replay["repeated_convergence_delta"])
    return {
        "backend": "captum",
        "version": captum.__version__,
        "samples": samples,
        "numpy_seed": numpy_seed,
        "torch_seed": torch_seed,
        "stdev": float(stdev),
        "source_ids": ids,
        "draws": draws,
        "author_phi": author_phi,
        "student_phi": replay["student_phi"],
        "student_contributions": replay["student_contributions"],
        "ordinary_standard_error": replay["ordinary_standard_error"],
        "author_convergence_delta": author_delta,
        "student_repeated_convergence_delta": replay["repeated_convergence_delta"],
        "prediction": replay["prediction"],
        "full_background_base": replay["full_background_base"],
        "sampled_background_base": replay["sampled_background_base"],
        "parity_max_abs_error": parity_error,
        "convergence_delta_max_abs_error": delta_error,
        "parity_passed": _close(author_phi, replay["student_phi"]),
        "warnings": CAPTUM_WARNINGS,
        "identity_sha256": _identity(
            model,
            queries,
            background,
            ids,
            draws,
            author_phi,
            replay,
            author_convergence_delta=author_delta,
            student_repeated_convergence_delta=replay["repeated_convergence_delta"],
        ),
    }


def _smoothed_relu() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(1, 1, bias=False, dtype=t.float64),
        nn.ReLU(),
        nn.Linear(1, 1, bias=False, dtype=t.float64),
    )
    with t.no_grad():
        model[0].weight.fill_(1.0)
        model[2].weight.fill_(1.0)
    return model.eval()


def positive_noise_reference(methods: Any) -> dict[str, Any]:
    """Check both positive-noise streams on an analytic one-dimensional ReLU.

    With a zero donor, every non-kink draw contributes ``ReLU(x + noise)``.
    This gives an exact finite-draw oracle for each package while the Gaussian
    population target remains ``sigma*pdf(x/sigma) + x*cdf(x/sigma)``.
    """
    model = _smoothed_relu()
    query = t.tensor([[0.15]], dtype=t.float64)
    background = t.tensor([[0.0]], dtype=t.float64)
    sigma = 0.2
    samples = 257
    ids = np.array([915001], dtype=np.int64)
    shap_result = shap_same_draw_reference(
        model,
        query,
        background,
        methods,
        samples=samples,
        numpy_seed=SHAP_NUMPY_SEED,
        torch_seed=SHAP_TORCH_SEED,
        batch_size=257,
        local_smoothing=sigma,
        source_ids=ids,
    )
    captum_result = captum_same_draw_reference(
        model,
        query,
        background,
        methods,
        samples=samples,
        numpy_seed=CAPTUM_NUMPY_SEED,
        torch_seed=CAPTUM_TORCH_SEED,
        stdev=sigma,
        source_ids=ids,
    )
    ratio = float(query[0, 0] / sigma)
    normal_pdf = math.exp(-0.5 * ratio**2) / math.sqrt(2 * math.pi)
    normal_cdf = 0.5 * (1.0 + math.erf(ratio / math.sqrt(2.0)))
    population_target = sigma * normal_pdf + float(query[0, 0]) * normal_cdf

    package_results = {}
    all_passed = True
    for name, result in (("shap", shap_result), ("captum", captum_result)):
        finite_draw_oracle = np.maximum(
            float(query[0, 0]) + result["draws"]["noise"][0, :, 0], 0.0
        ).mean()
        finite_draw_error = abs(float(result["author_phi"][0, 0]) - finite_draw_oracle)
        passed = bool(result["parity_passed"] and finite_draw_error <= AUTHOR_ATOL)
        all_passed &= passed
        package_results[name] = {
            "same_draw": result,
            "finite_draw_analytic_phi": finite_draw_oracle,
            "finite_draw_analytic_max_abs_error": finite_draw_error,
            "population_smoothed_relu_phi": population_target,
            "finite_draw_minus_population": finite_draw_oracle - population_target,
            "positive_noise_confirmed": bool(np.any(result["draws"]["noise"] != 0)),
            "passed": passed,
        }
    return {
        "status": "PASS" if all_passed else "FAIL",
        "fixture": "ReLU(x), zero donor, x=0.15, sigma=0.2",
        "samples": samples,
        "packages": package_results,
        "warning": "SHAP and Captum use different draw tables and noise dtypes; their finite Monte Carlo vectors are not cross-package equality targets.",
    }


def _compact_model_result(result: Mapping[str, Any], backend: str) -> dict[str, Any]:
    common = {
        "author_phi": result["author_phi"],
        "student_phi": result["student_phi"],
        "student_contributions": result["student_contributions"],
        "ordinary_standard_error": result["ordinary_standard_error"],
        "prediction": result["prediction"],
        "full_background_base": result["full_background_base"],
        "sampled_background_base": result["sampled_background_base"],
        "parity_max_abs_error": result["parity_max_abs_error"],
        "parity_passed": result["parity_passed"],
        "identity_sha256": result["identity_sha256"],
    }
    if backend == "shap":
        common.update(
            author_phi_raw=result["author_phi_raw"],
            author_shap_variance_proxy=result["author_shap_variance_proxy"],
            student_shap_variance_proxy=result["student_shap_variance_proxy"],
            variance_proxy_max_abs_error=result["variance_proxy_max_abs_error"],
        )
    else:
        common.update(
            author_convergence_delta=result["author_convergence_delta"],
            student_repeated_convergence_delta=result[
                "student_repeated_convergence_delta"
            ],
            convergence_delta_max_abs_error=result["convergence_delta_max_abs_error"],
        )
    return common


def validate_reference(
    models: Mapping[str, nn.Module],
    queries: Tensor,
    background: Tensor,
    source_ids: Tensor | np.ndarray,
    methods: Any,
    *,
    samples: int = REFERENCE_SAMPLES,
) -> dict[str, Any]:
    """Run complete same-draw package parity over all seven models and 272 rows."""
    if not isinstance(models, Mapping) or tuple(models) != MODEL_KEYS:
        raise ValueError("models must contain the seven frozen model keys in protocol order.")
    _float64_tensor("queries", queries, 2)
    _float64_tensor("background", background, 2)
    ids = _source_ids(source_ids, len(queries))
    if tuple(queries.shape) != (ALL_ROWS, FEATURES):
        raise ValueError(f"queries must have frozen shape ({ALL_ROWS},{FEATURES}).")
    if tuple(background.shape) != (BACKGROUND_ROWS, FEATURES):
        raise ValueError(
            f"background must contain all {BACKGROUND_ROWS} primary donors and {FEATURES} features."
        )
    if samples != REFERENCE_SAMPLES:
        raise ValueError(f"Full package reference coverage is frozen at K={REFERENCE_SAMPLES}.")
    environment = reference_environment()

    package_results: dict[str, dict[str, Any]] = {
        "shap": {"models": {}},
        "captum": {"models": {}},
    }
    maxima = {"shap": 0.0, "captum": 0.0}
    all_passed = True
    shared_draw_hashes: dict[str, dict[str, str]] = {}
    for key in MODEL_KEYS:
        model = _validate_model(models[key], FEATURES)
        shap_result = shap_same_draw_reference(
            model,
            queries,
            background,
            methods,
            samples=samples,
            numpy_seed=SHAP_NUMPY_SEED,
            torch_seed=SHAP_TORCH_SEED,
            batch_size=SHAP_BATCH_SIZE,
            local_smoothing=0.0,
            source_ids=ids,
        )
        captum_result = captum_same_draw_reference(
            model,
            queries,
            background,
            methods,
            samples=samples,
            numpy_seed=CAPTUM_NUMPY_SEED,
            torch_seed=CAPTUM_TORCH_SEED,
            stdev=0.0,
            source_ids=ids,
        )
        for backend, result in (("shap", shap_result), ("captum", captum_result)):
            package_results[backend]["models"][key] = _compact_model_result(result, backend)
            maxima[backend] = max(maxima[backend], float(result["parity_max_abs_error"]))
            all_passed &= bool(result["parity_passed"])
            hashes = {
                name: result["identity_sha256"][name]
                for name in ("source_ids", "queries", "background", "donor_indices", "alphas", "noise")
            }
            if backend not in shared_draw_hashes:
                shared_draw_hashes[backend] = hashes
                package_results[backend]["draws"] = result["draws"]
            elif hashes != shared_draw_hashes[backend]:
                raise AssertionError(f"{backend} draw identity changed between models.")

    package_results["shap"].update(
        version=EXPECTED_VERSIONS["shap"],
        warnings=SHAP_WARNINGS,
        max_parity_error=maxima["shap"],
        all_models_passed=all(
            item["parity_passed"] for item in package_results["shap"]["models"].values()
        ),
        draw_identity_sha256=shared_draw_hashes["shap"],
    )
    package_results["captum"].update(
        version=EXPECTED_VERSIONS["captum"],
        warnings=CAPTUM_WARNINGS,
        max_parity_error=maxima["captum"],
        all_models_passed=all(
            item["parity_passed"] for item in package_results["captum"]["models"].values()
        ),
        draw_identity_sha256=shared_draw_hashes["captum"],
    )

    positive_noise = positive_noise_reference(methods)
    package_draws_distinct = any(
        shared_draw_hashes["shap"][name] != shared_draw_hashes["captum"][name]
        for name in ("donor_indices", "alphas")
    )
    passes = {
        "sources_pinned": True,
        "shap_same_draw_all_models": bool(package_results["shap"]["all_models_passed"]),
        "captum_same_draw_all_models": bool(package_results["captum"]["all_models_passed"]),
        "positive_noise_same_draw": positive_noise["status"] == "PASS",
        "package_draw_tables_distinct": package_draws_distinct,
    }
    all_passed &= all(passes.values())
    counts = {
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
    return {
        "status": "PASS" if all_passed else "FAIL",
        "environment": environment,
        "versions": {
            "shap": EXPECTED_VERSIONS["shap"],
            "captum": EXPECTED_VERSIONS["captum"],
        },
        "counts": counts,
        "tolerances": {
            "same_draw_atol": AUTHOR_ATOL,
            "same_draw_rtol": AUTHOR_RTOL,
        },
        "passes": passes,
        "scope": {
            "models": len(MODEL_KEYS),
            "model_keys": MODEL_KEYS,
            "rows_per_model": ALL_ROWS,
            "background_rows": BACKGROUND_ROWS,
            "features": FEATURES,
            "samples_per_query": samples,
            "model_rows_per_package": len(MODEL_KEYS) * ALL_ROWS,
            "local_coordinates_per_package": len(MODEL_KEYS) * ALL_ROWS * FEATURES,
            "draws_per_package": len(MODEL_KEYS) * ALL_ROWS * samples,
            "contribution_coordinates_per_package": (
                len(MODEL_KEYS) * ALL_ROWS * samples * FEATURES
            ),
        },
        "identity_sha256": {
            "source_ids": _array_sha256(ids),
            "queries": _array_sha256(queries),
            "background": _array_sha256(background),
        },
        "packages": package_results,
        "positive_noise": positive_noise,
        "acceptance": {
            "atol": AUTHOR_ATOL,
            "rtol": AUTHOR_RTOL,
            "requires_same_draws": True,
            "shap_variance_proxy_is_acceptance_metric": False,
            "captum_convergence_delta_is_acceptance_metric": False,
            "cross_package_same_seed_parity_is_claimed": False,
        },
    }


def reference_check(
    models: Mapping[str, nn.Module],
    queries: Tensor,
    background: Tensor,
    source_ids: Tensor | np.ndarray,
    methods: Any,
) -> dict[str, Any]:
    """Visible learner call for the frozen all-model, all-row package check."""
    return validate_reference(
        models,
        queries,
        background,
        source_ids,
        methods,
        samples=REFERENCE_SAMPLES,
    )
