"""Independent checks: permutation enumeration, constrained WLS, NumPy, author code."""

from pathlib import Path
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import types
import urllib.request
import numpy as np
import torch as t

HERE = Path(__file__).resolve().parent
COMMIT = "0e34e6c7e2a6b1666e39043e83b9929e68ee68f3"


def author_imputer():
    """Load two byte-pinned, unmodified author files with an isolated import namespace."""
    manifest = json.loads((HERE / "author_reference.json").read_text())
    cache = Path.home() / ".cache/arena_extended/sage"
    modules = {}
    for name in ["utils.py", "imputers.py"]:
        path = cache / ("author_" + name)
        if not path.exists():
            with urllib.request.urlopen(manifest[name]["url"], timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != manifest[name]["sha256"]:
                raise ValueError("Author reference checksum changed.")
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]["sha256"]:
            raise ValueError("Cached author reference checksum changed.")
        spec = importlib.util.spec_from_file_location("arena_sage_author_" + name[:-3], path)
        module = importlib.util.module_from_spec(spec)
        if name == "imputers.py":
            previous = sys.modules.get("sage")
            namespace = types.ModuleType("sage")
            namespace.utils = modules["utils.py"]
            sys.modules["sage"] = namespace
            try:
                spec.loader.exec_module(module)
            finally:
                if previous is None:
                    sys.modules.pop("sage", None)
                else:
                    sys.modules["sage"] = previous
        else:
            spec.loader.exec_module(module)
        modules[name] = module
    return modules["imputers.py"].MarginalImputer


def permutation_reference(game):
    """Independent complete-ordering oracle with explicit nonaliasing mask buffers."""
    d = len(game).bit_length() - 1
    permutations = np.array(list(itertools.permutations(range(d))))
    current = np.zeros(len(permutations), dtype=np.int64)
    contributions = np.zeros((len(permutations), d))
    for step in range(d):
        features = permutations[:, step]
        # In the tested Python 3.14 / NumPy 1.26 Jupyter kernel, the | operator
        # reused the initial current buffer. Explicit output storage preserves it.
        added = np.empty_like(current)
        np.bitwise_or(current, np.left_shift(1, features), out=added)
        if np.shares_memory(current, added):
            raise ValueError("New coalition masks must not alias predecessor masks.")
        contributions[np.arange(len(permutations)), features] = game[added] - game[current]
        current = added
    np.testing.assert_allclose(contributions.sum(1), game[-1] - game[0], rtol=1e-10, atol=1e-10)
    return contributions.mean(0), len(permutations)


def checks(prepared, methods, predictor):
    game = prepared["explanations"][0]["game"].numpy()
    d = 8
    permutation_values, permutation_count = permutation_reference(game)
    masks = np.array([[bool(m & (1 << j)) for j in range(d)] for m in range(1, 255)], dtype=float)
    sizes = masks.sum(1).astype(int)
    weights = np.array([(d - 1) / (math.comb(d, k) * k * (d - k)) for k in sizes])
    total = game[-1] - game[0]
    design = masks[:, :-1] - masks[:, -1:]
    target = game[1:-1] - game[0] - masks[:, -1] * total
    first = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]), target * np.sqrt(weights), rcond=None
    )[0]
    regression_values = np.r_[first, total - first.sum()]
    expected = prepared["explanations"][0]["sage"].numpy()
    np.testing.assert_allclose(permutation_values, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(regression_values, expected, rtol=1e-9, atol=1e-9)
    model = prepared["models"][0]
    state = {k: v.numpy() for k, v in model["state"].items()}
    x = prepared["explain_x"][:64].numpy()
    hidden = np.maximum(x @ state["0.weight"].T + state["0.bias"], 0)
    hidden = np.maximum(hidden @ state["2.weight"].T + state["2.bias"], 0)
    numpy_prediction = (hidden @ state["4.weight"].T + state["4.bias"]).ravel() * float(
        model["y_std"]
    ) + float(model["y_mean"])
    torch_prediction = predictor(t.from_numpy(x)).numpy()
    np.testing.assert_allclose(numpy_prediction, torch_prediction, rtol=1e-10, atol=1e-10)
    masks = methods.coalition_masks(8)[t.arange(64) * 4]
    imputer = author_imputer()(
        lambda a: predictor(t.from_numpy(a)).numpy(), prepared["background"].numpy()
    )
    theirs = imputer(x, masks.numpy())
    ours = methods.marginal_predict(
        predictor, t.from_numpy(x), prepared["background"], masks
    ).numpy()
    np.testing.assert_allclose(theirs, ours, rtol=1e-10, atol=1e-10)
    return {
        "status": "PASS",
        "author_commit": COMMIT,
        "permutations": permutation_count,
        "weighted_regression_coalitions": len(weights),
        "numpy_native_predictions": len(x),
        "author_native_imputations": len(x),
        "permutation_max_abs_error": float(np.max(np.abs(permutation_values - expected))),
        "weighted_regression_max_abs_error": float(np.max(np.abs(regression_values - expected))),
        "numpy_prediction_max_abs_error": float(
            np.max(np.abs(numpy_prediction - torch_prediction))
        ),
        "author_imputation_max_abs_error": float(np.max(np.abs(theirs - ours))),
    }
