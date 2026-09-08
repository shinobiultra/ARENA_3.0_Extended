"""Prospective query, fold and shuffle construction: no model calls or fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_data

SEEDS = (0, 1, 2)
HASH_BITS = (3, 4, 5)
AUDIT_SIZE = 512


def uniform_unique_masks(n, count, *, seed, excluded=(), endpoints=False):
    """Draw one uint8 PCG64 Bernoulli row at a time, rejecting duplicate IDs.

    Drawing representation and row order are part of the prospective contract.
    Endpoints, when requested, occupy rows zero and one without consuming RNG.
    """
    if n != native_data.N_FEATURES or count < (2 if endpoints else 1):
        raise ValueError("This native plan requires 28 features and a positive budget.")
    seen = {int(value) for value in excluded}
    rows = []
    if endpoints:
        for row in (np.zeros(n, dtype=np.uint8), np.ones(n, dtype=np.uint8)):
            key = int(native_data.mask_ids(row[None])[0])
            if key in seen:
                raise ValueError("An explicitly included endpoint is excluded.")
            rows.append(row)
            seen.add(key)
    rng = np.random.Generator(np.random.PCG64(seed))
    while len(rows) < count:
        row = rng.integers(0, 2, size=n, dtype=np.uint8)
        key = int(native_data.mask_ids(row[None])[0])
        if key not in seen:
            rows.append(row)
            seen.add(key)
    return np.stack(rows)


def make_plan(spex_methods, proxy_methods):
    """Freeze every method's ordered requests before drawing untouched audit rows."""
    arrays, conditions, training_union = {}, [], set()
    for seed in SEEDS:
        proxy_stream = uniform_unique_masks(28, 1824, seed=9200 + seed, endpoints=True)
        for bits in HASH_BITS:
            design = spex_methods.make_query_design(n=28, t=3, b=bits, C=3, seed=seed)
            budget = design.requested_count
            for method, masks in (("spex", design.masks), ("proxyspex", proxy_stream[:budget])):
                key = f"{method}_s{seed}_k{budget}"
                masks = np.asarray(masks, dtype=np.uint8)
                ids = native_data.mask_ids(masks)
                unique_ids, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
                training_union.update(map(int, unique_ids))
                arrays[key + "_masks"] = masks
                arrays[key + "_ids"] = ids
                arrays[key + "_unique_ids"] = unique_ids
                arrays[key + "_unique_first"] = first
                arrays[key + "_inverse"] = inverse
                # Permute labels of ascending unique IDs, then restore raw request order.
                arrays[key + "_shuffle"] = np.random.default_rng(3300 + seed).permutation(len(unique_ids))
                if method == "spex":
                    arrays[key + "_matrices"] = design.matrices
                    arrays[key + "_delays"] = design.delays
                else:
                    for fold_index, fold in enumerate(proxy_methods.grouped_mask_folds(masks, fold_seed=seed)):
                        arrays[f"{key}_fold{fold_index}_train"] = fold["train_indices"]
                        arrays[f"{key}_fold{fold_index}_validation"] = fold["validation_indices"]
                conditions.append({"key": key, "method": method, "seed": seed, "b": bits,
                                   "budget": budget, "unique_count": len(unique_ids)})
    arrays["training_union_ids"] = np.asarray(sorted(training_union), dtype=np.uint64)
    arrays["audit_masks"] = uniform_unique_masks(28, AUDIT_SIZE, seed=7719, excluded=training_union)
    arrays["audit_ids"] = native_data.mask_ids(arrays["audit_masks"])
    if len(set(map(int, arrays["audit_ids"]))) != AUDIT_SIZE:
        raise AssertionError("Audit masks are not unique.")
    if np.isin(arrays["audit_ids"], arrays["training_union_ids"]).any():
        raise AssertionError("Audit overlaps a training design.")
    metadata = {
        "schema_version": 1, "status": "NO_NATIVE_OUTCOMES", "n": 28,
        "conditions": conditions, "native_queries": 0, "native_proxy_fits": 0,
        "training_union_per_prompt": len(training_union), "audit_rows_per_prompt": AUDIT_SIZE,
        "primary_training_and_audit_rows": 32 * (len(training_union) + AUDIT_SIZE),
        "counterfactual_rows": 32, "spex_recoveries": 576, "proxy_pipelines": 576,
        "proxy_ensemble_fits": 7488, "proxy_residual_tree_fits_min": 2285568,
        "proxy_residual_tree_fits_max": 2506752, "ridge_candidate_fits": 9216,
        "ridge_matrix_decompositions": 2304,
        "baseline_fits": 2304,
        "baseline_fit_detail": "constant and additive for every real and shuffled method condition",
        "reference_stages": 9, "reference_new_model_rows": 0,
        "reference_scope": "prompt 0, seed 0, three budgets; three actual reference variants",
        "probe_rows": "Bound by the separately frozen oracle contract; additional to this inventory.",
    }
    return metadata, arrays


def array_record(array):
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(), "elements": int(array.size)}


def write_plan(output_dir, spex_methods, proxy_methods):
    """Write once. The manifest binds array values, dtypes, shapes and ordering."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata, arrays = make_plan(spex_methods, proxy_methods)
    path = output_dir / "query_plan.npz"
    np.savez_compressed(path, **arrays)
    metadata["arrays"] = {key: array_record(value) for key, value in arrays.items()}
    metadata["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "query_plan.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_plan(output_dir):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "query_plan.json").read_text())
    path = output_dir / "query_plan.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["archive_sha256"]:
        raise ValueError("Frozen query archive changed.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if {key: array_record(value) for key, value in arrays.items()} != metadata["arrays"]:
        raise ValueError("Frozen query arrays changed.")
    return metadata, arrays


def validate_student_plan(output_dir, spex_methods, proxy_methods):
    """Check the student's sampler/folds against the source-bound native design."""
    expected, arrays = load_plan(output_dir)
    actual_metadata, actual = make_plan(spex_methods, proxy_methods)
    for key, value in actual_metadata.items():
        if expected[key] != value:
            raise ValueError(f"Native query metadata changed: {key}")
    if set(arrays) != set(actual):
        raise ValueError("Native query inventory changed.")
    for key in arrays:
        if array_record(arrays[key]) != array_record(actual[key]):
            raise ValueError(f"Student query/fold design differs from the native freeze: {key}")
    return expected, arrays
