"""Bounded, source-bound reference algorithms on supplied training values only.

CLI contract: ``contract --output contract.json``. The host API
``run_fixed_query_stages(output_dir, inputs, arrays, freeze)`` dispatches nine
isolated ``_worker`` commands through ``run_native.run_reference_stage``.
Audit labels are forbidden. Scientific source is never patched.

Author ProxySPEX's actual monolithic function is called on fixed samples. A
read-only Python profiler captures its return/exception frame's fitted objects
for receipts; it does not change inputs, globals, source, return values or errors.
Maintained ProxySPEX's actual fitting, extraction and refinement methods are
called separately because its public sampler cannot accept a supplied design.
SPEX uses the source's supported saved-M/D path and an exact value-replay callback.
None of these adapters claims that the public samplers generated the same inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

SCHEMA_VERSION = 1
AUTHOR_REVISION = "61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8"
VARIANTS = ("author_spex", "author_proxy", "maintained_proxy")
STAGE_SECONDS = 300
STAGE_RSS_BYTES = 8 * 1024**3
STAGE_OUTPUT_BYTES = 256 * 1024**2
THREADS = 4
REQUIRED_FIELDS = frozenset({
    "schema_version", "n", "t", "b", "seed", "prompt_index", "spex_matrices",
    "spex_delays", "spex_masks", "spex_values", "proxy_masks", "proxy_values",
    "audit_masks",
})
REQUIRED_VERSIONS = {
    "spectralexplain": "0.0.4", "sparse-transform": "0.2.1", "shapiq": "1.5.2",
    "galois": "0.4.11", "lightgbm": "4.7.0", "numpy": "2.5.3",
    "scipy": "1.18.1", "scikit-learn": "1.9.0", "pandas": "3.0.5",
    "numba": "0.67.0", "llvmlite": "0.49.0",
}


def configure_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:THREADS])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def canonical_bytes(value):
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def contract():
    if platform.python_version() != "3.12.12":
        raise RuntimeError("Use the isolated reference_environment: Python 3.12.12 is required")
    for name, expected in REQUIRED_VERSIONS.items():
        found = importlib.metadata.version(name)
        if found != expected:
            raise RuntimeError(f"Reference version mismatch: {name}={found}, expected {expected}")
    distribution = importlib.metadata.distribution("spectralexplain")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != AUTHOR_REVISION:
        raise RuntimeError("Reference environment must install spectralexplain from the pinned Git revision")
    sources = {}
    for package_name in ("spectralexplain", "sparse_transform", "shapiq"):
        package = importlib.import_module(package_name)
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*.py")):
            sources[f"{package_name}/{path.relative_to(root)}"] = sha256(path)
    environment = Path(__file__).resolve().parent / "reference_environment"
    versions = {d.metadata["Name"].lower().replace("_", "-"): d.version
                for d in importlib.metadata.distributions()}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "bridge_source_sha256": sha256(__file__),
        "reference_pyproject_sha256": sha256(environment / "pyproject.toml"),
        "reference_lock_sha256": sha256(environment / "uv.lock"),
        "author_revision": AUTHOR_REVISION, "python": platform.python_version(),
        "distribution_versions": versions, "scientific_sources_sha256": sources,
        "input_fields": sorted(REQUIRED_FIELDS),
        "native_scope": {"prompt_index": 0, "seed": 0, "n": 28, "t": 3,
                         "C": 3, "b": [3, 4, 5], "budgets": [456, 912, 1824],
                         "audit_rows": 512, "proxy_masks_unique": True,
                         "audit_disjoint_from_training": True},
        "stage_caps": {"wall_seconds": STAGE_SECONDS, "summed_process_tree_rss_bytes": STAGE_RSS_BYTES,
                       "retained_output_bytes": STAGE_OUTPUT_BYTES, "cpu_threads": THREADS,
                       "monitor_interval_seconds": 0.05,
                       "supervisor": "run_native.run_reference_stage; inherited joint deadline; each ID at most once",
                       "nine_native_stages_max_wall_seconds": 2700},
        "source_dispatch": {
            "author_spex": "SubsampledSignal fixed saved Ms/Ds + exact value replay; spectralexplain.support_recovery('soft', signal,b,t)",
            "author_proxy": "spectralexplain.utils.proxy_spex((proxy_masks,proxy_values),max_order=t); read-only return-frame observation",
            "maintained_proxy": "ProxySPEX(n=n,max_order=t,index='SII',proxy_model='lightgbm',hpo=True,random_state=seed); actual proxy_model.fit, convert_tree_model, _sklearn_to_fourier, _refine",
        },
        "spex_defaults": {"decoder": "author bounded Chase depth 2*t", "regress_argument": "lasso",
                          "report": False, "actual_final_regression_calls": 0,
                          "res_energy_cutoff": 0.9, "trap_exit": True, "peel_average": True,
                          "random_state_adapter": "reset legacy RNG and consume the source C*n*b binary matrix draws before support_recovery"},
        "author_proxy_grid": {"max_depth_at_t3": [1, 3], "n_estimators": [500, 1000, 5000],
                              "learning_rate": [0.01, 0.1], "cv": 5, "scoring": "r2",
                              "declared_cv_fits": 60, "full_data_refits": 1},
        "maintained_proxy_grid": {"max_depth": [3, 5], "max_iter": [500, 1000],
                                  "learning_rate": [0.01, 0.1], "cv": 5, "scoring": "r2",
                                  "declared_cv_fits": 40, "full_data_refits": 1},
        "ridge": {"alphas": "np.logspace(-6,6,100)", "fit_intercept": False,
                  "cv": "source RidgeCV default generalized/leave-one-out CV; no new game calls",
                  "basis_selection": "source full-training proxy spectrum; not nested foldwise basis learning"},
        "prediction_tolerances": {"tree_vs_full_spectrum_atol": 1e-8, "tree_vs_full_spectrum_rtol": 1e-7,
                                  "course_vs_author_spectrum": "no required native equality; recorded source discrepancies remain explicit"},
        "audit_labels": "forbidden; predictions only; no metrics against held-out responses",
        "source_failures": "retained verbatim; no fallback estimator, tolerance relaxation or replacement spectrum",
    }
    return {"contract_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(), "contract": payload}


def load_input(path, mode):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_FIELDS:
            raise ValueError(f"NPZ fields must match the contract exactly; extra={sorted(set(archive.files)-REQUIRED_FIELDS)}, missing={sorted(REQUIRED_FIELDS-set(archive.files))}")
        data = {key: archive[key].copy() for key in archive.files}
    for name in ("schema_version", "n", "t", "b", "seed", "prompt_index"):
        value = data[name]
        if value.ndim != 0 or value.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer scalar")
        data[name] = int(value)
    n, t, b = data["n"], data["t"], data["b"]
    if data["schema_version"] != 1 or not (1 <= n <= 511 and 1 <= t <= 5 and 1 <= b <= min(n, 12)):
        raise ValueError("Invalid schema or bounded BCH dimensions")
    for name in ("spex_matrices", "spex_delays", "spex_masks", "proxy_masks", "audit_masks"):
        array = data[name]
        if array.dtype.kind not in "bui" or not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} must contain binary integers")
        data[name] = array.astype(np.int64)
    M, D = data["spex_matrices"], data["spex_delays"]
    if M.shape != (3, n, b) or D.ndim != 2 or D.shape[1] != n:
        raise ValueError("Expected spex_matrices=(3,n,b) and spex_delays=(p,n)")
    for name in ("spex_masks", "proxy_masks", "audit_masks"):
        if data[name].ndim != 2 or data[name].shape[1] != n or len(data[name]) == 0:
            raise ValueError(f"Invalid {name} shape")
    for name, masks_name in (("spex_values", "spex_masks"), ("proxy_values", "proxy_masks")):
        values = np.asarray(data[name], dtype=np.float64)
        if values.shape != (len(data[masks_name]),) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain one finite scalar per training mask")
        data[name] = values
        unique, inverse = np.unique(data[masks_name], axis=0, return_inverse=True)
        for i in range(len(unique)):
            selected = values[inverse == i]
            if not np.all(selected == selected[0]):
                raise ValueError(f"Duplicate masks in {masks_name} have inconsistent values")
    L = ((np.arange(2**b)[:, None] >> np.arange(b-1, -1, -1)) & 1)
    expected_masks = np.concatenate([((L @ matrix.T)[None, :, :] + D[:, None, :]).reshape(-1, n) % 2 for matrix in M])
    if not np.array_equal(expected_masks, data["spex_masks"]):
        raise ValueError("Supplied SPEX masks do not match matrices/delays in group-delay-subcube order")
    if mode == "native_fixed":
        if (n, t, data["seed"], data["prompt_index"]) != (28, 3, 0, 0) or b not in (3, 4, 5):
            raise ValueError("Native reference scope is prompt0,seed0,n28,t3,b3..5")
        count = 3 * 19 * 2**b
        if len(data["spex_masks"]) != count or len(data["proxy_masks"]) != count or len(data["audit_masks"]) != 512:
            raise ValueError("Native training/audit row counts differ from the frozen contract")
        if len(np.unique(data["proxy_masks"], axis=0)) != count or len(np.unique(data["audit_masks"], axis=0)) != 512:
            raise ValueError("Native proxy and audit masks must each be unique")
        training = {tuple(x) for x in np.vstack((data["spex_masks"], data["proxy_masks"]))}
        if any(tuple(x) in training for x in data["audit_masks"]):
            raise ValueError("Audit masks overlap reference training masks")
    for value in data.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return data


def spectrum_records(spectrum, n, bit_keys):
    import numpy as np
    rows = []
    for key, value in spectrum.items():
        support = tuple(int(i) for i in (np.flatnonzero(key) if bit_keys else key))
        coefficient = float(np.real(value))
        if not math.isfinite(coefficient):
            raise ValueError("Source returned a nonfinite Fourier coefficient")
        rows.append({"support": list(support), "coefficient": coefficient})
    return sorted(rows, key=lambda row: row["support"])


def predict_spectrum(masks, records):
    import numpy as np
    result = np.zeros(len(masks), dtype=np.float64)
    for row in records:
        support = row["support"]
        signs = 1 - 2 * (masks[:, support].sum(axis=1) % 2)
        result += row["coefficient"] * signs
    return result


def observe_return_frame(function, *args, names, **kwargs):
    """Read actual frame locals on normal return or exception propagation."""
    if sys.getprofile() is not None:
        raise RuntimeError("Refusing to replace an existing profiler")
    target_code = getattr(function, "__func__", function).__code__
    captured = {}
    def observer(frame, event, arg):
        if event == "return" and frame.f_code is target_code:
            captured.update({name: frame.f_locals[name] for name in names if name in frame.f_locals})
    result, failure = None, None
    sys.setprofile(observer)
    try:
        result = function(*args, **kwargs)
    except Exception:
        failure = traceback.format_exc()
    finally:
        sys.setprofile(None)
    return result, captured, failure


def model_receipt(search, model, directory):
    import numpy as np
    receipt = {"search_class": f"{type(search).__module__}.{type(search).__name__}",
               "grid": search.param_grid, "cv": search.cv, "scoring": search.scoring,
               "best_params": getattr(search, "best_params_", None),
               "best_cv_score": getattr(search, "best_score_", None)}
    if hasattr(search, "cv_results_"):
        cv = search.cv_results_
        keys = [key for key in cv if key in ("params", "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time") or key.startswith("split")]
        write_json(directory / "cv_results.json", {key: cv[key] for key in keys})
        split_keys = [key for key in cv if key.startswith("split") and key.endswith("_test_score")]
        receipt.update(cv_candidate_count=len(cv["params"]), declared_cv_fit_count=len(cv["params"])*len(split_keys),
                       finite_cv_scores=sum(int(np.isfinite(cv[key]).sum()) for key in split_keys))
    if model is not None:
        booster = model.booster_
        path = directory / "lightgbm_model.txt"
        booster.save_model(str(path))
        receipt.update(backend=f"{type(model).__module__}.{type(model).__name__}", actual_tree_count=booster.num_trees(),
                       actual_booster_iterations=booster.params.get("num_iterations"), booster_num_threads=booster.params.get("num_threads"),
                       sklearn_n_estimators=model.get_params().get("n_estimators"), max_iter_alias=model.get_params().get("max_iter"),
                       full_data_refit_completed=True, model_sha256=sha256(path))
        if receipt["backend"] != "lightgbm.sklearn.LGBMRegressor" or receipt["booster_num_threads"] != THREADS:
            raise RuntimeError("Reference backend/thread contract was not honored")
    return receipt


def execute_spex(data, directory):
    import numpy as np
    import spectralexplain
    from sparse_transform.qsft.codes.BCH import BCH
    from sparse_transform.qsft.signals.input_signal_subsampled import SubsampledSignal
    from sparse_transform.qsft.utils.general import save_data
    n, t, b = data["n"], data["t"], data["b"]
    D = np.asarray(BCH(n, t).get_delay_matrix(), dtype=int)
    if not np.array_equal(D, data["spex_delays"]):
        raise ValueError("Supplied delay matrix is not the actual author BCH delay matrix")
    cache = directory / "fixed_signal"
    cache.mkdir()
    matrices = [m.copy() for m in data["spex_matrices"]]
    delays = [[D.copy()] for _ in matrices]
    save_data((matrices, delays), cache / "Ms_and_Ds.pickle")
    offset, batches = 0, []
    def replay(masks):
        nonlocal offset
        masks = np.asarray(masks, dtype=int)
        count = len(masks)
        if not np.array_equal(masks, data["spex_masks"][offset:offset+count]):
            raise ValueError("Actual author signal requested a mask outside the exact supplied sequence")
        result = data["spex_values"][offset:offset+count].copy()
        offset += count
        batches.append(count)
        return result
    query_args = {"query_method": "complex", "num_subsample": 3, "delays_method_source": "joint-coded",
                  "subsampling_method": "qsft", "delays_method_channel": "identity-siso", "num_repeat": 1,
                  "b": b, "all_bs": [b], "t": t}
    fixed = SubsampledSignal(func=replay, n=n, q=2, query_args=query_args, folder=str(cache))
    if offset != len(data["spex_values"]):
        raise ValueError("Not all supplied SPEX values were replayed")
    np.random.seed(data["seed"])
    np.random.randint(2, size=(3, n, b))  # same RNG advance as source's independently generated matrices
    final = spectralexplain.support_recovery("soft", fixed, b, t)
    records = spectrum_records(final, n, bit_keys=True)
    predictions = {"final": predict_spectrum(data["audit_masks"], records)}
    return {"status": "COMPLETED", "spectra": {"final": records}, "predictions": predictions,
            "adapter": "source-supported saved Ms/Ds path + exact ordered replay; no public random sampler invoked",
            "replayed_logical_rows": offset, "replay_batch_rows": batches,
            "unique_training_masks": len(np.unique(data["spex_masks"], axis=0)), "native_oracle_calls": 0,
            "final_regression": "Not executed: actual author report=False early return retained"}


def execute_author_proxy(data, directory):
    import spectralexplain
    final, frame, failure = observe_return_frame(
        spectralexplain.utils.proxy_spex, (data["proxy_masks"], data["proxy_values"]),
        max_order=data["t"], names=("grid_search", "best_model", "four_dict", "four_dict_trunc", "reg", "thresh"))
    result = {"status": "SOURCE_FAILURE" if failure else "COMPLETED", "exception": failure,
              "adapter": "Actual proxy_spex fixed-samples branch; read-only profiler captures actual source return/exception locals",
              "native_oracle_calls": 0, "spectra": {}, "predictions": {}}
    model = frame.get("best_model")
    if "grid_search" in frame:
        result["fit"] = model_receipt(frame["grid_search"], model, directory)
    for name, spectrum in (("full", frame.get("four_dict")), ("selected", frame.get("four_dict_trunc")), ("final", final)):
        if spectrum is not None:
            records = spectrum_records(spectrum, data["n"], bit_keys=True)
            result["spectra"][name] = records
            result["predictions"][name] = predict_spectrum(data["audit_masks"], records)
    result["threshold"] = frame.get("thresh")
    if "reg" in frame:
        result["ridge"] = {"alpha": frame["reg"].alpha_, "fit_intercept": False, "candidate_alphas": frame["reg"].alphas}
    finish_proxy_predictions(result, model, data)
    return result


def execute_maintained_proxy(data, directory):
    from shapiq import ProxySPEX
    from shapiq.tree.conversion import convert_tree_model
    reference = ProxySPEX(n=data["n"], max_order=data["t"], index="SII", proxy_model="lightgbm", hpo=True, random_state=data["seed"])
    result = {"status": "STARTED", "adapter": "Actual maintained fit/extraction/_refine methods on supplied queries; public sampler and SII conversion bypassed",
              "native_oracle_calls": 0, "spectra": {}, "predictions": {}}
    search, model = reference.proxy_model, None
    try:
        search.fit(data["proxy_masks"], data["proxy_values"])
        model = search.best_estimator_
        tree_models = convert_tree_model(model)
        if not isinstance(tree_models, list):
            tree_models = [tree_models]
        full = reference._sklearn_to_fourier(tree_models)
        full_records = spectrum_records(full, data["n"], bit_keys=False)
        result["spectra"]["full"] = full_records
        result["predictions"]["full"] = predict_spectrum(data["audit_masks"], full_records)
        final, frame, failure = observe_return_frame(reference._refine, full, data["proxy_masks"], data["proxy_values"],
                                                     names=("four_dict_trunc", "reg", "thresh"))
        selected = frame.get("four_dict_trunc")
        if selected is not None:
            selected_records = spectrum_records(selected, data["n"], bit_keys=True)
        elif not failure:
            selected_records = full_records  # source short-spectrum/zero-energy early return
        else:
            selected_records = None
        if selected_records is not None:
            result["spectra"]["selected"] = selected_records
            result["predictions"]["selected"] = predict_spectrum(data["audit_masks"], selected_records)
        if final is not None:
            records = spectrum_records(final, data["n"], bit_keys=False)
            result["spectra"]["final"] = records
            result["predictions"]["final"] = predict_spectrum(data["audit_masks"], records)
        result.update(status="SOURCE_FAILURE" if failure else "COMPLETED", exception=failure, threshold=frame.get("thresh"))
        if "reg" in frame:
            result["ridge"] = {"alpha": frame["reg"].alpha_, "fit_intercept": False, "candidate_alphas": frame["reg"].alphas}
    except Exception:
        result.update(status="SOURCE_FAILURE", exception=traceback.format_exc())
    result["fit"] = model_receipt(search, model, directory)
    finish_proxy_predictions(result, model, data)
    return result


def finish_proxy_predictions(result, model, data):
    import numpy as np
    if model is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tree_prediction = model.predict(data["audit_masks"])
        result["predictions"]["tree"] = tree_prediction
        if "full" in result["predictions"]:
            error = result["predictions"]["full"] - tree_prediction
            result["extraction_check"] = {"max_abs_difference": float(np.max(abs(error))),
                                          "rmse": float(np.sqrt(np.mean(error**2))),
                                          "atol": 1e-8, "rtol": 1e-7,
                                          "within_tolerance": bool(np.allclose(result["predictions"]["full"], tree_prediction, atol=1e-8, rtol=1e-7))}
    full, selected = result["spectra"].get("full"), result["spectra"].get("selected")
    if full is not None and selected is not None:
        total = sum(row["coefficient"]**2 for row in full if row["support"])
        result["selection"] = {"full_count": len(full), "selected_count": len(selected),
                               "constant_retained": any(not row["support"] for row in selected),
                               "nonconstant_energy_retained": sum(row["coefficient"]**2 for row in selected if row["support"])/total if total else None}


def worker(args):
    import numpy as np
    output = Path(args.output_dir)
    identity = contract()
    if identity["contract_sha256"] != args.expected_contract:
        raise RuntimeError("Worker contract hash differs from the prospective freeze")
    data = load_input(args.input, args.mode)
    start = time.monotonic()
    receipt = {"variant": args.variant, "status": "STARTED", "contract_sha256": identity["contract_sha256"],
               "input_sha256": sha256(args.input), "mode": args.mode, "prompt_index": data["prompt_index"],
               "seed": data["seed"], "n": data["n"], "t": data["t"], "b": data["b"],
               "spex_training_rows": len(data["spex_masks"]), "proxy_training_rows": len(data["proxy_masks"]),
               "audit_mask_rows": len(data["audit_masks"]), "audit_values_received": False,
               "native_oracle_calls": 0, "cpu_affinity": sorted(os.sched_getaffinity(0))}
    training_name = "spex" if args.variant == "author_spex" else "proxy"
    training_masks = data[training_name + "_masks"]
    receipt.update(requested_training_rows=len(training_masks),
                   unique_training_masks=len(np.unique(training_masks, axis=0)),
                   training_masks_uint8_sha256=hashlib.sha256(training_masks.astype(np.uint8).tobytes()).hexdigest(),
                   training_values_float64_sha256=hashlib.sha256(data[training_name + "_values"].tobytes()).hexdigest(),
                   model_queries_performed_by_bridge=0)
    write_json(output / "result.json", receipt)
    try:
        result = {"author_spex": execute_spex, "author_proxy": execute_author_proxy,
                  "maintained_proxy": execute_maintained_proxy}[args.variant](data, output)
        predictions = result.pop("predictions", {})
        spectra = result.pop("spectra", {})
        write_json(output / "spectra.json", spectra)
        np.savez_compressed(output / "audit_predictions.npz", **predictions)
        receipt.update(result, prediction_names=sorted(predictions), spectra_names=sorted(spectra),
                       spectra_sha256=sha256(output / "spectra.json"), predictions_sha256=sha256(output / "audit_predictions.npz"))
    except Exception:
        receipt.update(status="BRIDGE_FAILURE", exception=traceback.format_exc())
    receipt["elapsed_seconds"] = time.monotonic() - start
    write_json(output / "result.json", receipt)
    return 0 if receipt["status"] in ("COMPLETED", "SOURCE_FAILURE") else 2


def run_supervised_stage(python_path, input_path, output, *, variant, b, mode, expected_contract):
    """Use the shared supervisor; preserve scientific and process statuses separately."""
    import numpy as np
    from chapter16_shapley_attribution_baselines.exercises.part18_spex.run_native import run_reference_stage
    output = Path(output).resolve()
    stage_id = f"{variant}_b{b}"
    command = [str(python_path), str(Path(__file__).resolve()), "_worker",
               "--input", str(Path(input_path).resolve()), "--output-dir", str(output),
               "--mode", mode, "--variant", variant, "--expected-contract", expected_contract]
    process = run_reference_stage(command, output, stage_id=stage_id,
                                  cwd=Path(__file__).resolve().parents[3])
    result_path = output / "result.json"
    scientific = json.loads(result_path.read_text()) if result_path.exists() else {}
    if process["status"] != "COMPLETE":
        status = "TIMEOUT" if "wall_seconds" in process["resource_violations"] else "PROCESS_FAILURE"
    else:
        status = scientific.get("status", "MISSING_RESULT")
    predictions_path = output / "audit_predictions.npz"
    spectra_path = output / "spectra.json"
    operational = {"process_receipt": str(output / "execution.json"),
               "result_path": str(result_path) if result_path.exists() else None,
               "predictions_path": str(predictions_path) if predictions_path.exists() else None,
               "spectra_path": str(spectra_path) if spectra_path.exists() else None,
               "input_path": str(Path(input_path).resolve()), "input_sha256": sha256(input_path),
               "elapsed_seconds": process["elapsed_seconds"],
               "sampled_peak_rss_bytes": process["sampled_peak_rss_bytes"],
               "source_exception_traceback": scientific.get("exception")}
    receipt = {"stage_id": stage_id, "variant": variant, "b": b, "status": status,
               "process_status": process["status"], "scientific_status": scientific.get("status"),
               "predictions_valid": process["status"] == "COMPLETE" and status in ("COMPLETED", "SOURCE_FAILURE"),
               "prediction_names": scientific.get("prediction_names", []),
               "contract_sha256": expected_contract, "returncode": process["returncode"],
               "resource_violations": process["resource_violations"], "operational": operational,
               "fallback_used": False, "audit_values_received": False, "native_oracle_calls": 0,
               "source_exception_type_message": (scientific.get("exception") or "").strip().split("\n")[-1]}
    for key in ("selection", "fit", "ridge", "threshold", "extraction_check", "adapter",
                "replayed_logical_rows", "replay_batch_rows", "requested_training_rows", "unique_training_masks",
                "training_masks_uint8_sha256", "training_values_float64_sha256", "model_queries_performed_by_bridge",
                "final_regression"):
        if key in scientific:
            receipt[key] = scientific[key]
    # Process failures retain partial files but expose no substitute prediction.
    receipt["spectra"], receipt["predictions"] = {}, {}
    if receipt["predictions_valid"]:
        receipt["spectra"] = json.loads(spectra_path.read_text())
        with np.load(predictions_path, allow_pickle=False) as archive:
            receipt["predictions"] = {key: archive[key].tolist() for key in archive.files}
    write_json(output / "bridge_stage.json", receipt)
    return receipt


def run_fixed_query_stages(output_dir, inputs, arrays, freeze):
    """Host API: supplied prompt-0/seed-0 values; no model or audit labels imported.

    ``inputs`` contains exactly six (condition, training_values) pairs, with
    methods spex/proxyspex and b=3,4,5. Query arrays use condition["key"] plus
    _masks, and for SPEX _matrices/_delays. Only common audit_masks is read.
    Freeze must contain reference_contract_sha256, reference_python_path, and
    the ordered nine reference_stage_ids. All paths in the returned receipts
    identify actual retained files; a timeout never promotes partial predictions.
    """
    import numpy as np
    expected_ids = [f"{variant}_b{b}" for b in (3, 4, 5) for variant in VARIANTS]
    if freeze["reference_stage_ids"] != expected_ids:
        raise ValueError("Freeze must name the ordered nine fixed reference stages")
    expected_contract = freeze["reference_contract_sha256"]
    if len(expected_contract) != 64 or any(c not in "0123456789abcdef" for c in expected_contract):
        raise ValueError("Invalid frozen reference contract SHA256")
    python_path = Path(freeze["reference_python_path"])
    if not python_path.is_absolute() or not python_path.is_file():
        raise ValueError("Frozen reference Python must identify the isolated environment executable")
    by_condition = {}
    for condition, values in inputs:
        method, b = condition["method"], condition["b"]
        budget = 57 * 2**b
        key = f"{method}_s0_k{budget}"
        if method not in ("spex", "proxyspex") or b not in (3, 4, 5) or condition["seed"] != 0:
            raise ValueError("Reference inputs must be prompt0/seed0 for the six selected conditions")
        if condition["key"] != key or condition["budget"] != budget or (method, b) in by_condition:
            raise ValueError("Reference condition key, budget or multiplicity differs from freeze")
        masks = arrays[key + "_masks"]
        if condition["unique_count"] != len(np.unique(masks, axis=0)):
            raise ValueError("Reference unique count differs from supplied masks")
        by_condition[method, b] = (condition, np.asarray(values, dtype=np.float64))
    if set(by_condition) != {(method, b) for method in ("spex", "proxyspex") for b in (3, 4, 5)}:
        raise ValueError("Exactly six reference training inputs are required")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    prepared = {}
    # Validate every exact input before launching the first scientific stage.
    for b in (3, 4, 5):
        spex_condition, spex_values = by_condition["spex", b]
        proxy_condition, proxy_values = by_condition["proxyspex", b]
        spex_key, proxy_key = spex_condition["key"], proxy_condition["key"]
        input_path = output_dir / f"fixed_queries_b{b}.npz"
        np.savez_compressed(input_path, schema_version=SCHEMA_VERSION, n=28, t=3, b=b, seed=0,
                            prompt_index=0, spex_matrices=arrays[spex_key + "_matrices"],
                            spex_delays=arrays[spex_key + "_delays"], spex_masks=arrays[spex_key + "_masks"],
                            spex_values=spex_values, proxy_masks=arrays[proxy_key + "_masks"],
                            proxy_values=proxy_values, audit_masks=arrays["audit_masks"])
        load_input(input_path, "native_fixed")
        prepared[b] = input_path
    receipts = []
    for b in (3, 4, 5):
        for variant in VARIANTS:
            receipts.append(run_supervised_stage(python_path, prepared[b], output_dir / f"{variant}_b{b}",
                            variant=variant, b=b, mode="native_fixed", expected_contract=expected_contract))
            write_json(output_dir / "stages.json", receipts)
    return receipts


def main():
    configure_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    describe = commands.add_parser("contract")
    describe.add_argument("--output", type=Path)
    command = commands.add_parser("_worker")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--mode", choices=("analytic", "native_fixed"), required=True)
    command.add_argument("--variant", choices=VARIANTS, required=True)
    command.add_argument("--expected-contract", required=True)
    args = parser.parse_args()
    if args.command == "contract":
        result = contract()
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
