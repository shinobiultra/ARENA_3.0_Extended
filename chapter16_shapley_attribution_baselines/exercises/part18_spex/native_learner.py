"""Portable, hash-checked native scores and explicit student estimator replay.

This module never imports an oracle or a model. Loading is read-only. A replay
passes only ordered training responses and audit *masks* to the frozen fitting
boundary, with the estimator supplied by the notebook. Audit targets are attached
after fitting. Full study outputs stay external; the portable JSON contains no
fitted estimators and no pickled or executable objects.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import native_data, native_query_plan, native_study

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "expected_outputs/native_query_fixture.json.gz"
SCHEMA = "arena.spex-native-query-fixture.v1"
BUDGETS = (456, 912, 1824)
STORED_REPLAY_CASES = ((0, 0, 1824), (1, 0, 912), (0, 0, 456))
PREDICTION_TOLERANCE = {"atol": 1e-8, "rtol": 1e-7}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _array_digest(values):
    return native_query_plan.array_record(np.asarray(values))


def _masks_from_ids(ids):
    ids = np.asarray(ids, dtype=np.uint64)
    if ids.ndim != 1 or np.any(ids >= 2**28):
        raise ValueError("Mask IDs must be 28-bit nonnegative integers.")
    shifts = np.arange(27, -1, -1, dtype=np.uint64)
    return ((ids[:, None] >> shifts) & 1).astype(np.uint8)


def _condition(plan, method, seed, budget):
    selected = [row for row in plan["conditions"]
                if (row["method"], row["seed"], row["budget"]) == (method, seed, budget)]
    if len(selected) != 1:
        raise ValueError("Choose a declared method, seed and requested-mask budget.")
    return selected[0]


def _stage_errors(row):
    if row["method"] != "proxyspex":
        return {}
    actual, predictions = row["audit_values"], row["predictions"]
    pairs = {
        "model_to_proxy_rmse": (actual, predictions["proxy"]),
        "proxy_to_extraction_rmse": (predictions["proxy"], predictions["extracted"]),
        "extraction_to_truncation_rmse": (predictions["extracted"], predictions["truncated"]),
        "truncation_to_refinement_rmse": (predictions["truncated"], predictions["refined"]),
        "model_to_extraction_rmse": (actual, predictions["extracted"]),
        "model_to_truncation_rmse": (actual, predictions["truncated"]),
        "model_to_refinement_rmse": (actual, predictions["refined"]),
    }
    return {name: float(np.sqrt(np.mean((np.asarray(a) - b) ** 2)))
            for name, (a, b) in pairs.items()}


def _compact_result(row):
    """Keep every case's measured errors, but omit fitted trees and long vectors."""
    fields = ("prompt_index", "prompt_id", "method", "seed", "budget", "condition", "query_receipt")
    result = {name: row[name] for name in fields}
    predictions = {**row["predictions"], **row["baseline_predictions"]}
    result["metrics"] = native_study.evaluate_predictions(row["audit_values"], predictions)
    result["prediction_digests"] = {name: _array_digest(values) for name, values in predictions.items()}
    if (row["prompt_index"], row["seed"], row["budget"]) in STORED_REPLAY_CASES:
        result["replay_reference_predictions"] = predictions
    result["stage_errors"] = _stage_errors(row)
    spectrum = row["spectrum"]
    ordered = sorted(spectrum, key=lambda support: (-abs(spectrum[support]), support))
    result["largest_terms"] = [{"support": list(support), "coefficient": float(spectrum[support])}
                               for support in ordered[:12]]
    result["spectrum_count"] = len(spectrum)
    result["spectrum_terms_are_truncated_for_display"] = len(spectrum) > 12
    result["record_origin"] = "recorded_complete_native_study"
    return result


def _compact_reference(row, audit_values):
    result = {name: value for name, value in row.items() if name not in ("spectra", "predictions")}
    result["metrics"] = native_study.evaluate_predictions(audit_values, row["predictions"])
    result["prediction_digests"] = {name: _array_digest(np.asarray(values))
                                    for name, values in row["predictions"].items()}
    result["record_origin"] = "recorded_prompt0_seed0_reference_stage"
    return result


def _training_score_table(payload, union_ids):
    physical = payload["oracle_values"]["physical"]
    scores = []
    for prompt_index in range(len(payload["prompts"])):
        selected = ((physical["game"] == "primary") & (physical["prompt_index"] == prompt_index))
        ids, values = physical["mask_id"][selected], physical["difference_float64"][selected]
        order = np.argsort(ids)
        ids, values = ids[order], values[order]
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Primary cached physical mask IDs must be unique within a prompt.")
        positions = np.searchsorted(ids, union_ids)
        if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], union_ids):
            raise ValueError("Native physical rows do not cover every declared training mask.")
        scores.append(values[positions])
    return np.asarray(scores, dtype=np.float64)


def _read_full_study(run_directory, binding):
    """Require caller-bound top-level bytes before following internal record hashes."""
    run_directory = Path(run_directory).resolve()
    if _sha(run_directory / "execution.json") != binding["native_execution_sha256"]:
        raise ValueError("Native execution receipt differs from the supplied binding.")
    execution = json.loads((run_directory / "execution.json").read_text())
    if (execution.get("status") != "COMPLETE" or execution.get("changed_sources") or execution.get("failures")
            or execution.get("resource_violations") or execution.get("returncode") != 0):
        raise ValueError("A completed, source-stable supervised native run is required.")
    study_directory = run_directory / "study"
    native_study.load_record(study_directory / "summary", expected_hashes=binding["summary_hashes"])
    payload = native_study.load_study(study_directory)
    if payload["freeze_sha256"] != binding["freeze_sha256"]:
        raise ValueError("The completed study is not bound to this frozen protocol.")
    plan_directory = Path(binding["query_plan_directory"])
    if _sha(plan_directory / "query_plan.json") != binding["query_plan_manifest_sha256"]:
        raise ValueError("The frozen query-plan manifest changed.")
    plan, arrays = native_query_plan.load_plan(plan_directory)
    if payload["query_plan"] != plan:
        raise ValueError("The completed study and supplied mask plan disagree.")
    database = payload["oracle_inventory"]["database"]
    database_path = study_directory / "oracle/oracle.sqlite3"
    if (_sha(database_path) != database["sha256"]
            or database_path.stat().st_size != database["size_bytes"]):
        raise ValueError("The native physical query receipt changed.")
    return {"payload": payload, "plan": plan, "arrays": arrays, "binding": dict(binding),
            "training_values": _training_score_table(payload, arrays["training_union_ids"]),
            "evidence_kind": "complete_external_study"}


def export_native_fixture(run_directory, destination, *, binding):
    """Derive exact portable numeric scores after the complete native study.

    The caller supplies completed native and fresh-reproduction hashes. This
    function does not approve or publish an artifact. The publisher seals its
    bytes with the existing bounded gzip-JSON contract.
    """
    for name in ("native_execution_sha256", "fresh_reproduction_sha256"):
        if len(binding.get(name, "")) != 64:
            raise ValueError(f"Require the exact completed {name} binding.")
    evidence = _read_full_study(run_directory, binding)
    payload, plan, arrays = evidence["payload"], evidence["plan"], evidence["arrays"]
    encoded_arrays = {}
    for name, values in arrays.items():
        if name.endswith(("_masks", "_matrices", "_delays")):
            continue
        encoded_arrays[name] = {"dtype": str(values.dtype), "shape": list(values.shape),
                                "values": values.tolist()}
    audits = [{"prompt_index": index, **_plain(payload["audit"][index])}
              for index in range(len(payload["prompts"]))]
    fixture = {
        "schema": SCHEMA, "status": "COMPLETE_NATIVE_SCORES",
        "binding": dict(binding), "source_inputs": payload["source_inputs"],
        "prompts": payload["prompts"], "query_plan": plan, "plan_arrays": encoded_arrays,
        "training_values": evidence["training_values"], "audit": audits,
        "results": [_compact_result(row) for row in payload["results"]],
        "references": [_compact_reference(row, payload["audit"][0]["audit_values"])
                       for row in payload["references"]],
        "inventory": {"prompts": 32, "conditions": 1152, "audit_rows_per_prompt": 512,
                      "training_union_per_prompt": len(arrays["training_union_ids"]),
                      "native_physical_rows": payload["oracle_inventory"]["physical_evaluations"],
                      "selected_replay": {"prompt_index": 0, "seed": 0, "budget": 1824},
                      "pointwise_replay_comparison_cases": [list(case) for case in STORED_REPLAY_CASES],
                      "replay_coverage": "all 32 prompts, both methods, all 3 seeds and all 3 budgets"},
        "omitted_from_portable_fixture": [
            "fitted trees and regression state",
            "saved prediction vectors except three predeclared default/play cases for pointwise replay checks",
            "full spectra beyond the largest 12 displayed terms per recorded condition",
            "physical batch logits and timing receipts; retained in the complete external study",
            "BCH matrices and delay arrays; student make_query_design reconstructs and checks queried masks",
        ],
    }
    # Python's float JSON encoder round-trips each finite float64 exactly.
    decoded = json.dumps(_plain(fixture), allow_nan=False, separators=(",", ":")).encode()
    from scripts.evidence_bundles import MAX_DECODED_JSON_BYTES, MAX_MEMBER_BYTES
    if len(decoded) > MAX_DECODED_JSON_BYTES:
        raise ValueError("Portable JSON exceeds the existing decoded-data bound.")
    encoded = gzip.compress(decoded, compresslevel=9, mtime=0)
    if len(encoded) > MAX_MEMBER_BYTES:
        raise ValueError("Portable gzip exceeds the existing encoded-data bound.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(encoded)
    return {"path": str(destination.resolve()), "sha256": _sha(destination),
            "size_bytes": len(encoded), "decoded_bytes": len(decoded),
            "decoded_sha256": hashlib.sha256(decoded).hexdigest(), "inventory": fixture["inventory"]}


def _decode_fixture(fixture):
    if fixture.get("schema") != SCHEMA or fixture.get("status") != "COMPLETE_NATIVE_SCORES":
        raise ValueError("Unsupported or incomplete native query fixture.")
    for name in ("native_execution_sha256", "fresh_reproduction_sha256"):
        digest = fixture["binding"].get(name, "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Completed native and reproduction identities are required.")
    prompts, plan = fixture["prompts"], fixture["query_plan"]
    if len(prompts) != 32 or len({p["id"] for p in prompts}) != 32 or plan["n"] != 28:
        raise ValueError("Require all 32 declared 28-word prompts.")
    if any(len(p["feature_names"]) != 28 for p in prompts):
        raise ValueError("Every native prompt needs 28 displayed word features.")
    expected_conditions = {(method, seed, budget)
                           for method in ("spex", "proxyspex") for seed in range(3) for budget in BUDGETS}
    if (len(plan["conditions"]) != 18
            or {(r["method"], r["seed"], r["budget"]) for r in plan["conditions"]} != expected_conditions):
        raise ValueError("Portable fixture must preserve every frozen sampling condition.")
    expected_arrays = {name for name in plan["arrays"]
                       if not name.endswith(("_masks", "_matrices", "_delays"))}
    if set(fixture["plan_arrays"]) != expected_arrays:
        raise ValueError("Portable query array inventory differs from the frozen plan.")
    arrays = {}
    for name, record in fixture["plan_arrays"].items():
        if np.dtype(record["dtype"]).kind not in "ui":
            raise ValueError("Frozen mask, permutation and fold arrays must be integer arrays.")
        values = np.asarray(record["values"], dtype=record["dtype"])
        if list(values.shape) != record["shape"] or _array_digest(values) != plan["arrays"][name]:
            raise ValueError(f"Portable query array changed: {name}")
        arrays[name] = values
    arrays["audit_masks"] = _masks_from_ids(arrays["audit_ids"])
    for condition in plan["conditions"]:
        key = condition["key"]
        arrays[key + "_masks"] = _masks_from_ids(arrays[key + "_ids"])
    for name, values in arrays.items():
        if _array_digest(values) != plan["arrays"][name]:
            raise ValueError(f"Reconstructed query array changed: {name}")
    if len(arrays["audit_ids"]) != 512 or np.isin(arrays["audit_ids"], arrays["training_union_ids"]).any():
        raise ValueError("Require 512 untouched audit masks outside all training designs.")
    scores = np.asarray(fixture["training_values"], dtype=np.float64)
    if scores.shape != (32, len(arrays["training_union_ids"])) or not np.isfinite(scores).all():
        raise ValueError("Require every finite raw training response, without broadcasting.")
    audits = {}
    for row in fixture["audit"]:
        converted = {name: np.asarray(value, dtype=np.float64) if name != "prompt_index" else value
                     for name, value in row.items()}
        if converted["audit_values"].shape != (512,) or not np.isfinite(converted["audit_values"]).all():
            raise ValueError("Audit labels must be 512 finite values.")
        audits[row["prompt_index"]] = converted
    if len(fixture["audit"]) != 32 or set(audits) != set(range(32)):
        raise ValueError("Audit rows must cover every prompt once.")
    results = fixture["results"]
    identities = [(row["prompt_index"], row["method"], row["seed"], row["budget"], row["condition"])
                  for row in results]
    expected = {(index, method, seed, budget, label) for index in range(32)
                for method, seed, budget in expected_conditions for label in ("real", "shuffled")}
    if len(results) != len(expected) or set(identities) != expected:
        raise ValueError("Recorded curves must retain all 1,152 real and shuffled cases.")
    reference_ids = [f"{variant}_b{bits}" for bits in (3, 4, 5)
                     for variant in ("author_spex", "author_proxy", "maintained_proxy")]
    if [row["stage_id"] for row in fixture["references"]] != reference_ids:
        raise ValueError("All nine declared reference-stage outcomes must remain visible.")
    for row in results:
        row["largest_spectrum"] = {tuple(term["support"]): term["coefficient"] for term in row["largest_terms"]}
        row["spectrum"] = row["largest_spectrum"]
        needs_vectors = (row["prompt_index"], row["seed"], row["budget"]) in STORED_REPLAY_CASES
        vectors = row.get("replay_reference_predictions", {})
        if needs_vectors and set(vectors) != set(row["prediction_digests"]):
            raise ValueError("The predeclared learner examples require every saved stage prediction.")
        for name, values in vectors.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (512,) or not np.isfinite(values).all():
                raise ValueError("Stored replay predictions must be 512 finite values.")
            if _array_digest(values) != row["prediction_digests"][name]:
                raise ValueError("Stored replay prediction values differ from their recorded digest.")
            vectors[name] = values
    payload = {"prompts": prompts, "results": results, "audit": audits, "query_plan": plan,
               "source_inputs": fixture["source_inputs"], "inventory": fixture["inventory"],
               "references": fixture["references"],
               "record_origin": "hash_bound_complete_native_study_summary"}
    return {"payload": payload, "plan": plan, "arrays": arrays, "training_values": scores,
            "binding": fixture["binding"], "evidence_kind": "portable_exact_query_scores"}


def load_native_evidence(path=None, *, binding=None):
    """Load the portable default without needing the author's home-directory cache.

    A full run directory is also accepted with an explicit top-level binding.
    The portable route uses the repository's existing bounded gzip-JSON contract.
    """
    path = Path(path or os.environ.get("ARENA_NATIVE_QUERY_FIXTURE", DEFAULT_FIXTURE))
    if path.is_dir():
        if binding is None:
            raise ValueError("A full external study requires an explicit receipt binding.")
        return _read_full_study(path, binding)
    from scripts.evidence_bundles import validate_json_data
    try:
        contract = validate_json_data(path)
    except (OSError, EOFError) as error:
        raise ValueError("Portable native data is missing or its gzip stream is damaged.") from error
    if binding is not None and _sha(path) != binding["fixture_sha256"]:
        raise ValueError("The portable fixture differs from the explicit binding.")
    fixture = json.loads(gzip.decompress(path.read_bytes()))
    provenance = contract["provenance"]
    if (provenance["kind"] != "native_execution_packaging"
            or fixture["binding"]["native_execution_sha256"] != provenance["source_execution"]["sha256"]):
        raise ValueError("Fixture lineage differs from the bound native execution receipt.")
    evidence = _decode_fixture(fixture)
    evidence["binding"] = {**evidence["binding"], "fixture_sha256": _sha(path),
                           "fixture_contract_sha256": _sha(str(path) + ".data-contract.json")}
    evidence["fixture_contract"] = contract
    return evidence


def training_values_for(evidence, condition, prompt_index):
    """Retrieve exact ordered training labels, preserving repeated mask requests."""
    if not 0 <= prompt_index < 32:
        raise ValueError("Choose a declared prompt index from 0 through 31.")
    arrays = evidence["arrays"]
    ids, union = arrays[condition["key"] + "_ids"], arrays["training_union_ids"]
    positions = np.searchsorted(union, ids)
    if np.any(positions >= len(union)) or not np.array_equal(union[positions], ids):
        raise ValueError("The condition requests masks outside the saved training union.")
    return evidence["training_values"][prompt_index, positions].copy()


def replay_condition(evidence, *, method, methods, prompt_index=0, seed=0, budget=1824, output_dir=None):
    """Recompute one declared real/shuffled pair with the student's actual method.

    Loading the other method's recorded metrics never fits that method. This
    explicit replay is exploratory after audit access and cannot reselect the
    prospective native study's settings.
    """
    condition = _condition(evidence["plan"], method, seed, budget)
    values = training_values_for(evidence, condition, prompt_index)
    # This boundary receives no audit labels or completed fit objects.
    rows = native_study.fit_condition(condition, evidence["arrays"], values,
                                      spex_methods=methods if method == "spex" else None,
                                      proxy_methods=methods if method == "proxyspex" else None)
    payload = evidence["payload"]
    recorded = [row for row in payload["results"]
                if (row["prompt_index"], row["seed"], row["budget"]) == (prompt_index, seed, budget)]
    comparisons = []
    for row in rows:
        row.update(prompt_index=prompt_index, prompt_id=payload["prompts"][prompt_index]["id"],
                   audit_values=payload["audit"][prompt_index]["audit_values"],
                   query_receipt={"raw_requested": budget, "unique_model_masks": condition["unique_count"],
                                  "control_reuses_training_values": row["condition"] == "shuffled"},
                   record_origin="student_replay_from_exact_native_training_scores")
        row["metrics"] = native_study.evaluate_predictions(
            row["audit_values"], {**row["predictions"], **row["baseline_predictions"]})
        row["stage_errors"] = _stage_errors(row)
        saved = next(item for item in recorded if item["method"] == method and item["condition"] == row["condition"])
        saved_metrics = saved.get("metrics") or native_study.evaluate_predictions(
            saved["audit_values"], {**saved["predictions"], **saved["baseline_predictions"]})
        reference_predictions = saved.get("replay_reference_predictions",
                                          {**saved.get("predictions", {}), **saved.get("baseline_predictions", {})})
        all_predictions = {**row["predictions"], **row["baseline_predictions"]}
        deltas = {stage: float(np.max(np.abs(prediction - np.asarray(reference_predictions[stage]))))
                  for stage, prediction in all_predictions.items() if stage in reference_predictions}
        pointwise = {stage: bool(np.allclose(prediction, reference_predictions[stage], **PREDICTION_TOLERANCE))
                     for stage, prediction in all_predictions.items() if stage in reference_predictions}
        comparisons.append({"condition": row["condition"], "recorded_metrics": saved_metrics,
                            "student_metrics": row["metrics"],
                            "maximum_prediction_absolute_delta": deltas,
                            "prediction_allclose": pointwise, "prediction_tolerance": PREDICTION_TOLERANCE,
                            "prediction_digest_matches": {
                                stage: _array_digest(prediction) == saved.get("prediction_digests", {}).get(stage)
                                for stage, prediction in row["predictions"].items()}})
    replay = {"status": "COMPLETE", "method": method, "prompt_index": prompt_index, "seed": seed,
              "budget": budget, "new_model_queries": 0, "new_physical_model_rows": 0,
              "estimator_refits": 2, "baseline_fits": 4, "requested_training_masks_per_condition": budget,
              "unique_training_masks": condition["unique_count"],
              "audit_labels_received_by_fit": False, "selection_status": "exploratory_replay_of_fixed_condition",
              "binding": evidence["binding"], "comparisons": comparisons,
              "fit_accounting": [row["fit_accounting"] for row in rows]}
    result = {"prompts": payload["prompts"], "results": rows, "audit": payload["audit"],
              "recorded_comparators": [row for row in recorded if row["method"] != method],
              "replay": replay, "query_plan": evidence["plan"]}
    if output_dir is not None:
        base = destination = Path(output_dir)
        repeat = 1
        while destination.exists():
            destination = base.with_name(base.name + f"_repeat{repeat:03d}")
            repeat += 1
        replay["output_directory"] = str(destination.resolve())
        native_study.save_record(destination, result)
    return result
