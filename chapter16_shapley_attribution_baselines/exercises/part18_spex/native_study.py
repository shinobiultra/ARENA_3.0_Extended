"""Shared fixed-query native comparison, with an explicit fit/audit boundary.

The algorithms are supplied by the learner namespace. This module owns only
study ordering, baselines, persistence and calls to the frozen model oracle.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import sys

import numpy as np

from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_data, native_query_plan

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        key = f"a{len(arrays):06d}"
        arrays[key] = value
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {"__mapping__": [[_encode(key, arrays), _encode(item, arrays)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode(item, arrays) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Do not serialize hidden fitted objects: {type(value).__name__}")


def _decode(value, arrays):
    if isinstance(value, dict):
        if "__array__" in value:
            return arrays[value["__array__"]]
        if "__tuple__" in value:
            return tuple(_decode(item, arrays) for item in value["__tuple__"])
        if "__mapping__" in value:
            return {_decode(key, arrays): _decode(item, arrays) for key, item in value["__mapping__"]}
        raise ValueError("Unknown study encoding.")
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    return value


def save_record(path, record):
    """Numeric arrays plus structured JSON, without pickle or fitted estimators."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    arrays = {}
    encoded = _encode(record, arrays)
    np.savez_compressed(path / "arrays.npz", **arrays)
    (path / "record.json").write_text(json.dumps(encoded, allow_nan=False, separators=(",", ":")) + "\n")
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in ("arrays.npz", "record.json")}
    (path / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return hashes


def load_record(path, *, expected_hashes=None):
    path = Path(path)
    hashes = json.loads((path / "hashes.json").read_text())
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError("Record no longer matches its pre-audit or summary binding.")
    if set(hashes) != {"arrays.npz", "record.json"}:
        raise ValueError("Record inventory changed.")
    for name, expected in hashes.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Study record changed: {path / name}")
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return _decode(json.loads((path / "record.json").read_text()), arrays)


def baseline_predictions(train_masks, train_values, audit_masks):
    """Query-matched constant and unregularized minimum-norm additive baselines."""
    train_masks, train_values, audit_masks = map(np.asarray, (train_masks, train_values, audit_masks))
    unique, first = np.unique(train_masks, axis=0, return_index=True)
    values = train_values[first].astype(np.float64)
    features = np.column_stack([np.ones(len(unique)), unique.astype(np.float64)])
    coefficients, _, rank, singular = np.linalg.lstsq(features, values, rcond=None)
    predictions = {
        "constant": np.full(len(audit_masks), values.mean(), dtype=np.float64),
        "additive": np.column_stack([np.ones(len(audit_masks)), audit_masks]) @ coefficients,
    }
    return predictions, {"unique_training_masks": len(unique), "coefficients": coefficients,
                         "rank": int(rank), "singular_values": singular, "fits": 2,
                         "solver": "numpy.linalg.lstsq float64 rcond=None; intercept; no regularization"}


def fit_condition(condition, arrays, train_values, *, spex_methods, proxy_methods):
    """Fit and predict WITHOUT audit labels; callable separately for learner plays."""
    key = condition["key"]
    masks, audit = arrays[key + "_masks"], arrays["audit_masks"]
    first, inverse, permutation = (arrays[key + suffix] for suffix in ("_unique_first", "_inverse", "_shuffle"))
    values = np.asarray(train_values, dtype=np.float64)
    if values.shape != (condition["budget"],) or not np.isfinite(values).all():
        raise ValueError("Require the frozen, finite ordered query values.")
    if not np.array_equal(values, values[first][inverse]):
        raise ValueError("Repeated masks have inconsistent labels.")
    records = []
    for label in ("real", "shuffled"):
        labels = values if label == "real" else values[first][permutation][inverse]
        start = time.perf_counter()
        baselines, baseline_record = baseline_predictions(masks, labels, audit)
        common = {"method": condition["method"], "seed": condition["seed"], "budget": condition["budget"],
                  "condition": label, "baseline_predictions": baselines, "baseline_fit": baseline_record}
        if condition["method"] == "spex":
            design = spex_methods.make_query_design(n=28, t=3, b=condition["b"], C=3, seed=condition["seed"])
            if not np.array_equal(design.masks, masks):
                raise ValueError("Student SPEX masks differ from the frozen query plan.")
            fit = spex_methods.recover_spectrum(design, labels)
            prediction = spex_methods.predict_spectrum(audit, fit.spectrum)
            compact_trace = [{"iteration": step["iteration"], "energy_before": step["energy_before"],
                              "energy_after": step["energy_after"], "updates": step["updates"],
                              "accepted_bins": len(step["events"]), "rejected_bins": step["rejected_bins"],
                              "zero_bins": step["zero_bins"], "support_count": len(step["spectrum"])}
                             for step in fit.trace]
            common.update({"spectrum": fit.spectrum, "predictions": {"final": prediction},
                           "recovery": {"trace": compact_trace, "initial_bins": fit.initial_bins,
                                        "residual_bins": fit.residual_bins, "stop_reason": fit.stop_reason,
                                        "residual_energy": fit.residual_energy},
                           "fit_accounting": {"spex_recoveries": 1, "oracle_calls_by_methods": 0}})
        else:
            fit = proxy_methods.fit_proxy(masks, labels, fold_seed=condition["seed"])
            if len(fit["folds"]) != 3:
                raise ValueError("Exactly three frozen Proxy CV folds are required.")
            for fold_index, fold in enumerate(fit["folds"]):
                for field, suffix in (("train_indices", "train"), ("validation_indices", "validation")):
                    if not np.array_equal(fold[field], arrays[f"{key}_fold{fold_index}_{suffix}"]):
                        raise ValueError("Proxy CV folds changed after the prospective freeze.")
            predictions = {"proxy": proxy_methods.predict_boosted_trees(fit["model"], audit)}
            for stage, field in (("extracted", "full_spectrum"), ("truncated", "truncated_spectrum"),
                                 ("refined", "refined_spectrum")):
                predictions[stage] = proxy_methods.predict_spectrum(audit, fit[field])
            np.testing.assert_allclose(predictions["proxy"], predictions["extracted"], atol=1e-8, rtol=1e-7,
                                       err_msg="Exact tree extraction failed; retain this run.")
            common.update({"spectrum": fit["refined_spectrum"], "predictions": predictions,
                           "fit": {name: value for name, value in fit.items() if name != "model"},
                           "fit_accounting": fit["accounting"]})
        for prediction in (*common["predictions"].values(), *baselines.values()):
            if np.asarray(prediction).shape != (len(audit),) or not np.isfinite(prediction).all():
                raise ValueError("Require one finite fitted prediction per audit row, without broadcasting.")
        common["fit_seconds"] = time.perf_counter() - start
        records.append(common)
    return records


def evaluate_predictions(values, predictions):
    """Descriptive held-out error; no outcome-dependent passing threshold."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Audit values must be a nonempty finite vector.")
    variance = float(np.mean((values - values.mean()) ** 2))
    result = {}
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction)
        if prediction.shape != values.shape or not np.isfinite(prediction).all():
            raise ValueError("Prediction and audit vectors must have exactly equal shapes and finite values.")
        mse = float(np.mean((np.asarray(prediction) - values) ** 2))
        result[name] = {"rmse": float(np.sqrt(mse)), "nrmse": float(np.sqrt(mse / variance)) if variance else None,
                        "r2": 1 - mse / variance if variance else None}
    return result


def validate_freeze(path):
    """Check the original scientific approval or its narrowly bound cap amendment."""
    import ast
    from collections import Counter
    from datetime import datetime
    import sqlite3

    def file_hash(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def linked_json(path, expected, label):
        path = Path(path)
        if file_hash(path) != expected:
            raise ValueError(f"{label} bytes changed.")
        return json.loads(path.read_text())

    def approved(candidate_record, status):
        if candidate_record.get("status") != "APPROVED" or not candidate_record.get("independent_review_sha256"):
            raise ValueError("An actual source-bound independent protocol approval is required.")
        review = linked_json(candidate_record["independent_review_path"],
                             candidate_record["independent_review_sha256"], "Independent review")
        candidate = linked_json(candidate_record["candidate_manifest_path"],
                                candidate_record["candidate_manifest_sha256"], "Candidate manifest")
        if (review.get("candidate_manifest_sha256") != candidate_record["candidate_manifest_sha256"]
                or review.get("status") != status):
            raise ValueError("Independent approval does not cover this exact native candidate.")
        if any(candidate_record.get(key) != value for key, value in candidate.items()):
            raise ValueError("Approved scientific or execution fields changed.")
        return candidate

    record = json.loads(Path(path).read_text())
    amendment = record.get("freeze_kind") == "RESOURCE_ONLY_AMENDMENT"
    if record.get("freeze_kind") not in (None, "RESOURCE_ONLY_AMENDMENT"):
        raise ValueError("Unknown native freeze kind.")
    candidate = approved(record, "APPROVE_RESOURCE_ONLY_AMENDMENT" if amendment else "APPROVE_NATIVE_PROTOCOL")
    if record.get("native_queries_before_freeze") != 0 or record.get("native_fits_before_freeze") != 0:
        raise ValueError("The original scientific freeze must predate native outcomes.")
    for name, expected in record["source_inputs"].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or file_hash(source) != expected:
            raise ValueError(f"Frozen native source changed: {name}")
    if amendment:
        envelope_fields = {"status", "candidate_manifest_path", "candidate_manifest_sha256",
                           "independent_review_path", "independent_review_sha256", "approved_envelope_written_at_utc"}
        if set(record) - envelope_fields != set(candidate):
            raise ValueError("Every amendment field must be bound by its independently approved candidate.")
        original = linked_json(record["original_scientific_freeze_path"],
                               record["original_scientific_freeze_sha256"], "Original scientific freeze")
        approved(original, "APPROVE_NATIVE_PROTOCOL")
        if (original.get("freeze_kind") is not None or record.get("counts_scope") != "original_scientific_freeze"
                or original.get("native_queries_before_freeze") != 0 or original.get("native_fits_before_freeze") != 0):
            raise ValueError("Legacy zero counts must refer to the original, unamended scientific freeze.")
        extra_fields = {"freeze_kind", "counts_scope", "original_scientific_freeze_path",
                        "original_scientific_freeze_sha256", "prior_execution_path", "prior_execution_sha256",
                        "prior_interruption_inventory_path", "prior_interruption_inventory_sha256",
                        "native_queries_before_amendment", "native_fits_before_amendment",
                        "native_fits_before_amendment_units"}
        if set(record) != set(original) | extra_fields:
            raise ValueError("Resource amendment field inventory differs from the explicit schema.")
        mutable = envelope_fields | {"created_at_utc", "source_inputs", "resources", "native_execution_started"}
        if any(record.get(key) != value for key, value in original.items() if key not in mutable):
            raise ValueError("Resource amendment changed an original scientific field.")
        if original["resources"]["wall_seconds"] != 5400:
            raise ValueError("The historical joint cap must be 5400 seconds.")
        if record["resources"] != {**original["resources"], "wall_seconds": 10800}:
            raise ValueError("Only the joint wall cap may change, from 5400 to 10800 seconds.")
        if record.get("native_execution_started") is not True:
            raise ValueError("The resource envelope must acknowledge prior native execution.")
        inventory = linked_json(record["prior_interruption_inventory_path"],
                                record["prior_interruption_inventory_sha256"], "Prior interruption inventory")
        execution = linked_json(record["prior_execution_path"], record["prior_execution_sha256"], "Prior execution")
        prior = Path(record["prior_execution_path"]).resolve().parent
        # Each historical link is physically checked, not accepted as a label.
        for label in ("execution", "original_scientific_freeze", "oracle_database", "child_log", "runtime_basis"):
            item = inventory[label]
            actual = Path(item["path"])
            if actual.stat().st_size != item["size_bytes"] or file_hash(actual) != item["sha256"]:
                raise ValueError(f"Prior interruption evidence changed: {label}")
        if (inventory["execution"]["path"] != str(Path(record["prior_execution_path"]))
                or inventory["execution"]["sha256"] != record["prior_execution_sha256"]
                or inventory["original_scientific_freeze"]["path"] != record["original_scientific_freeze_path"]
                or inventory["original_scientific_freeze"]["sha256"] != record["original_scientific_freeze_sha256"]):
            raise ValueError("Prior inventory belongs to a different execution or scientific freeze.")
        original_path = str(Path(record["original_scientific_freeze_path"]).resolve())
        if (execution["fixed_inputs"]["freeze_path"] != original_path
                or execution["source_inputs"].get(original_path, {}).get("sha256") != record["original_scientific_freeze_sha256"]
                or execution["fixed_inputs"]["plan_dir"] != original["query_plan_directory"]
                or execution["fixed_inputs"]["snapshot_path"] != original["model_snapshot_path"]
                or execution["joint_limits"]["wall_seconds"] != 5400):
            raise ValueError("The interrupted execution must have used the original frozen scientific inputs and cap.")
        if (execution.get("status") != "FAIL" or execution.get("returncode") != -2
                or execution.get("changed_sources") or execution.get("resource_violations")
                or inventory["status"] != "INTERRUPTED_RESOURCE_FEASIBILITY"
                or inventory["partial_values_or_fits_will_be_reused_for_new_cold_execution"] is not False
                or inventory["proposed_joint_wall_seconds"] != 10800):
            raise ValueError("The retained prior execution must be the unchanged-source SIGINT interruption.")
        times = [datetime.fromisoformat(value) for value in (
            original["created_at_utc"], execution["finished_at_utc"], inventory["recorded_at_utc"], record["created_at_utc"])]
        if any(value.tzinfo is None for value in times) or times != sorted(times) or times[2] == times[3]:
            raise ValueError("The amendment creation time must follow the retained interrupted execution and inventory.")
        relative = "chapter16_shapley_attribution_baselines/exercises/part18_spex/"
        allowed = {relative + name for name in ("run_native.py", "native_study.py", "native_protocol.md")}
        if set(record["source_inputs"]) != set(original["source_inputs"]):
            raise ValueError("Resource amendment changed the native source inventory.")
        changed = {name for name in record["source_inputs"] if record["source_inputs"][name] != original["source_inputs"][name]}
        if changed != allowed:
            raise ValueError("Only the supervisor, freeze validator and protocol appendix may change.")
        old_sources = {}
        for name, expected in original["source_inputs"].items():
            old_path = prior / "source_snapshot" / name
            receipt_entry = execution["source_inputs"].get(str((ROOT / name).resolve()), {})
            if file_hash(old_path) != expected or receipt_entry.get("sha256") != expected:
                raise ValueError(f"Historical native source is not bound to the original freeze: {name}")
            if name in allowed:
                old_sources[name] = old_path.read_text()

        class DoubleJointCap(ast.NodeTransformer):
            count = 0

            def visit_Constant(self, node):
                if type(node.value) in (int, float) and node.value == 5400:
                    node.value *= 2
                    self.count += 1
                return node

        transform = DoubleJointCap()
        old_runner = transform.visit(ast.parse(old_sources[relative + "run_native.py"]))
        new_runner = ast.parse((ROOT / relative / "run_native.py").read_text())
        if transform.count != 4 or ast.dump(old_runner) != ast.dump(new_runner):
            raise ValueError("Supervisor AST changed beyond the four joint wall-cap substitutions.")
        trees = [ast.parse(text) for text in (old_sources[relative + "native_study.py"],
                                             (ROOT / relative / "native_study.py").read_text())]
        for tree in trees:
            validators = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_freeze"]
            if len(validators) != 1:
                raise ValueError("Require exactly one freeze-validation function.")
            validators[0].body = [ast.Pass()]
        if ast.dump(trees[0]) != ast.dump(trees[1]):
            raise ValueError("Study AST changed outside the freeze-validation body.")
        old_protocol = old_sources[relative + "native_protocol.md"]
        protocol = (ROOT / relative / "native_protocol.md").read_text()
        appendix = "\n\n## Resource-only execution amendment: 180-minute joint cap\n"
        if not protocol.startswith(old_protocol + appendix):
            raise ValueError("The original protocol must be an exact prefix of its explicit resource appendix.")
        database = Path(inventory["oracle_database"]["path"]).resolve()
        if database != prior / "study/oracle/oracle.sqlite3":
            raise ValueError("Prior inventory must bind the interrupted execution's own oracle database.")
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            table_counts = {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                            for name in ("calls", "batches", "physical_rows", "cache")}
            batch_status = dict(connection.execute("SELECT status,COUNT(*) FROM batches GROUP BY status"))
            call_status = dict(connection.execute("SELECT json_extract(receipt,'$.status'),COUNT(*) FROM calls GROUP BY 1"))
            call_stages = dict(connection.execute("SELECT json_extract(receipt,'$.stage'),COUNT(*) FROM calls GROUP BY 1"))
            attempted = connection.execute("SELECT SUM(json_extract(identity,'$.rows')) FROM batches").fetchone()[0]
        if (table_counts != inventory["table_counts"] or batch_status != inventory["batch_status_counts"]
                or call_status != inventory["call_status_counts"] or call_stages != inventory["call_stages"]
                or attempted != inventory["attempted_physical_rows_from_started_batches"]
                or table_counts["physical_rows"] != inventory["completed_physical_rows"]
                or table_counts["cache"] != inventory["completed_cached_rows"]):
            raise ValueError("Prior query counts differ from the retained physical database.")
        if (set(call_stages) != {"invariance", "training"} or inventory["audits_saved"] != 0
                or inventory["reference_stage_records"] != 0 or (prior / "study/audit").exists()
                or (prior / "study/references").exists()):
            raise ValueError("The interrupted envelope must precede every audit and reference outcome.")
        accounting = Counter()
        identities, paths, seconds = [], [], 0.0
        for entry in inventory["saved_conditions"]:
            condition_path = (prior / entry["path"]).resolve()
            if not condition_path.is_relative_to(prior / "study/conditions"):
                raise ValueError("A retained fit must belong to the interrupted study.")
            if json.loads((condition_path / "hashes.json").read_text()) != entry["hashes"]:
                raise ValueError("Prior condition hash binding changed.")
            for name, expected in entry["hashes"].items():
                if name not in {"arrays.npz", "record.json"} or file_hash(condition_path / name) != expected:
                    raise ValueError("A prior saved fit changed after the interruption inventory.")
            if set(entry["hashes"]) != {"arrays.npz", "record.json"}:
                raise ValueError("Incomplete retained fit payload inventory.")
            fields = dict(json.loads((condition_path / "record.json").read_text())["__mapping__"])
            identity = [fields[name] for name in ("prompt_index", "method", "seed", "budget", "condition")]
            if identity != entry["identity"]:
                raise ValueError("Prior saved fit identity changed.")
            identities.append(tuple(identity))
            paths.append(condition_path)
            work = _decode(fields["fit_accounting"], {})
            for key in ("oracle_calls_by_methods", "ensemble_fits", "residual_tree_fits", "ridge_candidate_fits",
                        "ridge_matrix_decompositions", "spex_recoveries"):
                accounting[key] += work.get(key, 0)
            accounting["baseline_fits"] += dict(fields["baseline_fit"]["__mapping__"])["fits"]
            seconds += fields["fit_seconds"]
        actual_paths = {path.parent.resolve() for path in (prior / "study/conditions").glob("*/hashes.json")}
        if (len(set(identities)) != len(identities) or set(paths) != actual_paths
                or len(paths) != len(set(paths)) or dict(accounting) != inventory["saved_fit_accounting"]
                or not np.isclose(seconds, inventory["saved_fit_seconds"], rtol=0, atol=1e-9)
                or len(identities) != inventory["native_fits_before_amendment"]):
            raise ValueError("Prior saved-fit counts or physical inventory changed.")
        units = "completed, saved method-condition fits; lower bound on attempted work, not an exact attempted fit count"
        if (type(record["native_queries_before_amendment"]) is not int or record["native_queries_before_amendment"] <= 0
                or record["native_queries_before_amendment"] != table_counts["physical_rows"]
                or type(record["native_fits_before_amendment"]) is not int or record["native_fits_before_amendment"] <= 0
                or record["native_fits_before_amendment"] != len(identities)
                or record["native_fits_before_amendment_units"] != units
                or inventory["native_fits_before_amendment_units"] != units):
            raise ValueError("Amendment counts must disclose actual completed queries and saved fits with explicit units.")
    validate_environment(record)
    return record

def environment_snapshot():
    return {"python": sys.version, "executable": sys.executable,
            "versions": {distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
                         for distribution in importlib.metadata.distributions()}}


def validate_environment(record):
    """The actual running environment must equal the pre-native environment."""
    path = Path(record["main_environment_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["main_environment_sha256"]:
        raise ValueError("The frozen main environment file changed.")
    if environment_snapshot() != json.loads(path.read_text()):
        raise ValueError("The actual Python executable/version or installed distributions differ from the freeze.")


def validate_inventory(output_dir, conditions, audits, references, plan, inventory):
    """Reconcile observed saved fits/queries; never copy planned counts as evidence."""
    if len(conditions) != 1152 or len(audits) != 32 or len(references) != 9:
        raise ValueError("Incomplete condition, audit or fixed-reference inventory.")
    audit_ids = [load_record(Path(output_dir) / entry["path"], expected_hashes=entry["hashes"])["prompt_index"]
                 for entry in audits]
    if sorted(audit_ids) != list(range(32)):
        raise ValueError("Audit records must cover every prompt exactly once.")
    expected_reference_ids = [f"{variant}_b{bits}" for bits in (3, 4, 5)
                              for variant in ("author_spex", "author_proxy", "maintained_proxy")]
    if [row["stage_id"] for row in references] != expected_reference_ids:
        raise ValueError("Reference stages differ from the prospective ordered inventory.")
    if any(row["audit_values_received"] or row["native_oracle_calls"] or row["fallback_used"] for row in references):
        raise ValueError("Reference fitting crossed the frozen query/audit boundary.")
    observed = {"spex_recoveries": 0, "proxy_pipelines": 0, "baseline_fits": 0,
                "proxy_ensemble_fits": 0, "ridge_candidate_fits": 0,
                "ridge_matrix_decompositions": 0, "proxy_residual_tree_fits": 0}
    identities = set()
    for entry in conditions:
        row = load_record(Path(output_dir) / entry["path"], expected_hashes=entry["hashes"])
        identity = (row["prompt_index"], row["method"], row["seed"], row["budget"], row["condition"])
        if identity in identities:
            raise ValueError("Duplicate saved method condition.")
        identities.add(identity)
        if row["baseline_fit"]["fits"] != 2:
            raise ValueError("Both query-matched baselines must be fitted in every condition.")
        observed["baseline_fits"] += 2
        accounting = row["fit_accounting"]
        if accounting["oracle_calls_by_methods"] != 0:
            raise ValueError("An estimator made undeclared model queries.")
        if row["method"] == "spex":
            if accounting["spex_recoveries"] != 1:
                raise ValueError("Each SPEX condition requires one complete recovery.")
            observed["spex_recoveries"] += 1
        else:
            observed["proxy_pipelines"] += 1
            expected_trees = 3840 + row["fit"]["proxy_cv"]["selected_parameters"]["n_estimators"]
            expected = {"ensemble_fits": 13, "ridge_candidate_fits": 16,
                        "ridge_matrix_decompositions": 4, "residual_tree_fits": expected_trees}
            for name, count in expected.items():
                if accounting[name] != count:
                    raise ValueError(f"Proxy fit inventory differs from its frozen grid: {name}")
            for target, source in (("proxy_ensemble_fits", "ensemble_fits"),
                                   ("ridge_candidate_fits", "ridge_candidate_fits"),
                                   ("ridge_matrix_decompositions", "ridge_matrix_decompositions"),
                                   ("proxy_residual_tree_fits", "residual_tree_fits")):
                observed[target] += accounting[source]
    expected_identities = {(prompt, row["method"], row["seed"], row["budget"], condition)
                           for prompt in range(32) for row in plan["conditions"] for condition in ("real", "shuffled")}
    if identities != expected_identities:
        raise ValueError("Saved fits do not cover the complete prospective grid.")
    for name, value in observed.items():
        if name != "proxy_residual_tree_fits" and value != plan[name]:
            raise ValueError(f"Incomplete observed fit inventory: {name}")
    if not plan["proxy_residual_tree_fits_min"] <= observed["proxy_residual_tree_fits"] <= plan["proxy_residual_tree_fits_max"]:
        raise ValueError("Observed tree fits lie outside the fixed candidate grid.")
    physical = plan["primary_training_and_audit_rows"] + plan["counterfactual_rows"] + 15
    if inventory["physical_evaluations"] != physical or inventory["attempted_physical_evaluations"] != physical:
        raise ValueError("Physical native queries differ from the complete planned inventory.")
    if inventory["cached_unique_masks"] != plan["primary_training_and_audit_rows"] + plan["counterfactual_rows"]:
        raise ValueError("The primary/counterfactual cache contains missing or undeclared masks.")
    if any(batch["status"] != "PASS" for batch in inventory["batches"]):
        raise ValueError("A native model batch did not complete.")
    observed.update(physical_evaluations=physical, completed_condition_records=len(conditions),
                    audit_records=len(audits), reference_stage_records=len(references))
    return observed


def run_study(output_dir, *, spex_methods, proxy_methods, freeze_path=None, plan_dir=None, snapshot_path=None):
    """Execute the complete joint study from a cold cache, using learner methods.

    No audit labels are requested until ALL primary/reference fits and all audit
    predictions have been saved. A completed run is read with load_study; this
    function never silently resumes an incomplete or mismatched old run.
    """
    if os.environ.get("ARENA_SPEX_SUPERVISED_CHILD") != "1":
        raise ValueError("Run the native study through run_native.run_native so its complete resource caps are enforced.")
    from transformers import AutoTokenizer
    from chapter16_shapley_attribution_baselines.exercises.part18_spex.native_oracle import Oracle

    freeze_path = Path(freeze_path or HERE / "native_freeze.json")
    freeze = validate_freeze(freeze_path)
    plan_dir = Path(plan_dir or freeze["query_plan_directory"])
    if hashlib.sha256((plan_dir / "query_plan.json").read_bytes()).hexdigest() != freeze["query_plan_manifest_sha256"]:
        raise ValueError("Prospective query plan identity changed.")
    plan, arrays = native_query_plan.validate_student_plan(plan_dir, spex_methods, proxy_methods)
    snapshot_path = Path(snapshot_path or freeze["model_snapshot_path"])
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    prompts = native_data.prompt_records()
    tokenized = []
    for prompt in prompts:
        counterfactual = dict(prompt, prompt=prompt["counterfactual_prompt"])
        tokenized.append({**prompt, **native_data.tokenize_record(prompt, tokenizer), "feature_names": prompt["words"],
                          "counterfactual": native_data.tokenize_record(counterfactual, tokenizer)})
    oracle = Oracle(output_dir / "oracle", tokenized, snapshot_path, freeze_path=freeze_path,
                    student_mask_inputs=getattr(spex_methods, "masked_word_inputs", native_data.masked_word_inputs))
    oracle.validate_invariance()
    saved_conditions, reference_inputs = [], []
    for prompt_index, prompt in enumerate(prompts):
        for condition in plan["conditions"]:
            key = condition["key"]
            values = oracle.query(prompt_index, arrays[key + "_masks"], stage="training", method=condition["method"],
                                  seed=condition["seed"], budget=condition["budget"])
            records = fit_condition(condition, arrays, values, spex_methods=spex_methods, proxy_methods=proxy_methods)
            for record in records:
                record.update({"prompt_index": prompt_index, "prompt_id": prompt["id"],
                               "query_receipt": {"raw_requested": condition["budget"],
                                                 "unique_model_masks": condition["unique_count"],
                                                 "control_reuses_training_values": record["condition"] == "shuffled"}})
                relative = f"conditions/p{prompt_index:02d}_{key}_{record['condition']}"
                hashes = save_record(output_dir / relative, record)
                saved_conditions.append({"path": relative, "hashes": hashes})
            if prompt_index == 0 and condition["seed"] == 0:
                reference_inputs.append((condition, values.copy()))
        print(f"Training and prediction saved: prompt {prompt_index + 1}/32", flush=True)
    references = run_references(output_dir / "references", reference_inputs, arrays, freeze)
    boundary_hashes = save_record(output_dir / "prediction_boundary",
                                 {"condition_records": saved_conditions, "references": references,
                                  "audit_model_queries_so_far": 0})
    audit_records = []
    for prompt_index, prompt in enumerate(prompts):
        values = oracle.query(prompt_index, arrays["audit_masks"], stage="held_out_audit", method="evaluation", seed=7719,
                              budget=len(arrays["audit_masks"]))
        unmasked = oracle.query(prompt_index, np.ones((1, 28), dtype=np.uint8), stage="primary_sign", method="control",
                                seed=0, budget=1)
        counterfactual = oracle.counterfactual(prompt_index)
        record = {"prompt_index": prompt_index, "audit_values": values, "unmasked": unmasked,
                  "counterfactual": counterfactual}
        relative = f"audit/p{prompt_index:02d}"
        hashes = save_record(output_dir / relative, record)
        audit_records.append({"path": relative, "hashes": hashes})
    inventory = oracle.inventory()
    observed = validate_inventory(output_dir, saved_conditions, audit_records, references, plan, inventory)
    oracle_value_hashes = save_record(output_dir / "oracle_values", oracle.export_values())
    summary = {"status": "COMPLETE", "prompts": [{**prompt, "feature_names": prompt["words"]} for prompt in prompts],
               "conditions": saved_conditions, "references": references, "query_plan": plan,
               "oracle_inventory": inventory, "oracle_receipts": oracle.receipts, "observed_inventory": observed,
               "oracle_value_hashes": oracle_value_hashes,
               "source_inputs": freeze["source_inputs"], "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
               "audit_records": audit_records, "prediction_boundary_hashes": boundary_hashes}
    validate_freeze(freeze_path)
    save_record(output_dir / "summary", summary)
    oracle.close()
    return load_study(output_dir)


def run_references(output_dir, inputs, arrays, freeze):
    """Fixed-query bridge contract is supplied by the independently tested adapter."""
    from chapter16_shapley_attribution_baselines.exercises.part18_spex import reference_bridge
    return reference_bridge.run_fixed_query_stages(output_dir, inputs, arrays, freeze)


def load_study(output_dir):
    output_dir = Path(output_dir)
    summary = load_record(output_dir / "summary")
    if summary["status"] != "COMPLETE":
        raise ValueError("Native study is incomplete.")
    boundary = load_record(output_dir / "prediction_boundary", expected_hashes=summary["prediction_boundary_hashes"])
    if boundary["condition_records"] != summary["conditions"]:
        raise ValueError("Condition predictions changed across the audit boundary.")
    if boundary["references"] != summary["references"]:
        raise ValueError("Reference predictions changed across the audit boundary.")
    audit_rows = [load_record(output_dir / entry["path"], expected_hashes=entry["hashes"])
                  for entry in summary["audit_records"]]
    audits = {row["prompt_index"]: row for row in audit_rows}
    results = []
    for entry in summary["conditions"]:
        row = load_record(output_dir / entry["path"], expected_hashes=entry["hashes"])
        row["audit_values"] = audits[row["prompt_index"]]["audit_values"]
        row["metrics"] = evaluate_predictions(row["audit_values"], {**row["predictions"], **row["baseline_predictions"]})
        results.append(row)
    oracle_values = load_record(output_dir / "oracle_values", expected_hashes=summary["oracle_value_hashes"])
    return {**summary, "results": results, "audit": audits, "oracle_values": oracle_values}
