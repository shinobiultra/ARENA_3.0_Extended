"""Read-only fixture checks for the explicitly published native CPU lessons.

These small JSON receipts replace legacy smoke/reference fixtures only in the
listed sections. This gate checks their contents and report bindings; the full
report audit and section runners still own source freshness and array replay.
No models, cached tensors or fitted artifacts are loaded here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence


CHAPTER = "chapter16_shapley_attribution_baselines/exercises"
# Explicit migration list: the presence of a receipt must never opt an unrelated
# lesson out of its established fixture requirements.
NATIVE_RECEIPT_SECTIONS = {
    "part9_influence_and_tracin": ("16_9_influence_functions_and_tracin", "reference_refits", "reference_refits"),
    "part10_datamodels": ("16_10_datamodels", "classifier_refits", "reference_refits"),
    "part11_dataset_cartography": ("16_11_dataset_cartography", "main_training_runs", "reference_refits"),
    "part12_sage_global_importance": ("16_12_sage_global_importance", "training_runs", "reference_refits"),
    "part13_treeshap": ("16_13_treeshap_from_first_principles", "model_fits", "model_fits"),
    "part14_deepshap": ("16_14_deepshap_when_does_fast_attribution_match_exact_shap", "model_fits", "model_fits"),
    "part15_gradientshap": ("16_15_expected_gradients_which_expectation_does_gradientshap_estimate", "model_fits", "model_fits"),
    "part16_shapley_taylor_faith_shap": ("16_16_where_does_higher_order_credit_go", "model_fits", "model_fits"),
    "part17_shapiq_estimators": ("16_17_one_query_many_interactions", "models", "model_fits"),
}
NATIVE_FIXTURE_NAMES = ("native_summary.json", "learner_execution.json", "fresh_reproduction.json")
LEGACY_FIXTURE_NAMES = ("smoke_test.json", "reference_metrics.json")
TENSOR_RECEIPT_SECTIONS = {
    "part9_influence_and_tracin", "part10_datamodels", "part11_dataset_cartography",
    "part12_sage_global_importance",
}


def uses_native_receipts(root: Path, exercise_dir: Path) -> bool:
    relative = exercise_dir.relative_to(root)
    return relative.parent.as_posix() == CHAPTER and relative.name in NATIVE_RECEIPT_SECTIONS


def fixture_paths(root: Path, exercise_dir: Path) -> list[Path]:
    names = NATIVE_FIXTURE_NAMES if uses_native_receipts(root, exercise_dir) else LEGACY_FIXTURE_NAMES
    return [exercise_dir / "expected_outputs" / name for name in names]


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _source_hashes(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(path, str) and bool(path.strip())
        and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
        for path, digest in value.items()
    )


def native_receipt_blockers(
    root: Path, exercise_dir: Path, lock: dict[str, Any], report: dict[str, Any]
) -> list[str]:
    """Validate an accepted native triplet without executing or fitting anything."""
    if not uses_native_receipts(root, exercise_dir):
        return ["section has not migrated from the legacy fixture contract"]
    notebook_id, summary_fit_key, report_fit_key = NATIVE_RECEIPT_SECTIONS[exercise_dir.name]
    blockers: list[str] = []
    if lock.get("notebook_id") != notebook_id or report.get("notebook_id") != notebook_id:
        blockers.append("native receipt section identity does not match lock/report")
    if report.get("accepted") is not True or report.get("tests_passed") is not True:
        blockers.append("native report must be accepted with passing tests")
    evidence = report.get("metrics", {})
    evidence = evidence.get("gpu_evidence", {}) if isinstance(evidence, dict) else {}
    if not isinstance(evidence, dict) or evidence.get("placeholder_only") is True:
        blockers.append("native report cannot use placeholder evidence")
    try:
        supported = has_native_cpu_model_evidence(lock, report)
    except (AttributeError, KeyError, TypeError, ValueError):
        supported = False
    if not supported:
        blockers.append("report lacks the explicit native CPU model evidence contract")

    manifest = report.get("report_inputs", {})
    if not isinstance(manifest, dict) or manifest.get("algorithm") != "sha256":
        return blockers + ["native receipts require a sha256 report manifest"]
    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        return blockers + ["report manifest files must be a list"]
    receipts: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for path in fixture_paths(root, exercise_dir):
        try:
            raw = path.read_bytes()
            receipt = json.loads(raw)
        except (OSError, ValueError, UnicodeError) as exc:
            blockers.append(f"{path.name}: missing or invalid JSON ({type(exc).__name__})")
            continue
        if not isinstance(receipt, dict) or not receipt:
            blockers.append(f"{path.name}: receipt must be a nonempty object")
            continue
        relative = path.relative_to(root).as_posix()
        matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("path") == relative]
        digest = hashlib.sha256(raw).hexdigest()
        if len(matches) != 1:
            blockers.append(f"{path.name}: requires exactly one report input binding")
        elif matches[0].get("sha256") != digest or matches[0].get("size_bytes") != len(raw):
            blockers.append(f"{path.name}: report input hash/size does not match receipt")
        receipts[path.name] = receipt
        digests[path.name] = digest
    if len(receipts) != len(NATIVE_FIXTURE_NAMES):
        return blockers

    summary, learner, fresh = (receipts[name] for name in NATIVE_FIXTURE_NAMES)
    metrics = report.get("metrics", {})
    measured = metrics.get("gpu_test", {}) if isinstance(metrics, dict) else {}
    if not isinstance(measured, dict):
        measured = {}
    for name, key in (("learner_execution.json", "learner_evidence_sha256"),
                      ("fresh_reproduction.json", "fresh_evidence_sha256")):
        if measured.get(key) != digests[name]:
            blockers.append(f"{name}: measured evidence hash does not match receipt")
    if not _positive_integer(summary.get(summary_fit_key)) or summary.get(summary_fit_key) != measured.get(report_fit_key):
        blockers.append("native_summary.json: model fitting count does not match native report")

    if learner.get("status") != "PASS" or learner.get("completed_from_own_answers") is not True:
        blockers.append("learner_execution.json: complete own-answer execution is required")
    learner_sources, fresh_sources = learner.get("source_inputs"), fresh.get("source_inputs")
    if learner.get("changed_sources") != [] or not _source_hashes(learner_sources):
        blockers.append("learner_execution.json: unchanged execution sources are required")
    for key, report_key in (("code_cells", "own_answer_code_cells"), ("embedded_figures", "embedded_figures")):
        if not _positive_integer(learner.get(key)) or learner.get(key) != measured.get(report_key):
            blockers.append(f"learner_execution.json: {key} does not match native report")
    if fresh.get("status") != "PASS" or not _source_hashes(fresh_sources):
        blockers.append("fresh_reproduction.json: passing source-based reproduction is required")
    if _source_hashes(learner_sources) and _source_hashes(fresh_sources):
        if any(fresh_sources.get(path) != digest for path, digest in learner_sources.items()):
            blockers.append("fresh_reproduction.json: sources do not match the executed learner")

    if exercise_dir.name == "part17_shapiq_estimators":
        if (learner.get("analytic_only") is not False or learner.get("native_summary_exists") is not True
                or learner.get("execution_device") != "cpu"
                or learner.get("executed_code_cells") != learner.get("code_cells")):
            blockers.append("learner_execution.json: complete native CPU execution is required")
        if (summary.get("status") != "PASS" or summary.get("games") != measured.get("model_games")
                or summary.get("estimator_inputs_only_queried_values") is not True
                or summary.get("reference_precomputed_cost_disclosed") is not True):
            blockers.append("native_summary.json: complete disclosed logical-query study is required")
        if (fresh.get("native_models_refit") is not True or fresh.get("cached_model_inputs_used") is not False
                or fresh.get("child_received_learner_payloads") is not False or fresh.get("mismatches") != []
                or fresh.get("comparison") != "exact_recursive_dtype_shape_value_sha256_including_container_order"):
            blockers.append("fresh_reproduction.json: exact source-only native refit is required")
        comparisons = fresh.get("counts", {})
        count_key, element_key, report_count_key = "arrays", "array_elements", "fresh_compared_arrays"
    else:
        if learner.get("native_model_executed") is not True:
            blockers.append("learner_execution.json: native model execution is required")
        if (learner.get("native_summary") != summary
                or learner.get("native_summary_sha256") != digests["native_summary.json"]):
            blockers.append("native_summary.json: learner summary contents/hash do not match")
        if fresh.get("native_model_retrained") is not True or fresh.get("changed_sources") != []:
            blockers.append("fresh_reproduction.json: unchanged-source native refit is required")
        if "cached_model_inputs_used" in fresh and fresh["cached_model_inputs_used"] is not False:
            blockers.append("fresh_reproduction.json: fitted cache reuse is forbidden")
        comparisons = fresh.get("comparisons", {})
        count_key, element_key, report_count_key = (
            ("tensors", "tensor_elements", "fresh_compared_tensors")
            if exercise_dir.name in TENSOR_RECEIPT_SECTIONS
            else ("arrays", "array_elements", "fresh_compared_arrays")
        )
        error = comparisons.get("max_absolute_error") if isinstance(comparisons, dict) else None
        if (type(error) not in (int, float) or not math.isfinite(error)
                or not 0 <= error <= 1e-10 or error != measured.get("fresh_max_error")):
            blockers.append("fresh_reproduction.json: numerical comparison error is invalid or inconsistent")
    if not isinstance(comparisons, dict):
        comparisons = {}
    for key, report_key in ((count_key, report_count_key), (element_key, "fresh_compared_elements")):
        if not _positive_integer(comparisons.get(key)) or comparisons.get(key) != measured.get(report_key):
            blockers.append(f"fresh_reproduction.json: {key} does not match native report")
    return blockers
