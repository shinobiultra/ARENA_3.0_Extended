"""Exercise the shared audit adapters with explicitly synthetic native receipts."""

import hashlib
import json
from copy import deepcopy

import pytest

from arena_ext.legacy_sparse_evidence import native_report
from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence
from scripts.verification_fixture_contract import (
    CHAPTER, LEGACY_FIXTURE_NAMES, NATIVE_FIXTURE_NAMES, SPARSE_RECEIPT_SECTIONS,
    fixture_paths, native_receipt_blockers, uses_native_receipts, verification_entrypoint,
)


reports = native_report


@pytest.fixture(params=SPARSE_RECEIPT_SECTIONS)
def sparse_case(request, synthetic_native_context, tmp_path):
    context = synthetic_native_context
    directory = tmp_path / CHAPTER / request.param
    outputs = directory / "expected_outputs"
    outputs.mkdir(parents=True)
    section, identity = SPARSE_RECEIPT_SECTIONS[request.param]

    def write(name, value):
        path = outputs / name
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    fresh_hash = write("fresh_reproduction.json", _fresh(context))
    context["bindings"]["fresh_reproduction_sha256"] = fresh_hash
    context["learner"]["fresh_reproduction_sha256"] = fresh_hash
    learner_hash = write("learner_execution.json", context["learner"])
    context["bindings"]["learner_execution_sha256"] = learner_hash
    summary, report = _report(context, section)
    summary_hash = write("native_summary.json", summary)
    report["metrics"]["gpu_test"]["native_summary_sha256"] = summary_hash
    report["metrics"]["gpu_evidence"] = {
        "category": "cpu_or_budget_metric", "uses_cuda": False, "placeholder_only": False,
    }
    report.update(accepted=True, tests_passed=True)
    report["report_inputs"]["algorithm"] = "sha256"
    for entry in report["report_inputs"]["files"]:
        path = tmp_path / entry["path"]
        if path.exists():
            entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    lock = {"notebook_id": identity, "execution_device": "cpu", "exercise_metadata": {"REQUIRES_GPU": False}}
    return tmp_path, directory, lock, report


def test_sparse_receipt_adapter_accepts_separate_model_study_and_student_replay(sparse_case):
    root, directory, lock, report = sparse_case
    assert has_native_cpu_model_evidence(lock, report)
    assert native_receipt_blockers(*sparse_case) == []
    assert [path.name for path in fixture_paths(root, directory)] == list(NATIVE_FIXTURE_NAMES)
    assert verification_entrypoint(root, directory) == root / "arena_ext/legacy_sparse_evidence/native_report.py"


@pytest.mark.parametrize("name", NATIVE_FIXTURE_NAMES)
def test_sparse_receipt_adapter_keeps_physical_manifest_checks(sparse_case, name):
    root, directory, _, report = sparse_case
    relative = (directory / "expected_outputs" / name).relative_to(root).as_posix()
    entry = next(row for row in report["report_inputs"]["files"] if row["path"] == relative)
    entry["size_bytes"] += 1
    assert any(name in message for message in native_receipt_blockers(*sparse_case))


def test_sparse_adapter_does_not_reclassify_placeholder_or_gpu_claims(sparse_case):
    _, _, lock, report = sparse_case
    report["metrics"]["gpu_evidence"]["uses_cuda"] = True
    assert not has_native_cpu_model_evidence(lock, report)
    report["metrics"]["gpu_evidence"]["uses_cuda"] = False
    report["metrics"]["gpu_test"]["native_execution_complete"] = False
    assert not has_native_cpu_model_evidence(lock, report)
    assert not native_report.has_native_evidence(report)


def test_sparse_names_outside_the_explicit_chapter_keep_legacy_contract(tmp_path):
    directory = tmp_path / "unrelated" / "part18_spex"
    assert not uses_native_receipts(tmp_path, directory)
    assert [path.name for path in fixture_paths(tmp_path, directory)] == list(LEGACY_FIXTURE_NAMES)
    assert verification_entrypoint(tmp_path, directory) == directory / "solutions.py"


@pytest.mark.parametrize("stem", [
    "16.18_Find_the_Interactions_Without_Trying_Every_Subset",
    "16.19_Explain_the_Proxy_Check_the_Model",
])
def test_course_surface_dispatches_to_the_strict_sparse_learner_contract(tmp_path, monkeypatch, stem):
    from scripts import sparse_interaction_learner_contract
    from scripts.audit_course_surface import _exercise_notebook_declares_verification_contract

    observed = []

    def reject(path):
        observed.append(path)
        return False

    monkeypatch.setattr(sparse_interaction_learner_contract, "has_learner_contract", reject)
    path = tmp_path / (stem + "_exercises.ipynb")
    assert not _exercise_notebook_declares_verification_contract(path)
    assert observed == [path]


# Synthetic fixtures ported from historical native_report_tests.py; see
# arena_ext/legacy_sparse_evidence/PROVENANCE.json.
def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

@pytest.fixture
def synthetic_native_context(tmp_path):
    bindings = {key: _digest(key) for key in ("native_execution_sha256", "fresh_execution_sha256",
                "fresh_reproduction_sha256", "learner_execution_sha256", "fixture_sha256")}
    rows = []
    for prompt, method, seed, budget, label in sorted(reports.GRID):
        stages = ("final", "constant", "additive") if method == "spex" else (
            "proxy", "extracted", "truncated", "refined", "constant", "additive")
        # Deliberately poor fidelity and a shuffled result beating the real fit.
        rmse = 7.0 if label == "real" else 2.0
        rows.append({"prompt_index": prompt, "prompt_id": f"synthetic_{prompt}", "method": method,
                     "seed": seed, "budget": budget, "condition": label,
                     "query_receipt": {"raw_requested": budget, "unique_model_masks": budget - 1},
                     "metrics": {stage: {"rmse": rmse, "nrmse": rmse / 2, "r2": 1 - rmse**2 / 4} for stage in stages},
                     "stage_errors": {f"synthetic_stage_{index}": 0.5 for index in range(7)} if method == "proxyspex" else {},
                     "record_origin": "recorded_complete_native_study"})
    references = [{"stage_id": stage, "status": "SOURCE_FAILURE" if index == 0 else "COMPLETED",
                   "process_status": "COMPLETE", "scientific_status": "SOURCE_FAILURE" if index == 0 else "COMPLETED",
                   "exception": "retained synthetic source failure" if index == 0 else None,
                   "metrics": {} if index == 0 else {"final": {"rmse": 3.0, "nrmse": None, "r2": None}},
                   "record_origin": "recorded_prompt0_seed0_reference_stage"}
                  for index, stage in enumerate(reports.REFERENCE_IDS)]
    learner = {"status": "passed", "sources_unchanged": True, "executed_code_cells": 31, "total_code_cells": 31,
               "figures": {"1618_synthetic.png": _digest("figure")},
               "native_execution_sha256": bindings["native_execution_sha256"],
               "fresh_reproduction_sha256": bindings["fresh_reproduction_sha256"],
               "portable_replay": {"status": "PASS", "fixture_sha256": bindings["fixture_sha256"],
                                   "no_full_study_cache_required": True, "new_model_queries": 0,
                                   "replay_records": [str(tmp_path / "learner/default"), str(tmp_path / "learner/play")]}}
    replays = [{"record_path": path, "hashes": {name: _digest(path + name) for name in ("record.json", "arrays.npz")},
                "replay": {"status": "COMPLETE", "method": "spex", "prompt_index": index, "seed": 0,
                           "budget": 1824 if index == 0 else 912, "new_model_queries": 0, "new_physical_model_rows": 0,
                           "audit_labels_received_by_fit": False, "estimator_refits": 2, "baseline_fits": 4,
                           "comparisons": [{"condition": label, "prediction_tolerance": {"atol": 1e-8, "rtol": 1e-7},
                                            "prediction_allclose": dict.fromkeys(("final", "constant", "additive"), True),
                                            "maximum_prediction_absolute_delta": dict.fromkeys(("final", "constant", "additive"), 1e-10)}
                                           for label in ("real", "shuffled")]}}
               for index, path in enumerate(learner["portable_replay"]["replay_records"])]
    counts = {"arrays": 1500, "array_elements": 123456, "scalars": 1234, "containers": 4000}
    comparison = {"status": "PASS", "mismatch_count": 0, "counts": {"original": counts, "fresh": counts.copy()},
                  "bitwise_exact": False, "numerical_difference_leaves": 3, "comparison_sha256": _digest("full comparison")}
    result = {"bindings": bindings, "learner": learner, "replay_outcomes": replays, "comparison": comparison,
              "portable": {"results": rows, "references": references,
                           "audit": [{"prompt_index": index, "audit_values": [0.25] * 512,
                                      "unmasked": [-1.25], "counterfactual": [0.75]} for index in range(32)]},
              "native": {"observed_inventory": {**reports.MEASURED_INVENTORY, "proxy_residual_tree_fits": 2400000}},
              "native_receipt": {"elapsed_seconds": 6000.0, "started_at_utc": "2026-09-08T00:00:00+00:00", "source_inputs": {}},
              "fresh_receipt": {"elapsed_seconds": 6100.0, "started_at_utc": "2026-09-08T03:00:00+00:00"}}
    return result

def _fresh(context):
    comparison = context["comparison"]
    return {"status": "PASS", "failures": [], "native_child_received_learner_payloads": False,
            "cached_learner_models_or_values_used_for_native_execution": False,
            "learner_execution_sha256": context["bindings"]["native_execution_sha256"],
            "fresh_execution_sha256": context["bindings"]["fresh_execution_sha256"],
            "comparison_sha256": comparison["comparison_sha256"], "comparison_counts": comparison["counts"],
            "bitwise_exact": comparison["bitwise_exact"], "numerical_difference_leaves": comparison["numerical_difference_leaves"],
            "mismatch_count": comparison["mismatch_count"]}

def _report(context, section="16.18"):
    if section == "16.19":
        for row in context["replay_outcomes"]:
            row["replay"]["method"] = "proxyspex"
            for comparison in row["replay"]["comparisons"]:
                stages = ("proxy", "extracted", "truncated", "refined", "constant", "additive")
                comparison["prediction_allclose"] = dict.fromkeys(stages, True)
                comparison["maximum_prediction_absolute_delta"] = dict.fromkeys(stages, 1e-10)
    summary = reports._summary(section, context)
    values = reports._metric_summary(summary, context["learner"])
    values.update(native_summary_sha256=_digest(summary), comparison_summary_sha256=_digest(context["comparison"]))
    inputs = [{"path": reports._prefix(section) + "expected_outputs/" + name,
               "sha256": values[field], "size_bytes": 1} for field, name in reports.RECEIPT_FIELDS.items()]
    inputs.append({"path": reports.publish.FIXTURE, "sha256": values["fixture_sha256"], "size_bytes": 1})
    report = {"notebook_id": reports.IDS[section], "metrics": {"gpu_test": values,
              "native_outcomes": {key: summary[key] for key in ("condition_outcomes", "reference_outcomes", "prompt_outcomes")}},
              "report_inputs": {"files": inputs}}
    return summary, report


@pytest.mark.parametrize("section", ["16.18", "16.19"])
def test_complete_synthetic_contract_preserves_failures_poor_fidelity_and_nonexact_reproduction(synthetic_native_context, section):
    summary, report = _report(synthetic_native_context, section)
    assert reports.has_native_evidence(report)
    assert reports.receipt_blockers(section, summary, synthetic_native_context["learner"], _fresh(synthetic_native_context), report) == []
    assert summary["reference_outcomes"][0]["exception"] == "retained synthetic source failure"
    assert len(summary["condition_outcomes"]) == 1152
    assert min(row["metrics"][next(iter(row["metrics"]))]["r2"] for row in summary["condition_outcomes"]) < 0
    assert report["metrics"]["gpu_test"]["fresh_bitwise_exact"] is False
    assert report["metrics"]["gpu_test"]["recorded_study_reexecuted_in_notebook"] is False


@pytest.mark.parametrize("mutation", ["drop_condition", "duplicate_condition", "drop_reference", "drop_prompt", "drop_control",
                                     "missing_stage", "nonfinite", "wrong_queries", "wrong_fits", "missing_reference_failure"])
def test_outcome_mutations_are_rejected(synthetic_native_context, mutation):
    summary, _ = _report(synthetic_native_context)
    rows = summary["condition_outcomes"]
    if mutation == "drop_condition":
        rows.pop()
    elif mutation == "duplicate_condition":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "drop_reference":
        summary["reference_outcomes"].pop()
    elif mutation == "drop_prompt":
        summary["prompt_outcomes"].pop()
    elif mutation == "drop_control":
        rows[0]["metrics"].pop("constant")
    elif mutation == "missing_stage":
        next(row for row in rows if row["method"] == "proxyspex")["stage_errors"].pop("synthetic_stage_0")
    elif mutation == "nonfinite":
        rows[0]["metrics"]["constant"]["rmse"] = float("nan")
    elif mutation == "wrong_queries":
        summary["inventory"]["physical_evaluations"] -= 1
    elif mutation == "wrong_fits":
        summary["inventory"]["proxy_ensemble_fits"] -= 1
    else:
        summary["reference_outcomes"][0].pop("scientific_status")
    with pytest.raises(ValueError):
        reports._validate_outcomes(summary)


@pytest.mark.parametrize("change", ["mismatch", "false_exact", "missing_counts", "cached", "unfinished", "other_run", "metric_lie",
                                   "replay_query", "replay_leak", "replay_method", "missing_hash", "duplicate_hash",
                                   "replay_wrong_predictions", "erased_reference_outcomes"])
def test_cross_receipt_and_report_lies_reject(synthetic_native_context, change):
    summary, report = _report(synthetic_native_context)
    learner, fresh = synthetic_native_context["learner"], _fresh(synthetic_native_context)
    if change == "mismatch":
        summary["comparison"]["mismatch_count"] = 1
    elif change == "false_exact":
        report["metrics"]["gpu_test"]["fresh_bitwise_exact"] = True
    elif change == "missing_counts":
        summary["comparison"]["counts"]["original"]["arrays"] = 0
    elif change == "cached":
        fresh["cached_learner_models_or_values_used_for_native_execution"] = True
    elif change == "unfinished":
        learner["executed_code_cells"] -= 1
    elif change == "other_run":
        fresh["fresh_execution_sha256"] = "f" * 64
    elif change == "metric_lie":
        report["metrics"]["gpu_test"]["reference_status_counts"] = {"COMPLETED": 9}
    elif change == "replay_query":
        summary["replay_outcomes"][0]["replay"]["new_model_queries"] = 1
    elif change == "replay_leak":
        summary["replay_outcomes"][0]["replay"]["audit_labels_received_by_fit"] = True
    elif change == "replay_method":
        summary["replay_outcomes"][0]["replay"]["method"] = "proxyspex"
    elif change == "missing_hash":
        report["report_inputs"]["files"].pop()
    elif change == "duplicate_hash":
        report["report_inputs"]["files"].append(report["report_inputs"]["files"][0].copy())
    elif change == "replay_wrong_predictions":
        summary["replay_outcomes"][0]["replay"]["comparisons"][0]["prediction_allclose"]["final"] = False
    else:
        report["metrics"]["native_outcomes"]["reference_outcomes"] = []
    assert reports.receipt_blockers("16.18", summary, learner, fresh, report)


def test_constant_audit_variance_preserves_undefined_metrics(synthetic_native_context):
    summary, _ = _report(synthetic_native_context)
    for metric in summary["condition_outcomes"][0]["metrics"].values():
        metric.update(nrmse=None, r2=None)
    reports._validate_outcomes(summary)


def test_actual_proxy_learner_json_receipt_layout_is_preserved(synthetic_native_context):
    synthetic_native_context["learner"]["model_queries"] = 0
    synthetic_native_context["learner"]["portable_replay"].pop("new_model_queries")
    synthetic_native_context["learner"]["portable_replay"].pop("replay_records")
    names = ["native_default_replay.json", "native_play_replay.json"]
    synthetic_native_context["learner"]["replay_receipts"] = {name: _digest(name) for name in names}
    for name, outcome in zip(names, synthetic_native_context["replay_outcomes"], strict=True):
        outcome.update(record_path=name, hashes={name: _digest(name)})
    summary, report = _report(synthetic_native_context, "16.19")
    assert reports.receipt_blockers("16.19", summary, synthetic_native_context["learner"], _fresh(synthetic_native_context), report) == []
