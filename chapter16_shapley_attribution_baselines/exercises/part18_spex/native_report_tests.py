"""Synthetic evidence-contract tests; no native model, fitting or publication."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_report as reports


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def context(tmp_path):
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
def test_complete_synthetic_contract_preserves_failures_poor_fidelity_and_nonexact_reproduction(context, section):
    summary, report = _report(context, section)
    assert reports.has_native_evidence(report)
    assert reports.receipt_blockers(section, summary, context["learner"], _fresh(context), report) == []
    assert summary["reference_outcomes"][0]["exception"] == "retained synthetic source failure"
    assert len(summary["condition_outcomes"]) == 1152
    assert min(row["metrics"][next(iter(row["metrics"]))]["r2"] for row in summary["condition_outcomes"]) < 0
    assert report["metrics"]["gpu_test"]["fresh_bitwise_exact"] is False
    assert report["metrics"]["gpu_test"]["recorded_study_reexecuted_in_notebook"] is False


@pytest.mark.parametrize("mutation", ["drop_condition", "duplicate_condition", "drop_reference", "drop_prompt", "drop_control",
                                     "missing_stage", "nonfinite", "wrong_queries", "wrong_fits", "missing_reference_failure"])
def test_outcome_mutations_are_rejected(context, mutation):
    summary, _ = _report(context)
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
def test_cross_receipt_and_report_lies_reject(context, change):
    summary, report = _report(context)
    learner, fresh = context["learner"], _fresh(context)
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


def test_constant_audit_variance_preserves_undefined_metrics(context):
    summary, _ = _report(context)
    for metric in summary["condition_outcomes"][0]["metrics"].values():
        metric.update(nrmse=None, r2=None)
    reports._validate_outcomes(summary)


def test_actual_proxy_learner_json_receipt_layout_is_preserved(context):
    context["learner"]["model_queries"] = 0
    context["learner"]["portable_replay"].pop("new_model_queries")
    context["learner"]["portable_replay"].pop("replay_records")
    names = ["native_default_replay.json", "native_play_replay.json"]
    context["learner"]["replay_receipts"] = {name: _digest(name) for name in names}
    for name, outcome in zip(names, context["replay_outcomes"], strict=True):
        outcome.update(record_path=name, hashes={name: _digest(name)})
    summary, report = _report(context, "16.19")
    assert reports.receipt_blockers("16.19", summary, context["learner"], _fresh(context), report) == []


def test_prediction_stream_checks_actual_saved_arrays_and_rejects_changed_portable_values(tmp_path):
    from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_study, native_learner
    import numpy as np

    root = tmp_path / "native"
    audit = {"prompt_index": 0, "audit_values": np.linspace(-1, 1, 512), "unmasked": np.array([-2.0]),
             "counterfactual": np.array([0.5])}
    row = {"prompt_index": 0, "prompt_id": "synthetic", "method": "spex", "seed": 0, "budget": 1824,
           "condition": "real", "query_receipt": {"raw_requested": 1824, "unique_model_masks": 1000},
           "predictions": {"final": audit["audit_values"] * 2},
           "baseline_predictions": {"constant": np.zeros(512), "additive": audit["audit_values"] * 0.25},
           "spectrum": {(): -0.1, (2,): 0.5}, "fit": {"unused_state": np.ones((2, 3))}}
    references = [{"stage_id": "author_spex_b3", "status": "SOURCE_FAILURE", "process_status": "COMPLETE",
                   "scientific_status": "SOURCE_FAILURE", "exception": "synthetic failure", "predictions": {}, "spectra": {}}]
    original = {"audit_records": [{"path": "audit/p00", "hashes": native_study.save_record(root / "study/audit/p00", audit)}],
                "conditions": [{"path": "conditions/one", "hashes": native_study.save_record(root / "study/conditions/one", row)}],
                "references": references}
    portable = {"audit": [reports._plain(audit)],
                "results": [reports._plain(native_learner._compact_result({**row, "audit_values": audit["audit_values"]}))],
                "references": [native_learner._compact_reference(references[0], audit["audit_values"])]}
    reports._verify_portable_outcomes(root, portable, original)
    portable["results"][0]["metrics"]["final"]["rmse"] += 1
    with pytest.raises(ValueError, match="physical native predictions"):
        reports._verify_portable_outcomes(root, portable, original)


def test_manifest_requires_real_bytes_and_safe_canonical_paths(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"synthetic": true}\n')
    first = reports._manifest({"expected_outputs/receipt.json": path})
    path.write_text('{"synthetic": false}\n')
    assert reports._manifest({"expected_outputs/receipt.json": path}) != first
    with pytest.raises(ValueError):
        reports._manifest({"../escaped.json": path})
    with pytest.raises(ValueError):
        reports._manifest({"missing.json": tmp_path / "missing.json"})


def test_existing_or_repository_output_rejects_before_native_validation(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("No native validator should run")
    monkeypatch.setattr(reports, "_validated_inputs", forbidden)
    for output in (tmp_path, reports.HERE / "never_create_test_output"):
        with pytest.raises(ValueError, match="new external"):
            reports.build_report("16.18", "unused", "unused", "unused", "unused", {}, output)


def test_synthetic_external_assembly_passes_universal_schema_and_existing_audit_with_explicit_new_predicate(context, tmp_path, monkeypatch):
    """Mock only upstream physical validation: this is not actual native evidence."""
    from arena_ext.verification import verify_report_input_manifest
    from scripts import audit_report_evidence_contracts as audit

    original, fresh = tmp_path / "original", tmp_path / "fresh"
    learner = tmp_path / "learner.json"
    fixture = tmp_path / "fixture.json.gz"
    for directory, receipt in ((original, context["native_receipt"]), (fresh, context["fresh_receipt"])):
        reports.publish.write_json(directory / "execution.json", receipt)
    reports.publish.write_json(fresh / "reproduction.json", _fresh(context))
    reports.publish.write_json(fresh / "comparison_process/comparison_summary.json", context["comparison"])
    reports.publish.write_json(learner, context["learner"])
    fixture.write_bytes(b"synthetic fixture; upstream validation deliberately mocked")
    Path(str(fixture) + ".data-contract.json").write_text("{}\n")
    bindings = context["bindings"]
    for key, path in (("native_execution_sha256", original / "execution.json"), ("fresh_execution_sha256", fresh / "execution.json"),
                      ("fixture_sha256", fixture)):
        bindings[key] = reports.publish.bundles.sha256(path)
    context["learner"]["native_execution_sha256"] = bindings["native_execution_sha256"]
    context["learner"]["portable_replay"]["fixture_sha256"] = bindings["fixture_sha256"]
    reports.publish.write_json(fresh / "reproduction.json", _fresh(context))
    bindings["fresh_reproduction_sha256"] = reports.publish.bundles.sha256(fresh / "reproduction.json")
    context["learner"]["fresh_reproduction_sha256"] = bindings["fresh_reproduction_sha256"]
    reports.publish.write_json(learner, context["learner"])
    bindings["learner_execution_sha256"] = reports.publish.bundles.sha256(learner)
    context["native"].update(native_execution=reports.publish.physical_record(original / "execution.json"),
                             fresh_execution=reports.publish.physical_record(fresh / "execution.json"))
    inputs = {}
    for role in ("exercises", "solutions", "page"):
        path = tmp_path / (role + ".txt")
        path.write_text("synthetic artifact\n")
        inputs[reports.publish.artifact_target("16.18", role)] = path
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"synthetic figure")
    inputs[f"{reports.publish.CHAPTER}/instructions/assets/1618_synthetic.png"] = figure
    context["learner"]["figures"] = {"1618_synthetic.png": reports.publish.bundles.sha256(figure)}
    reports.publish.write_json(learner, context["learner"])
    bindings["learner_execution_sha256"] = reports.publish.bundles.sha256(learner)
    monkeypatch.setattr(reports, "_validated_inputs", lambda *args: context)
    output = tmp_path / "report_output"
    result = reports.build_report("16.18", original, fresh, learner, fixture, inputs, output)
    artifacts = reports.publish.read_json(output / "report_artifacts.json")
    assert result["date_run"] == context["fresh_receipt"]["started_at_utc"]
    assert result["metrics"]["gpu_test"]["native_model_queries_in_report"] == 0
    assert len(result["metrics"]["native_outcomes"]["condition_outcomes"]) == 1152
    assert result["metrics"]["native_outcomes"]["reference_outcomes"][0]["status"] == "SOURCE_FAILURE"
    overlay = tmp_path / "canonical_test_overlay"
    for name, record in artifacts["physical_report_inputs"].items():
        target = overlay / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(record["absolute_path"]).read_bytes())
    assert verify_report_input_manifest(overlay, result["report_inputs"]) == []
    monkeypatch.setattr(audit, "ROOT", overlay)
    old_predicate = audit.has_native_cpu_model_evidence
    monkeypatch.setattr(audit, "has_native_cpu_model_evidence", lambda lock, report: reports.has_native_evidence(report) or old_predicate(lock, report))
    lock = reports.yaml.safe_load((output / "artifacts.lock.yml").read_text())
    assert audit.report_contract_blockers(lock, result) == []


def test_actual_cli_rejects_incomplete_native_receipt_without_creating_report(tmp_path):
    import subprocess
    import sys

    native = tmp_path / "incomplete_native"
    reports.publish.write_json(native / "execution.json", {"status": "FAIL", "returncode": -2})
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}\n")
    output = tmp_path / "must_not_exist"
    command = [sys.executable, str(Path(reports.__file__)), "--section", "16.18", "--native-run", str(native),
               "--fresh-run", str(tmp_path / "unrun_fresh"), "--learner-execution", str(tmp_path / "unrun_learner"),
               "--fixture", str(tmp_path / "unexported_fixture"), "--artifact-inputs", str(inputs), "--output", str(output)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
    assert result.returncode != 0
    assert "Native execution must have completed successfully" in result.stderr
    assert not output.exists()
