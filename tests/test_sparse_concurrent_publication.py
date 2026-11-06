"""Synthetic scheduling evidence; no model evaluation or native outcome claim."""
import gzip
import json
import sqlite3
import zipfile

from PIL import Image

from copy import deepcopy
from pathlib import Path
import sys

import pytest

from arena_ext.legacy_sparse_evidence import publish_study as publish
from arena_ext.legacy_sparse_evidence import receipts


@pytest.fixture
def concurrent_case(tmp_path):
    outer, original = tmp_path / "outer", tmp_path / "original"
    fresh = outer / "native"
    fresh.mkdir(parents=True)
    coordinator = tmp_path / "synthetic_coordinator.py"
    coordinator.write_text("# Synthetic test source; never executed.\n")
    supplement = tmp_path / "supplement.json"
    publish.write_json(supplement, {"primary_run": str(original), "reproduction_root": str(outer),
                                    "coordinator_sha256": publish.bundles.sha256(coordinator)})
    approval = tmp_path / "approval.json"
    publish.write_json(approval, {"status": "APPROVE_CONCURRENT_COLD_REPRODUCTION", "independent_review": True,
                                 "supplement_sha256": publish.bundles.sha256(supplement), "synthetic_only": True})
    scheduling = {"mode": "concurrent_cold_native_then_frozen_comparison",
                  "unchanged_sequential_reproduce_function_used": False,
                  "outer_supervision_receipt": str(outer / "execution.json"),
                  "supplement_path": str(supplement), "supplement_sha256": publish.bundles.sha256(supplement),
                  "independent_approval_path": str(approval), "independent_approval_sha256": publish.bundles.sha256(approval),
                  "coordinator_sha256": publish.bundles.sha256(coordinator)}
    reproduction = {"scheduling": scheduling, "elapsed_seconds": 1.0, "learner_directory": str(original)}
    publish.write_json(fresh / "reproduction.json", reproduction)
    sources = {str(path): publish.bundles.file_record(path) for path in (coordinator, supplement, approval)}
    receipt = {"status": "COMPLETE", "returncode": 0, "execution_kind": "concurrent_cold_reproduction_coordinator",
               "failures": [], "resource_violations": [], "changed_sources": [],
               "source_inputs": sources, "source_inputs_after": deepcopy(sources),
               "limits": {"wall_seconds": 10800, "rss_bytes": 8 * 1024**3, "output_bytes": 8 * 1024**3},
               "elapsed_seconds": 1.1, "sampled_peak_rss_bytes": 4096,
               "supplement_sha256": scheduling["supplement_sha256"], "approval_sha256": scheduling["independent_approval_sha256"],
               "command": [sys.executable, str(coordinator), "--child"]}
    receipts._finalize_receipt(outer, receipt, 8 * 1024**3)
    return reproduction, fresh, receipt


def test_concurrent_receipts_bind_outer_resources_source_and_approval(concurrent_case):
    reproduction, fresh, _ = concurrent_case
    evidence = publish.validate_concurrent_schedule(reproduction, fresh)
    assert set(evidence) == {"supervision", "supplement", "approval", "coordinator"}
    assert all(publish.verify_record(record).is_file() for record in evidence.values())


@pytest.mark.parametrize("change", ["failed_outer", "resource_breach", "source_drift", "too_long", "rss", "output_limit", "wrong_kind"])
def test_inner_success_cannot_bypass_outer_failure(concurrent_case, change):
    reproduction, fresh, receipt = concurrent_case
    if change == "failed_outer":
        receipt["status"] = "FAIL"
    elif change == "resource_breach":
        receipt["resource_violations"] = ["wall_seconds"]
    elif change == "source_drift":
        receipt["source_inputs_after"] = {}
    elif change == "too_long":
        receipt["elapsed_seconds"] = 10801
    elif change == "rss":
        receipt["sampled_peak_rss_bytes"] = 8 * 1024**3 + 1
    elif change == "output_limit":
        receipt["limits"]["output_bytes"] *= 2
    elif change == "wrong_kind":
        receipt["execution_kind"] = "cold_source_native"
    receipts._finalize_receipt(fresh.parent, receipt, 8 * 1024**3)
    with pytest.raises(ValueError):
        publish.validate_concurrent_schedule(reproduction, fresh)


@pytest.mark.parametrize("change", ["new_file", "approval", "supplement", "coordinator", "foreign_outer", "sequential_claim", "reproduction_time"])
def test_scheduling_tampering_is_rejected(concurrent_case, change):
    reproduction, fresh, receipt = concurrent_case
    scheduling = reproduction["scheduling"]
    if change == "new_file":
        (fresh.parent / "unrecorded.log").write_text("extra bytes")
    elif change in {"approval", "supplement"}:
        key = "independent_approval_path" if change == "approval" else "supplement_path"
        path = Path(scheduling[key])
        path.write_text(path.read_text() + "\n")
    elif change == "coordinator":
        Path(receipt["command"][1]).write_text("# Changed source.\n")
    elif change == "foreign_outer":
        scheduling["outer_supervision_receipt"] = str(fresh / "execution.json")
    elif change == "sequential_claim":
        scheduling["unchanged_sequential_reproduce_function_used"] = True
    elif change == "reproduction_time":
        reproduction["elapsed_seconds"] = 10801
    with pytest.raises(ValueError):
        publish.validate_concurrent_schedule(reproduction, fresh)


def test_existing_sequential_reproduction_keeps_its_contract(tmp_path):
    assert publish.validate_concurrent_schedule({"status": "PASS"}, tmp_path) == {}


def test_concurrent_inner_receipt_cannot_omit_schedule(concurrent_case):
    reproduction, fresh, _ = concurrent_case
    del reproduction["scheduling"]
    with pytest.raises(ValueError, match="omitted"):
        publish.validate_concurrent_schedule(reproduction, fresh)


def test_stager_copies_the_authoritative_scheduling_evidence_for_both_lessons(tmp_path, monkeypatch):

    original, fresh = _native_pair(tmp_path)
    manifests = _section_inputs(tmp_path, original, fresh)
    outer = tmp_path / "synthetic_outer"
    outer.mkdir()
    evidence = {}
    for role in ("supervision", "supplement", "approval", "coordinator"):
        path = outer / (role + (".py" if role == "coordinator" else ".json"))
        if role == "coordinator":
            path.write_text("# Synthetic packaging source; never executed.\n")
        else:
            publish.write_json(path, {"synthetic_only": True, "role": role})
        evidence[role] = publish.physical_record(path)
    monkeypatch.setattr(publish, "validate_concurrent_schedule", lambda *args: evidence)
    output = tmp_path / "candidate"
    candidate = publish.stage_candidate(original, fresh, manifests, output)
    for section in publish.SECTIONS:
        for role, source in evidence.items():
            name = "scheduling_" + role + (".py" if role == "coordinator" else ".json")
            target = publish.artifact_target(section, "supporting_evidence", name)
            assert candidate["artifacts"][target]["sha256"] == source["sha256"]
            assert (output / "tree" / target).read_bytes() == Path(source["absolute_path"]).read_bytes()


# Synthetic fixtures ported from the historical publication_tests.py; see
# arena_ext/legacy_sparse_evidence/PROVENANCE.json. No model code is imported.
def _encode(value):
    if isinstance(value, dict):
        return {"__mapping__": [[key, _encode(item)] for key, item in value.items()]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value

def _record(path, value):
    path.mkdir(parents=True)
    publish.write_json(path / "record.json", _encode(value))
    with zipfile.ZipFile(path / "arrays.npz", "w"):
        pass
    hashes = {name: publish.bundles.sha256(path / name) for name in ("record.json", "arrays.npz")}
    publish.write_json(path / "hashes.json", hashes)
    return hashes

def _run(directory, source):
    study = directory / "study"
    database = study / "oracle/oracle.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE synthetic (value REAL)")
        connection.execute("INSERT INTO synthetic VALUES (1.25)")
    conditions = [{"path": "conditions/0", "hashes": _record(study / "conditions/0", {"prediction": 1.25})}]
    audits = [{"path": "audits/0", "hashes": _record(study / "audits/0", {"actual": 1.25})}]
    references = [{"stage_id": "author_spex_b3", "status": "SOURCE_FAILURE", "process_status": "FAIL",
                   "scientific_status": "NO_RESULT", "predictions": []}]
    boundary = _record(study / "prediction_boundary", {"condition_records": conditions, "references": references})
    oracle = _record(study / "oracle_values", {"logits": [1.0, 2.25]})
    _record(study / "summary", {"status": "COMPLETE", "conditions": conditions, "audit_records": audits,
                                "references": references, "prediction_boundary_hashes": boundary,
                                "oracle_value_hashes": oracle, "observed_inventory": {"synthetic_conditions": 1},
                                "oracle_inventory": {"database": {"path": str(database), **publish.bundles.file_record(database)}}})
    sources = {str(source): publish.bundles.file_record(source)}
    execution = {"status": "COMPLETE", "returncode": 0, "execution_kind": "cold_source_native",
                 "child_received_learner_payloads": False, "failures": [], "resource_violations": [], "changed_sources": [],
                 "source_inputs": sources, "source_inputs_after": sources, "method_modules": {"synthetic": "tiny"}}
    publish.write_json(directory / "execution.json", execution)

def _native_pair(tmp_path):
    source = tmp_path / "synthetic_source.py"
    source.write_text("SYNTHETIC_ONLY = True\n")
    original, fresh = tmp_path / "original", tmp_path / "fresh"
    _run(original, source)
    _run(fresh, source)
    process = fresh / "comparison_process"
    publish.write_json(process / "execution.json", {"status": "COMPLETE", "returncode": 0, "resource_violations": []})
    publish.write_json(process / "comparison.json", {"status": "PASS", "mismatches": [], "exact_matches": ["tiny"]})
    summary = {"status": "PASS", "mismatch_count": 0, "bitwise_exact": True, "numerical_difference_leaves": 0,
               "comparison_sha256": publish.bundles.sha256(process / "comparison.json")}
    publish.write_json(process / "comparison_summary.json", summary)
    reproduction = {"status": "PASS", "failures": [], "native_child_received_learner_payloads": False,
                    "cached_learner_models_or_values_used_for_native_execution": False,
                    "learner_directory": str(original), "fresh_directory": str(fresh),
                    "learner_execution_sha256": publish.bundles.sha256(original / "execution.json"),
                    "fresh_execution_sha256": publish.bundles.sha256(fresh / "execution.json"),
                    "comparison_execution": {"path": str(process / "execution.json"),
                                             "sha256": publish.bundles.sha256(process / "execution.json")},
                    "comparison_sha256": summary["comparison_sha256"]}
    publish.write_json(fresh / "reproduction.json", reproduction)
    return original, fresh

def _section_inputs(tmp_path, original, fresh):
    artifact_root = tmp_path / "lesson_inputs"
    artifact_root.mkdir()
    fixture = artifact_root / "fixture.json.gz"
    fixture.write_bytes(gzip.compress(json.dumps({"schema": "synthetic-only", "scores": [1.25, -0.5]}).encode(), mtime=0))
    manifests = []
    for section in publish.SECTIONS:
        directory = artifact_root / section
        directory.mkdir()
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"arena": {"section": section}},
                    "cells": [{"cell_type": "code", "id": "tiny", "source": "print('synthetic')", "metadata": {},
                               "execution_count": 1, "outputs": [{"output_type": "stream", "name": "stdout", "text": "synthetic\n"}]}]}
        executed = directory / "executed.ipynb"
        publish.write_json(executed, notebook)
        artifacts = []
        for role, name in (("exercises", "exercises.ipynb"), ("solutions", "solutions.ipynb"),
                           ("page", "page.md"), ("verification_report", "report.json"),
                           ("data_instructions", "DATA_ACCESS.md"), ("figure", publish.SECTIONS[section][3] + "result.png"),
                           ("native_summary", "summary.json"), ("artifacts_lock", "artifacts.lock.yml"),
                           ("verification_schema", "schema.json")):
            source = directory / name
            if source.suffix == ".ipynb":
                publish.write_json(source, notebook)
            elif source.suffix == ".png":
                Image.new("RGB", (4, 4), color="blue").save(source)
            elif source.suffix == ".json":
                publish.write_json(source, {"status": "SYNTHETIC_PUBLICATION_TEST"})
            else:
                source.write_text("Synthetic fixture and regenerate instructions; no real native outcome.\n")
            artifacts.append({"role": role, "name": name, **publish.physical_record(source)})
        if section == "16.18":
            artifacts.append({"role": "learner_data", **publish.physical_record(fixture)})
        bindings = {"native_execution_sha256": publish.bundles.sha256(original / "execution.json"),
                    "fresh_reproduction_sha256": publish.bundles.sha256(fresh / "reproduction.json")}
        execution = directory / "execution.json"
        publish.write_json(execution, {"status": "passed", "sources_unchanged": True, "executed_code_cells": 1,
                                        "total_code_cells": 1, "executed_notebook": str(executed),
                                        "notebook_sha256": publish.bundles.sha256(executed), **bindings,
                                        "figures": {item["name"]: item["sha256"] for item in artifacts if item["role"] == "figure"},
                                        "portable_replay": {"status": "PASS", "no_full_study_cache_required": True,
                                                            "fixture_sha256": publish.bundles.sha256(fixture)}})
        manifest = directory / "publication_inputs.json"
        publish.write_json(manifest, {"schema": publish.SECTION_SCHEMA, "section": section, **bindings,
                                      "learner_execution": publish.physical_record(execution), "artifacts": artifacts,
                                      "data_access": {"fixture_target": publish.FIXTURE, "fixture_sha256": publish.bundles.sha256(fixture),
                                                      "default_requires_full_study": False,
                                                      "full_study_access": "verified_external_cache_or_source_regeneration",
                                                      "regeneration_instructions_target": publish.artifact_target(section, "data_instructions", "DATA_ACCESS.md")}})
        manifests.append(manifest)
    return manifests
