"""Tiny publication-contract tests; no native imports, model queries or fitting."""

import gzip
import json
from pathlib import Path
import sqlite3
import zipfile

from PIL import Image
import pytest

from chapter16_shapley_attribution_baselines.exercises.part18_spex import publish_study as publish


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


@pytest.fixture
def inputs(tmp_path):
    original, fresh = _native_pair(tmp_path)
    return original, fresh, _section_inputs(tmp_path, original, fresh)


def test_stage_without_approval_preserves_source_failures_and_portable_data(inputs, tmp_path):
    original, fresh, manifests = inputs
    staged = tmp_path / "candidate"
    record = publish.stage_candidate(original, fresh, manifests, staged)
    assert record["status"] == "STAGED_PENDING_INDEPENDENT_REVIEW"
    assert record["native"]["reference_outcomes"][0]["status"] == "SOURCE_FAILURE"
    assert record["native"]["reference_outcomes"][0]["scientific_status"] == "NO_RESULT"
    assert publish.validate_candidate(staged)["artifacts"] == record["artifacts"]
    contract = publish.bundles.validate_review_bundle(staged / "execution_review.tar.gz")
    assert not any(name.endswith((".npz", ".sqlite3")) for name in contract["members"])
    assert any(name.endswith(".npz") for name in contract["external_payloads"])
    assert any(name.endswith(".sqlite3") for name in contract["external_payloads"])
    publish.bundles.validate_json_data(staged / "tree" / publish.FIXTURE)
    assert not (staged / "installation.json").exists()


@pytest.mark.parametrize("change", ["native_incomplete", "source_drift", "condition_bytes", "database_bytes", "comparison_bytes", "cached_reproduction"])
def test_invalid_native_evidence_never_creates_candidate(inputs, tmp_path, change):
    original, fresh, manifests = inputs
    if change == "native_incomplete":
        record = publish.read_json(original / "execution.json")
        record["status"] = "RUNNING"
        publish.write_json(original / "execution.json", record)
    elif change == "source_drift":
        (tmp_path / "synthetic_source.py").write_text("CHANGED = True\n")
    elif change == "condition_bytes":
        (original / "study/conditions/0/record.json").write_text("{}\n")
    elif change == "database_bytes":
        with (original / "study/oracle/oracle.sqlite3").open("ab") as stream:
            stream.write(b"tamper")
    elif change == "comparison_bytes":
        (fresh / "comparison_process/comparison.json").write_text("{}\n")
    else:
        record = publish.read_json(fresh / "reproduction.json")
        record["cached_learner_models_or_values_used_for_native_execution"] = True
        publish.write_json(fresh / "reproduction.json", record)
    with pytest.raises(ValueError):
        publish.stage_candidate(original, fresh, manifests, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("change", ["unsafe_name", "raw_payload", "missing_fixture", "wrong_fixture", "failed_learner"])
def test_invalid_lesson_artifacts_reject_before_staging(inputs, tmp_path, change):
    original, fresh, manifests = inputs
    record = publish.read_json(manifests[0])
    if change == "unsafe_name":
        record["artifacts"][4]["name"] = "../other.md"
    elif change == "raw_payload":
        source = tmp_path / "fitted.pt"
        source.write_bytes(b"raw model must remain external")
        record["artifacts"].append({"role": "supporting_evidence", "name": source.name, **publish.physical_record(source)})
    elif change == "missing_fixture":
        record["artifacts"] = [item for item in record["artifacts"] if item["role"] != "learner_data"]
    elif change == "wrong_fixture":
        record["data_access"]["fixture_sha256"] = "f" * 64
    else:
        execution_path = Path(record["learner_execution"]["absolute_path"])
        execution = publish.read_json(execution_path)
        execution["portable_replay"]["status"] = "FAIL"
        publish.write_json(execution_path, execution)
        record["learner_execution"] = publish.physical_record(execution_path)
    publish.write_json(manifests[0], record)
    with pytest.raises(ValueError):
        publish.stage_candidate(original, fresh, manifests, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def _approval(tmp_path, staged):
    evidence = tmp_path / "synthetic_review.md"
    evidence.write_text("Synthetic approval validation only; no actual scientific publication approval.\n")
    approval = tmp_path / "approval.json"
    publish.write_json(approval, {"status": publish.APPROVAL_STATUS, "independent_review": True,
                                  "candidate_manifest_sha256": publish.bundles.sha256(staged / "candidate_manifest.json"),
                                  "sections": ["16.18", "16.19"], "review_evidence": [publish.physical_record(evidence)],
                                  "checks": dict.fromkeys(("scientific_outcomes", "visible_figures", "controls_and_failures",
                                                            "portable_student_replay", "source_and_reproduction"), True)})
    return approval


def test_install_requires_exact_approval_and_preserves_prior_target(inputs, tmp_path):
    original, fresh, manifests = inputs
    staged = tmp_path / "candidate"
    candidate = publish.stage_candidate(original, fresh, manifests, staged)
    approval = _approval(tmp_path, staged)
    bad = publish.read_json(approval)
    bad["candidate_manifest_sha256"] = "0" * 64
    publish.write_json(approval, bad)
    repository = tmp_path / "fake_repository"
    with pytest.raises(ValueError, match="different candidate"):
        publish.install_candidate(staged, approval, repository=repository)
    assert not repository.exists()
    approval = _approval(tmp_path, staged)
    target = next(iter(candidate["artifacts"]))
    destination = repository / target
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"prior exact artifact")
    receipt = publish.install_candidate(staged, approval, repository=repository)
    assert receipt["status"] == "INSTALLED_APPROVED_CANDIDATE"
    assert (staged / "previous_canonical" / target).read_bytes() == b"prior exact artifact"
    assert destination.read_bytes() == (staged / "tree" / target).read_bytes()
    with pytest.raises(ValueError, match="already recorded"):
        publish.install_candidate(staged, approval, repository=repository)


@pytest.mark.parametrize("target", ["tree", "archive", "external", "review"])
def test_post_stage_tampering_blocks_install(inputs, tmp_path, target):
    original, fresh, manifests = inputs
    staged = tmp_path / "candidate"
    candidate = publish.stage_candidate(original, fresh, manifests, staged)
    approval = _approval(tmp_path, staged)
    if target == "tree":
        changed = staged / "tree" / next(iter(candidate["artifacts"]))
    elif target == "archive":
        changed = staged / "execution_review.tar.gz"
    elif target == "external":
        changed = original / "study/conditions/0/arrays.npz"
    else:
        changed = tmp_path / "synthetic_review.md"
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError):
        publish.install_candidate(staged, approval, repository=tmp_path / "fake_repository")
    assert not (tmp_path / "fake_repository").exists()


def test_existing_stage_directory_is_preserved(inputs, tmp_path):
    original, fresh, manifests = inputs
    staged = tmp_path / "candidate"
    staged.mkdir()
    sentinel = staged / "earlier_failed_review.json"
    sentinel.write_text('{"status":"FAIL"}\n')
    with pytest.raises(ValueError, match="new directory"):
        publish.stage_candidate(original, fresh, manifests, staged)
    assert sentinel.read_text() == '{"status":"FAIL"}\n'


def test_install_rejects_linked_section_parent_and_preserves_original(inputs, tmp_path):
    original, fresh, manifests = inputs
    staged = tmp_path / "candidate"
    publish.stage_candidate(original, fresh, manifests, staged)
    approval = _approval(tmp_path, staged)
    repository = tmp_path / "fake_repository"
    sacred = repository / "original_arena"
    sacred.mkdir(parents=True)
    sentinel = sacred / "original.ipynb"
    sentinel.write_bytes(b"original ARENA exact bytes")
    linked = repository / publish.CHAPTER / "exercises/part18_spex"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(sacred, target_is_directory=True)
    with pytest.raises(ValueError, match="linked parent"):
        publish.install_candidate(staged, approval, repository=repository)
    assert list(sacred.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"original ARENA exact bytes"


def test_figure_inventory_mismatch_is_rejected(inputs, tmp_path):
    original, fresh, manifests = inputs
    record = publish.read_json(manifests[0])
    execution_path = Path(record["learner_execution"]["absolute_path"])
    execution = publish.read_json(execution_path)
    execution["figures"] = {"different.png": "1" * 64}
    publish.write_json(execution_path, execution)
    record["learner_execution"] = publish.physical_record(execution_path)
    publish.write_json(manifests[0], record)
    with pytest.raises(ValueError, match="figure bytes"):
        publish.stage_candidate(original, fresh, manifests, tmp_path / "candidate")
