"""Mutation coverage for the small, read-only native receipt fixture gate."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from arena_ext.verification import build_report_input_manifest
from scripts.verification_fixture_contract import (
    CHAPTER,
    LEGACY_FIXTURE_NAMES,
    NATIVE_FIXTURE_NAMES,
    NATIVE_RECEIPT_SECTIONS,
    fixture_paths,
    native_receipt_blockers,
    uses_native_receipts,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=NATIVE_RECEIPT_SECTIONS)
def native_case(request, tmp_path):
    """Copy only metadata; no native cache or fitted array is needed by this gate."""
    source = ROOT / CHAPTER / request.param
    target = tmp_path / CHAPTER / request.param
    (target / "expected_outputs").mkdir(parents=True)
    for name in NATIVE_FIXTURE_NAMES:
        shutil.copyfile(source / "expected_outputs" / name, target / "expected_outputs" / name)
    lock = yaml.safe_load((source / "artifacts.lock.yml").read_text())
    report = json.loads((source / "verification_report.json").read_text())
    return tmp_path, target, lock, report


def _rebind(case):
    """Keep hashes current so semantic mutants cannot fail only on stale bytes."""
    root, directory, _, report = case
    paths = fixture_paths(root, directory)
    manifest = build_report_input_manifest(root, paths)
    replacements = {entry["path"]: entry for entry in manifest["files"]}
    report["report_inputs"]["files"] = [
        replacements.get(entry["path"], entry) for entry in report["report_inputs"]["files"]
    ]
    for name, key in (("learner_execution.json", "learner_evidence_sha256"),
                      ("fresh_reproduction.json", "fresh_evidence_sha256")):
        report["metrics"]["gpu_test"][key] = replacements[(directory / "expected_outputs" / name).relative_to(root).as_posix()]["sha256"]


def _write(case, name, value):
    (case[1] / "expected_outputs" / name).write_text(json.dumps(value))
    _rebind(case)


def test_saved_native_triplet_is_sufficient_without_legacy_fixtures_or_cache(native_case):
    assert native_receipt_blockers(*native_case) == []
    assert not any((native_case[1] / "expected_outputs" / name).exists() for name in LEGACY_FIXTURE_NAMES)


@pytest.mark.parametrize("name", NATIVE_FIXTURE_NAMES)
def test_missing_receipt_is_rejected(native_case, name):
    (native_case[1] / "expected_outputs" / name).unlink()
    assert any(name in blocker and "missing" in blocker for blocker in native_receipt_blockers(*native_case))


@pytest.mark.parametrize("name", NATIVE_FIXTURE_NAMES)
def test_corrupt_receipt_is_rejected_even_with_updated_binding(native_case, name):
    (native_case[1] / "expected_outputs" / name).write_text("{broken JSON")
    _rebind(native_case)
    assert any(name in blocker and "invalid JSON" in blocker for blocker in native_receipt_blockers(*native_case))


@pytest.mark.parametrize("name", NATIVE_FIXTURE_NAMES)
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hash", "size"])
def test_missing_duplicate_or_wrong_report_binding_is_rejected(native_case, name, mutation):
    root, directory, _, report = native_case
    path = (directory / "expected_outputs" / name).relative_to(root).as_posix()
    entries = report["report_inputs"]["files"]
    entry = next(item for item in entries if item["path"] == path)
    if mutation == "missing":
        entries.remove(entry)
    elif mutation == "duplicate":
        entries.append(dict(entry))
    elif mutation == "hash":
        entry["sha256"] = "0" * 64
    else:
        entry["size_bytes"] += 1
    assert any(name in blocker for blocker in native_receipt_blockers(*native_case))


@pytest.mark.parametrize("name", NATIVE_FIXTURE_NAMES)
@pytest.mark.parametrize("placeholder", [{}, [], {"status": "PASS", "accepted": True, "tests_passed": True}])
def test_placeholder_receipt_is_rejected_even_with_updated_binding(native_case, name, placeholder):
    _write(native_case, name, placeholder)
    assert native_receipt_blockers(*native_case)


@pytest.mark.parametrize("name,field,value", [
    ("learner_execution.json", "completed_from_own_answers", False),
    ("learner_execution.json", "changed_sources", ["modified.py"]),
    ("learner_execution.json", "code_cells", 0),
    ("learner_execution.json", "embedded_figures", 0),
    ("learner_execution.json", "source_inputs", "source.py"),
    ("learner_execution.json", "source_inputs", {"source.py": "placeholder"}),
    ("fresh_reproduction.json", "status", "FAIL"),
    ("fresh_reproduction.json", "source_inputs", {}),
])
def test_failed_or_incomplete_receipt_is_rejected_even_with_updated_binding(native_case, name, field, value):
    payload = json.loads((native_case[1] / "expected_outputs" / name).read_text())
    payload[field] = value
    _write(native_case, name, payload)
    assert native_receipt_blockers(*native_case)


def test_reproduction_source_drift_is_rejected_even_with_updated_binding(native_case):
    path = native_case[1] / "expected_outputs" / "fresh_reproduction.json"
    payload = json.loads(path.read_text())
    source = next(iter(payload["source_inputs"]))
    payload["source_inputs"][source] = "0" * 64
    _write(native_case, path.name, payload)
    assert any("sources do not match" in blocker for blocker in native_receipt_blockers(*native_case))


@pytest.mark.parametrize("mutation", ["placeholder_report", "failed_report", "gpu_lock", "wrong_identity", "wrong_metric_hash"])
def test_triplet_cannot_bypass_native_report_contract(native_case, mutation):
    _, _, lock, report = native_case
    if mutation == "placeholder_report":
        report["metrics"]["gpu_test"] = {"device": "cpu", "preflight_passed": True}
    elif mutation == "failed_report":
        report["accepted"] = False
    elif mutation == "gpu_lock":
        lock["exercise_metadata"]["REQUIRES_GPU"] = True
    elif mutation == "wrong_identity":
        lock["notebook_id"] = report["notebook_id"] = "different_lesson"
    else:
        report["metrics"]["gpu_test"]["learner_evidence_sha256"] = "0" * 64
    assert native_receipt_blockers(*native_case)


def test_legacy_contract_is_retained_outside_explicit_migration_list(tmp_path):
    directory = tmp_path / CHAPTER / "part3_shapley_interactions_shapiq"
    (directory / "expected_outputs").mkdir(parents=True)
    for name in NATIVE_FIXTURE_NAMES:
        (directory / "expected_outputs" / name).write_text('{"status":"PASS"}')
    assert not uses_native_receipts(tmp_path, directory)
    assert [path.name for path in fixture_paths(tmp_path, directory)] == list(LEGACY_FIXTURE_NAMES)
    assert native_receipt_blockers(tmp_path, directory, {}, {})
    # Even a matching directory basename in a different chapter is not migrated.
    assert not uses_native_receipts(tmp_path, tmp_path / "unrelated" / "part17_shapiq_estimators")
