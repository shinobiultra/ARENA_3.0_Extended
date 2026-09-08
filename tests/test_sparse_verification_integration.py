"""Exercise the shared audit adapters with explicitly synthetic native receipts."""

import hashlib
import json

import pytest

from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_report
from chapter16_shapley_attribution_baselines.exercises.part18_spex.native_report_tests import (
    _fresh, _report, context as synthetic_native_context,
)
from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence
from scripts.verification_fixture_contract import (
    CHAPTER, LEGACY_FIXTURE_NAMES, NATIVE_FIXTURE_NAMES, SPARSE_RECEIPT_SECTIONS,
    fixture_paths, native_receipt_blockers, uses_native_receipts, verification_entrypoint,
)


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
    assert verification_entrypoint(root, directory) == root / CHAPTER / "part18_spex/native_report.py"


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
