"""Reject incomplete, stale or tampered learner evidence before GPU evaluation."""
from __future__ import annotations

import gzip
import json

import pytest

from . import edge_evidence as evidence


@pytest.fixture
def complete_record(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "HERE", tmp_path)
    (tmp_path / "lesson.md").write_text("The exact executed lesson.\n")
    (tmp_path / "method.py").write_text("def method():\n    return 1\n")
    for suffix in ("exercises", "solutions"):
        (tmp_path / f"{evidence.STEM}_{suffix}.ipynb").write_text("{}")
    search = {"complete": True, "trace": [{"decision": "keep"}], "initial_loss": 0.}
    payload = {
        "fresh_searches": True, "model_revision": evidence.real_edges.GPT2_REVISION,
        "ioi": {"searches": [{**search, "threshold": threshold} for threshold in (.05, .004, .001, .0005)],
                "development_rows": [{"threshold": .001, "recovery": .8815, "pairwise_accuracy": 27 / 32,
                                      "clean_pairwise_accuracy": 31 / 32, "kl_clean_to_circuit": .4088}],
                "selection_passed": False, "heldout": None, "frozen_selection": None,
                "evaluation_split": "development", "diagnostic_candidate": {"threshold": .001}},
        "greater_than": {"search": {**search, "threshold": .004}},
    }
    directory = tmp_path / "expected_outputs"
    directory.mkdir()
    raw = directory / "completed_edge_study.json.gz"
    raw.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
    manifest = {
        "status": "PASS", "scope": "full", "completed_from_own_answers": True,
        "fresh_searches_required": True, "sources_changed_during_run": [],
        "input_files": {path.name: evidence.sha256(path) for path in (tmp_path / "lesson.md", tmp_path / "method.py")},
        "published_evidence_sha256": evidence.sha256(raw),
        "published_exercises_sha256": evidence.sha256(tmp_path / f"{evidence.STEM}_exercises.ipynb"),
        "published_notebook_sha256": evidence.sha256(tmp_path / f"{evidence.STEM}_solutions.ipynb"),
    }
    record = directory / "learner_execution.json"
    record.write_text(json.dumps(manifest))
    return directory, record, raw, manifest


def test_completed_evidence_rejects_partial_or_stale_sources(complete_record):
    directory, record, _, manifest = complete_record
    evidence.read_completed_learner_evidence(directory)
    for change in ({"scope": "greater-than"}, {"status": "FAIL"},
                   {"fresh_searches_required": False}, {"sources_changed_during_run": ["method.py"]}):
        record.write_text(json.dumps({**manifest, **change}))
        with pytest.raises(ValueError, match="successful full"):
            evidence.read_completed_learner_evidence(directory)
    record.write_text(json.dumps(manifest))
    (directory.parent / "method.py").write_text("def method():\n    return 2\n")
    with pytest.raises(ValueError, match="stale"):
        evidence.read_completed_learner_evidence(directory)


def test_completed_evidence_rejects_missing_source_and_changed_payload(complete_record):
    directory, record, raw, manifest = complete_record
    reduced = {**manifest, "input_files": {"lesson.md": manifest["input_files"]["lesson.md"]}}
    record.write_text(json.dumps(reduced))
    with pytest.raises(ValueError, match="every section source"):
        evidence.read_completed_learner_evidence(directory)
    record.write_text(json.dumps(manifest))
    raw.write_bytes(raw.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="raw learner evidence"):
        evidence.read_completed_learner_evidence(directory)


def test_completed_evidence_requires_actual_complete_sweeps(complete_record):
    directory, record, raw, manifest = complete_record
    original = json.loads(gzip.decompress(raw.read_bytes()))
    for change in ("missing_threshold", "empty_trace", "partial_search", "cached_search"):
        payload = json.loads(json.dumps(original))
        if change == "missing_threshold":
            payload["ioi"]["searches"].pop()
        elif change == "empty_trace":
            payload["ioi"]["searches"][0]["trace"] = []
        elif change == "partial_search":
            payload["greater_than"]["search"]["complete"] = False
        else:
            payload["fresh_searches"] = False
        raw.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
        record.write_text(json.dumps({**manifest, "published_evidence_sha256": evidence.sha256(raw)}))
        with pytest.raises(ValueError):
            evidence.read_completed_learner_evidence(directory)


def test_completed_evidence_rejects_changed_published_notebook(complete_record):
    directory, _, _, _ = complete_record
    (directory.parent / f"{evidence.STEM}_solutions.ipynb").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Published solutions"):
        evidence.read_completed_learner_evidence(directory)


def test_failed_selection_cannot_be_relabelled_as_heldout_transfer(complete_record):
    directory, record, raw, manifest = complete_record
    original = json.loads(gzip.decompress(raw.read_bytes()))
    for change in ({"selection_passed": True}, {"heldout": {"prompts": ["opened"]}},
                   {"evaluation_split": "heldout"}, {"frozen_selection": {"threshold": .001}},
                   {"diagnostic_candidate": {"threshold": .004}}):
        payload = json.loads(json.dumps(original))
        payload["ioi"].update(change)
        raw.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
        record.write_text(json.dumps({**manifest, "published_evidence_sha256": evidence.sha256(raw)}))
        with pytest.raises(ValueError, match="IOI"):
            evidence.read_completed_learner_evidence(directory)
