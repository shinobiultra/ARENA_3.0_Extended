"""Fail-closed mutation tests for the named cosmetic publication bridge."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from arena_ext import readability_compatibility as compat


def sha(data):
    return hashlib.sha256(data).hexdigest()


def install_fixture(tmp_path, monkeypatch, *, markdown=False):
    name = "lesson.md" if markdown else "method.py"
    before = (
        "A measured result with a random control.\n\n```python\nx = 1e-6\n```\n"
        if markdown else 'def method(x):\n    return x + 1, "preserve", 1e-6\n'
    )
    after = (
        "A measured result with a\nrandom control.\n\n```python\nx = 1e-6\n```\n"
        if markdown else 'def method(x):\n    return (\n        x + 1, "preserve", 1e-6\n    )\n'
    )
    path = tmp_path / name
    path.write_text(after)
    snapshot = tmp_path / "source_snapshot" / name
    snapshot.parent.mkdir()
    snapshot.write_text(before)
    record = {
        "schema_version": 1,
        "native_experiments_rerun": False,
        "source_files": [{
            "path": name,
            "before_source": before,
            "before_sha256": sha(before.encode()),
            "after_sha256": sha(after.encode()),
            "equivalence": "commonmark_whitespace_only" if markdown else "python_ast",
        }],
    }
    record_path = tmp_path / "compatibility.json"
    monkeypatch.setattr(compat, "ROOT", tmp_path)
    monkeypatch.setattr(compat, "RECORD_PATH", record_path)

    def pin(value):
        record_path.write_text(json.dumps(value))
        monkeypatch.setattr(compat, "RECORD_SHA256", sha(record_path.read_bytes()))

    pin(record)
    return path, snapshot, record, pin


def test_every_real_registered_publication_has_exact_semantic_proof():
    record = json.loads(compat.RECORD_PATH.read_text())
    assert sha(compat.RECORD_PATH.read_bytes()) == compat.RECORD_SHA256
    assert len(record["source_files"]) == 11
    for entry in record["source_files"]:
        path = compat.ROOT / entry["path"]
        assert sha(path.read_bytes()) == entry["after_sha256"]
        assert compat.source_matches_execution(path, entry["before_sha256"])


def test_exact_execution_needs_no_compatibility_record(tmp_path, monkeypatch):
    path = tmp_path / "method.py"
    path.write_text("x = 1\n")
    monkeypatch.setattr(compat, "RECORD_PATH", tmp_path / "missing.json")
    assert compat.source_matches_execution(path, sha(path.read_bytes()))


@pytest.mark.parametrize("markdown", [False, True])
def test_only_registered_cosmetic_version_passes(tmp_path, monkeypatch, markdown):
    path, snapshot, record, _ = install_fixture(tmp_path, monkeypatch, markdown=markdown)
    expected = record["source_files"][0]["before_sha256"]
    assert compat.source_matches_execution(path, expected, snapshot)
    path.write_text(path.read_text() + "\n")
    assert not compat.source_matches_execution(path, expected, snapshot)


@pytest.mark.parametrize("old,new", [("+ 1", "- 1"), ('"preserve"', '"changed"'), ("1e-6", "1e-3")])
def test_arithmetic_literal_and_tolerance_changes_fail_even_with_rebound_hash(tmp_path, monkeypatch, old, new):
    path, snapshot, record, pin = install_fixture(tmp_path, monkeypatch)
    path.write_text(path.read_text().replace(old, new))
    record["source_files"][0]["after_sha256"] = sha(path.read_bytes())
    pin(record)
    assert not compat.source_matches_execution(path, record["source_files"][0]["before_sha256"], snapshot)


@pytest.mark.parametrize("old,new", [("random control", "successful control"), ("x = 1e-6", "x = 1e-3")])
def test_markdown_claim_and_code_fence_changes_fail_even_with_rebound_hash(tmp_path, monkeypatch, old, new):
    path, snapshot, record, pin = install_fixture(tmp_path, monkeypatch, markdown=True)
    path.write_text(path.read_text().replace(old, new))
    record["source_files"][0]["after_sha256"] = sha(path.read_bytes())
    pin(record)
    assert not compat.source_matches_execution(path, record["source_files"][0]["before_sha256"], snapshot)


def test_tampered_missing_and_duplicate_receipts_fail(tmp_path, monkeypatch):
    path, snapshot, record, pin = install_fixture(tmp_path, monkeypatch)
    expected = record["source_files"][0]["before_sha256"]
    compat.RECORD_PATH.write_text(compat.RECORD_PATH.read_text() + " ")
    assert not compat.source_matches_execution(path, expected, snapshot)
    compat.RECORD_PATH.unlink()
    assert not compat.source_matches_execution(path, expected, snapshot)
    record["source_files"] *= 2
    pin(record)
    assert not compat.source_matches_execution(path, expected, snapshot)


def test_execution_snapshot_and_expected_hash_are_still_authoritative(tmp_path, monkeypatch):
    path, snapshot, record, _ = install_fixture(tmp_path, monkeypatch)
    expected = record["source_files"][0]["before_sha256"]
    assert not compat.source_matches_execution(path, "0" * 64, snapshot)
    snapshot.write_text(snapshot.read_text().replace("+ 1", "+ 2"))
    assert not compat.source_matches_execution(path, expected, snapshot)
    snapshot.unlink()
    assert not compat.source_matches_execution(path, expected, snapshot)


def test_unknown_path_has_no_future_cosmetic_bypass(tmp_path, monkeypatch):
    path, snapshot, record, _ = install_fixture(tmp_path, monkeypatch)
    other = tmp_path / "unlisted.py"
    other.write_bytes(path.read_bytes())
    assert not compat.source_matches_execution(other, record["source_files"][0]["before_sha256"], snapshot)


def test_record_cannot_supply_arbitrary_guard_replacements(tmp_path, monkeypatch):
    path, snapshot, record, pin = install_fixture(tmp_path, monkeypatch)
    entry = record["source_files"][0]
    path.write_text(path.read_text().replace("+ 1", "- 1"))
    entry.update(
        after_sha256=sha(path.read_bytes()),
        equivalence="python_ast_with_reviewed_validation_guard",
        guard_replacements=[["+ 1", "- 1"]],
    )
    pin(record)
    assert not compat.source_matches_execution(path, entry["before_sha256"], snapshot)


def test_reviewed_guard_does_not_hide_scientific_edits():
    record = json.loads(compat.RECORD_PATH.read_text())
    entries = [e for e in record["source_files"] if e["equivalence"] == "python_ast_with_reviewed_validation_guard"]
    assert len(entries) == 4
    for entry in entries:
        current = (compat.ROOT / entry["path"]).read_text()
        assert compat._equivalent_source(entry["path"], entry["before_source"], current, entry["equivalence"])
        assert not compat._equivalent_source(
            entry["path"], entry["before_source"], current + "\nNUMERICAL_TOLERANCE = 1e-3\n", entry["equivalence"]
        )
