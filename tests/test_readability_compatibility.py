"""Fail-closed mutation tests for the named cosmetic publication bridge."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

import pytest

from arena_ext import readability_compatibility as compat
from scripts.evidence_bundles import file_record, validate_review_bundle

RETIRED_PROOF_DIR = Path("docs/mastery/evidence/2026-10-16-historical-readability")
RETIRED_MAP_SHA256 = "4d84a8186612065a7ba043aa214fc4fdc73b099231d2cdd1aba876974f2654da"
RETIRED_PATHS = frozenset({
    "chapter8_automated_circuits/exercises/part3_acdc_circuit_metrics/edge_evidence.py",
    "chapter8_automated_circuits/exercises/part3_acdc_circuit_metrics/lesson.md",
})


def sha(data):
    return hashlib.sha256(data).hexdigest()


def historical_publication_bytes(entry, *, root=None):
    """Test-only historical proof; never a production execution-source fallback."""
    root = compat.ROOT if root is None else root
    record_bytes = compat.RECORD_PATH.read_bytes()
    if sha(record_bytes) != compat.RECORD_SHA256:
        raise ValueError("Pinned historical manifest changed")
    registered = json.loads(record_bytes)["source_files"]
    if [item for item in registered if item["path"] == entry["path"]] != [entry]:
        raise ValueError("Unknown historical publication identity")
    path = root / entry["path"]
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Current publication is not a regular file")
        data = path.read_bytes()
        if sha(data) != entry["after_sha256"]:
            raise ValueError("Present publication changed; archive cannot hide it")
        return data
    if entry["path"] not in RETIRED_PATHS:
        raise ValueError("Missing publication has no retired historical mapping")
    mapping_bytes = (root / RETIRED_PROOF_DIR / "retired-paths.json").read_bytes()
    if sha(mapping_bytes) != RETIRED_MAP_SHA256:
        raise ValueError("Pinned retired mapping changed")
    mapping = json.loads(mapping_bytes)
    if (mapping["schema"] != "arena.retired-readability-publications.v1"
            or set(mapping["retired_paths"]) != RETIRED_PATHS
            or mapping["historical_record"]["sha256"] != compat.RECORD_SHA256):
        raise ValueError("Unknown retired mapping or historical record")
    member = mapping["retired_paths"][entry["path"]]
    if (member["archive_member"] != entry["path"]
            or member["sha256"] != entry["after_sha256"]
            or member["replacement_for_this_proof"] is not None):
        raise ValueError("Historical proof identity cannot be replaced")
    archive = root / mapping["bundle"]["repository_path"]
    sidecar = Path(str(archive) + ".review-bundle.json")
    if (not archive.resolve().is_relative_to(root.resolve())
            or archive.is_symlink() or sidecar.is_symlink()
            or mapping["contract"]["repository_path"] != mapping["bundle"]["repository_path"] + ".review-bundle.json"):
        raise ValueError("Invalid historical archive locator")
    for path, expected in ((archive, mapping["bundle"]), (sidecar, mapping["contract"])):
        if file_record(path) != {key: expected[key] for key in ("sha256", "size_bytes")}:
            raise ValueError("Historical archive or contract hash changed")
    contract = validate_review_bundle(archive)
    if set(contract["members"]) != RETIRED_PATHS:
        raise ValueError("Unexpected historical archive member inventory")
    with tarfile.open(archive, "r:gz") as saved:
        data = saved.extractfile(entry["path"]).read()
    if sha(data) != entry["after_sha256"]:
        raise ValueError("Archived publication changed")
    return data


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
        publication = historical_publication_bytes(entry)
        assert sha(publication) == entry["after_sha256"]
        assert sha(entry["before_source"].encode()) == entry["before_sha256"]
        assert compat._equivalent_source(
            entry["path"], entry["before_source"], publication.decode(), entry["equivalence"]
        )
        if path.exists():
            assert compat.source_matches_execution(path, entry["before_sha256"])
        else:
            assert not compat.source_matches_execution(path, entry["before_sha256"])


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
        current = historical_publication_bytes(entry).decode()
        assert compat._equivalent_source(entry["path"], entry["before_source"], current, entry["equivalence"])
        assert not compat._equivalent_source(
            entry["path"], entry["before_source"], current + "\nNUMERICAL_TOLERANCE = 1e-3\n", entry["equivalence"]
        )


@pytest.fixture
def retired_proof_fixture(tmp_path):
    """Copy immutable artifacts into a temporary root for destructive mutations."""
    original = compat.ROOT / RETIRED_PROOF_DIR
    destination = tmp_path / RETIRED_PROOF_DIR
    destination.mkdir(parents=True)
    for name in ("retired-paths.json", "retired_8_3_readability_publications.tar.gz",
                 "retired_8_3_readability_publications.tar.gz.review-bundle.json"):
        shutil.copyfile(original / name, destination / name)
    record = json.loads(compat.RECORD_PATH.read_text())
    entry = next(item for item in record["source_files"] if item["path"] == sorted(RETIRED_PATHS)[0])
    return tmp_path, destination, entry


def test_archived_proof_never_resurrects_production_execution_sources():
    capture = json.loads((compat.ROOT / RETIRED_PROOF_DIR / "git-capture.json").read_text())
    assert file_record(Path(compat.__file__)) == capture["production_helper_before"]
    record = json.loads(compat.RECORD_PATH.read_text())
    assert len(record["source_files"]) == 11
    for entry in record["source_files"]:
        if entry["path"] not in RETIRED_PATHS:
            continue
        path = compat.ROOT / entry["path"]
        assert not path.exists()
        assert not compat.source_matches_execution(path, entry["before_sha256"])
        assert sha(historical_publication_bytes(entry)) == entry["after_sha256"]


def test_historical_archive_cannot_hide_a_changed_present_publication(retired_proof_fixture):
    root, _, entry = retired_proof_fixture
    current = root / entry["path"]
    current.parent.mkdir(parents=True)
    current.write_text("NUMERICAL_TOLERANCE = 0.1\n")
    with pytest.raises(ValueError, match="Present publication changed"):
        historical_publication_bytes(entry, root=root)
    assert not compat.source_matches_execution(current, entry["before_sha256"])


@pytest.mark.parametrize("kind", ["missing", "changed"])
def test_historical_proof_requires_unchanged_pinned_mapping(retired_proof_fixture, kind):
    root, destination, entry = retired_proof_fixture
    mapping = destination / "retired-paths.json"
    if kind == "missing":
        mapping.unlink()
    else:
        mapping.write_bytes(mapping.read_bytes() + b" ")
    with pytest.raises((ValueError, OSError)):
        historical_publication_bytes(entry, root=root)


@pytest.mark.parametrize("kind", ["missing_entry", "unknown_entry", "renamed_member", "replacement", "wrong_record"])
def test_mapping_schema_cannot_expand_or_redirect_historical_proof(retired_proof_fixture, monkeypatch, kind):
    root, destination, entry = retired_proof_fixture
    path = destination / "retired-paths.json"
    mapping = json.loads(path.read_text())
    if kind == "missing_entry":
        mapping["retired_paths"].pop(sorted(RETIRED_PATHS)[-1])
    elif kind == "unknown_entry":
        mapping["retired_paths"]["unknown.py"] = deepcopy(mapping["retired_paths"][entry["path"]])
    elif kind == "renamed_member":
        mapping["retired_paths"][entry["path"]]["archive_member"] = "chapter8_automated_circuits/exercises/part4_acdc_circuit_metrics/solutions.py"
    elif kind == "replacement":
        mapping["retired_paths"][entry["path"]]["replacement_for_this_proof"] = "8.4"
    else:
        mapping["historical_record"]["sha256"] = "0" * 64
    path.write_text(json.dumps(mapping))
    # Rebind only the synthetic fixture pin to exercise the deeper identity
    # checks. The real protected mapping is never modified.
    monkeypatch.setattr(sys.modules[__name__], "RETIRED_MAP_SHA256", sha(path.read_bytes()))
    with pytest.raises(ValueError):
        historical_publication_bytes(entry, root=root)


@pytest.mark.parametrize("name", ["retired_8_3_readability_publications.tar.gz",
                                  "retired_8_3_readability_publications.tar.gz.review-bundle.json"])
def test_archived_publication_or_contract_tampering_fails(retired_proof_fixture, name):
    root, destination, entry = retired_proof_fixture
    path = destination / name
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash changed"):
        historical_publication_bytes(entry, root=root)


def test_missing_active_publication_does_not_gain_archive_fallback(retired_proof_fixture):
    root, _, _ = retired_proof_fixture
    record = json.loads(compat.RECORD_PATH.read_text())
    entry = next(item for item in record["source_files"] if item["path"] not in RETIRED_PATHS)
    with pytest.raises(ValueError, match="no retired historical mapping"):
        historical_publication_bytes(entry, root=root)


def test_proof_cannot_transfer_to_8_4_even_with_identical_historical_bytes(retired_proof_fixture):
    root, _, entry = retired_proof_fixture
    unrelated = deepcopy(entry)
    unrelated["path"] = "chapter8_automated_circuits/exercises/part4_acdc_circuit_metrics/solutions.py"
    path = root / unrelated["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(historical_publication_bytes(entry, root=root))
    with pytest.raises(ValueError, match="Unknown historical publication identity"):
        historical_publication_bytes(unrelated, root=root)
    assert not compat.source_matches_execution(path, entry["before_sha256"])


def test_historical_proof_does_not_accept_a_changed_original_manifest(retired_proof_fixture, monkeypatch):
    root, destination, entry = retired_proof_fixture
    record = destination / "changed-original-manifest.json"
    record.write_bytes(compat.RECORD_PATH.read_bytes() + b" ")
    monkeypatch.setattr(compat, "RECORD_PATH", record)
    with pytest.raises(ValueError, match="Pinned historical manifest changed"):
        historical_publication_bytes(entry, root=root)
