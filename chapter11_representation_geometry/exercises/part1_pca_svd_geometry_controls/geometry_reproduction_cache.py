"""Reuse independently executed CUDA evidence after strictly non-scientific edits.

This does not execute a model. It verifies an archived native run, every source
dependency of that run, the current learner's executed Python, and every one of
its 6,656 actual comparison rows. Report metadata explicitly records reuse.
"""

from __future__ import annotations

from arena_ext.readability_compatibility import source_matches_execution

import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry_evidence import read_gzip, sha256, validate_learner

HERE = Path(__file__).resolve().parent
NATIVE_INPUTS = {
    "geometry_cuda.py",
    "geometry_evidence.py",
    "geometry_study.py",
    "geometry_methods.py",
    "geometry_data.py",
    "geometry_protocol.json",
}


def compare_rows(actual: list[dict], expected: list[dict]) -> dict:
    """Require the exact condition/example inventory and compare measured outputs."""

    def key(row):
        return row["family"], row["row_id"], row["condition"], row["draw"]

    wanted = {key(row): row for row in expected}
    observed = {key(row): row for row in actual}
    if len(wanted) != len(expected) or len(observed) != len(actual):
        raise ValueError("A reproduction contains repeated condition/example rows.")
    if observed.keys() != wanted.keys():
        raise ValueError("The reproduction must contain every expected row and no others.")
    error = 0.0
    mismatches = 0
    for row_key, row in observed.items():
        reference = wanted[row_key]
        values = np.asarray(row["calendar_logits"], dtype=np.float64)
        correct = np.asarray(reference["calendar_logits"], dtype=np.float64)
        if values.shape != correct.shape or not np.isfinite(values).all():
            raise ValueError("Missing or nonfinite reproduced logits.")
        error = max(error, float(np.max(np.abs(values - correct))))
        mismatches += row["top_token_id"] != reference["top_token_id"]
        for name in (
            "original_answer",
            "desired_answer",
            "restricted_answer",
            "original_full_correct",
            "desired_full_correct",
            "shift",
        ):
            if row[name] != reference[name]:
                raise ValueError("Reproduced answer metadata differs: " + name)
    if error > 1e-6 or mismatches:
        raise ValueError("Cached native logits or winning tokens differ from the learner.")
    return {
        "compared_examples": len(actual),
        "max_calendar_logit_error": error,
        "top_token_mismatches": mismatches,
    }


def validate_native_sources(source: Path, state: dict, here: Path = HERE):
    if set(state["source_inputs"]) != NATIVE_INPUTS:
        raise ValueError("The native reproduction source inventory is incomplete.")
    for name, digest in state["source_inputs"].items():
        if not source_matches_execution(here / name, digest, source / "source_snapshot" / name):
            raise ValueError("Native scientific source changed: " + name)


def validate_cached_reproduction(learner: Path, manifest_path: Path) -> tuple[dict, Path]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("all_archive_members_read_back_and_verified"):
        raise ValueError("Require an archive which was read back and verified.")
    archive = manifest_path.parent / "execution.tar.gz"
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Archived native evidence changed.")
    source = Path(manifest["source_directory"])
    for name, digest in manifest["members_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError("The retained native artifact differs from its archive: " + name)
    if not {"reproduction.json", "actual_outcomes.json.gz"} <= manifest["members_sha256"].keys():
        raise ValueError("Both the native execution and actual outcomes must be archived.")
    state = json.loads((source / "reproduction.json").read_text())
    if state["status"] != "PASS" or state["changed_sources"]:
        raise ValueError("Cannot reuse an incomplete or failed native reproduction.")
    validate_native_sources(source, state)
    audit = validate_learner(learner)
    if audit != state["learner_audit"]:
        raise ValueError("The completed learner differs from the independently reproduced study.")
    study = read_gzip(learner / "experiments/measured_study.json.gz")
    protocol = study["protocol"]
    first_batch = {
        family: {row["id"] for row in task["rows"]["final"][: protocol["batch_size"]]}
        for family, task in study["tasks"].items()
    }
    expected = [
        row
        for row in study["outcomes"]
        if row["row_id"] in first_batch[row["family"]] and row["shift"] == 2
    ]
    actual = read_gzip(source / "actual_outcomes.json.gz")
    counts = compare_rows(actual, expected)
    if counts["compared_examples"] != 6656 or state["condition_batches"] != 208:
        raise ValueError("The full first-batch native comparison is required.")
    for name, value in counts.items():
        if state[name] != value:
            raise ValueError("Native summary disagrees with the actual reproduced rows: " + name)
    conditions = {(row["family"], row["condition"], row["draw"]) for row in actual}
    if len(conditions) != 208:
        raise ValueError("A native control or null draw is missing.")
    result = {
        **state,
        "reused_native_reproduction": True,
        "native_cuda_executed_this_report": False,
        "reproduction_manifest_sha256": sha256(manifest_path),
        "reproduction_manifest_path": str(manifest_path),
        "reproduction_record_sha256": sha256(source / "reproduction.json"),
        "reproduction_actual_rows_sha256": sha256(source / "actual_outcomes.json.gz"),
    }
    return result, source
