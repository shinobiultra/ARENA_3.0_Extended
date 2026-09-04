"""Strict release audit for the ARENA extension.

This gate is intentionally stricter than the normal verification runner. The
runner proves each current report's declared claim. This script fails when a
required roadmap target is still gated, pending, or explicitly not ready.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_hard_exercise_ladders import ladder_blockers
from scripts.audit_course_surface import course_surface_blockers
from scripts.audit_extension_artifact_hygiene import artifact_hygiene_blockers
from scripts.audit_report_evidence_contracts import report_evidence_blockers
from scripts.audit_roadmap_final_completeness import roadmap_final_completeness_blockers


def iter_leaf_values(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_leaf_values(value, child)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from iter_leaf_values(value, child)
    else:
        yield prefix, obj


def optional_false_ready_metric(notebook_id: str, key: str) -> bool:
    """Return true for optional readiness checks that should not block release.

    The 5.5 roadmap requirement is the quantized DiffusionGemma generation path
    under the 24GB GPU budget. The full Google BF16 checkpoint is still reported
    for transparency, but direct BF16 local loading is not a required release
    path because it exceeds the intended local tier.
    """

    return (
        notebook_id == "5_5_diffusion_language_models"
        and key == "gpu_test.diffusiongemma_bf16_local_ready_for_direct_loading"
    )


def report_blockers() -> list[str]:
    blockers: list[str] = []
    reports = sorted(
        set(ROOT.glob("chapter*/exercises/part*/verification_report.json"))
        | set(ROOT.glob("docs/evidence/*/verification_report.json"))
    )
    for path in reports:
        report = json.loads(path.read_text())
        notebook_id = report.get("notebook_id", str(path))
        evidence = report.get("metrics", {}).get("gpu_evidence", {})
        if report.get("accepted") is not True or report.get("tests_passed") is not True:
            blockers.append(f"{notebook_id}: report is not accepted")
        if evidence.get("category") != "cuda_section_metric" or evidence.get("uses_cuda") is not True:
            blockers.append(f"{notebook_id}: lacks section-specific CUDA evidence")

        for key, value in iter_leaf_values(report.get("metrics", {})):
            if optional_false_ready_metric(notebook_id, key):
                continue
            if key.endswith("ready_for_real_activations") and value is False:
                blockers.append(f"{notebook_id}: {key}=False")
            if key.endswith("ready_for_direct_loading") and value is False:
                blockers.append(f"{notebook_id}: {key}=False")
            if key.endswith("generation_ready") and value is False:
                blockers.append(f"{notebook_id}: {key}=False")
            if key.endswith("gated_unavailable") and value is True:
                blockers.append(f"{notebook_id}: {key}=True")
    return blockers


def required_artifact_blocker_for_row(row: dict[str, str]) -> str | None:
    local_status = row.get("local_status", "")
    revision = row.get("revision", "")
    if local_status == "REQUIRED_GATED_PENDING":
        return (
            f"{row['name']}: required artifact is gated/pending "
            f"({row['repo_or_source_id']})"
        )
    if local_status == "REQUIRED_PENDING_VERIFIED_REPORT":
        return (
            f"{row['name']}: required artifact lacks a modern verified report "
            f"({row['repo_or_source_id']})"
        )
    if local_status == "REQUIRED" and revision.startswith("pin_before_"):
        return (
            f"{row['name']}: required artifact revision is unresolved "
            f"({revision}; {row['repo_or_source_id']})"
        )
    return None


def required_artifact_blockers() -> list[str]:
    blockers: list[str] = []
    path = ROOT / "docs/artifact_registry.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            blocker = required_artifact_blocker_for_row(row)
            if blocker is not None:
                blockers.append(blocker)
    return blockers


def main() -> None:
    report_failures = report_blockers()
    artifact_failures = required_artifact_blockers()
    ladder_failures = [f"hard exercise ladder: {blocker}" for blocker in ladder_blockers()]
    evidence_failures = [
        f"report evidence contract: {blocker}" for blocker in report_evidence_blockers()
    ]
    surface_failures = [f"course surface: {blocker}" for blocker in course_surface_blockers()]
    hygiene_failures = [
        f"artifact hygiene: {blocker}" for blocker in artifact_hygiene_blockers()
    ]
    roadmap_failures = [
        f"roadmap final completeness: {blocker}"
        for blocker in roadmap_final_completeness_blockers()
    ]
    blockers = (
        report_failures
        + artifact_failures
        + ladder_failures
        + evidence_failures
        + surface_failures
        + hygiene_failures
        + roadmap_failures
    )

    report_count = len(
        set(ROOT.glob("chapter*/exercises/part*/verification_report.json"))
        | set(ROOT.glob("docs/evidence/*/verification_report.json"))
    )
    print(f"reports_checked={report_count}")
    print(f"report_blockers={len(report_failures)}")
    print(f"required_artifact_blockers={len(artifact_failures)}")
    print(f"hard_exercise_ladder_blockers={len(ladder_failures)}")
    print(f"report_evidence_contract_blockers={len(evidence_failures)}")
    print(f"course_surface_blockers={len(surface_failures)}")
    print(f"artifact_hygiene_blockers={len(hygiene_failures)}")
    print(f"roadmap_final_completeness_blockers={len(roadmap_failures)}")
    if blockers:
        print("STRICT_COMPLETION=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("STRICT_COMPLETION=PASS")


if __name__ == "__main__":
    main()
