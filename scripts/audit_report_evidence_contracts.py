"""Audit verification reports for non-placeholder evidence contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_extension_verification_reports import extension_records, expected_metric_failures
from arena_ext.verification import verify_report_input_manifest


PLACEHOLDER_EVIDENCE_LEVELS = {
    "notebook_contract",
    "placeholder_only",
}
PLACEHOLDER_CLAIM_TERMS = (
    "placeholder",
    "starter report",
)


def _has_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _has_nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _has_negative_control(controls: list[Any]) -> bool:
    text = " ".join(str(control).lower() for control in controls)
    return any(term in text for term in ("negative", "random", "shuffle", "permut", "control"))


def verification_report_records() -> list[dict[str, Path]]:
    records = [
        {
            "lock_path": record["lock_path"],
            "report_path": record["report_path"],
        }
        for record in extension_records()
    ]
    for report_path in sorted((ROOT / "docs/evidence").glob("*/verification_report.json")):
        lock_path = report_path.with_name("artifacts.lock.yml")
        if lock_path.exists():
            records.append({"lock_path": lock_path, "report_path": report_path})
    return records


def report_freshness_blockers(report: dict[str, Any]) -> list[str]:
    notebook_id = str(report.get("notebook_id", "<unknown>"))
    return [
        f"{notebook_id}: {blocker}"
        for blocker in verify_report_input_manifest(ROOT, report.get("report_inputs", {}))
    ]


def report_contract_blockers(lock: dict[str, Any], report: dict[str, Any]) -> list[str]:
    notebook_id = str(lock.get("notebook_id", report.get("notebook_id", "<unknown>")))
    blockers: list[str] = []

    if report.get("notebook_id") != lock.get("notebook_id"):
        blockers.append(f"{notebook_id}: report notebook_id does not match lockfile")
    if report.get("gt_tier") != lock.get("gt_tier"):
        blockers.append(f"{notebook_id}: report gt_tier does not match lockfile")
    if report.get("evidence_level") != lock.get("evidence_level"):
        blockers.append(f"{notebook_id}: report evidence_level does not match lockfile")
    if report.get("claim_scope") != lock.get("claim_scope"):
        blockers.append(f"{notebook_id}: report claim_scope does not match lockfile")

    evidence_level = str(report.get("evidence_level", ""))
    claim_scope = str(report.get("claim_scope", ""))
    if evidence_level in PLACEHOLDER_EVIDENCE_LEVELS:
        blockers.append(f"{notebook_id}: placeholder evidence_level={evidence_level}")
    if len(claim_scope) < 80:
        blockers.append(f"{notebook_id}: claim_scope is too thin")
    lower_claim_scope = claim_scope.lower()
    for term in PLACEHOLDER_CLAIM_TERMS:
        if term in lower_claim_scope:
            blockers.append(f"{notebook_id}: claim_scope contains placeholder term {term!r}")

    if report.get("tests_passed") is not True or report.get("accepted") is not True:
        blockers.append(f"{notebook_id}: report is not accepted")
    if report.get("known_failures") not in ([], None):
        blockers.append(f"{notebook_id}: accepted report has known_failures")

    metrics = report.get("metrics", {})
    evidence = metrics.get("gpu_evidence", {}) if isinstance(metrics, dict) else {}
    if evidence.get("category") != "cuda_section_metric":
        blockers.append(f"{notebook_id}: gpu_evidence.category={evidence.get('category')!r}")
    if evidence.get("uses_cuda") is not True:
        blockers.append(f"{notebook_id}: gpu_evidence.uses_cuda is not true")
    if evidence.get("placeholder_only") is True:
        blockers.append(f"{notebook_id}: gpu_evidence is placeholder_only")
    if not evidence.get("section_specific_metric_keys"):
        blockers.append(f"{notebook_id}: gpu_evidence lacks section-specific metric keys")

    controls = lock.get("controls", [])
    if not _has_nonempty_list(controls):
        blockers.append(f"{notebook_id}: lockfile controls are empty")
    elif not _has_negative_control(controls):
        blockers.append(f"{notebook_id}: lockfile controls lack a negative/random control")

    baselines = report.get("baselines")
    if not _has_mapping(baselines):
        blockers.append(f"{notebook_id}: baselines are empty")
    else:
        if not baselines.get("declared_controls"):
            blockers.append(f"{notebook_id}: baselines.declared_controls is empty")
        if not baselines.get("expected_metrics"):
            blockers.append(f"{notebook_id}: baselines.expected_metrics is empty")

    negative_controls = report.get("negative_controls")
    if not _has_mapping(negative_controls):
        blockers.append(f"{notebook_id}: negative_controls are empty")
    elif not negative_controls.get("declared_controls"):
        blockers.append(f"{notebook_id}: negative_controls.declared_controls is empty")

    if not _has_mapping(report.get("ood_tests")):
        blockers.append(f"{notebook_id}: ood_tests are empty")

    if not _has_nonempty_list(report.get("models")):
        blockers.append(f"{notebook_id}: models are empty")
    if not _has_nonempty_list(report.get("datasets")):
        blockers.append(f"{notebook_id}: datasets are empty")
    if not _has_nonempty_list(report.get("safety_notes")):
        blockers.append(f"{notebook_id}: safety_notes are empty")

    blockers.extend(f"{notebook_id}: {failure}" for failure in expected_metric_failures(
        lock.get("expected_metrics", {}),
        report.get("metrics", {}),
    ))
    blockers.extend(report_freshness_blockers(report))
    return blockers


def report_evidence_blockers() -> list[str]:
    blockers: list[str] = []
    for record in verification_report_records():
        lock = yaml.safe_load(record["lock_path"].read_text())
        report = json.loads(record["report_path"].read_text())
        blockers.extend(report_contract_blockers(lock, report))
    return blockers


def main() -> None:
    records = verification_report_records()
    blockers = report_evidence_blockers()
    print(f"reports_checked={len(records)}")
    print(f"report_evidence_blockers={len(blockers)}")
    if blockers:
        print("REPORT_EVIDENCE_CONTRACTS=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("REPORT_EVIDENCE_CONTRACTS=PASS")


if __name__ == "__main__":
    main()
