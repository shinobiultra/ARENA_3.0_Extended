"""Audit GPU evidence in generated extension verification reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def classify_gpu_result(gpu_result: Any) -> dict[str, Any]:
    """Match the report-runner classification for already-written reports."""

    if not isinstance(gpu_result, dict):
        return {
            "category": "missing",
            "uses_cuda": False,
            "placeholder_only": True,
            "section_specific_metric_keys": [],
        }

    generic_keys = {
        "cuda_available",
        "cuda_version",
        "device",
        "full_path",
        "gpu_total_memory_gb",
        "gpu_name",
        "peak_vram_gb",
        "smoke_test_available",
        "torch_version",
        "within_vram_budget",
    }
    section_specific_keys = sorted(key for key in gpu_result if key not in generic_keys)
    placeholder_only = (
        gpu_result.get("smoke_test_available") is True
        and not section_specific_keys
        and gpu_result.get("cuda_available") is not True
    )
    uses_cuda = gpu_result.get("cuda_available") is True
    if uses_cuda and section_specific_keys:
        category = "cuda_section_metric"
    elif uses_cuda:
        category = "cuda_environment_or_budget"
    elif placeholder_only:
        category = "placeholder_only"
    elif section_specific_keys:
        category = "cpu_or_budget_metric"
    else:
        category = "missing"

    return {
        "category": category,
        "uses_cuda": uses_cuda,
        "placeholder_only": placeholder_only,
        "section_specific_metric_keys": section_specific_keys,
    }


def report_paths() -> list[Path]:
    """Return all committed verification reports, including legacy evidence docs."""

    chapter_reports = ROOT.glob("chapter*/exercises/part*/verification_report.json")
    evidence_reports = ROOT.glob("docs/evidence/*/verification_report.json")
    return sorted(set(chapter_reports) | set(evidence_reports))


def _missing_runtime_keys(gpu_result: Any) -> list[str]:
    if not isinstance(gpu_result, dict):
        return ["gpu_test"]
    required = ("torch_version", "cuda_version", "gpu_total_memory_gb")
    return [key for key in required if gpu_result.get(key) in (None, "")]


def report_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in report_paths():
        report = json.loads(path.read_text())
        metrics = report.get("metrics", {})
        gpu_result = metrics.get("gpu_test")
        recomputed_evidence = classify_gpu_result(gpu_result)
        declared_evidence = metrics.get("gpu_evidence")
        rows.append(
            {
                "notebook_id": report.get("notebook_id", path.parent.name),
                "gt_tier": report.get("gt_tier", "unknown"),
                "accepted": report.get("accepted") is True,
                "tests_passed": report.get("tests_passed") is True,
                "known_failures": report.get("known_failures") or [],
                "category": recomputed_evidence["category"],
                "uses_cuda": recomputed_evidence["uses_cuda"],
                "keys": recomputed_evidence["section_specific_metric_keys"],
                "declared_category": (
                    declared_evidence.get("category")
                    if isinstance(declared_evidence, dict)
                    else None
                ),
                "declared_uses_cuda": (
                    declared_evidence.get("uses_cuda")
                    if isinstance(declared_evidence, dict)
                    else None
                ),
                "peak_vram_gb": (
                    gpu_result.get("peak_vram_gb") if isinstance(gpu_result, dict) else None
                ),
                "missing_runtime_keys": _missing_runtime_keys(gpu_result),
                "path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def strict_report_blockers(rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for row in rows:
        prefix = f"{row['notebook_id']} ({row['path']})"
        if not row["accepted"]:
            blockers.append(f"{prefix}: accepted is not true")
        if not row["tests_passed"]:
            blockers.append(f"{prefix}: tests_passed is not true")
        if row["known_failures"]:
            blockers.append(f"{prefix}: known_failures is nonempty")
        if row["category"] != "cuda_section_metric":
            blockers.append(f"{prefix}: recomputed gpu_evidence.category={row['category']!r}")
        if row["uses_cuda"] is not True:
            blockers.append(f"{prefix}: recomputed gpu_evidence.uses_cuda is not true")
        if row["declared_category"] is not None and row["declared_category"] != row["category"]:
            blockers.append(
                f"{prefix}: declared gpu_evidence.category={row['declared_category']!r} "
                f"but recomputed {row['category']!r}"
            )
        if row["declared_uses_cuda"] is not None and row["declared_uses_cuda"] != row["uses_cuda"]:
            blockers.append(
                f"{prefix}: declared gpu_evidence.uses_cuda={row['declared_uses_cuda']!r} "
                f"but recomputed {row['uses_cuda']!r}"
            )
        if not isinstance(row["peak_vram_gb"], float | int):
            blockers.append(f"{prefix}: gpu_test.peak_vram_gb is missing")
    return blockers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda-section-metrics",
        action="store_true",
        help="Fail unless every report has a section-specific CUDA metric.",
    )
    parser.add_argument(
        "--require-runtime-metadata",
        action="store_true",
        help="Fail unless every gpu_test records torch_version, cuda_version, and gpu_total_memory_gb.",
    )
    args = parser.parse_args()

    rows = report_rows()
    counts = Counter(row["category"] for row in rows)
    print(f"reports={len(rows)}")
    for category, count in sorted(counts.items()):
        print(f"{category}={count}")

    for row in rows:
        print(
            f"{row['notebook_id']} | {row['gt_tier']} | {row['category']} | "
            f"uses_cuda={row['uses_cuda']} | path={row['path']} | "
            f"keys={','.join(row['keys'])}"
        )

    if args.require_cuda_section_metrics:
        blockers = strict_report_blockers(rows)
        if blockers:
            print("\nReports lacking strict section-specific CUDA evidence:")
            for blocker in blockers:
                print(f"- {blocker}")
            raise SystemExit(1)

    if args.require_runtime_metadata:
        incomplete = [row for row in rows if row["missing_runtime_keys"]]
        if incomplete:
            print("\nReports lacking runtime CUDA metadata:")
            for row in incomplete:
                print(
                    f"- {row['notebook_id']} ({row['path']}): "
                    f"{', '.join(row['missing_runtime_keys'])}"
                )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
