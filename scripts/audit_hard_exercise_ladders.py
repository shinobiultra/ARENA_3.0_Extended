"""Audit hard-extension exercises against the roadmap's verification ladder.

This is a structural gate for learner-facing ARENA-style feedback. It does not
pretend to prove research claims; it checks that each difficulty-3+ section has
non-placeholder visible tests, fixture provenance, and a CUDA-backed report
before the section is treated as released.
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs/hard_exercise_ladder_registry.csv"

REQUIRED_FIXTURE_TERMS = (
    "How this fixture was produced",
    "Trusted implementation",
    "Random seed",
    "Allowed tolerances",
    "When to regenerate",
    "rtol=",
    "atol=",
)
FORBIDDEN_VISIBLE_TEST_TERMS = (
    "looks_reasonable",
    "NotImplementedError",
)


@dataclass(frozen=True)
class VisibleTestSummary:
    test_function_count: int
    assert_count: int
    success_print_count: int
    placeholder_tests: tuple[str, ...]


def registry_rows(path: Path = REGISTRY_PATH) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _path_from_source(source: str) -> Path:
    return ROOT / source.split(":", maxsplit=1)[0]


def visible_test_summary(path: Path) -> VisibleTestSummary:
    tree = ast.parse(path.read_text(), filename=str(path))
    test_functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert_count = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    success_print_count = 0
    placeholder_tests: list[str] = []

    for function in test_functions:
        meaningful_body = [
            node
            for node in function.body
            if not isinstance(node, (ast.Expr, ast.Pass))
            or not isinstance(getattr(node, "value", None), ast.Constant)
        ]
        if len(meaningful_body) == 0 or all(isinstance(node, ast.Pass) for node in function.body):
            placeholder_tests.append(function.name)

        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "print":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            if "All tests in `" in str(node.args[0].value):
                success_print_count += 1

    return VisibleTestSummary(
        test_function_count=len(test_functions),
        assert_count=assert_count,
        success_print_count=success_print_count,
        placeholder_tests=tuple(placeholder_tests),
    )


def fixture_readme_missing_terms(path: Path) -> list[str]:
    text = path.read_text()
    return [term for term in REQUIRED_FIXTURE_TERMS if term not in text]


def report_has_cuda_section_metric(path: Path) -> bool:
    report = json.loads(path.read_text())
    evidence = report.get("metrics", {}).get("gpu_evidence", {})
    return (
        report.get("accepted") is True
        and report.get("tests_passed") is True
        and evidence.get("category") == "cuda_section_metric"
        and evidence.get("uses_cuda") is True
    )


def row_blockers(row: dict[str, str]) -> list[str]:
    notebook_id = row["notebook_id"]
    difficulty = int(row["difficulty"])
    blockers: list[str] = []

    if row["release_status"] != "released_with_cuda_section_metric":
        blockers.append(f"{notebook_id}: release_status={row['release_status']}")

    fixture_path = _path_from_source(row["fixture_provenance_source"])
    tests_path = _path_from_source(row["visible_tests_source"])
    report_path = _path_from_source(row["report_source"])

    for source_name, path in (
        ("fixture_provenance_source", fixture_path),
        ("visible_tests_source", tests_path),
        ("report_source", report_path),
    ):
        if not path.exists():
            blockers.append(f"{notebook_id}: {source_name} missing at {path.relative_to(ROOT)}")
            return blockers

    missing_terms = fixture_readme_missing_terms(fixture_path)
    if missing_terms:
        blockers.append(f"{notebook_id}: fixture README missing {missing_terms}")

    tests_text = tests_path.read_text()
    forbidden_terms = [term for term in FORBIDDEN_VISIBLE_TEST_TERMS if term in tests_text]
    if forbidden_terms:
        blockers.append(f"{notebook_id}: visible tests contain forbidden terms {forbidden_terms}")

    summary = visible_test_summary(tests_path)
    if summary.test_function_count < 3:
        blockers.append(
            f"{notebook_id}: only {summary.test_function_count} visible tests; expected >= 3"
        )
    if summary.assert_count < difficulty + 3:
        blockers.append(
            f"{notebook_id}: only {summary.assert_count} assertions; expected >= {difficulty + 3}"
        )
    if summary.success_print_count < summary.test_function_count:
        blockers.append(
            f"{notebook_id}: {summary.success_print_count}/{summary.test_function_count} tests "
            "print ARENA-style success messages"
        )
    if summary.placeholder_tests:
        blockers.append(f"{notebook_id}: placeholder tests {list(summary.placeholder_tests)}")

    if not report_has_cuda_section_metric(report_path):
        blockers.append(f"{notebook_id}: report lacks accepted CUDA section metric")

    return blockers


def ladder_blockers(rows: list[dict[str, str]] | None = None) -> list[str]:
    blockers: list[str] = []
    for row in rows or registry_rows():
        blockers.extend(row_blockers(row))
    return blockers


def main() -> None:
    rows = registry_rows()
    blockers = ladder_blockers(rows)
    print(f"hard_exercises_checked={len(rows)}")
    print(f"ladder_blockers={len(blockers)}")
    if blockers:
        print("HARD_EXERCISE_LADDERS=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("HARD_EXERCISE_LADDERS=PASS")


if __name__ == "__main__":
    main()
