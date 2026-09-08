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
import re
import sys
import yaml
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

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


def _success_feedback(call: ast.AST) -> bool:
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "print" and call.args):
        return False
    message = call.args[0]
    if isinstance(message, ast.Constant):
        return isinstance(message.value, str) and bool(message.value.strip())
    if isinstance(message, ast.JoinedStr):
        # Keep the explanation visible in source. Interpolating a raw result or
        # report is not feedback, but named success plus dynamic case details is.
        literal_text = " ".join(
            part.value for part in message.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        return bool(re.search(
            r"\b(?:passed|succeeded|matches|matched|correct|agrees?|verified|rejected)\b",
            literal_text, flags=re.IGNORECASE,
        ))
    return False


def visible_test_summary(path: Path) -> VisibleTestSummary:
    tree = ast.parse(path.read_text(), filename=str(path))
    test_functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert_count = sum(
        isinstance(node, ast.Assert) or (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"assert_close", "assert_allclose", "assert_array_equal"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "testing"
        ) for node in ast.walk(tree)
    )
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

        # Feedback need not repeat one exact sentence. Require a nonempty
        # final print after an actual assertion / numerical oracle, so a
        # progress print or a print-only fake test cannot satisfy this check.
        checks = any(isinstance(node, (ast.Assert, ast.Raise)) or (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"assert_close", "assert_allclose", "raises", "parse"}
        ) or (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id.startswith(("test_", "check_", "assert_", "_assert_"))
        ) for node in ast.walk(function))
        feedback = [function.body[-1]] if function.body else []
        if len(function.body) > 1 and isinstance(function.body[-1], ast.Return):
            feedback.append(function.body[-2])
        # Expected-exception tests may print and return inside the handler,
        # followed by a failing assertion if the expected error did not occur.
        feedback.extend(statement for node in ast.walk(function)
                        if isinstance(node, ast.ExceptHandler)
                        for statement in node.body)
        calls = [node.value for node in feedback if isinstance(node, ast.Expr)]
        if checks and any(_success_feedback(call) for call in calls):
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

    cpu_release = row.get("requires_gpu", "").lower() == "false" and row["release_status"] == "released_with_cpu_model_metric"
    if row["release_status"] != "released_with_cuda_section_metric" and not cpu_release:
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

    cpu_accepted = False
    if cpu_release:
        lock_path = _path_from_source(row["metadata_source"])
        report = json.loads(report_path.read_text())
        cpu_accepted = (lock_path.is_file() and report.get("accepted") is True and report.get("tests_passed") is True
                        and has_native_cpu_model_evidence(yaml.safe_load(lock_path.read_text()), report))
    if not report_has_cuda_section_metric(report_path) and not cpu_accepted:
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
