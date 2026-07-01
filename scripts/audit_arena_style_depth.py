"""Audit roadmap-level ARENA style depth for extension instruction pages.

The normal course-surface audit checks that pages exist and have the broad
shape of a lesson. This stricter audit tracks the roadmap appendix requirements:
visible expected outputs, solution dropdowns, common-bug guidance, exercise
metadata near learner tasks, and no "released" hard-exercise rows with remaining
style/ladder evidence still marked as TODO.
"""

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_course_surface import is_extension_section, load_config


REQUIRED_PAGE_TERMS = {
    "expected output": "expected-output block",
    "solution": "solution dropdown or solution pointer",
    "common bug": "common-bug note",
    "difficulty": "exercise difficulty metadata",
    "importance": "exercise importance metadata",
}
REQUIRED_METADATA_TERMS = {
    "gt_tier": "GT_TIER learner-facing metadata",
    "exercise_id": "EXERCISE_ID learner-facing metadata",
    "expected_runtime": "EXPECTED_RUNTIME learner-facing metadata",
    "requires_gpu": "REQUIRES_GPU learner-facing metadata",
}
UNRESOLVED_LADDER_EVIDENCE = {
    "ongoing_visible_notebook_cell_spot_audit",
    "debug_cache_for_complex_functions_where_needed",
    "periodic_real_model_revision_refresh",
}
PAGE_METADATA_PATTERNS = {
    "GT_TIER": re.compile(r"GT_TIER\s*=\s*[\"']([^\"']+)[\"']"),
    "EXERCISE_ID": re.compile(r"EXERCISE_ID\s*=\s*[\"']([^\"']+)[\"']"),
}


def extension_page_paths(config: dict[str, Any] | None = None) -> list[tuple[str, Path]]:
    config = config or load_config()
    paths: list[tuple[str, Path]] = []
    for chapter_name, chapter in config["chapters"].items():
        for section in chapter.get("sections", []):
            number = str(section.get("number", ""))
            if not is_extension_section(number):
                continue
            page_path = ROOT / chapter_name / "instructions/pages" / section["page_file"]
            paths.append((number, page_path))
    return paths


def page_style_depth_blockers(config: dict[str, Any] | None = None) -> list[str]:
    blockers: list[str] = []
    for number, page_path in extension_page_paths(config):
        if not page_path.exists():
            blockers.append(f"{number}: missing page {page_path.relative_to(ROOT)}")
            continue
        text = page_path.read_text().lower()
        missing = [
            label
            for term, label in {**REQUIRED_PAGE_TERMS, **REQUIRED_METADATA_TERMS}.items()
            if term not in text
        ]
        if missing:
            blockers.append(
                f"{number}: page lacks roadmap style elements {missing} "
                f"({page_path.relative_to(ROOT)})"
            )
    return blockers


def _section_locks() -> dict[str, tuple[Path, dict[str, Any]]]:
    locks: dict[str, tuple[Path, dict[str, Any]]] = {}
    for lock_path in ROOT.glob("chapter*/exercises/part*/artifacts.lock.yml"):
        lock = yaml.safe_load(lock_path.read_text())
        section = str(lock.get("section", ""))
        if section:
            locks[section] = (lock_path, lock)
    return locks


def page_metadata_consistency_blockers(config: dict[str, Any] | None = None) -> list[str]:
    blockers: list[str] = []
    locks = _section_locks()
    for number, page_path in extension_page_paths(config):
        if not page_path.exists() or number not in locks:
            continue
        text = page_path.read_text()
        lock_path, lock = locks[number]
        expected_metadata = lock.get("exercise_metadata", {})
        for field, pattern in PAGE_METADATA_PATTERNS.items():
            match = pattern.search(text)
            if match is None:
                continue
            actual = match.group(1)
            expected = str(expected_metadata.get(field, ""))
            if actual != expected:
                blockers.append(
                    f"{number}: page {field}={actual!r} does not match "
                    f"{lock_path.relative_to(ROOT)} {field}={expected!r} "
                    f"({page_path.relative_to(ROOT)})"
                )
    return blockers


def hard_registry_depth_blockers(
    registry_path: Path = ROOT / "docs/hard_exercise_ladder_registry.csv",
) -> list[str]:
    blockers: list[str] = []
    with registry_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            remaining = set(row.get("remaining_release_evidence", "").split(";"))
            unresolved = sorted(remaining & UNRESOLVED_LADDER_EVIDENCE)
            if unresolved:
                blockers.append(
                    f"{row['notebook_id']}: released hard-exercise row still lists "
                    f"unresolved evidence {unresolved}"
                )
    return blockers


def _path_from_source(source: str) -> Path:
    return ROOT / source.split(":", maxsplit=1)[0]


def bare_assertion_blockers(
    registry_path: Path = ROOT / "docs/hard_exercise_ladder_registry.csv",
) -> list[str]:
    blockers: list[str] = []
    seen: set[Path] = set()
    with registry_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            tests_path = _path_from_source(row["visible_tests_source"])
            if tests_path in seen or not tests_path.exists():
                continue
            seen.add(tests_path)
            tree = ast.parse(tests_path.read_text(), filename=str(tests_path))
            bare_asserts = [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Assert) and node.msg is None
            ]
            if bare_asserts:
                preview = ", ".join(str(line) for line in bare_asserts[:5])
                suffix = "" if len(bare_asserts) <= 5 else f", +{len(bare_asserts) - 5} more"
                blockers.append(
                    f"{tests_path.relative_to(ROOT)}: bare assertions without educational "
                    f"failure messages at lines {preview}{suffix}"
                )
    return blockers


def style_depth_blockers(config: dict[str, Any] | None = None) -> list[str]:
    return (
        page_style_depth_blockers(config)
        + page_metadata_consistency_blockers(config)
        + hard_registry_depth_blockers()
        + bare_assertion_blockers()
    )


def main() -> None:
    page_count = len(extension_page_paths())
    blockers = style_depth_blockers()
    print(f"extension_pages_checked={page_count}")
    print(f"style_depth_blockers={len(blockers)}")
    if blockers:
        print("ARENA_STYLE_DEPTH=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("ARENA_STYLE_DEPTH=PASS")


if __name__ == "__main__":
    main()
