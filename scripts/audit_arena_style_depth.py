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
import json
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
STYLE_STATUS_PATH = ROOT / "docs/arena_style_rewrite_status.yml"
COURSE_READY_STATUSES = {"course_ready"}
REQUIRED_COURSE_READY_PAGE_PATTERNS = {
    "expected output dropdown": re.compile(
        r"<details>\s*<summary>\s*expected output",
        re.IGNORECASE,
    ),
    "solution dropdown": re.compile(
        r"<details>\s*<summary>\s*solution",
        re.IGNORECASE,
    ),
    "help or interpretation dropdown": re.compile(
        r"<details>\s*<summary>\s*(help|interpreting)",
        re.IGNORECASE,
    ),
    "signature result": re.compile(r"signature result", re.IGNORECASE),
    "limitations": re.compile(r"limitations", re.IGNORECASE),
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


def load_style_status(path: Path = STYLE_STATUS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"sections": {}}
    status = yaml.safe_load(path.read_text())
    if not isinstance(status, dict) or not isinstance(status.get("sections"), dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a sections mapping")
    return status


def style_status_blockers(config: dict[str, Any] | None = None) -> list[str]:
    blockers: list[str] = []
    status = load_style_status()
    sections = status["sections"]
    for number, page_path in extension_page_paths(config):
        entry = sections.get(number)
        if not isinstance(entry, dict):
            blockers.append(f"{number}: missing docs/arena_style_rewrite_status.yml entry")
            continue
        if entry.get("page") != page_path.relative_to(ROOT).as_posix():
            blockers.append(f"{number}: style-status page path does not match config")
        if entry.get("status") not in {"prototype", "course_ready"}:
            blockers.append(f"{number}: unknown style status {entry.get('status')!r}")
    return blockers


def _page_course_ready_blockers(number: str, page_path: Path) -> list[str]:
    text = page_path.read_text()
    blockers: list[str] = []
    for label, pattern in REQUIRED_COURSE_READY_PAGE_PATTERNS.items():
        if pattern.search(text) is None:
            blockers.append(f"{number}: course-ready page lacks {label}")
    if "<img" not in text and "|" not in text:
        blockers.append(f"{number}: course-ready page lacks visible figure or table")
    return blockers


def _notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if isinstance(cell, dict)
    )


def notebook_setup_contract_blockers() -> list[str]:
    """Catch notebook report cells that reference setup variables never defined."""

    blockers: list[str] = []
    for notebook_path in sorted(ROOT.glob("chapter*/exercises/part*/*.ipynb")):
        text = _notebook_text(notebook_path)
        if "section_dir" in text and "section_dir =" not in text:
            blockers.append(
                f"{notebook_path.relative_to(ROOT)} references section_dir without defining it"
            )
        if "section_dir = exercises_dir / section" in text and "section =" not in text:
            blockers.append(
                f"{notebook_path.relative_to(ROOT)} defines section_dir from undefined section"
            )
    return blockers


def _notebook_course_ready_blockers(number: str, page_path: Path) -> list[str]:
    blockers: list[str] = []
    section_dir = page_path.parents[2] / "exercises"
    exercise_dirs = [
        path
        for path in section_dir.glob("part*")
        if any(path.glob("*_exercises.ipynb")) and any(path.glob("*_solutions.ipynb"))
    ]
    matching_dirs = [
        path for path in exercise_dirs if page_path.stem.split("]_", maxsplit=1)[-1] in " ".join(p.name for p in path.glob("*.ipynb"))
    ]
    if not matching_dirs:
        # Fall back to artifact locks, which carry the section number exactly.
        matching_dirs = [
            path
            for path in exercise_dirs
            if (path / "artifacts.lock.yml").exists()
            and str(yaml.safe_load((path / "artifacts.lock.yml").read_text()).get("section")) == number
        ]
    if not matching_dirs:
        blockers.append(f"{number}: could not locate paired notebooks for course-ready section")
        return blockers
    for notebook_dir in matching_dirs[:1]:
        for notebook_path in sorted(notebook_dir.glob("*.ipynb")):
            text = _notebook_text(notebook_path).lower()
            required = {
                "expected output": "expected-output content",
                "<details>": "dropdown content",
                "help -": "help dropdown",
                "signature result": "signature result",
                "limitations": "limitations",
            }
            missing = [label for term, label in required.items() if term not in text]
            if missing:
                blockers.append(
                    f"{number}: {notebook_path.relative_to(ROOT)} lacks {missing}"
                )
    return blockers


def course_ready_depth_blockers(config: dict[str, Any] | None = None) -> list[str]:
    status = load_style_status()
    sections = status["sections"]
    blockers: list[str] = []
    for number, page_path in extension_page_paths(config):
        entry = sections.get(number, {})
        if entry.get("status") not in COURSE_READY_STATUSES:
            continue
        if not page_path.exists():
            blockers.append(f"{number}: missing course-ready page {page_path.relative_to(ROOT)}")
            continue
        blockers.extend(_page_course_ready_blockers(number, page_path))
        blockers.extend(_notebook_course_ready_blockers(number, page_path))
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
        + style_status_blockers(config)
        + course_ready_depth_blockers(config)
        + page_metadata_consistency_blockers(config)
        + hard_registry_depth_blockers()
        + bare_assertion_blockers()
        + notebook_setup_contract_blockers()
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
