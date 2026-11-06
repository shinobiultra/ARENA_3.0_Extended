#!/usr/bin/env python
"""Run one lesson's focused tests: one CPU command, no model downloads.

    uv run python infrastructure/extension/focused_tests.py 7.1
    uv run python infrastructure/extension/focused_tests.py 7.1 12.3 --list

For each section it runs, in a single pytest call:

1. the structural checks in `tests/test_extension_sections.py` for that section:
   the committed notebooks and `solutions.py` match the master, the solutions
   notebook is one complete fresh-kernel run without errors, the exercises
   notebook has the ARENA shape, the folder is clean, the expected figures exist,
   and every local file the lesson links to or shows exists;
2. the lesson's own pytest files: `tests/test_lesson_<N>_<M>_*.py`, plus any other
   `tests/test_*.py` that names the lesson's exercise folder.

The notebook's own `tests.test_x(x)` calls run inside the fresh-kernel run
(`build.py <section> --execute`), so they are not repeated here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
STRUCTURAL = TESTS / "test_extension_sections.py"
# Course-wide tooling tests that mention exercise folders only as fixtures.
NOT_LESSON_TESTS = {"test_extension_sections.py", "test_extension_page_links.py", "test_extension_page_media.py",
                    "test_original_arena_preservation_audit.py", "test_course_surface_audit.py"}


def exercise_dir(section: str) -> str:
    config = yaml.safe_load((ROOT / "infrastructure/core/config.yaml").read_text(encoding="utf-8"))
    try:
        return config["conversion_map"][section]["exercise_dir"]
    except KeyError:
        raise SystemExit(f"Unknown section {section!r}") from None


def lesson_test_files(section: str) -> list[Path]:
    prefix = "test_lesson_" + section.replace(".", "_") + "_"
    folder = exercise_dir(section)
    files = set(TESTS.glob(prefix + "*.py"))
    for path in TESTS.glob("test_*.py"):
        if path.name in NOT_LESSON_TESTS or path.name.startswith("test_lesson_"):
            continue
        if folder in path.read_text(encoding="utf-8", errors="ignore"):
            files.add(path)
    return sorted(files)


def structural_node_ids(sections: list[str]) -> list[str]:
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(STRUCTURAL)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    wanted = {f"[{section}]" for section in sections}
    return [line.strip() for line in collected if any(line.strip().endswith(tag) for tag in wanted)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sections", nargs="+", help="Section numbers, e.g. 7.1")
    parser.add_argument("--list", action="store_true", help="Only print what would run")
    args = parser.parse_args()

    targets = structural_node_ids(args.sections)
    for section in args.sections:
        targets += [str(path.relative_to(ROOT)) for path in lesson_test_files(section)]
    if args.list:
        print("\n".join(targets))
        return
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets]
    offline = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=offline).returncode)


if __name__ == "__main__":
    main()
