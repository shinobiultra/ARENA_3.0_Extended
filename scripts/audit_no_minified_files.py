"""Reject unreadable generated extension sources.

ARENA is teaching material, so reviewability is part of correctness. This audit
is deliberately scoped to extension-owned release payloads; original upstream
ARENA files can keep their historical formatting.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_extension_artifact_hygiene import iter_extension_files


PY_MAX_LINE_LENGTH = 240
MD_MAX_LINE_LENGTH = 400
IPYNB_MAX_LINE_LENGTH = 5000


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _is_allowed_markdown_long_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("|")
        or stripped.startswith("<img")
        or stripped.startswith("<iframe")
        or "http://" in stripped
        or "https://" in stripped
    )


def minified_file_blockers(files: list[Path] | None = None) -> list[str]:
    """Return blockers for extension-owned source files with unreadable lines."""

    blockers: list[str] = []
    source_files = files if files is not None else iter_extension_files()
    for path in source_files:
        if path.suffix not in {".py", ".md", ".ipynb"}:
            continue
        if path.suffix == ".ipynb":
            blockers.extend(_notebook_blockers(path))
            continue
        limit = PY_MAX_LINE_LENGTH if path.suffix == ".py" else MD_MAX_LINE_LENGTH
        for line_number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if len(line) <= limit:
                continue
            if path.suffix == ".md" and _is_allowed_markdown_long_line(line):
                continue
            blockers.append(
                f"{_display_path(path)}:{line_number}: line length {len(line)} "
                f"exceeds {limit}"
            )
    return blockers


def _notebook_blockers(path: Path) -> list[str]:
    blockers: list[str] = []
    text = path.read_text(errors="ignore")
    lines = text.splitlines()
    if len(lines) <= 3 and any(len(line) > IPYNB_MAX_LINE_LENGTH for line in lines):
        blockers.append(f"{_display_path(path)}: notebook JSON is collapsed into too few lines")
        return blockers

    for line_number, line in enumerate(lines, 1):
        if len(line) > IPYNB_MAX_LINE_LENGTH:
            blockers.append(
                f"{_display_path(path)}:{line_number}: notebook line length {len(line)} "
                f"exceeds {IPYNB_MAX_LINE_LENGTH}"
            )

    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as exc:
        blockers.append(f"{_display_path(path)}: invalid notebook JSON ({exc.msg})")
        return blockers

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        blockers.append(f"{_display_path(path)}: notebook has no readable cells")
    return blockers


def main() -> None:
    blockers = minified_file_blockers()
    print(f"minified_file_blockers={len(blockers)}")
    if blockers:
        print("NO_MINIFIED_FILES=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("NO_MINIFIED_FILES=PASS")


if __name__ == "__main__":
    main()
