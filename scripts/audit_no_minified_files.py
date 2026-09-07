"""Reject unreadable generated extension sources.

ARENA is teaching material, so reviewability is part of correctness. This audit
is deliberately scoped to extension-owned release payloads; original upstream
ARENA files can keep their historical formatting.
"""

from __future__ import annotations

import sys
import hashlib
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


def _frozen_protocol_has_readable_mirror(path: Path) -> bool:
    """Preserve experiment freeze bytes only when a text-identical readable copy exists."""
    if path.name != "protocol.md":
        return False
    freeze = path.with_name("protocol_freeze.json")
    mirror = path.with_name("protocol_readable.md")
    try:
        checksum = json.loads(freeze.read_text())["sha256"]["protocol.md"]
        return (hashlib.sha256(path.read_bytes()).hexdigest() == checksum
                and path.read_text().split() == mirror.read_text().split()
                and all(len(line) <= MD_MAX_LINE_LENGTH or _is_allowed_markdown_long_line(line)
                        for line in mirror.read_text().splitlines()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


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
        if _frozen_protocol_has_readable_mirror(path):
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

    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as exc:
        blockers.append(f"{_display_path(path)}: invalid notebook JSON ({exc.msg})")
        return blockers

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        blockers.append(f"{_display_path(path)}: notebook has no readable cells")
        return blockers

    # Notebook images are encoded on one line by nbformat. They are results the
    # learner needs to see, not minified source. Limit the exemption to binary
    # MIME values actually present in an output or attachment; source and HTML
    # still receive the ordinary serialized-line check.
    binary_mimes = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"}
    rendered_values: dict[str, set[str]] = {}
    for cell in cells:
        bundles = [output.get("data", {}) for output in cell.get("outputs", [])]
        bundles.extend(cell.get("attachments", {}).values())
        for bundle in bundles:
            for mime in binary_mimes:
                value = bundle.get(mime)
                if isinstance(value, str):
                    rendered_values.setdefault(mime, set()).add(value)

    for line_number, line in enumerate(lines, 1):
        if len(line) <= IPYNB_MAX_LINE_LENGTH:
            continue
        try:
            entry = json.loads("{" + line.strip().rstrip(",") + "}")
        except json.JSONDecodeError:
            entry = {}
        if len(entry) == 1:
            mime, value = next(iter(entry.items()))
            if isinstance(value, str) and value in rendered_values.get(mime, set()):
                continue
        blockers.append(
            f"{_display_path(path)}:{line_number}: notebook line length {len(line)} "
            f"exceeds {IPYNB_MAX_LINE_LENGTH}"
        )
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
