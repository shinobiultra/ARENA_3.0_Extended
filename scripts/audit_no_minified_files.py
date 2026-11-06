"""Reject unreadable (minified) lines in learner-facing extension chapter files.

ARENA is teaching material, so readability is part of correctness. The audit
covers the extension chapters' pages, notebooks and Python files (chapters
5-18 and the 0.6 and 1.6 lessons); tooling, tests and docs are out of scope,
and original upstream ARENA files keep their own formatting.
"""

from __future__ import annotations

import sys
import base64
import binascii
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_extension_artifact_hygiene import iter_extension_files


PY_MAX_LINE_LENGTH = 240
MD_MAX_LINE_LENGTH = 400
IPYNB_MAX_LINE_LENGTH = 5000


def _is_image_only_html(value: str) -> bool:
    """Allow inert images only, after checking each data URL and actual bytes."""
    from PIL import Image

    class ImageHTML(HTMLParser):
        valid = True
        sources: list[str]

        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.sources = []

        def handle_starttag(self, tag, attrs):
            names = [name for name, _ in attrs]
            if (tag != "img" or len(names) != len(set(names))
                    or set(names) - {"src", "width", "height", "alt", "title"}):
                self.valid = False
                return
            attributes = dict(attrs)
            for name, item in attrs:
                if (item is None or (name in {"width", "height"}
                                     and re.fullmatch(r"[1-9][0-9]{0,4}", item) is None)
                        or (name in {"alt", "title"} and any(
                            ord(char) < 32 or char in "<>" for char in item))):
                    self.valid = False
            self.sources.append(attributes.get("src", ""))

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)

        def handle_data(self, data):
            if data.strip():
                self.valid = False

        def handle_endtag(self, tag):
            self.valid = False

        def handle_comment(self, data):
            self.valid = False

        def handle_decl(self, decl):
            self.valid = False

        def handle_pi(self, data):
            self.valid = False

        def unknown_decl(self, data):
            self.valid = False

    parser = ImageHTML()
    try:
        parser.feed(value)
        parser.close()
        # HTMLParser recovers malformed markup; also require complete lexical tags.
        if (not parser.valid or not parser.sources
                or re.fullmatch(r"(?:\s*<img\s+[^<>]+/?>)+\s*", value) is None):
            return False
        for source in parser.sources:
            match = re.fullmatch(r"data:image/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/]+={0,2})", source)
            if match is None:
                return False
            mime, encoded = match.groups()
            data = base64.b64decode(encoded, validate=True)
            signatures = {
                "png": data.startswith(b"\x89PNG\r\n\x1a\n") and data.endswith(b"IEND\xaeB`\x82"),
                "jpeg": data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"),
                "gif": data.startswith((b"GIF87a", b"GIF89a")) and data.endswith(b";"),
                "webp": (data.startswith(b"RIFF") and data[8:12] == b"WEBP"
                         and int.from_bytes(data[4:8], "little") + 8 == len(data)),
            }
            if not signatures[mime]:
                return False
            with Image.open(io.BytesIO(data)) as picture:
                if (picture.format != {"png": "PNG", "jpeg": "JPEG", "gif": "GIF", "webp": "WEBP"}[mime]
                        or picture.width * picture.height > 80 * 1024 * 1024):
                    return False
                picture.verify()
            # verify() checks the container; loading also rejects invalid compressed
            # pixel data masquerading as an otherwise well-formed image container.
            with Image.open(io.BytesIO(data)) as picture:
                picture.load()
        return True
    except (ValueError, binascii.Error, OSError, SyntaxError, AssertionError, Image.DecompressionBombError):
        return False


def _json_strings(text: str):
    """Yield string value spans with their JSON paths (the JSON is already valid)."""
    decoder = json.JSONDecoder()

    def whitespace(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    def visit(index, path):
        index = whitespace(index)
        if text[index] == "{":
            index = whitespace(index + 1)
            while text[index] != "}":
                key, end = decoder.raw_decode(text, index)
                index = yield from visit(whitespace(end) + 1, (*path, key))
                index = whitespace(index)
                if text[index] == ",":
                    index = whitespace(index + 1)
            return index + 1
        if text[index] == "[":
            index, item = whitespace(index + 1), 0
            while text[index] != "]":
                index = yield from visit(index, (*path, item))
                item += 1
                index = whitespace(index)
                if text[index] == ",":
                    index = whitespace(index + 1)
            return index + 1
        value, end = decoder.raw_decode(text, index)
        if isinstance(value, str):
            yield path, value, index, end
        return end

    yield from visit(0, ())


def _html_image_output_lines(text: str, cells: list) -> set[int]:
    """Only exempt serialized strings belonging to valid complete HTML outputs."""
    allowed: dict[tuple, str] = {}
    for cell_index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        for output_index, output in enumerate(cell.get("outputs", [])):
            if output.get("output_type") not in {"display_data", "execute_result"}:
                continue
            value = output.get("data", {}).get("text/html")
            parts = value if isinstance(value, list) else [value]
            if not all(isinstance(part, str) for part in parts):
                continue
            if not _is_image_only_html("".join(parts)):
                continue
            prefix = ("cells", cell_index, "outputs", output_index, "data", "text/html")
            allowed.update({(*prefix, i): part for i, part in enumerate(parts)}
                           if isinstance(value, list) else {prefix: value})
    exempt: set[int] = set()
    # Require the rest of the physical line to be only the corresponding JSON
    # key/list punctuation. A collapsed source or another value is never exempt.
    for path, value, start, end in _json_strings(text):
        if value != allowed.get(path):
            continue
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)
        before, after = text[line_start:start].strip(), text[end:line_end].strip()
        if before in {"", '"text/html":', '"text/html": ['} and after in {"", ",", "]", "],"}:
            exempt.add(text.count("\n", 0, start) + 1)
    return exempt


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


def learner_chapter_files() -> list[Path]:
    """Extension files that live in chapter folders (pages, notebooks, lesson code)."""
    return [path for path in iter_extension_files()
            if _display_path(path).startswith("chapter")]


def minified_file_blockers(files: list[Path] | None = None) -> list[str]:
    """Return blockers for extension-owned source files with unreadable lines."""

    blockers: list[str] = []
    source_files = files if files is not None else learner_chapter_files()
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
    # still receive the ordinary serialized-line check, except for independently
    # validated image-only HTML at an actual code-cell output location.
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

    html_image_lines = _html_image_output_lines(text, cells)
    for line_number, line in enumerate(lines, 1):
        if len(line) <= IPYNB_MAX_LINE_LENGTH:
            continue
        if line_number in html_image_lines:
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
