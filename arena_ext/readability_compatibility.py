"""Validate only the explicitly recorded 2026-09-08 cosmetic publications.

Historical execution hashes remain the authority for the measured source.
This narrow bridge does not declare a new execution or accept future edits.
The record digest is pinned here; its hashes do not include this helper or any
report, avoiding a circular source/receipt binding.
"""
from __future__ import annotations

import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "docs/arena_salvage/2026-09-08_readability_compatibility.json"
RECORD_SHA256 = "f6d6fc026d139f46697ee3626ac96869ab49401f600fe5d3c9ddc5e36c941708"

# These are reviewed validation changes, not AST-identical formatting. Reversing
# only these exact snippets must restore the complete original AST, constants
# included. The record cannot supply or expand this list of permitted changes.
IMPORT_BEFORE = "from __future__ import annotations\n"
IMPORT_AFTER = (
    "from __future__ import annotations\n\n"
    "from arena_ext.readability_compatibility import source_matches_execution\n"
)
GUARD_REPLACEMENTS = {
    "chapter5_modern_architectures/exercises/part5_diffusion_language_models/study_evidence.py": (
        (IMPORT_BEFORE, IMPORT_AFTER),
        (
            "        if sha256(HERE/name)!=digest or sha256(run/'source_snapshot'/name)!=digest:",
            "        if not source_matches_execution(HERE/name, digest, run/'source_snapshot'/name):",
        ),
    ),
    "chapter8_automated_circuits/exercises/part3_acdc_circuit_metrics/edge_evidence.py": (
        (IMPORT_BEFORE, IMPORT_AFTER),
        (
            "        if sha256(HERE / name) != expected:",
            "        if not source_matches_execution(HERE / name, expected):",
        ),
    ),
    "chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/geometry_evidence.py": (
        (IMPORT_BEFORE, IMPORT_AFTER),
        (
            "        if name=='geometry_plotting.py' and sha256(HERE/name)!=digest:",
            "        if name=='geometry_plotting.py' and not source_matches_execution(HERE/name, digest):",
        ),
        (
            "        elif name!='lesson.md' and sha256(HERE/name)!=digest:raise ValueError('Executed dependency changed: '+name)",
            "        elif name!='lesson.md' and not source_matches_execution(HERE/name, digest):raise ValueError('Executed dependency changed: '+name)",
        ),
    ),
    "chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/geometry_reproduction_cache.py": (
        (IMPORT_BEFORE, IMPORT_AFTER),
        (
            '        if sha256(source / "source_snapshot" / name) != digest or sha256(here / name) != digest:',
            '        if not source_matches_execution(here / name, digest, source / "source_snapshot" / name):',
        ),
    ),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _RenderedMarkdown(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.events = []
        self.code_depth = 0

    def handle_starttag(self, tag, attrs):
        self.events.append(("start", tag, attrs))
        if tag in {"pre", "code"}:
            self.code_depth += 1

    def handle_endtag(self, tag):
        self.events.append(("end", tag))
        if tag in {"pre", "code"}:
            self.code_depth -= 1

    def handle_startendtag(self, tag, attrs):
        self.events.append(("empty", tag, attrs))

    def handle_data(self, data):
        self.events.append(("data", data if self.code_depth else re.sub(r"\s+", " ", data)))

    def handle_entityref(self, name):
        self.events.append(("entity", name))

    def handle_charref(self, name):
        self.events.append(("charref", name))

    def handle_comment(self, data):
        self.events.append(("comment", data))

    def handle_decl(self, decl):
        self.events.append(("declaration", decl))


def _markdown_semantics(source: str):
    from markdown_it import MarkdownIt

    parser = _RenderedMarkdown()
    parser.feed(MarkdownIt("commonmark").render(source))
    parser.close()
    return parser.events


def _equivalent_source(name: str, before: str, after: str, mode: str) -> bool:
    if mode == "commonmark_whitespace_only":
        return name.endswith(".md") and _markdown_semantics(before) == _markdown_semantics(after)
    if mode == "python_ast_with_reviewed_validation_guard":
        patches = GUARD_REPLACEMENTS.get(name)
        if not patches:
            return False
        for old, new in reversed(patches):
            if before.count(old) != 1 or after.count(new) != 1:
                return False
            after = after.replace(new, old, 1)
    elif mode != "python_ast":
        return False
    return name.endswith(".py") and ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))


def source_matches_execution(path: Path, expected_sha256: str, snapshot: Path | None = None) -> bool:
    """Accept exact execution bytes or one pinned, independently checked publication."""
    try:
        actual = path.read_bytes()
        historical = snapshot.read_bytes() if snapshot is not None else None
        if historical is not None and _sha(historical) != expected_sha256:
            return False
        if _sha(actual) == expected_sha256:
            return True
        name = path.resolve().relative_to(ROOT.resolve()).as_posix()
        record_bytes = RECORD_PATH.read_bytes()
        if _sha(record_bytes) != RECORD_SHA256:
            return False
        record = json.loads(record_bytes)
        if record.get("schema_version") != 1 or record.get("native_experiments_rerun") is not False:
            return False
        entries = record["source_files"]
        matching = [entry for entry in entries if entry["path"] == name]
        if len(matching) != 1:
            return False
        entry = matching[0]
        before = entry["before_source"].encode()
        if entry["before_sha256"] != expected_sha256 or _sha(before) != expected_sha256:
            return False
        if historical is not None and historical != before:
            return False
        if entry["after_sha256"] != _sha(actual):
            return False
        return _equivalent_source(name, before.decode(), actual.decode(), entry["equivalence"])
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, ImportError, UnicodeError):
        return False
