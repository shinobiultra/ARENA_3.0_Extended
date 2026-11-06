"""Finite pinned prose and reviewed blank-Practice setup editions; never execute.

Canonical source bytes, reference code/results, and existing Blank Practice
providers remain independent parents of the invertible derivative.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PurePosixPath

from infrastructure.extension import blank_practice

_KINDS = ("page", "exercises", "reference")
_KEY = "arena_teaching_edition"
_DELIMITER = "=== NEW CHAPTER ==="
_RUNTIME = {"_root", "_manifest_sha256", "_document_sha256", "_configuration_sha256"}
_FOLD_HELPERS = {
    "tree_curriculum_namespace", "deep_curriculum_namespace", "gradient_curriculum_namespace",
    "refresh_tree_default_methods", "deep_module_deepshap_methods_namespace",
    "gradient_module_gradientshap_methods_namespace",
}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _typed(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed UTF-8 teaching-edition JSON") from exc


def _relative(value: str) -> str:
    if (type(value) is not str or not value or "\\" in value or "\0" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or PurePosixPath(value).as_posix() != value or value == "."):
        raise ValueError("Expected exact repository-relative teaching-edition path")
    return value


def _read(root: Path, relative: str, *, optional: bool = False) -> bytes | None:
    """Retain nofollow directory descriptors, including every ancestor of root."""
    relative = _relative(relative)
    root = Path(root).absolute()
    if ".." in root.parts:
        raise ValueError("Teaching-edition root must not contain parent traversal")
    descriptors = []
    edges = []
    root_ready = root == Path("/")
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    directory_identity = lambda s: (s.st_dev, s.st_ino, s.st_mode)
    try:
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        descriptors.append(directory)
        for index, part in enumerate((*root.parts[1:], *PurePosixPath(relative).parts[:-1])):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            descriptors.append(child)
            edges.append((directory, part, child))
            directory = child
            if index == len(root.parts) - 2:
                root_ready = True
        name = PurePosixPath(relative).name
        before = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Teaching-edition input is a symlink or nonregular file: {relative}")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        descriptors.append(fd)
        if identity(os.fstat(fd)) != identity(before):
            raise ValueError(f"Teaching-edition input changed before read: {relative}")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Teaching-edition input shrank during read: {relative}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1) or identity(os.fstat(fd)) != identity(before):
            raise ValueError(f"Teaching-edition input changed during read: {relative}")
        if identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) != identity(before):
            raise ValueError(f"Teaching-edition path changed during read: {relative}")
        for parent, part, child in edges:
            bound = os.stat(part, dir_fd=parent, follow_symlinks=False)
            if directory_identity(bound) != directory_identity(os.fstat(child)):
                raise ValueError(f"Teaching-edition directory changed during read: {relative}")
        return b"".join(chunks)
    except FileNotFoundError as exc:
        if optional and root_ready:
            return None
        raise ValueError(f"Missing teaching-edition input: {relative}") from exc
    except OSError as exc:
        raise ValueError(f"Teaching-edition symlink, unreadable root/path or changed identity: {relative}") from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _validate(document: dict) -> None:
    required = {"schema_version", "section", "sources", "replacements"}
    if type(document) is not dict or not required <= document.keys() or document.keys() - required - {"rationale", "drop_empty_markdown_cells", "practice_code_replacements", "presentation_collapses"}:
        raise ValueError("Teaching-edition manifest has missing or unknown keys")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported teaching-edition schema_version; expected 1")
    if type(document["section"]) is not str or re.fullmatch(r"[0-9]+\.[0-9]+", document["section"]) is None:
        raise ValueError("Invalid teaching-edition section identity")
    if type(document["sources"]) is not dict or set(document["sources"]) != set(_KINDS):
        raise ValueError("Teaching-edition sources must specify page, exercises and reference")
    for kind, source in document["sources"].items():
        if type(source) is not dict or set(source) != {"path", "sha256"}:
            raise ValueError(f"Invalid {kind} source keys")
        _relative(source["path"])
        suffix = ".md" if kind == "page" else ("_exercises.ipynb" if kind == "exercises" else "_solutions.ipynb")
        if not source["path"].endswith(suffix):
            raise ValueError(f"Invalid {kind} source suffix")
        if type(source["sha256"]) is not str or re.fullmatch("[0-9a-f]{64}", source["sha256"]) is None:
            raise ValueError(f"Invalid {kind} source sha256")
    if "rationale" in document and (type(document["rationale"]) is not list or any(type(x) is not str or not x for x in document["rationale"])):
        raise ValueError("Teaching-edition rationale must be a list of nonempty strings")
    if "drop_empty_markdown_cells" in document and type(document["drop_empty_markdown_cells"]) is not bool:
        raise ValueError("drop_empty_markdown_cells must be boolean")
    collapses = document.get("presentation_collapses", [])
    if type(collapses) is not list or (collapses and document["section"] != "16.3"):
        raise ValueError("presentation_collapses is a finite list restricted to section 16.3")
    collapse_hashes = set()
    for item in collapses:
        if (type(item) is not dict or set(item) != {"source_sha256", "reason"}
                or type(item["source_sha256"]) is not str or re.fullmatch("[0-9a-f]{64}", item["source_sha256"]) is None
                or type(item["reason"]) is not str or not item["reason"].strip()
                or item["source_sha256"] in collapse_hashes):
            raise ValueError("Presentation collapses need unique source SHA256s and nonempty reasons")
        collapse_hashes.add(item["source_sha256"])
    guards = document.get("practice_code_replacements", [])
    if type(guards) is not list:
        raise ValueError("practice_code_replacements must be a finite list")
    seen = set()
    for index, item in enumerate(guards):
        if (type(item) is not dict or set(item) != {"old", "new", "reason"}
                or any(type(item[key]) is not str or not item[key].strip() for key in item)
                or item["old"] == item["new"] or item["old"] in seen):
            raise ValueError(f"Practice code replacement {index} needs unique exact old/new sources and a reason")
        seen.add(item["old"])
        _validate_practice_change(document["section"], item["old"], item["new"])
    replacements = document["replacements"]
    if type(replacements) is not list or (not replacements and not document.get("drop_empty_markdown_cells", False) and not guards and not collapses):
        raise ValueError("Teaching edition requires finite replacements or explicit empty-Markdown removal")
    for index, item in enumerate(replacements):
        if type(item) is not dict or set(item) != {"old", "new", "counts"}:
            raise ValueError(f"Replacement {index} has missing or unknown keys")
        if type(item["old"]) is not str or not item["old"] or type(item["new"]) is not str or item["old"] == item["new"]:
            raise ValueError(f"Replacement {index} needs distinct exact nonempty old and string new")
        if item["old"].count(_DELIMITER) != item["new"].count(_DELIMITER):
            raise ValueError(f"Replacement {index} must preserve page part delimiters")
        counts = item["counts"]
        if (type(counts) is not dict or set(counts) != set(_KINDS)
                or any(type(n) is not int or n < 0 for n in counts.values()) or not any(counts.values())):
            raise ValueError(f"Replacement {index} needs explicit nonnegative integer counts for all surfaces")


def _ast(node) -> str:
    return ast.dump(node, include_attributes=False)


def _flag_assignment(node, name: str) -> bool:
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)


def _disabled_flag(node, name: str) -> bool:
    return _flag_assignment(node, name) and isinstance(node.value, ast.Constant) and node.value.value is False


def _native_guard(node) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
            and node.test.id == "RUN_OPTIONAL_NATIVE" and not node.orelse)


@lru_cache(maxsize=128)
def _validate_practice_change(section: str, old: str, new: str) -> None:
    """Prove the exact finite AST transform; never compile or execute it."""
    try:
        before, after = ast.parse(old), ast.parse(new)
    except SyntaxError as exc:
        raise ValueError("Practice plumbing preimage and replacement must be complete Python code cells") from exc
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if sorted(_ast(n) for n in ast.walk(before) if isinstance(n, definitions)) != sorted(_ast(n) for n in ast.walk(after) if isinstance(n, definitions)):
        raise ValueError("Practice plumbing must preserve every function/class AST")
    before_has_flag = any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == "RUN_OPTIONAL_NATIVE" for n in ast.walk(before))
    unwrapped = []
    added_flag = False
    for node in after.body:
        if _disabled_flag(node, "RUN_OPTIONAL_NATIVE") and not before_has_flag:
            if added_flag:
                raise ValueError("Practice plumbing may add the disabled native flag only once per cell")
            added_flag = True
        elif _native_guard(node):
            unwrapped.extend(node.body)
        else:
            unwrapped.append(node)
    if len(unwrapped) != len(before.body):
        raise ValueError("Practice plumbing may only add a disabled flag and wrap original top-level statements")
    numerical_old = _ast(ast.parse("NUMERICAL_ONLY = os.environ.get('ARENA_DIFFUSION_NUMERICAL_ONLY', '0') == '1'").body[0])
    numerical_new = _ast(ast.parse("NUMERICAL_ONLY = os.environ.get('ARENA_DIFFUSION_NUMERICAL_ONLY', '1') == '1'").body[0])
    for original, candidate in zip(before.body, unwrapped, strict=True):
        if _ast(original) == _ast(candidate):
            continue
        if (_flag_assignment(original, "RUN_TEXT_STUDY") and _disabled_flag(candidate, "RUN_TEXT_STUDY")
                and isinstance(original.value, ast.Constant) and type(original.value.value) is bool):
            continue
        if section == "5.8" and _ast(original) == numerical_old and _ast(candidate) == numerical_new:
            continue
        raise ValueError("Practice plumbing changed original statements, imports, calls or core code outside the finite setup whitelist")


def load_edition(root: Path, row: dict) -> dict | None:
    """Resolve a manifest only for one already validated reader inventory row."""
    root = Path(root).absolute()
    section = row.get("number")
    if type(section) is not str or re.fullmatch(r"[0-9]+\.[0-9]+", section) is None:
        raise ValueError("Invalid configured teaching-edition section")
    relative = f"mastery_hub/teaching_editions/{section.replace('.', '_')}.json"
    raw = _read(root, relative, optional=True)
    if raw is None:
        return None
    document = _json(raw)
    _validate(document)
    if document["section"] != section or row.get("available") is not True:
        raise ValueError("Teaching-edition section differs from available configured lesson")
    authority = blank_practice.config_identity(root)
    for kind in _KINDS:
        try:
            configured = Path(row[kind]).absolute().relative_to(root).as_posix()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid configured {kind} teaching-edition path") from exc
        if document["sources"][kind]["path"] != configured:
            raise ValueError(f"Teaching-edition {kind} path differs from configured identity")
        source_bytes = _read(root, configured)
        _pin(document, kind, source_bytes)
        if kind == "exercises" and document.get("practice_code_replacements"):
            _practice_preimages(document, blank_practice._load(source_bytes))
    if blank_practice.config_identity(root) != authority:
        raise ValueError("Configuration changed while capturing teaching-edition sources")
    return {**document, "_root": str(root), "_manifest_sha256": _digest(raw),
            "_document_sha256": _digest(_typed(document).encode("utf-8")),
            "_configuration_sha256": authority}


def _document(spec: dict) -> dict:
    if type(spec) is not dict or not _RUNTIME <= spec.keys():
        raise ValueError("Use load_edition to capture teaching-edition authority")
    document = {key: value for key, value in spec.items() if key not in _RUNTIME}
    _validate(document)
    if _digest(_typed(document).encode("utf-8")) != spec["_document_sha256"]:
        raise ValueError("Captured teaching-edition manifest changed")
    return document


def _pin(document: dict, kind: str, raw: bytes | None) -> None:
    if type(raw) is not bytes or _digest(raw) != document["sources"][kind]["sha256"]:
        raise ValueError(f"Stale teaching-edition {kind} source pin: {document['sources'][kind]['path']}")


def _replace(document: dict, texts: list[str], kind: str) -> list[str]:
    for index, item in enumerate(document["replacements"]):
        expected = item["counts"][kind]
        if expected == 0:  # Explicitly inapplicable surface; do not alter its text.
            continue
        count = sum(text.count(item["old"]) for text in texts)
        if count != expected:
            raise ValueError(f"Teaching-edition {document['section']} replacement {index} {kind} preimage count: expected {expected}, found {count}")
        texts = [text.replace(item["old"], item["new"]) for text in texts]
    return texts


def _source(cell: dict) -> str:
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def _practice_preimages(document: dict, notebook: dict) -> list[tuple[int, dict]]:
    targets = []
    for index, item in enumerate(document.get("practice_code_replacements", [])):
        matches = [i for i, cell in enumerate(notebook["cells"])
                   if cell["cell_type"] == "code" and _source(cell) == item["old"]]
        if len(matches) != 1:
            raise ValueError(f"Practice code replacement {index} requires one exact whole-cell preimage; found {len(matches)}")
        targets.append((matches[0], item))
    return targets


def _practice_code(document: dict, result: dict) -> list[dict]:
    records = []
    for index, item in _practice_preimages(document, result):
        cell = result["cells"][index]
        if cell["outputs"] != [] or cell["execution_count"] is not None:
            raise ValueError("Practice plumbing requires blank parent code outputs and execution counts")
        cell["source"] = item["new"] if isinstance(cell["source"], str) else item["new"].splitlines(keepends=True)
        records.append({"parent_cell_index": index, "source_sha256": _digest(item["old"].encode()),
                        "guarded_source_sha256": _digest(item["new"].encode()), "reason": item["reason"]})
    if records:
        ready = False
        for cell in result["cells"]:
            if cell["cell_type"] != "code":
                continue
            try:
                nodes = ast.parse(_source(cell)).body
            except SyntaxError:
                continue  # Untouched notebook magics are outside this finite Python transform.
            for node in nodes:
                if _flag_assignment(node, "RUN_OPTIONAL_NATIVE"):
                    if not _disabled_flag(node, "RUN_OPTIONAL_NATIVE"):
                        raise ValueError("Practice native flag must default to False")
                    ready = True
                if _native_guard(node) and not ready:
                    raise ValueError("Practice native guard needs an earlier RUN_OPTIONAL_NATIVE=False setup")
    return records


def _presentation_folds(document: dict, result: dict) -> list[dict]:
    records = []
    for item in document.get("presentation_collapses", []):
        matches = [cell for cell in result["cells"] if cell["cell_type"] == "code"
                   and _digest(_source(cell).encode("utf-8")) == item["source_sha256"]]
        if len(matches) != 1:
            raise ValueError(f"Presentation collapse needs one exact helper source; found {len(matches)}")
        cell = matches[0]
        try:
            tree = ast.parse(_source(cell))
        except SyntaxError as exc:
            raise ValueError("Presentation collapse requires a complete Python helper cell") from exc
        if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
                or tree.body[0].name not in _FOLD_HELPERS
                or any(isinstance(node, ast.Name) and node.id == "NotImplementedError" for node in ast.walk(tree))):
            raise ValueError("Presentation collapse is restricted to the six complete 16.3 setup helpers")
        namespace = cell["metadata"].setdefault("jupyter", {})
        if type(namespace) is not dict:
            raise ValueError("Presentation collapse requires object-valued Jupyter metadata")
        namespace["source_hidden"] = True
        records.append(dict(item, helper=tree.body[0].name))
    return records


def apply_page(spec: dict, raw_page: str) -> str:
    """Apply exact replacements to the pinned raw page before media rendering."""
    document = _document(spec)
    if type(raw_page) is not str:
        raise ValueError("Teaching-edition page must be decoded UTF-8 text")
    _pin(document, "page", raw_page.encode("utf-8"))
    result = _replace(document, [raw_page], "page")[0]
    if result.count(_DELIMITER) != raw_page.count(_DELIMITER):
        raise ValueError("Teaching edition changed page part boundaries")
    return result


def _base(spec: dict, document: dict, canonical_bytes: bytes, kind: str, base_bytes: bytes | None) -> tuple[bytes, dict, str]:
    if kind not in ("exercises", "reference"):
        raise ValueError("Teaching-edition notebook kind must be exercises or reference")
    _pin(document, kind, canonical_bytes)
    canonical = blank_practice._load(canonical_bytes)
    raw = canonical_bytes if base_bytes is None else base_bytes
    if type(raw) is not bytes:
        raise ValueError("Teaching-edition notebook parent must be bytes")
    base = blank_practice._load(raw)
    parent = "canonical"
    if raw != canonical_bytes:
        if kind != "exercises":
            raise ValueError("Only exercises may compose after validated Blank Practice")
        provenance = base["metadata"].get("arena_blank_practice", {})
        authority = blank_practice.config_identity(Path(spec["_root"]))
        if authority != spec["_configuration_sha256"]:
            raise ValueError("Configuration changed after capturing teaching-edition authority")
        if type(provenance) is not dict or provenance.get("configuration_sha256") != authority:
            raise ValueError("Blank Practice parent configuration differs from captured authority")
        restored = blank_practice.restore_download(raw, canonical_bytes, document["sources"][kind]["path"], authority)
        if _typed(restored) != _typed(canonical):
            raise ValueError("Blank Practice parent did not restore canonical source")
        parent = "blank_practice_download"
    if _KEY in base["metadata"]:
        raise ValueError("Teaching-edition provenance metadata collision")
    return raw, base, parent


def project_notebook(spec: dict, canonical_bytes: bytes, kind: str, base_bytes: bytes | None = None) -> bytes:
    """Project prose and explicitly reviewed blank-Practice setup plumbing."""
    document = _document(spec)
    raw, base, parent = _base(spec, document, canonical_bytes, kind, base_bytes)
    result = deepcopy(base)
    plumbing = _practice_code(document, result) if kind == "exercises" and parent == "blank_practice_download" else []
    folding = _presentation_folds(document, result)
    markdown = [cell for cell in result["cells"] if cell["cell_type"] == "markdown"]
    texts = [cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"]) for cell in markdown]
    revised = _replace(document, texts, kind)
    for cell, old, new in zip(markdown, texts, revised, strict=True):
        if old != new:
            cell["source"] = new if isinstance(cell["source"], str) else new.splitlines(keepends=True)
    if document.get("drop_empty_markdown_cells", False):
        result["cells"] = [cell for cell in result["cells"] if not (
            cell["cell_type"] == "markdown" and not "".join(cell["source"]).strip())]
    result["metadata"][_KEY] = {
        "schema_version": 1, "section": document["section"], "kind": kind,
        "source_path": document["sources"][kind]["path"],
        "source_sha256": document["sources"][kind]["sha256"],
        "manifest_sha256": spec["_manifest_sha256"], "base_sha256": _digest(raw),
        "parent_operation": parent, "execution_claim": "none",
        "output_qualification": (
            "PUBLIC Practice parent code preserved; parent outputs and execution counts were already cleared by validated Blank Practice. Historical metadata retained."
            if parent == "blank_practice_download" else "Historical code/results copied verbatim from the parent; prose revised."
        ) + " No fresh execution, scientific qualification or learner acceptance.",
    }
    if plumbing:
        result["metadata"][_KEY]["practice_plumbing_revised"] = True
        result["metadata"][_KEY]["practice_code_replacements"] = plumbing
        result["metadata"][_KEY]["output_qualification"] = (
            "PUBLIC Practice: reviewed optional-run setup plumbing revised; core function/class ASTs retained. "
            "Parent outputs and execution counts remain blank. No fresh execution, scientific qualification or learner acceptance."
        )
    if folding:
        result["metadata"][_KEY]["presentation_folding"] = True
        result["metadata"][_KEY]["presentation_collapses"] = folding
    blank_practice._validate(result)
    return (json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")


def restore_notebook(spec: dict, projected_bytes: bytes, canonical_bytes: bytes, kind: str, base_bytes: bytes | None = None) -> bytes:
    """Accept exactly the recomputed typed derivative and return exact parent bytes."""
    expected = blank_practice._load(project_notebook(spec, canonical_bytes, kind, base_bytes))
    actual = blank_practice._load(projected_bytes)
    if _typed(actual) != _typed(expected):
        raise ValueError("Teaching-edition derivative differs from the exact finite projection")
    return canonical_bytes if base_bytes is None else base_bytes
