"""Fail-closed, source-only recipes for reviewed historical lesson books.

This module never imports taught code or reads the current published notebooks.
A changed registered master or converter result requires a new reviewed recipe.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from nbformat.validator import get_validator

from scripts.evidence_bundles import validate_review_bundle

SCHEMA = "arena.extension-source-preservation.v1"
CONTRACTS = "infrastructure/extension/source_preservation_contracts.json"


def _require(condition, message):
    if not condition:
        raise ValueError("Source preservation: " + message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _same_json(left, right):
    """Ignore object key order while preserving array layout and scalar types."""
    return json.dumps(left, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _path(root, relative):
    _require(isinstance(relative, str) and bool(relative), "missing relative path")
    p = PurePosixPath(relative)
    _require(not p.is_absolute() and ".." not in p.parts and "\\" not in relative,
             "unsafe relative path")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), "path escapes repository")
    return path


def _archive(root, binding):
    archive = _path(root, binding["path"])
    sidecar = Path(str(archive) + ".review-bundle.json")
    manifest = _path(root, binding["manifest_path"])
    _require(_digest(archive.read_bytes()) == binding["sha256"], "archive bytes changed")
    _require(_digest(sidecar.read_bytes()) == binding["contract_sha256"], "archive contract changed")
    _require(_digest(manifest.read_bytes()) == binding["manifest_sha256"], "archive manifest changed")
    contract = validate_review_bundle(archive)
    source = contract["provenance"]["source_data"]
    _require(source["repository_path"] == binding["manifest_path"], "provenance path changed")
    _require(source["sha256"] == binding["manifest_sha256"], "provenance hash changed")
    with tarfile.open(archive) as handle:
        return {name: handle.extractfile(name).read() for name in handle.getnames()}


def _cells(book):
    return [{"cell_type": cell["cell_type"],
             "source_sha256": _digest("".join(cell["source"]).encode())}
            for cell in book["cells"]]


def _navigation(book):
    indices = [i for i, cell in enumerate(book["cells"])
               if cell["cell_type"] == "markdown" and "> **Colab:" in "".join(cell["source"])]
    _require(len(indices) == 1, "expected exactly one navigation cell")
    return indices[0]


def _noncode_fields(book, *, raw=False):
    allowed = {"cell_type", "source", "metadata", "id", "attachments"}
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            extras = set(cell) - allowed
            if raw:
                _require(extras <= {"execution_count", "outputs"}, "unexpected noncode field")
                _require(cell.get("execution_count") is None and cell.get("outputs", []) == [],
                         "nonempty noncode execution evidence")
            else:
                _require(not extras, "undeclared noncode field")


def apply_compatibility(book, pin):
    """Apply only explicit invertible metadata operations to a copied book."""
    result = deepcopy(book)
    seen = set()
    for removal in pin["compatibility_removals"]:
        index, field = removal["cell"], removal["field"]
        cell = result["cells"][index]
        _require((index, field) not in seen, "duplicate field removal")
        seen.add((index, field))
        _require(cell["cell_type"] == "markdown" and field in {"execution_count", "outputs"},
                 "invalid compatibility removal")
        expected = None if field == "execution_count" else []
        _require(removal["value"] == expected and field in cell and cell[field] == expected,
                 "nonempty or absent compatibility field")
        del cell[field]
    for addition in pin["id_additions"]:
        cell = result["cells"][addition["cell"]]
        _require(cell["cell_type"] == "markdown" and "id" not in cell,
                 "ID addition would change an existing ID or code cell")
        cell["id"] = addition["id"]
    _noncode_fields(result)
    _require(all("id" in cell for cell in result["cells"]), "undeclared missing cell ID")
    ids = [cell["id"] for cell in result["cells"]]
    _require(len(ids) == len(set(ids)), "duplicate cell ID")
    _require(not list(get_validator(4, 5).iter_errors(result)), "invalid canonical notebook schema")
    return result


def restore_notebooks(*, master_path, identity, raw_notebooks, raw_modules,
                      root, contracts_path=None, existing_notebooks=None):
    """Return exact archived books plus generated navigation, or None if unregistered.

    The actual master_path may be a copied temporary master. All registered input
    mismatches raise before any output is returned; there is no fallback or write.
    """
    root = Path(root)
    path = Path(contracts_path) if contracts_path is not None else root / CONTRACTS
    contracts = _json(path.read_bytes())
    _require(set(contracts) == {"schema", "sections"} and contracts["schema"] == SCHEMA,
             "unsupported recipe manifest")
    section = identity["section"]
    registered_names = {"master_" + key.replace(".", "_") + ".py": key
                        for key in contracts["sections"]}
    registered_section = registered_names.get(Path(master_path).name)
    _require(registered_section is None or registered_section == section,
             "registered master section identity mismatch")
    recipe = contracts["sections"].get(section)
    if recipe is None:
        return None
    _require(identity == recipe["identity"], "section identity mismatch")
    _require(Path(master_path).name == "master_" + section.replace(".", "_") + ".py",
             "master basename mismatch")
    _require(_digest(Path(master_path).read_bytes()) == recipe["master_sha256"], "actual master bytes changed")
    before = _archive(root, recipe["archives"]["before"])
    archived_raw = _archive(root, recipe["archives"]["raw"])
    _require(set(raw_notebooks) == set(recipe["books"]) == {"exercises", "solutions"},
             "notebook set changed")
    _require(set(raw_modules) == set(recipe["modules"]), "solution module set changed")
    for name, pin in recipe["modules"].items():
        _require(_digest(raw_modules[name].encode()) == pin["sha256"], "raw solution module changed")
        _require(archived_raw[pin["member"]] == raw_modules[name].encode(), "raw module archive mismatch")
    planned = {}
    for kind, pin in recipe["books"].items():
        original_bytes = before[pin["member"]]
        _require(_digest(original_bytes) == pin["before_sha256"], "original notebook identity changed")
        original = _json(original_bytes)
        generated = _json(raw_notebooks[kind])
        recorded = _json(archived_raw[pin["member"]])
        _noncode_fields(generated, raw=True)
        _require(_cells(generated) == pin["raw_cells"] == _cells(recorded),
                 "ordered raw cell source/type fingerprints changed")
        old_code = [c for c in original["cells"] if c["cell_type"] == "code"]
        new_code = [c for c in generated["cells"] if c["cell_type"] == "code"]
        _require(len(old_code) == len(new_code), "code cell count changed")
        for old, new in zip(old_code, new_code, strict=True):
            _require(ast.dump(ast.parse("".join(old["source"])))
                     == ast.dump(ast.parse("".join(new["source"]))), "archived teaching code mismatch")
        old_index, new_index = _navigation(original), _navigation(generated)
        navigation = generated["cells"][new_index]["source"]
        text = "".join(navigation)
        _require("start the guided workbook" in text and "#section-" + section.replace(".", "-") in text,
                 "guided navigation identity changed")
        restored = deepcopy(original)
        restored["cells"][old_index]["source"] = deepcopy(navigation)
        navigation_updated = deepcopy(restored)
        restored = apply_compatibility(restored, pin)
        if existing_notebooks is not None and existing_notebooks.get(kind) is not None:
            existing = _json(existing_notebooks[kind])
            _require(any(_same_json(existing, allowed)
                         for allowed in (original, navigation_updated, restored)),
                     "published notebook has unreviewed changes: " + kind)
        planned[kind] = json.dumps(restored, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    return planned
