"""Exact approved ROME prose descendant -> frozen historical test checkpoints.

Test-only file plumbing: no taught import, method invocation or file mutation.
Only the exact declared current state is accepted; no old-or-new fallback.
"""
from pathlib import Path, PurePosixPath
import copy
import hashlib
import json
import os
import stat
import tarfile

CONTRACT_PATH = "docs/mastery/evidence/rome_runtime_metadata_20261021_reconstructed_v1/checkpoint_inverse_contract.json"
CONTRACT_SHA256 = "8dc502197c5fac858ad609429836f1ecf0a12a706269c4a1dc5a2d911e85b018"
MASTER = "infrastructure/chapters/chapter9_alignment_interpretability/master_9_6.ipynb"
MASTER_PY = MASTER.removesuffix(".ipynb") + ".py"
DIRECTORY = "chapter9_alignment_interpretability/exercises/part6_rome_memit"
STEM = "9.6_Model_Editing_with_ROME_and_MEMIT"
BOOKS = tuple(DIRECTORY + "/" + STEM + "_" + kind + ".ipynb" for kind in ("exercises", "solutions"))
PAGE = "chapter9_alignment_interpretability/instructions/pages/06_[9.6]_Model_Editing_with_ROME_and_MEMIT"
JSON_TARGETS = (MASTER, *BOOKS)
PRODUCTS = (MASTER_PY, MASTER, *BOOKS, PAGE + ".md")
UNCHANGED = (DIRECTORY + "/solutions.py", PAGE + ".py")
CONTRACTS = "infrastructure/extension/source_preservation_contracts.json"
CASES = "mastery_hub/extension_checkpoints.json"
MODULE_SHA256 = "0e5fc0d94677487c4657919bba50aca893f9cdbfeca05593c435e85c8cc8f97d"
TOP_KEYS = {"schema", "section", "products", "unchanged_products", "cases", "contracts", "artifacts", "intro_inverses", "text_inverse", "module_sha256", "case_objects_changed", "other17_recipe_objects_changed", "only_approved_current_state_is_accepted"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON constant " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def _typed(value):
    # bool/int/float and array order remain distinguishable in canonical JSON.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _read(root, relative):
    _require(type(relative) is str, "Path must be a declared relative string")
    parts = PurePosixPath(relative)
    _require(not parts.is_absolute() and ".." not in parts.parts and "\\" not in relative and parts.as_posix() == relative, "Unsafe relative path")
    root = Path(root)
    _require(root.is_absolute() and not root.is_symlink(), "Root must be an absolute ordinary directory")
    target = root / relative
    for path in [target, *target.parents]:
        if path == root.parent:
            break
        _require(not path.is_symlink(), "Symlink input refused")
    _require(target.resolve().is_relative_to(root.resolve()), "Path escapes root")
    def parent_fd():
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts.parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd
        except BaseException:
            os.close(fd)
            raise
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)
    directory_fd = parent_fd()
    try:
        before = os.stat(parts.parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        _require(stat.S_ISREG(before.st_mode), "Input is not a regular file")
        fd = os.open(parts.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        try:
            _require(identity(os.fstat(fd)) == identity(before), "Input identity changed before read")
            chunks = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            _require(identity(os.fstat(fd)) == identity(before), "Input identity changed during read")
        finally:
            os.close(fd)
        _require(identity(os.stat(parts.parts[-1], dir_fd=directory_fd, follow_symlinks=False)) == identity(before), "Input path changed during read")
        final_fd = parent_fd()
        try:
            _require(identity(os.stat(parts.parts[-1], dir_fd=final_fd, follow_symlinks=False)) == identity(before), "Input parent path changed during read")
        finally:
            os.close(final_fd)
        return b"".join(chunks)
    finally:
        os.close(directory_fd)


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def _set(value, path, replacement):
    node = value
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement


def _artifact(root, description):
    _require(set(description) == {"archive_path", "archive", "sidecar_path", "sidecar", "manifest_path", "manifest", "files"}, "Unknown artifact field")
    for key, record_key in (("archive_path", "archive"), ("sidecar_path", "sidecar"), ("manifest_path", "manifest")):
        _require(_record(_read(root, description[key])) == description[record_key], "Artifact identity changed: " + key)
    sidecar = _json(_read(root, description["sidecar_path"]))
    manifest = _json(_read(root, description["manifest_path"]))
    _require(set(manifest) == {"scope", "files"} and _typed(manifest["files"]) == _typed(description["files"]), "Manifest members changed")
    expected_source = {"absolute_path": str(Path(root) / description["manifest_path"]), "repository_path": description["manifest_path"], **description["manifest"]}
    _require(_typed(sidecar["provenance"]["source_data"]) == _typed(expected_source), "Source provenance changed")
    from scripts.evidence_bundles import validate_review_bundle
    validate_review_bundle(Path(root) / description["archive_path"])
    payload = {}
    with tarfile.open(Path(root) / description["archive_path"]) as handle:
        members = handle.getmembers()
        _require(len(members) == len(description["files"]) and {m.name for m in members} == set(description["files"]), "Unknown or missing archive members")
        for member in members:
            _require(member.isfile(), "Nonregular archive member")
            data = handle.extractfile(member).read()
            _require(_record(data) == description["files"][member.name], "Archive member changed")
            payload[member.name] = data
    return payload


def _guard(root):
    raw = _read(root, CONTRACT_PATH)
    _require(hashlib.sha256(raw).hexdigest() == CONTRACT_SHA256, "Inverse contract changed")
    contract = _json(raw)
    _require(set(contract) == TOP_KEYS and contract["schema"] == "arena.rome.runtime-wording-checkpoint.v1" and contract["section"] == "9.6", "Unknown inverse contract schema")
    _require(contract["case_objects_changed"] is False and contract["other17_recipe_objects_changed"] is False and contract["only_approved_current_state_is_accepted"] is True, "Unexpected inverse flags")
    _require(set(contract["products"]) == set(PRODUCTS) and set(contract["unchanged_products"]) == set(UNCHANGED), "Product set changed")
    current = {}
    for relative, pin in {**contract["products"], **contract["unchanged_products"]}.items():
        current[relative] = _read(root, relative)
        _require(_record(current[relative]) == pin, "Current approved product changed: " + relative)
    _require(contract["module_sha256"] == MODULE_SHA256 and _record(current[UNCHANGED[0]])["sha256"] == MODULE_SHA256, "Unchanged method module changed")
    _require(set(contract["cases"]) == {"path", "sha256", "size_bytes", "count"} and contract["cases"]["path"] == CASES and type(contract["cases"]["count"]) is int and contract["cases"]["count"] == 60, "Case binding changed")
    case_bytes = _read(root, CASES)
    _require(_record(case_bytes) == {k: contract["cases"][k] for k in ("sha256", "size_bytes")}, "Current complete60 case bytes changed")
    _require(len(_json(case_bytes)["checkpoints"]) == 60, "Current case cohort changed")
    binding = contract["contracts"]
    _require(set(binding) == {"path", "before", "after", "ordered_sections", "top_keys", "schema_value", "before_recipe", "after_recipe", "allowed_recipe_paths"} and binding["path"] == CONTRACTS, "Contract binding fields changed")
    actual_contract_bytes = _read(root, CONTRACTS)
    _require(_record(actual_contract_bytes) == binding["after"], "Current complete18 contract bytes changed")
    actual_contracts = _json(actual_contract_bytes)
    _require(list(actual_contracts) == binding["top_keys"] and actual_contracts["schema"] == binding["schema_value"], "Contracts top-level fields changed")
    _require(list(actual_contracts["sections"]) == binding["ordered_sections"] and len(actual_contracts["sections"]) == 18, "Complete18 recipe set changed")
    _require(set(contract["artifacts"]) == {"actual60_preimages", "before", "raw"}, "Artifact role set changed")
    payloads = {name: _artifact(root, row) for name, row in contract["artifacts"].items()}
    originals = payloads["actual60_preimages"]
    before_contract_bytes = originals[CONTRACTS]
    _require(_record(before_contract_bytes) == binding["before"], "Preserved full18 contracts changed")
    before_contracts = _json(before_contract_bytes)
    _require(list(before_contracts) == binding["top_keys"] and list(before_contracts["sections"]) == binding["ordered_sections"] and len(before_contracts["sections"]) == 18, "Preserved contracts shape changed")
    _require(originals[CASES] == case_bytes, "Any of60 cases changed")
    _require(_typed(before_contracts["sections"]["9.6"]) == _typed(binding["before_recipe"]) and _typed(actual_contracts["sections"]["9.6"]) == _typed(binding["after_recipe"]), "Old/new9.6 recipe pair changed")
    recovered = copy.deepcopy(actual_contracts)
    recovered["sections"]["9.6"] = copy.deepcopy(binding["before_recipe"])
    _require(_typed(recovered) == _typed(before_contracts), "One of17 other recipes or top fields changed")
    allowed = [["master_sha256"], ["archives", "before"], ["archives", "raw"]]
    for kind in ("exercises", "solutions"):
        allowed.extend([["books", kind, "before_sha256"], ["books", kind, "raw_cells", 2, "source_sha256"]])
    _require(binding["allowed_recipe_paths"] == allowed, "Recipe operation set changed")
    undo_recipe = copy.deepcopy(binding["after_recipe"])
    for path in allowed:
        _set(undo_recipe, path, _at(binding["before_recipe"], path))
    _require(_typed(undo_recipe) == _typed(binding["before_recipe"]), "Recipe changed beyond finite approved fields")
    expected_members = set(BOOKS) | {UNCHANGED[0]}
    for name in ("before", "raw"):
        _require(set(payloads[name]) == expected_members, "Before/raw archive contains unknown scientific payload")
        _require(payloads[name][UNCHANGED[0]] == current[UNCHANGED[0]], "Method module archive bytes changed")
    inverses = contract["intro_inverses"]
    _require(len(inverses) == 3 and [v["path"] for v in inverses] == list(JSON_TARGETS), "JSON inverse path set changed")
    for inverse in inverses:
        _require(set(inverse) == {"path", "cell", "cell_count", "ordered_cell_ids_before", "ordered_cell_ids_after", "fields"}, "Unknown source inverse field")
        relative = inverse["path"]
        original = _json(originals[relative])
        actual = _json(current[relative])
        is_master = relative == MASTER
        _require(type(inverse["cell"]) is int and inverse["cell"] == (3 if is_master else 1) and type(inverse["cell_count"]) is int and inverse["cell_count"] == (110 if is_master else 105), "Inverse cell/index changed")
        _require(len(actual["cells"]) == len(original["cells"]) == inverse["cell_count"], "Whole-cell count changed")
        _require([c["id"] for c in actual["cells"]] == inverse["ordered_cell_ids_after"] and [c["id"] for c in original["cells"]] == inverse["ordered_cell_ids_before"], "Ordered IDs changed")
        index = inverse["cell"]
        _require(actual["cells"][index]["cell_type"] == original["cells"][index]["cell_type"] == "markdown", "Inverse target is not Markdown")
        if not is_master:
            _require(actual["cells"][index]["id"] == original["cells"][index]["id"] == "rome96-intro", "Known intro ID changed")
        _require(set(inverse["fields"]) == ({"source", "id"} if is_master else {"source"}), "Source inverse operation set changed")
        undo = copy.deepcopy(actual)
        for field, operation in inverse["fields"].items():
            _require(set(operation) == {"before", "after"} and _typed(undo["cells"][index][field]) == _typed(operation["after"]) and _typed(original["cells"][index][field]) == _typed(operation["before"]), "Source inverse input mismatch")
            undo["cells"][index][field] = copy.deepcopy(operation["before"])
        _require(_typed(undo) == _typed(original), "Another whole-cell object or top field changed")
        if not is_master:
            _require(_typed(_json(payloads["before"][relative])) == _typed(actual), "Transformed before template differs from current whole book object")
            old_ids = [c["id"] for c in original["cells"] if c["id"].startswith("rome-")]
            _require(old_ids == ["rome-" + str(i).zfill(3) for i in range(69)], "Original69 source cell identities changed")
    text = contract["text_inverse"]
    _require(set(text) == {"paths", "before", "after"} and text["paths"] == [MASTER_PY, PAGE + ".md"], "Text inverse paths changed")
    for relative in text["paths"]:
        before = originals[relative]
        _require(before.count(text["before"].encode()) == 1 and before.replace(text["before"].encode(), text["after"].encode()) == current[relative], "Text changed beyond one approved sentence")
    return contract, current, originals, actual_contract_bytes, before_contracts


def recover_intro_checkpoint_bytes(root, relative_path, actual_bytes) -> bytes:
    """Return exact original bytes only for the three approved current JSON targets."""
    _require(type(relative_path) is str and relative_path in JSON_TARGETS, "Unknown historical checkpoint path")
    _require(type(actual_bytes) is bytes, "Checkpoint input must be original bytes")
    contract, current, originals, _, _ = _guard(root)
    _require(actual_bytes == current[relative_path], "Supplied bytes differ from current approved file")
    return originals[relative_path]


def recover_recipe_checkpoint_contracts(root, actual_contract_bytes) -> dict:
    """Return the complete preserved18-recipe object after exact-current-state proof."""
    _require(type(actual_contract_bytes) is bytes, "Complete contracts input must be bytes")
    _, _, _, current_bytes, before_contracts = _guard(root)
    _require(actual_contract_bytes == current_bytes, "Supplied contracts differ from exact complete current file")
    return before_contracts
