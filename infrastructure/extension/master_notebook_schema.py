"""Pure, fail-closed schema compatibility for extension saved master notebooks.

The upstream converter remains byte-exact. Only its in-memory master JSON passes
through here; generated lesson notebooks and their preservation recipes do not.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from nbformat.validator import get_validator


_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Extension master notebook: " + message)


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("Extension master notebook: non-JSON number: " + value)


def _float(value: str) -> float:
    result = float(value)
    _require(math.isfinite(result), "nonfinite JSON number")
    return result


def canonical_master_notebook(raw_json: str, *, master_basename: str) -> str:
    """Return strict notebook 4.5 JSON without changing teaching content/evidence.

    Allowed changes are exactly: bump 4.2 to 4.5; remove markdown outputs=[] and
    execution_count=null when present; add absent cell IDs. Existing IDs survive
    unchanged and must be valid and unique. An absent ID is 'ext-' followed by the
    first 32 hex digits of SHA-256 over UTF-8 compact JSON of
    ["arena.extension-master-cell.v1", master_basename, zero_based_index,
    cell_type, source]. JSON uses ensure_ascii=False and allow_nan=False; source
    retains its exact string-or-list representation. Including basename and index
    distinguishes sections and repeated cells without any existing-notebook read.
    A collision is an error, including a collision with a supplied ID.

    Strict JSON parsing and direct get_validator(4, 5).iter_errors checks perform
    no nbformat normalization, repair, implicit field removal, or output changes.
    The complete serialized result must strictly encode as UTF-8 in memory,
    before the caller can open the existing master for its single write.
    """
    _require(isinstance(master_basename, str) and bool(master_basename)
             and Path(master_basename).name == master_basename
             and "\\" not in master_basename and master_basename.endswith(".py"),
             "expected a Python master basename")
    book = json.loads(raw_json, object_pairs_hook=_unique,
                      parse_constant=_constant, parse_float=_float)
    _require(isinstance(book, dict), "notebook must be an object")
    _require(type(book.get("nbformat")) is int and book["nbformat"] == 4,
             "expected notebook major version 4")
    _require(type(book.get("nbformat_minor")) is int and book["nbformat_minor"] in {2, 5},
             "expected notebook minor version 2 or 5")
    _require(isinstance(book.get("cells"), list), "cells must be an array")
    ids = set()
    for index, cell in enumerate(book["cells"]):
        _require(isinstance(cell, dict), f"cell {index} must be an object")
        if cell.get("cell_type") == "markdown":
            if "execution_count" in cell:
                _require(cell["execution_count"] is None,
                         f"markdown cell {index} contains execution_count evidence")
                del cell["execution_count"]
            if "outputs" in cell:
                _require(isinstance(cell["outputs"], list) and not cell["outputs"],
                         f"markdown cell {index} contains outputs evidence")
                del cell["outputs"]
        elif cell.get("cell_type") == "code":
            count = cell.get("execution_count")
            _require(count is None or type(count) is int,
                     f"code cell {index} execution_count must be an integer or null")
            outputs = cell.get("outputs")
            if isinstance(outputs, list):
                for output in outputs:
                    if isinstance(output, dict) and "execution_count" in output:
                        _require(type(output["execution_count"]) is int,
                                 f"code cell {index} output execution_count must be an integer")
        if "id" not in cell:
            payload = ["arena.extension-master-cell.v1", master_basename, index,
                       cell.get("cell_type"), cell.get("source")]
            identity = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            cell["id"] = "ext-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        identifier = cell["id"]
        _require(isinstance(identifier, str) and _ID.fullmatch(identifier) is not None,
                 f"cell {index} has invalid ID")
        _require(identifier not in ids, f"cell {index} has duplicate ID")
        ids.add(identifier)
    book["nbformat_minor"] = 5
    failures = list(get_validator(4, 5).iter_errors(book))
    _require(not failures, "invalid notebook schema: " + "; ".join(str(error) for error in failures))
    serialized = json.dumps(book, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    serialized.encode("utf-8", errors="strict")
    return serialized
