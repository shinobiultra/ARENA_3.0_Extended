"""Offline source navigation; upstream publication never implies companion readiness."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {"id", "title", "source_url", "checked_on", "source_status", "companion_status",
          "prerequisites", "original_paths", "entrypoint"}


def load_catalog(path: Path, *, root: Path = ROOT) -> list[dict[str, object]]:
    """Validate a version-1 catalog without contacting any external source."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schema_version", "units"}:
        raise ValueError("Catalog requires schema_version and units")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported catalog schema version")
    rows = data["units"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Catalog units must be a nonempty list")
    known = {}
    base = root.resolve()

    def local_path(value: object) -> Path:
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("Catalog paths must remain inside the repository")
        target = (base / value).resolve()
        if not target.is_relative_to(base) or not target.exists():
            raise ValueError(f"Missing or unsafe catalog path: {value}")
        return target

    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Unexpected or missing unit fields")
        unit_id = row["id"]
        if not isinstance(unit_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", unit_id) or unit_id in known:
            raise ValueError("Unit IDs must be unique safe identifiers")
        known[unit_id] = row
        if not isinstance(row["title"], str) or not row["title"].strip():
            raise ValueError("Unit title is required")
        if not isinstance(row["source_url"], str):
            raise ValueError("Source URL must be HTTPS")
        url = urlparse(row["source_url"])
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            raise ValueError("Source URL must be HTTPS without credentials")
        if not isinstance(row["checked_on"], str):
            raise ValueError("checked_on must be an ISO date")
        try:
            checked = dt.date.fromisoformat(row["checked_on"])
        except ValueError as exc:
            raise ValueError("checked_on must be an ISO date") from exc
        if checked.isoformat() != row["checked_on"]:
            raise ValueError("checked_on must use YYYY-MM-DD")
        if row["source_status"] not in {"published", "pending_upstream", "link_only"}:
            raise ValueError("Unknown source status")
        if row["companion_status"] not in {"planned", "prototype", "validated", "learner_ready"}:
            raise ValueError("Unknown companion status")
        if not isinstance(row["prerequisites"], list) or any(not isinstance(p, str) for p in row["prerequisites"]):
            raise ValueError("Prerequisites must be unit IDs")
        if not isinstance(row["original_paths"], list) or not row["original_paths"]:
            raise ValueError("Original source paths are required")
        for original in row["original_paths"]:
            local_path(original)
        entry = row["entrypoint"]
        if row["companion_status"] == "planned" and entry is not None:
            raise ValueError("A planned companion has no runnable entrypoint")
        if row["source_status"] == "pending_upstream" and (entry is not None or row["companion_status"] != "planned"):
            raise ValueError("Pending upstream material cannot claim a ready companion")
        if row["companion_status"] in {"validated", "learner_ready"} and entry is None:
            raise ValueError("A validated companion requires an entrypoint")
        if entry is not None and not local_path(entry).is_file():
            raise ValueError("Companion entrypoint must be a file")

    visited, active = set(), set()

    def visit(unit_id: str) -> None:
        if unit_id not in known:
            raise ValueError(f"Unknown prerequisite: {unit_id}")
        if unit_id in active:
            raise ValueError("Prerequisite cycle")
        if unit_id in visited:
            return
        active.add(unit_id)
        for prereq in known[unit_id]["prerequisites"]:
            visit(prereq)
        active.remove(unit_id)
        visited.add(unit_id)

    for unit_id in known:
        visit(unit_id)
    return rows
