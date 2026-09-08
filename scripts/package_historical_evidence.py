"""Preserve original archives by digest, then derive strict review-only bundles.

This is a packaging migration, not a native execution or evidence refresh. The
source archive and every retained member remain recoverable at their original
hashes. Historical manifests and scientific receipts are not rewritten.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evidence_bundles import (
    RAW_SUFFIXES, contract_path, file_record, inspect_member, resolve_historical_archive,
    seal_json_data, seal_review_bundle, sha256, validate_review_bundle,
)


def preserve_original(path: Path, cache: Path) -> Path:
    """Copy before changing anything and verify the independent copy's exact bytes."""
    checksum = sha256(path)
    saved = cache / (checksum + ".tar.gz")
    saved.parent.mkdir(parents=True, exist_ok=True)
    if not saved.exists():
        shutil.copyfile(path, saved)
    if file_record(saved) != file_record(path):
        raise ValueError(f"original preservation failed: {path}")
    return saved


def _readable_json(value, depth: int = 0) -> str:
    """Keep each primitive array on one line; retain indentation for object records."""
    if not isinstance(value, (dict, list)) or not value:
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list) and all(not isinstance(item, (dict, list)) for item in value):
        return json.dumps(value, ensure_ascii=False)
    indent, child_indent = "  " * depth, "  " * (depth + 1)
    if isinstance(value, dict):
        records = [child_indent + json.dumps(key, ensure_ascii=False) + ": "
                   + _readable_json(item, depth + 1) for key, item in value.items()]
        return "{\n" + ",\n".join(records) + "\n" + indent + "}"
    records = [child_indent + _readable_json(item, depth + 1) for item in value]
    return "[\n" + ",\n".join(records) + "\n" + indent + "]"


def package_json_evidence(path: Path, cache: Path, *, compressed: bool = False) -> dict:
    """Retain gzip outcomes unchanged, or format JSON without changing any value."""
    before = file_record(path)
    original = cache / (before["sha256"] + (".json.gz" if compressed else ".json"))
    original.parent.mkdir(parents=True, exist_ok=True)
    if not original.exists():
        shutil.copyfile(path, original)
    if file_record(original) != before:
        raise ValueError("JSON original preservation failed")
    provenance = {
        "kind": "historical_data_packaging",
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_claim": "Packaging only; all recorded outcomes, values and measurement dates are unchanged.",
        "source_data": {"repository_path": path.relative_to(ROOT).as_posix(),
                        "absolute_path": str(original.resolve()), **before},
    }
    if compressed:
        return seal_json_data(path, provenance=provenance)
    value = json.loads(path.read_text())
    formatted = _readable_json(value) + "\n"
    # Canonical reserialization also handles preserved NaN values, if a historical report uses them.
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if json.dumps(json.loads(formatted), sort_keys=True, ensure_ascii=False, separators=(",", ":")) != canonical:
        raise ValueError("JSON values changed during cosmetic formatting")
    path.write_text(formatted)
    record = {"schema": "arena.lossless-json-formatting.v1", "provenance": provenance,
              "formatted": file_record(path), "value_equivalence": True,
              "canonical_values_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
              "format": "Two-space object indentation; primitive array records stay on one line."}
    Path(str(path) + ".packaging.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def package_archive(path: Path, original: Path, *, git_commit: str = "", git_blob: str = "") -> dict:
    relative = path.relative_to(ROOT).as_posix()
    original_record = file_record(original)
    if git_blob:
        blob_hash = hashlib.sha1(f"blob {original.stat().st_size}\0".encode())
        with original.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                blob_hash.update(chunk)
        if blob_hash.hexdigest() != git_blob:
            raise ValueError(f"original archive differs from the proposed Git recovery blob: {relative}")
    source_members, external = {}, {}
    retained = []
    with tarfile.open(original, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or member.issparse() or member.name in source_members:
                raise ValueError(f"unsupported original member: {relative}: {member.name}")
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable original member: {member.name}")
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            record = {"sha256": checksum, "size_bytes": member.size}
            source_members[member.name] = record
            if Path(member.name.lower()).suffix in RAW_SUFFIXES:
                external[member.name] = {
                    **record, "source_archive_member": member.name,
                    "source_archive_sha256": original_record["sha256"],
                    "reason": "Raw model/tensor serialization is retained only in the exact original archive.",
                }
            else:
                stream = handle.extractfile(member)
                inspect_member(member.name, stream.read())
                retained.append(member.name)
    if external:
        temporary = path.with_name(path.name + ".packaging-tmp")
        with temporary.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as destination:
                    with tarfile.open(original, "r:gz") as source:
                        for member in source:
                            if member.name in retained:
                                destination.addfile(copy.copy(member), source.extractfile(member))
        temporary.replace(path)
    provenance = {
        "kind": "historical_archive_packaging",
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_claim": (
            "Packaging only. This is not a native rerun, scientific revalidation or new review. "
            "Retained receipts, failure labels, values, scientific hashes and measurement dates "
            "are byte-identical to the original. Historical embedded archive inventories "
            "describe that original; this adjacent contract describes the derived review bundle."
        ),
        "source_archive": {"repository_path": relative, "absolute_path": str(original.resolve()),
                           **original_record, "git_commit": git_commit, "git_blob": git_blob},
        "source_archive_members": source_members,
    }
    contract = seal_review_bundle(path, provenance=provenance, external_payloads=external)
    assert resolve_historical_archive(path, original_record["sha256"]).is_file()
    return {
        "repository_path": relative, "original": provenance["source_archive"],
        "review_bundle": contract["archive"],
        "review_contract": contract_path(path).relative_to(ROOT).as_posix(),
        "retained_members": len(retained), "externalized_members": len(external),
        "externalized_bytes": sum(record["size_bytes"] for record in external.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--exclude-part", action="append", default=[])
    parser.add_argument("--format-json", action="append", type=Path, default=[])
    parser.add_argument("--seal-json-data", action="append", type=Path, default=[])
    args = parser.parse_args()
    paths = sorted(path for path in ROOT.glob("chapter*/**/*.tar.gz")
                   if not any(part in path.parts for part in args.exclude_part))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "ls-tree", "-r", "HEAD"], cwd=ROOT, text=True)
    blobs = {line.split("\t", 1)[1]: line.split("\t", 1)[0].split()[2]
             for line in tree.splitlines()}
    pending = [path for path in paths if not contract_path(path).exists()]
    # The entire original byte set is safe before the first in-repository archive is replaced.
    originals = {path: preserve_original(path, args.cache) for path in pending}
    results = []
    for path in paths:
        if path not in originals:
            contract = validate_review_bundle(path)
            source = contract["provenance"]["source_archive"]
            resolve_historical_archive(path, source["sha256"])
            results.append({"repository_path": path.relative_to(ROOT).as_posix(),
                            "original": source, "review_bundle": contract["archive"],
                            "review_contract": contract_path(path).relative_to(ROOT).as_posix(),
                            "retained_members": len(contract["members"]),
                            "externalized_members": len(contract["external_payloads"]),
                            "externalized_bytes": sum(r["size_bytes"] for r in contract["external_payloads"].values())})
        else:
            results.append(package_archive(path, originals[path], git_commit=commit,
                                           git_blob=blobs.get(path.relative_to(ROOT).as_posix(), "")))
    inventory = {
        "schema": "arena.historical-packaging-inventory.v1",
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_claim": "Packaging only; no native computation or scientific evidence refresh.",
        "excluded_parts": args.exclude_part, "archives": results,
        "archive_count": len(results),
        "original_bytes": sum(r["original"]["size_bytes"] for r in results),
        "review_bytes": sum(r["review_bundle"]["size_bytes"] for r in results),
        "externalized_members": sum(r["externalized_members"] for r in results),
        "externalized_bytes": sum(r["externalized_bytes"] for r in results),
    }
    inventory["json_artifacts"] = [
        package_json_evidence(path.resolve(), args.cache.parent / "original_json", compressed=compressed)
        for compressed, paths in [(False, args.format_json), (True, args.seal_json_data)]
        for path in paths
    ]
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({key: value for key, value in inventory.items() if key != "archives"}, indent=2))


if __name__ == "__main__":
    main()
