"""Bounded, byte-verified review bundles; never deserialize model or tensor payloads.

The adjacent ``.review-bundle.json`` contract is mandatory, not a suffix exemption.
Historical source receipts are immutable. A derived archive may omit raw numerical
payloads only when its provenance identifies the exact preserved original archive.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any


SCHEMA = "arena.review-bundle.v1"
DATA_SCHEMA = "arena.bounded-json-data.v1"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_TAR_BYTES = MAX_TOTAL_BYTES + 32 * 1024 * 1024
MAX_DECODED_JSON_BYTES = 64 * 1024 * 1024
MAX_TOTAL_DECODED_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_CONTRACT_BYTES = 4 * 1024 * 1024
RAW_SUFFIXES = {
    ".safetensors", ".ckpt", ".gguf", ".onnx", ".pth", ".pt", ".npz",
    ".npy", ".bin", ".zip", ".tar", ".joblib", ".pickle", ".pkl", ".mp4",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".log", ".yml", ".yaml", ".csv", ".tsv", ".txt",
    ".html", ".css", ".js", ".h", ".toml", ".lock", ".svg",
}
LIMITS = {
    "archive_bytes": MAX_ARCHIVE_BYTES,
    "member_bytes": MAX_MEMBER_BYTES,
    "total_member_bytes": MAX_TOTAL_BYTES,
    "tar_container_bytes": MAX_TAR_BYTES,
    "decoded_json_bytes": MAX_DECODED_JSON_BYTES,
    "total_decoded_bytes": MAX_TOTAL_DECODED_BYTES,
    "members": MAX_MEMBERS,
}


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def contract_path(archive: Path) -> Path:
    return Path(str(archive) + ".review-bundle.json")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_name(name: str) -> None:
    _require(isinstance(name, str) and bool(name), "empty member path")
    path = PurePosixPath(name)
    _require(
        not path.is_absolute() and "\\" not in name and ":" not in name
        and not any(ord(char) < 32 for char in name)
        and all(part not in {"", ".", ".."} for part in name.split("/"))
        and path.as_posix() == name,
        f"unsafe member path: {name!r}",
    )


def _hash_record(record: Any, label: str) -> None:
    _require(isinstance(record, dict), f"missing {label} record")
    _require(
        isinstance(record.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
        f"invalid {label} hash",
    )
    _require(type(record.get("size_bytes")) is int and record["size_bytes"] >= 0,
             f"invalid {label} size")


def _json_object(data: bytes) -> Any:
    result = json.loads(data.decode("utf-8"))
    _require(isinstance(result, (dict, list)), "JSON evidence must contain an object or array")
    return result


def inspect_member(name: str, data: bytes) -> dict[str, Any]:
    """Validate the allowed format as well as the name; gzip permits JSON only."""
    _safe_name(name)
    _require(len(data) <= MAX_MEMBER_BYTES, f"oversized review member: {name}")
    lower = name.lower()
    suffix = PurePosixPath(lower).suffix
    _require(suffix not in RAW_SUFFIXES and not lower.endswith(".tar.gz"),
             f"raw serialization or nested archive: {name}")
    _require(not data.startswith((b"PK\x03\x04", b"\x93NUMPY", b"\x80\x02", b"\x80\x03",
                                  b"\x80\x04", b"\x80\x05")),
             f"raw serialization disguised as review evidence: {name}")
    record: dict[str, Any] = {
        "type": "regular", "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if lower.endswith(".json.gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
            decoded = handle.read(MAX_DECODED_JSON_BYTES + 1)
        _require(len(decoded) <= MAX_DECODED_JSON_BYTES, f"gzip expansion limit: {name}")
        _json_object(decoded)
        record.update(format="gzip-json", decoded_size_bytes=len(decoded),
                      decoded_sha256=hashlib.sha256(decoded).hexdigest())
    elif suffix in {".json", ".ipynb"}:
        value = _json_object(data)
        if suffix == ".ipynb":
            _require(isinstance(value, dict) and isinstance(value.get("cells"), list)
                     and value.get("nbformat") == 4, f"invalid notebook: {name}")
        record["format"] = "notebook-json" if suffix == ".ipynb" else "json"
    elif suffix == ".jsonl":
        lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
        _require(bool(lines), f"empty JSONL evidence: {name}")
        for line in lines:
            _json_object(line.encode("utf-8"))
        record["format"] = "jsonl"
    elif suffix in {".png", ".jpg", ".jpeg"}:
        from PIL import Image

        is_png = data.startswith(b"\x89PNG\r\n\x1a\n") and data.endswith(b"IEND\xaeB`\x82")
        is_jpeg = data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9")
        _require(is_png or is_jpeg, f"invalid PNG/JPEG evidence: {name}")
        with Image.open(io.BytesIO(data)) as picture:
            _require(picture.format in {"PNG", "JPEG"}, f"unsupported image content: {name}")
            _require(picture.width * picture.height <= 80 * 1024 * 1024,
                     f"oversized review image dimensions: {name}")
            picture.verify()
        record["format"] = "png" if is_png else "jpeg"
        if (suffix == ".png") != is_png:
            # Some browser screenshot providers return JPEG despite a caller's .png name.
            # Preserve reviewed bytes and disclose the actual format instead of re-encoding.
            record["filename_format_mismatch"] = {"suffix": suffix, "detected_format": record["format"]}
    else:
        _require(suffix in TEXT_SUFFIXES
                 or (not suffix and PurePosixPath(lower).name.endswith("license")),
                 f"unsupported review member format: {name}")
        decoded = data.decode("utf-8")
        _require("\0" not in decoded, f"binary data in text evidence: {name}")
        record["format"] = "utf8-text"
    return record


def inspect_archive(archive: Path) -> dict[str, dict[str, Any]]:
    archive = Path(archive)
    _require(archive.is_file() and not archive.is_symlink(), "bundle is missing or a symlink")
    _require(archive.name.endswith(".tar.gz"), "unsupported review archive format")
    _require(archive.stat().st_size <= MAX_ARCHIVE_BYTES, "review archive exceeds byte limit")
    records: dict[str, dict[str, Any]] = {}
    total = decoded_total = 0
    # Bound the entire expanded container, including PAX metadata and padding, before
    # tarfile can honor a hostile extended-header length. Also inspect trailing bytes.
    with gzip.open(archive, "rb") as compressed:
        tar_bytes = compressed.read(MAX_TAR_BYTES + 1)
    _require(len(tar_bytes) <= MAX_TAR_BYTES, "tar container expansion limit")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as handle:
        for member in handle:
            _safe_name(member.name)
            _require(member.name not in records, f"duplicate review member: {member.name}")
            _require(member.isfile() and not member.issparse(),
                     f"unsupported review member type: {member.name}")
            _require(0 <= member.size <= MAX_MEMBER_BYTES, f"oversized review member: {member.name}")
            _require(len(records) < MAX_MEMBERS, "too many review members")
            total += member.size
            _require(total <= MAX_TOTAL_BYTES, "review bundle expansion limit")
            stream = handle.extractfile(member)
            _require(stream is not None, f"unreadable review member: {member.name}")
            data = stream.read(MAX_MEMBER_BYTES + 1)
            _require(len(data) == member.size, f"review member byte count changed: {member.name}")
            record = inspect_member(member.name, data)
            decoded_total += record.get("decoded_size_bytes", member.size)
            _require(decoded_total <= MAX_TOTAL_DECODED_BYTES, "decoded bundle expansion limit")
            records[member.name] = record
        _require(not any(tar_bytes[handle.offset:]), "nonzero data after tar members")
    _require(bool(records), "empty review bundle")
    return records


def _validate_provenance(provenance: Any) -> None:
    _require(isinstance(provenance, dict), "missing explicit bundle provenance")
    kind = provenance.get("kind")
    _require(kind in {"historical_archive_packaging", "historical_data_packaging", "native_execution_packaging"},
             "unsupported bundle provenance kind")
    _require(isinstance(provenance.get("measurement_claim"), str)
             and bool(provenance["measurement_claim"].strip()), "missing measurement boundary")
    field = {"historical_archive_packaging": "source_archive",
             "historical_data_packaging": "source_data",
             "native_execution_packaging": "source_execution"}[kind]
    source = provenance.get(field)
    _hash_record(source, field)
    _require(isinstance(source.get("absolute_path"), str)
             and Path(source["absolute_path"]).is_absolute(), f"missing {field} locator")
    if kind.startswith("historical_"):
        _require(isinstance(source.get("repository_path"), str), "missing original archive path")
        _safe_name(source["repository_path"])


def _verify_provenance_source(provenance: dict) -> None:
    field = {"historical_archive_packaging": "source_archive",
             "historical_data_packaging": "source_data",
             "native_execution_packaging": "source_execution"}[provenance["kind"]]
    source = provenance[field]
    path = Path(source["absolute_path"])
    _require(path.is_file() and file_record(path) == {
        "sha256": source["sha256"], "size_bytes": source["size_bytes"]
    }, f"provenance source bytes missing or changed: {field}")
    if field == "source_archive":
        observed = {}
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                _safe_name(member.name)
                _require(member.isfile() and not member.issparse() and member.name not in observed,
                         "unsupported or duplicate original archive member")
                stream = archive.extractfile(member)
                _require(stream is not None, "unreadable original archive member")
                observed[member.name] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                                         "size_bytes": member.size}
        _require(observed == provenance.get("source_archive_members"),
                 "original archive member provenance mismatch")


def seal_review_bundle(
    archive: Path, *, provenance: dict[str, Any], external_payloads: dict | None = None
) -> dict[str, Any]:
    """Inspect all archive bytes and record a bounded manifest without changing them."""
    archive = Path(archive)
    _validate_provenance(provenance)
    _verify_provenance_source(provenance)
    members = inspect_archive(archive)
    payloads = {}
    for name, item in (external_payloads or {}).items():
        _safe_name(name)
        checksum = item.get("sha256", item.get("artifact_sha256"))
        record = {**item, "sha256": checksum}
        _hash_record(record, f"external payload {name}")
        if record.get("absolute_path"):
            source = Path(record["absolute_path"])
            _require(source.is_file() and file_record(source) == {
                "sha256": record["sha256"], "size_bytes": record["size_bytes"]
            }, f"external payload missing or changed: {name}")
        payloads[name] = record
    contract = {
        "schema": SCHEMA, "archive": {"name": archive.name, **file_record(archive)},
        "limits": LIMITS, "provenance": provenance, "members": members,
        "external_payloads": payloads,
        "member_count": len(members),
        "total_member_bytes": sum(item["size_bytes"] for item in members.values()),
    }
    contract_path(archive).write_text(json.dumps(contract, indent=2) + "\n")
    validate_review_bundle(archive)
    return contract


def validate_review_bundle(archive: Path) -> dict[str, Any]:
    archive = Path(archive)
    sidecar = contract_path(archive)
    _require(sidecar.is_file() and not sidecar.is_symlink(), "missing review bundle contract")
    _require(sidecar.stat().st_size <= MAX_CONTRACT_BYTES, "oversized review bundle contract")
    contract = json.loads(sidecar.read_text())
    _require(isinstance(contract, dict) and contract.get("schema") == SCHEMA,
             "unsupported review bundle schema")
    _require(contract.get("limits") == LIMITS, "review bundle bounds were changed")
    _validate_provenance(contract.get("provenance"))
    expected = contract.get("archive", {})
    _hash_record(expected, "archive")
    _require(expected == {"name": archive.name, **file_record(archive)}, "archive hash or size mismatch")
    members = inspect_archive(archive)
    _require(contract.get("members") == members, "member paths, types, sizes, formats or hashes mismatch")
    _require(contract.get("member_count") == len(members), "member count mismatch")
    _require(contract.get("total_member_bytes") == sum(r["size_bytes"] for r in members.values()),
             "total member size mismatch")
    payloads = contract.get("external_payloads")
    _require(isinstance(payloads, dict), "missing external payload inventory")
    for name, record in payloads.items():
        _safe_name(name)
        _require(name not in members, f"external payload also present in review bundle: {name}")
        _hash_record(record, f"external payload {name}")
        _require(bool(record.get("absolute_path") or record.get("source_archive_member")),
                 f"missing external payload locator: {name}")
        if record.get("absolute_path"):
            _require(isinstance(record["absolute_path"], str) and Path(record["absolute_path"]).is_absolute(),
                     f"external payload locator must be absolute: {name}")
        if record.get("source_archive_member"):
            _require(record["source_archive_member"] == name, "external archive member locator mismatch")
        if "source_archive_sha256" in record:
            _require(record["source_archive_sha256"] == contract["provenance"].get("source_archive", {}).get("sha256"),
                     "external source archive hash mismatch")
    if contract["provenance"]["kind"] == "historical_archive_packaging":
        original_members = contract["provenance"].get("source_archive_members")
        _require(isinstance(original_members, dict) and bool(original_members), "missing original member inventory")
        _require(set(original_members) == set(members) | set(payloads), "original member partition mismatch")
        for name, record in original_members.items():
            _safe_name(name)
            _hash_record(record, f"original member {name}")
            retained = members.get(name, payloads.get(name))
            _require(all(record[key] == retained[key] for key in ("sha256", "size_bytes")),
                     f"historical member bytes changed: {name}")
    return contract


def resolve_historical_archive(archive: Path, expected_sha256: str) -> Path:
    """Resolve an unchanged historical receipt to its exact bytes, never new measurements."""
    archive = Path(archive)
    if archive.is_file() and sha256(archive) == expected_sha256:
        return archive
    contract = validate_review_bundle(archive)
    source = contract["provenance"].get("source_archive", {})
    _require(source.get("sha256") == expected_sha256, "receipt does not identify the preserved original")
    original = Path(source["absolute_path"])
    _require(original.is_file() and file_record(original) == {
        "sha256": source["sha256"], "size_bytes": source["size_bytes"]
    }, "original archive is unavailable or changed; restore the recorded Git blob or external cache")
    return original


def seal_json_data(path: Path, *, provenance: dict[str, Any]) -> dict[str, Any]:
    """Keep all JSON outcomes in a bounded gzip data artifact, with its decoded hash."""
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_MEMBER_BYTES,
             "JSON data missing or oversized")
    record = inspect_member(path.name, path.read_bytes())
    _require(record["format"] == "gzip-json", "bounded data artifact must be gzip JSON")
    _validate_provenance(provenance)
    _verify_provenance_source(provenance)
    contract = {"schema": DATA_SCHEMA, "name": path.name, "record": record,
                "provenance": provenance,
                "limits": {"encoded_bytes": MAX_MEMBER_BYTES, "decoded_bytes": MAX_DECODED_JSON_BYTES}}
    Path(str(path) + ".data-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def validate_json_data(path: Path) -> dict[str, Any]:
    path = Path(path)
    sidecar = Path(str(path) + ".data-contract.json")
    _require(sidecar.is_file() and not sidecar.is_symlink(), "missing bounded JSON data contract")
    _require(sidecar.stat().st_size <= MAX_CONTRACT_BYTES, "oversized JSON data contract")
    contract = json.loads(sidecar.read_text())
    _require(isinstance(contract, dict) and contract.get("schema") == DATA_SCHEMA,
             "unsupported bounded JSON data schema")
    _require(contract.get("name") == path.name, "JSON data name mismatch")
    _require(contract.get("limits") == {"encoded_bytes": MAX_MEMBER_BYTES,
                                        "decoded_bytes": MAX_DECODED_JSON_BYTES}, "JSON data bounds changed")
    _validate_provenance(contract.get("provenance"))
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_MEMBER_BYTES,
             "JSON data missing or oversized")
    _require(contract.get("record") == inspect_member(path.name, path.read_bytes()),
             "JSON data hash, size, format or decoded content mismatch")
    return contract
