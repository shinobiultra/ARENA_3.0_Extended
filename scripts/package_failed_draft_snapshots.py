"""Lossless review wrappers for two known failed draft-output snapshots.

The originals remain unchanged. Historical-data packaging describes artifact
bytes, not successful execution. Malformed failed stdout stays labelled text.
"""

from __future__ import annotations
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from scripts import evidence_bundles as bundles

PREFIX = "docs/mastery/evidence/provenance_inventory_20261019/draft_contracts/"
SOURCES = {
    "pre_metadata_restore_refresh_20261019.tar.gz": {
        "sha256": "868a60ff165782f8f6ac710b3ad87ec64d7c4ba609194c13be7790548b748cfb",
        "size_bytes": 74901,
        "exceptions": {
            PREFIX + "strict_verifier_refreshed.stderr": (
                "stderr-text",
                "strict_verifier_refreshed.stderr.txt",
            )
        },
    },
    "pre_publication_refresh_20261019.tar.gz": {
        "sha256": "85f9fbce85577ebbd017fdd305276fb01ddd0f0a59d03b5d9ccc1b071d07c4c9",
        "size_bytes": 73915,
        "exceptions": {
            PREFIX + "strict_verifier_final.stderr": (
                "stderr-text",
                "strict_verifier_final.stderr.txt",
            ),
            PREFIX + "strict_verifier_final.json": (
                "empty-failed-json-stdout",
                "strict_verifier_final.failed-json.txt",
            ),
        },
    },
}
CLAIM = (
    "Lossless packaging only of draft artifact snapshots, including failed/malformed outputs. No scientific execution, numerical validation, human/course "
    "acceptance or successful-run claim."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def content_record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def original_members(path: Path) -> dict[str, bytes]:
    """Bounded regular-file read, without attempting to parse known failed JSON."""
    require(path.is_file() and not path.is_symlink(), "original missing or symlink")
    require(path.stat().st_size <= bundles.MAX_ARCHIVE_BYTES, "original oversized")
    with gzip.open(path, "rb") as f:
        expanded = f.read(bundles.MAX_TAR_BYTES + 1)
    require(len(expanded) <= bundles.MAX_TAR_BYTES, "original expansion limit")
    members = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as t:
        for m in t:
            bundles._safe_name(m.name)
            require(
                m.isfile() and not m.issparse() and m.name not in members,
                "unsafe/duplicate original member",
            )
            require(
                len(members) < bundles.MAX_MEMBERS and 0 <= m.size <= bundles.MAX_MEMBER_BYTES,
                "original member limit",
            )
            b = t.extractfile(m).read(bundles.MAX_MEMBER_BYTES + 1)
            require(len(b) == m.size, "original member size mismatch")
            total += len(b)
            require(total <= bundles.MAX_TOTAL_BYTES, "original total limit")
            members[m.name] = b
        require(not any(expanded[t.offset :]), "nonzero original trailing bytes")
    require(bool(members), "empty original")
    return members


def mapping_for(name: str, members: dict[str, bytes]) -> dict:
    require(name in SOURCES, "unreviewed source archive")
    exceptions = SOURCES[name]["exceptions"]
    require(set(exceptions) <= set(members), "missing declared failed output")
    mapping = {}
    for original, data in members.items():
        if original in exceptions:
            category, newbase = exceptions[original]
            review = str(Path(original).with_name(newbase))
            if category == "empty-failed-json-stdout":
                require(data == b"", "known failed JSON must remain exactly empty")
                try:
                    json.loads(data)
                except json.JSONDecodeError:
                    pass
                else:
                    raise ValueError("failed JSON unexpectedly valid")
            else:
                text = data.decode("utf-8")
                require("\0" not in text, "stderr binary data")
            bundles.inspect_member(review, data)
        else:
            # Invalid JSON outside the exact allowlist fails, never becomes text.
            bundles.inspect_member(original, data)
            review = original
            category = "unchanged-valid-review-artifact"
        mapping[original] = {"review_name": review, **content_record(data), "category": category}
    require(
        len({r["review_name"] for r in mapping.values()}) == len(mapping),
        "nonbijective review names",
    )
    return mapping


def verify_wrapper(review: Path) -> dict:
    contract = bundles.validate_review_bundle(review)
    p = contract["provenance"]
    require(
        set(p)
        == {
            "kind",
            "measurement_claim",
            "source_data",
            "original_archive_name",
            "member_mapping",
            "accepted",
            "scientific_execution",
            "all_original_members_retained",
        },
        "unknown provenance fields",
    )
    require(
        p["kind"] == "historical_data_packaging" and p["measurement_claim"] == CLAIM,
        "wrong packaging semantics",
    )
    require(
        p["accepted"] is False
        and p["scientific_execution"] is False
        and p["all_original_members_retained"] is True,
        "false acceptance or retention claim",
    )
    name = p["original_archive_name"]
    require(name in SOURCES, "wrong original identity")
    expected = SOURCES[name]
    source = p["source_data"]
    require(
        set(source) == {"absolute_path", "repository_path", "sha256", "size_bytes"},
        "unknown original locator fields",
    )
    require(source["repository_path"] == PREFIX + name, "wrong original repository path")
    require(
        {k: source[k] for k in ("sha256", "size_bytes")}
        == {k: expected[k] for k in ("sha256", "size_bytes")},
        "wrong original hash/size",
    )
    original = Path(source["absolute_path"])
    require(original.is_absolute(), "original locator must be absolute")
    require(
        original.is_file()
        and not original.is_symlink()
        and bundles.file_record(original) == {k: expected[k] for k in ("sha256", "size_bytes")},
        "original missing or changed",
    )
    originals = original_members(original)
    mapping = mapping_for(name, originals)
    require(p["member_mapping"] == mapping, "mapping altered or undeclared malformed member")
    require(contract["external_payloads"] == {}, "externalization is forbidden")
    reviewed = original_members(review)
    require(
        set(reviewed) == {r["review_name"] for r in mapping.values()},
        "dropped, added or renamed review member",
    )
    for old, record in mapping.items():
        require(reviewed[record["review_name"]] == originals[old], "review member bytes changed")
    return {
        "original_archive": name,
        "original": bundles.file_record(original),
        "review": bundles.file_record(review),
        "members_covered": len(mapping),
        "members_preserved_exactly": len(mapping),
        "bijective": True,
        "accepted": False,
        "scientific_execution": False,
        "renamed_failures": {
            old: r
            for old, r in mapping.items()
            if r["category"] != "unchanged-valid-review-artifact"
        },
    }


def resolve_original(source: Path, *, original_name: str | None = None) -> Path:
    """Accept only a pinned original or a fully verified wrapper for that original."""
    name = original_name or source.name
    require(name in SOURCES, "unreviewed source identity")
    expected = {k: SOURCES[name][k] for k in ("sha256", "size_bytes")}
    require(source.is_file() and not source.is_symlink(), "source missing or symlink")
    if bundles.file_record(source) == expected:
        return source
    result = verify_wrapper(source)
    require(result["original_archive"] == name, "wrapper resolves different pinned original")
    contract = json.loads(bundles.contract_path(source).read_text())
    original = Path(contract["provenance"]["source_data"]["absolute_path"])
    require(bundles.file_record(original) == expected, "resolved original changed")
    return original


def verify_for_publication(review: Path, repository_root: Path) -> dict:
    """Require durable repository-local recovery before root publishes a wrapper."""
    result = verify_wrapper(review)
    contract = json.loads(bundles.contract_path(review).read_text())
    source = Path(contract["provenance"]["source_data"]["absolute_path"])
    expected = (
        repository_root.resolve()
        / ".arena_artifacts"
        / "failed-draft-snapshot-originals"
        / (result["original"]["sha256"] + ".tar.gz")
    )
    require(
        source == expected and not any(p.is_symlink() for p in (source, *source.parents)),
        "publication requires durable original store",
    )
    return result


def prepare(
    source: Path, workspace: Path, *, repository_root: Path | None = None
) -> tuple[Path, dict]:
    """Write only to a separate temporary workspace; preserve before wrapping."""
    name = source.name
    require(name in SOURCES, "unreviewed source")
    expected = {k: SOURCES[name][k] for k in ("sha256", "size_bytes")}
    published_before = bundles.file_record(source)
    raw_source = resolve_original(source, original_name=name)
    workspace = workspace.resolve()
    require(workspace.is_relative_to(Path("/tmp")), "candidates must remain temporary")
    workspace.mkdir(parents=True, exist_ok=True)
    store = (
        (repository_root.resolve() / ".arena_artifacts" / "failed-draft-snapshot-originals")
        if repository_root is not None
        else workspace / "originals"
    )
    original = store / (expected["sha256"] + ".tar.gz")
    original.parent.mkdir(parents=True, exist_ok=True)
    require(
        not any(p.is_symlink() for p in (original, *original.parents)),
        "original store may not contain symlinks",
    )
    if not original.exists():
        shutil.copyfile(raw_source, original)
    else:
        require(bundles.file_record(original) == expected, "preserved copy changed")
    require(bundles.file_record(original) == expected, "independent preservation failed")
    members = original_members(original)
    mapping = mapping_for(name, members)
    review = workspace / (name.removesuffix(".tar.gz") + ".lossless-review.tar.gz")
    require(not review.exists(), "candidate already exists")
    with review.open("wb") as f:
        with gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as t:
                for old, entry in sorted(mapping.items()):
                    info = tarfile.TarInfo(entry["review_name"])
                    info.size = len(members[old])
                    info.mtime = 0
                    info.mode = 0o644
                    t.addfile(info, io.BytesIO(members[old]))
    provenance = {
        "kind": "historical_data_packaging",
        "measurement_claim": CLAIM,
        "source_data": {
            "absolute_path": str(original),
            "repository_path": PREFIX + name,
            **expected,
        },
        "original_archive_name": name,
        "member_mapping": mapping,
        "accepted": False,
        "scientific_execution": False,
        "all_original_members_retained": True,
    }
    bundles.seal_review_bundle(review, provenance=provenance)
    receipt = verify_wrapper(review)
    require(bundles.file_record(source) == published_before, "published source path changed")
    return review, receipt
