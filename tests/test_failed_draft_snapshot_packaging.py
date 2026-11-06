import copy
import io
import json
from pathlib import Path
import tarfile
import shutil
import pytest
from scripts import package_failed_draft_snapshots as p
from scripts import evidence_bundles as b

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[False, True], ids=["before-publication", "after-publication"])
def candidate(tmp_path, request):
    source = ROOT / p.PREFIX / "pre_publication_refresh_20261019.tar.gz"
    if not request.param:
        return p.prepare(source, tmp_path / "candidate")[0]
    repo = tmp_path / "repository"
    published = repo / p.PREFIX / source.name
    published.parent.mkdir(parents=True)
    shutil.copyfile(p.resolve_original(source), published)
    review, _ = p.prepare(published, tmp_path / "first-candidate", repository_root=repo)
    p.verify_for_publication(review, repo)
    shutil.copyfile(review, published)
    shutil.copyfile(b.contract_path(review), b.contract_path(published))
    # Published filename changes, so its strict sidecar must use that actual name.
    provenance = json.loads(b.contract_path(review).read_text())["provenance"]
    b.seal_review_bundle(published, provenance=provenance)
    original = p.resolve_original(published)
    assert original == repo / ".arena_artifacts" / "failed-draft-snapshot-originals" / (
        p.SOURCES[source.name]["sha256"] + ".tar.gz"
    )
    new_review, _ = p.prepare(published, tmp_path / "second-candidate", repository_root=repo)
    p.verify_for_publication(new_review, repo)
    return new_review


def test_lossless_wrapper(candidate):
    r = p.verify_wrapper(candidate)
    assert r["members_covered"] == r["members_preserved_exactly"] and r["bijective"]
    assert len(r["renamed_failures"]) == 2 and r["accepted"] is False


@pytest.mark.parametrize("mutation", ["drop", "rename", "change"])
def test_review_member_mutation_rejected(candidate, mutation):
    provenance = json.loads(b.contract_path(candidate).read_text())["provenance"]
    members = p.original_members(candidate)
    names = list(members)
    name = next(n for n in names if n.endswith(".py"))
    if mutation == "drop":
        del members[name]
    elif mutation == "rename":
        members[name + ".txt"] = members.pop(name)
    else:
        members[name] += b"altered"
    with tarfile.open(candidate, "w:gz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    b.seal_review_bundle(candidate, provenance=provenance)
    with pytest.raises(ValueError):
        p.verify_wrapper(candidate)


def test_mapping_tamper_even_after_strict_reseal_rejected(candidate):
    c = json.loads(b.contract_path(candidate).read_text())
    prov = c["provenance"]
    first = next(iter(prov["member_mapping"]))
    prov["member_mapping"][first]["category"] = "empty-failed-json-stdout"
    b.seal_review_bundle(candidate, provenance=prov)
    with pytest.raises(ValueError, match="mapping"):
        p.verify_wrapper(candidate)


def test_missing_original_rejected(candidate):
    c = json.loads(b.contract_path(candidate).read_text())
    Path(c["provenance"]["source_data"]["absolute_path"]).unlink()
    with pytest.raises(ValueError, match="original missing"):
        p.verify_wrapper(candidate)


def test_undeclared_malformed_json_rejected():
    name = "pre_publication_refresh_20261019.tar.gz"
    members = p.original_members(p.resolve_original(ROOT / p.PREFIX / name))
    members["unexpected.json"] = b""
    with pytest.raises(json.JSONDecodeError):
        p.mapping_for(name, members)


def test_known_failed_json_cannot_hide_nonempty_or_valid_json():
    name = "pre_publication_refresh_20261019.tar.gz"
    members = p.original_members(p.resolve_original(ROOT / p.PREFIX / name))
    for value in (b"{bad", b"{}"):
        members[p.PREFIX + "strict_verifier_final.json"] = value
        with pytest.raises(ValueError, match="exactly empty"):
            p.mapping_for(name, members)


def test_publication_requires_durable_original(candidate, tmp_path):
    with pytest.raises(ValueError, match="durable original"):
        p.verify_for_publication(candidate, tmp_path / "different-repository")


def test_resolver_rejects_changed_wrapper_and_wrong_original(candidate):
    with pytest.raises(ValueError):
        p.resolve_original(candidate, original_name="pre_metadata_restore_refresh_20261019.tar.gz")
    candidate.write_bytes(candidate.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        p.resolve_original(candidate, original_name="pre_publication_refresh_20261019.tar.gz")
