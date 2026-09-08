import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.audit_extension_artifact_hygiene import (
    ROOT,
    artifact_hygiene_blockers,
    forbidden_artifact_blockers,
    large_file_blockers,
    prompt_jsonl_blockers,
    release_payload_pathspecs,
    untracked_extension_payload_blockers,
)
from scripts.evidence_bundles import (
    contract_path,
    file_record,
    inspect_archive,
    resolve_historical_archive,
    seal_json_data,
    seal_review_bundle,
    validate_json_data,
    validate_review_bundle,
)
from scripts.package_historical_evidence import _readable_json


def test_hygiene_audit_flags_raw_model_artifacts():
    path = ROOT / "chapter5_modern_architectures/exercises/example.safetensors"

    blockers = forbidden_artifact_blockers([path])

    assert blockers == [
        "raw artifact forbidden in extension path: "
        "chapter5_modern_architectures/exercises/example.safetensors"
    ]


def test_hygiene_audit_flags_large_extension_files(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b"0" * (5 * 1024 * 1024 + 1))

    blockers = large_file_blockers([path])

    assert blockers == [f"oversized extension file: {path} ({path.stat().st_size} bytes)"]


def test_hygiene_audit_rejects_unredacted_sensitive_prompt(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    records = [
        {
            "id": "a",
            "task": "refusal",
            "prompt": "[REDACTED]",
            "prompt_hash": "0" * 64,
            "label": "refusal_expected",
            "safe_to_display": False,
            "contains_sensitive_content": True,
            "target_token": None,
            "counterfactual_id": "b",
            "metadata": {
                "template": "t1",
                "split": "train",
                "source": "generated",
                "seed": 1,
            },
        },
        {
            "id": "b",
            "task": "refusal",
            "prompt": "unsafe raw prompt",
            "label": "refusal_expected",
            "safe_to_display": False,
            "contains_sensitive_content": True,
            "target_token": None,
            "counterfactual_id": "a",
            "metadata": {
                "template": "t2",
                "split": "validation",
                "source": "generated",
                "seed": 1,
            },
        },
        {
            "id": "c",
            "task": "refusal",
            "prompt": "safe test",
            "label": "compliance_expected",
            "safe_to_display": True,
            "contains_sensitive_content": False,
            "target_token": None,
            "counterfactual_id": "d",
            "metadata": {
                "template": "t3",
                "split": "test",
                "source": "generated",
                "seed": 1,
            },
        },
        {
            "id": "d",
            "task": "refusal",
            "prompt": "safe ood",
            "label": "compliance_expected",
            "safe_to_display": True,
            "contains_sensitive_content": False,
            "target_token": None,
            "counterfactual_id": "c",
            "metadata": {
                "template": "t4",
                "split": "ood",
                "source": "generated",
                "seed": 1,
            },
        },
    ]
    prompt_path.write_text("\n".join(json.dumps(record) for record in records))

    blockers = prompt_jsonl_blockers(prompt_path)

    assert f"{prompt_path}: unredacted sensitive prompt b" in blockers
    assert f"{prompt_path}: missing prompt_hash for sensitive prompt b" in blockers


def test_untracked_payload_check_ignores_caches_and_external_payloads():
    blockers = untracked_extension_payload_blockers(
        [
            "arena_ext/foo.py",
            "arena_ext/__pycache__/foo.pyc",
            "external/model.bin",
            ".venv/bin/python",
            "chapter7_activation_to_language/.pytest_cache/state",
        ]
    )

    assert blockers == ["required extension payload is untracked: arena_ext/foo.py"]


def test_release_payload_pathspecs_cover_required_extension_roots():
    pathspecs = set(release_payload_pathspecs())

    assert "arena_ext" in pathspecs
    assert "docs" in pathspecs
    assert "scripts" in pathspecs
    assert "tests" in pathspecs
    assert "data/generated" in pathspecs
    assert ".github/workflows/extension-quality.yml" in pathspecs
    assert "Extension-Roadmap.md" in pathspecs
    assert "uv.lock" in pathspecs
    assert any(path.startswith("chapter7_activation_to_language") for path in pathspecs)


def test_current_extension_artifact_hygiene_passes():
    assert artifact_hygiene_blockers() == []


def _archive(path, members):
    with tarfile.open(path, "w:gz") as handle:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    return path


def _provenance(tmp_path):
    source = tmp_path / "execution.json"
    source.write_text('{"status": "FAIL", "date_run": "2020-01-01"}\n')
    return {
        "kind": "native_execution_packaging",
        "source_execution": {"absolute_path": str(source), **file_record(source)},
        "measurement_claim": "Packaging only; the recorded failure and date are unchanged.",
    }


def _sealed(tmp_path):
    archive = _archive(tmp_path / "execution.tar.gz", [
        ("receipt.json", b'{"status": "FAIL", "date_run": "2020-01-01"}'),
        ("source/method.py", b"def method(x):\n    return x\n"),
    ])
    seal_review_bundle(archive, provenance=_provenance(tmp_path))
    return archive


def test_legitimate_review_bundle_is_accepted_and_failure_bytes_remain(tmp_path):
    archive = _sealed(tmp_path)
    assert forbidden_artifact_blockers([archive]) == []
    contract = validate_review_bundle(archive)
    with tarfile.open(archive) as handle:
        receipt = json.load(handle.extractfile("receipt.json"))
    assert receipt == {"status": "FAIL", "date_run": "2020-01-01"}
    assert contract["members"]["source/method.py"]["type"] == "regular"


@pytest.mark.parametrize("name", ["../x.json", "/x.json", "a/../x.json", "a//x.json",
                                 "./x.json", "a\\x.json", "C:/x.json", "a\nx.json"])
def test_bundle_rejects_unsafe_member_paths(tmp_path, name):
    archive = _archive(tmp_path / "unsafe.tar.gz", [(name, b"{}")])
    with pytest.raises(ValueError, match="unsafe member path"):
        inspect_archive(archive)


@pytest.mark.parametrize("name,data", [
    ("weights.pt", b"payload"), ("small.joblib", b"payload"), ("arrays.npz", b"payload"),
    ("small.pkl", b"payload"), ("nested.tar.gz", b"payload"),
    ("unknown.exe", b"payload"), ("forged.json", b"PK\x03\x04payload"),
    ("forged.md", b"\x80\x04payload"), ("forged.png", b"not an image"),
    ("forged.json.gz", gzip.compress(b"not JSON")),
])
def test_bundle_rejects_raw_unsupported_or_disguised_payloads(tmp_path, name, data):
    archive = _archive(tmp_path / "unsupported.tar.gz", [(name, data)])
    with pytest.raises((ValueError, UnicodeDecodeError)):
        seal_review_bundle(archive, provenance=_provenance(tmp_path))


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE,
                                        tarfile.CHRTYPE, tarfile.DIRTYPE])
def test_bundle_rejects_links_and_special_member_types(tmp_path, member_type):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("member.json")
        info.type = member_type
        info.linkname = "elsewhere"
        handle.addfile(info)
    with pytest.raises(ValueError, match="unsupported review member type"):
        inspect_archive(archive)


def test_bundle_rejects_duplicate_member_names(tmp_path):
    archive = _archive(tmp_path / "duplicate.tar.gz", [("x.json", b"{}"), ("x.json", b"{}")])
    with pytest.raises(ValueError, match="duplicate review member"):
        inspect_archive(archive)


def test_unsealed_tar_archive_is_not_exempt(tmp_path):
    archive = _archive(tmp_path / "raw.tar.gz", [("receipt.json", b"{}")])
    assert "missing review bundle contract" in forbidden_artifact_blockers([archive])[0]


def test_browser_jpeg_named_png_keeps_bytes_and_discloses_actual_format(tmp_path):
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 128, 0)).save(stream, format="JPEG")
    data = stream.getvalue()
    archive = _archive(tmp_path / "browser.tar.gz", [("browser.png", data)])
    contract = seal_review_bundle(archive, provenance=_provenance(tmp_path))
    record = contract["members"]["browser.png"]
    assert record["format"] == "jpeg"
    assert record["filename_format_mismatch"] == {"suffix": ".png", "detected_format": "jpeg"}
    assert record["sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("mutation", ["archive_hash", "member_hash", "member_size", "member_type",
                                     "missing_member", "extra_member", "bounds", "schema", "provenance"])
def test_bundle_rejects_tampered_contract(tmp_path, mutation):
    archive = _sealed(tmp_path)
    sidecar = contract_path(archive)
    contract = json.loads(sidecar.read_text())
    if mutation == "archive_hash":
        contract["archive"]["sha256"] = "0" * 64
    elif mutation.startswith("member_"):
        key = {"member_hash": "sha256", "member_size": "size_bytes", "member_type": "type"}[mutation]
        contract["members"]["receipt.json"][key] = {"sha256": "0" * 64, "size_bytes": 0, "type": "symlink"}[key]
    elif mutation == "missing_member":
        contract["members"].pop("receipt.json")
    elif mutation == "extra_member":
        contract["members"]["extra.json"] = contract["members"]["receipt.json"]
    elif mutation == "bounds":
        contract["limits"]["member_bytes"] *= 2
    elif mutation == "schema":
        contract["schema"] = "arena.review-bundle.v999"
    else:
        contract.pop("provenance")
    sidecar.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        validate_review_bundle(archive)
    assert forbidden_artifact_blockers([archive])


def test_bundle_rejects_changed_archive_bytes_even_if_it_still_opens(tmp_path):
    archive = _sealed(tmp_path)
    _archive(archive, [("receipt.json", b'{"status": "PASS"}')])
    with pytest.raises(ValueError, match="archive hash or size mismatch"):
        validate_review_bundle(archive)


def test_historical_original_is_recoverable_with_all_external_members(tmp_path):
    original = _archive(tmp_path / "original.tar.gz", [("receipt.json", b'{"status":"FAIL"}'),
                                                       ("raw.pt", b"all tensor bytes")])
    archive = _archive(tmp_path / "review.tar.gz", [("receipt.json", b'{"status":"FAIL"}')])
    provenance = {
        "kind": "historical_archive_packaging", "measurement_claim": "Packaging only.",
        "source_archive": {"absolute_path": str(original), "repository_path": "chapter16/original.tar.gz",
                           **file_record(original)},
        "source_archive_members": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for name, data in [("receipt.json", b'{"status":"FAIL"}'), ("raw.pt", b"all tensor bytes")]
        },
    }
    external = {"raw.pt": {**provenance["source_archive_members"]["raw.pt"],
                           "source_archive_member": "raw.pt"}}
    seal_review_bundle(archive, provenance=provenance, external_payloads=external)
    assert resolve_historical_archive(archive, file_record(original)["sha256"]) == original
    sidecar = contract_path(archive)
    contract = json.loads(sidecar.read_text()); contract["external_payloads"] = {}
    sidecar.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="partition mismatch"):
        validate_review_bundle(archive)


def test_sealing_requires_physical_provenance_bytes(tmp_path):
    archive = _archive(tmp_path / "review.tar.gz", [("receipt.json", b"{}")])
    provenance = _provenance(tmp_path)
    Path(provenance["source_execution"]["absolute_path"]).write_text('{"status":"PASS"}')
    with pytest.raises(ValueError, match="provenance source bytes missing or changed"):
        seal_review_bundle(archive, provenance=provenance)


def test_bounded_json_data_preserves_all_outcomes_and_detects_tampering(tmp_path):
    path = tmp_path / "outcomes.json.gz"
    values = {"results": [{"status": "FAIL", "score": 1.25}, {"status": "PASS", "score": -2.5}]}
    path.write_bytes(gzip.compress(json.dumps(values).encode()))
    original = path.read_bytes()
    seal_json_data(path, provenance=_provenance(tmp_path))
    assert validate_json_data(path)["record"]["format"] == "gzip-json"
    assert path.read_bytes() == original
    assert json.loads(gzip.decompress(path.read_bytes())) == values
    path.write_bytes(gzip.compress(b'{"results":[]}'))
    with pytest.raises(ValueError, match="content mismatch"):
        validate_json_data(path)


def test_gzip_expansion_limit_is_enforced(tmp_path, monkeypatch):
    import scripts.evidence_bundles as bundles
    monkeypatch.setattr(bundles, "MAX_DECODED_JSON_BYTES", 128)
    archive = _archive(tmp_path / "bomb.tar.gz", [("huge.json.gz", gzip.compress(b'{"a":"' + b"a" * 1024 + b'"}'))])
    with pytest.raises(ValueError, match="gzip expansion limit"):
        inspect_archive(archive)


def test_tar_container_expansion_limit_includes_metadata_and_padding(tmp_path, monkeypatch):
    import scripts.evidence_bundles as bundles
    archive = _archive(tmp_path / "container.tar.gz", [("x.json", b"{}")])
    monkeypatch.setattr(bundles, "MAX_TAR_BYTES", 4096)
    with pytest.raises(ValueError, match="tar container expansion limit"):
        inspect_archive(archive)


def test_bundle_rejects_payload_hidden_after_tar_end_marker(tmp_path):
    archive = _archive(tmp_path / "trailing.tar.gz", [("x.json", b"{}")])
    unpacked = gzip.decompress(archive.read_bytes())
    archive.write_bytes(gzip.compress(unpacked + b"PK\x03\x04hidden binary payload"))
    with pytest.raises(ValueError, match="nonzero data after tar members"):
        inspect_archive(archive)


def test_bundle_enforces_member_size_count_and_total_limits(tmp_path, monkeypatch):
    import scripts.evidence_bundles as bundles
    archive = _archive(tmp_path / "limits.tar.gz", [("x.json", b'{"x":123}'), ("y.json", b"{}")])
    with monkeypatch.context() as scoped:
        scoped.setattr(bundles, "MAX_MEMBER_BYTES", 4)
        with pytest.raises(ValueError, match="oversized review member"):
            inspect_archive(archive)
    with monkeypatch.context() as scoped:
        scoped.setattr(bundles, "MAX_MEMBERS", 1)
        with pytest.raises(ValueError, match="too many review members"):
            inspect_archive(archive)
    with monkeypatch.context() as scoped:
        scoped.setattr(bundles, "MAX_TOTAL_BYTES", 10)
        with pytest.raises(ValueError, match="review bundle expansion limit"):
            inspect_archive(archive)


def test_bounded_json_large_data_contract_is_required_for_size_exception(tmp_path):
    import base64
    import random

    path = tmp_path / "large.json.gz"
    # Incompressible, deterministic synthetic text forces the encoded file over 5 MiB.
    content = base64.b64encode(random.Random(1).randbytes(6 * 1024 * 1024)).decode()
    path.write_bytes(gzip.compress(json.dumps({"outcomes": content}).encode()))
    assert path.stat().st_size > 5 * 1024 * 1024
    assert large_file_blockers([path])
    seal_json_data(path, provenance=_provenance(tmp_path))
    assert large_file_blockers([path]) == []
    sidecar = Path(str(path) + ".data-contract.json")
    contract = json.loads(sidecar.read_text()); contract["record"]["decoded_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(contract))
    assert large_file_blockers([path])


def test_readable_json_formatting_preserves_nested_values_and_record_order():
    values = {"date": "2020-01-01", "status": "FAIL", "records": [
        {"tokens": ["é", "a\\b", "\n"], "scores": [0.0, -0.0, 1e-12, 1.23456789]},
        {"absent": None, "active": False, "nested": {"array": [[1, 2], [3, 4]]}},
    ]}
    result = _readable_json(values)
    assert json.loads(result) == values
    assert list(json.loads(result)) == list(values)
    assert '"scores": [0.0, -0.0, 1e-12, 1.23456789]' in result


def test_gradientshap_publisher_packaging_excludes_tensor_payloads(tmp_path):
    # Execute only this plumbing function's AST; never import native models or fit code.
    path = ROOT / "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/publish_study.py"
    module = ast.parse(path.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_archive_members")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    learner, native = tmp_path / "learner", tmp_path / "learner/native"
    native.mkdir(parents=True)
    (native / "small.pt").write_bytes(b"small tensor")
    (native / "small.pt.manifest.json").write_text("{}")
    members = namespace["_archive_members"](learner, native)
    assert not any(member.suffix == ".pt" for member, _ in members)
    assert (native / "small.pt.manifest.json", "native/small.pt.manifest.json") in members


def test_gradientshap_archive_publisher_seals_review_and_hashes_all_external_payloads(tmp_path):
    path = ROOT / "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/publish_study.py"
    module = ast.parse(path.read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"require", "_archive_members", "archive_evidence"}]
    learner = tmp_path / "learner"
    native = learner / "native"
    native.mkdir(parents=True)
    (learner / "execution.json").write_text(json.dumps({"native_directory": str(native)}))
    (learner / "completed_learner.ipynb").write_text('{"nbformat":4,"cells":[]}')
    for name in ("before_audit.json", "summary.json", "artifact_manifest.json"):
        (native / name).write_text("{}")
    payload_names = ("fitted.pt", "replicate_samples.pt", "timings.pt")
    manifests = {}
    for name in payload_names:
        payload = native / name
        payload.write_bytes(b"raw numerical payload " + name.encode())
        (native / (name + ".manifest.json")).write_text("{}")
        manifests[name] = {"artifact_sha256": file_record(payload)["sha256"],
                           "size_bytes": payload.stat().st_size}
    namespace = {
        "Path": Path, "Any": object, "json": json, "io": io, "tarfile": tarfile,
        "digest": lambda p: file_record(p)["sha256"],
        "manifest_path": lambda p: Path(str(p) + ".manifest.json"),
        "validate_artifact_directory": lambda p: manifests,
        "DETERMINISTIC_ARTIFACTS": payload_names[:2], "TIMING_ARTIFACT": payload_names[-1],
        "EXTERNAL_DETERMINISTIC_ARTIFACT": payload_names[1], "ARCHIVE_INLINE_LIMIT_BYTES": 0,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    archive = tmp_path / "published.tar.gz"
    result = namespace["archive_evidence"](learner, archive)
    assert set(result["external_payloads"]) == set(payload_names)
    assert all(record["archived_bytes"] is False for record in result["external_payloads"].values())
    contract = validate_review_bundle(archive)
    assert set(contract["external_payloads"]) == set(payload_names)
    assert not any(name.endswith(".pt") for name in contract["members"])
    assert forbidden_artifact_blockers([archive]) == []
