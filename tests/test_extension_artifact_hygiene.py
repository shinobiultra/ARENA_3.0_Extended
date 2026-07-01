import json
from pathlib import Path

from scripts.audit_extension_artifact_hygiene import (
    ROOT,
    artifact_hygiene_blockers,
    forbidden_artifact_blockers,
    large_file_blockers,
    prompt_jsonl_blockers,
    release_payload_pathspecs,
    untracked_extension_payload_blockers,
)


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
