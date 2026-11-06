"""Artifact-hygiene audit: raw weights, 5 MB limit, prompt data contracts, .gitignore."""

import json

from scripts.audit_extension_artifact_hygiene import (
    ROOT,
    artifact_hygiene_blockers,
    forbidden_artifact_blockers,
    iter_extension_files,
    large_file_blockers,
    prompt_jsonl_blockers,
    required_gitignore_blockers,
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


def test_ignored_local_caches_are_not_scanned():
    files = {path.relative_to(ROOT).as_posix() for path in iter_extension_files()}
    assert not any(path.startswith("docs/mastery/evidence/") for path in files)
    assert any(path.startswith("chapter7_activation_to_language/") for path in files)


def test_gitignore_keeps_caches_and_weights_out():
    assert required_gitignore_blockers() == []


def test_current_extension_artifact_hygiene_passes():
    assert artifact_hygiene_blockers() == []
