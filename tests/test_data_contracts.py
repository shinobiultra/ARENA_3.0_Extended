import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/generated/refusal_proxy_prompts_v1"


def test_refusal_proxy_prompt_artifact_matches_jsonl_contract():
    from arena_ext.data_contracts import load_jsonl_records, validate_prompt_records

    records = load_jsonl_records(DATASET_DIR / "prompts.jsonl")
    report = validate_prompt_records(records)

    assert report.total_records == 12
    assert report.valid_records == 12
    assert set(report.splits) == {"train", "validation", "test", "ood"}
    assert report.valid


def test_prompt_contract_rejects_unredacted_sensitive_records():
    from arena_ext.data_contracts import load_jsonl_records, validate_prompt_records

    records = load_jsonl_records(DATASET_DIR / "prompts.jsonl")
    records[-2] = {
        **records[-2],
        "prompt": "unsafe placeholder should have been redacted",
    }
    report = validate_prompt_records(records)

    assert not report.valid
    assert report.unsafe_display_violations == ("refusal_0006",)


def test_dataset_manifest_has_required_generation_metadata():
    from arena_ext.data_contracts import validate_dataset_manifest

    manifest = yaml.safe_load((DATASET_DIR / "manifest.yml").read_text())
    report = validate_dataset_manifest(manifest)

    assert report.has_seed
    assert report.has_split_policy
    assert report.has_ood_split
    assert report.has_counterfactual_mapping
    assert report.has_label_function
    assert report.has_example_preview
    assert report.has_required_controls
    assert report.valid


def test_prompt_schema_names_required_fields():
    schema = json.loads((DATASET_DIR / "prompt_schema.json").read_text())

    assert "safe_to_display" in schema["required"]
    assert "contains_sensitive_content" in schema["required"]
    assert "counterfactual_id" in schema["required"]
    assert set(schema["properties"]["metadata"]["required"]) == {
        "template",
        "split",
        "source",
        "seed",
    }
