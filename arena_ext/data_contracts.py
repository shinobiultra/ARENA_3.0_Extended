"""Prompt and generated-dataset contract helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_PROMPT_FIELDS = {
    "id",
    "task",
    "prompt",
    "label",
    "safe_to_display",
    "contains_sensitive_content",
    "target_token",
    "counterfactual_id",
    "metadata",
}
REQUIRED_METADATA_FIELDS = {"template", "split", "source", "seed"}
REQUIRED_SPLITS = {"train", "validation", "test", "ood"}
REQUIRED_DATASET_CONTROLS = {
    "label_permutation",
    "template_permutation",
    "spurious_correlation_split",
    "counterfactual_split",
    "random_input_split",
}
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PromptDatasetValidationReport:
    total_records: int
    valid_records: int
    splits: tuple[str, ...]
    missing_fields: tuple[str, ...]
    missing_metadata_fields: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    missing_counterfactuals: tuple[str, ...]
    unsafe_display_violations: tuple[str, ...]
    missing_prompt_hashes: tuple[str, ...]
    valid: bool


@dataclass(frozen=True)
class DatasetManifestValidationReport:
    has_seed: bool
    has_schema_path: bool
    has_split_policy: bool
    has_ood_split: bool
    has_counterfactual_mapping: bool
    has_label_function: bool
    has_example_preview: bool
    has_required_controls: bool
    missing_controls: tuple[str, ...]
    valid: bool


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from `path` with one object per nonblank line."""

    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def validate_prompt_records(
    records: list[dict[str, Any]],
) -> PromptDatasetValidationReport:
    """Validate prompt JSONL records against the roadmap prompt contract."""

    ids: list[str] = []
    splits: set[str] = set()
    missing_fields: list[str] = []
    missing_metadata_fields: list[str] = []
    unsafe_display_violations: list[str] = []
    missing_prompt_hashes: list[str] = []
    valid_records = 0

    for index, record in enumerate(records):
        record_id = str(record.get("id", f"record_{index}"))
        ids.append(record_id)
        missing = sorted(REQUIRED_PROMPT_FIELDS - set(record))
        missing_fields.extend(f"{record_id}:{field}" for field in missing)
        metadata = record.get("metadata", {})
        metadata_missing: list[str] = []
        if isinstance(metadata, dict):
            metadata_missing = sorted(REQUIRED_METADATA_FIELDS - set(metadata))
            missing_metadata_fields.extend(
                f"{record_id}:{field}" for field in metadata_missing
            )
            split = metadata.get("split")
            if isinstance(split, str):
                splits.add(split)
        else:
            missing_metadata_fields.append(f"{record_id}:metadata_not_object")

        safe_to_display = record.get("safe_to_display")
        sensitive = record.get("contains_sensitive_content")
        prompt = record.get("prompt")
        prompt_hash = record.get("prompt_hash")
        if (safe_to_display is False or sensitive is True) and prompt != "[REDACTED]":
            unsafe_display_violations.append(record_id)
        if safe_to_display is False or sensitive is True:
            if not isinstance(prompt_hash, str) or HEX_64.fullmatch(prompt_hash) is None:
                missing_prompt_hashes.append(record_id)

        if not missing and not metadata_missing:
            valid_records += 1

    id_counts = {record_id: ids.count(record_id) for record_id in ids}
    duplicates = tuple(sorted(key for key, count in id_counts.items() if count > 1))
    id_set = set(ids)
    missing_counterfactuals = tuple(
        sorted(
            str(record.get("id"))
            for record in records
            if record.get("counterfactual_id") not in id_set
        )
    )
    missing_splits = REQUIRED_SPLITS - splits
    missing_fields.extend(f"split:{split}" for split in sorted(missing_splits))
    valid = not (
        missing_fields
        or missing_metadata_fields
        or duplicates
        or missing_counterfactuals
        or unsafe_display_violations
        or missing_prompt_hashes
    )
    return PromptDatasetValidationReport(
        total_records=len(records),
        valid_records=valid_records,
        splits=tuple(sorted(splits)),
        missing_fields=tuple(missing_fields),
        missing_metadata_fields=tuple(missing_metadata_fields),
        duplicate_ids=duplicates,
        missing_counterfactuals=missing_counterfactuals,
        unsafe_display_violations=tuple(unsafe_display_violations),
        missing_prompt_hashes=tuple(missing_prompt_hashes),
        valid=valid,
    )


def validate_dataset_manifest(
    manifest: dict[str, Any],
) -> DatasetManifestValidationReport:
    """Validate generated-dataset metadata required by the roadmap."""

    split_policy = manifest.get("split_policy", {})
    controls = set(manifest.get("controls", []))
    missing_controls = tuple(sorted(REQUIRED_DATASET_CONTROLS - controls))
    has_split_policy = isinstance(split_policy, dict) and all(
        key in split_policy
        for key in ("train", "validation", "test")
    )
    report = DatasetManifestValidationReport(
        has_seed="seed" in manifest,
        has_schema_path=bool(manifest.get("schema_path")),
        has_split_policy=has_split_policy,
        has_ood_split=bool(manifest.get("ood_split")),
        has_counterfactual_mapping=bool(manifest.get("counterfactual_mapping")),
        has_label_function=bool(manifest.get("label_function")),
        has_example_preview=bool(manifest.get("example_preview")),
        has_required_controls=not missing_controls,
        missing_controls=missing_controls,
        valid=False,
    )
    valid = (
        report.has_seed
        and report.has_schema_path
        and report.has_split_policy
        and report.has_ood_split
        and report.has_counterfactual_mapping
        and report.has_label_function
        and report.has_example_preview
        and report.has_required_controls
    )
    return DatasetManifestValidationReport(
        has_seed=report.has_seed,
        has_schema_path=report.has_schema_path,
        has_split_policy=report.has_split_policy,
        has_ood_split=report.has_ood_split,
        has_counterfactual_mapping=report.has_counterfactual_mapping,
        has_label_function=report.has_label_function,
        has_example_preview=report.has_example_preview,
        has_required_controls=report.has_required_controls,
        missing_controls=report.missing_controls,
        valid=valid,
    )
