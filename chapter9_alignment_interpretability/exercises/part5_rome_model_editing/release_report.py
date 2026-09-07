"""Section-local inputs for future parent-owned release report registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena_ext.verification import build_report_input_manifest


ROOT = Path(__file__).resolve().parents[3]
SECTION = Path(__file__).resolve().parent
PAGE = (
    ROOT
    / "chapter9_alignment_interpretability"
    / "instructions"
    / "pages"
    / "05_[9.5]_ROME_Model_Editing.md"
)
ASSETS = (
    ROOT
    / "chapter9_alignment_interpretability"
    / "instructions"
    / "assets"
)

REPORT_INPUT_PATHS = (
    SECTION / "release_report.py",
    SECTION / "solutions.py",
    SECTION / "tests.py",
    SECTION / "build_notebooks.py",
    SECTION / "README.md",
    SECTION / "artifacts.lock.yml",
    SECTION / "verification_report.schema.json",
    SECTION / "9.5_ROME_Model_Editing_exercises.ipynb",
    SECTION / "9.5_ROME_Model_Editing_solutions.ipynb",
    SECTION / "expected_outputs" / "README.md",
    SECTION / "expected_outputs" / "smoke_test.json",
    SECTION / "expected_outputs" / "reference_metrics.json",
    PAGE,
    ASSETS / "rome_rank_one_toy_oracle.png",
    ASSETS / "rome_model_editing_signature.png",
)


def release_report_input_paths() -> list[Path]:
    """Return the complete owned input set, failing on absent or out-of-scope paths."""
    allowed_roots = (SECTION.resolve(), PAGE.parent.resolve(), ASSETS.resolve())
    paths = [path.resolve() for path in REPORT_INPUT_PATHS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing 9.5 release inputs: {missing}")
    outside_scope = [
        str(path)
        for path in paths
        if not any(path.is_relative_to(allowed_root) for allowed_root in allowed_roots)
    ]
    if outside_scope:
        raise ValueError(f"9.5 release inputs escape the owned scope: {outside_scope}")
    return paths


def build_release_report_generation_inputs() -> dict[str, Any]:
    """Build the hashed manifest and exact CPU runner contract for the parent."""
    paths = release_report_input_paths()
    return {
        "notebook_id": "9_5_rome_model_editing",
        "runner_module": (
            "chapter9_alignment_interpretability.exercises."
            "part5_rome_model_editing.solutions"
        ),
        "runner_callable": "run_cpu_release",
        "implementation_factory": "reference_learner_functions",
        "runner_kwargs": {"steps": 12, "patching_samples": 3},
        "requires_gpu": False,
        "schema_path": str((SECTION / "verification_report.schema.json").relative_to(ROOT)),
        "metric_artifact_path": str(
            (SECTION / "expected_outputs" / "reference_metrics.json").relative_to(ROOT)
        ),
        "report_inputs": build_report_input_manifest(ROOT, paths),
        "shared_registration": "parent_owned_not_modified_by_section",
    }


if __name__ == "__main__":
    print(json.dumps(build_release_report_generation_inputs(), indent=2, sort_keys=True))
