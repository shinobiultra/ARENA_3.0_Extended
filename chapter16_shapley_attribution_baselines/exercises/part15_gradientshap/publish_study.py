"""Publish only a GradientSHAP lesson that was executed, reproduced, and reviewed."""

from __future__ import annotations

import argparse
import ast
import copy
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.execute_learner import (
    EXPECTED_FIGURES,
    digest,
    learner_source_paths,
)
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.reproduce_native import (
    validate_artifact_directory,
)
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.run_native import (
    DETERMINISTIC_ARTIFACTS,
    EXTERNAL_DETERMINISTIC_ARTIFACT,
    REPLICATE_ELEMENTS,
    REPLICATE_SHAPE,
    TIMING_ARTIFACT,
    load_payload,
    manifest_path,
)


PAGE = (
    HERE.parents[1]
    / "instructions/pages/15_[16.15]_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate.md"
)
# Review archives contain receipts, source and figures; numerical payloads stay hash-bound externally.
ARCHIVE_INLINE_LIMIT_BYTES = 0


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def published_asset_name(learner_name: str) -> str:
    return f"gradientshap_{learner_name}"


def _verify_snapshot(directory: Path, state: dict) -> None:
    for name, checksum in state["source_inputs"].items():
        source = ROOT / name
        snapshot = directory / "source_snapshot" / name
        require(source.is_file() and digest(source) == checksum, "Current source changed: " + name)
        require(
            snapshot.is_file() and digest(snapshot) == checksum,
            "Source snapshot changed or is missing: " + name,
        )


def _show_figure_stems(source: str) -> list[str]:
    """Find only literal show_figure(figure, 'stem') calls in executable code."""
    stems: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "show_figure" or len(node.args) < 2:
            continue
        argument = node.args[1]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            stems.append(argument.value)
    return stems


def _reference_contract(reference: dict[str, Any]) -> None:
    require(reference.get("status") == "PASS", "Package reference validation did not pass.")
    passes = reference.get("passes", {})
    require(bool(passes) and all(value is True for value in passes.values()), "Reference checks failed.")
    counts = reference.get("counts", {})
    expected = {
        "models": 7,
        "model_rows": 1904,
        "background_rows": 64,
        "samples_per_package": 128,
        "shap_vectors": 1904,
        "shap_coordinates": 15232,
        "captum_vectors": 1904,
        "captum_coordinates": 15232,
        "package_vectors": 3808,
        "package_coordinates": 30464,
    }
    for key, value in expected.items():
        require(counts.get(key) == value, f"Reference scope mismatch: {key}.")
    versions = reference.get("versions", {})
    require(versions.get("shap") == "0.49.1", "The SHAP reference version changed.")
    require(versions.get("captum") == "0.9.0", "The Captum reference version changed.")
    tolerances = reference.get("tolerances", {})
    require(
        tolerances.get("same_draw_atol") == 1.0e-10
        and tolerances.get("same_draw_rtol") == 1.0e-10,
        "Same-draw reference tolerances changed.",
    )


def validate_fresh(learner: Path, fresh: Path) -> tuple[dict, dict]:
    learner, fresh = Path(learner).resolve(), Path(fresh).resolve()
    execution_path = learner / "execution.json"
    reproduction_path = fresh / "reproduction.json"
    require(execution_path.is_file(), "Learner execution evidence is missing.")
    require(reproduction_path.is_file(), "Fresh reproduction evidence is missing.")
    execution = json.loads(execution_path.read_text())
    reproduction = json.loads(reproduction_path.read_text())
    expected_sources = {str(path.relative_to(ROOT)) for path in learner_source_paths()}
    require(execution.get("status") == "PASS", "Learner execution failed.")
    require(not execution.get("changed_sources"), "Learner sources changed during execution.")
    require(execution.get("completed_from_own_answers") is True, "Own-answer learner required.")
    require(execution.get("native_model_executed") is True, "Fresh native learner fit required.")
    require(execution.get("execution_device") == "cpu", "The canonical study must run on CPU.")
    require(execution.get("code_cells", 0) > 0, "The executed learner has no code cells.")
    require(execution.get("embedded_figures") == len(EXPECTED_FIGURES), "Figure count changed.")
    require(set(execution.get("source_inputs", {})) == expected_sources, "Incomplete learner sources.")
    require(set(reproduction.get("source_inputs", {})) == expected_sources, "Incomplete fresh sources.")
    _verify_snapshot(learner, execution)
    _verify_snapshot(fresh, reproduction)
    require(reproduction.get("status") == "PASS", "Fresh reproduction failed.")
    require(not reproduction.get("changed_sources"), "Sources changed during reproduction.")
    require(reproduction.get("native_model_retrained") is True, "All native models must be refit.")
    require(
        reproduction.get("cached_model_inputs_used") is False,
        "Fresh reproduction cannot consume cached learner models.",
    )
    require(reproduction.get("learner_directory") == str(learner), "Wrong learner was reproduced.")
    require(
        reproduction.get("learner_execution_sha256") == digest(execution_path),
        "Learner execution evidence changed.",
    )
    completed = learner / "completed_learner.ipynb"
    require(
        completed.is_file() and execution.get("completed_learner_sha256") == digest(completed),
        "Executed notebook changed or is missing.",
    )
    native_summary = learner / "figures/native_summary.json"
    require(
        native_summary.is_file() and execution.get("native_summary_sha256") == digest(native_summary),
        "Native summary changed or is missing.",
    )
    summary = json.loads(native_summary.read_text())
    require(summary.get("replicate_sample_shape") == list(REPLICATE_SHAPE), "Replicate shape changed.")
    require(
        summary.get("replicate_sample_coordinates") == REPLICATE_ELEMENTS,
        "The complete full replicate array is missing.",
    )
    comparisons = reproduction.get("comparisons", {})
    require(
        comparisons.get("arrays", 0) > 0
        and comparisons.get("array_elements", 0) >= REPLICATE_ELEMENTS
        and comparisons.get("max_absolute_error") == 0.0
        and comparisons.get("comparison") == "exact_recursive_value_sha256",
        "Every deterministic native array must match the fresh refit exactly.",
    )
    original = Path(execution["native_directory"]).resolve()
    original.relative_to(learner)
    fresh_native = fresh / "fresh_run"
    validate_artifact_directory(original)
    validate_artifact_directory(fresh_native)
    external = execution.get("external_replicate_artifact", {})
    external_path = learner / external.get("relative_path", "")
    require(
        external_path.name == EXTERNAL_DETERMINISTIC_ARTIFACT
        and external_path.is_file()
        and external.get("size_bytes") == external_path.stat().st_size
        and external.get("sha256") == digest(external_path),
        "The external full replicate payload changed or is missing.",
    )
    reference = load_payload(original / "reference.pt")
    _reference_contract(reference)
    return execution, reproduction


def _archive_members(learner: Path, native: Path) -> list[tuple[Path, str]]:
    members = [
        (learner / "execution.json", "execution.json"),
        (learner / "completed_learner.ipynb", "completed_learner.ipynb"),
        (native / "before_audit.json", "native/before_audit.json"),
        (native / "summary.json", "native/summary.json"),
        (native / "artifact_manifest.json", "native/artifact_manifest.json"),
    ]
    # Standalone run_native.py records this orchestration file; the direct executed-notebook
    # workflow records its equivalent state in execution.json and intentionally has no run.json.
    # Archive the runner record when present without making it a requirement for learner evidence.
    runner_record = native / "run.json"
    if runner_record.is_file():
        members.append((runner_record, "native/run.json"))
    members.extend(
        (path, str(path.relative_to(learner)))
        for path in sorted((learner / "figures").glob("*"))
        if path.is_file()
    )
    members.extend(
        (path, str(path.relative_to(learner)))
        for path in sorted((learner / "source_snapshot").rglob("*"))
        if path.is_file()
    )
    members.extend(
        (path, f"native/{path.name}") for path in sorted(native.glob("*.manifest.json"))
    )
    return members


def archive_evidence(learner: Path, archive: Path) -> dict[str, Any]:
    """Archive reviewable evidence while hash-binding cache-resident numerical payloads."""
    learner = Path(learner).resolve()
    execution = json.loads((learner / "execution.json").read_text())
    native = Path(execution["native_directory"]).resolve()
    native.relative_to(learner)
    manifests = validate_artifact_directory(native)
    members = _archive_members(learner, native)
    included = {
        name: {"sha256": digest(path), "size_bytes": path.stat().st_size}
        for path, name in members
    }
    payloads = {
        name: {
            "absolute_path": str(native / name),
            "artifact_sha256": manifests[name]["artifact_sha256"],
            "size_bytes": manifests[name]["size_bytes"],
            "manifest": f"native/{manifest_path(Path(name)).name}",
            "manifest_sha256": digest(manifest_path(native / name)),
            "external_large_payload": name == EXTERNAL_DETERMINISTIC_ARTIFACT,
            "archived_bytes": False,
        }
        for name in (*DETERMINISTIC_ARTIFACTS, TIMING_ARTIFACT)
    }
    external = {name: record for name, record in payloads.items() if not record["archived_bytes"]}
    archive_manifest = {
        "schema_version": 1,
        "included_files": included,
        "payloads": payloads,
        "external_payloads": external,
        "inline_payload_limit_bytes": ARCHIVE_INLINE_LIMIT_BYTES,
    }
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        for path, name in members:
            handle.add(path, arcname=name, recursive=False)
        payload = (json.dumps(archive_manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("archive_manifest.json")
        info.size = len(payload)
        info.mtime = 0
        handle.addfile(info, io.BytesIO(payload))
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
        require(names == set(included) | {"archive_manifest.json"}, "Archive member set changed.")
        for name, record in included.items():
            member = handle.extractfile(name)
            require(member is not None, "Archive member cannot be read: " + name)
            data = member.read()
            require(
                len(data) == record["size_bytes"]
                and __import__("hashlib").sha256(data).hexdigest() == record["sha256"],
                "Archive member failed byte verification: " + name,
            )
    from scripts.evidence_bundles import seal_review_bundle

    seal_review_bundle(
        archive,
        provenance={
            "kind": "native_execution_packaging",
            "source_execution": {
                "absolute_path": str(learner / "execution.json"),
                "sha256": digest(learner / "execution.json"),
                "size_bytes": (learner / "execution.json").stat().st_size,
            },
            "measurement_claim": "Packaging only; execution dates and scientific receipts are unchanged.",
        },
        external_payloads=external,
    )
    return {
        "path": str(archive),
        "sha256": digest(archive),
        "size_bytes": archive.stat().st_size,
        "external_payloads": external,
    }


def publish(learner: Path, fresh: Path) -> dict:
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.build_notebooks import (
        STEM,
        build_notebooks,
    )

    learner, fresh = Path(learner).resolve(), Path(fresh).resolve()
    execution, reproduction = validate_fresh(learner, fresh)
    review_path = learner / "visual_review.json"
    require(review_path.is_file(), "Actual browser visual review is required before publishing.")
    review = json.loads(review_path.read_text())
    require(review.get("status") == "PASS", "Visual review did not pass.")
    require(
        review.get("browser_math_and_dropdowns_checked") is True,
        "Browser math and dropdown review is required.",
    )
    require(review.get("lesson_sha256") == digest(HERE / "lesson.md"), "Reviewed lesson changed.")
    require(
        review.get("completed_learner_sha256") == digest(learner / "completed_learner.ipynb"),
        "Reviewed notebook changed.",
    )
    require(
        review.get("figures") == execution.get("figures") and bool(review.get("figures")),
        "Every generated figure must be reviewed.",
    )
    require(bool(review.get("screenshots")), "Browser-review screenshots are required.")

    student, solved = build_notebooks()
    executed = json.loads((learner / "completed_learner.ipynb").read_text())
    require(
        len(student["cells"]) == len(solved["cells"]) == len(executed["cells"]),
        "Draft, own-answer, and executed cell counts differ.",
    )
    assets = HERE.parents[1] / "instructions/assets"
    assets.mkdir(parents=True, exist_ok=True)
    pieces: list[str] = []
    displayed: list[str] = []
    for exercise, answer, actual in zip(student["cells"], solved["cells"], executed["cells"], strict=True):
        require(answer["cell_type"] == actual["cell_type"], "Executed cell type changed.")
        require(
            "".join(answer["source"]) == "".join(actual["source"]),
            "Published source differs from executed own answers.",
        )
        source = "".join(exercise["source"]).strip()
        if exercise["cell_type"] == "markdown":
            def link(match):
                target = match[2]
                if "://" in target or target.startswith("#") or not (HERE / target).exists():
                    return match[0]
                return f"[{match[1]}]({os.path.relpath((HERE / target).resolve(), PAGE.parent)})"

            pieces.append(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, source))
            continue
        answer.update(outputs=copy.deepcopy(actual["outputs"]), execution_count=actual["execution_count"])
        if "raise NotImplementedError" not in source:
            exercise.update(
                outputs=copy.deepcopy(actual["outputs"]), execution_count=actual["execution_count"]
            )
        pieces.append("```python\n" + source + "\n```")
        stems = _show_figure_stems(source)
        matches = [name for name in execution["figures"] if name.removesuffix(".png") in stems]
        image_outputs = sum("image/png" in output.get("data", {}) for output in actual["outputs"])
        require(
            len(matches) == len(stems) == image_outputs,
            "Every figure must map to an actual literal show_figure call.",
        )
        for name in matches:
            asset_name = published_asset_name(name)
            (assets / asset_name).write_bytes((learner / "figures" / name).read_bytes())
            pieces.append(
                f"![Actual learner result: {name.removesuffix('.png').replace('_', ' ')}]"
                f"(../assets/{asset_name})"
            )
            displayed.append(name)
    require(
        sorted(displayed) == sorted(execution["figures"]) == sorted(EXPECTED_FIGURES),
        "A generated figure is missing from the lesson or displayed more than once.",
    )
    for suffix, notebook in (("exercises", student), ("solutions", solved)):
        (HERE / f"{STEM}_{suffix}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text("\n\n".join(pieces) + "\n")
    expected = HERE / "expected_outputs"
    expected.mkdir(exist_ok=True)
    for origin, name in (
        (learner / "figures/native_summary.json", "native_summary.json"),
        (fresh / "reproduction.json", "fresh_reproduction.json"),
    ):
        (expected / name).write_bytes(origin.read_bytes())
    archive = archive_evidence(learner, expected / "execution.tar.gz")
    publication = {
        **execution,
        "learner_directory": str(learner),
        "fresh_directory": str(fresh),
        "published_exercises_sha256": digest(HERE / f"{STEM}_exercises.ipynb"),
        "published_solutions_sha256": digest(HERE / f"{STEM}_solutions.ipynb"),
        "published_page_sha256": digest(PAGE),
        "published_figures": {
            published_asset_name(name): checksum for name, checksum in execution["figures"].items()
        },
        "visual_review": review,
        "fresh_reproduction_sha256": digest(fresh / "reproduction.json"),
        "archive": archive,
        "full_release_validation": True,
    }
    output = expected / "learner_execution.json"
    output.write_text(json.dumps(publication, indent=2) + "\n")
    print(
        f"Published {len(student['cells'])} cells, {execution['code_cells']} own-answer code cells, "
        f"and {len(displayed)} reviewed figures."
    )
    return publication


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", type=Path, required=True)
    parser.add_argument("--fresh-directory", type=Path, required=True)
    arguments = parser.parse_args()
    publish(arguments.learner_directory, arguments.fresh_directory)
