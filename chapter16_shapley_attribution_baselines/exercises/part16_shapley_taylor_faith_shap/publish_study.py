"""Publish 16.16 only after exact execution, reproduction, and byte-level visual review."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.execute_learner import (
    EXPECTED_FIGURES,
    learner_source_paths,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.reproduce_native import (
    validate_artifact_directory,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.run_native import (
    DETERMINISTIC_ARTIFACTS,
    EXTERNAL_DETERMINISTIC_ARTIFACTS,
    MODEL_KEYS,
    TIMING_ARTIFACT,
    digest,
    load_payload,
    manifest_path,
    validate_freeze,
)


PAGE = (
    HERE.parents[1]
    / "instructions/pages/16_[16.16]_Where_Does_Higher_Order_Credit_Go.md"
)
ARCHIVE_INLINE_LIMIT_BYTES = 128 * 1024 * 1024
FULL_HYBRID_COORDINATES = 50_135_040


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def published_asset_name(learner_name: str) -> str:
    return f"shapley_interactions_{learner_name}"


def _verify_snapshot(directory: Path, state: Mapping[str, Any]) -> None:
    for name, checksum in state["source_inputs"].items():
        source = ROOT / name
        snapshot = directory / "source_snapshot" / name
        require(source.is_file() and digest(source) == checksum, "Current source changed: " + name)
        require(
            snapshot.is_file() and digest(snapshot) == checksum,
            "Source snapshot changed or is missing: " + name,
        )


def _show_figure_stems(source: str) -> list[str]:
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


def _reference_contract(reference: Mapping[str, Any]) -> None:
    require(reference.get("status") == "PASS", "Independent interaction reference did not pass.")
    passes = reference.get("passes", {})
    require(bool(passes) and all(value is True for value in passes.values()), "Reference checks failed.")
    counts = reference.get("counts", {})
    require(counts.get("games") == 2_992, "Reference must cover all 2,992 unique empirical games.")
    require(
        counts.get("primary_games") == len(MODEL_KEYS) * 272,
        "Reference primary model/row scope changed.",
    )
    require(
        counts.get("nonprimary_clean0_games") == 4 * 272,
        "Reference background scope changed.",
    )
    require(
        counts.get("direct_shapiq_exact_computers") == 2_992
        and counts.get("direct_shapiq_index_calls") == 2_992 * 3,
        "Reference must report its actual direct package call scope.",
    )
    require(
        counts.get("all_reference_direct_shapiq_exact_computers") == 3_050
        and counts.get("all_reference_direct_shapiq_index_calls") == 9_150,
        "Augmented exact-fixture and empirical package scope changed.",
    )
    tolerances = reference.get("tolerances", {})
    strict = tolerances.get("strict", {})
    require(
        strict.get("absolute_floor") == 1.0e-10
        and strict.get("scale_coefficient") == 1.0e-12,
        "Strict same-index tolerance changed.",
    )


def validate_fresh(learner: Path, fresh: Path) -> tuple[dict, dict]:
    """Validate current source bytes and exact original/fresh scientific evidence."""

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
    require(execution.get("native_model_executed") is True, "Native learner execution required.")
    require(execution.get("execution_device") == "cpu", "The canonical study must run on CPU.")
    require(execution.get("code_cells", 0) > 0, "The executed learner has no code cells.")
    require(execution.get("embedded_figures") == len(EXPECTED_FIGURES), "Figure count changed.")
    require(set(execution.get("source_inputs", {})) == expected_sources, "Incomplete learner sources.")
    require(set(reproduction.get("source_inputs", {})) == expected_sources, "Incomplete fresh sources.")
    _verify_snapshot(learner, execution)
    _verify_snapshot(fresh, reproduction)
    require(reproduction.get("status") == "PASS", "Fresh reproduction failed.")
    require(not reproduction.get("changed_sources"), "Sources changed during reproduction.")
    require(reproduction.get("native_model_retrained") is True, "All seven models must be refit.")
    require(
        reproduction.get("cached_model_inputs_used") is False
        and reproduction.get("child_received_learner_payloads") is False,
        "Fresh reproduction cannot consume learner fitted payloads.",
    )
    require(reproduction.get("learner_directory") == str(learner), "Wrong learner was reproduced.")
    require(
        reproduction.get("learner_execution_sha256") == digest(execution_path),
        "Learner execution record changed.",
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
    require(
        summary.get("full_hybrid_coordinates") == FULL_HYBRID_COORDINATES,
        "The complete ten-shard donor prediction scope is missing.",
    )
    comparisons = reproduction.get("comparisons", {})
    require(
        comparisons.get("arrays", 0) > 0
        and comparisons.get("array_elements", 0) >= FULL_HYBRID_COORDINATES
        and comparisons.get("max_absolute_error") == 0.0
        and comparisons.get("comparison") == "exact_recursive_value_sha256",
        "Every deterministic native leaf must match the fresh refit exactly.",
    )
    original = Path(execution["native_directory"]).resolve()
    original.relative_to(learner)
    fresh_native = fresh / "fresh_run"
    validate_artifact_directory(original)
    validate_artifact_directory(fresh_native)
    external = execution.get("external_hybrid_artifacts", {})
    require(set(external) == set(EXTERNAL_DETERMINISTIC_ARTIFACTS), "Hybrid shard inventory changed.")
    for name, record in external.items():
        path = learner / record.get("relative_path", "")
        require(
            path.name == name
            and path.is_file()
            and record.get("size_bytes") == path.stat().st_size
            and record.get("sha256") == digest(path),
            "A complete raw hybrid shard changed or is missing: " + name,
        )
    reference = load_payload(original / "reference.pt")
    _reference_contract(reference)
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_reference import (
        validate_reference_payload,
    )

    validation = validate_reference_payload(reference, expected_game_count=2_992)
    require(validation.get("status") == "PASS", "Stored reference arrays failed live revalidation.")
    return execution, reproduction


def validate_publication_review(execution: Mapping[str, Any]) -> dict:
    """Keep a locally resolved development run distinct from independent approval."""
    freeze = validate_freeze()
    if freeze.get("independent_approval") is True:
        return freeze
    path = HERE / "independent_recheck.json"
    require(path.is_file(), "Development execution is complete, but independent publication review is still pending.")
    review = json.loads(path.read_text())
    require(review.get("decision") == "APPROVE_PUBLICATION", "Independent publication review did not approve.")
    require(review.get("reviewer_role") == "independent", "Root resolution cannot substitute for independent publication review.")
    require(review.get("review_timing") == "after_development_execution", "Review timing must be stated honestly.")
    require(review.get("protocol_freeze_sha256") == digest(HERE / "protocol_freeze.json"), "Review binds a different frozen design.")
    require(review.get("source_inputs") == execution.get("source_inputs"), "Independent review binds different source bytes.")
    return review


def _resolve_review_file(
    record: Mapping[str, Any], learner: Path, fresh: Path, label: str
) -> Path:
    require(set(record) >= {"path", "sha256", "size_bytes"}, f"Incomplete review record: {label}")
    path = Path(str(record["path"]))
    candidates = [path.resolve()] if path.is_absolute() else [
        (learner / path).resolve(),
        (fresh / path).resolve(),
    ]
    valid: list[Path] = []
    for candidate in candidates:
        try:
            candidate.relative_to(learner)
            valid.append(candidate)
            continue
        except ValueError:
            pass
        try:
            candidate.relative_to(fresh)
            valid.append(candidate)
        except ValueError:
            pass
    path = next((candidate for candidate in valid if candidate.is_file()), None)
    require(path is not None, f"Review file is missing or outside evidence roots: {label}")
    require(path.stat().st_size == record["size_bytes"], f"Review file size changed: {label}")
    require(digest(path) == record["sha256"], f"Review file bytes changed: {label}")
    return path


def validate_visual_review(
    learner: Path,
    fresh: Path,
    execution: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    """Rehash the notebook, every figure, screenshot, contact, and review report."""

    review_path = learner / "visual_review.json"
    require(review_path.is_file(), "Actual browser visual review is required before publishing.")
    review = json.loads(review_path.read_text())
    require(review.get("status") == "PASS", "Visual review did not pass.")
    require(
        review.get("browser_math_and_dropdowns_checked") is True,
        "Browser math and dropdown review is required.",
    )
    require(review.get("lesson_sha256") == digest(HERE / "lesson.md"), "Reviewed lesson changed.")
    completed = learner / "completed_learner.ipynb"
    require(
        review.get("completed_learner_sha256") == digest(completed),
        "Reviewed completed notebook changed.",
    )
    figure_hashes = {
        name: digest(learner / "figures" / name) for name in EXPECTED_FIGURES
    }
    require(figure_hashes == execution.get("figures"), "Execution figure bytes changed.")
    require(review.get("figures") == figure_hashes, "Reviewed figure map changed.")

    evidence = review.get("evidence_files")
    require(isinstance(evidence, Mapping), "Visual review needs explicit evidence_files records.")
    files: list[Path] = [review_path]
    singleton_keys = (
        "completed_notebook",
        "browser_report",
        "browser_stdout",
        "independent_figure_review",
        "fresh_reproduction",
    )
    for key in singleton_keys:
        record = evidence.get(key)
        require(isinstance(record, Mapping), f"Missing review evidence record: {key}")
        files.append(_resolve_review_file(record, learner, fresh, key))
    for key in ("screenshots", "inspection_contacts"):
        records = evidence.get(key)
        require(isinstance(records, Mapping) and bool(records), f"Missing review evidence set: {key}")
        for name, record in records.items():
            require(isinstance(record, Mapping), f"Invalid review evidence record: {key}/{name}")
            files.append(_resolve_review_file(record, learner, fresh, f"{key}/{name}"))
    require(completed in files, "The explicit completed-notebook record resolves elsewhere.")
    require(
        (fresh / "reproduction.json") in files,
        "The explicit fresh-reproduction record resolves elsewhere.",
    )
    return review, list(dict.fromkeys(files))


def _archive_members(
    learner: Path,
    native: Path,
    review_files: Iterable[Path],
) -> list[tuple[Path, str]]:
    members = [
        (learner / "execution.json", "execution.json"),
        (learner / "completed_learner.ipynb", "completed_learner.ipynb"),
        (learner / "visual_review.json", "visual_review.json"),
        (native / "before_audit.json", "native/before_audit.json"),
        (native / "summary.json", "native/summary.json"),
        (native / "artifact_manifest.json", "native/artifact_manifest.json"),
    ]
    runner = native / "run.json"
    if runner.is_file():
        members.append((runner, "native/run.json"))
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
    members.extend((path, f"native/{path.name}") for path in sorted(native.glob("*.manifest.json")))
    members.extend(
        (path, f"native/{path.name}")
        for path in sorted(native.glob("*.pt"))
        if path.stat().st_size <= ARCHIVE_INLINE_LIMIT_BYTES
    )
    for index, path in enumerate(review_files):
        if path in {item[0] for item in members}:
            continue
        members.append((path, f"visual_evidence/{index:03d}_{path.name}"))
    names = [name for _, name in members]
    require(len(names) == len(set(names)), "Archive paths must be unique.")
    return members


def archive_evidence(
    learner: Path,
    archive: Path,
    review_files: Iterable[Path],
) -> dict[str, Any]:
    """Archive small evidence and bind cache-resident large tensor bytes exactly."""

    learner = Path(learner).resolve()
    execution = json.loads((learner / "execution.json").read_text())
    native = Path(execution["native_directory"]).resolve()
    native.relative_to(learner)
    manifests = validate_artifact_directory(native)
    members = _archive_members(learner, native, review_files)
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
            "external_hybrid_payload": name in EXTERNAL_DETERMINISTIC_ARTIFACTS,
            "archived_bytes": manifests[name]["size_bytes"] <= ARCHIVE_INLINE_LIMIT_BYTES,
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
    require(not archive.exists(), "Refusing to overwrite an evidence archive.")
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
                and hashlib.sha256(data).hexdigest() == record["sha256"],
                "Archive member failed byte verification: " + name,
            )
    return {
        "path": str(archive),
        "sha256": digest(archive),
        "size_bytes": archive.stat().st_size,
        "external_payloads": external,
    }


def publish(learner: Path, fresh: Path) -> dict[str, Any]:
    """Publish the executed pair, page, actual figures, evidence, and archive."""

    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.build_notebooks import (
        STEM,
        build_notebooks,
        expected_page_path,
        render_instruction_code,
        validate_page_surface,
    )

    learner, fresh = Path(learner).resolve(), Path(fresh).resolve()
    execution, reproduction = validate_fresh(learner, fresh)
    validate_publication_review(execution)
    review, review_files = validate_visual_review(learner, fresh, execution)
    student, solved = build_notebooks()
    executed = json.loads((learner / "completed_learner.ipynb").read_text())
    require(len(solved["cells"]) == len(executed["cells"]), "Own-answer execution lost a cell.")
    student_by_id = {cell["id"]: cell for cell in student["cells"]}
    solved_by_id = {cell["id"]: cell for cell in solved["cells"]}
    executed_by_id = {cell["id"]: cell for cell in executed["cells"]}
    require(len(student_by_id) == len(student["cells"]), "Exercise cell IDs are not unique.")
    require(len(solved_by_id) == len(solved["cells"]), "Solution cell IDs are not unique.")
    require(list(solved_by_id) == list(executed_by_id), "Executed solution cell order changed.")
    require(set(solved_by_id) < set(student_by_id), "Solutions must omit exercise-only dropdown cells.")
    require(
        len(student_by_id) - len(solved_by_id) == 8,
        "Solutions must omit exactly the eight redundant Solution dropdown cells.",
    )
    assets = HERE.parents[1] / "instructions/assets"
    pieces: list[str] = []
    displayed: list[str] = []
    asset_payloads: dict[str, bytes] = {}
    for exercise in student["cells"]:
        answer = solved_by_id.get(exercise["id"])
        actual = executed_by_id.get(exercise["id"])
        if answer is None:
            require(
                exercise["cell_type"] == "markdown"
                and "<summary>Solution</summary>" in "".join(exercise["source"]),
                "Only exercise Solution dropdowns may be absent from the solutions notebook.",
            )
            source = "".join(exercise["source"]).strip()
            pieces.append(source)
            continue
        require(actual is not None, "An own-answer cell was not executed.")
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
                relative = os.path.relpath((HERE / target).resolve(), PAGE.parent)
                return f"[{match[1]}]({relative})"

            pieces.append(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, source))
            continue
        answer.update(outputs=copy.deepcopy(actual["outputs"]), execution_count=actual["execution_count"])
        if "raise NotImplementedError" not in source:
            exercise.update(
                outputs=copy.deepcopy(actual["outputs"]), execution_count=actual["execution_count"]
            )
        pieces.append(render_instruction_code(exercise))
        stems = _show_figure_stems(source)
        matches = [name for name in execution["figures"] if name.removesuffix(".png") in stems]
        image_outputs = sum("image/png" in output.get("data", {}) for output in actual["outputs"])
        require(
            len(matches) == len(stems) == image_outputs,
            "Every figure must map to one actual literal show_figure call.",
        )
        for name in matches:
            asset_name = published_asset_name(name)
            data = (learner / "figures" / name).read_bytes()
            require(
                hashlib.sha256(data).hexdigest() == execution["figures"][name],
                "Staged figure bytes changed.",
            )
            asset_payloads[asset_name] = data
            pieces.append(
                f"![Actual learner result: {name.removesuffix('.png').replace('_', ' ')}]"
                f"(../assets/{asset_name})"
            )
            displayed.append(name)
    require(
        sorted(displayed) == sorted(execution["figures"]) == sorted(EXPECTED_FIGURES),
        "A generated figure is missing from the lesson or displayed more than once.",
    )
    page = "\n\n".join(pieces) + "\n"
    validate_page_surface(page)
    require(expected_page_path() == PAGE, "Builder and publisher page paths differ.")

    # All source, execution, reproduction, review, mapping, and page checks have now passed.
    assets.mkdir(parents=True, exist_ok=True)
    for name, data in asset_payloads.items():
        (assets / name).write_bytes(data)
    for suffix, notebook in (("exercises", student), ("solutions", solved)):
        (HERE / f"{STEM}_{suffix}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page)
    expected = HERE / "expected_outputs"
    expected.mkdir(exist_ok=True)
    for origin, name in (
        (learner / "figures/native_summary.json", "native_summary.json"),
        (fresh / "reproduction.json", "fresh_reproduction.json"),
    ):
        (expected / name).write_bytes(origin.read_bytes())
        require(digest(expected / name) == digest(origin), "Published evidence copy changed: " + name)
    archive = archive_evidence(learner, expected / "execution.tar.gz", review_files)
    publication = {
        **execution,
        "learner_directory": str(learner),
        "fresh_directory": str(fresh),
        "published_exercises_sha256": digest(HERE / f"{STEM}_exercises.ipynb"),
        "published_solutions_sha256": digest(HERE / f"{STEM}_solutions.ipynb"),
        "published_page_sha256": digest(PAGE),
        "published_figures": {
            published_asset_name(name): digest(assets / published_asset_name(name))
            for name in EXPECTED_FIGURES
        },
        "visual_review": review,
        "visual_review_sha256": digest(learner / "visual_review.json"),
        "fresh_reproduction_sha256": digest(fresh / "reproduction.json"),
        "archive": archive,
        "full_release_validation": True,
    }
    output = expected / "learner_execution.json"
    output.write_text(json.dumps(publication, indent=2) + "\n")
    print(
        f"Published {len(student['cells'])} cells, {execution['code_cells']} own-answer code cells, "
        f"and {len(displayed)} byte-reviewed figures."
    )
    return publication


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", type=Path, required=True)
    parser.add_argument("--fresh-directory", type=Path, required=True)
    arguments = parser.parse_args()
    publish(arguments.learner_directory, arguments.fresh_directory)
