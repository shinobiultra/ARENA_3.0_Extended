"""Publish only the TreeSHAP lesson that was executed, reproduced, and reviewed."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.build_notebooks import (
    STEM,
    complete_learner,
    lesson_cells,
)
from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.execute_learner import (
    digest,
    learner_source_paths,
)
from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.run_native import (
    DETERMINISTIC_ARTIFACTS,
    TIMING_ARTIFACT,
    load_payload,
)

PAGE = HERE.parents[1] / "instructions/pages/13_[16.13]_TreeSHAP_from_First_Principles.md"


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def published_asset_name(learner_name: str) -> str:
    """Namespace section assets while preserving concise learner-cache stems."""
    return f"treeshap_{learner_name}"


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
    """Find only literal show_figure(figure, 'stem') calls in the code AST."""
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
    require(execution.get("code_cells", 0) > 0, "The executed learner has no code cells.")
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
    comparisons = reproduction.get("comparisons", {})
    require(
        comparisons.get("arrays", 0) > 0
        and comparisons.get("array_elements", 0) > 0
        and comparisons.get("max_absolute_error") == 0.0,
        "Every deterministic native array must match the fresh refit exactly.",
    )
    timing = reproduction.get("timing_comparison", {})
    require(
        timing.get("measured_arrays", 0) + timing.get("measured_scalars", 0) > 0,
        "Measured timings must be validated separately.",
    )
    expected_artifacts = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
    for directory, records in (
        (learner, execution.get("native_artifacts", {})),
        (fresh, reproduction.get("artifacts", {})),
    ):
        require(
            {Path(name).name for name in records} == expected_artifacts,
            "A required native artifact is missing or an unexpected artifact was supplied.",
        )
        for name, record in records.items():
            path = directory / name
            require(
                path.is_file()
                and path.stat().st_size == record["size_bytes"]
                and digest(path) == record["sha256"],
                "Native artifact changed or is missing: " + name,
            )
    reference = load_payload(Path(execution["native_directory"]) / "reference.pt")
    require(isinstance(reference, dict) and reference, "Reference evidence is empty.")
    passes = reference.get("passes", {})
    require(
        reference.get("status") == "PASS_WITH_EXPECTED_RAW_REFERENCE_FAILURE",
        "Reference evidence must retain the expected raw SHAP threshold failure.",
    )
    require(
        passes.get("path_compiled") is True
        and passes.get("interventional_unmodified") is False
        and passes.get("interventional_threshold_corrected") is True
        and passes.get("exhaustive_path") is True
        and passes.get("exhaustive_interventional") is True
        and passes.get("sklearn_models_unchanged") is True,
        "Raw, corrected, exhaustive, and model-integrity reference boundaries are incomplete.",
    )
    require(
        reference.get("unmodified_interventional_passed") is False
        and reference.get("threshold_corrected_interventional_passed") is True
        and reference.get("counts", {}).get("raw_affected_model_rows", 0) >= 2,
        "The raw threshold-routing discrepancy must remain visible and quantified.",
    )
    return execution, reproduction


def publish(learner: Path, fresh: Path) -> dict:
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
    require(bool(review.get("screenshots")), "Actual browser screenshots are required.")
    for name, checksum in review["screenshots"].items():
        path = learner / "html_review" / name
        require(path.is_file() and digest(path) == checksum, "Reviewed screenshot changed: " + name)
    for name, checksum in execution["figures"].items():
        path = learner / "figures" / name
        require(path.is_file() and digest(path) == checksum, "Generated figure changed: " + name)

    student = {
        "cells": lesson_cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    solved = complete_learner(student)
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
            "Every figure must map to an actual literal show_figure(figure, 'stem') call.",
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
        sorted(displayed) == sorted(execution["figures"]),
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
