"""Publish the exact executed lesson after native reproduction and visual review."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.build_notebooks import STEM, lesson_cells, complete_learner
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.execute_learner import learner_source_paths, digest

PAGE = HERE.parents[1] / "instructions/pages/10_[16.10]_Datamodels.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_fresh(learner, fresh):
    learner, fresh = Path(learner).resolve(), Path(fresh).resolve()
    execution = json.loads((learner / "execution.json").read_text())
    reproduction = json.loads((fresh / "reproduction.json").read_text())
    expected_sources = {str(path.relative_to(ROOT)) for path in learner_source_paths()}
    require(execution["status"] == "PASS" and not execution["changed_sources"], "Learner execution failed.")
    require(execution["native_model_executed"] and execution["code_cells"] == 31, "Full learner required.")
    require(set(execution["source_inputs"]) == expected_sources, "Incomplete learner source set.")
    expected_sources.add(str((HERE / "reproduce_native.py").relative_to(ROOT)))
    require(set(reproduction["source_inputs"]) == expected_sources, "Incomplete fresh source set.")
    for directory, state in [(learner, execution), (fresh, reproduction)]:
        for name, checksum in state["source_inputs"].items():
            require(digest(ROOT / name) == checksum, "Current source changed: " + name)
            require(digest(directory / "source_snapshot" / name) == checksum, "Source snapshot changed: " + name)
    require(reproduction["status"] == "PASS" and not reproduction["changed_sources"], "Fresh reproduction failed.")
    require(reproduction["learner_directory"] == str(learner), "Fresh run used a different learner.")
    require(reproduction["learner_execution_sha256"] == digest(learner / "execution.json"), "Learner evidence changed.")
    require(execution["completed_learner_sha256"] == digest(learner / "completed_learner.ipynb"), "Executed notebook changed.")
    require(execution["native_summary_sha256"] == digest(learner / "figures/native_summary.json"), "Native summary changed.")
    counts = reproduction["comparisons"]
    require(counts["tensors"] == 263 and counts["tensor_elements"] == 5546159 and counts["scalars"] == 327,
            "Every prescribed native comparison is required.")
    require(counts["max_absolute_error"] <= 1e-10, "Fresh native outcomes differ.")
    for directory, records in [(learner, execution["tensor_artifacts"]), (fresh, reproduction["artifacts"])]:
        require(len(records) == 3, "A native artifact is missing.")
        for name, record in records.items():
            path = directory / name
            require(path.stat().st_size == record["size_bytes"] and digest(path) == record["sha256"], "Native tensor changed: " + name)
    return execution, reproduction


def publish(learner, fresh):
    learner, fresh = Path(learner).resolve(), Path(fresh).resolve()
    execution, reproduction = validate_fresh(learner, fresh)
    review = json.loads((learner / "visual_review.json").read_text())
    require(review["status"] == "PASS" and review["browser_math_and_dropdowns_checked"], "Visual review required.")
    require(review["lesson_sha256"] == digest(HERE / "lesson.md"), "Reviewed lesson changed.")
    require(review["completed_learner_sha256"] == digest(learner / "completed_learner.ipynb"), "Reviewed execution changed.")
    require(review["figures"] == execution["figures"] and len(review["figures"]) == 15, "All figures must be reviewed.")
    require(bool(review["screenshots"]), "Browser screenshots required.")
    for name, checksum in review["screenshots"].items():
        require(digest(learner / "html_review" / name) == checksum, "Reviewed screenshot changed.")
    for name, checksum in execution["figures"].items():
        require(digest(learner / "figures" / name) == checksum, "Generated figure changed.")
    student = dict(cells=lesson_cells(), metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, nbformat=4, nbformat_minor=5)
    solved = complete_learner(student)
    executed = json.loads((learner / "completed_learner.ipynb").read_text())
    assets = HERE.parents[1] / "instructions/assets"
    assets.mkdir(exist_ok=True)
    pieces, displayed = [], []
    for exercise, answer, actual in zip(student["cells"], solved["cells"], executed["cells"], strict=True):
        require(answer["cell_type"] == actual["cell_type"], "Cell type changed.")
        require("".join(answer["source"]) == "".join(actual["source"]), "Published source differs from executed own answers.")
        source = "".join(exercise["source"]).strip()
        if exercise["cell_type"] == "markdown":
            # Resolve notebook-relative links from the instruction-page directory.
            def link(match):
                target = match[2]
                if "://" in target or target.startswith("#") or not (HERE / target).exists():
                    return match[0]
                return f"[{match[1]}]({os.path.relpath((HERE / target).resolve(), PAGE.parent)})"
            pieces.append(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, source))
            continue
        answer.update(outputs=copy.deepcopy(actual["outputs"]), execution_count=actual["execution_count"])
        if "raise NotImplementedError" not in source:
            exercise.update(outputs=copy.deepcopy(actual["outputs"]), execution_count=actual["execution_count"])
        pieces.append("```python\n" + source + "\n```")
        matches = []
        for name in execution["figures"]:
            stem = name.removesuffix(".png")
            pattern = re.sub(r"_(clean|corrupted)$", "_{condition}", stem)
            if stem in source or pattern in source:
                matches.append(name)
        count = sum("image/png" in output.get("data", {}) for output in actual["outputs"])
        require(len(matches) == count, "Every figure must map to its actual producing cell.")
        for name in matches:
            (assets / name).write_bytes((learner / "figures" / name).read_bytes())
            pieces.append(f"![Actual learner result: {name.removesuffix('.png').replace('_', ' ')}](../assets/{name})")
            displayed.append(name)
    require(sorted(displayed) == sorted(execution["figures"]), "A figure is missing or duplicated.")
    for suffix, notebook in [("exercises", student), ("solutions", solved)]:
        (HERE / f"{STEM}_{suffix}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    PAGE.write_text("\n\n".join(pieces) + "\n")
    expected = HERE / "expected_outputs"
    expected.mkdir(exist_ok=True)
    for origin, name in [(learner / "figures/native_summary.json", "native_summary.json"),
                         (fresh / "reproduction.json", "fresh_reproduction.json")]:
        (expected / name).write_bytes(origin.read_bytes())
    publication = dict(execution, learner_directory=str(learner), fresh_directory=str(fresh),
                       published_exercises_sha256=digest(HERE / f"{STEM}_exercises.ipynb"),
                       published_solutions_sha256=digest(HERE / f"{STEM}_solutions.ipynb"),
                       published_page_sha256=digest(PAGE), published_figures=execution["figures"],
                       visual_review=review, full_release_validation=True)
    (expected / "learner_execution.json").write_text(json.dumps(publication, indent=2) + "\n")
    print("Published 71 cells, 31 executed own-answer code cells and 15 controlled figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", type=Path, required=True)
    parser.add_argument("--fresh-directory", type=Path, required=True)
    args = parser.parse_args()
    publish(args.learner_directory, args.fresh_directory)
