"""Publish the visually reviewed, independently reproduced own-answer learner."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.build_notebooks import (
    HERE,
    STEM,
    lesson_cells,
    complete_learner,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.das_reproduction import (
    validate_learner,
    file_digest,
)

FIGURES = (
    "das_exact_interchanges.png",
    "das_learning_and_controls.png",
    "das_coordinate_and_specificity.png",
    "das_angle_play.png",
    "das_native_site_comparison.png",
    "das_native_site_play.png",
    "das_native_rank_selection.png",
    "das_native_final_controls.png",
    "das_native_specificity.png",
    "das_native_generalization.png",
    "das_native_token_examples.png",
    "das_native_pair_play.png",
)


def publish(learner, fresh):
    execution, tensors, methods = validate_learner(learner)
    del tensors, methods
    reproduction = json.loads((fresh / "reproduction.json").read_text())
    if (
        reproduction["status"] != "PASS"
        or reproduction["changed_sources"]
        or reproduction["learner_directory"] != str(learner)
    ):
        raise ValueError("Require a fresh successful reproduction of this exact learner.")
    if reproduction["learner_execution_sha256"] != file_digest(learner / "execution.json"):
        raise ValueError("The reproduced learner execution changed.")
    for name, digest in reproduction["source_inputs"].items():
        if file_digest(fresh / "source_snapshot" / name) != digest:
            raise ValueError("Fresh reproduction source snapshot changed: " + name)
        # validate_learner above proves every current own-answer code AST still
        # matches the executed learner. Prose-only rendering repairs do not
        # change a native experiment; the current prose is reviewed separately.
        if name != "lesson.md" and file_digest(HERE / name) != digest:
            raise ValueError("Fresh reproduction input changed: " + name)
    for filename, key in [
        ("actual_comparisons.jsonl", "actual_comparisons_sha256"),
        ("actual_fresh_native_tensors.pt", "actual_tensor_sha256"),
    ]:
        if file_digest(fresh / filename) != reproduction[key]:
            raise ValueError("Fresh native evidence changed: " + filename)
    if len(reproduction["conditions"]) != 43 or reproduction["actual_intervention_rows"] != 16512:
        raise ValueError("Every one of 43 final conditions must have been reproduced.")
    review = json.loads((learner / "visual_review.json").read_text())
    if review["status"] != "PASS" or not review["browser_math_and_dropdowns_checked"]:
        raise ValueError("Require actual figure and rendered notebook review.")
    if review["lesson_sha256"] != file_digest(HERE / "lesson.md"):
        raise ValueError("The reviewed learner prose changed.")
    actual_figures = {name: file_digest(learner / "figures" / name) for name in FIGURES}
    if review["figures"] != actual_figures:
        raise ValueError("The reviewed figures changed or are incomplete.")
    student = dict(
        cells=lesson_cells(),
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        nbformat=4,
        nbformat_minor=5,
    )
    solved = complete_learner(student)
    actual = json.loads((learner / "completed_learner.ipynb").read_text())
    for exercise, answer, executed in zip(
        student["cells"], solved["cells"], actual["cells"], strict=True
    ):
        if exercise["cell_type"] != "code":
            continue
        answer.update(
            outputs=copy.deepcopy(executed["outputs"]), execution_count=executed["execution_count"]
        )
        if "raise NotImplementedError" not in "".join(exercise["source"]):
            exercise.update(
                outputs=copy.deepcopy(executed["outputs"]),
                execution_count=executed["execution_count"],
            )
    for suffix, notebook in [("exercises", student), ("solutions", solved)]:
        (HERE / f"{STEM}_{suffix}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    assets = HERE.parents[1] / "instructions/assets"
    assets.mkdir(exist_ok=True)
    for name in FIGURES:
        (assets / name).write_bytes((learner / "figures" / name).read_bytes())
    pieces, displayed = [], []
    for cell in student["cells"]:
        source = "".join(cell["source"]).strip()
        if cell["cell_type"] == "markdown":
            pieces.append(source)
        else:
            pieces.append("```python\n" + source + "\n```")
            for name in FIGURES:
                if name in source:
                    pieces.append(
                        f"![Actual learner result: {name.removesuffix('.png').replace('_', ' ')}](../assets/{name})"
                    )
                    displayed.append(name)
    if sorted(displayed) != sorted(FIGURES):
        raise ValueError("Every actual figure must appear once in the instruction page.")
    page = (
        HERE.parents[1]
        / "instructions/pages/03_[11.3]_Causal_Abstraction_and_Distributed_Alignment_Search.md"
    )
    page.write_text("\n\n".join(pieces) + "\n")
    expected = HERE / "expected_outputs"
    (expected / "native_summary.json").write_bytes(
        (learner / "figures/native_summary.json").read_bytes()
    )
    (expected / "fresh_reproduction.json").write_bytes((fresh / "reproduction.json").read_bytes())
    record = {
        **execution,
        "learner_directory": str(learner),
        "fresh_directory": str(fresh),
        "full_release_validation": True,
        "published_code_matches_executed_own_answers": True,
        "published_exercises_sha256": file_digest(HERE / f"{STEM}_exercises.ipynb"),
        "published_solutions_sha256": file_digest(HERE / f"{STEM}_solutions.ipynb"),
        "published_page_sha256": file_digest(page),
        "published_figures": actual_figures,
        "visual_review": review,
    }
    (expected / "learner_execution.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"Published {len(student['cells'])} cells, {execution['code_cells']} executed code cells, {len(FIGURES)} actual figures."
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", type=Path, required=True)
    parser.add_argument("--fresh-directory", type=Path, required=True)
    args = parser.parse_args()
    publish(args.learner_directory.resolve(), args.fresh_directory.resolve())
