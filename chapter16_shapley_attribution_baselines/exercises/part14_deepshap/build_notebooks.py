"""Build the paired 16.14 DeepSHAP notebooks from one readable lesson source.

This builder is draft-only. ``--check`` validates the learner arc and its own
answers; ``--draft-output`` writes review notebooks to a new directory without
publishing course surfaces.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP"
MARKER = re.compile(r"^<!-- ARENA:cell (markdown|code) -->\s*$", re.M)
INSERT = re.compile(r"^# ARENA:insert ([A-Za-z0-9_.-]+):([a-z_]+)\s*$", re.M)
EXERCISE_FUNCTIONS = (
    "paired_forward",
    "rescale_slope",
    "rescale_multipliers",
    "single_reference",
    "background_deepshap",
    "coalition_values",
    "exact_shapley",
    "reveal_cancel_groups",
    "ranked_replacement_curves",
)

# These groups are intentionally semantic. They keep the learner-facing fitting
# and diagnostic algorithms visible without dropping whole modules into giant cells.
VISIBLE_GROUPS = {
    "deepshap_methods.py": {
        "coalition_masks": ("coalition_masks",),
        "gradient_delta": ("gradient_delta",),
    },
    "deepshap_diagnostics.py": {
        "helpers": ("DTYPE", "_method", "_linear", "_prediction", "_max_abs"),
        "affine_fixture": ("affine_fixture",),
        "threshold_fixture": ("threshold_fixture",),
        "near_zero_fixture": ("near_zero_fixture",),
        "background_average_fixture": ("background_average_fixture",),
        "min_fixture": ("min_fixture",),
        "replacement_fixture": ("replacement_fixture",),
        "rescale_limits": ("diagnose_rescale_limits",),
        "near_zero": ("diagnose_near_zero",),
        "affine_controls": ("diagnose_affine_controls",),
        "background_average": ("diagnose_background_average",),
        "threshold": ("diagnose_threshold",),
        "cancellation": ("diagnose_cancellation",),
        "replacement_control": ("diagnose_replacement_control",),
        "threshold_play": ("threshold_play",),
        "aggregate": ("run_diagnostics",),
    },
    "deepshap_study.py": {
        "constants": (
            "FEATURES",
            "HIDDEN",
            "TRAINING_STEPS",
            "LEARNING_RATE",
            "WEIGHT_DECAY",
            "RIDGE_PENALTY",
            "MODEL_SEEDS",
            "SHUFFLE_SEEDS",
            "PRIMARY_BACKGROUND_SEED",
            "INDEPENDENT_BACKGROUND_SEEDS",
            "RANDOM_ORDER_SEED",
        ),
        "metadata": ("MODEL_KEYS", "FEATURE_NAMES", "MODEL_LABELS", "METHOD_NAMES", "UNITS"),
        "model_adapter": (
            "_model_from_state",
            "make_model",
        ),
        "normalization": (
            "_fold_mlp_state",
            "_fold_ridge_state",
            "_folding_error",
        ),
        "training_normalization": ("_training_normalization",),
        "train_mlp": ("_train_mlp",),
        "fold_mlp_result": ("_fold_mlp_result",),
        "fit_mlp": ("fit_one_mlp_control",),
        "fit_ridge": ("fit_ridge_control",),
        "fitting": ("fit_models",),
        "explanation": ("_explain_model", "explain_block"),
        "backgrounds": ("_background_indices",),
        "preparation": ("prepare_study",),
        "ranked_helpers": ("_ranked_curves",),
        "random_and_combine": ("_random_curves",),
        "performance": ("_performance_table", "_fixed_random_orders"),
        "replacement_evaluation": ("_evaluate_replacements",),
        "background_evaluation": ("_evaluate_background_variants",),
        "evaluation": ("evaluate_study",),
        "play": ("native_play",),
    },
}


def visible_module_source(filename: str, group: str) -> str:
    """Extract one coherent, current AST group for a learner-visible cell."""
    if Path(filename).name != filename or group not in VISIBLE_GROUPS.get(filename, {}):
        raise ValueError(f"Unsupported visible source insertion: {filename}:{group}")
    names = VISIBLE_GROUPS[filename][group]
    first_group = (filename, group) in {
        ("deepshap_diagnostics.py", "helpers"),
        ("deepshap_study.py", "constants"),
    }
    source = (HERE / filename).read_text()
    lines = source.splitlines(True)
    pieces: list[str] = []
    found: list[str] = []
    for index, node in enumerate(ast.parse(source).body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.ImportFrom) and (node.module == "__future__" or node.level > 0):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not first_group:
                continue
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assignment_names = [target.id for target in targets if isinstance(target, ast.Name)]
            selected = [name for name in assignment_names if name in names]
            if not selected:
                continue
            found.extend(selected)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name not in names:
                continue
            found.append(node.name)
        else:
            continue
        pieces.append("".join(lines[node.lineno - 1 : node.end_lineno]).rstrip())
    if tuple(found) != names:
        raise ValueError(
            f"Visible group {filename}:{group} drifted: expected {names}, found {tuple(found)}"
        )
    result = "\n\n".join(pieces).strip() + "\n"
    ast.parse(result)
    return result


def _expand_visible_insert(source: str) -> str:
    matches = INSERT.findall(source)
    if not matches:
        return source
    if len(matches) != 1:
        raise ValueError("A code cell may contain at most one visible-source insertion.")
    return visible_module_source(*matches[0])


def lesson_cells() -> list[dict]:
    source = (HERE / "lesson.md").read_text()
    if any(ord(character) < 32 and character not in "\n\t\r" for character in source):
        raise ValueError("lesson contains control characters; check LaTeX escaping")
    pieces = MARKER.split(source)
    if pieces[0].strip():
        raise ValueError("lesson content must begin with an ARENA cell marker")
    cells = []
    for kind, text in zip(pieces[1::2], pieces[2::2], strict=True):
        text = text.strip() + "\n"
        if kind == "code":
            if not text.startswith("```python\n") or not text.rstrip().endswith("```"):
                raise ValueError("each code cell needs one Python fence")
            text = text[len("```python\n") :].rstrip()[:-3].rstrip() + "\n"
            text = _expand_visible_insert(text)
            ast.parse(text)
        cell = {
            "id": hashlib.sha256(f"{len(cells)}:{kind}:{text}".encode()).hexdigest()[:12],
            "cell_type": kind,
            "metadata": {},
            "source": text.splitlines(True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return cells


def complete_from_answer(source: str, answer: str) -> str:
    """Replace only definitions containing ``NotImplementedError``."""
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    lines, answer_lines = source.splitlines(True), answer.splitlines(True)
    answers = {}
    for node in ast.parse(answer).body:
        if isinstance(node, definitions):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            answers[node.name] = "".join(answer_lines[start : node.end_lineno])
    for node in reversed(ast.parse(source).body):
        if isinstance(node, definitions):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            block = "".join(lines[start : node.end_lineno])
            if "raise NotImplementedError" in block:
                if node.name not in answers:
                    raise ValueError(f"this exercise's solution is missing {node.name}")
                lines[start : node.end_lineno] = [answers[node.name] + "\n"]
    completed = "".join(lines)
    if "raise NotImplementedError" in completed:
        raise ValueError("an exercise was not completed from its own answer")
    ast.parse(completed)
    return completed


def complete_learner(notebook: dict) -> dict:
    """Complete every exercise from its immediately following Solution dropdown."""
    result = copy.deepcopy(notebook)
    completed_names: list[str] = []
    for index, cell in enumerate(result["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or "raise NotImplementedError" not in source:
            continue
        following = []
        for other in result["cells"][index + 1 :]:
            if other["cell_type"] != "markdown":
                break
            following.append("".join(other["source"]))
        blocks = re.findall(
            r"<summary>Solution</summary>(.*?)</details>", "\n".join(following), re.S
        )
        if len(blocks) != 1:
            raise ValueError(f"cell {index} must have exactly one following exercise answer")
        answer = "\n\n".join(re.findall(r"```python\n(.*?)```", blocks[0], re.S))
        parsed = ast.parse(source)
        completed_names.extend(
            node.name
            for node in parsed.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and "raise NotImplementedError" in ast.get_source_segment(source, node)
        )
        cell["source"] = complete_from_answer(source, answer).splitlines(True)
    if tuple(completed_names) != EXERCISE_FUNCTIONS:
        raise ValueError(f"expected the nine exercises in order, got {tuple(completed_names)}")
    return result


def _definitions(source: str) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def validate_answer_parity(solved: dict) -> None:
    """Require the nine own answers to equal the readable method references."""
    reference = _definitions((HERE / "deepshap_methods.py").read_text())
    notebook: dict[str, ast.AST] = {}
    for cell in solved["cells"]:
        if cell["cell_type"] == "code":
            notebook.update(_definitions("".join(cell["source"])))
    for name in EXERCISE_FUNCTIONS:
        if name not in reference or name not in notebook:
            raise ValueError(f"missing answer/reference definition for {name}")
        if ast.dump(notebook[name], include_attributes=False) != ast.dump(
            reference[name], include_attributes=False
        ):
            raise ValueError(f"lesson answer drifted from deepshap_methods.py: {name}")


def build_notebooks() -> tuple[dict, dict]:
    metadata = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    learner = {**metadata, "cells": lesson_cells()}
    solved = complete_learner(learner)
    validate_answer_parity(solved)
    return learner, solved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--draft-output", type=Path)
    args = parser.parse_args()
    learner, solved = build_notebooks()
    if args.draft_output:
        args.draft_output.mkdir(parents=True, exist_ok=False)
        for suffix, notebook in (("exercises", learner), ("solutions", solved)):
            (args.draft_output / f"{STEM}_{suffix}.ipynb").write_text(
                json.dumps(notebook, indent=1) + "\n"
            )
    print(
        "Draft parses, exposes study/diagnostic source, and completes from its own "
        f"answers: {len(learner['cells'])} cells, {len(EXERCISE_FUNCTIONS)} exercises"
    )


if __name__ == "__main__":
    main()
