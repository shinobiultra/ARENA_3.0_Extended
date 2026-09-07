"""Build draft 16.15 Expected Gradients notebooks from one readable lesson source.

The checkpoint can validate its nine current method answers before diagnostics and
study modules exist. Strict ``--check`` and ``--draft-output`` refuse unresolved
source markers, study calls, or evidence placeholders.
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
STEM = "16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate"
MARKER = re.compile(r"^<!-- ARENA:cell (markdown|code) -->\s*$", re.M)
INSERT = re.compile(r"^# ARENA:insert ([A-Za-z0-9_.-]+):([a-z_]+)\s*$", re.M)
BLOCKER = "CHECKPOINT BLOCKER"
PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|PLACEHOLDER)\b|FILL FROM|unknown until execution",
    re.I,
)
EXERCISE_FUNCTIONS = (
    "sample_path_schedule",
    "interpolate_paths",
    "input_gradients",
    "expected_gradient_samples",
    "summarize_expected_gradients",
    "expected_gradients",
    "gauss_legendre_pairs",
    "noisy_expected_gradient_samples",
    "ranked_coalition_curves",
)

# These groups are intentionally small and semantic. They expose the diagnostic
# fixtures and checks at the first point where the learner has implemented their
# dependencies, while keeping plotting and serialization plumbing in modules.
VISIBLE_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "gradientshap_diagnostics.py": {
        "mechanical_helpers": (
            "DTYPE",
            "_method",
            "_linear",
            "_prediction",
            "_max_abs",
            "_relu_model",
            "_threshold_model",
            "_quadratic_model",
            "_two_feature_shapley",
        ),
        "schedule_fixture": ("schedule_fixture",),
        "schedule_diagnostic": ("diagnose_schedule",),
        "path_fixture": ("path_fixture",),
        "path_diagnostic": ("diagnose_interpolation",),
        "gradient_fixture": ("gradient_fixture",),
        "gradient_diagnostic": ("diagnose_gradients",),
        "sample_fixture": ("sample_fixture",),
        "sample_diagnostic": ("diagnose_expected_gradient_samples",),
        "convergence_diagnostic": ("diagnose_convergence",),
        "estimator_fixture": ("estimator_fixture",),
        "estimator_diagnostic": ("diagnose_expected_gradients",),
        "quadrature_fixture": ("quadrature_fixture",),
        "quadrature_diagnostic": ("diagnose_quadrature",),
        "background_diagnostic": ("diagnose_background_average",),
        "noise_fixture": ("noise_fixture",),
        "noise_diagnostic": ("diagnose_noise_target",),
        "threshold_fixture": ("threshold_fixture",),
        "kink_fixture": ("kink_fixture",),
        "replacement_fixture": ("replacement_fixture",),
        "game_diagnostic": ("diagnose_path_vs_coalition",),
        "threshold_play": ("threshold_play", "run_diagnostics"),
    },
    "gradientshap_study.py": {
        "protocol_constants": (
            "FEATURES",
            "HIDDEN",
            "TRAINING_STEPS",
            "LEARNING_RATE",
            "WEIGHT_DECAY",
            "RIDGE_PENALTY",
            "MODEL_KEYS",
            "MODEL_SEEDS",
            "SHUFFLE_SEEDS",
            "BACKGROUND_SPECS",
            "BUDGETS",
            "QUADRATURE_NODES",
            "NOISE_SIGMAS",
            "PRIMARY_SCHEDULE_SEED",
            "REPLICATE_SCHEDULE_SEED",
            "NOISE_SCHEDULE_SEED",
            "REFERENCE_SEEDS",
            "RANDOM_ORDER_SEED",
            "REPLICATES",
            "REPLICATE_MODELS",
            "PATH_DISTANCE_SAMPLES",
            "COALITION_DISTANCE_DONORS",
            "FORWARD_BATCH",
            "DISTANCE_BATCH",
        ),
        "study_metadata": (
            "FEATURE_NAMES",
            "MODEL_LABELS",
            "UNITS",
            "METHOD_NAMES",
            "DEEP_METHOD_NAMES",
        ),
        "model_reconstruction": ("_model_from_state",),
        "training_normalization": ("_training_normalization",),
        "train_and_fold": ("_train_mlp", "_fold_mlp_state", "_fold_ridge_state"),
        "fit_mlp": ("fit_one_mlp_control",),
        "fit_ridge": ("fit_ridge_control",),
        "fit_models": ("fit_models",),
        "verify_states": ("verify_16_14_reproduction",),
        "backgrounds_and_schedule": ("_background_indices", "_primary_schedule"),
        "monte_carlo_block": ("_monte_carlo_block",),
        "quadrature_block": ("_quadrature_block",),
        "coalition_hybrids": ("_coalition_hybrid_predictions",),
        "comparison_baselines": ("_comparison_baselines",),
        "global_rankings": ("_global_rankings",),
        "prepare_study": ("prepare_study",),
        "performance": ("_performance_table",),
        "background_evaluation": ("_background_block", "_background_comparison"),
        "noise_evaluation": ("_standard_normal_schedule", "_noise_comparison"),
        "replicate_evaluation": ("_replicate_seed", "_replicate_comparison"),
        "distance_evaluation": ("_nearest_train_distance", "_distance_comparison"),
        "replacement_evaluation": ("_replacement_evaluation",),
        "evaluate_study": ("evaluate_study",),
        "native_play": ("native_play",),
    },
    "gradientshap_reference.py": {
        "reference_check": ("reference_check",),
    },
}


def visible_module_source(filename: str, group: str) -> str:
    """Extract one current, coherent group for a learner-visible code cell."""
    if Path(filename).name != filename or group not in VISIBLE_GROUPS.get(filename, {}):
        raise ValueError(f"Unsupported visible source insertion: {filename}:{group}")
    names = VISIBLE_GROUPS[filename][group]
    source = (HERE / filename).read_text()
    lines = source.splitlines(True)
    pieces: list[str] = []
    found: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
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


def unresolved_requirements(source: str | None = None) -> list[str]:
    """List every unresolved source marker or placeholder blocking generation."""
    source = source if source is not None else (HERE / "lesson.md").read_text()
    unresolved = []
    for filename, group in INSERT.findall(source):
        if not (HERE / filename).is_file():
            unresolved.append(f"missing source {filename} for group {group}")
        elif group not in VISIBLE_GROUPS.get(filename, {}):
            unresolved.append(f"unmapped visible group {filename}:{group}")
    for line_number, line in enumerate(source.splitlines(), 1):
        if BLOCKER in line:
            unresolved.append(f"lesson.md:{line_number}: {line.strip()}")
        elif PLACEHOLDER.search(line):
            unresolved.append(f"lesson.md:{line_number}: unresolved placeholder: {line.strip()}")
    return unresolved


def _expand_visible_insert(source: str, allow_unresolved: bool) -> str:
    matches = INSERT.findall(source)
    if not matches:
        return source
    if len(matches) != 1:
        raise ValueError("A code cell may contain at most one visible-source insertion.")
    filename, group = matches[0]
    if (HERE / filename).is_file() and group in VISIBLE_GROUPS.get(filename, {}):
        return visible_module_source(filename, group)
    if allow_unresolved:
        return source
    raise ValueError(f"Unresolved visible source insertion: {filename}:{group}")


def lesson_cells(*, allow_unresolved: bool = False) -> list[dict]:
    source = (HERE / "lesson.md").read_text()
    if any(character in source for character in ("\x00", "\x08", "\x0c")):
        raise ValueError("lesson contains a control character; check LaTeX escaping")
    if not allow_unresolved:
        blockers = unresolved_requirements(source)
        if blockers:
            raise ValueError("Strict build blocked:\n- " + "\n- ".join(blockers))
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
            text = _expand_visible_insert(text, allow_unresolved)
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
        if not isinstance(node, definitions):
            continue
        start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
        block = "".join(lines[start : node.end_lineno])
        if "raise NotImplementedError" not in block:
            continue
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
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                block = ast.get_source_segment(source, node) or ""
                if "raise NotImplementedError" in block:
                    completed_names.append(node.name)
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
    """Require all nine own answers to equal the current readable method source."""
    reference = _definitions((HERE / "gradientshap_methods.py").read_text())
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
            raise ValueError(f"lesson answer drifted from gradientshap_methods.py: {name}")


def validate_exercise_surface(source: str) -> None:
    """Check the exact original-ARENA metadata and dropdown surface at checkpoint time."""
    for token in ("Difficulty:", "Importance:", "You should spend", "Expected output"):
        if source.count(token) < len(EXERCISE_FUNCTIONS):
            raise ValueError(f"lesson is missing exercise surface token {token!r}")
    if source.count("<summary>Solution</summary>") != len(EXERCISE_FUNCTIONS):
        raise ValueError("every exercise needs exactly one following Solution dropdown")
    if "## Core Question" not in source or "runs on CPU" not in source:
        raise ValueError("lesson needs its Core Question and CPU contract")


def build_checkpoint() -> tuple[dict, dict, list[str]]:
    """Parse and own-answer-complete the current checkpoint without expanding missing modules."""
    source = (HERE / "lesson.md").read_text()
    validate_exercise_surface(source)
    cells = lesson_cells(allow_unresolved=True)
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
    learner = {**metadata, "cells": cells}
    solved = complete_learner(learner)
    validate_answer_parity(solved)
    return learner, solved, unresolved_requirements(source)


def build_notebooks() -> tuple[dict, dict]:
    """Build only after every visible source and execution placeholder is resolved."""
    learner, solved, blockers = build_checkpoint()
    if blockers:
        raise ValueError("Strict build blocked:\n- " + "\n- ".join(blockers))
    # Reparse with strict expansion rather than returning checkpoint marker cells.
    learner["cells"] = lesson_cells(allow_unresolved=False)
    solved = complete_learner(learner)
    validate_answer_parity(solved)
    return learner, solved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--checkpoint-check", action="store_true")
    parser.add_argument("--draft-output", type=Path)
    args = parser.parse_args()
    if args.checkpoint_check:
        learner, _, blockers = build_checkpoint()
        print(
            f"Checkpoint parses and completes {len(EXERCISE_FUNCTIONS)} AST-matched answers "
            f"across {len(learner['cells'])} cells."
        )
        print("Unresolved strict-build requirements:")
        for blocker in blockers:
            print(f"- {blocker}")
        return
    learner, solved = build_notebooks()
    if args.draft_output:
        args.draft_output.mkdir(parents=True, exist_ok=False)
        for suffix, notebook in (("exercises", learner), ("solutions", solved)):
            (args.draft_output / f"{STEM}_{suffix}.ipynb").write_text(
                json.dumps(notebook, indent=1) + "\n"
            )
    print(
        "Strict draft parses, exposes study/diagnostic source, and completes from its own "
        f"answers: {len(learner['cells'])} cells, {len(EXERCISE_FUNCTIONS)} exercises"
    )


if __name__ == "__main__":
    main()
