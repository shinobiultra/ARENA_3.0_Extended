"""Build draft 16.16 Shapley-Taylor / Faith-Shap notebooks from one lesson source.

Checkpoint and draft modes keep the learner surface reviewable while method, study,
reference, figure, and native-evidence contracts are still being frozen. Strict checks
refuse every unresolved marker. This builder never writes canonical course surfaces.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import html
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
STEM = "16.16_Where_Does_Higher_Order_Credit_Go"
PAGE_FILENAME = "16_[16.16]_Where_Does_Higher_Order_Credit_Go.md"
METHOD_MODULE = "interaction_methods.py"
MARKER = re.compile(r"^<!-- ARENA:cell (markdown|code) -->\s*$", re.M)
INSERT = re.compile(r"^# ARENA:insert ([A-Za-z0-9_.-]+):([a-z_]+)\s*$", re.M)
BLOCKER = "CHECKPOINT BLOCKER"
PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|PLACEHOLDER)\b|FILL FROM|unknown until execution|NATIVE_RESULT",
    re.I,
)

EXERCISE_GROUPS = (
    ("mobius_transform",),
    ("shapley_taylor_interactions",),
    ("faith_design_matrix", "faith_kernel_weights"),
    ("faith_shap_interactions",),
    ("faith_shap_closed_form",),
    ("reconstruct_game", "faithfulness_error"),
    ("masked_model_game",),
    ("contextual_pair_synergy",),
)
EXERCISE_FUNCTIONS = tuple(name for group in EXERCISE_GROUPS for name in group)
FIGURE_STEMS = (
    "01_exact_allocation_ledger",
    "02_why_faith_moves_singletons",
    "10_dividend_play",
    "03_native_game_and_controls",
    "04_fixed_row_interaction_graph",
    "05_allrow_pair_heatmaps",
    "06_heldout_pair_order_capture",
    "07_order_two_reconstruction",
    "08_background_changes_the_game",
    "09_contextual_pair_interventions",
    "11_donor_context_play",
)

# Group names are intentionally semantic and short. They are candidate bindings until
# their owners stabilize the source modules; unresolved files and groups block strict mode.
VISIBLE_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "interaction_diagnostics.py": {
        "exact_game_fixture": ("_fixture_mobius", "exact_game_fixture"),
        "mobius_diagnostic": ("diagnose_mobius_games",),
        "taylor_diagnostic": ("diagnose_shapley_taylor",),
        "faith_design_diagnostic": ("diagnose_faith_design",),
        "faith_solver_diagnostic": ("diagnose_faith_solver",),
        "reconstruction_diagnostic": ("diagnose_reconstruction",),
        "model_game_diagnostic": (
            "_AdditiveDiagnosticModel",
            "_additive_formula",
            "diagnose_model_game",
        ),
        "pair_context_diagnostic": ("diagnose_pair_context",),
    },
    "interaction_study.py": {
        'protocol_constants': ('FEATURES', 'HIDDEN', 'EMPIRICAL_ORDER', 'TRAINING_STEPS', 'LEARNING_RATE', 'WEIGHT_DECAY', 'RIDGE_PENALTY', 'FORWARD_BATCH', 'DISTANCE_BATCH', 'MODEL_KEYS', 'MODEL_SEEDS', 'SHUFFLE_SEEDS', 'BACKGROUND_SPECS', 'METHOD_KEYS', 'METHOD_NAMES', 'FEATURE_NAMES', 'FEATURE_UNITS', 'TARGET_NAME', 'TARGET_UNIT', 'RNG_NAMESPACE', 'RANDOM_EXACT_SEED', 'WITHIN_SIZE_SEED', 'RANDOM_PAIR_SEED', 'WITHIN_SIZE_REPLICATES', 'RANDOM_PAIR_ORDERS', 'SUPPORT_DONORS', 'DEFAULT_PLAY'),
        'model_fit_and_hash_gate': ('_train_mlp', 'fit_known_models', 'verify_known_reproduction'),
        'donor_game_construction': ('donor_replacement_games', 'evaluate_model_games'),
        'exact_fixtures_and_methods': ('pairwise_sii', 'all_pair_context_synergies', 'explain_complete_games'),
        'explanation_preparation': ('explanation_pair_rankings', 'fixed_random_pair_orders', 'prepare_study'),
        'audit_evaluation': ('pair_capture_curves', 'within_size_shuffle_control', 'background_sensitivity', 'evaluate_study'),
        'native_play': ('native_play',),
        'empirical_play': ('empirical_context_play',),
    },
    "interaction_reference.py": {
        "reference_check": (
            "reference_check",
            "validate_exact_reference_suite",
            "finalize_study_reference",
        ),
    },
}

REQUIRED_SOURCES = (
    "interaction_data.py",
    METHOD_MODULE,
    "interaction_methods_tests.py",
    "interaction_diagnostics.py",
    "interaction_diagnostics_tests.py",
    "interaction_plots.py",
    "interaction_plots_tests.py",
    "interaction_study.py",
    "interaction_study_tests.py",
    "interaction_reference.py",
    "interaction_reference_tests.py",
    "protocol.md",
    "protocol_freeze.json",
    "pre_run_resolution.json",
    "data_manifest.json",
    "run_native.py",
    "execute_learner.py",
    "reproduce_native.py",
    "publish_study.py",
)

PROVIDED_TITLES = {
    "protocol_constants": "Provided study settings and fixed defaults",
    "model_fit_and_hash_gate": "Provided model training and reproduction checks",
    "donor_game_construction": "Provided complete-game runner using your donor method",
    "exact_fixtures_and_methods": "Provided comparison runner using your interaction methods",
    "explanation_preparation": "Provided explanation preparation and pair ranking",
    "audit_evaluation": "Provided held-out evaluation and controls",
    "reference_check": "Provided primary-paper and package reference adapter",
    "native_play": "Provided assembly of the fixed local comparison",
    "empirical_play": "Provided donor-context microscope using your methods",
}


def visible_module_source(filename: str, group: str) -> str:
    """Extract one current source group for a learner-visible code cell."""
    if Path(filename).name != filename or group not in VISIBLE_GROUPS.get(filename, {}):
        raise ValueError(f"Unsupported visible source insertion: {filename}:{group}")
    names = VISIBLE_GROUPS[filename][group]
    if not names:
        raise ValueError(f"Visible source group is not frozen: {filename}:{group}")
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
    """Return all blockers which prevent a strict executable build."""
    source = source if source is not None else (HERE / "lesson.md").read_text()
    unresolved: list[str] = []
    for filename in REQUIRED_SOURCES:
        if not (HERE / filename).is_file():
            unresolved.append(f"missing required source {filename}")
    for filename, group in INSERT.findall(source):
        if not (HERE / filename).is_file():
            unresolved.append(f"missing source {filename} for group {group}")
        elif group not in VISIBLE_GROUPS.get(filename, {}):
            unresolved.append(f"unmapped visible group {filename}:{group}")
        elif not VISIBLE_GROUPS[filename][group]:
            unresolved.append(f"unfrozen visible group {filename}:{group}")
        else:
            try:
                visible_module_source(filename, group)
            except ValueError as error:
                unresolved.append(str(error))
    for line_number, line in enumerate(source.splitlines(), 1):
        if BLOCKER in line:
            unresolved.append(f"lesson.md:{line_number}: {line.strip()}")
        elif PLACEHOLDER.search(line):
            unresolved.append(f"lesson.md:{line_number}: unresolved placeholder: {line.strip()}")
    return list(dict.fromkeys(unresolved))


def _expand_visible_insert(source: str, allow_unresolved: bool) -> str:
    matches = INSERT.findall(source)
    if not matches:
        return source
    if len(matches) != 1:
        raise ValueError("A code cell may contain at most one visible-source insertion.")
    filename, group = matches[0]
    if (
        (HERE / filename).is_file()
        and group in VISIBLE_GROUPS.get(filename, {})
        and VISIBLE_GROUPS[filename][group]
    ):
        try:
            return visible_module_source(filename, group)
        except ValueError:
            if not allow_unresolved:
                raise
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
        provided_title = None
        if kind == "code":
            if not text.startswith("```python\n") or not text.rstrip().endswith("```"):
                raise ValueError("each code cell needs one Python fence")
            text = text[len("```python\n") :].rstrip()[:-3].rstrip() + "\n"
            inserts = INSERT.findall(text)
            if inserts:
                provided_title = PROVIDED_TITLES.get(inserts[0][1])
            elif text.startswith("from __future__ import annotations"):
                provided_title = "Setup: imports and supplied helpers"
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
            if provided_title:
                cell["metadata"] = {
                    "jupyter": {"source_hidden": True},
                    "arena": {"provided_code": provided_title},
                }
        cells.append(cell)
    return cells


def complete_from_answer(source: str, answer: str) -> str:
    """Replace only top-level definitions containing ``NotImplementedError``."""
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    lines, answer_lines = source.splitlines(True), answer.splitlines(True)
    answers: dict[str, str] = {}
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
    completed_groups: list[tuple[str, ...]] = []
    solution_dropdown_cells: set[int] = set()
    for index, cell in enumerate(result["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or "raise NotImplementedError" not in source:
            continue
        following = []
        for other_index, other in enumerate(result["cells"][index + 1 :], index + 1):
            if other["cell_type"] != "markdown":
                break
            other_source = "".join(other["source"])
            following.append(other_source)
            if "<summary>Solution</summary>" in other_source:
                solution_dropdown_cells.add(other_index)
        blocks = re.findall(
            r"<summary>Solution</summary>(.*?)</details>", "\n".join(following), re.S
        )
        if len(blocks) != 1:
            raise ValueError(f"cell {index} must have exactly one following exercise answer")
        answer = "\n\n".join(re.findall(r"```python\n(.*?)```", blocks[0], re.S))
        names = []
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                block = ast.get_source_segment(source, node) or ""
                if "raise NotImplementedError" in block:
                    names.append(node.name)
        completed_groups.append(tuple(names))
        cell["source"] = complete_from_answer(source, answer).splitlines(True)
    if tuple(completed_groups) != EXERCISE_GROUPS:
        raise ValueError(f"expected exercise groups {EXERCISE_GROUPS}, got {completed_groups}")
    result["cells"] = [
        cell for index, cell in enumerate(result["cells"]) if index not in solution_dropdown_cells
    ]
    _set_variant_title(result, "Solutions")
    return result


def _set_variant_title(notebook: dict, variant: str) -> None:
    """Give exercise and solution notebooks distinct original-ARENA titles."""
    if variant not in {"Exercises", "Solutions"}:
        raise ValueError(f"Unknown notebook variant: {variant}")
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        title = "# [16.16] Where Does Higher-Order Credit Go?"
        if not source.startswith(title):
            continue
        first, *rest = source.splitlines(True)
        first = re.sub(r"\s+—\s+(?:Exercises|Solutions)\s*$", "", first.rstrip("\n"))
        cell["source"] = [first + f" — {variant}\n", *rest]
        return
    raise ValueError("Could not find the 16.16 notebook title cell.")


def _definitions(source: str) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def validate_answer_parity(solved: dict, *, required: bool) -> None:
    """Require all ten own answers to equal the stabilized readable method source."""
    path = HERE / METHOD_MODULE
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return
    reference = _definitions(path.read_text())
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
            raise ValueError(f"lesson answer drifted from {METHOD_MODULE}: {name}")


def validate_exercise_surface(source: str) -> None:
    """Require the original-ARENA exercise and dropdown surface."""
    expected = len(EXERCISE_GROUPS)
    headings = list(re.finditer(r"^### Exercise [1-8]\b", source, re.M))
    if len(headings) != expected:
        raise ValueError(f"expected {expected} numbered implementation exercises")
    for index, (heading, group) in enumerate(zip(headings, EXERCISE_GROUPS, strict=True)):
        stop = headings[index + 1].start() if index + 1 < len(headings) else len(source)
        segment = source[heading.start() : stop]
        for token in ("Difficulty:", "Importance:", "You should spend", "Expected output"):
            if token not in segment:
                raise ValueError(f"exercise {index + 1} is missing {token!r}")
        if segment.count("<summary>Solution</summary>") != 1:
            raise ValueError(f"exercise {index + 1} needs one Solution dropdown")
        if segment.count("<summary>Help -") < 1:
            raise ValueError(f"exercise {index + 1} needs at least one Help dropdown")
        if segment.count("<summary>Interpretation</summary>") != 1:
            raise ValueError(f"exercise {index + 1} needs one Interpretation dropdown")
        for name in group:
            call = f"tests.test_{name}(fn={name})"
            if segment.count(call) != 1:
                raise ValueError(f"exercise {index + 1} must call its test exactly once: {call}")
            if segment.index(call) > segment.index("<summary>Expected output</summary>"):
                raise ValueError(f"exercise {index + 1} must run its test before expected output")
        help_positions = [match.start() for match in re.finditer("<summary>Help -", segment)]
        if not (
            segment.index("<summary>Expected output</summary>")
            < min(help_positions)
            <= max(help_positions)
            < segment.index("<summary>Interpretation</summary>")
            < segment.index("<summary>Solution</summary>")
        ):
            raise ValueError(f"exercise {index + 1} must keep expected output, help, interpretation, then solution")
    if source.count("## Try It Yourself") < 2:
        raise ValueError("lesson needs both editable play cells")
    if "## Core Question" not in source or "runs on CPU" not in source:
        raise ValueError("lesson needs its Core Question and CPU contract")
    stems = tuple(re.findall(r'show_figure\(figure, "([0-9]{2}_[a-z0-9_]+)"\)', source))
    if stems != FIGURE_STEMS:
        raise ValueError(f"expected the frozen figure order {FIGURE_STEMS}, got {stems}")


def expected_page_path() -> Path:
    """Return the only canonical instruction-page path accepted for 16.16."""
    return HERE.parents[1] / "instructions" / "pages" / PAGE_FILENAME


def render_instruction_code(cell: dict) -> str:
    """Keep supplied code expandable on the page, with every byte inspectable."""
    source = "".join(cell["source"]).strip()
    code = "```python\n" + source + "\n```"
    title = cell.get("metadata", {}).get("arena", {}).get("provided_code")
    if title:
        return f"<details>\n<summary>{html.escape(title)}</summary>\n\n{code}\n\n</details>"
    return code


def validate_page_surface(page_source: str) -> None:
    """Check the rendered page keeps the complete learner-facing course surface."""
    required = (
        "# [16.16] Where Does Higher-Order Credit Go?",
        "## Core Question",
        "## Try It Yourself: reroute a dividend",
        "## Try It Yourself: donor-context microscope",
    )
    missing = [item for item in required if item not in page_source]
    if missing:
        raise ValueError(f"rendered page is missing required sections: {missing}")
    if page_source.count("<summary>Solution</summary>") != len(EXERCISE_GROUPS):
        raise ValueError("rendered page must retain all eight learner Solution dropdowns")
    for name in EXERCISE_FUNCTIONS:
        if page_source.count(f"tests.test_{name}(fn={name})") != 1:
            raise ValueError(f"rendered page lost the immediate test for {name}")
    for stem in FIGURE_STEMS:
        if f'show_figure(figure, "{stem}")' not in page_source:
            raise ValueError(f"rendered page lost the literal figure call for {stem}")
    cursor = 0
    for cell in lesson_cells(allow_unresolved=True):
        if cell["cell_type"] != "code":
            continue
        fragment = render_instruction_code(cell)
        position = page_source.find(fragment, cursor)
        if position < 0:
            raise ValueError(f"rendered page lost or reordered learner code cell {cell['id']}")
        cursor = position + len(fragment)


def build_checkpoint() -> tuple[dict, dict, list[str]]:
    """Parse and own-answer-complete the draft without requiring frozen modules."""
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
    _set_variant_title(learner, "Exercises")
    solved = complete_learner(learner)
    validate_answer_parity(solved, required=False)
    return learner, solved, unresolved_requirements(source)


def build_notebooks() -> tuple[dict, dict]:
    """Build executable notebooks only after every strict requirement is resolved."""
    learner, _, blockers = build_checkpoint()
    if blockers:
        raise ValueError("Strict build blocked:\n- " + "\n- ".join(blockers))
    learner["cells"] = lesson_cells(allow_unresolved=False)
    _set_variant_title(learner, "Exercises")
    solved = complete_learner(learner)
    validate_answer_parity(solved, required=True)
    return learner, solved


def write_draft(directory: Path) -> tuple[int, list[str]]:
    """Write explicitly named draft notebooks while preserving unresolved markers."""
    learner, solved, blockers = build_checkpoint()
    directory.mkdir(parents=True, exist_ok=False)
    for suffix, notebook in (("exercises", learner), ("solutions", solved)):
        (directory / f"DRAFT_{STEM}_{suffix}.ipynb").write_text(
            json.dumps(notebook, indent=1) + "\n"
        )
    return len(learner["cells"]), blockers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--checkpoint-check", action="store_true")
    parser.add_argument("--draft-output", type=Path)
    args = parser.parse_args()
    if args.draft_output:
        count, blockers = write_draft(args.draft_output)
        print(f"Wrote explicit draft notebooks with {count} cells.")
        print("Canonical publication remains blocked by:")
        for blocker in blockers:
            print(f"- {blocker}")
        return
    if args.checkpoint_check:
        learner, _, blockers = build_checkpoint()
        print(
            f"Checkpoint parses and completes {len(EXERCISE_FUNCTIONS)} functions across "
            f"{len(EXERCISE_GROUPS)} exercises and {len(learner['cells'])} cells."
        )
        print("Unresolved strict-build requirements:")
        for blocker in blockers:
            print(f"- {blocker}")
        return
    if not args.check:
        parser.error("choose --checkpoint-check, --draft-output, or --check")
    learner, _ = build_notebooks()
    page_path = expected_page_path()
    if not page_path.is_file():
        raise ValueError(f"Publication is incomplete: missing canonical instruction page {page_path}")
    validate_page_surface(page_path.read_text())
    print(
        "Strict learner build passes from its own answers: "
        f"{len(learner['cells'])} cells, {len(EXERCISE_GROUPS)} exercises, "
        f"{len(EXERCISE_FUNCTIONS)} functions"
    )


if __name__ == "__main__":
    main()
