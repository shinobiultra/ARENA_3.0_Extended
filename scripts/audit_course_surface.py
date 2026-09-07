"""Audit the learner-facing course surface for extension chapters."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "infrastructure/core/config.yaml"
EXTENSION_ORIGINAL_SECTIONS = {"0.6", "1.6"}
REQUIRED_STREAMLIT_THEME_KEYS = {
    "primaryColor",
    "backgroundColor",
    "secondaryBackgroundColor",
    "textColor",
}
REQUIRED_INSTRUCTION_REQUIREMENTS = {
    "torch",
    "streamlit==1.45.0",
    "pyyaml",
}
REQUIRED_NOTEBOOK_VERIFICATION_STRINGS = (
    "def run_gpu_test(max_vram_gb: float = 24.0)",
    "def run_full_experiment(max_vram_gb: float = 24.0)",
)
REQUIRED_NOTEBOOK_EVIDENCE_SURFACES = (
    "verification_report.json",
    "run_gelu1l_signature_result",
    "prepare_gelu1l_state",
    "run_transformerlens_acdc_preflight",
)
TREESHAP_STUDENT_METHODS = {
    "tree_predict",
    "tree_path_game",
    "exact_shapley",
    "extend_path",
    "unwind_path",
    "tree_shap",
    "ensemble_shap",
    "interventional_game",
    "saabas",
}
DEEPSHAP_STUDENT_METHODS = {
    "paired_forward",
    "rescale_slope",
    "rescale_multipliers",
    "single_reference",
    "background_deepshap",
    "coalition_values",
    "exact_shapley",
    "reveal_cancel_groups",
    "ranked_replacement_curves",
}
GRADIENTSHAP_STUDENT_METHODS = {
    "sample_path_schedule",
    "interpolate_paths",
    "input_gradients",
    "expected_gradient_samples",
    "summarize_expected_gradients",
    "expected_gradients",
    "gauss_legendre_pairs",
    "noisy_expected_gradient_samples",
    "ranked_coalition_curves",
}
SHAPLEY_INTERACTION_STUDENT_METHODS = {
    "mobius_transform",
    "shapley_taylor_interactions",
    "faith_design_matrix",
    "faith_kernel_weights",
    "faith_shap_interactions",
    "faith_shap_closed_form",
    "reconstruct_game",
    "faithfulness_error",
    "masked_model_game",
    "contextual_pair_synergy",
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def is_extension_section(number: str) -> bool:
    if number in EXTENSION_ORIGINAL_SECTIONS:
        return True
    try:
        return int(number.split(".", maxsplit=1)[0]) >= 5
    except ValueError:
        return False


def extension_chapter_names(config: dict[str, Any]) -> list[str]:
    chapter_names = config["chapter_names"]
    return [chapter_names[index] for index in sorted(chapter_names) if int(index) >= 5]


def _chapter_number_from_name(config: dict[str, Any], chapter_name: str) -> int:
    for number, name in config["chapter_names"].items():
        if name == chapter_name:
            return int(number)
    raise KeyError(chapter_name)


def _requirement_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _page_has_arena_shape(text: str, number: str, title: str) -> bool:
    has_title = f"# [{number}] {title}" in text
    has_intro = re.search(r"^#{1,2}\s+(?:Introduction|Core Question)\s*$", text, re.M | re.I) is not None
    has_learning_objectives = re.search(r"^#{2,5}\s+(?:Content & )?Learning Objectives\s*$", text, re.M | re.I) is not None
    has_exercise = "Exercise" in text
    has_visible_test = "tests." in text or "run_smoke_test" in text
    has_runtime_contract = ("run_gpu_test" in text or "CUDA" in text or "GPU Contract" in text
                            or re.search(r"\b(?:runs? on|use) CPU\b", text) is not None
                            or re.search(r"REQUIRES_GPU\s*=\s*(?:True|False)\b", text) is not None)
    return all(
        [
            has_title,
            has_intro,
            has_learning_objectives,
            has_exercise,
            has_visible_test,
            has_runtime_contract,
        ]
    )


def _exercise_notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _exercise_notebook_declares_verification_contract(path: Path) -> bool:
    if path.name == "16.17_One_Query_Many_Interactions_exercises.ipynb":
        from scripts.shapiq_learner_contract import has_learner_contract

        return has_learner_contract(path)
    source = _exercise_notebook_source(path)
    has_contract = all(required in source for required in REQUIRED_NOTEBOOK_VERIFICATION_STRINGS)
    has_evidence_surface = any(surface in source for surface in REQUIRED_NOTEBOOK_EVIDENCE_SURFACES)
    # Legacy interface detection is only a surface check, never scientific
    # acceptance. Modern lessons execute their methods and keep reports outside
    # the learning arc; their full executor supplies the verification interface.
    return (has_contract and has_evidence_surface) or _has_direct_experiment_contract(path)


def _reachable_local_calls(trees: list[ast.Module], roots: set[str]) -> set[str]:
    """Follow calls through notebook-defined functions from visible experiment cells."""
    functions = {
        node.name: node
        for tree in trees
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reached: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        function = functions.get(name)
        if function is None:
            continue
        pending.extend(
            call.func.id
            for call in ast.walk(function)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        )
    return reached


def _reachable_student_method_calls(
    trees: list[ast.Module], roots: set[str], student_methods: set[str]
) -> set[str]:
    """Follow direct and methods-namespace calls through visible local functions."""
    functions = {
        node.name: node
        for tree in trees
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reached: set[str] = set()
    invoked: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        function = functions.get(name)
        if function is None:
            continue
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        pending.extend(
            call.func.id for call in calls if isinstance(call.func, ast.Name)
        )
        invoked.update(
            call.func.id
            for call in calls
            if isinstance(call.func, ast.Name) and call.func.id in student_methods
        )
        invoked.update(
            call.func.attr
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "methods"
            and call.func.attr in student_methods
        )
        for assignment in ast.walk(function):
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            value = assignment.value
            targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            if (
                not isinstance(value, ast.Call)
                or not isinstance(value.func, ast.Name)
                or value.func.id != "_method"
                or len(value.args) < 2
                or not isinstance(value.args[0], ast.Name)
                or value.args[0].id != "methods"
                or not isinstance(value.args[1], ast.Constant)
                or value.args[1].value not in student_methods
            ):
                continue
            aliases = {target.id for target in targets if isinstance(target, ast.Name)}
            if any(
                isinstance(call.func, ast.Name) and call.func.id in aliases
                for call in calls
            ):
                invoked.add(value.args[1].value)
        for method in invoked - reached:
            if method in functions:
                pending.append(method)
    return invoked


def _visible_callable_method_arguments(
    trees: list[ast.Module], roots: set[str], student_methods: set[str]
) -> set[str]:
    """Recognize a visible method passed to a helper that actually invokes it."""
    functions = {
        node.name: node
        for tree in trees
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    invoked: set[str] = set()
    for tree in trees:
        for outer in tree.body:
            if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for call in (node for node in ast.walk(outer) if isinstance(node, ast.Call)):
                if not isinstance(call.func, ast.Name) or call.func.id not in roots:
                    continue
                function = functions.get(call.func.id)
                if function is None:
                    continue
                parameters = [item.arg for item in function.args.args]
                bindings = dict(zip(parameters, call.args))
                bindings.update(
                    (keyword.arg, keyword.value)
                    for keyword in call.keywords
                    if keyword.arg is not None
                )
                called_names = {
                    item.func.id
                    for item in ast.walk(function)
                    if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                }
                invoked.update(
                    value.id
                    for parameter, value in bindings.items()
                    if parameter in called_names
                    and isinstance(value, ast.Name)
                    and value.id in student_methods
                )
    return invoked


def _has_visible_method_namespace(trees: list[ast.Module], methods: set[str]) -> bool:
    """Require the visible namespace to bind every taught name to its own answer."""
    for tree in trees:
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "methods"
                for target in node.targets
            ):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "SimpleNamespace"
            ):
                continue
            bindings = {
                keyword.arg: keyword.value.id
                for keyword in value.keywords
                if keyword.arg and isinstance(keyword.value, ast.Name)
            }
            return all(bindings.get(name) == name for name in methods)
    return False


def _has_exact_visible_method_namespace(
    trees: list[ast.Module], methods: set[str]
) -> bool:
    """Require one namespace whose complete public surface is learner-defined."""
    namespaces: list[dict[str, str]] = []
    for tree in trees:
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "methods"
                for target in node.targets
            ):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "SimpleNamespace"
            ):
                return False
            if any(
                keyword.arg is None or not isinstance(keyword.value, ast.Name)
                for keyword in value.keywords
            ):
                return False
            namespaces.append(
                {keyword.arg: keyword.value.id for keyword in value.keywords}
            )
    expected = {name: name for name in methods}
    return len(namespaces) == 1 and namespaces[0] == expected


def _visible_answers_match_source(
    here: Path,
    filename: str,
    answers: list[ast.Module],
    methods: set[str],
) -> bool:
    """Bind each visible solution to the course-owned implementation AST."""
    try:
        source = {
            node.name: node
            for node in ast.parse((here / filename).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    except (OSError, SyntaxError):
        return False
    visible = {
        node.name: node
        for tree in answers
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return all(
        name in source
        and name in visible
        and ast.dump(source[name], include_attributes=False)
        == ast.dump(visible[name], include_attributes=False)
        for name in methods
    )


def _deepshap_visible_sequence(cells: list[dict[str, Any]]) -> bool:
    """Check the actual fit, preparation, audit, reference and persistence calls."""
    positions: dict[str, list[int]] = {}
    saved: dict[str, list[int]] = {"fitted.pt": [], "prepared.pt": []}
    audit_loads: list[int] = []
    for cell_index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            return False
        for outer in tree.body:
            if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for call in (node for node in ast.walk(outer) if isinstance(node, ast.Call)):
                name = ast.unparse(call.func)
                positions.setdefault(name, []).append(cell_index)
                if (
                    name == "save_payload"
                    and len(call.args) >= 2
                    and isinstance(call.args[1], ast.Constant)
                    and call.args[1].value in saved
                ):
                    saved[call.args[1].value].append(cell_index)
                if (
                    name == "load_split"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "audit"
                ):
                    audit_loads.append(cell_index)
    required = {
        "fit_models",
        "prepare_study",
        "write_before_audit",
        "evaluate_study",
        "validate_reference",
        "run_diagnostics",
        "native_play",
        "persist_completed",
        "threshold_play",
    }
    if not required <= positions.keys() or not audit_loads or not all(saved.values()):
        return False
    first = {name: min(positions[name]) for name in required}
    boundary = max(min(saved["fitted.pt"]), min(saved["prepared.pt"]), first["write_before_audit"])
    return (
        first["fit_models"] < first["prepare_study"] <= boundary < min(audit_loads)
        < first["evaluate_study"] < first["validate_reference"]
        <= first["persist_completed"]
    )


def _gradientshap_visible_sequence(cells: list[dict[str, Any]]) -> bool:
    """Require the visible pre-audit boundary and complete GradientSHAP study path."""
    positions: dict[str, list[tuple[int, int]]] = {}
    audit_assignments: list[tuple[int, int]] = []
    for cell_index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            return False
        for outer in tree.body:
            if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for call in (node for node in ast.walk(outer) if isinstance(node, ast.Call)):
                positions.setdefault(ast.unparse(call.func), []).append(
                    (cell_index, call.lineno)
                )
            for assignment in (
                node for node in ast.walk(outer) if isinstance(node, (ast.Assign, ast.AnnAssign))
            ):
                targets = (
                    assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
                )
                value = assignment.value
                if (
                    any(isinstance(target, ast.Name) and target.id == "audit" for target in targets)
                    and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "load_split"
                    and value.args
                    and isinstance(value.args[0], ast.Constant)
                    and value.args[0].value == "audit"
                ):
                    audit_assignments.append((cell_index, assignment.lineno))
    required = {
        "fit_models",
        "verify_16_14_reproduction",
        "prepare_study",
        "persist_prepared",
        "evaluate_study",
        "reference_check",
        "run_diagnostics",
        "native_play",
        "persist_completed",
        "write_notebook_summary",
        "threshold_play",
    }
    if not required <= positions.keys() or not audit_assignments:
        return False
    first = {name: min(positions[name]) for name in required}
    audit = min(audit_assignments)
    return (
        first["fit_models"]
        < first["verify_16_14_reproduction"]
        < first["prepare_study"]
        < first["persist_prepared"]
        < audit
        < first["evaluate_study"]
        < first["reference_check"]
        < first["run_diagnostics"]
        < first["native_play"]
        < first["persist_completed"]
        < first["write_notebook_summary"]
    )


def _shapley_interaction_visible_sequence(cells: list[dict[str, Any]]) -> bool:
    """Check the frozen, sealed 16.16 study and both editable play paths."""
    calls: dict[str, list[tuple[tuple[int, int], ast.Call]]] = {}
    split_assignments: dict[str, list[tuple[int, int]]] = {
        "train": [],
        "explain": [],
        "audit": [],
    }
    all_audit_loads: list[tuple[int, int]] = []
    for cell_index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            return False
        for outer in tree.body:
            if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for call in (node for node in ast.walk(outer) if isinstance(node, ast.Call)):
                position = (cell_index, call.lineno)
                name = ast.unparse(call.func)
                calls.setdefault(name, []).append((position, call))
                if (
                    name == "load_split"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "audit"
                ):
                    all_audit_loads.append(position)
            for assignment in (
                node
                for node in ast.walk(outer)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            ):
                targets = (
                    assignment.targets
                    if isinstance(assignment, ast.Assign)
                    else [assignment.target]
                )
                value = assignment.value
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "load_split"
                    and value.args
                    and isinstance(value.args[0], ast.Constant)
                    and value.args[0].value in split_assignments
                ):
                    continue
                split = value.args[0].value
                if any(
                    isinstance(target, ast.Name) and target.id == split
                    for target in targets
                ):
                    split_assignments[split].append((cell_index, assignment.lineno))

    ordered = (
        "validate_freeze",
        "load_manifest",
        "fit_known_models",
        "verify_known_reproduction",
        "prepare_study",
        "persist_prepared",
        "evaluate_study",
        "reference_check",
        "finalize_study_reference",
        "interaction_diagnostics",
        "native_play",
        "persist_completed",
        "write_notebook_summary",
    )
    plays = ("exact_dividend_play", "empirical_context_play")
    if any(name not in calls or len(calls[name]) != 1 for name in (*ordered, *plays)):
        return False
    if any(len(split_assignments[name]) != 1 for name in split_assignments):
        return False
    if all_audit_loads != split_assignments["audit"]:
        return False
    first = {name: calls[name][0][0] for name in (*ordered, *plays)}
    if not (
        first["validate_freeze"]
        < first["load_manifest"]
        < first["fit_known_models"]
        < first["verify_known_reproduction"]
        < first["prepare_study"]
        < first["persist_prepared"]
        < split_assignments["audit"][0]
        < first["evaluate_study"]
        < first["reference_check"]
        < first["finalize_study_reference"]
        < first["interaction_diagnostics"]
        < first["native_play"]
        < first["persist_completed"]
        < first["write_notebook_summary"]
    ):
        return False
    if not (
        first["validate_freeze"] < split_assignments["train"][0]
        < first["fit_known_models"]
        and first["validate_freeze"] < split_assignments["explain"][0]
        < first["fit_known_models"]
        and first["exact_dividend_play"] < first["empirical_context_play"]
        and first["evaluate_study"] < first["empirical_context_play"]
    ):
        return False

    def _positional_names(name: str, expected: tuple[str, ...]) -> bool:
        node = calls[name][0][1]
        return len(node.args) >= len(expected) and all(
            isinstance(value, ast.Name) and value.id == expected[index]
            for index, value in enumerate(node.args[: len(expected)])
        )

    def _keyword_name(name: str, keyword: str, expected: str) -> bool:
        node = calls[name][0][1]
        return any(
            item.arg == keyword
            and isinstance(item.value, ast.Name)
            and item.value.id == expected
            for item in node.keywords
        )

    exact_scopes = {
        "fit_known_models": ("train",),
        "verify_known_reproduction": ("fitted", "manifest"),
        "prepare_study": ("fitted", "train", "explain"),
        "persist_prepared": ("native_directory", "fitted", "prepared"),
        "evaluate_study": ("fitted", "prepared", "audit"),
        "finalize_study_reference": ("prepared", "empirical_reference"),
        "native_play": ("evaluated",),
        "empirical_context_play": ("evaluated",),
    }
    if not all(_positional_names(name, values) for name, values in exact_scopes.items()):
        return False
    return all(
        _keyword_name(name, "methods", "methods")
        for name in (
            "prepare_study",
            "evaluate_study",
            "native_play",
            "exact_dividend_play",
            "empirical_context_play",
        )
    )


def _has_direct_experiment_contract(path: Path) -> bool:
    """Recognize an own-answer lesson without requiring report-reading cells.

    Require actual exercise definitions/answers, calls to learner methods in
    experiment cells, generated figures, and a separate complete execution path.
    Source/output hashes and fresh CUDA evidence are checked by the report and
    section evidence audits, not inferred from this structural check.
    """
    here = path.parent
    required = [here / name for name in ("lesson.md", "execute_learner.py", "solutions.py")]
    if not all(file.is_file() for file in required):
        return False
    notebook = json.loads(path.read_text())
    cells = notebook.get("cells", [])
    markdown = "\n".join("".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "markdown")
    try:
        trees = [ast.parse("".join(cell.get("source", []))) for cell in cells if cell.get("cell_type") == "code"]
        source_tree = ast.parse(required[2].read_text())
        executor_tree = ast.parse(required[1].read_text())
        answers = [ast.parse(code) for block in re.findall(r"<summary>Solution</summary>(.*?)</details>", markdown, re.S)
                   for code in re.findall(r"```python\s*\n(.*?)```", block, re.S)]
    except SyntaxError:
        return False
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    stubs = {node.name for tree in trees for node in tree.body if isinstance(node, definitions)
             and any(isinstance(child, ast.Raise) and "NotImplementedError" in ast.unparse(child)
                     for child in ast.walk(node))}
    answered = {node.name for tree in answers for node in tree.body if isinstance(node, definitions)}
    if len(stubs) < 3 or not stubs <= answered:
        return False
    # Exclude method definitions and test calls: a method must be used by a
    # visible experiment, rather than appear only in a stub or its own test.
    experiment_nodes = [node for tree in trees for node in tree.body if not isinstance(node, definitions)]
    calls = [node for outer in experiment_nodes for node in ast.walk(outer) if isinstance(node, ast.Call)]
    root_calls = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
    used = root_calls & stubs
    visible_method_use = len(used) >= 3
    if SHAPLEY_INTERACTION_STUDENT_METHODS <= stubs:
        reached = _reachable_student_method_calls(
            [*trees, *answers], root_calls, SHAPLEY_INTERACTION_STUDENT_METHODS
        )
        reached |= _visible_callable_method_arguments(
            [*trees, *answers], root_calls, SHAPLEY_INTERACTION_STUDENT_METHODS
        )
        visible_figures = sum(
            isinstance(call.func, ast.Name) and call.func.id == "show_figure"
            for call in calls
        )
        visible_method_use = (
            path.name
            == "16.16_Where_Does_Higher_Order_Credit_Go_exercises.ipynb"
            and SHAPLEY_INTERACTION_STUDENT_METHODS <= answered
            and SHAPLEY_INTERACTION_STUDENT_METHODS <= reached
            and _visible_answers_match_source(
                here,
                "interaction_methods.py",
                answers,
                SHAPLEY_INTERACTION_STUDENT_METHODS,
            )
            and _has_exact_visible_method_namespace(
                trees, SHAPLEY_INTERACTION_STUDENT_METHODS
            )
            and _shapley_interaction_visible_sequence(cells)
            and visible_figures >= 11
        )
    elif GRADIENTSHAP_STUDENT_METHODS <= stubs:
        reached = _reachable_student_method_calls(
            [*trees, *answers], root_calls, GRADIENTSHAP_STUDENT_METHODS
        )
        reached |= _visible_callable_method_arguments(
            [*trees, *answers], root_calls, GRADIENTSHAP_STUDENT_METHODS
        )
        visible_figures = sum(
            isinstance(call.func, ast.Name) and call.func.id == "show_figure"
            for call in calls
        )
        visible_method_use = (
            GRADIENTSHAP_STUDENT_METHODS <= answered
            and GRADIENTSHAP_STUDENT_METHODS <= reached
            and _visible_answers_match_source(
                here,
                "gradientshap_methods.py",
                answers,
                GRADIENTSHAP_STUDENT_METHODS,
            )
            and _has_visible_method_namespace(trees, GRADIENTSHAP_STUDENT_METHODS)
            and _gradientshap_visible_sequence(cells)
            and visible_figures >= 20
        )
    elif DEEPSHAP_STUDENT_METHODS <= stubs:
        reached = _reachable_student_method_calls(
            [*trees, *answers], root_calls, DEEPSHAP_STUDENT_METHODS
        )
        visible_figures = sum(
            isinstance(call.func, ast.Name) and call.func.id == "show_figure"
            for call in calls
        )
        visible_method_use = (
            DEEPSHAP_STUDENT_METHODS <= answered
            and DEEPSHAP_STUDENT_METHODS <= reached
            and _has_visible_method_namespace(trees, DEEPSHAP_STUDENT_METHODS)
            and _deepshap_visible_sequence(cells)
            and visible_figures >= 17
        )
    elif TREESHAP_STUDENT_METHODS <= stubs:
        # The learner executor substitutes these immediately following answers
        # into the stubs. Follow that completed, still-visible call graph.
        reached = _reachable_local_calls([*trees, *answers], root_calls)
        visible_figures = sum(
            isinstance(call.func, ast.Name) and call.func.id == "show_figure"
            for call in calls
        )
        visible_method_use = (
            TREESHAP_STUDENT_METHODS <= stubs
            and TREESHAP_STUDENT_METHODS <= reached
            and {"tiny_tree_play", "native_play"} <= root_calls
            and visible_figures >= 16
        )
    figures = any(isinstance(node.func, ast.Attribute) and node.func.attr == "savefig" for node in calls)
    # A visible figure helper is ordinary notebook plumbing. Follow its call
    # only when it saves the figure object passed by an experiment cell.
    called = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in called and node.args.args:
                figure_arg = node.args.args[0].arg
                figures |= any(
                    isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                    and item.func.attr == "savefig" and isinstance(item.func.value, ast.Name)
                    and item.func.value.id == figure_arg for item in ast.walk(node)
                )
                # A called pyplot helper can also save the active figure.
                # Follow its actual save operation, not its name or a comment.
                figures |= any(
                    isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                    and item.func.attr == "savefig"
                    and isinstance(item.func.value, ast.Call)
                    and isinstance(item.func.value.func, ast.Attribute)
                    and item.func.value.func.attr == "gcf"
                    and isinstance(item.func.value.func.value, ast.Name)
                    and item.func.value.func.value.id in {"plt", "pyplot"}
                    for item in ast.walk(node)
                )
    functions = {node.name: node for node in source_tree.body if isinstance(node, ast.FunctionDef)}
    if not {"run_gpu_test", "run_full_experiment"} <= functions.keys():
        return False
    full = ast.unparse(functions["run_full_experiment"])
    executor = ast.unparse(executor_tree)
    own_executor = ("complete_learner(" in executor and ".execute(" in executor
                    and "completed_from_own_answers" in executor)
    integrated_publication = "--publish" in executor and "execute_learner.py" in full
    # A separate publisher can validate and copy the exact completed learner.
    # Follow the GPU entry point's local imported orchestration module to check
    # that it runs the full executor and separate CUDA path; filenames alone
    # do not count as an execution/publication interface.
    separate_publisher = False
    for file in here.glob("publish_*.py"):
        try:
            publisher_tree = ast.parse(file.read_text())
            publisher = ast.unparse(publisher_tree)
        except SyntaxError:
            continue
        completed_publication = (
            "complete_learner(" in publisher
            and "validate_" in publisher
            and ".write_text(" in publisher
            and "completed_learner.ipynb" in publisher
        )
        publish_node = next(
            (
                node
                for node in publisher_tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "publish"
            ),
            None,
        )
        publish_calls = (
            {
                ast.unparse(call.func)
                for call in ast.walk(publish_node)
                if isinstance(call, ast.Call)
            }
            if publish_node is not None
            else set()
        )
        publish_literals = (
            {
                node.value
                for node in ast.walk(publish_node)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            if publish_node is not None
            else set()
        )
        rebuilt_reviewed_publication = (
            {"build_notebooks", "validate_fresh", "archive_evidence"} <= publish_calls
            and any(name.endswith(".write_text") for name in publish_calls)
            and {"visual_review.json", "completed_learner.ipynb", "execution.tar.gz"}
            <= publish_literals
        )
        if completed_publication or rebuilt_reviewed_publication:
            separate_publisher = True
    delegated_execution = False
    for node in ast.walk(functions["run_gpu_test"]):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        delegate_path = here / (node.module.rsplit(".", 1)[-1] + ".py")
        if not delegate_path.is_file():
            continue
        try:
            delegate = ast.unparse(ast.parse(delegate_path.read_text()))
        except SyntaxError:
            continue
        if ("execute_learner.py" in delegate and "subprocess.run(" in delegate
                and "--scope" in delegate and "full" in delegate and "cuda.py" in delegate):
            delegated_execution = True
        # A completed learner can be validated once and independently reproduced
        # without rerunning the entire learner inside every report command.
        imported_names = {alias.asname or alias.name: alias.name for alias in node.names}
        called_names = {call.func.id for call in ast.walk(functions["run_gpu_test"])
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}
        delegate_functions = {item.name: item for item in ast.parse(delegate_path.read_text()).body
                              if isinstance(item, ast.FunctionDef)}
        for local_name, remote_name in imported_names.items():
            if local_name not in called_names or remote_name not in delegate_functions:
                continue
            fresh_calls = [call for call in ast.walk(delegate_functions[remote_name]) if isinstance(call, ast.Call)]
            fresh_names = {call.func.id for call in fresh_calls if isinstance(call.func, ast.Name)}
            loads_model = any(isinstance(call.func, ast.Attribute) and call.func.attr == "from_pretrained"
                              and isinstance(call.func.value, ast.Name) and call.func.value.id == "AutoModelForCausalLM"
                              for call in fresh_calls)
            native_measurements = ("measure_calendar_replacements" in fresh_names
                                   or {"capture_rows", "replace_rows", "fit_candidates"} <= fresh_names)
            # A frozen-alignment study validates saved bases and reruns its
            # own visible intervention and continuation, checking independent
            # full-forward hooks rather than refitting CCA maps.
            invoked_methods = {
                call.func.slice.value for call in fresh_calls
                if isinstance(call.func, ast.Subscript)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "methods"
                and isinstance(call.func.slice, ast.Constant)
                and isinstance(call.func.slice.value, str)
            }
            native_measurements |= (
                {"capture_native", "full_forward_intervention"} <= fresh_names
                and {"subspace_interchange", "continue_native"} <= invoked_methods
            )
            # Concept erasure refits maps and probes before a separate native
            # inference helper. Follow those called helpers; filenames and a
            # loaded checkpoint alone are not an experiment contract.
            fit_node = delegate_functions.get("construct_conditions")
            infer_node = delegate_functions.get("infer")
            edit_node = delegate_functions.get("edit")
            if fit_node is not None and infer_node is not None and edit_node is not None:
                fit_attributes = {
                    call.func.attr for call in ast.walk(fit_node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "m"
                }
                infer_names = {
                    call.func.id for call in ast.walk(infer_node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                }
                edit_attributes = {
                    call.func.attr for call in ast.walk(edit_node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "m"
                }
                native_measurements |= (
                    {"capture_native", "full_forward_intervention", "construct_conditions",
                     "compare_fitted_conditions", "infer"} <= fresh_names
                    and {"fit_leace", "fit_binary_probe", "fit_orthogonal_eraser"} <= fit_attributes
                    and {"edit", "continue_native"} <= infer_names
                    and {"apply_eraser", "matched_random_edit"} <= edit_attributes
                )
            if loads_model and "validate_learner" in fresh_names and native_measurements:
                delegated_execution = True
            # CCS reloads through token-capture plumbing and refits through
            # provided control orchestration. Inspect those actual called
            # functions as well as the fresh entry point.
            if _has_ccs_native_refit_contract(here, delegate_functions[remote_name], fresh_names):
                delegated_execution = True
            if _has_data_attribution_refit_contract(here, fresh_names):
                delegated_execution = True
            if _has_datamodel_refit_contract(here, fresh_names):
                delegated_execution = True
            if _has_cartography_refit_contract(here, fresh_names):
                delegated_execution = True
            if _has_sage_refit_contract(here, fresh_names):
                delegated_execution = True
            if _has_treeshap_refit_contract(here, fresh_names):
                delegated_execution = True
            if _has_deepshap_refit_contract(here, fresh_names):
                delegated_execution = True
            if _has_gradientshap_refit_contract(here, fresh_names):
                delegated_execution = True
            if _has_shapley_interaction_refit_contract(here, fresh_names):
                delegated_execution = True
    return (visible_method_use and figures and own_executor and "run_gpu_test(" in full
            and (integrated_publication or (separate_publisher and delegated_execution)))


def _has_sage_refit_contract(here: Path, fresh_names: set[str]) -> bool:
    """Require actual native model training, performance games and independent checks."""
    if not {"run", "compare"} <= fresh_names:
        return False
    requirements = {
        "run_native.py": {"run": {"validate_freeze", "sage_data.load_split",
            "sage_study.prepare_study", "sage_study.evaluate_study", "sage_study.diagnostics",
            "sage_study.native_play", "sage_reference.checks"}},
        "sage_data.py": {"load_split": {"source_data"}},
        "sage_methods.py": {
            "marginal_predict": {"t.where", "model"},
            "loss_game": {"(predictions - y[:, None]).square"},
            "exact_shapley": {"math.comb"},
            "gaussian_game": {"t.linalg.pinv", "t.einsum"},
            "permutation_sage": {"marginal_predict", "path_contributions", "t.cat"},
            "summarize_paths": {"math.sqrt"},
        },
        "sage_study.py": {
            "fit_models": {"t.nn.Linear", "t.optim.Adam", "loss.backward", "optimizer.step", "t.linalg.solve"},
            "prepare_study": {"fit_models", "methods.coalition_predictions", "explain_predictions", "t.argsort", "t.save"},
            "explain_predictions": {"methods.loss_game", "methods.exact_shapley"},
            "evaluate_study": {"methods.coalition_predictions", "methods.prefix_values"},
            "diagnostics": {"methods.permutation_sage", "methods.table_path_contributions", "methods.summarize_paths"},
            "native_play": {"methods.coalition_predictions", "explain_predictions"},
        },
        "sage_reference.py": {
            "checks": {"permutation_reference", "np.linalg.lstsq", "np.testing.assert_allclose", "author_imputer"},
            "permutation_reference": {"itertools.permutations", "np.bitwise_or", "np.testing.assert_allclose"},
        },
    }
    for filename, functions in requirements.items():
        try:
            nodes = {node.name: node for node in ast.parse((here / filename).read_text()).body
                     if isinstance(node, ast.FunctionDef)}
        except (OSError, SyntaxError):
            return False
        for name, required in functions.items():
            if name not in nodes:
                return False
            calls = {ast.unparse(node.func) for node in ast.walk(nodes[name]) if isinstance(node, ast.Call)}
            if not required <= calls:
                return False
    return True


def _has_treeshap_refit_contract(here: Path, fresh_names: set[str]) -> bool:
    """Follow fresh tree fits, both games, controls and independent references."""
    if not {"run", "compare"} <= fresh_names:
        return False
    requirements = {
        "run_native.py": {
            "run": {
                "validate_freeze", "data.load_split", "study.fit_models",
                "study.prepare_study", "persist_prepared", "study.evaluate_study",
                "reference_module.check_reference", "diagnostics_module.run_diagnostics",
                "study.native_play", "persist_completed",
            },
            "persist_prepared": {"save_payload", "write_before_audit"},
            "persist_completed": {"save_payload", "summarize"},
        },
        "treeshap_data.py": {
            "load_split": {"source_data"},
            "source_data": {"np.loadtxt", "hashlib.sha256"},
            "background_order": {"hashlib.sha256", "sorted"},
        },
        "treeshap_methods.py": {
            "tree_predict": {"np.asarray", "np.flatnonzero", "np.where"},
            "tree_path_game": {"np.asarray", "np.full", "np.where", "visit"},
            "exact_shapley": {"math.comb"},
            "extend_path": {"np.asarray", "np.zeros"},
            "unwind_path": {"np.broadcast_arrays", "np.where"},
            "tree_shap": {"extend_path", "unwind_path", "visit"},
            "ensemble_shap": {"tree_shap", "tree_path_game", "tree_predict"},
            "interventional_game": {"np.where", "predict"},
            "saabas": {"expectation"},
        },
        "treeshap_study.py": {
            "fit_models": {
                "DecisionTreeRegressor", "RandomForestRegressor",
                "GradientBoostingRegressor", "model.fit", "model.predict", "adapt_ensemble",
            },
            "explain_block": {
                "coalition_masks", "ensemble_path_game", "ensemble_predict",
                "ensemble_shap", "interventional_game", "exact_shapley", "ensemble_saabas",
                "np.testing.assert_allclose",
            },
            "prepare_study": {"background_order", "explain_block", "np.argsort"},
            "evaluate_study": {"explain_block", "ensemble_shap", "rng.permutation"},
            "native_play": {"background_order", "ensemble_shap", "ensemble_predict"},
        },
        "treeshap_diagnostics.py": {
            "run_diagnostics": {
                "correlated_games", "path_cardinality_diagnostic",
                "repeated_feature_diagnostic", "benchmark_runtime", "tiny_tree_play",
            },
            "correlated_games": {
                "_conditional_binary_game", "_interventional_binary_game", "path_game",
                "exact", "saabas_fn",
            },
            "repeated_feature_diagnostic": {"np.max"},
            "benchmark_runtime": {"_benchmark_axis"},
        },
        "treeshap_reference.py": {
            "check_reference": {
                "_reference_environment", "_validate_identity", "_validate_block",
                "_compiled_reference", "subset_reference", "_require_close",
                "_isclose_rows",
            },
            "_compiled_reference": {
                "shap.TreeExplainer", "path_explainer.shap_values", "raw.shap_values",
                "corrected.shap_values", "_threshold_copies", "model.predict",
                "_require_close",
            },
            "_threshold_copies": {
                "_floor_float32_thresholds", "np.nextafter", "corrected.model.predict",
                "nearest_model.model.predict", "sklearn_model.predict", "_require_close",
            },
            "subset_reference": {"math.factorial"},
        },
    }
    for filename, functions in requirements.items():
        try:
            nodes = {
                node.name: node
                for node in ast.parse((here / filename).read_text()).body
                if isinstance(node, ast.FunctionDef)
            }
        except (OSError, SyntaxError):
            return False
        for name, required in functions.items():
            if name not in nodes:
                return False
            calls = {
                ast.unparse(node.func)
                for node in ast.walk(nodes[name])
                if isinstance(node, ast.Call)
            }
            if not required <= calls:
                return False
    try:
        reference_node = next(
            node
            for node in ast.parse((here / "treeshap_reference.py").read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "check_reference"
        )
        run_node = next(
            node
            for node in ast.parse((here / "run_native.py").read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
    except (OSError, SyntaxError, StopIteration):
        return False
    exact_assignments = {
        target.id
        for assignment in ast.walk(reference_node)
        if isinstance(assignment, ast.Assign)
        and isinstance(assignment.value, ast.Call)
        and isinstance(assignment.value.func, ast.Name)
        and assignment.value.func.id == "subset_reference"
        for target in assignment.targets
        if isinstance(target, ast.Name)
    }
    if not {"exact_path", "exact_interventional"} <= exact_assignments:
        return False
    # A generic load call is insufficient: the audit split must be loaded only
    # after the prepared rankings have been persisted.
    prepared_lines = [
        call.lineno
        for call in ast.walk(run_node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "persist_prepared"
    ]
    audit_lines = [
        call.lineno
        for call in ast.walk(run_node)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "data.load_split"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "audit"
    ]
    return bool(prepared_lines and audit_lines and min(prepared_lines) < min(audit_lines))


def _has_deepshap_refit_contract(here: Path, fresh_names: set[str]) -> bool:
    """Follow fresh neural fits, full games, controls and independent DeepExplainer."""
    if not {"run", "compare"} <= fresh_names:
        return False
    requirements = {
        "run_native.py": {
            "run": {
                "validate_freeze",
                "data.load_split",
                "study.fit_models",
                "study.prepare_study",
                "persist_prepared",
                "study.evaluate_study",
                "reference_module.validate_reference",
                "diagnostics_module.run_diagnostics",
                "study.native_play",
                "persist_completed",
            },
            "persist_prepared": {"save_payload", "write_before_audit"},
            "persist_completed": {"save_payload", "summarize"},
        },
        "deepshap_data.py": {
            "source_data": {"np.loadtxt", "hashlib.sha256"},
            "load_split": {"source_data"},
            "background_order": {"hashlib.sha256", "sorted"},
        },
        "deepshap_methods.py": {
            "paired_forward": {"_layers", "_method_tensor", "_check_widths", "layer"},
            "rescale_slope": {"_method_tensor", "t.broadcast_tensors", "t.relu", "t.where"},
            "rescale_multipliers": {"paired_forward", "rescale_slope", "t.einsum"},
            "single_reference": {"_method_tensor", "rescale_multipliers"},
            "background_deepshap": {"rescale_multipliers", "_model_output"},
            "coalition_values": {"t.where", "_model_output", "t.cat"},
            "exact_shapley": {"math.comb"},
            "reveal_cancel_groups": {"t.broadcast_tensors", "activate"},
            "ranked_replacement_curves": {"t.argsort", "values.gather"},
            "coalition_masks": {"t.arange"},
            "gradient_delta": {"t.autograd.grad"},
        },
        "deepshap_study.py": {
            "_train_mlp": {"t.optim.Adam", "loss.backward", "optimizer.step"},
            "fit_one_mlp_control": {
                "t.manual_seed",
                "t.nn.Linear",
                "t.nn.ReLU",
                "_train_mlp",
                "_fold_mlp_result",
            },
            "fit_ridge_control": {"t.linalg.solve", "_fold_ridge_state", "_folding_error"},
            "fit_models": {
                "_training_normalization",
                "fit_one_mlp_control",
                "fit_ridge_control",
            },
            "_explain_model": {
                "methods.background_deepshap",
                "methods.rescale_multipliers",
                "methods.coalition_masks",
                "methods.coalition_values",
                "methods.exact_shapley",
                "methods.gradient_delta",
            },
            "prepare_study": {"_background_indices", "explain_block", "t.argsort"},
            "_evaluate_replacements": {"_ranked_curves", "_random_curves"},
            "_evaluate_background_variants": {"_combine_explanations", "_explain_model"},
            "evaluate_study": {
                "explain_block",
                "_performance_table",
                "_fixed_random_orders",
                "_evaluate_replacements",
                "_evaluate_background_variants",
            },
            "native_play": {"background_order", "_explain_model", "methods.ranked_replacement_curves"},
        },
        "deepshap_diagnostics.py": {
            "run_diagnostics": {
                "diagnose_rescale_limits",
                "diagnose_near_zero",
                "diagnose_affine_controls",
                "diagnose_background_average",
                "diagnose_threshold",
                "diagnose_cancellation",
                "diagnose_replacement_control",
                "threshold_play",
            },
            "diagnose_near_zero": {"single_reference"},
            "diagnose_threshold": {
                "single_reference",
                "coalition_values",
                "exact_shapley",
                "gradient_delta",
            },
            "diagnose_cancellation": {"single_reference", "reveal_cancel", "exact_shapley"},
        },
        "deepshap_reference.py": {
            "validate_reference": {
                "_reference_environment",
                "_validate_identity",
                "_validate_block",
                "_author_deep",
                "_bounded_pair_author",
                "subset_reference",
                "ridge_closed_form",
                "known_diagnostics",
            },
            "_author_deep": {"shap.DeepExplainer", "explainer.shap_values", "_require_close"},
            "_bounded_pair_author": {"_author_deep"},
            "subset_reference": {"math.comb"},
        },
    }
    for filename, functions in requirements.items():
        try:
            nodes = {
                node.name: node
                for node in ast.parse((here / filename).read_text()).body
                if isinstance(node, ast.FunctionDef)
            }
        except (OSError, SyntaxError):
            return False
        for name, required in functions.items():
            if name not in nodes:
                return False
            calls = {
                ast.unparse(node.func)
                for node in ast.walk(nodes[name])
                if isinstance(node, ast.Call)
            }
            if not required <= calls:
                return False
    try:
        method_nodes = {
            node.name: node
            for node in ast.parse((here / "deepshap_methods.py").read_text()).body
            if isinstance(node, ast.FunctionDef)
        }
        reveal_activation_calls = sum(
            isinstance(call.func, ast.Name) and call.func.id == "activate"
            for call in ast.walk(method_nodes["reveal_cancel_groups"])
            if isinstance(call, ast.Call)
        )
        run_node = next(
            node
            for node in ast.parse((here / "run_native.py").read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
    except (OSError, SyntaxError, StopIteration):
        return False
    prepared_lines = [
        call.lineno
        for call in ast.walk(run_node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "persist_prepared"
    ]
    audit_lines = [
        call.lineno
        for call in ast.walk(run_node)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "data.load_split"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "audit"
    ]
    return bool(
        reveal_activation_calls == 4
        and prepared_lines
        and audit_lines
        and min(prepared_lines) < min(audit_lines)
    )


def _has_gradientshap_refit_contract(here: Path, fresh_names: set[str]) -> bool:
    """Follow the fresh fits, frozen audit boundary, full study, and external draws."""
    if not {"compare_manifests", "validate_artifact_directory"} <= fresh_names:
        return False
    requirements = {
        "run_native.py": {
            "run": {
                "validate_freeze",
                "data.load_split",
                "data.load_manifest",
                "study.fit_models",
                "study.verify_16_14_reproduction",
                "study.prepare_study",
                "_require_reproduction_gate",
                "persist_prepared",
                "study.evaluate_study",
                "reference_module.validate_reference",
                "diagnostics_module.run_diagnostics",
                "study.native_play",
                "persist_completed",
            },
            "persist_prepared": {"save_payload", "write_before_audit"},
            "_require_reproduction_gate": {"sum", "len", "isinstance"},
            "write_before_audit": {"_require_reproduction_gate", "digest", "_array_identity"},
            "persist_completed": {
                "_verify_boundary",
                "_validate_replicate_branch",
                "_array",
                "save_payload",
                "_external_reference",
                "_persisted_evaluated",
                "summarize",
            },
            "_external_reference": {"_array_identity", "manifest_path"},
            "_array_identity": {"_array", "_array_value_sha256"},
            "_array_value_sha256": {"_array", "hashlib.sha256"},
            "_validate_replicate_branch": {
                "_require_array",
                "_array",
                "np.array_equal",
            },
            "_persisted_evaluated": {"dict"},
        },
        "reproduce_native.py": {
            "reproduce": {
                "subprocess.run",
                "compare_manifests",
                "validate_artifact_directory",
                "load_payload",
                "_validate_timing",
                "digest",
            },
            "compare_manifests": {"_manifest_identity"},
            "validate_artifact_directory": {"_load_manifest", "digest", "manifest_path"},
        },
        "gradientshap_data.py": {
            "source_data": {"np.loadtxt", "hashlib.sha256"},
            "load_split": {"source_data"},
            "background_order": {"hashlib.sha256", "sorted"},
            "tensor_mapping_sha256": {"hashlib.sha256"},
        },
        "gradientshap_methods.py": {
            "_eg_seed_words": {"hashlib.sha256", "int.from_bytes"},
            "_eg_preserve_execution_state": {
                "random.getstate",
                "np.random.get_state",
                "t.random.get_rng_state",
                "t.get_default_dtype",
                "model.modules",
                "model.buffers",
                "model.parameters",
                "random.setstate",
                "np.random.set_state",
                "t.random.set_rng_state",
                "t.set_default_dtype",
            },
            "sample_path_schedule": {
                "_eg_index_tensor",
                "_eg_positive_int",
                "np.random.Generator",
                "np.random.PCG64",
                "_eg_seed_words",
                "donor_rng.integers",
                "alpha_rng.random",
                "t.tensor",
            },
            "interpolate_paths": {"_eg_validate_paths"},
            "input_gradients": {
                "_eg_model",
                "_eg_float_tensor",
                "_eg_positive_int",
                "_eg_preserve_execution_state",
                "_eg_model_output",
                "t.autograd.grad",
                "t.zeros_like",
                "t.isfinite",
            },
            "expected_gradient_samples": {"interpolate_paths", "input_gradients"},
            "summarize_expected_gradients": {
                "prefix.mean",
                "prefix.std",
                "math.sqrt",
                "t.stack",
            },
            "expected_gradients": {
                "_eg_validate_paths",
                "expected_gradient_samples",
                "summarize_expected_gradients",
                "_eg_model_output",
            },
            "gauss_legendre_pairs": {
                "np.polynomial.legendre.leggauss",
                "input_gradients",
                "t.tensor",
            },
            "noisy_expected_gradient_samples": {"interpolate_paths", "input_gradients"},
            "ranked_coalition_curves": {
                "t.where",
                "t.argsort",
                "t.arange",
                "values.gather",
            },
        },
        "gradientshap_study.py": {
            "_train_mlp": {"t.optim.Adam", "loss.backward", "optimizer.step"},
            "fit_one_mlp_control": {
                "t.manual_seed",
                "t.nn.Linear",
                "t.nn.ReLU",
                "_train_mlp",
                "_fold_mlp_state",
            },
            "fit_ridge_control": {"t.linalg.solve", "_fold_ridge_state", "_folding_error"},
            "fit_models": {
                "t.set_num_threads",
                "t.use_deterministic_algorithms",
                "_training_normalization",
                "fit_one_mlp_control",
                "fit_ridge_control",
                "tensor_mapping_sha256",
            },
            "verify_16_14_reproduction": {
                "_validate_fitted",
                "tensor_mapping_sha256",
                "t.equal",
                "_training_normalization",
                "_model_from_state",
            },
            "_monte_carlo_block": {
                "methods.expected_gradient_samples",
                "methods.summarize_expected_gradients",
            },
            "_quadrature_block": {"methods.gauss_legendre_pairs"},
            "prepare_study": {
                "verify_16_14_reproduction",
                "load_manifest",
                "_background_indices",
                "_primary_schedule",
                "_monte_carlo_block",
                "_quadrature_block",
                "_comparison_baselines",
                "_global_rankings",
            },
            "evaluate_study": {
                "_require_namespace",
                "_validate_evaluation_inputs",
                "_primary_schedule",
                "_model_from_state",
                "_monte_carlo_block",
                "_quadrature_block",
                "_comparison_baselines",
                "_combine_primary",
                "_background_comparison",
                "_noise_comparison",
                "_replicate_comparison",
                "_distance_comparison",
                "_replacement_evaluation",
                "_performance_table",
            },
            "native_play": {
                "_require_namespace",
                "_validate_evaluation_inputs",
                "methods.sample_path_schedule",
                "_model_from_state",
                "methods.expected_gradient_samples",
                "methods.noisy_expected_gradient_samples",
                "methods.summarize_expected_gradients",
                "_quadrature_block",
                "_comparison_baselines",
                "methods.ranked_coalition_curves",
            },
            "_replicate_comparison": {
                "methods.sample_path_schedule",
                "methods.expected_gradient_samples",
                "methods.summarize_expected_gradients",
            },
            "_noise_comparison": {
                "methods.noisy_expected_gradient_samples",
                "methods.summarize_expected_gradients",
            },
        },
        "gradientshap_reference.py": {
            "reference_environment": {"_sha256"},
            "shap_draw_table": {"np.random.RandomState", "legacy.choice", "legacy.uniform"},
            "captum_draw_table": {
                "np.random.RandomState",
                "legacy.choice",
                "legacy.uniform",
                "t.normal",
            },
            "replay_draw_table": {
                "_preserve_execution_state",
                "_method",
                "_model_values",
                "contributions.mean",
                "contributions.std",
            },
            "shap_same_draw_reference": {
                "reference_environment",
                "shap_draw_table",
                "_preserve_execution_state",
                "shap.GradientExplainer",
                "explainer.shap_values",
                "replay_draw_table",
                "_close",
            },
            "captum_same_draw_reference": {
                "reference_environment",
                "captum_draw_table",
                "_preserve_execution_state",
                "GradientShap",
                "GradientShap(model).attribute",
                "replay_draw_table",
                "_close",
            },
            "positive_noise_reference": {
                "shap_same_draw_reference",
                "captum_same_draw_reference",
            },
            "validate_reference": {
                "reference_environment",
                "shap_same_draw_reference",
                "captum_same_draw_reference",
                "positive_noise_reference",
                "_validate_model",
            },
            "reference_check": {"validate_reference"},
        },
        "gradientshap_diagnostics.py": {
            "run_diagnostics": {
                "diagnose_schedule",
                "diagnose_interpolation",
                "diagnose_gradients",
                "diagnose_expected_gradient_samples",
                "diagnose_convergence",
                "diagnose_expected_gradients",
                "diagnose_quadrature",
                "diagnose_background_average",
                "diagnose_noise_target",
                "diagnose_path_vs_coalition",
                "threshold_play",
            },
        },
    }
    parsed: dict[str, tuple[ast.Module, dict[str, ast.FunctionDef]]] = {}
    for filename, functions in requirements.items():
        try:
            tree = ast.parse((here / filename).read_text())
        except (OSError, SyntaxError):
            return False
        nodes = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        parsed[filename] = tree, nodes
        for name, required in functions.items():
            if name not in nodes:
                return False
            calls = {
                ast.unparse(node.func)
                for node in ast.walk(nodes[name])
                if isinstance(node, ast.Call)
            }
            if not required <= calls:
                return False

    study_tree, study_nodes = parsed["gradientshap_study.py"]
    constants: dict[str, Any] = {}
    for node in study_tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        try:
            constants[targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    model_keys = constants.get("MODEL_KEYS")
    if model_keys != (
        "clean_0",
        "clean_1",
        "clean_2",
        "shuffled_0",
        "shuffled_1",
        "shuffled_2",
        "ridge",
    ):
        return False
    state_fields = None
    for assignment in ast.walk(study_nodes["verify_16_14_reproduction"]):
        if not isinstance(assignment, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "state_fields"
            for target in assignment.targets
        ):
            continue
        try:
            state_fields = ast.literal_eval(assignment.value)
        except (ValueError, TypeError):
            return False
    if state_fields != {
        "initial_state": "initial_state",
        "standardized_state": "standardized_state",
        "raw_folded_state": "state",
    } or len(model_keys) * len(state_fields) != 21:
        return False

    reproduction_gate = ast.unparse(parsed["run_native.py"][1]["_require_reproduction_gate"])
    if not all(
        fragment in reproduction_gate
        for fragment in (
            "state_hash_count != 21",
            "expected != actual",
            "len(mismatches) != 0",
        )
    ):
        return False

    reproduce = ast.unparse(parsed["reproduce_native.py"][1]["reproduce"])
    if not all(
        fragment in reproduce
        for fragment in (
            "subprocess.run(",
            "str(HERE / 'run_native.py')",
            "str(fresh_directory)",
            "compare_manifests(",
        )
    ):
        return False

    reference_tree, reference_nodes = parsed["gradientshap_reference.py"]
    reference_constants: dict[str, Any] = {}
    for node in reference_tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        try:
            reference_constants[targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    if (
        reference_constants.get("MODEL_KEYS") != model_keys
        or reference_constants.get("ALL_ROWS") != 272
        or reference_constants.get("BACKGROUND_ROWS") != 64
        or reference_constants.get("REFERENCE_SAMPLES") != 128
        or reference_constants.get("FEATURES") != 8
    ):
        return False
    full_reference_loop = any(
        isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "MODEL_KEYS"
        and {
            ast.unparse(call.func)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        }
        >= {"shap_same_draw_reference", "captum_same_draw_reference"}
        for node in ast.walk(reference_nodes["validate_reference"])
    )
    if not full_reference_loop:
        return False

    run_tree, run_nodes = parsed["run_native.py"]
    run_node = run_nodes["run"]
    call_lines: dict[str, list[int]] = {}
    audit_lines: list[int] = []
    for node in ast.walk(run_node):
        if isinstance(node, ast.Call):
            call_lines.setdefault(ast.unparse(node.func), []).append(node.lineno)
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "audit" for target in node.targets
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and ast.unparse(value.func) == "data.load_split"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == "audit"
        ):
            audit_lines.append(node.lineno)
    ordered = (
        "study.fit_models",
        "study.verify_16_14_reproduction",
        "study.prepare_study",
        "persist_prepared",
        "study.evaluate_study",
        "reference_module.validate_reference",
        "diagnostics_module.run_diagnostics",
        "study.native_play",
        "persist_completed",
    )
    if not audit_lines or any(name not in call_lines for name in ordered):
        return False
    first = {name: min(call_lines[name]) for name in ordered}
    if not (
        first["study.fit_models"]
        < first["study.verify_16_14_reproduction"]
        < first["study.prepare_study"]
        < first["persist_prepared"]
        < min(audit_lines)
        < first["study.evaluate_study"]
        < first["reference_module.validate_reference"]
        < first["diagnostics_module.run_diagnostics"]
        < first["study.native_play"]
        < first["persist_completed"]
    ):
        return False

    prepared_saves = {
        call.args[1].value
        for call in ast.walk(run_nodes["persist_prepared"])
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "save_payload"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
    }
    if prepared_saves != {"fitted.pt", "prepared.pt"}:
        return False

    run_constants: dict[str, Any] = {}
    for node in run_tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        try:
            run_constants[targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    if (
        run_constants.get("EXTERNAL_DETERMINISTIC_ARTIFACT") != "replicate_samples.pt"
        or run_constants.get("REPLICATE_SHAPE") != (64, 3, 64, 2048, 8)
    ):
        return False
    completed = ast.unparse(run_nodes["persist_completed"])
    external = ast.unparse(run_nodes["_external_reference"])
    if not all(
        fragment in completed
        for fragment in (
            "replicates = evaluated.get('sampling_replicates', {})",
            "_validate_replicate_branch(replicates)",
            "samples = replicates.get('samples')",
            "array.shape != REPLICATE_SHAPE",
            "array.dtype != np.dtype('float64')",
            "save_payload(directory, EXTERNAL_DETERMINISTIC_ARTIFACT, samples)",
            "_external_reference(samples, replicate_manifest)",
            "_persisted_evaluated(evaluated, external)",
            "summary['replicate_sample_coordinates'] != REPLICATE_ELEMENTS",
        )
    ):
        return False
    if not all(
        fragment in external
        for fragment in (
            "'storage': 'external_pt'",
            "'payload_path': '/sampling_replicates/samples'",
            "'artifact_sha256'",
        )
    ):
        return False
    required_sources = {
        "gradientshap_data.py",
        "gradientshap_methods.py",
        "gradientshap_methods_tests.py",
        "gradientshap_study.py",
        "gradientshap_reference.py",
        "gradientshap_diagnostics.py",
        "protocol.md",
        "protocol_freeze.json",
        "data_manifest.json",
        "run_native.py",
    }
    source_literals = {
        node.value
        for node in ast.walk(run_nodes["source_paths"])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    return required_sources <= source_literals


def _has_shapley_interaction_refit_contract(
    here: Path, fresh_names: set[str]
) -> bool:
    """Follow the 16.16 refit, physical audit seal, references, and play paths.

    This is a source contract. Completeness and values of native tensors are
    established by the separate artifact/report audits after execution.
    """
    if not {"compare_manifests", "validate_artifact_directory"} <= fresh_names:
        return False
    requirements = {
        "run_native.py": {
            "run": {
                "validate_freeze",
                "data_module.load_split",
                "data_module.load_manifest",
                "study.fit_known_models",
                "study.verify_known_reproduction",
                "_state_hash_gate",
                "study.prepare_study",
                "persist_prepared",
                "study.evaluate_study",
                "reference_module.validate_study_reference",
                "diagnostics_module.interaction_diagnostics",
                "study.native_play",
                "persist_completed",
            },
            "_state_hash_gate": {"sum", "len", "isinstance"},
            "write_before_audit": {
                "_state_hash_gate",
                "hybrid_artifact_name",
                "manifest_path",
                "digest",
                "_array_identity",
                "_write_json",
            },
            "persist_prepared": {
                "_state_hash_gate",
                "_persist_phase_hybrids",
                "save_payload",
                "write_before_audit",
            },
            "persist_completed": {
                "_verify_boundary",
                "_persist_phase_hybrids",
                "_replace_logical_all_rows",
                "save_payload",
                "summarize",
                "_inventory_record",
            },
        },
        "reproduce_native.py": {
            "reproduce": {
                "subprocess.run",
                "validate_artifact_directory",
                "compare_manifests",
                "load_payload",
                "_validate_timing",
                "digest",
            },
            "compare_manifests": {"_manifest_identity"},
            "validate_artifact_directory": {
                "_load_manifest",
                "_validate_external_descriptors",
                "digest",
                "manifest_path",
            },
        },
        "interaction_data.py": {
            "source_data": {"load_manifest", "np.loadtxt", "hashlib.sha256"},
            "load_split": {"load_manifest", "source_data", "recipe_group_ids"},
            "background_order": {"hashlib.sha256", "sorted"},
            "tensor_mapping_sha256": {"hashlib.sha256", "array.tobytes"},
        },
        "interaction_methods.py": {
            "_interaction_preserve_model_state": {
                "random.getstate",
                "np.random.get_state",
                "torch.random.get_rng_state",
                "torch.get_default_dtype",
                "model.modules",
                "model.buffers",
                "model.parameters",
                "random.setstate",
                "np.random.set_state",
                "torch.random.set_rng_state",
                "torch.set_default_dtype",
            },
            "mobius_transform": {
                "_interaction_n_players",
                "_interaction_tensor",
                "torch.arange",
            },
            "shapley_taylor_interactions": {
                "_interaction_order",
                "_interaction_tensor",
                "_interaction_counts",
                "math.comb",
                "torch.zeros_like",
            },
            "faith_design_matrix": {
                "_interaction_order",
                "_interaction_counts",
                "torch.arange",
            },
            "faith_kernel_weights": {"math.comb", "weights.sum"},
            "faith_shap_interactions": {
                "faith_design_matrix",
                "faith_kernel_weights",
                "torch.linalg.solve",
            },
            "faith_shap_closed_form": {"math.comb", "torch.zeros_like"},
            "reconstruct_game": {"_interaction_tensor", "torch.arange"},
            "faithfulness_error": {"reconstruct_game", "faith_kernel_weights"},
            "masked_model_game": {
                "_interaction_game_batches",
                "_interaction_model_output",
                "torch.where",
                "sums.index_add_",
            },
            "_interaction_game_batches": {
                "_interaction_model",
                "_interaction_float_input",
                "_interaction_preserve_model_state",
                "torch.no_grad",
                "flat.remainder",
                "flat.div",
            },
            "contextual_pair_synergy": {
                "_interaction_tensor",
                "contexts.bitwise_or",
            },
        },
        "interaction_study.py": {
            "_train_mlp": {"t.optim.Adam", "loss.backward", "optimizer.step"},
            "_fit_one_mlp": {
                "t.manual_seed",
                "t.nn.Linear",
                "t.nn.ReLU",
                "_train_mlp",
                "_fold_mlp_state",
            },
            "_fit_ridge": {"t.linalg.solve", "_fold_ridge_state", "_folding_check"},
            "fit_known_models": {
                "t.set_num_threads",
                "t.use_deterministic_algorithms",
                "_fit_one_mlp",
                "_fit_ridge",
                "tensor_mapping_sha256",
            },
            "verify_known_reproduction": {
                "_validate_fitted",
                "tensor_mapping_sha256",
                "t.equal",
            },
            "donor_replacement_games": {"methods.masked_model_game"},
            "explain_complete_games": {
                "methods.mobius_transform",
                "methods.shapley_taylor_interactions",
                "methods.faith_shap_interactions",
                "methods.faith_shap_closed_form",
                "methods.reconstruct_game",
                "methods.faithfulness_error",
                "all_pair_context_synergies",
                "pairwise_sii",
            },
            "_exact_suite": {
                "fixed_exact_games",
                "random_exact_games",
                "explain_complete_games",
            },
            "prepare_study": {
                "verify_known_reproduction",
                "_background_records",
                "_evaluate_primary",
                "_evaluate_backgrounds",
                "_exact_suite",
                "explanation_pair_rankings",
                "fixed_random_pair_orders",
                "_direct_full_context_four",
                "methods.faith_design_matrix",
                "methods.faith_kernel_weights",
            },
            "within_size_shuffle_control": {
                "methods.mobius_transform",
                "methods.shapley_taylor_interactions",
                "methods.faith_shap_interactions",
                "pairwise_sii",
            },
            "evaluate_study": {
                "verify_known_reproduction",
                "_evaluate_primary",
                "_evaluate_backgrounds",
                "_combine_game_blocks",
                "_direct_full_context_four",
                "pair_capture_curves",
                "within_size_shuffle_control",
                "nearest_training_diagnostics",
                "background_sensitivity",
            },
            "exact_dividend_play": {"_zeta_transform", "explain_complete_games"},
            "empirical_context_play": {
                "_logical_hybrid_row",
                "explain_complete_games",
            },
            "native_play": {"exact_dividend_play", "empirical_context_play"},
        },
        "interaction_reference.py": {
            "reference_environment": {"_sha256", "_source_path"},
            "_direct_stii": {"_derivative", "math.comb"},
            "_direct_fsii": {"_mobius", "math.comb"},
            "_direct_sii_pairs": {"_derivative", "math.factorial"},
            "_operator": {"_direct_stii", "_direct_fsii", "_direct_sii_pairs"},
            "_faith_kkt": {"_faith_system", "np.linalg.solve"},
            "_package_three": {"ExactComputer", "exact", "_package_array"},
            "reference_check": {"_preserve_execution_state", "_reference_check_impl"},
            "validate_reference_payload": {
                "_preserve_execution_state",
                "reference_environment",
                "_check",
            },
            "_validate_exact_suite": {"validate_reference_payload"},
            "validate_exact_reference_suite": {"_validate_exact_suite"},
            "finalize_study_reference": {
                "_preserve_execution_state",
                "validate_reference_payload",
                "validate_exact_reference_suite",
            },
            "validate_study_reference": {
                "_preserve_execution_state",
                "reference_check",
                "finalize_study_reference",
                "_study_block",
            },
        },
        "interaction_diagnostics.py": {
            "interaction_diagnostics": {
                "_native_mapping",
                "_native_tensor",
                "torch.unique",
            },
        },
    }
    parsed: dict[str, tuple[ast.Module, dict[str, ast.FunctionDef]]] = {}
    for filename, functions in requirements.items():
        try:
            tree = ast.parse((here / filename).read_text())
        except (OSError, SyntaxError):
            return False
        nodes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        parsed[filename] = tree, nodes
        for name, required_calls in functions.items():
            if name not in nodes:
                return False
            actual_calls = {
                ast.unparse(call.func)
                for call in ast.walk(nodes[name])
                if isinstance(call, ast.Call)
            }
            if not required_calls <= actual_calls:
                return False

    def _constants(tree: ast.Module) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            try:
                result[targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
        return result

    expected_models = (
        "clean_0",
        "clean_1",
        "clean_2",
        "shuffled_0",
        "shuffled_1",
        "shuffled_2",
        "ridge",
    )
    study_tree, study_nodes = parsed["interaction_study.py"]
    play_assignments = {
        target.id: node.value
        for node in ast.walk(study_nodes["native_play"])
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for result_name in ("empirical", "donor"):
        call = play_assignments.get(result_name)
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != "empirical_context_play":
            return False
    study_constants = _constants(study_tree)
    if study_constants.get("MODEL_KEYS") != expected_models:
        return False
    fit_loops = [
        node
        for node in ast.walk(study_nodes["fit_known_models"])
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "enumerate"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "MODEL_SEEDS"
    ]
    if len(fit_loops) != 2:
        return False
    fields = None
    for assignment in ast.walk(study_nodes["verify_known_reproduction"]):
        if not isinstance(assignment, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "fields"
            for target in assignment.targets
        ):
            continue
        try:
            fields = ast.literal_eval(assignment.value)
        except (ValueError, TypeError):
            return False
    if fields != {
        "initial_state": "initial_state",
        "standardized_state": "standardized_state",
        "raw_folded_state": "state",
    } or len(expected_models) * len(fields) != 21:
        return False

    method_nodes = parsed["interaction_methods.py"][1]
    method_fragments = {
        "mobius_transform": "mobius[..., present] -= mobius[..., lower]",
        "shapley_taylor_interactions": (
            "mobius[..., superset] / math.comb(superset.bit_count(), order)"
        ),
        "faith_kernel_weights": (
            "math.comb(n_players, size) * size * (n_players - size)"
        ),
        "faith_shap_interactions": (
            "solution = torch.linalg.solve(kkt, right_hand_side.T).T"
        ),
        "faith_shap_closed_form": (
            "math.comb(superset_size + max_order - 1, max_order + size)"
        ),
        "reconstruct_game": "values[..., present] += values[..., lower]",
        "faithfulness_error": "(weights * residual.square()).sum(dim=-1)",
        "masked_model_game": "sums.index_add_(0, slots, output)",
        "contextual_pair_synergy": (
            "game_values[..., with_pair] - game_values[..., with_first] - "
            "game_values[..., with_second] + game_values[..., contexts]"
        ),
    }
    if any(
        fragment not in ast.unparse(method_nodes[name])
        for name, fragment in method_fragments.items()
    ):
        return False

    run_tree, run_nodes = parsed["run_native.py"]
    run_node = run_nodes["run"]
    call_lines: dict[str, list[int]] = {}
    split_lines: dict[str, list[int]] = {"train": [], "explain": [], "audit": []}
    for node in ast.walk(run_node):
        if isinstance(node, ast.Call):
            call_lines.setdefault(ast.unparse(node.func), []).append(node.lineno)
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and ast.unparse(value.func) == "data_module.load_split"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value in split_lines
        ):
            continue
        split = value.args[0].value
        if any(
            isinstance(target, ast.Name) and target.id == split
            for target in node.targets
        ):
            split_lines[split].append(node.lineno)
    ordered = (
        "validate_freeze",
        "study.fit_known_models",
        "study.verify_known_reproduction",
        "_state_hash_gate",
        "study.prepare_study",
        "persist_prepared",
        "study.evaluate_study",
        "reference_module.validate_study_reference",
        "diagnostics_module.interaction_diagnostics",
        "study.native_play",
        "persist_completed",
    )
    if any(name not in call_lines for name in ordered):
        return False
    if any(len(lines) != 1 for lines in split_lines.values()):
        return False
    first = {name: min(call_lines[name]) for name in ordered}
    if not (
        first["validate_freeze"]
        < split_lines["train"][0]
        < first["study.fit_known_models"]
        and first["validate_freeze"]
        < split_lines["explain"][0]
        < first["study.fit_known_models"]
        < first["study.verify_known_reproduction"]
        < first["_state_hash_gate"]
        < first["study.prepare_study"]
        < first["persist_prepared"]
        < split_lines["audit"][0]
        < first["study.evaluate_study"]
        < first["reference_module.validate_study_reference"]
        < first["diagnostics_module.interaction_diagnostics"]
        < first["study.native_play"]
        < first["persist_completed"]
    ):
        return False

    state_gate = ast.unparse(run_nodes["_state_hash_gate"])
    if not all(
        fragment in state_gate
        for fragment in (
            "tuple(expected) != MODEL_KEYS",
            "expected != actual",
            "count != 21",
            "len(mismatches)",
        )
    ):
        return False
    prepared_saves = {
        call.args[1].value
        for call in ast.walk(run_nodes["persist_prepared"])
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "save_payload"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
    }
    if prepared_saves != {"fitted.pt", "prepared.pt"}:
        return False
    boundary_source = ast.unparse(run_nodes["write_before_audit"])
    explain_shard_write = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_persist_phase_hybrids"
        and len(call.args) >= 3
        and isinstance(call.args[2], ast.Constant)
        and call.args[2].value == "explain"
        for call in ast.walk(run_nodes["persist_prepared"])
    )
    if (
        not explain_shard_write
        or "before_audit.json" not in boundary_source
        or "audit_loaded" not in boundary_source
        or "state_hash_count" not in boundary_source
    ):
        return False

    reproduce_source = ast.unparse(parsed["reproduce_native.py"][1]["reproduce"])
    if not all(
        fragment in reproduce_source
        for fragment in (
            "subprocess.run(",
            "str(HERE / 'run_native.py')",
            "str(fresh_directory)",
            "compare_manifests(",
            "child_received_learner_payloads",
        )
    ):
        return False
    reference_nodes = parsed["interaction_reference.py"][1]
    package_source = ast.unparse(reference_nodes["_package_three"])
    if not all(label in package_source for label in ("'STII'", "'FSII'", "'SII'")):
        return False
    reference_source = ast.unparse(reference_nodes["_reference_check_impl"])
    if "values - values[:, :1]" not in reference_source:
        return False
    diagnostic_node = parsed["interaction_diagnostics.py"][1]["interaction_diagnostics"]
    pair_shape_checks = [
        call for call in ast.walk(diagnostic_node)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "_native_tensor"
        and len(call.args) >= 3
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "primary64.pair_values"
    ]
    if len(pair_shape_checks) != 1 or ast.unparse(pair_shape_checks[0].args[2]) != "(7, 272, 3, 28)":
        return False

    required_sources = {
        "interaction_data.py",
        "interaction_methods.py",
        "interaction_methods_tests.py",
        "interaction_study.py",
        "interaction_study_tests.py",
        "interaction_reference.py",
        "interaction_reference_tests.py",
        "interaction_diagnostics.py",
        "interaction_plots.py",
        "interaction_plots_tests.py",
        "protocol.md",
        "protocol_freeze.json",
        "data_manifest.json",
        "run_native.py",
    }
    source_literals = {
        node.value
        for node in ast.walk(run_nodes["source_paths"])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    } if "source_paths" in run_nodes else set()
    return required_sources <= source_literals


def _has_cartography_refit_contract(here: Path, fresh_names: set[str]) -> bool:
    """Follow actual SGD traces, data selection, held-out evaluation and reproduction."""
    if not {"run", "compare"} <= fresh_names:
        return False
    requirements = {
        "run_native.py": {
            "run": {"validate_freeze", "load_training", "map_training_data", "retrain_selections",
                    "load_evaluation", "evaluate_study", "retraining_play", "reference_checks"},
            "reference_checks": {"implementation.train_ensemble", "np.exp",
                                 "np.testing.assert_allclose", "torch.testing.assert_close"},
        },
        "cartography_data.py": {
            "load_training": {"source_arrays", "training_rows", "make_split"},
            "load_evaluation": {"source_arrays", "make_split"},
        },
        "cartography_methods.py": {
            "training_statistics": {"gold_probabilities", "probabilities.mean", "probabilities.std"},
            "train_ensemble": {"initialize_parameters", "batched_logits", "F.cross_entropy",
                               "torch.autograd.grad", "align_epoch", "value.add_"},
            "evaluate_models": {"batched_logits", "F.cross_entropy"},
        },
        "cartography_study.py": {
            "map_training_data": {"methods.train_ensemble", "methods.training_statistics", "methods.retrieval_curve"},
            "make_selections": {"methods.select_indices", "methods.matched_random_indices", "torch.randperm"},
            "retrain_selections": {"make_selections", "methods.train_ensemble", "torch.save"},
            "evaluate_study": {"methods.evaluate_models"},
            "retraining_play": {"methods.select_indices", "methods.matched_random_indices",
                                "methods.train_ensemble", "methods.evaluate_models"},
        },
        "cartography_reference.py": {
            "standard_training": {"torch.optim.SGD", "model", "optimizer.step", "F.cross_entropy"},
        },
    }
    for filename, functions in requirements.items():
        try:
            nodes = {node.name: node for node in ast.parse((here / filename).read_text()).body
                     if isinstance(node, ast.FunctionDef)}
        except (OSError, SyntaxError):
            return False
        for name, required in functions.items():
            if name not in nodes:
                return False
            calls = {ast.unparse(node.func) for node in ast.walk(nodes[name]) if isinstance(node, ast.Call)}
            if not required <= calls:
                return False
    return True


def _has_datamodel_refit_contract(here: Path, fresh_names: set[str]) -> bool:
    """Follow subset training, validation, frozen interventions and full comparison."""
    if not {"run", "compare"} <= fresh_names:
        return False
    requirements = {
        "run_native.py": {
            "run": {"load_data", "data_manifest", "study.make_designs", "study.prepare_study",
                    "study.evaluate_study", "study.deletion_play", "reference_checks"},
            "reference_checks": {"study.refit_masks", "scipy_logistic_optimum", "Lasso", "independent.predict"},
        },
        "datamodel_data.py": {"load_data": {"source_arrays"}},
        "datamodel_methods.py": {
            "fit_datamodel_path": {"fit_lasso"},
            "fit_lasso": {"soft_threshold", "lasso_kkt"},
        },
        "datamodel_study.py": {
            "make_designs": {"methods.sample_subsets"},
            "refit_masks": {"fit_weighted_logistic"},
            "model_margins": {"methods.correct_class_margins"},
            "fit_response_models": {"methods.fit_datamodel_path", "least_squares_datamodel"},
            "build_deletion_masks": {"methods.select_supportive", "methods.matched_random_removals"},
            "prepare_study": {"refit_masks", "model_margins", "fit_response_models", "build_deletion_masks"},
            "evaluate_study": {"refit_masks", "model_margins", "response_predictions"},
            "deletion_play": {"methods.select_supportive", "methods.matched_random_removals",
                              "refit_masks", "model_margins"},
        },
    }
    for filename, functions in requirements.items():
        try:
            nodes = {node.name: node for node in ast.parse((here / filename).read_text()).body
                     if isinstance(node, ast.FunctionDef)}
        except (OSError, SyntaxError):
            return False
        for name, required in functions.items():
            if name not in nodes:
                return False
            calls = {ast.unparse(node.func) for node in ast.walk(nodes[name]) if isinstance(node, ast.Call)}
            if not required <= calls:
                return False
            if filename == "run_native.py" and name == "reference_checks":
                fitted_lasso = any(
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "fit" and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == "Lasso"
                    for node in ast.walk(nodes[name])
                )
                if not fitted_lasso:
                    return False
    return True


def _has_data_attribution_refit_contract(here: Path, fresh_names: set[str]) -> bool:
    """Follow actual scratch-model fitting, counterfactuals and comparisons.

    A pretrained-model loader is not part of a from-scratch digit experiment.
    Require its complete data/refit/evaluation path instead. This recognizes
    structure; source hashes and numerical reproduction remain separate gates.
    """
    if not {"load_study_data", "data_manifest", "prepare_study", "evaluate_final",
            "query_deletion_play", "compare", "scipy_logistic_optimum"} <= fresh_names:
        return False
    requirements = {
        "attribution_study.py": {
            "prepare_study": {"fit_condition"},
            "fit_condition": {"m.fit_weighted_logistic", "m.sgd_trace", "score_queries",
                              "m.class_matched_removals", "m.conjugate_gradient", "m.logistic_hvp"},
            "score_queries": {"query_outputs", "m.upweight_influence",
                              "m.trace_loss_reductions", "checkpoint_analysis"},
            "checkpoint_analysis": {"m.tracin_checkpoint_scores", "m.ghost_linear_products",
                                    "torch.testing.assert_close"},
            "evaluate_final": {"score_queries", "query_outputs"},
        },
        "attribution_data.py": {
            "load_study_data": {"source_arrays"},
            "source_arrays": {"sha256", "zipfile.ZipFile", "np.loadtxt"},
        },
        "attribution_play.py": {
            "query_deletion_play": {"m.upweight_influence", "m.class_matched_removals",
                                    "m.fit_weighted_logistic"},
        },
    }
    for filename, functions in requirements.items():
        try:
            nodes = {node.name: node for node in ast.parse((here / filename).read_text()).body
                     if isinstance(node, ast.FunctionDef)}
        except (OSError, SyntaxError):
            return False
        for name, required in functions.items():
            if name not in nodes:
                return False
            calls = {ast.unparse(call.func) for call in ast.walk(nodes[name]) if isinstance(call, ast.Call)}
            if not required <= calls:
                return False
    return True


def _has_ccs_native_refit_contract(here: Path, fresh_node: ast.FunctionDef, names: set[str]) -> bool:
    """Structural coverage only; execution and artifact hashes have separate gates."""
    if not {"validate_learner", "compare_nested", "validate_complete_result"} <= names:
        return False

    def calls(node):
        return {ast.unparse(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)}

    if not {"native.load_model", "native.check_capture", "native.capture_pairs", "native.weight_digest",
            "study.fit_reporters", "study.calibrate_reporters", "study.reporter_scores",
            "study.evaluate_reporters", "study.summarize_native_outputs"} <= calls(fresh_node):
        return False
    try:
        native_tree = ast.parse((here / "ccs_native.py").read_text())
        study_tree = ast.parse((here / "ccs_study.py").read_text())
    except (OSError, SyntaxError):
        return False
    native = {node.name: node for node in native_tree.body if isinstance(node, ast.FunctionDef)}
    study = {node.name: node for node in study_tree.body if isinstance(node, ast.FunctionDef)}
    requirements = [
        (native, "load_model", {"AutoModelForCausalLM.from_pretrained"}),
        (native, "capture_pairs", {"capture_texts"}),
        (native, "capture_texts", {"model.transformer", "model.lm_head", "gather"}),
        (native, "check_capture", {"model", "capture_texts", "torch.testing.assert_close"}),
        (study, "fit_reporters", {"m.fit_pair_centering", "m.apply_pair_centering", "m.fit_ccs_restarts",
                                  "m.select_restart", "m.fit_difference_pca", "fit_binary_probe", "torch.randn"}),
        (study, "calibrate_reporters", {"reporter_scores", "m.calibrate_orientation"}),
        (study, "reporter_scores", {"m.apply_pair_centering", "m.paired_probabilities"}),
        (study, "evaluate_reporters", {"m.binary_metrics", "nm.paired_opinion_change"}),
        (study, "summarize_native_outputs", {"nm.model_answer_metrics"}),
    ]
    return all(name in functions and required <= calls(functions[name])
               for functions, name, required in requirements)


def chapter_surface_blockers(config: dict[str, Any], chapter_name: str) -> list[str]:
    chapter_dir = ROOT / chapter_name
    instructions_dir = chapter_dir / "instructions"
    blockers: list[str] = []

    home_path = instructions_dir / "Home.py"
    if not home_path.exists():
        blockers.append(f"{chapter_name}: missing instructions/Home.py")
    elif "render_extension_home(__file__)" not in home_path.read_text():
        blockers.append(f"{chapter_name}: Home.py does not use shared extension renderer")

    config_path = instructions_dir / ".streamlit/config.toml"
    if not config_path.exists():
        blockers.append(f"{chapter_name}: missing instructions/.streamlit/config.toml")
    else:
        config_text = config_path.read_text()
        if "[theme]" not in config_text:
            blockers.append(f"{chapter_name}: streamlit config missing [theme]")
        for key in REQUIRED_STREAMLIT_THEME_KEYS:
            if key not in config_text:
                blockers.append(f"{chapter_name}: streamlit config missing {key}")

    requirements_path = instructions_dir / "requirements.txt"
    if not requirements_path.exists():
        blockers.append(f"{chapter_name}: missing instructions/requirements.txt")
    else:
        missing = REQUIRED_INSTRUCTION_REQUIREMENTS - _requirement_lines(requirements_path)
        if missing:
            blockers.append(f"{chapter_name}: instruction requirements missing {sorted(missing)}")

    return blockers


def section_surface_blockers(
    config: dict[str, Any],
    chapter_name: str,
    section: dict[str, Any],
) -> list[str]:
    chapter_dir = ROOT / chapter_name
    number = str(section.get("number", ""))
    title = str(section.get("title", ""))
    blockers: list[str] = []

    conversion = config["conversion_map"].get(number)
    if conversion is None:
        blockers.append(f"{number}: missing conversion_map entry")
        return blockers
    if conversion.get("exercise_dir") != section.get("exercise_dir"):
        blockers.append(f"{number}: conversion_map exercise_dir mismatch")
    expected_page = f"{conversion.get('streamlit_page')}.md"
    if expected_page != section.get("page_file"):
        blockers.append(f"{number}: conversion_map streamlit_page does not match page_file")

    page_path = chapter_dir / "instructions/pages" / str(section.get("page_file", ""))
    if not page_path.exists():
        blockers.append(f"{number}: missing instruction page {page_path.relative_to(ROOT)}")
    elif not _page_has_arena_shape(page_path.read_text(), number, title):
        blockers.append(f"{number}: instruction page lacks ARENA-style learner surface")

    exercise_dir = chapter_dir / "exercises" / str(section.get("exercise_dir", ""))
    if not exercise_dir.exists():
        blockers.append(f"{number}: missing exercise dir {exercise_dir.relative_to(ROOT)}")
    for required in ("README.md", "solutions.py", "tests.py", "verification_report.json"):
        required_path = exercise_dir / required
        if not required_path.exists():
            blockers.append(f"{number}: missing {required_path.relative_to(ROOT)}")

    solutions_path = exercise_dir / "solutions.py"
    if solutions_path.exists() and "def run_gpu_test" in solutions_path.read_text():
        exercise_notebooks = sorted(exercise_dir.glob("*_exercises.ipynb"))
        if not exercise_notebooks:
            blockers.append(f"{number}: missing exercise notebook with verification contract")
        else:
            for notebook_path in exercise_notebooks:
                try:
                    has_contract = _exercise_notebook_declares_verification_contract(notebook_path)
                except json.JSONDecodeError as exc:
                    blockers.append(
                        f"{number}: invalid notebook JSON in {notebook_path.relative_to(ROOT)}: {exc}"
                    )
                    continue
                if not has_contract:
                    blockers.append(
                        f"{number}: exercise notebook lacks a complete learner execution or GPU/full verification surface"
                    )

    return blockers


def course_surface_blockers(config: dict[str, Any] | None = None) -> list[str]:
    config = config or load_config()
    blockers: list[str] = []

    for chapter_name in extension_chapter_names(config):
        blockers.extend(chapter_surface_blockers(config, chapter_name))

    for chapter_name, chapter in config["chapters"].items():
        for section in chapter.get("sections", []):
            number = str(section.get("number", ""))
            if is_extension_section(number):
                blockers.extend(section_surface_blockers(config, chapter_name, section))

    return blockers


def main() -> None:
    blockers = course_surface_blockers()
    print(f"extension_chapters_checked={len(extension_chapter_names(load_config()))}")
    print(f"course_surface_blockers={len(blockers)}")
    if blockers:
        print("COURSE_SURFACE=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("COURSE_SURFACE=PASS")


if __name__ == "__main__":
    main()
