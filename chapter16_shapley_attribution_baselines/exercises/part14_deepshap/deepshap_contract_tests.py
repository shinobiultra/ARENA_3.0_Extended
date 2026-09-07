"""Contracts for frozen data, visible own answers, fresh fits, and evidence gates."""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chapter16_shapley_attribution_baselines.exercises.part14_deepshap import run_native


HERE = Path(__file__).resolve().parent
METHOD_NAMES = (
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
PLUMBING_IMPORTS = {
    "deepshap_methods": {
        "_check_widths",
        "_epsilon",
        "_layers",
        "_method_tensor",
        "_model_output",
    },
    "deepshap_diagnostics": {"DTYPE", "_linear", "_max_abs", "_method", "_prediction"},
    "deepshap_study": {
        "_clone_state",
        "_clone_tree",
        "_combine_explanations",
        "_combine_row_blocks",
        "_float_tensor",
        "_ids_tensor",
        "_normalization_tensors",
        "_output_tensor",
        "_require_disjoint_ids",
        "_require_methods",
        "_rows",
        "_state_shapes",
        "_validate_evaluation_inputs",
        "_validate_fitted",
        "_validate_rankings",
    },
}
VISIBLE_PRIVATE_STUDY_FUNCTIONS = {
    "_model_from_state",
    "_fold_mlp_state",
    "_fold_ridge_state",
    "_folding_error",
    "_training_normalization",
    "_train_mlp",
    "_fold_mlp_result",
    "_explain_model",
    "_background_indices",
    "_ranked_curves",
    "_random_curves",
    "_performance_table",
    "_fixed_random_orders",
    "_evaluate_replacements",
    "_evaluate_background_variants",
}


def _module(name: str):
    return importlib.import_module(
        f"chapter16_shapley_attribution_baselines.exercises.part14_deepshap.{name}"
    )


def test_frozen_data_and_recipe_separation():
    freeze = run_native.validate_freeze()
    assert set(freeze["sha256"]) == {"protocol.md", "data_manifest.json", "deepshap_data.py"}, (
        "The pre-fit freeze must bind the protocol, manifest, and exact data loader."
    )
    data_module = _module("deepshap_data")
    data = data_module.source_data()
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    groups = []
    all_ids = []
    for name, count in (("train", 758), ("explain", 153), ("audit", 119)):
        ids = manifest["splits"][name]
        assert len(ids) == count, f"The frozen {name} split must retain exactly {count} rows."
        groups.append({tuple(row) for row in data[ids, :7]})
        all_ids.extend(ids)
    assert sorted(all_ids) == list(range(1030)), "The three splits must cover all source rows once."
    assert all(
        not groups[left] & groups[right]
        for left in range(3)
        for right in range(left)
    ), "Recipe groups must remain disjoint across train, explanation, and audit splits."
    assert len(set.union(*groups)) == 427, "The frozen data must retain all 427 recipe groups."
    assert manifest["cross_split_recipe_overlap"] == 0, (
        "The manifest must explicitly record zero cross-split recipe overlap."
    )


def test_study_source_declares_all_fits_without_loading_evaluation_data():
    study = _module("deepshap_study")
    signature = inspect.signature(study.fit_models)
    assert list(signature.parameters) == ["train"], (
        "fit_models must accept training data only; explanation and audit data cannot influence fits."
    )
    source = (HERE / "deepshap_study.py").read_text()
    tree = ast.parse(source)
    fit_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fit_models"
    )
    fit_source = ast.get_source_segment(source, fit_node) or ""
    assert "load_split" not in fit_source, "fit_models must not load hidden evaluation splits."
    assert tuple(study.MODEL_KEYS) == (
        "clean_0",
        "clean_1",
        "clean_2",
        "shuffled_0",
        "shuffled_1",
        "shuffled_2",
        "ridge",
    ), "The visible study must retain all clean, shuffled-target, and ridge model branches."


def test_visible_study_call_graph_keeps_training_explanation_and_controls_explicit():
    tree = ast.parse((HERE / "deepshap_study.py").read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    def calls(name: str) -> set[str]:
        result = set()
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                result.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                result.add(node.func.attr)
        return result

    expected = {
        "_train_mlp": {"Adam", "zero_grad", "backward", "step"},
        "fit_one_mlp_control": {
            "Linear",
            "ReLU",
            "Sequential",
            "manual_seed",
            "_train_mlp",
            "_fold_mlp_result",
        },
        "fit_ridge_control": {"solve", "_fold_ridge_state"},
        "fit_models": {
            "_training_normalization",
            "fit_one_mlp_control",
            "fit_ridge_control",
        },
        "_explain_model": {
            "background_deepshap",
            "rescale_multipliers",
            "coalition_masks",
            "coalition_values",
            "exact_shapley",
            "gradient_delta",
        },
        "evaluate_study": {
            "explain_block",
            "_performance_table",
            "_fixed_random_orders",
            "_evaluate_replacements",
            "_combine_row_blocks",
            "_evaluate_background_variants",
        },
        "native_play": {
            "background_order",
            "_explain_model",
            "ranked_replacement_curves",
        },
    }
    for name, required in expected.items():
        missing = required - calls(name)
        assert not missing, f"The visible {name} implementation is missing calls {sorted(missing)}."


def _visible_functions(notebook: dict) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assert_only_plumbing_imports(source: str) -> None:
    """Reject shortcuts into the three learner-owned scientific modules."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.rsplit(".", 1)[-1]
                assert module not in PLUMBING_IMPORTS, (
                    f"The learner cannot import the complete {module} module; "
                    "its scientific functions must remain visible."
                )
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").rsplit(".", 1)[-1]
            imported_names = {alias.name for alias in node.names}
            assert not (imported_names & set(PLUMBING_IMPORTS)), (
                "The learner cannot import a complete scientific module from its package."
            )
            if module not in PLUMBING_IMPORTS:
                continue
            assert "*" not in imported_names, (
                f"Wildcard imports from {module} can hide the method being taught."
            )
            unexpected = imported_names - PLUMBING_IMPORTS[module]
            assert not unexpected, (
                f"Only declared mechanical plumbing may be imported from {module}; "
                f"found {sorted(unexpected)}."
            )
            assert all(alias.asname is None for alias in node.names), (
                f"Plumbing imports from {module} must keep their declared names."
            )


def test_core_study_and_diagnostics_are_visible_in_own_answers():
    build = _module("build_notebooks")
    complete = build.complete_learner({"cells": build.lesson_cells()})
    visible = _visible_functions(complete)
    methods_tree = ast.parse((HERE / "deepshap_methods.py").read_text())
    method_functions = [node for node in methods_tree.body if isinstance(node, ast.FunctionDef)]
    assert tuple(node.name for node in method_functions if node.name in METHOD_NAMES) == METHOD_NAMES, (
        "The methods module must define the nine learner APIs in the frozen pedagogical order."
    )
    for filename in ("deepshap_methods.py", "deepshap_study.py", "deepshap_diagnostics.py"):
        for node in ast.parse((HERE / filename).read_text()).body:
            required_private = (
                filename == "deepshap_study.py"
                and node.name in VISIBLE_PRIVATE_STUDY_FUNCTIONS
                if isinstance(node, ast.FunctionDef)
                else False
            )
            if isinstance(node, ast.FunctionDef) and (
                not node.name.startswith("_") or required_private
            ):
                assert visible.get(node.name) == ast.dump(node, include_attributes=False), (
                    f"The own-answer learner must visibly contain the exact {node.name} definition."
                )
    for cell in complete["cells"]:
        if cell["cell_type"] == "code":
            _assert_only_plumbing_imports("".join(cell["source"]))


def test_core_module_and_wildcard_import_shortcuts_are_rejected():
    forbidden = (
        "from part14_deepshap.deepshap_methods import paired_forward",
        "from part14_deepshap.deepshap_study import fit_models",
        "from part14_deepshap.deepshap_diagnostics import *",
        "import part14_deepshap.deepshap_methods as methods",
        "from part14_deepshap import deepshap_study as study",
    )
    for source in forbidden:
        with pytest.raises(AssertionError):
            _assert_only_plumbing_imports(source)


def test_missing_own_answer_is_rejected():
    build = _module("build_notebooks")
    notebook = {"cells": copy.deepcopy(build.lesson_cells())}
    index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code" and "raise NotImplementedError" in "".join(cell["source"])
    )
    notebook["cells"][index + 1]["source"] = ["The required answer was removed."]
    with pytest.raises(ValueError):
        build.complete_learner(notebook)


def _top_level_calls(source: str) -> list[tuple[str, list[object]]]:
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        arguments = [arg.value if isinstance(arg, ast.Constant) else None for arg in node.args]
        calls.append((node.func.id, arguments))
    return calls


def test_own_answer_seals_prepared_payload_before_literal_audit_load():
    build = _module("build_notebooks")
    complete = build.complete_learner({"cells": build.lesson_cells()})
    calls = []
    for cell_index, cell in enumerate(complete["cells"]):
        if cell["cell_type"] == "code":
            calls.extend(
                (cell_index, name, arguments)
                for name, arguments in _top_level_calls("".join(cell["source"]))
            )
    fitted_save = next(
        index
        for index, name, arguments in calls
        if name == "save_payload" and "fitted.pt" in arguments
    )
    prepared_save = next(
        index
        for index, name, arguments in calls
        if name == "save_payload" and "prepared.pt" in arguments
    )
    boundary = next(index for index, name, _ in calls if name == "write_before_audit")
    audit_load = next(
        index
        for index, name, arguments in calls
        if name == "load_split" and arguments and arguments[0] == "audit"
    )
    evaluation = next(index for index, name, _ in calls if name == "evaluate_study")
    assert max(fitted_save, prepared_save) <= boundary < audit_load <= evaluation, (
        "Fitted states and explanation rankings must be sealed before the literal audit load."
    )


def test_prepared_boundary_saves_states_and_rejects_live_modules(tmp_path):
    fitted = {
        "models": {"clean_0": {"state": {"0.weight": torch.ones(1, 2)}}},
        "train_ids": np.array([1, 2], dtype=np.int64),
    }
    prepared = {
        "audit_used_for_selection": False,
        "explain_ids": np.array([3], dtype=np.int64),
        "background_ids": np.array([1], dtype=np.int64),
    }
    run_native.persist_prepared(tmp_path, fitted, prepared)
    saved = run_native.load_payload(tmp_path / "fitted.pt")
    assert torch.equal(saved["models"]["clean_0"]["state"]["0.weight"], torch.ones(1, 2)), (
        "The fitted payload must preserve the complete numerical state needed to rebuild a model."
    )
    boundary = json.loads((tmp_path / "before_audit.json").read_text())
    assert boundary["audit_loaded"] is False, (
        "The persisted boundary must state that audit data had not yet been loaded."
    )
    with pytest.raises(TypeError, match="states"):
        run_native.save_payload(
            tmp_path,
            "fitted.pt",
            {"models": {"bad": {"model": torch.nn.Sequential(torch.nn.Linear(2, 1))}}},
        )
    run_native.save_payload(tmp_path, "prepared.pt", {**prepared, "changed": True})
    with pytest.raises(ValueError, match="changed"):
        run_native.persist_completed(
            tmp_path, fitted, prepared, {}, {"mechanism": {}, "timing": {}}, {}, {}
        )


@pytest.mark.parametrize(
    "mutation", ["value", "shape", "dtype", "missing", "scalar_type", "order", "tensor"]
)
def test_complete_reproduction_rejects_mutations(mutation):
    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.reproduce_native import (
        compare,
    )

    expected = {
        "values": np.array([1.0, 2.0], dtype=np.float64),
        "ids": np.array([3, 5], dtype=np.int64),
        "tensor": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "count": 2,
    }
    actual = copy.deepcopy(expected)
    if mutation == "value":
        actual["values"][0] = 10
    elif mutation == "shape":
        actual["values"] = actual["values"][None]
    elif mutation == "dtype":
        actual["values"] = actual["values"].astype(np.float32)
    elif mutation == "missing":
        actual.pop("ids")
    elif mutation == "scalar_type":
        actual["count"] = 2.0
    elif mutation == "order":
        actual["ids"] = actual["ids"][::-1]
    elif mutation == "tensor":
        actual["tensor"][0] = 9
    counts = {"arrays": 0, "array_elements": 0, "scalars": 0, "max_absolute_error": 0.0}
    with pytest.raises((ValueError, AssertionError)):
        compare(actual, expected, "native", counts)


def test_publisher_uses_ast_literal_figure_calls_and_namespaces_assets():
    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.publish_study import (
        _show_figure_stems,
        published_asset_name,
    )

    source = "mentioned = \"show_figure(fig, 'fake')\"\nshow_figure(fig, '01_real')\n"
    assert _show_figure_stems(source) == ["01_real"], (
        "Only actual literal show_figure calls may satisfy publication evidence."
    )
    assert published_asset_name("01_real.png") == "deepshap_01_real.png", (
        "Published DeepSHAP figures must be namespaced in the shared chapter asset directory."
    )


def test_no_gpu_execution_calls_in_native_pipeline():
    for name in ("run_native.py", "execute_learner.py", "reproduce_native.py", "solutions.py"):
        source = (HERE / name).read_text()
        assert "torch.cuda" not in source, f"{name} must remain a CPU-only execution path."
        assert ".cuda(" not in source, f"{name} must never move a model or tensor onto CUDA."


def test_every_scientific_source_is_snapshot_bound():
    source = (HERE / "run_native.py").read_text()
    listed = {
        item.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Tuple)
        for item in node.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    required = {
        "deepshap_data.py",
        "deepshap_methods.py",
        "deepshap_methods_tests.py",
        "deepshap_study.py",
        "deepshap_study_tests.py",
        "deepshap_reference.py",
        "deepshap_reference_tests.py",
        "deepshap_diagnostics.py",
        "deepshap_plots.py",
        "author_reference.json",
        "protocol.md",
        "protocol_freeze.json",
        "data_manifest.json",
        "run_native.py",
    }
    assert required <= listed, "Every scientific source must be copied and hash-bound before fitting."
