"""Semantic contracts for the frozen study, learner, and exact evidence harness."""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import inspect
import json
from pathlib import Path
import symtable

import numpy as np
import pytest
import torch

from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap import run_native


HERE = Path(__file__).resolve().parent
METHOD_NAMES = (
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
MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
PLUMBING_IMPORTS = {
    "gradientshap_methods": {
        "_eg_float_tensor",
        "_eg_index_tensor",
        "_eg_model",
        "_eg_model_output",
        "_eg_positive_int",
        "_eg_preserve_execution_state",
        "_eg_source_ids",
        "_eg_seed_words",
        "_eg_validate_paths",
    },
    "gradientshap_study": {
        "_batched_prediction",
        "_checked_curves",
        "_clone_state",
        "_combine_baselines",
        "_combine_mc",
        "_combine_primary",
        "_combine_quadrature",
        "_float_tensor",
        "_fixed_random_orders",
        "_folding_error",
        "_identity_hash",
        "_ids_tensor",
        "_normalization_tensors",
        "_output_tensor",
        "_require_disjoint_ids",
        "_require_namespace",
        "_rows",
        "_seed_words",
        "_state_shapes",
        "_validate_evaluation_inputs",
        "_validate_fitted",
    },
}


def _module(name: str):
    return importlib.import_module(
        f"chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.{name}"
    )


def test_zero_native_run_freeze_and_all_predeclared_identities():
    freeze = run_native.validate_freeze()
    assert set(freeze["sha256"]) == {
        "protocol.md",
        "data_manifest.json",
        "gradientshap_data.py",
    }, "The freeze must bind the protocol, data declaration, and exact loader."
    assert freeze["native_fits_before_freeze"] == 0, (
        "The frozen protocol must precede every native 16.15 fit."
    )
    assert freeze["native_attribution_runs_before_freeze"] == 0, (
        "The frozen protocol must precede every native GradientSHAP attribution run."
    )
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    states = manifest["known_16_14_evidence"]["model_state_hashes"]
    assert tuple(states) == MODEL_KEYS, "All seven known 16.14 predictors must remain declared."
    assert sum(len(item) for item in states.values()) == 21, (
        "Initial, standardized, and raw-folded hashes are required for all seven predictors."
    )
    design = manifest["sampling_design_candidate"]
    assert len(design["replicate_seeds"]) == len(set(design["replicate_seeds"])) == 64, (
        "The manifest must retain all 64 independently derived replicate seeds."
    )
    assert design["nested_budgets"] == [8, 32, 128, 512, 2048], (
        "Nested Monte Carlo budgets must remain literal prefixes."
    )
    assert design["same_draw_tolerance"] == {"atol": 1e-10, "rtol": 1e-10}, (
        "Pinned package parity must keep its reviewed same-draw tolerance."
    )


def test_frozen_data_recipe_groups_and_predeclared_replicate_rows():
    data_module = _module("gradientshap_data")
    data = data_module.source_data()
    manifest = data_module.load_manifest()
    groups = []
    all_ids = []
    for name, count in (("train", 758), ("explain", 153), ("audit", 119)):
        ids = manifest["splits"][name]
        assert len(ids) == count, f"The {name} split must retain exactly {count} rows."
        groups.append({tuple(row) for row in data[ids, :7]})
        all_ids.extend(ids)
    assert sorted(all_ids) == list(range(1030)), "The split must cover every source row once."
    assert all(
        not groups[left] & groups[right]
        for left in range(3)
        for right in range(left)
    ), "Recipe groups must remain disjoint across train, explanation, and audit."
    replicate = manifest["replicate_diagnostic_candidate"]
    assert len(replicate["explanation_source_ids"]) == 32, (
        "The replicate diagnostic needs its first 32 explanation identities."
    )
    assert len(replicate["audit_source_ids"]) == 32, (
        "The replicate diagnostic needs its separately declared 32 audit identities."
    )
    assert set(replicate["explanation_source_ids"]) <= set(manifest["splits"]["explain"]), (
        "Replicate explanation identities must come only from the explanation split."
    )
    assert set(replicate["audit_source_ids"]) <= set(manifest["splits"]["audit"]), (
        "Replicate audit identities must come only from the audit split."
    )


def _calls(function: ast.FunctionDef) -> set[str]:
    result = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            result.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            result.add(node.func.attr)
    return result


def test_visible_study_declares_seven_refits_full_arrays_and_controls():
    study = _module("gradientshap_study")
    assert list(inspect.signature(study.fit_models).parameters) == ["train"], (
        "fit_models may receive training data only."
    )
    assert "audit" not in inspect.signature(study.prepare_study).parameters, (
        "Preparation cannot receive audit values."
    )
    assert "audit" in inspect.signature(study.evaluate_study).parameters, (
        "Audit data must enter only the post-seal evaluation."
    )
    assert tuple(study.MODEL_KEYS) == MODEL_KEYS, (
        "The visible study must retain three clean, three shuffled, and one ridge fit."
    )
    source = (HERE / "gradientshap_study.py").read_text()
    functions = {
        node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    }
    expected_calls = {
        "_train_mlp": {"Adam", "zero_grad", "backward", "step"},
        "fit_one_mlp_control": {"Linear", "ReLU", "Sequential", "manual_seed", "_train_mlp"},
        "fit_models": {"_training_normalization", "fit_one_mlp_control", "fit_ridge_control"},
        "prepare_study": {
            "verify_16_14_reproduction",
            "_primary_schedule",
            "_monte_carlo_block",
            "_quadrature_block",
            "_comparison_baselines",
            "_global_rankings",
        },
        "evaluate_study": {
            "_performance_table",
            "_background_comparison",
            "_noise_comparison",
            "_replicate_comparison",
            "_distance_comparison",
            "_replacement_evaluation",
        },
        "native_play": {
            "sample_path_schedule",
            "summarize_expected_gradients",
            "_quadrature_block",
            "_comparison_baselines",
            "ranked_coalition_curves",
        },
    }
    for name, required in expected_calls.items():
        missing = required - _calls(functions[name])
        assert not missing, f"The visible {name} call graph is missing {sorted(missing)}."
    comparison_source = ast.get_source_segment(source, functions["_comparison_baselines"]) or ""
    assert "hybrid_prediction" in comparison_source, (
        "Every query/coalition/donor prediction must be retained, not only its donor mean."
    )


def test_immediate_input_gradient_check_is_stable_under_lesson_default_dtype():
    method_tests = _module("gradientshap_methods_tests")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        method_tests.test_input_gradients()
    finally:
        torch.set_default_dtype(previous)


def _assert_only_plumbing_imports(source: str) -> None:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.rsplit(".", 1)[-1]
                assert module not in PLUMBING_IMPORTS, (
                    f"The learner cannot import the complete {module} implementation."
                )
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").rsplit(".", 1)[-1]
            imported = {alias.name for alias in node.names}
            if module not in PLUMBING_IMPORTS:
                continue
            assert "*" not in imported, f"Wildcard import hides the taught {module} logic."
            unexpected = imported - PLUMBING_IMPORTS[module]
            assert not unexpected, f"Unexpected hidden {module} imports: {sorted(unexpected)}."


def _visible_functions(notebook: dict) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_core_methods_and_study_are_visible_in_complete_own_answers():
    build = _module("build_notebooks")
    complete = build.complete_learner({"cells": build.lesson_cells()})
    visible = _visible_functions(complete)
    method_tree = ast.parse((HERE / "gradientshap_methods.py").read_text())
    method_nodes = [
        node for node in method_tree.body if isinstance(node, ast.FunctionDef) and node.name in METHOD_NAMES
    ]
    assert tuple(node.name for node in method_nodes) == METHOD_NAMES, (
        "The nine methods must stay in the frozen teaching order."
    )
    for node in method_nodes:
        assert visible.get(node.name) == ast.dump(node, include_attributes=False), (
            f"The own-answer learner must visibly contain the exact {node.name} implementation."
        )
    required_study = {
        "_training_normalization",
        "_train_mlp",
        "_fold_mlp_state",
        "_fold_ridge_state",
        "fit_one_mlp_control",
        "fit_ridge_control",
        "fit_models",
        "verify_16_14_reproduction",
        "_primary_schedule",
        "_monte_carlo_block",
        "_quadrature_block",
        "_comparison_baselines",
        "prepare_study",
        "_background_comparison",
        "_noise_comparison",
        "_replicate_comparison",
        "_distance_comparison",
        "_replacement_evaluation",
        "evaluate_study",
        "native_play",
    }
    for node in ast.parse((HERE / "gradientshap_study.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in required_study:
            assert visible.get(node.name) == ast.dump(node, include_attributes=False), (
                f"The own-answer learner must visibly contain the exact {node.name} study logic."
            )
    for cell in complete["cells"]:
        if cell["cell_type"] == "code":
            _assert_only_plumbing_imports("".join(cell["source"]))


def _top_level_calls(source: str) -> list[tuple[str, list[object]]]:
    calls = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
            else:
                continue
        else:
            call = node.value
        if isinstance(call.func, ast.Name):
            arguments = [arg.value if isinstance(arg, ast.Constant) else None for arg in call.args]
            calls.append((call.func.id, arguments))
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
    preparation = next(index for index, name, _ in calls if name == "prepare_study")
    boundary = next(index for index, name, _ in calls if name == "persist_prepared")
    audit_load = next(
        index
        for index, name, arguments in calls
        if name == "load_split" and arguments and arguments[0] == "audit"
    )
    evaluation = next(index for index, name, _ in calls if name == "evaluate_study")
    completion = next(index for index, name, _ in calls if name == "persist_completed")
    assert preparation <= boundary < audit_load <= evaluation <= completion, (
        "Fitted states and explanation rankings must be saved and sealed before literal audit load."
    )


def test_complete_own_answer_has_no_unresolved_function_globals():
    build = _module("build_notebooks")
    complete = build.complete_learner({"cells": build.lesson_cells()})
    source = "\n\n".join(
        "".join(cell["source"])
        for cell in complete["cells"]
        if cell["cell_type"] == "code"
    )
    table = symtable.symtable(source, "completed_gradientshap.py", "exec")
    module_defined = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    builtin_names = set(dir(builtins))
    missing: dict[str, list[str]] = {}

    def walk(scope) -> None:
        for child in scope.get_children():
            unresolved = sorted(
                {
                    symbol.get_name()
                    for symbol in child.get_symbols()
                    if symbol.is_global() and symbol.is_referenced()
                }
                - module_defined
                - builtin_names
            )
            if unresolved:
                missing[f"{child.get_type()}:{child.get_name()}:{child.get_lineno()}"] = unresolved
            walk(child)

    walk(table)
    assert not missing, f"Visible own-answer functions have unresolved globals: {missing}."


def _fake_reproduction() -> dict:
    hashes = {
        key: {
            "initial_state": f"{3 * index:064x}",
            "standardized_state": f"{3 * index + 1:064x}",
            "raw_folded_state": f"{3 * index + 2:064x}",
        }
        for index, key in enumerate(MODEL_KEYS)
    }
    return {
        "status": "PASS",
        "expected_model_state_hashes": hashes,
        "model_state_hashes": copy.deepcopy(hashes),
        "mismatches": (),
    }


def test_pre_audit_boundary_and_external_replicate_payload_are_exact(tmp_path, monkeypatch):
    small_shape = (2, 3, 2, 4, 2)
    monkeypatch.setattr(run_native, "REPLICATE_SHAPE", small_shape)
    monkeypatch.setattr(run_native, "REPLICATE_ELEMENTS", int(np.prod(small_shape)))
    monkeypatch.setattr(run_native, "BUDGETS", (2, 4))
    fitted = {"models": {}, "train_ids": torch.tensor([10, 11], dtype=torch.int64)}
    prepared = {
        "phase": "prepared_before_audit",
        "audit_used_for_selection": False,
        "reproduction": _fake_reproduction(),
        "explain_ids": torch.tensor([20], dtype=torch.int64),
        "backgrounds": {
            "primary64": {"source_ids": torch.tensor([10], dtype=torch.int64)}
        },
        "primary": {},
    }
    run_native.persist_prepared(tmp_path, fitted, prepared)
    boundary = json.loads((tmp_path / "before_audit.json").read_text())
    assert boundary["audit_loaded"] is False, "The durable seal must precede audit loading."
    assert boundary["state_hash_count"] == 21, "The durable seal must bind all state hashes."

    samples = torch.arange(np.prod(small_shape), dtype=torch.float64).reshape(small_shape)
    evaluated = {
        "phase": "evaluated_after_audit",
        "audit_used_for_selection": False,
        "audit_ids": torch.tensor([30], dtype=torch.int64),
        "sampling_replicates": {
            "source_ids": torch.tensor([20, 30], dtype=torch.int64),
            "row_split": torch.tensor([0, 1], dtype=torch.int64),
            "split_labels": ("explanation", "audit"),
            "models": run_native.REPLICATE_MODELS,
            "model_labels": ("clean", "shuffle", "ridge"),
            "budgets": torch.tensor([2, 4], dtype=torch.int64),
            "replicate_seeds": torch.tensor([1, 2], dtype=torch.uint64),
            "baseline_index": torch.zeros((2, 2, 4), dtype=torch.int64),
            "alpha": torch.zeros((2, 2, 4), dtype=torch.float64),
            "samples": samples,
            "phi_prefix": torch.zeros((2, 3, 2, 2, 2), dtype=torch.float64),
            "se_prefix": torch.zeros((2, 3, 2, 2, 2), dtype=torch.float64),
            "q512": torch.zeros((3, 2, 2), dtype=torch.float64),
            "q512_error": torch.zeros((2, 3, 2, 2, 2), dtype=torch.float64),
            "coverage_95": torch.zeros((2, 3, 2, 2, 2), dtype=torch.bool),
            "coverage_rate": torch.zeros((3, 2), dtype=torch.float64),
        },
        "audit_primary": {},
        "background_variants": {},
        "noise_variants": {},
    }
    run_native.persist_completed(
        tmp_path, fitted, prepared, evaluated, {}, {"status": "PASS"}, {}, {}
    )
    assert evaluated["sampling_replicates"]["samples"] is samples, (
        "Externalization must not replace or copy the canonical in-memory sample branch."
    )
    stored = run_native.load_payload(tmp_path / "evaluated.pt")
    reference = stored["sampling_replicates"]["samples"]
    assert reference["storage"] == "external_pt", "evaluated.pt needs an explicit external ref."
    assert reference["shape"] == list(small_shape) and reference["dtype"] == "float64", (
        "The external reference must retain exact shape and dtype."
    )
    external = run_native.load_payload(tmp_path / "replicate_samples.pt")
    assert torch.equal(external, samples), "The full external sample array must be retained exactly."
    assert reference["value_sha256"] == run_native._array_value_sha256(external), (
        "The evaluated reference must bind the external array values."
    )
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.reproduce_native import (
        validate_artifact_directory,
    )

    validate_artifact_directory(tmp_path)


def test_recursive_manifest_rejects_value_shape_dtype_and_order_mutations(tmp_path):
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.reproduce_native import (
        compare_manifests,
    )

    expected = {
        "values": np.array([1.0, 2.0], dtype=np.float64),
        "ids": torch.tensor([3, 5], dtype=torch.int64),
        "labels": ("a", "b"),
    }
    _, expected_manifest = run_native.save_payload(tmp_path, "expected.pt", expected)
    mutations = []
    value = copy.deepcopy(expected)
    value["values"][0] = 9
    mutations.append(value)
    value = copy.deepcopy(expected)
    value["values"] = value["values"][None]
    mutations.append(value)
    value = copy.deepcopy(expected)
    value["values"] = value["values"].astype(np.float32)
    mutations.append(value)
    value = {"ids": expected["ids"], "values": expected["values"], "labels": expected["labels"]}
    mutations.append(value)
    for index, mutation in enumerate(mutations):
        _, actual_manifest = run_native.save_payload(tmp_path, f"mutation_{index}.pt", mutation)
        actual_manifest["artifact"] = expected_manifest["artifact"]
        with pytest.raises(AssertionError):
            compare_manifests(
                actual_manifest,
                expected_manifest,
                "payload",
                {"arrays": 0, "array_elements": 0, "records": 0, "scalars": 0},
            )


def test_large_array_and_file_hashes_are_streamed():
    source = inspect.getsource(run_native._array_value_sha256)
    assert ".tobytes(" not in source, "Large array hashing must use bounded memory views."
    source = inspect.getsource(run_native.digest)
    assert ".read_bytes(" not in source, "Large artifact hashing must stream bounded chunks."


def test_summary_counts_only_the_three_sigma_variants(monkeypatch):
    monkeypatch.setattr(run_native, "REPLICATE_SHAPE", (1, 1, 1, 1, 1))
    fitted = {"models": {}, "train_ids": torch.tensor([], dtype=torch.int64)}
    prepared = {
        "explain_ids": torch.tensor([], dtype=torch.int64),
        "audit_used_for_selection": False,
        "primary": {},
    }
    evaluated = {
        "phase": "evaluated_after_audit",
        "audit_ids": torch.tensor([], dtype=torch.int64),
        "sampling_replicates": {"samples": torch.zeros((1, 1, 1, 1, 1), dtype=torch.float64)},
        "background_variants": {},
        "noise_variants": {
            "standard_normal": torch.zeros(1),
            "source_ids": torch.zeros(1, dtype=torch.int64),
            "sigma_0": {},
            "sigma_0_02": {},
            "sigma_0_10": {},
        },
        "audit_primary": {},
    }
    summary = run_native.summarize(fitted, prepared, evaluated, {}, {})
    assert summary["noise_variants"] == 3, (
        "Noise schedule metadata must not be counted as additional sigma conditions."
    )


def test_cpu_only_pipeline_and_complete_scientific_source_snapshot():
    for name in ("run_native.py", "execute_learner.py", "reproduce_native.py", "solutions.py"):
        path = HERE / name
        if not path.is_file():
            continue
        source = path.read_text()
        assert "torch.cuda" not in source, f"{name} must remain a CPU-only execution path."
        assert ".cuda(" not in source, f"{name} cannot move study state onto CUDA."
    listed = {path.name for path in run_native.source_paths()}
    required = {
        "gradientshap_data.py",
        "gradientshap_methods.py",
        "gradientshap_methods_tests.py",
        "gradientshap_study.py",
        "gradientshap_study_tests.py",
        "gradientshap_reference.py",
        "gradientshap_reference_tests.py",
        "gradientshap_diagnostics.py",
        "gradientshap_plots.py",
        "deepshap_methods.py",
        "protocol.md",
        "protocol_freeze.json",
        "data_manifest.json",
        "run_native.py",
    }
    assert required <= listed, "Every scientific dependency must be snapshotted before fitting."


def test_archive_inventory_accepts_direct_learner_without_runner_record(tmp_path):
    publish = _module("publish_study")
    learner = tmp_path / "learner"
    native = learner / "figures" / "native"
    native.mkdir(parents=True)
    for path in (
        learner / "execution.json",
        learner / "completed_learner.ipynb",
        native / "before_audit.json",
        native / "summary.json",
        native / "artifact_manifest.json",
    ):
        path.write_text("{}\n")

    members = publish._archive_members(learner, native)
    assert "native/run.json" not in {name for _, name in members}, (
        "Direct own-answer execution has no standalone runner record."
    )
    assert all(path.is_file() for path, _ in members), "Every archive member must exist."

    (native / "run.json").write_text("{}\n")
    members = publish._archive_members(learner, native)
    assert "native/run.json" in {name for _, name in members}, (
        "Standalone runner state should be archived when present."
    )


def test_runner_attaches_state_reproduction_and_fresh_run_uses_a_child_process():
    runner_source = (HERE / "run_native.py").read_text()
    runner_tree = ast.parse(runner_source)
    run_function = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    run_text = ast.get_source_segment(runner_source, run_function) or ""
    assert 'fitted["reproduction"] = reproduction' in run_text, (
        "fitted.pt must retain the 21-state comparison and known-16.14 provenance."
    )
    reproduce_source = (HERE / "reproduce_native.py").read_text()
    reproduce_tree = ast.parse(reproduce_source)
    reproduce_function = next(
        node
        for node in reproduce_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "reproduce"
    )
    calls = _calls(reproduce_function)
    assert "run" in calls, "Fresh reproduction must launch the native runner as a child process."
    assert "subprocess.run" in (ast.get_source_segment(reproduce_source, reproduce_function) or ""), (
        "Fresh reproduction must isolate the complete refit in another CPU process."
    )
