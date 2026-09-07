"""Contracts for frozen data, visible own answers, fresh refits, and evidence gates."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chapter16_shapley_attribution_baselines.exercises.part13_treeshap import (
    build_notebooks,
    run_native,
    treeshap_data,
    treeshap_methods,
    treeshap_study,
)

HERE = Path(__file__).resolve().parent


def test_frozen_data_and_recipe_separation():
    freeze = run_native.validate_freeze()
    assert set(freeze["sha256"]) == {"protocol.md", "data_manifest.json", "treeshap_data.py"}
    data = treeshap_data.source_data()
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    groups = []
    all_ids = []
    for name, count in (("train", 758), ("explain", 153), ("audit", 119)):
        ids = manifest["splits"][name]
        assert len(ids) == count
        groups.append({tuple(row) for row in data[ids, :7]})
        all_ids.extend(ids)
    assert sorted(all_ids) == list(range(1030))
    assert all(not groups[left] & groups[right] for left in range(3) for right in range(left))
    assert len(set.union(*groups)) == 427
    assert manifest["cross_split_recipe_overlap"] == 0


def test_full_fresh_fit_contract_has_18_models_and_486_trees():
    fitted = treeshap_study.fit_models(treeshap_data.load_split("train"))
    assert list(fitted["models"]) == [
        f"{family}_{target}_{seed}"
        for family in ("tree", "forest", "boost")
        for seed in range(3)
        for target in ("clean", "shuffled")
    ]
    assert sum(len(item["ensemble"]["trees"]) for item in fitted["models"].values()) == 486
    assert sum("_clean_" in key for key in fitted["models"]) == 9
    assert sum("_shuffled_" in key for key in fitted["models"]) == 9
    for item in fitted["models"].values():
        assert item["train_prediction"].shape == (758,)
        assert item["training_targets"].shape == (758,)


def _visible_functions(notebook: dict) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.FunctionDef)
    }


def test_core_and_complete_study_are_visible_in_own_answers():
    complete = build_notebooks.complete_learner({"cells": build_notebooks.lesson_cells()})
    visible = _visible_functions(complete)
    for name in ("treeshap_methods.py", "treeshap_study.py"):
        for node in ast.parse((HERE / name).read_text()).body:
            if isinstance(node, ast.FunctionDef):
                assert visible.get(node.name) == ast.dump(node, include_attributes=False), node.name
    for cell in complete["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(name.endswith(".treeshap_methods") for name in imported)
        assert not any(name.endswith(".treeshap_study") for name in imported)


def test_missing_own_answer_is_rejected():
    notebook = {"cells": copy.deepcopy(build_notebooks.lesson_cells())}
    index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code" and "raise NotImplementedError" in "".join(cell["source"])
    )
    notebook["cells"][index + 1]["source"] = ["The required answer was removed."]
    with pytest.raises(ValueError):
        build_notebooks.complete_learner(notebook)


def _top_level_calls(source: str) -> list[tuple[str, list[object]]]:
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        arguments = [arg.value if isinstance(arg, ast.Constant) else None for arg in node.args]
        calls.append((node.func.id, arguments))
    return calls


def test_own_answer_seals_prepared_payload_before_literal_audit_load():
    complete = build_notebooks.complete_learner({"cells": build_notebooks.lesson_cells()})
    calls = []
    for cell_index, cell in enumerate(complete["cells"]):
        if cell["cell_type"] != "code":
            continue
        calls.extend((cell_index, name, arguments) for name, arguments in _top_level_calls("".join(cell["source"])))
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
    assert prepared_save <= boundary < audit_load <= evaluation


def test_prepared_boundary_strips_sklearn_and_detects_changes(tmp_path):
    fitted = {
        "models": {"tree_clean_0": {"sklearn": object(), "ensemble": {"trees": []}}},
        "train_ids": np.array([1, 2], dtype=np.int64),
    }
    prepared = {
        "audit_used_for_selection": False,
        "explain_ids": np.array([3], dtype=np.int64),
        "background_ids": np.array([1], dtype=np.int64),
    }
    run_native.persist_prepared(tmp_path, fitted, prepared)
    saved = run_native.load_payload(tmp_path / "fitted.pt")
    assert "sklearn" not in saved["models"]["tree_clean_0"]
    boundary = json.loads((tmp_path / "before_audit.json").read_text())
    assert boundary["audit_loaded"] is False
    run_native.save_payload(tmp_path, "prepared.pt", {**prepared, "changed": True})
    with pytest.raises(ValueError, match="changed"):
        run_native.persist_completed(
            tmp_path,
            fitted,
            prepared,
            {},
            {"mechanism": {}, "timing": {}},
            {},
            {},
        )


@pytest.mark.parametrize("mutation", ["value", "shape", "dtype", "missing", "scalar_type", "order"])
def test_complete_reproduction_rejects_mutations(mutation):
    from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.reproduce_native import (
        compare,
    )

    expected = {
        "values": np.array([1.0, 2.0], dtype=np.float64),
        "ids": np.array([3, 5], dtype=np.int64),
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
    counts = {"arrays": 0, "array_elements": 0, "scalars": 0, "max_absolute_error": 0.0}
    with pytest.raises((ValueError, AssertionError)):
        compare(actual, expected, "native", counts)


def test_publisher_uses_ast_literal_figure_calls():
    from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.publish_study import (
        _show_figure_stems,
    )

    source = "mentioned = \"show_figure(fig, 'fake')\"\nshow_figure(fig, '01_real')\n"
    assert _show_figure_stems(source) == ["01_real"]


def test_no_gpu_execution_calls_in_native_pipeline():
    for name in ("run_native.py", "execute_learner.py", "reproduce_native.py", "solutions.py"):
        source = (HERE / name).read_text()
        assert "torch.cuda" not in source
        assert ".cuda(" not in source
