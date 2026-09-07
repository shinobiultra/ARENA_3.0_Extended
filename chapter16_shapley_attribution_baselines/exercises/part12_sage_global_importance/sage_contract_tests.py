"""Meaningful boundaries: visible answers, pinned groups, selection and complete comparisons."""

from pathlib import Path
import ast
import copy
import hashlib
import json
import numpy as np
import pytest
import torch as t
from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance import (
    sage_data,
    sage_methods,
    sage_study,
    build_notebooks,
)

HERE = Path(__file__).resolve().parent


def test_frozen_data_and_recipe_separation():
    freeze = json.loads((HERE / "protocol_freeze.json").read_text())
    for name, checksum in freeze["sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == checksum
    data = sage_data.source_data()
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    groups = []
    all_ids = []
    for name, count in [("train", 758), ("explain", 153), ("audit", 119)]:
        ids = manifest["splits"][name]
        assert len(ids) == count
        groups.append({tuple(row) for row in data[ids, :7]})
        all_ids.extend(ids)
    assert sorted(all_ids) == list(range(1030))
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    assert len(set.union(*groups)) == 427


def test_core_and_study_are_visible_in_own_answers():
    notebook = {"cells": build_notebooks.lesson_cells()}
    complete = build_notebooks.complete_learner(notebook)
    visible = {
        node.name: ast.dump(node, include_attributes=False)
        for cell in complete["cells"]
        if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ["sage_methods.py", "sage_study.py"]:
        for node in ast.parse((HERE / name).read_text()).body:
            if isinstance(node, ast.FunctionDef):
                assert visible[node.name] == ast.dump(node, include_attributes=False), node.name
    for cell in complete["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            assert "import sage_study" not in source
            assert "import sage_methods as" not in source


def test_missing_own_answer_is_rejected():
    notebook = {"cells": copy.deepcopy(build_notebooks.lesson_cells())}
    index = next(
        i
        for i, c in enumerate(notebook["cells"])
        if c["cell_type"] == "code" and "raise NotImplementedError" in "".join(c["source"])
    )
    notebook["cells"][index + 1]["source"] = ["Missing answer."]
    with pytest.raises(ValueError, match="following exercise answer"):
        build_notebooks.complete_learner(notebook)


def test_unfinished_ranking_rejects_audit():
    with pytest.raises(ValueError, match="Freeze explanation rankings"):
        sage_study.evaluate_study({"phase": "training"}, {"y": np.zeros(2)}, sage_methods)


def test_global_identity_and_changed_targets():
    predictions = t.tensor([[2.0, 1.0, 3.0, 4.0], [2.0, 3.0, 1.0, 0.0]], dtype=t.float64)
    y = t.tensor([4.0, 0.0], dtype=t.float64)
    local = sage_methods.loss_game(predictions, y)
    t.testing.assert_close(
        sage_methods.exact_shapley(local).mean(0), sage_methods.exact_shapley(local.mean(0))
    )
    changed = sage_methods.exact_shapley(sage_methods.loss_game(predictions, y.flip(0)).mean(0))
    assert not t.allclose(changed, sage_methods.exact_shapley(local.mean(0)))
    # Predictive sensitivity is unchanged because no predictions changed.
    t.testing.assert_close(
        sage_methods.exact_shapley(predictions), sage_methods.exact_shapley(predictions.clone())
    )


@pytest.mark.parametrize("mutation", ["value", "shape", "dtype", "missing", "scalar_type", "order"])
def test_complete_reproduction_rejects_mutations(mutation):
    from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance.reproduce_native import (
        compare,
    )

    expected = {
        "values": t.tensor([1.0, 2.0], dtype=t.float64),
        "ids": t.tensor([3, 5]),
        "count": 2,
    }
    actual = copy.deepcopy(expected)
    if mutation == "value":
        actual["values"][0] = 10
    elif mutation == "shape":
        actual["values"] = actual["values"][None]
    elif mutation == "dtype":
        actual["values"] = actual["values"].float()
    elif mutation == "missing":
        actual.pop("ids")
    elif mutation == "scalar_type":
        actual["count"] = 2.0
    elif mutation == "order":
        actual["ids"] = actual["ids"].flip(0)
    counts = dict(tensors=0, tensor_elements=0, scalars=0, max_absolute_error=0.0)
    with pytest.raises((ValueError, AssertionError)):
        compare(actual, expected, "native", counts)


def test_independent_permutation_reference_keeps_first_feature():
    from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance.sage_reference import (
        permutation_reference,
    )

    game = np.array([0.0, 3.0, -2.0, 8.0, 1.0, 7.0, -4.0, 12.0])
    original = game.copy()
    phi, count = permutation_reference(game)
    assert count == 6
    np.testing.assert_array_equal(game, original)
    np.testing.assert_allclose(
        phi, sage_methods.exact_shapley(t.from_numpy(game)).numpy(), atol=1e-12
    )
    np.testing.assert_allclose(phi.sum(), 12.0, atol=1e-12)
