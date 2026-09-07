"""Keep the own-answer and separate reference paths aligned and auditable."""
import ast
import copy
import inspect
from pathlib import Path

import pytest
import torch

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.build_notebooks import lesson_cells, complete_learner
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.reproduce_native import compare


def normalized(node):
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(child, clean=False) is not None:
            child.body[0].value.value = inspect.cleandoc(child.body[0].value.value)
    return ast.dump(node)


def test_own_answers_expose_every_native_core_operation():
    completed = complete_learner({"cells": lesson_cells()})
    visible = {}
    for cell in completed["cells"]:
        if cell["cell_type"] == "code":
            visible.update({node.name: node for node in ast.parse("".join(cell["source"])).body if isinstance(node, ast.FunctionDef)})
    for filename in ("attribution_methods.py", "attribution_study.py", "attribution_play.py"):
        for node in ast.parse((Path(__file__).parent / filename).read_text()).body:
            if isinstance(node, ast.FunctionDef):
                assert node.name in visible, f"The learner cannot see {node.name}."
                assert normalized(node) == normalized(visible[node.name]), f"Own answer differs from reference: {node.name}"


def test_recursive_comparison_keeps_errors_shapes_and_missing_cases():
    expected = {"scores": torch.tensor([1., 2.], dtype=torch.float64), "ids": ["row:1", "row:2"]}
    for actual in [
        {**expected, "scores": torch.tensor([1., 3.], dtype=torch.float64)},
        {**expected, "scores": torch.ones(1, 2, dtype=torch.float64)},
        {**expected, "scores": expected["scores"].float()},
        {**expected, "ids": ["row:2", "row:1"]},
        {"scores": expected["scores"]},
    ]:
        counts = dict(tensors=0, tensor_elements=0, scalars=0, max_absolute_error=0.)
        with pytest.raises((ValueError, AssertionError)):
            compare(actual, expected, "test", counts)
