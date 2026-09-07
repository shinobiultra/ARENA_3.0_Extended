"""Data provenance, student visibility and actual train/evaluation boundaries."""

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from . import datamodel_methods as methods
from . import datamodel_study as study
from .build_notebooks import complete_learner, lesson_cells
from .datamodel_data import load_data, data_manifest
from ..part9_influence_and_tracin.reproduce_native import compare

HERE = Path(__file__).resolve().parent


def test_pinned_data_and_freeze():
    data = load_data()
    manifest = data_manifest(data)
    assert manifest == json.loads((HERE / "data_manifest.json").read_text())
    assert manifest["splits"]["train"]["class_counts"] == [64, 64]
    assert manifest["splits"]["query"]["class_counts"] == [32, 32]
    assert manifest["exact_duplicate_image_pairs"] == []
    assert set(data["train"]["ids"]).isdisjoint(data["query"]["ids"])
    assert (data["train"]["x"][:, -1] == 1).all()
    assert (data["query"]["x"][:, -1] == 1).all()
    freeze = json.loads((HERE / "protocol_freeze.json").read_text())
    for name, expected in freeze["inputs"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected


def test_mask_splits_have_the_prescribed_distribution_and_no_reused_models():
    designs = study.make_designs(methods)
    for fraction in study.FRACTIONS:
        for split, count in (("train", 1024), ("validation", 128), ("test", 256)):
            masks = designs[fraction][split]
            assert masks.shape == (count, 128)
            assert (masks.sum(1) == int(128 * fraction)).all()
        joined = torch.cat(list(designs[fraction].values()))
        assert len(joined.unique(dim=0)) == len(joined)


def normalized(node):
    result = copy.deepcopy(node)
    if isinstance(result, ast.FunctionDef) and isinstance(result.body[0], ast.Expr):
        value = result.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            value.value = inspect.cleandoc(value.value)
    return ast.dump(result, include_attributes=False)


def test_every_method_and_study_function_is_visible_and_uses_own_answers():
    cells = complete_learner({"cells": lesson_cells()})["cells"]
    definitions = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] = node
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").endswith(("datamodel_methods", "datamodel_study"))
    for filename in ("datamodel_methods.py", "datamodel_study.py"):
        for node in ast.parse((HERE / filename).read_text()).body:
            if isinstance(node, ast.FunctionDef):
                assert node.name in definitions, f"Hidden taught function: {node.name}"
                assert normalized(definitions[node.name]) == normalized(node), node.name
    # A missing local solution must not be silently completed from a later exercise.
    broken = {"cells": lesson_cells()}
    first = next(
        i
        for i, c in enumerate(broken["cells"])
        if c["cell_type"] == "code" and "raise NotImplementedError" in "".join(c["source"])
    )
    broken["cells"][first + 1]["source"] = ["No solution supplied.\n"]
    with pytest.raises(ValueError, match="following exercise answer"):
        complete_learner(broken)


class UnreadTestDesign(dict):
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("Preparation accessed held-out subset inputs.")
        return super().__getitem__(key)


def test_preparation_does_not_access_test_subsets(monkeypatch, tmp_path):
    data = load_data()
    designs = {
        fraction: UnreadTestDesign(
            train=torch.ones(2, 128, dtype=torch.bool),
            validation=torch.ones(3, 128, dtype=torch.bool),
            test="UNREAD",
        )
        for fraction in study.FRACTIONS
    }
    calls = []

    def fake_refit(train, masks):
        calls.append(len(masks))
        return {"theta": torch.zeros(len(masks), 65, dtype=torch.float64)}

    def fake_response(x, y, xv, yv, method_module):
        assert len(x) == len(y) and len(xv) == len(yv)
        return {"lasso": {"selected_weights": torch.zeros(128, 64, dtype=torch.float64)}}

    monkeypatch.setattr(study, "refit_masks", fake_refit)
    monkeypatch.setattr(study, "fit_response_models", fake_response)
    prepared = study.prepare_study(
        data["train"], data["query"], designs, methods, tmp_path / "study"
    )
    assert calls == [2, 3, 2, 3, 2, 3]
    assert prepared["test_subset_responses_seen"] is False
    assert prepared["counterfactual_responses_seen"] is False
    assert len(prepared["deletions"]["masks"]) == 6528
    deletion = prepared["deletions"]
    assert torch.equal(128 - deletion["masks"].sum(1), deletion["budget"])
    counts = deletion["removed_class_counts"].reshape(-1, 17, 2)
    assert torch.equal(counts, counts[:, :1].expand_as(counts))


def test_evaluation_rejects_unselected_state_before_any_refit(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("A refit ran before checking the evaluation boundary.")

    monkeypatch.setattr(study, "refit_masks", forbidden)
    with pytest.raises(ValueError, match="Complete all datamodel"):
        study.evaluate_study({"phase": "STILL_SELECTING"}, {}, {}, methods, tmp_path)


@pytest.mark.parametrize("mutation", ["value", "shape", "dtype", "missing", "id_order"])
def test_reproduction_comparison_rejects_mismatches(mutation):
    expected = {
        "margin": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        "ids": ["image:1", "image:2"],
    }
    actual = copy.deepcopy(expected)
    if mutation == "value":
        actual["margin"][0, 0] += 0.1
    elif mutation == "shape":
        actual["margin"] = actual["margin"].T
    elif mutation == "dtype":
        actual["margin"] = actual["margin"].float()
    elif mutation == "missing":
        actual.pop("margin")
    else:
        actual["ids"].reverse()
    counts = dict(tensors=0, tensor_elements=0, scalars=0, max_absolute_error=0.0)
    with pytest.raises((ValueError, AssertionError)):
        compare(actual, expected, "fixture", counts)
