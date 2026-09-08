"""Behavioral checks for the learner, real training boundary and full reproduction."""

import ast
import copy
import inspect
from types import SimpleNamespace

import pytest
import torch

from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography import (
    cartography_methods as methods,
)
from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography import (
    cartography_study as study,
)
from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography.build_notebooks import (
    complete_learner,
    lesson_cells,
)
from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography.run_native import (
    validate_freeze,
)
from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography.reproduce_native import (
    compare,
)


def normalized(node):
    node = copy.deepcopy(node)
    if isinstance(node, ast.FunctionDef) and ast.get_docstring(node):
        node.body = node.body[1:]
    return ast.dump(node, include_attributes=False)


def test_frozen_source_and_split():
    validate_freeze()


def test_every_core_and_study_function_visible_from_own_answers():
    draft = dict(cells=lesson_cells())
    solved = complete_learner(draft)
    source = "\n\n".join(
        "".join(cell["source"]) for cell in solved["cells"] if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    visible = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for module in (methods, study):
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ != module.__name__:
                continue
            original = ast.parse(inspect.getsource(function)).body[0]
            assert name in visible, f"Learner cannot inspect and edit {name}."
            assert normalized(visible[name]) == normalized(
                original
            ), f"Notebook and native implementation diverged: {name}."
    assert (
        "cartography_methods import" not in source
    ), "The learner must implement the taught operations."
    assert source.count("tests.test_") == 10, "Keep every immediate exercise test in the learner."
    broken = copy.deepcopy(draft)
    stub = next(
        i
        for i, cell in enumerate(broken["cells"])
        if cell["cell_type"] == "code" and "raise NotImplementedError" in "".join(cell["source"])
    )
    broken["cells"][stub + 1]["source"] = ["The answer was removed."]
    with pytest.raises(ValueError, match="following exercise answer"):
        complete_learner(broken)


def test_selection_and_all_refits_need_training_data_only(tmp_path):
    train = dict(
        x=torch.arange(3000).float()[:, None],
        y=torch.arange(3000) % 10,
        ids=[f"train:{i}" for i in range(3000)],
    )
    mapping = dict(
        statistics=dict(
            clean=dict(
                end=dict(
                    confidence=torch.linspace(0, 1, 3000).expand(5, -1),
                    variability=torch.linspace(1, 0, 3000).expand(5, -1),
                )
            )
        )
    )
    calls = []

    def fake_fit(x, labels, seeds, epochs=30, record=False):
        assert (
            not record
        ), "Selection refits need final parameters and losses, not large map traces."
        calls.append(dict(n=x.shape[1], seeds=list(seeds), epochs=epochs))
        return dict(
            seeds=list(seeds),
            loss=torch.zeros(epochs, 5),
            steps=((x.shape[1] + 127) // 128) * epochs,
        )

    implementation = SimpleNamespace(
        select_indices=methods.select_indices,
        matched_random_indices=methods.matched_random_indices,
        train_ensemble=fake_fit,
    )
    result = study.retrain_selections(train, mapping, implementation, tmp_path)
    assert (
        len(calls) == 24 and result["refits"] == 120
    ), "Preserve all three sizes, regions, controls, seeds and budget checks."
    assert all(
        call["seeds"] == list(range(161200, 161205)) for call in calls
    ), "Every condition uses the prescribed five fresh seeds."
    assert (
        sum(call["epochs"] == 90 for call in calls) == 2
    ), "Both ambiguous and matched-random equal-update refits are required."
    for name, indices in result["selections"].items():
        if name.endswith("_matched"):
            selected = result["selections"][name.removesuffix("_matched")]
            for a, b in zip(indices, selected):
                torch.testing.assert_close(
                    torch.bincount(train["y"][a], minlength=10),
                    torch.bincount(train["y"][b], minlength=10),
                )
    assert not result[
        "heldout_predictions_computed"
    ], "All selections and training finish before held-out predictions."
    assert (
        tmp_path / "selections_before_refits.pt"
    ).exists(), "Persist the complete selection boundary."


def test_evaluation_rejects_incomplete_training(tmp_path):
    prepared = dict(phase="INCOMPLETE", heldout_predictions_computed=False)
    with pytest.raises(ValueError, match="before evaluating"):
        study.evaluate_study({}, prepared, {}, SimpleNamespace(), tmp_path)


@pytest.mark.parametrize("mutation", ["value", "shape", "dtype", "missing", "ids", "scalar_type"])
def test_reproduction_checks_entire_payload(mutation):
    expected = dict(values=torch.arange(12).float().reshape(3, 4), ids=["a", "b", "c"], flag=True)
    actual = copy.deepcopy(expected)
    if mutation == "value":
        actual["values"][2, 3] += 0.01
    if mutation == "shape":
        actual["values"] = actual["values"].flatten()
    if mutation == "dtype":
        actual["values"] = actual["values"].double()
    if mutation == "missing":
        del actual["flag"]
    if mutation == "ids":
        actual["ids"] = ["b", "a", "c"]
    if mutation == "scalar_type":
        actual["flag"] = 1
    counts = dict(tensors=0, tensor_elements=0, scalars=0, max_absolute_error=0.0)
    with pytest.raises((ValueError, AssertionError)):
        compare(actual, expected, "fixture", counts)
