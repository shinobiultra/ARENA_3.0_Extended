"""Current capstone contracts; bounded CPU checks of the visible lesson functions.

See docs/mastery/migrations/capstone.md for the old-to-current guarantee mapping.
"""

import ast
import copy
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch as t

SECTION = (
    Path(__file__).resolve().parents[1]
    / "chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint"
)
if str(SECTION.parent) not in sys.path:
    sys.path.insert(0, str(SECTION.parent))
sprint = importlib.import_module("part1_capstone_research_sprint.solutions")
checks = importlib.import_module("part1_capstone_research_sprint.tests")

CORE_FUNCTIONS = [
    "split_examples", "make_pairs", "patch_coordinates", "patch_direction",
    "evaluate_patch", "mean_difference_direction", "rank_coordinates",
    "clopper_pearson", "claim_verdict", "paired_bootstrap",
    "evaluate_interventions", "select_on_development", "assess_methodology",
    "logit_contributions", "principal_directions",
]


@pytest.mark.parametrize("name", CORE_FUNCTIONS)
def test_student_contract(name):
    getattr(checks, f"test_{name}")(getattr(sprint, name))


def test_seeded_control_contract():
    checks.test_build_controls(sprint.build_controls, sprint.mean_difference_direction)


def test_plan_fingerprint_detects_nested_tampering():
    plan = copy.deepcopy(sprint.PLAN)
    registered = sprint.plan_fingerprint(plan)
    assert sprint.plan_fingerprint(dict(reversed(list(plan.items())))) == registered
    for field, replacement in [
        ("H1", {**plan["H1"], "threshold": 0.1}),
        ("model", {**plan["model"], "seed": 100}),
        ("candidates", list(reversed(plan["candidates"]))),
    ]:
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        assert sprint.plan_fingerprint(changed) != registered


@pytest.mark.parametrize("digits", [(3, 8), (1, 7)])
def test_real_task_split_is_exhaustive_disjoint_and_has_independent_pairs(digits):
    x, y, rows = sprint.utils.load_digit_pair(digits)
    splits = sprint.split_examples(y, **sprint.PLAN["split"])
    all_ids = np.concatenate(list(splits.values()))
    np.testing.assert_array_equal(np.sort(all_ids), np.arange(len(y)))
    assert len(np.unique(rows)) == len(rows) == len(x)
    idx0, idx1 = sprint.make_pairs(y[splits["confirm"]])
    paired = np.concatenate([idx0, idx1])
    assert len(idx0) >= sprint.PLAN["instrument_checks"]["min_confirm_pairs"]
    assert len(np.unique(paired)) == len(paired)
    assert t.all(y[splits["confirm"]][idx0] == 0)
    assert t.all(y[splits["confirm"]][idx1] == 1)
    if digits == (3, 8):
        assert len(idx0) == 44  # 44 independent pairs, rather than 88 directions


def test_zero_events_retains_population_uncertainty():
    low, high = sprint.clopper_pearson(0, 44)
    assert low == 0.0
    assert 0.08 < high < 0.081
    assert sprint.claim_verdict(low, high, sprint.PLAN["H1"]["threshold"]) == "ruled out"


def test_intervention_assembly_retains_every_control_and_draw():
    generator = t.Generator().manual_seed(19)
    h = t.randn(40, 8, generator=generator, dtype=t.float64)
    y = t.tensor([0, 1] * 20)
    direction = sprint.mean_difference_direction(h, y)
    controls = sprint.build_controls(h, y, [1, 2, 4, 8], n_random=16, seed=31)
    interventions = sprint.make_interventions(t.arange(8), direction, controls, [1, 2, 4, 8])
    assert set(interventions) == {
        "no_patch", "full_patch", "direction", "shuffled", "random_dirs",
        *[f"units_{k}" for k in [1, 2, 4, 8]],
        *[f"random_units_{k}" for k in [1, 2, 4, 8]],
    }
    recipient, donor = h[:3], h[3:6]
    t.testing.assert_close(interventions["no_patch"][0](recipient, donor), recipient)
    t.testing.assert_close(interventions["full_patch"][0](recipient, donor), donor)
    t.testing.assert_close(interventions["units_8"][0](recipient, donor), donor)
    assert len(interventions["random_dirs"]) == 16
    for k in [1, 2, 4, 8]:
        assert len(interventions[f"random_units_{k}"]) == 16
    t.testing.assert_close(
        interventions["shuffled"][0](recipient, donor),
        sprint.patch_direction(recipient, donor, controls["shuffled_direction"]),
    )


def test_training_and_pilot_keep_confirmation_out_of_fitting():
    """Check data routing without authorizing or executing a training run."""
    source = SECTION.joinpath("solutions.py").read_text()
    training_calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "train_model"
    ]
    assert len(training_calls) == 2  # worked example and exploratory pilot
    for call in training_calls:
        for argument in call.args[:2]:
            assert isinstance(argument, ast.Subscript)
            assert isinstance(argument.slice, ast.Subscript)
            assert argument.slice.slice.value == "train"
    pilot = inspect.getsource(sprint.run_pilot)
    assert 'splits_["confirm"]' not in pilot
    assert 'splits_["dev"]' in pilot
    assert source.count("verify_registration(REGISTRATION, PLAN, splits, x, y, rows)") == 2
    assert '"selection" in log.stages() and "confirmation" not in log.stages()' in source


def test_source_notebook_answers_are_visible():
    def notebook(suffix):
        return json.loads(SECTION.joinpath(f"10.1_Capstone_Research_Sprint_{suffix}.ipynb").read_text())

    exercise, solution = notebook("exercises"), notebook("solutions")
    text = "\n".join("".join(cell["source"]) for cell in exercise["cells"])
    for marker in ["Pre-registration", "negative result", "shuffled-label", "Clopper", "write-up", "PLAY_"]:
        assert marker in text
    for name in [*CORE_FUNCTIONS, "train_model", "build_controls"]:
        assert f"def {name}(" in text
        assert f"test_{name}(" in text
    solution_code = "\n".join(
        "".join(cell["source"]) for cell in solution["cells"] if cell["cell_type"] == "code"
    )
    for name in [*CORE_FUNCTIONS, "train_model", "build_controls"]:
        assert f"def {name}(" in solution_code
    assert "from sprint import" not in solution_code
    assert "import sprint\n" not in solution_code


def test_training_contract():
    checks.test_train_model(sprint.train_model)


def test_registration_is_bound_to_plan_data_rows_and_splits():
    x, y, rows = sprint.utils.load_digit_pair((3, 8))
    splits = sprint.split_examples(y, **sprint.PLAN["split"])
    registration = sprint.register_plan(sprint.PLAN, splits, x, y, rows)
    sprint.verify_registration(registration, sprint.PLAN, splits, x, y, rows)
    changed_plan = copy.deepcopy(sprint.PLAN)
    changed_plan["H1"]["threshold"] = 0.1
    changed_x = x.clone()
    changed_x[0, 0] += 0.01
    changed_y = y.clone()
    changed_y[0] = 1 - changed_y[0]
    changed_rows = rows.copy()
    changed_rows[[0, 1]] = changed_rows[[1, 0]]
    changed_splits = {name: ids.copy() for name, ids in splits.items()}
    changed_splits["train"][0], changed_splits["dev"][0] = (
        changed_splits["dev"][0], changed_splits["train"][0]
    )
    for arguments in [
        (changed_plan, splits, x, y, rows),
        (sprint.PLAN, splits, changed_x, y, rows),
        (sprint.PLAN, splits, x, changed_y, rows),
        (sprint.PLAN, splits, x, y, changed_rows),
        (sprint.PLAN, changed_splits, x, y, rows),
    ]:
        with pytest.raises(ValueError, match="changed"):
            sprint.verify_registration(registration, *arguments)
    modified = copy.deepcopy(registration)
    modified["payload"]["budget"]["readout_rows"] += 1
    with pytest.raises(ValueError, match="changed"):
        sprint.verify_registration(modified, sprint.PLAN, splits, x, y, rows)
    overlap = {name: ids.copy() for name, ids in splits.items()}
    overlap["confirm"][0] = overlap["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        sprint.register_plan(sprint.PLAN, overlap, x, y, rows)


@pytest.mark.parametrize("kind,digits", [("mlp", (3, 8)), ("linear", (1, 7))])
def test_real_choice_negative_result_and_registered_budget(kind, digits):
    plan = copy.deepcopy(sprint.PLAN)
    plan["digits"] = list(digits)
    plan["model"]["kind"] = kind
    x, y, rows = sprint.utils.load_digit_pair(digits)
    splits = sprint.split_examples(y, **plan["split"])
    model, _ = sprint.train_model(x[splits["train"]], y[splits["train"]], **plan["model"])
    registration = sprint.register_plan(plan, splits, x, y, rows)
    with t.no_grad():
        h_train = model.encoder(x[splits["train"]])
    y_train = y[splits["train"]]
    readout = (model.head.weight[1] - model.head.weight[0]).detach()
    order = sprint.rank_coordinates(h_train, y_train, readout)
    direction = sprint.mean_difference_direction(h_train, y_train)
    controls = sprint.build_controls(
        h_train, y_train, plan["coordinate_counts"], plan["controls"]["n_random"], plan["controls"]["seed"]
    )
    interventions = sprint.make_interventions(order, direction, controls, plan["coordinate_counts"])
    usage = {"readout_rows": 0, "completed": []}
    actual_head_rows = 0

    def observed_head(values):
        nonlocal actual_head_rows
        actual_head_rows += len(values)
        return model.head(values)

    results = {}
    for stage in ["dev", "confirm"]:
        sprint.verify_registration(registration, plan, splits, x, y, rows)
        with t.no_grad():
            hidden = model.encoder(x[splits[stage]])
        results[stage] = sprint.evaluate_registered(
            observed_head, hidden, y[splits[stage]], interventions, registration, stage, usage
        )
        if stage == "dev":
            selected = sprint.select_on_development(
                {name: results[stage][name]["swap"].mean() for name in plan["candidates"]}, plan["candidates"]
            )
    confirm = results["confirm"]
    n_pairs = confirm["units_1"]["swap"].shape[1]
    interval = sprint.clopper_pearson(int(confirm["units_1"]["swap"].sum()), n_pairs)
    assert sprint.claim_verdict(*interval, plan["H1"]["threshold"]) == "ruled out"
    assert selected in plan["candidates"]
    assert n_pairs >= 40
    assert confirm["full_patch"]["swap"].mean() >= 0.95
    assert confirm["no_patch"]["swap"].sum() == 0
    assert usage["completed"] == ["dev", "confirm"]
    assert actual_head_rows == usage["readout_rows"] == registration["payload"]["budget"]["readout_rows"]
    assert not sprint.assess_methodology({
        "budget respected": usage["readout_rows"] <= registration["payload"]["budget"]["readout_rows"],
        "all registered interventions retained": set(confirm) == set(interventions),
        "full patch works": bool(confirm["full_patch"]["swap"].mean() >= 0.95),
        "no patch clean": bool(confirm["no_patch"]["swap"].sum() == 0),
    })
    if kind == "mlp":
        assert n_pairs == 44
        assert confirm["shuffled"]["swap"].mean() == 1.0
        assert confirm["units_1"]["swap"].mean() == 0.0
    # Both repeated confirmation and an exhausted ledger fail before any head calls.
    for stage, ledger, expected_error in [
        ("confirm", usage, "stage once"),
        ("dev", {"readout_rows": usage["readout_rows"], "completed": []}, "budget exceeded"),
    ]:
        with t.no_grad():
            stage_hidden = model.encoder(x[splits[stage]])
        before = actual_head_rows
        with pytest.raises(ValueError, match=expected_error):
            sprint.evaluate_registered(
                observed_head, stage_hidden, y[splits[stage]], interventions, registration, stage, ledger
            )
        assert actual_head_rows == before
