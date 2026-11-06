"""Tiny fixed CPU DEV fixtures for source-visible guards; never run the VLM lesson."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADDON = ROOT / "mastery_hub/practice_addons/mini_vlm_checked_recovery"


def notebook():
    path = ADDON / "exercises.ipynb"
    assert path.is_file(), "The source-only checked-recovery addon must exist"
    return json.loads(path.read_text())


@pytest.fixture
def api():
    # Only test invocation imports Torch; collection/generation remains source-only.
    import torch as t

    text = "\n".join("".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "markdown")
    blocks = re.findall(r'<details data-guard-solution="[^"]+">.*?```python\n(.*?)```', text, re.S)
    assert len(blocks) == 3
    namespace = {"t": t}
    for block in blocks:
        tree = ast.parse(block)
        assert all(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body)
        exec(compile(tree, "<visible checked-recovery solution dropdown>", "exec"), namespace)
    return t, namespace


def literal_inputs(t, dtype):
    clean = t.tensor([[3., 0.], [2., 0.]], dtype=dtype, device="cpu")
    corrupt = t.tensor([[-1., 0.], [0., 0.]], dtype=dtype, device="cpu")
    clean_ids = t.tensor([0, 0], device="cpu")
    corrupt_ids = t.tensor([1, 1], device="cpu")
    return clean, corrupt, clean_ids, corrupt_ids


def sweep_args(t, ns, dtype):
    clean, corrupt, a, b = literal_inputs(t, dtype)
    return dict(layers=(0, 1), masks={"object": None, "random": None}, pair_indices=(7, 9),
                clean_logits=clean, corrupt_logits=corrupt, clean_ids=a, corrupt_ids=b)


def test_source_exists_and_is_unexecuted():
    nb = notebook()
    assert nb["nbformat"] == 4
    assert all(c.get("execution_count") is None and c.get("outputs", []) == [] for c in nb["cells"])
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    for required in ("Forecast", "Reflect", "Try It Yourself", "saved historical Mini reference",
                     "my_recovery", "metric_callback=my_recovery", "patch_logits_callback=my_patch_logits"):
        assert required in text
    assert "from chapter12" not in text
    assert "import solutions" not in text


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_baseline_valid_dtype_and_whole_row_order(api, dtype_name):
    t, ns = api
    args = literal_inputs(t, getattr(t, dtype_name))
    clean_ld, corrupt_ld, gap = ns["validate_recovery_baselines"](*args, pair_indices=(7, 9))
    assert gap.dtype == getattr(t, dtype_name)
    assert gap.tolist() == [4., 2.]
    assert clean_ld.tolist() == [3., 2.]
    assert corrupt_ld.tolist() == [-1., 0.]


@pytest.mark.parametrize("bad", [0., -0., -1., float("nan"), float("inf"), -float("inf")])
def test_bad_baseline_stops_before_patch_or_metric(api, bad):
    t, ns = api
    kwargs = sweep_args(t, ns, t.float64)
    kwargs["clean_logits"][1, 0] = bad
    calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError("Callback reached before baseline validation")
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_sweep"](**kwargs, patch_logits_callback=forbidden, metric_callback=forbidden)
    assert calls == []
    assert caught.value.stage == "baseline"
    assert caught.value.pair_indices == (9,)
    assert caught.value.cohort_pair_indices == (7, 9)
    assert caught.value.completed_conditions == ()
    assert caught.value.unattempted_conditions == ((0, "object"), (0, "random"), (1, "object"), (1, "random"))


def test_all_bad_pairs_and_finite_subtraction_overflow(api):
    t, ns = api
    clean, corrupt, a, b = literal_inputs(t, t.float64)
    clean[:, 0] = t.tensor([1., 1e308], dtype=t.float64)
    corrupt[:, 0] = t.tensor([1., -1e308], dtype=t.float64)
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["validate_recovery_baselines"](clean, corrupt, a, b, pair_indices=(7, 9))
    assert caught.value.pair_indices == (7, 9)
    assert {r["reason"] for r in caught.value.invalid_pairs} == {"zero_gap", "nonfinite_gap"}


@pytest.mark.parametrize("dtype_name,tiny", [("float32", 1e-30), ("float64", 1e-300)])
def test_tiny_positive_and_outside_unit_interval_are_unchanged(api, dtype_name, tiny):
    t, ns = api
    dtype = getattr(t, dtype_name)
    clean, corrupt, a, b = literal_inputs(t, dtype)
    clean[1, 0] = tiny
    expected = t.tensor([-0.25, 1.25], dtype=dtype)
    calls = []
    def own_metric(*received):
        calls.append(received)
        return expected
    actual = ns["checked_learner_recovery"](
        clean, clean, corrupt, a, b, pair_indices=(7, 9), metric_callback=own_metric, condition=(0, "object")
    )
    assert len(calls) == 1 and calls[0][0] is clean
    assert actual.dtype == dtype
    assert actual.tolist() == [-0.25, 1.25]
    expected.fill_(99.)
    assert actual.tolist() == [-0.25, 1.25]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_late_metric_failure_precedes_every_mean(api, monkeypatch, bad):
    t, ns = api
    kwargs = sweep_args(t, ns, t.float32)
    patches, metrics, means = [], [], []
    def patch(layer, name):
        patches.append((layer, name))
        return kwargs["clean_logits"]
    def metric(*args):
        metrics.append(1)
        return t.tensor([0., bad if len(metrics) == 3 else 1.])
    def forbidden_mean(*args, **kw):
        means.append(1)
        raise AssertionError("A mean ran before all conditions passed")
    monkeypatch.setattr(t.Tensor, "mean", forbidden_mean)
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_sweep"](**kwargs, patch_logits_callback=patch, metric_callback=metric)
    error = caught.value
    assert error.stage == "metric" and error.condition == (1, "object")
    assert error.pair_indices == (9,) and error.cohort_pair_indices == (7, 9)
    assert error.completed_conditions == ((0, "object"), (0, "random"))
    assert error.unattempted_conditions == ((1, "random"),)
    assert len(patches) == len(metrics) == 3 and means == []
    assert not hasattr(error, "curves")


def test_full_success_uses_own_callback_order_and_clones_reused_buffers(api):
    t, ns = api
    kwargs = sweep_args(t, ns, t.float64)
    buffer = t.tensor([0., 0.], dtype=t.float64)
    calls = []
    def patch(layer, name):
        calls.append((layer, name))
        return kwargs["clean_logits"]
    def metric(*args):
        buffer.fill_(len(calls))
        return buffer
    result = ns["checked_learner_sweep"](**kwargs, patch_logits_callback=patch, metric_callback=metric)
    assert result["status"] == "complete_checked"
    assert result["condition_order"] == tuple(calls) == ((0, "object"), (0, "random"), (1, "object"), (1, "random"))
    assert result["curves"] == {"object": [1., 3.], "random": [2., 4.]}
    assert result["pair_indices"] == (7, 9)
    assert result["gap"].tolist() == [4., 2.]
    buffer.fill_(99.)
    kwargs["clean_logits"].fill_(99.)
    assert [r["recovery"].tolist() for r in result["rows"]] == [[1., 1.], [2., 2.], [3., 3.], [4., 4.]]
    assert result["clean_ld"].tolist() == [3., 2.]


def test_nonfinite_mean_publishes_no_curves(api):
    t, ns = api
    kwargs = sweep_args(t, ns, t.float32)
    values = t.tensor([3e38, 3e38], dtype=t.float32)
    assert t.isfinite(values).all().item()
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_sweep"](**kwargs, patch_logits_callback=lambda *a: kwargs["clean_logits"],
                                    metric_callback=lambda *a: values)
    assert caught.value.stage == "aggregation"
    assert caught.value.pair_indices == (7, 9)
    assert caught.value.unattempted_conditions == ()
    assert not hasattr(caught.value, "curves")


@pytest.mark.parametrize("change", ["empty", "duplicate_pair", "bool_pair", "bad_id", "short_id",
                                     "integer_logits", "mismatched_dtype", "empty_layers", "duplicate_layers", "empty_masks"])
def test_structural_errors_precede_callbacks(api, change):
    t, ns = api
    kwargs = sweep_args(t, ns, t.float32)
    if change == "empty": kwargs["pair_indices"] = ()
    if change == "duplicate_pair": kwargs["pair_indices"] = (7, 7)
    if change == "bool_pair": kwargs["pair_indices"] = (True, 9)
    if change == "bad_id": kwargs["clean_ids"][1] = 2
    if change == "short_id": kwargs["clean_ids"] = kwargs["clean_ids"][:1]
    if change == "integer_logits": kwargs["clean_logits"] = kwargs["clean_logits"].long()
    if change == "mismatched_dtype": kwargs["corrupt_logits"] = kwargs["corrupt_logits"].double()
    if change == "empty_layers": kwargs["layers"] = ()
    if change == "duplicate_layers": kwargs["layers"] = (0, 0)
    if change == "empty_masks": kwargs["masks"] = {}
    calls = []
    def forbidden(*args):
        calls.append(1)
        raise AssertionError("Malformed input reached callback")
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_sweep"](**kwargs, patch_logits_callback=forbidden, metric_callback=forbidden)
    assert caught.value.stage == "structure" and calls == []


@pytest.mark.parametrize("output_kind", ["shape", "integer", "boolean", "not_tensor"])
def test_bad_metric_output_is_rejected(api, output_kind):
    t, ns = api
    clean, corrupt, a, b = literal_inputs(t, t.float32)
    outputs = {"shape": t.tensor([[0., 1.]]), "integer": t.tensor([0, 1]),
               "boolean": t.tensor([False, True], dtype=t.bool), "not_tensor": [0., 1.]}
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_recovery"](clean, clean, corrupt, a, b, pair_indices=(7, 9),
                                       metric_callback=lambda *a: outputs[output_kind], condition=(0, "object"))
    assert caught.value.stage == "structure"


def test_callback_error_preserves_primary_exception(api):
    t, ns = api
    kwargs = sweep_args(t, ns, t.float32)
    primary = RuntimeError("Distinct DEV callback exception")
    def fail(*args): raise primary
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_sweep"](**kwargs, patch_logits_callback=fail, metric_callback=fail)
    assert caught.value.stage == "callback_exception"
    assert caught.value.__cause__ is primary
    assert caught.value.pair_indices == (7, 9)


def test_manifest_binds_addon_bytes_and_historical_boundary():
    manifest = json.loads((ADDON / "source_manifest.json").read_text())
    for name, expected in manifest["addon_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert len(manifest["unchanged_canonical_inputs"]) == 47
    assert manifest["scientific_execution"] is False
    assert manifest["historical_canonical_files_modified"] == []


@pytest.mark.parametrize("test_name,argument", [
    ("test_my_baseline_guard", "validate_recovery_baselines"),
    ("test_my_metric_guard", "checked_learner_recovery"),
    ("test_my_sweep_guard", "checked_learner_sweep"),
])
def test_visible_immediate_controls_against_visible_answers(api, test_name, argument):
    _, namespace = api
    # Compile only the named control definition, never execute a notebook cell.
    matches = [node for cell in notebook()["cells"] if cell["cell_type"] == "code"
               for node in ast.parse("".join(cell["source"])).body
               if isinstance(node, ast.FunctionDef) and node.name == test_name]
    assert len(matches) == 1
    exec(compile(ast.Module(body=matches, type_ignores=[]), "<visible tiny DEV control>", "exec"), namespace)
    namespace[test_name](namespace[argument])


def test_metric_callback_exception_and_patch_shape_failure(api):
    t, ns = api
    kwargs = sweep_args(t, ns, t.float32)
    primary = ArithmeticError("Distinct DEV metric exception")
    def metric(*args): raise primary
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_sweep"](**kwargs, patch_logits_callback=lambda *a: kwargs["clean_logits"],
                                    metric_callback=metric)
    assert caught.value.__cause__ is primary and caught.value.stage == "callback_exception"
    assert caught.value.condition == (0, "object")
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_sweep"](**kwargs, patch_logits_callback=lambda *a: kwargs["clean_logits"][:1],
                                    metric_callback=metric)
    assert caught.value.stage == "structure" and caught.value.__cause__ is None


def test_same_dtype_tiny_gap_overflow_of_own_callback_result_fails(api):
    t, ns = api
    clean, corrupt, a, b = literal_inputs(t, t.float64)
    clean[1, 0] = 1e-300
    def own_dev_metric(*args):
        # Fixed literal callback arithmetic only, not the taught VLM method.
        return t.tensor([0., 1e300], dtype=t.float64) / t.tensor([4., 1e-300], dtype=t.float64)
    with pytest.raises(ns["RecoveryCohortFailure"]) as caught:
        ns["checked_learner_recovery"](clean, clean, corrupt, a, b, pair_indices=(7, 9),
                                       metric_callback=own_dev_metric, condition=(0, "object"))
    assert caught.value.stage == "metric" and caught.value.pair_indices == (9,)


def test_future_failed_rerun_cannot_reuse_old_success_by_source_order():
    cells = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]}
    run = ast.parse(cells["future-run"])
    first = run.body[0]
    assert isinstance(first, ast.Assign)
    assert ast.unparse(first) == "prospective_checked_recovery = None"
    display = ast.parse(cells["future-display"])
    assert isinstance(display.body[0], ast.If)
    assert ast.unparse(display.body[0].test) == "prospective_checked_recovery is None"
    assert isinstance(display.body[0].body[0], ast.Raise)
    assert "patching_sweep(" not in cells["future-run"]


def test_own_floating_callback_dtype_is_preserved_with_float32_baselines(api):
    t, ns = api
    clean = t.tensor([[3., 0.], [2., 0.], [1., 0.]], dtype=t.float32, device="cpu")
    corrupt = t.tensor([[-1., 0.], [0., 0.], [0., 0.]], dtype=t.float32, device="cpu")
    a = t.tensor([0, 0, 0], device="cpu")
    b = t.tensor([1, 1, 1], device="cpu")
    buffer = t.tensor([-0.125, 0.625, 1.125], dtype=t.float64, device="cpu")
    calls = []
    def own_metric(*args):
        calls.append(1)
        return buffer
    result = ns["checked_learner_sweep"](
        layers=(0,), masks={"object": None}, pair_indices=(7, 9, 11),
        clean_logits=clean, corrupt_logits=corrupt, clean_ids=a, corrupt_ids=b,
        patch_logits_callback=lambda *args: clean, metric_callback=own_metric,
    )
    recovery = result["rows"][0]["recovery"]
    assert recovery.dtype == t.float64 and recovery.device.type == "cpu"
    assert recovery.tolist() == [-0.125, 0.625, 1.125]
    assert result["curves"]["object"] == [buffer.mean().item()]
    assert result["gap"].dtype == t.float32 and result["pair_indices"] == (7, 9, 11)
    buffer.fill_(99.)
    assert recovery.tolist() == [-0.125, 0.625, 1.125]
    assert calls == [1]
