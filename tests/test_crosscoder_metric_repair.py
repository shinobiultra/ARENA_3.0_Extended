"""Independent literal oracles and deliberately incorrect method mutations."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from mastery_hub.practice_addons.crosscoder_metric_repair import lesson
from mastery_hub.practice_addons.crosscoder_metric_repair.generate import build_notebook, generate


def methods():
    # Test the functions actually emitted into the worked notebook, not hidden helpers.
    namespace = {"np": np}
    for cell in build_notebook(True)["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            tree = ast.parse(source)
            if tree.body and isinstance(tree.body[0], ast.FunctionDef):
                exec(compile(tree, "<worked-notebook-method>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("fixture", range(4))
def test_immediate_hand_oracles(fixture):
    namespace = methods()
    for check in lesson.CHECKS[:fixture + 1]:
        exec(check, namespace)


def test_pearson_general_shape_offset_scale_and_constants():
    metric = methods()["pearson_columns"]
    np.testing.assert_allclose(metric([2, 4], [[0, 5], [8, 5]]), [1, 0])
    np.testing.assert_allclose(metric([0, 0, 1, 1], [[0], [1], [0], [1]]), [0])
    for x, y in [([1], [[1]]), ([1, 2], [[1]]), ([np.nan, 2], [[1], [2]])]:
        with pytest.raises(ValueError):
            metric(x, y)


def test_auc_invalid_inputs_and_orientation():
    namespace = methods()
    for scores, labels in [([], []), ([1, 2], [0, 0]), ([1, 2], [0, 2]),
                           ([np.inf, 1], [0, 1]), ([[1, 2]], [[0, 1]])]:
        with pytest.raises(ValueError):
            namespace["rank_auc"](scores, labels)
    with pytest.raises(ValueError):
        namespace["evaluate_frozen"]([1, 2], [0, 1], 0)


def test_immediate_oracles_reject_sample_scaling_mutant():
    namespace = methods()
    namespace["pearson_columns"] = lambda x, y: np.abs(np.mean(
        ((x - np.mean(x)) / np.std(x, ddof=1))[:, None] *
        ((y - np.mean(y, axis=0)) / np.maximum(np.std(y, axis=0, ddof=1), 1e-12)), axis=0))
    with pytest.raises(AssertionError):
        exec(lesson.CHECKS[0], namespace)


def test_immediate_oracles_reject_always_absolute_auc_mutant():
    namespace = methods()
    correct = namespace["rank_auc"]
    namespace["rank_auc"] = lambda s, y: max(correct(s, y), 1-correct(s, y))
    with pytest.raises(AssertionError):
        exec(lesson.CHECKS[1], namespace)


def test_immediate_oracles_reject_eval_reorientation_mutant():
    namespace = methods()
    exec(lesson.CHECKS[2], namespace)
    namespace["evaluate_frozen"] = lambda s, y, sign: max(
        namespace["rank_auc"](s, y), 1-namespace["rank_auc"](s, y))
    with pytest.raises(AssertionError):
        exec(lesson.CHECKS[3], namespace)


def test_eval_does_not_fit_or_use_eval_labels_for_orientation():
    namespace = methods()
    sign = namespace["fit_orientation"]([3, 2, 1, 0], [0, 0, 1, 1])

    def forbidden(*args):
        raise AssertionError("EVAL called TRAIN fitting")

    namespace["fit_orientation"] = forbidden
    assert namespace["evaluate_frozen"]([0, 1, 2, 3], [0, 0, 1, 1], sign) == 0
    assert namespace["evaluate_frozen"]([0, 1, 2, 3], [1, 1, 0, 0], sign) == 1


def test_generator_preserves_filled_answers_and_recorded_outputs(tmp_path):
    generate(tmp_path)
    exercises = tmp_path / "exercises.ipynb"
    document = json.loads(exercises.read_text())
    code = next(c for c in document["cells"] if c["cell_type"] == "code")
    code.update(source=["my_answer = 42\n"], execution_count=19,
                outputs=[{"output_type": "stream", "name": "stdout", "text": "saved work"}])
    exercises.write_text(json.dumps(document))
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.ipynb")}
    generate(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.ipynb")}
    assert sum("raise NotImplementedError" in "".join(c["source"])
               for c in build_notebook(False)["cells"]) == 4


def test_play_changes_eval_without_refitting_train():
    namespace = methods()
    namespace["orientation"] = namespace["fit_orientation"]([3, 2, 1, 0], [0, 0, 1, 1])
    play = next(source for kind, source in lesson.notebook_cells(True)
                if kind == "code" and source.startswith("EVAL_SIGN ="))
    exec(play.replace("EVAL_SIGN = 1", "EVAL_SIGN = -1").replace("N = 4", "N = 8"), namespace)
    assert namespace["play_auc"] == 1
    assert namespace["play_legacy"] == pytest.approx(.875)
    assert namespace["play_pearson"] == pytest.approx(1)
