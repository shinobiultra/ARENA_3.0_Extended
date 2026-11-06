"""build.py keeps saved outputs when a rebuild leaves the code cells' programs unchanged."""

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infrastructure" / "extension"))
sys.path.insert(0, str(ROOT / "infrastructure" / "core"))

import build  # noqa: E402


def test_formatting_only_changes_keep_the_program():
    old = "x = 1\r\ny = x + 1  # add one\r\nprint(y)"
    assert build.same_code(old, "x = 1\n\ny = x + 1\n\nprint(y)\n")


def test_code_and_string_changes_are_changes():
    assert not build.same_code("print(x + 1)", "print(x + 2)")
    assert not build.same_code('print("a\\n\\nb")', 'print("a\\nb")')


def test_magics_compare_as_text():
    assert build.same_code("%matplotlib inline\nx = 1", "%matplotlib inline   \nx = 1")
    assert not build.same_code("%pip install a", "%pip install b")


def _notebook(sources, outputs=False):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("# Title")]
    for index, source in enumerate(sources, 1):
        cell = nbformat.v4.new_code_cell(source)
        if outputs:
            cell.execution_count = index
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=f"{index}\n")]
        nb.cells.append(cell)
    return nb


def test_carry_over_keeps_outputs_for_reformatted_code(tmp_path):
    previous = _notebook(["a = 1\r\nprint(a)", "print(2)"], outputs=True)
    solutions = tmp_path / "solutions.ipynb"
    nbformat.write(_notebook(["a = 1\n\nprint(a)", "print(2)"]), solutions)
    assert build.carry_over_outputs(previous, solutions)
    fresh = nbformat.read(solutions, as_version=4)
    code = [cell for cell in fresh.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == [1, 2]
    assert code[0].source == "a = 1\n\nprint(a)"


def test_carry_over_refuses_changed_code(tmp_path):
    previous = _notebook(["print(1)", "print(2)"], outputs=True)
    solutions = tmp_path / "solutions.ipynb"
    nbformat.write(_notebook(["print(1)", "print(3)"]), solutions)
    assert not build.carry_over_outputs(previous, solutions)
