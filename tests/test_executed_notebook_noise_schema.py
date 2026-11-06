"""Saved execution cleanup must preserve the public notebook schema and evidence."""
from copy import deepcopy

import nbformat

from infrastructure.extension.build import strip_environment_noise


def test_execution_cleanup_keeps_markdown_valid_and_code_evidence():
    markdown=nbformat.v4.new_markdown_cell('A visible worked result')
    code=nbformat.v4.new_code_cell('print(3)',execution_count=1,outputs=[
        nbformat.v4.new_output('stream',name='stdout',text='3\n'),
        nbformat.v4.new_output('stream',name='stderr',text='A genuine warning remains\n'),
    ])
    notebook=nbformat.v4.new_notebook(cells=[markdown,code])
    expected_markdown=deepcopy(markdown)
    expected_code=deepcopy(code)
    strip_environment_noise(notebook)
    nbformat.validate(notebook)
    assert notebook.cells[0]==expected_markdown
    assert notebook.cells[1]==expected_code


def test_known_environment_warning_does_not_erase_a_real_warning():
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('Read first'),
        nbformat.v4.new_code_cell('pass',execution_count=2,outputs=[
            nbformat.v4.new_output('stream',name='stderr',text='[transformers] `torch_dtype` is deprecated.\nImportant model mismatch\n'),
            nbformat.v4.new_output('stream',name='stdout',text='kept result\n'),
        ])])
    strip_environment_noise(notebook)
    nbformat.validate(notebook)
    assert notebook.cells[1].execution_count==2
    assert notebook.cells[1].outputs[0].text=='Important model mismatch\n'
    assert notebook.cells[1].outputs[1].text=='kept result\n'
