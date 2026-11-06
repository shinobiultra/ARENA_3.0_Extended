"""Current extension lessons must point learners to the prediction pathway."""
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'infrastructure/extension'))
import build


def test_extension_reference_links_offer_guided_prediction_workbook(tmp_path):
    # Link generation never reads the master; the generated .ipynb is not checked in.
    master_path = tmp_path / 'master_7_1.ipynb'
    master_path.write_text('{}')
    master = build.ExtensionMasterFile(
        master_path=master_path,
        chapter_dir=ROOT/'chapter7_activation_to_language',
        chapter_name_long='Chapter 7 - Activation to Language',
        exercise_dir_name='part1_lenses_patchscopes',
        streamlit_page_name='01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes',
    )
    streamlit, notebooks = master.get_page_links()
    assert any('extension_predictions/exercises.ipynb' in line for line in notebooks)
    assert any('guided prediction' in line.lower() for line in streamlit)
    assert any('practice' in line.lower() for line in notebooks)


def test_extension_normalization_preserves_code_evidence_and_validates(tmp_path):
    path = tmp_path / 'generated.ipynb'
    nb = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('Historical result'),
        nbformat.v4.new_code_cell('print(2)', execution_count=7,
            outputs=[nbformat.v4.new_output('stream', name='stdout', text='2\n')]),
    ])
    nb.cells[0].update(execution_count=None, outputs=[])
    # The raw converter output is deliberately schema-invalid here.
    path.write_text(__import__('json').dumps(nb))
    build.normalize_generated_notebook(path)
    fixed = nbformat.read(path, as_version=4)
    nbformat.validate(fixed)
    assert fixed.cells[1] == nb.cells[1]
    assert 'outputs' not in fixed.cells[0] and 'execution_count' not in fixed.cells[0]
    before = path.read_bytes()
    build.normalize_generated_notebook(path)
    assert path.read_bytes() == before


def test_extension_normalization_rejects_noncode_execution_evidence(tmp_path):
    path = tmp_path / 'unexpected.ipynb'
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('Keep this')])
    nb.cells[0].update(execution_count=3, outputs=[{'text': 'retained evidence'}])
    path.write_text(__import__('json').dumps(nb))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='execution evidence'):
        build.normalize_generated_notebook(path)
    assert path.read_bytes() == before
