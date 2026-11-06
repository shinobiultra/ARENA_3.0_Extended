"""Execution display state belongs to retained outputs, never changed code."""
import importlib.util
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'infrastructure/core'))
sys.path.insert(0, str(ROOT / 'infrastructure/extension'))
spec = importlib.util.spec_from_file_location('extension_output_build', ROOT / 'infrastructure/extension/build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def executed_notebook():
    cell = nbformat.v4.new_code_cell('display(widget)', execution_count=3,
        metadata={'historical_reference': {'source_sha256': 'archived-source'},
                  'execution': {'iopub.status.idle': 'historical-time'}},
        outputs=[nbformat.v4.new_output('display_data', data={
            'application/vnd.jupyter.widget-view+json': {'model_id': 'view', 'version_major': 2, 'version_minor': 0}
        })])
    return nbformat.v4.new_notebook(cells=[cell], metadata={
        'widgets': {'application/vnd.jupyter.widget-state+json': {
            'state': {'view': {'model_name': 'DOMWidgetModel', 'state': {'layout': 'IPY_MODEL_layout'}},
                      'layout': {'model_name': 'LayoutModel', 'state': {}}},
            'version_major': 2, 'version_minor': 0}},
        'execution_provenance': {'kind': 'historical', 'receipt': 'unchanged-receipt'},
        'language_info': {'name': 'python', 'version': '3.14.2'}})


def test_retained_output_keeps_widget_dependency_and_execution_provenance(tmp_path):
    old = executed_notebook()
    path = tmp_path / 'solutions.ipynb'
    fresh = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('new explanation'),
        nbformat.v4.new_code_cell('display(widget)', metadata={'tags': ['new-teaching-tag']})],
        metadata={'language_info': {'name': 'python'}})
    nbformat.write(fresh, path)
    assert build.carry_over_outputs(old, path)
    result = nbformat.read(path, as_version=4)
    assert result.metadata.widgets == old.metadata.widgets
    assert result.metadata.execution_provenance == old.metadata.execution_provenance
    assert result.metadata.language_info == old.metadata.language_info
    assert result.cells[1].metadata.historical_reference == old.cells[0].metadata.historical_reference
    assert result.cells[1].metadata.execution == old.cells[0].metadata.execution
    assert result.cells[1].metadata.tags == ['new-teaching-tag']
    assert result.cells[1].outputs == old.cells[0].outputs
    assert result.cells[1].execution_count == 3
    assert result.cells[0].source == 'new explanation'


def test_changed_code_cannot_inherit_widget_state_or_execution(tmp_path):
    path = tmp_path / 'solutions.ipynb'
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('different_method()')]), path)
    before = path.read_bytes()
    assert not build.carry_over_outputs(executed_notebook(), path)
    assert path.read_bytes() == before


def test_restored_widget_states_match_archives_and_resolve_view_dependencies():
    import json
    import tarfile

    receipt = json.loads((ROOT / 'docs/mastery/evidence/widget_state_repair_20261016/receipt.json').read_text())
    for record in receipt.values():
        with tarfile.open(ROOT / record['archive']) as tar:
            old = json.loads(tar.extractfile(record['archive_member']).read())
        current = nbformat.read(ROOT / record['destination'], as_version=4)
        assert current.metadata.widgets == old['metadata']['widgets']
        states = current.metadata.widgets['application/vnd.jupyter.widget-state+json']['state']
        assert set(record['view_ids']).issubset(states)

        def references(value):
            if isinstance(value, str) and value.startswith('IPY_MODEL_'):
                yield value.removeprefix('IPY_MODEL_')
            elif isinstance(value, dict):
                for item in value.values():
                    yield from references(item)
            elif isinstance(value, list):
                for item in value:
                    yield from references(item)

        assert set(references(states)).issubset(states)
