"""No-execute generation retains reference provenance without learner answers."""
from copy import deepcopy
from types import SimpleNamespace
import sys
import nbformat
import pytest
from infrastructure.extension import build


def test_generator_keeps_reference_metadata_and_never_carries_answers(tmp_path, monkeypatch):
    chapter = 'chapter5_modern_architectures'
    directory = tmp_path/chapter/'exercises/part2_mixture_of_experts'
    directory.mkdir(parents=True)
    path = directory/'5.2_Mixture_of_Experts_from_Scratch_exercises.ipynb'
    old = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise NotImplementedError')])
    old.metadata['fresh_reference'] = {'kind': 'saved reference; student unexecuted', 'section': '5.2'}
    old.metadata['learner_answers'] = {'private': 'do not carry'}
    old.cells[0].execution_count = 7
    old.cells[0].outputs = [nbformat.v4.new_output('stream', name='stdout', text='old learner answer')]
    nbformat.write(old, path)
    generated = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise NotImplementedError')])
    generated.cells[0].metadata['new_teacher_hint'] = 'retain generated fields'
    expected_cells = deepcopy(generated.cells)
    masters = tmp_path/'masters'
    masters.mkdir()
    (masters/'master_5_2.py').write_text('# static fake generator fixture\n')
    mapping = {'exercise_dir': directory.name, 'streamlit_page': '02_[5.2]_Mixture_of_Experts_from_Scratch'}
    def generate(section, config):
        nbformat.write(generated, path)
        nbformat.write(nbformat.v4.new_notebook(), path.with_name(path.name.replace('_exercises', '_solutions')))
        return SimpleNamespace(exercises_dir=directory, colab_name='5.2_Mixture_of_Experts_from_Scratch', source_preservation_applied=False)
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(build, 'MASTERS', masters)
    monkeypatch.setattr(build, 'load_config', lambda: {'conversion_map': {'5.2': mapping}, 'chapter_names': {5: chapter}})
    monkeypatch.setattr(build, 'generate', generate)
    monkeypatch.setattr(build, 'rewrite_page_media', lambda master: None)
    monkeypatch.setattr(build, 'inline_media', lambda *args: None)
    monkeypatch.setattr(build, 'summarise', lambda master: 'static fixture')
    monkeypatch.setattr(sys, 'argv', ['build.py', '5.2'])
    build.main()
    actual = nbformat.read(path, as_version=4)
    assert actual.metadata['fresh_reference'] == old.metadata['fresh_reference']
    assert 'learner_answers' not in actual.metadata
    assert actual.cells == expected_cells
    assert actual.cells[0].execution_count is None and not actual.cells[0].outputs


@pytest.mark.parametrize('changed_source', ['return_value = 2', 'return_value = 1\n# edited'])
def test_changed_exercise_code_does_not_keep_stale_provenance(tmp_path, changed_source):
    old = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('return_value = 1')])
    old.metadata['fresh_reference'] = {'section': '5.2'}
    current = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(changed_source)])
    path = tmp_path/'exercise.ipynb'
    nbformat.write(current, path)
    before = path.read_bytes()
    assert not build.carry_over_exercise_provenance(old, path)
    assert path.read_bytes() == before


def test_teaching_metadata_merges_missing_fields_but_generated_values_win(tmp_path):
    old = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise NotImplementedError')])
    old.metadata['migration'] = {
        'execution_kind': 'unexecuted_with_expected_displays', 'fresh_execution': False,
        'source_notebook_metadata': {'source': {'arena': {'section': 'old'}, 'trusted': True}},
        'source_cell_metadata': {'source': [{'id': 'old-source-id', 'cell_type': 'code',
            'metadata': {'arena': {'exercise_functions': ['fit'], 'answers': ['secret']},
                         'trusted': True, 'execution': {'iopub.status.busy': 'old timing'}}}]},
        'learner_answers': 'secret', 'execution_count': 5, 'trusted': True}
    old.metadata['arena'] = {'section': 'old', 'status': 'prior-reviewed', 'answers': 'secret'}
    old.cells[0].metadata = {
        'arena': {'exercise_functions': ['old-fit'], 'calling_cell': 'old-call',
                  'provided_code': 'Supplied setup', 'answers': 'secret'},
        'jupyter': {'source_hidden': True, 'outputs_hidden': True, 'trusted': True},
        'tags': ['old-teaching-tag'], 'trusted': True,
        'execution': {'shell.execute_reply': 'old timing'}, 'answers': 'secret'}
    old.cells[0].execution_count = 8
    old.cells[0].outputs = [nbformat.v4.new_output('stream', name='stdout', text='old learner answer')]
    new = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise NotImplementedError')])
    new.metadata['migration'] = {'execution_kind': 'new-generated-kind'}
    new.metadata['arena'] = {'section': 'new', 'new_generated_hint': 'keep'}
    new.cells[0].metadata = {'arena': {'exercise_functions': ['new-fit']},
                             'jupyter': {'source_hidden': False}, 'tags': ['new-tag']}
    new.cells[0].outputs = [nbformat.v4.new_output('display_data', data={'text/plain': 'new expected display'})]
    original_current_cell = deepcopy(new.cells[0])
    path = tmp_path/'exercise.ipynb'
    nbformat.write(new, path)
    assert build.carry_over_exercise_provenance(old, path)
    actual = nbformat.read(path, as_version=4)
    assert actual.metadata['arena'] == {'section': 'new', 'status': 'prior-reviewed', 'new_generated_hint': 'keep'}
    assert actual.metadata['migration']['execution_kind'] == 'new-generated-kind'
    assert actual.metadata['migration']['fresh_execution'] is False
    assert actual.metadata['migration']['source_notebook_metadata'] == {'source': {'arena': {'section': 'old'}}}
    assert actual.metadata['migration']['source_cell_metadata']['source'][0]['metadata'] == {'arena': {'exercise_functions': ['fit']}}
    assert not {'learner_answers', 'execution_count', 'trusted'} & actual.metadata['migration'].keys()
    assert actual.cells[0].metadata == {
        'arena': {'exercise_functions': ['new-fit'], 'calling_cell': 'old-call', 'provided_code': 'Supplied setup'},
        'jupyter': {'source_hidden': False, 'outputs_hidden': True}, 'tags': ['new-tag']}
    assert actual.cells[0].source == original_current_cell.source
    assert actual.cells[0].outputs == original_current_cell.outputs
    assert actual.cells[0].execution_count == original_current_cell.execution_count is None
    assert actual.cells[0].id == original_current_cell.id
    # A second no-execute carry is idempotent, including presentation and provenance.
    first = path.read_bytes()
    assert build.carry_over_exercise_provenance(old, path)
    assert path.read_bytes() == first


def test_changed_student_answer_code_and_extra_cells_never_carry_metadata(tmp_path):
    old = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('answer = 42')])
    old.metadata['migration'] = {'fresh_execution': False}
    old.cells[0].metadata['arena'] = {'calling_cell': 'old role'}
    for cells in ([nbformat.v4.new_code_cell('raise NotImplementedError')],
                  [nbformat.v4.new_code_cell('answer = 42'), nbformat.v4.new_code_cell('extra = 1')]):
        current = nbformat.v4.new_notebook(cells=cells)
        path = tmp_path/'exercise.ipynb'
        nbformat.write(current, path)
        before = path.read_bytes()
        assert not build.carry_over_exercise_provenance(old, path)
        assert path.read_bytes() == before


def test_notebook_kernel_and_language_provenance_preserves_only_allowed_fields(tmp_path):
    old = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise NotImplementedError')])
    old.metadata['kernelspec'] = {
        'display_name': 'Original Python 3.11', 'language': 'python', 'name': 'python311',
        'trusted': True, 'execution_count': 12, 'learner_answers': 'secret'}
    old.metadata['language_info'] = {
        'name': 'old-python', 'version': '3.11', 'pygments_lexer': 'ipython3',
        'timing': 'old execution', 'answers': 'secret', 'trusted': True}
    old.metadata['execution'] = {'iopub.execute_input': 'old runtime state'}
    old.metadata['trusted'] = True
    generated = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise NotImplementedError')])
    generated.metadata['language_info'] = {'name': 'python'}
    generated.metadata['kernelspec'] = {'name': 'generated-kernel', 'display_name': 'Generated kernel'}
    path = tmp_path/'exercise.ipynb'
    nbformat.write(generated, path)
    assert build.carry_over_exercise_provenance(old, path)
    actual = nbformat.read(path, as_version=4)
    assert actual.metadata['kernelspec'] == {
        'name': 'generated-kernel', 'display_name': 'Generated kernel', 'language': 'python'}
    assert actual.metadata['language_info'] == {
        'name': 'python', 'version': '3.11', 'pygments_lexer': 'ipython3'}
    assert 'execution' not in actual.metadata and 'trusted' not in actual.metadata
    assert actual.cells == generated.cells
    # Freshly generated lexer identity wins the same recursive conflict.
    generated.metadata['language_info']['pygments_lexer'] = 'generated-lexer'
    nbformat.write(generated, path)
    assert build.carry_over_exercise_provenance(old, path)
    actual = nbformat.read(path, as_version=4)
    assert actual.metadata['language_info'] == {
        'name': 'python', 'version': '3.11', 'pygments_lexer': 'generated-lexer'}
    assert actual.cells == generated.cells
    # The same original provenance must not follow changed student source.
    changed = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('student_answer = 1')])
    nbformat.write(changed, path)
    before = path.read_bytes()
    assert not build.carry_over_exercise_provenance(old, path)
    assert path.read_bytes() == before


def test_cot_original_cell_provenance_sanitized_and_generated_conflicts_win(tmp_path):
    old = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise NotImplementedError')])
    old.metadata['migration'] = {
        'status': 'unexecuted with expected displays',
        'original_all_cell_metadata': [{
            'legacy_cell': 12, 'id': 'original-id', 'cell_type': 'code',
            'metadata': {'arena': {'exercise_functions': ['fit'], 'answers': 'secret'},
                         'jupyter': {'source_hidden': True, 'trusted': True},
                         'tags': ['exercise'], 'execution': {'busy': 'old timing'},
                         'trusted': True, 'learner_answers': 'secret'},
            'answers': 'secret', 'execution_count': 9, 'trusted': True, 'timing': 'old'}],
        'learner_answers': 'secret'}
    generated = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise NotImplementedError')])
    path = tmp_path/'exercise.ipynb'
    nbformat.write(generated, path)
    assert build.carry_over_exercise_provenance(old, path)
    actual = nbformat.read(path, as_version=4)
    assert actual.metadata['migration'] == {
        'status': 'unexecuted with expected displays',
        'original_all_cell_metadata': [{
            'legacy_cell': 12, 'id': 'original-id', 'cell_type': 'code',
            'metadata': {'arena': {'exercise_functions': ['fit']},
                         'jupyter': {'source_hidden': True}, 'tags': ['exercise']}}]}
    assert actual.cells == generated.cells
    generated.metadata['migration'] = {
        'status': 'new generated status',
        'original_all_cell_metadata': [{'legacy_cell': 99, 'metadata': {'tags': ['new']}}]}
    nbformat.write(generated, path)
    assert build.carry_over_exercise_provenance(old, path)
    actual = nbformat.read(path, as_version=4)
    assert actual.metadata['migration'] == generated.metadata['migration']
