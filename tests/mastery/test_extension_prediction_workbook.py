"""Prediction-workbook invariants; no model, network, training or hidden answers."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
LESSON = ROOT / 'mastery_hub/units/extension_predictions/lesson.py'
GENERATOR = ROOT / 'scripts/build_extension_prediction_workbook.py'
SECTIONS = {
    '0.6', '1.6',
    '5.1', '5.2', '5.3', '5.4', '5.5', '5.6', '5.7', '5.8', '5.9',
    '6.1', '6.2', '6.3', '6.4', '6.5',
    '7.1', '7.2', '7.3', '7.4', '7.5', '7.6',
    '8.1', '8.2', '8.3', '8.4', '8.5', '8.6', '8.7', '8.8',
    '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', '9.7', '10.1',
    '11.1', '11.2', '11.3', '11.4', '11.5', '11.6',
    '12.1', '12.2', '12.3', '12.4', '12.5', '12.6', '12.7',
    '13.1', '13.2', '13.3', '13.4', '13.5', '13.6',
    '14.1', '14.2', '14.3', '14.4', '14.5', '14.6', '14.7',
    '15.1', '15.2', '15.3', '15.4', '15.5',
    '16.1', '16.2', '16.3', '16.4', '16.5', '16.6', '16.7', '16.8', '16.9', '16.10', '16.11',
    '17.1', '17.2', '17.3', '17.4', '18.1', '18.2',
}


def load(path):
    spec = importlib.util.spec_from_file_location('extension_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lesson():
    return load(LESSON)


def setup_case(lesson, section='7.1'):
    case = next(c for c in lesson.load_cases() if c['section'] == section)
    inputs = {key: 'exact fixture input: ' + key for key in case['config_fields']}
    config = lesson.freeze_inputs(case, inputs)
    prediction = {'numeric': 0.5, 'ranking': case['forecast_items'], 'binary': True}[case['prediction_type']]
    record = lesson.commit_prediction(case, config, prediction, 0.7, 'Causal fixture explanation.')
    return case, config, record


def observe(lesson, case, config, record):
    return lesson.record_observation(record, config, record['prediction'],
        {name: 0.25 for name in case['controls']},
        'learner_run', 'fixture measurement from my visible implementation; fixture only')


def complete(lesson):
    case, config, record = setup_case(lesson)
    measured = observe(lesson, case, config, record)
    return lesson.reflect(measured, 'My estimate matched; mechanism remains uncertain.',
        'Change the inspection task; prompt priors could dominate.',
        {'numeric': 0.5, 'binary': True}[case['transfer_prediction_type']], 0.6,
        'Text-only controls may explain the answer.')


def test_case_coverage_and_existing_current_paths(lesson):
    manifest = json.loads((ROOT / 'mastery_hub/extension_checkpoints.json').read_text())
    assert manifest['schema_version'] == 1
    assert {c['section'] for c in lesson.load_cases()} == SECTIONS
    assert len(lesson.load_cases()) == len(SECTIONS)
    assert [case['section'] for case in lesson.load_cases()] == sorted(
        SECTIONS, key=lambda section: tuple(map(int, section.split('.')))
    ), 'Cases must retain configured numeric order, including 16.9 before 16.10'
    for case in lesson.load_cases():
        for field in ('title', 'question', 'config_fields', 'controls', 'method_functions',
                      'experiment', 'transfer', 'known_limitations', 'observation_prompt'):
            assert case[field], (case['section'], field)
        assert case['status'] == 'guided_practice'
        assert case['prediction_type'] in ('numeric', 'ranking', 'binary')
        assert case['evidence_class'] in ('analytic_toy', 'recorded_real_run')
        assert case['data_role'] in ('historical', 'monitored', 'unknown')
        for key in ('master_path', 'exercise_path', 'solutions_path'):
            path = ROOT / case[key]
            assert path.is_file() and path.resolve().is_relative_to(ROOT)
        master = (ROOT / case['master_path']).read_text()
        for function in case['method_functions']:
            assert 'def ' + function + '(' in master or 'class ' + function + '(' in master
        assert not {'answer', 'expected_answer', 'reference_answer'} & set(case)


def test_notebook_prompts_precede_commit_and_references(lesson):
    cells = lesson.build_cells()
    notebook = nbformat.v4.new_notebook(cells=[nbformat.from_dict(c) for c in cells])
    nbformat.validate(notebook)
    assert len({c['id'] for c in cells}) == len(cells)
    assert cells == lesson.build_cells()
    text = '\n'.join(c['source'] for c in cells)
    assert 'guided practice' in text.lower() and 'unknown' in text
    assert 'not compatible' in text and 'learner_reported' in text
    for section in SECTIONS:
        relevant = [c for c in cells if c['metadata'].get('section') == section]
        assert [c['metadata']['stage'] for c in relevant] == ['forecast', 'commit', 'experiment', 'observe', 'reflect']
        assert 'section-' + section.replace('.', '-') in relevant[0]['source']
        assert '_solutions.ipynb' in relevant[0]['source']
        assert 'answer-rich' in relevant[0]['source']
        assert 'confidence' in relevant[1]['source'] and 'reasoning' in relevant[1]['source']
        assert 'NotImplementedError' in relevant[1]['source']
        assert 'export_record(' not in relevant[-1]['source']
    for cell in cells:
        assert 'reference_answer' not in cell['metadata']
        if cell['cell_type'] == 'code':
            compile(cell['source'], '<workbook>', 'exec')
            assert cell['execution_count'] is None and cell['outputs'] == []


def test_commit_is_copy_and_cannot_be_replaced(lesson):
    case, config, record = setup_case(lesson)
    original = deepcopy(record)
    config['inputs'][next(iter(config['inputs']))] = 'changed'
    assert record == original
    assert record['prior_exposure'] == 'unknown'
    assert record['assessment'] == 'guided_practice'
    assert 'score' not in record
    record['prediction'] = 999
    with pytest.raises(ValueError, match='seal'):
        lesson.validate_record(record)


def test_configuration_change_rejected_before_observation(lesson):
    case, config, record = setup_case(lesson)
    config['inputs'][next(iter(config['inputs']))] = 'changed'
    with pytest.raises(ValueError, match='configuration changed'):
        observe(lesson, case, config, record)
    assert record['status'] == 'committed'


def test_observation_and_reflection_are_immutable(lesson):
    case, config, record = setup_case(lesson)
    measured = observe(lesson, case, config, record)
    assert record['status'] == 'committed' and record['observation'] is None
    assert measured['status'] == 'observed' and measured['observation']['verification'] == 'learner_reported'
    with pytest.raises(ValueError):
        observe(lesson, case, config, measured)
    with pytest.raises(ValueError):
        lesson.reflect(record, 'x', 'y', True, 0.5, 'z')
    completed = complete(lesson)
    assert completed['status'] == 'completed'
    assert len(completed['history']) == 3
    assert completed['transfer']['prediction'] is True
    assert completed['reflection']


def test_completed_roundtrip_is_reported_not_new_execution(lesson):
    record = complete(lesson)
    restored = lesson.loads_record(lesson.dumps_record(record))
    assert restored == record
    replay = lesson.replay_record(restored)
    assert replay['verification'] == 'learner_reported'
    assert replay['executed'] is False
    assert replay['record'] == record
    assert 'arena-extension-guided-practice' == record['schema']
    assert record['schema_version'] == 2


@pytest.mark.parametrize('section', sorted(SECTIONS))
def test_every_case_can_be_committed_and_observed(lesson, section):
    case, config, record = setup_case(lesson, section)
    assert observe(lesson, case, config, record)['status'] == 'observed'


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), {'__reduce__': 'code'}, 'not numeric'])
def test_invalid_numeric_predictions_rejected(lesson, bad):
    case, config, _ = setup_case(lesson)
    case = deepcopy(case)
    case['prediction_type'] = 'numeric'
    config = lesson.freeze_inputs(case, {key: 'fixture' for key in case['config_fields']})
    with pytest.raises(ValueError):
        lesson.commit_prediction(case, config, bad, 0.6, 'Reason')


def test_strict_record_parser_and_history_tampering(lesson):
    record = complete(lesson)
    for altered in (dict(record, unexpected=True), dict(record, schema_version=True),
                    dict(record, status='unseen')):
        with pytest.raises(ValueError):
            lesson.loads_record(json.dumps(altered))
    with pytest.raises(ValueError):
        lesson.loads_record('{"schema":1,"schema":2}')
    record['history'][0]['payload']['reasoning'] = 'rewrite history'
    with pytest.raises(ValueError, match='seal'):
        lesson.validate_record(record)


def test_missing_control_and_inputs_rejected(lesson):
    case, config, record = setup_case(lesson)
    with pytest.raises(ValueError):
        lesson.freeze_inputs(case, {})
    with pytest.raises(ValueError):
        lesson.record_observation(record, config, record['prediction'], {}, 'learner_run', 'measured')
    with pytest.raises(ValueError):
        lesson.commit_prediction(case, config, record['prediction'], 1.1, 'Reason')


def test_explicit_private_export_only(lesson, tmp_path):
    record = complete(lesson)
    target = lesson.export_record(record, tmp_path / 'private')
    assert target.stat().st_mode & 0o777 == 0o600
    assert lesson.load_record(target) == record
    with pytest.raises(FileExistsError):
        lesson.export_record(record, tmp_path / 'private')
    with pytest.raises(ValueError, match='outside'):
        lesson.export_record(record, ROOT / 'mastery_hub')
    other_repo = tmp_path / 'other-repo'
    (other_repo / '.git').mkdir(parents=True)
    (other_repo / '.git/HEAD').write_text('ref: refs/heads/main\n')
    with pytest.raises(ValueError, match='outside'):
        lesson.export_record(record, other_repo / 'private')


def test_generator_check_is_read_only_and_confined():
    generator = load(GENERATOR)
    path = ROOT / 'mastery_hub/units/extension_predictions/exercises.ipynb'
    assert generator.OUTPUT == path
    before = path.read_bytes()
    result = subprocess.run([sys.executable, str(GENERATOR), '--check'], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.read_bytes() == before
    assert nbformat.reads(before.decode(), as_version=4) == generator.make_notebook()


def test_notebook_output_changes_do_not_change_experiment_source(lesson, tmp_path):
    path = tmp_path / 'source.ipynb'
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('x = 1')])
    nbformat.write(notebook, path)
    initial = lesson.source_digest(path)
    notebook.cells[0].execution_count = 4
    notebook.cells[0].outputs = [nbformat.v4.new_output('stream', name='stdout', text='a measured result')]
    notebook.cells[0].metadata['execution'] = {'timestamp': 'runtime'}
    notebook.metadata['language_info'] = {'name': 'python', 'version': '3.14'}
    nbformat.write(notebook, path)
    assert lesson.source_digest(path) == initial
    notebook.cells[0].source = 'x = 2'
    nbformat.write(notebook, path)
    assert lesson.source_digest(path) != initial


def test_transfer_is_a_separate_frozen_forecast(lesson):
    record = complete(lesson)
    case = lesson.transfer_case(record)
    assert case['question'] == record['case']['transfer']
    assert case['prediction_type'] == 'binary'
    inputs = deepcopy(record['config']['inputs'])
    inputs['patchscope'] = 'exact changed language prompt and candidate-label manifest'
    config = lesson.freeze_inputs(case, inputs)
    new = lesson.commit_prediction(case, config, record['transfer']['prediction'],
        record['transfer']['confidence'], record['transfer']['reasoning'], prior_exposure='seen')
    assert new['record_id'] != record['record_id']
    assert new['config_fingerprint'] != record['config_fingerprint']
    assert new['status'] == 'committed' and record['status'] == 'completed'
    assert observe(lesson, case, config, new)['observation']['verification'] == 'learner_reported'


def test_malformed_ranking_items_fail_cleanly(lesson):
    case = next(c for c in lesson.load_cases() if c['prediction_type'] == 'ranking')
    case['forecast_items'] = [{}]
    with pytest.raises(ValueError):
        lesson.validate_case(case)


def test_completed_workbook_cells_execute_as_synthetic_plumbing(lesson, capsys):
    """Fill worksheet stubs with explicit fixtures; do not claim model reproduction."""
    namespace = {'__name__': '__main__'}
    for cell in lesson.build_cells():
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']
        stage = cell['metadata'].get('stage')
        if stage == 'commit':
            source = source.replace("{field: None for field in case['config_fields']}",
                                    "{field: 'synthetic plumbing fixture' for field in case['config_fields']}")
            source = source.replace('prediction = None',
                "prediction = {'numeric': 0.5, 'binary': True, 'ranking': case['forecast_items']}[case['prediction_type']]")
            source = source.replace('confidence = None', 'confidence = 0.7')
            source = source.replace("reasoning = ''", "reasoning = 'Synthetic fixture; no scientific interpretation.'")
        if stage == 'observe':
            source = source.replace('observed = None', "observed = committed['prediction']")
            source = source.replace("{name: None for name in case['controls']}",
                                    "{name: 0.0 for name in case['controls']}")
            source = source.replace('source_kind = None', "source_kind = 'historical_reference'")
            source = source.replace("source_receipt = ''", "source_receipt = 'Synthetic worksheet fixture; not a scientific run.'")
        if stage == 'reflect':
            source = source.replace("mismatch = ''", "mismatch = 'Synthetic fixture does not establish a result.'")
            source = source.replace("changed_condition = ''", "changed_condition = 'Synthetic changed-condition plan.'")
            source = source.replace('transfer_prediction = None',
                "transfer_prediction = {'numeric': 0.5, 'binary': True}[measured['case']['transfer_prediction_type']]")
            source = source.replace('transfer_confidence = None', 'transfer_confidence = 0.6')
            source = source.replace("transfer_reasoning = ''", "transfer_reasoning = 'Synthetic causal rationale.'")
        exec(compile(source, '<synthetic-workbook-cell>', 'exec'), namespace)
    assert set(namespace['records']) == SECTIONS
    assert all(record['status'] == 'completed' for record in namespace['records'].values())
    output = capsys.readouterr().out
    assert output.count('EXTENSION_RECORD_JSON_BEGIN') == 0
    assert output.count('EXTENSION_RECORD_JSON_END') == 0
    namespace['inspect_record'](next(iter(namespace['records'].values())))
    assert capsys.readouterr().out.count('EXTENSION_RECORD_JSON_BEGIN') == 1
    for section, record in namespace['records'].items():
        assert lesson.loads_record(lesson.dumps_record(record)) == record
        assert record['observation']['verification'] == 'learner_reported'
        assert record['transfer']['status'] == 'forecast_only'


def test_generator_preserves_saved_records_and_runtime_metadata(tmp_path):
    generator = load(GENERATOR)
    canonical = generator.make_notebook()
    saved = deepcopy(canonical)
    cell = next(c for c in saved.cells if c.cell_type == 'code')
    cell.execution_count = 4
    cell.outputs = [nbformat.v4.new_output('stream', name='stdout',
                                          text='EXTENSION_RECORD_JSON_BEGIN\nprivate worksheet\n')]
    cell.metadata['execution'] = {'timestamp': 'runtime only'}
    cell.metadata['trusted'] = True
    saved.metadata['language_info']['version'] = '3.14.2'
    saved.metadata['widgets'] = {'runtime': 'state'}
    target = tmp_path / 'saved-record.ipynb'
    nbformat.write(saved, target)
    before = target.read_bytes()
    assert generator.synchronize(target, canonical, check=True)
    assert generator.synchronize(target, canonical, check=False)
    assert target.read_bytes() == before


def test_generator_refuses_to_overwrite_filled_or_unverifiable_source(tmp_path):
    generator = load(GENERATOR)
    canonical = generator.make_notebook()
    filled = deepcopy(canonical)
    cell = next(c for c in filled.cells if c.metadata.get('stage') == 'commit')
    cell.source = cell.source.replace('prediction = None', 'prediction = 0.85')
    target = tmp_path / 'filled.ipynb'
    nbformat.write(filled, target)
    before = target.read_bytes()
    assert not generator.synchronize(target, canonical, check=True)
    assert not generator.synchronize(target, canonical, check=False)
    assert target.read_bytes() == before
    target.write_text('not a valid notebook; preserve this artifact')
    before = target.read_bytes()
    assert not generator.synchronize(target, canonical, check=False)
    assert target.read_bytes() == before


def test_generator_updates_pristine_stale_source_but_preserves_recorded_work(tmp_path):
    generator = load(GENERATOR)
    old = generator.make_notebook()
    expected = deepcopy(old)
    expected.cells[0].source += '\nA new canonical teaching clarification.\n'
    expected.metadata['extension_predictions']['source_baseline_sha256'] = generator.source_fingerprint(expected)
    target = tmp_path / 'stale-pristine.ipynb'
    nbformat.write(old, target)
    assert not generator.synchronize(target, expected, check=True)
    assert generator.synchronize(target, expected, check=False)
    assert nbformat.read(target, as_version=4) == expected
    cell = next(c for c in old.cells if c.cell_type == 'code')
    cell.execution_count = 1
    cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text='a saved prediction record')]
    nbformat.write(old, target)
    before = target.read_bytes()
    assert not generator.synchronize(target, expected, check=False)
    assert target.read_bytes() == before


def test_record_writer_reader_share_utf8_limit_including_export_newline(lesson, monkeypatch, tmp_path):
    case, config, _ = setup_case(lesson)
    record = lesson.commit_prediction(case, config, 0.5, 0.7, '予測した因果関係。' * 30)
    payload = lesson.dumps_record(record)
    byte_count = len((payload + '\n').encode('utf-8'))
    assert byte_count > len(payload) + 1
    monkeypatch.setattr(lesson, 'MAX_RECORD_BYTES', byte_count)
    assert lesson.loads_record(payload + '\n') == record
    target = lesson.export_record(record, tmp_path / 'within-limit')
    assert target.stat().st_size == byte_count
    assert lesson.load_record(target) == record
    monkeypatch.setattr(lesson, 'MAX_RECORD_BYTES', byte_count - 1)
    with pytest.raises(ValueError, match='byte limit'):
        lesson.dumps_record(record)
    rejected_directory = tmp_path / 'no-orphan-artifact'
    with pytest.raises(ValueError, match='byte limit'):
        lesson.export_record(record, rejected_directory)
    assert not rejected_directory.exists()
    with pytest.raises(ValueError):
        lesson.loads_record(payload + '\n')
    with pytest.raises(ValueError):
        lesson.load_record(target)


def test_preparation_and_subfunction_tests_precede_commitment(lesson):
    cells = lesson.build_cells()
    for case in lesson.load_cases():
        relevant = [c for c in cells if c['metadata'].get('section') == case['section']]
        prompt = relevant[0]['source']
        assert case['exercise_path'] in prompt
        assert 'sub-function tests before freezing' in prompt
        assert 'answer-rich' in prompt
        assert 'seen' in prompt
        after = relevant[2]['source']
        assert 'Run the frozen' in after
        assert 'edits require a new commitment' in after


def legacy_commitment(lesson, section='7.1'):
    """The original v1 commitment shape, with its original version and sealed history."""
    case, config, record = setup_case(lesson, section)
    record['schema_version'] = 1
    return case, config, lesson._seal(record)


def test_legacy_v1_is_readable_without_migration_or_resealing(lesson):
    case, config, committed = legacy_commitment(lesson)
    measured = observe(lesson, case, config, committed)
    assert set(measured['observation']) == {'value', 'controls', 'source_kind', 'source_receipt', 'verification'}
    completed = lesson.reflect(measured, 'Original reflection.', 'Original transfer condition.',
                               True, 0.6, 'Original transfer explanation.')
    original = lesson.dumps_record(completed)
    restored = lesson.loads_record(original)
    assert restored['schema_version'] == 1
    assert lesson.dumps_record(restored) == original
    assert restored['history'] == completed['history'] and restored['seal'] == completed['seal']
    assert lesson.prediction_feedback(restored)['confidence_meaning'] == 'legacy unscored self-rating'
    with pytest.raises(ValueError, match='legacy v1'):
        lesson.record_observation(committed, config, None, {k: 'missing' for k in case['controls']},
                                  'historical_reference', 'No evidence', observation_status='unresolved')
    restored['observation']['value'] = 999
    with pytest.raises(ValueError, match='seal'):
        lesson.validate_record(restored)


def test_binary_probability_is_for_selected_false_event_and_legacy_is_unscored(lesson):
    section = next(c['section'] for c in lesson.load_cases() if c['prediction_type'] == 'binary')
    case, config, _ = setup_case(lesson, section)
    committed = lesson.commit_prediction(case, config, False, 0.8, 'Predict the event is false.')
    controls = {k: 'observed control outcome' for k in case['controls']}
    correct = lesson.record_observation(committed, config, False, controls, 'learner_run', 'fixture')
    wrong = lesson.record_observation(committed, config, True, controls, 'learner_run', 'fixture')
    assert lesson.prediction_feedback(correct)['score'] == pytest.approx(0.04)
    assert lesson.prediction_feedback(wrong)['score'] == pytest.approx(0.64)
    case, config, legacy = legacy_commitment(lesson, section)
    assert lesson.prediction_feedback(observe(lesson, case, config, legacy))['score'] is None


def test_point_and_ranking_feedback_do_not_score_self_rating(lesson):
    for kind in ('numeric', 'ranking'):
        section = next(c['section'] for c in lesson.load_cases() if c['prediction_type'] == kind)
        case, config, committed = setup_case(lesson, section)
        actual = 0.2 if kind == 'numeric' else list(reversed(committed['prediction']))
        observed = lesson.record_observation(committed, config, actual,
            {k: 'measured' for k in case['controls']}, 'learner_run', 'fixture only')
        feedback = lesson.prediction_feedback(observed)
        assert feedback['confidence_meaning'] == 'unscored self-rating'
        if kind == 'numeric':
            assert feedback['metric'] == 'absolute_error' and feedback['score'] == pytest.approx(0.3)
        else:
            assert feedback['metric'] == 'pairwise_ordering_errors'
            assert feedback['score'] == len(actual) * (len(actual)-1) // 2
        assert 'brier' not in str(feedback)


@pytest.mark.parametrize('status', ['unresolved', 'unavailable'])
def test_uncertain_observation_can_complete_without_invented_value_or_score(lesson, status):
    case, config, committed = setup_case(lesson)
    observed = lesson.record_observation(committed, config, None,
        {k: 'unavailable: raw values not published' for k in case['controls']},
        'historical_reference', 'Figure does not resolve this quantity.', observation_status=status)
    completed = lesson.reflect(observed, 'Evidence cannot distinguish the explanations.',
                               'Use an available resolved quantity next.', True, 0.5,
                               'No result inferred from missing evidence.')
    assert lesson.loads_record(lesson.dumps_record(completed)) == completed
    assert lesson.prediction_feedback(completed)['score'] is None
    assert completed['observation']['value'] is None
    assert status in lesson.forecast_card(completed)
    assert 'forecast_only' in lesson.forecast_card(completed)
    with pytest.raises(ValueError, match='invented'):
        lesson.record_observation(committed, config, 0.5,
            {k: 'unavailable' for k in case['controls']}, 'historical_reference', 'figure',
            observation_status=status)


def test_approximate_figure_range_has_error_bounds_without_point_score(lesson):
    case, config, committed = setup_case(lesson)
    observed = lesson.record_observation(committed, config, None,
        {k: 'approximately read from figure' for k in case['controls']},
        'historical_reference', 'Figure axis reading, raw data unavailable.',
        observation_status='approximately_read', interval=[0.46, 0.53])
    feedback = lesson.prediction_feedback(observed)
    assert feedback['score'] is None
    assert feedback['absolute_error_bounds'] == pytest.approx([0, 0.04])
    assert lesson.loads_record(lesson.dumps_record(observed)) == observed
    tampered = deepcopy(observed)
    tampered['observation']['interval'][0] = 0.1
    with pytest.raises(ValueError, match='seal'):
        lesson.validate_record(tampered)
    for value, interval in [(None, None), (0.3, [0.46, 0.53]), (None, [0.7, 0.2]),
                             (None, [False, 1]), (None, [0.2, float('inf')])]:
        with pytest.raises(ValueError):
            lesson.record_observation(committed, config, value,
                {k: 'figure' for k in case['controls']}, 'historical_reference', 'figure',
                observation_status='approximately_read', interval=interval)


def test_worked_cycle_and_readable_cards_precede_case_source_metadata(lesson, capsys):
    cells = lesson.build_cells()
    assert f"{len(lesson.load_cases())} guided extension cases" in cells[0]['source']
    worked = next(c for c in cells if c['id'] == 'worked-cycle')
    namespace = {}
    exec(compile(worked['source'], '<exposed-worked-example>', 'exec'), namespace)
    assert namespace['observed'] == 2 and namespace['control'] == 4
    assert namespace['changed_observed'] == namespace['revised_forecast'] == 3
    assert 'changed-rate' in capsys.readouterr().out
    assert next(i for i,c in enumerate(cells) if c['id']=='worked-cycle') < next(
        i for i,c in enumerate(cells) if c['metadata'].get('section'))
    for case in lesson.load_cases():
        prompt = next(c['source'] for c in cells if c['metadata'].get('section') == case['section'])
        assert prompt.index('**Forecast') < prompt.index('exact pre-experiment freeze')
        details = prompt.index('<details><summary>Help: exact pre-experiment freeze')
        assert details < prompt.index('Freeze these exact inputs') < prompt.index('Evidence:')
    _, _, record = setup_case(lesson)
    card = lesson.forecast_card(record)
    assert record['reasoning'] in card and 'Forecast:' in card
    assert 'seal' not in card and 'sha256' not in card


def test_source_or_helper_change_rejects_observation_but_keeps_legacy_readable(lesson, monkeypatch):
    case, config, committed = setup_case(lesson)
    monkeypatch.setattr(lesson, 'source_digest', lambda path: 'f' * 64)
    changed = lesson.freeze_inputs(case, config['inputs'])
    with pytest.raises(ValueError, match='configuration changed'):
        observe(lesson, case, changed, committed)
    assert lesson.loads_record(lesson.dumps_record(committed)) == committed
    changed = deepcopy(config)
    changed['workbook_helper_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='configuration changed'):
        observe(lesson, case, changed, committed)


def test_approximate_binary_and_ranking_have_no_probability_or_order_score(lesson):
    for kind in ('binary', 'ranking'):
        section = next(c['section'] for c in lesson.load_cases() if c['prediction_type'] == kind)
        case, config, committed = setup_case(lesson, section)
        observed = lesson.record_observation(committed, config, committed['prediction'],
            {k: 'approximately_read' for k in case['controls']},
            'historical_reference', 'Approximate visual impression, not resolved measurements.',
            observation_status='approximately_read')
        assert lesson.prediction_feedback(observed)['score'] is None
        assert lesson.loads_record(lesson.dumps_record(observed)) == observed
        with pytest.raises(ValueError, match='numeric interval'):
            lesson.record_observation(committed, config, None,
                {k: 'figure' for k in case['controls']}, 'historical_reference', 'figure',
                observation_status='approximately_read', interval=[0, 1])


def test_actual_legacy_transfer_requires_fresh_v2_confidence_in_documented_route(lesson):
    """Execute the public route using the exact captured pre-v2 sealed fixture."""
    import hashlib

    fixture = ROOT / 'tests/mastery/fixtures/extension_prediction_v1.json'
    original_bytes = fixture.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == '4e995bc478eb743924375ab1028753ce83257823bf0b9dadbb7be3784deee41b'
    legacy = lesson.loads_record(original_bytes.decode())
    assert legacy['schema_version'] == 1
    assert legacy['transfer']['confidence'] == 0.6
    assert legacy['case']['transfer_prediction_type'] == 'binary'
    assert lesson.dumps_record(legacy).encode() == original_bytes
    route = next(c['source'] for c in lesson.build_cells() if c['id'] == 'transfer-route')
    source = route.split('```python\n', 1)[1].split('```', 1)[0]
    # The fixture was captured for section 0.6; the public example can choose any completed section.
    source = source.replace("records['7.1']", "records[legacy['case']['section']]")
    source = source.replace("{field: None for field in tcase['config_fields']}",
                            "{field: 'fresh exact synthetic transfer input' for field in tcase['config_fields']}")
    namespace = {'worksheet': lesson, 'legacy': legacy,
                 'records': {legacy['case']['section']: legacy}, 'show_record': lambda record: None}
    with pytest.raises(NotImplementedError, match='Judge/reaffirm confidence'):
        exec(compile(source, '<documented-transfer-route>', 'exec'), namespace)
    assert 'tforecast' not in namespace  # The existing legacy 0.6 cannot unlock the new commitment.
    assert "original['transfer']['confidence']" not in source
    source = source.replace('fresh_transfer_confidence = None', 'fresh_transfer_confidence = 0.9')
    exec(compile(source, '<reaffirmed-transfer-route>', 'exec'), namespace)
    transfer = namespace['tforecast']
    assert transfer['schema_version'] == 2 and transfer['confidence'] == 0.9
    assert transfer['prediction'] == legacy['transfer']['prediction']
    measured = lesson.record_observation(transfer, namespace['tconfig'], transfer['prediction'],
        {key: 'synthetic control only' for key in transfer['case']['controls']},
        'learner_run', 'Synthetic confidence-semantics regression; no target experiment.')
    assert lesson.prediction_feedback(measured)['metric'] == 'selected_event_brier'
    assert lesson.prediction_feedback(measured)['score'] == pytest.approx(0.01)
    assert lesson.dumps_record(legacy).encode() == original_bytes
    assert fixture.read_bytes() == original_bytes
