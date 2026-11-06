"""Strict local JSON persistence and reproducible analytic-toy replay."""
from copy import deepcopy
from dataclasses import asdict
import json
import socket

import numpy as np
import pytest

from arena_ext.mastery.challenges import commit_prediction, expose_case, new_attempt, reveal
from arena_ext.mastery.records import (
    default_data_dir, empty_session, export_attempt, fingerprint, load_attempt, load_session,
    replay, save_session, validate_record,
)
from arena_ext.mastery.quadratic import QuadraticConfig, SIMULATOR_VERSION, simulate



def example_record():
    config = asdict(QuadraticConfig(noise_std=.1, seed=12))
    config['x0'] = list(config['x0'])
    config['simulator_version'] = SIMULATOR_VERSION
    fp = fingerprint(config)
    attempt = reveal(commit_prediction(new_attempt('stable', fp), True, .8, 'Small step'), fp, True)
    return dict(schema_version=1, unit_id='u02_optimization', challenge_id='stable',
                attempt_id=attempt['attempt_id'], config=config, config_fingerprint=fp,
                source_revision='local-analytic-recurrence-v1', code_revision='abc123', seeds=[12],
                evidence_class='analytic_toy', attempt=attempt, control_outcomes={'eta_zero':'fixed'},
                next_action='Try a larger step', reflection='Noise causes fluctuations')


def test_key_order_independent_fingerprints_and_scientific_changes():
    config = example_record()['config']
    assert fingerprint(config) == fingerprint(dict(reversed(list(config.items()))))
    for field in config:
        changed = deepcopy(config)
        changed[field] = [2., 3.] if field == 'x0' else 'other' if field == 'simulator_version' else config[field] + 1
        assert fingerprint(changed) != fingerprint(config)


def test_json_round_trip_explicit_export_and_no_overwrite(tmp_path):
    record = example_record()
    path = export_attempt(record, tmp_path)
    assert path.name == f"stable-{record['attempt_id']}.json"
    assert load_attempt(path) == record
    with pytest.raises(FileExistsError):
        export_attempt(record, tmp_path)
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('field,value', [('schema_version',2), ('schema_version', True),
    ('evidence_class','pretend_real'), ('unit_id','../escape'), ('attempt_id','/escape')])
def test_reject_unknown_schema_evidence_and_traversal(tmp_path, field, value):
    record = example_record()
    record[field] = value
    with pytest.raises(ValueError):
        export_attempt(record, tmp_path)
    assert not list(tmp_path.iterdir())


def test_replay_matches_seed_and_config(tmp_path):
    record = example_record()
    loaded = load_attempt(export_attempt(record, tmp_path))
    config = dict(record['config'])
    config.pop('simulator_version')
    expected = simulate(QuadraticConfig(**config))
    actual = replay(loaded)
    np.testing.assert_allclose(actual.states, expected.states, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual.losses, expected.losses, atol=1e-12, rtol=0)
    other = deepcopy(record)
    other['config']['simulator_version'] = 'unknown-v2'
    other['config_fingerprint'] = fingerprint(other['config'])
    other['attempt']['config_fingerprint'] = other['config_fingerprint']
    with pytest.raises(ValueError, match='simulator version'):
        replay(other)


def test_reject_nonfinite_oversize_missing_unknown_and_executable(tmp_path):
    record = example_record()
    path = tmp_path / 'input.json'
    for raw in ('{"x":NaN}', '{"x":Infinity}', '__import__("os").system("false")', '\\x80pickle',
                json.dumps(record).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')):
        path.write_text(raw)
        with pytest.raises(ValueError):
            load_attempt(path)
    path.write_bytes(b' ' * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match='1 MiB'):
        load_attempt(path)
    for change in ('missing', 'unknown', 'nonfinite', 'mismatch'):
        bad = deepcopy(record)
        if change == 'missing':
            bad.pop('seeds')
        elif change == 'unknown':
            bad['script'] = 'exec this'
        elif change == 'nonfinite':
            bad['control_outcomes'] = {'loss':float('nan')}
        else:
            bad['config']['seed'] = 99
        with pytest.raises(ValueError):
            validate_record(bad)


def test_interrupted_write_does_not_publish_partial_file(tmp_path, monkeypatch):
    import arena_ext.mastery.records as records
    def fail(*args, **kwargs):
        raise OSError('interrupted before publish')
    monkeypatch.setattr(records.os, 'link', fail)
    with pytest.raises(OSError, match='interrupted'):
        export_attempt(example_record(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_persistence_and_exports_do_not_use_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network attempted')
    monkeypatch.setattr(socket, 'socket', forbidden)
    record = example_record()
    assert load_attempt(export_attempt(record, tmp_path)) == record
    replay(record)


def test_pending_session_survives_reload_with_exposures_and_history(tmp_path):
    completed = example_record()
    config = completed['config']
    fp = fingerprint(config)
    exposed = expose_case([], 'stable', fp)
    pending = commit_prediction(new_attempt('stable', fp, exposed_cases=exposed), False, .9, 'Retry practice')
    session = dict(schema_version=1, config=config, current_attempt=pending,
                   history=[completed], exposed_cases=exposed, next_action='Reveal my prediction', mode='Challenge', reflections={})
    assert load_session(tmp_path)['current_attempt'] is None
    save_session(session, tmp_path)
    loaded = load_session(tmp_path)
    assert loaded == session
    assert loaded['current_attempt']['assessment'] == 'practice'
    session['next_action'] = 'Changed action'
    save_session(session, tmp_path)
    assert load_session(tmp_path)['next_action'] == 'Changed action'
    assert (tmp_path / 'session.json').stat().st_mode & 0o777 == 0o600


def test_session_rejects_unknown_fields_inconsistent_config_and_corruption(tmp_path):
    session = load_session(tmp_path)
    session['unknown'] = 'x'
    with pytest.raises(ValueError):
        save_session(session, tmp_path)
    (tmp_path / 'session.json').write_text('{broken')
    with pytest.raises(ValueError):
        load_session(tmp_path)


def test_private_data_directory_is_outside_checkout_and_environment_override(tmp_path, monkeypatch):
    monkeypatch.delenv('ARENA_MASTERY_DATA_DIR', raising=False)
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    assert default_data_dir() == tmp_path / 'data' / 'arena-mastery'
    monkeypatch.setenv('ARENA_MASTERY_DATA_DIR', str(tmp_path / 'isolated'))
    assert default_data_dir() == tmp_path / 'isolated'
    assert not (tmp_path / 'isolated').exists()


def test_uploaded_bytes_use_same_strict_record_validation():
    from arena_ext.mastery.records import loads_attempt
    record = example_record()
    assert loads_attempt(json.dumps(record).encode()) == record
    with pytest.raises(ValueError, match='1 MiB'):
        loads_attempt(b' ' * (1024 * 1024 + 1))


def test_session_updates_preserve_completed_history_and_exposures(tmp_path):
    record = example_record()
    session = dict(schema_version=1, config=record['config'], current_attempt=None,
        history=[record], exposed_cases=expose_case([], record['challenge_id'], record['config_fingerprint']), next_action='', mode='Explore', reflections={})
    save_session(session, tmp_path)
    modified = deepcopy(session)
    modified['history'][0]['reflection'] = 'overwrite immutable history'
    with pytest.raises(ValueError, match='history'):
        save_session(modified, tmp_path)
    modified = deepcopy(session)
    modified['history'] = []
    modified['exposed_cases'] = []
    with pytest.raises(ValueError):
        save_session(modified, tmp_path)
    assert load_session(tmp_path) == session


def test_interrupted_session_write_preserves_previous_session(tmp_path, monkeypatch):
    import arena_ext.mastery.records as records
    initial = load_session(tmp_path)
    save_session(initial, tmp_path)
    changed = deepcopy(initial)
    changed['next_action'] = 'new action'
    def fail(*args, **kwargs):
        raise OSError('interrupted session write')
    monkeypatch.setattr(records.os, 'replace', fail)
    with pytest.raises(OSError):
        save_session(changed, tmp_path)
    assert load_session(tmp_path) == initial
    assert len(list(tmp_path.iterdir())) == 1


def test_mode_and_reflections_survive_session_reload(tmp_path):
    record = example_record()
    session = load_session(tmp_path)
    session.update(mode='Challenge', history=[record],
        exposed_cases=expose_case([], record['challenge_id'], record['config_fingerprint']),
        reflections={record['attempt_id']: 'Revised explanation'})
    save_session(session, tmp_path)
    assert load_session(tmp_path) == session
    for field, value in (('mode','Wrong'), ('reflections',{'unknown':'Note'}), ('reflections',{record['attempt_id']: True})):
        bad = deepcopy(session)
        bad[field] = value
        with pytest.raises(ValueError):
            save_session(bad, tmp_path)


def pending_session():
    session = empty_session()
    config = example_record()['config']
    session.update(config=config, mode='Challenge', current_attempt=commit_prediction(
        new_attempt('stable', fingerprint(config)), True, .8, 'Original causal explanation'))
    return session


@pytest.mark.parametrize('field,value', [('prediction',False), ('confidence',.99),
    ('reasoning','Changed explanation'), ('attempt_id','different'), ('challenge_id','different')])
def test_saved_commitment_cannot_be_overwritten(tmp_path, field, value):
    original = pending_session()
    save_session(original, tmp_path)
    changed = deepcopy(original)
    changed['current_attempt'][field] = value
    with pytest.raises(ValueError, match='committed'):
        save_session(changed, tmp_path)
    assert load_session(tmp_path) == original


@pytest.mark.parametrize('replacement', ['erase', 'restart', 'uncommit'])
def test_saved_commitment_cannot_be_erased_on_same_config(tmp_path, replacement):
    original = pending_session()
    save_session(original, tmp_path)
    changed = deepcopy(original)
    if replacement == 'erase':
        changed['current_attempt'] = None
    elif replacement == 'restart':
        changed['current_attempt'] = new_attempt('stable', fingerprint(changed['config']))
    else:
        changed['current_attempt'] = new_attempt('stable', fingerprint(changed['config']))
        changed['current_attempt']['attempt_id'] = original['current_attempt']['attempt_id']
    with pytest.raises(ValueError, match='committed'):
        save_session(changed, tmp_path)
    assert load_session(tmp_path) == original


def test_saved_commitment_allows_reveal_and_one_way_practice_exposure(tmp_path):
    original = pending_session()
    save_session(original, tmp_path)
    changed = deepcopy(original)
    changed['mode'] = 'Explore'
    changed['current_attempt']['assessment'] = 'practice'
    changed['exposed_cases'] = expose_case([], 'stable', fingerprint(changed['config']))
    save_session(changed, tmp_path)
    promoted = deepcopy(changed)
    promoted['current_attempt']['assessment'] = 'unseen'
    with pytest.raises(ValueError, match='practice'):
        save_session(promoted, tmp_path)
    changed['current_attempt'] = reveal(changed['current_attempt'], fingerprint(changed['config']), True)
    record = example_record()
    record.update(attempt_id=changed['current_attempt']['attempt_id'], attempt=deepcopy(changed['current_attempt']))
    changed['history'].append(record)
    save_session(changed, tmp_path)
    assert load_session(tmp_path)['current_attempt']['status'] == 'revealed'


def test_saved_commitment_new_config_and_explicit_invalidation_retry(tmp_path):
    original = pending_session()
    save_session(original, tmp_path)
    changed = deepcopy(original)
    changed['config']['eta'] += .01
    changed['current_attempt'] = new_attempt('stable', fingerprint(changed['config']))
    save_session(changed, tmp_path)
    assert load_session(tmp_path) == changed
    save_session(original, tmp_path)
    invalidated = deepcopy(original)
    invalidated['current_attempt']['status'] = 'invalidated'
    save_session(invalidated, tmp_path)
    retry = deepcopy(original)
    retry['current_attempt'] = new_attempt('stable', fingerprint(original['config']))
    save_session(retry, tmp_path)
    assert load_session(tmp_path) == retry


@pytest.mark.parametrize('status', ['uncommitted', 'committed', 'invalidated'])
def test_portable_records_require_completed_revealed_state(tmp_path, status):
    from arena_ext.mastery.records import loads_attempt
    record = example_record()
    attempt = new_attempt(record['challenge_id'], record['config_fingerprint'])
    attempt['attempt_id'] = record['attempt_id']
    if status != 'uncommitted':
        attempt = commit_prediction(attempt, True, .8, 'Pending explanation')
        attempt['status'] = status
    record['attempt'] = attempt
    with pytest.raises(ValueError, match='revealed'):
        validate_record(record)
    with pytest.raises(ValueError, match='revealed'):
        export_attempt(record, tmp_path)
    with pytest.raises(ValueError, match='revealed'):
        loads_attempt(json.dumps(record))
    path = tmp_path / 'pending.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='revealed'):
        load_attempt(path)


def test_revealed_current_requires_exact_completed_history_envelope(tmp_path):
    session = pending_session()
    fp = fingerprint(session['config'])
    session['current_attempt'] = reveal(session['current_attempt'], fp, True)
    session['exposed_cases'] = expose_case([], 'stable', fp)
    with pytest.raises(ValueError, match='revealed.*history'):
        save_session(session, tmp_path)


def test_two_stage_reveal_cannot_rewrite_completed_current_or_history(tmp_path):
    original = pending_session()
    save_session(original, tmp_path)
    revealed = deepcopy(original)
    revealed['current_attempt'] = reveal(revealed['current_attempt'], fingerprint(revealed['config']), True)
    record = example_record()
    record.update(attempt_id=revealed['current_attempt']['attempt_id'], attempt=deepcopy(revealed['current_attempt']))
    revealed['history'].append(record)
    revealed['exposed_cases'] = expose_case([], 'stable', fingerprint(revealed['config']))
    save_session(revealed, tmp_path)
    changed = deepcopy(revealed)
    changed['current_attempt'].update(prediction=False, confidence=.99, reasoning='Changed after reveal', score=.9801)
    with pytest.raises(ValueError):
        save_session(changed, tmp_path)
    changed['history'][0]['attempt'] = deepcopy(changed['current_attempt'])
    with pytest.raises(ValueError):
        save_session(changed, tmp_path)
    assert load_session(tmp_path) == revealed
