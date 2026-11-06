"""Strict, private local JSON records. No network, executable imports, or implicit export."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .challenges import expose_case, json_value, validate_attempt, validate_fingerprint, validate_identifier

MAX_BYTES = 1024 * 1024
RECORD_KEYS = frozenset({'schema_version', 'unit_id', 'challenge_id', 'attempt_id', 'config',
    'config_fingerprint', 'source_revision', 'code_revision', 'seeds', 'evidence_class', 'attempt',
    'control_outcomes', 'next_action', 'reflection'})
SESSION_KEYS = frozenset({'schema_version', 'config', 'current_attempt', 'history', 'exposed_cases', 'next_action', 'mode', 'reflections'})


def _canonical(value) -> bytes:
    json_value(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError('invalid strict JSON value') from exc


def fingerprint(config: dict[str, object]) -> str:
    """SHA-256 of sorted, compact UTF-8 JSON; include simulator_version in saved configs."""
    if type(config) is not dict:
        raise ValueError('config must be a JSON object')
    return hashlib.sha256(_canonical(config)).hexdigest()


def _config(config):
    from .quadratic import QuadraticConfig
    if type(config) is not dict or set(config) != {f.name for f in fields(QuadraticConfig)} | {'simulator_version'}:
        raise ValueError('config has missing or unknown scientific fields')
    if type(config['simulator_version']) is not str or not config['simulator_version']:
        raise ValueError('simulator version must be explicit')
    if type(config['x0']) is not list:
        raise ValueError('config x0 must be a JSON list')
    try:
        return QuadraticConfig(**{k:v for k,v in config.items() if k != 'simulator_version'})
    except (ValueError, TypeError) as exc:
        raise ValueError(f'invalid scientific config: {exc}') from exc


def _schema(value, keys, label):
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f'{label} has missing or unknown fields')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('unsupported schema version')
    json_value(value)


def validate_record(record: dict[str, object]) -> dict[str, object]:
    _schema(record, RECORD_KEYS, 'record')
    for name in ('unit_id', 'challenge_id', 'attempt_id'):
        validate_identifier(record[name])
    config = _config(record['config'])
    validate_fingerprint(record['config_fingerprint'])
    if record['config_fingerprint'] != fingerprint(record['config']):
        raise ValueError('configuration fingerprint does not match scientific inputs')
    if record['evidence_class'] != 'analytic_toy':
        raise ValueError('pilot supports only analytic_toy evidence class')
    for name in ('source_revision', 'code_revision', 'next_action', 'reflection'):
        if type(record[name]) is not str:
            raise ValueError(f'{name} must be text')
    if not record['source_revision'].strip() or not record['code_revision'].strip():
        raise ValueError('source and code revisions must be explicit')
    if type(record['seeds']) is not list or any(type(s) is not int for s in record['seeds']) or record['seeds'] != [config.seed]:
        raise ValueError('seeds must match the scientific config seed')
    if type(record['control_outcomes']) is not dict:
        raise ValueError('control_outcomes must be a JSON object')
    attempt = validate_attempt(record['attempt'])
    if attempt['status'] != 'revealed':
        raise ValueError('portable record requires a completed revealed attempt')
    for name in ('challenge_id', 'attempt_id', 'config_fingerprint'):
        if attempt[name] != record[name]:
            raise ValueError(f'attempt {name} does not match envelope')
    return deepcopy(record)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _parse(data: bytes | str):
    try:
        if type(data) is str:
            data = data.encode('utf-8')
        if type(data) is not bytes:
            raise ValueError('strict JSON input must be bytes or text')
        if len(data) > MAX_BYTES:
            raise ValueError('JSON import exceeds 1 MiB limit')
        def invalid_constant(value):
            raise ValueError(f'nonfinite JSON constant: {value}')
        result = json.loads(data.decode('utf-8'), parse_constant=invalid_constant,
                            object_pairs_hook=_no_duplicates)
        json_value(result)
        return result
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise ValueError('invalid strict JSON input') from exc


def loads_attempt(data: bytes | str) -> dict[str, object]:
    return validate_record(_parse(data))


def _read(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink():
        raise ValueError('symlink record paths are not accepted')
    with path.open('rb') as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('JSON import exceeds 1 MiB limit')
    return data


def load_attempt(path: Path) -> dict[str, object]:
    return loads_attempt(_read(path))


def _atomic_json(value, path: Path, *, replace: bool = False) -> Path:
    data = _canonical(value) + b'\n'
    if len(data) > MAX_BYTES:
        raise ValueError('JSON record exceeds 1 MiB limit')
    path = Path(path)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError('symlink destinations are not accepted')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix='.mastery-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # Hard-link publication is atomic and refuses even a racing overwrite.
            os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def export_attempt(attempt: dict[str, object], destination: Path) -> Path:
    """Explicit local export to a chosen directory; filename is never caller input."""
    record = validate_record(attempt)
    filename = f"{record['challenge_id']}-{record['attempt_id']}.json"
    return _atomic_json(record, Path(destination) / filename)


def replay(record: dict[str, object]):
    """Recompute the analytic toy; this is never a fresh real-model training run."""
    from .quadratic import SIMULATOR_VERSION, simulate
    record = validate_record(record)
    if record['config']['simulator_version'] != SIMULATOR_VERSION:
        raise ValueError('simulator version mismatch; cannot silently replay this record')
    return simulate(_config(record['config']))


def default_data_dir() -> Path:
    """Resolve a private location without creating it or writing to the checkout."""
    override = os.environ.get('ARENA_MASTERY_DATA_DIR')
    if override:
        path = Path(override).expanduser()
    else:
        base = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share').expanduser()
        path = base / 'arena-mastery'
    if not path.is_absolute():
        raise ValueError('private data directory must be absolute')
    return path


def empty_session() -> dict:
    return dict(schema_version=1, config=None, current_attempt=None, history=[], exposed_cases=[], next_action='', mode='Explore', reflections={})


def validate_session(session: dict) -> dict:
    _schema(session, SESSION_KEYS, 'session')
    if type(session['next_action']) is not str or type(session['history']) is not list or type(session['exposed_cases']) is not list:
        raise ValueError('invalid session text/history/exposed cases')
    if session['mode'] not in ('Explore', 'Challenge') or type(session['reflections']) is not dict:
        raise ValueError('invalid session mode or reflections')
    if session['config'] is not None:
        _config(session['config'])
    current = session['current_attempt']
    if current is not None:
        validate_attempt(current)
        if session['config'] is None or current['config_fingerprint'] != fingerprint(session['config']):
            raise ValueError('pending attempt configuration does not match session')
    seen_ids = set()
    for record in session['history']:
        validate_record(record)
        if record['attempt']['status'] != 'revealed':
            raise ValueError('completed history must contain revealed attempts')
        if record['attempt_id'] in seen_ids:
            raise ValueError('duplicate attempt in completed history')
        seen_ids.add(record['attempt_id'])
    if current is not None and current['status'] == 'revealed':
        completed = next((record for record in session['history'] if record['attempt_id'] == current['attempt_id']), None)
        if completed is None or _canonical(completed['attempt']) != _canonical(current):
            raise ValueError('revealed current attempt must match its completed history envelope exactly')
    if any(key not in seen_ids or type(value) is not str for key, value in session['reflections'].items()):
        raise ValueError('reflections require a completed history attempt ID and text')
    exposures = []
    for pair in session['exposed_cases']:
        if type(pair) is not dict or set(pair) != {'challenge_id', 'config_fingerprint'}:
            raise ValueError('invalid exposed case')
        exposures = expose_case(exposures, pair['challenge_id'], pair['config_fingerprint'])
    if exposures != session['exposed_cases']:
        raise ValueError('duplicate exposed cases')
    for record in session['history']:
        pair = {'challenge_id': record['challenge_id'], 'config_fingerprint': record['config_fingerprint']}
        if pair not in exposures:
            raise ValueError('revealed history must retain case exposure')
    return deepcopy(session)


def _pending_transition(previous: dict, session: dict) -> None:
    """A reload/rerun may advance a commitment, never silently rewrite it."""
    old, new = previous['current_attempt'], session['current_attempt']
    if old is None:
        return
    same_id = new is not None and new['attempt_id'] == old['attempt_id']
    if same_id and old['status'] == 'revealed' and _canonical(old) != _canonical(new):
        raise ValueError('revealed current attempt is immutable')
    if same_id and old['assessment'] == 'practice' and new['assessment'] != 'practice':
        raise ValueError('practice assessment cannot become unseen')
    if old['status'] not in ('committed', 'invalidated') or old['confidence'] is None:
        return
    if not same_id:
        if old['status'] == 'committed' and (new is None or new['config_fingerprint'] == old['config_fingerprint']):
            raise ValueError('committed prediction cannot be erased or replaced on the same configuration; invalidate first')
        return
    fixed = ('attempt_id', 'challenge_id', 'config_fingerprint', 'prediction', 'confidence', 'reasoning')
    if _canonical({key: old[key] for key in fixed}) != _canonical({key: new[key] for key in fixed}):
        raise ValueError('committed prediction, confidence, reasoning and identity are immutable')
    allowed = ('invalidated',) if old['status'] == 'invalidated' else ('committed', 'revealed', 'invalidated')
    if new['status'] not in allowed:
        raise ValueError('committed prediction cannot return to an earlier state')
    if old['assessment'] == 'unseen' and new['assessment'] == 'practice':
        pair = {'challenge_id': old['challenge_id'], 'config_fingerprint': old['config_fingerprint']}
        if pair not in session['exposed_cases']:
            raise ValueError('practice reassessment requires a retained case exposure')


def save_session(session: dict, directory: Path | None = None) -> Path:
    validated = validate_session(session)
    target = Path(directory if directory is not None else default_data_dir())
    previous = load_session(target)
    _pending_transition(previous, validated)
    existing = {record['attempt_id']: record for record in validated['history']}
    if any(existing.get(record['attempt_id']) != record for record in previous['history']):
        raise ValueError('completed history is immutable; prior records must be retained')
    if any(pair not in validated['exposed_cases'] for pair in previous['exposed_cases']):
        raise ValueError('previous case exposures must be retained')
    return _atomic_json(validated, target / 'session.json', replace=True)


def load_session(directory: Path | None = None) -> dict:
    path = Path(directory if directory is not None else default_data_dir()) / 'session.json'
    if not path.exists():
        return empty_session()
    return validate_session(_parse(_read(path)))
