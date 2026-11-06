"""Distinct tiny DEV fixtures; never construct or evaluate the public target."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / 'mastery_hub/units/clip_prediction'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lesson():
    assert (UNIT / 'lesson.py').is_file(), 'six-file companion implementation is missing'
    return load(UNIT / 'lesson.py', 'clip_dev_lesson')


def source():
    return '''def zero_shot_probabilities(logits, kind):
    if kind == "softmax":
        return logits.softmax(dim=-1)
    if kind == "sigmoid":
        return logits.sigmoid()
    raise ValueError(kind)
'''


def committed(lesson, prediction=-0.25):
    manual = {'prediction': prediction, 'confidence': 0.5,
              'reasoning': 'More candidates compete for row weight.', 'revision': ''}
    return lesson.commit_prediction(
        lesson.freeze_inputs(source()), source(), prediction, 0.5,
        manual['reasoning'], 'seen', manual_forecast=manual,
    )


def test_inert_source_lifecycle(lesson, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('learner/target execution during inert lifecycle')
    monkeypatch.setattr(lesson, '_execute', forbidden)
    text = committed(lesson)
    assert lesson.read_record(text)['status'] == 'committed'
    assert 'logits.softmax(dim=-1)' in lesson.reveal_methods(text)
    with pytest.raises(ValueError, match='BV03'):
        lesson.run_experiment(text, source())
    assert all(c.get('execution_count') is None and not c.get('outputs')
               for c in lesson.build_cells())


@pytest.mark.parametrize('value', [True, None, float('nan'), float('inf'), '1'])
def test_strict_numeric_forecast(lesson, value):
    with pytest.raises(ValueError):
        committed(lesson, value)


@pytest.mark.parametrize('value', [-1000, 0, 1000.25])
def test_unbounded_finite_forecast(lesson, value):
    assert lesson.read_record(committed(lesson, value))['commitment']['prediction'] == value


def test_source_inspection_never_evaluates_annotations_defaults(lesson):
    for text in ['def zero_shot_probabilities(logits, kind=print("bad")):\n    pass',
                 '@print("bad")\ndef zero_shot_probabilities(logits, kind):\n    pass',
                 'def zero_shot_probabilities(logits: print("bad"), kind):\n    pass']:
        with pytest.raises(ValueError):
            lesson.freeze_inputs(text)


def test_exposure_and_revision_are_explicit(lesson):
    identity = lesson.freeze_inputs(source())
    manual = {'prediction': 1, 'confidence': 0.5, 'reasoning': 'Initial forecast', 'revision': ''}
    for exposure in ['', 'not_seen', None]:
        with pytest.raises(ValueError):
            lesson.commit_prediction(identity, source(), 1, 0.5, manual['reasoning'], exposure,
                                     manual_forecast=manual)
    with pytest.raises(ValueError, match='revision'):
        lesson.commit_prediction(identity, source(), 2, 0.5, 'Reason', 'seen',
                                 manual_forecast=manual)


def test_duplicate_keys_and_resealed_type_corruption(lesson):
    text = committed(lesson)
    with pytest.raises(ValueError):
        lesson.read_record(text.replace('"version":1', '"version":1,"version":1'))
    record = json.loads(text)
    record['version'] = True
    with pytest.raises(ValueError):
        lesson.read_record(lesson._seal(record))


def dev_inputs():
    import torch as t
    t.set_num_threads(1)
    base = t.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=t.float64)
    expanded = t.tensor([[0.0, 1.0, -2.0], [2.0, -1.0, 0.5]], dtype=t.float64)
    return base, expanded, ['a', 'b'], ['a', 'b', 'c'], ['a', 'b'], [2, 0, 1]


def dev_collect(lesson, callback):
    base, expanded, ids, expanded_ids, selected, permutation = dev_inputs()
    return lesson._collect(callback, base, expanded, ['u', 'v'], ids,
                           expanded_ids, selected, permutation, 1.0)


def test_dev_positive_and_all_twelve_calls(lesson):
    calls = []
    def callback(x, kind):
        calls.append((tuple(x.shape), kind))
        return x.softmax(-1) if kind == 'softmax' else x.sigmoid()
    result = dev_collect(lesson, callback)
    assert len(calls) == 12
    assert result['delta'] < 0
    assert result['sigmoid_delta'] == 0
    assert result['row_ids'] == ['u', 'v']
    assert len(result['conditions']) == 6


@pytest.mark.parametrize('bad', ['axis', 'sigmoid_normalized', 'nan', 'shape', 'bool', 'dtype'])
def test_dev_rejects_wrong_implementations(lesson, bad):
    import torch as t
    def callback(x, kind):
        out = x.softmax(-1) if kind == 'softmax' else x.sigmoid()
        if bad == 'axis' and kind == 'softmax':
            return x.softmax(0)
        if bad == 'sigmoid_normalized' and kind == 'sigmoid':
            return out / out.sum(-1, keepdim=True)
        if bad == 'nan':
            return t.full_like(out, float('nan'))
        if bad == 'shape':
            return out[:, :1]
        if bad == 'bool':
            return out > 0
        if bad == 'dtype':
            return out.float()
        return out
    with pytest.raises(lesson.CohortFailure):
        dev_collect(lesson, callback)


def test_dev_input_mutation_and_late_failure(lesson):
    def mutate(x, kind):
        x[0, 0] += 1
        return x.softmax(-1)
    with pytest.raises(lesson.CohortFailure, match='mutated'):
        dev_collect(lesson, mutate)
    calls = []
    def late(x, kind):
        calls.append(kind)
        if len(calls) == 12:
            raise RuntimeError('last condition failed')
        return x.softmax(-1) if kind == 'softmax' else x.sigmoid()
    with pytest.raises(lesson.CohortFailure) as failure:
        dev_collect(lesson, late)
    assert len(calls) == 12
    assert failure.value.details['condition'] == 'shifted_expansion'
    assert 'delta' not in failure.value.details
    assert failure.value.details['row_ids'] == ['u', 'v']


def test_dev_reused_buffers_are_cloned(lesson):
    import torch as t
    buffers = {}
    def callback(x, kind):
        out = x.softmax(-1) if kind == 'softmax' else x.sigmoid()
        buffer = buffers.setdefault(tuple(x.shape), t.empty_like(x))
        buffer.copy_(out)
        return buffer
    result = dev_collect(lesson, callback)
    baseline = deepcopy(result['conditions'][0]['outputs'])
    for buffer in buffers.values():
        buffer.fill_(99)
    assert result['conditions'][0]['outputs'] == baseline


def test_attempt_anchor_independent_of_commitment_and_home(lesson, tmp_path, monkeypatch):
    lesson.SESSION_ANCHOR = tmp_path / 'account'
    first = lesson._consume_attempt()
    monkeypatch.setenv('HOME', str(tmp_path / 'other'))
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'elsewhere'))
    with pytest.raises(FileExistsError):
        lesson._consume_attempt()
    assert first.exists()


def test_export_is_private_exclusive_and_outside_repo(lesson, tmp_path):
    text = committed(lesson)
    path = tmp_path / 'receipt.json'
    lesson.export_record(text, path)
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        lesson.export_record(text, path)
    with pytest.raises(ValueError):
        lesson.export_record(text, UNIT / 'forbidden.json')


def test_generation_preserves_edits_and_outputs(lesson, tmp_path):
    builder = load(ROOT / 'scripts/build_clip_prediction_notebook.py', 'clip_builder_dev')
    notebook = builder.make_notebook()
    path = tmp_path / 'exercise.ipynb'
    assert not builder.synchronize(path, notebook, check=True)
    assert builder.synchronize(path, notebook, check=False)
    notebook['cells'][0]['source'] += '\nmy work'
    path.write_text(json.dumps(notebook))
    before = path.read_bytes()
    assert not builder.synchronize(path, builder.make_notebook(), check=False)
    assert path.read_bytes() == before


def test_no_mean_until_all_conditions_and_controls_pass(lesson, monkeypatch):
    calls = []
    means = []
    original = lesson.math.fsum
    def spy(values):
        means.append(True)
        return original(values)
    monkeypatch.setattr(lesson.math, 'fsum', spy)
    def callback(x, kind):
        assert not means
        calls.append(kind)
        return x.softmax(-1) if kind == 'softmax' else x.sigmoid()
    dev_collect(lesson, callback)
    assert len(calls) == 12 and len(means) == 2
    means.clear()
    def wrong(x, kind):
        assert not means
        return x.softmax(-1)  # Deliberately normalize sigmoid.
    with pytest.raises(lesson.CohortFailure):
        dev_collect(lesson, wrong)
    assert means == []


def test_baseline_invalidity_blocks_every_callback(lesson):
    import torch as t
    base, expanded, ids, expanded_ids, selected, permutation = dev_inputs()
    expanded[1, 2] = float('nan')
    calls = []
    with pytest.raises(ValueError, match='finite'):
        lesson._collect(lambda *args: calls.append(args), base, expanded, ['u', 'v'],
                        ids, expanded_ids, selected, permutation, 1.0)
    assert calls == []
    # Malformed IDs are rejected before learner execution too.
    with pytest.raises(ValueError):
        lesson._collect(lambda *args: calls.append(args), base, base.clone(), ['u', 'u'],
                        ids, ids, ['a', 'b'], [0, 1], 1.0)
    assert calls == []


def test_failure_receipt_portable_and_attempt_not_reset(lesson, tmp_path, monkeypatch):
    # Explicit DEV-only orchestration sentinel: never enter the public target body.
    lesson.SESSION_ANCHOR = tmp_path / 'private-account'
    def sentinel(record):
        raise RuntimeError('DEV sentinel failed before any target construction')
    monkeypatch.setattr(lesson, '_execute', sentinel)
    def forbidden(*args):
        raise AssertionError('DEV forbids public collector')
    monkeypatch.setattr(lesson, '_collect', forbidden)
    assert lesson._execute is sentinel and lesson._collect is forbidden
    monkeypatch.setattr(lesson, 'TARGET_EXECUTION_AUTHORIZED', True)
    first = committed(lesson)
    with pytest.raises(lesson.AttemptFailure) as failed:
        lesson.run_experiment(first, source())
    retained = failed.value.receipt
    assert lesson.read_record(retained)['status'] == 'failed'
    assert 'DEV sentinel' in lesson.replay(retained)['message']
    assert len(list((tmp_path / 'private-account').rglob('*.receipt.json'))) == 1
    # A new forecast and export/HOME choices cannot reset the started attempt.
    second = committed(lesson, 1.25)
    monkeypatch.setenv('HOME', str(tmp_path / 'alternate-home'))
    with pytest.raises(FileExistsError):
        lesson.run_experiment(second, source())
    def drift():
        raise AssertionError('portable record consulted current source')
    monkeypatch.setattr(lesson, '_snapshots', drift)
    monkeypatch.setattr(lesson, 'runtime_identity', drift)
    assert lesson.replay(retained)['kind'] == 'failed_cpu_attempt'
    lesson.export_record(retained, tmp_path / 'portable-failure.json')
    with pytest.raises(ValueError):
        lesson.plot_receipt(retained)
    with pytest.raises(ValueError):
        lesson.reflect(retained, 'No observation', 'New condition', 0, 0.5, 'Reason')


def test_source_drift_blocks_before_attempt_consumption(lesson, monkeypatch):
    record = committed(lesson)
    def target_forbidden(*args):
        raise AssertionError('DEV source drift must stop before public target')
    monkeypatch.setattr(lesson, '_execute', target_forbidden)
    monkeypatch.setattr(lesson, '_collect', target_forbidden)
    assert lesson._execute is target_forbidden and lesson._collect is target_forbidden
    monkeypatch.setattr(lesson, 'TARGET_EXECUTION_AUTHORIZED', True)
    changed = lesson.freeze_inputs(source())
    changed['runtime']['torch'] = 'changed-version'
    monkeypatch.setattr(lesson, 'freeze_inputs', lambda source: changed)
    def forbidden():
        raise AssertionError('drifted commitment consumed an attempt')
    monkeypatch.setattr(lesson, '_consume_attempt', forbidden)
    with pytest.raises(ValueError, match='drift'):
        lesson.run_experiment(record, source())


@pytest.mark.parametrize('change', ['outputs', 'count', 'trusted', 'widgets', 'edited'])
def test_generator_preserves_every_learner_work_kind(lesson, tmp_path, change):
    builder = load(ROOT / 'scripts/build_clip_prediction_notebook.py', 'clip_preservation_dev')
    notebook = builder.make_notebook()
    code = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
    if change == 'outputs':
        code['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': 'DEV saved work'}]
    elif change == 'count':
        code['execution_count'] = 1
    elif change == 'trusted':
        code['metadata']['trusted'] = False
    elif change == 'widgets':
        notebook['metadata']['widgets'] = {}
    else:
        code['source'] += '\n# learner edit'
    path = tmp_path / 'saved.ipynb'
    path.write_text(json.dumps(notebook))
    before = path.read_bytes()
    assert not builder.synchronize(path, builder.make_notebook(), check=False)
    assert not builder.synchronize(path, builder.make_notebook(), check=True)
    assert path.read_bytes() == before


def test_notebook_stale_success_reset_and_source_parse(lesson):
    import ast
    cells = {cell['id']: cell for cell in lesson.build_cells()}
    run = cells['clip-future-run']['source']
    assert ast.parse(run).body[0].targets[0].id == 'receipt'
    assert ast.parse(run).body[0].value.value is None
    assert 'lesson.run_experiment' in run
    for cell in cells.values():
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    assert 'REFERENCE' not in lesson._execute.__code__.co_names


def test_stored_dev_summary_tamper_and_signed_error(lesson):
    result = dev_collect(lesson, lambda x, kind: x.softmax(-1) if kind == 'softmax' else x.sigmoid())
    # Test the stored-output validator on the distinct two-row DEV cohort only.
    record = {'identity': {'protocol': {'row_ids': ['u', 'v'], 'base_ids': ['a', 'b'],
               'candidate_ids': ['a', 'b', 'c'], 'selected_ids': ['a', 'b'],
               'permutation': [2, 0, 1]}, 'runtime': {'DEV': 'fixture'}},
              'commitment': {'prediction': 0.125}}
    error = result['delta'] - record['commitment']['prediction']
    observation = {'kind': 'fresh_cpu_analytic', 'result': result, 'runtime': {'DEV': 'fixture'},
                   'signed_error': error, 'absolute_error': abs(error)}
    lesson._observation(observation, record)
    for mutate in ('sign', 'mean', 'row', 'bool'):
        changed = deepcopy(observation)
        if mutate == 'sign':
            changed['signed_error'] = -error
        elif mutate == 'mean':
            changed['result']['delta'] += 0.1
        elif mutate == 'row':
            changed['result']['row_ids'].pop()
        else:
            changed['absolute_error'] = True
        with pytest.raises(ValueError):
            lesson._observation(changed, record)


def test_inert_import_in_fresh_python_process():
    import subprocess
    import sys
    program = r'''import importlib.util, sys
from pathlib import Path
path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('inert_clip', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
module.build_cells()
source = 'def zero_shot_probabilities(logits, kind):\n    raise NotImplementedError()\n'
identity = module.freeze_inputs(source)
manual = {'prediction': 0.0, 'confidence': 0.5, 'reasoning': 'DEV source-only record', 'revision': ''}
record = module.commit_prediction(identity, source, 0.0, 0.5, manual['reasoning'],
                                  'seen', manual_forecast=manual)
module.read_record(record)
module.reveal_methods(record)
assert 'torch' not in sys.modules
assert module.TARGET_EXECUTION_AUTHORIZED is False
'''
    result = subprocess.run([sys.executable, '-B', '-c', program, str(UNIT / 'lesson.py')],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_invalid_return_captured_before_validation(lesson):
    import torch as t
    with pytest.raises(lesson.CohortFailure) as failed:
        dev_collect(lesson, lambda x, kind: t.full_like(x, float('nan')))
    capture = failed.value.details['attempted_return']
    assert capture['tag'] == 'tensor'
    assert capture['dtype'] == 'torch.float64'
    assert capture['shape'] == [2, 2]
    assert capture['nonfinite_count'] == 4
    assert capture['values'][0] == {'tag': 'nonfinite', 'value': 'nan'}


def test_mutate_then_raise_keeps_primary_and_postcheck(lesson):
    def broken(x, kind):
        x[0, 0] = 42
        raise RuntimeError('DEV PRIMARY mutation then raise')
    with pytest.raises(lesson.CohortFailure) as failed:
        dev_collect(lesson, broken)
    assert 'DEV PRIMARY' in str(failed.value)
    details = failed.value.details
    assert details['primary']['error_type'] == 'RuntimeError'
    assert details['secondary'][0]['error_type'] == 'ValueError'
    assert 'mutated input' in details['secondary'][0]['message']
    assert details['input_after']['sha256'] != details['input_before']['sha256']


@pytest.mark.parametrize('mutation', ['dtype', 'signed_zero'])
def test_exact_input_dtype_and_bytes(lesson, mutation):
    def broken(x, kind):
        saved = x.clone()
        if mutation == 'dtype':
            x.data = x.float()
        else:
            x[0, 0] = -0.0  # original DEV element is +0.0
        return saved.softmax(-1) if kind == 'softmax' else saved.sigmoid()
    with pytest.raises(lesson.CohortFailure, match='mutated input') as failed:
        dev_collect(lesson, broken)
    assert failed.value.details['input_before'] != failed.value.details['input_after']


def _pretarget_failure(lesson, monkeypatch, tmp_path, execute):
    # Install the entire-body replacement and guards BEFORE opening the local gate.
    assert lesson.TARGET_EXECUTION_AUTHORIZED is False
    def forbidden(*args, **kwargs):
        raise AssertionError('public collector/factory is forbidden in DEV sentinels')
    monkeypatch.setattr(lesson, '_execute', execute)
    monkeypatch.setattr(lesson, '_collect', forbidden)
    assert lesson._execute is execute and lesson._collect is forbidden
    lesson.SESSION_ANCHOR = tmp_path / 'account'
    record = committed(lesson)
    monkeypatch.setattr(lesson, 'TARGET_EXECUTION_AUTHORIZED', True)
    try:
        with pytest.raises(lesson.AttemptFailure) as failed:
            lesson.run_experiment(record, source())
        assert len(list((tmp_path / 'account').rglob('*.started.json'))) == 1
        return failed.value
    finally:
        lesson.TARGET_EXECUTION_AUTHORIZED = False


def test_nonfinite_failure_detail_seals_receipt(lesson, tmp_path, monkeypatch):
    def sentinel(record):
        raise lesson.CohortFailure('DEV PRIMARY NaN diagnostic', {'nan': float('nan')})
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    record = lesson.read_record(failed.receipt)
    assert record['status'] == 'failed'
    assert 'DEV PRIMARY' in record['observation']['message']
    diagnostic = record['observation']['details']['primary']['details']
    assert diagnostic['tag'] == 'mapping' and diagnostic['length'] == 1
    key, value = diagnostic['items'][0]
    assert key['value'] == 'nan'
    assert value == {'tag': 'nonfinite', 'value': 'nan'}
    assert len(list((tmp_path / 'account').rglob('*.receipt.json'))) == 1


def test_secondary_storage_failure_keeps_receipt_and_primary(lesson, tmp_path, monkeypatch):
    def sentinel(record):
        raise RuntimeError('DEV PRIMARY before target')
    def unwritable(*args):
        raise OSError('DEV SECONDARY storage failure')
    monkeypatch.setattr(lesson, '_exclusive', unwritable)
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    observation = lesson.replay(failed.receipt)
    assert 'DEV PRIMARY' in observation['message']
    assert observation['details']['persistence'] == 'memory_only'
    assert len(observation['details']['secondary']) == 2
    assert all('DEV SECONDARY' in item['message']
               for item in observation['details']['secondary'])
    assert not list((tmp_path / 'account').rglob('*.receipt.json'))
    with pytest.raises(FileExistsError):
        lesson._consume_attempt()


def test_unprintable_and_deep_exception_details_remain_bounded(lesson, tmp_path, monkeypatch):
    class Unprintable(RuntimeError):
        def __str__(self):
            raise ValueError('DEV error formatter failed')
    def sentinel(record):
        detail = {'huge': 'x' * 100_000, 'unsupported': object()}
        for _ in range(40):
            detail = {'nested': detail}
        exc = Unprintable()
        exc.details = detail
        raise exc
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    record = lesson.read_record(failed.receipt)
    assert record['status'] == 'failed'
    assert record['observation']['error_type'] == 'Unprintable'
    assert 'exception text unavailable' in record['observation']['message']
    primary = record['observation']['details']['primary']
    assert primary['message_capture']['items'][0][0]['value'] == 'formatting_failed'
    assert primary['message_capture']['items'][0][1]['value'] == 'ValueError'
    assert 'depth_limit' in lesson.canonical(primary['details'])
    assert len(failed.receipt) < lesson.LIMIT


@pytest.mark.parametrize('collision', ['file', 'symlink'])
def test_generator_refuses_late_creation(lesson, tmp_path, monkeypatch, collision):
    import os
    builder = load(ROOT / 'scripts/build_clip_prediction_notebook.py', 'clip_race_v2')
    path = tmp_path / 'late.ipynb'
    precious = tmp_path / 'precious.txt'
    precious.write_text('DEV precious learner work')
    original_open = os.open
    original_write = Path.write_text
    raced = []
    def open_with_collision(target, flags, *args, **kwargs):
        if Path(target) == path and flags & os.O_CREAT:
            raced.append(True)
            if collision == 'file':
                original_write(path, precious.read_text())
            else:
                path.symlink_to(precious)
        return original_open(target, flags, *args, **kwargs)
    def write_with_collision(target, data, *args, **kwargs):
        if target == path and not raced:
            raced.append(True)
            if collision == 'file':
                original_write(path, precious.read_text())
            else:
                path.symlink_to(precious)
        return original_write(target, data, *args, **kwargs)
    monkeypatch.setattr(os, 'open', open_with_collision)
    monkeypatch.setattr(Path, 'write_text', write_with_collision)
    assert not builder.synchronize(path, builder.make_notebook(), check=False)
    assert raced == [True]
    assert path.read_text() == 'DEV precious learner work'
    assert precious.read_text() == 'DEV precious learner work'
    assert path.is_symlink() == (collision == 'symlink')


def test_generator_refuses_outdated_pristine_bytes(lesson, tmp_path):
    builder = load(ROOT / 'scripts/build_clip_prediction_notebook.py', 'clip_old_v2')
    old = builder.make_notebook()
    old['metadata']['clip_prediction']['old_template'] = True
    old['metadata']['clip_prediction'][builder.BASELINE] = builder.fingerprint(old)
    path = tmp_path / 'old.ipynb'
    path.write_text(json.dumps(old))
    before = path.read_bytes()
    assert not builder.synchronize(path, builder.make_notebook(), check=False)
    assert path.read_bytes() == before


def test_storage_fallback_is_persisted_without_retrying_target(lesson, tmp_path, monkeypatch):
    calls = []
    original = lesson._exclusive
    def flaky(raw, path):
        calls.append(Path(path))
        if len(calls) == 1:
            raise OSError('DEV preferred storage failed')
        return original(raw, path)
    monkeypatch.setattr(lesson, '_exclusive', flaky)
    def sentinel(record):
        raise RuntimeError('DEV PRIMARY one attempt')
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    assert lesson.replay(failed.receipt)['details']['persistence'] == 'fallback_persisted'
    assert len(calls) == 2
    stored = list((tmp_path / 'account').rglob('*.fallback.receipt.json'))
    assert len(stored) == 1 and stored[0].read_text() == failed.receipt
    assert stored[0].stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('stage', ['context', 'pipe', 'start', 'recv', 'cleanup'])
def test_orchestration_failures_always_keep_receipt(lesson, tmp_path, monkeypatch, stage):
    class Connection:
        def __init__(self, child_side=False):
            self.child_side = child_side
            self.closes = 0
        def close(self):
            self.closes += 1
            if stage == 'cleanup' and (not self.child_side or self.closes > 1):
                raise OSError('DEV cleanup SECONDARY')
        def poll(self, seconds):
            assert seconds == 30
            return True
        def recv(self):
            if stage == 'recv':
                raise EOFError('DEV IPC PRIMARY')
            return {'ok': False, 'failure': lesson._error_info(RuntimeError('DEV worker PRIMARY'))}
    class Process:
        pid = None
        def start(self):
            if stage == 'start':
                raise RuntimeError('DEV startup PRIMARY')
            self.pid = 1
        def is_alive(self):
            return False
        def join(self, timeout):
            assert timeout == 2
    class Context:
        def Pipe(self, duplex):
            assert duplex is False
            if stage == 'pipe':
                raise OSError('DEV pipe PRIMARY')
            return Connection(), Connection(child_side=True)
        def Process(self, target, args):
            assert target is lesson._worker
            return Process()
    def context(kind):
        assert kind == 'fork'
        if stage == 'context':
            raise RuntimeError('DEV context PRIMARY')
        return Context()
    monkeypatch.setattr(lesson.multiprocessing, 'get_context', context)
    def sentinel(record):
        raise AssertionError('even the DEV execute sentinel must not run')
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    observation = lesson.replay(failed.receipt)
    assert 'PRIMARY' in observation['message'] or stage == 'cleanup'
    if stage == 'cleanup':
        assert 'worker PRIMARY' in observation['message']
        assert any('SECONDARY' in item['message'] for item in observation['details']['secondary'])


def test_sanitation_failure_has_valid_fallback(lesson, tmp_path, monkeypatch):
    def broken_capture(value, *args):
        raise ValueError('DEV sanitizer failure')
    def sentinel(record):
        raise lesson.CohortFailure('DEV PRIMARY survives sanitizer', {'x': object()})
    monkeypatch.setattr(lesson, '_diagnostic', broken_capture)
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    observation = lesson.replay(failed.receipt)
    assert 'DEV PRIMARY' in observation['message']
    diagnostic = observation['details']['primary']['details']
    assert diagnostic['tag'] == 'text'
    assert diagnostic['value'] == 'diagnostic_capture_failed: ValueError'


def test_preferred_failure_sealing_falls_back(lesson, tmp_path, monkeypatch):
    original = lesson._failure_receipt
    calls = []
    def reject_first(*args):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError('DEV preferred failure validation rejected')
        return original(*args)
    def sentinel(record):
        raise RuntimeError('DEV PRIMARY survives envelope fallback')
    monkeypatch.setattr(lesson, '_failure_receipt', reject_first)
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    assert 'DEV PRIMARY' in lesson.replay(failed.receipt)['message']
    observation = lesson.replay(failed.receipt)
    assert len(calls) == 2
    assert observation['details']['secondary'][0]['message'] == 'DEV preferred failure validation rejected'


def test_bounded_capture_discloses_omissions_and_unsupported_values(lesson):
    import torch as t
    captured = lesson._capture_output(t.zeros((2, 100), dtype=t.float64))
    assert len(captured['values']) == lesson.DIAGNOSTIC_ITEMS
    assert captured['truncated'] is True
    assert captured['nonfinite_count_scope'] == 'sample'
    class NoRepresentation:
        def __repr__(self):
            raise AssertionError('arbitrary repr must not run')
    captured = lesson._capture_output(NoRepresentation())
    assert captured == {'tag': 'unsupported', 'type': 'NoRepresentation'}
    diagnostic = lesson._safe_diagnostic({'huge': 'x' * 100_000,
                                         'unsupported': NoRepresentation(),
                                         'inf': float('inf')})
    lesson._validate_diagnostic(diagnostic)
    assert '100000' in lesson.canonical(diagnostic)
    assert 'unsupported' in lesson.canonical(diagnostic)
    assert len(lesson.canonical(diagnostic)) < 10_000


def test_failure_diagnostics_remain_strict_after_resealing(lesson, tmp_path, monkeypatch):
    def sentinel(record):
        raise RuntimeError('DEV PRIMARY validator fixture')
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    original = lesson.read_record(failed.receipt)
    for mutation in ('unknown', 'nonfinite', 'persistence', 'primary_mismatch'):
        changed = deepcopy(original)
        details = changed['observation']['details']
        if mutation == 'unknown':
            details['extra'] = True
        elif mutation == 'nonfinite':
            details['primary']['details'] = {'tag': 'scalar', 'value': float('nan')}
            with pytest.raises(ValueError):
                lesson._seal(changed)
            continue
        elif mutation == 'persistence':
            details['persistence'] = True
        else:
            details['primary']['message'] = 'different'
        with pytest.raises(ValueError):
            lesson.read_record(lesson._seal(changed))


def test_capture_failure_does_not_hide_validation_or_input_failure(lesson, monkeypatch):
    import torch as t
    def unavailable(output):
        raise RuntimeError('DEV SECONDARY capture failed')
    monkeypatch.setattr(lesson, '_capture_output', unavailable)
    def bad(x, kind):
        x[0, 0] = 20
        return t.full_like(x, float('nan'))
    with pytest.raises(lesson.CohortFailure) as failed:
        dev_collect(lesson, bad)
    assert 'nonfinite output' in str(failed.value)
    details = failed.value.details
    assert details['attempted_return']['tag'] == 'capture_failed'
    assert len(details['secondary']) == 2
    assert 'capture failed' in details['secondary'][0]['message']
    assert 'mutated input' in details['secondary'][1]['message']


def test_successful_capture_is_not_accepted_after_input_shape_drift(lesson):
    def bad(x, kind):
        saved = x.clone()
        x.resize_(1, x.numel())
        return saved.softmax(-1) if kind == 'softmax' else saved.sigmoid()
    with pytest.raises(lesson.CohortFailure, match='mutated input'):
        dev_collect(lesson, bad)


@pytest.mark.parametrize('field', ['details', 'message', 'type'])
@pytest.mark.parametrize('storage', ['normal', 'preferred_fails', 'both_fail'])
def test_surrogate_primary_failure_keeps_lossless_receipt(lesson, tmp_path, monkeypatch,
                                                        field, storage):
    bad = '\ud800\udc00|\\ud800|\U0001f642|\udfff'
    raw_message = 'DEV PRIMARY ' + (bad if field == 'message' else field)
    expected_type = 'Surrogate' + bad if field == 'type' else 'CohortFailure'
    # CPython rejects assigning a raw-surrogate __name__; a metaclass can expose
    # one during the diagnostic lookup without any source/target execution.
    class SurrogateName(type):
        def __getattribute__(cls, name):
            if name == '__name__':
                return expected_type
            return super().__getattribute__(name)
    class SurrogateError(RuntimeError, metaclass=SurrogateName):
        pass
    def sentinel(record):
        if field == 'type':
            raise SurrogateError(raw_message)
        raise lesson.CohortFailure(raw_message, {'text': bad} if field == 'details' else {})
    writes = []
    original = lesson._exclusive
    def storage_probe(raw, path):
        writes.append(Path(path))
        if storage == 'both_fail' or storage == 'preferred_fails' and len(writes) == 1:
            raise OSError('DEV SECONDARY storage \ud800')
        return original(raw, path)
    monkeypatch.setattr(lesson, '_exclusive', storage_probe)
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    failed.receipt.encode('utf-8', errors='strict')
    observation = lesson.replay(failed.receipt)
    primary = observation['details']['primary']
    expected_message = raw_message.encode('utf-8', errors='backslashreplace').decode('utf-8')
    assert observation['message'] == primary['message'] == expected_message
    if field == 'message':
        capture = primary['message_capture']
        assert capture['tag'] == 'codepoint_text'
        assert capture['codepoints'] == list(map(ord, raw_message))
        assert capture['length'] == len(raw_message) and capture['truncated'] is False
    elif field == 'type':
        capture = primary['error_type_capture']
        assert capture['tag'] == 'codepoint_text'
        assert capture['codepoints'] == list(map(ord, expected_type))
        assert primary['error_type'] == expected_type.encode('utf-8', 'backslashreplace').decode()
    else:
        capture = primary['details']['items'][0][1]
        assert capture['tag'] == 'codepoint_text'
        assert capture['codepoints'] == list(map(ord, bad))
        # A raw surrogate pair remains two distinct codepoints, unlike a JSON decode.
        assert capture['codepoints'][:2] == [0xd800, 0xdc00]
    persistence = {'normal': 'persisted', 'preferred_fails': 'fallback_persisted',
                   'both_fail': 'memory_only'}[storage]
    assert observation['details']['persistence'] == persistence
    assert len(list((tmp_path / 'account').rglob('*.receipt.json'))) == (storage != 'both_fail')
    assert len(observation['details']['secondary']) == {'normal': 0, 'preferred_fails': 1,
                                                       'both_fail': 2}[storage]
    for secondary in observation['details']['secondary']:
        assert secondary['message_capture']['tag'] == 'codepoint_text'
        assert 0xd800 in secondary['message_capture']['codepoints']
    with pytest.raises(FileExistsError):
        lesson._consume_attempt()


def test_surrogate_preferred_envelope_fallback_keeps_primary(lesson, tmp_path, monkeypatch):
    original = lesson._failure_receipt
    calls = []
    def reject_once(*args):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError('DEV SECONDARY preferred seal \udfff')
        return original(*args)
    monkeypatch.setattr(lesson, '_failure_receipt', reject_once)
    def sentinel(record):
        raise lesson.CohortFailure('DEV PRIMARY \ud800', {'raw': '\udc00'})
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    observation = lesson.replay(failed.receipt)
    assert observation['message'] == 'DEV PRIMARY \\ud800'
    assert observation['error_type'] == 'CohortFailure'
    assert observation['details']['secondary'][0]['message'] == 'DEV SECONDARY preferred seal \\udfff'
    assert observation['details']['secondary'][0]['message_capture']['codepoints'][-1] == 0xdfff
    # Original raw diagnostics stay encoded inside the fallback primary evidence.
    def codepoint_values(node):
        if type(node) is dict:
            if node.get('tag') == 'codepoint_text':
                yield node['codepoints']
            for value in node.values():
                yield from codepoint_values(value)
        elif type(node) is list:
            for value in node:
                yield from codepoint_values(value)
    assert any(0xdc00 in values for values in codepoint_values(observation))
    assert len(calls) == 2


@pytest.mark.parametrize('stage', ['context', 'recv', 'cleanup', 'rejected_response'])
def test_surrogate_orchestration_paths_retain_receipt(lesson, tmp_path, monkeypatch, stage):
    class Connection:
        def __init__(self, child_side=False):
            self.child_side = child_side
            self.closes = 0
        def close(self):
            self.closes += 1
            if stage == 'cleanup' and (not self.child_side or self.closes > 1):
                raise OSError('DEV SECONDARY cleanup \udfff')
        def poll(self, seconds):
            assert seconds == 30
            return True
        def recv(self):
            if stage == 'recv':
                raise EOFError('DEV PRIMARY IPC \ud800')
            if stage == 'rejected_response':
                # Malformed external worker data is rejected, then captured safely.
                info = lesson._error_info(RuntimeError('DEV worker'))
                info['message'] = 'DEV malformed worker \ud800'
                return {'ok': False, 'failure': info}
            return {'ok': False, 'failure': lesson._error_info(RuntimeError('DEV PRIMARY worker \ud800'))}
    class Process:
        pid = None
        def start(self):
            self.pid = 1
        def is_alive(self):
            return False
        def join(self, timeout):
            assert timeout == 2
    class Context:
        def Pipe(self, duplex):
            assert duplex is False
            return Connection(), Connection(child_side=True)
        def Process(self, target, args):
            assert target is lesson._worker
            return Process()
    def context(kind):
        assert kind == 'fork'
        if stage == 'context':
            raise RuntimeError('DEV PRIMARY context \ud800')
        return Context()
    monkeypatch.setattr(lesson.multiprocessing, 'get_context', context)
    def never(record):
        raise AssertionError('even sentinel execution forbidden')
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, never)
    observation = lesson.replay(failed.receipt)
    failed.receipt.encode('utf-8', errors='strict')
    if stage == 'rejected_response':
        assert observation['error_type'] == 'ValueError'
        assert observation['message'] == 'UTF-8 text required'
        captured = lesson.canonical(observation['details']['primary']['details'])
        assert '55296' in captured
    else:
        assert observation['message'].endswith('\\ud800')
        assert observation['details']['primary']['message_capture']['codepoints'][-1] == 0xd800
    if stage == 'cleanup':
        secondary = observation['details']['secondary']
        assert len(secondary) == 2
        assert all(item['message_capture']['codepoints'][-1] == 0xdfff for item in secondary)


def test_codepoint_diagnostic_preserves_bounds_and_utf8(lesson):
    raw = '\ud800' * 100_000
    diagnostic = lesson._diagnostic(raw)
    assert diagnostic == {'tag': 'codepoint_text', 'codepoints': [0xd800] * lesson.DIAGNOSTIC_TEXT,
                           'length': 100_000, 'truncated': True}
    lesson._validate_diagnostic(diagnostic)
    encoded = lesson.canonical(diagnostic).encode('utf-8', errors='strict')
    assert len(encoded) < 20_000
    normal = lesson._diagnostic('\U0001f642é')
    assert normal == {'tag': 'text', 'value': '\U0001f642é', 'length': 2, 'truncated': False}
    assert lesson._utf8_display('\U0001f642é') == '\U0001f642é'
    # The huge tail is deliberately unobserved and its extent remains explicit.
    truncated = lesson._diagnostic('a' * lesson.DIAGNOSTIC_TEXT + '\ud800')
    assert truncated['tag'] == 'text' and truncated['truncated'] is True
    assert truncated['length'] == lesson.DIAGNOSTIC_TEXT + 1


def test_surrogate_unsupported_type_is_tagged_losslessly(lesson):
    name = 'Opaque\ud800'
    class Named(type):
        def __getattribute__(cls, key):
            return name if key == '__name__' else super().__getattribute__(key)
    class Opaque(metaclass=Named):
        def __repr__(self):
            raise AssertionError('must not render arbitrary objects')
    diagnostic = lesson._diagnostic(Opaque())
    assert diagnostic['tag'] == 'unsupported_type'
    assert diagnostic['type'] == 'Opaque\\ud800'
    assert diagnostic['type_capture']['codepoints'] == list(map(ord, name))
    lesson._validate_diagnostic(diagnostic)
    lesson.canonical(diagnostic).encode('utf-8', errors='strict')


def test_codepoint_diagnostic_and_failure_summary_stay_strict(lesson):
    original = lesson._diagnostic('a\ud800')
    for field, value in [('codepoints', [True]), ('codepoints', [0x110000]),
                         ('codepoints', [-1]), ('codepoints', [97]),
                         ('length', True), ('length', 1), ('truncated', True)]:
        invalid = {**original, field: value}
        with pytest.raises(ValueError):
            lesson._validate_diagnostic(invalid)
    with pytest.raises(ValueError, match='UTF-8'):
        lesson._validate_diagnostic({'tag': 'text', 'value': '\ud800', 'length': 1,
                                     'truncated': False})
    info = lesson._error_info(RuntimeError('safe'))
    info['message'] = '\ud800'
    with pytest.raises(ValueError, match='UTF-8'):
        lesson._validate_error_info(info)
    info = lesson._error_info(RuntimeError('safe'))
    info['error_type'] = '\ud800'
    with pytest.raises(ValueError, match='UTF-8'):
        lesson._validate_error_info(info)


def test_surrogate_method_source_rejected_before_attempt(lesson, tmp_path, monkeypatch):
    lesson.SESSION_ANCHOR = tmp_path / 'account'
    def forbidden(*args):
        raise AssertionError('invalid source must not consume or dispatch')
    monkeypatch.setattr(lesson, '_consume_attempt', forbidden)
    monkeypatch.setattr(lesson, '_execute', forbidden)
    bad_source = 'def zero_shot_probabilities(logits, kind):\n    # \ud800\n    pass\n'
    with pytest.raises(UnicodeEncodeError):
        lesson.freeze_inputs(bad_source)
    assert not (tmp_path / 'account').exists()
    assert lesson.TARGET_EXECUTION_AUTHORIZED is False


def test_aggregate_opaque_diagnostic_bound_and_receipt(lesson, tmp_path, monkeypatch):
    Opaque = type('T' * 2048, (), {})
    detail = [[Opaque()] * 64] * 64
    captured = lesson._safe_diagnostic(detail)
    lesson._validate_diagnostic(captured)
    size = len(lesson.canonical(captured).encode('utf-8'))
    assert size <= 65_536
    assert 'truncated' in lesson.canonical(captured)
    def sentinel(record):
        raise lesson.CohortFailure('DEV PRIMARY opaque aggregate', detail)
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    observation = lesson.replay(failed.receipt)
    assert observation['message'] == 'DEV PRIMARY opaque aggregate'
    assert observation['details']['persistence'] == 'persisted'
    assert len(failed.receipt.encode('utf-8')) <= 4_000_000


@pytest.mark.parametrize('shape', ['wide', 'deep', 'surrogate_names', 'mapping_keys'])
def test_all_diagnostic_branches_share_global_budget(lesson, shape):
    class SurrogateType(type):
        def __getattribute__(cls, name):
            if name == '__name__':
                return '\ud800' * 2048
            return super().__getattribute__(name)
    class SurrogateOpaque(metaclass=SurrogateType):
        pass
    opaque = type('Opaque' * 340, (), {})()
    if shape == 'wide':
        detail = [[opaque] * 64] * 64
    elif shape == 'deep':
        detail = [opaque] * 64
        for _ in range(40):
            detail = [detail] * 64
    elif shape == 'surrogate_names':
        detail = [[SurrogateOpaque()] * 64] * 64
    else:
        detail = {('key' + str(i)) * 300: [opaque, '\ud800' * 2048] for i in range(64)}
    captured = lesson._safe_diagnostic(detail)
    lesson._validate_diagnostic(captured)
    assert len(lesson.canonical(captured).encode('utf-8')) <= 65_536
    # Independently count every diagnostic node, including type captures and markers.
    nodes = []
    dynamic = []
    def visit(node):
        nodes.append(node)
        tag = node['tag']
        if tag == 'text':
            dynamic.append(len(node['value']))
        elif tag == 'codepoint_text':
            dynamic.append(len(node['codepoints']))
        elif tag in ('unsupported', 'unsupported_type'):
            dynamic.append(len(node['type']))
            if tag == 'unsupported_type':
                visit(node['type_capture'])
        elif tag in ('sequence', 'mapping'):
            for item in node['items']:
                for child in (item if tag == 'mapping' else [item]):
                    visit(child)
    visit(captured)
    assert len(nodes) <= lesson.DIAGNOSTIC_NODES
    assert len(nodes) * 64 + sum(dynamic) <= lesson.DIAGNOSTIC_CHARACTERS


def test_aggregate_failure_fallback_and_both_storage_failures(lesson, tmp_path, monkeypatch):
    opaque = type('T' * 2048, (), {})()
    detail = [[opaque] * 64] * 64
    preferred = lesson._failure_receipt
    calls = []
    def reject_first(*args):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError('DEV SECONDARY preferred envelope failed')
        return preferred(*args)
    monkeypatch.setattr(lesson, '_failure_receipt', reject_first)
    def unavailable(*args):
        raise OSError('DEV SECONDARY storage unavailable \ud800')
    monkeypatch.setattr(lesson, '_exclusive', unavailable)
    def sentinel(record):
        raise lesson.CohortFailure('DEV PRIMARY aggregate \ud800', detail)
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    observation = lesson.replay(failed.receipt)
    assert observation['message'] == 'DEV PRIMARY aggregate \\ud800'
    assert observation['details']['primary']['message_capture']['codepoints'][-1] == 0xd800
    assert observation['details']['persistence'] == 'memory_only'
    assert len(failed.receipt.encode('utf-8')) <= 4_000_000
    assert any('storage unavailable' in item['message'] for item in observation['details']['secondary'])


def test_validator_rejects_legacy_aggregate_and_fallback_is_independent(lesson):
    # The exact V3 failure shape, as stored tagged data rather than arbitrary objects.
    opaque = {'tag': 'unsupported', 'type': 'T' * 2048}
    row = {'tag': 'sequence', 'items': [opaque] * 64, 'length': 64, 'truncated': False}
    oversized = {'tag': 'sequence', 'items': [row] * 64, 'length': 64, 'truncated': False}
    assert len(lesson.canonical(oversized).encode()) > lesson.LIMIT
    with pytest.raises(ValueError, match='aggregate'):
        lesson._validate_diagnostic(oversized)
    primary = lesson._error_info(RuntimeError('DEV PRIMARY oversized legacy diagnostic'))
    primary['details'] = oversized
    record = lesson.read_record(committed(lesson))
    raw = lesson._best_failure_receipt(record, primary, [], 'memory_only')
    observation = lesson.replay(raw)
    assert observation['message'] == 'DEV PRIMARY oversized legacy diagnostic'
    assert observation['details']['secondary'][0]['message'] == 'record too large'
    assert len(raw.encode()) <= lesson.LIMIT


def test_commitment_reserves_capacity_and_fallback_keeps_sources(lesson, monkeypatch):
    identity = lesson.freeze_inputs(source())
    initial = committed(lesson)
    path = 'mastery_hub/units/clip_prediction/README.md'
    padding = lesson.COMMITMENT_LIMIT - len(initial.encode()) - 4096
    identity['sources'][path]['utf8'] += 'P' * padding
    identity['sources'][path]['sha256'] = lesson.sha(identity['sources'][path]['utf8'].encode())
    manual = {'prediction': -0.25, 'confidence': 0.5,
              'reasoning': 'More candidates compete for row weight.', 'revision': ''}
    near_limit = lesson.commit_prediction(identity, source(), -0.25, 0.5, manual['reasoning'],
                                          'seen', manual_forecast=manual)
    assert lesson.COMMITMENT_LIMIT - 8192 < len(near_limit.encode()) <= lesson.COMMITMENT_LIMIT
    original = lesson._failure_receipt
    attempts = []
    def reject_preferred(*args):
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError('DEV SECONDARY prefer fallback at source capacity')
        return original(*args)
    monkeypatch.setattr(lesson, '_failure_receipt', reject_preferred)
    primary = lesson._error_info(lesson.CohortFailure('DEV PRIMARY source reserve \ud800',
                                                     ['\x00' * 2048] * 64))
    raw = lesson._best_failure_receipt(lesson.read_record(near_limit), primary,
                                     [primary] * 16, 'memory_only')
    record = lesson.read_record(raw)
    assert record['observation']['message'] == 'DEV PRIMARY source reserve \\ud800'
    assert record['identity']['sources'] == identity['sources']
    assert len(raw.encode()) <= lesson.LIMIT == 4_000_000
    secondary = record['observation']['details']['secondary']
    assert len(secondary) == 5  # preferred failure, three retained, explicit omitted-count record
    assert secondary[-1]['message'] == 'Additional secondary failures truncated'
    counts = {key['value']: value['value'] for key, value in secondary[-1]['details']['items']}
    assert counts == {'retained_secondary_count': 3, 'omitted_secondary_count': 13}
    identity['sources'][path]['utf8'] += 'Q' * 8192
    identity['sources'][path]['sha256'] = lesson.sha(identity['sources'][path]['utf8'].encode())
    with pytest.raises(ValueError, match='reserve failure receipt capacity'):
        lesson.commit_prediction(identity, source(), -0.25, 0.5, manual['reasoning'],
                                 'seen', manual_forecast=manual)


def test_summary_encoded_byte_bounds_remain_strict(lesson):
    info = lesson._error_info(RuntimeError('safe'))
    info['message'] = '\U0001f642' * lesson.DIAGNOSTIC_DISPLAY
    with pytest.raises(ValueError, match='UTF-8 byte bound'):
        lesson._validate_error_info(info)


def test_json_escaping_cannot_bypass_independent_fallback_bound(lesson):
    info = lesson._error_info(RuntimeError('DEV PRIMARY control characters'))
    raw_message = 'DEV PRIMARY ' + '\x00' * 12_000
    info['message'] = raw_message
    marker = 'diagnostic_capture_failed: ValueError'
    info['message_capture'] = {'tag': 'text', 'value': marker, 'length': len(marker),
                               'truncated': False}
    with pytest.raises(ValueError, match='serialized byte bound'):
        lesson._validate_error_info(info)
    raw = lesson._best_failure_receipt(lesson.read_record(committed(lesson)), info,
                                     [info] * 16, 'memory_only')
    observation = lesson.replay(raw)
    assert observation['message'] == raw_message[:lesson.DIAGNOSTIC_TEXT]
    capture = observation['details']['primary']['message_capture']
    assert capture['value'] == raw_message[:lesson.DIAGNOSTIC_TEXT]
    assert capture['length'] == len(raw_message) and capture['truncated'] is True
    assert len(raw.encode()) <= lesson.LIMIT


@pytest.mark.parametrize('detail_kind', ['partial_unicode', 'partial_aggregate'])
def test_combined_seal_and_both_write_failures_keep_ordered_history(lesson, tmp_path,
                                                                 monkeypatch, detail_kind):
    partial = {'state': 'DEV partial failure state', 'raw': '\udfff'}
    if detail_kind == 'partial_aggregate':
        opaque = type('T' * 2048, (), {})()
        partial['wide'] = [[opaque] * 64] * 64
    preferred = lesson._failure_receipt
    seals = []
    writes = []
    def reject_once(*args):
        seals.append(True)
        if len(seals) == 1:
            raise ValueError('DEV SECONDARY preferred seal \ud800')
        return preferred(*args)
    def no_storage(raw, path):
        writes.append(Path(path))
        raise OSError('DEV SECONDARY write ' + str(len(writes)))
    monkeypatch.setattr(lesson, '_failure_receipt', reject_once)
    monkeypatch.setattr(lesson, '_exclusive', no_storage)
    def sentinel(record):
        raise lesson.CohortFailure('DEV PRIMARY callback \ud800', partial)
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    observation = lesson.replay(failed.receipt)
    assert observation['message'] == 'DEV PRIMARY callback \\ud800'
    primary = observation['details']['primary']
    assert primary['message_capture']['codepoints'][-1] == 0xd800
    assert 'DEV partial failure state' in lesson.canonical(primary['details'])
    secondary = observation['details']['secondary']
    assert [item['message'] for item in secondary] == [
        'DEV SECONDARY preferred seal \\ud800', 'DEV SECONDARY write 1', 'DEV SECONDARY write 2']
    assert secondary[0]['message_capture']['codepoints'][-1] == 0xd800
    assert observation['details']['persistence'] == 'memory_only'
    assert len(writes) == 2 and len(seals) == 4
    assert len(failed.receipt.encode()) <= lesson.LIMIT
    assert not list((tmp_path / 'account').rglob('*.receipt.json'))


def test_repeated_fallback_appends_after_existing_causes_and_keeps_omissions(lesson, monkeypatch):
    record = lesson.read_record(committed(lesson))
    primary = lesson._error_info(lesson.CohortFailure('DEV PRIMARY history', {'state': 'partial'}))
    secondary = [lesson._error_info(RuntimeError('DEV old cause ' + str(i))) for i in range(16)]
    original = lesson._failure_receipt
    calls = []
    def alternating_failure(*args):
        calls.append(True)
        if len(calls) % 2:
            raise ValueError('DEV new seal ' + str(len(calls)))
        return original(*args)
    monkeypatch.setattr(lesson, '_failure_receipt', alternating_failure)
    first = lesson._best_failure_receipt(record, primary, secondary, 'memory_only')
    observation = lesson.replay(first)
    messages = [item['message'] for item in observation['details']['secondary']]
    assert messages[:4] == ['DEV old cause 0', 'DEV old cause 1', 'DEV old cause 2', 'DEV new seal 1']
    canonical_primary = observation['details']['primary']
    canonical_history = observation['details']['secondary']
    canonical_history.append(lesson._error_info(OSError('DEV later storage')))
    second = lesson._best_failure_receipt(record, canonical_primary, canonical_history, 'memory_only')
    observation = lesson.replay(second)
    messages = [item['message'] for item in observation['details']['secondary']]
    assert messages[:4] == ['DEV old cause 0', 'DEV old cause 1', 'DEV old cause 2', 'DEV new seal 3']
    counts = {key['value']: value['value'] for key, value in observation['details']['secondary'][-1]['details']['items']}
    assert counts == {'retained_secondary_count': 3, 'omitted_secondary_count': 15}
    assert observation['message'] == 'DEV PRIMARY history'


@pytest.mark.parametrize('failed_seal_calls', [(1, 3), (1, 4)])
def test_history_survives_later_receipt_rebuild_failures(lesson, tmp_path, monkeypatch,
                                                       failed_seal_calls):
    original = lesson._failure_receipt
    calls = []
    writes = []
    def fail_selected(*args):
        calls.append(True)
        if len(calls) in failed_seal_calls:
            raise ValueError('DEV seal ' + str(len(calls)))
        return original(*args)
    def unavailable(*args):
        writes.append(True)
        raise OSError('DEV write ' + str(len(writes)))
    monkeypatch.setattr(lesson, '_failure_receipt', fail_selected)
    monkeypatch.setattr(lesson, '_exclusive', unavailable)
    def sentinel(record):
        raise lesson.CohortFailure('DEV PRIMARY later rebuild', {'state': 'partial'})
    failed = _pretarget_failure(lesson, monkeypatch, tmp_path, sentinel)
    observation = lesson.replay(failed.receipt)
    messages = [item['message'] for item in observation['details']['secondary']]
    expected = (['DEV seal 1', 'DEV write 1', 'DEV seal 3', 'DEV write 2']
                if failed_seal_calls == (1, 3)
                else ['DEV seal 1', 'DEV write 1', 'DEV write 2', 'DEV seal 4'])
    assert messages == expected
    assert observation['message'] == 'DEV PRIMARY later rebuild'
    assert 'partial' in lesson.canonical(observation['details']['primary']['details'])
    assert observation['details']['persistence'] == 'memory_only'
