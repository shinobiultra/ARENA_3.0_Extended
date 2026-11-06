"""Guided extension worksheets and strict, self-contained learner-reported records.

No models run here. Hashes detect accidental edits; this is local study tooling,
not tamper-proof assessment. Method implementations stay in the source exercises.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'mastery_hub/extension_checkpoints.json'
SCHEMA = 'arena-extension-guided-practice'
VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
MAX_RECORD_BYTES = 2_000_000
CASE_KEYS = frozenset({'section', 'title', 'master_path', 'question', 'prediction_type',
    'forecast_items', 'config_fields', 'controls', 'method_functions', 'exercise_path',
    'solutions_path', 'evidence_class', 'evidence_detail', 'data_role', 'known_limitations',
    'status', 'experiment', 'observation_prompt', 'transfer', 'transfer_prediction_type'})
RECORD_KEYS = frozenset({'schema', 'schema_version', 'record_id', 'case', 'config',
    'config_fingerprint', 'prediction', 'confidence', 'reasoning', 'prior_exposure',
    'assessment', 'status', 'observation', 'reflection', 'transfer', 'history', 'seal'})


def _json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json(item)
        return
    raise ValueError('expected finite JSON-native values')


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError('nonempty text required')


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('missing or unknown fields')


def _digest(value):
    _json(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _fingerprint(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected lowercase SHA-256 digest')


def _relative(value):
    _text(value)
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) != value or '\\' in value:
        raise ValueError('source paths must be confined repository-relative paths')


def _strings(value):
    if (type(value) is not list or not value or not all(type(item) is str for item in value)
            or len(value) != len(set(value))):
        raise ValueError('expected distinct nonempty text items')
    for item in value:
        _text(item)


def validate_case(case):
    _keys(case, CASE_KEYS)
    _json(case)
    if type(case['section']) is not str or not re.fullmatch(r'[0-9]+\.[0-9]+', case['section']):
        raise ValueError('section must be N.M')
    for key in ('title', 'question', 'evidence_detail', 'experiment', 'observation_prompt', 'transfer'):
        _text(case[key])
    for key in ('master_path', 'exercise_path', 'solutions_path'):
        _relative(case[key])
    for key in ('controls', 'method_functions', 'known_limitations'):
        _strings(case[key])
    if type(case['config_fields']) is not dict or not case['config_fields']:
        raise ValueError('configuration fields required')
    for key, value in case['config_fields'].items():
        _text(key)
        _text(value)
    if case['prediction_type'] not in ('numeric', 'ranking', 'binary'):
        raise ValueError('unknown prediction type')
    if case['prediction_type'] == 'ranking':
        _strings(case['forecast_items'])
    elif case['forecast_items'] != []:
        raise ValueError('non-ranking cases cannot have ranking items')
    if case['transfer_prediction_type'] not in ('binary', 'numeric'):
        raise ValueError('this worksheet uses binary transfer forecasts or finite numeric transfer forecasts; other types are unsupported')
    if case['evidence_class'] not in ('analytic_toy', 'recorded_real_run'):
        raise ValueError('unknown evidence class')
    if case['data_role'] not in ('historical', 'monitored', 'unknown'):
        raise ValueError('unknown data role')
    if case['status'] != 'guided_practice':
        raise ValueError('only guided practice is supported')
    return deepcopy(case)


def load_cases():
    data = _parse(MANIFEST.read_text())
    _keys(data, {'schema_version', 'checkpoints'})
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('unsupported checkpoint manifest version')
    cases = [validate_case(case) for case in data['checkpoints']]
    if len({case['section'] for case in cases}) != len(cases):
        raise ValueError('duplicate section')
    return cases


def _config(case, config):
    _keys(config, {'files', 'inputs', 'workbook_helper_sha256'})
    _json(config)
    _keys(config['inputs'], case['config_fields'])
    _keys(config['files'], [case[k] for k in ('master_path', 'exercise_path', 'solutions_path')])
    for digest in [*config['files'].values(), config['workbook_helper_sha256']]:
        _fingerprint(digest)
    for value in config['inputs'].values():
        if value is None or value == '' or value == {} or value == []:
            raise ValueError('fill every frozen input before commitment; label historical unknowns explicitly')


def source_digest(path):
    """Hash experiment source; notebook runtime outputs do not alter its inputs."""
    path = Path(path)
    if path.suffix != '.ipynb':
        return hashlib.sha256(path.read_bytes()).hexdigest()
    notebook = json.loads(path.read_text())
    # Exact cell text and teaching tags bind the source. Execution counts, outputs,
    # cell IDs and kernel-added metadata are observations rather than method inputs.
    source = {'nbformat': notebook['nbformat'], 'cells': [
        {'cell_type': cell['cell_type'],
         'source': ''.join(cell['source']) if type(cell['source']) is list else cell['source'],
         'tags': cell.get('metadata', {}).get('tags', [])}
        for cell in notebook['cells']]}
    return _digest(source)


def freeze_inputs(case, inputs):
    """Hash local sources and copy *all* learner-specified experimental inputs."""
    case = validate_case(case)
    files = {}
    for key in ('master_path', 'exercise_path', 'solutions_path'):
        path = (ROOT / case[key]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('source path escaped repository')
        files[case[key]] = source_digest(path)
    config = {'files': files, 'inputs': deepcopy(inputs),
              'workbook_helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _config(case, config)
    return config


def _prediction(case, prediction):
    kind = case['prediction_type']
    if kind == 'binary' and type(prediction) is bool:
        return
    if kind == 'numeric' and (type(prediction) is int
                              or type(prediction) is float and math.isfinite(prediction)):
        return
    if kind == 'ranking' and type(prediction) is list and all(type(v) is str for v in prediction):
        if len(prediction) == len(set(prediction)) and set(prediction) == set(case['forecast_items']):
            return
    raise ValueError('prediction must match the declared numeric, binary or ranking type')


def _probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('confidence must be finite and in [0, 1]')


def _event(stage, payload, history):
    event = {'stage': stage, 'payload': deepcopy(payload),
             'previous': history[-1]['seal'] if history else None}
    event['seal'] = _digest(event)
    return event


def _seal(record):
    record['seal'] = _digest({k: v for k, v in record.items() if k != 'seal'})
    return record


def commit_prediction(case, config, prediction, confidence, reasoning, *, prior_exposure='unknown'):
    """New commitment; never rewrite an earlier record to change its forecast."""
    case = validate_case(case)
    _config(case, config)
    _prediction(case, prediction)
    _probability(confidence)
    _text(reasoning)
    if prior_exposure not in ('unknown', 'seen', 'self_reported_not_seen'):
        raise ValueError('unknown exposure category')
    fingerprint = _digest({'case': case, 'config': config})
    commitment = dict(prediction=deepcopy(prediction), confidence=float(confidence),
                      reasoning=reasoning, config_fingerprint=fingerprint)
    record = dict(schema=SCHEMA, schema_version=VERSION, record_id=uuid4().hex,
        case=case, config=deepcopy(config), **commitment, prior_exposure=prior_exposure,
        assessment='guided_practice', status='committed', observation=None,
        reflection=None, transfer=None, history=[_event('committed', commitment, [])])
    return validate_record(_seal(record))


def validate_record(record):
    """Validate a portable snapshot without executing code or consulting mutable sources."""
    _keys(record, RECORD_KEYS)
    _json(record)
    if record['schema'] != SCHEMA or type(record['schema_version']) is not int or record['schema_version'] not in SUPPORTED_VERSIONS:
        raise ValueError('unsupported record schema/version; quadratic app records use a different schema')
    if type(record['record_id']) is not str or not re.fullmatch('[0-9a-f]{32}', record['record_id']):
        raise ValueError('invalid record identity')
    _fingerprint(record['seal'])
    if record['seal'] != _digest({k: v for k, v in record.items() if k != 'seal'}):
        raise ValueError('record seal mismatch: preserve the original record and create a new commitment')
    case = validate_case(record['case'])
    _config(case, record['config'])
    if record['config_fingerprint'] != _digest({'case': case, 'config': record['config']}):
        raise ValueError('configuration fingerprint mismatch')
    _prediction(case, record['prediction'])
    _probability(record['confidence'])
    _text(record['reasoning'])
    if record['prior_exposure'] not in ('unknown', 'seen', 'self_reported_not_seen') or record['assessment'] != 'guided_practice':
        raise ValueError('records remain guided practice for every exposure category')
    stages = {'committed': ['committed'], 'observed': ['committed', 'observed'],
              'completed': ['committed', 'observed', 'completed']}
    if type(record['status']) is not str or record['status'] not in stages:
        raise ValueError('unknown record status')
    history = record['history']
    if type(history) is not list or len(history) != len(stages[record['status']]):
        raise ValueError('invalid transition history')
    previous = None
    for event, expected_stage in zip(history, stages[record['status']]):
        _keys(event, {'stage', 'payload', 'previous', 'seal'})
        if event['previous'] != previous or event['stage'] != expected_stage:
            raise ValueError('invalid transition history')
        if event['seal'] != _digest({k: v for k, v in event.items() if k != 'seal'}):
            raise ValueError('history seal mismatch')
        previous = event['seal']
    commitment = {key: record[key] for key in ('prediction', 'confidence', 'reasoning', 'config_fingerprint')}
    if history[0]['payload'] != commitment:
        raise ValueError('committed prediction cannot change')
    if record['status'] == 'committed':
        if record['observation'] is not None:
            raise ValueError('commitment cannot contain an observation')
    else:
        observation = record['observation']
        _validate_observation(case, observation, record['schema_version'])
        _keys(observation['controls'], case['controls'])
        for value in observation['controls'].values():
            if value is None or value == '' or value == {} or value == []:
                raise ValueError('record every control outcome, including failures')
        if observation['source_kind'] not in ('historical_reference', 'learner_run') or observation['verification'] != 'learner_reported':
            raise ValueError('observations are learner reported, not verified new runs')
        _text(observation['source_receipt'])
        if history[1]['payload'] != observation:
            raise ValueError('observation cannot change')
    if record['status'] != 'completed':
        if record['reflection'] is not None or record['transfer'] is not None:
            raise ValueError('reflect after recording observations and controls')
    else:
        _text(record['reflection'])
        transfer = record['transfer']
        _keys(transfer, {'question', 'changed_condition', 'prediction', 'confidence', 'reasoning', 'status'})
        if transfer['question'] != case['transfer'] or transfer['status'] != 'forecast_only':
            raise ValueError('transfer is a new forecast, not a completed transfer experiment')
        _text(transfer['changed_condition'])
        _text(transfer['reasoning'])
        _probability(transfer['confidence'])
        _prediction(dict(case, prediction_type=case['transfer_prediction_type']),
                    transfer['prediction'])
        if history[2]['payload'] != {'reflection': record['reflection'], 'transfer': transfer}:
            raise ValueError('reflection cannot change')
    return deepcopy(record)


def _validate_observation(case, observation, version):
    keys = {'value', 'controls', 'source_kind', 'source_receipt', 'verification'}
    _keys(observation, keys if version == 1 else keys | {'status', 'interval'})
    if version == 1:
        _prediction(case, observation['value'])
        return
    status, value, interval = (observation[key] for key in ('status', 'value', 'interval'))
    if status not in ('measured', 'approximately_read', 'unresolved', 'unavailable'):
        raise ValueError('unknown observation status')
    if status in ('unresolved', 'unavailable'):
        if value is not None or interval is not None:
            raise ValueError('unresolved/unavailable evidence cannot contain an invented value or interval')
        return
    if status == 'measured':
        _prediction(case, value)
        if interval is not None:
            raise ValueError('use approximately_read for a figure-reading range')
        return
    if value is not None:
        _prediction(case, value)
    if interval is not None:
        if case['prediction_type'] != 'numeric' or type(interval) is not list or len(interval) != 2:
            raise ValueError('approximate numeric interval must be [lower, upper]')
        for endpoint in interval:
            _prediction(case, endpoint)
        if interval[0] > interval[1] or value is not None and not interval[0] <= value <= interval[1]:
            raise ValueError('approximate interval must be ordered and contain any reported value')
    if value is None and interval is None:
        raise ValueError('approximately_read requires a supported value or numeric interval; otherwise use unresolved')


def prediction_feedback(record):
    """Descriptive case feedback only; never calibration, mastery or empirical acceptance.

    Legacy confidence has no retroactive probability interpretation. Approximate
    intervals bound possible point errors; they are not predictive intervals.
    """
    record = validate_record(record)
    kind = record['case']['prediction_type']
    semantics = ('legacy unscored self-rating' if record['schema_version'] == 1 else
                 'probability of selected binary event' if kind == 'binary' else 'unscored self-rating')
    feedback = {'confidence_meaning': semantics, 'score': None}
    observation = record['observation']
    if observation is None:
        return dict(feedback, message='Awaiting observation.')
    status = observation.get('status', 'measured')
    if status != 'measured':
        feedback['message'] = status + ': no point error or probability score.'
        if status == 'approximately_read' and observation['interval'] is not None:
            lower, upper = observation['interval']
            point = record['prediction']
            feedback['absolute_error_bounds'] = [max(lower - point, point - upper, 0),
                                                 max(abs(point - lower), abs(point - upper))]
        return feedback
    predicted, actual = record['prediction'], observation['value']
    if kind == 'numeric':
        feedback.update(metric='absolute_error', score=abs(predicted - actual),
                        message='Absolute error, in the forecast quantity’s units; confidence is unscored.')
    elif kind == 'ranking':
        positions = {item: index for index, item in enumerate(actual)}
        errors = sum(positions[left] > positions[right]
                     for index, left in enumerate(predicted) for right in predicted[index + 1:])
        feedback.update(metric='pairwise_ordering_errors', score=errors,
                        pairs=len(predicted) * (len(predicted) - 1) // 2,
                        message='Discordant pairs for the frozen complete order and tie policy; confidence is unscored.')
    elif record['schema_version'] == 2:
        feedback.update(metric='selected_event_brier', score=(record['confidence'] - int(predicted == actual)) ** 2,
                        message='Squared probability error for the selected event. One case cannot establish calibration.')
    else:
        feedback['message'] = 'Legacy binary confidence remains unscored; no event contract was frozen.'
    return feedback


def forecast_card(record):
    """Readable study view; the immutable full snapshot remains available separately."""
    record = validate_record(record)
    feedback = prediction_feedback(record)
    lines = [f"{record['case']['section']} — {record['case']['title']}",
             record['case']['question'], f"Forecast: {record['prediction']!r}",
             f"Confidence: {record['confidence']} ({feedback['confidence_meaning']})",
             f"Because: {record['reasoning']}", f"Worksheet: {record['status']} · guided practice"]
    if record['observation'] is not None:
        observation = record['observation']
        lines += [f"Observation: {observation.get('status', 'measured')} · {observation['value']!r}",
                  'Controls:']
        lines += [f'  {name}: {value}' for name, value in observation['controls'].items()]
        lines.append('Feedback: ' + feedback['message'])
        if feedback['score'] is not None:
            lines.append(f"{feedback['metric']}: {feedback['score']:.6g}")
        if 'absolute_error_bounds' in feedback:
            lower, upper = feedback['absolute_error_bounds']
            lines.append(f'Possible absolute error: {lower:.6g} to {upper:.6g} (reading range only)')
        if observation.get('interval') is not None:
            lines.append('Figure-reading range: ' + str(observation['interval']))
    if record['reflection'] is not None:
        lines += ['Causal revision: ' + record['reflection'],
                  'Transfer: forecast_only; a separate frozen experiment is still required.']
    return '\n'.join(lines)


def record_observation(record, config, observed, control_outcomes, source_kind, source_receipt,
                       *, observation_status='measured', interval=None):
    result = validate_record(record)
    if result['status'] != 'committed':
        raise ValueError('observation requires one committed prediction; start a new record for another run')
    _config(result['case'], config)
    if _digest({'case': result['case'], 'config': config}) != result['config_fingerprint']:
        raise ValueError('configuration changed: preserve this prediction and commit a new record before observing')
    result['observation'] = dict(value=deepcopy(observed), controls=deepcopy(control_outcomes),
        source_kind=source_kind, source_receipt=source_receipt, verification='learner_reported')
    if result['schema_version'] == 2:
        result['observation'].update(status=observation_status, interval=deepcopy(interval))
    elif observation_status != 'measured' or interval is not None:
        raise ValueError('legacy v1 records retain their exact measured observation contract; start a new v2 record')
    result['history'].append(_event('observed', result['observation'], result['history']))
    result['status'] = 'observed'
    return validate_record(_seal(result))


def reflect(record, mismatch, changed_condition, prediction, confidence, reasoning):
    result = validate_record(record)
    if result['status'] != 'observed':
        raise ValueError('record observations and controls, including unresolved evidence, before reflection')
    result['reflection'] = mismatch
    result['transfer'] = dict(question=result['case']['transfer'], changed_condition=changed_condition,
        prediction=prediction, confidence=confidence, reasoning=reasoning, status='forecast_only')
    payload = {'reflection': mismatch, 'transfer': result['transfer']}
    result['history'].append(_event('completed', payload, result['history']))
    result['status'] = 'completed'
    return validate_record(_seal(result))


def transfer_case(record):
    """Return the transfer case; a new commitment needs a fresh confidence judgment.

    In particular, v1 confidence remains an unscored self-rating and cannot be
    silently reused as a v2 binary event probability. This does not migrate records.
    """
    record = validate_record(record)
    if record['status'] != 'completed':
        raise ValueError('reflect and forecast the transfer first')
    case = deepcopy(record['case'])
    case.update(title=case['title'] + ' — transfer', question=case['transfer'],
                prediction_type=case['transfer_prediction_type'], forecast_items=[])
    return validate_case(case)


def _parse(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON number: ' + value)
    if type(payload) is not str or len(payload.encode('utf-8')) > MAX_RECORD_BYTES:
        raise ValueError('record exceeds UTF-8 byte limit')
    try:
        return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid JSON record') from exc


def dumps_record(record):
    payload = json.dumps(validate_record(record), indent=2, ensure_ascii=False, allow_nan=False)
    # Reserve the newline export_record writes, so every writer result is readable.
    if len((payload + '\n').encode('utf-8')) > MAX_RECORD_BYTES:
        raise ValueError('record exceeds UTF-8 byte limit, including export newline')
    return payload


def loads_record(payload):
    return validate_record(_parse(payload))


def load_record(path):
    path = Path(path)
    if path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError('record exceeds UTF-8 byte limit')
    return loads_record(path.read_text(encoding='utf-8'))


def replay_record(record):
    """Restore a worksheet; this does not replay or verify its scientific experiment."""
    return {'record': validate_record(record), 'verification': 'learner_reported', 'executed': False}


def export_record(record, directory):
    """Explicit opt-in only: a new mode-0600 JSON file outside every Git checkout."""
    payload = dumps_record(record)
    directory = Path(directory).expanduser().resolve()
    if directory.is_relative_to(ROOT) or any((p / '.git').is_file() or (p / '.git/HEAD').is_file()
                                               for p in [directory, *directory.parents]):
        raise ValueError('choose a private directory outside Git repositories')
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = directory / f"extension-{record['record_id']}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(payload + '\n')
    return target


def _cell(kind, identifier, source, **metadata):
    cell = {'cell_type': kind, 'id': identifier, 'metadata': metadata, 'source': source}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    return cell


def build_cells():
    cases_list = load_cases()
    cells = [_cell('markdown', 'intro', f'''# Prepare → predict → run → explain

**Can one wrong forecast improve your next explanation?** Start with the exposed example
below, then choose **one** of the {len(cases_list)} guided extension cases in the index.
You will prepare a method in its source lesson, predict a named result, inspect its controls,
explain a mismatch and forecast a changed condition. This companion records that reasoning;
the student implementations and visible scientific results belong to the linked lessons.

Original ARENA chapters 0–4 remain canonical prerequisites. Public references are
answer-rich: seeing them is guided practice, never evidence of unseen performance or mastery.
No model, API, GPU, download, cluster or new target experiment is authorized by this workbook.

## A complete exposed example: why did the update move twice as far?

For the scalar loss L(x)=x², start at x=4 and take one gradient descent step of size 0.25.
A learner predicts **3**, reasoning “the slope is x, so the step subtracts 1.” Their numeric
confidence **0.8 is an UNSCORED self-rating**, not an 80% probability of an exact real value.
Freeze this tiny example: x=4, loss=x², one step, rate=0.25; control: rate=0.
Run the visible arithmetic below. All answers are deliberately exposed practice.

| Condition | Forecast x after one step | Observed x | What it distinguishes |
| --- | ---: | ---: | --- |
| Rate 0.25 | 3 | 2 | Slope x versus slope 2x |
| Zero-rate control | 4 | 4 | Any update versus no update |
| Changed rate 0.125, after revision | 3 | 3 | Revised 2x explanation predicts a new condition |

**Causal revision:** “I omitted the factor of two in d(x²)/dx. The zero-rate control
rules out a change that happens even without an update. With slope 8, halving the rate
predicts x=3; the changed case agrees.” This checks the arithmetic explanation, not
convergence, stochastic training, or generalization. The optimizer lessons develop those ideas.
'''), _cell('code', 'worked-cycle', '''# Exposed practice only: visible arithmetic, no target experiment or record export.
x, rate = 4.0, 0.25
first_forecast, control_forecast = 3.0, 4.0
observed = x - rate * (2 * x)
control = x - 0 * (2 * x)
revised_rate, revised_forecast = 0.125, 3.0  # Forecast before evaluating changed condition.
changed_observed = x - revised_rate * (2 * x)
print('Condition       Forecast  Observation  Absolute error')
for name, forecast, value in [('initial', first_forecast, observed),
                               ('zero-rate', control_forecast, control),
                               ('changed-rate', revised_forecast, changed_observed)]:
    print(f'{name:14}  {forecast:8.2f}  {value:11.2f}  {abs(forecast-value):14.2f}')
assert (observed, control, changed_observed) == (2.0, 4.0, 3.0)
'''), _cell('markdown', 'worked-interpretation', '''<details><summary>Help / solution: forecast error is different from confidence</summary>

The first absolute error is |3−2|=1 in x units; the changed-case error is 0.
Neither says whether the self-rating 0.8 was calibrated.
For a **binary** selected event, confidence is the probability assigned to the selected answer:
if you forecast `False` with 0.8, assign probability 0.8 to the event being false (0.2 to true).
A measured outcome earns selected-event Brier error `(confidence − correctness)²`:
0.04 for a correct 0.8 forecast, 0.64 for a wrong one. Lower is better;
one case cannot establish calibration. Numeric point forecasts use absolute error in the
question's units. Complete rankings use pairwise ordering errors under the frozen tie policy.
Their confidence is an **UNSCORED self-rating**; no tolerance or probability event is invented.
Legacy v1 confidence is always treated as an unscored self-rating.
</details>

**Try It Yourself:** change `x` and `rate` in a copy of the arithmetic cell. Write both the
“slope x” and “slope 2x” forecasts before running it. Keep rate=0 as a control. Which
settings fail to distinguish the explanations? At x=0 both predict no movement.

## Choose one case, then follow five steps

1. Read the question; prepare the visible student method and its immediate tests.
2. Explain a causal forecast, its confidence meaning and what would falsify it.
3. Freeze **all** experimental inputs and code, then commit before the target observation.
4. Record the supported observation and controls, including uncertainty or missing evidence.
5. Revise your explanation. A transfer forecast completes the worksheet; a **separate frozen,
   observed transfer run** is required to test the changed-condition explanation.

### If a figure cannot answer the question

A plot may support only a numeric range, e.g. `[0.46, 0.53]`, rather than a precise value.
Choose `approximately_read`, set `observed=None`, and record that range in `observation_interval`.
This is a reading range, **not** a predictive confidence interval: a forecast 0.50 has possible
absolute error from 0 to 0.04, without a point score. If the ordering of a near-tie cannot be
read, use `unresolved` with no value or interval. If the required evidence is absent, use
`unavailable`. Both permit reflection without an error score. A failed control is an outcome;
a missing control is `unavailable` with a reason, never a fabricated pass.

<details><summary>Help: immutable records, source hashes and private snapshots</summary>

New records use `arena-extension-guided-practice` v2; legacy v1 records remain readable
without conversion or resealing. A legacy transfer forecast keeps its original unscored
confidence; creating a new v2 transfer requires a fresh or explicitly reaffirmed confidence
judgment under v2 semantics. Never automatically copy a v1 self-rating into a binary
probability field. This is **not compatible** with the optimizer app importer.
Exact code and inputs are frozen before observation. Changing any seed, prompt, asset,
checkpoint, metric, tie/selection rule, implementation or budget requires a new commitment.
Source hashes exclude notebook runtime outputs. Hashes detect edits; they do not independently
verify an experiment. Every record remains `guided_practice` and every observation
`learner_reported`. `replay_record` executes no scientific code.

A readable card appears after each step. To inspect a full resumable JSON snapshot explicitly,
call `inspect_record(records['7.1'])`; preserve it before leaving the session. Export is an
explicit private action described at the end. Run All reaches intentional learner stubs;
it never silently fills inputs, saves files or executes a scientific target.
</details>
'''), _cell('code', 'setup', '''from pathlib import Path
import importlib.util
import sys

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'mastery_hub/extension_checkpoints.json').is_file()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook inside the ARENA extension checkout.')
path = ROOT / 'mastery_hub/units/extension_predictions/lesson.py'
spec = importlib.util.spec_from_file_location('extension_worksheet', path)
worksheet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worksheet)
cases = {case['section']: case for case in worksheet.load_cases()}
records = {}  # Per-section immutable snapshots; restarting can restore printed JSON.

def show_record(record):
    print(worksheet.forecast_card(record))

def inspect_record(record):
    print('EXTENSION_RECORD_JSON_BEGIN')
    print(worksheet.dumps_record(record))
    print('EXTENSION_RECORD_JSON_END')
''')]
    index = '\n'.join(f"- [{case['section']} — {case['title']}](#section-{case['section'].replace('.', '-')})"
                      for case in cases_list)
    cells.append(_cell('markdown', 'case-index', '## Case index — choose one\n\n' + index))
    for case in cases_list:
        section = case['section']
        anchor = 'section-' + section.replace('.', '-')
        fields = '\n'.join(f'- **{key}**: {description}' for key, description in case['config_fields'].items())
        order = ('\nRanking items: ' + ', '.join(f'`{item}`' for item in case['forecast_items'])
                 + '. Use the order specified in the question; freeze tie-breaking before measuring.') if case['forecast_items'] else ''
        link_base = '../../..'
        confidence_guidance = ('Confidence is the probability (0–1) you assign to your selected binary answer: True means the event, False means its complement. This is not a calibration claim.' if case['prediction_type'] == 'binary' else 'Confidence (0–1) is an **UNSCORED self-rating** of your reasoning, not a probability of an exact number or ranking. Feedback uses absolute error in the stated units or pairwise ordering errors; freeze ranking tie-breaking before measurement.')
        prompt = f'''<a id="{anchor}"></a>
## {section} — {case['title']}

**Forecast ({case['prediction_type']}):** {case['question']}{order}

### Prepare and test before freezing

Open the [student exercise]({link_base}/{case['exercise_path']}) and prepare
{', '.join('`' + name + '`' for name in case['method_functions'])}. Complete its
sub-function tests before freezing the method code and experimental inputs. Consult the
[reference solution]({link_base}/{case['solutions_path']}) if needed for preparation.
These links are **answer-rich** and can expose saved target results: record `seen` if you
encounter them. This remains guided practice for every exposure category. Stop before
running the target experiment, then return here to commit your forecast.

Write what would falsify your causal explanation.
{confidence_guidance}

<details><summary>Help: exact pre-experiment freeze and source identity</summary>

**Freeze these exact inputs:**

{fields}

The helper additionally hashes the current master and the exercise/solution cell source
(excluding runtime outputs). Prepare the student method before commitment; use a fresh
commitment after any code edits. Freeze the
actual student implementation and local assets in your input manifest: changing a seed,
checkpoint, prompt, cohort, donor, metric, selection rule, code or budget requires a new
commitment. Unknown historical revisions stay explicitly unknown; they limit reproducibility.

Evidence: `{case['evidence_class']}` — {case['evidence_detail']}
Data role: `{case['data_role']}`. All cases are `guided_practice`.
</details>

<details><summary>Help: validate a commitment</summary>

Supply a finite number, a boolean, or each ranking item exactly once, according to the
question. Fill every input field, confidence and causal reasoning. `freeze_inputs` copies
inputs and hashes source bytes; `commit_prediction` returns a new sealed record. It cannot
edit an earlier forecast. Keep the committed card; explicitly inspect/copy its full snapshot before leaving the session.
No answer is prefilled here. Honest failed controls belong in the final record.
</details>
'''
        cells.append(_cell('markdown', anchor + '-forecast', prompt, section=section, stage='forecast'))
        cells.append(_cell('code', anchor + '-commit', f'''# After preparing and testing the method, fill this cell before running its target experiment.
case = cases[{section!r}]
inputs = {{field: None for field in case['config_fields']}}  # Replace EVERY None with exact inputs.
prediction = None  # Your numeric / binary / ranking forecast, as specified above.
confidence = None  # Binary: probability of selected answer; numeric/ranking: UNSCORED self-rating.
reasoning = ''  # Causal mechanism and a result that would falsify it.
prior_exposure = 'unknown'  # Or 'seen' / 'self_reported_not_seen'; always guided practice.
if any(value is None for value in inputs.values()) or prediction is None or confidence is None or not reasoning.strip():
    raise NotImplementedError('Fill frozen inputs, typed forecast, confidence and causal reasoning first.')
config = worksheet.freeze_inputs(case, inputs)
committed = worksheet.commit_prediction(case, config, prediction, confidence, reasoning,
                                          prior_exposure=prior_exposure)
records[{section!r}] = committed
show_record(committed)
''', section=section, stage='commit', tags=['prediction-commit']))
        references = f'''### After commitment: run the frozen experiment and inspect

Run the frozen {', '.join('`' + name + '`' for name in case['method_functions'])}
implementations on the committed inputs. Code or input edits require a new commitment
before measuring; keep the earlier record. The implementation and sub-function tests
belong to the preparation step above.

{case['experiment']}

Inspect the lesson's visible tables/curves/examples and compare against the
[reference solution]({link_base}/{case['solutions_path']}).
[Canonical extension master]({link_base}/{case['master_path']}).

**Measure:** {case['observation_prompt']}

**Controls to record:** {', '.join(case['controls'])}.
Record actual failed or unexpected controls; do not convert them to passing labels.
For a missing control, enter an explicit description such as `unavailable: no per-policy residual
was published`; distinguish this from `failed: ...` or `unresolved: ...`.
When only reading saved output, choose `historical_reference`. When you run your own
visible implementation, choose `learner_run` and keep its exact measurement receipt.
Both remain learner reported; this helper does not independently verify either source.

**Limitations:** {' '.join(case['known_limitations'])}
'''
        cells.append(_cell('markdown', anchor + '-experiment', references, section=section, stage='experiment'))
        cells.append(_cell('code', anchor + '-observe', f'''committed = records[{section!r}]
case = committed['case']
observed = None  # Supported typed value, or None for unresolved/unavailable or interval-only reading.
observation_status = 'measured'  # Or 'approximately_read', 'unresolved', 'unavailable'.
observation_interval = None  # Numeric figure-reading range [lower, upper], only for approximately_read.
control_outcomes = {{name: None for name in case['controls']}}
source_kind = None  # 'historical_reference' or 'learner_run'
source_receipt = ''  # Exact output cell/run receipt, raw metric values, denominator and failures.
if any(v is None for v in control_outcomes.values()) or source_kind is None or not source_receipt.strip():
    raise NotImplementedError('Record supported evidence/status and every control, then cite the exact source.')
# Re-read your actual inputs and source after running; changed inputs invalidate this commitment.
# Update current_inputs if anything changed in the experiment. The helper will reject a changed fingerprint.
current_inputs = committed['config']['inputs']
current_config = worksheet.freeze_inputs(case, current_inputs)
measured = worksheet.record_observation(committed, current_config, observed, control_outcomes,
                                        source_kind, source_receipt,
                                        observation_status=observation_status, interval=observation_interval)
records[{section!r}] = measured
show_record(measured)
''', section=section, stage='observe', tags=['prediction-observe']))
        cells.append(_cell('code', anchor + '-reflect', f'''measured = records[{section!r}]
# Reflection: compare your forecast with the raw measurements and failed controls.
# Identify a mismatch (or what agreement still does not establish), then revise your causal account.
print(measured['case']['transfer'])  # Try It Yourself: forecast a changed condition BEFORE measuring it.
mismatch = ''
changed_condition = ''  # Exact changed settings and what remains frozen.
transfer_prediction = None  # {'Finite numeric' if case['transfer_prediction_type'] == 'numeric' else 'Boolean'} answer to the printed transfer question.
transfer_confidence = None  # Binary: probability of selected event; numeric: UNSCORED self-rating.
transfer_reasoning = ''
if not mismatch.strip() or not changed_condition.strip() or transfer_prediction is None or transfer_confidence is None or not transfer_reasoning.strip():
    raise NotImplementedError('Explain the mismatch, freeze a changed condition and forecast its result.')
completed = worksheet.reflect(measured, mismatch, changed_condition, transfer_prediction,
                              transfer_confidence, transfer_reasoning)
records[{section!r}] = completed
show_record(completed)
# Worksheet complete. Study cycle still needs the separate observed changed-condition run.
# Transfer remains forecast_only in this record even after a separate run.
# For the transfer: tcase = worksheet.transfer_case(completed); freeze ALL new inputs;
# Make a fresh/reaffirmed confidence judgment under the new case’s typed semantics.
# Never automatically copy a legacy v1 unscored self-rating as a binary probability.
# commit_prediction(tcase, new_config, transfer_prediction, fresh_transfer_confidence, transfer_reasoning),
# then record_observation and reflect. Keep the original record; do not overwrite its forecast.
''', section=section, stage='reflect', tags=['reflection', 'try-it-yourself']))
    cells.append(_cell('markdown', 'transfer-route', '''## Test a revised explanation with a separate transfer record

The first record's `completed` status means **worksheet complete**, including a changed-condition
forecast. Its transfer remains `forecast_only`: a new forecast is not a second observation.
After preparing and testing the changed case, copy this route into your own cell. It freezes
all changed and unchanged inputs again; keep the first record in `records` and the transfer
in a separate variable. Run only an experiment already authorized for the source lesson.

**Judge confidence again before the new commitment.** For a binary transfer, give the
probability of your selected event; for a numeric transfer, give an **UNSCORED self-rating**.
This is required even when resuming an old record: v1 confidence had no probability contract.
Keep its value and original forecast unchanged. You may consciously reaffirm the same number,
but the new judgment must be explicit; the route never carries confidence forward automatically.

```python
original = records['7.1']  # Choose the section you completed.
tcase = worksheet.transfer_case(original)
fresh_transfer_confidence = None  # Judge/reaffirm: binary event probability; numeric UNSCORED self-rating.
transfer_inputs = {field: None for field in tcase['config_fields']}
# Fill every input exactly, including changed_condition and all unchanged settings.
if fresh_transfer_confidence is None:
    raise NotImplementedError('Judge/reaffirm confidence under the new transfer semantics before committing.')
tconfig = worksheet.freeze_inputs(tcase, transfer_inputs)
tforecast = worksheet.commit_prediction(
    tcase, tconfig, original['transfer']['prediction'], fresh_transfer_confidence,
    original['transfer']['reasoning'], prior_exposure='unknown')
show_record(tforecast)
# Preserve/inspect this commitment BEFORE running the separately authorized changed case.
# After running, use record_observation with the freshly re-frozen actual inputs,
# supported observation/status, every control outcome and the changed run's exact receipt.
```

Compare the new observation to both the old and revised causal accounts. State which control
could still explain the apparent agreement. A second unresolved/unavailable result is an honest
second worksheet outcome; it does not establish an empirically confirmed learning cycle.
'''))
    cells.append(_cell('markdown', 'private-export', '''## Resume or explicitly export a private worksheet

Call `inspect_record(records['7.1'])` to print the complete resumable snapshot, including
observations, controls and reflection, between `EXTENSION_RECORD_JSON_BEGIN` and
`EXTENSION_RECORD_JSON_END`. Cards are the everyday reading view; the JSON is the record.
To resume, pass that JSON text to `worksheet.loads_record(payload)` or read an explicit file
with `worksheet.load_record(path)` and assign the validated result into `records[section]`.
`worksheet.replay_record(record)` returns the stored worksheet with `executed=False` and
`verification='learner_reported'`; it does not rerun your experiment.

**Export only on your explicit action:** open a new code cell and call
`worksheet.export_record(records['7.1'], Path.home() / '.local/share/arena-private-worksheets')`.
Choose a private directory outside Git. The function writes a new mode-0600 JSON file,
refuses overwrites and refuses every Git checkout. Run All never calls it automatically.
No app importer or automatic mastery adapter is provided for this separate schema.
'''))
    return cells
