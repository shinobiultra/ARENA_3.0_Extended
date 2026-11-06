"""Public analytic forecast, strict local receipts, and inert notebook generation.

Target execution is disabled pending BV03. Hashes detect drift, not dishonest
exposure claims or hostile code. The learner's normalization remains inline.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
from itertools import islice
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
UNIT = Path(__file__).resolve().parent
SCHEMA = 'arena-clip-candidate-set-prediction'
VERSION = 1
PROTOCOL_ID = '12.1-candidate-set-normalization-v1'
SESSION = 'learner-primary-v1'
SESSION_ANCHOR = None  # OS account home, independent of HOME/XDG; DEV may override.
TARGET_EXECUTION_AUTHORIZED = False  # BV03 is deferred; changing this needs review.
LIMIT = 4_000_000
FAILURE_RESERVE = 1_000_000
COMMITMENT_LIMIT = LIMIT - FAILURE_RESERVE
RTOL = 1e-10
ATOL = 1e-12
CONDITIONS = ('baseline', 'expansion', 'unchanged_baseline', 'permuted_expansion',
              'shifted_baseline', 'shifted_expansion')
KINDS = ('softmax', 'sigmoid')
SOURCE_PATHS = (
    'mastery_hub/units/clip_prediction/README.md',
    'mastery_hub/units/clip_prediction/protocol.json',
    'mastery_hub/units/clip_prediction/lesson.py',
    'mastery_hub/units/clip_prediction/exercises.ipynb',
    'scripts/build_clip_prediction_notebook.py',
    'tests/mastery/test_clip_prediction.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(canonical(value).encode())


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected), 'missing/unknown fields')


def number(value):
    require(type(value) in (int, float), 'finite number excluding bool required')
    try:
        require(math.isfinite(value), 'finite number required')
    except OverflowError as exc:
        raise ValueError('number exceeds supported finite range') from exc


def text(value):
    # Bound codepoints before whitespace scanning or strict UTF-8 encoding.
    require(type(value) is str and 0 < len(value) <= 20_000, 'bounded text required')
    require(bool(value.strip()), 'bounded text required')
    require(_valid_utf8(value), 'UTF-8 text required')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    require(type(raw) is str and 0 < len(raw) <= LIMIT, 'bounded JSON required')
    require(len(raw.encode()) <= LIMIT, 'bounded JSON required')
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda value: require(False, 'nonfinite JSON'))
    except (RecursionError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError('invalid JSON') from exc


def _seal(record):
    result = deepcopy(record)
    result.pop('seal', None)
    result['seal'] = digest(result)
    raw = canonical(result)
    require(len(raw.encode()) <= LIMIT, 'record too large')
    return raw


def validate_source(source):
    require(type(source) is str and 0 < len(source) <= 65_536, 'bounded source')
    require(len(source.encode()) <= 65_536, 'bounded source')
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('invalid source') from exc
    require(len(tree.body) == 1 and type(tree.body[0]) is ast.FunctionDef,
            'one plain function definition required')
    node = tree.body[0]
    args = node.args
    require(node.name == 'zero_shot_probabilities', 'original function name required')
    require(not node.decorator_list and not node.returns and not node.type_params,
            'no evaluated decorators, annotations or type parameters')
    require(not args.posonlyargs and not args.kwonlyargs and not args.vararg
            and not args.kwarg and not args.defaults and not args.kw_defaults,
            'plain original positional signature required')
    require(tuple(arg.arg for arg in args.args) == ('logits', 'kind')
            and all(arg.annotation is None for arg in args.args), 'original signature required')
    return tree


def validate_protocol(case):
    keys(case, ('id', 'row_ids', 'candidate_ids', 'base_ids', 'selected_ids', 'logit_rule',
                'raw_logits', 'permutation', 'shift', 'dtype', 'device', 'threads',
                'conditions', 'kinds', 'max_calls', 'timeout_seconds', 'rtol', 'atol',
                'metric', 'units', 'roles', 'target_status', 'source_pins'))
    fixed = {
        'id': PROTOCOL_ID, 'row_ids': ['q0', 'q1', 'q2', 'q3'],
        'candidate_ids': ['c0', 'c1', 'c2', 'c3', 'c4'], 'base_ids': ['c0', 'c1', 'c2'],
        'selected_ids': ['c1', 'c2', 'c0', 'c1'],
        'logit_rule': '(((7*i + 5*j + 3) mod 17) - 8) / 4; 0<=i<4, 0<=j<5',
        'permutation': [4, 1, 3, 0, 2], 'shift': 1, 'dtype': 'float64', 'device': 'cpu',
        'threads': 1, 'conditions': list(CONDITIONS), 'kinds': list(KINDS),
        'max_calls': 12, 'timeout_seconds': 30, 'rtol': RTOL, 'atol': ATOL,
        'metric': 'mean(selected_expanded_softmax-selected_baseline_softmax)',
        'units': 'weight units; 0.01 equals one percentage point',
        'roles': {'TRAIN': 'none', 'DEV': 'exposed distinct literal fixtures', 'TEST': 'none'},
        'target_status': 'UNCONSTRUCTED_UNEXECUTED',
    }
    for key, value in fixed.items():
        require(canonical(case[key]) == canonical(value), 'fixed protocol field: ' + key)
    # Raw JSON arithmetic only, never tensors or normalization.
    expected = [[(((7 * i + 5 * j + 3) % 17) - 8) / 4 for j in range(5)]
                for i in range(4)]
    require(canonical(case['raw_logits']) == canonical(expected), 'raw public logit expansion')
    require(type(case['source_pins']) is dict and case['source_pins'], 'source pins required')
    for path, pin in case['source_pins'].items():
        text(path)
        require(type(pin) is str and len(pin) == 64
                and all(c in '0123456789abcdef' for c in pin), 'source pin hash')
    return deepcopy(case)


def runtime_identity():
    """Metadata lookup only: no torch import or numerical call."""
    return {'python': sys.version, 'torch': importlib.metadata.version('torch'),
            'platform': platform.platform(), 'byteorder': sys.byteorder,
            'device': 'cpu', 'dtype': 'float64', 'threads': 1}


def _notebook_template():
    path = ROOT / 'scripts/build_clip_prediction_notebook.py'
    spec = importlib.util.spec_from_file_location('clip_template_builder', path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return (json.dumps(builder.make_notebook(), indent=1, ensure_ascii=False) + '\n').encode()


def _snapshots():
    snapshots = {}
    for relative in SOURCE_PATHS:
        raw = (_notebook_template() if relative.endswith('.ipynb')
               else (ROOT / relative).read_bytes())
        snapshots[relative] = {'utf8': raw.decode(), 'sha256': sha(raw)}
    return snapshots


def freeze_inputs(method_source):
    """No method execution, compilation, target allocation or session mutation."""
    validate_source(method_source)
    case = validate_protocol(parse((UNIT / 'protocol.json').read_text()))
    return {'protocol': case, 'protocol_sha256': digest(case), 'method_source': method_source,
            'method_sha256': sha(method_source.encode()), 'sources': _snapshots(),
            'runtime': runtime_identity(), 'session': SESSION}


def _identity(value):
    keys(value, ('protocol', 'protocol_sha256', 'method_source', 'method_sha256',
                 'sources', 'runtime', 'session'))
    validate_protocol(value['protocol'])
    require(value['protocol_sha256'] == digest(value['protocol']), 'protocol hash mismatch')
    validate_source(value['method_source'])
    require(value['method_sha256'] == sha(value['method_source'].encode()), 'method hash mismatch')
    keys(value['sources'], SOURCE_PATHS)
    for item in value['sources'].values():
        keys(item, ('utf8', 'sha256'))
        require(type(item['utf8']) is str and item['sha256'] == sha(item['utf8'].encode()),
                'recoverable source snapshot mismatch')
    keys(value['runtime'], ('python', 'torch', 'platform', 'byteorder', 'device', 'dtype', 'threads'))
    for key in ('python', 'torch', 'platform', 'byteorder', 'device', 'dtype'):
        text(value['runtime'][key])
    require(value['runtime']['device'] == 'cpu' and value['runtime']['dtype'] == 'float64'
            and type(value['runtime']['threads']) is int and value['runtime']['threads'] == 1,
            'frozen CPU runtime policy')
    require(value['session'] == SESSION, 'fixed session required')


def _forecast(value):
    keys(value, ('prediction', 'confidence', 'reasoning'))
    number(value['prediction'])
    number(value['confidence'])
    require(0 <= value['confidence'] <= 1, 'confidence range')
    text(value['reasoning'])


def _commitment(value):
    keys(value, ('prediction', 'confidence', 'reasoning', 'prior_exposure', 'manual_forecast'))
    _forecast({key: value[key] for key in ('prediction', 'confidence', 'reasoning')})
    require(value['prior_exposure'] in ('seen', 'unknown', 'self_reported_not_seen'),
            'explicit exposure choice required')
    manual = value['manual_forecast']
    keys(manual, ('prediction', 'confidence', 'reasoning', 'revision'))
    _forecast({key: manual[key] for key in ('prediction', 'confidence', 'reasoning')})
    require(type(manual['revision']) is str, 'revision explanation required')
    changed = any(canonical(manual[key]) != canonical(value[key])
                  for key in ('prediction', 'confidence', 'reasoning'))
    if changed:
        require(bool(manual['revision'].strip()), 'explain the manual forecast revision')
        text(manual['revision'])


def commit_prediction(identity, method_source, prediction, confidence, reasoning,
                      prior_exposure, *, manual_forecast):
    _identity(identity)
    require(identity['method_source'] == method_source, 'method changed since freeze')
    commitment = {'prediction': prediction, 'confidence': confidence, 'reasoning': reasoning,
                  'prior_exposure': prior_exposure, 'manual_forecast': deepcopy(manual_forecast)}
    _commitment(commitment)
    raw = _seal({'schema': SCHEMA, 'version': VERSION, 'status': 'committed',
                 'identity': deepcopy(identity), 'commitment': commitment,
                 'observation': None, 'reflection': None})
    read_record(raw)
    return raw


def _close(left, right):
    return math.isclose(left, right, rel_tol=RTOL, abs_tol=ATOL)


def _matrix(value, rows, columns):
    require(type(value) is list and len(value) == rows, 'matrix row count')
    for row in value:
        require(type(row) is list and len(row) == columns, 'matrix column count')
        for item in row:
            number(item)
            require(0 <= item <= 1, 'readout outside [0,1]')


def _same(left, right):
    return all(_close(a, b) for row_a, row_b in zip(left, right)
               for a, b in zip(row_a, row_b))


def _summarize(conditions, rows, base_ids, expanded_ids, selected, permutation):
    """Stored-output bookkeeping only; no normalization or hidden reference."""
    require(len(conditions) == 6, 'all six conditions required before any mean')
    expected_ids = [base_ids, expanded_ids, base_ids,
                    [expanded_ids[i] for i in permutation], base_ids, expanded_ids]
    for index, item in enumerate(conditions):
        keys(item, ('name', 'candidate_ids', 'outputs'))
        require(item['name'] == CONDITIONS[index] and item['candidate_ids'] == expected_ids[index],
                'condition order/IDs mismatch')
        keys(item['outputs'], KINDS)
        for kind in KINDS:
            _matrix(item['outputs'][kind], len(rows), len(expected_ids[index]))
        require(all(_close(sum(row), 1.0) for row in item['outputs']['softmax']),
                'softmax row sum control failed')
    by_name = {item['name']: item['outputs'] for item in conditions}
    for kind in KINDS:
        require(_same(by_name['baseline'][kind], by_name['unchanged_baseline'][kind]),
                'identity control failed')
        restored = [[row[permutation.index(i)] for i in range(len(expanded_ids))]
                    for row in by_name['permuted_expansion'][kind]]
        require(_same(by_name['expansion'][kind], restored), 'candidate-ID control failed')
    for base, shifted in (('baseline', 'shifted_baseline'), ('expansion', 'shifted_expansion')):
        require(_same(by_name[base]['softmax'], by_name[shifted]['softmax']),
                'common-shift softmax control failed')
    selected_values = {}
    for name in CONDITIONS:
        ids = expected_ids[CONDITIONS.index(name)]
        selected_values[name] = {
            kind: [by_name[name][kind][i][ids.index(label)] for i, label in enumerate(selected)]
            for kind in KINDS
        }
    require(all(_close(a, b) for a, b in zip(selected_values['baseline']['sigmoid'],
                                            selected_values['expansion']['sigmoid'])),
            'independent sigmoid control failed')
    # Every condition and mandatory control has passed before the first mean.
    differences = {kind: [b - a for a, b in zip(selected_values['baseline'][kind],
                                                selected_values['expansion'][kind])]
                   for kind in KINDS}
    means = {kind: math.fsum(values) / len(rows) for kind, values in differences.items()}
    for value in means.values():
        number(value)
    return {'selected': selected_values, 'per_row_changes': differences,
            'delta': means['softmax'], 'sigmoid_delta': means['sigmoid']}


class CohortFailure(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


DIAGNOSTIC_ITEMS = 64
DIAGNOSTIC_DEPTH = 12
DIAGNOSTIC_NODES = 4096
DIAGNOSTIC_CHARACTERS = 8192
DIAGNOSTIC_TEXT = 2048
DIAGNOSTIC_DISPLAY = 6 * DIAGNOSTIC_TEXT
DIAGNOSTIC_NODE_TEXT = 64  # covers all fixed keys/tags/reasons for each node
DIAGNOSTIC_BYTES = 65_536
DIAGNOSTIC_SUMMARY_BYTES = 7 * DIAGNOSTIC_TEXT + 2
FALLBACK_DETAILS_BYTES = 8192
FALLBACK_INFO_BYTES = 131_072


def _valid_utf8(value):
    try:
        value.encode('utf-8', errors='strict')
        return True
    except UnicodeEncodeError:
        return False


def _utf8_display(value):
    """Bound first; only surrogate codepoints become visible backslash escapes.

    The separate tagged capture preserves the exact original codepoints, including
    raw surrogate pairs and literal backslash-u text that a display could confuse.
    """
    prefix = value[:DIAGNOSTIC_TEXT]
    return prefix.encode('utf-8', errors='backslashreplace').decode('utf-8')


def _summary_text(value, fallback):
    display = _utf8_display(value)
    return display if display.strip() else fallback


def _captured_prefix(capture):
    require(capture['tag'] in ('text', 'codepoint_text'), 'text capture required')
    return (capture['value'] if capture['tag'] == 'text'
            else ''.join(map(chr, capture['codepoints'])))


def _text_capture(value, extent):
    prefix = value[:extent]
    if not _valid_utf8(prefix):
        return {'tag': 'codepoint_text', 'codepoints': list(map(ord, prefix)),
                'length': len(value), 'truncated': len(value) > extent}
    return {'tag': 'text', 'value': prefix,
            'length': len(value), 'truncated': len(value) > extent}



def _diagnostic(value, depth=0, budget=None):
    """One shared budget, including metadata/markers and opaque type captures."""
    if budget is None:
        budget = [DIAGNOSTIC_NODES, DIAGNOSTIC_CHARACTERS]
    captured = _diagnostic_node(value, depth, budget)
    extent = len(canonical(captured).encode('utf-8'))
    if extent > DIAGNOSTIC_BYTES:
        return {'tag': 'truncated', 'reason': 'byte_limit', 'captured_bytes': extent}
    return captured


def _diagnostic_node(value, depth, budget):
    # Parents check resources before visiting a child. Every emitted node,
    # including depth/text/item truncation markers, consumes this allowance.
    require(budget[0] > 0 and budget[1] >= DIAGNOSTIC_NODE_TEXT, 'diagnostic budget exhausted')
    budget[0] -= 1
    budget[1] -= DIAGNOSTIC_NODE_TEXT
    if depth >= DIAGNOSTIC_DEPTH:
        return {'tag': 'truncated', 'reason': 'depth_limit'}
    kind = type(value)
    if value is None or kind is bool:
        return {'tag': 'scalar', 'value': value}
    if kind is str:
        extent = min(len(value), DIAGNOSTIC_TEXT, budget[1])
        budget[1] -= extent
        return _text_capture(value, extent)
    if kind is int:
        if value.bit_length() > 1023:
            return {'tag': 'large_integer', 'bits': value.bit_length()}
        return {'tag': 'scalar', 'value': value}
    if kind is float:
        if not math.isfinite(value):
            label = 'nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf')
            return {'tag': 'nonfinite', 'value': label}
        return {'tag': 'scalar', 'value': value}
    if kind in (list, tuple, dict):
        length = len(value)
        items = islice(value.items(), DIAGNOSTIC_ITEMS) if kind is dict else value[:DIAGNOSTIC_ITEMS]
        output = []
        for item in items:
            if budget[0] <= 0 or budget[1] < DIAGNOSTIC_NODE_TEXT:
                break
            if kind is dict:
                key, entry = item
                captured_key = _diagnostic_node(key, depth + 1, budget)
                if budget[0] <= 0 or budget[1] < DIAGNOSTIC_NODE_TEXT:
                    break  # no incomplete mapping pair is published
                output.append([captured_key, _diagnostic_node(entry, depth + 1, budget)])
            else:
                output.append(_diagnostic_node(item, depth + 1, budget))
        return {'tag': 'mapping' if kind is dict else 'sequence', 'items': output,
                'length': length, 'truncated': len(output) < length}
    raw_name = _raw_type_name(value)
    prefix = raw_name[:DIAGNOSTIC_TEXT]
    if (len(raw_name) <= DIAGNOSTIC_TEXT and _valid_utf8(prefix)
            and prefix.strip() and len(prefix) <= budget[1]):
        budget[1] -= len(prefix)
        return {'tag': 'unsupported', 'type': prefix}
    if budget[0] <= 0 or budget[1] < DIAGNOSTIC_NODE_TEXT + 18:
        return {'tag': 'truncated', 'reason': 'text_limit'}
    # The extra typed-name capture is another node. Reserve its metadata and
    # enough shared characters for BOTH display and exact codepoints. The factor
    # 18 covers six-character escapes and the nonempty unnamed-type fallback.
    budget[0] -= 1
    budget[1] -= DIAGNOSTIC_NODE_TEXT
    extent = min(len(raw_name), DIAGNOSTIC_TEXT, budget[1] // 18)
    prefix = raw_name[:extent]
    display = _summary_text(prefix, 'UnnamedType')
    budget[1] -= len(display) + extent
    return {'tag': 'unsupported_type', 'type': display,
            'type_capture': _text_capture(raw_name, extent)}


def _validate_diagnostic(value, depth=0, count=None):
    """Validate captured data rather than accepting arbitrary JSON in a failure."""
    root = count is None
    if root:
        count = [0, 0]
    count[0] += 1
    count[1] += DIAGNOSTIC_NODE_TEXT
    require(depth <= DIAGNOSTIC_DEPTH + 1 and count[0] <= DIAGNOSTIC_NODES
            and count[1] <= DIAGNOSTIC_CHARACTERS, 'diagnostic aggregate bound')
    require(type(value) is dict and type(value.get('tag')) is str, 'tagged diagnostic')
    tag = value['tag']
    if tag == 'scalar':
        keys(value, ('tag', 'value'))
        item = value['value']
        require(item is None or type(item) in (bool, int, float), 'diagnostic scalar')
        if type(item) in (int, float):
            number(item)
    elif tag == 'text':
        keys(value, ('tag', 'value', 'length', 'truncated'))
        require(type(value['value']) is str and len(value['value']) <= DIAGNOSTIC_TEXT,
                'diagnostic text')
        require(_valid_utf8(value['value']), 'diagnostic text must be UTF-8')
        count[1] += len(value['value'])
        require(type(value['length']) is int and value['length'] >= len(value['value'])
                and type(value['truncated']) is bool
                and value['truncated'] == (value['length'] > len(value['value'])),
                'diagnostic text extent')
    elif tag == 'codepoint_text':
        keys(value, ('tag', 'codepoints', 'length', 'truncated'))
        points = value['codepoints']
        require(type(points) is list and len(points) <= DIAGNOSTIC_TEXT
                and all(type(point) is int and 0 <= point <= 0x10ffff for point in points),
                'bounded Unicode codepoints')
        require(any(0xd800 <= point <= 0xdfff for point in points), 'surrogate capture tag')
        count[1] += len(points)
        require(type(value['length']) is int and value['length'] >= len(points)
                and type(value['truncated']) is bool
                and value['truncated'] == (value['length'] > len(points)),
                'codepoint text extent')
    elif tag in ('mapping', 'sequence'):
        keys(value, ('tag', 'items', 'length', 'truncated'))
        items = value['items']
        require(type(items) is list and len(items) <= DIAGNOSTIC_ITEMS
                and type(value['length']) is int and value['length'] >= len(items)
                and type(value['truncated']) is bool
                and value['truncated'] == (value['length'] > len(items)), 'diagnostic items')
        for item in items:
            if tag == 'mapping':
                require(type(item) is list and len(item) == 2, 'diagnostic mapping entry')
                for entry in item:
                    _validate_diagnostic(entry, depth + 1, count)
            else:
                _validate_diagnostic(item, depth + 1, count)
    elif tag == 'nonfinite':
        keys(value, ('tag', 'value'))
        require(value['value'] in ('nan', '+inf', '-inf'), 'nonfinite tag')
    elif tag == 'large_integer':
        keys(value, ('tag', 'bits'))
        require(type(value['bits']) is int and value['bits'] > 1023, 'large integer extent')
    elif tag == 'truncated':
        if value.get('reason') in ('byte_limit', 'fallback_limit'):
            keys(value, ('tag', 'reason', 'captured_bytes'))
            threshold = DIAGNOSTIC_BYTES if value['reason'] == 'byte_limit' else FALLBACK_DETAILS_BYTES
            require(type(value['captured_bytes']) is int and value['captured_bytes'] > threshold,
                    'captured diagnostic byte extent')
        else:
            keys(value, ('tag', 'reason'))
            require(value['reason'] in ('depth_limit', 'item_limit', 'text_limit'), 'truncation reason')
    elif tag == 'unsupported_type':
        keys(value, ('tag', 'type', 'type_capture'))
        text(value['type'])
        require(len(value['type']) <= DIAGNOSTIC_DISPLAY, 'type display extent')
        count[1] += len(value['type'])
        _validate_diagnostic(value['type_capture'], depth + 1, count)
        capture = value['type_capture']
        require(capture['tag'] in ('text', 'codepoint_text'), 'type name capture')
        prefix = _captured_prefix(capture)
        require(value['type'] == _summary_text(prefix, 'UnnamedType'), 'type display/capture mismatch')
    elif tag == 'unsupported':
        keys(value, ('tag', 'type'))
        text(value['type'])
        require(len(value['type']) <= DIAGNOSTIC_TEXT, 'type name extent')
        count[1] += len(value['type'])
    else:
        raise ValueError('unknown diagnostic tag')

    require(count[1] <= DIAGNOSTIC_CHARACTERS, 'diagnostic aggregate text bound')
    if root:
        require(len(canonical(value).encode('utf-8')) <= DIAGNOSTIC_BYTES,
                'diagnostic encoded byte bound')


def _raw_type_name(value):
    try:
        name = type(value).__name__
        return name if type(name) is str and name else 'UnknownType'
    except BaseException:
        return 'UnavailableType'


def _type_name(value):
    return _summary_text(_raw_type_name(value), 'UnnamedType')


def _safe_diagnostic(value):
    try:
        captured = _diagnostic(value)
        _validate_diagnostic(captured)
        return captured
    except BaseException as exc:
        message = 'diagnostic_capture_failed: ' + _type_name(exc)
        # This fallback does not re-enter the failed sanitizer.
        prefix = message[:DIAGNOSTIC_TEXT]
        return {'tag': 'text', 'value': prefix, 'length': len(message),
                'truncated': len(message) > len(prefix)}


def _error_info(exc, *, details=None):
    """Formatting/sanitation failures cannot discard the primary exception."""
    raw_type = _raw_type_name(exc)
    error_type = _summary_text(raw_type, 'UnnamedErrorType')
    try:
        message = str(exc)
        message_capture = _safe_diagnostic(message)
        message = _summary_text(message, error_type)
    except BaseException as formatting:
        message = error_type + ': exception text unavailable'
        message_capture = _safe_diagnostic({'formatting_failed': _type_name(formatting)})
    if details is None:
        try:
            details = getattr(exc, 'details', {})
        except BaseException as access:
            details = {'details_access_failed': _type_name(access)}
    return {'error_type': error_type, 'error_type_capture': _safe_diagnostic(raw_type),
            'message': message, 'message_capture': message_capture,
            'details': _safe_diagnostic(details)}


def _validate_error_info(info):
    keys(info, ('error_type', 'error_type_capture', 'message', 'message_capture', 'details'))
    text(info['error_type'])
    text(info['message'])
    require(len(info['message']) <= DIAGNOSTIC_DISPLAY + 40
            and len(info['error_type']) <= DIAGNOSTIC_DISPLAY, 'bounded failure summary')
    require(len(info['error_type'].encode('utf-8')) <= DIAGNOSTIC_DISPLAY
            and len(info['message'].encode('utf-8')) <= DIAGNOSTIC_DISPLAY + 40,
            'failure summary UTF-8 byte bound')
    require(len(canonical(info['error_type']).encode('utf-8')) <= DIAGNOSTIC_SUMMARY_BYTES
            and len(canonical(info['message']).encode('utf-8')) <= DIAGNOSTIC_SUMMARY_BYTES + 40,
            'failure summary serialized byte bound')
    _validate_diagnostic(info['error_type_capture'])
    _validate_diagnostic(info['message_capture'])
    _validate_diagnostic(info['details'])

    type_capture = info['error_type_capture']
    prefix = _captured_prefix(type_capture)
    capture_failed = (type_capture['tag'] == 'text'
                      and prefix.startswith('diagnostic_capture_failed: '))
    if not capture_failed:
        require(info['error_type'] == _summary_text(prefix, 'UnnamedErrorType'),
                'error type display/capture mismatch')
    message_capture = info['message_capture']
    if message_capture['tag'] in ('text', 'codepoint_text'):
        prefix = _captured_prefix(message_capture)
        capture_failed = (message_capture['tag'] == 'text'
                          and prefix.startswith('diagnostic_capture_failed: '))
        if not capture_failed:
            require(info['message'] == _summary_text(prefix, info['error_type']),
                    'error message display/capture mismatch')
    else:
        require(message_capture['tag'] == 'mapping'
                and info['message'] == info['error_type'] + ': exception text unavailable',
                'exception formatting failure capture')


def _input_identity(argument):
    """Exact CPU logical-element bytes plus dtype/shape/device; signed zero matters."""
    import torch as t
    require(type(argument) is t.Tensor and argument.device.type == 'cpu',
            'input identity requires plain CPU tensor')
    identity = {'shape': list(argument.shape), 'dtype': str(argument.dtype),
                'device': str(argument.device), 'bytes': argument.numel() * argument.element_size()}
    if identity['bytes'] > 4096:
        return {**identity, 'sha256': None, 'capture_omitted': 'byte_limit'}
    data = argument.detach().contiguous().view(t.uint8).reshape(-1).tolist()
    return {**identity, 'sha256': sha(bytes(data))}


def _capture_output(output):
    """Capture malformed returns before validation, with explicit omission extent."""
    import torch as t
    if type(output) is not t.Tensor:
        return _diagnostic(output)
    captured = {'tag': 'tensor', 'shape': list(output.shape), 'dtype': str(output.dtype),
                'device': str(output.device), 'numel': output.numel()}
    if output.device.type != 'cpu' or output.layout != t.strided or output.is_quantized:
        return {**captured, 'values_omitted': 'device/layout/quantization unsupported'}
    extent = min(output.numel(), DIAGNOSTIC_ITEMS)
    sample = []
    for flat_index in range(extent):
        remainder = flat_index
        coordinates = []
        for size in reversed(output.shape):
            coordinates.append(remainder % size)
            remainder //= size
        sample.append(output.detach()[tuple(reversed(coordinates))].item())
    captured['values'] = [_diagnostic(item) for item in sample]
    captured['truncated'] = output.numel() > extent
    bad = [i for i, item in enumerate(sample)
           if type(item) is float and not math.isfinite(item)]
    captured['nonfinite_count'] = len(bad)
    captured['nonfinite_count_scope'] = 'sample' if captured['truncated'] else 'entire_tensor'
    captured['nonfinite_flat_indices'] = bad
    captured['nonfinite_indices_truncated'] = captured['truncated']
    return captured


def _collect(callback, base, expanded, rows, base_ids, expanded_ids, selected,
             permutation, shift):
    """DEV-capable collector. Public target construction is a separate gated action."""
    import torch as t
    require(callable(callback), 'own callback required')
    require(type(rows) is list and rows and len(set(rows)) == len(rows), 'unique row IDs')
    for ids in (rows, base_ids, expanded_ids, selected):
        require(type(ids) is list and all(type(x) is str and x for x in ids), 'string IDs')
    require(len(set(base_ids)) == len(base_ids) and len(set(expanded_ids)) == len(expanded_ids),
            'unique candidate IDs')
    require(expanded_ids[:len(base_ids)] == base_ids and len(selected) == len(rows)
            and all(label in base_ids for label in selected), 'original pair mapping')
    require(type(permutation) is list and all(type(i) is int for i in permutation)
            and sorted(permutation) == list(range(len(expanded_ids))), 'exact permutation')
    number(shift)
    for tensor, ids in ((base, base_ids), (expanded, expanded_ids)):
        require(isinstance(tensor, t.Tensor) and tuple(tensor.shape) == (len(rows), len(ids)),
                'input shape')
        require(tensor.device.type == 'cpu' and tensor.dtype == t.float64
                and bool(t.isfinite(tensor).all()), 'finite CPU float64 inputs required')
    require(t.equal(base, expanded[:, :len(base_ids)]), 'original entries must remain exact')
    inputs = [base, expanded, base, expanded[:, permutation], base + shift, expanded + shift]
    ids_by_condition = [base_ids, expanded_ids, base_ids,
                        [expanded_ids[i] for i in permutation], base_ids, expanded_ids]
    captured = []
    for name, original, ids in zip(CONDITIONS, inputs, ids_by_condition):
        outputs = {}
        for kind in KINDS:
            argument = original.detach().clone()
            before = _input_identity(argument)
            require(before['sha256'] is not None, 'input identity byte bound')
            attempted_return = {'tag': 'not_returned'}
            primary = None
            secondary = []
            after = None
            try:
                output = callback(argument, kind)
                # A malformed return is evidence even if validation rejects it.
                try:
                    attempted_return = _capture_output(output)
                except BaseException as capture_failure:
                    secondary.append(_error_info(capture_failure))
                    attempted_return = {'tag': 'capture_failed',
                                        'return_type': _type_name(output)}
                require(type(output) is t.Tensor and output.shape == original.shape,
                        'output shape')
                require(output.device.type == 'cpu' and output.dtype == t.float64,
                        'output must preserve CPU float64')
                require(bool(t.isfinite(output).all()), 'nonfinite output')
                require(bool(((output >= 0) & (output <= 1)).all()), 'output outside [0,1]')
                outputs[kind] = output.detach().clone().tolist()
                if secondary:
                    primary = secondary.pop(0)  # incomplete evidence cannot succeed
            except BaseException as exc:
                primary = _error_info(exc)
            finally:
                try:
                    after = _input_identity(argument)
                    require(after == before, 'learner mutated input dtype/shape/device/bytes')
                except BaseException as exc:
                    info = _error_info(exc)
                    if primary is None:
                        primary = info
                    else:
                        secondary.append(info)
            if primary is not None:
                details = {'condition': name, 'kind': kind, 'row_ids': list(rows),
                           'candidate_ids': list(ids), 'primary': primary, 'secondary': secondary,
                           'attempted_return': attempted_return, 'input_before': before,
                           'input_after': after, 'completed_conditions': deepcopy(captured),
                           'completed_outputs_in_condition': deepcopy(outputs)}
                raise CohortFailure(primary['message'], details)
        captured.append({'name': name, 'candidate_ids': list(ids), 'outputs': outputs})
    try:
        summary = _summarize(captured, rows, base_ids, expanded_ids, selected, permutation)
    except Exception as exc:
        raise CohortFailure(str(exc), {'condition': 'controls', 'kind': 'both',
                            'row_ids': list(rows), 'candidate_ids': list(expanded_ids),
                            'completed_conditions': captured,
                            'completed_outputs_in_condition': {}}) from exc
    return {'row_ids': list(rows), 'conditions': captured, **summary}


def _observation(value, record):
    keys(value, ('kind', 'result', 'runtime', 'signed_error', 'absolute_error'))
    require(value['kind'] == 'fresh_cpu_analytic', 'observation class')
    case = record['identity']['protocol']
    result = value['result']
    keys(result, ('row_ids', 'conditions', 'selected', 'per_row_changes', 'delta', 'sigmoid_delta'))
    require(result['row_ids'] == case['row_ids'], 'complete cohort required')
    expected = _summarize(result['conditions'], case['row_ids'], case['base_ids'],
                          case['candidate_ids'], case['selected_ids'], case['permutation'])
    for key, item in expected.items():
        require(canonical(result[key]) == canonical(item), 'stored summary mismatch')
    require(canonical(value['runtime']) == canonical(record['identity']['runtime']),
            'runtime receipt mismatch')
    error = result['delta'] - record['commitment']['prediction']
    number(error)
    number(value['signed_error'])
    number(value['absolute_error'])
    require(value['signed_error'] == error and value['absolute_error'] == abs(error),
            'forecast error mismatch; signed=observed-predicted')


def _check_failure_reserve(record):
    base = dict(record)
    base.update(status='committed', observation=None, reflection=None)
    require(len(_seal(base).encode('utf-8')) <= COMMITMENT_LIMIT,
            'source/commitment must reserve failure receipt capacity')


def read_record(raw):
    record = parse(raw)
    keys(record, ('schema', 'version', 'status', 'identity', 'commitment',
                  'observation', 'reflection', 'seal'))
    require(record['schema'] == SCHEMA and type(record['version']) is int
            and record['version'] == VERSION, 'schema/version')
    require(record['seal'] == digest({k: v for k, v in record.items() if k != 'seal'}),
            'record seal mismatch')
    _identity(record['identity'])
    _commitment(record['commitment'])
    _check_failure_reserve(record)
    status = record['status']
    require(status in ('committed', 'succeeded', 'failed', 'reflected'), 'record status')
    if status == 'committed':
        require(record['observation'] is None and record['reflection'] is None, 'unrun commitment')
    elif status == 'failed':
        failure = record['observation']
        keys(failure, ('kind', 'error_type', 'message', 'details'))
        require(failure['kind'] == 'failed_cpu_attempt', 'failure kind')
        text(failure['error_type'])
        text(failure['message'])
        details = failure['details']
        keys(details, ('primary', 'secondary', 'persistence'))
        _validate_error_info(details['primary'])
        require(failure['message'] == details['primary']['message']
                and failure['error_type'] == details['primary']['error_type'], 'primary mismatch')
        require(type(details['secondary']) is list and len(details['secondary']) <= 16,
                'bounded secondary failures')
        for item in details['secondary']:
            _validate_error_info(item)
        require(details['persistence'] in ('persisted', 'fallback_persisted', 'memory_only'),
                'failure persistence state')
        require(record['reflection'] is None, 'failed attempt has no success reflection')
    else:
        _observation(record['observation'], record)
        if status == 'succeeded':
            require(record['reflection'] is None, 'unexpected reflection')
        else:
            reflection = record['reflection']
            keys(reflection, ('parent_seal', 'explanation', 'changed_condition',
                              'prediction', 'confidence', 'reasoning'))
            parent = deepcopy(record)
            parent['status'] = 'succeeded'
            parent['reflection'] = None
            require(reflection['parent_seal'] == parse(_seal(parent))['seal'], 'reflection parent')
            text(reflection['explanation'])
            text(reflection['changed_condition'])
            _forecast({key: reflection[key] for key in ('prediction', 'confidence', 'reasoning')})
    return deepcopy(record)


def replay(receipt_text):
    record = read_record(receipt_text)
    require(record['status'] in ('succeeded', 'reflected', 'failed'), 'no observation yet')
    return deepcopy(record['observation'])


def reveal_methods(commitment_text):
    read_record(commitment_text)
    opening = '<details><summary>Public solution (seen exposure)</summary>\n\n```python\n'
    return opening + REFERENCE + '```\n</details>'


def _account_home():
    if SESSION_ANCHOR is not None:
        return Path(SESSION_ANCHOR).resolve()
    import pwd
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def _consume_attempt():
    directory = _account_home() / '.local/state/arena-mastery/clip_prediction' / SESSION
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / (sha(PROTOCOL_ID.encode()) + '.started.json')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(canonical({'protocol': PROTOCOL_ID, 'session': SESSION,
                                'status': 'started_unknown_until_receipt', 'time_ns': time.time_ns()}))
    return path


def _execute(record):
    """Future target body; unreachable through run_experiment while BV03 is deferred."""
    import torch as t
    t.set_num_threads(1)
    require(t.get_num_threads() == 1, 'one CPU thread required')
    require(runtime_identity() == record['identity']['runtime'], 'worker runtime drift')
    namespace = {'t': t, 'torch': t}
    source = record['identity']['method_source']
    validate_source(source)
    exec(compile(source, '<inline CLIP learner method>', 'exec'), namespace)
    case = record['identity']['protocol']
    expanded = t.tensor(case['raw_logits'], dtype=t.float64, device='cpu')
    return _collect(namespace['zero_shot_probabilities'], expanded[:, :3], expanded,
                    case['row_ids'], case['base_ids'], case['candidate_ids'],
                    case['selected_ids'], case['permutation'], case['shift'])


def _worker(record, connection):
    try:
        connection.send({'ok': True, 'result': _execute(record)})
    except BaseException as exc:
        # The error envelope contains only bounded builtin JSON-safe data.
        connection.send({'ok': False, 'failure': _error_info(exc)})
    finally:
        connection.close()


class AttemptFailure(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(read_record(receipt)['observation']['message'])


def _failure_receipt(record, primary, secondary, persistence):
    """Build and validate a complete failure; no observation survives a failed run."""
    failed = deepcopy(record)
    failed['status'] = 'failed'
    failed['reflection'] = None
    failed['observation'] = {
        'kind': 'failed_cpu_attempt', 'error_type': primary['error_type'],
        'message': primary['message'],
        'details': {'primary': primary, 'secondary': secondary, 'persistence': persistence},
    }
    raw = _seal(failed)
    read_record(raw)
    return raw


def _compact_failure_info(info):
    """An independently bounded fallback; valid huge diagnostics are not copied."""
    try:
        _validate_error_info(info)
        result = {key: deepcopy(info[key]) for key in
                  ('error_type', 'error_type_capture', 'message', 'message_capture')}
        details = info['details']
        extent = len(canonical(details).encode('utf-8'))
        result['details'] = (deepcopy(details) if extent <= FALLBACK_DETAILS_BYTES else
                             {'tag': 'truncated', 'reason': 'fallback_limit',
                              'captured_bytes': extent})
    except BaseException:
        result = _error_info(RuntimeError('Primary failure diagnostics unavailable'),
                             details={'original_failure': info})
        if type(info) is dict:
            for field in ('error_type', 'message'):
                original = info.get(field)
                if type(original) is str and original:
                    fallback = 'UnnamedErrorType' if field == 'error_type' else result['error_type']
                    result[field] = _summary_text(original, fallback)
                    capture = 'error_type_capture' if field == 'error_type' else 'message_capture'
                    result[capture] = _safe_diagnostic(original)
    _validate_error_info(result)
    # With independently bounded text/codepoint captures, 12 KB UTF-8 summaries,
    # 64 KB maximum message capture and 8 KB details, this stays below 128 KB.
    require(len(canonical(result).encode('utf-8')) <= FALLBACK_INFO_BYTES,
            'compact failure envelope bound')
    return result


def _secondary_omission_count(info):
    """Recognize our explicit history summary without discarding its old count."""
    if (type(info) is not dict or info.get('error_type') != 'RuntimeError'
            or info.get('message') != 'Additional secondary failures truncated'):
        return None
    details = info.get('details')
    if (type(details) is not dict or details.get('tag') != 'mapping'
            or type(details.get('items')) is not list or details.get('length') != 2
            or details.get('truncated') is not False):
        return None
    counts = {}
    for pair in details['items']:
        if type(pair) is not list or len(pair) != 2:
            return None
        key, value = pair
        if (type(key) is not dict or key.get('tag') != 'text'
                or type(value) is not dict or value.get('tag') != 'scalar'):
            return None
        name, count = key.get('value'), value.get('value')
        if type(name) is not str or type(count) is not int or count < 0:
            return None
        counts[name] = count
    if (set(counts) != {'retained_secondary_count', 'omitted_secondary_count'}
            or counts['retained_secondary_count'] > 3 or counts['omitted_secondary_count'] == 0):
        return None
    return counts['omitted_secondary_count']


def _retained_failure_state(raw):
    """A validated receipt is the sole state for every subsequent rebuild."""
    retained = read_record(raw)
    require(retained['status'] == 'failed', 'retained failure state required')
    details = retained['observation']['details']
    return details['primary'], details['secondary']


def _best_failure_receipt(record, primary, secondary, persistence):
    try:
        return _failure_receipt(record, primary, secondary, persistence)
    except BaseException as sealing:
        fallback = _compact_failure_info(primary)
        visible = []
        omitted = 0
        for item in secondary:
            prior_omitted = _secondary_omission_count(item)
            if prior_omitted is None:
                visible.append(item)
            else:
                omitted += prior_omitted
        omitted += max(0, len(visible) - 3)
        retained = [_compact_failure_info(item) for item in visible[:3]]
        # This sealing failure happened after the existing causes, not before.
        retained.append(_compact_failure_info(_error_info(sealing)))
        if omitted:
            retained.append(_error_info(RuntimeError('Additional secondary failures truncated'),
                            details={'retained_secondary_count': min(3, len(visible)),
                                     'omitted_secondary_count': omitted}))
        # At most six <128 KB infos fit the separately reserved 1 MB envelope.
        # No source snapshot or commitment is removed, and LIMIT remains 4 MB.
        return _failure_receipt(record, fallback, retained, persistence)


def run_experiment(commitment_text, method_source):
    require(TARGET_EXECUTION_AUTHORIZED, 'BV03 target execution remains deferred')
    record = read_record(commitment_text)
    require(record['status'] == 'committed', 'fresh committed record required')
    require(canonical(record['identity']) == canonical(freeze_inputs(method_source)),
            'source/runtime/protocol drift')
    marker = _consume_attempt()  # Before learner definition evaluation or target construction.
    response = None
    secondary = []
    process = parent = child = None
    try:
        context = multiprocessing.get_context('fork')
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(record, child))
        process.start()
        child.close()
        if parent.poll(30):
            response = parent.recv()
        else:
            raise TimeoutError('30-second outer deadline; attempt remains consumed; no retry')
    except BaseException as exc:
        response = {'ok': False, 'failure': _error_info(exc,
                    details={'partial_results': 'unknown; no automatic retry'})}
    finally:
        # Cleanup errors are secondary to a callback/process error, and cannot
        # bypass receipt construction. Each independent cleanup gets its chance.
        actions = []
        if process is not None:
            def stop():
                if process.is_alive():
                    process.terminate()
            def join():
                if process.pid is not None:
                    process.join(timeout=2)
            actions.extend((stop, join))
        actions.extend(connection.close for connection in (parent, child)
                       if connection is not None)
        for action in actions:
            try:
                action()
            except BaseException as exc:
                secondary.append(_error_info(exc))
    try:
        require(type(response) is dict and type(response.get('ok')) is bool,
                'worker response envelope')
        keys(response, ('ok', 'result') if response['ok'] else ('ok', 'failure'))
        if not response['ok']:
            _validate_error_info(response['failure'])
            primary = response['failure']
        else:
            delta = response['result']['delta']
            error = delta - record['commitment']['prediction']
            number(error)
            record['status'] = 'succeeded'
            record['observation'] = {'kind': 'fresh_cpu_analytic', 'result': response['result'],
                                     'runtime': record['identity']['runtime'],
                                     'signed_error': error, 'absolute_error': abs(error)}
            raw = _seal(record)
            read_record(raw)
            primary = secondary.pop(0) if secondary else None
    except BaseException as exc:
        primary = _error_info(exc, details={'rejected_worker_response': response})
    if primary is not None:
        raw = _best_failure_receipt(record, primary, secondary, 'persisted')
        primary, secondary = _retained_failure_state(raw)
    try:
        _exclusive(raw, marker.with_suffix('.receipt.json'))
    except BaseException as storage:
        if primary is None:
            primary = _error_info(storage, details={'rejected_success': record['observation']})
        else:
            secondary.append(_error_info(storage))
        raw = _best_failure_receipt(record, primary, secondary, 'fallback_persisted')
        primary, secondary = _retained_failure_state(raw)
        try:
            # Recovery writes evidence only; it never retries the experiment.
            _exclusive(raw, marker.with_suffix('.fallback.receipt.json'))
        except BaseException as fallback_storage:
            secondary.append(_error_info(fallback_storage))
            raw = _best_failure_receipt(record, primary, secondary, 'memory_only')
            primary, secondary = _retained_failure_state(raw)
        raise AttemptFailure(raw) from storage
    if primary is not None:
        raise AttemptFailure(raw)
    return raw


def _exclusive(raw, path):
    path = Path(path).expanduser().resolve()
    require(not path.is_relative_to(ROOT.resolve()), 'private export must be outside repository')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(raw)
    return path


def export_record(raw, destination):
    read_record(raw)
    return _exclusive(raw, destination)


def export_attempt(raw, destination):
    require(read_record(raw)['status'] == 'reflected', 'reflection required')
    return _exclusive(raw, destination)


def reflect(receipt_text, explanation, changed_condition, prediction, confidence, reasoning):
    record = read_record(receipt_text)
    require(record['status'] == 'succeeded', 'complete success required')
    record['reflection'] = {'parent_seal': record['seal'], 'explanation': explanation,
                            'changed_condition': changed_condition, 'prediction': prediction,
                            'confidence': confidence, 'reasoning': reasoning}
    record['status'] = 'reflected'
    raw = _seal(record)
    read_record(raw)
    return raw


def plot_receipt(receipt_text):
    record = read_record(receipt_text)
    require(record['status'] in ('succeeded', 'reflected'), 'complete success required')
    import matplotlib.pyplot as plt
    result = record['observation']['result']
    fig, panels = plt.subplots(2, 3, figsize=(12, 7))
    axes = list(panels.flat)
    positions = list(range(len(result['row_ids'])))
    for axis, kind in zip(axes[:2], KINDS):
        for offset, condition in ((-0.18, 'baseline'), (0.18, 'expansion')):
            axis.bar([x + offset for x in positions], result['selected'][condition][kind],
                     width=0.36, label=condition)
        axis.set_xticks(positions, result['row_ids'])
        axis.set_title('Softmax weight' if kind == 'softmax' else 'Sigmoid score')
        axis.set_ylim(0, 1)
        axis.legend()
    axes[2].imshow(record['identity']['protocol']['raw_logits'], aspect='auto')
    axes[2].set_title('Public input logits')
    axes[2].set_xticks(range(5), record['identity']['protocol']['candidate_ids'])
    axes[2].set_yticks(positions, result['row_ids'])
    observed = result['delta']
    predicted = record['commitment']['prediction']
    axes[3].bar(['Observed softmax', 'Observed sigmoid'],
                [observed, result['sigmoid_delta']])
    axes[3].scatter([0], [predicted], color='black', marker='x', s=70, label='Forecast')
    axes[3].axhline(0, color='gray', linewidth=0.7)
    axes[3].set_ylabel('Expanded minus baseline (weight/score units)')
    axes[3].legend()
    for kind, offset in (('softmax', -0.18), ('sigmoid', 0.18)):
        changes = [b - a for a, b in zip(result['selected']['baseline'][kind],
                                        result['selected']['shifted_baseline'][kind])]
        axes[4].bar([x + offset for x in positions], changes, width=0.36, label=kind)
    axes[4].set_xticks(positions, result['row_ids'])
    axes[4].set_title('Common +1 shift: designated-pair changes')
    axes[4].legend()
    axes[5].axis('off')
    axes[5].text(0, 1, 'All mandatory controls passed:\n'
                '• unchanged-baseline identity\n• coherent candidate-ID permutation\n'
                '• fixed-pair sigmoid independence\n• softmax common-shift invariance\n'
                '• all 12 complete finite readouts\n\n'
                f'Signed forecast error: {observed - predicted:.6g}\n'
                'Synthetic fixed logits; no native probability claim.', va='top')
    fig.suptitle('Public analytic candidate-set forecast; stored observations only')
    fig.tight_layout()
    return fig


REFERENCE = '''def zero_shot_probabilities(logits, kind):
    if kind == "softmax":
        return logits.softmax(dim=-1)
    if kind == "sigmoid":
        return logits.sigmoid()
    raise ValueError(f"Unknown kind {kind!r}")
'''


def build_cells():
    """Build source text only. No torch import, callback, target array or observation."""
    cells = []
    def add(kind, name, source):
        cell = {'cell_type': kind, 'id': 'clip-' + name, 'metadata': {}, 'source': source}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', 'question', '''# Candidate sets change a softmax weight

How much does the **mean softmax weight of four designated original pairs** change
when two public candidate columns are added? Compare independent sigmoid scores.
This is a fixed-logit analytic exercise, not calibrated native probabilities or a
comparison of full trained CLIP/SigLIP models. TRAIN: none; DEV: public distinct
fixtures; TEST: none. Public source and formulas are exposed. Record prior exposure.
Target construction/execution is currently **disabled pending BV03**.

By the end, implement both readouts, distinguish competition from independent
pair scoring, commit an exposure-aware numerical forecast, and interpret controls.
Cold open: imagine matching a red-bicycle query against three captions. Add two
extra captions while keeping every old pair logit fixed. Should the old bicycle
pair lose softmax weight? Must its independent score change? Explain your causal
prediction first; the public four-row exercise asks how large the mean change is.
''')
    add('code', 'setup', 'from mastery_hub.units.clip_prediction import lesson\nimport torch as t\n')
    add('markdown', 'inputs', '## Public protocol (plaintext, no target tensors)\n\n```json\n'
        + (UNIT / 'protocol.json').read_text() + '\n```\n\nDelta uses weight units: '
        '0.01 is one percentage point. All four rows remain in order. '
        'Signed forecast error is observed Delta minus predicted Delta. '
        'The input rule is public, but no target score or plot has been computed here.')
    add('code', 'manual', '''# Complete this before opening solutions or measuring the target.
manual_forecast = dict(prediction=None, confidence=None, reasoning='', revision='')
prior_exposure = None  # Choose 'seen', 'unknown', or 'self_reported_not_seen'.
# In reasoning, disclose prior target/reference/output or related-solution exposure.
''')
    add('markdown', 'exercise', '''## Implement the readout (about five minutes)

Implement the original `zero_shot_probabilities(logits, kind)` interface.
`kind='softmax'`: normalize over candidate columns in each row.
`kind='sigmoid'`: transform each pair independently. Reject an unknown kind.
Output keeps input shape, CPU device and float64 dtype. No reference fallback.

<details><summary>Help</summary>
The last axis contains candidates. Sigmoid has no candidate-sum denominator.
</details>

Expected immediate-test outputs: shape `(2, 3)`, softmax row sums one, all-zero
sigmoid row equal to one half, and unknown kind raises `ValueError`.
''')
    add('code', 'method', '''METHOD_SOURCE = """def zero_shot_probabilities(logits, kind):
    raise NotImplementedError("Implement both readouts")
"""
# This is your editable method. The explicit DEV cell below defines and calls it.
''')
    add('code', 'immediate-tests', '''# Explicit tiny DEV only: these are not the public four-row target.
lesson.validate_source(METHOD_SOURCE)
namespace = {'t': t, 'torch': t}
exec(compile(METHOD_SOURCE, '<my CLIP DEV method>', 'exec'), namespace)
my_readout = namespace['zero_shot_probabilities']
x_dev = t.tensor([[0., 0., 0.], [1., -2., 0.5]], dtype=t.float64, device='cpu')
y_soft = my_readout(x_dev.clone(), 'softmax')
y_sig = my_readout(x_dev.clone(), 'sigmoid')
t.testing.assert_close(y_soft, x_dev.softmax(-1))
t.testing.assert_close(y_sig, x_dev.sigmoid())
assert y_soft.shape == y_sig.shape == x_dev.shape
t.testing.assert_close(y_soft.sum(-1), t.ones(2, dtype=t.float64))
t.testing.assert_close(y_sig[0], t.full((3,), 0.5, dtype=t.float64))
try:
    my_readout(x_dev.clone(), 'unknown')
except ValueError:
    pass
else:
    raise AssertionError('Unknown kind must raise ValueError')
print('Immediate DEV tests passed; target remains unexecuted.')
''')
    add('markdown', 'commit-help', '''## Commit your completed method and numerical forecast

Keep the original manual forecast. If DEV feedback changed your prediction,
confidence or reason, explain the revision explicitly. A software commitment hashes
your actual method, public protocol and recoverable source snapshots. It does not
prove an adversarial chronology or unseen mastery. Changing source requires a new
commitment, but does **not** reset an already-started session attempt.
''')
    add('code', 'commit', '''prediction = manual_forecast['prediction']
confidence = manual_forecast['confidence']
reasoning = manual_forecast['reasoning']
identity = lesson.freeze_inputs(METHOD_SOURCE)
commitment = lesson.commit_prediction(
    identity, METHOD_SOURCE, prediction, confidence, reasoning, prior_exposure,
    manual_forecast=manual_forecast,
)
''')
    add('code', 'solution', '''# Optional public solution after commitment; never installs code.
from IPython.display import Markdown, display
display(Markdown(lesson.reveal_methods(commitment)))
''')
    add('markdown', 'controls', '''## Future run and visible result

Six conditions × two readouts = 12 calls: baseline, expansion, repeated baseline,
coherent candidate permutation, shifted baseline, shifted expansion. Every condition
and control passes before a mean or accepted figure. Identity and permutation must
preserve results; sigmoid must preserve fixed-pair scores under expansion; softmax
must preserve rows under a common shift. Sigmoid's shift response is a diagnostic.
Expected future signature: paired per-query before/after bars, mean changes, forecast
and input heatmap. No numerical target answer is supplied. A wrong prediction is
useful evidence, not a failed engineering test. Whole-cohort failures stay visible.
''')
    add('code', 'future-run', '''receipt = None  # Clear stale success before prerequisites or running.
# BV03 is deferred. After explicit authority and reviewed enabling change only:
# receipt = lesson.run_experiment(commitment, METHOD_SOURCE)
# Failed attempts raise AttemptFailure with its preserved .receipt; never auto-retry.
''')
    add('code', 'future-display', '''if receipt is not None:
    lesson.plot_receipt(receipt)
''')
    add('markdown', 'interpretation', '''<details><summary>Interpretation and limits</summary>
Adding alternatives changes a softmax denominator without changing original logits.
An independent sigmoid has no such denominator. This algebra does not show that a
trained SigLIP is calibrated, that CLIP ignores objects, or that either model has
better retrieval accuracy. Read the existing 12.1 loss/readout discussion and its
paper connections: [CLIP](https://arxiv.org/abs/2103.00020) and
[SigLIP](https://arxiv.org/abs/2303.15343). Existing real-image results remain historical, separate evidence.
</details>

## Reflect, then Try It Yourself
Explain the largest mismatch and what the controls rule out. Propose changing the
candidate set or common logit shift; commit a new prediction before changing inputs.
This first companion does not authorize or execute a second fixed target. Keep all
old commitments and failures. Replay and private export never run the learner.
''')
    add('code', 'reflection', '''# Try It Yourself: make this prospective prediction editable before any new run.
play_hypothesis = dict(changed_condition='', prediction=None, confidence=None,
                       reasoning='')
# No changed-input construction or normalization is authorized by this play cell.
# reflection = lesson.reflect(receipt, explanation='', changed_condition='',
#                             prediction=None, confidence=None, reasoning='')
# lesson.export_record(receipt, '/absolute/private/path/new-receipt.json')
# lesson.export_attempt(reflection, '/absolute/private/path/new-reflection.json')
# replayed = lesson.replay(receipt)  # Stored observations only, no method execution.
''')
    return cells
