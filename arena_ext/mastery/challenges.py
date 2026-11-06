"""Immutable, prediction-first local self-study transitions (not exam security)."""
from __future__ import annotations

from copy import deepcopy
import math
import re
from uuid import uuid4

# Declared before examples are tuned; these describe transfer, not achieved mastery.
HELD_OUT_TEMPLATES = (
    {'id': 'curvature-between', 'change': 'unseen curvature inside explored range', 'transfer': 'interpolation'},
    {'id': 'curvature-beyond', 'change': 'curvature outside explored range', 'transfer': 'extrapolation'},
    {'id': 'starting-direction', 'change': 'unseen initial eigendirection', 'transfer': 'extrapolation'},
    {'id': 'noise-regime', 'change': 'noiseless to noisy gradient', 'transfer': 'extrapolation'},
    {'id': 'update-convention', 'change': 'explicit alternative recurrence in a future lesson', 'transfer': 'extrapolation'},
)
ATTEMPT_KEYS = frozenset({'attempt_id', 'challenge_id', 'config_fingerprint', 'status', 'assessment',
                          'prediction', 'confidence', 'reasoning', 'observed', 'score', 'score_type'})


def validate_identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise ValueError('identifier must contain only letters, digits, underscores or hyphens')
    return value


def validate_fingerprint(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('configuration fingerprint must be a lowercase SHA-256 digest')
    return value


def json_value(value):
    """Validate JSON-native values without permitting object serialization hooks."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            json_value(item)
        return
    raise ValueError('expected finite JSON-native values')


def _probability(value):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError('confidence/probability must be finite and in [0, 1]')


def brier(p: float, y: int) -> float:
    _probability(p)
    if type(y) not in (int, bool) or y not in (0, 1):
        raise ValueError('binary outcome must be 0 or 1')
    return float((p - y) ** 2)


def validate_attempt(attempt: dict) -> dict:
    if type(attempt) is not dict or set(attempt) != ATTEMPT_KEYS:
        raise ValueError('attempt has missing or unknown fields')
    json_value(attempt)
    for field in ('attempt_id', 'challenge_id'):
        validate_identifier(attempt[field])
    validate_fingerprint(attempt['config_fingerprint'])
    status = attempt['status']
    if status not in ('uncommitted', 'committed', 'revealed', 'invalidated'):
        raise ValueError('unknown attempt status')
    if attempt['assessment'] not in ('practice', 'unseen'):
        raise ValueError('unknown assessment category')
    if not isinstance(attempt['reasoning'], str):
        raise ValueError('reasoning must be text')
    if status == 'uncommitted':
        if any(attempt[k] is not None for k in ('prediction', 'confidence', 'observed', 'score')) or attempt['reasoning']:
            raise ValueError('uncommitted attempt cannot contain an answer or outcome')
    elif attempt['confidence'] is not None:
        _probability(attempt['confidence'])
    elif status in ('committed', 'revealed'):
        raise ValueError('committed prediction requires confidence')
    if status in ('committed', 'revealed') and (attempt['prediction'] is None or not attempt['reasoning'].strip()):
        raise ValueError('committed attempt requires prediction and reasoning')
    if status != 'revealed':
        if attempt['observed'] is not None or attempt['score'] is not None or attempt['score_type'] != 'unscored':
            raise ValueError('only revealed attempts contain outcomes or scores')
    elif type(attempt['prediction']) is bool and type(attempt['observed']) is bool and attempt['score_type'] != 'brier':
        raise ValueError('binary outcomes require Brier scoring')
    elif attempt['score_type'] == 'brier':
        if type(attempt['prediction']) is not bool or type(attempt['observed']) is not bool:
            raise ValueError('Brier scoring requires binary predictions and outcomes')
        p = attempt['confidence'] if attempt['prediction'] else 1 - attempt['confidence']
        if type(attempt['score']) not in (int, float) or attempt['score'] != brier(p, attempt['observed']):
            raise ValueError('score does not match prediction and observation')
    elif attempt['score_type'] != 'unscored' or attempt['score'] is not None:
        raise ValueError('unknown score type')
    return deepcopy(attempt)


def expose_case(exposed_cases, challenge_id: str, config_fingerprint: str) -> list[dict]:
    validate_identifier(challenge_id)
    validate_fingerprint(config_fingerprint)
    result = []
    for pair in exposed_cases:
        if type(pair) is not dict or set(pair) != {'challenge_id', 'config_fingerprint'}:
            raise ValueError('invalid exposed case')
        validate_identifier(pair['challenge_id'])
        validate_fingerprint(pair['config_fingerprint'])
        if pair not in result:
            result.append(deepcopy(pair))
    pair = {'challenge_id': challenge_id, 'config_fingerprint': config_fingerprint}
    if pair not in result:
        result.append(pair)
    return result


def new_attempt(challenge_id: str, config_fingerprint: str, *, exposed_cases=()) -> dict:
    expose_case(exposed_cases, challenge_id, config_fingerprint)
    practice = {'challenge_id': challenge_id, 'config_fingerprint': config_fingerprint} in exposed_cases
    return validate_attempt(dict(attempt_id=uuid4().hex, challenge_id=challenge_id,
        config_fingerprint=config_fingerprint, status='uncommitted',
        assessment='practice' if practice else 'unseen', prediction=None, confidence=None,
        reasoning='', observed=None, score=None, score_type='unscored'))


def commit_prediction(attempt, prediction, confidence: float, reasoning: str) -> dict:
    result = validate_attempt(attempt)
    if result['status'] != 'uncommitted':
        raise ValueError('prediction is already committed or attempt is invalidated')
    _probability(confidence)
    json_value(prediction)
    if prediction is None or not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError('prediction and a brief reasoning are required')
    result.update(status='committed', prediction=deepcopy(prediction), confidence=float(confidence), reasoning=reasoning)
    return validate_attempt(result)


def invalidate(attempt, config_fingerprint: str) -> dict:
    """Keep original identity/prediction; callers start a new attempt for new inputs."""
    result = validate_attempt(attempt)
    validate_fingerprint(config_fingerprint)
    if result['status'] != 'revealed' and result['config_fingerprint'] != config_fingerprint:
        result['status'] = 'invalidated'
    return result


def reveal(attempt, config_fingerprint: str, observed) -> dict:
    result = validate_attempt(attempt)
    validate_fingerprint(config_fingerprint)
    if result['config_fingerprint'] != config_fingerprint:
        raise ValueError('configuration changed; invalidate prediction and start a new attempt')
    json_value(observed)
    if result['status'] == 'revealed':
        if result['observed'] != observed:
            raise ValueError('attempt already revealed with a different outcome')
        return result
    if result['status'] != 'committed':
        raise ValueError('commit a prediction first; invalidated attempts cannot reveal')
    result.update(status='revealed', observed=deepcopy(observed))
    if type(result['prediction']) is bool and type(observed) is bool:
        p = result['confidence'] if result['prediction'] else 1 - result['confidence']
        result.update(score_type='brier', score=brier(p, observed))
    return validate_attempt(result)
