"""Prediction transitions exercise the learner contract without a UI."""
import copy
import math
import pytest

from arena_ext.mastery.challenges import (
    HELD_OUT_TEMPLATES, brier, commit_prediction, expose_case,
    invalidate, new_attempt, reveal, validate_attempt,
)

FP = 'a' * 64
OTHER = 'b' * 64


def committed(**kwargs):
    return commit_prediction(new_attempt('stability', FP, **kwargs), True, .8, 'Roots inside unit circle.')


def test_reveal_requires_prediction_and_matching_config():
    attempt = new_attempt('stability', FP)
    with pytest.raises(ValueError, match='commit'):
        reveal(attempt, FP, True)
    pending = committed()
    with pytest.raises(ValueError, match='configuration'):
        reveal(pending, OTHER, True)
    assert reveal(pending, FP, True)['status'] == 'revealed'


def test_completed_attempt_is_immutable_and_duplicate_reveal_is_idempotent():
    pending = committed()
    before = copy.deepcopy(pending)
    result = reveal(pending, FP, True)
    assert pending == before
    assert result['score'] == pytest.approx(.04)
    assert reveal(result, FP, True) == result
    with pytest.raises(ValueError, match='already'):
        reveal(result, FP, False)
    with pytest.raises(ValueError):
        commit_prediction(result, False, .9, 'Changed mind')
    result['reasoning'] = 'Mutated copy'
    assert pending['reasoning'] == before['reasoning']


def test_explored_case_is_not_unseen():
    exposures = expose_case([], 'stability', FP)
    assert exposures == expose_case(exposures, 'stability', FP)
    assert new_attempt('stability', FP, exposed_cases=exposures)['assessment'] == 'practice'
    assert new_attempt('stability', OTHER, exposed_cases=exposures)['assessment'] == 'unseen'
    assert exposures == [{'challenge_id': 'stability', 'config_fingerprint': FP}]


def test_invalidation_retains_prediction_and_history():
    pending = committed()
    changed = invalidate(pending, OTHER)
    assert changed['status'] == 'invalidated'
    assert changed['config_fingerprint'] == FP
    assert pending['status'] == 'committed'
    with pytest.raises(ValueError, match='invalidated'):
        reveal(changed, FP, True)
    done = reveal(pending, FP, True)
    assert invalidate(done, OTHER) == done


def test_brier_reference_values():
    assert brier(1, 1) == 0
    assert brier(0, 1) == 1
    assert brier(.5, 0) == .25
    for p in (-1, 2, math.nan, math.inf, True):
        with pytest.raises(ValueError):
            brier(p, 1)
    for y in (-1, 2, .5, '1'):
        with pytest.raises(ValueError):
            brier(.5, y)


def test_confidence_validation_and_separate_metrics():
    for confidence in (-.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            commit_prediction(new_attempt('stability', FP), True, confidence, 'Because')
    result = reveal(commit_prediction(new_attempt('ranking', FP), ['a', 'b'], .5, 'Order'), FP, ['b', 'a'])
    assert result['score'] is None
    assert result['score_type'] == 'unscored'
    assert {t['transfer'] for t in HELD_OUT_TEMPLATES} == {'interpolation', 'extrapolation'}


def test_attempts_validate_ids_and_json_values():
    for name in ('../escape', '/escape', '', 'a/b'):
        with pytest.raises(ValueError):
            new_attempt(name, FP)
    with pytest.raises(ValueError):
        new_attempt('ok', 'not-a-sha256')
    bad = committed()
    bad['surprise'] = 1
    with pytest.raises(ValueError):
        validate_attempt(bad)
    with pytest.raises(ValueError):
        commit_prediction(new_attempt('ok', FP), math.nan, .5, 'Bad')


def test_duplicate_exposure_input_cannot_make_a_new_case_practice():
    exposures = [{'challenge_id':'stable', 'config_fingerprint':FP}] * 2
    assert new_attempt('other', OTHER, exposed_cases=exposures)['assessment'] == 'unseen'


def test_imported_committed_state_requires_prediction_and_reasoning():
    for field, value in (('prediction', None), ('reasoning', '')):
        bad = committed()
        bad[field] = value
        with pytest.raises(ValueError):
            validate_attempt(bad)


def test_score_schema_cannot_suppress_binary_metrics_or_use_boolean_score():
    result = reveal(committed(), FP, True)
    bad = copy.deepcopy(result)
    bad.update(score_type='unscored', score=None)
    with pytest.raises(ValueError):
        validate_attempt(bad)
    correct = reveal(commit_prediction(new_attempt('stable', FP), True, 1., 'Certain'), FP, True)
    correct['score'] = False
    with pytest.raises(ValueError):
        validate_attempt(correct)


def test_huge_integer_confidence_is_rejected_as_invalid_probability():
    with pytest.raises(ValueError):
        brier(10**1000, 1)
