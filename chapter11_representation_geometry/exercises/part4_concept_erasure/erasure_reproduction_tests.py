"""Reject missing controls, omitted rows and substituted native evidence."""
import copy
import pytest
from .erasure_reproduction import validate_summary, expected_conditions
from . import erasure_data as data


def complete_summary():
    conditions = expected_conditions()
    return dict(
        status='PASS', weights_unchanged=True, model_gradients_absent=True,
        model_id=data.MODEL_ID, revision=data.REVISION,
        conditions=106, final_role_rows=288, final_object_rows=384, final_outcomes=71232,
        condition_tensor_sha256={filename: 'hash-placeholder' for _, _, filename in conditions},
        final=[dict(site=site, method=name, tensor_file=filename,
                    final=dict(rows=288, native_hook_max_error=0.),
                    object_final=dict(rows=384, native_hook_max_error=0.))
               for site, name, filename in conditions],
    )


def test_complete_native_inventory():
    validate_summary(complete_summary())


@pytest.mark.parametrize('mutation', ['missing_control', 'duplicate', 'relabel', 'omit_rows',
    'wrong_model', 'wrong_revision', 'changed_weights', 'gradients', 'hook_error', 'missing_tensor', 'wrong_total'])
def test_native_inventory_mutations_rejected(mutation):
    summary = copy.deepcopy(complete_summary())
    if mutation == 'missing_control':
        summary['final'].pop()
    elif mutation == 'duplicate':
        summary['final'][-1] = summary['final'][0]
    elif mutation == 'relabel':
        summary['final'][0]['method'] = 'Only favorable examples'
    elif mutation == 'omit_rows':
        summary['final'][0]['final']['rows'] = 287
    elif mutation == 'wrong_model':
        summary['model_id'] = 'another-model'
    elif mutation == 'wrong_revision':
        summary['revision'] = 'main'
    elif mutation == 'changed_weights':
        summary['weights_unchanged'] = False
    elif mutation == 'gradients':
        summary['model_gradients_absent'] = False
    elif mutation == 'hook_error':
        summary['final'][0]['object_final']['native_hook_max_error'] = .001
    elif mutation == 'missing_tensor':
        summary['condition_tensor_sha256'].pop(next(iter(summary['condition_tensor_sha256'])))
    elif mutation == 'wrong_total':
        summary['final_outcomes'] = 71231
    with pytest.raises(ValueError):
        validate_summary(summary)
