"""Independent gradient/game checks and publication evidence rejection checks."""
import copy
from pathlib import Path

import pytest
import torch

from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run import inrun_methods as methods
from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run import inrun_tests as checks
from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run import inrun_report as report


def test_independent_gradient_oracle():
    checks.test_forward_and_factors(methods.mlp_forward, methods.mlp_factors, methods.per_example_gradients)
    checks.test_ghost_products(methods.ghost_dot_products)


def test_fixed_denominator_exact_local_game():
    checks.test_fixed_denominator(methods.sgd_step)
    checks.test_local_game(methods.local_coalition_values)
    checks.test_batch_shapley(methods.batch_shapley)


def test_taylor_self_cross_and_complete_trajectory():
    checks.test_taylor_allocations(methods.first_order_scores, methods.second_order_scores)
    checks.test_accounting_trajectory(methods.run_accountant)


def test_numeric_comparison_rejects_changes_and_unknown_timing_fields():
    tensor={'tensor':True,'dtype':'torch.float64','shape':[2],'values':[1.,2.]}
    payload={'digit_runs':{seed:{'run':{'wall_seconds':1.}} for seed in report.SEEDS},
             'timing_rows':[{'seconds':1.} for _ in range(14)],
             'training_times':[{'seconds':1.} for _ in range(6)], 'actual':tensor}
    fresh=copy.deepcopy(payload);fresh['timing_rows'][0]['seconds']=2.
    counts, excluded=report.compare(payload,fresh)
    assert counts['tensor_elements']==2 and len(excluded)==23
    fresh['actual']['values'][0]+=1e-12
    with pytest.raises(ValueError,match='Tensor changed'):report.compare(payload,fresh)
    fresh=copy.deepcopy(payload);fresh['other_wall_seconds']=2.
    with pytest.raises(ValueError,match='Keys/order changed'):report.compare(payload,fresh)
    fresh=copy.deepcopy(payload);fresh['actual']['dtype']='torch.float32'
    with pytest.raises(ValueError,match='Tensor changed'):report.compare(payload,fresh)


def test_invalid_tensor_shape_and_nonfinite_rejected():
    with pytest.raises(ValueError,match='shape'):report.array({'tensor':True,'shape':[3],'values':[1.,2.]})
    with pytest.raises(ValueError,match='Nonfinite'):report.array({'tensor':True,'shape':[1],'values':[float('nan')]})
