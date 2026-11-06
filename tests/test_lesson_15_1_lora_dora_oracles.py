"""15.1 LoRA and DoRA from scratch: tiny CPU oracles and wrong-method rejection on the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = 'chapter15_peft_misalignment.exercises.part1_lora_dora'


@pytest.fixture(scope='module')
def lesson():return importlib.import_module(MODULE+'.solutions')


@pytest.mark.parametrize('name,args',[
('test_lora_delta_uses_scaled_b_matrix_times_a_matrix',['lora_delta']),
('test_adapter_delta_report_records_rank_alpha_and_nonzero_update',['adapter_delta_report']),
('test_lora_delta_rank_bound_and_svd_spectrum',['lora_delta']),
('test_lora_merge_unmerge_parity_is_independent_of_report',['lora_merge_max_abs_diff']),
('test_dora_recompose_weight_preserves_target_row_magnitudes',['dora_recompose_weight','dora_weight_report']),
('test_dora_recomposition_handles_nonzero_delta_direction',['dora_recompose_weight']),
('test_intruder_dimension_report_measures_projection_fraction',['intruder_dimension_report']),
('test_adapter_mechanism_report_requires_accuracy_and_mechanism',['adapter_mechanism_report']),
('test_proxy_batch_has_planted_target_and_ood_conflict',['make_proxy_batch']),
('test_evaluation_on_known_weights',['evaluate_proxy_adapter','ProxyFullFinetuneClassifier']),
('test_same_norm_control_without_training',['same_norm_random_adapter_control','ProxyLoRAClassifier']),
('test_signature_gate_without_training',['build_signature_payload']),
])
def test_local_independent_cpu_oracles(lesson,name,args):
    checks=importlib.import_module(MODULE+'.tests')
    getattr(checks,name)(*[getattr(lesson,x) for x in args])


def test_oracles_reject_nonzero_and_high_accuracy_but_wrong_methods(lesson):
    import torch as t
    from types import SimpleNamespace
    checks=importlib.import_module(MODULE+'.tests')
    with pytest.raises(AssertionError):checks.test_lora_delta_uses_scaled_b_matrix_times_a_matrix(lambda a,b,alpha:t.zeros(2,2))
    with pytest.raises(AssertionError):checks.test_dora_recomposition_handles_nonzero_delta_direction(lambda w,d,m:(w+d)*m[:,None])
    with pytest.raises(AssertionError):checks.test_adapter_mechanism_report_requires_accuracy_and_mechanism(lambda **kw:SimpleNamespace(accuracy_delta=.2,mechanism_delta=.05,accuracy_improved=True,mechanism_preserved=True,adapter_acceptable=True))
    with pytest.raises(AssertionError):checks.test_intruder_dimension_report_measures_projection_fraction(lambda *a,**kw:SimpleNamespace(projection_fraction=0.0,intruder_detected=False))
    wrong = type(lesson.build_signature_payload)((lambda **kw: {'signature_passed': True}).__code__, {'compare_proxy_methods': None})
    with pytest.raises(AssertionError):checks.test_signature_gate_without_training(wrong)
