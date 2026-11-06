"""5.4 Mamba-2 and SSD: tiny CPU oracles, wrong-answer rejection and direct operator fixtures."""
from __future__ import annotations

import importlib
import math

import pytest
import torch as t

MODULE = 'chapter5_modern_architectures.exercises.part4_mamba2_ssd'


@pytest.fixture(scope='module')
def lesson():return importlib.import_module(MODULE+'.solutions')
@pytest.fixture(scope='module')
def checks():return importlib.import_module(MODULE+'.tests')


@pytest.mark.parametrize('name,args',[
('test_segment_sums_exact',['segment_sums']),
('test_expand_grouped_parameters',['expand_grouped_parameters']),
('test_semiseparable_matrix_hand_oracle',['semiseparable_matrix']),
('test_recurrent_ssd_hand_oracle',['ssd_recurrent']),
('test_quadratic_ssd_parity_and_causality',['ssd_recurrent','ssd_quadratic']),
('test_chunked_ssd_parity_state_and_gradients',['ssd_recurrent','ssd_chunked']),
('test_mamba2_block_shape_and_causality',['Mamba2BlockFromScratch','Mamba2Config']),
('test_mamba2_step_cache_parity',['Mamba2BlockFromScratch','Mamba2Config']),
('test_delayed_recall_batch_semantics',['make_delayed_recall_batch']),
('test_recall_training_updates_parameters',['train_recall_model']),
('test_recall_orchestration',['run_delayed_recall_experiment']),
('test_scaling_storage_without_benchmark',['measure_ssd_scaling','ScalingResult']),
('test_exact_metrics_on_hand_oracle',['exact_parity_metrics']),
('test_released_mapping_on_synthetic_tensors',['native_to_released_state_dict']),
('test_released_gate_order',['ReleasedGatedRMSNorm']),
('test_released_mixer_on_tiny_zero_projection',['ReleasedMamba2MixerFromScratch','ReleasedMamba2Config']),
('test_released_mixer_literal_scalar_oracle',['ReleasedMamba2MixerFromScratch','ReleasedMamba2Config']),
('test_released_lm_on_tiny_configuration',['ReleasedMamba2LMFromScratch','ReleasedMamba2Config']),
('test_checkpoint_parity_assembly_without_weights',['verify_released_mamba2_parity','ReleasedCheckpointParity']),
('test_tiny_recall_carry_intervention',['TinyMamba2RecallModel']),
('test_recurrent_quadratic_chunked_gradient_parity',['exact_gradient_parity_metrics']),
('test_tiny_stacked_lm_full_vs_recurrent_cache',['TinyStackedMamba2LM']),
('test_parameter_matched_attention_and_mamba1_controls',['parameter_matched_control_counts','TinyCausalAttentionControl','TinyMamba1Control']),
('test_cuda_hooks_are_strict',[]),
])
def test_independent_cpu_fixtures(lesson,checks,name,args):
    getattr(checks,name)(*[getattr(lesson,arg) for arg in args])


def test_incorrect_independent_answers_are_rejected(lesson,checks):
    with pytest.raises(AssertionError):checks.test_recurrent_ssd_hand_oracle(lambda x,*args,**kwargs:(x,x[:,-1]))
    class WrongGate(t.nn.Module):
        def __init__(self,*args,**kwargs):super().__init__();self.weight=t.nn.Parameter(t.ones(2))
        def forward(self,x,gate):return x*self.weight
    with pytest.raises(AssertionError):checks.test_released_gate_order(WrongGate)
    with pytest.raises(AssertionError):checks.test_released_mapping_on_synthetic_tensors(lambda v:v)
    with pytest.raises(AssertionError):checks.test_scaling_storage_without_benchmark(lambda *args,**kw:lesson.ScalingResult([5,8,9],[7.]*3,[7.]*3,[7.]*3,[0.]*3,[0.]*3))
    with pytest.raises(AssertionError):checks.test_exact_metrics_on_hand_oracle(lambda **kwargs:{})


def test_decay_controls_match_direct_operator_fixtures(lesson):
    x = t.tensor([2., 3., 5.], dtype=t.float64).reshape(1, 3, 1, 1)
    bc = t.ones(1, 3, 1, 1, dtype=t.float64)
    for decay in (.25, .5, .9):
        a = t.full((1, 3, 1), math.log(decay), dtype=t.float64)
        expected = t.tensor([2., 2 * decay + 3., (2 * decay + 3) * decay + 5], dtype=t.float64)
        t.testing.assert_close(lesson.ssd_quadratic(x, a, bc, bc).flatten(), expected, atol=1e-12, rtol=0)
        for q in (1, 2, 4):
            y, state = lesson.ssd_chunked(x, a, bc, bc, q)
            t.testing.assert_close(y.flatten(), expected, atol=1e-12, rtol=0)
            assert abs(state.item() - expected[-1]) < 1e-12
        for carry in (0., .5, 1.):
            y, state = lesson.ssd_recurrent(x, a, bc, bc, carry_scale=carry)
            value = 0.
            expected_values = []
            for token in (2., 3., 5.):
                value = carry * decay * value + token
                expected_values.append(value)
            t.testing.assert_close(y.flatten(), t.tensor(expected_values, dtype=t.float64), atol=1e-12, rtol=0)
            assert abs(state.item() - value) < 1e-12
