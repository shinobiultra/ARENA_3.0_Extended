"""5.3 Mamba from scratch: tiny CPU primitive oracles and wrong-answer rejection."""
from __future__ import annotations

import importlib

import pytest
import torch as t

MODULE = 'chapter5_modern_architectures.exercises.part3_mamba_from_scratch'


@pytest.fixture(scope='module')
def lesson():return importlib.import_module(MODULE+'.solutions')
@pytest.fixture(scope='module')
def checks():return importlib.import_module(MODULE+'.tests')

@pytest.mark.parametrize('name,args',[
('test_continuous_discretization_exact',['stable_continuous_A','discretize_scalar_ssm_exact']),
('test_sequential_affine_scan_manual',['sequential_affine_scan']),
('test_compose_affine_updates',['compose_affine_updates']),
('test_parallel_affine_scan_parity',['sequential_affine_scan','parallel_affine_scan']),
('test_parallel_scan_gradients',['parallel_affine_scan']),
('test_causal_depthwise_conv_manual_and_no_future',['causal_depthwise_conv1d']),
('test_project_selective_parameters_is_input_dependent',['project_selective_parameters']),
('test_selective_discretization_shapes_and_stability',['discretize_selective_scan']),
('test_selective_scan_single_step_manual',['selective_scan_recurrent']),
('test_recurrent_parallel_selective_scan_parity',['selective_scan_recurrent','selective_scan_parallel']),
('test_mamba_block_has_norm_residual_and_gate',['MambaBlockFromScratch']),
('test_mamba_block_recurrent_step_parity',['MambaBlockFromScratch']),
('test_tiny_stacked_mamba_lm_and_cache',['MambaForCausalLMFromScratch']),
('test_state_intervention_is_causal',['intervene_on_state']),
('test_selective_parameters_literal',['project_selective_parameters','discretize_selective_scan','_expand_bc']),
('test_rms_norm_literal',['MambaRMSNorm']),
('test_mixer_literal_oracle',['MambaMixerFromScratch','MambaConfig']),
('test_mixer_cache_storage',['MambaMixerFromScratch','MambaConfig']),
('test_weight_mapping_without_checkpoint',['hf_to_local_key','load_hf_mamba_weights']),
('test_parity_assembly_without_checkpoint',['pinned_mamba_cpu_parity']),
('test_event_controls_literal',['build_event_coefficients','selective_readout','intervene_on_state','causal_attention_memory','finite_causal_conv_memory']),
('test_delay_benchmark_literal',['run_memory_delay_benchmark']),
])
def test_independent_cpu_primitives(lesson,checks,name,args):getattr(checks,name)(*[getattr(lesson,arg) for arg in args])


def test_bad_independent_answers_are_rejected(checks,lesson):
    with pytest.raises(AssertionError):checks.test_sequential_affine_scan_manual(lambda a,b,initial_state=None:b)
    with pytest.raises(AssertionError):checks.test_weight_mapping_without_checkpoint(lambda key:'wrong',lesson.load_hf_mamba_weights)
    with pytest.raises(AssertionError):checks.test_event_controls_literal(lambda events,values:(t.ones(1,5,1,dtype=t.float64),)*3,lesson.selective_readout,lesson.intervene_on_state,lesson.causal_attention_memory,lesson.finite_causal_conv_memory)


def test_effective_step_is_the_recurrent_step(lesson):
    assert lesson.MambaMixerFromScratch.step is lesson.mamba_recurrent_step
