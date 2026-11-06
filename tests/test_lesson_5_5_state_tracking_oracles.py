"""5.5 State tracking in SSMs and Transformers: untrained, CPU-only primitive checks on the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = 'chapter5_modern_architectures.exercises.part5_state_tracking'


@pytest.fixture
def primitives(monkeypatch):
    import torch as t
    t.set_num_threads(1)
    module=importlib.import_module(MODULE+'.solutions')
    state=t.get_rng_state();dtype=t.get_default_dtype()
    def forbidden(*a,**k):raise AssertionError('No optimizer,CUDA,pretrained or model study authorized')
    monkeypatch.setattr(t.optim,'AdamW',forbidden)
    monkeypatch.setattr(t.cuda,'init',forbidden)
    monkeypatch.setattr(module,'train_state_tracker_cpu',forbidden)
    monkeypatch.setattr(module,'run_gpu_test',forbidden)
    monkeypatch.setattr(module,'run_full_experiment',forbidden)
    t.set_default_dtype(t.float32)
    yield module
    t.set_rng_state(state);t.set_default_dtype(dtype)

@pytest.mark.parametrize('seed',[0,1])
def test_bounded_bracket_primitive(primitives,seed):
    import torch as t
    m=primitives;b=m.generate_bracket_depth_task(4,8,max_depth=3,seed=seed)
    assert b.tokens.shape==b.states.shape==(4,8)
    t.testing.assert_close(b.states,t.where(b.tokens.bool(),1,-1).cumsum(-1))
    assert int(b.states.min())>=0 and int(b.states.max())<=3
    from chapter5_modern_architectures.exercises.part5_state_tracking.tests import _reference_brackets
    t.testing.assert_close(b.states,_reference_brackets(4,8,max_depth=3,seed=seed).states)

def test_literal_probe_primitive(primitives):
    import torch as t
    labels=t.tensor([[0,1,2,3],[3,2,1,0]]).repeat(4,1)
    x=t.nn.functional.one_hot(labels,4).float()
    probe=primitives.fit_state_probe(x,labels,ridge=.001)
    assert primitives.state_probe_accuracy(x,labels,probe)==1.
    assert probe.mean.shape==probe.scale.shape==(4,)

def test_handcrafted_anomaly_primitive(primitives):
    tests=importlib.import_module(MODULE+'.tests')
    tests.test_find_confident_errors_orders_real_mistakes(primitives.find_confident_errors)

def test_untrained_cached_forward_primitive(primitives):
    tests=importlib.import_module(MODULE+'.tests')
    tests.test_collect_recurrent_states_matches_full_forward(primitives.collect_recurrent_states)

def test_untrained_matched_transplant_primitive(primitives):
    tests=importlib.import_module(MODULE+'.tests')
    tests.test_state_transplant_matches_donor_dynamics(primitives.run_state_transplant,primitives.make_matched_state_pair)
