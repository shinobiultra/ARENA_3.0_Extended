"""16.10 Influence functions and TracIn: tiny CPU oracles run against the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = "chapter16_shapley_attribution_baselines.exercises.part10_influence_tracin"


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module(MODULE + ".solutions")


ORACLES = ('logistic_losses','logistic_gradients','weighted_objective','objective_derivatives',
           'logistic_hvp','conjugate_gradient','fit_weighted_logistic','upweight_influence',
           'sgd_trace','trace_loss_reductions','tracin_checkpoint_scores','ghost_linear_products','class_matched_removals')


@pytest.mark.parametrize('name',ORACLES)
def test_visible_generated_methods_against_independent_cpu_oracles(methods,name,monkeypatch):
    checks=importlib.import_module(MODULE+'.tests')
    torch=importlib.import_module('torch');previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        monkeypatch.setattr(checks,'answers',methods)
        getattr(checks,'test_'+name)(getattr(methods,name))
    finally:
        torch.set_default_dtype(previous)


def test_regularized_removal_normalization_against_independent_refits(methods,monkeypatch):
    checks=importlib.import_module(MODULE+'.tests')
    monkeypatch.setattr(checks,'answers',methods)
    checks.test_regularized_removal_normalization()
