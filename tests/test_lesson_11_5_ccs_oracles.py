"""11.5 Contrast-consistent search: tiny CPU oracles run against the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = 'chapter11_representation_geometry.exercises.part5_contrast_consistent_search'


# Independent hand-worked, NumPy-gradient, AdamW-single-restart and library-SVD oracles.
ORACLES = [
    ('ccs_methods_tests', 'test_pair_centering', ['fit_pair_centering', 'apply_pair_centering']),
    ('ccs_methods_tests', 'test_ccs_loss_values', ['ccs_loss']),
    ('ccs_methods_tests', 'test_ccs_loss_gradient', ['ccs_loss']),
    ('ccs_methods_tests', 'test_paired_inference', ['paired_probabilities']),
    ('ccs_methods_tests', 'test_two_feature_counterexample', ['paired_probabilities', 'ccs_loss']),
    ('ccs_methods_tests', 'test_batched_restarts', ['fit_ccs_restarts']),
    ('ccs_methods_tests', 'test_restart_selection', ['select_restart']),
    ('ccs_methods_tests', 'test_orientation_and_metrics', ['calibrate_orientation', 'binary_metrics']),
    ('ccs_methods_tests', 'test_pca_distractor', ['fit_difference_pca']),
    ('ccs_native_methods_tests', 'test_last_token_rows', ['last_token_rows']),
    ('ccs_native_methods_tests', 'test_paired_opinion_change', ['paired_opinion_change']),
    ('ccs_native_methods_tests', 'test_model_answer_metrics', ['model_answer_metrics']),
]


@pytest.mark.parametrize('test_module,name,functions', ORACLES)
def test_visible_generated_functions_against_independent_cpu_oracles(test_module, name, functions):
    torch = importlib.import_module('torch')
    methods = importlib.import_module(MODULE + '.solutions')
    tests = importlib.import_module(MODULE + '.' + test_module)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        getattr(tests, name)(*[getattr(methods, function) for function in functions])
    finally:
        torch.set_default_dtype(previous)
