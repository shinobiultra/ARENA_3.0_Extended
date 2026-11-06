"""Regression checks for immediate learner oracles, including plausible wrong methods."""
from importlib import import_module

import numpy as np
import pytest

CODES = 'chapter17_training_dynamics.exercises.part3_finetuning_diffs_emergence'
PROX = 'chapter18_intrinsic_interpretability.exercises.part2_sparse_modular_kans'


def shrink(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0)


def test_reference_iterative_methods_pass_strengthened_immediate_oracles():
    for package, method in [(CODES, 'infer_codes'), (PROX, 'proximal_fit')]:
        reference = import_module(package + '.solutions')
        checks = import_module(package + '.tests')
        getattr(checks, 'test_' + method)(getattr(reference, method))


def test_general_dictionary_oracle_rejects_orthogonal_closed_form():
    def wrong(x, dictionary, penalty=.05, steps=200):
        norms = np.square(dictionary).sum(axis=1)
        if not norms.any():
            return np.zeros((len(x), len(dictionary)))
        return shrink(x @ dictionary.T, penalty) / norms
    with pytest.raises(AssertionError):
        import_module(CODES + '.tests').test_infer_codes(wrong)


@pytest.mark.parametrize('mutation', ['ignore_steps', 'extra_step', 'diagonal_gradient'])
def test_general_dictionary_oracle_rejects_wrong_iterations(mutation):
    def wrong(x, dictionary, penalty=.05, steps=200):
        rate = 1 / np.linalg.norm(dictionary, 2) ** 2 if dictionary.any() else 0
        codes = np.zeros((len(x), len(dictionary)))
        if mutation == 'ignore_steps':
            steps = 200
        if mutation == 'extra_step':
            steps += 1
        for _ in range(steps):
            if mutation == 'diagonal_gradient':
                gradient = codes * np.square(dictionary).sum(axis=1) - x @ dictionary.T
            else:
                gradient = (codes @ dictionary - x) @ dictionary.T
            codes = shrink(codes - rate * gradient, rate * penalty)
        return codes
    with pytest.raises(AssertionError):
        import_module(CODES + '.tests').test_infer_codes(wrong)


def test_proximal_oracle_rejects_whitened_closed_form():
    def wrong(features, targets, costs, strength, steps=80, learning_rate=.5):
        return shrink(np.asarray(features).T @ targets / len(features), strength * np.asarray(costs))
    with pytest.raises(AssertionError):
        import_module(PROX + '.tests').test_proximal_fit(wrong)


@pytest.mark.parametrize('mutation', ['ignore_rate', 'extra_step', 'output_denominator', 'unscaled_threshold'])
def test_proximal_oracle_rejects_wrong_updates(mutation):
    def wrong(features, targets, costs, strength, steps=80, learning_rate=.5):
        x, y = np.asarray(features), np.asarray(targets)
        weights = np.zeros((x.shape[1], y.shape[1]))
        if mutation == 'ignore_rate':
            learning_rate = .5
        if mutation == 'extra_step':
            steps += 1
        for _ in range(steps):
            denominator = len(x) * (y.shape[1] if mutation == 'output_denominator' else 1)
            gradient = x.T @ (x @ weights - y) / denominator
            threshold = strength * np.asarray(costs)
            if mutation != 'unscaled_threshold':
                threshold = learning_rate * threshold
            weights = shrink(weights - learning_rate * gradient, threshold)
        return weights
    with pytest.raises(AssertionError):
        import_module(PROX + '.tests').test_proximal_fit(wrong)
