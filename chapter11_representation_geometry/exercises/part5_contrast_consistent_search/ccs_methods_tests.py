"""Immediate mathematical and optimization checks for the visible CCS methods."""

import inspect
import math

import numpy as np
import torch
from torch.testing import assert_close

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_methods as methods


def exact_pairs(distractor_strength=3.0):
    """Every combination is present; the target and distractor are independent."""
    target, distractor, content = torch.tensor(
        [(y, s, u) for y in (-1, 1) for s in (-1, 1) for u in (-1, 1)], dtype=torch.float64
    ).T
    plus = torch.stack((torch.full_like(target, 8), target, distractor_strength * distractor, content), -1)
    minus = torch.stack((torch.full_like(target, -8), -target, -distractor_strength * distractor, content), -1)
    return plus, minus, (target > 0).long(), (distractor > 0).long()


def test_pair_centering(fit=methods.fit_pair_centering, apply=methods.apply_pair_centering):
    plus, minus, _, _ = exact_pairs()
    centering = fit(plus, minus)
    p, m = apply(plus, minus, centering)
    assert_close(p.mean(0), torch.zeros(4, dtype=p.dtype))
    assert_close(m.mean(0), torch.zeros(4, dtype=p.dtype))
    assert_close(p[:, 0], torch.zeros(8, dtype=p.dtype))
    # A new distribution keeps its shift; silently re-centering it would hide it.
    held_p, held_m = apply(plus[:2] + 4, minus[:2] + 7, centering)
    assert_close(held_p, plus[:2] + 4 - plus.mean(0))
    assert_close(held_m, minus[:2] + 7 - minus.mean(0))
    assert_close(plus[:, 0], torch.full((8,), 8.0, dtype=plus.dtype))
    print('All tests in test_pair_centering passed!')


def test_ccs_loss_values(loss=methods.ccs_loss):
    p = torch.tensor([1.0, 0.0, 0.5, 0.0, 1.0, 0.4], dtype=torch.float64)
    m = torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0, 0.4], dtype=torch.float64)
    total, consistency, confidence = loss(p[None], m[None])
    assert_close(total, torch.tensor([0.0, 0.0, 0.25, 1.0, 2.0, 0.2], dtype=p.dtype))
    assert_close(total, consistency + confidence)
    p, m = torch.tensor([0.8], dtype=p.dtype), torch.tensor([0.6], dtype=p.dtype)
    # Complementing arbitrary soft scores is not a symmetry of the original loss.
    assert_close(loss(p, m)[0], torch.tensor(0.52, dtype=p.dtype))
    assert_close(loss(1 - p, 1 - m)[0], torch.tensor(0.20, dtype=p.dtype))
    print('All tests in test_ccs_loss_values passed!')


def test_ccs_loss_gradient(loss=methods.ccs_loss):
    # Independent NumPy derivative, deliberately away from minimum's tie points.
    plus = np.array([[1.0, 2.0], [-1.0, 0.5], [0.3, -0.7]])
    minus = np.array([[-2.0, 0.2], [1.4, -0.3], [-0.9, 1.3]])
    parameters = np.array([0.6, -0.4, 0.2])
    sigmoid = lambda a: 1 / (1 + np.exp(-a))
    p, m = sigmoid(plus @ parameters[:2] + parameters[2]), sigmoid(minus @ parameters[:2] + parameters[2])
    error = p + m - 1
    dp = (2 * error + 2 * p * (p < m)) * p * (1 - p) / len(p)
    dm = (2 * error + 2 * m * (m < p)) * m * (1 - m) / len(m)
    expected = np.concatenate((plus.T @ dp + minus.T @ dm, [dp.sum() + dm.sum()]))
    w = torch.tensor(parameters, requires_grad=True)
    tplus, tminus = torch.tensor(plus), torch.tensor(minus)
    actual = loss((tplus @ w[:2] + w[2]).sigmoid(), (tminus @ w[:2] + w[2]).sigmoid())[0]
    actual.backward()
    assert_close(w.grad, torch.tensor(expected), atol=1e-12, rtol=1e-12)
    assert (
        torch.autograd.gradcheck(lambda v: loss((tplus @ v[:2] + v[2]).sigmoid(), (tminus @ v[:2] + v[2]).sigmoid())[0], (w,))
    ), 'Differentiate both branches, average over examples and keep the original asymmetric confidence term.'
    print('All tests in test_ccs_loss_gradient passed!')


def test_paired_inference(infer=methods.paired_probabilities):
    plus = torch.tensor([[2.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    minus = torch.tensor([[-1.0, 2.0], [0.0, 1.0]], dtype=torch.float64)
    weight, bias = torch.tensor([0.4, -0.3], dtype=plus.dtype), torch.tensor(0.2, dtype=plus.dtype)
    score, p, m = infer(plus, minus, weight, bias)
    assert_close(score, (torch.sigmoid(plus @ weight + bias) + torch.sigmoid(-(minus @ weight + bias))) / 2)
    assert_close(infer(plus, minus, -weight, -bias)[0], 1 - score)
    assert_close(infer(minus, plus, weight, bias)[0], 1 - score)
    assert_close(p, torch.sigmoid(plus @ weight + bias))
    assert_close(m, torch.sigmoid(minus @ weight + bias))
    print('All tests in test_paired_inference passed!')


def test_two_feature_counterexample(infer=methods.paired_probabilities, loss=methods.ccs_loss):
    plus, minus, target, distractor = exact_pairs()
    plus, minus = methods.apply_pair_centering(plus, minus, methods.fit_pair_centering(plus, minus))
    scores = []
    for weight in ([0.0, 15.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]):
        score, p, m = infer(plus, minus, torch.tensor(weight, dtype=plus.dtype), torch.tensor(0.0, dtype=plus.dtype))
        assert (
            loss(p, m)[0] < 1e-12
        ), 'The target and independent distractor must both satisfy consistency; only one predicts the target.'
        scores.append(score)
    assert (
        ((scores[0] > 0.5) == target).all()
    ), 'The target and independent distractor must both satisfy consistency; only one predicts the target.'
    assert (
        ((scores[1] > 0.5) == distractor).all()
    ), 'The target and independent distractor must both satisfy consistency; only one predicts the target.'
    assert (
        float(((scores[1] > 0.5) == target).double().mean()) == 0.5
    ), 'The target and independent distractor must both satisfy consistency; only one predicts the target.'
    # Before separate centering, the answer-token axis also attains low loss.
    plus, minus, _, _ = exact_pairs()
    score, p, m = infer(plus, minus, torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=plus.dtype), torch.tensor(0.0))
    assert (
        loss(p, m)[0] < 1e-12
    ), 'The target and independent distractor must both satisfy consistency; only one predicts the target.'
    assert (
        float(((score > 0.5) == target).double().mean()) == 0.5
    ), 'The target and independent distractor must both satisfy consistency; only one predicts the target.'
    print('All tests in test_two_feature_counterexample passed!')


def test_batched_restarts(fit=methods.fit_ccs_restarts):
    plus, minus, _, _ = exact_pairs()
    plus, minus = methods.apply_pair_centering(plus, minus, methods.fit_pair_centering(plus, minus))
    original_plus, original_minus = plus.clone(), minus.clone()
    before = torch.random.get_rng_state().clone()
    batched = fit(plus, minus, seeds=[5, 17, 29], steps=40)
    assert_close(torch.random.get_rng_state(), before)
    assert (
        batched['history'].shape == (41, 3, 3)
    ), 'Each restart needs separate optimizer state, a final post-update loss and no label input.'
    for i, seed in enumerate(batched["seeds"]):
        single = fit(plus, minus, seeds=[seed], steps=40)
        assert_close(batched["weights"][i], single["weights"][0], atol=1e-11, rtol=1e-11)
        assert_close(batched["biases"][i], single["biases"][0], atol=1e-11, rtol=1e-11)
        assert_close(batched["history"][:, i], single["history"][:, 0], atol=1e-11, rtol=1e-11)
    _, p, m = methods.paired_probabilities(plus, minus, batched["weights"], batched["biases"])
    assert_close(batched["history"][-1, :, 0], methods.ccs_loss(p, m)[0])
    assert (
        (batched['history'][-1, :, 0] < batched['history'][0, :, 0]).all()
    ), 'Each restart needs separate optimizer state, a final post-update loss and no label input.'
    assert_close(plus, original_plus)
    assert_close(minus, original_minus)
    assert (
        all(('label' not in name for name in inspect.signature(fit).parameters))
    ), 'Each restart needs separate optimizer state, a final post-update loss and no label input.'
    print('All tests in test_batched_restarts passed!')


def test_restart_selection(select=methods.select_restart):
    history = torch.tensor([[[0.3, 0.1, 0.2], [0.1, 0.05, 0.05], [0.1, 0.0, 0.1]]])
    fits = {"history": history, "weights": torch.eye(3), "biases": torch.arange(3.0), "seeds": [9, 6, 1]}
    chosen = select(fits)
    assert (
        chosen['index'] == 1 and chosen['seed'] == 6
    ), 'Select the first minimum final training loss and return a copy of the chosen parameters.'
    assert_close(chosen["weight"], torch.tensor([0.0, 1.0, 0.0]))
    chosen["weight"][0] = 99
    assert (
        fits['weights'][1, 0] == 0
    ), 'Select the first minimum final training loss and return a copy of the chosen parameters.'
    print('All tests in test_restart_selection passed!')


def test_orientation_and_metrics(orient=methods.calibrate_orientation, evaluate=methods.binary_metrics):
    calibration_scores = torch.tensor([0.8, 0.1, 0.9, 0.2], dtype=torch.float64)
    calibration_labels = torch.tensor([0, 1, 0, 1])
    orientation = orient(calibration_scores, calibration_labels)
    assert (
        orientation == -1
    ), 'Calibrate the sign once; final labels must not change primary accuracy or tie handling.'
    # The final set prefers the opposite orientation; the primary metric must not switch.
    result = evaluate(calibration_scores, 1 - calibration_labels, orientation)
    assert (
        result['accuracy'] == 0 and result['test_oracle_orientation_accuracy'] == 1
    ), 'Calibrate the sign once; final labels must not change primary accuracy or tie handling.'
    tied = torch.full((4,), 0.5, dtype=torch.float64)
    assert (
        orient(tied, calibration_labels) == 1
    ), 'Calibrate the sign once; final labels must not change primary accuracy or tie handling.'
    tied_result = evaluate(tied, calibration_labels)
    assert (
        tied_result['accuracy'] == 0.5
    ), 'Calibrate the sign once; final labels must not change primary accuracy or tie handling.'
    assert (
        math.isclose(tied_result['cross_entropy'], math.log(2))
    ), 'Calibrate the sign once; final labels must not change primary accuracy or tie handling.'
    assert (
        tied_result['brier'] == 0.25
    ), 'Calibrate the sign once; final labels must not change primary accuracy or tie handling.'
    assert (
        math.isfinite(evaluate(torch.tensor([0.0, 1.0]), torch.tensor([1, 0]))['cross_entropy'])
    ), 'Calibrate the sign once; final labels must not change primary accuracy or tie handling.'
    print('All tests in test_orientation_and_metrics passed!')


def test_pca_distractor(fit=methods.fit_difference_pca):
    plus, minus, _, _ = exact_pairs()
    result = fit(plus, minus)
    assert_close(result["weight"], torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=plus.dtype))
    assert_close(result["singular_values"][0].square(), torch.tensor(8 * 36.0, dtype=plus.dtype))
    print('All tests in test_pca_distractor passed!')


def test_invalid_inputs():
    import pytest

    with pytest.raises(ValueError, match="broadcasting"):
        methods.ccs_loss(torch.full((3, 1), 0.5), torch.full((3,), 0.5))
    with pytest.raises(ValueError, match="finite"):
        methods.ccs_loss(torch.tensor([float("nan")]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="variance"):
        methods.fit_difference_pca(torch.ones(3, 2), torch.ones(3, 2))
    with pytest.raises(ValueError, match="unique"):
        methods.fit_ccs_restarts(torch.ones(3, 2), torch.ones(3, 2), seeds=[1, 1])
    print('All tests in test_invalid_inputs passed!')


def run_all_tests():
    checks = [function for name, function in globals().items() if name.startswith("test_") and callable(function)]
    for check in checks:
        check()
        print(f"{check.__name__}: passed")
    return len(checks)


if __name__ == "__main__":
    torch.set_num_threads(4)
    print(f"{run_all_tests()} CCS method checks passed")
