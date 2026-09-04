import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.state_tracking import (
        apply_state_intervention,
        evaluate_probe_generalization,
        fit_linear_probe,
        generate_bracket_depth_task,
        generate_parity_task,
        intervention_report,
        make_position_split,
        one_hot_state_features,
        probe_accuracy,
        probe_predictions,
        random_direction_control,
    )


def test_parity_task_matches_cumulative_xor():
    batch = generate_parity_task(batch=2, seq_len=5, seed=0)
    expected = batch.tokens.cumsum(dim=-1) % 2

    assert batch.task == "parity"
    assert t.equal(batch.states, expected)


def test_bracket_depth_task_is_bounded_and_consistent():
    batch = generate_bracket_depth_task(batch=8, seq_len=20, max_depth=3, seed=1)
    deltas = t.where(batch.tokens.bool(), 1, -1)

    assert batch.states.min() >= 0
    assert batch.states.max() <= 3
    assert t.equal(batch.states, deltas.cumsum(dim=-1))


def test_linear_probe_generalizes_to_later_positions():
    task = generate_bracket_depth_task(batch=16, seq_len=12, max_depth=3, seed=2)
    features = one_hot_state_features(task.states, noise_scale=0.01, seed=3)
    train_mask, test_mask = make_position_split(task.states, train_fraction=0.5)

    probe = fit_linear_probe(features, task.states, train_mask=train_mask)
    report = evaluate_probe_generalization(features, task.states, probe, train_mask, test_mask)

    assert report.train_accuracy > 0.99
    assert report.test_accuracy > 0.99


def test_probe_predictions_match_labels_on_one_hot_states():
    task = generate_parity_task(batch=4, seq_len=6, seed=4)
    features = one_hot_state_features(task.states)
    probe = fit_linear_probe(features, task.states)

    predictions = probe_predictions(features, probe)

    assert t.equal(predictions, task.states)
    assert probe_accuracy(features, task.states, probe) == pytest.approx(1.0)


def test_state_intervention_flips_probe_prediction():
    labels = t.tensor([[0, 1]])
    features = one_hot_state_features(labels)
    probe = fit_linear_probe(features, labels)
    hidden_state = features[0, 0]

    intervened = apply_state_intervention(
        hidden_state,
        probe,
        source_state=0,
        target_state=1,
        coefficient=4.0,
    )
    prediction = probe_predictions(intervened, probe)
    report = intervention_report(
        hidden_state,
        probe,
        source_state=0,
        target_state=1,
        coefficient=4.0,
    )

    assert int(prediction.item()) == 1
    assert report.passed
    assert report.target_logit_delta > 0


def test_random_direction_control_is_not_targeted():
    labels = t.tensor([[0, 1]])
    features = one_hot_state_features(labels)
    probe = fit_linear_probe(features, labels)
    hidden_state = features[0, 0]

    random_prediction = random_direction_control(
        hidden_state,
        probe,
        target_state=1,
        coefficient=0.1,
        seed=0,
    )

    assert random_prediction == 0
