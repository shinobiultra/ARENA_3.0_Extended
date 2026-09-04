from collections.abc import Callable

import torch as t

from arena_ext import state_tracking as reference


def _solutions():
    from chapter5_modern_architectures.exercises.part4_mamba_state_tracking import (
        solutions,
    )

    return solutions


def test_generate_parity_task_matches_cumulative_xor(
    generate_parity_task: Callable | None = None,
):
    solutions = _solutions()
    generate_parity_task = generate_parity_task or solutions.generate_parity_task
    batch = generate_parity_task(batch=3, seq_len=9, seed=10)
    expected = batch.tokens.cumsum(dim=-1) % 2
    reference_batch = reference.generate_parity_task(batch=3, seq_len=9, seed=10)
    assert batch.task == "parity", "Parity batches should identify their task."
    assert batch.vocab == {0: "0", 1: "1"}, "Parity vocab should map binary tokens."
    t.testing.assert_close(
        batch.states,
        expected,
        msg="Parity state should be cumulative XOR of all tokens seen so far.",
    )
    t.testing.assert_close(
        batch.states,
        reference_batch.states,
        msg="Parity generator should match the independent reference implementation.",
    )
    print("All tests in `test_generate_parity_task_matches_cumulative_xor` passed!")


def test_generate_bracket_depth_task_is_bounded_and_consistent(
    generate_bracket_depth_task: Callable | None = None,
):
    solutions = _solutions()
    generate_bracket_depth_task = (
        generate_bracket_depth_task or solutions.generate_bracket_depth_task
    )
    batch = generate_bracket_depth_task(batch=8, seq_len=20, max_depth=3, seed=1)
    deltas = t.where(batch.tokens.bool(), 1, -1)
    reference_batch = reference.generate_bracket_depth_task(
        batch=8,
        seq_len=20,
        max_depth=3,
        seed=1,
    )
    assert int(batch.states.min().item()) >= 0, "Bracket depth should never go negative."
    assert int(batch.states.max().item()) <= 3, "Bracket depth should respect max_depth."
    t.testing.assert_close(
        batch.states,
        deltas.cumsum(dim=-1),
        msg="Bracket states should equal the cumulative sum of open/close deltas.",
    )
    t.testing.assert_close(
        batch.states,
        reference_batch.states,
        msg="Bracket-depth generator should match the independent reference path.",
    )
    print("All tests in `test_generate_bracket_depth_task_is_bounded_and_consistent` passed!")


def test_one_hot_state_features_shape_noise_and_reference(
    one_hot_state_features: Callable | None = None,
):
    solutions = _solutions()
    one_hot_state_features = one_hot_state_features or solutions.one_hot_state_features
    states = t.tensor([[0, 1, 3], [2, 3, 0]])
    features = one_hot_state_features(states, num_states=4)
    noisy = one_hot_state_features(states, num_states=4, noise_scale=0.01, seed=5)
    reference_noisy = reference.one_hot_state_features(
        states,
        num_states=4,
        noise_scale=0.01,
        seed=5,
    )
    assert features.shape == (2, 3, 4), (
        f"Expected one-hot features with shape (2, 3, 4), got {tuple(features.shape)}."
    )
    t.testing.assert_close(
        features.argmax(dim=-1),
        states,
        msg="Noise-free one-hot features should decode exactly to the latent states.",
    )
    assert not t.equal(noisy, features), "Nonzero noise_scale should perturb the features."
    t.testing.assert_close(
        noisy,
        reference_noisy,
        msg="Noisy feature generation should be deterministic and match the reference.",
    )
    print("All tests in `test_one_hot_state_features_shape_noise_and_reference` passed!")


def test_make_position_split_masks_are_ordered_and_disjoint(
    make_position_split: Callable | None = None,
):
    solutions = _solutions()
    make_position_split = make_position_split or solutions.make_position_split
    states = t.zeros(2, 10, dtype=t.long)
    train_mask, test_mask = make_position_split(states, train_fraction=0.6)
    assert train_mask.shape == states.shape, "Train mask should match the state tensor."
    assert test_mask.shape == states.shape, "Test mask should match the state tensor."
    assert bool((train_mask & test_mask).sum().item() == 0), (
        "Train and test masks should be disjoint."
    )
    assert bool((train_mask | test_mask).all().item()), (
        "Train and test masks should cover every position."
    )
    assert bool(train_mask[:, :6].all().item()), "Train split should use early positions."
    assert bool(test_mask[:, 6:].all().item()), "Test split should use later positions."
    print("All tests in `test_make_position_split_masks_are_ordered_and_disjoint` passed!")


def test_fit_linear_probe_recovers_held_out_one_hot_states(
    fit_linear_probe: Callable | None = None,
    one_hot_state_features: Callable | None = None,
    make_position_split: Callable | None = None,
    evaluate_probe_generalization: Callable | None = None,
    probe_predictions: Callable | None = None,
):
    solutions = _solutions()
    fit_linear_probe = fit_linear_probe or solutions.fit_linear_probe
    one_hot_state_features = one_hot_state_features or solutions.one_hot_state_features
    make_position_split = make_position_split or solutions.make_position_split
    evaluate_probe_generalization = (
        evaluate_probe_generalization or solutions.evaluate_probe_generalization
    )
    probe_predictions = probe_predictions or solutions.probe_predictions
    task = solutions.generate_bracket_depth_task(batch=16, seq_len=12, max_depth=3, seed=2)
    features = one_hot_state_features(task.states, noise_scale=0.01, seed=3)
    train_mask, test_mask = make_position_split(task.states, train_fraction=0.5)
    probe = fit_linear_probe(features, task.states, train_mask=train_mask)
    report = evaluate_probe_generalization(features, task.states, probe, train_mask, test_mask)
    predictions = probe_predictions(features, probe)
    assert report.train_accuracy > 0.99, (
        f"Probe should recover train-position states; got {report.train_accuracy:.3f}."
    )
    assert report.test_accuracy > 0.99, (
        f"Probe should generalize to held-out positions; got {report.test_accuracy:.3f}."
    )
    t.testing.assert_close(
        predictions,
        task.states,
        msg="Closed-form probe should decode every noisy one-hot state in this control.",
    )
    print("All tests in `test_fit_linear_probe_recovers_held_out_one_hot_states` passed!")


def test_probe_intervention_flips_decoded_state_with_random_control(
    intervention_report: Callable | None = None,
    random_direction_control: Callable | None = None,
):
    solutions = _solutions()
    intervention_report = intervention_report or solutions.intervention_report
    random_direction_control = random_direction_control or solutions.random_direction_control
    labels = t.tensor([[0, 1]])
    features = solutions.one_hot_state_features(labels)
    probe = solutions.fit_linear_probe(features, labels)
    hidden_state = features[0, 0]
    report = intervention_report(
        hidden_state,
        probe,
        source_state=0,
        target_state=1,
        coefficient=4.0,
    )
    random_prediction = random_direction_control(
        hidden_state,
        probe,
        target_state=1,
        coefficient=0.1,
        seed=0,
    )
    assert report.source_prediction == 0, "The source hidden state should decode as state 0."
    assert report.passed, (
        f"Probe-derived intervention should flip to target state; report={report}."
    )
    assert report.target_logit_delta > 0, "Target-state logit should increase."
    assert random_prediction == report.source_prediction, (
        "Small matched random-direction control should not flip the decoded state."
    )
    print("All tests in `test_probe_intervention_flips_decoded_state_with_random_control` passed!")


def test_tiny_mamba_state_classifier_forward_shapes(TinyMambaStateClassifier: type | None = None):
    solutions = _solutions()
    TinyMambaStateClassifier = TinyMambaStateClassifier or solutions.TinyMambaStateClassifier
    model = TinyMambaStateClassifier(num_states=4).eval()
    input_ids = t.tensor([[1, 0, 1, 1, 0], [1, 1, 0, 0, 1]])
    with t.inference_mode():
        logits, hidden_states = model(input_ids, return_hidden_states=True)
    assert logits.shape == (2, 5, 4), (
        f"Classifier logits should have shape (batch, seq, num_states), got {tuple(logits.shape)}."
    )
    assert hidden_states.shape[:2] == input_ids.shape, (
        "Hidden states should preserve batch and sequence dimensions."
    )
    print("All tests in `test_tiny_mamba_state_classifier_forward_shapes` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["parity_task"]["matches_cumulative_xor"], (
        "Smoke test should verify parity labels exactly."
    )
    assert result["bracket_depth"]["bounded"], (
        "Smoke test should verify bracket-depth bounds."
    )
    assert result["bracket_depth"]["consistent"], (
        "Smoke test should verify bracket-depth recurrence consistency."
    )
    assert result["probe_generalization"]["test_accuracy"] > 0.99, (
        "Smoke test should verify held-out-position probe generalization."
    )
    assert result["intervention"]["passed"], (
        "Smoke test should verify probe-derived latent-state intervention."
    )
    assert result["parity_probe"]["predictions_match"], (
        "Smoke test should verify the parity probe control."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_mamba_state_tracker_shapes(TinyMambaStateClassifier: type | None = None):
    solutions = _solutions()
    TinyMambaStateClassifier = (
        TinyMambaStateClassifier or solutions.TinyMambaStateClassifier
    )
    model = TinyMambaStateClassifier(num_states=4).cpu().eval()
    tokens = t.tensor([[1, 0, 1, 1, 0], [1, 1, 0, 0, 1]])
    with t.inference_mode():
        logits, hidden = model(tokens, return_hidden_states=True)
    assert logits.shape == (2, 5, 4), "return one depth logit vector per token"
    assert hidden.shape[:2] == tokens.shape, "preserve batch and sequence axes"
    assert logits.device.type == "cpu", "the learner path must remain CPU runnable"
    print("All tests in `test_mamba_state_tracker_shapes` passed!")


def test_cpu_training_reduces_loss(train_state_tracker_cpu: Callable | None = None):
    solutions = _solutions()
    train_state_tracker_cpu = train_state_tracker_cpu or solutions.train_state_tracker_cpu
    model, losses = train_state_tracker_cpu(
        steps=12,
        batch_size=32,
        seq_len=16,
        max_depth=3,
        seed=17,
    )
    assert len(losses) == 12, "record one loss per optimizer step"
    assert all(float("-inf") < loss < float("inf") for loss in losses), (
        "training losses must remain finite"
    )
    assert losses[-1] < losses[0] - 0.25, (
        f"loss should fall during the semantic smoke run; got {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
    assert next(model.parameters()).device.type == "cpu"
    print("All tests in `test_cpu_training_reduces_loss` passed!")


def test_collect_recurrent_states_matches_full_forward(
    collect_recurrent_states: Callable | None = None,
):
    solutions = _solutions()
    collect_recurrent_states = collect_recurrent_states or solutions.collect_recurrent_states
    t.manual_seed(3)
    model = solutions.TinyMambaStateClassifier(num_states=4).cpu().eval()
    tokens = solutions.generate_bracket_depth_task(
        batch=3,
        seq_len=9,
        max_depth=3,
        seed=4,
    ).tokens
    with t.inference_mode():
        full_logits = model(tokens)
        cached_logits, recurrent_states = collect_recurrent_states(model, tokens)
    t.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-5)
    expected_features = model.backbone.config.d_inner * model.backbone.config.d_state
    assert recurrent_states.shape == (3, 9, expected_features)
    print("All tests in `test_collect_recurrent_states_matches_full_forward` passed!")


def test_state_probe_recovers_exact_control(
    fit_state_probe: Callable | None = None,
    state_probe_accuracy: Callable | None = None,
):
    solutions = _solutions()
    fit_state_probe = fit_state_probe or solutions.fit_state_probe
    state_probe_accuracy = state_probe_accuracy or solutions.state_probe_accuracy
    labels = t.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]).repeat(8, 1)
    exact_features = t.nn.functional.one_hot(labels, num_classes=4).float()
    probe = fit_state_probe(exact_features, labels, ridge=1e-3)
    accuracy = state_probe_accuracy(exact_features, labels, probe)
    assert accuracy == 1.0, f"the exact-state control should decode perfectly, got {accuracy:.3f}"
    print("All tests in `test_state_probe_recovers_exact_control` passed!")


def test_state_transplant_matches_donor_dynamics(
    run_state_transplant: Callable | None = None,
    make_matched_state_pair: Callable | None = None,
):
    solutions = _solutions()
    run_state_transplant = run_state_transplant or solutions.run_state_transplant
    make_matched_state_pair = make_matched_state_pair or solutions.make_matched_state_pair
    t.manual_seed(5)
    model = solutions.TinyMambaStateClassifier(num_states=4).cpu().eval()
    source, donor, edit_position = make_matched_state_pair()
    result = run_state_transplant(
        model,
        source,
        donor,
        edit_position,
        random_seed=0,
    )
    t.testing.assert_close(
        result["patched_logits"],
        result["donor_logits"],
        atol=1e-5,
        rtol=1e-5,
        msg=(
            "with identical convolutional history and suffix, transplanting the donor SSM "
            "state should exactly reproduce donor continuation logits"
        ),
    )
    assert not t.allclose(result["random_logits"], result["donor_logits"])
    print("All tests in `test_state_transplant_matches_donor_dynamics` passed!")


def test_find_confident_errors_orders_real_mistakes(
    find_confident_errors: Callable | None = None,
):
    solutions = _solutions()
    find_confident_errors = find_confident_errors or solutions.find_confident_errors
    tokens = t.tensor([[1, 1, 0, 1]])
    labels = t.tensor([[1, 2, 1, 2]])
    logits = t.tensor(
        [[[5.0, 1.0, 0.0], [0.0, 0.0, 6.0], [0.0, 4.0, 1.0], [0.0, 4.0, 3.0]]]
    )
    records = find_confident_errors(tokens, labels, logits, k=2)
    assert [record["position"] for record in records] == [0, 3]
    assert records[0]["prefix"] == "("
    assert records[0]["confidence"] >= records[1]["confidence"]
    print("All tests in `test_find_confident_errors_orders_real_mistakes` passed!")
