"""Original seven immediate learner tests plus an independent scalar recurrence oracle.

The exact historical test bodies are retained. Only canonical lazy solution routing
and the local scalar generator reference replace prototype-library plumbing.
The optional training test is never run by migration verification.
"""
from collections.abc import Callable
from types import SimpleNamespace
import torch as t


def _solutions():
    from chapter5_modern_architectures.exercises.part5_state_tracking import solutions
    return solutions


def _reference_brackets(batch, seq_len, *, max_depth=4, seed=0):
    """Scalar transition enumeration using the same declared random-action tape."""
    rng = t.Generator().manual_seed(seed)
    depths = [0] * batch
    labels = [[] for _ in range(batch)]
    for _ in range(seq_len):
        proposals = t.randint(0, 2, (batch,), generator=rng).tolist()
        for row, proposal in enumerate(proposals):
            if depths[row] == 0:
                step = 1
            elif depths[row] == max_depth:
                step = -1
            else:
                step = 1 if proposal else -1
            depths[row] += step
            labels[row].append(depths[row])
    return SimpleNamespace(states=t.tensor(labels,dtype=t.long))


reference = SimpleNamespace(generate_bracket_depth_task=_reference_brackets)


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
    assert next(model.parameters()).device.type == "cpu", (
        "The learner training path should stay CPU-runnable; move the model and generated "
        "batches to CPU before optimizing."
    )
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
    assert recurrent_states.shape == (3, 9, expected_features), (
        "Collect one flattened recurrent SSM state per batch item and token; expected "
        f"(3, 9, {expected_features}), got {tuple(recurrent_states.shape)}."
    )
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
    assert not t.allclose(result["random_logits"], result["donor_logits"]), (
        "A matched-norm random edit should not reproduce the donor continuation; if it does, "
        "check that the random direction is independent of the donor state."
    )
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
    positions = [record["position"] for record in records]
    assert positions == [0, 3], (
        "Anomaly hunting should retain only wrong predictions and sort them by confidence; "
        f"expected positions [0, 3], got {positions}."
    )
    assert records[0]["prefix"] == "(", (
        "Each anomaly should retain the exact prefix ending at its position; "
        f"expected '(', got {records[0]['prefix']!r}."
    )
    assert records[0]["confidence"] >= records[1]["confidence"], (
        "Confident errors should be returned in descending confidence order; "
        f"got {records[0]['confidence']:.3f} then {records[1]['confidence']:.3f}."
    )
    print("All tests in `test_find_confident_errors_orders_real_mistakes` passed!")
