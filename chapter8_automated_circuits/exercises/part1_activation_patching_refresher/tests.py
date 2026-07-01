from collections.abc import Callable

import torch as t

from arena_ext import activation_patching as reference


def _solutions():
    from chapter8_automated_circuits.exercises.part1_activation_patching_refresher import (
        solutions,
    )

    return solutions


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} fields should match the independent reference implementation."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, float):
            assert abs(actual_value - expected_value) < 1e-6, (
                f"{msg} field {key!r} should be {expected_value}, got {actual_value}."
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_answer_logit_diff_validates_token_ids(
    answer_logit_diff: Callable | None = None,
):
    answer_logit_diff = answer_logit_diff or _solutions().answer_logit_diff
    logits = t.tensor([[4.0, 1.0], [3.0, 2.0]])
    result = answer_logit_diff(logits, positive_token_id=0, negative_token_id=1)
    expected = reference.answer_logit_diff(
        logits,
        positive_token_id=0,
        negative_token_id=1,
    )
    assert result == expected == 2.0, (
        "Logit-diff metric should average positive-minus-negative logits over the batch."
    )
    try:
        answer_logit_diff(logits, positive_token_id=2, negative_token_id=1)
    except ValueError as exc:
        assert "positive_token_id" in str(exc), (
            "Out-of-range positive token ids should raise a helpful ValueError."
        )
    else:
        raise AssertionError("Out-of-range positive_token_id should raise ValueError.")
    print("All tests in `test_answer_logit_diff_validates_token_ids` passed!")


def test_patch_activation_slice_replaces_one_component_without_mutating_inputs(
    patch_activation_slice: Callable | None = None,
):
    patch_activation_slice = patch_activation_slice or _solutions().patch_activation_slice
    clean = t.tensor([[10.0, 20.0], [30.0, 40.0]])
    corrupt = t.zeros_like(clean)
    patched = patch_activation_slice(
        clean,
        corrupt,
        component_index=1,
        component_dim=0,
    )
    expected = reference.patch_activation_slice(
        clean,
        corrupt,
        component_index=1,
        component_dim=0,
    )
    assert patched.tolist() == expected.tolist() == [[0.0, 0.0], [30.0, 40.0]], (
        "Patching component 1 along dim 0 should copy exactly that clean row."
    )
    assert corrupt.tolist() == [[0.0, 0.0], [0.0, 0.0]], (
        "patch_activation_slice should return a patched clone and leave corrupt inputs unchanged."
    )
    try:
        patch_activation_slice(clean, corrupt, component_index=2, component_dim=0)
    except ValueError as exc:
        assert "component_index" in str(exc), (
            "Out-of-range component indices should raise a helpful ValueError."
        )
    else:
        raise AssertionError("Out-of-range component_index should raise ValueError.")
    print(
        "All tests in `test_patch_activation_slice_replaces_one_component_without_mutating_inputs` passed!"
    )


def test_patching_recovery_report_and_sweep_normalize_by_clean_corrupt_gap(
    patching_recovery_report: Callable | None = None,
    activation_patching_sweep: Callable | None = None,
    recovery_fraction: Callable | None = None,
):
    solutions = _solutions()
    patching_recovery_report = patching_recovery_report or solutions.patching_recovery_report
    activation_patching_sweep = activation_patching_sweep or solutions.activation_patching_sweep
    recovery_fraction = recovery_fraction or solutions.recovery_fraction

    clean_logits = t.tensor([4.0, 1.0])
    corrupt_logits = t.tensor([1.0, 3.0])
    patched_logits = t.tensor([3.0, 1.0])
    report = patching_recovery_report(
        clean_logits,
        corrupt_logits,
        patched_logits,
        positive_token_id=0,
        negative_token_id=1,
        min_recovered_fraction=0.75,
    )
    expected_report = reference.patching_recovery_report(
        clean_logits,
        corrupt_logits,
        patched_logits,
        positive_token_id=0,
        negative_token_id=1,
        min_recovered_fraction=0.75,
    )
    _assert_report_close(report, expected_report, msg="Patching recovery report")
    assert report.clean_metric == 3.0 and report.corrupt_metric == -2.0, (
        "Clean and corrupt metrics should use the same answer logit-diff readout."
    )
    assert abs(report.recovered_fraction - 0.8) < 1e-6 and report.passes_recovery, (
        "Patched metric 2.0 should recover 80% of the clean-corrupt gap."
    )

    sweep = activation_patching_sweep(
        clean_metric=3.0,
        corrupt_metric=-2.0,
        patched_metrics=t.tensor([-1.0, 2.0, 0.0]),
    )
    expected_sweep = reference.activation_patching_sweep(
        clean_metric=3.0,
        corrupt_metric=-2.0,
        patched_metrics=t.tensor([-1.0, 2.0, 0.0]),
    )
    assert t.allclose(sweep.patch_scores, expected_sweep.patch_scores), (
        "Sweep scores should be normalized recovered fractions for each component."
    )
    assert [round(score, 6) for score in sweep.patch_scores.tolist()] == [0.2, 0.8, 0.4], (
        "Patched metrics [-1, 2, 0] should map to recovery scores [0.2, 0.8, 0.4]."
    )
    assert sweep.best_index == 1 and abs(sweep.best_score - 0.8) < 1e-6, (
        "The best component should be the component with the highest recovery score."
    )
    try:
        recovery_fraction(clean_metric=1.0, corrupt_metric=1.0, patched_metric=1.0)
    except ValueError as exc:
        assert "differ" in str(exc), (
            "Zero clean-corrupt gaps should fail instead of dividing by zero."
        )
    else:
        raise AssertionError("Equal clean/corrupt metrics should raise ValueError.")
    print(
        "All tests in `test_patching_recovery_report_and_sweep_normalize_by_clean_corrupt_gap` passed!"
    )


def test_localization_and_random_controls_require_top_components_to_win(
    patching_localization_report: Callable | None = None,
    random_patch_control_report: Callable | None = None,
):
    solutions = _solutions()
    patching_localization_report = (
        patching_localization_report or solutions.patching_localization_report
    )
    random_patch_control_report = (
        random_patch_control_report or solutions.random_patch_control_report
    )
    patch_scores = t.tensor([0.2, 0.9, 0.8, 0.1])
    localization = patching_localization_report(
        patch_scores,
        target_indices=[1, 2],
        top_k=2,
        min_overlap=1.0,
    )
    expected_localization = reference.patching_localization_report(
        patch_scores,
        target_indices=[1, 2],
        top_k=2,
        min_overlap=1.0,
    )
    _assert_report_close(
        localization,
        expected_localization,
        msg="Patching localization report",
    )
    assert localization.top_indices == (1, 2) and localization.localizes_target, (
        "Top-k patching scores should recover the known target components in this toy task."
    )
    miss = patching_localization_report(
        patch_scores,
        target_indices=[0, 3],
        top_k=2,
        min_overlap=1.0,
    )
    assert not miss.localizes_target, (
        "Localization should fail when top patches do not overlap the target set."
    )

    control = random_patch_control_report(
        patch_scores,
        random_indices=[0, 3],
        top_k=2,
    )
    expected_control = reference.random_patch_control_report(
        patch_scores,
        random_indices=[0, 3],
        top_k=2,
    )
    _assert_report_close(control, expected_control, msg="Random patch control report")
    assert abs(control.top_patch_score - 0.85) < 1e-6, (
        "Top patch score should average the two largest component scores."
    )
    assert abs(control.random_patch_score - 0.15) < 1e-6 and control.top_beats_random, (
        "Top patches should beat the same-size random control."
    )
    losing_control = random_patch_control_report(
        patch_scores,
        random_indices=[1, 2],
        top_k=2,
    )
    assert not losing_control.top_beats_random, (
        "A random set equal to the top set should not count as top beating random."
    )
    print(
        "All tests in `test_localization_and_random_controls_require_top_components_to_win` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["logit_diff"] == 2.0, (
        "Notebook contract should include the answer logit-diff metric."
    )
    assert result["patch_slice"] == [[0.0, 0.0], [30.0, 40.0]], (
        "Notebook contract should include one clean slice patched into corrupt activations."
    )
    assert result["recovery"]["passes_recovery"], (
        "Notebook contract should include a recovered-fraction report that passes."
    )
    assert result["sweep"]["best_index"] == 1, (
        "Notebook contract should rank the best patching component."
    )
    assert result["localization"]["localizes_target"], (
        "Notebook contract should include target-component localization."
    )
    assert result["random_control"]["top_beats_random"], (
        "Notebook contract should include a top-vs-random patching control."
    )
    print("All tests in `test_notebook_contract` passed!")
