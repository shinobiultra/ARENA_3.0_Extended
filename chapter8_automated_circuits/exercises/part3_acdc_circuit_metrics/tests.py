from collections.abc import Callable

import torch as t

from arena_ext import circuit_metrics as reference


def _solutions():
    from chapter8_automated_circuits.exercises.part3_acdc_circuit_metrics import (
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


def test_position_patching_helpers_score_recovery(
    answer_logit_diff: Callable | None = None,
    activation_patching_sweep: Callable | None = None,
):
    solutions = _solutions()
    answer_logit_diff = answer_logit_diff or solutions.answer_logit_diff
    activation_patching_sweep = activation_patching_sweep or solutions.activation_patching_sweep
    logits = t.tensor([[1.0, 4.0, -2.0], [0.0, 3.0, 1.0]])
    diff = answer_logit_diff(logits, positive_token_id=1, negative_token_id=2)
    assert diff == 4.0, (
        "answer_logit_diff should average positive-minus-negative logits over leading dims."
    )
    sweep = activation_patching_sweep(
        clean_metric=3.0,
        corrupt_metric=-1.0,
        patched_metrics=t.tensor([-1.0, 1.0, 3.0]),
    )
    assert sweep.patch_scores.tolist() == [0.0, 0.5, 1.0], (
        "Patch scores should be normalized recovery between corrupt and clean metrics."
    )
    assert sweep.best_index == 2 and sweep.best_score == 1.0, (
        "The sweep should expose the highest-recovery component."
    )
    try:
        activation_patching_sweep(
            clean_metric=1.0,
            corrupt_metric=1.0,
            patched_metrics=t.tensor([1.0]),
        )
    except ValueError as exc:
        assert "must differ" in str(exc), (
            "Degenerate clean/corrupt metrics should raise a useful denominator error."
        )
    else:
        raise AssertionError("Equal clean/corrupt metrics should raise ValueError.")
    print("All tests in `test_position_patching_helpers_score_recovery` passed!")


def test_acdc_pruning_report_keeps_threshold_edges(
    acdc_pruning_report: Callable | None = None,
):
    acdc_pruning_report = acdc_pruning_report or _solutions().acdc_pruning_report
    scores = t.tensor([0.9, 0.5, 0.2])
    names = ["name-mover", "backup", "negative"]
    report = acdc_pruning_report(scores, names, threshold=0.5)
    expected = reference.acdc_pruning_report(scores, names, threshold=0.5)
    _assert_report_close(report, expected, msg="ACDC pruning report")
    assert report.kept_edges == ("name-mover", "backup"), (
        "ACDC pruning should keep scores equal to or above the threshold."
    )
    assert report.removed_edges == ("negative",) and report.num_kept == 2, (
        "ACDC pruning should report both removed edge names and the kept count."
    )
    try:
        acdc_pruning_report(scores, names[:2], threshold=0.5)
    except ValueError as exc:
        assert "align" in str(exc), (
            "Mismatched edge score/name lengths should raise an alignment error."
        )
    else:
        raise AssertionError("Mismatched edge scores and names should raise ValueError.")
    print("All tests in `test_acdc_pruning_report_keeps_threshold_edges` passed!")


def test_faithfulness_report_normalizes_clean_corrupt_gap(
    circuit_faithfulness_report: Callable | None = None,
):
    circuit_faithfulness_report = (
        circuit_faithfulness_report or _solutions().circuit_faithfulness_report
    )
    report = circuit_faithfulness_report(
        full_metric=3.0,
        corrupt_metric=-1.0,
        circuit_metric=2.2,
        min_preserved_fraction=0.75,
    )
    expected = reference.circuit_faithfulness_report(
        full_metric=3.0,
        corrupt_metric=-1.0,
        circuit_metric=2.2,
        min_preserved_fraction=0.75,
    )
    _assert_report_close(report, expected, msg="Faithfulness report")
    assert abs(report.preserved_fraction - 0.8) < 1e-6, (
        "Faithfulness should be normalized by the clean-corrupt metric gap."
    )
    assert report.passes_faithfulness, (
        "A circuit preserving 80% should pass a 75% faithfulness threshold."
    )
    weak_report = circuit_faithfulness_report(
        full_metric=3.0,
        corrupt_metric=-1.0,
        circuit_metric=1.0,
        min_preserved_fraction=0.75,
    )
    assert not weak_report.passes_faithfulness, (
        "A weak circuit should fail the preserved-fraction threshold."
    )
    try:
        circuit_faithfulness_report(
            full_metric=1.0,
            corrupt_metric=1.0,
            circuit_metric=1.0,
        )
    except ValueError as exc:
        assert "must differ" in str(exc), (
            "Degenerate clean/corrupt gaps should raise a helpful denominator error."
        )
    else:
        raise AssertionError("Equal full/corrupt metrics should raise ValueError.")
    print("All tests in `test_faithfulness_report_normalizes_clean_corrupt_gap` passed!")


def test_minimality_and_completeness_reports_distinguish_failure_modes(
    circuit_minimality_report: Callable | None = None,
    circuit_completeness_report: Callable | None = None,
):
    solutions = _solutions()
    circuit_minimality_report = (
        circuit_minimality_report or solutions.circuit_minimality_report
    )
    circuit_completeness_report = (
        circuit_completeness_report or solutions.circuit_completeness_report
    )
    minimality = circuit_minimality_report(
        circuit_metric=2.2,
        ablated_metric=0.5,
        min_metric_damage=1.0,
    )
    expected_minimality = reference.circuit_minimality_report(
        circuit_metric=2.2,
        ablated_metric=0.5,
        min_metric_damage=1.0,
    )
    _assert_report_close(minimality, expected_minimality, msg="Minimality report")
    assert abs(minimality.metric_damage - 1.7) < 1e-6, (
        "Minimality damage should be circuit_metric - ablated_metric."
    )
    assert minimality.passes_minimality, (
        "A large ablation damage should pass the minimality threshold."
    )
    bloated = circuit_minimality_report(
        circuit_metric=2.2,
        ablated_metric=2.0,
        min_metric_damage=1.0,
    )
    assert not bloated.passes_minimality, (
        "A circuit should fail minimality when removing nodes barely changes behavior."
    )

    completeness = circuit_completeness_report(
        circuit_metric=2.2,
        expanded_metric=2.35,
        max_omitted_node_gain=0.2,
    )
    expected_completeness = reference.circuit_completeness_report(
        circuit_metric=2.2,
        expanded_metric=2.35,
        max_omitted_node_gain=0.2,
    )
    _assert_report_close(completeness, expected_completeness, msg="Completeness report")
    assert abs(completeness.omitted_node_gain - 0.15) < 1e-6, (
        "Completeness gain should be expanded_metric - circuit_metric."
    )
    assert completeness.passes_completeness, (
        "A small omitted-node gain should pass the completeness threshold."
    )
    incomplete = circuit_completeness_report(
        circuit_metric=2.2,
        expanded_metric=2.8,
        max_omitted_node_gain=0.2,
    )
    assert not incomplete.passes_completeness, (
        "A circuit should fail completeness when adding omitted nodes helps a lot."
    )
    print(
        "All tests in `test_minimality_and_completeness_reports_distinguish_failure_modes` passed!"
    )


def test_random_circuit_baseline_report_requires_margin(
    random_circuit_baseline_report: Callable | None = None,
):
    random_circuit_baseline_report = (
        random_circuit_baseline_report or _solutions().random_circuit_baseline_report
    )
    report = random_circuit_baseline_report(
        circuit_metric=2.2,
        random_metric=0.8,
        min_margin=1.0,
    )
    expected = reference.random_circuit_baseline_report(
        circuit_metric=2.2,
        random_metric=0.8,
        min_margin=1.0,
    )
    _assert_report_close(report, expected, msg="Random baseline report")
    assert abs(report.margin - 1.4) < 1e-6 and report.circuit_beats_random, (
        "The discovered circuit should beat a same-size random circuit by the margin."
    )
    weak = random_circuit_baseline_report(
        circuit_metric=2.2,
        random_metric=1.8,
        min_margin=1.0,
    )
    assert not weak.circuit_beats_random, (
        "A circuit should fail the random baseline if the margin is too small."
    )
    print("All tests in `test_random_circuit_baseline_report_requires_margin` passed!")


def test_ood_template_report_tracks_worst_template(
    ood_template_report: Callable | None = None,
):
    ood_template_report = ood_template_report or _solutions().ood_template_report
    logits = t.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [2.0, 0.0],
        ]
    )
    answer_ids = t.tensor([0, 1, 0, 1])
    template_ids = t.tensor([0, 0, 1, 1])
    report = ood_template_report(logits, answer_ids, template_ids, min_accuracy=0.75)
    assert report.per_template_accuracy == {0: 1.0, 1: 0.5}, (
        "OOD reports should expose per-template accuracy, not only the average."
    )
    assert report.worst_template_accuracy == 0.5 and not report.passes_ood, (
        "OOD should fail when the worst held-out template falls below threshold."
    )
    passing_logits = t.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    passing = ood_template_report(
        passing_logits,
        answer_ids,
        template_ids,
        min_accuracy=1.0,
    )
    expected_passing = reference.ood_template_report(
        passing_logits,
        answer_ids,
        template_ids,
        min_accuracy=1.0,
    )
    _assert_report_close(passing, expected_passing, msg="OOD template report")
    assert passing.per_template_accuracy == {0: 1.0, 1: 1.0} and passing.passes_ood, (
        "Perfect per-template accuracy should pass a 100% OOD threshold."
    )
    try:
        ood_template_report(passing_logits, answer_ids[:2], template_ids, min_accuracy=1.0)
    except ValueError as exc:
        assert "answer_ids" in str(exc), (
            "Mismatched answer ids should raise a shape-specific error."
        )
    else:
        raise AssertionError("Mismatched answer ids should raise ValueError.")
    print("All tests in `test_ood_template_report_tracks_worst_template` passed!")


def test_circuit_method_comparison_report_matches_exact_patching(
    circuit_method_comparison_report: Callable | None = None,
):
    circuit_method_comparison_report = (
        circuit_method_comparison_report
        or _solutions().circuit_method_comparison_report
    )
    exact = t.tensor([0.95, 0.8, 0.15, 0.05])
    eap_ig = t.tensor([0.9, 0.7, 0.2, 0.1])
    weak_eap = t.tensor([0.2, 0.1, 0.9, 0.8])
    names = ["name-mover", "backup-name-mover", "mlp-noise", "wrong-position"]

    report = circuit_method_comparison_report(
        exact,
        {"eap_ig": eap_ig, "weak_eap": weak_eap},
        names,
        top_k=2,
        min_topk_overlap=0.5,
        min_score_correlation=0.5,
    )
    expected = reference.circuit_method_comparison_report(
        exact,
        {"eap_ig": eap_ig, "weak_eap": weak_eap},
        names,
        top_k=2,
        min_topk_overlap=0.5,
        min_score_correlation=0.5,
    )
    _assert_report_close(report, expected, msg="Circuit method comparison")
    assert report.exact_top_edges == ("name-mover", "backup-name-mover"), (
        "Exact patching should define the reference top-k circuit."
    )
    assert report.topk_overlap["eap_ig"] == 1.0, (
        "A good approximate method should recover the exact top-k edges."
    )
    assert report.topk_overlap["weak_eap"] == 0.0, (
        "A bad approximate method should be exposed by top-k overlap."
    )
    assert report.score_correlations["eap_ig"] > 0.95, (
        "The comparison should report score correlation, not only set overlap."
    )
    assert not report.passes_comparison, (
        "The report should fail when any required comparison method fails."
    )
    print("All tests in `test_circuit_method_comparison_report_matches_exact_patching` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["acdc"]["num_kept"] == 2, (
        "The notebook smoke contract should include the ACDC pruning result."
    )
    assert result["faithfulness"]["passes_faithfulness"], (
        "The notebook smoke contract should include a passing faithfulness check."
    )
    assert result["minimality"]["passes_minimality"], (
        "The notebook smoke contract should include a passing minimality check."
    )
    assert result["completeness"]["passes_completeness"], (
        "The notebook smoke contract should include a passing completeness check."
    )
    assert result["random_baseline"]["circuit_beats_random"], (
        "The notebook smoke contract should include a passing random baseline."
    )
    assert result["ood"]["passes_ood"], (
        "The notebook smoke contract should include a passing OOD template check."
    )
    assert result["method_comparison"]["passes_comparison"], (
        "The notebook smoke contract should include exact-vs-approximate circuit comparison."
    )
    print("All tests in `test_notebook_contract` passed!")
