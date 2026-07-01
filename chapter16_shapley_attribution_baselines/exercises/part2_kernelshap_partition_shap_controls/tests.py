from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import torch as t

from arena_ext import shapley_attribution as reference


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partition_shap_controls import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _assert_tensor_close(actual: t.Tensor, expected: t.Tensor, *, msg: str) -> None:
    assert actual.dtype == expected.dtype, (
        f"{msg} should preserve dtype {expected.dtype}, got {actual.dtype}."
    )
    assert actual.shape == expected.shape, (
        f"{msg} should have shape {tuple(expected.shape)}, got {tuple(actual.shape)}."
    )
    assert t.allclose(actual, expected, atol=1e-8, rtol=0.0), (
        f"{msg} should equal {expected.tolist()}, got {actual.tolist()}."
    )


def _assert_report_matches_reference(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} should expose the same fields as the independent reference."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, t.Tensor):
            _assert_tensor_close(actual_value, expected_value, msg=f"{msg} field {key!r}")
        elif isinstance(expected_value, float):
            assert abs(actual_value - expected_value) < 1e-8, (
                f"{msg} field {key!r} should be {expected_value}, got {actual_value}."
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_kernelshap_kernel_weight_uses_finite_coalition_formula(
    kernelshap_kernel_weight: Callable | None = None,
):
    kernelshap_kernel_weight = (
        kernelshap_kernel_weight or _solutions().kernelshap_kernel_weight
    )
    assert kernelshap_kernel_weight(1, 4) == reference.kernelshap_kernel_weight(1, 4), (
        "Singleton coalitions in a four-player KernelSHAP table should receive the exact finite weight."
    )
    assert kernelshap_kernel_weight(2, 4) == reference.kernelshap_kernel_weight(2, 4), (
        "Middle-size coalitions should use the KernelSHAP combinatorial weight, not a uniform weight."
    )
    try:
        kernelshap_kernel_weight(0, 4)
    except ValueError as exc:
        assert "coalition_size" in str(exc) or "0 <" in str(exc), (
            "Invalid KernelSHAP coalition sizes should raise a helpful coalition_size error."
        )
    else:
        raise AssertionError("Empty coalitions should not use finite KernelSHAP weights.")
    print("All tests in `test_kernelshap_kernel_weight_uses_finite_coalition_formula` passed!")


def test_kernelshap_approximation_report_matches_exact_additive_game(
    kernelshap_approximation_report: Callable | None = None,
    additive_game: Callable | None = None,
):
    solutions = _solutions()
    kernelshap_approximation_report = (
        kernelshap_approximation_report or solutions.kernelshap_approximation_report
    )
    additive_game = additive_game or solutions.additive_game
    values = additive_game(t.tensor([1.0, -2.0, 0.5]))
    report = kernelshap_approximation_report(values, num_players=3)
    expected = reference.kernelshap_approximation_report(values, num_players=3)
    _assert_report_matches_reference(report, expected, msg="KernelSHAP additive report")
    assert report.approximates_exact, (
        "Full-table KernelSHAP should exactly recover an additive game's feature weights."
    )
    assert report.shapley_values.tolist() == [1.0, -2.0, 0.5], (
        "Additive KernelSHAP values should equal the original weights."
    )
    print("All tests in `test_kernelshap_approximation_report_matches_exact_additive_game` passed!")


def test_kernelshap_interaction_report_splits_conjunction_credit(
    kernelshap_approximation_report: Callable | None = None,
    conjunction_game: Callable | None = None,
):
    solutions = _solutions()
    kernelshap_approximation_report = (
        kernelshap_approximation_report or solutions.kernelshap_approximation_report
    )
    conjunction_game = conjunction_game or solutions.conjunction_game
    values = conjunction_game(3)
    report = kernelshap_approximation_report(values, num_players=3)
    expected = reference.kernelshap_approximation_report(values, num_players=3)
    _assert_report_matches_reference(report, expected, msg="KernelSHAP interaction report")
    assert report.approximates_exact, (
        "Full-table KernelSHAP should match exact Shapley even on the conjunction game."
    )
    assert all(abs(value - (1 / 3)) < 1e-8 for value in report.shapley_values.tolist()), (
        "A symmetric three-player conjunction should split credit equally."
    )
    print("All tests in `test_kernelshap_interaction_report_splits_conjunction_credit` passed!")


def test_partition_shap_report_recovers_additive_groups(
    partition_shap_report: Callable | None = None,
    additive_game: Callable | None = None,
):
    solutions = _solutions()
    partition_shap_report = partition_shap_report or solutions.partition_shap_report
    additive_game = additive_game or solutions.additive_game
    values = additive_game(t.tensor([1.0, 2.0, 3.0, 4.0]))
    report = partition_shap_report(values, groups=((0, 1), (2, 3)))
    expected = reference.partition_shap_report(values, groups=((0, 1), (2, 3)))
    _assert_report_matches_reference(report, expected, msg="PartitionSHAP additive report")
    assert report.recovers_exact, (
        "PartitionSHAP should not distort player credit in a purely additive game."
    )
    assert report.group_values.tolist() == [3.0, 7.0], (
        "The group-level Shapley values should equal the sums of each additive group."
    )
    assert report.player_values.tolist() == [1.0, 2.0, 3.0, 4.0], (
        "The final player values should recover every additive feature weight."
    )
    print("All tests in `test_partition_shap_report_recovers_additive_groups` passed!")


def test_partition_shap_report_splits_interaction_group_symmetrically(
    partition_shap_report: Callable | None = None,
    conjunction_game: Callable | None = None,
):
    solutions = _solutions()
    partition_shap_report = partition_shap_report or solutions.partition_shap_report
    conjunction_game = conjunction_game or solutions.conjunction_game
    values = conjunction_game(2)
    report = partition_shap_report(values, groups=((0, 1),))
    expected = reference.partition_shap_report(values, groups=((0, 1),))
    _assert_report_matches_reference(report, expected, msg="PartitionSHAP interaction report")
    assert report.recovers_exact, (
        "A single grouped two-player conjunction should still recover exact Shapley credit."
    )
    assert report.player_values.tolist() == [0.5, 0.5], (
        "Within-group symmetric interaction credit should split equally between both players."
    )
    print("All tests in `test_partition_shap_report_splits_interaction_group_symmetrically` passed!")


def test_partition_shap_report_handles_cross_group_interaction_without_irrelevant_credit(
    partition_shap_report: Callable | None = None,
    coalition_values_from_function: Callable | None = None,
):
    solutions = _solutions()
    partition_shap_report = partition_shap_report or solutions.partition_shap_report
    coalition_values_from_function = (
        coalition_values_from_function or solutions.coalition_values_from_function
    )
    values = coalition_values_from_function(
        3,
        lambda coalition: {0, 2}.issubset(coalition),
    )
    report = partition_shap_report(values, groups=((0, 1), (2,)))
    expected = reference.partition_shap_report(values, groups=((0, 1), (2,)))
    _assert_report_matches_reference(report, expected, msg="PartitionSHAP cross-group report")
    assert report.recovers_exact, (
        "The exact Owen calculation should recover ordinary Shapley values for this simple interaction."
    )
    assert report.player_values.tolist() == [0.5, 0.0, 0.5], (
        "The irrelevant grouped player should receive zero credit for an interaction between players 0 and 2."
    )
    print(
        "All tests in `test_partition_shap_report_handles_cross_group_interaction_without_irrelevant_credit` passed!"
    )


def test_partition_shap_report_rejects_invalid_grouping(
    partition_shap_report: Callable | None = None,
    additive_game: Callable | None = None,
):
    solutions = _solutions()
    partition_shap_report = partition_shap_report or solutions.partition_shap_report
    additive_game = additive_game or solutions.additive_game
    values = additive_game(t.tensor([1.0, 2.0, 3.0]))
    try:
        partition_shap_report(values, groups=((0, 1), (1, 2)))
    except ValueError as exc:
        assert "partition players" in str(exc), (
            "Overlapping PartitionSHAP groups should raise an explicit partition error."
        )
    else:
        raise AssertionError("Overlapping PartitionSHAP groups should raise ValueError.")
    print("All tests in `test_partition_shap_report_rejects_invalid_grouping` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["kernel_additive"]["approximates_exact"], (
        "The notebook contract should prove KernelSHAP additive parity."
    )
    assert result["kernel_interaction"]["approximates_exact"], (
        "The notebook contract should prove KernelSHAP conjunction parity."
    )
    assert result["partition_additive"]["recovers_exact"], (
        "The notebook contract should prove PartitionSHAP additive parity."
    )
    assert result["partition_interaction"]["recovers_exact"], (
        "The notebook contract should prove PartitionSHAP interaction-group parity."
    )
    assert result["partition_cross_group_interaction"]["recovers_exact"], (
        "The notebook contract should prove PartitionSHAP does not assign cross-group interaction credit to irrelevant players."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_report_has_real_cuda_kernel_partition_evidence():
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    gpu = report["metrics"]["gpu_test"]
    evidence = report["metrics"]["gpu_evidence"]
    assert report["accepted"] is True, (
        "The committed verification report should be accepted."
    )
    assert report["gt_tier"] == "GT-0", (
        "16.2 should remain scoped to GT-0 finite-table SHAP controls."
    )
    assert evidence["uses_cuda"] is True and evidence["placeholder_only"] is False, (
        "16.2 evidence should come from a real CUDA section metric, not a placeholder path."
    )
    assert gpu["cuda_available"] is True and gpu["preflight_passed"] is True, (
        "The GPU preflight should run on CUDA and pass all SHAP-control gates."
    )
    assert gpu["kernel_approximates_exact"] is True, (
        "KernelSHAP should match exact Shapley on the trained model coalition table."
    )
    assert gpu["kernel_vs_true_max_abs_error"] <= 1e-4, (
        "Trained-model KernelSHAP should remain close to the analytic ground-truth game."
    )
    assert gpu["partition_recovers_exact"] is True, (
        "Singleton PartitionSHAP should recover exact player Shapley values."
    )
    assert gpu["aligned_partition_recovers_exact"] is True, (
        "Grouped PartitionSHAP should recover exact player credit for this trained pairwise game."
    )
    assert gpu["mismatched_partition_recovers_exact"] is True, (
        "The mismatched grouping should still be computed by exact Owen values, not the old within-group heuristic."
    )
    assert gpu["cross_group_partition_recovers_exact"] is True, (
        "The cross-group interaction counterexample should recover exact credit."
    )
    assert gpu["cross_group_irrelevant_credit"] <= 1e-8, (
        "An irrelevant player inside a mixed group should receive zero cross-group interaction credit."
    )
    assert gpu["shuffled_control_rejected"] is True, (
        "The shuffled-label trained-model attribution should be rejected."
    )
    assert gpu["within_vram_budget"] is True, (
        "The CUDA preflight should stay inside the declared VRAM budget."
    )
    print("All tests in `test_committed_report_has_real_cuda_kernel_partition_evidence` passed!")
