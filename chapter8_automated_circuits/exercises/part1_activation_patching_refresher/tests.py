"""Semantic tests for [8.1] Activation Patching Refresher."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import nbformat
import torch as t


def _solutions():
    from chapter8_automated_circuits.exercises.part1_activation_patching_refresher import (
        solutions,
    )

    return solutions


def _expected_route_matrix() -> t.Tensor:
    return t.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_answer_logit_diff(
    answer_logit_diff: Callable | None = None,
) -> None:
    answer_logit_diff = answer_logit_diff or _solutions().answer_logit_diff
    logits = t.tensor([[0.0, 4.0, 1.0], [0.0, 3.0, 2.0]])
    assert answer_logit_diff(logits, positive_token_id=1, negative_token_id=2) == 2.0
    for kwargs, message in [
        ({"positive_token_id": 1, "negative_token_id": 1}, "must differ"),
        ({"positive_token_id": 3, "negative_token_id": 2}, "positive_token_id"),
    ]:
        try:
            answer_logit_diff(logits, **kwargs)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("Invalid answer-token ids must fail clearly.")
    print("All tests in `test_answer_logit_diff` passed!")


def test_recovery_fraction(
    recovery_fraction: Callable | None = None,
) -> None:
    recovery_fraction = recovery_fraction or _solutions().recovery_fraction
    assert recovery_fraction(clean_metric=2.0, corrupt_metric=-2.0, patched_metric=-2.0) == 0.0
    assert recovery_fraction(clean_metric=2.0, corrupt_metric=-2.0, patched_metric=0.0) == 0.5
    assert recovery_fraction(clean_metric=2.0, corrupt_metric=-2.0, patched_metric=2.0) == 1.0
    try:
        recovery_fraction(clean_metric=1.0, corrupt_metric=1.0, patched_metric=1.0)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("A zero clean-corrupt gap must not be normalized.")
    print("All tests in `test_recovery_fraction` passed!")


def test_causal_copy_model_has_exact_ground_truth(
    run_causal_copy_model: Callable | None = None,
) -> None:
    solutions = _solutions()
    run_causal_copy_model = run_causal_copy_model or solutions.run_causal_copy_model
    clean, corrupt = solutions.make_copy_task_pair()
    clean_run = run_causal_copy_model(clean)
    corrupt_run = run_causal_copy_model(corrupt)
    assert clean_run.logits.tolist() == [0.0, 2.0, 0.0, 0.0, 0.0]
    assert corrupt_run.logits.tolist() == [0.0, 0.0, 2.0, 0.0, 0.0]
    assert clean_run.cache.shape == (3, 5, 2)
    assert clean_run.cache[0, solutions.SOURCE_POS].tolist() == [1.0, 0.0]
    assert clean_run.cache[1, solutions.QUERY_POS].tolist() == [1.0, 0.0]
    assert clean_run.cache[2, solutions.ANSWER_POS].tolist() == [1.0, 0.0]
    assert corrupt_run.cache[2, solutions.ANSWER_POS].tolist() == [0.0, 1.0]
    print("All tests in `test_causal_copy_model_has_exact_ground_truth` passed!")


def test_causal_copy_model_rejects_partial_patch_spec(
    run_causal_copy_model: Callable | None = None,
) -> None:
    solutions = _solutions()
    run_causal_copy_model = run_causal_copy_model or solutions.run_causal_copy_model
    clean, corrupt = solutions.make_copy_task_pair()
    donor = run_causal_copy_model(clean).cache
    try:
        run_causal_copy_model(corrupt, patch_layer=0, donor_cache=donor)
    except ValueError as exc:
        assert "required together" in str(exc)
    else:
        raise AssertionError("Partial patch specifications must fail.")
    print("All tests in `test_causal_copy_model_rejects_partial_patch_spec` passed!")


def test_patch_residual_cell_is_local_and_non_mutating(
    patch_residual_cell: Callable | None = None,
) -> None:
    solutions = _solutions()
    patch_residual_cell = patch_residual_cell or solutions.patch_residual_cell
    clean, corrupt = solutions.make_copy_task_pair()
    donor = solutions.run_causal_copy_model(clean).cache
    donor_before = donor.clone()
    patched = patch_residual_cell(
        corrupt,
        donor,
        layer=0,
        position=solutions.SOURCE_POS,
    )
    assert solutions.answer_logit_diff(patched.logits) == 2.0
    assert t.equal(donor, donor_before)
    wrong = patch_residual_cell(
        corrupt,
        donor,
        layer=0,
        position=solutions.DISTRACTOR_POS,
    )
    assert solutions.answer_logit_diff(wrong.logits) == -2.0
    print("All tests in `test_patch_residual_cell_is_local_and_non_mutating` passed!")


def test_denoising_sweep_recovers_exact_route(
    denoising_patch_sweep: Callable | None = None,
) -> None:
    solutions = _solutions()
    denoising_patch_sweep = denoising_patch_sweep or solutions.denoising_patch_sweep
    clean, corrupt = solutions.make_copy_task_pair()
    scores = denoising_patch_sweep(clean, corrupt)
    t.testing.assert_close(scores, _expected_route_matrix(), rtol=0.0, atol=0.0)
    print("All tests in `test_denoising_sweep_recovers_exact_route` passed!")


def test_noising_sweep_matches_exact_route(
    noising_patch_sweep: Callable | None = None,
) -> None:
    solutions = _solutions()
    noising_patch_sweep = noising_patch_sweep or solutions.noising_patch_sweep
    clean, corrupt = solutions.make_copy_task_pair()
    scores = noising_patch_sweep(clean, corrupt)
    t.testing.assert_close(scores, _expected_route_matrix(), rtol=0.0, atol=0.0)
    print("All tests in `test_noising_sweep_matches_exact_route` passed!")


def test_wrong_position_donor_control_fails(
    make_wrong_position_donor: Callable | None = None,
    denoising_patch_sweep: Callable | None = None,
) -> None:
    solutions = _solutions()
    make_wrong_position_donor = make_wrong_position_donor or solutions.make_wrong_position_donor
    denoising_patch_sweep = denoising_patch_sweep or solutions.denoising_patch_sweep
    clean, corrupt = solutions.make_copy_task_pair()
    clean_cache = solutions.run_causal_copy_model(clean).cache
    wrong_donor = make_wrong_position_donor(clean_cache)
    scores = denoising_patch_sweep(clean, corrupt, donor_cache=wrong_donor)
    assert t.equal(scores, t.zeros_like(scores))
    print("All tests in `test_wrong_position_donor_control_fails` passed!")


def test_localization_report_recovers_route(
    localization_report: Callable | None = None,
) -> None:
    solutions = _solutions()
    localization_report = localization_report or solutions.localization_report
    report = localization_report(_expected_route_matrix())
    assert set(report.top_cells) == set(solutions.ROUTE_CELLS)
    assert report.topk_overlap == 1.0
    assert report.route_mean == 1.0
    assert report.off_route_max == 0.0
    assert report.separation == 1.0
    print("All tests in `test_localization_report_recovers_route` passed!")


def test_localization_report_exposes_false_positive(
    localization_report: Callable | None = None,
) -> None:
    localization_report = localization_report or _solutions().localization_report
    scores = _expected_route_matrix()
    scores[0, 0] = 1.25
    report = localization_report(scores)
    assert report.topk_overlap < 1.0
    assert report.off_route_max == 1.25
    assert report.separation < 0.0
    print("All tests in `test_localization_report_exposes_false_positive` passed!")


def test_signature_contract(run_toy_signature_result: Callable | None = None) -> None:
    run_toy_signature_result = run_toy_signature_result or _solutions().run_toy_signature_result
    result = run_toy_signature_result()
    assert result["clean_metric"] == 2.0 and result["corrupt_metric"] == -2.0
    assert result["topk_overlap"] == 1.0
    assert result["separation"] == 1.0
    assert result["wrong_position_donor_max"] == 0.0
    assert result["denoising_noising_max_error"] == 0.0
    assert result["exact_ground_truth_passed"]
    print("All tests in `test_signature_contract` passed!")


def test_solution_notebook_exposes_taught_implementations() -> None:
    section_dir = Path(__file__).resolve().parent
    notebook = nbformat.read(
        section_dir / "8.1_Activation_Patching_Refresher_solutions.ipynb",
        as_version=4,
    )
    source = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    taught = {
        "answer_logit_diff",
        "recovery_fraction",
        "run_causal_copy_model",
        "patch_residual_cell",
        "denoising_patch_sweep",
        "noising_patch_sweep",
        "make_wrong_position_donor",
        "localization_report",
    }
    defined = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert taught <= defined, f"Solved notebook hides taught functions: {sorted(taught - defined)}"
    assert "solutions." not in source and "import solutions" not in source
    print("All tests in `test_solution_notebook_exposes_taught_implementations` passed!")


def test_notebook_cells_compile_and_learner_contract_is_visible() -> None:
    section_dir = Path(__file__).resolve().parent
    for filename in [
        "8.1_Activation_Patching_Refresher_exercises.ipynb",
        "8.1_Activation_Patching_Refresher_solutions.ipynb",
    ]:
        notebook = nbformat.read(section_dir / filename, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{filename}:cell-{index}", "exec")
        text = "\n".join(cell.source for cell in notebook.cells)
        assert text.lower().count("### exercise -") == 8
        assert text.lower().count("<summary>expected output") >= 8
        assert text.lower().count("<summary>help") >= 8
        assert text.lower().count("<summary>interpretation") >= 8
        assert text.lower().count("<summary>solution") >= 8
        assert "Try It Yourself" in text and "Bonus Anomaly Hunt" in text
        assert "Signature Result" in text
    print("All tests in `test_notebook_cells_compile_and_learner_contract_is_visible` passed!")
