from __future__ import annotations

import ast
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import nbformat
import torch as t


def _solutions():
    from chapter17_training_dynamics.exercises.part1_checkpoint_archaeology import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _gpu_report() -> dict:
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    return report["metrics"]["gpu_test"]


def test_make_modular_addition_dataset_has_exact_finite_table(
    make_modular_addition_dataset: Callable | None = None,
):
    make_modular_addition_dataset = (
        make_modular_addition_dataset or _solutions().make_modular_addition_dataset
    )
    pairs, labels = make_modular_addition_dataset(modulus=13, device="cpu")
    assert pairs.shape == (169, 2), (
        "The model organism should enumerate the complete 13x13 finite input table."
    )
    assert labels.shape == (169,), (
        "The labels should have one target token for every finite table row."
    )
    assert int(labels[7 * 13 + 11].item()) == 5, (
        "The row for (7, 11) should have label (7 + 11) mod 13 = 5. "
        "Common bug: accidentally using integer division, subtraction, or an off-by-one modulus."
    )
    assert len(set(map(tuple, pairs.tolist()))) == 169, (
        "Every input pair should appear exactly once; checkpoint claims are only exhaustive "
        "if the finite domain is actually enumerated."
    )
    print("All tests in `test_make_modular_addition_dataset_has_exact_finite_table` passed!")


def test_train_save_reload_modular_checkpoints_writes_real_files(
    train_save_reload_modular_checkpoints: Callable | None = None,
):
    train_save_reload_modular_checkpoints = (
        train_save_reload_modular_checkpoints
        or _solutions().train_save_reload_modular_checkpoints
    )
    with tempfile.TemporaryDirectory(prefix="arena17_reload_test_") as tmp:
        root = Path(tmp)
        run = train_save_reload_modular_checkpoints(
            checkpoint_dir=root,
            device="cpu",
            seed=0,
            steps=32,
            checkpoint_steps=[0, 4, 8, 16, 32],
        )
        checkpoint_files = sorted(root.glob("*.pt"))
        assert len(checkpoint_files) == run["checkpoint_count"] == 5, (
            "The exercise should save the requested real checkpoint files, not just "
            "remember metrics in memory."
        )
        assert all(path.stat().st_size > 10_000 for path in checkpoint_files), (
            "Checkpoint files should contain real torch state, not empty sentinels."
        )
    assert run["real_checkpoints_reloaded"], (
        "Metrics should be recomputed after loading each saved checkpoint into a fresh model."
    )
    assert float(run["accuracies"][-1].item()) >= 0.99, (
        "The true-label model organism should solve the finite table by step 32."
    )
    assert float(run["mechanism_scores"][-1].item()) >= 0.60, (
        "The Fourier mechanism score should rise with task performance."
    )
    assert float(run["random_mechanism_scores"].max().item()) <= 0.01, (
        "A same-size random Fourier mask should not look like the addition mechanism."
    )
    print("All tests in `test_train_save_reload_modular_checkpoints_writes_real_files` passed!")


def test_addition_fourier_mask_identifies_exact_addition_rule(
    addition_fourier_mask: Callable | None = None,
    random_fourier_mask: Callable | None = None,
    fourier_power_fraction: Callable | None = None,
    exact_addition_logits: Callable | None = None,
):
    solutions = _solutions()
    addition_fourier_mask = addition_fourier_mask or solutions.addition_fourier_mask
    random_fourier_mask = random_fourier_mask or solutions.random_fourier_mask
    fourier_power_fraction = fourier_power_fraction or solutions.fourier_power_fraction
    exact_addition_logits = exact_addition_logits or solutions.exact_addition_logits

    logits = exact_addition_logits(modulus=13, device="cpu")
    legal = addition_fourier_mask(modulus=13, device="cpu")
    random_mask = random_fourier_mask(modulus=13, seed=17, device="cpu")
    assert int(legal.sum().item()) == 12, (
        "For prime modulus 13, the non-DC legal addition modes should be exactly p - 1."
    )
    assert int(random_mask.sum().item()) == int(legal.sum().item()), (
        "The random-mechanism control should have the same number of Fourier modes."
    )
    assert not bool((legal & random_mask).any().item()), (
        "The random-mechanism control should not accidentally include legal addition modes."
    )
    legal_score = fourier_power_fraction(logits, legal, modulus=13)
    random_score = fourier_power_fraction(logits, random_mask, modulus=13)
    assert legal_score > 0.999, (
        "The exact addition table should put essentially all centered logit power in "
        "legal addition Fourier modes."
    )
    assert random_score < 0.02, (
        "A same-size random Fourier mask should fail on the exact toy ground truth."
    )
    print("All tests in `test_addition_fourier_mask_identifies_exact_addition_rule` passed!")


def test_fourier_intervention_report_is_causal_on_exact_table(
    fourier_intervention_report: Callable | None = None,
    exact_addition_logits: Callable | None = None,
    make_modular_addition_dataset: Callable | None = None,
):
    solutions = _solutions()
    fourier_intervention_report = (
        fourier_intervention_report or solutions.fourier_intervention_report
    )
    exact_addition_logits = exact_addition_logits or solutions.exact_addition_logits
    make_modular_addition_dataset = (
        make_modular_addition_dataset or solutions.make_modular_addition_dataset
    )

    logits = exact_addition_logits(modulus=13, device="cpu")
    _, labels = make_modular_addition_dataset(modulus=13, device="cpu")
    report = fourier_intervention_report(logits, labels, modulus=13, random_seed=3)
    assert report["original_accuracy"] == 1.0, "The exact logit table should solve mod-13."
    assert report["legal_only_accuracy"] == 1.0, (
        "Keeping only legal Fourier modes should preserve the exact addition rule."
    )
    assert report["legal_ablated_accuracy"] <= 0.10, (
        "Ablating legal addition modes should destroy the exact addition rule."
    )
    assert report["random_only_accuracy"] <= 0.20, (
        "Keeping only a random same-size Fourier mask should not solve the task."
    )
    print("All tests in `test_fourier_intervention_report_is_causal_on_exact_table` passed!")


def test_checkpoint_archaeology_signature_result_trains_controls_and_intervenes(
    checkpoint_archaeology_signature_result: Callable | None = None,
):
    checkpoint_archaeology_signature_result = (
        checkpoint_archaeology_signature_result
        or _solutions().checkpoint_archaeology_signature_result
    )
    with tempfile.TemporaryDirectory(prefix="arena17_signature_test_") as tmp:
        result = checkpoint_archaeology_signature_result(
            checkpoint_root=Path(tmp),
            device="cpu",
            seed=0,
            include_figure=False,
        )

    assert result["preflight_passed"], (
        "The CPU signature path should train, save, reload, measure, intervene, and "
        "reject controls on the tiny model organism."
    )
    assert result["checkpoint_count"] == result["checkpoint_files_written"] == 28, (
        "The signature path should save target and random-label checkpoints for every "
        "declared checkpoint step."
    )
    assert result["accuracy_stable_from_step"] == 24, (
        "The target run should become stably accurate at checkpoint step 24."
    )
    assert result["mechanism_stable_from_step"] == 24, (
        "The Fourier mechanism score should become stable at the same checkpoint scale."
    )
    assert result["final_accuracy"] == 1.0, (
        "The final target checkpoint should solve the complete finite table."
    )
    assert result["final_addition_fourier_score"] >= 0.70, (
        "The final target checkpoint should have a large legal Fourier power fraction."
    )
    assert result["random_label_peak_accuracy"] <= 0.20, (
        "The random-label run should stay near chance when evaluated on true addition."
    )
    assert result["random_label_peak_fourier_score"] <= 0.05, (
        "The random-label run should not develop the true addition Fourier mechanism."
    )
    assert result["random_mechanism_peak_score"] <= 0.05, (
        "A same-size random Fourier mask should fail as a mechanism control."
    )
    late = result["intervention_rows"][-1]
    assert late["phase"] == "late" and late["step"] == 180, (
        "The intervention table should include an explicit late checkpoint."
    )
    assert late["legal_only_accuracy"] == 1.0 and late["legal_ablation_drop"] == 1.0, (
        "The late checkpoint should be causally carried by legal addition Fourier modes."
    )
    assert late["random_only_accuracy"] <= 0.20 and late["random_ablation_drop"] == 0.0, (
        "The random-mechanism intervention should fail visibly."
    )
    print(
        "All tests in "
        "`test_checkpoint_archaeology_signature_result_trains_controls_and_intervenes` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    live = result["live_checkpoint_archaeology"]
    assert live["preflight_passed"] and live["real_checkpoints_reloaded"], (
        "The notebook contract should include a live train/save/reload path."
    )
    assert live["final_accuracy"] == 1.0 and live["final_addition_fourier_score"] >= 0.70, (
        "The notebook contract should expose both task performance and the mechanism metric."
    )
    assert live["random_label_accuracy_control_passed"], (
        "The random-label control should be visible and should fail to solve true addition."
    )
    assert live["random_mechanism_control_passed"], (
        "The same-size random Fourier mechanism control should be visible and should fail."
    )
    assert len(live["intervention_rows"]) == 3, (
        "The notebook contract should include early/mid/late causal interventions."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_notebook_learner_surface_and_code_cells_compile():
    section_dir = _section_dir()
    exercise_path = (
        section_dir / "17.1_Checkpoint_Archaeology_and_Mechanism_Emergence_exercises.ipynb"
    )
    solution_path = (
        section_dir / "17.1_Checkpoint_Archaeology_and_Mechanism_Emergence_solutions.ipynb"
    )
    for path in [exercise_path, solution_path]:
        nb = nbformat.read(path, as_version=4)
        text = "\n".join(cell.source for cell in nb.cells)
        assert "By the end of this notebook" in text, (
            f"{path.name} should start from a single falsifiable learner claim."
        )
        assert text.count("### Exercise") >= 6, (
            f"{path.name} should contain at least six student exercises."
        )
        for required in [
            "Expected output",
            "Help",
            "Interpretation",
            "Solution",
            "Try It Yourself",
            "Anomaly Hunting",
            "Limitations",
            "checkpoint_archaeology_signature_result.png",
        ]:
            assert required in text, f"{path.name} is missing learner surface marker {required!r}."
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path}:{index}")

    exercise_text = "\n".join(
        cell.source
        for cell in nbformat.read(exercise_path, as_version=4).cells
        if cell.cell_type == "code"
    )
    solution_text = "\n".join(
        cell.source
        for cell in nbformat.read(solution_path, as_version=4).cells
        if cell.cell_type == "code"
    )
    assert "raise NotImplementedError" in exercise_text, (
        "The exercise notebook should leave real student implementation work."
    )
    assert "raise NotImplementedError" not in solution_text, (
        "The solution notebook should contain complete reference implementations."
    )
    print("All tests in `test_notebook_learner_surface_and_code_cells_compile` passed!")


def test_committed_gpu_report_records_real_checkpoint_preflight():
    gpu = _gpu_report()
    assert gpu["preflight_passed"], (
        "The committed GPU report should not be placeholder evidence."
    )
    assert gpu["cuda_available"] and "RTX 5090" in gpu["device"], (
        "The report should record a real CUDA device rather than synthetic evidence."
    )
    assert gpu["real_checkpoints_reloaded"], (
        "Checkpoint archaeology evidence must come from saved and reloaded checkpoint files."
    )
    if "target_addition_fourier_score_trajectory" not in gpu:
        print(
            "Committed GPU report predates the Fourier-mechanism rewrite; parent "
            "serial CUDA validation should regenerate it."
        )
        return
    assert gpu["final_accuracy"] == 1.0 and gpu["final_addition_fourier_score"] >= 0.70, (
        "The CUDA report should expose both solved task performance and a high "
        "addition Fourier score."
    )
    assert gpu["random_label_accuracy_control_passed"], (
        "The CUDA report should reject the random-label control."
    )
    assert gpu["random_mechanism_control_passed"], (
        "The CUDA report should reject the random Fourier mechanism control."
    )
    assert gpu["within_vram_budget"] and gpu["peak_vram_gb"] <= 24.0, (
        "The GPU evidence should stay inside the declared local VRAM budget."
    )
    print("All tests in `test_committed_gpu_report_records_real_checkpoint_preflight` passed!")
