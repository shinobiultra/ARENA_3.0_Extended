"""Semantic and learner-surface tests for [10.1] Capstone Research Sprint."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import nbformat
import torch as t
from PIL import Image


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _solutions():
    module_name = "chapter10_capstone_research_sprint_reference_solutions"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _section_dir() / "solutions.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import solutions from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_make_parity_batch_has_exact_ground_truth(
    make_parity_batch: Callable | None = None,
    make_rotation: Callable | None = None,
) -> None:
    solutions = _solutions()
    make_parity_batch = make_parity_batch or solutions.make_parity_batch
    make_rotation = make_rotation or solutions.make_rotation
    rotation = make_rotation()
    batch = make_parity_batch([0, 1, 2], rotation)

    assert batch.bits.shape == (12, 2), "Each template should enumerate all four bit pairs."
    assert batch.latent_features.shape == (12, 8), "The declared latent space has width eight."
    assert batch.activations.shape == (12, 8), "Rotation must preserve the latent batch shape."
    assert batch.labels.shape == (12,), "Every enumerated example needs one XOR label."
    assert batch.activations.dtype == t.float64, "The exact organism uses float64 throughout."
    assert t.allclose(
        rotation.T @ rotation, t.eye(8, dtype=t.float64), atol=1e-10
    ), "The mixing matrix must remain orthogonal."
    assert t.equal(
        batch.latent_features[:, 2], batch.bits[:, 0] * batch.bits[:, 1]
    ), "Latent coordinate two is the declared signed XOR feature."
    for template_id in [0, 1, 2]:
        template_rows = batch.template_ids == template_id
        assert int(template_rows.sum()) == 4, "Every template must contain four balanced bit pairs."
        assert int(batch.labels[template_rows].sum()) == 2, "Each template must have balanced XOR labels."

    exact_direction = rotation[:, 2]
    expected_signed_label = batch.labels.to(t.float64) * 2 - 1
    assert t.allclose(
        batch.activations @ exact_direction, expected_signed_label, atol=1e-10
    ), "Projecting onto the planted direction must recover the exact signed label."
    print("All tests in `test_make_parity_batch_has_exact_ground_truth` passed!")


def test_ridge_probe_recovers_exact_direction_and_rejects_raw_baseline(
    make_parity_batch: Callable | None = None,
    make_rotation: Callable | None = None,
    fit_ridge_direction: Callable | None = None,
    direction_accuracy: Callable | None = None,
) -> None:
    solutions = _solutions()
    make_parity_batch = make_parity_batch or solutions.make_parity_batch
    make_rotation = make_rotation or solutions.make_rotation
    fit_ridge_direction = fit_ridge_direction or solutions.fit_ridge_direction
    direction_accuracy = direction_accuracy or solutions.direction_accuracy

    rotation = make_rotation()
    train = make_parity_batch(range(12), rotation)
    heldout = make_parity_batch(range(12, 20), rotation)
    learned = fit_ridge_direction(train.activations, train.labels)
    exact = rotation[:, 2]
    cosine = float(t.nn.functional.cosine_similarity(learned, exact, dim=0).abs())
    assert cosine > 0.999, "The ridge probe should recover the known distributed XOR direction."
    assert direction_accuracy(heldout.activations, heldout.labels, learned) == 1.0, (
        "The recovered direction must generalize to all held-out nuisance templates."
    )

    raw_probe = fit_ridge_direction(train.bits, train.labels)
    raw_accuracy = direction_accuracy(heldout.bits, heldout.labels, raw_probe)
    assert raw_accuracy == 0.5, "A linear classifier on the two raw bits cannot solve XOR."

    try:
        fit_ridge_direction(train.activations[:, None], train.labels)
    except ValueError:
        pass
    else:
        raise AssertionError("The probe should reject non-matrix features.")
    print(
        "All tests in `test_ridge_probe_recovers_exact_direction_and_rejects_raw_baseline` passed!"
    )


def test_paired_bootstrap_uses_example_pairs(
    paired_bootstrap_delta_ci: Callable | None = None,
) -> None:
    paired_bootstrap_delta_ci = (
        paired_bootstrap_delta_ci or _solutions().paired_bootstrap_delta_ci
    )
    method = t.tensor([True, True, True, True, True, True, True, True])
    baseline = t.tensor([True, False, True, False, True, False, True, False])
    estimate, low, high = paired_bootstrap_delta_ci(
        method, baseline, n_resamples=2_000, seed=3
    )
    assert estimate == 0.5, "The paired method-minus-baseline estimate should equal 0.5."
    assert 0.0 <= low <= estimate <= high <= 1.0, (
        "A paired accuracy interval must contain its estimate and remain in [0, 1]."
    )
    assert (estimate, low, high) == paired_bootstrap_delta_ci(
        method, baseline, n_resamples=2_000, seed=3
    ), "A fixed bootstrap seed must reproduce the exact paired interval."
    try:
        paired_bootstrap_delta_ci(method, baseline[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("The bootstrap should require aligned example pairs.")
    print("All tests in `test_paired_bootstrap_uses_example_pairs` passed!")


def test_directional_patch_replaces_only_the_named_projection(
    patch_along_direction: Callable | None = None,
) -> None:
    patch_along_direction = patch_along_direction or _solutions().patch_along_direction
    recipient = t.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]], dtype=t.float64)
    donor = t.tensor([[-4.0, 7.0, 8.0], [3.0, -1.0, 2.0]], dtype=t.float64)
    direction = t.tensor([2.0, 0.0, 0.0], dtype=t.float64)
    patched = patch_along_direction(recipient, donor, direction)

    unit = direction / direction.norm()
    recipient_residual = recipient - (recipient @ unit)[:, None] * unit
    patched_residual = patched - (patched @ unit)[:, None] * unit
    assert t.allclose(
        patched @ unit, donor @ unit
    ), "The named projection must take the donor value."
    assert t.allclose(
        patched_residual, recipient_residual
    ), "Every component orthogonal to the patch direction must stay unchanged."
    assert t.allclose(
        patched[:, 1:], recipient[:, 1:]
    ), "This axis-aligned fixture must preserve the two untouched coordinates exactly."
    print("All tests in `test_directional_patch_replaces_only_the_named_projection` passed!")


def test_noise_sweep_is_reproducible_and_exposes_failure(
    noise_sweep_accuracy: Callable | None = None,
) -> None:
    solutions = _solutions()
    noise_sweep_accuracy = noise_sweep_accuracy or solutions.noise_sweep_accuracy
    rotation = solutions.make_rotation()
    heldout = solutions.make_parity_batch(range(12, 20), rotation)
    direction = rotation[:, 2]
    sigmas = t.tensor([0.0, 0.5, 1.0, 2.0, 3.0], dtype=t.float64)
    first = noise_sweep_accuracy(
        heldout.activations, heldout.labels, direction, sigmas, repeats=128, seed=123
    )
    second = noise_sweep_accuracy(
        heldout.activations, heldout.labels, direction, sigmas, repeats=128, seed=123
    )
    assert t.equal(first, second), "The fixed-seed noise sweep must be reproducible."
    assert first[0] == 1.0, "The noiseless direction must solve the exact held-out task."
    assert bool(t.all(first[:-1] >= first[1:])), "The fixed-noise stress curve should be monotone."
    assert first[-1] < 0.70, "Strong activation noise should reveal the method's failure boundary."
    print("All tests in `test_noise_sweep_is_reproducible_and_exposes_failure` passed!")


def test_complete_cpu_study_meets_preregistered_claim() -> None:
    result = _solutions().run_smoke_test(cpu=True)
    assert result["model_family"] == "exact_rotated_parity_model", (
        "The capstone contract must name the exact model organism."
    )
    assert result["train_example_count"] == 48, "Twelve train templates should yield 48 examples."
    assert result["heldout_example_count"] == 32, "Eight held-out templates should yield 32 examples."
    assert result["heldout_accuracy"] == 1.0, "The recovered mechanism must solve every held-out example."
    assert result["raw_bits_accuracy"] == 0.5, "A linear raw-bit baseline must remain at XOR chance."
    assert result["template_only_accuracy"] == 0.5, "Nuisance templates must not predict the label."
    assert result["direction_cosine"] > 0.999, "The fitted direction must match planted ground truth."
    assert result["learned_patch_target_accuracy"] == 1.0, (
        "Patching the learned direction must transfer every donor target."
    )
    assert result["ablation_accuracy"] == 0.5, "Removing the causal direction must reduce accuracy to chance."
    assert result["random_patch_target_accuracy_mean"] < 0.20, (
        "Matched random directions must rarely transfer donor targets."
    )
    assert result["best_random_patch_target_accuracy"] >= 0.5, (
        "The anomaly audit must retain the strongest accidental random alignment."
    )
    assert result["contract_passed"] and result["accepted"], (
        "Every preregistered scientific criterion must pass together."
    )
    print("All tests in `test_complete_cpu_study_meets_preregistered_claim` passed!")


def test_committed_cpu_artifacts_match_the_live_study() -> None:
    section_dir = _section_dir()
    live = _solutions().run_study(device="cpu")
    committed = json.loads((section_dir / "results/metrics.json").read_text())
    assert committed == live, "Committed metrics should be a direct deterministic study output."

    failures = [
        json.loads(line)
        for line in (section_dir / "results/failure_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [failure["type"] for failure in failures] == [
        "random_direction_alignment",
        "activation_noise_boundary",
    ], "Committed failure cases must preserve both declared anomaly classes."
    report = (section_dir / "reports/capstone.md").read_text()
    assert "## Preregistered claim" in report, "The write-up must state its claim before results."
    assert "## Controls and failure analysis" in report, (
        "The write-up must interpret controls and failure boundaries."
    )
    assert "0.057" in report and "0.841" in report, (
        "The write-up must quote the random-patch mean and sigma-one stress result."
    )
    print("All tests in `test_committed_cpu_artifacts_match_the_live_study` passed!")


def test_notebooks_have_a_complete_live_research_arc() -> None:
    section_dir = _section_dir()
    exercise_path = section_dir / "10.1_Capstone_Research_Sprint_exercises.ipynb"
    solution_path = section_dir / "10.1_Capstone_Research_Sprint_solutions.ipynb"
    for path in [exercise_path, solution_path]:
        notebook = nbformat.read(path, as_version=4)
        text = "\n".join(cell.source for cell in notebook.cells)
        for marker in [
            "By the end of this notebook",
            "Preregistered claim",
            "Exact ground truth",
            "Expected output",
            "<summary>Help",
            "<summary>Interpretation",
            "<summary>Solution",
            "Signature Result",
            "Try It Yourself",
            "Anomaly Hunting",
            "Compact Write-up",
            "Limitations",
            "Reading links",
            "random direction",
        ]:
            assert marker in text, f"{path.name} is missing learner-surface marker {marker!r}."
        assert text.count("### Exercise") >= 5, f"{path.name} needs a genuine exercise progression."
        assert "json.loads" not in text, f"{path.name} must not make JSON loading the lesson."
        assert '["metrics"]["gpu_test"]' not in text, (
            f"{path.name} must generate its signature result rather than plotting report metrics."
        )
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path}:{index}")

    exercise_code = "\n".join(
        cell.source
        for cell in nbformat.read(exercise_path, as_version=4).cells
        if cell.cell_type == "code"
    )
    solution_code = "\n".join(
        cell.source
        for cell in nbformat.read(solution_path, as_version=4).cells
        if cell.cell_type == "code"
    )
    assert exercise_code.count("raise NotImplementedError") >= 5, (
        "The learner notebook must retain at least five substantive implementation stubs."
    )
    assert "raise NotImplementedError" not in solution_code, (
        "The solution notebook must provide every learner implementation."
    )
    assert "fit_ridge_direction" in exercise_code, "Learners must implement mechanism recovery."
    assert "patch_along_direction" in exercise_code, "Learners must implement the causal intervention."
    assert "plt.subplots" in solution_code, "The solution must generate the visible signature figure."

    page = (
        section_dir.parents[1]
        / "instructions/pages/01_[10.1]_Capstone_Research_Sprint.md"
    ).read_text()
    assert "Preregistered claim" in page, "The instruction page must foreground the falsifiable claim."
    assert "Signature Result" in page, "The instruction page must interpret the headline result."
    assert "capstone_parity_signature_result.png" in page, (
        "The instruction page must display the generated parity result."
    )

    for asset_name in [
        "capstone_parity_signature_result.png",
        "capstone_random_direction_anomaly.png",
    ]:
        asset_path = section_dir.parents[1] / "instructions/assets" / asset_name
        with Image.open(asset_path) as image:
            assert image.width >= 1000 and image.height >= 600, (
                f"{asset_name} must remain large enough for learner inspection."
            )
            assert image.convert("RGB").getbbox() is not None, (
                f"{asset_name} must contain rendered, nonblank pixels."
            )
    print("All tests in `test_notebooks_have_a_complete_live_research_arc` passed!")
