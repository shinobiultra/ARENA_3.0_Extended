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

    assert batch.bits.shape == (12, 2)
    assert batch.latent_features.shape == (12, 8)
    assert batch.activations.shape == (12, 8)
    assert batch.labels.shape == (12,)
    assert batch.activations.dtype == t.float64
    assert t.allclose(rotation.T @ rotation, t.eye(8, dtype=t.float64), atol=1e-10)
    assert t.equal(batch.latent_features[:, 2], batch.bits[:, 0] * batch.bits[:, 1])
    for template_id in [0, 1, 2]:
        template_rows = batch.template_ids == template_id
        assert int(template_rows.sum()) == 4
        assert int(batch.labels[template_rows].sum()) == 2

    exact_direction = rotation[:, 2]
    expected_signed_label = batch.labels.to(t.float64) * 2 - 1
    assert t.allclose(batch.activations @ exact_direction, expected_signed_label, atol=1e-10)
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
    assert direction_accuracy(heldout.activations, heldout.labels, learned) == 1.0

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
    assert estimate == 0.5
    assert 0.0 <= low <= estimate <= high <= 1.0
    assert (estimate, low, high) == paired_bootstrap_delta_ci(
        method, baseline, n_resamples=2_000, seed=3
    )
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
    assert t.allclose(patched @ unit, donor @ unit)
    assert t.allclose(patched_residual, recipient_residual)
    assert t.allclose(patched[:, 1:], recipient[:, 1:])
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
    assert t.equal(first, second)
    assert first[0] == 1.0
    assert bool(t.all(first[:-1] >= first[1:])), "The fixed-noise stress curve should be monotone."
    assert first[-1] < 0.70, "Strong activation noise should reveal the method's failure boundary."
    print("All tests in `test_noise_sweep_is_reproducible_and_exposes_failure` passed!")


def test_complete_cpu_study_meets_preregistered_claim() -> None:
    result = _solutions().run_smoke_test(cpu=True)
    assert result["model_family"] == "exact_rotated_parity_model"
    assert result["train_example_count"] == 48
    assert result["heldout_example_count"] == 32
    assert result["heldout_accuracy"] == 1.0
    assert result["raw_bits_accuracy"] == 0.5
    assert result["template_only_accuracy"] == 0.5
    assert result["direction_cosine"] > 0.999
    assert result["learned_patch_target_accuracy"] == 1.0
    assert result["ablation_accuracy"] == 0.5
    assert result["random_patch_target_accuracy_mean"] < 0.20
    assert result["best_random_patch_target_accuracy"] >= 0.5
    assert result["contract_passed"] and result["accepted"]
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
    ]
    report = (section_dir / "reports/capstone.md").read_text()
    assert "## Preregistered claim" in report
    assert "## Controls and failure analysis" in report
    assert "0.057" in report and "0.841" in report
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
        assert text.count("### Exercise") >= 5
        assert "json.loads" not in text
        assert '["metrics"]["gpu_test"]' not in text
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
    assert exercise_code.count("raise NotImplementedError") >= 5
    assert "raise NotImplementedError" not in solution_code
    assert "fit_ridge_direction" in exercise_code
    assert "patch_along_direction" in exercise_code
    assert "plt.subplots" in solution_code

    page = (
        section_dir.parents[1]
        / "instructions/pages/01_[10.1]_Capstone_Research_Sprint.md"
    ).read_text()
    assert "Preregistered claim" in page
    assert "Signature Result" in page
    assert "capstone_parity_signature_result.png" in page

    for asset_name in [
        "capstone_parity_signature_result.png",
        "capstone_random_direction_anomaly.png",
    ]:
        asset_path = section_dir.parents[1] / "instructions/assets" / asset_name
        with Image.open(asset_path) as image:
            assert image.width >= 1000 and image.height >= 600
            assert image.convert("RGB").getbbox() is not None
    print("All tests in `test_notebooks_have_a_complete_live_research_arc` passed!")
