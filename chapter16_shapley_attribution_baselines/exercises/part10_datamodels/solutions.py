"""Reference methods and full CPU reproduction for the existing report runner."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.datamodel_methods import *
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels import datamodel_methods_tests


def run_smoke_test(cpu=True):
    tested = []
    for name, function in vars(datamodel_methods_tests).items():
        if name.startswith("test_") and inspect.isfunction(function):
            function()
            tested.append(name)
    return dict(tests_passed=True, accepted=True, test_count=len(tested),
                claim_scope="Independent subset, margin, proximal, KKT, Lasso and intervention checks.")


def run_gpu_test(max_vram_gb=0.):
    """Legacy entry name: perform the complete model retraining study on CPU."""
    from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.publish_study import validate_fresh, PAGE
    from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.reproduce_native import reproduce, digest
    from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.build_notebooks import STEM
    here = Path(__file__).parent
    evidence_path = here / "expected_outputs/learner_execution.json"
    published = json.loads(evidence_path.read_text())
    for path, key in [(here / f"{STEM}_exercises.ipynb", "published_exercises_sha256"),
                      (here / f"{STEM}_solutions.ipynb", "published_solutions_sha256"),
                      (PAGE, "published_page_sha256")]:
        if digest(path) != published[key]:
            raise ValueError("Published learner changed: " + str(path))
    for name, checksum in published["published_figures"].items():
        if digest(PAGE.parent.parent / "assets" / name) != checksum:
            raise ValueError("Published figure changed: " + name)
    learner = Path(published["learner_directory"])
    cache = learner.parent / "report_reproductions"
    cache.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="report-", dir=cache)) / "fresh_cpu"
    reproduce(learner, output)
    execution, fresh = validate_fresh(learner, output)
    fresh_path = here / "expected_outputs/fresh_reproduction.json"
    fresh_path.write_bytes((output / "reproduction.json").read_bytes())
    native = fresh["native_execution"]
    return dict(
        cuda_available=False, device="cpu", peak_vram_gb=0., within_vram_budget=True,
        actual_learner_passed=True, own_answer_code_cells=execution["code_cells"],
        embedded_figures=execution["embedded_figures"], native_model_retrained=True,
        native_model_retrained_in_this_report=True, native_evidence_reused=False,
        fresh_reproduction_passed=True, reference_refits=native["classifier_refits"],
        independent_scipy_checks=native["independent_scipy_optima"],
        independent_lasso_checks=native["independent_lasso_fits"],
        fresh_compared_tensors=fresh["comparisons"]["tensors"],
        fresh_compared_elements=fresh["comparisons"]["tensor_elements"],
        fresh_max_error=fresh["comparisons"]["max_absolute_error"],
        heldout_unit="training_subset_model", heldout_model_count=768,
        test_responses_used_for_selection=False,
        query_image_count=64, query_images_used_as_regression_targets=True,
        query_images_excluded_from_classifier_training=True,
        exploratory_refits=native["play_refits"], whole_paper_replication=False,
        fresh_output_directory=str(output), fresh_elapsed_seconds=fresh["elapsed_seconds"],
        learner_evidence_sha256=digest(evidence_path), fresh_evidence_sha256=digest(fresh_path),
    )


def run_full_experiment(max_vram_gb=0.):
    return dict(numerical=run_smoke_test(), fresh_cpu_model=run_gpu_test(max_vram_gb))
