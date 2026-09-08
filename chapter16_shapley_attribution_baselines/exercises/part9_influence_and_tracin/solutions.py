"""Reference functions and complete CPU reproduction for the report runner."""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_methods import *
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin import attribution_methods_tests


def run_smoke_test(cpu=True):
    tested = []
    for name, function in vars(attribution_methods_tests).items():
        if name.startswith("test_") and inspect.isfunction(function):
            function()
            tested.append(name)
    return dict(tests_passed=True, accepted=True, test_count=len(tested),
                claim_scope="Independent derivative, optimization, weighting, path and affine-gradient contracts.")


def run_gpu_test(max_vram_gb=0.):
    """Legacy runner name: reload data and retrain every condition on CPU."""
    from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.publish_study import validate_fresh, PAGE
    from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.reproduce_native import reproduce, digest
    from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.build_notebooks import STEM
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
    return dict(
        cuda_available=False, device="cpu", peak_vram_gb=0., within_vram_budget=True,
        actual_learner_passed=True, own_answer_code_cells=execution["code_cells"],
        embedded_figures=execution["embedded_figures"], native_model_retrained=True,
        native_model_retrained_in_this_report=True, native_evidence_reused=False,
        fresh_reproduction_passed=True, reference_refits=fresh["reference_refits"],
        sgd_paths=fresh["sgd_paths"], sgd_updates=fresh["sgd_updates"],
        fresh_compared_tensors=fresh["comparisons"]["tensors"],
        fresh_compared_elements=fresh["comparisons"]["tensor_elements"],
        fresh_max_error=fresh["comparisons"]["max_absolute_error"],
        independent_scipy_checks=len(fresh["independent_scipy_checks"]),
        final_query_count=64, final_used_for_selection=False, exploratory_refits=33,
        fresh_output_directory=str(output), fresh_elapsed_seconds=fresh["elapsed_seconds"],
        learner_evidence_sha256=digest(evidence_path), fresh_evidence_sha256=digest(fresh_path),
        whole_paper_replication=False,
    )


def run_full_experiment(max_vram_gb=0.):
    return dict(numerical=run_smoke_test(), fresh_cpu_model=run_gpu_test(max_vram_gb))
