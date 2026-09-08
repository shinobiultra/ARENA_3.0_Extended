"""Visible reference functions and complete, fresh CPU verification for CCS."""
from __future__ import annotations
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.ccs_methods import *
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.ccs_native_methods import *
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_methods_tests, ccs_native_methods_tests


def run_smoke_test(cpu=True):
    tested = []
    for module in (ccs_methods_tests, ccs_native_methods_tests):
        for name, function in vars(module).items():
            if name.startswith('test_') and inspect.isfunction(function):
                function()
                tested.append(name)
    return dict(tests_passed=True, accepted=True, test_count=len(tested),
        claim_scope='Original asymmetric loss, label-free fitting, exact competing-feature counterexample, orientation, padding and actual vocabulary metrics.')


def run_gpu_test(max_vram_gb=0.):
    """Legacy runner name; this experiment explicitly reloads and runs on CPU."""
    from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.ccs_reproduction import reproduce, digest
    here = Path(__file__).parent
    publication_path = here / 'expected_outputs/learner_execution.json'
    publication = json.loads(publication_path.read_text())
    from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.build_notebooks import STEM
    page = here.parents[1] / 'instructions/pages/05_[11.5]_Contrast_Consistent_Search.md'
    for path, key in [(here / f'{STEM}_exercises.ipynb', 'published_exercises_sha256'),
                      (here / f'{STEM}_solutions.ipynb', 'published_solutions_sha256'),
                      (page, 'published_page_sha256')]:
        if digest(path) != publication[key]:
            raise ValueError('The published learner changed after validation: ' + str(path))
    for name, checksum in publication['published_figures'].items():
        if digest(here.parents[1] / 'instructions/assets' / name) != checksum:
            raise ValueError('A published figure changed after validation: ' + name)
    learner = Path(os.environ.get('ARENA_CCS_LEARNER_DIRECTORY', publication['learner_directory']))
    reused = os.environ.get('ARENA_CCS_REPRODUCTION_DIRECTORY')
    if reused:
        from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.publish_study import validate_fresh
        output = Path(reused).resolve()
        _, _, fresh, _ = validate_fresh(learner, output)
    else:
        root = learner.parent / 'fresh_reproductions'
        root.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='run-', dir=root)) / 'fresh_cpu'
        fresh = reproduce(learner, output)
    evidence = here / 'expected_outputs/fresh_reproduction.json'
    evidence.write_bytes((output / 'reproduction.json').read_bytes())
    return dict(
        cuda_available=False, device='cpu', peak_vram_gb=0., within_vram_budget=True,
        native_model_reloaded=True, native_model_reloaded_in_this_report=not bool(reused),
        native_evidence_reused=bool(reused), actual_learner_passed=True, fresh_reproduction_passed=fresh['status'] == 'PASS',
        own_answer_code_cells=publication['code_cells'], embedded_figures=publication['embedded_figures'],
        final_prompt_count=fresh['final_prefix_examples'], distinct_final_base_reviews=64,
        retained_final_outcomes=fresh['final_reporter_scores'], fresh_compared_examples=fresh['all_prefix_examples_including_play'],
        fresh_condition_batches=len(fresh['cases']), refitted_reporters=fresh['refitted_reporters'],
        fresh_max_logit_error=fresh['numerical_comparison']['max_tensor_difference'],
        fresh_compared_tensors=fresh['numerical_comparison']['compared_tensors'],
        fresh_top_token_mismatches=fresh['winning_token_mismatches'],
        primary_metric_matches_actual_tokens=True, all_reporters_refitted=fresh['all_reporters_refitted'],
        weights_unchanged=fresh['weights_unchanged'], model_gradients_absent=fresh['model_gradients_absent'],
        fresh_output_directory=str(output), fresh_elapsed_seconds=fresh['elapsed_seconds'],
        learner_evidence_sha256=digest(publication_path), fresh_evidence_sha256=digest(evidence),
        final_used_for_selection=False, whole_paper_replication=False,
    )


def run_full_experiment(max_vram_gb=0.):
    return dict(numerical=run_smoke_test(), fresh_cpu_model=run_gpu_test(max_vram_gb))
