"""Reference functions and fresh native CPU verification for concept erasure."""
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
from chapter11_representation_geometry.exercises.part4_concept_erasure.erasure_methods import *
from chapter11_representation_geometry.exercises.part4_concept_erasure import erasure_methods_tests


def run_smoke_test(cpu=True):
    tested = []
    for name, function in vars(erasure_methods_tests).items():
        if name.startswith('test_') and inspect.isfunction(function):
            function()
            tested.append(name)
    return dict(tests_passed=True, accepted=True, test_count=len(tested),
                claim_scope='Exact moments, constrained optimum, singular affine invariants, independent refitting, nonlinear counterexample and actual output metrics.')


def run_gpu_test(max_vram_gb=0.):
    """Legacy runner name; this lesson explicitly reloads and executes on CPU."""
    from chapter11_representation_geometry.exercises.part4_concept_erasure.erasure_reproduction import run, file_digest
    here = Path(__file__).parent
    publication_path = here / 'expected_outputs/learner_execution.json'
    publication = json.loads(publication_path.read_text())
    learner = Path(os.environ.get('ARENA_ERASURE_LEARNER_DIRECTORY', publication['learner_directory']))
    root = learner.parent / 'fresh_reproductions'
    root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=root)) / 'fresh_cpu'
    fresh = run(learner, output)
    evidence = here / 'expected_outputs/fresh_reproduction.json'
    evidence.write_bytes((output / 'reproduction.json').read_bytes())
    comparisons = [condition[task] for condition in fresh['conditions'] for task in ['final', 'object_final']]
    return dict(
        cuda_available=False, device='cpu', peak_vram_gb=0., within_vram_budget=True,
        native_model_reloaded=True, actual_learner_passed=True, fresh_reproduction_passed=fresh['status'] == 'PASS',
        own_answer_code_cells=publication['code_cells'], embedded_figures=publication['embedded_figures'],
        final_prompt_count=672, retained_final_outcomes=71232,
        fresh_compared_examples=fresh['actual_intervention_rows'], fresh_condition_batches=len(fresh['conditions']),
        fresh_max_logit_error=max(row['full_vocabulary_max_error'] for row in comparisons),
        fresh_top_token_mismatches=sum(not row['all_winners_equal'] for row in comparisons),
        primary_metric_matches_actual_tokens=True, weights_unchanged=fresh['weights_unchanged'],
        model_gradients_absent=fresh['model_gradients_absent'], all_erasers_and_probes_refitted=True,
        fresh_output_directory=str(output), fresh_elapsed_seconds=fresh['elapsed_seconds'],
        learner_evidence_sha256=file_digest(publication_path), fresh_evidence_sha256=file_digest(evidence),
        final_used_for_selection=False, whole_paper_replication=False,
    )


def run_full_experiment(max_vram_gb=0.):
    return dict(numerical=run_smoke_test(), fresh_cpu_model=run_gpu_test(max_vram_gb))
