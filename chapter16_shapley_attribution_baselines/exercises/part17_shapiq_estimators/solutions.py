"""Visible method answers and the registered fresh-CPU verification entry point."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.estimator_methods import *
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods_tests as tests


def run_smoke_test(cpu: bool = True) -> dict:
    names = ('coalition_design','sii_query_weights','sample_coalitions','shapiq_estimate',
             'query_sampled_game','constrained_wls','pair_residual_design','kernel_weights','kernelshap_iq')
    for name in names:
        getattr(tests, 'test_' + name)()
    return {'tests_passed': True, 'accepted': True, 'test_count': len(names),
            'student_method_exercises': 8, 'student_core_functions': 9,
            'claim_scope': 'Analytic immediate method checks; the native model study runs separately.'}


def run_gpu_test(max_vram_gb: float = 0.0) -> dict:
    """Legacy API name; actually refit all seven models and recompute the CPU study."""
    from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.execute_learner import digest
    from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.native_contracts import load_payloads, study_metrics
    from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.publish_study import validate_fresh, validate_published
    from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.reproduce_native import reproduce

    published = validate_published()
    learner = Path(published['learner_directory'])
    cache = learner.parent / 'report_reproductions'
    cache.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='report-', dir=cache)) / 'fresh_cpu'
    reproduce(learner, output)
    execution, fresh = validate_fresh(learner, output)
    metrics = study_metrics(load_payloads(output / 'fresh_native'))
    consumer_files = sorted(HERE.glob('*.py'))
    result = {'cuda_available': False, 'device': 'cpu', 'peak_vram_gb': 0.0,
              'within_vram_budget': max_vram_gb >= 0.0, 'actual_learner_passed': execution['status'] == 'PASS',
              'own_answer_code_cells': execution['executed_code_cells'], 'embedded_figures': execution['embedded_figures'],
              'native_model_retrained_in_this_report': fresh['native_models_refit'],
              'cached_model_inputs_used': fresh['cached_model_inputs_used'],
              'fresh_reproduction_passed': fresh['status'] == 'PASS',
              'fresh_compared_arrays': fresh['counts']['arrays'], 'fresh_compared_elements': fresh['counts']['array_elements'],
              'fresh_comparison': fresh['comparison'], 'fresh_output_directory': str(output),
              'learner_evidence_sha256': digest(HERE / 'expected_outputs/learner_execution.json'),
              'fresh_evidence_sha256': digest(output / 'reproduction.json'),
              'review_scope': 'author_review_not_independent', **metrics}
    # Report consumers are separate from the immutable code actually executed in the learner.
    receipt = {'result': result, 'consumer_inputs': {str(p.relative_to(ROOT)): digest(p) for p in consumer_files},
               'fresh_reproduction_sha256': digest(output / 'reproduction.json')}
    (output / 'report_metrics.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.copy2(output / 'reproduction.json', HERE / 'expected_outputs/fresh_reproduction.json')
    shutil.copy2(output / 'report_metrics.json', HERE / 'expected_outputs/reference_metrics.json')
    return result


def run_full_experiment(max_vram_gb: float = 0.0) -> dict:
    return {'numerical': run_smoke_test(), 'fresh_cpu_model': run_gpu_test(max_vram_gb)}
