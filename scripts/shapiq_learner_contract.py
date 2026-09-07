"""Read-only course integration checks for the executed 16.17 teaching surface."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

METHODS = {'coalition_design', 'sii_query_weights', 'sample_coalitions', 'shapiq_estimate',
           'query_sampled_game', 'constrained_wls', 'pair_residual_design', 'kernel_weights', 'kernelshap_iq'}
STEM = '16.17_One_Query_Many_Interactions'
PREFIX = 'chapter16_shapley_attribution_baselines/exercises/part17_shapiq_estimators/'


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_native_evidence(report):
    """Require full coverage, source-only refits and hashes bound by the report manifest."""
    values = report.get('metrics', {}).get('gpu_test', {})
    expected = {
        'study': '16.17_shapiq_estimators', 'device': 'cpu', 'actual_learner_passed': True,
        'native_model_retrained_in_this_report': True, 'cached_model_inputs_used': False,
        'fresh_reproduction_passed': True, 'fresh_compared_arrays': 663,
        'fresh_compared_elements': 118275724,
        'fresh_comparison': 'exact_recursive_dtype_shape_value_sha256_including_container_order',
        'own_answer_code_cells': 22, 'embedded_figures': 10, 'model_fits': 7,
        'known_model_state_hashes_matched': 21, 'train_rows': 758,
        'explanation_rows': 153, 'audit_rows': 119, 'model_games': 1904,
        'coalition_values': 487424, 'donor_predictions': 31195136,
        'replicates_per_proposal': 32, 'proposals': ['kernel', 'uniform'],
        'nested_draw_budgets': [16,32,64,128,256,512,1024],
        'signed_pair_estimates': 47767552, 'author_reference_version': '1.5.2',
        'author_reference_games': 7, 'author_reference_algorithms': 3,
        'author_reference_passed': True, 'logical_query_benchmark': True,
        'precomputed_oracle': True, 'cross_split_recipe_overlap': 0,
        'method_superiority_is_acceptance_gate': False, 'review_scope': 'author_review_not_independent',
    }
    if any(values.get(name) != value for name, value in expected.items()):
        return False
    entries = report.get('report_inputs', {}).get('files', [])
    for field, filename in [('learner_evidence_sha256', 'learner_execution.json'),
                            ('fresh_evidence_sha256', 'fresh_reproduction.json')]:
        matches = [item for item in entries if item.get('path') == PREFIX + 'expected_outputs/' + filename]
        checksum = values.get(field)
        if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
            return False
        if len(matches) != 1 or matches[0].get('sha256') != checksum:
            return False
    return True


def has_learner_contract(path):
    """Check visible methods, adjacent answers, experiment wiring and published provenance.

    This is a structural/source freshness check. Actual native-array and author-package
    comparisons are performed by the registered report, not inferred from names here.
    """
    path = Path(path)
    if path.name != STEM + '_exercises.ipynb':
        return False
    here, root = path.parent, path.parent.parents[2]
    try:
        notebook = json.loads(path.read_text())
        published = json.loads((here / 'expected_outputs/learner_execution.json').read_text())
        execution = json.loads((here / 'expected_outputs/own_answer_execution.json').read_text())
        if published['published_exercises_sha256'] != _digest(path) or execution['status'] != 'PASS':
            return False
        if not execution['completed_from_own_answers'] or execution['changed_sources'] or execution['analytic_only']:
            return False
        required = {PREFIX + name for name in ['estimator_methods.py','estimator_methods_tests.py',
                    'estimator_study.py','estimator_plots.py','lesson.md','lesson_cells.py',
                    'build_notebooks.py','execute_learner.py','protocol.md','data_manifest.json','protocol_freeze.json']}
        if not required <= execution['source_inputs'].keys():
            return False
        if any(_digest(root / name) != value for name, value in execution['source_inputs'].items()):
            return False
        definitions = {node.name: ast.dump(node) for node in ast.parse((here / 'estimator_methods.py').read_text()).body if isinstance(node, ast.FunctionDef)}
        answers, stubs, tests, calls, namespace = {}, set(), set(), [], None
        figures = 0
        for cell in notebook['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'markdown':
                if cell.get('metadata', {}).get('arena', {}).get('solution_for'):
                    match = re.search(r'<summary>Solution</summary>\s*```python\n(.*?)\n```', source, re.S)
                    if not match:
                        return False
                    answers.update({node.name: ast.dump(node) for node in ast.parse(match[1]).body if isinstance(node, ast.FunctionDef)})
                continue
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    if any(isinstance(child, ast.Raise) and 'NotImplementedError' in ast.unparse(child) for child in ast.walk(node)):
                        stubs.add(node.name)
                    continue
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'methods' for target in node.targets):
                    if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != 'SimpleNamespace':
                        return False
                    namespace = {kw.arg: ast.unparse(kw.value) for kw in node.value.keywords}
                for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
                    name = ast.unparse(call.func)
                    calls.append((name, [ast.unparse(arg) for arg in call.args]))
                    if name.startswith('tests.test_') and any(kw.arg == 'fn' and ast.unparse(kw.value) == name[11:] for kw in call.keywords):
                        tests.add(name[11:])
            figures += sum('image/png' in item.get('data', {}) for item in cell.get('outputs', []))
        return (stubs == METHODS and tests == METHODS and namespace == {name: name for name in METHODS}
                and all(answers.get(name) == definitions[name] for name in METHODS)
                and ('study.run_study', ['methods', "OUTPUT_DIR / 'native'"]) in calls
                and ('plots.make_native_play', ['payload', 'methods']) in calls
                and figures == execution['embedded_figures'] == 10)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError):
        return False
