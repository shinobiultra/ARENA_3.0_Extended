"""The direct learner contract must reject missing student ownership and stale evidence."""

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.shapiq_learner_contract import PREFIX, STEM, has_learner_contract, has_native_evidence
from scripts.audit_arena_style_depth import _has_signature_result

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / PREFIX


def test_actual_published_learner_contract():
    assert has_learner_contract(HERE / (STEM + '_exercises.ipynb'))


@pytest.mark.parametrize('mutation', ['hidden_namespace', 'missing_test', 'missing_figure', 'wrong_answer'])
def test_notebook_mutations_rejected_even_with_updated_publication_hash(tmp_path, mutation):
    execution = json.loads((HERE / 'expected_outputs/own_answer_execution.json').read_text())
    for name in execution['source_inputs']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    here = tmp_path / PREFIX
    (here / 'expected_outputs').mkdir(exist_ok=True)
    for name in ['learner_execution.json', 'own_answer_execution.json']:
        shutil.copy2(HERE / 'expected_outputs' / name, here / 'expected_outputs' / name)
    notebook = json.loads((HERE / (STEM + '_exercises.ipynb')).read_text())
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if mutation == 'hidden_namespace' and 'methods = SimpleNamespace(' in source:
            cell['source'] = source.replace('shapiq_estimate=shapiq_estimate', 'shapiq_estimate=hidden_method')
            break
        if mutation == 'missing_test' and 'tests.test_coalition_design(fn=coalition_design)' in source:
            cell['source'] = source.replace('tests.test_coalition_design(fn=coalition_design)', '')
            break
        if mutation == 'wrong_answer' and cell.get('metadata', {}).get('arena', {}).get('solution_for') == ['coalition_design']:
            cell['source'] = source.replace('return masks', 'return 2 * masks')
            break
        if mutation == 'missing_figure' and any('image/png' in item.get('data', {}) for item in cell.get('outputs', [])):
            cell['outputs'] = []
            break
    else:
        raise AssertionError('Mutation did not target the actual lesson.')
    path = here / (STEM + '_exercises.ipynb')
    path.write_text(json.dumps(notebook))
    record_path = here / 'expected_outputs/learner_execution.json'
    record = json.loads(record_path.read_text())
    record['published_exercises_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record))
    assert not has_learner_contract(path)


def test_plot_helper_requires_visible_result():
    assert _has_signature_result('plots.plot_convergence(payload)\n![actual plot](figure.png)')
    assert not _has_signature_result('plots.plot_convergence(payload)')
    assert not _has_signature_result('![actual plot](figure.png)')


def test_native_report_coverage_and_hashes_are_required():
    report = json.loads((HERE / 'verification_report.json').read_text())
    assert has_native_evidence(report)
    for field, value in [('donor_predictions', 16384), ('cached_model_inputs_used', True),
                         ('fresh_compared_elements', 118275723), ('embedded_figures', 0),
                         ('fresh_evidence_sha256', '0' * 64)]:
        bad = copy.deepcopy(report)
        bad['metrics']['gpu_test'][field] = value
        assert not has_native_evidence(bad), field
