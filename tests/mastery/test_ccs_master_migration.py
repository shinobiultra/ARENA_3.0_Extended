"""Preservation and tiny independent CPU checks; never run the real-model study."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / 'chapter11_representation_geometry/exercises/part5_contrast_consistent_search'
EVIDENCE = ROOT / 'docs/mastery/evidence/ccs_master_migration'
MASTER = ROOT / 'infrastructure/chapters/chapter11_representation_geometry/master_11_5.py'
MODULE = 'chapter11_representation_geometry.exercises.part5_contrast_consistent_search'
STEM = '11.5_Contrast-Consistent_Search'
OLD_STEM = '11.5_Contrast_Consistent_Search'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def source(cell):
    return ''.join(cell['source'])


def codes(notebook):
    return [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']


def read(kind):
    return nbformat.read(SECTION / f'{STEM}_{kind}.ipynb', as_version=4)


def archived(kind):
    before = json.loads((EVIDENCE / 'before.json').read_text())
    with tarfile.open(ROOT / before['archive']) as saved:
        return json.load(saved.extractfile(str((SECTION / f'{OLD_STEM}_{kind}.ipynb').relative_to(ROOT))))


def test_archive_is_exact_and_legacy_publication_remains_present():
    before = json.loads((EVIDENCE / 'before.json').read_text())
    path = ROOT / before['archive']
    assert sha(path.read_bytes()) == before['archive_sha256']
    with tarfile.open(path) as saved:
        for name, expected in before['files'].items():
            assert sha(saved.extractfile(name).read()) == expected, name
            if name != str((SECTION / 'solutions.py').relative_to(ROOT)):
                assert sha((ROOT / name).read_bytes()) == expected, name
    for name, expected in before['legacy_inventory'].items():
        if name != str((SECTION / 'solutions.py').relative_to(ROOT)):
            assert sha((ROOT / name).read_bytes()) == expected, name


def test_both_original_receipts_have_current_source_inputs():
    for filename, count in [('learner_execution.json', 13), ('fresh_reproduction.json', 14)]:
        receipt = json.loads((SECTION / 'expected_outputs' / filename).read_text())
        assert len(receipt['source_inputs']) == count
        for name, expected in receipt['source_inputs'].items():
            assert sha((ROOT / name).read_bytes()) == expected, name
    publication = json.loads((SECTION / 'expected_outputs/learner_execution.json').read_text())
    for kind in ('exercises', 'solutions'):
        assert sha((SECTION / f'{OLD_STEM}_{kind}.ipynb').read_bytes()) == publication[f'published_{kind}_sha256']
    assert publication['code_cells'] == 32 and publication['embedded_figures'] == 15


def test_all_code_ast_nonblank_lines_outputs_and_counts_match_archive():
    old, new = archived('solutions'), read('solutions')
    normalized = nbformat.reads(json.dumps(old), as_version=4)
    parity = json.loads((EVIDENCE / 'output_parity.json').read_text())
    assert parity['execution_kind'] == 'retained_historical_reference'
    assert parity['fresh_model_execution'] is False
    assert len(old['cells']) == 65
    assert len(codes(old)) == len(codes(new)) == len(parity['cells']) == 32
    for index, (a, b, pair) in enumerate(zip(codes(old), codes(new), parity['cells'], strict=True)):
        x, y = source(a), source(b)
        assert [line for line in x.splitlines() if line.strip()] == [line for line in y.splitlines() if line.strip()], index
        assert ast.dump(ast.parse(x)) == ast.dump(ast.parse(y)), index
        assert sha(x.encode()) == pair['published_source_sha256']
        assert sha(y.encode()) == pair['generated_source_sha256']
        assert sha(ast.dump(ast.parse(y)).encode()) == pair['ast_sha256']
        assert sha(json.dumps(a.get('outputs', []), sort_keys=True).encode()) == pair['outputs_sha256']
        assert b.outputs == codes(normalized)[index].outputs
        assert a['execution_count'] == b.execution_count == pair['execution_count']
    assert sum(cell.execution_count is not None for cell in codes(new)) == 32
    assert sum(bool(cell.outputs) for cell in codes(new)) == 30


def test_own_answer_source_matches_archive_and_stored_execution():
    builder = importlib.import_module(MODULE + '.build_notebooks')
    reconstructed = builder.complete_learner({'cells': builder.lesson_cells()})
    old = archived('solutions')
    assert [(c['cell_type'], source(c)) for c in reconstructed['cells']] == [(c['cell_type'], source(c)) for c in old['cells']]
    with tarfile.open(SECTION / 'expected_outputs/full_own_answer_v3/execution.tar.gz') as saved:
        executed = json.load(saved.extractfile('completed_learner.ipynb'))
    for a, b in zip(codes(executed), codes(read('solutions')), strict=True):
        assert ast.dump(ast.parse(source(a))) == ast.dump(ast.parse(source(b)))


def test_fifteen_figures_preserve_exact_original_assets_and_exercise_outputs():
    before = json.loads((EVIDENCE / 'before.json').read_text())
    assert len(before['figures']) == 15
    for name, expected in before['figures'].items():
        assert sha((SECTION / 'media' / name).read_bytes()) == expected
        assert sha((ROOT / 'chapter11_representation_geometry/instructions/assets' / name).read_bytes()) == expected
    exercises = read('exercises')
    html = '\n'.join(''.join(out.get('data', {}).get('text/html', '')) for cell in codes(exercises) for out in cell.outputs)
    embedded = re.findall(r'data:image/png;base64,([^\"]+)', html)
    assert len(embedded) == 15
    assert {sha(base64.b64decode(value)) for value in embedded} == set(before['figures'].values())
    assert set(re.findall(r'media/(ccs_[\w]+\.png)', MASTER.read_text())) == set(before['figures'])


def test_every_student_stub_completes_from_its_own_generated_answer():
    exercises, solutions = read('exercises'), read('solutions')
    nbformat.validate(exercises)
    nbformat.validate(solutions)
    legacy = importlib.import_module(MODULE + '.build_notebooks')
    actual, expected, stubs = [], [], set()
    for i, cell in enumerate(exercises.cells):
        if cell.cell_type != 'code':
            continue
        text = cell.source
        if 'raise NotImplementedError' in text:
            preceding = []
            for candidate in exercises.cells[i + 1:]:
                if candidate.cell_type == 'code':
                    break
                preceding.append(candidate.source)
            answers = re.findall(r'<summary>Solution</summary>(.*?)</details>', '\n'.join(preceding), re.S)
            assert len(answers) == 1
            answer = '\n\n'.join(re.findall(r'```python\n(.*?)```', answers[0], re.S))
            stubs.update(n.name for n in ast.parse(text).body if isinstance(n, ast.FunctionDef))
            text = legacy.complete_from_answer(text, answer)
        actual.append(ast.dump(ast.parse(text)))
    expected = [ast.dump(ast.parse(cell.source)) for cell in codes(solutions)]
    assert actual == expected
    assert stubs == {'fit_pair_centering', 'apply_pair_centering', 'ccs_loss', 'paired_probabilities', 'fit_ccs_restarts', 'select_restart', 'calibrate_orientation', 'binary_metrics', 'fit_difference_pca', 'last_token_rows', 'paired_opinion_change', 'model_answer_metrics'}


def test_original_teaching_lines_survive_and_prediction_pairs_are_specific():
    current = '\n'.join(c.source for c in read('exercises').cells if c.cell_type == 'markdown')
    lines = {line.strip() for line in current.splitlines() if line.strip()}
    for cell in archived('exercises')['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        text = re.sub(r'<details>\s*<summary>Solution</summary>.*?</details>', '', source(cell), flags=re.S)
        for line in text.splitlines():
            if line.strip() and line.strip() != '## Setup' and not line.startswith('# [11.5]'):
                assert line.strip() in lines, line
    assert current.count('### Forecast —') == current.count('### Reflect —') == 10
    for phrase in ('retained historical reference', 'not a fresh execution', 'confidence', 'causal reasoning', 'exposure status', 'prediction-flip fraction', 'whole-vocabulary', 'Try It Yourself', 'shuffled', 'distractor'):
        assert phrase in current, phrase
    cells = read('solutions').cells
    fit_index = next(i for i,c in enumerate(cells) if c.cell_type == 'code' and 'native_summary = ' in c.source)
    forecast_index = next(i for i,c in enumerate(cells) if c.cell_type == 'markdown' and '### Forecast — native-development' in c.source)
    assert forecast_index < fit_index


def test_generated_module_import_has_no_study_side_effects(tmp_path):
    script = '''
import contextlib, importlib, io, os, sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
os.environ['ARENA_CCS_OUTPUT_DIRECTORY'] = str(Path.cwd() / 'must_not_exist')
def forbidden(*args, **kwargs):
    raise AssertionError('Import ran study behavior')
torch.set_num_threads = forbidden
plt.style.use = forbidden
plt.subplots = forbidden
cuda_before = torch.cuda.is_initialized()
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    module = importlib.import_module('chapter11_representation_geometry.exercises.part5_contrast_consistent_search.solutions')
assert not capture.getvalue()
assert not list(Path.cwd().iterdir())
assert torch.cuda.is_initialized() == cuda_before
assert 'transformers' not in sys.modules
for name in ('ccs_loss', 'fit_ccs_restarts', 'fit_reporters', 'reporter_scores', 'last_token_rows', 'paired_opinion_change', 'model_answer_metrics', 'run_smoke_test', 'run_gpu_test'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'Import safety passed'


# Independent hand-worked, NumPy-gradient, AdamW-single-restart and library-SVD oracles.
ORACLES = [
    ('ccs_methods_tests', 'test_pair_centering', ['fit_pair_centering', 'apply_pair_centering']),
    ('ccs_methods_tests', 'test_ccs_loss_values', ['ccs_loss']),
    ('ccs_methods_tests', 'test_ccs_loss_gradient', ['ccs_loss']),
    ('ccs_methods_tests', 'test_paired_inference', ['paired_probabilities']),
    ('ccs_methods_tests', 'test_two_feature_counterexample', ['paired_probabilities', 'ccs_loss']),
    ('ccs_methods_tests', 'test_batched_restarts', ['fit_ccs_restarts']),
    ('ccs_methods_tests', 'test_restart_selection', ['select_restart']),
    ('ccs_methods_tests', 'test_orientation_and_metrics', ['calibrate_orientation', 'binary_metrics']),
    ('ccs_methods_tests', 'test_pca_distractor', ['fit_difference_pca']),
    ('ccs_native_methods_tests', 'test_last_token_rows', ['last_token_rows']),
    ('ccs_native_methods_tests', 'test_paired_opinion_change', ['paired_opinion_change']),
    ('ccs_native_methods_tests', 'test_model_answer_metrics', ['model_answer_metrics']),
]


@pytest.mark.parametrize('test_module,name,functions', ORACLES)
def test_visible_generated_functions_against_independent_cpu_oracles(test_module, name, functions):
    torch = importlib.import_module('torch')
    methods = importlib.import_module(MODULE + '.solutions')
    tests = importlib.import_module(MODULE + '.' + test_module)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        getattr(tests, name)(*[getattr(methods, function) for function in functions])
    finally:
        torch.set_default_dtype(previous)
