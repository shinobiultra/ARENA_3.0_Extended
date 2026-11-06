"""17.1 source/output preservation and tiny CPU oracles, never training."""
from __future__ import annotations

import ast
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

from scripts.verified_presentation import verified_pre_wrap_notebook

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / 'chapter17_training_dynamics/exercises/part1_checkpoint_archaeology'
EVIDENCE = ROOT / 'docs/mastery/evidence/checkpoint_master_migration'
MASTER = ROOT / 'infrastructure/chapters/chapter17_training_dynamics/master_17_1.py'
OLD = '17.1_Checkpoint_Archaeology_and_Mechanism_Emergence'
STEM = '17.1_Checkpoint_Archaeology_and_Grokking'
MODULE = 'chapter17_training_dynamics.exercises.part1_checkpoint_archaeology'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    return ''.join(cell['source'])


def codes(notebook):
    return [c for c in notebook['cells'] if c['cell_type'] == 'code']


def read(kind, stem=STEM):
    return nbformat.read(SECTION / f'{stem}_{kind}.ipynb', as_version=4)


def normalize(text):
    return text.replace(
        'from chapter17_training_dynamics.exercises.part1_checkpoint_archaeology.checkpoint_reports import (',
        'from arena_ext.training_dynamics import (',
    ).replace(
        'figure_path = section_dir / "media" / "checkpoint_archaeology_signature_result.png"',
        'figure_path = assets_dir / "checkpoint_archaeology_signature_result.png"',
    )


def test_archive_historical_inputs_and_legacy_artifacts_are_retained():
    before = json.loads((EVIDENCE / 'before.json').read_text())
    archive = ROOT / before['archive']
    assert sha(archive.read_bytes()) == before['archive_sha256']
    with tarfile.open(archive) as saved:
        for name, expected in before['files'].items():
            assert sha(saved.extractfile(name).read()) == expected, name
            # Only generated current solutions.py is replaced. The old module has an exact physical copy.
            current = ROOT / name
            if current == SECTION / 'solutions.py':
                current = SECTION / 'legacy_solutions.py'
            assert sha(current.read_bytes()) == expected, name
    for row in before['report_inputs']:
        current = ROOT / row['path']
        if current == SECTION / 'solutions.py':
            current = SECTION / 'legacy_solutions.py'
        assert sha(current.read_bytes()) == row['sha256'], row['path']
    assert before['report_date_run'] == '2026-09-04T23:53:46+00:00'
    assert before['published_cells'] == 35


def test_historical_output_and_scientific_source_parity():
    parity = json.loads((EVIDENCE / 'output_parity.json').read_text())
    oldraw = json.loads((SECTION / f'{OLD}_solutions.ipynb').read_text())
    old = read('solutions', OLD)
    new = read('solutions')
    assert parity['execution_kind'] == 'retained_historical_reference'
    assert parity['fresh_model_execution'] is False
    assert len(codes(old)) == len(codes(new)) == len(parity['cells']) == 14
    for a, b, raw, row in zip(codes(old), codes(new), codes(oldraw), parity['cells'], strict=True):
        x, y = a.source, b.source
        assert ast.dump(ast.parse(x)) == ast.dump(ast.parse(normalize(y)))
        assert [v for v in x.splitlines() if v.strip()] == [v for v in normalize(y).splitlines() if v.strip()]
        assert sha(source(raw).encode()) == row['published_source_sha256']
        assert sha(y.encode()) == row['generated_source_sha256']
        assert sha(ast.dump(ast.parse(normalize(y))).encode()) == row['normalized_ast_sha256']
        assert b.outputs == a.outputs
        assert b.execution_count == a.execution_count == row['execution_count']
        assert sha(json.dumps(raw.get('outputs', []), sort_keys=True).encode()) == row['outputs_sha256']
    assert sum(c.execution_count is not None for c in codes(new)) == 14
    assert sum(bool(c.outputs) for c in codes(new)) == 13
    # The CUDA wrapper is authoritative in the notebook, not the slightly different old module.
    functions = lambda text: {n.name: ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}
    published = functions('\n\n'.join(c.source for c in codes(old)))
    generated = functions((SECTION / 'solutions.py').read_text())
    legacy = functions((SECTION / 'legacy_solutions.py').read_text())
    assert published['run_modular_addition_checkpoint_archaeology_preflight'] == generated['run_modular_addition_checkpoint_archaeology_preflight']
    assert published['run_modular_addition_checkpoint_archaeology_preflight'] != legacy['run_modular_addition_checkpoint_archaeology_preflight']
    assert len(published) == 30  # 27 top-level functions, nested study runner and two model methods
    assert all(generated[name] == value for name, value in published.items())


def test_media_pixels_and_minimal_helper_dependency_closure():
    import base64

    old = read('solutions', OLD)
    for row in json.loads((EVIDENCE / 'figures.json').read_text()):
        pngs = [o.data['image/png'] for o in old.cells[row['published_cell']].outputs if o.get('data', {}).get('image/png')]
        assert len(pngs) == 1
        assert sha(base64.b64decode(pngs[0])) == row['sha256']
        assert sha((SECTION / 'media' / row['name']).read_bytes()) == row['sha256']
    provenance = json.loads((EVIDENCE / 'helper_provenance.json').read_text())
    old_helper = (ROOT / provenance['source']).read_bytes()
    new_helper = (ROOT / provenance['local_copy']).read_bytes()
    assert sha(old_helper) == provenance['source_sha256']
    assert sha(new_helper) == provenance['local_sha256']
    assert ast.dump(ast.parse(old_helper)) == ast.dump(ast.parse(new_helper))
    tree = ast.parse(new_helper)
    definitions = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    imports = next(n for n in ast.parse(codes(read('solutions'))[0].source).body if isinstance(n, ast.ImportFrom) and n.module.endswith('checkpoint_reports'))
    required = {n.name for n in imports.names}
    pending = list(required)
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    while pending:
        for node in ast.walk(nodes[pending.pop()]):
            if isinstance(node, ast.Name) and node.id in definitions - required:
                required.add(node.id)
                pending.append(node.id)
    assert required == definitions, 'Do not copy unused scientific helpers.'


def test_own_visible_solution_dropdowns_complete_every_exercise():
    ex, sol = read('exercises'), read('solutions')
    nbformat.validate(ex)
    nbformat.validate(sol)
    completed, count, stub_names = [], 0, set()
    for index, cell in enumerate(ex.cells):
        if cell.cell_type != 'code':
            continue
        text = cell.source
        if 'raise NotImplementedError' in text:
            count += 1
            following = []
            for next_cell in ex.cells[index + 1:]:
                if next_cell.cell_type == 'code':
                    break
                following.append(next_cell.source)
            blocks = re.findall(r'<summary>Solution</summary>(.*?)</details>', '\n'.join(following), re.S)
            assert len(blocks) == 1
            answer = '\n\n'.join(re.findall(r'```python\n(.*?)```', blocks[0], re.S))
            answers = {n.name: ast.get_source_segment(answer, n) for n in ast.parse(answer).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            lines = text.splitlines()
            for node in reversed(ast.parse(text).body):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and any(isinstance(n, ast.Raise) and 'NotImplementedError' in ast.unparse(n) for n in ast.walk(node)):
                    stub_names.add(node.name)
                    assert node.name in answers
                    lines[node.lineno - 1:node.end_lineno] = answers[node.name].splitlines()
            text = '\n'.join(lines)
            assert 'tests.test_' in text or any(isinstance(n, ast.Assert) for n in ast.parse(text).body), 'Each exercise retains immediate semantic tests or acceptance assertions.'
        completed.append(text)
    assert count == 6
    assert len(stub_names) == 27
    for a, b in zip(completed, codes(sol), strict=True):
        assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b.source))
    assert all(c.execution_count is None for c in codes(ex))


def test_explanations_forecasts_controls_and_play_survive():
    notebook = verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb')
    text = '\n'.join(c.source for c in notebook.cells)
    assert len(notebook.cells) >= 62
    assert len(re.findall(r'^### Exercise', text, re.M)) == 6
    for heading in ('## Forecast before the result', '## Reflect after the result'):
        assert len(re.findall('^' + heading, text, re.M)) == 6
    assert text.index('Forecast before the result — learned-checkpoints') < text.index('signature_result = checkpoint_archaeology_signature_result(')
    for term in (
        'finite-domain mechanism emergence',
        'Grokking coverage remains missing',
        'no held-out split',
        'retained historical reference',
        'not a new readiness promotion',
        'guided practice',
        'confidence (0–100%)',
        'Try It Yourself',
        'PLAY_RANDOM_MASK_SEED',
        'names in the next cell are unused',
        'hidden width 96→32',
        'same-size random',
        'random-label',
        'Limitations',
        'Reading Links'
    ):
        assert term in text, term
    assert 'from arena_ext' not in MASTER.read_text()
    current_lines = {line.strip() for line in text.splitlines() if line.strip()}
    for cell in read('solutions', OLD).cells:
        if cell.cell_type != 'markdown':
            continue
        original = re.sub(r'<details>\s*<summary>Solution</summary>\s*```python\n.*?```\s*</details>', '', cell.source, flags=re.S)
        for line in original.splitlines():
            line = line.strip()
            if not line or line.startswith(('# [17.1]', '> **Notebooks:', '## Learning Objectives', '> Difficulty:', '> Importance:', '<img src="../../instructions/assets/checkpoint_archaeology_signature_result.png"')):
                continue
            assert line in current_lines, line


def test_import_is_safe_outside_repository(tmp_path):
    script = r'''
import contextlib, importlib, io, os, sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import IPython.display
import nbformat
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
def forbidden(*args, **kwargs):
    raise AssertionError('Import triggered a lesson side effect')
torch.manual_seed = forbidden
torch.nn.Module.__init__ = forbidden
torch.load = forbidden
torch.save = forbidden
plt.subplots = forbidden
IPython.display.display = forbidden
state = torch.random.get_rng_state().clone()
cuda_initialized = torch.cuda.is_initialized()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    module = importlib.import_module('chapter17_training_dynamics.exercises.part1_checkpoint_archaeology.solutions')
assert not captured.getvalue(), captured.getvalue()
assert not list(Path.cwd().iterdir())
assert torch.equal(state, torch.random.get_rng_state())
assert torch.cuda.is_initialized() == cuda_initialized
assert 'transformers' not in sys.modules
for name in ('addition_fourier_mask', 'fourier_power_fraction', 'project_logits_to_fourier_mask', 'train_save_reload_modular_checkpoints', 'checkpoint_archaeology_signature_result', 'run_gpu_test'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT), str(tmp_path)], text=True, capture_output=True, check=True)
    assert 'Import safety passed' in result.stdout


@pytest.fixture(scope='module')
def methods():
    return importlib.import_module(MODULE + '.solutions')


def test_tiny_finite_table_and_exact_fourier_oracles(methods):
    import torch as t

    p = 5
    pairs, labels = methods.make_modular_addition_dataset(modulus=p)
    assert pairs.tolist() == [[a, b] for a in range(p) for b in range(p)]
    assert labels.tolist() == [(a + b) % p for a in range(p) for b in range(p)]
    logits = methods.exact_addition_logits(modulus=p)
    legal = methods.addition_fourier_mask(modulus=p)
    expected = t.zeros((p, p, p), dtype=t.bool)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                expected[a, b, c] = c != 0 and a == b == (-c) % p
    assert t.equal(legal, expected)
    assert methods.fourier_power_fraction(logits, legal, modulus=p) == pytest.approx(1.0, abs=1e-6)
    random = methods.random_fourier_mask(modulus=p, seed=7)
    assert int(random.sum()) == int(legal.sum())
    assert not bool((random & legal).any())
    for a, b, c in random.nonzero().tolist():
        assert random[-a % p, -b % p, -c % p]
    assert methods.fourier_power_fraction(logits, random, modulus=p) < 1e-10
    centered = logits - logits.mean(-1, keepdim=True)
    kept = methods.project_logits_to_fourier_mask(logits, legal, modulus=p, keep=True)
    ablated = methods.project_logits_to_fourier_mask(logits, legal, modulus=p, keep=False)
    assert t.allclose(kept, logits, atol=2e-6)
    assert t.allclose(ablated, logits.mean(-1, keepdim=True).expand_as(logits), atol=2e-6)
    assert t.allclose(kept + ablated - logits.mean(-1, keepdim=True), logits, atol=2e-6)
    assert methods.fourier_power_fraction(centered * 3, legal, modulus=p) == pytest.approx(1.0, abs=1e-6)
    assert methods.fourier_power_fraction(t.ones_like(logits), legal, modulus=p) == 0
    with pytest.raises(ValueError, match='shape'):
        methods.fourier_power_fraction(t.zeros((3, 3)), legal, modulus=p)


def test_stable_crossing_and_matched_control_oracles(methods):
    import torch as t

    steps = t.tensor([0, 2, 4, 8, 16])
    values = t.tensor([0.1, 0.6, 0.2, 0.7, 0.8])
    assert methods.first_threshold_crossing(steps, values, threshold=0.5) == 2
    assert methods.stable_threshold_step(steps, values, threshold=0.5, min_consecutive=2) == 8
    assert methods.monotonicity_violations(values) == 1
    report = methods.mechanism_emergence_report(steps, values, metric_name='fixture', threshold=0.5, min_consecutive=2)
    assert report.first_crossing_step == 2 and report.stable_from_step == 8
    assert report.emerged
    assert methods.random_control_report(t.tensor([0.01, 0.02]), metric_name='fixture', max_allowed_value=0.05).control_passed
