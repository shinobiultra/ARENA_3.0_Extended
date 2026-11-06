"""Independent learner-surface, generation, and CPU execution checks."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import nbformat
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
LESSON = ROOT / 'mastery_hub/units/u02_optimization/lesson.py'
GENERATOR = ROOT / 'scripts/build_mastery_notebooks.py'


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_contract():
    lesson = load(LESSON)
    for kind in ('exercises', 'solutions'):
        cells = lesson.build_cells(kind)
        assert cells == lesson.build_cells(kind)
        notebook = nbformat.v4.new_notebook(cells=[nbformat.from_dict(c) for c in cells])
        nbformat.validate(notebook)
        ids = [c['id'] for c in cells]
        assert len(ids) == len(set(ids))
        text = '\n'.join(c['source'] for c in cells)
        for phrase in ('0.405', 'analytic_toy', 'Try It Yourself', 'Falsify', 'Reflection',
                       'zero learning rate', 'fixed optimum', 'rotation', 'noise', 'Held-out'):
            assert phrase in text
        definitions = [c for c in cells if 'exercise-definition' in c['metadata'].get('tags', [])]
        assert len(definitions) >= 6
        for definition in definitions:
            index = cells.index(definition)
            assert 'exercise-test' in cells[index + 1]['metadata']['tags']
        assert text.count('<summary>Help') >= 6
        assert text.count('<summary>Solution') >= 6
        assert text.count('Expected output') >= 6
        if kind == 'exercises':
            assert sum('raise NotImplementedError' in c['source'] for c in definitions) >= 6
        else:
            assert all('raise NotImplementedError' not in c['source'] for c in definitions)
    with pytest.raises(ValueError):
        lesson.build_cells('other')


def execute_cells(cells, fill_solutions=False):
    namespace = {'__name__': '__main__', '__executed_sources__': []}
    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if fill_solutions and 'exercise-definition' in cell['metadata'].get('tags', []):
                source = re.search(r'<summary>Solution</summary>.*?```python\n(.*?)\n```',
                                   cells[index - 1]['source'], flags=re.S).group(1)
                assert source == cell['metadata']['reference_answer']
            if fill_solutions and 'prediction-commit' in cell['metadata'].get('tags', []):
                source = cell['metadata']['reference_answer']
            namespace['__executed_sources__'].append(source)
            exec(compile(source, '<notebook-cell>', 'exec'), namespace)
    return namespace


def test_solution_and_completed_learner_semantics():
    lesson = load(LESSON)
    for kind in ('solutions', 'exercises'):
        ns = execute_cells(lesson.build_cells(kind), fill_solutions=(kind == 'exercises'))
        assert len(ns['signature_figure'].axes) == 3
        assert ns['controls']['zero learning rate']
        assert ns['controls']['fixed optimum']
        assert ns['controls']['rotation']
        assert ns['controls']['noise reproducibility']
        assert ns['controls']['unstable step']
        # Exact independent scalar recurrence over a non-axis-aligned eigensystem.
        H = ns['quadratic_hessian'](0.8, 4.0, 23.0)
        result = ns['simulate_student'](H, [0.3, -0.7], 0.07, 0.4, 13)
        x, v = np.array([0.3, -0.7]), np.zeros(2)
        for k in range(13):
            v = 0.4 * v + H @ x
            x = x - 0.07 * v
            np.testing.assert_allclose(result['states'][k+1], x, atol=1e-14)
        assert ns['heldout_attempt']['status'] == 'revealed'


def test_generator_read_only_check_and_output_preservation(tmp_path):
    generator = load(GENERATOR)
    assert set(generator.OUTPUTS) == {'exercises', 'solutions'}
    assert all(path.parent == ROOT / 'mastery_hub/units/u02_optimization' for path in generator.OUTPUTS.values())
    canonical = generator.make_notebook('solutions')
    executed = copy.deepcopy(canonical)
    code = next(c for c in executed.cells if c.cell_type == 'code')
    code.execution_count = 1
    code.outputs = [nbformat.v4.new_output('stream', name='stdout', text='preserve execution\n')]
    target = tmp_path / 'solutions.ipynb'
    nbformat.write(executed, target)
    before = target.read_bytes()
    assert generator.synchronize(target, canonical, check=True)
    assert target.read_bytes() == before
    assert generator.synchronize(target, canonical, check=False)
    assert target.read_bytes() == before
    drifted = copy.deepcopy(canonical)
    drifted.cells[0].source += '\nDRIFT'
    nbformat.write(drifted, target)
    before = target.read_bytes()
    assert not generator.synchronize(target, canonical, check=True)
    assert target.read_bytes() == before


def test_committed_notebooks_match_and_check_is_read_only():
    before = {p: p.read_bytes() for p in (ROOT / 'mastery_hub/units/u02_optimization').glob('*.ipynb')}
    result = subprocess.run([sys.executable, str(GENERATOR), '--check'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(before) == 2
    assert all(p.read_bytes() == content for p, content in before.items())


def test_completed_learner_fresh_cpu_kernel(tmp_path):
    """Actually execute completed student cells from dropdowns in a fresh kernel."""
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    lesson = load(LESSON)
    cells = lesson.build_cells('exercises')
    for index, cell in enumerate(cells):
        tags = cell['metadata'].get('tags', [])
        if 'exercise-definition' in tags:
            cell['source'] = re.search(
                r'<summary>Solution</summary>.*?```python\n(.*?)\n```',
                cells[index - 1]['source'], flags=re.S).group(1)
        if 'prediction-commit' in tags:
            cell['source'] = cell['metadata']['reference_answer']
    notebook = nbformat.v4.new_notebook(cells=[nbformat.from_dict(c) for c in cells])
    kernel = tmp_path / 'mastery-cpu'
    kernel.mkdir()
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Mastery CPU', 'language': 'python'}))
    manager = KernelSpecManager(kernel_dirs=[str(tmp_path)])
    client = NotebookClient(notebook, timeout=120, startup_timeout=20, kernel_name='mastery-cpu',
                            resources={'metadata': {'path': str(ROOT)}})
    client.km = AsyncKernelManager(kernel_name='mastery-cpu', kernel_spec_manager=manager)
    executed = client.execute()
    nbformat.validate(executed)
    code_cells = [c for c in executed.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
    assert any('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs)
    from arena_ext.mastery.records import loads_attempt, export_attempt, load_attempt, replay
    outputs = ''.join(o.get('text', '') for c in code_cells for o in c.outputs)
    payload = outputs.split('PORTABLE_RECORD_JSON_BEGIN\n')[1].split('\nPORTABLE_RECORD_JSON_END')[0]
    record = loads_attempt(payload)
    assert record['attempt']['reasoning']
    assert record['attempt']['status'] == 'revealed'
    assert record['config']['noise_std'] == 0. and record['config']['seed'] == 17
    assert record['control_outcomes']['rotation']['passed'] is True
    executed_sources = ['']
    for cell in code_cells:
        executed_sources.append(cell.source)
        if 'prediction-reveal' in cell.metadata.get('tags', []):
            break
    import hashlib
    expected_hash = hashlib.sha256(json.dumps(executed_sources, ensure_ascii=False).encode()).hexdigest()
    assert record['code_revision'] == 'executed-notebook-inputs-sha256:' + expected_hash
    exported = export_attempt(record, tmp_path / 'explicit-export')
    assert load_attempt(exported) == record
    reference_trace = replay(record)
    # Independent semantic replay directly from the exact serialized config.
    q = record['config']
    ns = execute_cells(load(LESSON).build_cells('exercises'), fill_solutions=True)
    own = ns['simulate_student'](ns['quadratic_hessian'](q['lambda_min'], q['lambda_max'], q['angle_deg']),
                                q['x0'], q['eta'], q['beta'], q['steps'], q['noise_std'], q['seed'])
    np.testing.assert_allclose(reference_trace.states, own['states'], atol=1e-13)
    np.testing.assert_allclose(reference_trace.losses, own['losses'], atol=1e-13)
    np.testing.assert_allclose(reference_trace.update_norms, own['update_norms'], atol=1e-13)
    (tmp_path / 'completed_student.ipynb').write_text(nbformat.writes(executed))


def test_notebook_portable_record_round_trip(tmp_path):
    from arena_ext.mastery.records import loads_attempt, replay, load_attempt
    from arena_ext.mastery.quadratic import SIMULATOR_VERSION
    ns = execute_cells(load(LESSON).build_cells('exercises'), fill_solutions=True)
    record = loads_attempt(json.dumps(ns['portable_record'], allow_nan=False))
    assert record['config']['simulator_version'] == SIMULATOR_VERSION
    assert record['config']['angle_deg'] == ns['heldout_config']['rotation_deg']
    assert record['config']['noise_std'] == ns['heldout_config']['noise_std']
    assert record['config']['seed'] == ns['heldout_config']['seed']
    assert record['attempt']['reasoning'] == ns['heldout_attempt']['reasoning']
    assert record['attempt']['status'] == 'revealed'
    assert record['control_outcomes']['rotation']['passed'] is True
    assert 'sha256:' in record['source_revision'] and 'sha256:' in record['code_revision']
    trace = replay(record)
    np.testing.assert_allclose(trace.states, ns['heldout_run']['states'], atol=1e-13)
    np.testing.assert_allclose(trace.losses, ns['heldout_run']['losses'], atol=1e-13)
    np.testing.assert_allclose(trace.update_norms, ns['heldout_run']['update_norms'], atol=1e-13)
    exported = ns['export_notebook_record'](tmp_path)
    assert load_attempt(exported) == record
