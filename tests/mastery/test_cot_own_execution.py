"""AST/JSON and synthetic callback checks only; never execute the CoT notebook."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'chapter9_alignment_interpretability/exercises/part3_cot_faithfulness/execute_learner.py'


@pytest.fixture
def entry():
    spec = importlib.util.spec_from_file_location('cot_own_execution', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {'tags': ['synthetic']}, 'source': source.splitlines(True)}
    if kind == 'code':
        result.update(outputs=[], execution_count=None)
    return result


def answer(source):
    return cell('markdown', '<details><summary>Solution</summary>\n```python\n' + source + '\n```\n</details>')


def fixture_book():
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'fixture': 'distinct synthetic, no scientific method'}, 'cells': [
        cell('markdown', 'Synthetic source fixture only.'),
        cell('code', 'import math\n'),
        answer('def core(x, scale=2):\n    return x * scale'),
        cell('code', 'def core(x, scale=2):\n    raise NotImplementedError()\n\nassert core(2) == 4\n'),
        answer('class Thing:\n    def __init__(self, x):\n        self.x = x\n    def forward(self):\n        def local(value):\n            return value + 1\n        return local(self.x)'),
        cell('code', 'class Thing:\n    def __init__(self, x):\n        raise NotImplementedError()\n    def forward(self):\n        raise NotImplementedError()\n\nassert Thing(2).forward() == 3\n'),
        answer('def core(x, scale=2):\n    return x + x'),
        cell('code', 'def core(x, scale=2):\n    raise NotImplementedError()\n\nassert core(3) == 6\n')]}


def prepared(entry):
    source = fixture_book()
    plan = entry.build_plan(source)
    completed, receipt = entry.prepare_public_reference(source, plan)
    return source, plan, completed, receipt


def test_reference_preparation_preserves_nonanswers_and_scope(entry):
    source = fixture_book()
    before = deepcopy(source)
    plan = entry.build_plan(source)
    completed, receipt = entry.prepare_public_reference(source, plan)
    assert source == before and completed['metadata'] == before['metadata']
    for index in [0, 1, 2, 4, 6]:
        assert completed['cells'][index] == before['cells'][index]
    assert 'assert core(2) == 4' in ''.join(completed['cells'][3]['source'])
    assert 'assert Thing(2).forward() == 3' in ''.join(completed['cells'][5]['source'])
    assert 'return x * scale' in ''.join(completed['cells'][3]['source'])
    assert 'return x + x' in ''.join(completed['cells'][7]['source'])
    assert len(plan['occurrences']) == 3
    assert receipt['practice_kind'] == 'exposed_public_reference_completion'
    assert receipt['executed'] is False and receipt['authorship'] == 'not_inferred'
    assert 'completed_from_own_answers' not in receipt


def test_learner_can_submit_different_math_and_reviewed_import(entry):
    source, plan, book, _ = prepared(entry)
    text = ''.join(book['cells'][3]['source']).replace('return x * scale',
        'import math\n    return sum(math.floor(x) for _ in range(scale))')
    book['cells'][3]['source'] = text.splitlines(True)
    receipt = entry.validate_completed(source, book, plan)
    assert receipt['answer_occurrences'] == 3
    assert receipt['submitted_answer_sha256'] != receipt['public_reference_answer_sha256']
    assert receipt['authorship'] == 'not_inferred'


@pytest.mark.parametrize('mutation', ['test', 'order', 'default', 'stub', 'import', 'delegate', 'global', 'class_interface'])
def test_source_binding_rejects_mutations(entry, mutation):
    source, plan, book, _ = prepared(entry)
    text = ''.join(book['cells'][3]['source'])
    if mutation == 'test':
        text = text.replace('assert core(2) == 4', 'assert True')
    elif mutation == 'order':
        book['cells'][0], book['cells'][1] = book['cells'][1], book['cells'][0]
    elif mutation == 'default':
        text = text.replace('scale=2', 'scale=3')
    elif mutation == 'stub':
        text = text.replace('return x * scale', 'raise NotImplementedError()')
    elif mutation == 'import':
        text = text.replace('return x * scale', 'from hidden_solutions import core\n    return core(x)')
    elif mutation == 'delegate':
        text = text.replace('return x * scale', 'return tests._solutions().core(x)')
    elif mutation == 'global':
        text = text.replace('return x * scale', 'global core\n    return x * scale')
    else:
        book['cells'][5]['source'] = ''.join(book['cells'][5]['source']).replace('def forward(self)', 'def other(self)').splitlines(True)
    book['cells'][3]['source'] = text.splitlines(True)
    with pytest.raises(ValueError):
        entry.validate_completed(source, book, plan)


def test_missing_or_ambiguous_visible_answer_fails_closed(entry):
    book = fixture_book()
    book['cells'][2]['source'] = ['No answer.']
    with pytest.raises(ValueError):
        entry.build_plan(book)
    book = fixture_book()
    book['cells'][2]['source'] = answer('def core(x, scale=2):\n    return x\n\ndef core(x, scale=2):\n    return x')['source']
    with pytest.raises(ValueError):
        entry.build_plan(book)


def test_plan_is_bound_to_original_whole_source(entry):
    source, plan, book, _ = prepared(entry)
    source['cells'][0]['metadata']['changed'] = True
    with pytest.raises(ValueError):
        entry.validate_completed(source, book, plan)


def test_import_and_default_check_are_inert(entry, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Default check entered runtime callback')
    monkeypatch.setattr(entry, '_run_kernel', forbidden)
    assert entry.main(['--check']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['executed'] is False and output['status'] == 'source_check_passed'
    script = """import builtins, importlib.util, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','numpy','matplotlib','nbformat','nbclient','transformers'}:
        raise AssertionError('Forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec=importlib.util.spec_from_file_location('inert_entry',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert module.main(['--check']) == 0
"""
    result = subprocess.run([sys.executable, '-B', '-c', script, str(ENTRY)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_explicit_execution_never_fills_and_preserves_failed_receipt(entry, monkeypatch, tmp_path, capsys):
    source, plan, book, _ = prepared(entry)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, {}))
    callbacks = []
    def failed(book, directory):
        callbacks.append('explicit synthetic callback only')
        raise RuntimeError('synthetic kernel failure; no scientific method')
    monkeypatch.setattr(entry, '_run_kernel', failed)
    path = tmp_path / 'learner.json'
    raw = json.dumps(book).encode()
    path.write_bytes(raw)
    out = tmp_path / 'failed-run'
    assert entry.main(['--execute', str(path), '--output-dir', str(out)]) == 1
    capsys.readouterr()
    receipt = json.loads((out / 'execution_receipt.json').read_text())
    assert callbacks == ['explicit synthetic callback only']
    assert receipt['status'] == 'FAILED' and receipt['execution_completed'] is False
    assert receipt['authorship'] == 'not_inferred'
    assert 'completed_from_own_answers' not in receipt
    assert path.read_bytes() == raw and (out / 'submitted_input.ipynb').read_bytes() == raw
    assert receipt['submitted_input_sha256'] == entry.digest_bytes(raw)
    callbacks.clear()
    path.write_text(json.dumps(source))
    assert entry.main(['--execute', str(path), '--output-dir', str(tmp_path / 'unfilled')]) == 1
    capsys.readouterr()
    assert callbacks == []


def test_actual_source_preparation_has_all_10_core_occurrences(entry):
    source, plan, pins = entry.load_frozen()
    assert len(plan['occurrences']) == 10
    completed, receipt = entry.prepare_public_reference(source, plan)
    check = entry.validate_completed(source, completed, plan)
    assert check['answer_occurrences'] == 10 and receipt['executed'] is False
    assert source['metadata'] == completed['metadata']
    assert len(source['cells']) == len(completed['cells']) == 72
    assert pins and all('sha256' in record for record in pins.values())


def test_reference_hash_is_comparable_to_actual_submitted_source(entry):
    source, plan, completed, _ = prepared(entry)
    checked = entry.validate_completed(source, completed, plan)
    assert checked['submitted_answer_sha256'] == checked['public_reference_answer_sha256']


def test_explicit_synthetic_success_accepts_kernel_runtime_metadata(entry, monkeypatch, tmp_path, capsys):
    source, plan, completed, _ = prepared(entry)
    source['metadata']['language_info'] = {'name': 'python'}
    completed['metadata']['language_info'] = {'name': 'python'}
    plan = entry.build_plan(source)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, {}))
    def synthetic_kernel(book, directory):
        fake_figure(entry, directory)
        result = deepcopy(book)
        result['metadata']['language_info'].update(version='fixture-version', file_extension='.py')
        for i, c in enumerate(result['cells']):
            c['id'] = 'fixture-runtime-' + str(i)
            if c['cell_type'] == 'code':
                c['execution_count'] = i + 1
                c['metadata']['execution'] = {'fixture': 'synthetic only'}
        return result
    monkeypatch.setattr(entry, '_run_kernel', synthetic_kernel)
    original = tmp_path / 'completed.json'
    raw = json.dumps(completed).encode()
    original.write_bytes(raw)
    out = tmp_path / 'success'
    assert entry.main(['--execute', str(original), '--output-dir', str(out)]) == 0
    capsys.readouterr()
    receipt = json.loads((out / 'execution_receipt.json').read_text())
    assert receipt['status'] == 'EXECUTED' and receipt['execution_completed'] is True
    assert receipt['completed_from_supplied_answers'] is True
    assert receipt['authorship'] == 'not_inferred' and original.read_bytes() == raw


def test_reasonable_local_reference_alias_and_nested_algorithm_are_supported(entry):
    source, plan, book, _ = prepared(entry)
    text = ''.join(book['cells'][3]['source']).replace('return x * scale',
        'import math as solutions\n    reference = 0\n    def accumulate():\n        nonlocal reference\n        reference += solutions.floor(x)\n    for _ in range(scale):\n        accumulate()\n    return reference')
    book['cells'][3]['source'] = text.splitlines(True)
    assert entry.validate_completed(source, book, plan)['answer_occurrences'] == 3


@pytest.mark.parametrize('relative_output', [False, True])
def test_explicit_runtime_adapter_uses_isolated_snapshot_workspace(entry, monkeypatch, tmp_path, capsys, relative_output):
    import types
    source, plan, completed, _ = prepared(entry)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, {}))
    directory = tmp_path / 'isolated-run'
    monkeypatch.chdir(tmp_path)
    requested_directory = Path('isolated-run') if relative_output else directory
    notebook_format = types.ModuleType('nbformat')
    notebook_format.from_dict = deepcopy
    notebook_format.writes = json.dumps
    notebook_format.write = lambda book, path: Path(path).write_text(json.dumps(book))
    client_module = types.ModuleType('nbclient')
    class SyntheticClient:
        def __init__(self, book, **kwargs):
            assert kwargs['resources']['metadata']['path'] == str(directory / 'source_snapshot')
            assert (directory / 'source_snapshot').is_dir()
            self.book = book
        def execute(self, **kwargs):
            fake_figure(entry, directory)
            assert kwargs['env']['PYTHONPATH'].split(entry.os.pathsep) == [str(directory / 'source_snapshot'), str(directory / 'source_snapshot/chapter9_alignment_interpretability/exercises')]
            for index, c in enumerate(self.book['cells']):
                if c['cell_type'] == 'code':
                    c['execution_count'] = index + 1
    client_module.NotebookClient = SyntheticClient
    monkeypatch.setitem(sys.modules, 'nbformat', notebook_format)
    monkeypatch.setitem(sys.modules, 'nbclient', client_module)
    path = tmp_path / 'submitted.json'
    path.write_text(json.dumps(completed))
    assert entry.main(['--execute', str(path), '--output-dir', str(requested_directory)]) == 0
    capsys.readouterr()
    receipt = json.loads((directory / 'execution_receipt.json').read_text())
    assert receipt['execution_workspace'] == str(directory / 'source_snapshot')
    assert receipt['execution_workspace'] != str(entry.ROOT)


def test_actual_class_submission_signature_is_recorded(entry):
    source, plan, book, _ = prepared(entry)
    book['cells'][5]['source'] = ''.join(book['cells'][5]['source']).replace(
        'def __init__(self, x):', 'def __init__(self, x: int):').splitlines(True)
    receipt = entry.validate_completed(source, book, plan)
    row = next(row for row in receipt['answers'] if row['name'] == 'Thing')
    assert "annotation=Name(id='int'" in row['actual_signature_ast']


def test_relative_reference_output_is_resolved_before_save(entry, monkeypatch, tmp_path, capsys):
    source = fixture_book()
    plan = entry.build_plan(source)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, {}))
    monkeypatch.chdir(tmp_path)
    saved = []
    save = entry._save
    def require_absolute(path, value):
        assert path.is_absolute()
        saved.append(path)
        save(path, value)
    monkeypatch.setattr(entry, '_save', require_absolute)
    assert entry.main(['--prepare-reference', '--output-dir', 'reference-practice']) == 0
    capsys.readouterr()
    directory = tmp_path / 'reference-practice'
    assert saved == [directory / 'public_reference_practice.ipynb', directory / 'preparation_receipt.json']
    receipt = json.loads(saved[1].read_text())
    assert receipt['executed'] is False and receipt['authorship'] == 'not_inferred'



def fake_figure(entry, directory):
    path = directory / 'source_snapshot' / entry.FIGURE_RELATIVE
    assert path.parent.is_dir() and not path.exists()
    path.write_bytes(b'\x89PNG\r\n\x1a\nDEV synthetic fixture only; no scientific figure')


def test_actual_protocol_mutations_fail_before_any_runtime(entry):
    source, plan, _ = entry.load_frozen()
    complete, _ = entry.prepare_public_reference(source, plan)
    mutations = [(61, 'RUN_GPU_SIGNATURE = False', 'RUN_GPU_SIGNATURE = True'),
                 (55, "'heldout_prompt_count': 40", "'heldout_prompt_count': 41"),
                 (49, 'num_pairs=12', 'num_pairs=13'),
                 (49, 'min_target_control_gap=1.0', 'min_target_control_gap=0.0'),
                 (66, 'PLAY_LAYER = 2', 'PLAY_LAYER = 1')]
    for index, old, new in mutations:
        changed = deepcopy(complete)
        text = ''.join(changed['cells'][index]['source'])
        assert old in text
        changed['cells'][index]['source'] = text.replace(old, new).splitlines(True)
        with pytest.raises(ValueError):
            entry.validate_completed(source, changed, plan)
    assert len(plan['occurrences']) == 10
    assert source['cells'][32]['outputs'] == complete['cells'][32]['outputs']
    assert source['cells'][49]['outputs'] == complete['cells'][49]['outputs']


def test_missing_fresh_figure_cannot_claim_execution_success(entry, monkeypatch, tmp_path, capsys):
    source, plan, complete, _ = prepared(entry)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, {}))
    def no_figure(book, directory):
        result = deepcopy(book)
        for index, c in enumerate(result['cells']):
            if c['cell_type'] == 'code':
                c['execution_count'] = index + 1
        return result
    monkeypatch.setattr(entry, '_run_kernel', no_figure)
    path = tmp_path / 'completed.json'
    path.write_text(json.dumps(complete))
    out = tmp_path / 'missing-figure'
    assert entry.main(['--execute', str(path), '--output-dir', str(out)]) == 1
    capsys.readouterr()
    receipt = json.loads((out / 'execution_receipt.json').read_text())
    assert receipt['status'] == 'FAILED' and receipt['execution_completed'] is False
    assert 'completed_from_supplied_answers' not in receipt
    assert receipt['fresh_pythia_measurements'] is False


@pytest.mark.parametrize('delegate', ['run_gpu_test()', 'reference_solutions.run_full_experiment()', '_load_pythia70m_on_cuda()'])
def test_explicit_native_or_historical_delegation_in_answer_is_rejected(entry, delegate):
    source, plan, complete, _ = prepared(entry)
    complete['cells'][3]['source'] = ''.join(complete['cells'][3]['source']).replace('return x * scale', 'return ' + delegate).splitlines(True)
    with pytest.raises(ValueError):
        entry.validate_completed(source, complete, plan)


def test_real_source_topology_and_custody_use_only_explicit_runtime_paths(entry, monkeypatch, tmp_path, capsys):
    """Actual dependency bytes, distinct synthetic notebook; no taught source imports."""
    import ast
    import types
    from importlib.machinery import PathFinder
    actual_source, _, pins = entry.load_frozen()
    source, plan, complete, _ = prepared(entry)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, pins))
    seen = []
    def topology_only(book, directory):
        assert book['metadata']['fixture'] == 'distinct synthetic, no scientific method'
        workspace = directory / 'source_snapshot'
        assert list((workspace / entry.FIGURE_RELATIVE).parent.iterdir()) == []
        for relative, pin in pins.items():
            custody = directory / 'input_source_custody' / relative
            assert entry.digest_bytes(custody.read_bytes()) == pin['sha256']
            snapshot = workspace / relative
            if relative in entry.RUNTIME_SOURCE_FILES:
                assert snapshot.read_bytes() == custody.read_bytes()
            else:
                assert not snapshot.exists()
        tree = ast.parse(''.join(actual_source['cells'][9]['source']))
        targets = {'chapter', 'section', 'root_dir', 'exercises_dir', 'section_dir', 'asset_dir'}
        assignments = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in targets for t in n.targets)]
        assert len(assignments) == 6
        class WorkspacePath:
            @staticmethod
            def cwd():
                return workspace
        ns = {'Path': WorkspacePath}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<path-assignments-only>', 'exec'), ns)
        assert ns['root_dir'] == workspace and ns['asset_dir'] == (workspace / entry.FIGURE_RELATIVE).parent
        historical = ast.parse((workspace / entry.HISTORICAL_SECTION / 'solutions.py').read_text())
        old_assignments = [n for n in historical.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'chapter', 'root_dir'} for t in n.targets)]
        assert len(old_assignments) == 2
        old_ns = {'Path': WorkspacePath}
        exec(compile(ast.Module(body=old_assignments, type_ignores=[]), '<historical-path-assignments-only>', 'exec'), old_ns)
        assert old_ns['root_dir'] == workspace
        for name in ['chapter9_alignment_interpretability.exercises.part2_cot_faithfulness.' + n for n in ['tests', 'utils', 'solutions']]:
            search = [str(workspace)]
            parts = name.split('.')
            for index in range(len(parts)):
                full = '.'.join(parts[:index + 1])
                spec = PathFinder.find_spec(full, search)
                assert spec is not None
                search = spec.submodule_search_locations
                if search is not None:
                    placeholder = types.ModuleType(full)
                    placeholder.__path__ = search
                    placeholder.__spec__ = spec
                    monkeypatch.setitem(sys.modules, full, placeholder)
            assert Path(spec.origin).is_relative_to(workspace)
        assert not (workspace / entry.CANONICAL_SECTION / 'media').exists()
        fake_figure(entry, directory)
        result = deepcopy(book)
        for index, c in enumerate(result['cells']):
            if c['cell_type'] == 'code':
                c['execution_count'] = index + 1
        seen.append('path-discovery/spec-only; synthetic callback')
        return result
    monkeypatch.setattr(entry, '_run_kernel', topology_only)
    path = tmp_path / 'distinct-synthetic.json'
    path.write_text(json.dumps(complete))
    out = tmp_path / 'topology'
    assert entry.main(['--execute', str(path), '--output-dir', str(out)]) == 0
    capsys.readouterr()
    assert seen == ['path-discovery/spec-only; synthetic callback']
    receipt = json.loads((out / 'execution_receipt.json').read_text())
    assert receipt['fresh_figure_generated'] is True and receipt['fresh_pythia_measurements'] is False


@pytest.mark.parametrize('failure', ['unexecuted_cell', 'error_output', 'non_png', 'symlink'])
def test_execution_completion_fails_closed_on_partial_or_invalid_fresh_output(entry, monkeypatch, tmp_path, capsys, failure):
    source, plan, complete, _ = prepared(entry)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, {}))
    def failed_fixture(book, directory):
        result = deepcopy(book)
        for index, c in enumerate(result['cells']):
            if c['cell_type'] == 'code':
                c['execution_count'] = index + 1
        fake_figure(entry, directory)
        figure = directory / 'source_snapshot' / entry.FIGURE_RELATIVE
        if failure == 'unexecuted_cell':
            result['cells'][1]['execution_count'] = None
        elif failure == 'error_output':
            result['cells'][1]['outputs'] = [{'output_type': 'error', 'ename': 'DEV', 'evalue': 'synthetic failure', 'traceback': []}]
        elif failure == 'non_png':
            figure.write_bytes(b'not a PNG; DEV fixture')
        else:
            target = tmp_path / 'outside-workspace.png'
            target.write_bytes(figure.read_bytes())
            figure.unlink()
            figure.symlink_to(target)
        return result
    monkeypatch.setattr(entry, '_run_kernel', failed_fixture)
    path = tmp_path / 'synthetic.json'
    path.write_text(json.dumps(complete))
    out = tmp_path / failure
    assert entry.main(['--execute', str(path), '--output-dir', str(out)]) == 1
    capsys.readouterr()
    receipt = json.loads((out / 'execution_receipt.json').read_text())
    assert receipt['status'] == 'FAILED' and receipt['execution_completed'] is False
    assert receipt['fresh_figure_generated'] is False and 'completed_from_supplied_answers' not in receipt


@pytest.mark.parametrize('mutation', ['not_dict', 'cells_not_list', 'duplicate_id', 'syntax', 'bad_source'])
def test_malformed_submitted_notebooks_fail_closed(entry, mutation):
    source, plan, complete, _ = prepared(entry)
    if mutation == 'not_dict':
        complete = []
    elif mutation == 'cells_not_list':
        complete['cells'] = None
    elif mutation == 'duplicate_id':
        complete['cells'][0]['id'] = complete['cells'][1]['id'] = 'duplicate'
    elif mutation == 'syntax':
        complete['cells'][3]['source'] = ['def core(:']
    else:
        complete['cells'][3]['source'] = [5]
    with pytest.raises((ValueError, TypeError, SyntaxError)):
        entry.validate_completed(source, complete, plan)


def test_contract_and_dependency_tamper_are_rejected_without_runtime(entry, monkeypatch, tmp_path):
    import shutil
    original_here = entry.HERE
    contract_dir = tmp_path / 'contract'
    contract_dir.mkdir()
    contract = contract_dir / 'own_execution_contract.json'
    contract.write_text('{}')
    monkeypatch.setattr(entry, 'HERE', contract_dir)
    with pytest.raises(ValueError, match='source contract changed'):
        entry.load_frozen()
    shutil.copy2(original_here / contract.name, contract)
    pins = json.loads(contract.read_text())['source_files']
    copied = tmp_path / 'source-copy'
    for relative in pins:
        destination = copied / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.ROOT / relative, destination)
    changed = copied / entry.HISTORICAL_SECTION / 'tests.py'
    changed.write_text(changed.read_text() + '\n# DEV changed dependency\n')
    monkeypatch.setattr(entry, 'ROOT', copied)
    with pytest.raises(ValueError, match='frozen source file changed'):
        entry.load_frozen()


def test_runtime_adapter_failure_keeps_partial_synthetic_notebook(entry, monkeypatch, tmp_path, capsys):
    import types
    source, plan, complete, _ = prepared(entry)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, {}))
    nb = types.ModuleType('nbformat')
    nb.from_dict = deepcopy
    nb.writes = json.dumps
    nb.write = lambda book, path: Path(path).write_text(json.dumps(book))
    runtime = types.ModuleType('nbclient')
    class PartialClient:
        def __init__(self, book, **kwargs):
            self.book = book
        def execute(self, **kwargs):
            self.book['cells'][1]['execution_count'] = 1
            raise RuntimeError('DEV failure; no actual kernel')
    runtime.NotebookClient = PartialClient
    monkeypatch.setitem(sys.modules, 'nbformat', nb)
    monkeypatch.setitem(sys.modules, 'nbclient', runtime)
    path = tmp_path / 'synthetic.json'
    raw = json.dumps(complete).encode()
    path.write_bytes(raw)
    out = tmp_path / 'partial'
    assert entry.main(['--execute', str(path), '--output-dir', str(out)]) == 1
    capsys.readouterr()
    saved = json.loads((out / 'executed_copy.ipynb').read_text())
    assert saved['cells'][1]['execution_count'] == 1
    receipt = json.loads((out / 'execution_receipt.json').read_text())
    assert receipt['status'] == 'FAILED' and receipt['execution_completed'] is False
    assert path.read_bytes() == raw and (out / 'submitted_input.ipynb').read_bytes() == raw

def test_cot_plumbing_allowance_is_exact_and_clean_guard_stays_narrow(tmp_path):
    """Use only source map statements and the original directory guard on DEV files."""
    import ast
    tree = ast.parse((ROOT / 'tests/test_extension_sections.py').read_text())
    names = {'ALLOWED_FILES', 'ALLOWED_DIRS', 'SECTION_PLUMBING_FILES', 'SECTION_DOCUMENTATION_FILES'}
    def root_name(node):
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(root_name(target) in names for target in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and root_name(node.value.func) in names:
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<source-allowlists-only>', 'exec'), namespace)
    assert namespace['SECTION_PLUMBING_FILES']['9.3'] == {'execute_learner.py'}
    assert namespace['SECTION_DOCUMENTATION_FILES']['9.3'] == {'lesson.md'}
    assert namespace['SECTION_PLUMBING_FILES']['5.2'] == {'execute_learner.py'}
    assert namespace['SECTION_DOCUMENTATION_FILES']['5.2'] == {'README.md', 'lesson.md'}
    assert namespace['ALLOWED_FILES'] == {'solutions.py', 'tests.py', 'utils.py'}
    assert namespace['ALLOWED_DIRS'] == {'media', 'data', '__pycache__'}
    guard = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'test_section_directory_is_clean')
    guard.decorator_list = []
    retained_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RETAINED_HISTORICAL_FILES' for t in n.targets))
    historical_names = ast.literal_eval(retained_node.value)['9.3']
    assert 'README.md' in historical_names
    namespace.update(ROOT=tmp_path, RETAINED_HISTORICAL_FILES={'9.3': historical_names}, RETAINED_HISTORICAL_DIRS={}, SECTION_DATA_DIRECTORIES={})
    directory = tmp_path / 'synthetic-section'
    directory.mkdir()
    namespace['section_paths'] = lambda *args: (directory, '9.3_Chain-of-Thought_Faithfulness')
    for name in {'solutions.py', 'tests.py', 'utils.py', 'execute_learner.py', 'lesson.md', 'own_execution_contract.json'} | historical_names:
        (directory / name).write_text('DEV file only')
    exec(compile(ast.Module(body=[guard], type_ignores=[]), '<unchanged-clean-guard-only>', 'exec'), namespace)
    namespace['test_section_directory_is_clean']('synthetic', '9.3', {})
    (directory / 'unreviewed_helper.py').write_text('DEV file only')
    with pytest.raises(AssertionError, match='unreviewed_helper.py'):
        namespace['test_section_directory_is_clean']('synthetic', '9.3', {})


@pytest.mark.parametrize('malformed_metadata', [None, []])
def test_readonly_cli_malformed_metadata_is_structured_failure_without_writes(entry, monkeypatch, tmp_path, capsys, malformed_metadata):
    source, plan, complete, _ = prepared(entry)
    monkeypatch.setattr(entry, 'load_frozen', lambda: (source, plan, {}))
    runtime_calls = []
    def forbidden_runtime(*args, **kwargs):
        runtime_calls.append('forbidden')
        raise AssertionError('read-only validation entered runtime')
    def forbidden_save(*args, **kwargs):
        raise AssertionError('read-only validation wrote output')
    monkeypatch.setattr(entry, '_run_kernel', forbidden_runtime)
    monkeypatch.setattr(entry, '_save', forbidden_save)
    complete['metadata'] = malformed_metadata
    path = tmp_path / 'malformed-completed.json'
    raw = json.dumps(complete).encode()
    path.write_bytes(raw)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert entry.main(['--validate-completed', str(path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'FAILED' and result['executed'] is False
    assert result['error'] == 'notebook metadata must be an object'
    assert runtime_calls == [] and path.read_bytes() == raw
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
