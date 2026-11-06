"""AST/JSON and synthetic callback checks only; never execute the MoE notebook."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'chapter5_modern_architectures/exercises/part2_mixture_of_experts/execute_learner.py'


@pytest.fixture
def entry():
    spec = importlib.util.spec_from_file_location('moe_own_execution', ENTRY)
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
    if name.split('.')[0] in {'torch','numpy','nbformat','nbclient','transformers'}:
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


def test_actual_source_preparation_has_all_21_core_occurrences(entry):
    source, plan, pins = entry.load_frozen()
    assert len(plan['occurrences']) == 21
    completed, receipt = entry.prepare_public_reference(source, plan)
    check = entry.validate_completed(source, completed, plan)
    assert check['answer_occurrences'] == 21 and receipt['executed'] is False
    assert source['metadata'] == completed['metadata']
    assert len(source['cells']) == len(completed['cells']) == 94
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
            assert kwargs['env']['PYTHONPATH'].split(entry.os.pathsep) == [str(directory / 'source_snapshot'), str(directory / 'source_snapshot/chapter5_modern_architectures/exercises')]
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


def test_actual_setup_path_slice_and_local_import_topology_are_isolated(entry, tmp_path):
    """Execute only frozen pathlib discovery statements, never setup imports or methods."""
    import ast
    from importlib.machinery import PathFinder
    source, _, pins = entry.load_frozen()
    workspace = tmp_path / 'source_snapshot'
    for relative, pin in pins.items():
        data = (entry.ROOT / relative).read_bytes()
        assert entry.digest_bytes(data) == pin['sha256']
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    tree = ast.parse(''.join(source['cells'][7]['source']))
    names = {'chapter', 'section', 'root_dir', 'exercises_dir'}
    statements = [node for node in tree.body if isinstance(node, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    assert len(statements) == 4
    class WorkspacePath:
        @staticmethod
        def cwd():
            return workspace
    namespace = {'Path': WorkspacePath}
    exec(compile(ast.Module(body=statements, type_ignores=[]), '<path-discovery-only>', 'exec'), namespace)
    assert namespace['root_dir'] == workspace
    exercise_root = namespace['exercises_dir']
    assert exercise_root == workspace / 'chapter5_modern_architectures/exercises'
    for package, module in [('part2_mixture_of_experts', 'tests'),
                            ('part2_mixture_of_experts', 'solutions'),
                            ('part8_mixture_of_experts', 'solutions')]:
        package_spec = PathFinder.find_spec(package, [str(exercise_root)])
        spec = PathFinder.find_spec(package + '.' + module, package_spec.submodule_search_locations)
        assert spec.origin == str(exercise_root / package / (module + '.py'))
    assert not (exercise_root / 'part2_mixture_of_experts/media/mixture_of_experts_signature.png').exists()


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


def test_moe_plumbing_allowance_is_exact_and_clean_guard_stays_narrow(tmp_path):
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
    assert namespace['SECTION_PLUMBING_FILES']['5.2'] == {'execute_learner.py'}
    assert namespace['SECTION_DOCUMENTATION_FILES']['5.2'] == {'README.md', 'lesson.md'}
    assert namespace['ALLOWED_FILES'] == {'solutions.py', 'tests.py', 'utils.py'}
    assert namespace['ALLOWED_DIRS'] == {'media', 'data', '__pycache__'}
    guard = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'test_section_directory_is_clean')
    guard.decorator_list = []
    namespace.update(ROOT=tmp_path, RETAINED_HISTORICAL_FILES={}, RETAINED_HISTORICAL_DIRS={}, SECTION_DATA_DIRECTORIES={})
    directory = tmp_path / 'synthetic-section'
    directory.mkdir()
    namespace['section_paths'] = lambda *args: (directory, '5.2_Mixture_of_Experts')
    for name in ['solutions.py', 'tests.py', 'utils.py', 'README.md', 'execute_learner.py', 'lesson.md', 'own_execution_contract.json']:
        (directory / name).write_text('DEV file only')
    exec(compile(ast.Module(body=[guard], type_ignores=[]), '<unchanged-clean-guard-only>', 'exec'), namespace)
    namespace['test_section_directory_is_clean']('synthetic', '5.2', {})
    (directory / 'unreviewed_helper.py').write_text('DEV file only')
    with pytest.raises(AssertionError, match='unreviewed_helper.py'):
        namespace['test_section_directory_is_clean']('synthetic', '5.2', {})
