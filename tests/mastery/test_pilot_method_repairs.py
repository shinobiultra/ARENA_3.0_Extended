"""CPU-only method regression checks: extract definitions, never load/download a model."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch as t

from scripts.verified_publication_pages import historical_bytes

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = {
    '7.1': ('chapter7_activation_to_language', 'part1_lenses_patchscopes', 'master_7_1.py'),
    '8.4': ('chapter8_automated_circuits', 'part4_acdc_circuit_metrics', 'master_8_4.py'),
}
METHODS = {
    '7.1': ('train_tuned_lens', 'evaluate_lenses', 'shifted_translator_agreement', 'patchscope_controls'),
    '8.4': ('single_edge_effects', 'attention_by_offset', 'top_k_circuit', 'completeness_scores', 'edge_drops'),
}


def section_tests(section):
    chapter, part, _ = SECTIONS[section]
    path = ROOT / chapter / 'exercises' / part / 'tests.py'
    spec = importlib.util.spec_from_file_location('pilot_tests_' + section.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method(section, name, generated=False):
    chapter, part, filename = SECTIONS[section]
    path = ROOT / chapter / 'exercises' / part / 'solutions.py' if generated else ROOT / 'infrastructure/chapters' / chapter / filename
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    # Future annotations avoid needing model classes. Decorators still execute normally.
    namespace = {'t': t, 'tqdm': lambda iterable, **kwargs: iterable}
    source = ast.unparse(ast.Module(body=[node], type_ignores=[]))
    exec(compile('from __future__ import annotations\n' + source, str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('section,name', [(section, name) for section, names in METHODS.items() for name in names])
def test_independent_method_fixture(section, name):
    getattr(section_tests(section), 'test_' + name)(method(section, name))


@pytest.mark.parametrize('section,name', [(section, name) for section, names in METHODS.items() for name in names])
def test_generated_definition_matches_master(section, name):
    assert ast.dump(ast.parse(ast.unparse(method_ast(section, name, False)))) == ast.dump(ast.parse(ast.unparse(method_ast(section, name, True))))


def method_ast(section, name, generated):
    chapter, part, filename = SECTIONS[section]
    path = ROOT / chapter / 'exercises' / part / 'solutions.py' if generated else ROOT / 'infrastructure/chapters' / chapter / filename
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_invalid_lens_denominators_are_rejected():
    from inspect import unwrap
    from unittest.mock import patch

    fixture = section_tests('7.1')._metric_fixture()
    model, lens, tokens, _, _, get_resid = fixture
    for name in ('evaluate_lenses', 'shifted_translator_agreement'):
        fn = method('7.1', name)
        with patch.dict(unwrap(fn).__globals__, get_resid_and_logprobs=get_resid):
            for bad_tokens, batch_size in ((tokens[:0], 2), (tokens[:, :1], 2), (tokens, 0), (tokens, -1)):
                with pytest.raises(ValueError, match='positive batch size'):
                    fn(model, lens, bad_tokens, batch_size=batch_size, **({'shift': 0} if 'shifted' in name else {}))


def test_archived_and_repaired_equal_batch_metrics_agree():
    """The published 200x64/batch20 config has equal token counts, so weights cancel."""
    from inspect import unwrap
    import tarfile
    from types import SimpleNamespace
    from unittest.mock import patch

    model, lens, _, raw, final, _ = section_tests('7.1')._metric_fixture()
    tokens = t.arange(200).view(-1, 1).expand(-1, 64)

    def get_resid(model, batch):
        ids = batch[:, 0] % 3
        # Repeat one evaluated distribution over positions, retaining a distinct BOS.
        residual = raw[:, ids][:, :, [0] + [1] * 63].log()
        target = final[ids][:, [0] + [1] * 63].log()
        return residual, target

    archive = ROOT / 'docs/mastery/evidence/pilot_method_repairs_20261016/before.tar.gz'
    with tarfile.open(archive) as tar:
        source = tar.extractfile('infrastructure/chapters/chapter7_activation_to_language/master_7_1.py').read().decode()
    old_nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    for name in ('evaluate_lenses', 'shifted_translator_agreement'):
        namespace = {'t': t}
        old_source = ast.unparse(ast.Module(body=[old_nodes[name]], type_ignores=[]))
        exec('from __future__ import annotations\n' + old_source, namespace)
        old, new = namespace[name], method('7.1', name)
        with patch.dict(unwrap(old).__globals__, get_resid_and_logprobs=get_resid), patch.dict(unwrap(new).__globals__, get_resid_and_logprobs=get_resid):
            for extra in ([{}] if name == 'evaluate_lenses' else [{'shift': -1}, {'shift': 0}, {'shift': 1}]):
                a, b = old(model, lens, tokens, batch_size=20, **extra), new(model, lens, tokens, batch_size=20, **extra)
                assert a.keys() == b.keys()
                for key in a:
                    t.testing.assert_close(t.tensor(a[key]), t.tensor(b[key]), atol=2e-7, rtol=2e-7)


def verified_fresh_solutions(notebook, path):
    """Accept fresh state only through actual candidate/result/output/source identity."""
    import hashlib
    import json
    import nbformat

    metadata = notebook.metadata.get('fresh_execution')
    if not metadata:
        return False
    result_path = ROOT / metadata['result']
    result = json.loads(result_path.read_text())
    candidate_path = result_path.parent / 'executed.ipynb'
    assert result['status'] == 'success' and result['kernel_cleanup_complete']
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == result['executed_notebook_sha256']
    candidate = nbformat.read(candidate_path, as_version=4)
    actual = [cell for cell in notebook.cells if cell.cell_type == 'code']
    executed = [cell for cell in candidate.cells if cell.cell_type == 'code']
    assert len(actual) == len(executed) == result['expected_code_cells'] == result['completed_code_cells']
    assert [hashlib.sha256(cell.source.encode()).hexdigest() for cell in actual] == result['original_code_sha256s']
    for cell, reference in zip(actual, executed):
        assert cell.source == reference.source and cell.outputs == reference.outputs
        assert cell.execution_count == reference.execution_count and cell.execution_count is not None
        assert not any(output.output_type == 'error' for output in cell.outputs)
    assert notebook.metadata.get('widgets') == candidate.metadata.get('widgets')
    runtime = json.loads((result_path.parent / 'kernel_runtime.json').read_text())
    assert '5090' in runtime['gpu'] and runtime['allocation_fraction'] == .9
    assert runtime['cuda_visible_devices'] == '0' and runtime['default_dtype'] == 'torch.float32'
    assert result['peak_whole_gpu_mib'] < 23040
    publication = json.loads((ROOT / metadata['publication_receipt']).read_text())
    assert publication['candidate_sha256'] == result['executed_notebook_sha256']
    for entry in publication['files']:
        assert hashlib.sha256(historical_bytes(ROOT / entry['path'], entry['sha256'])).hexdigest() == entry['sha256']
    return True


def verified_fresh_reference(notebook, path):
    import hashlib
    import json
    import nbformat

    metadata = notebook.metadata.get('fresh_reference')
    if not metadata:
        return False
    publication = json.loads((ROOT / metadata['publication_receipt']).read_text())
    entry = next(entry for entry in publication['files'] if entry['path'] == str(path.relative_to(ROOT)))
    assert hashlib.sha256(historical_bytes(path, entry['sha256'])).hexdigest() == entry['sha256']
    result = json.loads((ROOT / metadata['executed_solution_result']).read_text())
    solution = ROOT / result['canonical']
    assert verified_fresh_solutions(nbformat.read(solution, as_version=4), solution)
    return True


@pytest.mark.parametrize('section', SECTIONS)
def test_added_checks_are_immediate_and_have_honest_execution_state(section):
    import json
    import nbformat

    chapter, part, _ = SECTIONS[section]
    directory = ROOT / chapter / 'exercises' / part
    for path in directory.glob('*.ipynb'):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        fresh = verified_fresh_solutions(nb, path)
        reference = verified_fresh_reference(nb, path)
        for name in METHODS[section]:
            definition = next(i for i, cell in enumerate(nb.cells) if cell.cell_type == 'code' and 'def ' + name + '(' in cell.source)
            following = nb.cells[definition + 1]
            assert following.cell_type == 'code' and f'tests.test_{name}({name})' in following.source
            if fresh:
                assert following.execution_count is not None
                assert any(f'All tests in `test_{name}` passed!' in output.get('text', '') for output in following.outputs)
            else:
                assert not following.get('outputs') and following.get('execution_count') is None, 'Do not invent notebook test execution evidence'


def test_historical_output_and_media_receipts():
    import hashlib
    import json
    import tarfile
    import nbformat

    evidence = ROOT / 'docs/mastery/evidence/pilot_method_repairs_20261016'
    before = json.loads((evidence / 'before_manifest.json').read_text())
    receipt = json.loads((evidence / 'reference_preservation.json').read_text())
    assert hashlib.sha256((evidence / 'before.tar.gz').read_bytes()).hexdigest() == before['archive_sha256'] == receipt['archive_sha256']
    with tarfile.open(evidence / 'before.tar.gz') as archive:
        for entry in before['files']:
            saved = archive.extractfile(entry['path']).read()
            assert hashlib.sha256(saved).hexdigest() == entry['sha256']
            if '/media/' in entry['path']:
                section_directory = ROOT / entry['path'].split('/media/')[0]
                solution = next(section_directory.glob('*_solutions.ipynb'))
                current = nbformat.read(solution, as_version=4)
                if verified_fresh_solutions(current, solution):
                    continue
                assert (ROOT / entry['path']).read_bytes() == saved, 'Do not change historical figures during CPU repairs'
        for notebook in receipt['notebooks']:
            path = ROOT / notebook['path']
            fresh = nbformat.read(path, as_version=4)
            if verified_fresh_solutions(fresh, path) or verified_fresh_reference(fresh, path):
                continue
            assert hashlib.sha256(path.read_bytes()).hexdigest() == notebook['current_notebook_sha256']
            old = nbformat.reads(archive.extractfile(notebook['path']).read().decode(), as_version=4)
            for pair in notebook['restored_output_cells']:
                a, b = old.cells[pair['old_index']], fresh.cells[pair['new_index']]
                assert a.source == b.source and a.outputs == b.outputs
                assert b.metadata['historical_reference']['fresh_execution'] is False
            for index in notebook['new_test_cells_unexecuted']:
                assert not fresh.cells[index].outputs and fresh.cells[index].execution_count is None


def test_published_configuration_evidence():
    import json
    import tarfile

    evidence = ROOT / 'docs/mastery/evidence/pilot_method_repairs_20261016'
    with tarfile.open(evidence / 'before.tar.gz') as archive:
        for section, (chapter, part, filename) in SECTIONS.items():
            old_source = archive.extractfile(f'infrastructure/chapters/{chapter}/{filename}').read().decode()
            new_source = (ROOT / 'infrastructure/chapters' / chapter / filename).read_text()
            old_funcs = {node.name: ast.dump(node) for node in ast.parse(old_source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            new_funcs = {node.name: ast.dump(node) for node in ast.parse(new_source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            changed = {name for name in old_funcs if old_funcs[name] != new_funcs[name]}
            assert changed == ({'evaluate_lenses', 'shifted_translator_agreement'} if section == '7.1' else {'attention_by_offset'})
            if section == '7.1':
                stem = '7.1_Logit_Lens_Tuned_Lens_and_Patchscopes'
                old_nb = json.loads(archive.extractfile(f'{chapter}/exercises/{part}/{stem}_solutions.ipynb').read())
                data = next(c for c in old_nb['cells'] if 'all_tokens = utils.load_tinystories_tokens' in ''.join(c['source']))
                assert 'torch.Size([200, 64])' in str(data['outputs'])
                assert 'n_sequences=1800, seq_len=64' in ''.join(data['source'])
                assert 'all_tokens[:1600], all_tokens[1600:]' in ''.join(data['source'])
                for source in (old_source, new_source):
                    assert 'results = evaluate_lenses(model, tuned_lens, test_tokens)' in source
                    assert 'shifted_translator_agreement(model, tuned_lens, test_tokens, shift)' in source
                    assert 'shift: int, batch_size: int = 20' in source and 'tokens: Int[Tensor, "n_seq seq"], batch_size: int = 20' in source
            else:
                for source in (old_source, new_source):
                    assert 'by_offset = attention_by_offset(model, clean, layer=0, max_offset=6)' in source
                    assert 'max_offset: int, start_pos: int = 8' in source


def test_reconstructed_historical_pages_match_pre_lane_hashes():
    import hashlib
    import json
    import tarfile

    evidence = ROOT / 'docs/mastery/evidence/pilot_method_repairs_20261016'
    receipt = json.loads((evidence / 'before_pages_reconstruction.json').read_text())
    assert receipt['kind'] == 'reconstructed and verified against pre-lane hash, not original byte capture'
    assert receipt['original_archive_unchanged'] is True
    archive = evidence / 'before_pages_reconstructed_verified.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['reconstructed_verified_archive_sha256']
    assert len(receipt['pages']) == 4
    with tarfile.open(archive) as saved:
        for page in receipt['pages']:
            data = saved.extractfile(page['path']).read()
            assert page['byte_identical_to_pre_lane_hash'] is True
            assert hashlib.sha256(data).hexdigest() == page['expected_pre_lane_sha256'] == page['reconstructed_sha256']


def test_training_feedback_rejects_accumulated_gradient_mutation():
    """The learner check must fail if zero_grad is accidentally omitted."""
    import copy

    node = copy.deepcopy(method_ast('7.1', 'train_tuned_lens', False))

    class OmitZeroGrad(ast.NodeTransformer):
        removed = 0

        def visit_Expr(self, expression):
            if (isinstance(expression.value, ast.Call)
                    and isinstance(expression.value.func, ast.Attribute)
                    and expression.value.func.attr == 'zero_grad'):
                self.removed += 1
                return None
            return self.generic_visit(expression)

    mutation = OmitZeroGrad()
    node = ast.fix_missing_locations(mutation.visit(node))
    assert mutation.removed == 1
    namespace = {'t': t, 'tqdm': lambda iterable, **kwargs: iterable}
    source = ast.unparse(ast.Module(body=[node], type_ignores=[]))
    exec('from __future__ import annotations\n' + source, namespace)
    with pytest.raises(AssertionError, match='Clear gradients'):
        section_tests('7.1').test_train_tuned_lens(namespace['train_tuned_lens'])
