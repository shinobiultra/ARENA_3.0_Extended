"""DEV source-conversion checks only; never run a taught attribution method."""
from __future__ import annotations

import ast
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import nbformat
import pytest

C = Path(__file__).resolve().parents[1]
R = C
W = C.parent
CH = 'chapter16_shapley_attribution_baselines'
P = C / CH / 'exercises/part8_shap_vs_patching'
E = C / 'docs/mastery/evidence/shap_patching_16_8_migration_20261021/v2'
STEM = '16.8_SHAP_vs_Activation_Patching'


def file_bytes(path):
    if path.exists():
        return path.read_bytes()
    if path.is_relative_to(W / 'preimages'):
        member = str(path.relative_to(W / 'preimages'))
        with tarfile.open(E / 'before.tar.gz') as handle:
            return handle.extractfile(member).read()
    if path.is_relative_to(W / 'raw_generated'):
        return (E / 'raw_generated' / path.name).read_bytes()
    raise FileNotFoundError(path)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f'duplicate JSON key: {key}'
            result[key] = value
        return result
    return json.loads(file_bytes(path), object_pairs_hook=unique,
                      parse_constant=lambda value: pytest.fail(f'non-finite JSON {value}'))


def source(cell):
    return ''.join(cell['source'])


def test_strict_saved_books_and_master():
    paths = [P / f'{STEM}_{kind}.ipynb' for kind in ['exercises', 'solutions']]
    paths.append(C / 'infrastructure/chapters' / CH / 'master_16_8.ipynb')
    for path in paths:
        book = read(path)
        assert (book['nbformat'], book['nbformat_minor']) == (4, 5)
        nbformat.validate(book)
        ids = [c['id'] for c in book['cells']]
        assert len(ids) == len(set(ids))
        for cell in book['cells']:
            if cell['cell_type'] == 'markdown':
                assert 'outputs' not in cell and 'execution_count' not in cell


def test_all_original_cell_objects_have_exact_inverses():
    mapping = read(E / 'cell_inverse_map.json')
    metadata = read(E / 'legacy_book_metadata.json')
    for kind in ['exercises', 'solutions']:
        actual = read(P / f'{STEM}_{kind}.ipynb')
        assert len(mapping[kind]) == 61
        recovered = {}
        for row in mapping[kind]:
            cell = actual['cells'][row['candidate_index']]
            assert cell == row['transformed_cell']
            old = row['original_cell']
            assert cell['metadata'] == old['metadata']
            if old['cell_type'] == 'code':
                assert cell['outputs'] == old['outputs']
                assert cell['execution_count'] == old['execution_count']
            recovered.setdefault(row['source_path'], []).append((row['original_index'], old))
        for path, pairs in recovered.items():
            original = read(W / 'preimages' / path)
            assert [c for _, c in sorted(pairs)] == original['cells']
            alias = 'relu' if '/part6_' in path else 'agreement'
            rebuilt = dict(metadata[alias + '_' + kind], cells=[c for _, c in sorted(pairs)])
            assert rebuilt == original


def test_student_solution_differences_and_historical_outputs_survive():
    mapping = read(E / 'cell_inverse_map.json')
    actual = {kind: read(P / f'{STEM}_{kind}.ipynb') for kind in ['exercises', 'solutions']}
    for alias, expected in [('relu', 8), ('agreement', 10)]:
        # A includes its original different title; seven executable exercise groups.
        originals = {kind: [r for r in mapping[kind] if r['legacy'] == alias]
                     for kind in ['exercises', 'solutions']}
        differences = 0
        for a, b in zip(originals['exercises'], originals['solutions'], strict=True):
            if a['original_cell']['source'] != b['original_cell']['source']:
                differences += 1
                assert actual['exercises']['cells'][a['candidate_index']]['source'] != actual['solutions']['cells'][b['candidate_index']]['source']
        assert differences == expected
    solutions = actual['solutions']
    assert sum(bool(c.get('outputs')) for c in solutions['cells'] if c['cell_type'] == 'code') == 21
    assert sum('image/png' in o.get('data', {}) for c in solutions['cells'] for o in c.get('outputs', [])) == 3


def test_normal_converter_preserves_all_24_code_asts():
    for kind in ['exercises', 'solutions']:
        final = read(P / f'{STEM}_{kind}.ipynb')
        raw = read(W / 'raw_generated' / f'{STEM}_{kind}.ipynb')
        old_code = [ast.dump(ast.parse(source(c))) for c in final['cells'] if c['cell_type'] == 'code']
        new_code = [ast.dump(ast.parse(source(c))) for c in raw['cells'] if c['cell_type'] == 'code']
        assert len(old_code) == len(new_code) == 24
        assert old_code == new_code


def test_default_callback_scopes_and_scientific_asts_are_exact():
    module = ast.parse((P / 'solutions.py').read_text())
    initializer = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize_agreement_defaults')
    names = {'finite_circuit_value', 'finite_circuit_table', 'mechanistic_endpoint_scores'}
    nested = {n.name: n for n in initializer.body if isinstance(n, ast.FunctionDef)}
    assert set(nested) == names
    original = read(W / 'preimages' / CH / 'exercises/part8_shapley_mechinterp_agreement/16.8_Do_SHAPley_and_Mechanistic_Interpretability_Agree_solutions.ipynb')
    old_nodes = {n.name: n for n in ast.parse(source(original['cells'][9])).body if isinstance(n, ast.FunctionDef)}
    for name in names:
        assert ast.dump(nested[name]) == ast.dump(old_nodes[name])
    calls = [n for n in initializer.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)]
    assert len(calls) == 1
    assert ast.unparse(calls[0]) == 'TOY_LINEAR_WEIGHTS = t.tensor([1.2, -0.7, 1.6, 0.9], dtype=t.float64)'
    globals_ = {name for n in initializer.body if isinstance(n, ast.Global) for name in n.names}
    assert names | {'TOY_LINEAR_WEIGHTS', '_agreement_defaults_initialized'} == globals_


def test_scoped_alpha_map_keeps_local_reference_and_attributes():
    module = ast.parse((P / 'solutions.py').read_text())
    definitions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    assert 'relu_run_gpu_test' in definitions and 'agreement_run_gpu_test' in definitions
    assert 'run_gpu_test' not in definitions and 'run_full_experiment' not in definitions
    a = definitions['attribution_comparison']
    assert any(isinstance(n, ast.Name) and n.id == 'reference' and isinstance(n.ctx, ast.Store) for n in ast.walk(a))
    for name in ['relu_run_gpu_test', 'agreement_run_gpu_test']:
        attrs = [n.attr for n in ast.walk(definitions[name]) if isinstance(n, ast.Attribute)]
        assert 'run_gpu_test' in attrs
        assert 'relu_run_gpu_test' not in attrs and 'agreement_run_gpu_test' not in attrs


def test_exact_legacy_test_modules_and_explicit_fallback_provenance():
    for alias, part in [('relu', 'part6_shap_vs_activation_patching'), ('agreement', 'part8_shapley_mechinterp_agreement')]:
        original = W / 'preimages' / CH / 'exercises' / part / 'tests.py'
        assert (P / f'{alias}_tests.py').read_bytes() == file_bytes(original)
    readme = (P / 'README.md').read_text()
    assert 'fallback' in readme and 'does not establish isolation' in readme


def test_plain_import_is_inert_and_explicit_init_uses_only_recording_stub():
    # Prewarm third-party imports outside the observed lesson-import boundary.
    import torch
    import matplotlib
    import matplotlib.pyplot
    import IPython.display
    baseline_rng = torch.random.get_rng_state().clone()
    baseline_dtype = torch.get_default_dtype()
    baseline_path = list(sys.path)
    sys.path[:0] = [str(C), str(R)]
    spec = importlib.util.spec_from_file_location('shap_patching_dev_import', P / 'solutions.py')
    mod = importlib.util.module_from_spec(spec)
    def forbidden(*args, **kwargs):
        raise AssertionError('lesson import attempted runtime work')
    try:
        with patch.object(torch, 'tensor', forbidden), patch.object(torch, 'zeros', forbidden), patch.object(torch, 'ones', forbidden), patch.object(torch, 'set_default_dtype', forbidden), patch.object(matplotlib, 'use', forbidden), patch.object(Path, 'cwd', forbidden):
            spec.loader.exec_module(mod)
        assert torch.equal(torch.random.get_rng_state(), baseline_rng)
        assert torch.get_default_dtype() == baseline_dtype
        assert sys.path == [str(C), str(R)] + baseline_path
        assert callable(mod.exact_shapley_values) and callable(mod.enumerate_coalitions)
        assert not hasattr(mod, 'finite_circuit_value')
        assert not hasattr(mod, 'TOY_LINEAR_WEIGHTS')
        assert not hasattr(mod, 'root_dir')
        plumbing_events = []
        class DEVPath(PurePosixPath):
            def exists(self):
                plumbing_events.append(('exists', str(self)))
                return str(self) == '/DEV_ROOT/' + CH
        def cwd_stub():
            plumbing_events.append(('cwd',))
            return DEVPath('/DEV_ROOT')
        mod.Path = SimpleNamespace(cwd=cwd_stub)
        with patch.object(torch, 'tensor', forbidden), patch.object(torch, 'zeros', forbidden):
            mod.initialize_lesson_plumbing()
        assert plumbing_events == [('cwd',), ('cwd',), ('exists', '/DEV_ROOT/' + CH)]
        assert str(mod.root_dir) == '/DEV_ROOT'
        expected = '/DEV_ROOT/' + CH + '/exercises/'
        assert str(mod.relu_legacy_section_dir) == expected + 'part6_shap_vs_activation_patching'
        assert str(mod.agreement_legacy_section_dir) == expected + 'part8_shapley_mechinterp_agreement'
        assert str(mod.asset_path) == expected + 'part8_shap_vs_patching/media/shap_vs_activation_patching_exact_signature.png'
        assert str(mod.artifact_dir) == expected + 'part8_shap_vs_patching/artifacts'
        assert str(mod.asset_dir) == expected + 'part8_shap_vs_patching/media'
        calls = []
        token = object()
        def tensor_stub(data, *, dtype):
            calls.append((data, dtype))
            return token
        mod.t = SimpleNamespace(tensor=tensor_stub, float64=torch.float64)
        mod.initialize_agreement_defaults()
        assert calls == [([1.2, -0.7, 1.6, 0.9], torch.float64)]
        functions = [mod.finite_circuit_value, mod.finite_circuit_table, mod.mechanistic_endpoint_scores]
        for function in functions:
            assert function.__kwdefaults__['linear_weights'] is token
            assert function.__kwdefaults__['pair_weights'] is mod.TOY_PAIR_WEIGHTS
            assert function.__closure__ is None
            assert function.__qualname__ == function.__name__
        mod.initialize_agreement_defaults()
        assert len(calls) == 1
        assert functions == [mod.finite_circuit_value, mod.finite_circuit_table, mod.mechanistic_endpoint_scores]
    finally:
        sys.path[:] = baseline_path


def test_standard_page_renderer_inlines_all_ten_historical_media_entries():
    sys.path.insert(0, str(R))
    try:
        from infrastructure.extension.render_media import inline_page_media
    finally:
        sys.path.pop(0)
    page = C / CH / 'instructions/pages/08_[16.8]_SHAP_vs_Activation_Patching.md'
    text = page.read_text()
    rendered = inline_page_media(text, page)
    assert '../../exercises/part8_shap_vs_patching/media/' not in rendered
    assert rendered.count('src="data:image/') >= 10
    for row in read(E / 'media_map.json'):
        payload = (C / row['target']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row['sha256']
        assert base64.b64encode(payload).decode() in rendered
    shell = page.with_suffix('.py').read_text()
    assert 'inline_page_media(markdown_content_file.read_text' in shell


def test_forecast_precedes_public_answers_and_both_play_cells_remain():
    for kind in ['exercises', 'solutions']:
        book = read(P / f'{STEM}_{kind}.ipynb')
        text = '\n'.join(source(c) for c in book['cells'])
        assert text.index('Your commitment') < text.index('Falsifiable claim')
        assert 'play_interaction_strength' in text
        assert 'positive_pair_strength' in text
        assert text.count('### Predict before inspection') == 16
        assert text.count('### Reflect after testing') == 17
        assert text.count('You should spend 10-20 minutes') == 9
        assert 'Human learner walkthrough and scientific reproduction remain deferred' in text


def test_every_original_nonwrapper_function_and_stub_ast_is_preserved():
    maps = read(E / 'alpha_maps.json')
    mapping = read(E / 'cell_inverse_map.json')
    checked = 0
    for kind in ['exercises', 'solutions']:
        for row in mapping[kind]:
            old = row['original_cell']
            new = row['transformed_cell']
            if old['cell_type'] != 'code':
                continue
            undo = {v: k for k, v in maps[row['legacy']].items()}
            class Undo(ast.NodeTransformer):
                def visit_Name(self, node):
                    node.id = undo.get(node.id, node.id)
                    return node
                def visit_FunctionDef(self, node):
                    node.name = undo.get(node.name, node.name)
                    return self.generic_visit(node)
            originals = {n.name: n for n in ast.parse(source(old)).body
                         if isinstance(n, ast.FunctionDef)}
            changed = Undo().visit(ast.parse(source(new)))
            transformed = {n.name: n for n in changed.body if isinstance(n, ast.FunctionDef)}
            assert set(originals) == set(transformed)
            for name, function in originals.items():
                if name in {'run_gpu_test', 'run_full_experiment'}:
                    continue  # Explicitly declared historical lazy-import wrapper compatibility.
                assert ast.dump(function) == ast.dump(transformed[name]), (kind, row['legacy'], name)
                checked += 1
    assert checked == 85


def test_all_35_original_inputs_remain_byte_exact():
    manifest = read(E / 'source_manifest.json')
    assert len(manifest) == 35
    for row in manifest:
        payload = file_bytes(W / 'preimages' / row['path'])
        assert len(payload) == row['size']
        assert hashlib.sha256(payload).hexdigest() == row['sha256']
        assert (R / row['path']).read_bytes() == payload


def test_main_entrypoint_initializes_plumbing_before_default_callbacks():
    module = ast.parse((P / 'solutions.py').read_text())
    hooks = []
    for index, node in enumerate(module.body):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'MAIN':
            for inner in node.body:
                if isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call):
                    if isinstance(inner.value.func, ast.Name):
                        hooks.append((index, inner.value.func.id))
    positions = {name: index for index, name in hooks}
    assert positions['initialize_lesson_plumbing'] < positions['initialize_agreement_defaults']
    initialized_at = positions['initialize_agreement_defaults']
    for index, node in enumerate(module.body):
        if not isinstance(node, ast.If):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id in {'finite_circuit_value', 'finite_circuit_table', 'mechanistic_endpoint_scores'}:
                    assert index > initialized_at
