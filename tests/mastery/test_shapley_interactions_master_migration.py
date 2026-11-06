"""Static-only proofs for the full three-track migration; no taught-module imports."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CH = ROOT / 'chapter16_shapley_attribution_baselines'
SECTION = CH / 'exercises/part5_shapley_interactions'
EVIDENCE = ROOT / 'docs/mastery/evidence/shapley_interactions_master_migration'
STEM = '16.5_Shapley_Interactions'
TRACKS = [('toy', 'part3_shapley_interactions_shapiq', '16.3_Shapley_Interactions_with_shapiq'), ('allocation', 'part16_shapley_taylor_faith_shap', '16.16_Where_Does_Higher_Order_Credit_Go'), ('estimator', 'part17_shapiq_estimators', '16.17_One_Query_Many_Interactions')]


def read(p):
    return json.loads(p.read_text())


def source(c):
    return ''.join(c['source'])


def codes(j):
    return [c for c in j['cells'] if c['cell_type'] == 'code']


def book(kind):
    return read(SECTION / f'{STEM}_{kind}.ipynb')


def helper():
    spec = importlib.util.spec_from_file_location('namespace_plumbing_only', EVIDENCE / 'namespace.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inverse(text, tag, old):
    tree = ast.parse(text)
    if tag == 'toy':
        for node in ast.walk(tree):
            if isinstance(node, ast.alias) and node.name == 'chapter16_shapley_attribution_baselines.exercises.part3_shapley_interactions_shapiq.tests':
                node.name = 'part3_shapley_interactions_shapiq.tests'
    return helper().inverse(ast.unparse(tree), read(EVIDENCE / 'namespace_maps.json')['maps'][tag], old)


def test_strict_archive_union_and_every_original_byte():
    from scripts.evidence_bundles import validate_review_bundle, sha256
    before = read(EVIDENCE / 'before.json')
    c = validate_review_bundle(ROOT / before['archive'])
    assert len(before['files']) == 141
    assert len(c['members']) == len(before['internal']) == 113
    assert len(c['external_payloads']) == len(before['external']) == 28
    assert set(c['members']).isdisjoint(c['external_payloads'])
    assert set(c['members']) | set(c['external_payloads']) == set(before['files'])
    assert sum(n.endswith('.tar.gz') for n in c['external_payloads']) == 2
    assert sha256(ROOT / before['archive']) == before['archive_sha256']
    for path, record in before['files'].items():
        assert sha256(ROOT / path) == record['sha256'], path
    assert before['report_closure'] == {'toy': {'report_inputs': 14, 'matching': 14}, 'allocation': {'report_inputs': 51, 'matching': 51}, 'estimator': {'report_inputs': 67, 'matching': 67}}


def test_all_ninety_code_cells_have_exact_inverse_ast_and_historical_output_identity():
    original = [(tag, c) for tag, part, stem in TRACKS for c in codes(read(CH / 'exercises' / part / f'{stem}_solutions.ipynb'))]
    current = codes(book('solutions'))
    assert len(original) == len(current) == 90
    for (tag, old), new in zip(original, current, strict=True):
        assert inverse(source(new), tag, source(old)), (tag, source(old)[:80])
        assert new.get('outputs', []) == old.get('outputs', [])
        assert new['execution_count'] == old['execution_count']
        assert new['metadata'] == old['metadata']
    assert sum(len(c.get('outputs', [])) for c in current) == 79
    assert sum('image/png' in o.get('data', {}) for c in current for o in c.get('outputs', [])) == 21
    assert not inverse(source(current[0]) + '\n_scientific_change = 1\n', 'toy', source(original[0][1]))
    # Real coefficient/metric mutation must fail, not receive a blanket exclusion.
    target = next((x, c) for x, c in zip(original, current) if 'TRAINING_STEPS = 800' in source(c))
    (tag, old), new = target
    assert not inverse(source(new).replace('800', '801', 1), tag, source(old))


def test_own_answers_complete_each_stub_without_cross_track_substitution():
    student = book('exercises')
    solutions = iter(codes(book('solutions')))
    groups = 0
    for i, cell in enumerate(student['cells']):
        if cell['cell_type'] != 'code':
            continue
        correct = next(solutions)
        if 'NotImplementedError' not in source(cell):
            continue
        after = []
        for c in student['cells'][i + 1:]:
            if c['cell_type'] != 'markdown':
                break
            after.append(source(c))
        matches = re.findall(r'<summary>Solution</summary>(.*?)</details>', '\n'.join(after), re.S)
        assert len(matches) == 1
        answer = '\n'.join(re.findall(r'```python\n(.*?)```', matches[0], re.S))
        definitions = {n.name: n for n in ast.parse(answer).body if isinstance(n, ast.FunctionDef)}
        tree = ast.parse(source(cell))
        tree.body = [definitions[n.name] if isinstance(n, ast.FunctionDef) and 'NotImplementedError' in ast.unparse(n) else n for n in tree.body]
        assert ast.dump(tree) == ast.dump(ast.parse(source(correct)))
        groups += 1
    assert groups == 23


def test_twenty_two_expected_figures_without_actual_exercise_answers_or_counts():
    expected = read(EVIDENCE / 'output_parity.json')['figures']
    assert len(expected) == 22
    assert sum(f['kind'] == 'historical_actual_png_output' for f in expected) == 21
    assert sum(f['kind'] == 'historical_saved_only_asset_expected_display_added' for f in expected) == 1
    found = []
    for cell in codes(book('exercises')):
        assert cell['execution_count'] is None
        for o in cell.get('outputs', []):
            assert o['output_type'] == 'display_data'
            assert set(o['data']) == {'text/plain', 'text/html'}
            assert ''.join(o['data']['text/plain']) == '<IPython.core.display.HTML object>'
            html = ''.join(o['data']['text/html'])
            assert not re.sub(r'<img\s+src="data:image/png;base64,[A-Za-z0-9+/=]+"\s+width="1050">', '', html).strip()
            found.extend(hashlib.sha256(base64.b64decode(s)).hexdigest() for s in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', html))
    assert sorted(found) == sorted(f['sha256'] for f in expected)
    assert len(found) == 22
    for f in expected:
        assert hashlib.sha256((SECTION / 'media' / f['name']).read_bytes()).hexdigest() == f['sha256']


def test_all_original_notebook_and_cell_metadata_are_retained_as_provenance():
    saved = read(EVIDENCE / 'source_metadata.json')
    for kind in ['exercises', 'solutions']:
        m = book(kind)['metadata']['migration']
        assert m['fresh_execution'] is False
        for tag, part, stem in TRACKS:
            original = read(CH / 'exercises' / part / f'{stem}_{kind}.ipynb')
            assert m['source_notebook_metadata'][tag] == original['metadata'] == saved['notebooks'][tag][kind]
            cells = [{k: c[k] for k in ['id', 'cell_type', 'metadata'] if k in c} for c in original['cells']]
            assert m['source_cell_metadata'][tag] == cells == saved['original_cells'][tag][kind]
    assert 'fresh_reference' not in book('exercises')['metadata']


def test_source_visible_methods_controls_plays_forecasts_and_original_interfaces():
    text = '\n'.join(source(c) for c in book('solutions')['cells'])
    for name in ['all_coalitions', 'polynomial_game', 'discrete_second_difference', 'pairwise_shapley_interactions', 'exact_shapley_values', 'sampled_pair_interaction', 'permute_coalition_values_within_sizes', 'shapiq_pairwise_interactions', 'interaction_recovery_report', 'mobius_transform', 'shapley_taylor_interactions', 'faith_design_matrix', 'faith_kernel_weights', 'faith_shap_interactions', 'faith_shap_closed_form', 'reconstruct_game', 'faithfulness_error', 'masked_model_game', 'contextual_pair_synergy', 'coalition_design', 'sii_query_weights', 'sample_coalitions', 'shapiq_estimate', 'query_sampled_game', 'constrained_wls', 'pair_residual_design', 'kernel_weights', 'kernelshap_iq', 'empirical_context_play', 'native_play']:
        assert re.search(r'def ' + name + r'\(', text), name
    assert 'PLAY_TRIPLE_WEIGHT' in text and 'PLAY_BUDGET' in text
    assert 'CONTEXT_MASK' in text and 'DONOR_SOURCE_ID' in text and 'DRAWS' in text
    assert 'make_native_play' in text and 'QUERY_MASK' in text
    assert 'arena16.16/shapley-interactions/v1' in text
    assert 'WITHIN_SIZE_REPLICATES = 128' in text and 'RANDOM_PAIR_ORDERS = 256' in text
    assert text.count('## Forecast before running') >= 45
    assert text.count('## Forecast before running') == text.count('## Reflect after the result')
    assert 'shuffled' in text and 'ridge' in text and 'permutation' in text
    assert 'saved-only asset' in text and '79' in text
    for part, module in [('part16_shapley_taylor_faith_shap', 'interaction_study.py'), ('part17_shapiq_estimators', 'estimator_study.py')]:
        assert 'getattr(methods' in (CH / 'exercises' / part / module).read_text()


def test_python_module_compiles_with_future_hoist_and_has_no_top_level_study_calls():
    path = SECTION / 'solutions.py'
    text = path.read_text()
    compile(text, str(path), 'exec')
    tree = ast.parse(text)
    assert isinstance(tree.body[0], ast.ImportFrom) and tree.body[0].module == '__future__'
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(n, ast.If):
            assert ast.unparse(n.test) in ['MAIN', 'str(_MIGRATION_ROOT) not in _migration_sys.path', "str(_MIGRATION_ROOT / 'chapter16_shapley_attribution_baselines/exercises') not in _migration_sys.path"]
            continue
        assert isinstance(n, ast.Assign)
        assert any(isinstance(t, ast.Name) and t.id in ['MAIN', '_MIGRATION_ROOT', 'FEATURES', 'EMPIRICAL_ORDER'] for t in n.targets)
    for _, part, _ in TRACKS:
        p = CH / 'exercises' / part
        for name in ['tests.py', 'interaction_methods_tests.py', 'estimator_methods_tests.py']:
            if (p / name).exists():
                assert 'part5_shapley_interactions' not in (p / name).read_text()
    mapping = read(EVIDENCE / 'namespace_maps.json')
    assert 'annotations' not in mapping['collisions']


def test_standard_guided_case_size_and_local_page_routes():
    from mastery_hub.units.extension_predictions.lesson import validate_case
    case = read(EVIDENCE / 'proposed_case.json')
    assert validate_case(case) == case
    assert case['prediction_type'] == 'numeric' and case['transfer_prediction_type'] == 'binary'
    assert case['evidence_class'] == 'analytic_toy' and case['data_role'] == 'historical'
    assert 'one selected pair' in case['config_fields']['denominators']
    assert not set(case) & {'answer', 'prediction', 'observation', 'confidence', 'reflection'}
    page = CH / 'instructions/pages/05_[16.5]_Shapley_Interactions.md'
    paths = re.findall(r'src="(../../exercises/[^\"]+\.png)"', page.read_text())
    assert len(paths) == 22 and all((page.parent / p).is_file() for p in paths)
    for path in [SECTION / f'{STEM}_exercises.ipynb', SECTION / f'{STEM}_solutions.ipynb', SECTION / 'solutions.py', page]:
        assert path.stat().st_size <= 5 * 1024**2, path


def test_multiline_stub_blank_lines_remain_executable_after_original_converter():
    """Regression for local emission, without changing or executing the shared converter's output."""
    from infrastructure.core.conversion.cell import Cell
    stub = 'def fixture():\n    """First line.\n\n    Second line.\n    """\n    raise NotImplementedError()'
    commented = ['# ' + line if line.strip() else '' for line in stub.splitlines()]
    answer = 'def fixture():\n    return 1'
    cell = Cell([], [], 'code', ['# EXERCISE', *commented, '# END EXERCISE', '# SOLUTION', *answer.splitlines(), '# END SOLUTION'], 'local multiline-stub fixture')
    generated = '\n'.join(cell.process_inline_filters()['colab-ex'])
    tree = ast.parse(generated)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == 'fixture'
    assert isinstance(tree.body[0].body[-1], ast.Raise)
    assert ast.get_docstring(tree.body[0]) == 'First line.\n\nSecond line.'
    master = (ROOT / 'infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_5.py').read_text()
    assert not any(line.strip() == '#' for block in re.findall(r'# EXERCISE\n(.*?)# END EXERCISE', master, re.S) for line in block.splitlines())


def test_definition_time_dependencies_have_prior_import_safe_bindings():
    """Audit defaults/decorators/bases in definition order, not merely AST syntax."""
    import builtins
    tree = ast.parse((SECTION / 'solutions.py').read_text())
    assert any(isinstance(n, ast.ImportFrom) and n.module == '__future__' and any(a.name == 'annotations' for a in n.names) for n in tree.body)
    available = set(vars(builtins)) | {'__name__', '__file__'}
    examined = []
    def dependencies(expr):
        return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    def definition(node, visible):
        exprs = list(node.decorator_list)
        if isinstance(node, ast.ClassDef):
            exprs += node.bases + [x.value for x in node.keywords]
        else:
            exprs += [x for x in [*node.args.defaults, *node.args.kw_defaults] if x is not None]
        for expression in exprs:
            missing = dependencies(expression) - visible
            assert not missing, (node.name, ast.unparse(expression), sorted(missing))
            examined.append((node.name, ast.unparse(expression)))
        if isinstance(node, ast.ClassDef):
            class_visible = set(visible)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                    definition(child, class_visible)
                    class_visible.add(child.name)
    for node in tree.body:
        if isinstance(node, ast.Import):
            available.update(a.asname or a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != '__future__':
            available.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            available.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definition(node, available)
            available.add(node.name)
        # if MAIN blocks deliberately contribute no import-time bindings.
    assert ('explain_complete_games', 'FEATURES') in examined
    assert ('explain_complete_games', 'EMPIRICAL_ORDER') in examined
    assert ('InteractionRecoveryReport', 'dataclass(frozen=True)') in examined
    assert ('_AdditiveDiagnosticModel', 'nn.Module') in examined
    assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    assert ast.literal_eval(assignments['FEATURES']) == 8
    assert ast.literal_eval(assignments['EMPIRICAL_ORDER']) == 2
    # These are exact literal copies of the original notebook protocol settings.
    native = read(CH / 'exercises/part16_shapley_taylor_faith_shap/16.16_Where_Does_Higher_Order_Credit_Go_solutions.ipynb')
    original = ast.parse('\n'.join(source(c) for c in codes(native)))
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in original.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ['FEATURES', 'EMPIRICAL_ORDER']}
    assert constants == {'FEATURES': 8, 'EMPIRICAL_ORDER': 2}


def test_one_bounded_inert_import_with_scientific_and_write_guards():
    import os
    import subprocess
    import sys
    probe = EVIDENCE / 'import_safety_v3/guarded_import.py'
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'MPLCONFIGDIR': '/tmp/arena165-inert-import-mpl'}
    result = subprocess.run([sys.executable, str(probe)], cwd=ROOT, env=environment, text=True, capture_output=True, timeout=18)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['status'] == 'PASS' and report['MAIN'] is False
    assert report['literal_defaults'] == {'FEATURES': 8, 'EMPIRICAL_ORDER': 2}
    assert report['scientific_functions_called'] is False and report['writes_or_network'] is False and report['TorchCUDA_calls'] is False


def test_semantic_test_imports_defer_only_to_main_and_resolve_when_requested():
    """Exercise original import statements with stub modules; never import/run legacy tests."""
    from types import SimpleNamespace
    tree = ast.parse((SECTION / 'solutions.py').read_text())
    aliases = {'toy_tests', 'allocation_tests', 'estimator_tests'}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert not aliases & {a.asname or a.name.split('.')[0] for a in node.names}
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert not aliases & {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'MAIN':
            for child in node.body:
                if isinstance(child, (ast.Import, ast.ImportFrom)) and aliases & {a.asname or a.name.split('.')[0] for a in child.names}:
                    imports.append(child)
    assert len(imports) == 3
    expected_sources = {'toy_tests': CH / 'exercises/part3_shapley_interactions_shapiq/tests.py', 'allocation_tests': CH / 'exercises/part16_shapley_taylor_faith_shap/interaction_methods_tests.py', 'estimator_tests': CH / 'exercises/part17_shapiq_estimators/estimator_methods_tests.py'}
    for statement in imports:
        alias = next(a.asname for a in statement.names if a.asname in aliases)
        declared_tests = [n.name for n in ast.parse(expected_sources[alias].read_text()).body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
        assert declared_tests
        semantic = SimpleNamespace(**{name: (lambda: None) for name in declared_tests})
        calls = []
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            calls.append((name, tuple(fromlist or ())))
            if fromlist:
                return SimpleNamespace(**{n: semantic for n in fromlist})
            # IMPORT_NAME/IMPORT_FROM traversal for the original qualified toy import.
            nested = semantic
            for field in reversed(name.split('.')[1:]):
                nested = SimpleNamespace(**{field: nested})
            return nested
        guard = ast.If(test=ast.Name(id='MAIN', ctx=ast.Load()), body=[statement], orelse=[])
        code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), 'semantic-import-only-fixture', 'exec')
        for main in [False, True]:
            scope = {'MAIN': main, '__builtins__': {'__import__': fake_import}}
            calls.clear()
            exec(code, scope)
            assert bool(calls) is main
            if main:
                assert all(callable(getattr(scope[alias], name)) for name in declared_tests)
            else:
                assert alias not in scope
    graph = read(EVIDENCE / 'import_safety_v3/eager_import_graph.json')
    assert graph['reachable_semantic_test_modules'] == []
    assert graph['eager_top_level_Torch_allocation_calls'] == []
