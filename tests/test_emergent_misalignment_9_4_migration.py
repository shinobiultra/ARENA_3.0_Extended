"""Static source regression only; never imports a lesson, tensor library or runtime test."""
from pathlib import Path
import ast
import copy
import gzip
import hashlib
import json
import re
import tarfile

import pytest

from infrastructure.extension.source_preservation import apply_compatibility

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / 'chapter9_alignment_interpretability/exercises/part4_emergent_misalignment'
EVIDENCE = ROOT / 'docs/mastery/evidence/emergent_misalignment_migration_20261020'
STEM = '9.4_Emergent_Misalignment_Detection'
PAGE = '04_[9.4]_Emergent_Misalignment_Detection'

def source(cell):
    return ''.join(cell['source'])

def digest(value):
    return hashlib.sha256(value).hexdigest()

def json_digest(value):
    return digest(json.dumps(value, sort_keys=True).encode())

def definitions(tree):
    return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def assert_navigation_only_change(current, original, generated, compatibility=None):
    """Permit exact navigation and declared format metadata; preserve all teaching objects."""
    original_nav = [i for i, cell in enumerate(original['cells']) if '> **Colab:' in source(cell)]
    generated_nav = [cell for cell in generated['cells'] if '> **Colab:' in source(cell)]
    assert len(original_nav) == len(generated_nav) == 1
    index = original_nav[0]
    assert current['cells'][index]['source'] == generated_nav[0]['source']
    inverse = copy.deepcopy(current)
    inverse['cells'][index]['source'] = copy.deepcopy(original['cells'][index]['source'])
    expected = original if compatibility is None else apply_compatibility(original, compatibility)
    assert inverse == expected


def archived_navigation_books():
    directory = ROOT / 'docs/mastery/evidence/lesson_integration_20261020'
    expected = {
        'navigation_before.tar.gz': '2596e3f49dbffe16ae69f1298ef37c9470c8f31db996b0ffc858ed9573003093',
        'navigation_generated_raw.tar.gz': '6707b00a90fc7e3bd918fb611363350cc3650fd30a51d13f9509263710c833a9',
    }
    books = []
    for name, sha256 in expected.items():
        path = directory / name
        assert digest(path.read_bytes()) == sha256
        with tarfile.open(path) as handle:
            books.append({
                str(path.relative_to(ROOT)): handle.extractfile(str(path.relative_to(ROOT))).read()
                for path in [SECTION / f'{STEM}_{kind}.ipynb' for kind in ['exercises', 'solutions']]
            })
    return books


def test_navigation_inverse_rejects_non_navigation_mutations():
    original = {
        'metadata': {'preserved': True}, 'nbformat': 4, 'nbformat_minor': 5,
        'cells': [
            {'cell_type': 'markdown', 'source': ['> **Colab: old'], 'metadata': {}, 'id': 'nav'},
            {'cell_type': 'markdown', 'source': ['Original teaching'], 'metadata': {}, 'id': 'text'},
            {'cell_type': 'code', 'source': ['answer = 7'], 'metadata': {}, 'id': 'code',
             'execution_count': 2, 'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': ['7']}]},
        ],
    }
    generated = copy.deepcopy(original)
    generated['cells'][0]['source'] = ['> **Colab: exact generated navigation']
    assert_navigation_only_change(generated, original, generated)
    mutations = [
        lambda book: book['cells'][2].update(source=['answer = 8']),
        lambda book: book['cells'][2].update(outputs=[]),
        lambda book: book['cells'][2].update(execution_count=3),
        lambda book: book['cells'][2].update(metadata={'altered': True}),
        lambda book: book['cells'][2].update(id='altered'),
        lambda book: book['cells'][1].update(source=['Altered teaching']),
        lambda book: book['metadata'].update(preserved=False),
        lambda book: book['cells'][0].update(source=['> **Colab: arbitrary navigation']),
    ]
    for mutation in mutations:
        altered = copy.deepcopy(generated)
        mutation(altered)
        with pytest.raises(AssertionError):
            assert_navigation_only_change(altered, original, generated)

def test_canonical_paths_and_full_original_object_inverse():
    ledger = json.loads((EVIDENCE / 'parity_ledger.json').read_text())
    for kind in ['exercises','solutions']:
        record = ledger['books'][kind]
        notebook = json.loads((SECTION / f'{STEM}_{kind}.ipynb').read_text())
        assert notebook['metadata'] == record['metadata']
        assert len(record['mapping']) == 41
        assert sum(c['cell_type']=='code' for c in notebook['cells']) == 15
        assert sum(bool(c.get('outputs')) for c in notebook['cells']) == record['output_bearing_cells']
        assert sum(len(c.get('outputs',[])) for c in notebook['cells']) == record['output_records']
        for row in record['mapping']:
            c = notebook['cells'][row['canonical_cell']]
            assert c['source'] == row['canonical_source']
            assert json_digest(c) == row['canonical_cell_sha256']
            assert json_digest(c.get('outputs',[])) == row['outputs_sha256']
            assert c.get('execution_count') == row['execution_count']
            assert c.get('metadata',{}) == row['metadata']
            assert c.get('id') == row['id']
            inverse = dict(c, source=row['original_source'])
            assert json_digest(inverse) == row['original_cell_sha256']
        text = '\n'.join(source(c) for c in notebook['cells'])
        assert text.count('## Content & Learning Objectives') == 1
        assert text.count('## Prediction checkpoint ') == 8
        assert text.count('## Reflection checkpoint ') == 8
        assert text.count('Prior exposure to page, code, outputs or report: ______') == 8
        assert '<summary>Historical solution</summary>' in text
        assert '## Try It Yourself' in text and 'shuffle_heldout_labels = False' in text
        assert '## Historical report appendix' in text
    assert ledger['books']['solutions']['output_bearing_cells'] == 12
    assert len(ledger['books']['solutions']['png_outputs']) == 1
    case = json.loads((EVIDENCE/'proposed_case.json').read_text())
    assert ROOT/case['exercise_path'] == SECTION/f'{STEM}_exercises.ipynb'
    assert ROOT/case['solutions_path'] == SECTION/f'{STEM}_solutions.ipynb'
    assert case['forecast_items'] == [] and case['status'] == 'guided_practice'
    assert 'all ten held-out rows' in case['experiment']

def test_student_stubs_solution_functions_and_test_wiring():
    books = {kind:json.loads((SECTION/f'{STEM}_{kind}.ipynb').read_text()) for kind in ['exercises','solutions']}
    exercise_code = '\n'.join(source(c) for c in books['exercises']['cells'] if c['cell_type']=='code')
    solution_code = '\n'.join(source(c) for c in books['solutions']['cells'] if c['cell_type']=='code')
    original_exercise_stubs = sum('NotImplementedError' in ast.unparse(n) for n in ast.walk(ast.parse(exercise_code)) if isinstance(n,ast.FunctionDef))
    assert original_exercise_stubs > 0
    assert not any(isinstance(n,ast.Raise) and 'NotImplementedError' in ast.unparse(n) for n in ast.walk(ast.parse(solution_code)))
    lesson = ast.parse((SECTION/'solutions.py').read_text())
    nbdefs = definitions(ast.parse(solution_code)); pydefs = definitions(lesson)
    assert sum(isinstance(n,ast.FunctionDef) for n in nbdefs.values()) == 40
    assert sum(isinstance(n,ast.ClassDef) for n in nbdefs.values()) == 4
    for name,node in nbdefs.items():
        generated=copy.deepcopy(pydefs[name])
        if name=='toy_proxy_drift_timeline':
            assert ast.unparse(generated.body[1])=='_initialize_toy_proxy_constants()'
            del generated.body[1]
        assert ast.dump(node) == ast.dump(generated), name
    assert all(len(n.decorator_list)==1 for n in pydefs.values() if isinstance(n,ast.ClassDef))
    tests = ast.parse((SECTION/'tests.py').read_text())
    testdefs = definitions(tests)
    # Inspect required surface strings without invoking the legacy test functions.
    for function_name, kind in [('test_exercise_notebook_exposes_arena_learner_surface','exercises'),('test_solution_notebook_mirrors_progression_and_inlines_taught_methods','solutions')]:
        surface='\n'.join(source(c) for c in books[kind]['cells'])
        function=testdefs[function_name]
        required=[]
        for node in ast.walk(function):
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='required_strings' for target in node.targets):
                required.extend(ast.literal_eval(node.value))
            if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='text' and isinstance(node.iter,ast.List):
                required.extend(ast.literal_eval(node.iter))
        for literal in required:
            assert literal in surface,(kind,literal)
    calls = {n.func.attr for n in ast.walk(ast.parse(solution_code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tests'}
    assert calls and calls <= set(testdefs)
    test_text = (SECTION/'tests.py').read_text()
    assert 'exercises.part4_emergent_misalignment import (' in test_text
    assert f'{STEM}_exercises.ipynb' in test_text
    assert 'historical_verification_report.json' in test_text
    for path in [SECTION/'tests.py', SECTION/'solutions.py', SECTION/'validation_reference.py', SECTION/'notebook_checks.py']:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                assert 'arena_ext' not in ast.unparse(node), (path,node.lineno)
    gpu = pydefs['run_gpu_test']
    imports = [n for n in ast.walk(gpu) if isinstance(n,ast.ImportFrom)]
    assert len(imports)==1 and imports[0].module=='chapter9_alignment_interpretability.exercises.part3_emergent_misalignment_detection.solutions'
    assert [n.name for n in imports[0].names] == ['run_pythia_proxy_drift_preflight']
    assert not any(isinstance(n,ast.ImportFrom) and n.module and 'part3_emergent' in n.module for n in lesson.body)

def test_setup_guards_and_exact_inverse():
    ledger=json.loads((EVIDENCE/'parity_ledger.json').read_text())
    setup=next(r for r in ledger['books']['solutions']['mapping'] if r['legacy_cell']==4)
    routed=''.join(setup['canonical_source']);original=''.join(setup['original_source'])
    assert routed.replace('part4_emergent_misalignment','part3_emergent_misalignment_detection')==original
    assert 't.set_grad_enabled(False)' in routed
    module=ast.parse((SECTION/'solutions.py').read_text())
    guarded=[]
    def walk(nodes, inside_main=False):
        for node in nodes:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                continue
            if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='MAIN':
                walk(node.body, True)
                continue
            if isinstance(node,(ast.Import,ast.ImportFrom)) and 'part4_emergent' in ast.unparse(node):
                assert inside_main
            for call in ast.walk(node):
                if not isinstance(call,ast.Call):continue
                name=ast.unparse(call.func)
                if name in ['t.set_grad_enabled','Path.cwd','sys.path.append','json.loads','display','print','plot_toy_signature_result'] or name.startswith('tests.') or name.endswith('.read_text') or name.endswith('.mkdir'):
                    assert inside_main,(name,node.lineno)
                    guarded.append(name)
    walk(module.body)
    assert 't.set_grad_enabled' in guarded and 'Path.cwd' in guarded and 'json.loads' in guarded

def test_historical_reports_plumbing_and_master():
    assert not (SECTION/'verification_report.json').exists()
    report=json.loads((SECTION/'historical_verification_report.json').read_text())
    assert report['notebook_id']=='9_3_emergent_misalignment_detection'
    assert report['git_commit']=='8eab192b' and report['date_run'].startswith('2026-09-04')
    rows=json.loads((EVIDENCE/'local_validation_spans.json').read_text())
    for row in rows:
        module=ast.parse((ROOT/row['canonical_path']).read_text())
        assert digest(ast.dump(definitions(module)[row['name']]).encode())==row['ast_sha256']
    master=ROOT/'infrastructure/chapters/chapter9_alignment_interpretability/master_9_4.py'
    text=master.read_text()
    assert text.count('## Content & Learning Objectives')==1
    assert '> ##### Learning Objectives\n>\n> 1. Construct' in text
    assert (ROOT/'infrastructure/chapters/chapter9_alignment_interpretability/master_9_4.ipynb').is_file()
    page=(ROOT/f'chapter9_alignment_interpretability/instructions/pages/{PAGE}.md').read_text()
    assert page.count('## Content & Learning Objectives')==1
    assert page.count('## Prediction checkpoint ')==8
    assert '## Historical report appendix' in page
    assert 'def activation_difference_direction(' in page and 'def projection_mitigation_intervention(' in page
    for p in [SECTION/'solutions.py', SECTION/'tests.py', SECTION/'validation_reference.py', SECTION/'notebook_checks.py', master]:
        compile(p.read_text(),str(p),'exec')
    # The same strict hashes reject dropped or altered original objects.
    ledger=json.loads((EVIDENCE/'parity_ledger.json').read_text());row=ledger['books']['solutions']['mapping'][1]
    notebook=json.loads((SECTION/f'{STEM}_solutions.ipynb').read_text());c=notebook['cells'][row['canonical_cell']]
    mutation=dict(c,source=c['source']+['\nUNDECLARED_DEFAULT = True\n'])
    assert json_digest(mutation)!=row['canonical_cell_sha256']


def test_generated_python_zero_definition_time_allocations_and_v2_inverse():
    path=SECTION/'solutions.py'
    tree=ast.parse(path.read_text())
    eager_calls=[]
    defaults=[]
    def expressions(node):
        return [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)]
    for node in tree.body:
        if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='MAIN':
            continue
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            evaluated=[*node.decorator_list,*node.args.defaults,*[v for v in node.args.kw_defaults if v is not None]]
            evaluated += [a.annotation for a in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs] if a.annotation is not None]
            if node.returns is not None:evaluated.append(node.returns)
            for expr in evaluated:eager_calls.extend(expressions(expr))
            defaults.append({'name':node.name,'defaults':[ast.unparse(v)for v in [*node.args.defaults,*[x for x in node.args.kw_defaults if x is not None]]]})
            continue
        if isinstance(node,ast.ClassDef):
            for expr in [*node.decorator_list,*node.bases,*[k.value for k in node.keywords]]:
                eager_calls.extend(expressions(expr))
            for statement in node.body:eager_calls.extend(expressions(statement))
            continue
        eager_calls.extend(expressions(node))
    assert eager_calls == ['dataclass','dataclass','dataclass','dataclass'],eager_calls
    assert not any(call.startswith(('t.','torch.','np.','numpy.')) for call in eager_calls)
    proof=json.loads((EVIDENCE/'v3_repair_proof.json').read_text())
    before=gzip.decompress((EVIDENCE/proof['solutions_before_gzip']).read_bytes()).decode()
    initializer=proof['initializer_source'];constant=proof['original_constant_source']
    current=path.read_text()
    inverse=current.replace(initializer,constant).replace(proof['docstring']+'\n'+proof['initializer_call'],proof['docstring'])
    assert inverse==before
    assert digest(inverse.encode())==proof['solutions_before_sha256']
    # Preserve the original exact hash claim against its immutable pre-navigation bytes.
    # Current books must invert to those same complete objects after the one approved header change.
    before_books, generated_books = archived_navigation_books()
    contracts = ROOT / 'docs/mastery/evidence/notebook_schema_compatibility_20261020/reviewed_seven_contracts.json'
    assert digest(contracts.read_bytes()) == 'bb2ba063bcc8c960790e44b90cd81e6f03b6f23907dae2d20107195d79dd0cf6'
    compatibility = json.loads(contracts.read_bytes())['sections']['9.4']['books']
    for row in proof['unchanged_notebooks']:
        archived = before_books[row['path']]
        assert digest(archived)==row['sha256']
        kind = 'exercises' if row['path'].endswith('_exercises.ipynb') else 'solutions'
        assert_navigation_only_change(
            json.loads((ROOT/row['path']).read_bytes()),
            json.loads(archived), json.loads(generated_books[row['path']]),
            compatibility=compatibility[kind],
        )
    assert proof['earlier_pre_sentinel_freeze_status']=='LOST_NO_SURVIVING_RAW_FREEZE_FOUND'

if __name__=='__main__':
    for fn in [test_canonical_paths_and_full_original_object_inverse,test_student_stubs_solution_functions_and_test_wiring,test_setup_guards_and_exact_inverse,test_historical_reports_plumbing_and_master,test_generated_python_zero_definition_time_allocations_and_v2_inverse]:
        fn()
        print('PASS:',fn.__name__)
    print('Static checks only: no taught import, numerical method, test suite, model, training, data or network execution.')
