"""Static-only preservation and namespace proofs; never import a taught method or loader."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CH = ROOT / 'chapter16_shapley_attribution_baselines'
SECTION = CH / 'exercises/part11_datamodels_cartography'
EVIDENCE = ROOT / 'docs/mastery/evidence/datamodels_cartography_master_migration'
STEM = '16.11_Datamodels_and_Dataset_Cartography'
TRACKS = [('dm', 'part10_datamodels', '16.10_Datamodels'), ('cart', 'part11_dataset_cartography', '16.11_Dataset_Cartography')]


def read(path):
    return json.loads(path.read_text())


def source(cell):
    return ''.join(cell['source'])


def codes(book):
    return [c for c in book['cells'] if c['cell_type'] == 'code']


def book(kind):
    return read(SECTION / f'{STEM}_{kind}.ipynb')


def helper():
    spec = importlib.util.spec_from_file_location('migration_namespace_only', EVIDENCE / 'namespace.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_flat_archive_covers_union_and_originals_remain_exact():
    from scripts.evidence_bundles import validate_review_bundle, sha256
    before = read(EVIDENCE / 'before.json')
    contract = validate_review_bundle(ROOT / before['archive'])
    assert sha256(ROOT / before['archive']) == before['archive_sha256']
    assert set(before['internal']).isdisjoint(before['external'])
    assert set(before['files']) == set(before['internal']) | set(before['external'])
    assert set(contract['members']) == set(before['internal'])
    assert set(contract['external_payloads']) == set(before['external'])
    for name, record in before['files'].items():
        assert sha256(ROOT / name) == record['sha256'], name
    for name in before['external']:
        assert name.endswith('.tar.gz')
        assert contract['external_payloads'][name]['absolute_path'] == str(ROOT / name)
    for _, part, _ in TRACKS:
        actual = {str(p.relative_to(ROOT)) for p in (CH / 'exercises' / part).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        assert actual <= before['files'].keys()


def test_every_scientific_code_cell_has_exact_inverse_ast_and_output_identity():
    namespace = helper()
    maps = read(EVIDENCE / 'namespace_maps.json')['maps']
    canonical = codes(book('solutions'))
    originals = [(tag, c) for tag, part, stem in TRACKS for c in codes(read(CH / 'exercises' / part / f'{stem}_solutions.ipynb'))]
    assert len(canonical) == len(originals) == 61
    for new, (tag, old) in zip(canonical, originals, strict=True):
        assert namespace.inverse(source(new), maps[tag], source(old)), (tag, source(old)[:80])
        assert new['outputs'] == old.get('outputs', [])
        assert new['execution_count'] == old.get('execution_count')
    assert sum(len(c['outputs']) for c in canonical) == 59
    assert sum('image/png' in o.get('data', {}) for c in canonical for o in c['outputs']) == 35
    # The proof must reject a real scientific perturbation, not just assert the map exists.
    first = canonical[0]
    assert not namespace.inverse(source(first) + '\n_migration_scientific_change = 1\n', maps['dm'], source(originals[0][1]))


def test_exercises_have_unexecuted_stubs_and_expected_displays_only():
    ex = book('exercises')
    assert len(codes(ex)) == 61
    assert all(c['execution_count'] is None for c in codes(ex))
    images = []
    for c in codes(ex):
        for o in c.get('outputs', []):
            assert o['output_type'] == 'display_data'
            assert set(o['data']) == {'text/html', 'text/plain'}
            assert ''.join(o['data']['text/plain']) == '<IPython.core.display.HTML object>'
            html = ''.join(o['data']['text/html'])
            assert not re.sub(r'<img\s+src="data:image/png;base64,[A-Za-z0-9+/=]+"\s+width="1050">', '', html).strip()
            images.extend(re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', html))
    expected = read(EVIDENCE / 'output_parity.json')['figures']
    assert sorted(hashlib.sha256(base64.b64decode(x)).hexdigest() for x in images) == sorted(x['sha256'] for x in expected)
    assert len(images) == 35
    assert sum('raise NotImplementedError' in source(c) for c in codes(ex)) == 17


def test_each_stub_completes_from_its_own_next_solution():
    namespace = helper()
    ex, sol = book('exercises'), book('solutions')
    solved = iter(codes(sol))
    groups = 0
    for index, cell in enumerate(ex['cells']):
        if cell['cell_type'] != 'code':
            continue
        expected = next(solved)
        text = source(cell)
        if 'NotImplementedError' not in text:
            continue
        following = []
        for other in ex['cells'][index + 1:]:
            if other['cell_type'] != 'markdown':
                break
            following.append(source(other))
        matches = re.findall(r'<summary>Solution</summary>(.*?)</details>', '\n'.join(following), re.S)
        assert len(matches) == 1
        answer = '\n'.join(re.findall(r'```python\n(.*?)```', matches[0], re.S))
        answers = {n.name: n for n in ast.parse(answer).body if isinstance(n, ast.FunctionDef)}
        tree = ast.parse(text)
        tree.body = [answers[n.name] if isinstance(n, ast.FunctionDef) and 'NotImplementedError' in ast.unparse(n) else n for n in tree.body]
        assert ast.dump(tree) == ast.dump(ast.parse(source(expected)))
        groups += 1
    assert groups == 17


def test_visible_core_methods_controls_and_full_plays_survive():
    text = '\n'.join(source(c) for c in book('solutions')['cells'])
    for name in ['sample_subsets', 'correct_class_margins', 'soft_threshold', 'lasso_objective', 'lasso_kkt', 'fit_lasso', 'fit_datamodel_path', 'centered_weight_similarity', 'select_supportive', 'matched_random_removals', 'align_epoch', 'gold_probabilities', 'training_statistics', 'batched_logits', 'train_ensemble', 'select_indices', 'retrieval_curve', 'selection_overlap', 'matched_random_indices', 'evaluate_models', 'deletion_play', 'retraining_play']:
        assert re.search(r'def ' + name + r'\(', text), name
    assert 'INTERACTION' in text and 'SUBSET_FRACTION' in text
    assert 'class-matched' in text and 'equal' in text and 'shuffled' in text and 'permuted' in text
    assert text.count('## Forecast before running') >= 35
    assert text.count('## Forecast before running') == text.count('## Reflect after the result')
    assert 'retained historical reference' in text


def test_module_top_level_does_not_load_data_train_or_write():
    tree = ast.parse((SECTION / 'solutions.py').read_text())
    allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for n in tree.body:
        if isinstance(n, allowed):
            continue
        if isinstance(n, ast.If):
            assert ast.unparse(n.test) in ['MAIN', 'str(_MIGRATION_ROOT) not in _migration_sys.path']
            continue
        assert isinstance(n, ast.Assign)
        assert any(isinstance(t, ast.Name) and t.id in ['MAIN', '_MIGRATION_ROOT'] for t in n.targets)
    assert 'run_gpu_test(' not in (SECTION / 'solutions.py').read_text()


def test_cohort_and_helper_identities_preserved_without_number_substitution():
    for part, needle in [('part10_datamodels', 'arena16.10/v1'), ('part11_dataset_cartography', 'arena16.11/v1')]:
        p = CH / 'exercises' / part
        assert any(needle in f.read_text() for f in p.glob('*data.py'))
    text = (SECTION / 'solutions.py').read_text()
    assert 'part9_influence_and_tracin.attribution_methods' in text
    assert 'part10_datamodels.datamodel_data' in text and 'part11_dataset_cartography.cartography_data' in text
    assert 'datamodel_methods_tests as dm_tests' in text and 'cartography_methods_tests as cart_tests' in text
    assert 'globals()[name]' in text
    namespace = read(EVIDENCE / 'namespace_maps.json')
    assert 'disjoint' in namespace['dynamic_lookup']


def test_local_media_and_guided_proposal_are_reviewable():
    page = CH / 'instructions/pages/11_[16.11]_Datamodels_and_Dataset_Cartography.md'
    text = page.read_text()
    paths = re.findall(r'src="(../../exercises/[^\"]+\.png)"', text)
    assert len(paths) == 35
    assert all((page.parent / p).is_file() for p in paths)
    proposal = read(EVIDENCE / 'guided_proposal.json')
    assert proposal['status'] == 'proposal_not_adopted'
    assert proposal['answer_free'] is True
    assert proposal['learner_response'] is None
    assert {c['track'] for c in proposal['cases']} == {'datamodels', 'cartography'}
    assert all(c['forecast'] is None and c['observation'] is None and c['reflection'] is None for c in proposal['cases'])
    assert 'checkpoint' not in proposal and 'rubric' not in proposal
