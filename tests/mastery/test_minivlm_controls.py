"""Saved-output Mini VLM checks: no pretrained load, CUDA or training."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest
import torch as t

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / 'chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch'
BASE = ROOT / 'docs/mastery/evidence/minivlm_controls_20261016'
MASTER = ROOT / 'infrastructure/chapters/chapter12_vlm_interpretability/master_12_3.py'
STEM = '12.3_Mini_VLM_from_Scratch'


@pytest.fixture(autouse=True)
def restore_cpu_numeric_state():
    dtype, rng = t.get_default_dtype(), t.random.get_rng_state()
    try:
        yield
    finally:
        t.set_default_dtype(dtype)
        t.random.set_rng_state(rng)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def old_bytes(relative):
    with tarfile.open(BASE / 'before.tar.gz') as archive:
        return archive.extractfile(str(relative)).read()


def namespace():
    # Read and execute ONLY named visible definitions. Do not import solutions:
    # its setup chooses a device; the preparation tests must not probe CUDA.
    sys.path.insert(0, str(SECTION.parent))
    from part3_mini_vlm_from_scratch import utils
    from types import SimpleNamespace
    tree = ast.parse((SECTION / 'solutions.py').read_text())
    names = {'question_and_answer', 'target_blind_oracle', 'replay_question_targets',
             'evaluate_readout_controls', 'qwen_layer_schedule'}
    assignments = {'KINDS', 'NUMBER_WORDS', 'ANSWER_WORDS'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
             or isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in assignments for x in n.targets)]
    source = 'from __future__ import annotations\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    ns = dict(t=t, np=np, Counter=Counter, utils=utils, Scene=utils.Scene, COLOURS=utils.COLOURS, SHAPES=utils.SHAPES)
    exec(compile(source, '<visible MiniVLM control definitions only>', 'exec'), ns)
    test_tree = ast.parse((SECTION / 'tests.py').read_text())
    test_nodes = [node for node in test_tree.body if isinstance(node, ast.FunctionDef)
                  and node.name in {'test_target_blind_oracle', 'test_qwen_layer_schedule', 'test_evaluate_readout_controls'}]
    test_ns = dict(utils=utils, t=t, np=np)
    test_source = 'from __future__ import annotations\n' + ast.unparse(ast.Module(body=test_nodes, type_ignores=[]))
    exec(compile(test_source, '<three immediate CPU fixtures only>', 'exec'), test_ns)
    return ns, SimpleNamespace(**{node.name: test_ns[node.name] for node in test_nodes}), utils


def test_original_archive_is_exact_and_all_historical_outputs_remain_recoverable():
    manifest = json.loads((BASE / 'before_manifest.json').read_text())
    assert digest((BASE / 'before.tar.gz').read_bytes()) == manifest['archive_sha256']
    with tarfile.open(BASE / 'before.tar.gz') as archive:
        assert len(archive.getmembers()) == len(manifest['files']) == 24
        for row in manifest['files']:
            assert digest(archive.extractfile(row['path']).read()) == row['sha256']
    n = json.loads(old_bytes((SECTION / f'{STEM}_solutions.ipynb').relative_to(ROOT)))
    codes = [c for c in n['cells'] if c['cell_type'] == 'code']
    assert len(codes) == 57 and all(c['execution_count'] is not None for c in codes)
    assert sum(len(c['outputs']) for c in codes) == 97
    assert '27    27' in ''.join(''.join(o.get('text', [])) for c in codes for o in c['outputs'])


def test_every_original_method_and_training_test_is_unchanged():
    def definitions(raw):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(raw))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    before = definitions(old_bytes((SECTION / 'solutions.py').relative_to(ROOT)).decode())
    after = definitions((SECTION / 'solutions.py').read_text())
    assert before and all(after[name] == node for name, node in before.items())
    old_tests = old_bytes((SECTION / 'tests.py').relative_to(ROOT)).decode()
    old_call = 'SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")'
    pinned_call = 'SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")'
    assert old_tests.count(old_call) == 1
    pinned_original = old_tests.replace(old_call, pinned_call, 1)
    assert (SECTION / 'tests.py').read_text().startswith(pinned_original)
    assert (SECTION / 'tests.py').read_text().count(pinned_call) == 1
    source = (SECTION / 'solutions.py').read_text()
    assert 'n_steps=600, train_lm=False' in source and 'n_steps=1500, train_lm=True' in source
    assert 'n_steps=400, train_lm=True, use_image=False' in source
    assert 'n_steps=25' in old_tests and 'for train_lm in [False, True]' in old_tests


def test_published_lesson_has_exact_saved_execution_objects_and_actual_figures():
    publication = ROOT / 'docs/mastery/evidence/minivlm_publication_20261019'
    receipt = json.loads((publication / 'publication.json').read_text())
    raw = json.loads((BASE / '12.3_attempt2/recovered_completed.ipynb').read_text())
    canonical = json.loads((SECTION / f'{STEM}_solutions.ipynb').read_text())
    codes = [c for c in canonical['cells'] if c['cell_type'] == 'code']
    saved = [c for c in raw['cells'] if c['cell_type'] == 'code']
    assert len(codes) == 61 and len(saved) == 62
    for current, executed in zip(codes, saved[:61]):
        for field in ['source', 'outputs', 'execution_count', 'metadata']:
            assert current[field] == executed[field]
    assert canonical['metadata']['widgets'] == raw['metadata']['widgets']
    assert canonical['metadata']['minivlm_controls']['dispatch_required'] is False
    assert receipt['exact_source_output_count_cellmetadata_widget_parity'] is True
    assert 'Fresh monitored reference execution' in MASTER.read_text()
    for row in json.loads((BASE / '12.3_attempt2/media_receipt.json').read_text())['files']:
        assert digest((SECTION / 'media' / row['name']).read_bytes()) == row['sha256']
    assert len(list((SECTION / 'media').glob('*.png'))) == 16
    exercises = json.loads((SECTION / f'{STEM}_exercises.ipynb').read_text())
    exercise_codes = [c for c in exercises['cells'] if c['cell_type'] == 'code']
    assert all(c.get('execution_count') is None for c in exercise_codes)
    expected_outputs = [o for c in exercise_codes for o in c.get('outputs', [])]
    assert len(expected_outputs) == 16
    assert all(o['output_type'] == 'display_data' and 'text/html' in o['data']
               for o in expected_outputs)


def test_publication_preserves_frozen_evidence_and_answer_free_proposal():
    publication = ROOT / 'docs/mastery/evidence/minivlm_publication_20261019'
    preserved = json.loads((publication / 'prepublication_evidence_manifest.json').read_text())
    for row in preserved['files']:
        assert digest((ROOT / row['path']).read_bytes()) == row['sha256']
    proposal = json.loads((publication / 'guided_case_proposal.json').read_text())
    assert proposal['precommit_fields_contain_target_answers'] is False
    assert proposal['case']['data_role'] == 'monitored'
    top_level = {n.name for n in ast.parse((SECTION / 'solutions.py').read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert set(proposal['case']['method_functions']) <= top_level
    assert 'prospective' in proposal['case']['experiment']
    assert 'unexecuted' in proposal['case']['transfer']


def test_exact_cached_revisions_are_explicit_in_every_loader():
    source = (SECTION / 'solutions.py').read_text()
    helper = (SECTION / 'utils.py').read_text()
    assert 'revision="607a30d783dfa663caf39e06633721c8d4cfcd7e"' in source
    assert 'revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"' in helper
    assert helper.count('revision="89644892e4d85e24eaac8bacfd4f463576704203"') == 2
    assert 'SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")' in (SECTION / 'tests.py').read_text()
    old = old_bytes((SECTION / 'utils.py').relative_to(ROOT)).decode()
    for pin in ['7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed', '89644892e4d85e24eaac8bacfd4f463576704203']:
        helper = helper.replace(f', revision="{pin}"', '')
    old_tree, new_tree = ast.parse(old), ast.parse(helper)
    old_pca = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'plot_patch_pca')
    new_pca = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'plot_patch_pca')
    # The original explicit-reference implementation remains an exact AST suffix.
    assert ast.dump(ast.Module(body=new_pca.body[-len(old_pca.body[1:]):],type_ignores=[])) == ast.dump(ast.Module(body=old_pca.body[1:],type_ignores=[]))
    new_tree.body[new_tree.body.index(new_pca)] = old_pca
    assert ast.dump(new_tree) == ast.dump(old_tree)


def test_immediate_oracle_fixture():
    ns, tests, _ = namespace()
    tests.test_target_blind_oracle(ns['target_blind_oracle'])


def test_immediate_unique_schedule_fixture():
    ns, tests, _ = namespace()
    tests.test_qwen_layer_schedule(ns['qwen_layer_schedule'])


def test_immediate_spied_readout_fixture():
    ns, tests, _ = namespace()
    tests.test_evaluate_readout_controls(ns['evaluate_readout_controls'])


@pytest.mark.parametrize('sides,expected', [([(0, 0), (1, 0)], 1), ([(0, 0), (1, 1)], .5),
                                          ([(0, 0), (1, 0), (0, 1)], 2/3)])
def test_oracle_enumerates_possible_targets_instead_of_assuming_unique_positions(sides, expected):
    ns, _, utils = namespace()
    scene = utils.Scene([utils.SceneObject(c, s, q) for c, s, q in zip(['red','blue','green'], utils.SHAPES, sides)])
    assert ns['target_blind_oracle'](scene, 'horizontal') == expected
    assert ns['target_blind_oracle'](scene, 'colour') == 1 / len(sides)
    assert ns['target_blind_oracle'](scene, 'shape') == 1 / len(sides)
    assert ns['target_blind_oracle'](scene, 'count') == 1


def test_finite_realized_accuracy_may_exceed_expected_target_blind_oracle():
    ns, _, utils = namespace()
    scene = utils.Scene([utils.SceneObject('red','circle',(0,0)), utils.SceneObject('blue','square',(1,1))])
    # A fixed red answer happens to match the one frozen target. This is not an
    # oracle violation: across BOTH possible targets its mean agreement is .5.
    actual_target = 0
    realized = int(ns['question_and_answer'](scene, 'colour', actual_target)[1] == ' red')
    oracle = ns['target_blind_oracle'](scene, 'colour')
    assert realized == 1 > oracle == .5
    assert sum(ns['question_and_answer'](scene, 'colour', i)[1] == ' red' for i in range(2)) / 2 == oracle


def test_target_metadata_replays_original_kind_major_rng_including_count_draws():
    ns, _, utils = namespace()
    scenes = [utils.sample_scene(np.random.default_rng(seed)) for seed in range(200000, 200019)]
    expected = {}
    rng = np.random.default_rng(0)
    for kind in ['colour','shape','count','horizontal','vertical']:
        expected[kind] = []
        for scene in scenes:
            expected[kind].append(int(rng.integers(0, len(scene.objects))))
    actual = ns['replay_question_targets'](scenes, 0)
    assert actual == expected and actual['colour'] != actual['horizontal']


def test_qwen_literal_image_only_route_does_not_pass_question_target_or_candidates():
    source = (SECTION / 'solutions.py').read_text()
    assert 'utils.qwen_inputs(qwen_processor, q_images[i], "", device)' in source
    assert 'qwen_accuracy["image-only (no question)"] = correct / N_QWEN' in source
    assert 'qwen_layers = qwen_layer_schedule(N_QWEN_LAYERS)' in source
    assert 'denominator <= 0' in source and 'if n_valid == 0:' in source
    assert 'candidate_logits=scores.tolist()' in source
    assert 'not a finite-sample ceiling' in MASTER.read_text()


def test_historical_runner_identity_is_frozen_with_full_recipe_and_budget():
    protocol = json.loads((BASE / 'protocol.json').read_text())
    assert protocol['expected_lesson_code_cells'] == 61 and protocol['expected_evidence_probe_cells'] == 1
    assert protocol['walltime_seconds'] == 1755.411
    assert protocol['aggregate_initial_walltime_seconds'] == 1800
    assert protocol['prior_attempts'][0]['walltime_seconds'] == 44.58876825799962
    assert protocol['walltime_seconds'] + protocol['prior_attempts'][0]['walltime_seconds'] <= 1800
    assert protocol['cell_timeout_seconds'] == 600
    assert protocol['allocation_fraction'] == .90 and protocol['whole_gpu_ceiling_mib'] == 23040
    assert 'dispatch required' in protocol['status']
    assert protocol['candidate_sha256'] == digest((BASE / 'candidate.ipynb').read_bytes())
    assert protocol['runner_sha256'] == digest((BASE / 'runner.py').read_bytes())
    assert protocol['source_manifest_sha256'] == digest((BASE / 'source_manifest.json').read_bytes())
    frozen = json.loads((BASE / 'prelaunch_revisions/v6_executed_manifest.json').read_text())
    archive_path = BASE / 'prelaunch_revisions/v6_executed_before_publication.tar.gz'
    assert digest(archive_path.read_bytes()) == frozen['archive_sha256']
    with tarfile.open(archive_path) as archive:
        for entry in json.loads((BASE / 'source_manifest.json').read_text())['files']:
            assert digest(archive.extractfile(entry['path']).read()) == entry['sha256']
    cache = json.loads((BASE / 'cache_manifest.json').read_text())
    assert {entry['revision'] for entry in cache} == set(protocol['pins'].values())
    assert all(Path(entry['target']).exists() and Path(entry['target']).stat().st_size == entry['bytes'] for entry in cache)
    attempt = BASE / protocol['attempt_directory']
    if attempt.exists():
        assert json.loads((attempt / 'result.json').read_text())['protocol_sha256'] == digest((BASE / 'protocol.json').read_bytes())


@pytest.mark.parametrize('explicit_reference', [False, True])
def test_patch_pca_exact_fit_input_display_rows_and_no_mutation(monkeypatch, explicit_reference):
    import matplotlib.pyplot as plt
    ns, _, utils = namespace()
    images = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    features = t.sin(t.arange(4 * 196 * 8, dtype=t.float32) * .013).reshape(4, 196, 8)
    reference = t.cos(t.arange(3 * 196 * 8, dtype=t.float32) * .029).reshape(3, 196, 8)
    image_before, feature_before, reference_before = images.copy(), features.clone(), reference.clone()
    calls, figures = [], []
    def exact_svd(matrix, q, center):
        calls.append((matrix.clone(), q, center))
        u, values, vh = t.linalg.svd(matrix, full_matrices=False)
        return u[:, :q], values[:q], vh.T[:, :q]
    monkeypatch.setattr(t, 'pca_lowrank', exact_svd)
    monkeypatch.setattr(plt, 'show', lambda: figures.append(plt.gcf()))
    try:
        if explicit_reference:
            utils.plot_patch_pca(images, features, reference)
        else:
            utils.plot_patch_pca(images, features)
        assert len(figures) == 1 and len(figures[0].axes) == (3 if explicit_reference else 2) * 4
        expected_ref = reference.numpy() if explicit_reference else features.numpy()
        expected_matrices = [expected_ref.reshape(-1, 8)]
        if explicit_reference:
            centered_positions = expected_ref - expected_ref.mean(axis=0)
            expected_matrices.append(centered_positions.reshape(-1, 8))
        assert len(calls) == len(expected_matrices)
        for (actual, q, center), raw in zip(calls, expected_matrices):
            # Independent NumPy centering, all and only designated reference rows.
            np.testing.assert_allclose(actual.numpy(), raw - raw.mean(axis=0), atol=2e-7)
            assert q == 3 and center is False
        assert all(np.isfinite(np.asarray(ax.images[0].get_array())).all() for ax in figures[0].axes)
        np.testing.assert_array_equal(images, image_before)
        t.testing.assert_close(features, feature_before, atol=0, rtol=0)
        t.testing.assert_close(reference, reference_before, atol=0, rtol=0)
    finally:
        for figure in figures:plt.close(figure)


def test_patch_pca_two_argument_constant_features_remain_finite_and_non_tensor_rejects(monkeypatch):
    import matplotlib.pyplot as plt
    _, _, utils = namespace()
    figures = []
    monkeypatch.setattr(plt, 'show', lambda: figures.append(plt.gcf()))
    try:
        utils.plot_patch_pca(np.zeros((1,224,224,3),dtype=np.uint8), t.zeros(1,196,4))
        assert all(np.isfinite(np.asarray(ax.images[0].get_array())).all() for ax in figures[0].axes)
        with pytest.raises(TypeError, match='torch Tensor'):
            utils.plot_patch_pca(np.zeros((1,224,224,3),dtype=np.uint8), np.zeros((1,196,4)))
    finally:
        for figure in figures:plt.close(figure)
