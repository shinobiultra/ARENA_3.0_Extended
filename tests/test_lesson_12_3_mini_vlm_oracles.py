"""12.3 Mini VLM from scratch: CPU oracles for the visible control definitions and patch-PCA helper, no model loads."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest
import torch as t

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / 'chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch'


@pytest.fixture(autouse=True)
def restore_cpu_numeric_state():
    dtype, rng = t.get_default_dtype(), t.random.get_rng_state()
    try:
        yield
    finally:
        t.set_default_dtype(dtype)
        t.random.set_rng_state(rng)


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
