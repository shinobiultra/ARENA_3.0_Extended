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
             'evaluate_readout_controls', 'qwen_layer_schedule', 'logit_diff_recovery', 'top_attention_mask'}
    assignments = {'KINDS', 'NUMBER_WORDS', 'ANSWER_WORDS'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
             or isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in assignments for x in n.targets)]
    source = 'from __future__ import annotations\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    ns = dict(t=t, np=np, Counter=Counter, utils=utils, Scene=utils.Scene, COLOURS=utils.COLOURS, SHAPES=utils.SHAPES)
    ns['Float'] = ns['Int'] = ns['Bool'] = ns['Tensor'] = ns['Callable'] = ns['nn'] = ns['F'] = ns['einops'] = ns['HookedTransformer'] = object
    exec(compile(source, '<visible MiniVLM control definitions only>', 'exec'), ns)
    test_tree = ast.parse((SECTION / 'tests.py').read_text())
    test_nodes = [node for node in test_tree.body if isinstance(node, ast.FunctionDef)
                  and node.name in {'test_target_blind_oracle', 'test_qwen_layer_schedule', 'test_evaluate_readout_controls',
                                                           'test_logit_diff_recovery', 'test_top_attention_mask'}]
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


# ---------------------------------------------------------------------------------------------
# Whole-cohort guard in logit_diff_recovery, the highest-attention mask, the layer schedule, the bootstrap band.
# Each deliberately wrong variant must fail the lesson's own test.
# ---------------------------------------------------------------------------------------------


def _fails(test, wrong):
    try:
        test(wrong)
    except (AssertionError, ValueError, TypeError, IndexError, RuntimeError):
        return True
    return False


def _recovery_variants():
    def ld(logits, clean_ids, corrupt_ids):
        rows = t.arange(len(logits))
        return logits[rows, clean_ids] - logits[rows, corrupt_ids]

    def raw(patched, clean, corrupt, clean_ids, corrupt_ids):
        gap = ld(clean, clean_ids, corrupt_ids) - ld(corrupt, clean_ids, corrupt_ids)
        return (ld(patched, clean_ids, corrupt_ids) - ld(corrupt, clean_ids, corrupt_ids)) / gap, gap

    def no_guard(*args):
        return raw(*args)[0]

    def drops_bad_pairs(*args):
        recovery, gap = raw(*args)
        return recovery[gap > 0]

    def zeroes_bad_pairs(*args):
        recovery, gap = raw(*args)
        return t.where(gap > 0, recovery, t.zeros_like(recovery))

    def only_reversed(*args):
        recovery, gap = raw(*args)
        if (gap < 0).any():
            raise ValueError("reversed")
        return recovery

    def rejects_tiny_gaps(*args):
        recovery, gap = raw(*args)
        if not (gap > 1e-3).all() or not t.isfinite(recovery).all():
            raise ValueError("bad gap")
        return recovery

    def clips_recovery(*args):
        recovery, gap = raw(*args)
        if not (gap > 0).all() or not t.isfinite(recovery).all():
            raise ValueError("bad gap")
        return recovery.clamp(0, 1)

    def gap_only(*args):
        recovery, gap = raw(*args)
        if not (gap > 0).all():
            raise ValueError("bad gap")
        return recovery  # a nonfinite patched run slips through

    return dict(no_guard=no_guard, drops_bad_pairs=drops_bad_pairs, zeroes_bad_pairs=zeroes_bad_pairs,
                only_reversed=only_reversed, rejects_tiny_gaps=rejects_tiny_gaps, clips_recovery=clips_recovery,
                gap_only=gap_only)


def test_recovery_guard_accepts_the_solution_and_rejects_every_wrong_variant():
    ns, tests, _ = namespace()
    tests.test_logit_diff_recovery(ns['logit_diff_recovery'])
    for name, wrong in _recovery_variants().items():
        assert _fails(tests.test_logit_diff_recovery, wrong), f"wrong variant {name} passed the test"


def test_recovery_guard_stops_the_cohort_on_pair_7_gap_4_pair_9_gap_0():
    ns, _, _ = namespace()
    n = 10
    gaps = t.tensor([2.0, 3.0, 5.0, 1.0, 6.0, 2.5, 3.5, 4.0, 2.0, 0.0])
    clean, corrupt = t.zeros(n, 2), t.zeros(n, 2)
    clean[:, 0] = gaps
    ids0, ids1 = t.zeros(n, dtype=t.long), t.ones(n, dtype=t.long)
    with pytest.raises(ValueError):
        ns['logit_diff_recovery'](clean, clean, corrupt, ids0, ids1)
    # without pair 9 the cohort is fine, and a pair at exactly gap 4 is not what stopped it
    ok = slice(0, 9)
    out = ns['logit_diff_recovery'](clean[ok], clean[ok], corrupt[ok], ids0[ok], ids1[ok])
    t.testing.assert_close(out, t.ones(9))


def test_top_attention_mask_matches_hand_built_pattern_and_rejects_wrong_variants():
    ns, tests, _ = namespace()
    tests.test_top_attention_mask(ns['top_attention_mask'])

    def means(patterns, query_pos, image_slice):
        return patterns[:, :, query_pos, image_slice].mean(dim=(0, 1))

    def pick(scores, k, largest=True):
        out = t.zeros(len(scores), dtype=t.bool)
        out[scores.topk(k, largest=largest).indices] = True
        return out

    wrong = {
        "lowest": lambda p, q, s, k: pick(means(p, q, s), k, largest=False),
        "layer 0 only": lambda p, q, s, k: pick(p[0, :, q, s].mean(0), k),
        "last layer only": lambda p, q, s, k: pick(p[-1, :, q, s].mean(0), k),
        "wrong query": lambda p, q, s, k: pick(means(p, q - 1, s), k),
        "one too many": lambda p, q, s, k: pick(means(p, q, s), min(k + 1, 5)),
        "BOS competes": lambda p, q, s, k: pick(p[:, :, q, : s.stop].mean(dim=(0, 1)), k)[s.start :],
    }
    for name, fn in wrong.items():
        assert _fails(tests.test_top_attention_mask, fn), f"wrong variant {name} passed the test"


def test_layer_schedule_wrong_variants_fail():
    ns, tests, _ = namespace()
    duplicates_last = lambda n: [*range(0, n, 3), n - 1]  # the old bug: the final layer listed twice when n - 1 is a multiple of 3
    drops_last = lambda n: list(range(0, n, 3))
    every_layer = lambda n: list(range(n))
    for name, fn in dict(duplicates_last=duplicates_last, drops_last=drops_last, every_layer=every_layer).items():
        assert _fails(tests.test_qwen_layer_schedule, fn), f"wrong variant {name} passed the test"


def test_target_blind_oracle_hard_coded_uniform_fails():
    ns, tests, _ = namespace()
    assert _fails(tests.test_target_blind_oracle, lambda scene, kind: 1 / len(scene.objects))


def test_pair_bootstrap_band_is_a_pair_resampling_band():
    _, _, utils = namespace()
    band = utils.pair_bootstrap_band
    # two pairs with recovery 0 and 1: resampled means are 0, 1/2 or 1 with probabilities 1/4, 1/2, 1/4
    mean, low, high = band(t.tensor([0.0, 1.0]), n_boot=4000, seed=0)
    assert (float(mean), float(low), float(high)) == (0.5, 0.0, 1.0)
    # constant recovery has no uncertainty
    mean, low, high = band(t.full((20, 3), 0.7), n_boot=500, seed=1)
    np.testing.assert_allclose([mean.numpy(), low.numpy(), high.numpy()], 0.7)
    # pairs are resampled as whole rows: layers that move together keep moving together
    base = t.rand(30, 1, generator=t.Generator().manual_seed(0))
    mean, low, high = band(t.cat([base, 3 * base], dim=1), n_boot=500, seed=2)
    np.testing.assert_allclose(high[1].numpy(), 3 * high[0].numpy(), rtol=1e-5)
    np.testing.assert_allclose(low[1].numpy(), 3 * low[0].numpy(), rtol=1e-5)
    # the band is around the plain mean over pairs, and shrinks as pairs are added
    values = t.randn(400, generator=t.Generator().manual_seed(3)) * 0.2 + 0.5
    mean, low, high = band(values, n_boot=2000, seed=0)
    assert abs(float(mean) - float(values.mean())) < 1e-6 and float(low) < float(mean) < float(high)
    _, low_small, high_small = band(values[:20], n_boot=2000, seed=0)
    assert float(high_small - low_small) > 2 * float(high - low)
    # the same seed gives the same band
    assert t.equal(band(values, n_boot=300, seed=5)[1], band(values, n_boot=300, seed=5)[1])
