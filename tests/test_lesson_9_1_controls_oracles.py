"""9.1 Refusal directions: CPU fixtures for the shared control set and its summary, no model loads."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch as t
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / 'infrastructure/chapters/chapter9_alignment_interpretability/master_9_1.py'
CPU = t.device('cpu')


def master_function(name, replace=None, helpers=('random_unit_vectors', 'match_norm', 'shuffled_label_direction')):
    """Compile one master function (with the helpers it calls) without importing the master."""
    tree = ast.parse(MASTER.read_text())
    defs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    namespace = {'t': t, 'F': F, 'np': np, 'device': CPU}
    for fn_name in (*helpers, name):
        source = ast.unparse(ast.Module(body=[defs[fn_name]], type_ignores=[]))
        if replace and fn_name == name:
            old, new = replace
            assert old in source, f'{old!r} not found in {name}; update this wrong variant'
            source = source.replace(old, new)
        exec(compile('from __future__ import annotations\n' + source, str(MASTER), 'exec'), namespace)
    return namespace[name]


def toy_activations(seed=0, n=40, d=64):
    generator = t.Generator().manual_seed(seed)
    axis = t.randn(d, generator=generator)
    axis /= axis.norm()
    pos = t.randn(n, d, generator=generator) + 4.0 * axis
    neg = t.randn(n, d, generator=generator)
    return pos, neg, axis


def test_one_draw_set_has_six_random_and_six_shuffled_directions_of_the_reference_length():
    draw_controls = master_function('draw_controls')
    pos, neg, _ = toy_activations()
    reference = t.randn(64) * 3
    controls = draw_controls(pos, neg, reference)
    assert set(controls) == {'random direction', 'label-shuffled direction'}
    for vectors in controls.values():
        assert len(vectors) == 6
        for v in vectors:
            assert v.shape == (64,)
            assert abs(v.norm().item() - reference.norm().item()) < 1e-4


def test_the_same_call_gives_the_same_draws():
    """The signature, the side-effects table and the Part 5 table must see identical vectors."""
    draw_controls = master_function('draw_controls')
    pos, neg, _ = toy_activations()
    reference = t.randn(64)
    first, second = draw_controls(pos, neg, reference), draw_controls(pos, neg, reference)
    for name in first:
        for a, b in zip(first[name], second[name]):
            t.testing.assert_close(a, b)
    assert not t.allclose(first['random direction'][0], first['random direction'][1])
    assert not t.allclose(first['label-shuffled direction'][0], first['label-shuffled direction'][1])


def test_shuffled_draws_are_stratified():
    """A constant shift of the positive class cancels only if every draw puts half of each class in each group."""
    draw_controls = master_function('draw_controls')
    pos, neg, axis = toy_activations()
    reference = t.ones(64)
    base = draw_controls(pos, neg, reference)['label-shuffled direction']
    shifted = draw_controls(pos + 100.0 * axis, neg, reference)['label-shuffled direction']
    for a, b in zip(base, shifted):
        assert F.cosine_similarity(a, b, dim=0) > 0.999


def test_a_naive_shuffle_variant_fails_the_stratification_check():
    wrong = master_function('draw_controls', replace=('stratified=True', 'stratified=False'))
    pos, neg, axis = toy_activations()
    reference = t.ones(64)
    base = wrong(pos, neg, reference)['label-shuffled direction']
    shifted = wrong(pos + 100.0 * axis, neg, reference)['label-shuffled direction']
    cosines = [F.cosine_similarity(a, b, dim=0).item() for a, b in zip(base, shifted)]
    assert min(cosines) < 0.9, 'a naive shuffle must not survive the shift check'


def test_the_random_draws_do_not_depend_on_the_labels():
    draw_controls = master_function('draw_controls')
    pos, neg, _ = toy_activations()
    reference = t.ones(64)
    a = draw_controls(pos, neg, reference)['random direction']
    b = draw_controls(neg, pos, reference)['random direction']
    for x, y in zip(a, b):
        t.testing.assert_close(x, y)


def test_mean_and_range_hand_worked():
    mean_and_range = master_function('mean_and_range', helpers=())
    draws = [{'refusal': 0.0, 'loss': 3.0}, {'refusal': 0.5, 'loss': 3.5}, {'refusal': 1.0, 'loss': 5.5}]
    out = mean_and_range(draws)
    assert out['refusal'] == pytest.approx((0.5, 0.0, 1.0))
    assert out['loss'] == pytest.approx((4.0, 3.0, 5.5))


def test_mean_and_range_uses_the_mean_not_the_median_or_midrange():
    mean_and_range = master_function('mean_and_range', helpers=())
    out = mean_and_range([{'x': 0.0}, {'x': 0.0}, {'x': 9.0}])
    assert out['x'][0] == pytest.approx(3.0)


def test_a_median_variant_fails_the_mean_check():
    wrong = master_function('mean_and_range', replace=('np.mean', 'np.median'), helpers=())
    out = wrong([{'x': 0.0}, {'x': 0.0}, {'x': 9.0}])
    assert out['x'][0] != pytest.approx(3.0)
