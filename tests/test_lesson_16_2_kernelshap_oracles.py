"""16.2 KernelSHAP and PartitionSHAP: brute-force permutation and Owen-ordering oracles on finite games."""
from __future__ import annotations

import importlib
import itertools
import math

import numpy as np
import pytest

MODULE = 'chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partitionshap'


def permutation_oracle(values, players):
    result = np.zeros(players)
    for order in itertools.permutations(range(players)):
        before = frozenset()
        for player in order:
            after = before | {player}
            result[player] += values[after] - values[before]
            before = after
    return result / math.factorial(players)


def owen_ordering_oracle(values, groups):
    credits, count = np.zeros(sum(map(len, groups))), 0
    for group_order in itertools.permutations(range(len(groups))):
        for within in itertools.product(*(list(itertools.permutations(groups[i])) for i in group_order)):
            count += 1
            before = frozenset()
            for player in itertools.chain.from_iterable(within):
                after = before | {player}
                credits[player] += values[after] - values[before]
                before = after
    return credits / count


ORACLES = ['dividends', 'finite_kernel', 'rows_and_baseline', 'full_kernel', 'partial_kernel', 'rank_deficient', 'sampling', 'owen', 'partition_validation', 'group_expansion', 'shuffle_efficiency', 'live_play']


@pytest.mark.parametrize('oracle', ORACLES)
def test_visible_methods_against_independent_cpu_oracles(oracle):
    torch = importlib.import_module('torch')
    m = importlib.import_module(MODULE + '.solutions')
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        dividends = {frozenset({0}): 1.0, frozenset({1}): -.5, frozenset({2}): .25, frozenset({0, 1, 2}): 1.5}
        values = m.game_from_dividends(3, dividends)
        expected = torch.tensor(permutation_oracle(values, 3), dtype=torch.float64)
        if oracle == 'dividends':
            torch.testing.assert_close(m.shapley_from_dividends(3, dividends), expected)
            torch.testing.assert_close(m.exact_shapley_values(values, num_players=3), expected)
            assert len(values) == 8 and values[frozenset()] == 0
        elif oracle == 'finite_kernel':
            for size in range(1, 6):
                assert math.isclose(m.kernelshap_kernel_weight(size, 6), 5 / (math.comb(6, size) * size * (6-size)))
            for size in (0, 6):
                with pytest.raises(ValueError):
                    m.kernelshap_kernel_weight(size, 6)
        elif oracle == 'rows_and_baseline':
            shifted = {key: value + 7 for key, value in values.items()}
            x, y, w = m.kernelshap_regression_rows(shifted, num_players=3)
            assert x.shape == (6, 3) and y.shape == w.shape == (6,)
            for row, coalition in enumerate(m.all_coalitions(3)[1:-1]):
                assert x[row].tolist() == [float(i in coalition) for i in range(3)]
                assert y[row] == values[coalition]
            torch.testing.assert_close(m.kernelshap_values(shifted, num_players=3), expected)
        elif oracle == 'full_kernel':
            torch.testing.assert_close(m.kernelshap_values(values, num_players=3), expected, atol=1e-12, rtol=0)
        elif oracle == 'partial_kernel':
            sample = m.all_coalitions(3)[1:-2]
            x, y, w = m.kernelshap_regression_rows(values, num_players=3, coalitions=sample)
            x, y, w = x.numpy(), y.numpy(), w.numpy()
            basis = np.vstack([np.eye(2), -np.ones(2)])
            particular = np.full(3, values[frozenset(range(3))] / 3)
            theta = np.linalg.lstsq(np.sqrt(w)[:, None] * (x @ basis), np.sqrt(w) * (y - x @ particular), rcond=None)[0]
            result = m.kernelshap_values(values, num_players=3, coalitions=sample)
            np.testing.assert_allclose(result.numpy(), particular + basis @ theta, atol=1e-12)
        elif oracle == 'rank_deficient':
            game = m.additive_game(torch.tensor([1., 2., 3.]))
            previous_threads = torch.get_num_threads()
            try:
                for threads in (24, 1, 2):
                    torch.set_num_threads(threads)
                    for _ in range(5):
                        estimate = m.kernelshap_values(game, num_players=3, coalitions=(frozenset({0}),))
                        torch.testing.assert_close(estimate, torch.tensor([1., 2.5, 2.5], dtype=torch.float64), atol=1e-12, rtol=0)
            finally:
                torch.set_num_threads(previous_threads)
        elif oracle == 'sampling':
            for budget in (12, 24, 62):
                for strategy in ('paired_kernel', 'uniform'):
                    sample = m.sample_interior_coalitions(6, budget=budget, seed=4, strategy=strategy)
                    assert len(sample) == len(set(sample)) == budget
                    assert sample == m.sample_interior_coalitions(6, budget=budget, seed=4, strategy=strategy)
                    if strategy == 'paired_kernel':
                        assert all((frozenset(range(6)) - c) in sample for c in sample)
            with pytest.raises(ValueError):
                m.sample_interior_coalitions(6, budget=13, seed=0, strategy='paired_kernel')
        elif oracle == 'owen':
            six = m.game_from_dividends(6, m.GAME_DIVIDENDS)
            for groups in (((0,1,2),(3,4,5)), ((0,3,4),(1,2,5)), ((0,1),(2,3),(4,5))):
                np.testing.assert_allclose(m.partition_shap_values(six, groups=groups).numpy(), owen_ordering_oracle(six, groups), atol=1e-12)
        elif oracle == 'partition_validation':
            for groups in (((0,1),(1,2)), ((0,1),), ((),(0,1,2)), ((0,1),(3,))):
                with pytest.raises(ValueError):
                    m.validate_partition(groups, 3)
        elif oracle == 'group_expansion':
            game = m.additive_game(torch.tensor([1.,2.,3.,4.]))
            result = m.grouped_coalition_values(game, groups=((0,1),(2,3)))
            assert result == {frozenset():0.,frozenset({0}):3.,frozenset({1}):7.,frozenset({0,1}):10.}
        elif oracle == 'shuffle_efficiency':
            interior = m.all_coalitions(3)[1:-1]
            shuffled = dict(values)
            for coalition, value in zip(interior, [values[c] for c in interior][1:] + [values[interior[0]]], strict=True):
                shuffled[coalition] = value
            estimate = m.kernelshap_values(shuffled, num_players=3)
            assert abs(float(estimate.sum()) - values[frozenset(range(3))]) < 1e-12
            assert float((estimate-expected).abs().max()) > .1
        elif oracle == 'live_play':
            for budget, seed, groups in itertools.product((12,24,62), (0,4), (((0,1,2),(3,4,5)),((0,3,4),(1,2,5)))):
                result = m.compare_kernelshap_play(dividends=m.GAME_DIVIDENDS, budget=budget, seed=seed, groups=groups)
                m.check_kernelshap_play(result)
                np.testing.assert_allclose(result['oracle'].numpy(), permutation_oracle(result['values'],6), atol=1e-12)
    finally:
        torch.set_default_dtype(previous)
