"""Coverage and algebraic checks derived from the complete retained native arrays."""

from __future__ import annotations

import json
from pathlib import Path

import torch as t

from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap import gradientshap_study as known
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods as m
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.estimator_study import MODEL_KEYS, PROPOSALS, BUDGETS, load_manifest, recipe_weights
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.reproduce_native import PAYLOADS


def load_payloads(directory):
    return {name: t.load(Path(directory) / f'{name}.pt', map_location='cpu', weights_only=False) for name in PAYLOADS}


def _shape(value, shape, name):
    if not isinstance(value, t.Tensor) or tuple(value.shape) != shape or not bool(t.isfinite(value).all()):
        raise ValueError('Missing, incomplete or nonfinite array: ' + name)
    return value


def study_metrics(payload):
    t.set_num_threads(4)
    fitted, games, plans = (payload[key] for key in ('fitted', 'games', 'plans'))
    manifest = load_manifest()
    reproduced = known.verify_16_14_reproduction(fitted, manifest)
    if tuple(fitted['models']) != MODEL_KEYS or tuple(games['models']) != MODEL_KEYS:
        raise ValueError('The complete known seven-model study is required.')
    expected_ids = manifest['splits']['explain'] + manifest['splits']['audit']
    if games['source_ids'].tolist() != expected_ids:
        raise ValueError('Held-out source row identities changed.')
    values = _shape(games['values'], (7,272,256), 'games')
    hybrid = _shape(games['hybrid_predictions'], (7,272,256,64), 'donor predictions')
    t.testing.assert_close(values, hybrid.mean(-1), rtol=0, atol=0)
    exact = _shape(payload['exact']['pairs'], (7,272,28), 'exact pairs')
    t.testing.assert_close(exact, m.exact_sii(values, n_players=8), rtol=0, atol=0)
    boundary = m.boundary_masks(8)
    terms = m.interaction_masks(8)
    incidence = m.coalition_design(terms, n_players=8).T
    singletons = 1 << t.arange(8)
    pair_rhs = values[...,-1:] + values[...,:1] - values[...,singletons] - values[...,255-singletons]
    first_rhs = values[...,-1:] - values[...,:1]
    residual_bound = 1e-10 + 1e-12 * max(1., float(values.abs().max()))
    metrics = {'pair_constraint_error': 0., 'first_constraint_error': 0.}
    errors = {}
    for proposal in PROPOSALS:
        plan = plans[proposal]
        _shape(plan['masks'], (32,1024), 'masks')
        _shape(plan['probabilities'], (32,1024), 'probabilities')
        if plan['seeds'].tolist() != manifest['replicate_seeds']:
            raise ValueError('Replicate seeds changed.')
        block = payload['estimates'][proposal]
        for name in ('shapiq', 'standard_error', 'kernelshap_iq'):
            _shape(block[name], (32,7,7,272,28), proposal + '.' + name)
        _shape(block['kernelshap_singletons'], (32,7,7,272,8), 'first stage')
        _shape(block['pair_singular_values'], (32,7,20), 'pair singular values')
        if bool((block['standard_error'] < 0).any()):
            raise ValueError('Negative standard error.')
        pair_residual = block['kernelshap_iq'] @ incidence.T - pair_rhs
        first_residual = block['kernelshap_singletons'].sum(-1, keepdim=True) - first_rhs
        # Stored solver residuals are independently checked against the game boundaries.
        t.testing.assert_close(pair_residual, block['pair_constraint_residual'], rtol=0, atol=residual_bound)
        t.testing.assert_close(first_residual, block['first_constraint_residual'], rtol=0, atol=residual_bound)
        for key, residual in [('pair_constraint_error', pair_residual), ('first_constraint_error', first_residual)]:
            metrics[key] = max(metrics[key], float(residual.abs().max()))
            if metrics[key] > residual_bound:
                raise ValueError('Frozen equality tolerance exceeded: ' + key)
        for replicate, seed in enumerate(manifest['replicate_seeds']):
            masks, probability = m.sample_coalitions(n_players=8, draws=1024, seed=seed, proposal=proposal)
            t.testing.assert_close(masks, plan['masks'][replicate], rtol=0, atol=0)
            t.testing.assert_close(probability, plan['probabilities'][replicate], rtol=0, atol=0)
            for bi, draws in enumerate(BUDGETS):
                queried = t.unique(t.cat((boundary, masks[:draws])), sorted=True)
                t.testing.assert_close(queried, block['queried_masks'][replicate][bi], rtol=0, atol=0)
                if len(queried) != int(block['unique_queries'][replicate,bi]):
                    raise ValueError('Unique query count disagrees with actual retained masks.')
        errors[proposal] = {}
        for method in ('shapiq', 'kernelshap_iq'):
            errors[proposal][method] = {}
            for split, selected in [('explain', slice(0,153)), ('audit', slice(153,None))]:
                weights = recipe_weights(games['recipe_group_ids'][selected])
                per_replicate = ((block[method][...,selected,:] - exact[:,selected,:]).abs().mean(-1) * weights).sum(-1)
                errors[proposal][method][split] = per_replicate.mean(0).tolist()
    package = payload['package']
    actual = _shape(package['values'], (7,3,28), 'actual author-package values')
    index = expected_ids.index(0)
    t.testing.assert_close(package['expected'], exact[:,index], rtol=0, atol=0)
    t.testing.assert_close(package['absolute_error'], (actual-exact[:,index,None]).abs(), rtol=0, atol=0)
    for model in range(7):
        for kind in range(3):
            masks = package['actual_queried_masks'][model][kind]
            if set(masks.tolist()) != set(range(256)) or len(masks) != int(package['queried_coalitions'][model,kind]):
                raise ValueError('Incomplete author-package oracle receipt.')
            scale = max(1., float(values[model,index].abs().max()))
            bound = 1e-10+1e-12*scale if kind<2 else 1e-6*scale
            t.testing.assert_close(actual[model,kind], exact[model,index], rtol=0, atol=bound)
    return {
        'study': '16.17_shapiq_estimators', 'model_fits': len(fitted['models']),
        'known_model_state_hashes_matched': sum(len(v) for v in reproduced['model_state_hashes'].values()),
        'train_rows': len(fitted['train_ids']), 'explanation_rows':153, 'audit_rows':119,
        'model_games': values.shape[0]*values.shape[1], 'coalition_values': values.numel(),
        'donor_predictions': hybrid.numel(), 'replicates_per_proposal':32,
        'proposals': list(PROPOSALS), 'nested_draw_budgets':list(BUDGETS),
        'signed_pair_estimates': sum(payload['estimates'][p][name].numel() for p in PROPOSALS for name in ('shapiq','kernelshap_iq')),
        'author_reference_version': package['version'], 'author_reference_games':7,
        'author_reference_algorithms':3, 'author_reference_passed':True,
        'logical_query_benchmark':games['query_costs_are_logical'], 'precomputed_oracle':games['oracle_precomputed'],
        'fresh_predictive_discovery':False, 'method_superiority_is_acceptance_gate':False,
        'cross_split_recipe_overlap':manifest['cross_split_recipe_overlap'],
        'recipe_weighted_pair_mae_by_budget_and_model':errors, **metrics,
    }
