"""Primary-paper finite-penalty and actual package checks; analytic games only."""

import math

import numpy as np
import pytest
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods as m


@pytest.mark.parametrize('n',[4,6,8])
def test_finite_penalty_limit(n):
    generator=t.Generator().manual_seed(161720+n)
    values=t.randn(2**n,generator=generator,dtype=t.float64)
    values=values-values[0].clone()
    masks=t.arange(2**n)
    sizes=m.coalition_design(masks,n_players=n).sum(-1).long()
    single=m.exact_sii(values,n_players=n,order=1)
    target=m.exact_sii(values,n_players=n,order=2)
    residual=values-m.coalition_design(masks,n_players=n)@single
    design=m.pair_residual_design(masks,n_players=n)
    finite=t.tensor([1/math.comb(n-4,int(k)-2) if 2<=k<=n-2 else 0 for k in sizes],dtype=t.float64)
    errors=[]
    for penalty in [1e2,1e4,1e6,1e8]:
        weights=t.where((sizes<2)|(sizes>n-2),penalty,finite)
        weighted=weights.sqrt()[:,None]*design
        fit=t.linalg.lstsq(weighted,(weights.sqrt()*residual)[:,None],driver='gelsd').solution[:,0]
        errors.append(float((fit-target).abs().max()))
    assert errors[-1]<1e-6*max(1.,float(values.abs().max())), errors
    assert errors[-1]<errors[0]/1000, errors


@pytest.mark.parametrize('n',[4,6,8])
def test_actual_shapiq_package_exact_and_full_budget(n):
    import shapiq
    from importlib.metadata import version
    assert version('shapiq')=='1.5.2'
    generator=t.Generator().manual_seed(161730+n)
    games=t.randn((3,2**n),generator=generator,dtype=t.float64)
    games[0]=4.5
    games[1]=0;games[1,-1]=1
    terms=m.interaction_masks(n,2).tolist()
    expected=m.exact_sii(games,n_players=n,order=2)
    for row,values in enumerate(games):
        original=values.clone();queried=[]
        def game(coalitions):
            integers=(np.asarray(coalitions,dtype=np.int64)*(1<<np.arange(n))).sum(axis=-1)
            queried.append(integers.copy())
            return values[t.from_numpy(integers.copy())].numpy().copy()
        computers=[shapiq.ExactComputer(game=game,n_players=n),shapiq.SHAPIQ(n,index='SII',max_order=2,random_state=161740),shapiq.KernelSHAPIQ(n,index='SII',max_order=2,random_state=161740)]
        for index,computer in enumerate(computers):
            output=computer(index='SII',order=2) if index==0 else computer.approximate(budget=2**n,game=game)
            result=t.tensor([output[tuple(i for i in range(n) if mask&(1<<i))] for mask in terms],dtype=t.float64)
            bound=(1e-12 if index<2 else 1e-6)*max(1.,float(values.abs().max()))
            t.testing.assert_close(result,expected[row],rtol=0,atol=bound)
        t.testing.assert_close(values,original,rtol=0,atol=0)
        assert set(np.concatenate(queried).tolist())==set(range(2**n))
