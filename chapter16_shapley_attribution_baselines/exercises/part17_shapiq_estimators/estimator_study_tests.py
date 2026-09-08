"""No-fit checks of oracle isolation, paired sampling and real-input construction."""

import torch as t

from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods as m
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_study as s


def test_intact_donor_predictions():
    class Additive(t.nn.Module):
        def forward(self,x): return (x*t.arange(1,9,dtype=t.float64)).sum(-1,keepdim=True)+2
    query=t.arange(16,dtype=t.float64).reshape(2,8)
    donors=t.tensor([[0.]*8,[1.]*8,[-2.]*8],dtype=t.float64)
    masks=t.tensor([0,255,3,12,3],dtype=t.int64)
    actual=s.donor_predictions(Additive(),query,donors,masks,design_fn=m.coalition_design)
    expected=t.empty_like(actual)
    for row in range(2):
        for index,mask in enumerate(masks.tolist()):
            for donor in range(3):
                hybrid=t.tensor([query[row,i] if mask&(1<<i) else donors[donor,i] for i in range(8)],dtype=t.float64)
                expected[row,index,donor]=Additive()(hybrid).squeeze()
    t.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_group_weights_preserve_duplicate_rows():
    w=s.recipe_weights(t.tensor([1,1,1,2,3,3]))
    t.testing.assert_close(w,t.tensor([1/9,1/9,1/9,1/3,1/6,1/6],dtype=t.float64),rtol=0,atol=1e-15)


def test_paired_study_no_unqueried_values(monkeypatch):
    # Reduced analytic test workload; canonical constants and manifest stay unchanged.
    monkeypatch.setattr(s,'REPLICATES',2);monkeypatch.setattr(s,'BUDGETS',(16,32))
    plans={}
    for proposal in s.PROPOSALS:
        records=[m.sample_coalitions(n_players=8,draws=32,seed=161750+r,proposal=proposal) for r in range(2)]
        plans[proposal]={'masks':t.stack([x[0] for x in records]),'probabilities':t.stack([x[1] for x in records])}
    masks=t.arange(256);z=m.coalition_design(masks,n_players=8)
    values=(3+z@t.arange(1,9,dtype=t.float64))[None,None].expand(7,272,256).clone()
    results=s.estimate_all_games(values,plans,m)
    for proposal,record in results.items():
        assert record['shapiq'].shape==(2,2,7,272,28)
        assert bool(t.isfinite(record['standard_error']).all())
        t.testing.assert_close(record['kernelshap_iq'],t.zeros_like(record['kernelshap_iq']),rtol=0,atol=1e-10)
        for r in range(2):
            for b,k in enumerate((16,32)):
                expected=set(m.boundary_masks(8).tolist())|set(plans[proposal]['masks'][r,:k].tolist())
                assert set(record['queried_masks'][r][b].tolist())==expected
                assert int(record['unique_queries'][r,b])==len(expected)
                # Alter every unseen coalition. Neither estimator may change.
                altered=values[0,0].clone()
                altered[t.tensor([i for i in range(256) if i not in expected])]+=1000
                draw=plans[proposal]['masks'][r,:k];p=plans[proposal]['probabilities'][r,:k]
                receipt=m.query_sampled_game(lambda requested:altered[requested],draw,n_players=8)
                point,se=m.shapiq_estimate(receipt['boundary_values'],receipt['sampled_values'],draw,p,n_players=8)
                kiq=m.kernelshap_iq(receipt['boundary_values'],receipt['sampled_values'],draw,p,n_players=8)
                t.testing.assert_close(point,record['shapiq'][r,b,0,0],rtol=0,atol=1e-12)
                t.testing.assert_close(se,record['standard_error'][r,b,0,0],rtol=0,atol=1e-12)
                t.testing.assert_close(kiq['pairs'],record['kernelshap_iq'][r,b,0,0],rtol=0,atol=1e-10)


def test_variance_law_constant_and_scaling():
    generator=t.Generator().manual_seed(42)
    v=t.randn((7,272,256),generator=generator,dtype=t.float64)
    v[0]=3;v[2]=2*v[1]+9
    result=s.exact_sampling_variance(v)
    for proposal,record in result.items():
        assert record['one_draw_variance'].shape==(7,272,28)
        t.testing.assert_close(record['probabilities'].sum(),t.tensor(1.,dtype=t.float64),rtol=0,atol=1e-14)
        t.testing.assert_close(record['one_draw_variance'][0],t.zeros((272,28),dtype=t.float64),rtol=0,atol=0)
        t.testing.assert_close(record['one_draw_variance'][2],4*record['one_draw_variance'][1],rtol=1e-12,atol=1e-12)
