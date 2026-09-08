"""Fixed real-model study and counted oracle plumbing for the visible estimators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap import gradientshap_data as data
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap import gradientshap_study as known
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods as reference_methods


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_KEYS = ('clean_0','clean_1','clean_2','shuffled_0','shuffled_1','shuffled_2','ridge')
PROPOSALS = ('kernel','uniform')
BUDGETS = (16,32,64,128,256,512,1024)
REPLICATES = 32


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    manifest = json.loads((HERE/'data_manifest.json').read_text())
    source = ROOT / manifest['known_manifest_path']
    if digest(source) != manifest['known_manifest_sha256']:
        raise ValueError('The published known-model manifest changed.')
    if manifest['replicate_seeds'] != list(range(161750,161782)) or manifest['draw_budgets'] != list(BUDGETS):
        raise ValueError('Sampling protocol constants changed.')
    return manifest


def _check_methods(methods):
    names=('coalition_design','sii_query_weights','sample_coalitions','shapiq_estimate',
           'constrained_wls','pair_residual_design','kernel_weights','kernelshap_iq','query_sampled_game')
    if any(not callable(getattr(methods,name,None)) for name in names):
        raise TypeError('Supply all nine completed learner methods.')


def make_sampling_plans(methods) -> dict:
    """Plan all masks before looking at any model outputs."""
    manifest=load_manifest(); _check_methods(methods)
    result={}
    for proposal in PROPOSALS:
        masks=[];probabilities=[]
        for seed in manifest['replicate_seeds']:
            drawn,p=methods.sample_coalitions(n_players=8,draws=max(BUDGETS),seed=seed,proposal=proposal)
            masks.append(drawn);probabilities.append(p)
        result[proposal]={'masks':t.stack(masks),'probabilities':t.stack(probabilities),'seeds':t.tensor(manifest['replicate_seeds'],dtype=t.int64)}
    return result


def donor_predictions(model, queries, donors, masks, *, design_fn) -> t.Tensor:
    """Evaluate the exact requested intact-donor hybrids, retaining every donor.

    Model/query/donor data are fixed. Only the requested coalition mask axis is
    variable. This is ordinary inference plumbing; the estimator does not see
    model weights, complete tables, labels or a reference interaction value.
    """
    present=design_fn(masks,n_players=8).bool()
    output=t.empty((len(queries),len(masks),len(donors)),dtype=t.float64)
    with t.inference_mode():
        for start in range(0,len(queries),16):
            query=queries[start:start+16]
            for first in range(0,len(masks),32):
                keep=present[first:first+32]
                hybrid=t.where(keep[None,:,None,:],query[:,None,None,:],donors[None,None,:,:])
                prediction=model(hybrid.reshape(-1,8)).reshape(len(query),len(keep),len(donors))
                output[start:start+len(query),first:first+len(keep)]=prediction
    if not bool(t.isfinite(output).all()):
        raise ValueError('Nonfinite model-game value.')
    return output


def fit_and_build_games(methods) -> tuple[dict,dict]:
    """Reproduce the known seven models and construct all 1,904 oracle games."""
    _check_methods(methods);manifest=load_manifest()
    train=data.load_split('train')
    fitted=known.fit_models(train)
    fitted['known_reproduction']=known.verify_16_14_reproduction(fitted,manifest)
    # These held-out labels are only used for descriptive predictive context.
    explain,audit=data.load_split('explain'),data.load_split('audit')
    queries=t.from_numpy(np.concatenate((explain['x'],audit['x']))).double()
    targets=t.from_numpy(np.concatenate((explain['y'],audit['y']))).double()
    source_ids=t.from_numpy(np.concatenate((explain['ids'],audit['ids']))).long()
    donor_positions=data.background_order(train['ids'],manifest['background_seed'])[:64]
    donors=t.from_numpy(train['x'][donor_positions].copy()).double()
    donor_ids=t.from_numpy(train['ids'][donor_positions].copy()).long()
    masks=t.arange(256,dtype=t.int64)
    hybrid=[];predictions=[]
    for name in MODEL_KEYS:
        model=known.make_model(fitted['models'][name])
        hybrid.append(donor_predictions(model,queries,donors,masks,design_fn=methods.coalition_design))
        with t.inference_mode(): predictions.append(model(queries).squeeze(-1).clone())
    hybrid=t.stack(hybrid)
    games=hybrid.mean(-1)
    prediction=t.stack(predictions)
    t.testing.assert_close(games[...,-1],prediction,rtol=0,atol=1e-10)
    t.testing.assert_close(games[...,0],games[:,0:1,0].expand(-1,272),rtol=0,atol=1e-10)
    # Recipe grouping follows the published source definition (first seven inputs).
    raw=data.source_data()
    _,all_groups=np.unique(raw[:,:7],axis=0,return_inverse=True)
    groups=t.from_numpy(all_groups[source_ids.numpy()].copy()).long()
    return fitted,{'hybrid_predictions':hybrid,'values':games,'prediction':prediction,
        'targets':targets,'source_ids':source_ids,'recipe_group_ids':groups,'queries':queries,
        'donors':donors,'donor_source_ids':donor_ids,'feature_names':manifest['features'],
        'explanation_rows':153,'audit_rows':119,'models':MODEL_KEYS,
        'exact_reference_coalition_queries':7*272*256,
        'exact_reference_donor_predictions':hybrid.numel(),
        'oracle_precomputed':True,'query_costs_are_logical':True}


def estimate_all_games(game_values: t.Tensor, plans: dict, methods) -> dict:
    """Run every fixed replicate through the learner's counted subset query API."""
    _check_methods(methods)
    if game_values.shape!=(7,272,256):
        raise ValueError('Every one of the 1,904 complete model games is required.')
    results={}
    for proposal in PROPOSALS:
        shape=(REPLICATES,len(BUDGETS),7,272,28)
        shap=t.empty(shape,dtype=t.float64);se=t.empty_like(shap);kiq=t.empty_like(shap)
        single=t.empty((*shape[:-1],8),dtype=t.float64)
        unique_counts=t.empty((REPLICATES,len(BUDGETS)),dtype=t.int64)
        rank=t.empty_like(unique_counts);first_rank=t.empty_like(rank)
        singular_values=t.empty((REPLICATES,len(BUDGETS),20),dtype=t.float64)
        pair_constraint_residual=t.empty((*shape[:-1],8),dtype=t.float64)
        first_constraint_residual=t.empty((*shape[:-1],1),dtype=t.float64)
        query_masks=[]
        for replicate in range(REPLICATES):
            replicate_masks=[]
            for budget_index,draws in enumerate(BUDGETS):
                masks=plans[proposal]['masks'][replicate,:draws]
                p=plans[proposal]['probabilities'][replicate,:draws]
                receipt=methods.query_sampled_game(lambda requested:game_values[...,requested],masks,n_players=8)
                boundary=receipt['boundary_values'];sampled=receipt['sampled_values']
                unique_counts[replicate,budget_index]=receipt['unique_queries']
                replicate_masks.append(receipt['queried_masks'].clone())
                # Row chunks bound the temporary [rows,draws,pairs] variance array.
                boundary_flat=boundary.reshape(-1,18);sampled_flat=sampled.reshape(-1,draws)
                point=t.empty((1904,28),dtype=t.float64);uncertainty=t.empty_like(point)
                for start in range(0,1904,128):
                    point[start:start+128],uncertainty[start:start+128]=methods.shapiq_estimate(boundary_flat[start:start+128],sampled_flat[start:start+128],masks,p,n_players=8)
                shap[replicate,budget_index]=point.reshape(7,272,28)
                se[replicate,budget_index]=uncertainty.reshape(7,272,28)
                kernel=methods.kernelshap_iq(boundary,sampled,masks,p,n_players=8)
                kiq[replicate,budget_index]=kernel['pairs']
                single[replicate,budget_index]=kernel['shapley']
                rank[replicate,budget_index]=kernel['pair_stage']['free_rank']
                first_rank[replicate,budget_index]=kernel['first_stage']['free_rank']
                s=kernel['pair_stage']['singular_values']
                singular_values[replicate,budget_index]=0
                singular_values[replicate,budget_index,:len(s)]=s
                pair_constraint_residual[replicate,budget_index]=kernel['pair_stage']['constraint_residual']
                first_constraint_residual[replicate,budget_index]=kernel['first_stage']['constraint_residual']
            query_masks.append(replicate_masks)
        results[proposal]={'shapiq':shap,'standard_error':se,'kernelshap_iq':kiq,
            'kernelshap_singletons':single,'unique_queries':unique_counts,
            'queried_masks':query_masks,'pair_free_rank':rank,'first_free_rank':first_rank,
            'pair_free_parameters':20,'first_free_parameters':7,'pair_singular_values':singular_values,
            'pair_constraint_residual':pair_constraint_residual,'first_constraint_residual':first_constraint_residual}
    return results


def exact_sampling_variance(game_values: t.Tensor) -> dict:
    """Sum the actual finite iid law outside the estimator; no normal approximation."""
    n=8
    interior=t.tensor([mask for mask in range(256) if 2<=mask.bit_count()<=6],dtype=t.int64)
    sizes=reference_methods.coalition_design(interior,n_players=n).sum(-1).long()
    gamma=reference_methods.sii_query_weights(interior,n_players=n)
    centered=(game_values-game_values[...,:1]).reshape(-1,256)
    result={}
    for proposal in PROPOSALS:
        p=reference_methods.kernel_weights(interior,n_players=n,order=1) if proposal=='kernel' else t.ones(238,dtype=t.float64)
        p=p/p.sum()
        variance=t.empty((1904,28),dtype=t.float64)
        for start in range(0,1904,128):
            contributions=centered[start:start+128,interior,None]*(gamma/p[:,None])
            mean=(contributions*p[:,None]).sum(-2,keepdim=True)
            variance[start:start+128]=((contributions-mean).square()*p[:,None]).sum(-2)
        result[proposal]={'one_draw_variance':variance.reshape(7,272,28),'interior_masks':interior.clone(),'probabilities':p}
    return result


def package_reference(game_values: t.Tensor, source_ids: t.Tensor) -> dict:
    """Actual package calls on the predeclared source-0 row for all seven models."""
    import shapiq
    from importlib.metadata import version
    if version('shapiq')!='1.5.2': raise ValueError('Expected shapiq==1.5.2.')
    index=int(t.nonzero(source_ids==0).item())
    expected=reference_methods.exact_sii(game_values[:,index],n_players=8)
    terms=reference_methods.interaction_masks(8).tolist()
    output=t.empty((7,3,28),dtype=t.float64);counts=t.zeros((7,3),dtype=t.int64)
    observed_masks=[]
    for model in range(7):
        model_masks=[]
        for kind in range(3):
            calls=[]
            def game(coalitions):
                masks=(np.asarray(coalitions,dtype=np.int64)*(1<<np.arange(8))).sum(-1)
                calls.append(t.from_numpy(masks.copy()))
                return game_values[model,index,t.from_numpy(masks.copy())].numpy().copy()
            if kind==0:
                value=shapiq.ExactComputer(game=game,n_players=8)(index='SII',order=2)
            else:
                estimator=(shapiq.SHAPIQ if kind==1 else shapiq.KernelSHAPIQ)(8,index='SII',max_order=2,random_state=161740)
                value=estimator.approximate(budget=256,game=game)
            output[model,kind]=t.tensor([value[tuple(i for i in range(8) if term&(1<<i))] for term in terms],dtype=t.float64)
            masks=t.cat(calls);counts[model,kind]=len(masks);model_masks.append(masks)
            if set(masks.tolist())!=set(range(256)): raise AssertionError('Package reference omitted a coalition.')
            scale=max(1.,float(game_values[model,index].abs().max()))
            tolerance=(1e-10+1e-12*scale) if kind<2 else 1e-6*scale
            t.testing.assert_close(output[model,kind],expected[model],rtol=0,atol=tolerance)
        observed_masks.append(model_masks)
    return {'version':version('shapiq'),'source_id':0,'values':output,'expected':expected,
        'absolute_error':(output-expected[:,None]).abs(),'actual_queried_masks':observed_masks,
        'queried_coalitions':counts,'scope':'Seven fixed source-0 model games; three actual package algorithms at full budget.'}


def recipe_weights(groups: t.Tensor) -> t.Tensor:
    """Give every recipe group equal total weight, preserving all rows."""
    _,inverse,counts=t.unique(groups,sorted=True,return_inverse=True,return_counts=True)
    return 1.0/(len(counts)*counts[inverse].to(t.float64))


def run_study(methods, output_directory: Path) -> dict:
    """Complete the fixed study; all scientific operations use the learner namespace."""
    _check_methods(methods);directory=Path(output_directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Use a new empty output directory; evidence is not overwritten.')
    directory.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    plans=make_sampling_plans(methods)
    fitted,games=fit_and_build_games(methods)
    estimates=estimate_all_games(games['values'],plans,methods)
    # Reference answers are supplied only after sampled estimates are computed.
    exact={'pairs':reference_methods.exact_sii(games['values'],n_players=8),
           'shapley':reference_methods.exact_sii(games['values'],n_players=8,order=1),
           'sampling_variance':exact_sampling_variance(games['values'])}
    package=package_reference(games['values'],games['source_ids'])
    payloads={'fitted':fitted,'games':games,'plans':plans,'estimates':estimates,'exact':exact,'package':package}
    artifacts={}
    for name,payload in payloads.items():
        path=directory/(name+'.pt');t.save(payload,path)
        artifacts[path.name]={'sha256':digest(path),'bytes':path.stat().st_size}
    summary={'status':'PASS','scope':'complete fixed logical-query study on reproduced model games',
        'models':7,'rows':272,'games':1904,'pairs':28,'replicates_per_proposal':32,
        'proposals':list(PROPOSALS),'draw_budgets':list(BUDGETS),'known_model_state_checks':21,
        'reference_donor_predictions':31_195_136,'sampling_plans_data_independent':True,
        'estimator_inputs_only_queried_values':True,'reference_precomputed_cost_disclosed':True,
        'elapsed_seconds':time.perf_counter()-start,'artifacts':artifacts}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return payloads
