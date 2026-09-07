"""Figures expose estimates, controls and costs; no report JSON is a teaching result."""

from __future__ import annotations

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods as m
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_study as study


COLORS={'SHAP-IQ / kernel':'#0072B2','KernelSHAP-IQ / kernel':'#D55E00',
        'SHAP-IQ / uniform':'#56B4E9','KernelSHAP-IQ / uniform':'#CC79A7'}
SHORT=('Cement','Slag','Fly ash','Water','Superpl.','Coarse','Fine','Age')


def _finish(fig,title):
    fig.suptitle(title,fontsize=16,fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.97))
    return fig


def _labels(n=8):
    names=SHORT if n==8 else ('A','B','C','D')
    return [f'{names[i]} × {names[j]}' for i,j in itertools.combinations(range(n),2)]


def analytic_games(design_fn=m.coalition_design):
    z=design_fn(t.arange(16),n_players=4)
    additive=z@t.tensor([.2,.2,5.,4.],dtype=t.float64)
    mixed=additive+3*z[:,0]*z[:,1]+2*z[:,0]*z[:,1]*z[:,2]
    pure=t.zeros(16,dtype=t.float64);pure[-1]=1
    return {'constant':t.full((16,),2.,dtype=t.float64),'additive':additive,'mixed':mixed,'pure_four':pure}


def plot_shared_query(games,weights_fn,query_mask=3,game_name='mixed'):
    values=games[game_name]; masks=t.arange(16)
    weights=weights_fn(masks,n_players=4)
    update=(values[query_mask]-values[0])*weights[query_mask]
    exact=t.stack([(games[name]-games[name][0])@weights for name in ('constant','additive','mixed')])
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    labels=_labels(4); x=np.arange(6)
    axes[0].bar(x,weights[query_mask].numpy(),color=['#0072B2' if v>=0 else '#D55E00' for v in weights[query_mask]])
    axes[0].set(title=f'Query {query_mask:04b}: shared gamma weights',ylabel='Coefficient of one queried value')
    axes[1].bar(x,update.numpy(),color=['#0072B2' if v>=0 else '#D55E00' for v in update])
    axes[1].set(title=f'One term: v₀(T) = {float(values[query_mask]-values[0]):g}',ylabel='Contribution before sampling adjustment')
    limit=max(1e-12,float(exact.abs().max()))
    im=axes[2].imshow(exact.numpy(),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    axes[2].set_yticks(range(3),['constant','additive','pair + triple'])
    axes[2].set_title('Complete sums: unrelated terms cancel')
    for r in range(3):
        for col in range(6): axes[2].text(col,r,f'{float(exact[r,col]):.1f}',ha='center',va='center',color='white' if abs(float(exact[r,col]))>.6*limit else 'black')
    fig.colorbar(im,ax=axes[2],label='Exact pair SII',shrink=.8)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=55,ha='right')
    for ax in axes[:2]:ax.axhline(0,color='black',linewidth=.7)
    return _finish(fig,'One query updates every pair; a complete sum identifies the interaction')


def plot_sampling(sample_fn):
    n=8;fig,axes=plt.subplots(1,3,figsize=(15,4))
    size=np.arange(2,7)
    for proposal,color,offset in [('kernel','#0072B2',-.16),('uniform','#D55E00',.16)]:
        mass=np.array([1/(k*(8-k)) if proposal=='kernel' else math.comb(8,int(k)) for k in size],dtype=float);mass=mass/mass.sum()
        axes[0].bar(size+offset,mass,width=.3,label=proposal,color=color)
        axes[1].plot(size,mass/np.array([math.comb(8,int(k)) for k in size]),'o-',label=proposal,color=color)
        masks,_=sample_fn(n_players=8,draws=1024,seed=161750,proposal=proposal)
        counts=[len(set(m.boundary_masks(8).tolist())|set(masks[:k].tolist())) for k in study.BUDGETS]
        axes[2].plot(study.BUDGETS,counts,'o-',label=proposal,color=color)
    axes[0].set(title='Size probability',xlabel='Coalition size',ylabel='Probability of size k')
    axes[1].set(title='Per-coalition probability',xlabel='Coalition size',ylabel='Probability of one particular set')
    axes[2].axhline(256,color='black',linestyle='--',label='Exact enumeration')
    axes[2].set(title='One fixed draw stream; no model values',xlabel='IID draws (plus 18 exact boundaries)',ylabel='Unique coalition queries',xscale='log')
    for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.15)
    return _finish(fig,'Sampling probability and model-query cost are different quantities')


def plot_boundary_limit(kernel_fn):
    n=4;v=analytic_games()['pure_four'];masks=t.arange(16)
    edges=m.boundary_masks(n);inside=t.tensor([i for i in range(16) if i.bit_count()==2])
    p=t.full((6,),1/6,dtype=t.float64)
    result=kernel_fn(v[edges],v[inside],inside,p,n_players=n)
    residual=v-m.coalition_design(masks,n_players=n)@result['shapley']
    design=m.pair_residual_design(masks,n_players=n)
    prediction=design@result['pairs']
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    selected=t.tensor([1,14]);x=np.arange(2)
    axes[0].bar(x-.18,residual[selected].numpy(),width=.35,label='Residual to fit',color='#0072B2')
    axes[0].bar(x+.18,prediction[selected].numpy(),width=.35,label='Best common boundary fit',color='#D55E00')
    axes[0].set_xticks(x,['{A}','{B,C,D}']);axes[0].set(title='Two incompatible boundary targets',ylabel='Game value');axes[0].legend(fontsize=9)
    axes[1].bar(np.arange(6),result['pairs'].numpy(),color='#0072B2',label='Boundary-limit fit')
    axes[1].axhline(1/3,color='black',linestyle='--',label='Exact pair SII = 1/3')
    axes[1].axhline(0,color='#777777',linestyle=':',label='Additive control')
    axes[1].set_xticks(np.arange(6),_labels(4),rotation=55,ha='right');axes[1].set(title='The correct pair answer',ylabel='SII');axes[1].legend(fontsize=9)
    penalties=[1,10,1e2,1e3,1e4,1e5,1e6,1e7,1e8];errors=[]
    sizes=m.coalition_design(masks,n_players=n).sum(-1)
    for penalty in penalties:
        weights=t.where((sizes<2)|(sizes>2),penalty,t.ones(16,dtype=t.float64))
        fit=t.linalg.lstsq(weights.sqrt()[:,None]*design,(weights.sqrt()*residual)[:,None],driver='gelsd').solution[:,0]
        errors.append(float((fit-t.full((6,),1/3,dtype=t.float64)).abs().max()))
    axes[2].loglog(penalties,errors,'o-',color='#0072B2',label='Finite-penalty pair error')
    axes[2].axhline(max(1e-16,float((result['pairs']-1/3).abs().max())),color='#D55E00',linestyle='--',label='Direct limit (rounding scale)')
    axes[2].set(title='Approaching the paper’s boundary limit',xlabel='Boundary penalty',ylabel='Maximum absolute pair error');axes[2].legend(fontsize=9);axes[2].grid(alpha=.2)
    return _finish(fig,'Minimize boundary error; do not demand that incompatible rows both fit exactly')


def _split_weights(games,split):
    selection=slice(0,153) if split=='explain' else slice(153,272)
    return selection,study.recipe_weights(games['recipe_group_ids'][selection])


def plot_model_controls(payload):
    games,exact=payload['games'],payload['exact'];fig,axes=plt.subplots(1,2,figsize=(13,4.5));x=np.arange(7)
    for split,offset,color in [('explain',-.18,'#0072B2'),('audit',.18,'#D55E00')]:
        sel,w=_split_weights(games,split)
        rmse=((games['prediction'][:,sel]-games['targets'][sel]).square()*w).sum(-1).sqrt()
        pair=(exact['pairs'][:,sel].abs().mean(-1)*w).sum(-1)
        axes[0].bar(x+offset,rmse.numpy(),width=.35,label=split,color=color)
        axes[1].bar(x+offset,pair.numpy(),width=.35,label=split,color=color)
    for ax in axes:ax.set_xticks(x,study.MODEL_KEYS,rotation=35,ha='right');ax.legend();ax.grid(axis='y',alpha=.15)
    axes[0].set(title='All known predictors; no model selection',ylabel='Recipe-group-weighted prediction RMSE (MPa)')
    axes[1].set(title='Large interactions do not imply good predictions',ylabel='Mean absolute exact pair SII (MPa)')
    return _finish(fig,'Read clean, shuffled-label and additive-ridge controls together')


def plot_convergence(payload):
    games,exact,estimates=payload['games'],payload['exact']['pairs'],payload['estimates']
    fig,axes=plt.subplots(7,2,figsize=(14,21),squeeze=False)
    x=np.array(study.BUDGETS)+18
    for model,key in enumerate(study.MODEL_KEYS):
        for col,split in enumerate(('explain','audit')):
            ax=axes[model,col];sel,w=_split_weights(games,split)
            zero=float((exact[model,sel].abs().mean(-1)*w).sum())
            ax.axhline(zero,color='#555555',linestyle=':',label='Zero-pair baseline')
            for proposal in study.PROPOSALS:
                for field,label in [('shapiq','SHAP-IQ'),('kernelshap_iq','KernelSHAP-IQ')]:
                    err=(estimates[proposal][field][:,:,model,sel]-exact[model,sel]).abs().mean(-1)
                    per_rep=(err*w).sum(-1)
                    low,high=t.quantile(per_rep,t.tensor([.1,.9],dtype=t.float64),dim=0)
                    name=f'{label} / {proposal}'
                    ax.plot(x,per_rep.mean(0).numpy(),'o-',markersize=3,label=name,color=COLORS[name])
                    ax.fill_between(x,low.numpy(),high.numpy(),color=COLORS[name],alpha=.1)
            ax.axvline(256,color='black',alpha=.35,linestyle='--',label='256-query exact reference')
            ax.set(title=f'{key} · {split}',xscale='log',ylabel='Pair MAE (MPa)')
            if key=='ridge':
                ax.set_yscale('symlog',linthresh=1e-10);ax.set_ylim(bottom=0)
                maximum=max(float(estimates[prop]['kernelshap_iq'][:,:,model,sel].abs().max()) for prop in study.PROPOSALS)
                ax.text(.98,.1,f'KernelSHAP-IQ absolute pair maximum: {maximum:.1e} MPa',transform=ax.transAxes,ha='right',fontsize=8)
            else:
                ax.set_yscale('log')
            ax.grid(alpha=.15)
            if model==6:ax.set_xlabel('Nominal requests: 18 exact boundaries + K iid draws')
    axes[0,0].legend(fontsize=8,ncol=2)
    return _finish(fig,'All pair errors across query budgets · mean and 10–90% replicate range')


def plot_signed_comparison(payload):
    games,exact,estimates=payload['games'],payload['exact']['pairs'],payload['estimates']
    row=int(t.nonzero(games['source_ids']==0).item());budget=study.BUDGETS.index(128)
    arrays=[exact[0,row]];names=['Exact SII']
    for proposal in study.PROPOSALS:
        for field,label in [('shapiq','SHAP-IQ'),('kernelshap_iq','KernelSHAP-IQ')]:
            arrays.append(estimates[proposal][field][0,budget,0,row]);names.append(f'{label} / {proposal}')
    values=t.stack(arrays);errors=values-exact[0,row]
    fig,axes=plt.subplots(2,1,figsize=(16,6))
    for ax,array,title in zip(axes,[values,errors],['Signed pair values (MPa)','Signed error relative to exact SII (MPa)']):
        limit=max(1e-12,float(array.abs().max()));im=ax.imshow(array.numpy(),aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_yticks(range(5),names);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.8)
        ax.set_xticks(range(28),_labels(),rotation=65,ha='right',fontsize=8)
    return _finish(fig,'Fixed source 0 · clean_0 · replicate 0 · 128 draws · every pair retained')


def plot_variance(payload):
    games,exact,estimates=payload['games'],payload['exact'],payload['estimates']
    fig,axes=plt.subplots(7,2,figsize=(14,20))
    for model,key in enumerate(study.MODEL_KEYS):
        for col,proposal in enumerate(study.PROPOSALS):
            ax=axes[model,col]
            for split,style in [('explain','-'),('audit','--')]:
                sel,w=_split_weights(games,split)
                true=(exact['sampling_variance'][proposal]['one_draw_variance'][model,sel].mean(-1)*w).sum()/t.tensor(study.BUDGETS,dtype=t.float64)
                se2=(estimates[proposal]['standard_error'][:,:,model,sel].square().mean(-1)*w).sum(-1).mean(0)
                error2=((estimates[proposal]['shapiq'][:,:,model,sel]-exact['pairs'][model,sel]).square().mean(-1)*w).sum(-1).mean(0)
                for vals,label,color in [(true,'Exact iid SD','#333333'),(se2,'RMS reported SE','#0072B2'),(error2,'Observed replicate RMSE','#D55E00')]:
                    ax.plot(study.BUDGETS,vals.sqrt().numpy(),style,label=f'{label} · {split}',color=color)
            ax.set(title=f'{key} · {proposal} proposal',xscale='log',yscale='log',ylabel='Pair error / uncertainty (MPa)');ax.grid(alpha=.15)
            if model==6:ax.set_xlabel('IID draws K')
    axes[0,0].legend(fontsize=8)
    return _finish(fig,'SHAP-IQ sampling variability · exact finite law, reported SE and replicate errors')


def plot_query_costs(payload):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));x=np.array(study.BUDGETS)
    for proposal,color in [('kernel','#0072B2'),('uniform','#D55E00')]:
        record=payload['estimates'][proposal]
        for rep in range(32):
            axes[0].plot(x,record['unique_queries'][rep].numpy(),color=color,alpha=.12)
            axes[1].plot(x,record['pair_free_rank'][rep].numpy(),color=color,alpha=.12)
        axes[0].plot(x,record['unique_queries'].double().mean(0).numpy(),'o-',color=color,label=proposal)
        axes[1].plot(x,record['pair_free_rank'].double().mean(0).numpy(),'o-',color=color,label=proposal)
        repeated=1-record['unique_queries'].double()/(t.tensor(study.BUDGETS)+18)
        axes[2].plot(x,repeated.mean(0).numpy(),'o-',color=color,label=proposal)
    axes[0].axhline(256,color='black',linestyle='--',label='Complete reference')
    axes[1].axhline(20,color='black',linestyle='--',label='All free directions identified')
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('IID draws K');ax.legend(fontsize=9);ax.grid(alpha=.15)
    axes[0].set(title='Unique logical queries per model/row',ylabel='Coalitions (×64 donor predictions)')
    axes[1].set(title='Pair regression information',ylabel='Rank after eight incidence constraints',ylim=(-.5,21))
    axes[2].set(title='Caching preserves iid multiplicities',ylabel='Fraction of repeated requests',ylim=(0,1))
    return _finish(fig,'Count both queries and draws; a finite regression answer can still be unidentified')


def make_native_play(payload,methods,*,model_key='clean_0',source_id=0,draws=128,seed=161750,proposal='kernel'):
    if model_key not in study.MODEL_KEYS:raise ValueError('Unknown model key.')
    if type(draws) is not int or not 2<=draws<=4096:raise ValueError('Choose 2..4096 draws for this play.')
    positions=t.nonzero(payload['games']['source_ids']==source_id).flatten()
    if len(positions)!=1:raise ValueError('Choose a source ID from the held-out rows.')
    row=int(positions.item());model=study.MODEL_KEYS.index(model_key)
    values=payload['games']['values'][model,row]
    masks,p=methods.sample_coalitions(n_players=8,draws=draws,seed=seed,proposal=proposal)
    receipt=methods.query_sampled_game(lambda selected:values[selected],masks,n_players=8)
    point,se=methods.shapiq_estimate(receipt['boundary_values'],receipt['sampled_values'],masks,p,n_players=8)
    kernel=methods.kernelshap_iq(receipt['boundary_values'],receipt['sampled_values'],masks,p,n_players=8)
    return {'model_key':model_key,'source_id':source_id,'draws':draws,'seed':seed,'proposal':proposal,
        'exploratory_audit_reuse':row>=153,'shapiq':point,'se':se,'kernelshap_iq':kernel,
        'exact':payload['exact']['pairs'][model,row],'receipt':receipt,'sampled_masks':masks,'probabilities':p}


def plot_native_play(play):
    fig,axes=plt.subplots(2,1,figsize=(15,7));x=np.arange(28);exact=play['exact']
    axes[0].plot(x,exact.numpy(),'o-',color='black',label='Exact same-game SII',markersize=4)
    axes[0].errorbar(x-.1,play['shapiq'].numpy(),yerr=2*play['se'].numpy(),fmt='.',color='#0072B2',alpha=.7,label='SHAP-IQ ± 2 sample SE (diagnostic)')
    axes[0].plot(x+.1,play['kernelshap_iq']['pairs'].numpy(),'x',color='#D55E00',label='KernelSHAP-IQ')
    axes[0].set_ylabel('Pair SII (MPa)');axes[0].legend(fontsize=9);axes[0].grid(alpha=.15)
    axes[1].bar(x-.18,(play['shapiq']-exact).numpy(),width=.35,color='#0072B2',label='SHAP-IQ signed error')
    axes[1].bar(x+.18,(play['kernelshap_iq']['pairs']-exact).numpy(),width=.35,color='#D55E00',label='KernelSHAP-IQ signed error')
    axes[1].set_ylabel('Error (MPa)');axes[1].legend(fontsize=9)
    for ax in axes:ax.set_xticks(x,_labels(),rotation=65,ha='right',fontsize=8);ax.axhline(0,color='black',linewidth=.7)
    rank=play['kernelshap_iq']['pair_stage']['free_rank'];q=play['receipt']['unique_queries']
    title=f"Exploratory {play['model_key']} / source {play['source_id']} · {play['draws']} draws · {q} unique queries · pair rank {rank}/20"
    return _finish(fig,title)
