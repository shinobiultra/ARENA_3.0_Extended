"""Figure layout only: inputs are the notebook's actual fits and generated rows."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

COLORS = {'diffusion':'#2166ac','autoregressive':'#b35806',
          'diffusion-shuffled':'#7b3294','autoregressive-shuffled':'#777777',
          'ancestral':'#1b9e77','uniform':'#7570b3','confidence':'#d95f02'}
TASKS = ('arithmetic','parentheses','qa')
LABELS = {'arithmetic':'Exact addition + EOS','parentheses':'Valid completed grammar + EOS','qa':'Exact context answer + EOS'}


def metric(result,task):
    return result['summary'][task]['valid_grammar_rate' if task=='parentheses' else 'exact_answer_rate']


def save_figure(fig,directory,name):
    path=Path(directory)/name;path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150,bbox_inches='tight')
    plt.show()
    return path


def plot_training(fits,directory):
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for axis,kind in zip(axes,('diffusion','autoregressive')):
        for fit in fits:
            if fit['kind']!=kind:continue
            name=kind+('-shuffled' if fit['shuffled'] else '')
            history=[r for r in fit['history'] if 'development_objective' in r]
            x=[r['step'] for r in history];y=[r['development_objective'] for r in history]
            axis.plot(x,y,color=COLORS[name],alpha=.65,linestyle='--' if fit['shuffled'] else '-',
                      label=f"{name}, seed {fit['seed']}")
            axis.scatter([fit['selected_step']],[fit['selected_development_objective']],color=COLORS[name],marker='x',s=70)
        axis.set(title=f'{kind}: every development checkpoint',xlabel='Optimizer updates',
                 ylabel='Finite diffusion bound / target slot' if kind=='diffusion' else 'Causal NLL / target slot')
        axis.legend(fontsize=7,frameon=False)
    fig.suptitle('Within-objective checkpoint selection — crosses mark the earliest minimum; scales differ')
    return save_figure(fig,directory,'diffusion_text_training.png')


def baseline_rows(results,checkpoint='selected'):
    return [r for r in results if r['checkpoint']==checkpoint and r['config']['name'] in {'ancestral-16-greedy','ar-greedy'}]


def plot_task_quality(results,directory):
    chosen=baseline_rows(results)+baseline_rows(results,'untrained')+baseline_rows(results,'endpoint')
    order=[(c,l) for c in ('selected','endpoint','untrained') for l in ('diffusion','diffusion-shuffled','autoregressive','autoregressive-shuffled')
           if any(r['checkpoint']==c and r['label']==l for r in chosen)]
    fig,axes=plt.subplots(1,3,figsize=(17,5.8),layout='constrained')
    for ax,task in zip(axes,TASKS):
        for i,(checkpoint,label) in enumerate(order):
            values=[metric(r['result'],task) for r in chosen if r['checkpoint']==checkpoint and r['label']==label]
            color=COLORS[label]
            ax.bar(i,np.mean(values),color=color,alpha=.4 if checkpoint!='selected' else .85)
            ax.scatter(np.full(len(values),i)+np.linspace(-.12,.12,len(values)),values,color='black',s=15,zorder=3)
        ax.set(ylim=(0,1.03),title=LABELS[task],ylabel='Fraction of all 128 final examples')
        ax.set_xticks(range(len(order)),[f'{label}\n{checkpoint}' for checkpoint,label in order],rotation=65,ha='right',fontsize=7)
    fig.suptitle('Actual final generations — three seeds, shuffled supervision, late checkpoints and untrained models')
    return save_figure(fig,directory,'diffusion_text_task_controls.png')


def plot_budget_quality(results,directory):
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    chosen=[r for r in results if r['checkpoint']=='selected' and r['kind']=='diffusion'
            and r['config']['temperatures']==[0.,0.] and r['config']['block_size'] is None]
    for ax,task in zip(axes,TASKS):
        for sampler in ('ancestral','uniform','confidence'):
            for shuffled in (False,True):
                rows=[r for r in chosen if r['config']['sampler']==sampler and r['shuffled']==shuffled]
                x=sorted({r['config']['steps'] for r in rows})
                values=[[metric(r['result'],task) for r in rows if r['config']['steps']==steps] for steps in x]
                means=[np.mean(v) for v in values]
                ax.plot(x,means,marker='o',linestyle='--' if shuffled else '-',color=COLORS[sampler],
                        label=sampler+(' shuffled' if shuffled else ''))
                ax.fill_between(x,[min(v) for v in values],[max(v) for v in values],color=COLORS[sampler],alpha=.07)
        ar=[metric(r['result'],task) for r in baseline_rows(results) if r['label']=='autoregressive']
        ax.axhline(np.mean(ar),color=COLORS['autoregressive'],linestyle=':',label='Cached AR, selected')
        ax.set(title=LABELS[task],xlabel='Full-window denoising calls',ylabel='Fraction correct',ylim=(0,1.03),xticks=[1,4,8,16])
    axes[-1].legend(fontsize=7,frameon=False,loc='upper left',bbox_to_anchor=(1,1))
    fig.suptitle('All declared greedy budgets — lines are seed means; shading spans all three seeds')
    return save_figure(fig,directory,'diffusion_text_budget_quality.png')


def plot_cost_quality(results,timings,directory):
    lookup={(r['seed'],r['label'],r['checkpoint'],r['config']['name']):r for r in results}
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for row in timings:
        result=lookup[(row['seed'],row['label'],'selected',row['config'])]
        sampler=result['config'].get('sampler','autoregressive')
        color=COLORS[sampler];quality=metric(result['result'],'arithmetic')
        marker='x' if result['shuffled'] else 'o'
        axes[0].scatter(row['measurement']['forward_calls'],quality,color=color,marker=marker,alpha=.6)
        axes[1].scatter(row['measurement']['median_seconds']*1000,quality,color=color,marker=marker,alpha=.6)
    for ax in axes:
        for label,color in [(s,COLORS[s]) for s in ('ancestral','uniform','confidence','autoregressive')]:
            ax.scatter([],[],color=color,label=label)
        ax.scatter([],[],color='black',marker='x',label='Shuffled supervision')
        ax.set(ylabel='Exact final addition rate',ylim=(0,1.03))
    axes[0].set(xlabel='Actual batch forward calls',title='Calls are not equal-cost operations')
    axes[1].set(xlabel='Median milliseconds / fixed batch of 32',title='Cached AR; synchronized CUDA; two warmups + five repeats')
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Quality and measured cost together — all three seeds and matched shuffled controls')
    return save_figure(fig,directory,'diffusion_text_cost_quality.png')


def plot_trajectory_pair(main,control,tokenizer,directory,*,name='diffusion_text_trajectory.png'):
    fig,axes=plt.subplots(1,2,figsize=(23,8),layout='constrained')
    for ax,trace,label in zip(axes,(main,control),('Input-matched training','Shuffled-supervision control')):
        states=np.asarray(trace['states']);mask=states==1
        image=np.where(mask,0,np.where(states==states[-1:],2,1))
        ax.imshow(image,vmin=0,vmax=2,cmap=ListedColormap(['#e5e7eb','#d99489','#b5d2e3']),aspect='auto')
        for k,row in enumerate(states):
            for j,token in enumerate(row):
                text='■' if token==1 else tokenizer.id_to_token(int(token)).replace('Ġ','·').replace('Ċ','↵')
                ax.text(j,k,text,ha='center',va='center',fontsize=6,rotation=45 if len(text)>5 else 0)
        ax.set(xticks=np.arange(states.shape[1]),yticks=np.arange(len(states)),xlabel='All 24 target slots',ylabel='State (0 = all MASK)',
               title=label+'\n'+textwrap.fill(trace['outcome']['text'],80))
    fig.suptitle('Actual token decisions for a fixed example\n'+textwrap.fill(main['prompt'],120)
                 +'\nGray = MASK; blue = final-token match, which can still be wrong',fontsize=12)
    return save_figure(fig,directory,name)


def plot_commitment(main,control,directory):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for trace,label,color in [(main,'Matched training','#2166ac'),(control,'Shuffled supervision','#7b3294')]:
        values=[]
        for record in trace['steps']:
            h=np.asarray(record['entropy']);masked=np.asarray(record['masked_before'],dtype=bool)
            values.append(float(h[masked].mean()) if masked.any() else float('nan'))
        axes[0].plot(range(1,len(values)+1),values,marker='o',color=color,label=label)
        axes[1].plot(range(len(trace['stable_commitment'])),trace['stable_commitment'],marker='o',color=color,label=label)
    axes[0].set(title='Predictive uncertainty before revealing',xlabel='Model call',ylabel='Mean entropy at still-MASK slots (nats)')
    axes[1].set(title='Commitment does not certify correctness',xlabel='Target position',ylabel='First permanent final-token state')
    for ax in axes:ax.legend(frameon=False)
    return save_figure(fig,directory,'diffusion_text_entropy_commitment.png')


def plot_generation_gallery(source_rows,results,directory,*,task='story',count=3,name='diffusion_text_story_gallery.png'):
    selected=[r for r in source_rows if r['task']==task][:count]
    baselines=[r for r in baseline_rows(results) if r['seed']==0]
    entries=[]
    for row in selected:
        entries.append(('Input '+row['id'],row['prompt']))
        entries.append(('One reference',row.get('reference_continuation',row.get('answer',''))))
        for run in baselines:
            outcome=next(r for r in run['result']['rows'] if r['id']==row['id'])
            entries.append((run['label'],outcome['text']))
        entries.append(('',''))
    lines=[(label,textwrap.fill(text,105)) for label,text in entries]
    heights=[max(1,text.count('\n')+1) for _,text in lines]
    fig,ax=plt.subplots(figsize=(15,max(6,sum(heights)*.23+1.2)),layout='constrained')
    ax.set_axis_off();y=1.
    for (label,text),height in zip(lines,heights):
        ax.text(0,y,label,transform=ax.transAxes,fontsize=8,fontweight='bold',va='top')
        ax.text(.20,y,text,transform=ax.transAxes,fontsize=9,va='top',family='DejaVu Sans')
        y-=(height+.45)/(sum(heights)+.45*len(lines))
    title=(f'First {count} fixed TinyStories inputs — actual tokens, including errors; no story-accuracy claim'
           if task=='story' else f'First {count} fixed {task} inputs — exact reference and all actual baseline outputs')
    ax.set_title(title,pad=18)
    return save_figure(fig,directory,name)


def plot_temperature_blocks(results,directory):
    names=['confidence-16-greedy','confidence-16-annealed','confidence-16-warm',
           'confidence-block4-4','confidence-block8-4']
    labels=['Full window\n16 calls, greedy','Full window\n16 calls, 1→0.2',
            'Full window\n16 calls, 0.8','Block width 4\n24 calls, greedy','Block width 8\n12 calls, greedy']
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    for ax,task in zip(axes,TASKS):
        for shuffled,label,color,offset in [(False,'Matched training','#2166ac',-.15),(True,'Shuffled supervision','#7b3294',.15)]:
            values=[[metric(r['result'],task) for r in results if r['checkpoint']=='selected'
                and r['kind']=='diffusion' and r['shuffled']==shuffled and r['config']['name']==name] for name in names]
            for i,v in enumerate(values):
                ax.bar(i+offset,np.mean(v),width=.28,color=color,alpha=.65,label=label if i==0 else None)
                ax.scatter(np.full(len(v),i+offset)+np.linspace(-.07,.07,len(v)),v,color=color,s=18,zorder=3)
        ax.set(xticks=range(len(names)),xticklabels=labels,title=LABELS[task],ylim=(0,1.03),ylabel='Fraction correct on all final examples')
        ax.tick_params(axis='x',labelrotation=30,labelsize=8)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Frozen temperature and block ablations — all three seeds and matched controls; note the different call budgets')
    return save_figure(fig,directory,'diffusion_text_temperature_blocks.png')
