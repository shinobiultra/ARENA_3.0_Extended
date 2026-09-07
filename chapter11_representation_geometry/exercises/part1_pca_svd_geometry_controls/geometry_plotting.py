"""Figure layout only: inputs are actual learner-computed projections and rows."""
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS={'baseline':'Original','no_op':'No-op','mean_only':'Mean only','pure_coordinate':'Pure coordinate edit',
        'centered_mean_ablated':'Centered plane + mean','author_raw_mean_ablated':'Author plane + mean',
        'full_donor':'Full donor residual','pca5_donor':'Five-PC donor',
        'same_rank_random_donor_plane':'Random rank-two donor','equal_norm_random_edit':'Equal-norm random edit',
        'class_order_permuted_circular_probe':'Permuted calendar order'}
COLORS={'baseline':'#707c86','no_op':'#a8afb6','mean_only':'#a38871','pure_coordinate':'#37688c',
        'centered_mean_ablated':'#a470a5','author_raw_mean_ablated':'#b64e32','full_donor':'#438579',
        'pca5_donor':'#c49c45','same_rank_random_donor_plane':'#8c969e','equal_norm_random_edit':'#9276a3',
        'class_order_permuted_circular_probe':'#b99d63'}


def finish(fig,directory,name):
    path=Path(directory)/name;fig.savefig(path,dpi=160,bbox_inches='tight');plt.show();return str(path)


def plot_calendar_projections(projections,directory):
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for row,(family,data) in enumerate(projections.items()):
        count=len(data['names']);cmap=plt.get_cmap('tab20',count)
        for col,(train,test,title) in enumerate([(data['train_pca'],data['final_pca'],'Unsupervised training PCA'),
                                               (data['train_probe'],data['final_probe'],'Supervised circular probe')]):
            ax=axes[row,col]
            ax.scatter(test[:,0],test[:,1],c=data['final_labels'],cmap=cmap,vmin=0,vmax=count-1,marker='x',s=12,alpha=.18,label='New templates')
            ax.scatter(train[:,0],train[:,1],c=data['train_labels'],cmap=cmap,vmin=0,vmax=count-1,s=30,edgecolor='black',linewidth=.4,label='Training prompts')
            for i,name in enumerate(data['names']):
                center=train[np.asarray(data['train_labels'])==i].mean(0)
                ax.annotate(name[:3],center,xytext=(4,4),textcoords='offset points',fontsize=8)
            ax.set(title=f'{family.title()} | {title}',xlabel='Component 1' if col==0 else 'Predicted cosine',ylabel='Component 2' if col==0 else 'Predicted sine')
            ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8)
    fig.suptitle('Labels are added after PCA; training a circular probe does not by itself discover a circle')
    return finish(fig,directory,'geometry_mistral_projections.png')


def plot_calendar_layers(tasks,directory):
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for row,(family,task) in enumerate(tasks.items()):
        sweep=task['layer_sweep'];x=[point['boundary'] for point in sweep]
        axes[row,0].plot(x,[point['final_coordinate_r2'] for point in sweep],color='#37688c',label='New-template coordinate R²')
        axes[row,0].axhline(0,color='#777',linestyle=':',label='Mean-coordinate predictor')
        axes[row,0].set(ylabel=f'{family.title()} | coordinate R²',xlabel='Residual boundary')
        axes[row,1].plot(x,[point['training_pca2_order_agreement'] for point in sweep],color='#37688c',label='Chronological order')
        axes[row,1].plot(x,[np.quantile(point['order_null'],.95) for point in sweep],color='#a38871',linestyle='--',label='512-permutation 95th percentile')
        axes[row,1].set(ylabel='PCA distance-order correlation',xlabel='Residual boundary')
        for condition in ['pure_coordinate','author_raw_mean_ablated','full_donor','mean_only','equal_norm_random_edit']:
            axes[row,2].plot(x,[point['conditions'][condition]['desired_choice_accuracy'] for point in task['development_interventions']],label=LABELS[condition],color=COLORS[condition])
        axes[row,2].set(ylabel='Development shifted-answer accuracy',xlabel='Residual boundary',ylim=(-.03,1.03))
        for ax in axes[row]:
            ax.axvline(task['layer_input'],color='#444',linestyle=':',alpha=.5)
            ax.legend(fontsize=7)
    fig.suptitle('Decodable information can remain after the last causal use of this token')
    return finish(fig,directory,'geometry_mistral_layer_sweep.png')


def plot_calendar_final_controls(summary,null_summary,directory):
    conditions=['baseline','pure_coordinate','centered_mean_ablated','author_raw_mean_ablated','full_donor','pca5_donor','mean_only']
    fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    for row,family in enumerate(['weekday','month']):
        frame=summary[summary.family.eq(family)].set_index('condition')
        for col,(metric,label) in enumerate([('desired_accuracy','Desired calendar-answer accuracy'),('mean_margin','Desired − original answer logit')]):
            ax=axes[row,col]
            for i,name in enumerate(conditions):ax.bar(i,frame.loc[name,metric],color=COLORS[name])
            for i,name in enumerate(['same_rank_random_donor_plane','equal_norm_random_edit','class_order_permuted_circular_probe'],start=len(conditions)):
                values=null_summary[null_summary.family.eq(family)&null_summary.condition.eq(name)][metric]
                ax.scatter(i+np.linspace(-.15,.15,len(values)),values,color=COLORS[name],s=15,alpha=.65)
                ax.plot([i-.25,i+.25],[values.mean()]*2,color='black',linewidth=1)
            names=conditions+['same_rank_random_donor_plane','equal_norm_random_edit','class_order_permuted_circular_probe']
            ax.set_xticks(range(len(names)),[LABELS[name] for name in names],rotation=45,ha='right',fontsize=8)
            ax.set(ylabel=label,title=family.title());ax.axhline(0,color='#444',linewidth=.6)
            if col==0:ax.set_ylim(-.03,1.03)
    fig.suptitle('New templates, fixed +2 donor shift | bars: planned interventions; dots: all 32 null draws')
    return finish(fig,directory,'geometry_mistral_final_controls.png')


def plot_calendar_all_donors(frame,tasks,directory):
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    for row,(family,task) in enumerate(tasks.items()):
        count=len(task['names'])
        for col,condition in enumerate(['baseline','author_raw_mean_ablated','full_donor']):
            subset=frame[frame.family.eq(family)&frame.condition.eq(condition)]
            matrix=pd.crosstab(subset.desired_answer,subset.restricted_answer,normalize='index').reindex(index=range(count),columns=range(count),fill_value=0)
            image=axes[row,col].imshow(matrix,vmin=0,vmax=1,cmap='Blues');axes[row,col].grid(False)
            axes[row,col].set(title=f'{family.title()} | {LABELS[condition]}',xlabel='Actual calendar choice',ylabel='Requested counterfactual answer',
                              xticks=range(count),yticks=range(count))
            axes[row,col].set_xticklabels([name[:3] for name in task['names']],rotation=45,ha='right',fontsize=7)
            axes[row,col].set_yticklabels([name[:3] for name in task['names']],fontsize=7)
    fig.colorbar(image,ax=axes,shrink=.8,label='Fraction within requested-answer row')
    fig.suptitle('Every counterfactual donor is retained; a full residual transplant is a comparator, not a guaranteed ceiling')
    return finish(fig,directory,'geometry_mistral_all_donors.png')


def plot_calendar_examples(frame,tasks,directory):
    fig,axes=plt.subplots(2,1,figsize=(14,10),constrained_layout=True)
    for ax,(family,task) in zip(axes,tasks.items(),strict=True):
        ax.axis('off');y=1.
        chosen=task['rows']['final'][:3]
        for source in chosen:
            group=frame[frame.row_id.eq(source['id'])&frame['shift'].eq(2)&frame.draw.isna()].set_index('condition')
            ax.text(0,y,'\n'.join(textwrap.wrap(source['prompt'],95)),transform=ax.transAxes,va='top',fontsize=10,fontweight='bold');y-=.07
            ax.text(0,y,f"Original answer: {task['names'][source['answer']]} | change start to {task['names'][(source['start']+2)%len(task['names'])]} | desired answer: {task['names'][(source['answer']+2)%len(task['names'])]}",transform=ax.transAxes,va='top',fontsize=9);y-=.05
            for condition in ['baseline','pure_coordinate','author_raw_mean_ablated','full_donor','mean_only']:
                item=group.loc[condition]
                text=f"{LABELS[condition]:26s}  full-vocabulary top: {item.top_token_text!r:14s}  calendar choice: {task['names'][int(item.restricted_answer)]}"
                ax.text(.015,y,text,transform=ax.transAxes,va='top',fontsize=8.7,fontfamily='monospace',color=COLORS[condition]);y-=.035
            y-=.025
        ax.set_title(f'{family.title()}: first three fixed final inputs, including every error',loc='left',pad=10)
    return finish(fig,directory,'geometry_mistral_actual_answers.png')


def plot_projection_sweep(records,directory):
    frame=pd.DataFrame(records)
    fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    for row,family in enumerate(['weekday','month']):
        for col,(metric,label) in enumerate([('final_knn_accuracy','Final 3-NN accuracy'),('trustworthiness','Trustworthiness, k=5'),
                                             ('continuity','Continuity, k=5'),('procrustes_final_error_relative_to_seed0','Aligned error to seed 0')]):
            ax=axes[row,col]
            for group,(reducer,control) in enumerate([('umap','real'),('umap','independent_noise'),('tsne','real'),('tsne','independent_noise')]):
                subset=frame[frame.family.eq(family)&frame.reducer.eq(reducer)&frame.control.eq(control)]
                for setting in range(3):
                    points=subset[subset.setting_index.eq(setting)]
                    xs=group+(setting-1)*.18+np.linspace(-.04,.04,len(points))
                    ax.scatter(xs,points[metric],s=25,color=['#37688c','#b64e32','#438579'][setting],alpha=.7,label=f'Setting {setting+1}' if group==0 else None)
                    if col==0:
                        ax.scatter(xs,points.permuted_training_labels_accuracy,s=16,marker='x',color='#333',alpha=.55,label='Shuffled training labels' if group==0 and setting==0 else None)
                ax.plot([group-.3,group+.3],[subset[metric].mean()]*2,color='black',linewidth=1)
            ax.set_xticks(range(4),['UMAP\nreal','UMAP\nnoise','t-SNE\nreal','t-SNE\nnoise'],fontsize=8)
            ax.set(title=f'{family.title()} | {label}')
            if col<3:ax.set_ylim(-.03,1.03)
            if row==0 and col==0:ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('All five seeds and three settings | UMAP transforms new points; t-SNE fits them jointly')
    return finish(fig,directory,'geometry_projection_stability.png')


def plot_projection_gallery(records,study_tensors,directory,seed=0,setting_index=1):
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for row,family in enumerate(['weekday','month']):
        task=study_tensors[family];train_y=np.array([r['start'] for r in task['rows']['train']]);final_y=np.array([r['start'] for r in task['rows']['final']])
        count=len(NAMES_FOR_PLOTS[family])
        for col,(reducer,control) in enumerate([('umap','real'),('umap','independent_noise'),('tsne','real'),('tsne','independent_noise')]):
            record=next(r for r in records if r['family']==family and r['control']==control and r['reducer']==reducer and r['seed']==seed and r['setting_index']==setting_index)
            train=np.array(record['train_projection']);final=np.array(record['final_projection']);ax=axes[row,col]
            ax.scatter(final[:,0],final[:,1],c=final_y,cmap=plt.get_cmap('tab20',count),vmin=0,vmax=count-1,s=10,alpha=.3,marker='x')
            ax.scatter(train[:,0],train[:,1],c=train_y,cmap=plt.get_cmap('tab20',count),vmin=0,vmax=count-1,s=25,edgecolor='black',linewidth=.4)
            ax.set(title=f"{family.title()} | {reducer.upper()} | {control.replace('_',' ')}\n3-NN {record['final_knn_accuracy']:.2f}; shuffled {record['permuted_training_labels_accuracy']:.2f}",xlabel='Embedding coordinate 1',ylabel='Embedding coordinate 2')
            ax.set_aspect('equal',adjustable='datalim')
    fig.suptitle(f'Seed {seed}, setting {setting_index+1}: independent noise can also make an attractive picture')
    return finish(fig,directory,'geometry_projection_gallery.png')


NAMES_FOR_PLOTS={'weekday':['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
                 'month':['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}


def plot_angle_play(records,directory):
    from matplotlib.colors import ListedColormap,BoundaryNorm
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for row,family in enumerate(['weekday','month']):
        names=NAMES_FOR_PLOTS[family];count=len(names)
        cmap=ListedColormap(plt.get_cmap('tab20',count)(range(count)));norm=BoundaryNorm(np.arange(count+1)-.5,count)
        for col,condition in enumerate(['pure_coordinate','author_raw_mean_ablated','equal_norm_random_edit']):
            record=next(r for r in records if r['family']==family and r['condition']==condition)
            matrix=np.array(record['calendar_logits']).argmax(-1).reshape(len(record['radii']),-1)
            ax=axes[row,col];im=ax.imshow(matrix,aspect='auto',origin='lower',cmap=cmap,norm=norm);ax.grid(False)
            tick_x=np.arange(count)*len(record['angles'])/count
            ax.set(xticks=tick_x,yticks=range(len(record['radii'])),ylabel='Requested radius',xlabel='Requested input calendar coordinate',title=f'{family.title()} | {LABELS[condition]}')
            ax.set_xticklabels(names,rotation=45,ha='right',fontsize=8);ax.set_yticklabels(record['radii'])
            radius_one=record['radii'].index(1.)
            ax.axhline(radius_one-.5,color='black',linestyle=':',linewidth=.8);ax.axhline(radius_one+.5,color='black',linestyle=':',linewidth=.8)
            baseline=record['shared']['baseline']['top_token_text'];mean=record['shared']['mean_only']['top_token_text']
            ax.set_title(f'{family.title()} | {LABELS[condition]}\nFull-vocabulary original: {baseline!r}; mean only: {mean!r}',fontsize=10,pad=10)
        fig.colorbar(im,ax=axes[row],ticks=range(count),shrink=.8,label='Actual restricted calendar answer').ax.set_yticklabels(names)
    fig.suptitle('Turn a continuous coordinate and read the answer | first final input, offset +1; dotted row: unit circle')
    return finish(fig,directory,'geometry_mistral_angle_radius.png')
