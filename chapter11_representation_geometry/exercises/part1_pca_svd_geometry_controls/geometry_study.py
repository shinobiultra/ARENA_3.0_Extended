"""Readable experiment reference; these experiment bodies belong in the learner.

The published notebook must define these functions itself after implementing
the primitives. Data construction and serialization are the only plumbing.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import torch as t

from .geometry_data import NAMES,calendar_rows,paired_donor_indices
from .geometry_methods import (
    fit_pca,fit_ridge_coordinates,circular_coordinates,angular_predictions,coordinate_r2,
    cyclic_distance_agreement,cyclic_permutation_test,knn_predictions,neighborhood_scores,fit_procrustes,
    replace_coordinates,replace_subspace,mean_ablate_coordinates,author_raw_circle_replacement,
    positions_from_spans,capture_residual_positions,score_calendar_logits,forward_with_residual_replacement,
)


def encode_calendar_batch(tokenizer,rows,device):
    encoded=tokenizer([row['prompt'] for row in rows],padding=True,return_offsets_mapping=True,return_tensors='pt')
    offsets=encoded.pop('offset_mapping')
    positions=positions_from_spans(offsets,encoded['attention_mask'],t.tensor([row['source_span'] for row in rows]))
    encoded={key:value.to(device) for key,value in encoded.items()}
    if not encoded['attention_mask'][:,-1].bool().all():
        raise ValueError('Use left padding so the last model position is an actual prediction position.')
    return encoded,positions.to(device)


@t.inference_mode()
def capture_calendar_rows(model,tokenizer,rows,calendar_ids,batch_size=32):
    """All residual boundaries, reduced answer logits and full-vocabulary winners."""
    device=next(model.parameters()).device
    boundaries=[*model.model.layers,model.model.norm]
    hidden,logits,top=[],[],[]
    for start in range(0,len(rows),batch_size):
        encoded,positions=encode_calendar_batch(tokenizer,rows[start:start+batch_size],device)
        out,acts=capture_residual_positions(model,boundaries,encoded,positions)
        last=out[:,-1].float()
        hidden.append(acts);logits.append(last[:,calendar_ids].cpu());top.append(last.argmax(-1).cpu())
    return {'activations':t.cat(hidden),'calendar_logits':t.cat(logits),'top_token_ids':t.cat(top)}


@t.inference_mode()
def measure_calendar_replacements(model,tokenizer,rows,calendar_ids,layer,replacements,batch_size=32):
    """Same ordered batches as capture; no outcome-dependent example filtering."""
    device=next(model.parameters()).device
    boundary=model.model.layers[layer] if layer<len(model.model.layers) else model.model.norm
    logits,top=[],[]
    for start in range(0,len(rows),batch_size):
        encoded,positions=encode_calendar_batch(tokenizer,rows[start:start+batch_size],device)
        out=forward_with_residual_replacement(model,boundary,encoded,positions,replacements[start:start+batch_size])
        last=out[:,-1].float();logits.append(last[:,calendar_ids].cpu());top.append(last.argmax(-1).cpu())
    return {'calendar_logits':t.cat(logits),'top_token_ids':t.cat(top)}


def calendar_outcome_rows(rows,measured,calendar_ids,tokenizer,shift,condition,draw=None):
    """Retain all calendar logits and full-vocabulary winners for independent scoring."""
    count=len(calendar_ids);logits=measured['calendar_logits'];top=measured['top_token_ids']
    if logits.shape!=(len(rows),count) or top.shape!=(len(rows),):
        raise ValueError('Measured predictions must contain every requested row exactly once.')
    original=t.tensor([row['answer'] for row in rows]);desired=(original+shift)%count
    # The pure tensor scorer handles both vocabularies. The compact archive retains
    # full-vocabulary winners separately, so a calendar-only argmax cannot hide them.
    scores=score_calendar_logits(logits,original,desired,t.arange(count))
    result=[]
    for i,row in enumerate(rows):
        result.append({'row_id':row['id'],'family':row['family'],'template_index':row['template_index'],
                       'start':row['start'],'offset':row['offset'],'shift':shift,'condition':condition,'draw':draw,
                       'original_answer':int(original[i]),'desired_answer':int(desired[i]),
                       'restricted_answer':int(scores['restricted_token_ids'][i]),'top_token_id':int(top[i]),
                       'top_token_text':tokenizer.decode(int(top[i])),'calendar_logits':logits[i].tolist(),
                       'desired_minus_original_logit':float(scores['desired_minus_original_logit'][i]),
                       'original_full_correct':bool(top[i]==calendar_ids[original[i]]),
                       'desired_full_correct':bool(top[i]==calendar_ids[desired[i]])})
    return result


def fit_calendar_probe(activations,labels,period,components=5):
    mean,basis,variance=fit_pca(activations,components)
    coordinates=circular_coordinates(labels.to(activations.dtype),period)
    zmean,weights,bias=fit_ridge_coordinates((activations-mean)@basis,coordinates)
    return {'mean':mean,'basis':basis,'weights':weights,'bias':bias-zmean@weights,'variance':variance}


def development_calendar_sweep(model,tokenizer,train_rows,dev_rows,train_capture,dev_capture,calendar_ids,batch_size=32):
    """Recompute every development boundary; the published final choice stays frozen."""
    count=len(calendar_ids);train_labels=t.tensor([row['start'] for row in train_rows]);dev_labels=t.tensor([row['start'] for row in dev_rows])
    targets=circular_coordinates(((dev_labels+2)%count).float(),count)
    donors=t.tensor(paired_donor_indices(dev_rows,2,count))
    original=t.tensor([row['answer'] for row in dev_rows]);desired=(original+2)%count;indices=t.arange(len(dev_rows))
    baseline=dev_capture['calendar_logits'];baseline_margin=baseline[indices,desired]-baseline[indices,original]
    curves=[];outcomes=[]
    for boundary in range(train_capture['activations'].shape[1]):
        probe=fit_calendar_probe(train_capture['activations'][:,boundary],train_labels,count,5)
        mean,basis,w,bias=(probe[key] for key in ('mean','basis','weights','bias'))
        receiver=dev_capture['activations'][:,boundary]
        raw=author_raw_circle_replacement(mean,basis,w,targets)
        random=t.randn(receiver.shape,generator=t.Generator().manual_seed(1101+boundary))
        random=random/random.norm(dim=-1,keepdim=True)*(raw-receiver).norm(dim=-1,keepdim=True)
        changes={'pure_coordinate':replace_coordinates(receiver,mean,basis,w,bias,targets),
                 'centered_mean_ablated':mean_ablate_coordinates(mean,basis,w,bias,targets),
                 'author_raw_mean_ablated':raw,'full_donor':receiver[donors],
                 'mean_only':mean.expand_as(receiver),'equal_norm_random_edit':receiver+random}
        point={'boundary':boundary,'conditions':{}}
        for name,value in changes.items():
            measured=measure_calendar_replacements(model,tokenizer,dev_rows,calendar_ids,boundary,value,batch_size)
            saved=calendar_outcome_rows(dev_rows,measured,calendar_ids,tokenizer,2,name)
            for row in saved:row['layer_input']=boundary
            outcomes.extend(saved)
            logits=measured['calendar_logits'];margin=logits[indices,desired]-logits[indices,original]
            point['conditions'][name]={'desired_choice_accuracy':float(logits.argmax(-1).eq(desired).float().mean()),
                                      'mean_logit_effect':float((margin-baseline_margin).mean()),
                                      'desired_full_accuracy':sum(row['desired_full_correct'] for row in saved)/len(saved)}
        curves.append(point)
        if boundary%8==0:print(dev_rows[0]['family'],'development boundary',boundary,'complete',flush=True)
    return curves,outcomes


def run_calendar_study(model,tokenizer,protocol,output_directory):
    """Execute the frozen train/development/final experiment, keeping every control."""
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model.eval();batch_size=protocol['batch_size']
    result={'protocol':protocol,'tasks':{},'outcomes':[],'model_forward_rows':0,'nulls':[]}
    tensors={}
    def checkpoint():
        payload=json.dumps(result,separators=(',',':')).encode()
        (output/'measured_study.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    for family_index,(family,names) in enumerate(NAMES.items()):
        count=len(names)
        tokens=[tokenizer.encode(name,add_special_tokens=False) for name in names]
        if not all(len(ids)==1 for ids in tokens):raise ValueError('Every answer must be one actual model token.')
        choice_ids=t.tensor([ids[0] for ids in tokens])
        if len(choice_ids.unique())!=count:raise ValueError('Calendar classes must have distinct answer tokens.')
        rows={split:calendar_rows(family,split,protocol) for split in ('train','development','final')}
        prompt_inventory=[row['prompt'] for split in rows.values() for row in split]
        if len(set(prompt_inventory))!=len(prompt_inventory):raise ValueError('Prompt splits overlap.')
        captured={split:capture_calendar_rows(model,tokenizer,rows[split],choice_ids,batch_size) for split in ('train','development')}
        development_curves,development_outcomes=development_calendar_sweep(model,tokenizer,rows['train'],rows['development'],captured['train'],captured['development'],choice_ids,batch_size)
        # These fresh curves remain visible; they do not change the already-frozen layer choice.
        captured['final']=capture_calendar_rows(model,tokenizer,rows['final'],choice_ids,batch_size)
        result['outcomes'].extend(development_outcomes)
        result['model_forward_rows']+=sum(len(items) for items in rows.values())+len(development_outcomes)
        train_labels=t.tensor([r['start'] for r in rows['train']]);final_labels=t.tensor([r['start'] for r in rows['final']])
        layer=protocol['selected'][family]['layer_input']
        probe=fit_calendar_probe(captured['train']['activations'][:,layer],train_labels,count,5)
        mean,basis,w,bias=(probe[key] for key in ('mean','basis','weights','bias'))
        final=captured['final']['activations'][:,layer]
        final_targets=circular_coordinates(final_labels.float(),count)
        prediction=(final-mean)@basis@w+bias
        curves=[]
        permutations=t.stack([t.randperm(count,generator=t.Generator().manual_seed(protocol['circular_order_null_seed']+family_index*1000+draw))
                              for draw in range(protocol['circular_order_null_draws'])])
        for boundary in range(captured['train']['activations'].shape[1]):
            train_acts=captured['train']['activations'][:,boundary]
            other=fit_calendar_probe(train_acts,train_labels,count,5)
            pred=(captured['final']['activations'][:,boundary]-other['mean'])@other['basis']@other['weights']+other['bias']
            centroids=t.stack([train_acts[train_labels.eq(i)].mean(0) for i in range(count)])
            xy=(centroids-other['mean'])@other['basis'][:,:2]
            actual,null,p=cyclic_permutation_test(xy,permutations)
            curves.append({'boundary':boundary,'final_coordinate_r2':float(coordinate_r2(final_targets,pred)),
                           'final_angular_accuracy':float(angular_predictions(pred,count).eq(final_labels).float().mean()),
                           'training_pca2_order_agreement':float(actual),'order_null':null.tolist(),'order_p':float(p),
                           'training_variance':other['variance'].tolist()})
        leave_out=[]
        for heldout_class in range(count):
            keep=train_labels.ne(heldout_class);target_rows=final_labels.eq(heldout_class)
            other=fit_calendar_probe(captured['train']['activations'][keep,layer],train_labels[keep],count,5)
            pred=(final[target_rows]-other['mean'])@other['basis']@other['weights']+other['bias']
            leave_out.append({'class':heldout_class,'rows':int(target_rows.sum()),'predicted_coordinates':pred.tolist(),
                              'angular_correct':int(angular_predictions(pred,count).eq(heldout_class).sum()),
                              'mean_coordinate_squared_error':float((pred-final_targets[target_rows]).square().sum(-1).mean())})
        task={'names':list(names),'calendar_token_ids':choice_ids.tolist(),'rows':rows,'layer_input':layer,'layer_sweep':curves,
              'leave_one_class_out':leave_out,'final_coordinate_r2':float(coordinate_r2(final_targets,prediction)),
              'final_coordinate_predictions':prediction.tolist(),'final_angular_accuracy':float(angular_predictions(prediction,count).eq(final_labels).float().mean()),
              'training_pca_variance':probe['variance'].tolist(),'development_interventions':development_curves}
        result['tasks'][family]=task
        shared={'baseline':captured['final']}
        for name,value in [('no_op',final),('mean_only',mean.expand_as(final))]:
            shared[name]=measure_calendar_replacements(model,tokenizer,rows['final'],choice_ids,layer,value,batch_size)
            result['model_forward_rows']+=len(final)
        if not t.equal(shared['baseline']['calendar_logits'],shared['no_op']['calendar_logits']) or not t.equal(shared['baseline']['top_token_ids'],shared['no_op']['top_token_ids']):
            raise ValueError('No-op residual replacement changed the actual baseline.')
        task['no_op_max_logit_error']=0.
        for split in ('train','development'):
            result['outcomes'].extend(calendar_outcome_rows(rows[split],captured[split],choice_ids,tokenizer,2,split+'_baseline'))
        primary=None
        for shift in range(1,count):
            donors=t.tensor(paired_donor_indices(rows['final'],shift,count))
            target=circular_coordinates(((final_labels+shift)%count).float(),count)
            replacements={
                'pure_coordinate':replace_coordinates(final,mean,basis,w,bias,target),
                'centered_mean_ablated':mean_ablate_coordinates(mean,basis,w,bias,target),
                'author_raw_mean_ablated':author_raw_circle_replacement(mean,basis,w,target),
                'full_donor':final[donors],
                'pca5_donor':replace_subspace(final,final[donors],basis),
            }
            for name,value in replacements.items():
                measured=measure_calendar_replacements(model,tokenizer,rows['final'],choice_ids,layer,value,batch_size)
                result['model_forward_rows']+=len(final)
                result['outcomes'].extend(calendar_outcome_rows(rows['final'],measured,choice_ids,tokenizer,shift,name))
            for name,measured in shared.items():
                result['outcomes'].extend(calendar_outcome_rows(rows['final'],measured,choice_ids,tokenizer,shift,name))
            if shift==protocol['primary_control_donor_shift']:
                primary={'donors':donors,'target':target,'delta_norm':(replacements['author_raw_mean_ablated']-final).norm(dim=-1,keepdim=True)}
            checkpoint();print(family,'all-donor shift',shift,'complete',flush=True)
        shift=protocol['primary_control_donor_shift']
        for draw in range(protocol['null_draws']):
            seed=protocol['null_seed']+family_index*10000+draw
            random_plane=t.linalg.qr(t.randn(final.shape[1],2,generator=t.Generator().manual_seed(seed)),mode='reduced').Q
            noise=t.randn(final.shape,generator=t.Generator().manual_seed(seed+1000));noise=noise/noise.norm(dim=-1,keepdim=True)*primary['delta_norm']
            permutation=t.randperm(count,generator=t.Generator().manual_seed(seed+2000))
            perm_targets=circular_coordinates(permutation[train_labels].float(),count)
            zmean,perm_w,perm_b=fit_ridge_coordinates((captured['train']['activations'][:,layer]-mean)@basis,perm_targets)
            desired_perm=circular_coordinates(permutation[(final_labels+shift)%count].float(),count)
            changes={'same_rank_random_donor_plane':replace_subspace(final,final[primary['donors']],random_plane),
                     'equal_norm_random_edit':final+noise,
                     'class_order_permuted_circular_probe':author_raw_circle_replacement(mean,basis,perm_w,desired_perm)}
            result['nulls'].append({'family':family,'draw':draw,'seed':seed,'class_order_permutation':permutation.tolist(),
                                   'random_plane_sha256':hashlib.sha256(random_plane.numpy().tobytes()).hexdigest()})
            for name,value in changes.items():
                measured=measure_calendar_replacements(model,tokenizer,rows['final'],choice_ids,layer,value,batch_size)
                result['model_forward_rows']+=len(final)
                result['outcomes'].extend(calendar_outcome_rows(rows['final'],measured,choice_ids,tokenizer,shift,name,draw))
            if draw%8==7:checkpoint();print(family,'null draw',draw+1,'complete',flush=True)
        tensors[family]={'captured':captured,'probe':probe,'rows':rows,'calendar_token_ids':choice_ids}
        t.save(tensors,output/'actual_activations_and_probes.pt');checkpoint()
    result['complete']=True
    result['retained_outcomes']=len(result['outcomes'])
    result['final_base_rows']=sum(len(task['rows']['final']) for task in result['tasks'].values())
    result['all_donor_pairs']=sum(len(task['rows']['final'])*(len(task['names'])-1) for task in result['tasks'].values())
    if result['final_base_rows']!=protocol['final_base_rows'] or result['all_donor_pairs']!=protocol['all_donor_pairs']:
        raise ValueError('The frozen data inventory did not execute completely.')
    checkpoint()
    return result,tensors


def run_geometry_projection_sweep(study_tensors,protocol,output_directory):
    """Inductive UMAP and explicitly transductive t-SNE, including every null fit."""
    import umap
    from sklearn.manifold import TSNE
    records=[];references={}
    for family_index,(family,task) in enumerate(study_tensors.items()):
        layer=protocol['selected'][family]['layer_input']
        actual_train=task['captured']['train']['activations'][:,layer]
        actual_final=task['captured']['final']['activations'][:,layer]
        ytrain=t.tensor([row['start'] for row in task['rows']['train']]);yfinal=t.tensor([row['start'] for row in task['rows']['final']])
        mean=actual_train.mean(0)
        noise=t.randn(len(actual_train)+len(actual_final),actual_train.shape[-1],generator=t.Generator().manual_seed(3300+family_index))
        actual=t.cat([actual_train,actual_final])-mean
        noise=noise/noise.norm(dim=-1,keepdim=True)*actual.norm(dim=-1,keepdim=True)
        for control,train,final in [('real',actual_train,actual_final),('independent_noise',noise[:len(actual_train)],noise[len(actual_train):])]:
            center,axes,_=fit_pca(train,min(20,len(train)-1))
            train_input=((train-center)@axes).numpy();final_input=((final-center)@axes).numpy()
            high=t.cat([train,final])
            for reducer in ('umap','tsne'):
                settings=protocol['umap_settings'] if reducer=='umap' else [{'perplexity':p} for p in protocol['tsne_perplexities']]
                for setting_index,setting in enumerate(settings):
                    for seed in protocol['reducer_seeds']:
                        if reducer=='umap':
                            method=umap.UMAP(n_components=2,**setting,metric='euclidean',random_state=seed,transform_seed=seed,n_jobs=1)
                            train_low=t.tensor(method.fit_transform(train_input),dtype=t.float32)
                            final_low=t.tensor(method.transform(final_input),dtype=t.float32)
                            scope='inductive: fit training only, transform final'
                        else:
                            joined=t.cat([t.tensor(train_input),t.tensor(final_input)]).numpy()
                            method=TSNE(n_components=2,**setting,init=protocol['tsne_init'],learning_rate='auto',
                                        max_iter=protocol['tsne_iterations'],random_state=seed,metric='euclidean',n_jobs=1)
                            low=t.tensor(method.fit_transform(joined),dtype=t.float32)
                            train_low,final_low=low[:len(train)],low[len(train):]
                            scope='transductive: unlabeled final points included in fitting'
                        prediction=knn_predictions(train_low,ytrain,final_low,3)
                        shuffle=t.randperm(len(ytrain),generator=t.Generator().manual_seed(4400+family_index*100+seed*10+setting_index))
                        shuffled_prediction=knn_predictions(train_low,ytrain[shuffle],final_low,3)
                        trust,continuity=neighborhood_scores(high,t.cat([train_low,final_low]),5)
                        key=(family,control,reducer,setting_index)
                        if key not in references:references[key]=(train_low.clone(),final_low.clone())
                        ref_train,ref_final=references[key]
                        source_mean,target_mean,rotation=fit_procrustes(train_low,ref_train)
                        aligned=(final_low-source_mean)@rotation+target_mean
                        normalized_error=float((aligned-ref_final).square().mean().sqrt()/ref_final.std().clamp_min(1e-8))
                        records.append({'family':family,'control':control,'reducer':reducer,'scope':scope,'setting':setting,
                                        'setting_index':setting_index,'seed':seed,'train_projection':train_low.tolist(),
                                        'final_projection':final_low.tolist(),'final_knn_accuracy':float(prediction.eq(yfinal).float().mean()),
                                        'permuted_training_labels_accuracy':float(shuffled_prediction.eq(yfinal).float().mean()),
                                        'trustworthiness':float(trust),'continuity':float(continuity),
                                        'procrustes_final_error_relative_to_seed0':normalized_error,
                                        'final_predictions':prediction.tolist(),'permuted_training_label_predictions':shuffled_prediction.tolist()})
                    print(family,control,reducer,'setting',setting_index,'complete',flush=True)
    path=Path(output_directory)/'projection_study.json.gz'
    path.write_bytes(gzip.compress(json.dumps(records,separators=(',',':')).encode(),mtime=0))
    return records


def run_calendar_angle_play(model,tokenizer,study_tensors,protocol,output_directory):
    """A fixed source per family; sweep continuous coordinates with paired controls."""
    records=[]
    for family_index,(family,task) in enumerate(study_tensors.items()):
        row=task['rows']['final'][0];count=len(NAMES[family]);probe=task['probe']
        mean,basis,w,bias=(probe[key] for key in ('mean','basis','weights','bias'))
        layer=protocol['selected'][family]['layer_input']
        receiver=task['captured']['final']['activations'][0,layer]
        angles=t.arange(count*protocol['angle_play']['steps_per_class'])*2*t.pi/(count*protocol['angle_play']['steps_per_class'])
        radii=t.tensor(protocol['angle_play']['radii'])
        radius_grid,angle_grid=t.meshgrid(radii,angles,indexing='ij')
        target=radius_grid.flatten()[:,None]*t.stack([angle_grid.flatten().cos(),angle_grid.flatten().sin()],-1)
        receivers=receiver.expand(len(target),-1)
        raw=author_raw_circle_replacement(mean,basis,w,target)
        # One fixed direction across the grid: only its matched magnitude changes.
        direction=t.randn(receiver.shape,generator=t.Generator().manual_seed(5500+family_index))
        direction=direction/direction.norm()
        replacements={'pure_coordinate':replace_coordinates(receivers,mean,basis,w,bias,target),
                      'author_raw_mean_ablated':raw,
                      'equal_norm_random_edit':receivers+(raw-receivers).norm(dim=-1,keepdim=True)*direction}
        shared={}
        for name,value in [('baseline',receiver[None]),('mean_only',mean[None])]:
            measured=measure_calendar_replacements(model,tokenizer,[row],task['calendar_token_ids'],layer,value,protocol['batch_size'])
            shared[name]={'calendar_logits':measured['calendar_logits'][0].tolist(),
                          'top_token_id':int(measured['top_token_ids'][0]),'top_token_text':tokenizer.decode(int(measured['top_token_ids'][0]))}
        for condition,value in replacements.items():
            measured=measure_calendar_replacements(model,tokenizer,[row]*len(target),task['calendar_token_ids'],layer,value,protocol['batch_size'])
            records.append({'family':family,'row':row,'condition':condition,'layer_input':layer,
                            'radii':radii.tolist(),'angles':angles.tolist(),'shared':shared,
                            'calendar_logits':measured['calendar_logits'].tolist(),
                            'top_token_ids':measured['top_token_ids'].tolist(),
                            'top_token_text':[tokenizer.decode(int(token)) for token in measured['top_token_ids']]})
        print(family,'continuous angle/radius play complete',flush=True)
    (Path(output_directory)/'angle_play.json.gz').write_bytes(gzip.compress(json.dumps(records,separators=(',',':')).encode(),mtime=0))
    return records
