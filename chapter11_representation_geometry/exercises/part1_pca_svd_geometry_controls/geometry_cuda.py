"""Fresh native-model reproduction on the first unchanged batch of both families.

All 32 draws of every null are rebuilt; no teacher-forced expected logits enter
model calls. The full learner already evaluates every prompt/donor pair.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch as t

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_data import NAMES,calendar_rows,paired_donor_indices,load_protocol
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_methods import (
    circular_coordinates,fit_ridge_coordinates,replace_coordinates,mean_ablate_coordinates,author_raw_circle_replacement,replace_subspace,
)
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_study import capture_calendar_rows,fit_calendar_probe,measure_calendar_replacements,calendar_outcome_rows
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_evidence import read_gzip,sha256,validate_learner


def reproduce(learner_directory,output_directory):
    run=Path(learner_directory).resolve();output=Path(output_directory).resolve();output.mkdir(parents=True,exist_ok=False)
    audit=validate_learner(run);protocol=load_protocol();source=read_gzip(run/'experiments/measured_study.json.gz')
    names=['geometry_cuda.py','geometry_evidence.py','geometry_study.py','geometry_methods.py','geometry_data.py','geometry_protocol.json']
    inputs={name:sha256(Path(__file__).parent/name) for name in names}
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in names:(snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    reference={(r['family'],r['row_id'],r['condition'],r['draw']):r for r in source['outcomes'] if r['row_id'].startswith('final/') and r['shift']==2}
    state={'status':'RUNNING','source_inputs':inputs,'learner_audit':audit,'condition_batches':0,'compared_examples':0,
           'max_calendar_logit_error':0.,'top_token_mismatches':0,'comparisons':[]}
    def save():(output/'reproduction.json').write_text(json.dumps(state,indent=2)+'\n')
    save();started=time.perf_counter()
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer
        t.set_num_threads(4)
        cache=os.environ.get('ARENA_GEOMETRY_MODEL_CACHE')
        tokenizer=AutoTokenizer.from_pretrained(protocol['model_id'],revision=protocol['revision'],cache_dir=cache)
        tokenizer.padding_side='left';tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(protocol['model_id'],revision=protocol['revision'],cache_dir=cache,
                                                dtype=t.bfloat16,attn_implementation=protocol['attention']).to('cuda').eval()
        t.cuda.reset_peak_memory_stats();actual=[]
        for family_index,(family,names_) in enumerate(NAMES.items()):
            n=len(names_);layer=protocol['selected'][family]['layer_input'];batch_size=protocol['batch_size']
            train_rows=calendar_rows(family,'train',protocol);rows=calendar_rows(family,'final',protocol)
            choice_ids=t.tensor([tokenizer.encode(name,add_special_tokens=False)[0] for name in names_])
            train=capture_calendar_rows(model,tokenizer,train_rows,choice_ids,batch_size)
            final_capture=capture_calendar_rows(model,tokenizer,rows,choice_ids,batch_size)
            train_labels=t.tensor([r['start'] for r in train_rows]);labels=t.tensor([r['start'] for r in rows])
            probe=fit_calendar_probe(train['activations'][:,layer],train_labels,n,5)
            mean,basis,w,bias=(probe[key] for key in ('mean','basis','weights','bias'))
            final=final_capture['activations'][:,layer];donors=t.tensor(paired_donor_indices(rows,2,n));target=circular_coordinates(((labels+2)%n).float(),n)
            raw=author_raw_circle_replacement(mean,basis,w,target)
            replacements={'baseline':final,'no_op':final,'mean_only':mean.expand_as(final),
                          'pure_coordinate':replace_coordinates(final,mean,basis,w,bias,target),
                          'centered_mean_ablated':mean_ablate_coordinates(mean,basis,w,bias,target),
                          'author_raw_mean_ablated':raw,'full_donor':final[donors],'pca5_donor':replace_subspace(final,final[donors],basis)}
            selected_rows=rows[:batch_size]
            def compare(condition,replacement,draw=None):
                measured=measure_calendar_replacements(model,tokenizer,selected_rows,choice_ids,layer,replacement[:batch_size],batch_size)
                saved=calendar_outcome_rows(selected_rows,measured,choice_ids,tokenizer,2,condition,draw)
                expected=[reference[family,r['row_id'],condition,draw] for r in saved]
                error=float((measured['calendar_logits']-t.tensor([r['calendar_logits'] for r in expected])).abs().max())
                mismatch=int(measured['top_token_ids'].ne(t.tensor([r['top_token_id'] for r in expected])).sum())
                state['max_calendar_logit_error']=max(state['max_calendar_logit_error'],error)
                state['top_token_mismatches']+=mismatch;state['condition_batches']+=1;state['compared_examples']+=len(saved)
                state['comparisons'].append({'family':family,'condition':condition,'draw':draw,'examples':len(saved),'max_logit_error':error,'top_token_mismatches':mismatch})
                actual.extend(saved)
            for condition,replacement in replacements.items():compare(condition,replacement)
            delta_norm=(raw-final).norm(dim=-1,keepdim=True)
            for draw in range(protocol['null_draws']):
                seed=protocol['null_seed']+family_index*10000+draw
                plane=t.linalg.qr(t.randn(final.shape[1],2,generator=t.Generator().manual_seed(seed)),mode='reduced').Q
                noise=t.randn(final.shape,generator=t.Generator().manual_seed(seed+1000));noise=noise/noise.norm(dim=-1,keepdim=True)*delta_norm
                permutation=t.randperm(n,generator=t.Generator().manual_seed(seed+2000))
                _,perm_w,_=fit_ridge_coordinates((train['activations'][:,layer]-mean)@basis,circular_coordinates(permutation[train_labels].float(),n))
                desired=circular_coordinates(permutation[(labels+2)%n].float(),n)
                for condition,replacement in [('same_rank_random_donor_plane',replace_subspace(final,final[donors],plane)),
                                               ('equal_norm_random_edit',final+noise),
                                               ('class_order_permuted_circular_probe',author_raw_circle_replacement(mean,basis,perm_w,desired))]:
                    compare(condition,replacement,draw)
                if draw%8==7:save();print(family,'fresh draw',draw+1,'complete',flush=True)
            save()
        (output/'actual_outcomes.json.gz').write_bytes(gzip.compress(json.dumps(actual,separators=(',',':')).encode(),mtime=0))
        state.update(status='PASS',gpu=t.cuda.get_device_name(),peak_cuda_allocated_gib=t.cuda.max_memory_allocated()/2**30)
        if state['condition_batches']!=208 or state['compared_examples']!=6656 or state['max_calendar_logit_error']>1e-6 or state['top_token_mismatches']:
            raise ValueError('Fresh first-batch regeneration differs from the actual learner.')
    except BaseException as error:
        state.update(status='FAIL',error=str(error));raise
    finally:
        state['elapsed_seconds']=time.perf_counter()-started
        state['changed_sources']=[name for name,digest in inputs.items() if sha256(Path(__file__).parent/name)!=digest]
        if state['changed_sources']:state['status']='FAIL'
        save()
    if state['status']!='PASS':raise ValueError('Fresh CUDA inputs changed during execution.')
    return state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--learner-directory',type=Path,required=True);parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();result=reproduce(args.learner_directory,args.output_directory)
    print(json.dumps({key:value for key,value in result.items() if key not in ('comparisons','learner_audit')},indent=2))


if __name__=='__main__':main()
