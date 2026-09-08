"""Reproduce from source in a fresh child, then compare every deterministic leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch as t

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.execute_learner import digest,source_paths,validate_design

PAYLOADS=('fitted','games','plans','estimates','exact','package')


def tree_inventory(value):
    counts={'arrays':0,'array_elements':0,'scalars':0,'containers':0}
    arrays={}
    def visit(item,path):
        if isinstance(item,(t.Tensor,np.ndarray)):
            array=item.detach().cpu().contiguous().numpy() if isinstance(item,t.Tensor) else np.ascontiguousarray(item)
            if array.dtype.hasobject:raise ValueError('Object arrays are not evidence.')
            if not np.isfinite(array).all():raise ValueError('Nonfinite deterministic array: '+path)
            leaf={'dtype':array.dtype.str,'shape':list(array.shape),'sha256':hashlib.sha256(array.tobytes()).hexdigest()}
            arrays[path]=leaf;counts['arrays']+=1;counts['array_elements']+=array.size
            return {'array':leaf}
        if isinstance(item,dict):
            counts['containers']+=1
            return {'dict':[(str(key),visit(child,path+'.'+str(key))) for key,child in item.items()]}
        if isinstance(item,(tuple,list)):
            counts['containers']+=1
            return {type(item).__name__:[visit(child,path+f'[{index}]') for index,child in enumerate(item)]}
        if isinstance(item,np.generic):item=item.item()
        if item is None or isinstance(item,(str,int,float,bool)):
            counts['scalars']+=1
            if isinstance(item,float) and not np.isfinite(item):raise ValueError('Nonfinite scalar: '+path)
            return {type(item).__name__:item}
        raise TypeError(f'Unsupported evidence type {type(item)} at {path}')
    tree=visit(value,'root')
    checksum=hashlib.sha256(json.dumps(tree,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    return {'tree_sha256':checksum,'counts':counts,'arrays':arrays}


def validate_current(learner):
    learner=Path(learner).resolve();validate_design()
    record=json.loads((learner/'execution.json').read_text())
    if record['status']!='PASS' or record['analytic_only'] or record['changed_sources']:
        raise ValueError('A passing complete unchanged-source learner is required.')
    current={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    if current!=record['source_inputs']:raise ValueError('Current learner inputs differ from execution.')
    for name,checksum in current.items():
        if digest(learner/'source_snapshot'/name)!=checksum:raise ValueError('Source snapshot changed: '+name)
    if digest(learner/'completed_learner.ipynb')!=record['completed_learner_sha256']:
        raise ValueError('The executed notebook changed.')
    directory=learner/'figures/native';summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='PASS' or set(summary['artifacts'])!={name+'.pt' for name in PAYLOADS}:
        raise ValueError('Native payload inventory is incomplete.')
    for name,item in summary['artifacts'].items():
        p=directory/name
        if p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
            raise ValueError('Native evidence changed: '+name)
    return record


def run_child(output):
    from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import estimator_methods as methods,estimator_study as study
    validate_design();t.set_num_threads(4)
    study.run_study(methods,output)


def reproduce(learner,output):
    learner,output=Path(learner).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new fresh-reproduction directory.')
    before=validate_current(learner);output.mkdir(parents=True,exist_ok=True)
    for name in before['source_inputs']:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    script_hash=digest(Path(__file__))
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
    env.pop('ARENA_ESTIMATOR_OUTPUT_DIRECTORY',None)
    command=[sys.executable,str(Path(__file__).resolve()),'--child-output',str(output/'fresh_native')]
    start=time.perf_counter()
    with (output/'fresh_child.log').open('w') as log:
        child=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    if child.returncode:raise RuntimeError('Fresh child failed; inspect fresh_child.log.')
    totals={'arrays':0,'array_elements':0,'scalars':0,'containers':0};inventories={};mismatches=[]
    for name in PAYLOADS:
        original=t.load(learner/'figures/native'/f'{name}.pt',map_location='cpu',weights_only=False)
        a=tree_inventory(original);del original
        fresh=t.load(output/'fresh_native'/f'{name}.pt',map_location='cpu',weights_only=False)
        b=tree_inventory(fresh);del fresh
        if a['tree_sha256']!=b['tree_sha256']:mismatches.append(name)
        for key,count in a['counts'].items():totals[key]+=count
        inventories[name]={'original':a,'fresh':b}
    validate_current(learner)
    if digest(Path(__file__))!=script_hash:raise ValueError('Reproduction script changed during execution.')
    record={'status':'FAIL' if mismatches else 'PASS','learner_directory':str(learner),
        'learner_execution_sha256':digest(learner/'execution.json'),'source_inputs':before['source_inputs'],
        'reproduction_script_sha256':script_hash,'fresh_child_command':command,
        'native_models_refit':True,'child_received_learner_payloads':False,'cached_model_inputs_used':False,
        'comparison':'exact_recursive_dtype_shape_value_sha256_including_container_order',
        'counts':totals,'mismatches':mismatches,'elapsed_seconds':time.perf_counter()-start,
        'inventories':inventories}
    (output/'reproduction.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in {'source_inputs','inventories'}},indent=2))
    if mismatches:raise AssertionError('Deterministic payload mismatches: '+str(mismatches))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--learner',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--child-output',type=Path)
    args=parser.parse_args()
    if args.child_output:run_child(args.child_output)
    else:
        if not args.learner or not args.output:parser.error('--learner and --output are required')
        reproduce(args.learner,args.output)
