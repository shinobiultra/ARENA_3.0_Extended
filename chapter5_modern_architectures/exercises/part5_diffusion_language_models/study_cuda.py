"""Separate fresh CUDA refit and generation, after the full own-answer learner.

Checks all four modes at seed zero, all 42 frozen generation configurations,
and the selected and final-update weight states. No learner weights are used
for generation: they are read only as comparison targets after fresh fitting.
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

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch as t
from tokenizers import Tokenizer
from chapter5_modern_architectures.exercises.part5_diffusion_language_models import text_data as data
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.text_model import TextTransformer
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.text_training import fit_text_model,prepare_text_tensors
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.text_evaluation import generate_text_examples
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.study_evidence import validate_text_study,sha256

HERE=Path(__file__).resolve().parent


def compare_values(actual,expected,*,path='result',tolerance=1e-4):
    """All tokens, labels and inventories exact; finite numerical traces bounded."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():raise ValueError(f'Field inventory differs at {path}')
        return max((compare_values(actual[key],value,path=f'{path}.{key}',tolerance=tolerance) for key,value in expected.items()),default=0.)
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError(f'List inventory differs at {path}')
        return max((compare_values(a,e,path=f'{path}[{i}]',tolerance=tolerance) for i,(a,e) in enumerate(zip(actual,expected))),default=0.)
    if isinstance(expected,float):
        if not isinstance(actual,(float,int)) or not t.isfinite(t.tensor([actual,expected])).all():
            raise ValueError(f'Nonfinite numeric result at {path}')
        error=abs(actual-expected)
        if error>tolerance:raise ValueError(f'Numeric mismatch at {path}: {error}')
        return error
    if actual!=expected or type(actual)!=type(expected):raise ValueError(f'Exact outcome differs at {path}')
    return 0.


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--learner-directory',type=Path,required=True)
    parser.add_argument('--data-directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_directory.resolve();output.mkdir(parents=True,exist_ok=False)
    state={'status':'RUNNING','pid':os.getpid(),'started':time.time(),'fresh_training_runs':0,'fresh_generated_examples':0,
           'fresh_generation_conditions':0,'learner_directory':str(args.learner_directory),'all_fresh_outcomes_match_learner':False}
    def save(): (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    sources={p.name:sha256(p) for p in HERE.glob('*.py')}
    state['input_sources']=sources;save()
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in sources:(snapshot/name).write_bytes((HERE/name).read_bytes())
    try:
        state['offline_learner_audit']=validate_text_study(args.learner_directory,args.data_directory);save()
        if not t.cuda.is_available():raise RuntimeError('Fresh CUDA check requires a GPU')
        t.set_num_threads(4);t.set_float32_matmul_precision('highest');t.cuda.reset_peak_memory_stats()
        device=t.device('cuda')
        learner=json.loads(gzip.decompress((args.learner_directory/'experiments/completed_text_study.json.gz').read_bytes()))
        rows={split:json.loads((args.data_directory/f'{split}.json').read_text()) for split in ('train','development','final')}
        tokenizer=Tokenizer.from_file(str(args.data_directory/'tokenizer.json'))
        train=prepare_text_tensors(rows['train'],device=device);dev=prepare_text_tensors(rows['development'],device=device)
        def factory():return TextTransformer(768,width=64,heads=4,layers=2,max_length=64)
        fresh_states={};weight_errors=[];raw_files={}
        for kind,shuffled in [('diffusion',False),('diffusion',True),('autoregressive',False),('autoregressive',True)]:
            label=kind+('-shuffled' if shuffled else '')
            directory=output/label;directory.mkdir();state['active_fit']=label;save()
            def checkpoint(step,model,history):
                t.save(model.state_dict(),directory/f'checkpoint-{step}.pt')
                print(label,step,history['development_objective'],flush=True)
            model,fit=fit_text_model(factory,train,dev,kind=kind,shuffled=shuffled,seed=0,mask_id=data.MASK_ID,
                total_steps=16,updates=12000,batch_size=32,learning_rate=.002,weight_decay=.01,
                development_every=1000,checkpoint_callback=checkpoint)
            t.save(model.state_dict(),directory/'selected.pt')
            expected_fit=next(r for r in learner['fits'] if r['seed']==0 and r['kind']==kind and r['shuffled']==shuffled)
            for key in ['selected_step','initial_parameters_sha256','batch_indices_sha256','development_steps_sha256','development_noisy_sha256','target_permutation','preview_batches']:
                if fit[key]!=expected_fit[key]:raise ValueError(f'Fresh fit differs for {label}: {key}')
            for choice,filename in [('selected','selected.pt'),('endpoint','checkpoint-12000.pt')]:
                fresh=t.load(directory/filename,weights_only=True,map_location='cpu')
                expected=t.load(args.learner_directory/'experiments/weights'/f'seed0-{label}'/filename,weights_only=True,map_location='cpu')
                if fresh.keys()!=expected.keys():raise ValueError('Weight parameter inventories differ')
                error=max(float((fresh[key]-expected[key]).abs().max()) for key in fresh)
                if error>1e-4:raise ValueError(f'Fresh weight error exceeds tolerance: {label}, {choice}, {error}')
                fresh_states[label,choice]=fresh
                weight_errors.append({'label':label,'checkpoint':choice,'max_abs_error':error})
            payload=json.dumps(fit,indent=2).encode();name=f'{label}-fresh-fit.json.gz'
            (output/name).write_bytes(gzip.compress(payload,mtime=0));raw_files[name]=sha256(output/name)
            state['fresh_training_runs']+=1;state['weight_errors']=weight_errors;save();del model
        expected_runs=[r for r in learner['results'] if r['seed']==0]
        if len(expected_runs)!=42:raise ValueError('Require every seed-zero final configuration')
        trace_ids=[row['id'] for task in ('parentheses','arithmetic','qa','story') for row in [r for r in rows['final'] if r['task']==task][:2]]
        max_trace_error=0.
        for expected in expected_runs:
            label=expected['label'];choice=expected['checkpoint'];kind=expected['kind'];config=expected['config']
            with t.random.fork_rng(devices=[0]):
                t.manual_seed(0);model=factory().to(device)
            if choice!='untrained':model.load_state_dict(fresh_states[label,choice])
            model.eval()
            actual=generate_text_examples(model,rows['final'],tokenizer,kind=kind,seed=1000,
                sampler=config.get('sampler','ancestral'),steps=config.get('steps',16),temperatures=tuple(config['temperatures']),
                block_size=config.get('block_size'),batch_size=32,trace_ids=trace_ids)
            # All individual decoded outcomes and metrics must match exactly.
            if actual['rows']!=expected['result']['rows'] or actual['summary']!=expected['result']['summary']:
                raise ValueError(f'Fresh actual words or scores differ: {label}, {choice}, {config["name"]}')
            max_trace_error=max(max_trace_error,compare_values(actual,expected['result']))
            name=f'{label}-{choice}-{config["name"]}.json.gz'
            (output/name).write_bytes(gzip.compress(json.dumps(actual).encode(),mtime=0));raw_files[name]=sha256(output/name)
            state['fresh_generation_conditions']+=1;state['fresh_generated_examples']+=len(rows['final']);save()
            print('Fresh generation match',label,choice,config['name'],flush=True);del model
        state.update(status='PASS',all_fresh_outcomes_match_learner=True,max_trace_numeric_error=max_trace_error,
                     max_weight_error=max(r['max_abs_error'] for r in weight_errors),raw_files_sha256=raw_files,
                     device=t.cuda.get_device_name(0),torch_version=t.__version__,cuda_version=t.version.cuda,
                     peak_vram_gb=t.cuda.max_memory_allocated()/1024**3,comparison_tolerance=1e-4)
    except BaseException as error:
        state.update(status='FAIL',error_type=type(error).__name__,error=str(error));raise
    finally:
        changed=[name for name,digest in sources.items() if sha256(HERE/name)!=digest]
        state.update(elapsed_seconds=time.time()-state['started'],sources_changed_during_run=changed)
        if changed:state['status']='FAIL'
        save()
    print(json.dumps({k:state[k] for k in ['status','fresh_training_runs','fresh_generation_conditions','fresh_generated_examples','max_weight_error','max_trace_numeric_error','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
