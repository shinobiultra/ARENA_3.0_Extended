"""Independent development candidate; never loads final examples or publishes a lesson."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch as t
from tokenizers import Tokenizer
from chapter5_modern_architectures.exercises.part5_diffusion_language_models import text_data as data
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.text_model import TextTransformer
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.text_training import prepare_text_tensors,fit_text_model
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.text_evaluation import generate_text_examples

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--updates',type=int,default=1500)
    parser.add_argument('--development-every',type=int,default=300)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    output=args.output_directory.resolve();output.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((args.data_directory/'provenance.json').read_text())
    rows={}
    for split in ('train','development'):
        path=args.data_directory/f'{split}.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=provenance['inventory'][split]['sha256']:
            raise ValueError('dataset file differs from its frozen provenance')
        rows[split]=json.loads(path.read_text())
    tokenizer_path=args.data_directory/'tokenizer.json'
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()!=provenance['tokenizer_sha256']:
        raise ValueError('tokenizer differs from frozen data provenance')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    source_files=[*sorted(HERE.glob('text_*.py')),Path(__file__).resolve()]
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for path in source_files:(snapshot/path.name).write_bytes(path.read_bytes())
    protocol={'scope':'development candidate only','seed':args.seed,'model_width':64,'model_layers':2,'heads':4,
        'vocabulary':768,'context_slots':40,'target_slots':24,'updates':args.updates,'batch_size':32,
        'learning_rate':.002,'weight_decay':.01,'gradient_clip':1.,'development_every':args.development_every,
        'diffusion_training_steps':16,'development_generation_sampler':'confidence',
        'development_generation_steps':16,'temperatures':[0.,0.],
        'modes':['diffusion','diffusion-shuffled','autoregressive','autoregressive-shuffled'],
        'final_data_loaded':False,'data_provenance_sha256':hashlib.sha256((args.data_directory/'provenance.json').read_bytes()).hexdigest(),
        'input_sources':source_hashes,'dtype':'float32','device':args.device,
        'selection':'earliest minimum of each model kind\'s own fixed development objective'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'data_provenance.json').write_bytes((args.data_directory/'provenance.json').read_bytes())
    t.set_num_threads(4);t.set_float32_matmul_precision('highest')
    if args.device.startswith('cuda'):
        if not t.cuda.is_available():raise RuntimeError('CUDA unavailable')
        t.cuda.reset_peak_memory_stats()
    training=prepare_text_tensors(rows['train'],device=args.device)
    development=prepare_text_tensors(rows['development'],device=args.device)
    def factory():return TextTransformer(768,width=64,heads=4,layers=2,max_length=64)
    trace_ids=[row['id'] for task in ('parentheses','arithmetic','qa','story')
               for row in [r for r in rows['development'] if r['task']==task][:3]]
    state={'status':'running','pid':os.getpid(),'started':time.time(),'models':[]}
    state_path=output/'status.json'
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    for kind,shuffled in [('diffusion',False),('diffusion',True),('autoregressive',False),('autoregressive',True)]:
        label=kind+('-shuffled' if shuffled else '')
        directory=output/label;directory.mkdir();state['active_model']=label;save()
        def checkpoint(step,model,history):
            t.save(model.state_dict(),directory/f'checkpoint-{step}.pt')
            print(label,step,history,flush=True)
            state['last_checkpoint']=history;save()
        model,record=fit_text_model(factory,training,development,kind=kind,shuffled=shuffled,seed=args.seed,
            mask_id=data.MASK_ID,total_steps=16,updates=args.updates,batch_size=32,learning_rate=.002,
            weight_decay=.01,development_every=args.development_every,checkpoint_callback=checkpoint)
        t.save(model.state_dict(),directory/'selected.pt')
        (directory/'fit.json').write_text(json.dumps(record,indent=2)+'\n')
        result=generate_text_examples(model,rows['development'],tokenizer,kind=kind,seed=1000,
            sampler='confidence',steps=16,temperatures=(0.,0.),batch_size=32,trace_ids=trace_ids)
        (directory/'development_generation.json').write_text(json.dumps(result,indent=2)+'\n')
        state['models'].append({'label':label,'selected_step':record['selected_step'],
            'selected_development_objective':record['selected_development_objective'],'generation':result['summary']})
        print(label,json.dumps(result['summary']),flush=True);save()
        del model
    changed=[name for name,digest in source_hashes.items() if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest]
    state.update(status='COMPLETE',sources_changed=changed,final_data_loaded=False,
        elapsed_seconds=time.time()-state['started'],peak_vram_gb=t.cuda.max_memory_allocated()/1024**3 if args.device.startswith('cuda') else None)
    save()
    if changed:raise RuntimeError('Candidate sources changed while training; keep it explicitly scoped')


if __name__=='__main__':main()
