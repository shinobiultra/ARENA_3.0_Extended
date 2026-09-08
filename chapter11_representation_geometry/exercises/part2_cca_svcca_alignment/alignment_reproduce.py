"""Fresh CPU model/fit/native-hook reproduction of both ends of the final split."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch as t
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_evidence import validate_learner,sha256,HERE
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_ioi_data import paired_rows,donor_indices
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_model import capture_rows,replace_rows
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.run_alignment_development import fit_candidates


def reproduce(learner_directory,output_directory):
    run=Path(learner_directory).resolve();output=Path(output_directory).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.time();t.set_num_threads(4)
    names=['alignment_reproduce.py','alignment_evidence.py','alignment_methods.py','alignment_model.py','alignment_ioi_data.py','alignment_data.py','run_alignment_development.py','alignment_protocol.json']
    snapshot=output/'source_snapshot';snapshot.mkdir()
    inputs={name:sha256(HERE/name) for name in names}
    for name in names:(snapshot/name).write_bytes((HERE/name).read_bytes())
    state={'status':'RUNNING','source_inputs':inputs,'learner_directory':str(run),'device':'cpu','conditions':[]}
    def save():(output/'fresh_reproduction.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        audit=validate_learner(run);state['learner_audit']=audit;save()
        protocol=json.loads((HERE/'alignment_protocol.json').read_text())
        tokenizer=AutoTokenizer.from_pretrained(protocol['model_id'],revision=protocol['revision'],local_files_only=True)
        tokenizer.padding_side='left';tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(protocol['model_id'],revision=protocol['revision'],local_files_only=True,
                                                dtype=t.float32,attn_implementation='eager').cpu().eval()
        train_rows=paired_rows('train');final_rows=paired_rows('final')
        train=capture_rows(model,tokenizer,train_rows,position_mode='prediction')
        final=capture_rows(model,tokenizer,final_rows,position_mode='prediction')
        expected=t.load(run/'actual_alignment_study/actual_tensors.pt',map_location='cpu',weights_only=False)
        selection=t.tensor([*range(24),*range(72,96)]);selected_rows=[final_rows[i] for i in selection.tolist()]
        donors=t.tensor(donor_indices(final_rows));fresh={};fit_errors={};maximum=0.;mismatches=0;count=0
        def compare(key,name,measured):
            nonlocal maximum,mismatches,count
            reference=expected['logits'][key][name][selection]
            error=float((measured-reference).abs().max());wrong=int(measured.argmax(-1).ne(reference.argmax(-1)).sum())
            maximum=max(maximum,error);mismatches+=wrong;count+=len(measured)
            state['conditions'].append({'pair':key,'condition':name,'rows':len(measured),'max_logit_error':error,'top_token_mismatches':wrong})
            fresh[key][name]=measured
        for pair_index,(source_layer,target_layer) in enumerate(protocol['layer_pairs']):
            key=f'{source_layer}_to_{target_layer}';fresh[key]={}
            candidates=fit_candidates(train['activations'][:,source_layer].double(),train['activations'][:,target_layer].double())
            reference_candidates={c['name']:c for c in expected['fits'][key]}
            fit_errors[key]={c['name']:float((c['map']-reference_candidates[c['name']]['map']).abs().max()) for c in candidates}
            source=final['activations'][:,source_layer].double();target=final['activations'][:,target_layer].double()
            delta=source[donors]-source
            compare(key,'baseline',final['logits'][selection])
            for name,replacements in [('no_op',target),('native_donor',target[donors])]:
                measured=replace_rows(model,tokenizer,selected_rows,target_layer,replacements[selection],position_mode='prediction')
                compare(key,name,measured)
            for candidate in candidates:
                replacements=target+delta@candidate['map']
                measured=replace_rows(model,tokenizer,selected_rows,target_layer,replacements[selection],position_mode='prediction')
                compare(key,candidate['name'],measured)
            chosen=next(c for c in candidates if c['name']==protocol['selected'][key]['cca'])
            norm=(delta@chosen['map']).norm(dim=-1,keepdim=True)
            for draw in range(protocol['null_draws']):
                direction=t.randn(target.shape,generator=t.Generator().manual_seed(protocol['null_seed']+pair_index*100+draw),dtype=t.float64)
                replacements=target+direction/direction.norm(dim=-1,keepdim=True)*norm
                measured=replace_rows(model,tokenizer,selected_rows,target_layer,replacements[selection],position_mode='prediction')
                compare(key,f'random_{draw}',measured)
            save();print(key,'fresh model, fits, native donor and all random controls compared',flush=True)
        t.save({'train':train,'final':final,'selection':selection,'logits':fresh},output/'actual_fresh_tensors.pt')
        state.update(compared_rows=count,forward_rows=72+96+35*48*3,compared_conditions=len(state['conditions']),
                     selected_final_indices=selection.tolist(),fit_errors=fit_errors,maximum_logit_error=maximum,top_token_mismatches=mismatches,
                     actual_fresh_tensors_sha256=sha256(output/'actual_fresh_tensors.pt'))
        if maximum>1e-5 or mismatches or count!=5184:raise ValueError('Fresh native comparison failed; retain differences before changing any acceptance bound.')
        state['status']='PASS'
    except BaseException as error:state.update(status='FAIL',error=str(error));raise
    finally:
        state.update(elapsed_seconds=time.time()-started,changed_sources=[name for name,digest in inputs.items() if sha256(HERE/name)!=digest])
        if state['changed_sources']:state['status']='FAIL'
        save()
    return state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--learner-directory',type=Path,required=True);parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(reproduce(args.learner_directory,args.output_directory),indent=2))


if __name__=='__main__':main()
