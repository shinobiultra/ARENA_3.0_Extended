"""Check the executed learner against its actual native-model measurements."""
from __future__ import annotations
import ast,gzip,hashlib,json
from collections import Counter
from pathlib import Path
import torch as t
from .build_notebooks import HERE,lesson_cells,complete_learner
from . import alignment_methods as m

FIGURES=(
 'alignment_known_correlations.png','alignment_noise_overfit.png','alignment_variance_causal_failure.png',
 'alignment_real_correlations.png','alignment_real_transport_curves.png','alignment_real_name_changes.png',
 'alignment_real_token_gallery.png','alignment_real_name_generalization.png','alignment_real_play.png',
)

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition,message):
    if not condition:raise ValueError(message)


def validate_learner(directory):
    run=Path(directory);state=json.loads((run/'execution.json').read_text())
    require(state['status']=='PASS' and state['scope']=='full' and not state['changed_sources'],'A complete unchanged full learner is required.')
    require(state['completed_from_own_answers'],'Learner must complete its own answers.')
    for name,digest in state['source_inputs'].items():
        require(sha256(run/'source_snapshot'/name)==digest,'Snapshot hash mismatch: '+name)
        require(sha256(HERE/name)==digest,'Current learner source differs: '+name)
    expected=complete_learner({'cells':lesson_cells()})['cells']
    expected[1]['source'].append(f'\nassets_dir = Path({str(run/"figures")!r})\nassets_dir.mkdir(exist_ok=True)\n')
    actual=json.loads((run/'completed_learner.ipynb').read_text())['cells']
    require(len(expected)==len(actual),'Executed learner cell count changed.')
    code=0;images=0;definitions={}
    for e,a in zip(expected,actual,strict=True):
        require(e['cell_type']==a['cell_type'] and ''.join(e['source'])==''.join(a['source']),'Executed source does not match the current own-answer learner.')
        if a['cell_type']!='code':continue
        code+=1;require(a['execution_count'] is not None,'A code cell was not run.')
        require(not any(o.get('output_type')=='error' for o in a['outputs']),'Executed notebook contains an error.')
        images+=sum('image/png' in o.get('data',{}) for o in a['outputs'])
        for node in ast.parse(''.join(a['source'])).body:
            if isinstance(node,ast.FunctionDef):definitions[node.name]=ast.dump(node,include_attributes=False)
    for node in ast.parse((HERE/'alignment_methods.py').read_text()).body:
        if isinstance(node,ast.FunctionDef):require(definitions.get(node.name)==ast.dump(node,include_attributes=False),'Reference method differs from own answer: '+node.name)
    require(set(state['figures'])==set(FIGURES) and images==len(FIGURES),'All nine figures must be embedded in the executed learner.')
    protocol=json.loads((HERE/'alignment_protocol.json').read_text())
    development_source=Path(protocol['selection_source'])
    require(sha256(development_source)==protocol['selection_source_sha256'],'Frozen development selection source changed.')
    original_development=json.loads(development_source.read_text())
    for pair in original_development['pairs']:
        key=f"{pair['source_layer']}_to_{pair['target_layer']}"
        for method,name in protocol['selected'][key].items():
            eligible=[c for c in pair['candidates'] if c['method']==method]
            require(min(eligible,key=lambda c:(c['kl_mean'],c['name']))['name']==name,'Rank selection disagrees with development.')
    study=run/'actual_alignment_study'
    results=json.loads(gzip.decompress((study/'alignment_study.json.gz').read_bytes()))
    require(results['protocol']==protocol and results['complete'],'Model study did not finish its frozen protocol.')
    require(results['forward_rows']==12888 and results['retained_outcomes']==12816,'Experiment inventory changed.')
    tensors=t.load(study/'actual_tensors.pt',map_location='cpu',weights_only=False)
    rows=results['outcomes'];counts=Counter(r['split'] for r in rows)
    require(counts=={'development':2448,'final':10368},'Split inventory is incomplete.')
    identities=[(r['split'],r['row_id'],r['source_layer'],r['target_layer'],r['condition'],r['draw']) for r in rows]
    require(len(set(identities))==len(rows),'Outcome rows contain duplicates.')
    maximum_kl_error=0.;winning_tokens_checked=0
    for key,pair in results['pairs'].items():
        for split,raw in [('development',tensors['development_logits'][key]),('final',tensors['logits'][key])]:
            source_rows=results['rows'][split];lookup={r['id']:i for i,r in enumerate(source_rows)}
            these=[r for r in rows if r['split']==split and r['source_layer']==pair['source_layer'] and r['target_layer']==pair['target_layer']]
            require(len(these)==len(source_rows)*(17 if split=='development' else 36),'A layer-pair candidate/control is missing.')
            expected_conditions={c['name'] for c in tensors['fits'][key]}
            if split=='final':expected_conditions|={'baseline','no_op','native_donor','same_norm_random'}
            require({r['condition'] for r in these}==expected_conditions,'Condition identities differ from the candidate fits.')
            native=raw['native_donor'];native_top=native.argmax(-1)
            metric_cache={name:m.distribution_kl(native,value) for name,value in raw.items()}
            for row in these:
                index=lookup[row['row_id']];condition=row['condition']
                name=f"random_{row['draw']}" if condition=='same_norm_random' else condition
                require(name in raw,'A recorded outcome has no actual logits.')
                error=abs(float(metric_cache[name][index])-row['kl_to_native_donor']);maximum_kl_error=max(maximum_kl_error,error)
                require(int(raw[name][index].argmax())==row['top_token_id'],'A displayed winning token differs from actual logits.')
                require(int(native_top[index])==row['native_donor_top_token_id'],'A donor token differs from actual logits.')
                require(row['native_top_agreement']==(row['top_token_id']==row['native_donor_top_token_id']),'Token agreement was scored incorrectly.')
                winning_tokens_checked+=1
            if split=='final':
                require(t.equal(raw['no_op'],raw['baseline']),'Native no-op changed logits.')
                for draw in range(protocol['null_draws']):require(sum(r['condition']=='same_norm_random' and r['draw']==draw for r in these)==96,'Random-control draw is incomplete.')
                for summary in pair['final']:
                    subset=[r for r in these if r['condition']==summary['name']]
                    require(abs(sum(r['kl_to_native_donor'] for r in subset)/len(subset)-summary['mean_kl'])<1e-12,'A plotted summary differs from its rows.')
    require(maximum_kl_error<1e-7,'Recorded KL differs from retained full-vocabulary logits.')
    play=t.load(study/'actual_play.pt',map_location='cpu',weights_only=False)
    require(set(play['logits'])=={'Unchanged','Transported','Same-norm random','Actual donor'},'Fresh play is missing a control.')
    t.testing.assert_close(play['difference'].norm(),play['random_difference'].norm(),atol=1e-12,rtol=1e-12)
    return {'status':'PASS','own_answer_code_cells':code,'figures':list(FIGURES),'outcomes':len(rows),
            'main_forward_rows':results['forward_rows'],'play_forward_rows':5,'maximum_kl_record_error':maximum_kl_error,
            'winning_tokens_checked':winning_tokens_checked,'development_selected_before_final':True,
            'actual_tensors_sha256':sha256(study/'actual_tensors.pt'),'actual_play_sha256':sha256(study/'actual_play.pt')}
