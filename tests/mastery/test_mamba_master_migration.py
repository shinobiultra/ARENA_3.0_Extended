"""5.3 archive/surface parity and explicitly bounded independent CPU primitives."""
from __future__ import annotations
import ast,base64,hashlib,importlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path
import nbformat
import pytest
import torch as t
ROOT=Path(__file__).resolve().parents[2];EVIDENCE=ROOT/'docs/mastery/evidence/mamba_master_migration'
SECTION=ROOT/'chapter5_modern_architectures/exercises/part3_mamba_from_scratch'
MASTER=ROOT/'infrastructure/chapters/chapter5_modern_architectures/master_5_3.py'
PREFIX='chapter5_modern_architectures/exercises/part3_mamba_from_scratch/'
STEM='5.3_Mamba_from_Scratch';MODULE='chapter5_modern_architectures.exercises.part3_mamba_from_scratch'
def sha(v):return hashlib.sha256(v).hexdigest()
def source(c):return ''.join(c['source'])
def codes(n):return [c for c in n['cells'] if c['cell_type']=='code']
def archived(name):
    with tarfile.open(EVIDENCE/'published_before_master.tar.gz') as saved:return saved.extractfile(name).read()
def old(kind):return json.loads(archived(PREFIX+f'{STEM}_{kind}.ipynb'))
def read(kind):return nbformat.read(SECTION/f'{STEM}_{kind}.ipynb',as_version=4)
def normalize(s):return s.replace('chapter5_modern_architectures/exercises/part3_mamba_from_scratch/media/mamba_from_scratch_signature.png','chapter5_modern_architectures/instructions/assets/mamba_from_scratch_signature.png').replace('chapter5_modern_architectures.exercises.part3_mamba_from_scratch.historical_release','chapter5_modern_architectures.exercises.part3_mamba_from_scratch.solutions')
def is_stub(n):return any(isinstance(x,ast.Raise) and isinstance(x.exc,ast.Call) and isinstance(x.exc.func,ast.Name) and x.exc.func.id=='NotImplementedError' and not x.exc.args for x in ast.walk(n))
def segment(s,n):
    start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
    return ''.join(s.splitlines(True)[start:n.end_lineno]).rstrip()


def test_exact_archive_same_directory_overwrites_and_historical_identity():
    from scripts.evidence_bundles import validate_review_bundle
    before=json.loads((EVIDENCE/'before.json').read_text());archive=ROOT/before['archive']
    assert sha(archive.read_bytes())==before['archive_sha256']
    assert len(validate_review_bundle(archive)['members'])==17
    assert len(before['legacy_inventory'])==14 and len(before['report_input_physical_checks'])==14
    assert all(r['matches'] for r in before['report_input_physical_checks'])
    with tarfile.open(archive) as saved:
        assert set(saved.getnames())==set(before['files'])
        for name,digest in before['files'].items():assert sha(saved.extractfile(name).read())==digest
    assert (SECTION/'historical_release.py').read_bytes()==archived(PREFIX+'solutions.py')
    overwritten={PREFIX+name for name in ('solutions.py','tests.py',STEM+'_exercises.ipynb',STEM+'_solutions.ipynb')}
    overwritten.add('chapter5_modern_architectures/instructions/pages/03_[5.3]_Mamba_from_Scratch.md')
    for name,digest in before['legacy_inventory'].items():
        if name not in overwritten:assert sha((ROOT/name).read_bytes())==digest,name
    assert (SECTION/'tests.py').read_bytes().startswith(archived(PREFIX+'tests.py'))
    report=json.loads((SECTION/'verification_report.json').read_text())
    assert report['report_inputs'] == json.loads(archived(PREFIX+'verification_report.json'))['report_inputs']


def test_all17_historical_sources_outputs_counts_and_widgets_survive():
    raw=old('solutions');normalized=nbformat.reads(json.dumps(raw),as_version=4);current=read('solutions')
    receipt=json.loads((EVIDENCE/'output_parity.json').read_text())
    assert len(codes(raw))==receipt['historical_code_cells']==17
    assert len(codes(current))==23
    assert receipt['canonical_notebook_freshly_executed'] is receipt['fresh_model_execution'] is False
    used=[]
    for ordinal,(a,row) in enumerate(zip(codes(raw),receipt['cells'],strict=True)):
        c=codes(current)[row['canonical_code_ordinal']];used.append(row['canonical_code_ordinal'])
        s=source(a);dump=ast.dump(ast.parse(s))
        assert dump==ast.dump(ast.parse(normalize(c.source))),ordinal
        assert [v for v in s.splitlines() if v.strip()]==[v for v in normalize(c.source).splitlines() if v.strip()],ordinal
        assert row['source_sha256']==sha(s.encode()) and row['canonical_source_sha256']==sha(c.source.encode())
        assert row['normalized_ast_sha256']==sha(dump.encode())
        assert row['outputs_sha256']==sha(json.dumps(a.get('outputs',[]),sort_keys=True).encode())
        assert c.outputs==codes(normalized)[ordinal].outputs and c.execution_count==a['execution_count']==row['execution_count']
        assert bool(row['named_adaptation'])==(ordinal in (13,16))
    assert used==sorted(set(used))
    assert current.metadata['widgets']==normalized.metadata['widgets']
    # Before the explicitly authorized subset, no new-cell outputs/counts are invented.
    if not (EVIDENCE/'fresh_subset_publication.json').exists():
        assert receipt['appended_unexecuted_cells']==6
        for i,c in enumerate(codes(current)):
            if i not in used:assert c.execution_count is None and not c.outputs
    else:
        publication=json.loads((EVIDENCE/'fresh_subset_publication.json').read_text())
        executed=nbformat.read(EVIDENCE/'fresh_subset_execution.ipynb',as_version=4)
        assert {r['canonical_code_ordinal'] for r in publication['fresh_code_cells']}==set(range(23))-set(used)
        for row in publication['fresh_code_cells']:
            c=codes(current)[row['canonical_code_ordinal']];a=executed.cells[row['subset_cell_index']]
            assert c.source==a.source and c.outputs==a.outputs and c.execution_count==a.execution_count==row['execution_count']


def test_all_own_answers_complete_every_function_class_and_step_wiring():
    e,s=read('exercises'),read('solutions');nbformat.validate(e);nbformat.validate(s)
    completed=[];assigned=set();groups=0
    for i,c in enumerate(e.cells):
        if c.cell_type!='code':continue
        text=c.source;tree=ast.parse(text);targets=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and is_stub(n)]
        if targets:
            groups+=1;following=[]
            for later in e.cells[i+1:]:
                if later.cell_type=='markdown':
                    following.append(later.source)
                    if '<summary>Solution</summary>' in later.source:break
                    if later.source.startswith('### Exercise'):break
            answers=re.findall(r'<summary>Solution</summary>(.*?)</details>','\n'.join(following),re.S)
            assert len(answers)==1,i
            answer='\n\n'.join(re.findall(r'```python\n(.*?)```',answers[0],re.S));definitions={n.name:n for n in ast.parse(answer).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            lines=text.splitlines(True)
            for n in reversed(targets):
                assigned.add(n.name);start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
                lines[start:n.end_lineno]=[segment(answer,definitions[n.name])+'\n']
            text=''.join(lines)
        completed.append(ast.dump(ast.parse(text)))
    assert completed==[ast.dump(ast.parse(c.source)) for c in codes(s)]
    original={n.name for c in old('exercises')['cells'] if c['cell_type']=='code' for n in ast.parse(source(c)).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and is_stub(n)}
    assert assigned==original and groups==11
    assert {'MambaModelFromScratch','MambaForCausalLMFromScratch','MambaMixerFromScratch','mamba_recurrent_step'}<=assigned
    code='\n'.join(c.source for c in e.cells if c.cell_type=='code')
    cls=next(n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name=='MambaModelFromScratch')
    assert {'__init__','initial_states','forward'}<={n.name for n in cls.body if isinstance(n,ast.FunctionDef)}
    assert 'MambaMixerFromScratch.step = mamba_recurrent_step' in code


def test_every_original_prose_token_and_repaired_python_dropdown_survive():
    text='\n'.join(c.source for c in read('exercises').cells if c.cell_type=='markdown');compact=' '.join(text.split())
    for i,c in enumerate(old('solutions')['cells']):
        if c['cell_type']!='markdown':continue
        original=source(c)
        if i==0:original=original.split('\n',1)[1]
        original=original.replace('## Learning Objectives','## What you will implement').replace('<summary>Solution</summary>','<summary>Historical worked formulation</summary>').replace('../../instructions/assets/','media/')
        assert ' '.join(original.split()) in compact,i
    assert text.count('### Forecast')==text.count('### Reflect')==15
    historical=re.findall(r'<summary>Historical worked formulation</summary>(.*?)</details>',text,re.S)
    assert len(historical)==11
    for body in historical:
        blocks=re.findall(r'```python\n(.*?)```',body,re.S);assert blocks;ast.parse('\n'.join(blocks))
    for term in ('confidence','causal reasoning','exposure','guided practice','unseen mastery','not freshly executed','243','float32','float64','no model','all-context','stale','WRITE','ERASE','Try It Yourself','historical_release'):
        assert term in text,term
    for p in (MASTER,SECTION/'solutions.py',SECTION/'tests.py'):assert 'arena_ext' not in p.read_text()
    page=ROOT/'chapter5_modern_architectures/instructions/pages/03_[5.3]_Mamba_from_Scratch.md'
    assert page.exists() and page.with_suffix('.py').exists()
    assert 'part3_mamba_from_scratch/media/mamba_from_scratch_signature.png' in page.read_text()


def test_all_four_old_pngs_and_asset_bytes_survive():
    n=old('solutions');count=0
    for i,c in enumerate(n['cells']):
        for j,o in enumerate(c.get('outputs',[])):
            value=o.get('data',{}).get('image/png')
            if value:
                count+=1;assert (SECTION/f'media/mamba_historical_cell_{i}_output_{j}.png').read_bytes()==base64.b64decode(''.join(value))
    assert count==4
    original=ROOT/'chapter5_modern_architectures/instructions/assets/mamba_from_scratch_signature.png'
    assert (SECTION/'media/mamba_from_scratch_signature.png').read_bytes()==original.read_bytes()


@pytest.fixture(scope='module')
def lesson():return importlib.import_module(MODULE+'.solutions')
@pytest.fixture(scope='module')
def checks():return importlib.import_module(MODULE+'.tests')

@pytest.mark.parametrize('name,args',[
('test_continuous_discretization_exact',['stable_continuous_A','discretize_scalar_ssm_exact']),
('test_sequential_affine_scan_manual',['sequential_affine_scan']),
('test_compose_affine_updates',['compose_affine_updates']),
('test_parallel_affine_scan_parity',['sequential_affine_scan','parallel_affine_scan']),
('test_parallel_scan_gradients',['parallel_affine_scan']),
('test_causal_depthwise_conv_manual_and_no_future',['causal_depthwise_conv1d']),
('test_project_selective_parameters_is_input_dependent',['project_selective_parameters']),
('test_selective_discretization_shapes_and_stability',['discretize_selective_scan']),
('test_selective_scan_single_step_manual',['selective_scan_recurrent']),
('test_recurrent_parallel_selective_scan_parity',['selective_scan_recurrent','selective_scan_parallel']),
('test_mamba_block_has_norm_residual_and_gate',['MambaBlockFromScratch']),
('test_mamba_block_recurrent_step_parity',['MambaBlockFromScratch']),
('test_tiny_stacked_mamba_lm_and_cache',['MambaForCausalLMFromScratch']),
('test_state_intervention_is_causal',['intervene_on_state']),
('test_selective_parameters_literal',['project_selective_parameters','discretize_selective_scan','_expand_bc']),
('test_rms_norm_literal',['MambaRMSNorm']),
('test_mixer_literal_oracle',['MambaMixerFromScratch','MambaConfig']),
('test_mixer_cache_storage',['MambaMixerFromScratch','MambaConfig']),
('test_weight_mapping_without_checkpoint',['hf_to_local_key','load_hf_mamba_weights']),
('test_parity_assembly_without_checkpoint',['pinned_mamba_cpu_parity']),
('test_event_controls_literal',['build_event_coefficients','selective_readout','intervene_on_state','causal_attention_memory','finite_causal_conv_memory']),
('test_delay_benchmark_literal',['run_memory_delay_benchmark']),
])
def test_independent_cpu_primitives(lesson,checks,name,args):getattr(checks,name)(*[getattr(lesson,arg) for arg in args])


def test_effective_step_and_historical_release_are_inert_and_bound(lesson,tmp_path):
    assert lesson.MambaMixerFromScratch.step is lesson.mamba_recurrent_step
    assert sha((SECTION/'historical_release.py').read_bytes())==sha(archived(PREFIX+'solutions.py'))
    script='''import contextlib,importlib,io,os,sys
import torch as t
sys.path.insert(0,REPO)
t.manual_seed(17);rng=t.random.get_rng_state().clone();grad=t.is_grad_enabled()
def fail(*args,**kwargs):raise AssertionError('forbidden model/GPU/optimizer action on import')
t.cuda.init=fail;t.cuda.empty_cache=fail;t.optim.AdamW=fail;t.optim.Adam=fail
import huggingface_hub
huggingface_hub.hf_hub_download=fail;huggingface_hub.snapshot_download=fail
out=io.StringIO()
with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):m=importlib.import_module(MODULE)
assert not out.getvalue(),out.getvalue()
assert t.equal(rng,t.random.get_rng_state()) and grad==t.is_grad_enabled()
assert m.MambaMixerFromScratch.step is m.mamba_recurrent_step
assert 'chapter5_modern_architectures.exercises.part3_mamba_from_scratch.historical_release' not in sys.modules
'''
    script='REPO='+repr(str(ROOT))+'\nMODULE='+repr(MODULE+'.solutions')+'\n'+script
    result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr


def test_bad_independent_answers_are_rejected(checks,lesson):
    with pytest.raises(AssertionError):checks.test_sequential_affine_scan_manual(lambda a,b,initial_state=None:b)
    with pytest.raises(AssertionError):checks.test_weight_mapping_without_checkpoint(lambda key:'wrong',lesson.load_hf_mamba_weights)
    with pytest.raises(AssertionError):checks.test_event_controls_literal(lambda events,values:(t.ones(1,5,1,dtype=t.float64),)*3,lesson.selective_readout,lesson.intervene_on_state,lesson.causal_attention_memory,lesson.finite_causal_conv_memory)


def test_guided_proposal_binds_completed_subset_without_target_answers():
    from mastery_hub.units.extension_predictions.lesson import validate_case
    case=json.loads((EVIDENCE/'proposed_case.json').read_text())
    assert len(case)==20 and validate_case(case)==case
    execution=case['config_fields']['execution']
    assert 'Completed exact six-cell CPU subset' in execution
    assert '17 retained historical code cells' in execution
    assert 'not a freshly executed complete notebook' in execution
    assert 'must remain unexecuted' not in execution
    assert 'once available' not in case['experiment']
    assert 'target_answer' not in case and 'expected_answer' not in case
    assert case['status']=='guided_practice' and case['evidence_class']=='analytic_toy'
