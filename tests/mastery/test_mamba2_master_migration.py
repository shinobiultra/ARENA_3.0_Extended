"""Exact migration preservation and bounded CPU method checks; no model study."""
from __future__ import annotations
from scripts.verified_publication_pages import historical_bytes
import ast
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import nbformat
import pytest
import torch as t

from scripts.verified_presentation import verified_pre_wrap_notebook

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/'chapter5_modern_architectures'
OLD=CHAPTER/'exercises/part7_mamba2_ssd'
SECTION=CHAPTER/'exercises/part4_mamba2_ssd'
EVIDENCE=ROOT/'docs/mastery/evidence/mamba2_master_migration'
MASTER=ROOT/'infrastructure/chapters/chapter5_modern_architectures/master_5_4.py'
STEM='5.4_Mamba-2_and_State-Space_Duality'
MODULE='chapter5_modern_architectures.exercises.part4_mamba2_ssd'

def sha(v):return hashlib.sha256(v).hexdigest()
def source(c):return ''.join(c['source'])
def codes(n):return [c for c in n['cells'] if c['cell_type']=='code']
def read(kind):return nbformat.read(SECTION/f'{STEM}_{kind}.ipynb',as_version=4)
def old(kind):return json.loads((OLD/f'5.7_Mamba-2_and_SSD_{kind}.ipynb').read_text())
def normalize(s):
    return s.replace('import part4_mamba2_ssd.tests as tests','import part7_mamba2_ssd.tests as tests').replace('EXERCISE_ID = "5_4_mamba2_ssd"','EXERCISE_ID = "5_7_mamba2_ssd"').replace('chapter5_modern_architectures/exercises/part4_mamba2_ssd/media/mamba2_ssd_signature.png','chapter5_modern_architectures/instructions/assets/mamba2_ssd_signature.png')


def test_archive_and_all_legacy_files_remain_physical_and_identical():
    from scripts.evidence_bundles import validate_review_bundle
    before=json.loads((EVIDENCE/'before.json').read_text())
    path=ROOT/before['archive']
    assert sha(path.read_bytes())==before['archive_sha256']
    assert len(validate_review_bundle(path)['members'])==18
    with tarfile.open(path) as saved:
        assert set(saved.getnames())==set(before['files'])
        for name,digest in before['files'].items():assert sha(saved.extractfile(name).read())==digest
    assert len(before['legacy_inventory'])==15
    for name,digest in before['legacy_inventory'].items():assert sha((ROOT/name).read_bytes())==digest,name
    assert len(before['report_input_physical_checks'])==15
    assert all(r['matches'] for r in before['report_input_physical_checks'])
    for name in ('artifacts.lock.yml','verification_report.json','verification_report.schema.json'):
        assert (SECTION/name).read_bytes()==(OLD/name).read_bytes()


def test_all_21_old_code_cells_ast_nonblank_lines_outputs_counts_and_widgets_survive():
    raw=old('solutions');normalized=nbformat.reads(json.dumps(raw),as_version=4);current=read('solutions')
    receipt=json.loads((EVIDENCE/'output_parity.json').read_text())
    assert len(codes(raw))==receipt['historical_code_cells']==21
    assert len(codes(current))==27 and receipt['appended_unexecuted_cells']==0
    assert receipt['canonical_notebook_freshly_executed'] is receipt['fresh_model_execution'] is False
    assert len(receipt['cells'])==21
    used=[]
    for ordinal,(a,r) in enumerate(zip(codes(raw),receipt['cells'],strict=True)):
        b=codes(current)[r['canonical_code_ordinal']];used.append(r['canonical_code_ordinal'])
        before=source(a);after=normalize(b.source);dump=ast.dump(ast.parse(before))
        assert dump==ast.dump(ast.parse(after)),ordinal
        assert [v for v in before.splitlines() if v.strip()]==[v for v in after.splitlines() if v.strip()],ordinal
        assert r['source_sha256']==sha(before.encode()) and r['canonical_source_sha256']==sha(b.source.encode())
        assert r['normalized_ast_sha256']==sha(dump.encode())
        assert r['outputs_sha256']==sha(json.dumps(a.get('outputs',[]),sort_keys=True).encode())
        assert b.outputs==codes(normalized)[ordinal].outputs
        assert b.execution_count==a['execution_count']==r['execution_count']
        assert bool(r['named_adaptation'])==(ordinal in (0,12))
    assert used==sorted(set(used))
    publication=json.loads((EVIDENCE/'fresh_subset_publication.json').read_text())
    executed=nbformat.read(EVIDENCE/'fresh_subset_execution.ipynb',as_version=4)
    freshrows=publication['fresh_code_cells']
    assert {r['canonical_code_ordinal'] for r in freshrows}==set(range(27))-set(used)
    for row in freshrows:
        c=codes(current)[row['canonical_code_ordinal']];actual=executed.cells[row['subset_cell_index']]
        assert c.source==actual.source and c.outputs==actual.outputs and c.execution_count==actual.execution_count==row['execution_count']
        assert row['source_sha256']==sha(c.source.encode())
        assert row['outputs_sha256']==sha(json.dumps(c.outputs,sort_keys=True).encode())
    assert current.metadata.get('widgets')==raw['metadata'].get('widgets') is None
    # Legacy widget views have no saved backing state; do not fabricate it.
    assert sum('application/vnd.jupyter.widget-view+json' in o.get('data',{}) for c in codes(raw) for o in c.get('outputs',[])) == 6
    assert current.metadata['arena_execution_provenance']['execution_kind']=='mixed_historical_reference_and_fresh_cpu_subset'


def test_every_student_function_and_class_completes_from_its_own_answer():
    exercises,solutions=read('exercises'),read('solutions')
    nbformat.validate(exercises);nbformat.validate(solutions)
    completed=[];stubnames=set();groups=0
    for index,c in enumerate(exercises.cells):
        if c.cell_type!='code':continue
        text=c.source
        if 'raise NotImplementedError' in text:
            groups+=1
            following=[]
            for later in exercises.cells[index+1:]:
                if later.cell_type=='markdown':
                    following.append(later.source)
                    if '<summary>Solution</summary>' in later.source:break
                    if later.source.startswith('### Exercise'):break
            answers=re.findall(r'<summary>Solution</summary>(.*?)</details>','\n'.join(following),re.S)
            assert len(answers)==1,index
            answer='\n\n'.join(re.findall(r'```python\n(.*?)```',answers[0],re.S))
            parsed=ast.parse(answer)
            definitions={n.name:n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            lines=text.splitlines(True);answerlines=answer.splitlines(True)
            for node in reversed(ast.parse(text).body):
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and 'raise NotImplementedError' in ast.get_source_segment(text,node):
                    stubnames.add(node.name);replacement=definitions[node.name]
                    start=min([node.lineno]+[n.lineno for n in node.decorator_list])-1
                    astart=min([replacement.lineno]+[n.lineno for n in replacement.decorator_list])-1
                    lines[start:node.end_lineno]=[''.join(answerlines[astart:replacement.end_lineno]).rstrip()+'\n']
            text=''.join(lines);assert 'raise NotImplementedError' not in text,index
        completed.append(ast.dump(ast.parse(text)))
    assert completed==[ast.dump(ast.parse(c.source)) for c in codes(solutions)]
    assert groups==10
    original={name for c in old('exercises')['cells'] if c['cell_type']=='code' for name in c.get('metadata',{}).get('exercise_functions',[])}
    assert stubnames==original
    assert len(stubnames)==21
    # Replacing an assigned class restores *all* constructors/helpers/cache methods.
    for name in ('Mamba2BlockFromScratch','TinyMamba2RecallModel','ReleasedMamba2MixerFromScratch','ReleasedMamba2LMFromScratch'):
        assert name in stubnames


def test_all_original_prose_and_core_method_definitions_survive():
    text='\n'.join(c.source for c in verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb').cells if c.cell_type=='markdown')
    compact=' '.join(text.split())
    for i,c in enumerate(old('solutions')['cells']):
        if c['cell_type']!='markdown':continue
        original=source(c)
        if i==0:
            original=original.replace('# [5.7] Mamba-2 and Structured State-Space Duality\n','').replace('5_7_mamba2_ssd','5_4_mamba2_ssd').replace('## Learning Objectives','## What you will implement')
        original=original.replace('<summary>Solution</summary>','<summary>Historical worked formulation</summary>').replace('fresh CUDA\nmeasurements from the release harness','historical CUDA\nmeasurements from the release harness').replace('## Limitations','## Limitations — preserved historical scope')
        original=re.sub(r'Difficulty: ([1-5])/5',lambda m:'Difficulty: '+'🔴'*int(m[1])+'⚪'*(5-int(m[1])),original)
        original=re.sub(r'Importance: ([1-5])/5',lambda m:'Importance: '+'🔵'*int(m[1])+'⚪'*(5-int(m[1])),original)
        if '### Exercise 10' in original or '### Exercise 11' in original:
            original=re.sub(r'> \*\*Difficulty:\*\* ([🔴⚪]+)  ', r'> ```yaml\n> Difficulty: \1\n> Importance: 🔵🔵🔵🔵🔵\n> ```',original)
        assert ' '.join(original.split()) in compact,i
    assert text.count('### Forecast')==text.count('### Reflect')==16
    for needle in ('confidence','causal reasoning','exposure','guided practice','unseen mastery','not freshly executed','not a description of this particular panel','0.753','IEEE','219','hidden states:        25','shuffled','matched','Try It Yourself'):
        assert needle in text,needle
    # The entire legacy module's taught methods remain readable and AST-identical.
    a={n.name:ast.dump(n) for n in ast.parse((OLD/'solutions.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    b={n.name:ast.dump(n) for n in ast.parse((SECTION/'solutions.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert a==b and len(a)==49
    for path in (MASTER,SECTION/'solutions.py',SECTION/'tests.py'):assert 'arena_ext' not in path.read_text()


def test_exact_images_and_canonical_page_links_survive():
    media=SECTION/'media'
    assert (media/'mamba2_ssd_signature.png').read_bytes()==(CHAPTER/'instructions/assets/mamba2_ssd_signature.png').read_bytes()
    total=0
    for i,c in enumerate(old('solutions')['cells']):
        for j,o in enumerate(c.get('outputs',[])):
            value=o.get('data',{}).get('image/png')
            if value:
                total+=1
                assert (media/f'mamba2_historical_cell_{i}_output_{j}.png').read_bytes()==base64.b64decode(''.join(value))
    assert total==1
    assert (media/'mamba2_historical_cell_26_output_1.png').read_bytes()==(media/'mamba2_ssd_signature.png').read_bytes()
    page=CHAPTER/'instructions/pages/04_[5.4]_Mamba-2_and_State-Space_Duality.md'
    assert page.exists() and page.with_suffix('.py').exists()
    assert 'part4_mamba2_ssd/media/mamba2_ssd_signature.png' in page.read_text()


@pytest.fixture(scope='module')
def lesson():return importlib.import_module(MODULE+'.solutions')
@pytest.fixture(scope='module')
def checks():return importlib.import_module(MODULE+'.tests')


@pytest.mark.parametrize('name,args',[
('test_segment_sums_exact',['segment_sums']),
('test_expand_grouped_parameters',['expand_grouped_parameters']),
('test_semiseparable_matrix_hand_oracle',['semiseparable_matrix']),
('test_recurrent_ssd_hand_oracle',['ssd_recurrent']),
('test_quadratic_ssd_parity_and_causality',['ssd_recurrent','ssd_quadratic']),
('test_chunked_ssd_parity_state_and_gradients',['ssd_recurrent','ssd_chunked']),
('test_mamba2_block_shape_and_causality',['Mamba2BlockFromScratch','Mamba2Config']),
('test_mamba2_step_cache_parity',['Mamba2BlockFromScratch','Mamba2Config']),
('test_delayed_recall_batch_semantics',['make_delayed_recall_batch']),
('test_recall_training_updates_parameters',['train_recall_model']),
('test_recall_orchestration',['run_delayed_recall_experiment']),
('test_scaling_storage_without_benchmark',['measure_ssd_scaling','ScalingResult']),
('test_exact_metrics_on_hand_oracle',['exact_parity_metrics']),
('test_released_mapping_on_synthetic_tensors',['native_to_released_state_dict']),
('test_released_gate_order',['ReleasedGatedRMSNorm']),
('test_released_mixer_on_tiny_zero_projection',['ReleasedMamba2MixerFromScratch','ReleasedMamba2Config']),
('test_released_mixer_literal_scalar_oracle',['ReleasedMamba2MixerFromScratch','ReleasedMamba2Config']),
('test_released_lm_on_tiny_configuration',['ReleasedMamba2LMFromScratch','ReleasedMamba2Config']),
('test_checkpoint_parity_assembly_without_weights',['verify_released_mamba2_parity','ReleasedCheckpointParity']),
('test_tiny_recall_carry_intervention',['TinyMamba2RecallModel']),
('test_recurrent_quadratic_chunked_gradient_parity',['exact_gradient_parity_metrics']),
('test_tiny_stacked_lm_full_vs_recurrent_cache',['TinyStackedMamba2LM']),
('test_parameter_matched_attention_and_mamba1_controls',['parameter_matched_control_counts','TinyCausalAttentionControl','TinyMamba1Control']),
('test_cuda_hooks_are_strict',[]),
])
def test_independent_cpu_fixtures(lesson,checks,name,args):
    getattr(checks,name)(*[getattr(lesson,arg) for arg in args])


def test_incorrect_independent_answers_are_rejected(lesson,checks):
    with pytest.raises(AssertionError):checks.test_recurrent_ssd_hand_oracle(lambda x,*args,**kwargs:(x,x[:,-1]))
    class WrongGate(t.nn.Module):
        def __init__(self,*args,**kwargs):super().__init__();self.weight=t.nn.Parameter(t.ones(2))
        def forward(self,x,gate):return x*self.weight
    with pytest.raises(AssertionError):checks.test_released_gate_order(WrongGate)
    with pytest.raises(AssertionError):checks.test_released_mapping_on_synthetic_tensors(lambda v:v)
    with pytest.raises(AssertionError):checks.test_scaling_storage_without_benchmark(lambda *args,**kw:lesson.ScalingResult([5,8,9],[7.]*3,[7.]*3,[7.]*3,[0.]*3,[0.]*3))
    with pytest.raises(AssertionError):checks.test_exact_metrics_on_hand_oracle(lambda **kwargs:{})


def test_import_from_unrelated_directory_is_inert(tmp_path):
    script='''import contextlib, importlib, io, os, sys
from pathlib import Path
import torch as t
sys.path.insert(0, REPO)
t.manual_seed(123);before=t.random.get_rng_state().clone();grad=t.is_grad_enabled();precision=t.get_float32_matmul_precision();threads=t.get_num_threads();env=os.environ.get('TRITON_F32_DEFAULT')
def fail(*args,**kwargs): raise AssertionError('forbidden model/GPU/optimizer/download operation during import')
t.cuda.init=fail;t.cuda.empty_cache=fail;t.optim.AdamW=fail
import huggingface_hub
huggingface_hub.hf_hub_download=fail;huggingface_hub.snapshot_download=fail
out=io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):importlib.import_module(MODULE)
assert not out.getvalue(),out.getvalue()
assert t.equal(before,t.random.get_rng_state()) and grad==t.is_grad_enabled()
assert precision==t.get_float32_matmul_precision() and threads==t.get_num_threads()
assert env==os.environ.get('TRITON_F32_DEFAULT')
'''
    script='REPO='+repr(str(ROOT))+'\nMODULE='+repr(MODULE+'.solutions')+'\n'+script
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_added_play_is_exact_and_never_invokes_a_study(lesson):
    notebook=read('solutions')
    c=next(c for c in notebook.cells if c.cell_type=='code' and 'PLAY_DECAY = ' in c.source)
    assert c.execution_count==5 and any('image/png' in o.get('data',{}) for o in c.outputs)
    assert not any(x in c.source for x in ('run_gpu_test','train_recall_model','verify_released','download','cuda'))
    # All changed controls are evaluated as tiny direct operator fixtures only.
    x=t.tensor([2.,3.,5.],dtype=t.float64).reshape(1,3,1,1);bc=t.ones(1,3,1,1,dtype=t.float64)
    for decay in (.25,.5,.9):
        a=t.full((1,3,1),__import__('math').log(decay),dtype=t.float64)
        expected=t.tensor([2.,2*decay+3.,(2*decay+3)*decay+5],dtype=t.float64)
        t.testing.assert_close(lesson.ssd_quadratic(x,a,bc,bc).flatten(),expected,atol=1e-12,rtol=0)
        for q in (1,2,4):
            y,state=lesson.ssd_chunked(x,a,bc,bc,q)
            t.testing.assert_close(y.flatten(),expected,atol=1e-12,rtol=0)
            assert abs(state.item()-expected[-1])<1e-12
        for carry in (0.,.5,1.):
            y,state=lesson.ssd_recurrent(x,a,bc,bc,carry_scale=carry)
            value=0.;expected_values=[]
            for token in (2.,3.,5.):value=carry*decay*value+token;expected_values.append(value)
            t.testing.assert_close(y.flatten(),t.tensor(expected_values,dtype=t.float64),atol=1e-12,rtol=0)
            assert abs(state.item()-value)<1e-12


def test_fresh_six_cell_subset_receipts_complete_objects_and_real_figure_chain():
    from scripts.evidence_bundles import validate_review_bundle
    protocol_path=EVIDENCE/'fresh_subset_protocol.json'
    protocol=json.loads(protocol_path.read_text());result=json.loads((EVIDENCE/'fresh_subset_execution.json').read_text())
    publication=json.loads((EVIDENCE/'fresh_subset_publication.json').read_text())
    executed_path=EVIDENCE/'fresh_subset_execution.ipynb';executed=nbformat.read(executed_path,as_version=4)
    assert sha(protocol_path.read_bytes())=='fd77e2ffbff1a6995582781f6643c2246aa036853a0f99e4720add962814b95d'
    assert result['protocol_sha256']==publication['protocol_sha256']==sha(protocol_path.read_bytes())
    assert result['runner_sha256']==sha((EVIDENCE/'run_cpu_subset.py').read_bytes())
    assert result['execution_notebook_sha256']==publication['fresh_execution_notebook_sha256']==sha(executed_path.read_bytes())
    assert result['status']=='succeeded' and result['kernel_cleaned_up']
    assert result['seconds']<protocol['max_execution_seconds']==30
    assert result['canonical_notebook_freshly_executed'] is result['historical_scientific_cells_executed'] is result['training_executed'] is False
    assert len(executed.cells)==7 and len(protocol['definitions'])==31 and len(protocol['checks'])==12
    assert [c.execution_count for c in executed.cells]==list(range(1,8))
    runtime=json.loads(executed.cells[0].outputs[0].text.removeprefix('CPU_SUBSET_RUNTIME '))
    assert runtime==publication['runtime']
    assert runtime['threads']==runtime['interop_threads']==1 and runtime['cuda_visible_devices']=='' and runtime['cuda_initialized'] is False
    assert sha((EVIDENCE/'fresh_subset_definitions.py').read_bytes())==result['definitions_source_sha256']
    definitions={n.name:ast.dump(n) for n in ast.parse((SECTION/'solutions.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for row in protocol['definitions']:
        assert row['source_sha256']==sha(row['source'].encode())
        assert ast.dump(ast.parse(row['source']).body[0])==definitions[row['name']]
    checks={n.name:ast.dump(n) for n in ast.parse((SECTION/'tests.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    for row in protocol['checks']:
        assert row['source_sha256']==sha(row['source'].encode()) and ast.dump(ast.parse(row['source']).body[0])==checks[row['name']]
        assert row['name']!='test_recall_training_updates_parameters'
    archive=ROOT/protocol['snapshot_archive']
    assert sha(archive.read_bytes())==protocol['snapshot_archive_sha256']
    assert len(validate_review_bundle(archive)['members'])==11
    with tarfile.open(archive) as saved:
        for name,digest in protocol['snapshot_files'].items():assert sha(saved.extractfile(name).read())==digest
        before=nbformat.reads(saved.extractfile(protocol['canonical_solution']).read().decode(),as_version=4)
    current=read('solutions')
    assert len(publication['historical_full_cell_equality'])==21
    for row in publication['historical_full_cell_equality']:
        i=row['canonical_code_ordinal'];a=codes(before)[i];b=codes(current)[i]
        assert a==b
        assert row['before_full_cell_sha256']==row['after_full_cell_sha256']==sha(json.dumps(b,sort_keys=True).encode())
    for row,c in zip(protocol['fresh_code_cells'],executed.cells[1:],strict=True):
        assert row['source']==c.source and row['source_sha256']==sha(c.source.encode())
        assert c.source==codes(current)[row['canonical_code_ordinal']].source
    for name,digest in publication['artifacts'].items():assert sha(historical_bytes(ROOT/name,digest))==digest,name
    image=[base64.b64decode(o['data']['image/png']) for c in executed.cells for o in c.outputs if 'image/png' in o.get('data',{})]
    assert len(image)==1
    assert sha(image[0])==publication['play_image_sha256']=='0f282fe01e56ca0d9af383658c1977be6a975e6805a10ad6ddf406fd765374d0'
    assert (SECTION/'media/mamba2_cpu_memory_play.png').read_bytes()==image[0]
    assert (EVIDENCE/'fresh_cpu_memory_play.png').read_bytes()==image[0]
    stdout='\n'.join(o.text for c in executed.cells for o in c.outputs if o.output_type=='stream')
    assert all('All tests in `'+row['name']+'` passed!' in stdout for row in protocol['checks'])
    assert 'chosen carry outputs: [2.0, 4.0, 7.0]' in stdout
