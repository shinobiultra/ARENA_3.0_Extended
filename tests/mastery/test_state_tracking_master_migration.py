"""Exact historical retention plus a frozen bounded CPU primitive subset; no training."""
from __future__ import annotations
from scripts.verified_publication_pages import historical_bytes
import ast
import base64
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import nbformat
import pytest
R=Path(__file__).resolve().parents[2]
E=R/'docs/mastery/evidence/state_tracking_master_migration'
O=R/'chapter5_modern_architectures/exercises/part4_mamba_state_tracking'
N=R/'chapter5_modern_architectures/exercises/part5_state_tracking'
MASTER=R/'infrastructure/chapters/chapter5_modern_architectures/master_5_5.py'
STEM='5.5_State_Tracking_in_SSMs_and_Transformers'
MODULE='chapter5_modern_architectures.exercises.part5_state_tracking'
def sha(v):return hashlib.sha256(v).hexdigest()
def read(kind):return nbformat.read(N/f'{STEM}_{kind}.ipynb',4)
def codes(nb):return [c for c in nb.cells if c.cell_type=='code']
def old(kind):return nbformat.read(O/f'5.4_Mamba_State_Tracking_{kind}.ipynb',4)
def normalize(s):
    return s.replace('section = "part5_state_tracking"','section = "part4_mamba_state_tracking"').replace('assets_dir = exercises_dir / section / "media"','assets_dir = root_dir / chapter / "instructions" / "assets"').replace('import part5_state_tracking.tests as tests','import part4_mamba_state_tracking.tests as tests').replace('from part5_state_tracking.mamba_backbone import MambaConfig, TinyMambaModel','from arena_ext.mamba import MambaConfig, TinyMambaModel')

def test_exact_archive_and_every_old_file_preserved():
    m=json.loads((E/'before.json').read_text());assert len(m['legacy_files'])==16
    assert sha((R/m['archive']).read_bytes())==m['archive_sha256']
    with tarfile.open(R/m['archive']) as a:
        assert len(a.getmembers())==20
        for name,h in m['archive_files'].items():assert sha(a.extractfile(name).read())==h
    for name,h in m['legacy_files'].items():assert sha((R/name).read_bytes())==h
    assert len(m['report_inputs'])==15 and all(x['matches'] for x in m['report_inputs'])

def test_all14_original_scientific_cells_outputs_counts_retained():
    rows=json.loads((E/'output_parity.json').read_text())['cells']
    assert len(codes(old('solutions')))==len(codes(read('solutions')))==len(rows)==14
    for i,(a,b,row) in enumerate(zip(codes(old('solutions')),codes(read('solutions')),rows,strict=True)):
        assert ast.dump(ast.parse(a.source))==ast.dump(ast.parse(normalize(b.source))),i
        assert [x for x in a.source.splitlines() if x.strip()]==[x for x in normalize(b.source).splitlines() if x.strip()]
        assert a.outputs==b.outputs and a.execution_count==b.execution_count==row['execution_count']
        assert sha(json.dumps(b.outputs,sort_keys=True).encode())==row['outputs_sha256']
        assert b.metadata['arena_execution_origin']['execution_kind']=='retained_historical_reference'
    assert sum(len(c.outputs) for c in codes(read('solutions')))==13
    assert read('solutions').metadata['arena_execution_provenance']['canonical_notebook_freshly_executed'] is False

def test_original_exercise_cells_and_each_own_answer_preserved():
    ex=read('exercises');sc=codes(read('solutions'));ec=codes(ex)
    assert len(ec)==14
    for i,(a,b) in enumerate(zip(codes(old('exercises')),ec,strict=True)):
        if i:assert ast.dump(ast.parse(a.source))==ast.dump(ast.parse(b.source)),i
    pending=None;answers=0
    for cell in ex.cells:
        if cell.cell_type=='code' and 'raise NotImplementedError' in cell.source:
            pending=ec.index(cell)
        if cell.cell_type=='markdown' and '<summary>Solution</summary>' in cell.source:
            answer=re.search(r'```python\n(.*?)\n```',cell.source,re.S).group(1)
            assert pending is not None
            assert ast.dump(ast.parse(answer))==ast.dump(ast.parse(sc[pending].source)),pending
            answers+=1;pending=None
    assert answers==5
    assert 'max_depth: int = 3' in ec[1].source and 'max_depth: int = 4' in sc[1].source

def test_all_original_prose_retained_with_explicit_format_adaptations():
    text='\n'.join(c.source for c in read('solutions').cells if c.cell_type=='markdown')
    def plain(s):
        s=re.sub(r'(?:> )?```yaml\n.*?\n(?:> )?```','',s,flags=re.S)
        s=s.replace('## Learning Objectives','## What you will implement').replace('<summary>Solution</summary>','<summary>Historical worked formulation</summary>').replace('../../instructions/assets/','media/')
        return re.sub(r'\s+',' ',s).strip()
    actual=plain(text)
    for i,c in enumerate(old('solutions').cells):
        if c.cell_type!='markdown':continue
        s=c.source.split('\n',1)[1] if i==0 else c.source
        for paragraph in re.split(r'\n\s*\n',plain(s)):
            assert paragraph in actual,i
    assert text.count('### Forecast —')==text.count('### Reflect —')==7
    assert 'does not teach or establish an expanded Transformer/Mamba-3 comparison' in actual
    assert 'not substituted with a newer implementation from 5.3' in actual

def test_all_five_inline_figures_and_three_original_assets_preserved():
    count=0
    for i,c in enumerate(old('solutions').cells):
        for j,o in enumerate(c.get('outputs',[])):
            image=o.get('data',{}).get('image/png')
            if image:
                assert (N/f'media/state_tracking_historical_cell_{i}_output_{j}.png').read_bytes()==base64.b64decode(image);count+=1
    assert count==5
    for p in (R/'chapter5_modern_architectures/instructions/assets').glob('mamba_state_tracking*'):
        assert (N/'media'/p.name).read_bytes()==p.read_bytes()
    # Expected figures appear directly in the exercises notebook, with no network.
    assert sum('data:image/png;base64,' in o.get('data', {}).get('text/html', '')
               for c in codes(read('exercises')) for o in c.outputs) == 5

def test_exact_minimum_prerequisite_closure_has_no_state_tracking_methods():
    p=json.loads((E/'backbone_provenance.json').read_text());module=(N/'mamba_backbone.py').read_text()
    with tarfile.open(E/'published_before_master.tar.gz') as a:archived=a.extractfile('arena_ext/mamba.py').read().decode()
    assert sha(archived.encode())==p['source_sha256']
    nodes={n.name:n for n in ast.parse(module).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    oldnodes={n.name:n for n in ast.parse(archived).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert len(nodes)==len(p['definitions'])==10
    for row in p['definitions']:
        n=nodes[row['name']];start=min([n.lineno]+[d.lineno for d in n.decorator_list]);exact='\n'.join(module.splitlines()[start-1:n.end_lineno])
        assert exact==row['source'] and sha(exact.encode())==row['sha256']
        assert ast.dump(n,include_attributes=False)==ast.dump(oldnodes[n.name],include_attributes=False)==row['ast']
    assert not {'fit_state_probe','run_state_transplant','generate_bracket_depth_task'} & nodes.keys()

def test_all_seven_original_immediate_test_bodies_retained():
    oldtests=(O/'tests.py').read_text();newtests=(N/'tests.py').read_text()
    a={n.name:ast.dump(n) for n in ast.parse(oldtests).body if isinstance(n,ast.FunctionDef)}
    b={n.name:ast.dump(n) for n in ast.parse(newtests).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    assert len(b)==7 and all(a[k]==v for k,v in b.items())
    assert 'arena_ext' not in newtests

def test_import_is_inert_from_unrelated_directory(tmp_path):
    script='''import sys,importlib,io,contextlib
import torch as t
import matplotlib.pyplot as plt
sys.path.insert(0,sys.argv[1])
def fail(*a,**k):raise AssertionError('optimizer,CUDA or plot during import')
t.optim.AdamW=t.cuda.init=t.cuda.is_available=t.cuda.device_count=plt.show=fail
rng=t.get_rng_state().clone();threads=t.get_num_threads();out=io.StringIO()
with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):m=importlib.import_module('chapter5_modern_architectures.exercises.part5_state_tracking.solutions')
assert not m.MAIN and not out.getvalue()
assert t.equal(rng,t.get_rng_state()) and threads==t.get_num_threads()
assert 'arena_ext.mamba' not in sys.modules
assert 'chapter5_modern_architectures.exercises.part4_mamba_state_tracking.solutions' not in sys.modules
'''
    r=subprocess.run([sys.executable,'-c',script,str(R)],cwd=tmp_path,capture_output=True,text=True,timeout=12)
    assert r.returncode==0,r.stdout+r.stderr

@pytest.fixture
def primitives(monkeypatch):
    import torch as t
    t.set_num_threads(1)
    module=importlib.import_module(MODULE+'.solutions')
    state=t.get_rng_state();dtype=t.get_default_dtype()
    def forbidden(*a,**k):raise AssertionError('No optimizer,CUDA,pretrained or model study authorized')
    monkeypatch.setattr(t.optim,'AdamW',forbidden)
    monkeypatch.setattr(t.cuda,'init',forbidden)
    monkeypatch.setattr(module,'train_state_tracker_cpu',forbidden)
    monkeypatch.setattr(module,'run_gpu_test',forbidden)
    monkeypatch.setattr(module,'run_full_experiment',forbidden)
    t.set_default_dtype(t.float32)
    yield module
    t.set_rng_state(state);t.set_default_dtype(dtype)

@pytest.mark.parametrize('seed',[0,1])
def test_bounded_bracket_primitive(primitives,seed):
    import torch as t
    m=primitives;b=m.generate_bracket_depth_task(4,8,max_depth=3,seed=seed)
    assert b.tokens.shape==b.states.shape==(4,8)
    t.testing.assert_close(b.states,t.where(b.tokens.bool(),1,-1).cumsum(-1))
    assert int(b.states.min())>=0 and int(b.states.max())<=3
    from chapter5_modern_architectures.exercises.part5_state_tracking.tests import _reference_brackets
    t.testing.assert_close(b.states,_reference_brackets(4,8,max_depth=3,seed=seed).states)

def test_literal_probe_primitive(primitives):
    import torch as t
    labels=t.tensor([[0,1,2,3],[3,2,1,0]]).repeat(4,1)
    x=t.nn.functional.one_hot(labels,4).float()
    probe=primitives.fit_state_probe(x,labels,ridge=.001)
    assert primitives.state_probe_accuracy(x,labels,probe)==1.
    assert probe.mean.shape==probe.scale.shape==(4,)

def test_handcrafted_anomaly_primitive(primitives):
    tests=importlib.import_module(MODULE+'.tests')
    tests.test_find_confident_errors_orders_real_mistakes(primitives.find_confident_errors)

def test_untrained_cached_forward_primitive(primitives):
    tests=importlib.import_module(MODULE+'.tests')
    tests.test_collect_recurrent_states_matches_full_forward(primitives.collect_recurrent_states)

def test_untrained_matched_transplant_primitive(primitives):
    tests=importlib.import_module(MODULE+'.tests')
    tests.test_state_transplant_matches_donor_dynamics(primitives.run_state_transplant,primitives.make_matched_state_pair)

def test_frozen_primitive_protocol_matches_current_scientific_sources():
    p=json.loads((E/'cpu_primitive_protocol.json').read_text())
    assert p['budget_seconds']==30 and p['device']=='cpu' and p['threads']==1
    assert p['optimizer_steps']==0 and p['fresh_historical_study']==False
    for path,h in p['source_sha256'].items():
        target = E/'cpu_primitive_test_initial.py' if path == 'tests/mastery/test_state_tracking_master_migration.py' else R/path
        assert sha(target.read_bytes()) == h, path
    # The numerical primitive test bodies remain identical to the pre-run freeze.
    before = {n.name: ast.dump(n) for n in ast.parse((E/'cpu_primitive_test_initial.py').read_text()).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in ast.parse(Path(__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    for name in before:
        if name.endswith('_primitive') or name == 'primitives':
            assert before[name] == after[name], name


def test_guided_proposal_exact_schema_and_archived_denominator():
    from mastery_hub.units.extension_predictions.lesson import validate_case, _digest
    case = json.loads((E/'proposed_case.json').read_text())
    identity = json.loads((E/'proposed_case_identity.json').read_text())
    assert len(case) == 20 and validate_case(case) == case
    assert case['section'] == '5.5' and case['data_role'] == 'historical'
    assert case['status'] == 'guided_practice' and case['prediction_type'] == 'numeric'
    assert identity['case_fingerprint'] == _digest(case)
    assert identity['proposal_sha256'] == sha((E/'proposed_case.json').read_bytes())
    for path, checksum in identity['artifact_sha256'].items():
        assert sha(historical_bytes(R/path, checksum)) == checksum, path
    ref = identity['primary_reference']
    assert ref['legacy_notebook_cell'] == 13 and ref['code_ordinal'] == 8
    assert ref['denominator_each'] == 8
    cell = old('solutions').cells[13]
    assert cell == codes(old('solutions'))[8]
    outputs = ''.join(str(o.get('text', '')) for o in cell.outputs)
    assert 'targeted transplant donor-trajectory match' in outputs
    assert 'matched random donor-trajectory match' in outputs
    assert 'patched_match' in cell.source and 'random_match' in cell.source
    assert identity['transfer_status'] == 'unmeasured_future_authorization_required'
    assert 'matched_random_state_edit(source_cache, half_donor, seed=0)' in case['transfer']
    assert 'run_state_transplant has no strength parameter' in case['transfer']
    assert 'HALF the original delta norm' in case['transfer']


def test_proposal_metric_explains_resolution_without_revealing_target_values():
    case=json.loads((E/'proposed_case.json').read_text())
    metric=case['config_fields']['metric_and_selection']
    assert 'Denominator=8' in metric and '1/8' in metric
    assert '.000' not in metric and '1.000' not in metric
    assert 'post-commit historical reference' in metric
    identity=json.loads((E/'proposed_case_identity.json').read_text())
    assert '.000' not in identity['primary_reference']['precision']
    assert '1.000' not in identity['primary_reference']['precision']
