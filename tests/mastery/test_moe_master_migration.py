"""MoE migration fidelity and existing tiny CPU mechanics, never signature/play training."""
from __future__ import annotations
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

from scripts.verified_presentation import verified_pre_wrap_notebook

ROOT=Path(__file__).resolve().parents[2]
LEGACY=ROOT/'chapter5_modern_architectures/exercises/part8_mixture_of_experts'
SECTION=LEGACY.parent/'part2_mixture_of_experts'
MASTER=ROOT/'infrastructure/chapters/chapter5_modern_architectures/master_5_2.py'
EVIDENCE=ROOT/'docs/mastery/evidence/moe_master_migration'
STEM='5.2_Mixture_of_Experts_from_Scratch'
OLD_STEM='5.8_Mixture_of_Experts_from_Scratch'
MODULE='chapter5_modern_architectures.exercises.part2_mixture_of_experts'


def sha(value):return hashlib.sha256(value).hexdigest()
def source(cell):return ''.join(cell['source'])
def codes(notebook):return [c for c in notebook['cells'] if c['cell_type']=='code']
def notebook(kind):return nbformat.read(SECTION/f'{STEM}_{kind}.ipynb',as_version=4)
def original(kind):return json.loads((LEGACY/f'{OLD_STEM}_{kind}.ipynb').read_text())


def normalize_route(text,ordinal):
    changes=json.loads((EVIDENCE/'route_adaptations.json').read_text())['code_substitutions'].get(str(ordinal),[])
    for historical,canonical in changes:
        assert text.count(canonical)==1
        text=text.replace(canonical,historical)
    return text


def definitions(text):
    return {node.name:ast.dump(node) for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}


def test_original_snapshot_is_recoverable_and_all_originals_remain_unchanged():
    before=json.loads((EVIDENCE/'before.json').read_text());archive=ROOT/before['archive']
    assert sha(archive.read_bytes())==before['archive_sha256']
    assert len(before['files'])==13 and before['independent_full_execution_receipt_available'] is False
    with tarfile.open(archive) as stored:
        for name,digest in before['files'].items():
            assert sha(stored.extractfile(name).read())==digest,name
            assert sha((ROOT/name).read_bytes())==digest,name


def test_every_visible_definition_matches_current_archived_reference_source():
    reference=definitions((LEGACY/'solutions.py').read_text())
    old={name:tree for cell in codes(original('solutions')) for name,tree in definitions(source(cell)).items()}
    current={name:tree for cell in codes(notebook('solutions')) for name,tree in definitions(source(cell)).items()}
    assert len(old)==len(current)==49
    assert old==current
    assert all(reference[name]==tree for name,tree in old.items())
    # The old builder is checked via isolated output redirection below; it is never published in place.


def test_old_builder_reconstructs_exact_source_without_publishing(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('historical_moe_builder',LEGACY/'build_notebooks.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    for name,filename in [('EXERCISE_PATH','exercises.ipynb'),('SOLUTION_PATH','solutions.ipynb'),('PAGE_PATH','page.md')]:
        monkeypatch.setattr(builder,name,tmp_path/filename)
    builder.build()
    for kind in ('exercises','solutions'):
        restored=json.loads((tmp_path/f'{kind}.ipynb').read_text())
        published=original(kind)
        assert len(restored['cells'])==len(published['cells'])==47
        changed=[]
        for index,(new,old) in enumerate(zip(restored['cells'],published['cells'],strict=True)):
            assert new['cell_type']==old['cell_type']
            if source(new)!=source(old):
                changed.append(index)
                assert new['cell_type']=='markdown' and source(new).split()==source(old).split()
        # The current historical builder only reflows these three prose cells.
        assert changed==[0,27,30]


def test_all_original_code_outputs_and_counts_have_explicit_parity():
    parity=json.loads((EVIDENCE/'output_parity.json').read_text())
    assert parity['execution_kind']=='retained_historical_reference'
    assert parity['fresh_full_execution'] is False and parity['independent_full_execution_receipt_available'] is False
    old=original('solutions');normalized=nbformat.reads(json.dumps(old),as_version=4);new=notebook('solutions')
    assert len(codes(old))==len(codes(new))==len(parity['cells'])==22
    for ordinal,(a,b,norm,pair) in enumerate(zip(codes(old),codes(new),codes(normalized),parity['cells'],strict=True)):
        x,y=source(a),source(b);comparison=normalize_route(y,ordinal)
        assert [line for line in x.splitlines() if line.strip()]==[line for line in comparison.splitlines() if line.strip()]
        assert ast.dump(ast.parse(x))==ast.dump(ast.parse(comparison))
        assert sha(x.encode())==pair['published_source_sha256']
        assert sha(y.encode())==pair['generated_source_sha256']
        assert sha(ast.dump(ast.parse(x)).encode())==pair['published_ast_sha256']==pair['normalized_ast_sha256']
        assert sha(ast.dump(ast.parse(y)).encode())==pair['generated_ast_sha256']
        assert b.outputs==norm.outputs and b.execution_count==norm.execution_count==pair['historical_execution_count']
        assert sha(json.dumps(a.get('outputs',[]),sort_keys=True).encode())==pair['outputs_sha256']
    assert sum(bool(c.outputs) for c in codes(new))==17
    assert sum(c.execution_count is not None for c in codes(new))==22
    assert sum('image/png' in o.get('data',{}) for c in codes(new) for o in c.outputs)==2


def test_semantic_tests_preserve_all_immediate_independent_oracles():
    old=(LEGACY/'tests.py').read_text()
    boundary=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='test_short_training_run_is_real')
    start=min([boundary.lineno,*(d.lineno for d in boundary.decorator_list)])-1
    expected=''.join(old.splitlines(True)[:start]).rstrip()+'\n'
    assert (SECTION/'tests.py').read_text()==expected
    names=definitions(expected)
    calls={n.func.attr for cell in codes(notebook('solutions')) for n in ast.walk(ast.parse(cell.source))
           if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tests'}
    assert calls<=names.keys()
    assert 'test_top1_sparse_moe_language_path_reaches_router' in calls
    assert 'test_short_training_run_is_real' not in names and 'test_notebook_contract' not in names


def test_figure_and_negative_report_are_preserved():
    asset='mixture_of_experts_signature.png'
    assert (SECTION/'media'/asset).read_bytes()==(ROOT/'chapter5_modern_architectures/instructions/assets'/asset).read_bytes()
    embedded=[]
    for c in codes(notebook('exercises')):
        for o in c.outputs:
            embedded += re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',str(o.get('data',{}).get('text/html','')))
    assert len(embedded)==1 and base64.b64decode(embedded[0])==(SECTION/'media'/asset).read_bytes()
    report=json.loads((LEGACY/'expected_outputs/reference_metrics.json').read_text())
    assert report['original_positive_release_contract_accepted'] is False
    assert report['metrics']['preferred_expert_ablation_drop']==0
    output='\n'.join(str(o.get('text','')) for c in codes(notebook('solutions')) for o in c.outputs)
    assert 'Preferred-expert ablation hypothesis: REJECTED' in output
    assert 'Original positive-specialization release contract: False' in output


def _completed(source_text,answer):
    lines=source_text.splitlines(True);answer_lines=answer.splitlines(True)
    answers={}
    for node in ast.parse(answer).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            first=min([node.lineno,*(d.lineno for d in node.decorator_list)])-1
            answers[node.name]=''.join(answer_lines[first:node.end_lineno])
    for node in reversed(ast.parse(source_text).body):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            first=min([node.lineno,*(d.lineno for d in node.decorator_list)])-1
            if 'raise NotImplementedError' in ''.join(lines[first:node.end_lineno]):
                lines[first:node.end_lineno]=[answers[node.name]+'\n']
    return ''.join(lines)


def test_own_answers_complete_all_twelve_exercises_including_decorators():
    ex,sol=notebook('exercises'),notebook('solutions');nbformat.validate(ex);nbformat.validate(sol)
    completed=[];groups=0;taught=set()
    for index,cell in enumerate(ex.cells):
        if cell.cell_type!='code':continue
        text=cell.source
        if 'raise NotImplementedError' in text:
            groups+=1;taught.update(definitions(text))
            following=[]
            for other in ex.cells[index+1:]:
                if other.cell_type=='code':break
                following.append(other.source)
            blocks=re.findall(r'<summary>Solution</summary>(.*?)</details>','\n'.join(following),re.S)
            assert len(blocks)==1
            answer='\n\n'.join(re.findall(r'```python\n(.*?)```',blocks[0],re.S))
            text=_completed(text,answer)
            assert 'tests.test_' in text
        completed.append(text)
    assert groups==12 and len(taught)==21
    for a,b in zip(completed,codes(sol),strict=True):assert ast.dump(ast.parse(a))==ast.dump(ast.parse(source(b)))
    assert all(c.execution_count is None for c in codes(ex))


def test_original_teaching_prose_controls_plays_and_forecasts_survive():
    text='\n'.join(c.source for c in verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb').cells if c.cell_type=='markdown')
    lines={line.strip().removeprefix('> ').strip() for line in text.splitlines() if line.strip()}
    for cell in original('exercises')['cells']:
        if cell['cell_type']!='markdown':continue
        old=re.sub(r'<details>\s*<summary>Solution</summary>.*?</details>','',source(cell),flags=re.S)
        for line in old.splitlines():
            line=line.strip().removeprefix('> ').strip()
            if not line or line.startswith('# [5.8]') or line=='## Learning Objectives' or line.startswith('<img src="../assets/'):
                continue
            if line.startswith(('Difficulty:','Importance:')):continue
            if line.startswith('```'):continue
            assert line in lines,line
    assert text.count('## Forecast before the result')==10 and text.count('## Reflect after the result')==10
    assert '## Try It Yourself' in text
    for needle in ('REJECTED','rejected','capacity','current-token','near-chance','12.5%','does not prove zero logit effect','Finite capacity is allocated per forward call','not a fresh full execution','no independent source-hashed','current canonical 5.8','guided practice'):
        assert needle.lower() in text.lower(),needle
    all_text='\n'.join(c.source for c in verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb').cells)
    assert all_text.index('Forecast before the result — paired-negative-study')<all_text.index('signature = run_cpu_experiment(')


def test_import_safety_from_unrelated_directory(tmp_path):
    script=r'''
import contextlib,importlib,io,os,sys,tempfile
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import urllib.request
sys.path.insert(0,sys.argv[1]);os.chdir(sys.argv[2])
def forbidden(*a,**k):raise AssertionError('Import executed a lesson side effect')
torch.set_num_threads=forbidden
plt.rcParams.update=forbidden
plt.subplots=forbidden
tempfile.mkdtemp=forbidden
urllib.request.urlopen=forbidden
before=torch.cuda.is_initialized();capture=io.StringIO()
with contextlib.redirect_stdout(capture):
 module=importlib.import_module('chapter5_modern_architectures.exercises.part2_mixture_of_experts.solutions')
assert not capture.getvalue()
assert torch.cuda.is_initialized()==before
assert not hasattr(module,'signature') and not hasattr(module,'play_model')
for name in ('topk_router','compute_capacity_state','SparseMoETransformer','causal_expert_intervention_table','run_cpu_experiment'):
 assert callable(getattr(module,name))
print('Safe import; no training, plot, download or runtime mutation')
'''
    result=subprocess.run([sys.executable,'-c',script,str(ROOT),str(tmp_path)],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr


# Exclude historical trainability/full-publication tests: these are methods-only checks.
ORACLES=tuple(n.name for n in ast.parse((SECTION/'tests.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))


@pytest.mark.parametrize('name',ORACLES)
def test_existing_independent_cpu_semantic_oracle(name):
    import torch
    previous_threads=torch.get_num_threads();previous_dtype=torch.get_default_dtype()
    try:
        torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(devices=[]):getattr(importlib.import_module(MODULE+'.tests'),name)()
    finally:
        torch.set_default_dtype(previous_dtype);torch.set_num_threads(previous_threads)


def test_fresh_cpu_subset_is_a_separate_exact_source_bounded_receipt():
    protocol=json.loads((EVIDENCE/'cpu_subset_protocol.json').read_text())
    receipt=json.loads((EVIDENCE/'cpu_subset_execution.json').read_text())
    executed=nbformat.read(EVIDENCE/'cpu_methods_executed.ipynb',as_version=4)
    assert receipt['status']=='PASS' and receipt['method_cell_count']==20
    assert receipt['administrative_cell_count']==2 and receipt['scientific_training_or_play_runs']==0
    assert receipt['fresh_full_notebook_execution'] is False
    assert receipt['source_protocol_sha256']==sha((EVIDENCE/'cpu_subset_protocol.json').read_bytes())
    assert receipt['executed_notebook_sha256']==sha((EVIDENCE/'cpu_methods_executed.ipynb').read_bytes())
    assert protocol['selected_original_code_ordinals']==list(range(20))
    assert protocol['excluded_scientific_code_ordinals']==receipt['excluded_scientific_code_ordinals']==[20,21]
    assert protocol['CUDA_VISIBLE_DEVICES']=='' and protocol['cpu_threads']==1
    assert protocol['offline']=={'UV_OFFLINE':'1','HF_HUB_OFFLINE':'1'}
    assert receipt['runtime']['threads']==1 and receipt['runtime']['default_device']=='cpu'
    assert receipt['runtime']['cuda_initialized'] is False
    assert 0<receipt['elapsed_seconds']<60 and 0<receipt['aggregate_cpu_seconds']<60
    actual=[c for c in codes(executed) if 'original_code_ordinal' in c.metadata]
    assert len(codes(executed))==22 and len(actual)==20
    for ordinal,(cell,current,pair) in enumerate(zip(actual,codes(notebook('solutions'))[:20],receipt['cells'],strict=True)):
        assert cell.metadata['original_code_ordinal']==pair['original_code_ordinal']==ordinal
        assert cell.source==current.source
        assert cell.execution_count==pair['execution_count']==ordinal+2
        assert sha(cell.source.encode())==pair['source_sha256']==protocol['selected_code_sources_sha256'][ordinal]
        assert sha(ast.dump(ast.parse(cell.source)).encode())==pair['ast_sha256']==protocol['selected_code_ast_sha256'][ordinal]
        assert sha(json.dumps(cell.outputs,sort_keys=True).encode())==pair['outputs_sha256']
    # Fresh subset counts never replace the historical published counts1..22.
    assert [c.execution_count for c in codes(notebook('solutions'))]==list(range(1,23))


def test_current_publication_manifest_can_be_refreshed_after_navigation(tmp_path):
    spec=importlib.util.spec_from_file_location('moe_current_manifest',EVIDENCE/'refresh_current_manifest.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    current=json.loads((EVIDENCE/'current_publication.json').read_text())
    # Verify the original manifest against exact recovered publication bytes.
    # Current presentation must pass strict before/current and renderer/AST proofs.
    from scripts.verified_publication_pages import historical_bytes
    bindings = dict(current['files_sha256'])
    bindings.update({name: row['whole_file_sha256_descriptive'] for name, row in current['notebooks'].items()})
    for name, expected in bindings.items():
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(historical_bytes(ROOT/name, expected))
        assert sha(target.read_bytes()) == expected
    module.ROOT = tmp_path
    module.SECTION = tmp_path/SECTION.relative_to(ROOT)
    assert current==module.make_manifest()
    assert current['schema']=='arena-moe-current-publication'
    assert all(len(nb['code'])==22 for nb in current['notebooks'].values())


def test_proposed_guided_case_is_strict_and_uses_exact_retained_aggregate_evidence():
    from mastery_hub.units.extension_predictions.lesson import validate_case
    case=json.loads((EVIDENCE/'proposed_case.json').read_text())
    assert len(case)==20 and validate_case(case)==case
    assert case['section']=='5.2' and case['prediction_type']=='binary'
    assert case['forecast_items']==[] and case['status']=='guided_practice'
    assert case['data_role']=='historical'
    visible=definitions((SECTION/'solutions.py').read_text())
    assert set(case['method_functions'])<=visible.keys()
    assert 'historical_reference_json' in case['observation_prompt']
    assert 'not unseen transfer' in case['transfer']
    report=json.loads((LEGACY/'expected_outputs/reference_metrics.json').read_text())
    for key in ('preferred_expert_ablation_drop','mean_nonpreferred_expert_ablation_drop',
                'preferred_expert_forcing_drop','mean_nonpreferred_expert_forcing_drop'):
        values=report['seed_metrics'][key]
        assert len(values)==3 and abs(sum(values)/3-report['metrics'][key])<1e-15
