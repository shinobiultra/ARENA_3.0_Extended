"""Mutation checks for saved evidence only; no model, tensor, or lesson execution."""
import copy
import json
from pathlib import Path
import pytest
from scripts import verify_fresh_lesson_evidence as v

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/mastery/evidence/pilot_fresh_execution_20261016/7.1'
@pytest.fixture
def sequence():
    return json.loads((BASE/'executed.ipynb').read_text()),json.loads((BASE/'result.json').read_text())

def test_actual_cell_sequence_and_ast(sequence):
    raw,result=sequence
    v.verify_sequence(raw,result,40,900)
    assert len(v.notebook_assertions(raw))==1
    assert sum(len(v.test_calls(c)) for c in v.code_cells(raw))==14

@pytest.mark.parametrize('mutation', ['missing_receipt','duplicate_index','boolean_count','wrong_count','code_change','code_reorder','error_flag','error_output','over_budget','unknown_nested_field'])
def test_sequence_mutations_rejected(sequence,mutation):
    raw,result=sequence;cells=v.code_cells(raw)
    if mutation=='missing_receipt':result['cells'].pop()
    elif mutation=='duplicate_index':result['cells'][1]['index']=result['cells'][0]['index']
    elif mutation=='boolean_count':result['completed_code_cells']=True
    elif mutation=='wrong_count':cells[0]['execution_count']=42
    elif mutation=='code_change':cells[0]['source']=['print("tampered")']
    elif mutation=='code_reorder':result['original_code_sha256s'].reverse()
    elif mutation=='error_flag':result['cells'][0]['has_error']=True
    elif mutation=='error_output':cells[0]['outputs'].append({'output_type':'error','ename':'Error','evalue':'bad','traceback':[]})
    elif mutation=='over_budget':result['cells'][0]['duration_seconds']=901
    elif mutation=='unknown_nested_field':result['cells'][0]['unexpected']=False
    with pytest.raises(v.EvidenceError):v.verify_sequence(raw,result,40,900)

def test_ast_finds_multiline_calls_and_nested_assertions():
    cell={'source':['tests.\n' ]}
    cell['source']=['if True:\n    tests.test_example(\n        x\n    )\n    assert x > 0\n']
    assert v.test_calls(cell)==['test_example']
    assert len([n for n in v.ast.walk(v.cell_ast(cell)) if isinstance(n,v.ast.Assert)])==1

@pytest.mark.parametrize('name',['../outside','/tmp/outside','a/../../outside','./file','a\\b'])
def test_unsafe_paths_rejected(tmp_path,name):
    with pytest.raises(v.EvidenceError):v.safe_path(tmp_path,name)

def test_symlink_rejected(tmp_path):
    (tmp_path/'real').write_text('x');(tmp_path/'link').symlink_to(tmp_path/'real')
    with pytest.raises(v.EvidenceError):v.safe_path(tmp_path,'link')

def test_nested_contract_edits_and_wrong_method_rejected_by_exact_anchor(tmp_path):
    a=json.loads(v.ANCHORS_PATH.read_text())['8.4']
    name=next(p for p in a['files'] if p.endswith('.verification_report.draft.json'))
    original=json.loads((ROOT/name).read_text());target=tmp_path/name;target.parent.mkdir(parents=True)
    for mutate in [
        lambda d:d.update(notebook_id='8_4_circuit_tracing_with_attribution_graphs'),
        lambda d:d['tests'][0].update(extra=True),
        lambda d:d['criteria_results'][0].update(observed_pass=False),
        lambda d:d['observations'][0].update(plain_output='altered'),
        lambda d:d.update(raw_controller_status='failed')
    ]:
        d=copy.deepcopy(original);mutate(d);target.write_text(json.dumps(d))
        with pytest.raises(v.EvidenceError):v.require_hash(tmp_path,name,a['files'][name]['sha256'])

def test_anchor_manifest_cannot_be_rewritten(tmp_path):
    target=tmp_path/'scripts/fresh_lesson_evidence_anchors.json';target.parent.mkdir();d=json.loads(v.ANCHORS_PATH.read_text());d['8.4']['method']='8_4_circuit_tracing_with_attribution_graphs';target.write_text(json.dumps(d))
    with pytest.raises(v.EvidenceError):v.verify('8.4',tmp_path)

def test_exact_minivlm_recovery_and_probe_assertions():
    base=ROOT/'docs/mastery/evidence/minivlm_controls_20261016/12.3_attempt2'
    raw=json.loads((base/'FAILED.ipynb').read_text());recovered=json.loads((base/'recovered_completed.ipynb').read_text())
    v.verify_recovery(raw,recovered)
    assert len(v.code_cells(raw))==62
    assertions=v.notebook_assertions(raw)
    assert any(a['code_ordinal']==61 for a in assertions)
    for field,value in [('execution_count',None),('outputs',[]),('source',['pass'])]:
        changed=copy.deepcopy(recovered);v.code_cells(changed)[-1][field]=value
        with pytest.raises(v.EvidenceError):v.verify_recovery(raw,changed)
    changed=copy.deepcopy(recovered);changed['metadata']['widgets']={}
    with pytest.raises(v.EvidenceError):v.verify_recovery(raw,changed)

def test_observations_recomputed_not_pass_flags(sequence):
    raw,_=sequence;cells=v.code_cells(raw);before=v.observations(cells)
    next(c for c in cells if v.output_text(c).strip())['outputs']=[{'output_type':'stream','name':'stdout','text':['changed']}]
    assert v.observations(cells)!=before

def test_minivlm_saved_arithmetic_and_tampered_values():
    trace=json.loads((ROOT/'docs/mastery/evidence/minivlm_controls_20261016/12.3_attempt2/scientific_trace.json').read_text())
    result=v.verify_trace_arithmetic(trace)
    assert result['full question/count']=={'correct':998,'denominator':1000,'accuracy':.998}
    trace['qwen_pair_traces'][0]['recovery']+=.01
    with pytest.raises(v.EvidenceError):v.verify_trace_arithmetic(trace)

@pytest.fixture
def anchored_temp_root(tmp_path):
    """Full small-file/archival closure; intentionally no cache blobs copied."""
    import shutil
    a=json.loads(v.ANCHORS_PATH.read_text())['7.1']
    names=set(a['files'])|{'scripts/fresh_lesson_evidence_anchors.json',a['historical_source_archive']['path']}
    lock_name=next(n for n in a['files'] if n.endswith('.draft.yml'))
    import yaml
    contract=yaml.safe_load((ROOT/lock_name).read_text())
    names.update(e['path'] for e in contract['publication_files_separate_from_execution_inputs'])
    names.add(a['base']+'/7.1/executed.ipynb')
    for n in names:
        dest=tmp_path/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    return tmp_path,a

@pytest.mark.parametrize('target', ['runtime','kernel','runner','protocol','cache_manifest','dependency','wrong_method','report_output','consistent_report_and_anchor','missing_same_cell_success'])
def test_verify_entrypoint_rejects_mutations_before_cache(anchored_temp_root,target):
    root,a=anchored_temp_root
    if target=='runtime':name=next(n for n in a['files'] if n.endswith('/kernel_runtime.json'))
    elif target=='kernel':name=next(n for n in a['files'] if n.endswith('/kernel.json'))
    elif target=='runner':name=a['base']+'/runner.py'
    elif target=='protocol':name=a['base']+'/protocol.json'
    elif target=='cache_manifest':name=a['base']+'/cache_manifest.json'
    elif target=='dependency':name=a['directory']+'/tests.py'
    elif target=='missing_same_cell_success':name=a['base']+'/7.1/executed.ipynb'
    else:name=next(n for n in a['files'] if n.endswith('.verification_report.draft.json'))
    p=root/name
    if target in ('runner','dependency'):p.write_text(p.read_text()+'\n# changed frozen execution dependency\n')
    elif target=='cache_manifest':
        d=json.loads(p.read_text());d[0]['revision']='0'*40;p.write_text(json.dumps(d))
    else:
        d=json.loads(p.read_text())
        if target=='runtime':d['torch']='wrong-version'
        elif target=='protocol':d['versions']['torch']='wrong-version'
        elif target=='kernel':d['argv'][0]='/different/python'
        elif target=='wrong_method':d['notebook_id']='8_4_circuit_tracing_with_attribution_graphs'
        elif target in ('report_output','consistent_report_and_anchor'):d['observations'][0]['plain_output']='fabricated observation'
        else:
            c=next(c for c in v.code_cells(d) if v.test_calls(c));c['outputs']=[]
        p.write_text(json.dumps(d))
    if target=='consistent_report_and_anchor':
        anchor=root/'scripts/fresh_lesson_evidence_anchors.json'
        all_anchors=json.loads(anchor.read_text())
        entry=all_anchors['7.1']['files'][name]
        entry.update(sha256=v.sha(p.read_bytes()),size_bytes=p.stat().st_size)
        anchor.write_text(json.dumps(all_anchors))
    with pytest.raises(v.EvidenceError) as exc:v.verify('7.1',root)
    assert 'staged cache' not in str(exc.value), 'mutation must be caught before absent cache fixture'

def test_ast_ignores_comments_and_string_fake_calls():
    cell={'source':['# tests.test_fake_comment(x)\ntext = "tests.test_fake_string(x)"\nif True:\n    tests.test_real(x)\n']}
    assert v.test_calls(cell)==['test_real']
