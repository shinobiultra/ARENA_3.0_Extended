"""15.1 preservation and independent CPU math; no model training, CUDA or downloads."""
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

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / 'chapter15_peft_misalignment'
OLD = CHAPTER / 'exercises/part1_lora_dora_adapter_controls'
SECTION = CHAPTER / 'exercises/part1_lora_dora'
EVIDENCE = ROOT / 'docs/mastery/evidence/lora_dora_master_migration'
MASTER = ROOT / 'infrastructure/chapters/chapter15_peft_misalignment/master_15_1.py'
MODULE = 'chapter15_peft_misalignment.exercises.part1_lora_dora'
STEM = '15.1_LoRA_and_DoRA_from_Scratch'
OLD_STEM = '15.1_LoRA_DoRA_and_Adapter_Controls'

def sha(data): return hashlib.sha256(data).hexdigest()
def source(cell): return ''.join(cell['source'])
def codes(notebook): return [c for c in notebook['cells'] if c['cell_type']=='code']
def read(kind): return nbformat.read(SECTION/f'{STEM}_{kind}.ipynb',as_version=4)
def old(kind): return json.loads((OLD/f'{OLD_STEM}_{kind}.ipynb').read_text())
def normalize_route(text):
    return text.replace('section = "part1_lora_dora"','section = "part1_lora_dora_adapter_controls"').replace('asset_dir = section_dir / "media"','asset_dir = root_dir / chapter / "instructions" / "assets"').replace('import part1_lora_dora.tests as tests','import part1_lora_dora_adapter_controls.tests as tests').replace('from part1_lora_dora import historical_release as reference','from part1_lora_dora_adapter_controls import solutions as reference')


def test_archive_and_all_old_files_physically_unchanged():
    from scripts.evidence_bundles import validate_review_bundle
    before=json.loads((EVIDENCE/'before.json').read_text())
    archive=ROOT/before['archive']
    assert sha(archive.read_bytes())==before['archive_sha256']
    contract=validate_review_bundle(archive)
    assert len(contract['members'])==20
    assert len(before['legacy_inventory'])==17 and before['nested_original_receipts']==[]
    with tarfile.open(archive) as saved:
        assert set(saved.getnames())==set(before['files'])
        for name,digest in before['files'].items(): assert sha(saved.extractfile(name).read())==digest
    for name,digest in before['legacy_inventory'].items(): assert sha((ROOT/name).read_bytes())==digest,name
    for name in ('artifacts.lock.yml','verification_report.json','verification_report.schema.json'):
        assert (SECTION/name).read_bytes()==(OLD/name).read_bytes()
    assert all(r['matches'] for r in before['report_input_physical_checks'])
    assert len(before['report_input_physical_checks'])==14
    assert json.loads((SECTION/'verification_report.json').read_text())['report_inputs']==before['report_inputs']


def test_every_historical_ast_nonblank_line_output_and_execution_count_preserved():
    original=old('solutions'); normalized=nbformat.reads(json.dumps(original),as_version=4)
    current=read('solutions');receipt=json.loads((EVIDENCE/'output_parity.json').read_text())
    assert receipt['historical_code_cells']==len(codes(original))==17
    assert receipt['appended_unexecuted_cells']==13
    assert len(codes(current))==30
    assert receipt['fresh_model_execution'] is False and receipt['canonical_notebook_freshly_executed'] is False
    used=[]
    for ordinal,(a,record) in enumerate(zip(codes(original),receipt['cells'],strict=True)):
        index=record['canonical_code_ordinal'];used.append(index)
        b=codes(current)[index];adjusted=normalize_route(b.source) if ordinal==0 else b.source
        before=source(a);dump=ast.dump(ast.parse(adjusted))
        assert ast.dump(ast.parse(before))==dump,ordinal
        assert [s for s in before.splitlines() if s.strip()]==[s for s in adjusted.splitlines() if s.strip()]
        assert record['historical_ordinal']==ordinal
        assert record['published_source_sha256']==sha(before.encode())
        assert record['generated_source_sha256']==sha(b.source.encode())
        assert record['normalized_ast_sha256']==sha(dump.encode())
        assert record['outputs_sha256']==sha(json.dumps(a.get('outputs',[]),sort_keys=True).encode())
        assert b.execution_count==a['execution_count']==record['execution_count']
        assert b.outputs==codes(normalized)[ordinal].outputs
        assert bool(record['named_adaptation'])==(ordinal==0)
    assert used==sorted(set(used))
    assert sum(len(codes(current)[i].outputs) for i in used)==9
    fresh=json.loads((EVIDENCE/'fresh_subset_execution.json').read_text())
    fresh_rows={r['canonical_code_ordinal']:r for r in fresh['fresh_code_cells']}
    assert set(fresh_rows)==set(range(30))-set(used)
    for i,row in fresh_rows.items():
        cell=codes(current)[i]
        assert cell.execution_count==row['actual_kernel_execution_count']
        assert sha(json.dumps(cell.outputs,sort_keys=True).encode())==row['outputs_sha256']
    assert current.metadata['arena_execution_provenance']['execution_kind']=='mixed_historical_reference_and_fresh_cpu_subset'


def test_every_student_stub_completes_from_its_own_answer():
    exercises,solutions=read('exercises'),read('solutions');nbformat.validate(exercises);nbformat.validate(solutions)
    completed=[];stubs=set()
    for index,cell in enumerate(exercises.cells):
        if cell.cell_type!='code':continue
        text=cell.source
        if 'raise NotImplementedError' in text:
            following=[]
            for other in exercises.cells[index+1:]:
                if other.cell_type=='markdown':
                    following.append(other.source)
                    if '<summary>Solution</summary>' in other.source:break
                    if other.source.startswith('### Exercise'):break
            answers=re.findall(r'<summary>Solution</summary>(.*?)</details>','\n'.join(following),re.S)
            assert len(answers)==1
            answer='\n\n'.join(re.findall(r'```python\n(.*?)```',answers[0],re.S))
            definitions={n.name:ast.get_source_segment(answer,n) for n in ast.parse(answer).body if isinstance(n,ast.FunctionDef)}
            lines=text.splitlines(True)
            for n in reversed(ast.parse(text).body):
                if isinstance(n,ast.FunctionDef) and 'raise NotImplementedError' in ast.get_source_segment(text,n):
                    stubs.add(n.name);lines[n.lineno-1:n.end_lineno]=[definitions[n.name]+'\n']
            text=''.join(lines);assert 'raise NotImplementedError' not in text
        completed.append(ast.dump(ast.parse(text)))
    assert completed==[ast.dump(ast.parse(c.source)) for c in codes(solutions)]
    assert stubs=={'lora_delta','adapter_delta_report','lora_merge_max_abs_diff','dora_recompose_weight','dora_weight_report','intruder_dimension_report','adapter_mechanism_report','make_proxy_batch','evaluate_proxy_adapter','train_proxy_adapter','same_norm_random_adapter_control','compare_proxy_methods','build_signature_payload'}
    # Immediate subfunction checks survive as separate added cells, before the next exercise.
    for name in stubs:
        for i,c in enumerate(exercises.cells):
            if c.cell_type=='code' and re.search(r'^def '+name+r'\(',c.source,re.M):
                following='\n'.join(x.source for x in exercises.cells[i+1:i+6] if x.cell_type=='code')
                assert 'tests.test_' in following,name
                break


def test_full_prose_controls_and_method_specific_forecasts_survive():
    text='\n'.join(c.source for c in read('exercises').cells if c.cell_type=='markdown')
    compact=' '.join(text.split())
    for c in old('exercises')['cells']:
        if c['cell_type']!='markdown':continue
        original=source(c)
        if original.startswith('# [15.1]'):original=original.split('\n',1)[1]
        original=original.replace('## Learning Objectives','## What you will implement').replace('## Setup','## Setup code').replace('<summary>Solution</summary>','<summary>Historical worked formulation</summary>').replace('../../instructions/assets/','media/')
        original=re.sub(r'> Difficulty: (easy|medium|hard)\s*\n> Importance: high\s*\n> Time: (\d+) minutes',lambda m:'> ```yaml\n> Difficulty: '+{'easy':'🔴⚪⚪⚪⚪','medium':'🔴🔴🔴⚪⚪','hard':'🔴🔴🔴🔴⚪'}[m[1]]+'\n> Importance: 🔵🔵🔵🔵⚪\n>\n> You should spend '+m[2]+' minutes on this exercise.\n> ```',original)
        assert ' '.join(original.split()) in compact,original[:100]
    assert text.count('### Forecast')==text.count('### Reflect')==12
    for term in ('rank-matched','learning rate','protected coordinate','exposure','confidence','same-norm','random-label','guided practice','unseen mastery','90 degrees','zero update','full DoRA change'):
        assert term in text,term
    for p in (MASTER,SECTION/'solutions.py',SECTION/'tests.py'):
        assert 'arena_ext' not in p.read_text()
    page=CHAPTER/'instructions/pages/01_[15.1]_LoRA_and_DoRA_from_Scratch.md'
    assert page.exists()
    for name in ('lora_dora_adapter_controls_signature_panel.png', 'lora_dora_adapter_controls_behavior_table.png', 'lora_dora_cpu_direction_magnitude.png'):
        assert 'part1_lora_dora/media/' + name in page.read_text()
    assert (CHAPTER/'instructions/pages/01_[15.1]_LoRA_and_DoRA_from_Scratch.py').exists()
    assert 'Fresh CPU exact-matrix probe' in text
    assert 'media/lora_dora_cpu_direction_magnitude.png' in text


def test_every_historical_image_byte_preserved():
    media=SECTION/'media'
    for p in (CHAPTER/'instructions/assets').glob('lora_dora_adapter_controls*'):
        assert (media/p.name).read_bytes()==p.read_bytes()
    for i,c in enumerate(old('solutions')['cells']):
        for j,o in enumerate(c.get('outputs',[])):
            value=o.get('data',{}).get('image/png')
            if value:
                if isinstance(value,list):value=''.join(value)
                assert (media/f'lora_dora_historical_cell_{i}_output_{j}.png').read_bytes()==base64.b64decode(value)


@pytest.fixture(scope='module')
def lesson():return importlib.import_module(MODULE+'.solutions')


@pytest.mark.parametrize('name,args',[
('test_lora_delta_uses_scaled_b_matrix_times_a_matrix',['lora_delta']),
('test_adapter_delta_report_records_rank_alpha_and_nonzero_update',['adapter_delta_report']),
('test_lora_delta_rank_bound_and_svd_spectrum',['lora_delta']),
('test_lora_merge_unmerge_parity_is_independent_of_report',['lora_merge_max_abs_diff']),
('test_dora_recompose_weight_preserves_target_row_magnitudes',['dora_recompose_weight','dora_weight_report']),
('test_dora_recomposition_handles_nonzero_delta_direction',['dora_recompose_weight']),
('test_intruder_dimension_report_measures_projection_fraction',['intruder_dimension_report']),
('test_adapter_mechanism_report_requires_accuracy_and_mechanism',['adapter_mechanism_report']),
('test_proxy_batch_has_planted_target_and_ood_conflict',['make_proxy_batch']),
('test_evaluation_on_known_weights',['evaluate_proxy_adapter','ProxyFullFinetuneClassifier']),
('test_same_norm_control_without_training',['same_norm_random_adapter_control','ProxyLoRAClassifier']),
('test_signature_gate_without_training',['build_signature_payload']),
])
def test_local_independent_cpu_oracles(lesson,name,args):
    checks=importlib.import_module(MODULE+'.tests')
    getattr(checks,name)(*[getattr(lesson,x) for x in args])


def test_oracles_reject_nonzero_and_high_accuracy_but_wrong_methods(lesson):
    import torch as t
    from types import SimpleNamespace
    checks=importlib.import_module(MODULE+'.tests')
    with pytest.raises(AssertionError):checks.test_lora_delta_uses_scaled_b_matrix_times_a_matrix(lambda a,b,alpha:t.zeros(2,2))
    with pytest.raises(AssertionError):checks.test_dora_recomposition_handles_nonzero_delta_direction(lambda w,d,m:(w+d)*m[:,None])
    with pytest.raises(AssertionError):checks.test_adapter_mechanism_report_requires_accuracy_and_mechanism(lambda **kw:SimpleNamespace(accuracy_delta=.2,mechanism_delta=.05,accuracy_improved=True,mechanism_preserved=True,adapter_acceptable=True))
    with pytest.raises(AssertionError):checks.test_intruder_dimension_report_measures_projection_fraction(lambda *a,**kw:SimpleNamespace(projection_fraction=0.0,intruder_detected=False))
    wrong = type(lesson.build_signature_payload)((lambda **kw: {'signature_passed': True}).__code__, {'compare_proxy_methods': None})
    with pytest.raises(AssertionError):checks.test_signature_gate_without_training(wrong)


def test_import_is_inert_from_unrelated_directory(tmp_path):
    script='''
import contextlib, hashlib, importlib, io, pathlib, sys
import torch as t
import matplotlib.pyplot as plt
import pandas
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root))
section=root/'chapter15_peft_misalignment/exercises/part1_lora_dora'
def files():return {p.relative_to(section).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in section.rglob('*') if p.is_file()}
def forbidden(*args,**kwargs):raise AssertionError('CUDA or optimizer invoked during import')
t.cuda.is_available=t.cuda.init=t.cuda.device_count=forbidden
t.optim.AdamW=forbidden
rng=t.get_rng_state().clone();grad=t.is_grad_enabled();before=files();out=io.StringIO()
with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
    module=importlib.import_module('chapter15_peft_misalignment.exercises.part1_lora_dora.solutions')
assert out.getvalue()==''
assert t.equal(rng,t.get_rng_state()) and grad==t.is_grad_enabled()
assert before==files()
assert not module.MAIN
assert 'part1_lora_dora_adapter_controls.solutions' not in sys.modules
print('inert import passed')
'''
    result=subprocess.run([sys.executable,'-c',script,str(ROOT)],cwd=tmp_path,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'inert import passed' in result.stdout


def test_training_and_release_paths_are_visible_but_not_called_by_import(lesson):
    tree=ast.parse((SECTION/'solutions.py').read_text())
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    train=ast.get_source_segment((SECTION/'solutions.py').read_text(),functions['train_proxy_adapter'])
    for term in ('t.optim.AdamW','loss.backward()','optimizer.step()','random_labels=random_labels','eval_every','data_seed * 1000 + step'):
        assert term in train
    for name in ('run_gpu_test','run_full_experiment'):
        assert name in functions
    release=(SECTION/'historical_release.py').read_text()
    assert 'part1_lora_dora_adapter_controls import solutions' in release
    assert all(isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant)) for n in ast.parse(release).body)


def test_cpu_play_receipt_binds_exact_source_and_no_training_plot():
    receipt=json.loads((EVIDENCE/'cpu_play.json').read_text())
    play=next(c.source for c in codes(read('solutions')) if c.source.startswith('PLAY_UPDATE_SCALE ='))
    assert receipt['execution_kind']=='fresh_cpu_exact_matrix_probe'
    assert receipt['source_sha256']==sha(play.encode())
    assert receipt['training_performed'] is False and receipt['cuda_initialized'] is False
    assert receipt['elapsed_seconds']<30
    assert receipt['combinations_checked']==36
    assert receipt['default']['dora_row_norms']==[10.,5.]
    assert receipt['default']['projection_fraction']==0
    assert receipt['default']['control_projection_fraction']==1
    assert abs(receipt['default']['lora_row_norms'][0]-32**.5)<1e-6
    figure=ROOT/receipt['figure']['path']
    assert sha(figure.read_bytes())==receipt['figure']['sha256']


def test_guided_proposal_has_strict_schema_bound_metric_and_exposed_transfer():
    from mastery_hub.units.extension_predictions.lesson import validate_case, _digest
    proposal=json.loads((EVIDENCE/'proposed_case.json').read_text())
    identity=json.loads((EVIDENCE/'proposed_case_identity.json').read_text())
    assert len(proposal)==20
    assert validate_case(proposal)==proposal
    assert identity['proposal_sha256']==sha((EVIDENCE/'proposed_case.json').read_bytes())
    assert identity['case_fingerprint']==_digest(proposal)
    # Original whole-notebook hashes remain immutable historical metadata.
    # Current semantics are checked against the adopted integration binding below.
    binding=json.loads((EVIDENCE/'current_integration_binding.json').read_text())
    assert identity['proposal_sha256']==binding['historical_proposal_sha256']
    assert sha((EVIDENCE/'proposed_case_identity.json').read_bytes())==binding['historical_identity_sha256']
    assert_current_integration_binding(binding, read('exercises'), read('solutions'))
    assert proposal['prediction_type']=='numeric' and proposal['transfer_prediction_type']=='binary'
    assert proposal['evidence_class']=='analytic_toy' and proposal['status']=='guided_practice'
    assert 'BEFORE recomposition' in proposal['config_fields']['student_method']
    assert 'norm(delta)' in proposal['config_fields']['metric_and_selection']
    assert '90' in proposal['transfer'] and 'already computed' in proposal['transfer']
    assert 'float32' in proposal['transfer']
    assert all(key not in proposal for key in ('forecast','confidence','causal_reasoning','exposure','observed_controls','discrepancy_explanation','changed_condition_forecast','target','target_answer','answer'))
    receipt=json.loads((EVIDENCE/'cpu_play.json').read_text())
    def lookup(path):
        value=receipt
        for key in path:value=value[key]
        return value
    assert lookup(identity['primary_reference_path'])==receipt['default']['projection_fraction']
    row=receipt['changed_conditions'][25]
    assert (row['scale'],row['angle_degrees'],row['magnitude_scale'])==(1.,90.,1.)
    assert lookup(identity['transfer_reference_path'])>.2
    forecast=next(c.source for c in read('exercises').cells if c.cell_type=='markdown' and '### Forecast — magnitude is not mechanism' in c.source)
    assert 'sqrt(32)' not in forecast and 'projection zero' not in forecast and 'projection one' not in forecast


def semantic_code_records(notebook):
    return [{
        'source_sha256':sha(c.source.encode()),
        'outputs_sha256':sha(json.dumps(c.get('outputs',[]),sort_keys=True).encode()),
        'execution_count':c.get('execution_count'),
        'execution_origin':c.metadata.get('arena_execution_origin'),
    } for c in codes(notebook)]


def assert_current_integration_binding(binding, exercises, solutions):
    from mastery_hub.units.extension_predictions.lesson import validate_case, _digest
    from scripts.evidence_bundles import validate_review_bundle
    proposal=json.loads((EVIDENCE/'proposed_case.json').read_text())
    adopted=next(c for c in json.loads((ROOT/'mastery_hub/extension_checkpoints.json').read_text())['checkpoints'] if c['section']=='15.1')
    assert validate_case(adopted)==proposal
    assert _digest(adopted)==binding['case_fingerprint']
    for path,digest in binding['artifact_sha256'].items():assert sha((ROOT/path).read_bytes())==digest,path
    snapshot=binding['historical_metadata_snapshot']
    archive=ROOT/snapshot['archive']
    assert sha(archive.read_bytes())==snapshot['sha256']
    contract=validate_review_bundle(archive)
    assert len(contract['members'])==3
    with tarfile.open(archive) as saved:
        for name in ('proposed_case.json','proposed_case_identity.json'):
            key='docs/mastery/evidence/lora_dora_master_migration/'+name
            assert saved.extractfile(key).read()==(EVIDENCE/name).read_bytes()
        # The manifest had already been refreshed by root after navigation was added;
        # it is a separate historical record, not the proposal-time notebook bytes.
        manifest_member='docs/mastery/evidence/lora_dora_master_migration/manifest.json'
        assert sha(saved.extractfile(manifest_member).read())==snapshot['members'][manifest_member]['sha256']
        original_identity=json.loads(saved.extractfile('docs/mastery/evidence/lora_dora_master_migration/proposed_case_identity.json').read())
        frozen_notebook_digests={
            'exercises':'947bee60de49fa8a90509ffa941bdae71af807adbb294da18ef0c5ab60a62cf8',
            'solutions':'487c17d7a2e0474a6dbeac8e4667a9f28824cb4f07996d4765658b18aa441c59',
        }
        for kind,digest in frozen_notebook_digests.items():
            key=str((SECTION/f'{STEM}_{kind}.ipynb').relative_to(ROOT))
            assert original_identity['artifact_sha256'][key]==digest
    assert semantic_code_records(exercises)==binding['notebook_code_records']['exercises']
    assert semantic_code_records(solutions)==binding['notebook_code_records']['solutions']


def test_current_semantic_binding_rejects_code_or_output_changes_but_allows_navigation():
    from copy import deepcopy
    binding=json.loads((EVIDENCE/'current_integration_binding.json').read_text())
    exercises,solutions=read('exercises'),read('solutions')
    navigation=deepcopy(solutions)
    navigation.cells.insert(0,nbformat.v4.new_markdown_cell('A navigation-only link'))
    assert_current_integration_binding(binding,exercises,navigation)
    changed=deepcopy(solutions)
    play=next(c for c in codes(changed) if c.source.startswith('PLAY_UPDATE_SCALE ='))
    play.source=play.source.replace('PLAY_UPDATE_SCALE = 1.0','PLAY_UPDATE_SCALE = 2.0')
    with pytest.raises(AssertionError):assert_current_integration_binding(binding,exercises,changed)
    changed=deepcopy(solutions)
    codes(changed)[3].outputs.append(nbformat.v4.new_output('stream',name='stdout',text='fabricated pass'))
    with pytest.raises(AssertionError):assert_current_integration_binding(binding,exercises,changed)


def test_all13_added_cells_have_real_frozen_subset_execution_and_actual_origins():
    protocol=json.loads((EVIDENCE/'fresh_subset_protocol.json').read_text())
    receipt=json.loads((EVIDENCE/'fresh_subset_execution.json').read_text())
    execution=nbformat.read(EVIDENCE/'fresh_subset_execution.ipynb',as_version=4)
    assert receipt['status']=='passed' and receipt['failure'] is None
    assert receipt['protocol_sha256']==sha((EVIDENCE/'fresh_subset_protocol.json').read_bytes())
    assert receipt['execution_notebook_sha256']==sha((EVIDENCE/'fresh_subset_execution.ipynb').read_bytes())
    assert 0<receipt['elapsed_seconds']<protocol['budget_seconds']==30
    assert receipt['optimizer_steps']==protocol['training_protocol']['optimizer_steps_total']==440
    assert receipt['fresh_tiny_training_fits']==protocol['training_protocol']['fits']==6
    assert receipt['canonical_notebook_freshly_executed'] is False
    assert receipt['historical_runtime_statements_executed'] is False
    assert receipt['historical_study_rerun'] is False
    assert len(codes(execution))==14
    assert codes(execution)[0].source==protocol['support_cell_source']
    assert sha(codes(execution)[0].source.encode())==protocol['support_cell_sha256']
    current=codes(read('solutions'))
    assert len(protocol['fresh_code_cells'])==len(receipt['fresh_code_cells'])==13
    for index,(frozen,row) in enumerate(zip(protocol['fresh_code_cells'],receipt['fresh_code_cells'],strict=True),1):
        actual=codes(execution)[index]
        cell=current[frozen['canonical_code_ordinal']]
        assert actual.source==cell.source==frozen['source']
        assert sha(actual.source.encode())==frozen['source_sha256']==row['source_sha256']
        assert actual.execution_count==actual.metadata['actual_kernel_reply_execution_count']
        assert actual.execution_count==row['actual_kernel_execution_count']==cell.execution_count
        assert receipt['completed_kernel_replies'][str(index)]==actual.execution_count
        assert actual.outputs==cell.outputs and not any(o.output_type=='error' for o in actual.outputs)
        assert cell.metadata['arena_execution_origin']['session_id']==protocol['session_id']
        assert sha(json.dumps(actual.outputs,sort_keys=True).encode())==row['outputs_sha256']
    module=(SECTION/'solutions.py').read_text()
    nodes={n.name:n for n in ast.parse(module).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert len(protocol['minimum_definition_closure'])==29
    for frozen in protocol['minimum_definition_closure']:
        node=nodes[frozen['name']]
        start=min([node.lineno]+[d.lineno for d in node.decorator_list])
        actual='\n'.join(module.splitlines()[start-1:node.end_lineno])
        assert actual==frozen['source'] and sha(actual.encode())==frozen['sha256']
    assert len(receipt['figures'])==1
    for figure in receipt['figures']:assert sha((ROOT/figure['path']).read_bytes())==figure['sha256']
    banner='\n'.join(c.source for c in read('exercises').cells if c.cell_type=='markdown')
    assert 'Mixed execution provenance' in banner
    assert 'seventeen original code cells' in banner and 'thirteen added check/play cells' in banner
    failed=json.loads((EVIDENCE/'socket_startup_failure/fresh_subset_execution.json').read_text())
    assert failed['status']=='failed' and failed['completed_kernel_replies']=={}
