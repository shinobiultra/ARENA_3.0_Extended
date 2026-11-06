"""Engineering-only DEV, never the public target. See the module-local fingerprint guard."""
from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / 'mastery_hub/units/sae_prediction'

def test_candidate_has_reviewed_seven_method_contract():
    # A missing or reordered taught method must fail this engineering contract.
    assert (UNIT / 'lesson.py').is_file(), 'reviewed seven-method companion missing'
    from mastery_hub.units.sae_prediction import lesson
    assert lesson.METHOD_NAMES == ('affine','relu_encode','topk_encode','gated_encode','jump_encode','decode','measure_rows')

import json
import os
import sys
import copy
sys.path.insert(0, str(ROOT))

# Every numerical fixture is 2 examples, 2 dimensions, 3 features.
DEV = {
 'id':'DEV-two-by-two-three', 'example_order':['u','v'], 'feature_order':['a','b','c'],
 'method_order':['relu','topk','gated','jumprelu'],
 'X':[[1.,2.],[-1.,0.]], 'D':[[1.,0.],[0.,1.],[1.,1.]],
 'W_enc':[[1.,0.,1.],[0.,1.,-1.]], 'b_enc':[0.,0.,.5],
 'W_gate':[[1.,0.,0.],[0.,1.,1.]],'b_gate':[-.5,0.,-1.],
 'b_dec':[0.,0.], 'jump_threshold':[1.,.5,.25], 'k':1,
 'decoder_permutation':[1,2,0], 'coherent_feature_permutation':[2,0,1],
 'dtype':'float64','device':'cpu','threads':1,
 'tie_policy':'descending positive value then ascending original feature index',
 'gate_boundary':'strictly_greater_than_zero','jump_boundary':'strictly_greater_than_threshold',
 'activation_count_boundary':'strictly_greater_than_zero',
 'metric':'mean_squared_error(topk)-mean_squared_error(jumprelu)'}

@pytest.fixture
def api():
    from mastery_hub.units.sae_prediction import records, runner, lesson
    return records, runner, lesson

@pytest.fixture
def source(api):
    return api[2].SOLUTION_SOURCE

@pytest.fixture
def committed(api,source,tmp_path,monkeypatch):
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    r,_,_=api
    identity=r._freeze(source, DEV, 'engineering-dev')
    return r.commit(identity, -.25,.6,'A DEV-only arbitrary prediction.', 'engineering-exposed')

def tensor(values):
    import torch
    return torch.tensor(values,dtype=torch.float64)

def test_hand_calculated_forward_methods(api, source):
    import torch
    r,_,_=api; ns=r.namespace(source)
    M=ns['affine'](tensor(DEV['X']),tensor(DEV['W_enc']),tensor(DEV['b_enc']))
    assert M.tolist()==[[1.,2.,-.5],[-1.,0.,-.5]]
    assert ns['relu_encode'](M).tolist()==[[1.,2.,0.],[0.,0.,0.]]
    assert ns['topk_encode'](M,1,(0,1,2)).tolist()==[[0.,2.,0.],[0.,0.,0.]]
    assert ns['gated_encode'](tensor([[2.,-1.,3.],[1.,2.,0.]]),tensor([[-1.,1.,0.],[1.,0.,1.]])).tolist()==[[0.,0.,0.],[1.,0.,0.]]
    assert ns['jump_encode'](tensor([[1.,.5,.75],[0.,2.,-.5]]),tensor([1.,.5,.25])).tolist()==[[0.,0.,.75],[0.,2.,0.]]
    Z=tensor([[1.,2.,0.],[0.,0.,0.]])
    R=ns['decode'](Z,tensor(DEV['D']),tensor([.5,-.5]))
    assert R.tolist()==[[1.5,1.5],[.5,-.5]]
    E,L=ns['measure_rows'](tensor(DEV['X']),Z,R)
    assert E.tolist()==[.25,1.25] and L.tolist()==[2,0] and L.dtype==torch.int64
    assert ns['decode'](Z*0,tensor(DEV['D']),tensor([.5,-.5])).tolist()==[[.5,-.5],[.5,-.5]]

@pytest.mark.parametrize('k',[False,1.0,-1,4])
def test_topk_invalid_k(api,source,k):
    with pytest.raises(ValueError):api[0].namespace(source)['topk_encode'](tensor([[1.,2.,3.],[0.,0.,0.]]),k,(0,1,2))

def test_ties_original_ids_and_boundaries(api,source):
    ns=api[0].namespace(source); M=tensor([[2.,2.,-1.],[0.,-1.,-1.]])
    assert ns['topk_encode'](M,1,(9,2,4)).tolist()==[[0.,2.,0.],[0.,0.,0.]]
    assert ns['topk_encode'](M,0,(9,2,4)).tolist()==[[0.,0.,0.],[0.,0.,0.]]
    assert ns['topk_encode'](M,3,(9,2,4)).tolist()==[[2.,2.,0.],[0.,0.,0.]]
    assert ns['topk_encode'](M[:,[2,0,1]],1,(4,9,2)).tolist()==[[0.,0.,2.],[0.,0.,0.]]
    for theta in ([-1.,0.,0.],[float('inf'),0.,0.]):
        with pytest.raises(ValueError):ns['jump_encode'](M,tensor(theta))
    with pytest.raises(ValueError):ns['gated_encode'](M,tensor([[1.,2.],[2.,3.]]))
    with pytest.raises(ValueError):ns['topk_encode'](M,1,(True,2,3))
    with pytest.raises(ValueError):ns['topk_encode'](M,1,(2,2,3))

@pytest.mark.parametrize('suffix',["\nprint('side effect')",'\n@print\ndef extra():\n pass'])
def test_definition_only_guard(api,source,suffix):
    with pytest.raises(ValueError):api[0]._freeze(source+suffix,DEV,'dev')

@pytest.mark.parametrize('prediction,confidence,reasoning',[(None,.5,'x'),(True,.5,'x'),(float('inf'),.5,'x'),(1,True,'x'),(1,1.1,'x'),(1,.5,' ')])
def test_commit_rejects_blank_and_invalid(api,source,prediction,confidence,reasoning):
    identity=api[0]._freeze(source,DEV,'dev')
    with pytest.raises(ValueError):api[0].commit(identity,prediction,confidence,reasoning,'unknown')

def test_identity_snapshots_and_signed_forecast(api,committed):
    r,_,_=api; d=r.read(committed)
    assert d['payload']['prediction']==-.25
    identity=r.read(d['parent'])['payload']
    assert identity['session']=='engineering-dev'
    assert set(identity['sources'])==set(r.SOURCE_PATHS)
    assert identity['recipe']['id']==DEV['id'] and identity['raw_inputs']['X']['shape']==[2,2]
    assert identity['method_source'].startswith('def affine(')
    assert 'torch_config' in identity['runtime']

def test_run_dev_oracle_and_all_conditions(api,source,committed,tmp_path,monkeypatch):
    r,run,_=api; monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    text=run.run(committed,source,tmp_path/'exports',authorized=True)
    d=r.read(text)['payload']; assert d['metric']==0.0  # TopK and Jump keep the same sole positive on u.
    primary=d['conditions']['primary']
    assert primary['M']['values']==[[1.,2.,-.5],[-1.,0.,-.5]]
    assert primary['codes']['topk']['values']==[[0.,2.,0.],[0.,0.,0.]]
    assert primary['errors']['topk']['values']==[.5,.5]
    assert primary['counts']['topk']['values']==[1,0]
    assert set(d['conditions'])=={'primary','coherent','zero','mismatch'}
    assert d['diagnostics']=={'scalar_oracle':True,'coherent_invariance':True}
    assert os.stat(tmp_path/'exports').st_mode&0o777==0o700
    assert all(p.stat().st_mode&0o777==0o600 for p in (tmp_path/'exports').iterdir())
    with pytest.raises(ValueError,match='consumed'):run.run(committed,source,tmp_path/'different-export',authorized=True)

@pytest.mark.parametrize('patch',["return M.clamp(min=0)","return M.clamp(min=0) + 1"])
def test_failure_retains_wrong_finite_arrays(api,source,tmp_path,monkeypatch,patch):
    r,run,_=api; monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    changed=source.replace('return M.clamp(min=0)',patch,1)
    if changed==source:changed=source.replace('return X @ W + b','return X @ W + b + 1')
    c=r.commit(r._freeze(changed,DEV,'bad'),'0' if False else 0,.5,'Explicit incorrect DEV implementation','exposed')
    with pytest.raises(run.AttemptFailure) as exc:run.run(c,changed,tmp_path/'out',authorized=True)
    d=r.read(exc.value.record_text)
    assert d['kind']=='failure' and d['payload']['captured'] and 'forecast_error' not in d['payload']
    assert (tmp_path/'out'/'failure.json').read_text()==exc.value.record_text

def test_mutation_after_exception_and_partial_partner(api,source,tmp_path,monkeypatch):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    changed=source.replace('return X @ W + b',"X.add_(1)\n    raise RuntimeError('original-error')")
    c=r.commit(r._freeze(changed,DEV,'mutate'),0,.5,'DEV mutation case','exposed')
    with pytest.raises(run.AttemptFailure) as exc:run.run(c,changed,tmp_path/'out',authorized=True)
    fail=r.read(exc.value.record_text)['payload'];assert 'original-error' in fail['error']
    assert any('mutat' in v for v in fail['secondary'])
    changed=source.replace('return errors, counts','return errors, None')
    c=r.commit(r._freeze(changed,DEV,'partner'),0,.5,'DEV malformed partner','exposed')
    with pytest.raises(run.AttemptFailure) as exc:run.run(c,changed,tmp_path/'out2',authorized=True)
    assert any('/measure_rows/0' in k for k in r.read(exc.value.record_text)['payload']['captured'])

def test_stored_replay_without_live_sources_runtime_or_runner(api,source,committed,tmp_path,monkeypatch):
    r,run,lesson=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    text=run.run(committed,source,tmp_path/'out',authorized=True)
    def forbidden(*a,**k):raise AssertionError('live numerical operation during replay')
    for name in ('runtime_identity','snapshot_sources','_freeze'):monkeypatch.setattr(r,name,forbidden)
    for name in ('_evaluate','scalar_oracle','_tensors'):monkeypatch.setattr(run,name,forbidden)
    monkeypatch.setattr(r,'namespace',forbidden)
    assert r.read(text)['kind']=='observation'
    fig=lesson.plot(text);assert len(fig.axes)>=12
    assert [label.get_text() for label in fig.axes[0].get_yticklabels()]==['u','v']
    reflected=r.reflect(text,'The DEV rows explain this stored outcome.')
    assert r.read(reflected)['kind']=='reflection'
    p=r.export(reflected,tmp_path/'reflection.json');assert p.read_text()==reflected
    with pytest.raises(FileExistsError):r.export(reflected,p)
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(FileExistsError):r.export(reflected,link)

@pytest.mark.parametrize('alter',['metric','mean','counts','codes','residual','source'])
def test_fabricated_sealed_observation_rejected(api,source,committed,tmp_path,monkeypatch,alter):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    d=r.read(run.run(committed,source,tmp_path/'out',authorized=True));p=d['payload']
    if alter=='metric':p['metric']+=1
    elif alter=='mean':p['means']['relu']+=1
    elif alter=='source':
        c=r.read(d['parent']);ident=r.read(c['parent']);ident['payload']['method_source']+=' '
        c['parent']=r._seal(ident);d['parent']=r._seal(c)
    else:
        key={'counts':'counts','codes':'codes','residual':'residuals'}[alter]
        arr=p['conditions']['primary'][key]['relu'];arr['values'][0]=[10.,10.,10.] if alter=='codes' else ([10.,10.] if alter=='residual' else 3)
        arr['sha256']=r.array_digest(arr['values'],arr['dtype'])
    with pytest.raises(ValueError):r.read(r._seal(d))

@pytest.mark.parametrize('raw',['{"kind":1,"kind":2}','{} {}','{"x":NaN}','\ud800'])
def test_invalid_json(api,raw):
    with pytest.raises(ValueError):api[0].read(raw)

@pytest.mark.parametrize('replacement',[
 'return (X @ W + b).float()', 'return (X @ W + b)[0]',
 'return X @ W + b + float("nan")', 'return X @ W + b + float("inf")',
 'return (X @ W + b).to("meta")'])
def test_invalid_return_dtype_rank_finiteness_device(api,source,tmp_path,monkeypatch,replacement):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    bad=source.replace('return X @ W + b',replacement)
    c=r.commit(r._freeze(bad,DEV,'invalid'),0,.5,'Invalid DEV returned tensor','exposed')
    with pytest.raises(run.AttemptFailure) as exc:run.run(c,bad,tmp_path/'out',authorized=True)
    failure=r.read(exc.value.record_text)['payload'];assert failure['captured']=={}
    assert 'finite CPU float64' in failure['error']

@pytest.mark.parametrize('kind',['runtime','helper','test','source'])
def test_live_mutation_detected_and_secondary_retained(api,source,tmp_path,monkeypatch,kind):
    import shutil
    import torch
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    copyroot=tmp_path/'source-copy'
    for name in r.SOURCE_PATHS:
        dest=copyroot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    monkeypatch.setattr(r,'ROOT',copyroot)
    selected={'helper':r.SOURCE_PATHS[1],'test':'tests/mastery/test_sae_prediction.py','source':r.SOURCE_PATHS[3]}
    mutation='t.set_default_dtype(t.float64)' if kind=='runtime' else f"__import__('pathlib').Path({str(copyroot/selected[kind])!r}).write_text('changed DEV snapshot')"
    bad=source.replace('return X @ W + b',mutation+"\n    raise RuntimeError('primary-failure')")
    c=r.commit(r._freeze(bad,DEV,'mutation'),0,.5,'DEV live identity mutation','exposed')
    previous=torch.get_default_dtype()
    try:
        with pytest.raises(run.AttemptFailure) as exc:run.run(c,bad,tmp_path/'out',authorized=True)
    finally:torch.set_default_dtype(previous)
    failure=r.read(exc.value.record_text)['payload'];assert 'primary-failure' in failure['error']
    assert any('runtime' in v or 'source snapshot' in v for v in failure['secondary'])


def test_function_code_object_mutation_rejected(api,source,tmp_path,monkeypatch):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    bad=source.replace('return X @ W + b',"decode.__code__ = affine.__code__\n    return X @ W + b")
    c=r.commit(r._freeze(bad,DEV,'function-mutate'),0,.5,'DEV function code mutation','exposed')
    with pytest.raises(run.AttemptFailure,match='source'):run.run(c,bad,tmp_path/'out',authorized=True)


def test_input_alias_snapshot_detached_from_later_mutation(api,source,tmp_path,monkeypatch):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    bad=source.replace('return M.clamp(min=0)',"positive = M.clamp(min=0)\n    globals()['previous_return'] = positive\n    return positive",1)
    bad=bad.replace("require(type(k) is int", "previous_return.fill_(999)\n    require(type(k) is int")
    c=r.commit(r._freeze(bad,DEV,'alias'),0,.5,'DEV alias capture check','exposed')
    result=run.run(c,bad,tmp_path/'out',authorized=True)
    assert r.read(result)['payload']['conditions']['primary']['codes']['relu']['values']==[[1.,2.,0.],[0.,0.,0.]]


def test_scalar_oracle_independent_hand_expected(api):
    oracle=api[1].scalar_oracle(DEV)
    primary=oracle['primary']
    assert primary['M']['values']==[[1.,2.,-.5],[-1.,0.,-.5]]
    assert primary['G']['values']==[[.5,2.,1.],[-1.5,0.,-1.]]
    assert primary['reconstructions']['relu']['values']==[[1.,2.],[0.,0.]]
    assert primary['errors']['relu']['values']==[0.,.5]
    assert primary['errors']['topk']['values']==[.5,.5]
    assert primary['errors']['jumprelu']['values']==[.5,.5]
    assert oracle['zero']['errors']['relu']['values']==[2.5,.5]
    assert oracle['mismatch']['reconstructions']['relu']['values']==[[2.,3.],[0.,0.]]


def test_unknown_missing_nested_fields_and_bool_version(api,committed):
    r,_,_=api
    for mutation in (lambda x:x.update(extra=1),lambda x:x.pop('kind'),lambda x:x.update(version=True)):
        obj=r.read(committed);mutation(obj)
        with pytest.raises(ValueError):r.read(r._seal(obj))
    obj=r.read(committed);obj['payload']['extra']=1
    with pytest.raises(ValueError):r.read(r._seal(obj))
    obj=r.read(committed);obj['parent']=committed
    with pytest.raises(ValueError):r.read(r._seal(obj))


def test_source_utf8_limits_and_exact_roundtrip(api,source):
    r,_,_=api
    # A trailing Unicode comment makes exactly the allowed UTF-8 size.
    suffix='\n# ';remaining=r.SOURCE_LIMIT-len((source+suffix).encode());comment='界'*(remaining//3)+'x'*(remaining%3)
    largest=source+suffix+comment
    ident=r._freeze(largest,DEV,'unicode');assert r.read(ident)['payload']['method_source']==largest
    with pytest.raises(ValueError):r._freeze(largest+'x',DEV,'unicode')
    with pytest.raises(ValueError):r._freeze(source+'\ud800',DEV,'utf8')
    with pytest.raises(ValueError):r.read(' '* (r.RECORD_LIMIT+1))


def test_new_commitment_new_output_does_not_reset_session(api,source,committed,tmp_path,monkeypatch):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    run.run(committed,source,tmp_path/'first',authorized=True)
    newer=r.commit(r.read(committed)['parent'],123,.9,'A changed DEV prediction','exposed')
    with pytest.raises(ValueError,match='consumed'):run.run(newer,source,tmp_path/'second',authorized=True)
    # Reloading runner loses no ledger authority.
    import importlib;importlib.reload(run)
    with pytest.raises(ValueError,match='consumed'):run.run(newer,source,tmp_path/'third',authorized=True)


def test_unknown_start_and_failure_write_error_keep_consumed_ledger(api,source,committed,tmp_path,monkeypatch):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    occupied=tmp_path/'occupied';occupied.mkdir()
    with pytest.raises(run.AttemptFailure) as exc:run.run(committed,source,occupied,authorized=True)
    failure=r.read(exc.value.record_text)['payload'];assert failure['write_error']
    ledgers=list((tmp_path/'private').rglob('*.start.json'));assert len(ledgers)==1
    assert ledgers[0].stat().st_mode&0o777==0o600
    assert list((tmp_path/'private').rglob('*.failure.json'))
    with pytest.raises(ValueError,match='consumed'):run.run(committed,source,tmp_path/'retry',authorized=True)


def test_no_forecast_success_or_control_winner_gate(api,source,committed,tmp_path,monkeypatch):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    # Wrong signed forecast still yields an observation, no pass/accuracy field.
    c=r.commit(r.read(committed)['parent'],999.,0.,'Intentionally wrong DEV forecast','exposed')
    d=r.read(run.run(c,source,tmp_path/'out',authorized=True))['payload']
    assert d['forecast_error']==999. and d['metric']==0.
    assert not any('pass' in k or 'accuracy' in k for k in d)


def load_generator():
    spec=importlib.util.spec_from_file_location('candidate_sae_generator',ROOT/'scripts/build_sae_prediction_notebook.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_generator_blank_immediate_tests_dormant_forecast_and_no_numeric_calls(api,source,monkeypatch,tmp_path):
    r,run,lesson=api
    def forbidden(*a,**k):raise AssertionError('numerical entrypoint during source-only operation')
    for name in ('_evaluate','_tensors','scalar_oracle'):monkeypatch.setattr(run,name,forbidden)
    monkeypatch.setattr(r,'namespace',forbidden)
    g=load_generator();nb=g.make_notebook()
    codes=[c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c['execution_count'] is None and c['outputs']==[] for c in codes)
    for i,c in enumerate(nb['cells']):
        if c['id'].endswith('-source'):assert nb['cells'][i+1]['id']==c['id'].replace('-source','-dev-test')
    forecast=next(c['source'] for c in codes if c['id']=='forecast')
    assert "prediction = None\nconfidence = None\nreasoning = ''\nprior_exposure = 'unknown'" in forecast
    assert 'RUN_AUTHORIZED_TARGET = False' in next(c['source'] for c in codes if c['id']=='dormant-run')
    # Public target raw identity may be serialized, but no forward work happens.
    identity=r.freeze(source,'static-public-identity');c=r.commit(identity,1.,.5,'Static source-only commitment','engineering-exposed')
    assert r.read(c)['kind']=='commitment'
    assert g.synchronize(ROOT/'mastery_hub/units/sae_prediction/exercises.ipynb',nb,check=True)
    assert not g.synchronize(tmp_path/'missing.ipynb',nb,check=True)

@pytest.mark.parametrize('state',['trusted_false','trusted_true','execution','widgets','outputs','count','learner_source','record'])
def test_stale_generator_protects_raw_learner_metadata_and_bytes(tmp_path,state):
    g=load_generator();old=g.make_notebook();new=copy.deepcopy(old)
    new['cells'][0]['source']+=' New generation.';new['metadata']['sae_prediction'][g.BASELINE]=g.fingerprint(new)
    if state=='trusted_false':old['cells'][1]['metadata']['trusted']=False
    elif state=='trusted_true':old['cells'][1]['metadata']['trusted']=True
    elif state=='execution':old['cells'][1]['metadata']['execution']={}
    elif state=='widgets':old['metadata']['widgets']={}
    elif state=='outputs':old['cells'][1]['outputs']=[{'output_type':'stream','name':'stdout','text':'learner work'}]
    elif state=='count':old['cells'][1]['execution_count']=1
    elif state=='learner_source':old['cells'][1]['source']+='\n# learner edit'
    else:old['metadata']['sae_prediction']['record']='learner commitment'
    path=tmp_path/'work.ipynb';raw=json.dumps(old,indent=3)+'\n';path.write_text(raw)
    assert not g.synchronize(path,new,check=False)
    assert path.read_text()==raw


def test_generator_noop_and_check_never_write(tmp_path):
    g=load_generator();expected=g.make_notebook();path=tmp_path/'blank.ipynb'
    assert g.synchronize(path,expected,check=False)
    before=path.read_bytes();stat=path.stat()
    assert g.synchronize(path,expected,check=True) and g.synchronize(path,expected,check=False)
    assert path.read_bytes()==before and path.stat().st_mtime_ns==stat.st_mtime_ns
    changed=copy.deepcopy(expected);changed['cells'][0]['source']+=' stale'
    assert not g.synchronize(path,changed,check=True)
    assert path.read_bytes()==before


def test_public_guards_reject_id_and_raw_fingerprint_without_target_construction(api):
    r,run,_=api;target=json.loads((r.UNIT/'protocol.json').read_text())
    # Guard rejects before entering numerical implementation. The target is never evaluated.
    for func in (run.scalar_oracle,run._tensors):
        with pytest.raises(AssertionError,match='target ID'):func(target)
        renamed=copy.deepcopy(target);renamed['id']='renamed-would-be-leak'
        with pytest.raises(AssertionError,match='raw-input'):func(renamed)


def test_hand_assertions_are_immediately_visible_not_hidden_helper(api):
    cells=api[2].build_cells()
    for name in api[0].METHOD_NAMES:
        code=next(c['source'] for c in cells if c['id']==name+'-dev-test')
        assert 'assert ' in code and '.tolist()' in code, 'hand expected values must be learner-visible'
        source=next(c['source'] for c in cells if c['id']==name+'-source')
        assert "SOURCE = '''\ndef " in source, 'multiline editable plain definition'


def test_solutions_withheld_until_valid_commit_and_known_dev_control(api,source,committed):
    r,_,lesson=api
    with pytest.raises(ValueError):lesson.solutions(r.read(committed)['parent'])
    disclosed=lesson.solutions(committed)
    assert all('Postcommit solution: '+name in disclosed for name in r.METHOD_NAMES)
    assert 'fixed i,j,k order' in disclosed
    fig=lesson.dev_mismatch_panel(source)
    assert fig.axes[2].get_title()=='Known failure: row MSE 1, 6.5'
    assert fig.axes[2].images[0].get_array().tolist()==[[-1.,-1.],[-3.,2.]]


def test_runtime_unknown_fields_rejected_even_when_same_size(api,committed):
    r,_,_=api;commit=r.read(committed);identity=r.read(commit['parent'])
    runtime=identity['payload']['runtime'];runtime['invented_runtime_field']=runtime.pop('torch')
    commit['parent']=r._seal(identity)
    with pytest.raises(ValueError):r.read(r._seal(commit))


def test_all_new_private_directories_are_0700(api,source,committed,tmp_path,monkeypatch):
    r,run,_=api;root=tmp_path/'private';monkeypatch.setenv('XDG_DATA_HOME',str(root))
    run.run(committed,source,tmp_path/'out',authorized=True)
    for directory in [root,*[p for p in root.rglob('*') if p.is_dir()]]:
        assert directory.stat().st_mode&0o777==0o700


def test_stored_exact_support_rejects_joint_oracle_array_forgery(api,source,committed,tmp_path,monkeypatch):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    d=r.read(run.run(committed,source,tmp_path/'out',authorized=True))
    for group in ('conditions','oracle'):
        arr=d['payload'][group]['primary']['codes']['relu']
        arr['values'][0][0]=0.;arr['sha256']=r.array_digest(arr['values'])
    with pytest.raises(ValueError):r.read(r._seal(d))


def test_inline_immediate_tests_execute_against_own_source_only(api,source):
    r,_,lesson=api;ns=r.namespace(source)
    for name in r.METHOD_NAMES:assert lesson.practice_test(ns,name).endswith('passed')


def test_row_reduction_duplication_all_zero_and_negative_counts(api,source):
    ns=api[0].namespace(source)
    X=tensor([[1.,2.],[-1.,0.]]);Z=tensor([[0.,-2.,0.],[0.,0.,0.]]);R=X*0
    E,L=ns['measure_rows'](X,Z,R)
    assert E.tolist()==[2.5,.5] and L.tolist()==[0,0]
    import torch
    duplicate=ns['measure_rows'](X[:1].repeat(2,1),Z[:1].repeat(2,1),R[:1].repeat(2,1))[0]
    assert duplicate.mean().item()==E[0].item()==2.5
    assert E.mean().item()==1.5


def test_dev_play_cell_exposes_editable_fixture_gates_and_thresholds(api):
    cell=next(c['source'] for c in api[2].build_cells() if c['id']=='dev-play')
    for literal in ('DEV_X','DEV_D','DEV_W','DEV_GATE_BIAS','DEV_THETA','DEV_K'):
        assert literal in cell, 'play inputs and knobs must be editable in notebook'
    assert "ns['gated_encode']" in cell and "ns['jump_encode']" in cell


def test_frozen_registry_cannot_be_replenished_by_changing_environment(api,source,committed,tmp_path,monkeypatch):
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    run.run(committed,source,tmp_path/'first',authorized=True)
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'new-environment'))
    with pytest.raises(ValueError,match='consumed'):run.run(committed,source,tmp_path/'second',authorized=True)


@pytest.fixture(autouse=True)
def reject_public_target(monkeypatch):
    import torch
    from mastery_hub.units.sae_prediction import records as r,runner
    target=json.loads((r.UNIT/'protocol.json').read_text())
    fingerprints={(tuple(r.shape(target[k])),r.array_digest(target[k])) for k in r.ARRAY_NAMES}
    original_tensor=torch.tensor
    def guarded_tensor(value,*args,**kwargs):
        if type(value) in (list,tuple):
            def lists(v):return [lists(x) for x in v] if type(v) in (list,tuple) else v
            raw=lists(value)
            try:fingerprint=(tuple(r.shape(raw)),r.array_digest(raw))
            except (TypeError,ValueError):fingerprint=None
            assert fingerprint not in fingerprints,'forbidden public target raw-input construction'
        return original_tensor(value,*args,**kwargs)
    monkeypatch.setattr(torch,'tensor',guarded_tensor)
    for name in ('_tensors','scalar_oracle','_evaluate'):
        original=getattr(runner,name)
        def guard(case,*args,_original=original,**kwargs):
            assert case.get('id')!=target['id'],'forbidden public target ID dispatch'
            for key in r.ARRAY_NAMES:
                assert (tuple(r.shape(case[key])),r.array_digest(case[key])) not in fingerprints,'forbidden public target raw-input dispatch'
            return _original(case,*args,**kwargs)
        monkeypatch.setattr(runner,name,guard)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_identity_binds_pristine_template_not_evolving_saved_notebook(api,tmp_path,monkeypatch):
    import shutil
    r,_,_=api;copyroot=tmp_path/'copy'
    for name in r.SOURCE_PATHS:
        dest=copyroot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    monkeypatch.setattr(r,'ROOT',copyroot)
    before=r.snapshot_sources();path=copyroot/'mastery_hub/units/sae_prediction/exercises.ipynb'
    saved=json.loads(path.read_text());saved['metadata']['learner_commitment']='saved record'
    saved['cells'][1]['outputs']=[{'name':'stdout','output_type':'stream','text':'learner saved output'}]
    path.write_text(json.dumps(saved))
    assert r.snapshot_sources()==before, 'evolving notebook must not create a receipt self-hash cycle'
