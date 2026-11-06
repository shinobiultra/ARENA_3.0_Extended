"""Exact independent-review regressions; all numerical inputs are distinct DEV."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest
# Pytest importlib mode does not place the sibling test directory on sys.path.
# Load only this exact neighboring DEV support file, without a global module alias
# or a test-directory path insertion. Re-export its reviewed fixtures explicitly.
_dev_support_spec = importlib.util.spec_from_file_location(
    '_arena_sae_prediction_v2_review_dev_fixtures',
    Path(__file__).with_name('test_sae_prediction.py'),
)
_dev_support = importlib.util.module_from_spec(_dev_support_spec)
_dev_support_spec.loader.exec_module(_dev_support)
DEV = _dev_support.DEV
api = _dev_support.api
source = _dev_support.source
committed = _dev_support.committed
reject_public_target = _dev_support.reject_public_target

ROOT=Path(os.environ.get('SAE_CANDIDATE_ROOT',str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(ROOT))

def generator():
    spec=importlib.util.spec_from_file_location('sae_generator_review_v2',ROOT/'scripts/build_sae_prediction_notebook.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_s1_new_freeze_cannot_replenish_session_after_xdg_change(api,source,tmp_path,monkeypatch):
    r,run,_=api
    def commit_again():return r.commit(r._freeze(source,DEV,'same-authorized-dev-session'),0.,.5,'Same DEV exercise session','exposed')
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'first-environment'))
    first=commit_again();run.run(first,source,tmp_path/'first-export',authorized=True)
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'second-environment'))
    second=commit_again()
    with pytest.raises(ValueError,match='consumed'):run.run(second,source,tmp_path/'second-export',authorized=True)
    a=r.read(r.read(first)['parent'])['payload'];b=r.read(r.read(second)['parent'])['payload']
    assert a['attempt_registry']==b['attempt_registry']
    assert len(list(tmp_path.rglob('*.start.json')))==1


@pytest.mark.parametrize('mutation',['bool_ids','bool_raw','float_shape','bool_shape','tuple_shape','bool_label','tuple_labels'])
def test_s2_typed_identity_boundary_rejects_demonstrated_mutations(api,source,mutation):
    r,_,_=api;obj=r.read(r._freeze(source,DEV,'typed-dev'))
    p=obj['payload']
    if mutation=='bool_ids':p['original_feature_ids']=[False,True,2]
    elif mutation=='bool_raw':p['raw_inputs']['X']['values'][0][0]=True
    elif mutation=='float_shape':p['raw_inputs']['X']['shape']=[2.0,2]
    elif mutation=='bool_shape':p['raw_inputs']['b_enc']['shape']=[True,3]
    elif mutation=='tuple_shape':
        # Exercise typed validator directly: JSON cannot retain tuples.
        arr=deepcopy(p['raw_inputs']['X']);arr['shape']=(2,2)
        with pytest.raises(ValueError):r.validate_array(arr,[2,2])
        return
    elif mutation=='bool_label':
        p['recipe']['example_order']=[False,True];p['recipe_sha256']=r.digest(p['recipe'])
    else:
        recipe=deepcopy(DEV);recipe['example_order']=('u','v')
        with pytest.raises(ValueError):r.stored_recipe(recipe)
        return
    with pytest.raises(ValueError):r.read(r._seal(obj))


def test_s2_declared_numeric_int_float_values_remain_supported(api,source):
    r,_,_=api;obj=r.read(r._freeze(source,DEV,'typed-numeric-dev'))
    obj['payload']['raw_inputs']['X']['values'][0][0]=1
    # Values may be int or float; binary float64 bytes remain the invariant.
    assert r.read(r._seal(obj))['payload']['raw_inputs']['X']['values'][0][0]==1


def unicode_failure(api,source,tmp_path,monkeypatch,message='界'*7000):
    r,run,_=api
    bad=source.replace('return X @ W + b',f"raise RuntimeError({message!r})")
    c=r.commit(r._freeze(bad,DEV,'unicode-error-dev'),0.,.5,'Unicode DEV failure','exposed')
    with pytest.raises(run.AttemptFailure) as caught:run.run(c,bad,tmp_path/'failure-export',authorized=True)
    text=caught.value.record_text
    return text,r.read(text)['payload']


def test_s3_unicode_primary_retains_valid_exact_recoverable_message(api,source,tmp_path,monkeypatch):
    r,_,_=api;text,p=unicode_failure(api,source,tmp_path,monkeypatch)
    assert len(p['error'].encode('utf-8'))<=20_000
    assert '[truncated; see error_details]' in p['error']
    detail=p['error_details']['primary']
    assert detail['message']=='界'*7000 and detail['summary_truncated'] is True
    assert detail['message_truncated'] is False and detail['original_utf8_bytes']==21_000
    assert r.read(r.reflect(text,'The exact Unicode primary failure remains available.'))['kind']=='reflection'
    destination=r.export(text,tmp_path/'replayed-unicode.json')
    assert destination.read_text()==text


def test_s3_unicode_secondary_and_storage_failures_do_not_replace_primary(api,source,tmp_path,monkeypatch):
    r,run,_=api;original_current=r.assert_current
    # First call passes. The learner toggles a harmless namespace flag; the
    # post-call validator then raises a Unicode secondary error.
    def checking(*args):
        if getattr(run,'_dev_secondary_marker',False):raise ValueError('次'*7000)
        return original_current(*args)
    monkeypatch.setattr(r,'assert_current',checking)
    monkeypatch.setattr(run,'_dev_secondary_marker',False,raising=False)
    bad=source.replace('return X @ W + b',"__import__('mastery_hub.units.sae_prediction.runner',fromlist=[''])._dev_secondary_marker = True\n    raise RuntimeError('primary-stays-primary')")
    c=r.commit(r._freeze(bad,DEV,'unicode-secondary-dev'),0.,.5,'Unicode secondary and storage DEV','exposed')
    original_export=r.export
    def failing_export(text,path):
        if str(path).endswith('failure.json'):raise OSError('保'*7000)
        return original_export(text,path)
    monkeypatch.setattr(r,'export',failing_export)
    with pytest.raises(run.AttemptFailure) as caught:run.run(c,bad,tmp_path/'out',authorized=True)
    p=r.read(caught.value.record_text)['payload']
    assert 'primary-stays-primary' in p['error']
    assert p['error_details']['primary']['message']=='primary-stays-primary'
    assert p['error_details']['secondary'][0]['message']=='次'*7000
    assert p['error_details']['storage'][0]['message']=='保'*7000
    assert all(len(s.encode())<=20_000 for s in [p['error'],*p['secondary'],p['write_error']])
    assert list(tmp_path.rglob('*.start.json'))
    assert list(tmp_path.rglob('*.failure.json'))


@pytest.mark.parametrize('field,value',[('automatic_experiment_execution',0),('status',False),('nbformat',4.0),('nbformat_minor',5.0)])
def test_s4_type_drift_is_not_pristine(tmp_path,field,value):
    g=generator();expected=g.make_notebook();old=deepcopy(expected)
    if field in ('nbformat','nbformat_minor'):old[field]=value
    else:old['metadata']['sae_prediction'][field]=value
    path=tmp_path/'typed-drift.ipynb';path.write_text(json.dumps(old));before=path.read_bytes()
    assert g.synchronize(path,expected,check=True) is False
    assert path.read_bytes()==before


@pytest.mark.parametrize('raw',['{"metadata":{},"metadata":{}}','{"cells":[],"invalid":NaN}'])
def test_s4_generator_rejects_duplicate_or_nonfinite_json(tmp_path,raw):
    g=generator();path=tmp_path/'bad-json.ipynb';path.write_text(raw)
    assert g.synchronize(path,g.make_notebook(),check=True) is False
    assert path.read_text()==raw


def test_q1_q3_stored_dev_mse_panels_have_shared_nonnegative_scale_readable_diagnostics(api,source,committed,tmp_path):
    r,run,lesson=api;text=run.run(committed,source,tmp_path/'out',authorized=True);p=r.read(text)['payload']
    fig=lesson.plot(text);limits=[ax.get_ylim() for ax in fig.axes[10:14]]
    assert all(v==limits[0] for v in limits) and limits[0][0]==0
    highest=max(v for c in ('primary','zero','mismatch') for m in ('relu','topk','gated','jumprelu') for v in p['conditions'][c]['errors'][m]['values'])
    assert limits[0][1]>=highest
    title=fig._suptitle.get_text()
    assert 'scalar-oracle agreement' in title and 'coherent relabeling preserved' in title
    assert '{' not in title and 'True' not in title
    assert fig.axes[14].get_ylabel()!='squared activation units'


def test_q2_known_failure_panel_has_shared_reconstruction_scale_and_signed_residual_labels(api,source):
    fig=api[2].dev_mismatch_panel(source);a,b,res=fig.axes[:3]
    assert a.images[0].get_clim()==b.images[0].get_clim()
    assert [t.get_text() for t in a.get_xticklabels()]==['x0','x1']
    assert [t.get_text() for t in a.get_yticklabels()]==['u','v']
    assert 'mismatched - reference' in res.get_xlabel()
    lower,upper=res.images[0].get_clim();assert lower==-upper
    assert res.images[0].get_cmap().name=='RdBu_r'
    assert len(fig.axes)==5
    assert any('reconstruction' in ax.get_ylabel().lower() for ax in fig.axes[3:])
    assert any('signed residual' in ax.get_ylabel().lower() for ax in fig.axes[3:])
    assert res.images[0].get_array().tolist()==[[-1.,-1.],[-3.,2.]]
    # Keep all original 69 checks unchanged, including this MSE title assertion.
    assert res.get_title()=='Known failure: row MSE 1, 6.5'


def test_s1_new_process_reuses_authoritative_binding(api,source,tmp_path,monkeypatch):
    import subprocess
    r,run,_=api;monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'first-data'))
    c=r.commit(r._freeze(source,DEV,'cross-process-dev'),0.,.5,'DEV cross-process session','exposed')
    run.run(c,source,tmp_path/'first',authorized=True)
    child="""
import json,sys
from pathlib import Path
payload=json.loads(sys.stdin.read());sys.path.insert(0,payload['root'])
import torch;torch.set_num_threads(1)
from mastery_hub.units.sae_prediction import records as r,runner,lesson
r.SESSION_ANCHOR=Path(payload['anchor'])
# This sentinel cannot construct anything. Reaching it is a failed retry guard.
def denied(*a,**k):raise AssertionError('unexpected second attempt reached numerical dispatch boundary')
runner._evaluate=denied
c=r.commit(r._freeze(lesson.SOLUTION_SOURCE,payload['case'],'cross-process-dev'),4.,.2,'New DEV commitment in another process','exposed')
try:runner.run(c,lesson.SOLUTION_SOURCE,payload['out'],authorized=True)
except ValueError as error:
 assert 'consumed' in str(error),type(error).__name__+': '+str(error)
else:raise AssertionError('second process replenished the session')
print('Existing authoritative session consumed; no numerical dispatch.')
"""
    environment={**os.environ,'XDG_DATA_HOME':str(tmp_path/'new-child-data'),'PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run([sys.executable,'-c',child],input=json.dumps({'root':str(ROOT),'anchor':str(r.SESSION_ANCHOR),'case':DEV,'out':str(tmp_path/'child-export')}),text=True,capture_output=True,env=environment,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'no numerical dispatch' in result.stdout
    assert len(list(tmp_path.rglob('*.start.json')))==1


def test_s3_large_original_error_has_private_exact_recovery_artifact(api,source,tmp_path):
    r,run,_=api
    bad=source.replace('return X @ W + b',"raise RuntimeError('界' * 400000)")
    c=r.commit(r._freeze(bad,DEV,'large-original-dev'),0.,.5,'Bounded large DEV error recovery','exposed')
    with pytest.raises(run.AttemptFailure) as caught:run.run(c,bad,tmp_path/'out',authorized=True)
    p=r.read(caught.value.record_text)['payload'];detail=p['error_details']['primary']
    assert detail['summary_truncated'] and detail['message_truncated']
    artifact=detail['recovery_artifact'];path=Path(artifact['path'])
    assert path.read_bytes()==('界'*400000).encode('utf-8')
    assert artifact['bytes']==1_200_000 and artifact['sha256']==r.sha(path.read_bytes())
    assert path.stat().st_mode&0o777==0o600


def test_s3_failed_failure_write_and_failed_fallback_keep_all_error_details(api,source,tmp_path,monkeypatch):
    r,run,_=api;original=r.private_write
    def fail_only_failure_files(path,text):
        if str(path).endswith('failure.json'):raise OSError('保存不能界'*4000)
        return original(path,text)
    monkeypatch.setattr(r,'private_write',fail_only_failure_files)
    bad=source.replace('return X @ W + b',"raise RuntimeError('primary-never-overwritten')")
    c=r.commit(r._freeze(bad,DEV,'both-writes-fail-dev'),0.,.5,'DEV failure storage fault','exposed')
    with pytest.raises(run.AttemptFailure) as caught:run.run(c,bad,tmp_path/'out',authorized=True)
    p=r.read(caught.value.record_text)['payload']
    assert p['error_details']['primary']['message']=='primary-never-overwritten'
    assert len(p['error_details']['storage'])==2
    assert all(v['message']=='保存不能界'*4000 for v in p['error_details']['storage'])
    assert all(v['summary_truncated'] for v in p['error_details']['storage'])
    assert list(tmp_path.rglob('*.start.json'))
    assert isinstance(caught.value.__cause__,RuntimeError)


def test_s4_duplicate_existing_key_cannot_use_clean_noop_path(tmp_path):
    g=generator();expected=g.make_notebook();raw=json.dumps(expected)
    assert '"nbformat": 4' in raw
    raw=raw.replace('"nbformat": 4','"nbformat": 4, "nbformat": 4',1)
    path=tmp_path/'duplicate-valid-lookalike.ipynb';path.write_text(raw)
    assert g.synchronize(path,expected,check=True) is False
    assert path.read_text()==raw


def test_q1_known_dev_reversed_control_is_retained_and_same_scale(api,source,tmp_path):
    r,run,lesson=api;c=deepcopy(DEV)
    c.update(id='DEV-known-reversed-mismatch',X=[[1.,0.],[0.,1.]],
             D=[[0.,1.],[1.,0.],[1.,1.]],W_enc=[[1.,0.,0.],[0.,1.,0.]],
             b_enc=[0.,0.,-1.],jump_threshold=[.25,.5,.25],decoder_permutation=[1,0,2])
    committed=r.commit(r._freeze(source,c,'reversed-control-dev'),123.,.4,'Faithful reversed DEV control','exposed')
    text=run.run(committed,source,tmp_path/'out',authorized=True);p=r.read(text)['payload']
    for m in ('relu','topk','gated','jumprelu'):
        assert p['conditions']['primary']['errors'][m]['values']==[1.,1.]
        assert p['conditions']['mismatch']['errors'][m]['values']==[0.,0.]
        assert p['conditions']['zero']['errors'][m]['values']==[.5,.5]
    fig=lesson.plot(text)
    assert all(ax.get_ylim()==fig.axes[10].get_ylim() for ax in fig.axes[10:14])
    assert fig.axes[10].get_ylim()[0]==0 and fig.axes[10].get_ylim()[1]>=1


def test_s1_default_anchor_uses_os_account_home_not_environment(api,source,tmp_path,monkeypatch):
    import pwd
    from types import SimpleNamespace
    r,_,_=api
    monkeypatch.setattr(r,'SESSION_ANCHOR',None,raising=False)
    monkeypatch.setattr(pwd,'getpwuid',lambda uid:SimpleNamespace(pw_dir=str(tmp_path/'os-account-home')))
    monkeypatch.setenv('HOME',str(tmp_path/'first-environment-home'))
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'first-environment-data'))
    first=r.read(r._freeze(source,DEV,'os-bound-dev'))['payload']
    monkeypatch.setenv('HOME',str(tmp_path/'new-environment-home'))
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'new-environment-data'))
    second=r.read(r._freeze(source,DEV,'os-bound-dev'))['payload']
    assert first['attempt_registry']==second['attempt_registry']
    assert 'os-account-home' in second['session_binding']['anchor_path']
