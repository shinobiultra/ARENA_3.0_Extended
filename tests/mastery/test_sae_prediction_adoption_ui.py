"""The SAE sidebar serves reviewed blank bytes without importing or running a unit."""
import ast
import builtins
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.abc
from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[2]
NOTEBOOK_SHA256='84ada9a0705b2080f7e22a444bb741999cc63c677098c76bf97e65a90e440807'
LABEL='Download 6.1 SAE forward-rule prediction companion'
PREFIX='mastery_hub.units.sae_prediction'


def test_sidebar_serves_exact_blank_notebook_preserves_progress_and_never_imports_or_dispatches_sae(tmp_path,monkeypatch):
    import streamlit as st
    from arena_ext.mastery import quadratic,plots,records,challenges
    st.config.set_option('browser.gatherUsageStats',False)
    private=tmp_path/'private-progress';monkeypatch.setenv('ARENA_MASTERY_DATA_DIR',str(private))
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private-data'))
    config=asdict(quadratic.QuadraticConfig());config['x0']=list(config['x0']);config['simulator_version']=quadratic.SIMULATOR_VERSION
    key=records.fingerprint(config)
    exposures=[{'challenge_id':'deterministic-stability-v1','config_fingerprint':key},
               {'challenge_id':'preserved-prior-reading','config_fingerprint':'a'*64}]
    current=challenges.new_attempt('deterministic-stability-v1',key,exposed_cases=exposures)
    current=challenges.commit_prediction(current,True,.6,'Preserve this pending fixture; the UI must not reveal it.')
    session=records.empty_session()
    session.update(config=config,current_attempt=current,exposed_cases=exposures,mode='Challenge',
                   next_action='Preserve this next action without revealing a result.')
    records.save_session(session,private)
    before=deepcopy(records.load_session(private));before_bytes=(private/'session.json').read_bytes()
    blocked=[]
    def denied(*args,**kwargs):
        blocked.append('scientific dispatch')
        raise AssertionError('UI must not run SAE or optimizer scientific computations')
    # Challenge mode must preserve the pending state without computing a trace,
    # oracle, baseline, plot, control, or new scientific observation.
    monkeypatch.setattr(quadratic,'simulate',denied)
    monkeypatch.setattr(quadratic,'spectral_radius',denied)
    monkeypatch.setattr(plots,'build_figures',denied)
    for name,module in tuple(sys.modules.items()):
        if name==PREFIX or name.startswith(PREFIX+'.'):
            for symbol in ('run','_evaluate','_tensors','scalar_oracle','freeze','_freeze','namespace','plot','bind_session'):
                if symbol in vars(module):monkeypatch.setattr(module,symbol,denied)
    original_import=builtins.__import__
    def guarded_import(name,globals=None,locals=None,fromlist=(),level=0):
        requested=[name,*[name+'.'+item for item in (fromlist or ()) if type(item) is str]]
        if any(n==PREFIX or n.startswith(PREFIX+'.') for n in requested):
            blocked.append(name);raise AssertionError('sidebar attempted a SAE unit import')
        return original_import(name,globals,locals,fromlist,level)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    class RejectSAE(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname==PREFIX or fullname.startswith(PREFIX+'.'):
                blocked.append(fullname);raise AssertionError('sidebar attempted a SAE module load')
            return None
    monkeypatch.setattr(sys,'meta_path',[RejectSAE(),*sys.meta_path])
    downloaded=[];real_download=st.download_button
    def capture_download(label,data,*args,**kwargs):
        downloaded.append((label,data,kwargs.get('file_name'),kwargs.get('mime')))
        return real_download(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'download_button',capture_download)
    at=AppTest.from_file(str(ROOT/'mastery_hub/app.py'),default_timeout=20).run()
    assert not at.exception
    visible=[button.proto.label for button in at.get('download_button')]
    assert LABEL in visible, 'reviewed SAE companion download must be visible'
    item=next(item for item in downloaded if item[0]==LABEL)
    expected=(ROOT/'mastery_hub/units/sae_prediction/exercises.ipynb').read_bytes()
    assert type(item[1]) is bytes and item[1]==expected
    assert hashlib.sha256(item[1]).hexdigest()==NOTEBOOK_SHA256
    assert item[2:] == ('sae_prediction_exercises.ipynb','application/x-ipynb+json')
    caption=next(c.value for c in at.caption if c.value.startswith('6.1 companion:'))
    for phrase in ('four fixed forward rules','visible methods','DEV','target remains unexecuted','No automatic run','training','model/data download'):
        assert phrase in caption
    assert at.session_state['progress']==before
    assert records.load_session(private)==before
    assert (private/'session.json').read_bytes()==before_bytes
    # An ordinary rerun of the new download entry changes neither progress nor exposure.
    at.run();assert not at.exception and at.session_state['progress']==before
    assert records.load_session(private)==before and (private/'session.json').read_bytes()==before_bytes
    assert blocked==[]
    assert not list(tmp_path.rglob('*.binding.json')) and not list(tmp_path.rglob('*.start.json'))


def test_sidebar_sae_block_contains_only_file_download_and_caption():
    tree=ast.parse((ROOT/'mastery_hub/app.py').read_text())
    imports=[node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))]
    for node in imports:
        names=[a.name for a in node.names]
        assert not any('sae_prediction' in name for name in names)
        assert 'sae_prediction' not in (getattr(node,'module',None) or '')
    branch=next((node for node in ast.walk(tree) if isinstance(node,ast.If) and
                 isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Attribute) and
                 isinstance(node.test.func.value,ast.Name) and node.test.func.value.id=='sae_path'),None)
    assert branch is not None, 'explicit SAE notebook file-existence branch missing'
    calls={node.func.attr for node in ast.walk(branch) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    assert calls=={'exists','download_button','read_bytes','caption'}
    assert not any(isinstance(node,(ast.Import,ast.ImportFrom)) for node in ast.walk(branch))
