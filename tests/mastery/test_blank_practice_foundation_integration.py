"""Inert integrated-source contracts; generated plotting code is not executed."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys

import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'infrastructure/extension/syllabus.yaml').is_file())
TREE=Path(__file__).resolve().parents[2]

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.fixture
def lane():
 return load(TREE/'infrastructure/extension/blank_practice.py','live_integrated'),load(TREE/'mastery_hub/read.py','live_reader')

@pytest.mark.parametrize('section',['0.6','1.6'])
def test_exact_two_source_pinned_appends_and_typed_inverse(lane,section):
 h,r=lane;row=next(x for x in r.inventory(ROOT) if x['number']==section);raw=row['exercises'].read_bytes();source=json.loads(raw)
 after,name=h.download(ROOT,section,r.inventory);n=json.loads(after)
 assert len(n['cells'])==len(source['cells'])+7
 provider=load(TREE/'mastery_hub/practice_addons/foundation_live_signatures/lesson.py','reviewed_provider')
 assert h._typed(n['cells'][-6:])==h._typed(provider.additions(section))
 info=n['metadata']['arena_blank_practice']['foundation_live']
 assert info['cells_sha256']==hashlib.sha256(h._typed(n['cells'][-6:]).encode()).hexdigest()
 assert info['provider_sha256']=='5d54ed48161619e40d84e4a58ee961abce8cfced6b3c2aaeff695d14b5864b83'
 for before,cell in zip(source['cells'],n['cells'][1:-6],strict=True):
  expected=dict(before)
  if before['cell_type']=='code':expected.update(outputs=[],execution_count=None)
  assert h._typed(cell)==h._typed(expected)
 assert h._typed(h.restore_download(after,raw,row['exercises'].relative_to(ROOT).as_posix(),h.config_identity(ROOT)))==h._typed(source)
 assert all(c['outputs']==[] and c['execution_count'] is None for c in n['cells'] if c['cell_type']=='code')


def test_revised_study_copy_precedes_preserved_original_downloads(lane):
 h,r=lane
 class UI:
  sidebar=None
  def __init__(self):self.sidebar=self;self.downloads=[];self.text=[]
  def set_page_config(self,**kwargs):pass
  def title(self,text):pass
  def info(self,text):self.text.append(text)
  def markdown(self,text,**kwargs):self.text.append(text)
  def selectbox(self,label,options,**kwargs):return 'chapter5_modern_architectures' if label=='Chapter' else (next(x for x in options if x['number']=='5.1') if label=='Lesson' else 0)
  def download_button(self,label,data,file_name,**kwargs):self.downloads.append((label,data,file_name))
 ui=UI();r.render(ui,ROOT)
 labels=[x[0] for x in ui.downloads]
 assert labels[:2]==['Download revised Study Practice','Download revised worked reference']
 assert [label for label in labels if label in {'Download blank Practice exercises','Download exercises','Download worked reference'}]==['Download blank Practice exercises','Download exercises','Download worked reference']
 assert any('Start with blank Practice' in x and 'saved outputs' in x for x in ui.text)

@pytest.mark.parametrize('mutation',[
 lambda n:n['cells'][-6].__setitem__('source','changed live source'),
 lambda n:n['cells'][-5].__setitem__('id','changed-live-id'),
 lambda n:n['cells'][-5].__setitem__('execution_count',1),
 lambda n:n['cells'][-5].__setitem__('outputs',[{'output_type':'stream','name':'stdout','text':'invented'}]),
 lambda n:n['metadata']['arena_blank_practice']['foundation_live'].__setitem__('cells_sha256','f'*64),
 lambda n:n['cells'].pop(),
 lambda n:n['cells'].append(n['cells'][-1]),
])
def test_live_inverse_rejects_every_changed_appended_object(lane,mutation):
 h,r=lane;row=next(x for x in r.inventory(ROOT) if x['number']=='0.6');raw=row['exercises'].read_bytes();after,_=h.download(ROOT,'0.6',r.inventory);n=json.loads(after);mutation(n)
 with pytest.raises(ValueError):h.restore_download(json.dumps(n).encode(),raw,row['exercises'].relative_to(ROOT).as_posix(),h.config_identity(ROOT))


def test_source_drift_rejected_before_live_append(lane):
 h,r=lane;row=next(x for x in r.inventory(ROOT) if x['number']=='1.6');raw=row['exercises'].read_bytes()+b' '
 with pytest.raises(ValueError,match='source'):h.project_download(raw,row['exercises'].relative_to(ROOT).as_posix(),h.config_identity(ROOT))

@pytest.mark.parametrize('fault',['missing','changed'])
def test_missing_or_changed_provider_fails_without_import_cache(lane,tmp_path,monkeypatch,fault):
 h,r=lane;row=next(x for x in r.inventory(ROOT) if x['number']=='0.6');raw=row['exercises'].read_bytes()
 path=tmp_path/'mastery_hub/practice_addons/foundation_live_signatures/lesson.py';path.parent.mkdir(parents=True)
 if fault=='changed':path.write_text('raise AssertionError("unreviewed provider must never execute")')
 monkeypatch.setattr(h,'_PROVIDER_ROOT',tmp_path)
 with pytest.raises(ValueError):h.project_download(raw,row['exercises'].relative_to(ROOT).as_posix(),h.config_identity(ROOT))


def test_cached_provider_module_cannot_supply_appended_cells(lane,monkeypatch):
 h,r=lane;row=next(x for x in r.inventory(ROOT) if x['number']=='0.6')
 fake=SimpleNamespace(additions=lambda section: (_ for _ in ()).throw(AssertionError('stale imported function used')))
 monkeypatch.setitem(sys.modules,'mastery_hub.practice_addons.foundation_live_signatures.lesson',fake)
 after,_=h.download(ROOT,'0.6',r.inventory)
 assert json.loads(after)['cells'][-1]['id']=='foundation-live-0-6-limits'

@pytest.mark.parametrize('change',[
 lambda cells:cells[0].__setitem__('source','mutated generated object'),
 lambda cells:cells[1]['metadata'].__setitem__('unexpected',True),
 lambda cells:cells[1].__setitem__('id','other-generated-id'),
 lambda cells:cells[1].__setitem__('execution_count',1),
 lambda cells:cells.reverse(),
 lambda cells:cells.pop(),
])
def test_actual_generated_cell_json_digest_checked_after_provider_call(lane,monkeypatch,change):
 h,r=lane;relative=next(iter(h._FOUNDATIONS));real_exec=exec
 def altered_provider(code,namespace):
  real_exec(code,namespace);original=namespace['additions']
  def altered(section):
   cells=original(section);change(cells);return cells
  namespace['additions']=altered
 monkeypatch.setattr(h,'exec',altered_provider,raising=False)
 with pytest.raises(ValueError,match='cell JSON'):h._foundation_cells(relative)

@pytest.mark.parametrize('change',[
 lambda n:n['cells'][0]['source'].append('changed route note'),
 lambda n:n['metadata']['arena_blank_practice'].__setitem__('download_feature_version',3),
 lambda n:n['metadata']['arena_blank_practice'].__setitem__('study_cautions_sha256','0'*64),
 lambda n:n['metadata']['arena_blank_practice'].__setitem__('unexpected_allowed_field',True),
 lambda n:n['cells'][1].__setitem__('source','changed original Markdown'),
 lambda n:n['cells'][1]['metadata'].__setitem__('unexpected',1),
])
def test_v2_inverse_rejects_note_and_other_protected_changes(lane,change):
 h,r=lane;row=next(x for x in r.inventory(ROOT) if x['number']=='0.6');raw=row['exercises'].read_bytes();after,_=h.download(ROOT,'0.6',r.inventory);n=json.loads(after);change(n)
 with pytest.raises(ValueError):h.restore_download(json.dumps(n).encode(),raw,row['exercises'].relative_to(ROOT).as_posix(),h.config_identity(ROOT))

@pytest.mark.parametrize('section',['5.1','7.1','12.3'])
def test_unrelated_sections_get_no_live_cells_and_cautions_link_resolves_lexically(lane,section):
 import posixpath
 h,r=lane;row=next(x for x in r.inventory(ROOT) if x['number']==section);raw=row['exercises'].read_bytes();after,_=h.download(ROOT,section,r.inventory);n=json.loads(after)
 assert len(n['cells'])==len(json.loads(raw)['cells'])+1
 assert 'foundation_live' not in n['metadata']['arena_blank_practice']
 assert not any(c.get('id','').startswith('foundation-live-') for c in n['cells'])
 relative=row['exercises'].relative_to(ROOT).as_posix()
 target=posixpath.relpath('docs/mastery/study_cautions_20261023.md',posixpath.dirname(relative))
 assert f']({target})' in ''.join(n['cells'][0]['source'])
 assert posixpath.normpath(posixpath.join(posixpath.dirname(relative),target))=='docs/mastery/study_cautions_20261023.md'
