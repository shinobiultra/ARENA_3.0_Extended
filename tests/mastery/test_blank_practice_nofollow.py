"""V1-red/V1.1-green descriptor-bound input tests. Only private inert fixtures."""
from pathlib import Path
import copy
import importlib.util
import json
import os
import shutil

import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'infrastructure/extension/syllabus.yaml').is_file())
TREE=Path(__file__).resolve().parents[2]
HELPER=Path(os.environ.get('ARENA_BLANK_HELPER_SOURCE',TREE/'infrastructure/extension/blank_practice.py'))
READER=Path(os.environ.get('ARENA_READER_SOURCE',TREE/'mastery_hub/read.py'))

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.fixture
def lane(tmp_path):
 h=load(HELPER,'nofollow_helper');r=load(READER,'nofollow_reader');root=tmp_path/'repo';root.mkdir()
 for rel in h._CONFIG:
  f=root/rel;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,f)
 row=next(x for x in r.inventory(ROOT) if x['number']=='5.1')
 master=root/'infrastructure/chapters'/row['chapter']/'master_5_1.py';master.parent.mkdir(parents=True);master.write_text('# existence only')
 for role in ('page','reference'):
  f=root/row[role].relative_to(ROOT);f.parent.mkdir(parents=True,exist_ok=True);f.write_text('{}')
 source=root/row['exercises'].relative_to(ROOT)
 notebook={'nbformat':4,'nbformat_minor':2,'metadata':{},'cells':[{'cell_type':'code','metadata':{},'source':'# never executed','outputs':[],'execution_count':None}]}
 source.write_text(json.dumps(notebook));outside=tmp_path/'outside.ipynb';notebook['metadata']['outside_sentinel']=True;outside.write_text(json.dumps(notebook))
 return h,r,root,source,outside

@pytest.mark.parametrize('kind',['source','config'])
def test_late_symlink_after_validation_never_reads_outside(lane,monkeypatch,kind):
 h,r,root,source,outside=lane
 target=source if kind=='source' else root/h._CONFIG[0]
 if kind=='config':outside.write_text('outside synthetic config bytes')
 swaps=[];original_read=Path.read_bytes;original_open=os.open
 def swap():
  if not swaps:target.unlink();target.symlink_to(outside);swaps.append(True)
 def path_read(path,*args,**kwargs):
  if path==target:swap()
  return original_read(path,*args,**kwargs)
 def fd_open(path,flags,*args,**kwargs):
  if path==target.name and kwargs.get('dir_fd') is not None:swap()
  return original_open(path,flags,*args,**kwargs)
 monkeypatch.setattr(Path,'read_bytes',path_read);monkeypatch.setattr(os,'open',fd_open)
 with pytest.raises(ValueError):
  if kind=='source':h.download(root,'5.1',r.inventory)
  else:h.config_identity(root)
 assert swaps, 'Probe did not reach the actual input-open boundary'


def test_inventory_parses_only_captured_configuration_bytes(lane,monkeypatch):
 h,r,root,source,outside=lane;original=Path.read_text
 def no_config_reopen(path,*args,**kwargs):
  assert path not in [root/rel for rel in h._CONFIG], 'Fresh unchecked config pathname read'
  return original(path,*args,**kwargs)
 monkeypatch.setattr(Path,'read_text',no_config_reopen)
 raw,_=h.download(root,'5.1',r.inventory)
 assert not json.loads(raw)['metadata'].get('outside_sentinel')


def test_in_read_file_mutation_is_rejected(lane,monkeypatch):
 h,r,root,source,outside=lane;original=os.read;inode=source.stat().st_ino;changed=[]
 def read(fd,size):
  chunk=original(fd,size)
  if os.fstat(fd).st_ino==inode and chunk and not changed:
   with source.open('ab') as f:f.write(b' ')
   changed.append(True)
  return chunk
 monkeypatch.setattr(os,'read',read)
 with pytest.raises(ValueError,match='changed'):h.download(root,'5.1',r.inventory)
 assert changed


def test_directory_binding_changed_during_read_is_rejected(lane,monkeypatch):
 h,r,root,source,outside=lane;original=os.read;inode=source.stat().st_ino;changed=[]
 def read(fd,size):
  chunk=original(fd,size)
  if os.fstat(fd).st_ino==inode and chunk and not changed:
   parent=source.parent;renamed=parent.with_name(parent.name+'_old');parent.rename(renamed);parent.symlink_to(outside.parent,target_is_directory=True);changed.append(True)
  return chunk
 monkeypatch.setattr(os,'read',read)
 with pytest.raises(ValueError,match='changed|symlink'):h.download(root,'5.1',r.inventory)
 assert changed

@pytest.mark.parametrize('failure',[False,True])
def test_all_owned_descriptors_close_on_success_and_failure(lane,monkeypatch,failure):
 h,r,root,source,outside=lane;original_open=os.open;original_close=os.close;opened=[];closed=[]
 def opened_fd(*args,**kwargs):
  fd=original_open(*args,**kwargs);opened.append(fd);return fd
 def close(fd):closed.append(fd);return original_close(fd)
 monkeypatch.setattr(os,'open',opened_fd);monkeypatch.setattr(os,'close',close)
 if failure:
  source.unlink();source.symlink_to(outside)
  with pytest.raises(ValueError):h.download(root,'5.1',r.inventory)
 else:h.download(root,'5.1',r.inventory)
 assert opened and sorted(opened)==sorted(closed)
