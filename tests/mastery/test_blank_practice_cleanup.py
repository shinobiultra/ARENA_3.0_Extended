"""Actual source-reader cleanup adversaries; no kernels, imports or user files."""
from pathlib import Path
import errno
import importlib.util
import os

import pytest

HELPER=Path(os.environ.get('ARENA_BLANK_HELPER_SOURCE',Path(__file__).resolve().parents[2]/'infrastructure/extension/blank_practice.py'))

@pytest.mark.parametrize('read_kind,close_indexes,base_close',[
 ('none',(1,),False),
 ('oserror',(1,),False),
 ('none',(1,2,4),False),
 ('oserror',(1,2,4),False),
 ('valueerror',(1,2),False),
 ('none',(1,),True),
 ('ambient',(1,),False),
])
def test_every_close_attempted_once_and_all_failures_retained(tmp_path,monkeypatch,read_kind,close_indexes,base_close):
 spec=importlib.util.spec_from_file_location('cleanup_reader',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 target=tmp_path/'owned.bin';target.write_bytes(b'owned inert source')
 inode=target.stat().st_ino;opened=[];attempts=[];injected=[]
 real_open=os.open;real_close=os.close;real_read=os.read
 primary=OSError(errno.EIO,'primary read sentinel') if read_kind=='oserror' else ValueError('primary validation sentinel')
 def open_fd(*args,**kwargs):
  fd=real_open(*args,**kwargs);opened.append(fd);return fd
 def read_fd(fd,size):
  if read_kind in ('oserror','valueerror') and os.fstat(fd).st_ino==inode:raise primary
  return real_read(fd,size)
 def close_fd(fd):
  attempts.append(fd);real_close(fd)
  if len(attempts) in close_indexes:
   error=KeyboardInterrupt('close BaseException sentinel') if base_close else OSError(errno.EINTR,'released close sentinel '+str(len(attempts)))
   injected.append(error);raise error
 caught=None
 with monkeypatch.context() as m:
  m.setattr(os,'open',open_fd);m.setattr(os,'read',read_fd);m.setattr(os,'close',close_fd)
  try:
   if read_kind=='ambient':
    try:raise RuntimeError('ambient caller exception must not become primary')
    except RuntimeError:h._read_regular(tmp_path,'owned.bin')
   else:h._read_regular(tmp_path,'owned.bin')
  except BaseException as exc:caught=exc
 live=[]
 for fd in opened:
  try:os.fstat(fd)
  except OSError as exc:assert exc.errno==errno.EBADF
  else:live.append(fd)
 # Preserve the observed leak verdict, then reap only exact private probe FDs.
 for fd in live:real_close(fd)
 assert attempts==list(reversed(opened)), 'Cleanup stopped early or retried a possibly released descriptor'
 assert not live, 'Owned descriptors leaked before test-only cleanup'
 assert isinstance(caught,BaseExceptionGroup), 'Close failures must be reported after all cleanup'
 errors=list(caught.exceptions)
 if read_kind=='oserror':
  assert isinstance(errors[0],ValueError) and errors[0].__cause__ is primary
  assert primary.__traceback__ is not None
  errors=errors[1:]
 elif read_kind=='valueerror':
  assert errors[0] is primary and primary.__traceback__ is not None
  errors=errors[1:]
 assert errors==injected and len(errors)==len(close_indexes)
 assert isinstance(caught,ExceptionGroup) is not base_close
