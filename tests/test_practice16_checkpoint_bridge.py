"""Exact source-pinned 1.6 loader branch; no import hooks or notebook execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

_path=Path(__file__).with_name('test_practice16_execution_checkpoint.py')
_spec=importlib.util.spec_from_file_location('_practice16_bridge_fixtures',_path)
_f=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_f)
checkpoint=_f.checkpoint
actual_replica=_f.actual_replica


def bridge(d):
    p=_f.HERE/'scripts/practice06_execution_checkpoint.py'
    spec=importlib.util.spec_from_file_location('_practice06_exact_bridge',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    target=d.root/'scripts/practice16_execution_checkpoint.py';target.parent.mkdir(exist_ok=True)
    target.write_bytes((_f.HERE/'scripts/practice16_execution_checkpoint.py').read_bytes())
    return module,target


def test_exact_16_branch_returns_historical_inverse_and_preserves_physical_actual(actual_replica):
    d=actual_replica;c,adapter=bridge(d);physical=d.physical.read_bytes()
    expected=d.c._guard(d.root)['before']
    assert c._same(c.load_practice06_checkpoint_json(d.root,d.physical),expected)
    assert d.physical.read_bytes()==physical


@pytest.mark.parametrize('fault',['missing_adapter','altered_adapter','stale_physical','missing_physical'])
def test_exact_16_branch_rejects_missing_or_altered_identity_and_stale_state(actual_replica,fault):
    d=actual_replica;c,adapter=bridge(d)
    if fault=='missing_adapter':adapter.unlink()
    if fault=='altered_adapter':adapter.write_bytes(adapter.read_bytes()+b'\nraise AssertionError("unverified adapter executed")\n')
    if fault=='stale_physical':d.physical.write_bytes(d.c._guard(d.root)['before_bytes'])
    if fault=='missing_physical':d.physical.unlink()
    with pytest.raises((ValueError,OSError)):c.load_practice06_checkpoint_json(d.root,d.physical)


def test_neighbor_path_keeps_plain_json_behavior_without_loading_adapter(actual_replica):
    d=actual_replica;c,adapter=bridge(d);adapter.unlink()
    neighbor=d.physical.with_name('DEV_neighbor.json');neighbor.write_text('{"DEV_plain": true}')
    assert c.load_practice06_checkpoint_json(d.root,neighbor)=={'DEV_plain':True}


def test_source_verified_bridge_ignores_foreign_cached_adapter(actual_replica,monkeypatch):
    import sys
    d=actual_replica;c,adapter=bridge(d)
    def forbidden(*args,**kwargs):raise AssertionError('foreign cached adapter used')
    monkeypatch.setitem(sys.modules,'scripts.practice16_execution_checkpoint',SimpleNamespace(recover_practice16_checkpoint_bytes=forbidden))
    assert c._same(c.load_practice06_checkpoint_json(d.root,d.physical),d.c._guard(d.root)['before'])
