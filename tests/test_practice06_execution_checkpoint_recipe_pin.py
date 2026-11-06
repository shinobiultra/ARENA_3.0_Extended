"""Regression for V2's alternate-recipe bypass; saved actual files only, no execution."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# Reuse the byte-identical 79-case fixture source without editing its tests.
_fixture_path = Path(__file__).with_name('test_practice06_execution_checkpoint.py')
_spec = importlib.util.spec_from_file_location('_practice06_v3_unchanged_fixtures', _fixture_path)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
checkpoint = _fixtures.checkpoint
actual_replica = _fixtures.actual_replica


def copied_master(d, *, altered=False):
    canonical = d.root / 'infrastructure/chapters/chapter0_fundamentals/master_0_6.py'
    master = d.root / 'DEV_copied_source/master_0_6.py'
    master.parent.mkdir()
    master.write_bytes(canonical.read_bytes() + (b'\n# DEV changed copied master, never executed\n' if altered else b''))
    return master


def test_changed_copied_master_plus_resigned_alternate_recipe_is_rejected(actual_replica):
    import json
    d = actual_replica
    master = copied_master(d, altered=True)
    recipe_path = d.root / 'infrastructure/extension/source_preservation_contracts.json'
    recipe_before = recipe_path.read_bytes()
    contracts = json.loads(recipe_before)
    contracts['sections']['0.6']['master_sha256'] = hashlib.sha256(master.read_bytes()).hexdigest()
    override = master.parent / 'DEV_resigned_alternate_recipe.json'
    override.write_text(json.dumps(contracts))
    arguments = _fixtures.actual_restore_arguments(d, master)
    physical_before = d.physical.read_bytes()
    with pytest.raises(ValueError, match='contracts_path'):
        d.c.restore_practice06_notebooks(**arguments, contracts_path=override)
    assert recipe_path.read_bytes() == recipe_before
    assert d.physical.read_bytes() == physical_before


@pytest.mark.parametrize('form', ['same_bytes_alternate_path', 'canonical_explicit_path'])
def test_even_unchanged_recipe_path_overrides_are_rejected_for_registered_06(actual_replica, form):
    d = actual_replica
    master = copied_master(d)
    canonical = d.root / 'infrastructure/extension/source_preservation_contracts.json'
    alternate = master.parent / 'DEV_exact_recipe_copy.json'
    alternate.write_bytes(canonical.read_bytes())
    path = alternate if form == 'same_bytes_alternate_path' else canonical
    with pytest.raises(ValueError, match='contracts_path'):
        d.c.restore_practice06_notebooks(**_fixtures.actual_restore_arguments(d, master), contracts_path=path)


@pytest.mark.parametrize('explicit_default', [False, True])
def test_unmodified_copied_master_uses_exact_canonical_original_recipe(actual_replica, explicit_default):
    d = actual_replica
    master = copied_master(d)
    arguments = _fixtures.actual_restore_arguments(d, master)
    if explicit_default:
        arguments['contracts_path'] = None
    physical_before = d.physical.read_bytes()
    restored = d.c.restore_practice06_notebooks(**arguments)
    assert restored['solutions'].encode() == physical_before == d.physical.read_bytes()


def test_other_section_keeps_original_contracts_path_delegation(checkpoint, monkeypatch):
    import sys
    override = object()
    arguments = {'identity': {'section': 'DEV_other_section'}, 'contracts_path': override}
    seen = []
    expected = object()
    def original(**kwargs):
        seen.append(kwargs)
        return expected
    monkeypatch.setitem(sys.modules, 'infrastructure.extension.source_preservation', SimpleNamespace(restore_notebooks=original))
    assert checkpoint.restore_practice06_notebooks(**arguments) is expected
    assert seen == [arguments] and seen[0]['contracts_path'] is override
