"""Strict saved schemas and whole-object preservation after declared format repair."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

from nbformat.validator import get_validator
import pytest

from infrastructure.extension.source_preservation import apply_compatibility

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/mastery/evidence/notebook_schema_compatibility_20261020'
CONTRACT_SHA = 'bb2ba063bcc8c960790e44b90cd81e6f03b6f23907dae2d20107195d79dd0cf6'
BEFORE_SHA = '70210b9eb7680d5c517bc7a5318db147221c65be3e6d80e9ef28631c631abd10'
CONTRACTS = json.loads((BASE / 'reviewed_seven_contracts.json').read_bytes())
BOOKS = [(section, kind, pin) for section, recipe in CONTRACTS['sections'].items()
         for kind, pin in recipe['books'].items()]


def frozen_original(pin):
    archive = BASE / 'before.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == BEFORE_SHA
    assert hashlib.sha256((BASE / 'reviewed_seven_contracts.json').read_bytes()).hexdigest() == CONTRACT_SHA
    with tarfile.open(archive) as handle:
        return json.loads(handle.extractfile(pin['member']).read())


@pytest.mark.parametrize('section,kind,pin', BOOKS, ids=[f'{s}-{k}' for s, k, _ in BOOKS])
def test_saved_book_has_strict_schema_and_exact_declared_inverse(section, kind, pin):
    original = frozen_original(pin)
    current = json.loads((ROOT / pin['member']).read_bytes())
    assert list(get_validator(4, 5).iter_errors(current)) == []
    assert current == apply_compatibility(original, pin)
    inverse = deepcopy(current)
    for addition in pin['id_additions']:
        cell = inverse['cells'][addition['cell']]
        assert cell.pop('id') == addition['id']
    for removal in pin['compatibility_removals']:
        cell = inverse['cells'][removal['cell']]
        assert removal['field'] not in cell
        cell[removal['field']] = deepcopy(removal['value'])
    assert inverse == original
    assert [c for c in current['cells'] if c['cell_type'] == 'code'] == [
        c for c in original['cells'] if c['cell_type'] == 'code']


def test_nonempty_evidence_and_existing_identifiers_are_never_removed():
    for _, _, pin in BOOKS:
        original = frozen_original(pin)
        for addition in pin['id_additions']:
            changed = deepcopy(original)
            changed['cells'][addition['cell']]['id'] = 'already-present'
            with pytest.raises(ValueError):
                apply_compatibility(changed, pin)
        for removal in pin['compatibility_removals']:
            changed = deepcopy(original)
            changed['cells'][removal['cell']][removal['field']] = (
                1 if removal['field'] == 'execution_count' else [{'text': 'preserve evidence'}])
            with pytest.raises(ValueError):
                apply_compatibility(changed, pin)
