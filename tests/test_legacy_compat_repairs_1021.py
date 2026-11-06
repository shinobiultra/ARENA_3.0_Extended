"""Source/JSON custody regressions only; no method, model or target execution."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(os.environ.get('ARENA_MIGRATION_ROOT', Path(__file__).resolve().parents[1]))
LEGACY = Path(os.environ.get('ARENA_LEGACY_ROOT', ROOT))
E = ROOT / 'docs/mastery/evidence/rome_9_6_migration_20261021/v2'
CURRENT = ROOT / 'docs/mastery/evidence/rome_9_6_migration_20261021/post_header_mapping_20261021'
MAPPING_SHA256 = '7ed11953b3d9f3c9f99c9805b342831435f6703ce1874ce5cd956ecfbdb95a8b'
PROOF_SHA256 = '43c2df2f2b0fb73ec9547e5de65f736d95a12c553fb7cdf257b73c2b318bb5fc'
CHAPTER = 'chapter9_alignment_interpretability'


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def books(kind):
    old = read(LEGACY / CHAPTER / 'exercises/part5_rome_model_editing' /
               f'9.5_ROME_Model_Editing_{kind}.ipynb')
    current = read(ROOT / CHAPTER / 'exercises/part6_rome_memit' /
                   f'9.6_Model_Editing_with_ROME_and_MEMIT_{kind}.ipynb')
    return old, current


def guard_static_mapping(kind, current):
    """Validate fixed expectations; never infer or repair expected indices from input."""
    mapping_path = CURRENT / 'original_cell_mapping.json'
    if hashlib.sha256(mapping_path.read_bytes()).hexdigest() != MAPPING_SHA256:
        raise ValueError('static mapping bytes changed')
    rows = read(mapping_path)[kind]
    proof_path = CURRENT / 'mapping_proof.json'
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != PROOF_SHA256:
        raise ValueError('static mapping proof bytes changed')
    proof = read(proof_path)['books'][kind]
    if [cell['id'] for cell in current['cells']] != proof['canonical_cell_id_order']:
        raise ValueError('canonical cell order changed')
    old, _ = books(kind)
    declared = {change['id']: change for change in read(E / 'declared_source_changes.json')[kind]}
    restored = deepcopy(current)
    selected = []
    for row in rows:
        cell = deepcopy(current['cells'][row['canonical_index']])
        if cell['id'] != row['id']:
            raise ValueError('canonical index changed')
        if cell['id'] in declared:
            change = declared[cell['id']]
            if ''.join(cell['source']) != change['after_source']:
                raise ValueError('canonical source changed')
            cell['source'] = deepcopy(old['cells'][row['original_index']]['source'])
        if digest(cell) != row['original_cell_sha256']:
            raise ValueError('original whole cell changed')
        selected.append(cell)
    restored['cells'] = selected
    if restored != old:
        raise ValueError('original notebook object changed')
    return restored


def test_static_mapping_changes_only_indices_by_approved_single_header():
    old = read(E / 'original_cell_mapping.json')
    current_path = CURRENT / 'original_cell_mapping.json'
    current = read(current_path)
    proof_path = CURRENT / 'mapping_proof.json'
    proof = read(proof_path)
    assert hashlib.sha256(proof_path.read_bytes()).hexdigest() == PROOF_SHA256
    assert hashlib.sha256(current_path.read_bytes()).hexdigest() == MAPPING_SHA256
    assert proof['new_mapping_sha256'] == MAPPING_SHA256
    assert hashlib.sha256((E / 'original_cell_mapping.json').read_bytes()).hexdigest() == proof['frozen_v2_mapping_sha256']
    assert hashlib.sha256((E / 'declared_source_changes.json').read_bytes()).hexdigest() == proof['declared_source_changes_sha256']
    assert hashlib.sha256((E / 'source_freeze.json').read_bytes()).hexdigest() == proof['source_freeze_sha256']
    assert list(current) == list(old) == ['exercises', 'solutions']
    assert proof['changed_row_count'] == 138
    for kind in old:
        assert len(old[kind]) == len(current[kind]) == 69
        for before, after in zip(old[kind], current[kind], strict=True):
            assert list(before) == list(after)
            assert after['canonical_index'] == before['canonical_index'] + 1
            assert {k: v for k, v in after.items() if k != 'canonical_index'} == {
                k: v for k, v in before.items() if k != 'canonical_index'}


@pytest.mark.parametrize('kind', ['exercises', 'solutions'])
def test_static_indices_restore_every_original_whole_cell_and_top_metadata(kind):
    old, current = books(kind)
    assert len(current['cells']) == 105
    assert len(old['cells']) == 69
    assert guard_static_mapping(kind, current) == old
    header = read(CURRENT / 'mapping_proof.json')['approved_added_header']
    assert current['cells'][header['index']]['id'] == header['id']
    assert current['cells'][header['index']]['source'] == header['source']


@pytest.mark.parametrize('kind', ['exercises', 'solutions'])
@pytest.mark.parametrize('mutation', ['source', 'metadata', 'reorder_original', 'reorder_header'])
def test_static_mapping_rejects_mutated_cells_and_reordering(kind, mutation):
    _, current = books(kind)
    altered = deepcopy(current)
    rows = read(CURRENT / 'original_cell_mapping.json')[kind]
    index = rows[1]['canonical_index']
    if mutation == 'source':
        altered['cells'][index]['source'][0] += '\n# mutation fixture\n'
    elif mutation == 'metadata':
        altered['cells'][index]['metadata']['mutation_fixture'] = True
    elif mutation == 'reorder_original':
        other = rows[2]['canonical_index']
        altered['cells'][index], altered['cells'][other] = altered['cells'][other], altered['cells'][index]
    else:
        altered['cells'][1], altered['cells'][2] = altered['cells'][2], altered['cells'][1]
    with pytest.raises(ValueError, match='changed'):
        guard_static_mapping(kind, altered)
    assert guard_static_mapping(kind, current) == books(kind)[0]
