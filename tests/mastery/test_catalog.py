"""Catalog errors must not invent a runnable or released companion."""
import copy
import json
from pathlib import Path

import pytest

from arena_ext.mastery.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[2]


def catalog_file(tmp_path, records):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps({'schema_version': 1, 'units': records}))
    return path


def unit(**changes):
    row = dict(
        id='u01_basics', title='Basics', source_url='https://deep-learning-alchemy.github.io/',
        checked_on='2026-10-15', source_status='published', companion_status='planned',
        prerequisites=[], original_paths=['README.md'], entrypoint=None
    )
    row.update(changes)
    return row


def test_catalog_statuses_are_independent(tmp_path):
    rows = load_catalog(catalog_file(tmp_path, [unit()]), root=ROOT)
    assert rows[0]['companion_status'] == 'planned'
    assert rows[0]['entrypoint'] is None


@pytest.mark.parametrize('changes', [
    dict(id='../escape'), dict(source_status='ready'), dict(companion_status='complete'),
    dict(prerequisites=['missing']), dict(original_paths=['missing.md']),
    dict(original_paths=['../README.md']),
    dict(entrypoint='mastery_hub/missing.py', companion_status='validated')
])
def test_catalog_rejects_unknown_ids_cycles_and_missing_paths(tmp_path, changes):
    with pytest.raises(ValueError):
        load_catalog(catalog_file(tmp_path, [unit(**changes)]), root=ROOT)


def test_catalog_rejects_duplicates_and_cycles(tmp_path):
    first = unit(prerequisites=['u02_optimization'])
    second = unit(id='u02_optimization', prerequisites=['u01_basics'])
    for rows in ([unit(), unit()], [first, second]):
        with pytest.raises(ValueError):
            load_catalog(catalog_file(tmp_path, rows), root=ROOT)


def test_pending_source_has_no_fabricated_release(tmp_path):
    with pytest.raises(ValueError):
        load_catalog(catalog_file(tmp_path, [unit(source_status='pending_upstream', companion_status='validated', entrypoint='README.md')]), root=ROOT)
    with pytest.raises(ValueError):
        load_catalog(catalog_file(tmp_path, [unit(assignment_url='https://example.org/guessed.pdf')]), root=ROOT)


def test_shipped_catalog_offline_and_pilot_present():
    rows = load_catalog(ROOT / 'mastery_hub/catalog.json')
    assert len(rows) == 7
    pilot = next(row for row in rows if row['id'] == 'u02_optimization')
    assert (ROOT / pilot['entrypoint']).is_file()
    exotic = next(row for row in rows if row['id'] == 'u07_exotic')
    assert exotic['source_status'] == 'pending_upstream'
    assert exotic['entrypoint'] is None
