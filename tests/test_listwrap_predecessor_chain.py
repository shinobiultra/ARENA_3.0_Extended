"""Exact archived predecessors and current-byte mutation checks; no lesson execution."""
from pathlib import Path
import hashlib
import importlib.util
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/mastery/evidence/listwrap_preservation_20261019'
RECEIPT_SHA256 = '57e6a93e9d06f6e65d11ce825d5bff53293566cdf003c540e7ea00342511403d'
HELPER_SHA256 = 'd29dcf22047dbc7eddda4e6977c75792807b15326eff10dda0312424ccb48fb9'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def chain():
    raw = (BASE / 'chain_proposal.py').read_bytes()
    assert digest(raw) == HELPER_SHA256
    assert digest((BASE / 'proposed_chain.json').read_bytes()) == RECEIPT_SHA256
    spec = importlib.util.spec_from_file_location('listwrap_chain_regression', BASE / 'chain_proposal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_ten_immediate_and_eight_original_edges(chain):
    rows = json.loads((BASE / 'proposed_chain.json').read_text())['rows']
    original = json.loads((ROOT / 'docs/mastery/evidence/root_cpu_20261019/'
                           'canonical_markdown_refresh/manifest.json').read_text())
    old_rows = {r['path']: r for s in original['sections'] for r in s['files']}
    assert len(rows) == 10
    composed = 0
    for row in rows:
        assert digest(chain.historical_bytes(ROOT, row['path'], row['before_sha256'])) == row['before_sha256']
        if row['kind'] == 'canonical':
            expected = old_rows[row['path']]['before_sha256']
            assert digest(chain.historical_bytes(ROOT, row['path'], expected)) == expected
            composed += 1
    assert composed == 8


@pytest.mark.parametrize('member', [
    'proposed_chain.json', 'predecessors.tar.gz',
    'predecessors.tar.gz.review-bundle.json', 'member_map.json', 'current',
])
def test_warm_lookup_rejects_changed_bound_bytes(chain, monkeypatch, member):
    row = json.loads((BASE / 'proposed_chain.json').read_text())['rows'][0]
    chain.historical_bytes(ROOT, row['path'], row['before_sha256'])
    target = ROOT / row['path'] if member == 'current' else BASE / member
    original = Path.read_bytes

    def changed(path):
        raw = original(path)
        return raw + b' ' if path == target else raw

    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError):
        chain.historical_bytes(ROOT, row['path'], row['before_sha256'])


def test_unknown_paths_hashes_and_older_page_edges_fail(chain):
    rows = json.loads((BASE / 'proposed_chain.json').read_text())['rows']
    with pytest.raises(ValueError, match='Unknown'):
        chain.historical_bytes(ROOT, 'README.md', '0' * 64)
    canonical = next(r for r in rows if r['kind'] == 'canonical')
    with pytest.raises(ValueError, match='Requested original identity'):
        chain.historical_bytes(ROOT, canonical['path'], '0' * 64)
    page = next(r for r in rows if r['kind'] == 'standalone_page')
    with pytest.raises(ValueError, match='separately pinned older'):
        chain.historical_bytes(ROOT, page['path'], '0' * 64)
