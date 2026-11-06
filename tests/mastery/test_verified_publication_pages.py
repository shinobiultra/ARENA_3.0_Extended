"""Historical pages must bind exact bytes and a narrow current display bridge."""
from pathlib import Path
import ast
import hashlib
import json
import pytest

from scripts import verified_publication_pages as bridge

ROWS = json.loads(bridge.MANIFEST.read_text())['pages']


@pytest.mark.parametrize('row', ROWS, ids=lambda row: row['path'])
def test_each_page_recovers_exact_immutable_publication(row):
    path = bridge.ROOT/row['path']
    assert hashlib.sha256(bridge.historical_bytes(path, row['before_sha256'])).hexdigest() == row['before_sha256']


def test_unrelated_current_page_change_is_rejected(monkeypatch):
    row = ROWS[0]
    target = (bridge.ROOT/row['path']).resolve()
    original = Path.read_bytes
    def altered(path):
        raw = original(path)
        return raw+b'\nUnrelated new claim.\n' if path.resolve() == target else raw
    monkeypatch.setattr(Path, 'read_bytes', altered)
    with pytest.raises(ValueError, match='Unbound'):
        bridge.historical_bytes(target, row['before_sha256'])


def test_wrong_requested_historical_identity_is_rejected():
    with pytest.raises(ValueError, match='Unbound'):
        bridge.historical_bytes(bridge.ROOT/ROWS[0]['path'], '0'*64)


def test_media_hash_change_is_rejected(monkeypatch):
    row = next(row for row in ROWS if row['media'])
    target = (bridge.ROOT/row['media'][0]['file']).resolve()
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: original(p)+b'x' if p.resolve()==target else original(p))
    with pytest.raises(ValueError, match='media changed'):
        bridge.historical_bytes(bridge.ROOT/row['path'], row['before_sha256'])


def test_loader_bridge_rejects_unrelated_ast_or_alias_changes():
    row = next(row for row in ROWS if row['path'].endswith('.py'))
    import gzip
    before = gzip.decompress((bridge.ROOT/row['before_gzip']).read_bytes())
    current = (bridge.ROOT/row['path']).read_bytes()
    with pytest.raises(ValueError, match='Non-loader'):
        bridge._verify_loader(before, current+b'\nUNRELATED = 1\n')
    with pytest.raises(ValueError, match='Unknown media-loader'):
        bridge._verify_loader(before, current.replace(b'import inline_page_media', b'import inline_page_media as changed'))
