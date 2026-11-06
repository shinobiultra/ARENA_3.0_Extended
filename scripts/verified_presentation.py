"""Invert only the pinned, independently reviewed Markdown presentation refresh."""
from __future__ import annotations

import copy
import gzip
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from markdown_it import MarkdownIt

from arena_ext.readability_compatibility import _markdown_semantics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/mastery/evidence/root_cpu_20261019/canonical_markdown_refresh'
MANIFEST_SHA256 = '63b6586cdd9f4e98537c9505ee2563aefae0783c731098b4afc32e97c4c40e29'
RESTORATION = ROOT / 'docs/mastery/evidence/metadata_restore_20261019/restoration.json'
RESTORATION_SHA256 = 'bba26be16468fd8d4d9330d6e03e79a2c6908f04a897617977e5f1a1afc9852f'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=32)
def _markdown(before, after):
    if _markdown_semantics(before) != _markdown_semantics(after):
        raise ValueError('Markdown rendering changed')
    parser = MarkdownIt('commonmark')
    def fences(text):
        return [(t.type, t.markup, t.info, t.content) for t in parser.parse(text)
                if t.type in {'fence', 'code_block'}]
    if fences(before) != fences(after):
        raise ValueError('Literal code block changed')


@lru_cache(maxsize=16)
def _verify(before: bytes, after: bytes, suffix: str):
    if suffix == '.ipynb':
        old, new = json.loads(before), json.loads(after)
        if len(old['cells']) != len(new['cells']):
            raise ValueError('Cell count changed')
        if old.get('metadata', {}) != new.get('metadata', {}):
            raise ValueError('Notebook metadata changed')
        if old.get('metadata', {}).get('widgets') != new.get('metadata', {}).get('widgets'):
            raise ValueError('Widget graph changed')
        for a, b in zip(old['cells'], new['cells']):
            if a['cell_type'] != b['cell_type']:
                raise ValueError('Cell type changed')
            if a['cell_type'] == 'markdown':
                _markdown(''.join(a['source']), ''.join(b['source']))
            else:
                a, b = copy.deepcopy(a), copy.deepcopy(b)
                a.pop('id', None)
                b.pop('id', None)
                if a != b:
                    raise ValueError('Complete code object changed')
    elif suffix == '.py':
        split = lambda text: re.split(r'(?=^# ! CELL TYPE: )', text, flags=re.M)
        old, new = split(before.decode()), split(after.decode())
        if len(old) != len(new):
            raise ValueError('Master cell count changed')
        for a, b in zip(old, new):
            if a.startswith('# ! CELL TYPE: markdown'):
                pattern = r"(?ms)^r?(['\"]{3})(.*?)^\1"
                x, y = re.search(pattern, a), re.search(pattern, b)
                if x is None or y is None:
                    raise ValueError('Unknown Markdown wrapper')
                if a[:x.start(2)] + a[x.end(2):] != b[:y.start(2)] + b[y.end(2):]:
                    raise ValueError('Markdown wrapper changed')
                _markdown(x[2], y[2])
            elif a != b:
                raise ValueError('Non-Markdown source changed')
    elif suffix == '.md':
        _markdown(before.decode(), after.decode())
    else:
        raise ValueError('Unrecognized publication type')


def _verify_metadata_restoration(name, canonical_row, before, current):
    """Verify the immutable broken publication and precisely one restored field."""
    raw = RESTORATION.read_bytes()
    if _sha(raw) != RESTORATION_SHA256:
        raise ValueError('Metadata restoration manifest changed')
    rows = [r for r in json.loads(raw)['restorations'] if r['path'] == name]
    if len(rows) != 1:
        raise ValueError('Publication or before-source hash changed')
    row = rows[0]
    broken = gzip.decompress((ROOT / row['broken_after_gzip']).read_bytes())
    if (row['canonical_before_sha256'] != canonical_row['before_sha256']
            or row['canonical_after_sha256'] != canonical_row['after_sha256']
            or _sha(broken) != canonical_row['after_sha256']
            or _sha(current) != row['restored_current_sha256']):
        raise ValueError('Metadata restoration source binding changed')
    original, old, new = json.loads(before), json.loads(broken), json.loads(current)
    if 'fresh_reference' in old.get('metadata', {}):
        raise ValueError('Broken publication already contains reference metadata')
    restored = new.get('metadata', {}).get('fresh_reference')
    if restored != original.get('metadata', {}).get('fresh_reference') or restored != row['exact_original_value']:
        raise ValueError('Reference metadata restoration differs from original')
    inverse = copy.deepcopy(new)
    inverse['metadata'].pop('fresh_reference')
    if inverse != old:
        raise ValueError('Unrelated metadata restoration change')


def verified_pre_wrap_bytes(path: Path) -> bytes:
    """Return exact old bytes only after strict current-publication validation."""
    name = path.resolve().relative_to(ROOT.resolve()).as_posix()
    raw = (BASE / 'manifest.json').read_bytes()
    if _sha(raw) != MANIFEST_SHA256:
        raise ValueError('Unrecognized presentation manifest')
    rows = [row for section in json.loads(raw)['sections'] for row in section['files']
            if row['path'] == name]
    if len(rows) != 1:
        raise ValueError('File is not bound to reviewed presentation')
    row = rows[0]
    before = (BASE / 'before' / (name + '.before')).read_bytes()
    after = path.read_bytes()
    if _sha(before) != row['before_sha256']:
        raise ValueError('Publication or before-source hash changed')
    if _sha(after) != row['after_sha256']:
        _verify_metadata_restoration(name, row, before, after)
    _verify(before, after, path.suffix)
    return before


def verified_pre_wrap_notebook(path: Path):
    """Read the verified inverse for unchanged historical teaching assertions."""
    import nbformat

    return nbformat.reads(verified_pre_wrap_bytes(path).decode(), as_version=4)


def historical_bytes(path: Path, expected_sha256: str) -> bytes:
    """Require exact historical identity, with only the pinned inverse fallback."""
    current = path.read_bytes()
    if _sha(current) == expected_sha256:
        return current
    before = verified_pre_wrap_bytes(path)
    if _sha(before) != expected_sha256:
        raise ValueError('Requested historical hash does not match reviewed before source')
    return before


def verified_before_source(path: Path) -> str:
    """Expose verified old source text for historical line assertions only."""
    return verified_pre_wrap_bytes(path).decode()
