"""Fail-closed exact historical bindings across reviewed presentation changes."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path

from scripts.verified_presentation import _markdown, historical_bytes as _canonical_historical_bytes

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/mastery/evidence/historical_bindings_20261019/page_bridge.json'
MANIFEST_SHA256 = '92bcc655e0d56797bec74a3541f96bfd233ddf0f2e3dbfa00ced2fc7a074d5e2'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _verify_loader(before, current):
    """Invert precisely the new inline-page-media loader, retaining all other AST."""
    old, new = ast.parse(before.decode()), ast.parse(current.decode())
    removed = []
    for node in list(new.body):
        if isinstance(node, ast.ImportFrom) and node.module == 'infrastructure.extension.render_media':
            if ast.dump(node) != ast.dump(ast.parse('from infrastructure.extension.render_media import inline_page_media').body[0]):
                raise ValueError('Unknown media-loader import')
            new.body.remove(node)
            removed.append(node)
    class InvertLoader(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'inline_page_media':
                expected = ast.parse('inline_page_media(markdown_content_file.read_text(encoding="utf-8"), markdown_content_file)', mode='eval').body
                if ast.dump(node) != ast.dump(expected):
                    raise ValueError('Unknown media-loader invocation')
                self.count += 1
                return node.args[0]
            return self.generic_visit(node)
    transform = InvertLoader()
    transform.visit(new)
    if len(removed) != 1 or transform.count != 1 or ast.dump(old) != ast.dump(new):
        raise ValueError('Non-loader page Python changed')


def historical_bytes(path: Path, expected_sha256: str) -> bytes:
    """Require exact old hash plus complete current-source/presentation proof."""
    path = path.resolve()
    current = path.read_bytes()
    if _sha(current) == expected_sha256:
        return current
    rel = path.relative_to(ROOT).as_posix()
    manifest_raw = MANIFEST.read_bytes()
    if _sha(manifest_raw) != MANIFEST_SHA256:
        raise ValueError('Historical page manifest changed')
    rows = [r for r in json.loads(manifest_raw)['pages'] if r['path'] == rel]
    if not rows:
        return _canonical_historical_bytes(path, expected_sha256)
    if len(rows) != 1:
        raise ValueError('Ambiguous historical page')
    row = rows[0]
    before = gzip.decompress((ROOT / row['before_gzip']).read_bytes())
    if _sha(before) != row['before_sha256'] or row['before_sha256'] != expected_sha256 or _sha(current) != row['current_sha256']:
        raise ValueError('Unbound historical/current page bytes')
    if path.suffix == '.py':
        _verify_loader(before, current)
    elif path.suffix == '.md':
        previous = before.decode()
        for media in row['media']:
            if _sha((ROOT / media['file']).read_bytes()) != media['sha256']:
                raise ValueError('Bound local media changed')
            if media['old_url'] not in previous:
                raise ValueError('Unknown historical media mapping')
            previous = previous.replace(media['old_url'], media['current_local_src'])
        _markdown(previous, current.decode())
    else:
        raise ValueError('Unknown historical page kind')
    return before
