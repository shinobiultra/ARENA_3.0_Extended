"""The inverse accepts only a fully bound, semantically identical publication."""
import json
from pathlib import Path

import pytest

from scripts import verified_presentation as vp


def test_bound_real_notebook_and_unbound_path():
    path = vp.ROOT / ('chapter17_training_dynamics/exercises/part1_checkpoint_archaeology/'
                      '17.1_Checkpoint_Archaeology_and_Grokking_exercises.ipynb')
    assert vp.verified_pre_wrap_notebook(path).cells
    with pytest.raises(ValueError, match='not bound'):
        vp.verified_pre_wrap_bytes(vp.ROOT / 'README.md')


@pytest.mark.parametrize('mutation', ['prose', 'code', 'output', 'widgets', 'fence'])
def test_semantic_or_execution_changes_are_rejected(mutation):
    old = {'cells': [{'cell_type': 'markdown', 'source': 'A teaching claim.\n\n```python\nx = 1\n```'},
                     {'cell_type': 'code', 'source': 'x = 1', 'outputs': [], 'execution_count': None}],
           'metadata': {}}
    new = json.loads(json.dumps(old))
    if mutation == 'prose':
        new['cells'][0]['source'] = new['cells'][0]['source'].replace('claim', 'falsehood')
    elif mutation == 'fence':
        new['cells'][0]['source'] = new['cells'][0]['source'].replace('x = 1', 'x = 2')
    elif mutation == 'code':
        new['cells'][1]['source'] = 'x = 2'
    elif mutation == 'output':
        new['cells'][1]['outputs'] = [{'output_type': 'stream', 'text': 'fabricated'}]
    else:
        new['metadata']['widgets'] = {'fabricated': True}
    with pytest.raises(ValueError):
        vp._verify(json.dumps(old).encode(), json.dumps(new).encode(), '.ipynb')


def test_manifest_and_current_hash_rejected(monkeypatch):
    monkeypatch.setattr(vp, 'MANIFEST_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='manifest'):
        vp.verified_pre_wrap_bytes(vp.ROOT / 'README.md')


def test_expected_historical_hash_and_current_mutation(monkeypatch):
    path = vp.ROOT / ('chapter17_training_dynamics/exercises/part1_checkpoint_archaeology/'
                      '17.1_Checkpoint_Archaeology_and_Grokking_exercises.ipynb')
    before = vp.verified_pre_wrap_bytes(path)
    assert vp.historical_bytes(path, vp._sha(before)) == before
    with pytest.raises(ValueError, match='Requested historical'):
        vp.historical_bytes(path, '0' * 64)
    original = Path.read_bytes
    def changed(self):
        result = original(self)
        return result + b' ' if self == path else result
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError, match='hash changed'):
        vp.verified_pre_wrap_bytes(path)


def test_master_code_and_literal_fence_changes_rejected():
    old = b"# ! CELL TYPE: markdown\nr'''\nA claim.\n'''\n# ! CELL TYPE: code\nx = 1\n"
    with pytest.raises(ValueError, match='Non-Markdown'):
        vp._verify(old, old.replace(b'x = 1', b'x = 2'), '.py')


def test_before_backup_hash_change_rejected(monkeypatch):
    path = vp.ROOT / ('chapter17_training_dynamics/exercises/part1_checkpoint_archaeology/'
                      '17.1_Checkpoint_Archaeology_and_Grokking_exercises.ipynb')
    original = Path.read_bytes
    def changed(self):
        result = original(self)
        return result + b' ' if self.name.endswith('.before') else result
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError, match='hash changed'):
        vp.verified_pre_wrap_bytes(path)


def test_unknown_top_metadata_change_rejected_after_warm_verification():
    import copy
    import json
    from scripts import verified_presentation as presentation
    raw = json.dumps({'cells': [], 'metadata': {'language_info': {'name': 'python'}}}).encode()
    presentation._verify(raw, raw, '.ipynb')
    changed = json.loads(raw)
    changed['metadata']['unexpected_provenance'] = {'answer': 'must fail'}
    with pytest.raises(ValueError, match='Notebook metadata changed'):
        presentation._verify(raw, json.dumps(changed).encode(), '.ipynb')


def test_restored_exercise_rejects_unknown_metadata_after_warm_path_lookup(monkeypatch):
    from pathlib import Path
    import json
    from scripts import verified_presentation as presentation
    row = json.loads(presentation.RESTORATION.read_text())['restorations'][0]
    path = (presentation.ROOT/row['path']).resolve()
    presentation.verified_pre_wrap_bytes(path)
    original = Path.read_bytes
    def altered(p):
        raw = original(p)
        if p.resolve() == path:
            notebook = json.loads(raw)
            notebook['metadata']['unexpected'] = 'new unknown metadata'
            return json.dumps(notebook).encode()
        return raw
    monkeypatch.setattr(Path, 'read_bytes', altered)
    with pytest.raises(ValueError, match='Metadata restoration source binding changed'):
        presentation.verified_pre_wrap_bytes(path)
