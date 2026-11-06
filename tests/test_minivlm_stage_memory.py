"""MiniVLM stage handoff contracts, using fake owners and AST only (no model imports)."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMPANION = ROOT / 'mastery_hub/practice_addons/mini_vlm_memory_handoff'
SOURCE = COMPANION / 'stage_memory.py'


class TensorOwner:
    def __init__(self, device='cuda', payload=7):
        self.device, self.payload = device, payload

    def detach(self):
        return self

    def to(self, device):
        return TensorOwner(str(device), self.payload)


class ModuleOwner:
    def __init__(self):
        self.device = 'cuda'

    def to(self, device):
        self.device = str(device)
        return self


def handoff():
    tree = ast.parse(SOURCE.read_text())
    node = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'move_minivlm_stage'), None)
    assert node is not None, 'stage handoff must actually move the live prior owners'
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<MiniVLM memory helper only>', 'exec'), namespace)
    return namespace['move_minivlm_stage']


def test_large_prior_owners_move_to_cpu_without_losing_handles_or_qwen():
    move = handoff()
    names = ['gpt2', 'lm_a', 'vlm', 'siglip', 'projector_a', 'projector',
             'untrained_projector', 'random_projector', 'text_only_lm']
    namespace = {name: ModuleOwner() for name in names}
    namespace.update({name: TensorOwner() for name in
                      ['train_features', 'test_features', 'blank_features', 'noise_features',
                       'shuffled_features', 'clean_features', 'corrupt_features', 'image_resid']})
    originals = namespace.copy()
    namespace.update(t=SimpleNamespace(Tensor=TensorOwner), nn=SimpleNamespace(Module=ModuleOwner))
    selected_qwen = ModuleOwner()
    namespace['qwen'] = selected_qwen
    qwen_inputs = {'input_ids': TensorOwner()}
    namespace['example_inputs'] = qwen_inputs
    curves = {'object tokens': [0.2, 0.5], 'background': [0.0, 0.01]}
    namespace['patch_curves'] = curves
    move(namespace, 'cpu')
    assert all(namespace[name].device == 'cpu' for name in originals)
    assert all(namespace[name] is originals[name] for name in names)
    assert namespace['patch_curves'] is curves
    assert selected_qwen.device == 'cuda' and namespace['qwen'] is selected_qwen
    assert namespace['example_inputs'] is qwen_inputs and qwen_inputs['input_ids'].device == 'cuda'


def test_nested_cache_aliases_and_play_inputs_survive_round_trip():
    move = handoff()
    shared = TensorOwner(payload=42)
    clean_resids = {0: shared}
    cache = SimpleNamespace(cache_dict={'resid_pre': shared})
    masks = {'object': shared}
    namespace = {'clean_cache': cache, 'clean_resids': clean_resids,
                 'clean_embeds': shared, 'a_masks': masks, 'colour_ids': shared,
                 'a_tokens': TensorOwner(), 'pair_tokens': TensorOwner()}
    namespace.update(t=SimpleNamespace(Tensor=TensorOwner), nn=SimpleNamespace(Module=ModuleOwner))
    move(namespace, 'cpu')
    moved = namespace['clean_embeds']
    assert moved.device == 'cpu' and moved.payload == 42
    assert cache.cache_dict['resid_pre'] is clean_resids[0] is masks['object'] is moved
    assert namespace['colour_ids'] is moved
    assert namespace['clean_cache'] is cache and namespace['clean_resids'] is clean_resids
    move(namespace, 'cuda')
    assert namespace['a_tokens'].device == namespace['pair_tokens'].device == 'cuda'
    assert cache.cache_dict['resid_pre'] is clean_resids[0] is namespace['clean_embeds']
    assert namespace['clean_embeds'].payload == 42


def test_selected_qwen_is_protected_even_if_an_old_model_name_aliases_it():
    selected = ModuleOwner()
    namespace = dict(t=SimpleNamespace(Tensor=TensorOwner), nn=SimpleNamespace(Module=ModuleOwner),
                     qwen=selected, vlm=selected)
    handoff()(namespace, 'cpu')
    assert namespace['qwen'] is namespace['vlm'] is selected and selected.device == 'cuda'


def test_guide_replaces_legacy_cleanup_before_qwen_and_keeps_historical_binding():
    guide = (COMPANION / 'README.md').read_text()
    assert 'Replace the old cleanup; do not run both.' in guide
    assert 'qwen, qwen_processor = utils.load_qwen(device)' in guide
    assert 'exact historical 61-cell execution' in guide
    assert 'No GPU peak-memory or out-of-memory guarantee' in guide
    source = SOURCE.read_text()
    tree = ast.parse(source)
    assert all(not isinstance(node, (ast.Import, ast.ImportFrom)) or
               isinstance(node, ast.ImportFrom) and node.module == '__future__' for node in tree.body)
    assert 'globals().pop' not in source
    handoff()


def test_copy_paste_recipe_imports_from_notebook_directory_without_cuda(tmp_path, monkeypatch):
    import sys

    guide = (COMPANION / 'README.md').read_text()
    snippet = guide.split('```python\n', 1)[1].split('```', 1)[0]
    model, features = ModuleOwner(), TensorOwner()
    namespace = dict(root_dir=ROOT, device=SimpleNamespace(type='cpu'),
                     t=SimpleNamespace(Tensor=TensorOwner), nn=SimpleNamespace(Module=ModuleOwner),
                     gpt2=model, test_features=features)
    monkeypatch.chdir(tmp_path)
    old_path = sys.path.copy()
    try:
        exec(compile(snippet, '<README copy/paste recipe>', 'exec'), namespace)
    finally:
        sys.path[:] = old_path
    assert model.device == namespace['test_features'].device == 'cpu'
    assert namespace['move_minivlm_stage'].__module__.endswith('stage_memory')


def test_return_recipe_parks_existing_qwen_and_allows_already_deleted_qwen():
    guide = (COMPANION / 'README.md').read_text()
    snippet = guide.split('```python\n')[2].split('```', 1)[0]
    class CPUDevice(str):
        type = 'cpu'

    for qwen_exists in (True, False):
        namespace = dict(move_minivlm_stage=handoff(), device=CPUDevice('cpu'),
                         t=SimpleNamespace(Tensor=TensorOwner), nn=SimpleNamespace(Module=ModuleOwner),
                         gc=SimpleNamespace(collect=lambda: None), vlm=ModuleOwner())
        selected = ModuleOwner()
        if qwen_exists:
            namespace['qwen'] = selected
        exec(compile(snippet, '<README return-to-MiniVLM recipe>', 'exec'), namespace)
        assert namespace['vlm'].device == 'cpu'
        if qwen_exists:
            assert selected.device == 'cpu'
        else:
            assert 'qwen' not in namespace
