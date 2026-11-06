"""LoRA canonical migration fidelity and tiny CPU oracles; no pretrained study."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import nbformat
import pytest

from scripts.verified_presentation import verified_pre_wrap_notebook

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / 'chapter15_peft_misalignment/exercises/part2_lora_vs_full_finetuning'
EVIDENCE = ROOT / 'docs/mastery/evidence/lora_master_migration'
MASTER = ROOT / 'infrastructure/chapters/chapter15_peft_misalignment/master_15_2.py'
MODULE = 'chapter15_peft_misalignment.exercises.part2_lora_vs_full_finetuning'
STEM = '15.2_LoRA_vs_Full_Fine-Tuning'
OLD_STEM = '15.2_LoRA_vs_Full_Finetuning'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    return ''.join(cell['source'])


def codes(notebook):
    return [c for c in notebook['cells'] if c['cell_type'] == 'code']


def read(kind):
    return nbformat.read(SECTION / f'{STEM}_{kind}.ipynb', as_version=4)


def archived(name):
    before = json.loads((EVIDENCE / 'before.json').read_text())
    with tarfile.open(ROOT / before['archive']) as saved:
        return saved.extractfile(name).read()


def old_notebook(kind):
    return json.loads(archived(str((SECTION / f'{OLD_STEM}_{kind}.ipynb').relative_to(ROOT))))


def normalize_route(text, ordinal):
    if ordinal == 0:
        text = text.replace('import time\n', '', 1)
    if ordinal == 25:
        text = text.replace('asset_dir / "lora_full_signature_panel.png"', 'Path(__file__).resolve().parents[2] / "instructions/assets/lora_full_signature_panel.png"')
    return text


def test_archive_is_exact_and_legacy_sources_and_publication_remain_present():
    before = json.loads((EVIDENCE / 'before.json').read_text())
    path = ROOT / before['archive']
    assert sha(path.read_bytes()) == before['archive_sha256']
    with tarfile.open(path) as saved:
        for name, expected in before['files'].items():
            assert sha(saved.extractfile(name).read()) == expected, name
            if name != str((SECTION / 'solutions.py').relative_to(ROOT)):
                assert sha((ROOT / name).read_bytes()) == expected, name
    report = json.loads((SECTION / 'verification_report.json').read_text())
    assert report['report_inputs'] == before['report_inputs']
    for item in report['report_inputs']['files']:
        assert sha(archived(item['path'])) == item['sha256'], item['path']


def test_all_31_sources_outputs_and_counts_match_except_two_named_setup_routes():
    old, new = old_notebook('solutions'), read('solutions')
    normalized = nbformat.reads(json.dumps(old), as_version=4)
    parity = json.loads((EVIDENCE / 'output_parity.json').read_text())
    assert parity['fresh_model_execution'] is False
    assert parity['execution_kind'] == 'retained_historical_reference'
    assert len(old['cells']) == 62
    assert len(codes(old)) == len(parity['cells']) == 31
    assert len(codes(new)) == 32
    for ordinal, (a, b, pair) in enumerate(zip(codes(old), codes(new)[:31], parity['cells'], strict=True)):
        x, y = source(a), source(b)
        z = normalize_route(y, ordinal)
        assert [line for line in x.splitlines() if line.strip()] == [line for line in z.splitlines() if line.strip()], ordinal
        assert ast.dump(ast.parse(x)) == ast.dump(ast.parse(z)), ordinal
        assert sha(x.encode()) == pair['published_source_sha256']
        assert sha(y.encode()) == pair['generated_source_sha256']
        assert sha(ast.dump(ast.parse(z)).encode()) == pair['normalized_ast_sha256']
        assert sha(json.dumps(a.get('outputs', []), sort_keys=True).encode()) == pair['outputs_sha256']
        assert b.outputs == codes(normalized)[ordinal].outputs
        assert a['execution_count'] == b.execution_count == pair['execution_count']
        assert bool(pair['named_adaptation']) == (ordinal in {0, 25})
    assert sum(c.execution_count is not None for c in codes(new)[:31]) == 31
    assert sum(bool(c.outputs) for c in codes(new)[:31]) == 14
    assert sum(len(c.outputs) for c in codes(new)[:31]) == 20


def test_all_original_36_notebook_definitions_match_archived_module():
    old_module = archived(str((SECTION / 'solutions.py').relative_to(ROOT))).decode()
    defs = lambda text: {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    reference = defs(old_module)
    old_defs = {name: node for cell in codes(old_notebook('solutions')) for name, node in defs(source(cell)).items()}
    new_defs = {name: node for ordinal, cell in enumerate(codes(read('solutions'))[:31]) for name, node in defs(normalize_route(cell.source, ordinal)).items()}
    assert len(old_defs) == len(new_defs) == 36
    for name, node in old_defs.items():
        assert ast.dump(node) == ast.dump(reference[name]), name
        assert ast.dump(node) == ast.dump(new_defs[name]), name


def test_original_figure_bytes_and_embedded_reference_survive():
    before = json.loads((EVIDENCE / 'before.json').read_text())
    expected = before['figures']['lora_full_signature_panel.png']
    assert sha((SECTION / 'media/lora_full_signature_panel.png').read_bytes()) == expected
    assert sha((ROOT / 'chapter15_peft_misalignment/instructions/assets/lora_full_signature_panel.png').read_bytes()) == expected
    html = '\n'.join(''.join(out.get('data', {}).get('text/html', '')) for cell in codes(read('exercises')) for out in cell.outputs)
    encoded = re.findall(r'data:image/png;base64,([^\"]+)', html)
    assert len(encoded) == 2
    assert expected in {sha(base64.b64decode(value)) for value in encoded}


def complete_from_own_answer(text, answer):
    definitions = (ast.FunctionDef, ast.ClassDef)
    lines = text.splitlines(True)
    answer_lines = answer.splitlines(True)
    answers = {n.name: ''.join(answer_lines[min([n.lineno, *[d.lineno for d in n.decorator_list]]) - 1:n.end_lineno]).rstrip() for n in ast.parse(answer).body if isinstance(n, definitions)}
    for node in reversed(ast.parse(text).body):
        if isinstance(node, definitions) and 'raise NotImplementedError' in ast.get_source_segment(text, node):
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            lines[start:node.end_lineno] = [answers[node.name] + '\n']
    return ''.join(lines)


def test_all_student_exercises_complete_from_their_own_answers():
    exercises, solutions = read('exercises'), read('solutions')
    nbformat.validate(exercises)
    nbformat.validate(solutions)
    completed, stub_cells = [], 0
    for index, cell in enumerate(exercises.cells):
        if cell.cell_type != 'code':
            continue
        text = cell.source
        if 'raise NotImplementedError' in text:
            stub_cells += 1
            following = []
            for candidate in exercises.cells[index + 1:]:
                if candidate.cell_type == 'code':
                    break
                following.append(candidate.source)
            blocks = re.findall(r'<summary>Solution</summary>(.*?)</details>', '\n'.join(following), re.S)
            assert len(blocks) == 1
            answer = '\n\n'.join(re.findall(r'```python\n(.*?)```', blocks[0], re.S))
            text = complete_from_own_answer(text, answer)
            assert 'raise NotImplementedError' not in text
        if len(completed) == 0:
            assert 'RUN_BOUNDED_CPU = False' in text
            text = text.replace('RUN_BOUNDED_CPU = False', 'RUN_BOUNDED_CPU = True')
        completed.append(ast.dump(ast.parse(text)))
    assert stub_cells == 10
    assert completed == [ast.dump(ast.parse(c.source)) for c in codes(solutions)]


def test_original_teaching_lines_controls_and_specific_prediction_pairs_survive():
    text = '\n'.join(c.source for c in verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb').cells if c.cell_type == 'markdown')
    lines = {line.strip() for line in text.splitlines() if line.strip()}
    for cell in old_notebook('exercises')['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        old = re.sub(r'<details>\s*<summary>Solution</summary>.*?</details>', '', source(cell), flags=re.S)
        for line in old.splitlines():
            if line.strip() and not line.startswith('# [15.2]') and line != '## Learning Objectives':
                assert line.strip() in lines, line
    assert text.count('### Forecast —') == text.count('### Reflect —') == 12
    for required in ('historical reference evidence', 'not a fresh execution', 'method-specific learning rates', '0.02', '0.00005', 'same-norm', 'random labels', 'protected', 'Try It Yourself', 'confidence', 'causal reasoning', 'exposure status', 'two-update'):
        assert required.lower() in text.lower(), required
    assert 'printing these settings is not an experiment' in text


def test_import_is_safe_outside_the_repository(tmp_path):
    script = '''
import contextlib, importlib, io, os, sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
def forbidden(*args, **kwargs):
    raise AssertionError('Import ran study behavior')
plt.subplots = forbidden
torch.cuda.is_available = forbidden
torch.cuda.empty_cache = forbidden
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    module = importlib.import_module('chapter15_peft_misalignment.exercises.part2_lora_vs_full_finetuning.solutions')
assert not captured.getvalue()
assert not list(Path.cwd().iterdir())
assert 'transformers' not in sys.modules
assert not module.RUN_BOUNDED_CPU and not module.RUN_FULL_CUDA
assert module.time.perf_counter() > 0
for name in ('LoRALinear', 'train_on_codebook', 'evaluate_direction_ablation', 'run_reference_experiment', 'run_smoke_test', 'run_gpu_test', 'run_full_experiment'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'Import safety passed'


def test_notebook_wrappers_have_timing_and_notebook_safe_figure_route(monkeypatch):
    module = importlib.import_module(MODULE + '.solutions')
    assert module.time.perf_counter() > 0
    assert 'asset_dir / "lora_full_signature_panel.png"' in inspect.getsource(module.run_reference_experiment)
    assert '__file__' not in inspect.getsource(module.run_reference_experiment)
    calls = []
    metrics = {'accuracy': 1.0, 'target_probability': 1.0}
    fake = {'behavior': {name: dict(metrics) for name in ('base', 'lora_r1', 'lora_r4', 'full', 'random_labels', 'same_norm_random')}, 'causal_intervention': {'lora_r4': {'target_probability_drop': .2, 'random_direction_drop': .01}}, 'spectra': {'lora_r4': {'effective_rank': 2.0}}, 'protected_nll_change': {'lora_r4': 0.0}, 'dataset': {'heldout_examples': 20}, 'activation_drift': {'lora_r4': {'top_variance_fraction': .5}}}
    def fake_experiment(**kwargs):
        calls.append(kwargs)
        return fake
    monkeypatch.setattr(module, '_cuda_metadata', lambda budget: {'device': 'fixture', 'max_vram_gb': budget})
    monkeypatch.setattr(module, 'run_reference_experiment', fake_experiment)
    monkeypatch.setattr(module.t.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(module.t.cuda, 'max_memory_allocated', lambda device: 0)
    monkeypatch.setattr(module.t.cuda, 'get_device_name', lambda device: 'test-fixture')
    module.run_gpu_test()
    module.run_full_experiment()
    assert calls == [dict(device='fixture', steps=96, batch_size=16, seed=0, make_figure=True)] * 2


ORACLES = [
    ('test_toy_lora_delta_uses_scaled_b_times_a', ['toy_lora_delta']),
    ('test_lora_linear_starts_as_exact_base_and_trains_only_factors', ['LoRALinear']),
    ('test_merge_unmerge_preserves_logits_and_base_weight', ['merge_unmerge_max_diff', 'LoRALinear']),
    ('test_inject_lora_replaces_real_named_linear_modules', ['inject_lora', 'LoRALinear']),
    ('test_freeze_except_lora_exposes_exact_parameter_budget', ['freeze_except_lora', 'LoRALinear']),
    ('test_codebook_split_has_disjoint_templates_and_twenty_heldout_examples', ['make_codebook_examples']),
    ('test_random_label_control_preserves_prompts_and_balance', ['make_codebook_examples', 'randomize_training_labels']),
    ('test_matched_training_loop_learns_exact_next_token_organism', ['train_on_codebook']),
    ('test_spectrum_distinguishes_rank_one_from_full_rank', ['summarize_spectrum']),
    ('test_dominant_activation_drift_recovers_planted_direction', ['dominant_activation_drift']),
    ('test_direction_ablation_removes_exact_adaptation_component', ['evaluate_direction_ablation']),
    ('test_same_norm_random_lora_matches_learned_norm', ['make_same_norm_random_lora']),
]


@pytest.mark.parametrize('oracle,functions', ORACLES)
def test_generated_visible_methods_against_independent_cpu_oracles(oracle, functions):
    torch = importlib.import_module('torch')
    module = importlib.import_module(MODULE + '.solutions')
    tests = importlib.import_module(MODULE + '.tests')
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        getattr(tests, oracle)(*[getattr(module, function) for function in functions])
    finally:
        torch.set_default_dtype(previous)


def test_additive_cpu_play_knobs_match_independent_diagonal_oracles(monkeypatch):
    torch = importlib.import_module('torch')
    module = importlib.import_module(MODULE + '.solutions')
    def forbidden(*args, **kwargs):
        raise AssertionError('Matrix play attempted model loading or training')
    monkeypatch.setattr(module, 'load_pinned_model', forbidden)
    monkeypatch.setattr(module, 'train_on_codebook', forbidden)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for layer, diagonal in enumerate(((3., 1., .3, .1), (1.5, 1.2, .8, .4))):
            for rank in (1, 2, 3, 4):
                for replay in (0., .2, 1.):
                    result = module.lora_matrix_play(rank=rank, layer=layer, protected_weight=replay)
                    expected = torch.diag(torch.tensor([value / (1 + replay) if i < rank else 0 for i, value in enumerate(diagonal)], dtype=torch.float64))
                    torch.testing.assert_close(result['update'], expected, atol=1e-12, rtol=1e-12)
                    module.check_lora_matrix_play(result)
                    assert int(torch.linalg.matrix_rank(result['update'])) == rank
        weak = module.lora_matrix_play(rank=2, layer=0, protected_weight=0.)
        strong = module.lora_matrix_play(rank=2, layer=0, protected_weight=1.)
        assert strong['metrics']['Rank-constrained fit']['replay_displacement'] < weak['metrics']['Rank-constrained fit']['replay_displacement']
        assert strong['metrics']['Rank-constrained fit']['task_error'] > weak['metrics']['Rank-constrained fit']['task_error']
        for invalid in ({'rank': 0, 'layer': 0, 'protected_weight': 0.}, {'rank': 1, 'layer': 2, 'protected_weight': 0.}, {'rank': 1, 'layer': 0, 'protected_weight': -1.}):
            with pytest.raises(ValueError):
                module.lora_matrix_play(**invalid)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_additive_cpu_play_has_separate_bound_execution_and_figure_receipt():
    receipt = json.loads((EVIDENCE / 'fixture_execution.json').read_text())
    cell = codes(read('solutions'))[31]
    assert 'PLAY_FIXTURE_RANK' in cell.source
    assert receipt['execution_kind'] == 'fresh_exact_cpu_fixture'
    assert receipt['evidence_class'] == 'analytic_matrix_oracle'
    assert receipt['source_sha256'] == sha(cell.source.encode())
    assert receipt['outputs_sha256'] == sha(json.dumps(cell.outputs, sort_keys=True).encode())
    assert receipt['figure_sha256'] == sha((ROOT / receipt['figure']).read_bytes())
    assert receipt['code_ordinal'] == 31
    assert receipt['device'] == 'cpu'
    assert receipt['model_loader_called'] is receipt['optimizer_used'] is False
    assert receipt['historical_model_execution'] is False
    assert cell.execution_count == 1 and len(cell.outputs) == 2
    module = importlib.import_module(MODULE + '.solutions')
    for name, expected in receipt['method_source_sha256'].items():
        assert sha(inspect.getsource(getattr(module, name)).encode()) == expected, name
    assert receipt['configuration'] == {'rank': 2, 'layer': 0, 'protected_weight': .2, 'seed': 0}
    assert 'Exact diagonal, factor-forward, same-norm and objective controls passed.' in receipt['stdout']
    assert len(json.loads((EVIDENCE / 'output_parity.json').read_text())['cells']) == 31
