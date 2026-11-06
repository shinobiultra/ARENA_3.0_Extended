"""Canonical finite-game migration parity and independent bounded CPU oracles."""
from __future__ import annotations

import ast
import base64
import hashlib
import gzip
import importlib
import inspect
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import nbformat
import numpy as np
import pytest

from scripts.verified_presentation import verified_pre_wrap_notebook

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'chapter16_shapley_attribution_baselines/exercises/part2_kernelshap_partition_shap_controls'
SECTION = OLD.parent / 'part2_kernelshap_partitionshap'
MASTER = ROOT / 'infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_2.py'
EVIDENCE = ROOT / 'docs/mastery/evidence/kernelshap_master_migration'
MODULE = 'chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partitionshap'
STEM = '16.2_KernelSHAP_and_PartitionSHAP'
OLD_STEM = STEM + '_Controls'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    return ''.join(cell['source'])


def codes(notebook):
    return [c for c in notebook['cells'] if c['cell_type'] == 'code']


def read(kind):
    return nbformat.read(SECTION / f'{STEM}_{kind}.ipynb', as_version=4)


def old_read(kind):
    return json.loads((OLD / f'{OLD_STEM}_{kind}.ipynb').read_text())


def normalize_routes(text, ordinal):
    if ordinal == 0:
        text = text.replace('assets_dir = root_dir / chapter / \"exercises\" / section / \"media\"', 'assets_dir = root_dir / chapter / \"instructions\" / \"assets\"').replace('section = "part2_kernelshap_partitionshap"', 'section = "part2_kernelshap_partition_shap_controls"').replace('from chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partitionshap import tests', 'import part2_kernelshap_partition_shap_controls.tests as tests')
    if ordinal == 4:
        fixed = '''    # CPU SVD gives the declared minimum-norm KKT solution reliably.
    # CUDA supports only gels, so solve this small system on CPU and return
    # the coefficients to the original device.
    solution = t.linalg.lstsq(
        kkt.cpu(), constrained_rhs.cpu(), driver="gelsd"
    ).solution
    return solution[:num_players].to(kkt.device)'''
        assert fixed in text
        text = text.replace(fixed, '    return t.linalg.lstsq(kkt, constrained_rhs).solution[:num_players]')
    if ordinal == 11:
        text = text.replace('from chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partition_shap_controls import solutions as reference', 'from part2_kernelshap_partition_shap_controls import solutions as reference')
    return text


def test_exact_archive_and_all_legacy_sources_receipts_remain_unchanged():
    before = json.loads((EVIDENCE / 'before.json').read_text())
    archive = ROOT / before['archive']
    assert sha(archive.read_bytes()) == before['archive_sha256']
    with tarfile.open(archive) as saved:
        assert {m.name for m in saved.getmembers()} == set(before['files'])
        for name, expected in before['files'].items():
            assert sha(saved.extractfile(name).read()) == expected, name
            assert sha((ROOT / name).read_bytes()) == expected, name
    report = json.loads((OLD / 'verification_report.json').read_text())
    assert report['report_inputs'] == before['report_inputs']
    for item in report['report_inputs']['files']:
        assert sha((ROOT / item['path']).read_bytes()) == item['sha256']
    nested = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in OLD.rglob('*') if p.is_file() and p.name.endswith(('.tar.gz', '.zip', '.tgz', '.tar.zst'))}
    assert nested == before['nested_archives_retained_at_original_paths']


def test_all_twelve_original_sources_and_outputs_have_strict_routed_parity():
    old, new = old_read('solutions'), read('solutions')
    normalized = nbformat.reads(json.dumps(old), as_version=4)
    receipt = json.loads((EVIDENCE / 'output_parity.json').read_text())
    assert receipt['execution_kind'] == 'fresh_corrected_finite_game_cpu'
    assert receipt['historical_reference_preserved'] is True
    executed = nbformat.reads(gzip.decompress((EVIDENCE / 'fresh_cpu_executed.ipynb.gz').read_bytes()).decode(), as_version=4)
    assert receipt['fresh_model_execution'] is False
    assert receipt['historical_code_ordinals'] == list(range(12))
    assert receipt['additive_code_ordinals'] == [12]
    assert len(old['cells']) == 26 and len(codes(old)) == 12 and len(codes(new)) == 13
    for ordinal, (a, b, pair) in enumerate(zip(codes(old), codes(new)[:12], receipt['cells'], strict=True)):
        x, y = source(a), source(b)
        z = normalize_routes(y, ordinal)
        assert [line for line in x.splitlines() if line.strip()] == [line for line in z.splitlines() if line.strip()], ordinal
        assert ast.dump(ast.parse(x)) == ast.dump(ast.parse(z)), ordinal
        assert sha(x.encode()) == pair['published_source_sha256']
        assert sha(y.encode()) == pair['generated_source_sha256']
        assert sha(ast.dump(ast.parse(z)).encode()) == pair['normalized_ast_sha256']
        assert sha(json.dumps(a.get('outputs', []), sort_keys=True).encode()) == pair['published_outputs_sha256']
        assert b.outputs == codes(executed)[ordinal].outputs
        assert sha(json.dumps(b.outputs, sort_keys=True).encode()) == pair['current_outputs_sha256']
        assert a['execution_count'] == pair['historical_execution_count']
        assert b.execution_count == pair['current_execution_count']
        assert bool(pair['named_adaptation']) == (ordinal in {0, 4, 11})
    assert sum(c.execution_count is not None for c in codes(new)[:12]) == 12
    assert sum(len(c.outputs) for c in codes(new)[:12]) == 12
    assert sum(bool(c.outputs) for c in codes(new)[:12]) == 10


def test_copied_tests_helpers_have_only_explicit_canonical_routes():
    old = (OLD / 'tests.py').read_text()
    actual = (SECTION / 'tests.py').read_text()
    assert 'from arena_ext' not in actual
    without_oracles = re.sub(r'# BEGIN INDEPENDENT ORDERING ORACLES.*?# END INDEPENDENT ORDERING ORACLES', 'from arena_ext import shapley_attribution as reference', actual, flags=re.S)
    expected = old.replace('exercises.part2_kernelshap_partition_shap_controls import (', 'exercises.part2_kernelshap_partitionshap import (').replace('return Path(__file__).resolve().parent', 'return Path(__file__).resolve().parent.parent / "part2_kernelshap_partition_shap_controls"')
    assert without_oracles == expected
    assert (SECTION / 'utils.py').read_bytes() == (OLD / 'utils.py').read_bytes()
    current_defs = {node.name: node for ordinal, c in enumerate(codes(read('solutions'))[:12]) for node in ast.parse(normalize_routes(c.source, ordinal)).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    old_defs = {node.name: node for c in codes(old_read('solutions')) for node in ast.parse(source(c)).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert len(old_defs) == 22
    assert set(current_defs) == set(old_defs)
    for name in old_defs:
        assert ast.dump(current_defs[name]) == ast.dump(old_defs[name]), name


def test_historical_and_fresh_figures_have_separate_exact_provenance():
    before = json.loads((EVIDENCE / 'before.json').read_text())
    historical = before['figures']['kernelshap_partition_controls_signature.png']
    assert sha((SECTION / 'media/kernelshap_partition_controls_signature_historical.png').read_bytes()) == historical
    assert sha((ROOT / 'chapter16_shapley_attribution_baselines/instructions/assets/kernelshap_partition_controls_signature.png').read_bytes()) == historical
    play = json.loads((EVIDENCE / 'play_execution.json').read_text())
    assert sha((ROOT / play['figure']).read_bytes()) == play['figure_sha256']
    html = '\n'.join(''.join(out.get('data', {}).get('text/html', '')) for c in codes(read('exercises')) for out in c.outputs)
    embedded = re.findall(r'data:image/png;base64,([^\"]+)', html)
    assert len(embedded) == 2
    current = json.loads((EVIDENCE / 'fresh_cpu_execution.json').read_text())
    assert {sha(base64.b64decode(value)) for value in embedded} == set(current['figures'].values())


def test_six_implementation_cells_complete_from_own_answers_and_seventh_control_answer_survives():
    exercises, solutions = read('exercises'), read('solutions')
    nbformat.validate(exercises)
    nbformat.validate(solutions)
    actual, stub_count = [], 0
    for index, cell in enumerate(exercises.cells):
        if cell.cell_type != 'code':
            continue
        text = cell.source
        if 'raise NotImplementedError' in text:
            stub_count += 1
            following = []
            for candidate in exercises.cells[index + 1:]:
                if candidate.cell_type == 'code' or candidate.source.startswith('### Exercise'):
                    break
                following.append(candidate.source)
            blocks = re.findall(r'<summary>Solution</summary>(.*?)</details>', '\n'.join(following), re.S)
            assert len(blocks) == 1
            answer = '\n\n'.join(re.findall(r'```python\n(.*?)```', blocks[0], re.S))
            answers = {n.name: ast.get_source_segment(answer, n) for n in ast.parse(answer).body if isinstance(n, ast.FunctionDef)}
            lines = text.splitlines(True)
            for node in reversed(ast.parse(text).body):
                if isinstance(node, ast.FunctionDef) and 'raise NotImplementedError' in ast.get_source_segment(text, node):
                    lines[node.lineno - 1:node.end_lineno] = [answers[node.name] + '\n']
            text = ''.join(lines)
            assert 'raise NotImplementedError' not in text
        actual.append(ast.dump(ast.parse(text)))
    assert stub_count == 6
    assert actual == [ast.dump(ast.parse(c.source)) for c in codes(solutions)]
    markdown = '\n'.join(c.source for c in exercises.cells if c.cell_type == 'markdown')
    assert 'random_errors = []' in markdown and 'for seed in range(100):' in markdown


def test_original_teaching_lines_forecasts_controls_and_transfer_are_preserved():
    current = '\n'.join(c.source for c in verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb').cells if c.cell_type == 'markdown')
    lines = {line.strip() for line in current.splitlines() if line.strip()}
    for ordinal, cell in enumerate(old_read('exercises')['cells']):
        if cell['cell_type'] != 'markdown':
            continue
        text = source(cell)
        if ordinal in {4, 6, 8, 10, 12, 14}:
            text = re.sub(r'<details>\s*<summary>Solution</summary>.*?</details>', '', text, flags=re.S)
        for line in text.splitlines():
            if line.strip() and not line.startswith('# [16.2]') and line != '## Learning Objectives':
                assert line.strip() in lines, line
    assert current.count('### Forecast —') == current.count('### Reflect —') == 10
    for phrase in ('retained historical analytic references', 'fresh complete CPU execution', 'seed standard deviations', 'confidence', 'causal reasoning', 'exposure status', 'four-way dividend', '+0.8', 'fresh analytic play', 'unexecuted'):
        assert phrase in current, phrase
    master = MASTER.read_text()
    assert 'from arena_ext' not in master and 'import *' not in master


def test_unrelated_directory_import_preserves_grad_mode_and_never_runs_study(tmp_path):
    script = '''
import contextlib, importlib, io, os, sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
def forbidden(*args, **kwargs):
    raise AssertionError('Import ran study or plotting behavior')
plt.rcParams.update = forbidden
plt.subplots = forbidden
torch.cuda.is_available = forbidden
torch.cuda.empty_cache = forbidden
before = torch.is_grad_enabled()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    module = importlib.import_module('chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partitionshap.solutions')
assert not captured.getvalue()
assert torch.is_grad_enabled() == before
assert not list(Path.cwd().iterdir())
assert 'transformers' not in sys.modules
for name in ('kernelshap_values', 'partition_shap_values', 'compare_kernelshap_play', 'run_smoke_test', 'run_gpu_test'):
    assert callable(getattr(module, name))
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'Import safety passed'


def test_fresh_cpu_play_receipt_is_bound_to_exact_source_dependencies_outputs_and_inputs():
    receipt = json.loads((EVIDENCE / 'play_execution.json').read_text())
    cell = codes(read('solutions'))[12]
    assert receipt['code_ordinal'] == 12 and receipt['execution_kind'] == 'fresh_exact_cpu_play_from_full_notebook'
    assert receipt['source_sha256'] == sha(cell.source.encode())
    assert receipt['outputs_sha256'] == sha(json.dumps(cell.outputs, sort_keys=True).encode())
    module = importlib.import_module(MODULE + '.solutions')
    for name, expected in receipt['method_source_sha256'].items():
        assert sha(inspect.getsource(getattr(module, name)).encode()) == expected
    assert receipt['configuration']['budget'] == 24 and receipt['configuration']['seed'] == 4
    assert receipt['configuration']['groups'] == [[0, 3, 4], [1, 2, 5]]
    assert receipt['pretrained_model_used'] is receipt['training_used'] is receipt['transfer_game_executed'] is False
    assert cell.execution_count == 13 and len(cell.outputs) == 3
    assert 'Permutation oracle, full-table parity, efficiency and complement controls passed.' in receipt['stdout']


def permutation_oracle(values, players):
    result = np.zeros(players)
    for order in itertools.permutations(range(players)):
        before = frozenset()
        for player in order:
            after = before | {player}
            result[player] += values[after] - values[before]
            before = after
    return result / math.factorial(players)


def owen_ordering_oracle(values, groups):
    credits, count = np.zeros(sum(map(len, groups))), 0
    for group_order in itertools.permutations(range(len(groups))):
        for within in itertools.product(*(list(itertools.permutations(groups[i])) for i in group_order)):
            count += 1
            before = frozenset()
            for player in itertools.chain.from_iterable(within):
                after = before | {player}
                credits[player] += values[after] - values[before]
                before = after
    return credits / count


ORACLES = ['dividends', 'finite_kernel', 'rows_and_baseline', 'full_kernel', 'partial_kernel', 'rank_deficient', 'sampling', 'owen', 'partition_validation', 'group_expansion', 'shuffle_efficiency', 'live_play']


@pytest.mark.parametrize('oracle', ORACLES)
def test_visible_methods_against_independent_cpu_oracles(oracle):
    torch = importlib.import_module('torch')
    m = importlib.import_module(MODULE + '.solutions')
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        dividends = {frozenset({0}): 1.0, frozenset({1}): -.5, frozenset({2}): .25, frozenset({0, 1, 2}): 1.5}
        values = m.game_from_dividends(3, dividends)
        expected = torch.tensor(permutation_oracle(values, 3), dtype=torch.float64)
        if oracle == 'dividends':
            torch.testing.assert_close(m.shapley_from_dividends(3, dividends), expected)
            torch.testing.assert_close(m.exact_shapley_values(values, num_players=3), expected)
            assert len(values) == 8 and values[frozenset()] == 0
        elif oracle == 'finite_kernel':
            for size in range(1, 6):
                assert math.isclose(m.kernelshap_kernel_weight(size, 6), 5 / (math.comb(6, size) * size * (6-size)))
            for size in (0, 6):
                with pytest.raises(ValueError):
                    m.kernelshap_kernel_weight(size, 6)
        elif oracle == 'rows_and_baseline':
            shifted = {key: value + 7 for key, value in values.items()}
            x, y, w = m.kernelshap_regression_rows(shifted, num_players=3)
            assert x.shape == (6, 3) and y.shape == w.shape == (6,)
            for row, coalition in enumerate(m.all_coalitions(3)[1:-1]):
                assert x[row].tolist() == [float(i in coalition) for i in range(3)]
                assert y[row] == values[coalition]
            torch.testing.assert_close(m.kernelshap_values(shifted, num_players=3), expected)
        elif oracle == 'full_kernel':
            torch.testing.assert_close(m.kernelshap_values(values, num_players=3), expected, atol=1e-12, rtol=0)
        elif oracle == 'partial_kernel':
            sample = m.all_coalitions(3)[1:-2]
            x, y, w = m.kernelshap_regression_rows(values, num_players=3, coalitions=sample)
            x, y, w = x.numpy(), y.numpy(), w.numpy()
            basis = np.vstack([np.eye(2), -np.ones(2)])
            particular = np.full(3, values[frozenset(range(3))] / 3)
            theta = np.linalg.lstsq(np.sqrt(w)[:, None] * (x @ basis), np.sqrt(w) * (y - x @ particular), rcond=None)[0]
            result = m.kernelshap_values(values, num_players=3, coalitions=sample)
            np.testing.assert_allclose(result.numpy(), particular + basis @ theta, atol=1e-12)
        elif oracle == 'rank_deficient':
            game = m.additive_game(torch.tensor([1., 2., 3.]))
            previous_threads = torch.get_num_threads()
            try:
                for threads in (24, 1, 2):
                    torch.set_num_threads(threads)
                    for _ in range(5):
                        estimate = m.kernelshap_values(game, num_players=3, coalitions=(frozenset({0}),))
                        torch.testing.assert_close(estimate, torch.tensor([1., 2.5, 2.5], dtype=torch.float64), atol=1e-12, rtol=0)
            finally:
                torch.set_num_threads(previous_threads)
        elif oracle == 'sampling':
            for budget in (12, 24, 62):
                for strategy in ('paired_kernel', 'uniform'):
                    sample = m.sample_interior_coalitions(6, budget=budget, seed=4, strategy=strategy)
                    assert len(sample) == len(set(sample)) == budget
                    assert sample == m.sample_interior_coalitions(6, budget=budget, seed=4, strategy=strategy)
                    if strategy == 'paired_kernel':
                        assert all((frozenset(range(6)) - c) in sample for c in sample)
            with pytest.raises(ValueError):
                m.sample_interior_coalitions(6, budget=13, seed=0, strategy='paired_kernel')
        elif oracle == 'owen':
            six = m.game_from_dividends(6, m.GAME_DIVIDENDS)
            for groups in (((0,1,2),(3,4,5)), ((0,3,4),(1,2,5)), ((0,1),(2,3),(4,5))):
                np.testing.assert_allclose(m.partition_shap_values(six, groups=groups).numpy(), owen_ordering_oracle(six, groups), atol=1e-12)
        elif oracle == 'partition_validation':
            for groups in (((0,1),(1,2)), ((0,1),), ((),(0,1,2)), ((0,1),(3,))):
                with pytest.raises(ValueError):
                    m.validate_partition(groups, 3)
        elif oracle == 'group_expansion':
            game = m.additive_game(torch.tensor([1.,2.,3.,4.]))
            result = m.grouped_coalition_values(game, groups=((0,1),(2,3)))
            assert result == {frozenset():0.,frozenset({0}):3.,frozenset({1}):7.,frozenset({0,1}):10.}
        elif oracle == 'shuffle_efficiency':
            interior = m.all_coalitions(3)[1:-1]
            shuffled = dict(values)
            for coalition, value in zip(interior, [values[c] for c in interior][1:] + [values[interior[0]]], strict=True):
                shuffled[coalition] = value
            estimate = m.kernelshap_values(shuffled, num_players=3)
            assert abs(float(estimate.sum()) - values[frozenset(range(3))]) < 1e-12
            assert float((estimate-expected).abs().max()) > .1
        elif oracle == 'live_play':
            for budget, seed, groups in itertools.product((12,24,62), (0,4), (((0,1,2),(3,4,5)),((0,3,4),(1,2,5)))):
                result = m.compare_kernelshap_play(dividends=m.GAME_DIVIDENDS, budget=budget, seed=seed, groups=groups)
                m.check_kernelshap_play(result)
                np.testing.assert_allclose(result['oracle'].numpy(), permutation_oracle(result['values'],6), atol=1e-12)
    finally:
        torch.set_default_dtype(previous)


def test_complete_fresh_cpu_execution_archives_runtime_seed_arrays_and_failed_start_are_retained():
    receipt = json.loads((EVIDENCE / 'fresh_cpu_execution.json').read_text())
    freeze = json.loads((EVIDENCE / 'fresh_cpu_freeze.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['hard_total_timeout_seconds'] == 120
    assert receipt['elapsed_seconds'] < 120
    assert receipt['code_cells_executed'] == 13
    assert receipt['training_used'] is receipt['pretrained_model_used'] is receipt['four_way_transfer_executed'] is False
    assert receipt['runtime']['cuda_initialized'] is False and receipt['runtime']['cuda_visible_devices'] == ''
    assert receipt['runtime']['threads'] == 2 and receipt['runtime']['dtype'] == 'torch.float32'
    assert sha((EVIDENCE / 'fresh_cpu_freeze.json').read_bytes()) == receipt['freeze_sha256']
    executed_bytes = gzip.decompress((EVIDENCE / 'fresh_cpu_executed.ipynb.gz').read_bytes())
    assert sha(executed_bytes) == receipt['notebook_sha256']
    current = codes(read('solutions'))
    assert [sha(c.source.encode()) for c in current] == receipt['source_sha256'] == freeze['code_sources_sha256']
    for ordinal, cell in enumerate(current):
        assert sha(json.dumps(cell.outputs, sort_keys=True).encode()) == receipt['cells'][ordinal]['outputs_sha256']
        assert cell.execution_count == ordinal + 1
    assert receipt['cells'][11]['scope'] == 'legacy_finite_contract_rerun'
    metrics = receipt['current_metrics']
    paired, uniform = np.array(metrics['sampling_errors']['paired_kernel']), np.array(metrics['sampling_errors']['uniform'])
    assert paired.shape == uniform.shape == (6, 50)
    np.testing.assert_allclose(paired.mean(1), metrics['paired_mean'], atol=1e-14)
    np.testing.assert_allclose(uniform.mean(1), metrics['uniform_mean'], atol=1e-14)
    assert math.isclose(float(paired[2].mean() - uniform[2].mean()), metrics['budget24_paired_minus_uniform'], abs_tol=1e-14)
    assert freeze['configuration']['sampler_seeds'] == list(range(50))
    assert freeze['configuration']['interaction_dividends'] == [[[0, 1, 2], 1.8], [[3, 4, 5], -.9]]
    assert freeze['configuration']['four_way_transfer_executed'] is False
    attempts = json.loads((EVIDENCE / 'execution_attempts.json').read_text())
    assert len(attempts['successful_complete_cpu_runs']) == 2
    assert math.isclose(sum(item['seconds'] for item in attempts['successful_complete_cpu_runs']), attempts['aggregate_successful_complete_cpu_wall_seconds'])
    for item in attempts['successful_complete_cpu_runs']:
        saved = json.loads((EVIDENCE / item['receipt']).read_text())
        assert saved['status'] == 'PASS'
        assert sha(gzip.decompress((EVIDENCE / item['executed_notebook']).read_bytes())) == saved['notebook_sha256']
    failure = json.loads((EVIDENCE / 'fresh_cpu_socket_blocked_failure.json').read_text())
    assert 'PermissionError' in failure['error']
    assert (EVIDENCE / 'fresh_cpu_socket_blocked.FAILED.ipynb.gz').exists()
    diagnosis = json.loads((EVIDENCE / 'rank_deficient_review_failure.json').read_text())
    assert diagnosis['suite_result_before_repair']['failed'] == 1
    assert diagnosis['regression_kept'].endswith('[rank_deficient]')
