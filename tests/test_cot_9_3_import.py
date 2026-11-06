"""Fixture-only guard review; actual9.3 import requires reviewed explicit opt-in."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/mastery/evidence/cot_9_3_import_20261019'
SOURCE = ROOT/'chapter9_alignment_interpretability/exercises/part3_cot_faithfulness/solutions.py'
SOURCE_SHA256 = '260544bc66ddba46bb92744df903ae09bf8f2f5875f9132f681c7651bfe0822f'
LEGACY_TESTS_SHA256 = '7eea323a3e543c37af10553c2984e0fc779df867d2d533b9edfb3bbbed0428ec'


def run_guard(tmp_path, fixture=None, authorize=False):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MPLBACKEND': 'Agg',
           'MPLCONFIGDIR': str(tmp_path/'mpl-cache')}
    command = [sys.executable, '-B', str(EVIDENCE/'guarded_import.py'), '--root', str(ROOT)]
    if fixture is not None:
        source = tmp_path/'guard_fixture.py'
        source.write_text(fixture)
        command += ['--fixture-source', str(source)]
    if authorize:
        command += ['--authorize-target-import', '--expected-source-sha256', SOURCE_SHA256,
                    '--expected-legacy-tests-sha256', LEGACY_TESTS_SHA256]
    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=40)


def test_source_bound_definition_time_contract_without_import():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
    tree = ast.parse(raw)
    assert any(isinstance(n, ast.ImportFrom) and n.module == '__future__'
               and any(alias.name == 'annotations' for alias in n.names) for n in tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults+[x for x in node.args.kw_defaults if x is not None]:
                ast.literal_eval(default)
        if isinstance(node, ast.ClassDef):
            assert not node.bases and not node.keywords
            assert all(ast.dump(decorator) == ast.dump(ast.parse('dataclass(frozen=True)', mode='eval').body)
                       for decorator in node.decorator_list)


def test_guard_accepts_definition_only_fixture_without_target_import(tmp_path):
    result = run_guard(tmp_path, 'MAIN = False\nVALUE = 3\ndef untouched():\n    raise RuntimeError("must never call")\n')
    assert result.returncode == 0, result.stdout+result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['status'] == 'guarded_fixture_passed'
    assert receipt['before'] == receipt['after']
    assert not receipt['target_import_authorized'] and not receipt['method_or_study_called']
    assert receipt['lazy_historical_helper_unloaded']


@pytest.mark.parametrize('statement,expected', [
    ('import torch\nx = torch.tensor([1.0])', 'torch.tensor'),
    ('import torch\nx = torch.Tensor([1.0])', 'torch.Tensor.__new__'),
    ('import numpy\nx = numpy.ndarray((2, 2))', 'numpy.ndarray'),
    ('def study():\n    return 1\nstudy()', 'repository scientific/data method'),
    ('from pathlib import Path\nPath("forbidden.txt").write_text("x")', 'file write'),
    ('from pathlib import Path\nPath("data.json").read_text()', 'non-source/data/cache read'),
    ('import socket\nsocket.create_connection(("example.com", 80))', 'network'),
    ('import torch\nx = torch.nn.Linear(2, 2)', 'model construction/forward'),
])
def test_guard_rejects_fixture_side_effects_without_target_import(tmp_path, statement, expected):
    result = run_guard(tmp_path, 'MAIN = False\n'+statement+'\n')
    assert result.returncode != 0
    assert expected in result.stderr, result.stdout+result.stderr


def test_actual_import_is_inert_only_after_explicit_root_authorization(tmp_path):
    if os.environ.get('ARENA_COT_9_3_IMPORT_AUTHORIZED') != '1':
        pytest.skip('Actual9.3 import awaits independent harness GO and explicit root authorization')
    result = run_guard(tmp_path, authorize=True)
    assert result.returncode == 0, result.stdout+result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['status'] == 'inert_target_import_passed'
    assert receipt['before'] == receipt['after'] and not receipt['guard_calls']
    assert not receipt['stdout'] and not receipt['stderr']
    assert receipt['legacy_tests_actual_origin'] == str(
        ROOT/'chapter9_alignment_interpretability/exercises/part2_cot_faithfulness/tests.py')
    assert receipt['legacy_tests_imported_separately_under_guards']
    assert receipt['lazy_historical_helper_unloaded']
