"""Source-only review and fixture guards; actual6.4 import requires explicit opt-in."""
from pathlib import Path
import ast
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/mastery/evidence/transcoder_6_4_import_20261019'
SOURCE = ROOT/'chapter6_sparse_feature_methods/exercises/part4_transcoders/solutions.py'


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
        command += ['--authorize-target-import',
                    '--expected-source-sha256', os.environ['ARENA_TRANSCODER_EXPECTED_SOURCE_SHA256'],
                    '--expected-legacy-tests-sha256', os.environ['ARENA_TRANSCODER_EXPECTED_TESTS_SHA256']]
    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=40)


def test_guard_accepts_definition_only_fixture_without_target_import(tmp_path):
    result = run_guard(tmp_path, 'MAIN = False\nVALUE = 3\ndef untouched():\n    raise RuntimeError("must never call")\n')
    assert result.returncode == 0, result.stdout+result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['status'] == 'guarded_fixture_passed'
    assert receipt['before'] == receipt['after']
    assert not receipt['target_import_authorized'] and not receipt['method_or_study_called']


@pytest.mark.parametrize('statement,expected', [
    ('import torch\nx = torch.tensor([1.0])', 'torch.tensor'),
    ('import torch\nx = torch.Tensor([1.0])', 'torch.Tensor.__new__'),
    ('def study():\n    return 1\nstudy()', 'repository scientific/data method'),
    ('from pathlib import Path\nPath("forbidden.txt").write_text("x")', 'file write'),
    ('from pathlib import Path\nPath("data.json").read_text()', 'non-source/data/cache read'),
])
def test_guard_rejects_fixture_side_effects_before_target_import(tmp_path, statement, expected):
    result = run_guard(tmp_path, 'MAIN = False\n'+statement+'\n')
    assert result.returncode != 0
    assert expected in result.stderr, result.stdout+result.stderr


def test_actual_import_is_inert_under_hard_guards_only_when_authorized(tmp_path):
    if os.environ.get('ARENA_TRANSCODER_IMPORT_AUTHORIZED') != '1':
        pytest.skip('Actual6.4 import awaits production adoption or explicit root authorization')
    assert SOURCE.is_file(), 'Adopted canonical source must exist before actual import'
    result = run_guard(tmp_path, authorize=True)
    assert result.returncode == 0, result.stdout+result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['status'] == 'inert_target_import_passed'
    assert receipt['before'] == receipt['after'] and not receipt['guard_calls']
    assert not receipt['stdout'] and not receipt['stderr']
