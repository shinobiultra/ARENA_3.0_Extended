"""Canonical16.11 definition-time dependencies and one actual guarded inert import."""
from pathlib import Path
import ast
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'chapter16_shapley_attribution_baselines/exercises/part11_datamodels_cartography/solutions.py'
EVIDENCE = ROOT/'docs/mastery/evidence/datamodels_cartography_import_20261019'


def test_all_definition_time_defaults_bases_and_decorators_are_closed():
    tree = ast.parse(SOURCE.read_text())
    definitions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert definitions
    for node in definitions:
        # Every default here is literal: a guarded MAIN-only symbol must never resolve at import.
        for value in getattr(node, 'args', ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])).defaults:
            ast.literal_eval(value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for value in node.args.kw_defaults:
                if value is not None:
                    ast.literal_eval(value)
        assert not node.decorator_list, (node.name, 'new definition-time decorator needs audit')
        if isinstance(node, ast.ClassDef):
            assert not node.bases and not node.keywords, (node.name, 'new definition-time class dependency needs audit')


def test_actual_import_is_inert_under_hard_guards(tmp_path):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MPLBACKEND': 'Agg',
           'MPLCONFIGDIR': str(tmp_path/'mpl-cache')}
    result = subprocess.run([sys.executable, '-B', str(EVIDENCE/'guarded_import.py'), str(ROOT)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr
    receipt = json.loads(result.stdout)
    assert not result.stderr
    assert receipt['status'] == 'inert_actual_import_passed'
    assert receipt['before'] == receipt['after'] and receipt['files_and_source_caches_unchanged']
    assert not receipt['guard_calls'] and not receipt['captured_stdout'] and not receipt['captured_stderr']
