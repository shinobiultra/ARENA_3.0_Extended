"""8.4 ACDC circuit metrics: CPU fixtures for the master method definitions, no model loads."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch as t

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = 'chapter8_automated_circuits'
PART = 'part4_acdc_circuit_metrics'
MASTER = 'master_8_4.py'
METHODS = ('single_edge_effects', 'attention_by_offset', 'top_k_circuit', 'completeness_scores', 'edge_drops')


def section_tests():
    path = ROOT / CHAPTER / 'exercises' / PART / 'tests.py'
    spec = importlib.util.spec_from_file_location('lesson_tests_' + PART, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method_ast(name):
    path = ROOT / 'infrastructure/chapters' / CHAPTER / MASTER
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name)


def method(name):
    # Future annotations avoid needing model classes. Decorators still execute normally.
    namespace = {'t': t, 'tqdm': lambda iterable, **kwargs: iterable}
    source = ast.unparse(ast.Module(body=[method_ast(name)], type_ignores=[]))
    exec(compile('from __future__ import annotations\n' + source, MASTER, 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('name', METHODS)
def test_independent_method_fixture(name):
    getattr(section_tests(), 'test_' + name)(method(name))
