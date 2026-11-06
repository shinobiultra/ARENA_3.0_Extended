"""5.2 Mixture of experts: the lesson own immediate CPU oracles, run one by one on the solutions."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / 'chapter5_modern_architectures/exercises/part2_mixture_of_experts'
MODULE = 'chapter5_modern_architectures.exercises.part2_mixture_of_experts'



# Exclude historical trainability/full-publication tests: these are methods-only checks.
ORACLES=tuple(n.name for n in ast.parse((SECTION/'tests.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))


@pytest.mark.parametrize('name',ORACLES)
def test_existing_independent_cpu_semantic_oracle(name):
    import torch
    previous_threads=torch.get_num_threads();previous_dtype=torch.get_default_dtype()
    try:
        torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(devices=[]):getattr(importlib.import_module(MODULE+'.tests'),name)()
    finally:
        torch.set_default_dtype(previous_dtype);torch.set_num_threads(previous_threads)
