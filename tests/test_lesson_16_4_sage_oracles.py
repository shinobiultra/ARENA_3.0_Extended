"""16.4 SAGE global importance: tiny CPU oracles run against the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = "chapter16_shapley_attribution_baselines.exercises.part4_sage"


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module(MODULE + ".solutions")


ORACLES = ("coalition_masks", "marginal_predict", "loss_game", "exact_shapley", "gaussian_game", "path_contributions", "permutation_sage", "summarize_paths", "prefix_values")


@pytest.mark.parametrize("name", ORACLES)
def test_visible_generated_methods_against_independent_cpu_oracles(methods, name):
    checks = importlib.import_module(MODULE + ".tests")
    torch = importlib.import_module("torch")
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        getattr(checks, "test_" + name)(getattr(methods, name))
    finally:
        torch.set_default_dtype(old_dtype)
