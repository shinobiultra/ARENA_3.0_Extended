"""Bounded CPU mutation checks for the public 8.8 scoring exercise tests.

No compiled transformer construction, model downloads, or native-model claims.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch as t

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "chapter8_automated_circuits/exercises/part8_circuit_benchmarks/tests.py"
MASTER = ROOT / "infrastructure/chapters/chapter8_automated_circuits/master_8_8.py"


def _public_tests():
    spec = importlib.util.spec_from_file_location("circuit_benchmark_public_tests", TESTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _learner_functions():
    # Extract only visible student method definitions: importing the whole lesson
    # would import model-loading libraries and its demonstration setup.
    tree = ast.parse(MASTER.read_text())
    names = {"patching_scores", "eap_scores", "eap_ig_scores"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"t": t}
    exec(compile(module, str(MASTER), "exec"), namespace)
    return [namespace[name] for name in ("patching_scores", "eap_scores", "eap_ig_scores")]


def _run(patching, eap, ig):
    public = _public_tests()
    graph = public._AnalyticEdgeGraph()
    clean = t.zeros((1, 1), dtype=t.long)
    public.test_edge_methods(patching, eap, ig, graph, clean, clean, lambda logits: logits.sum())


def test_visible_methods_pass_independent_cpu_oracles():
    _run(*_learner_functions())


@pytest.mark.parametrize("mutation", ["all_zero", "permute"])
def test_wrong_exact_scores_are_rejected(mutation):
    patching, eap, ig = _learner_functions()

    def mutant(graph, tokens, corrupt_out, metric):
        scores = patching(graph, tokens, corrupt_out, metric)
        if mutation == "all_zero":
            return t.zeros_like(scores)
        output = scores.clone()
        output[graph.edge_mask] = scores[graph.edge_mask].roll(1)
        return output

    with pytest.raises(AssertionError, match="exact full-minus-one"):
        _run(mutant, eap, ig)


def test_wrong_eap_endpoint_is_rejected():
    patching, _, ig = _learner_functions()
    # Correct path allocation is not the endpoint derivative on this graph.
    with pytest.raises(AssertionError, match="endpoint EAP"):
        _run(patching, patching, ig)


def test_complete_but_edgewise_wrong_ig_is_rejected():
    patching, eap, ig = _learner_functions()

    def mutant(graph, tokens, corrupt_out, metric, steps=8):
        scores = ig(graph, tokens, corrupt_out, metric, steps=steps)
        output = scores.clone()
        output[graph.edge_mask] = scores[graph.edge_mask].roll(1)
        t.testing.assert_close(output.sum(), scores.sum())
        return output

    with pytest.raises(AssertionError, match="edgewise IG"):
        _run(patching, eap, mutant)


def test_independent_edge_removals_are_not_accepted_as_ig():
    patching, eap, _ = _learner_functions()

    def mutant(graph, tokens, corrupt_out, metric, steps=8):
        return patching(graph, tokens, corrupt_out, metric)

    with pytest.raises(AssertionError, match="edgewise IG"):
        _run(patching, eap, mutant)
