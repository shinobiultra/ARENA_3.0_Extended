"""Bounded CPU sensitivity checks for the taught noising and IRIS callback tests."""
from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as t
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NOISING = ROOT / 'chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching'
IRIS = ROOT / 'chapter14_jepa_world_models/exercises/part7_recurrent_world_models'


def _test_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path / 'tests.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _methods(path: Path, names: tuple[str, ...]):
    """Execute only selected definitions, never setup, model loading or main cells."""
    tree = ast.parse((path / 'solutions.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = {'t': t, 'Tensor': t.Tensor, 'F': F, 'np': np, 'math': math,
                 'N_DIFFUSION_STEPS': 8, 'DiffusionSchedule': lambda **kw: SimpleNamespace(**kw)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path / 'solutions.py'), 'exec'), namespace)
    return namespace


def _no_reference_import():
    raise AssertionError('Explicit callbacks must not import a reference implementation.')


def test_noising_callbacks_pass_without_reference_import():
    tests = _test_module(NOISING, 'generation_noising_fixture')
    tests._solutions = _no_reference_import
    methods = _methods(NOISING, ('make_diffusion_schedule', 'q_sample'))
    tests.test_diffusion_schedule_has_exact_endpoint_and_valid_noise(
        methods['make_diffusion_schedule'], methods['q_sample'])


@pytest.mark.parametrize('mutation', ['identity', 'ignore_noise', 'share_first_time', 'no_square_roots'])
def test_noising_fixture_rejects_wrong_forward_process(mutation):
    tests = _test_module(NOISING, 'generation_noising_mutant_fixture')
    tests._solutions = _no_reference_import
    methods = _methods(NOISING, ('make_diffusion_schedule', 'q_sample'))

    def mutant(clean, timesteps, noise, schedule):
        if mutation == 'identity':
            return clean.clone()
        times = timesteps[:1].expand_as(timesteps) if mutation == 'share_first_time' else timesteps
        signal = schedule.alpha_bars[times].reshape(-1, 1, 1, 1)
        if mutation == 'ignore_noise':
            return signal.sqrt() * clean
        if mutation == 'no_square_roots':
            return signal * clean + (1 - signal) * noise
        return signal.sqrt() * clean + (1 - signal).sqrt() * noise

    with pytest.raises(AssertionError):
        tests.test_diffusion_schedule_has_exact_endpoint_and_valid_noise(
            methods['make_diffusion_schedule'], mutant)


def test_iris_nonzero_fixture_passes_existing_predictor():
    tests = _test_module(IRIS, 'generation_iris_fixture')
    methods = _methods(IRIS, ('causal_attention', 'iris_logits'))
    tests.test_iris_logits(methods['iris_logits'])


@pytest.mark.parametrize('mutation', ['no_branches', 'no_attention', 'no_mlp', 'bidirectional'])
def test_iris_fixture_rejects_missing_or_noncausal_computation(mutation):
    tests = _test_module(IRIS, 'generation_iris_mutant_fixture')
    methods = _methods(IRIS, ('causal_attention',))

    def mutant(model, tokens):
        x = model['embedding'](tokens) + model['position'](t.arange(tokens.shape[1]))[None]
        if mutation in ('no_branches', 'no_attention'):
            if mutation == 'no_attention':
                x = x + model['mlp'](model['norm'](x))
            return model['readout'](x)
        if mutation == 'bidirectional':
            q, k, v = (model[name](x) for name in ('query', 'key', 'value'))
            x = x + (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1) @ v
        else:
            x = x + methods['causal_attention'](model['query'](x), model['key'](x), model['value'](x))
        if mutation != 'no_mlp':
            x = x + model['mlp'](model['norm'](x))
        return model['readout'](x)

    with pytest.raises(AssertionError):
        tests.test_iris_logits(mutant)
