"""7.1 Logit lens, tuned lens and Patchscopes: CPU fixtures for the master method definitions, no model loads."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch as t

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = 'chapter7_activation_to_language'
PART = 'part1_lenses_patchscopes'
MASTER = 'master_7_1.py'
METHODS = ('train_tuned_lens', 'evaluate_lenses', 'shifted_translator_agreement', 'patchscope_controls')


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




def test_invalid_lens_denominators_are_rejected():
    from inspect import unwrap
    from unittest.mock import patch

    fixture = section_tests()._metric_fixture()
    model, lens, tokens, _, _, get_resid = fixture
    for name in ('evaluate_lenses', 'shifted_translator_agreement'):
        fn = method(name)
        with patch.dict(unwrap(fn).__globals__, get_resid_and_logprobs=get_resid):
            for bad_tokens, batch_size in ((tokens[:0], 2), (tokens[:, :1], 2), (tokens, 0), (tokens, -1)):
                with pytest.raises(ValueError, match='positive batch size'):
                    fn(model, lens, bad_tokens, batch_size=batch_size, **({'shift': 0} if 'shifted' in name else {}))


def test_training_feedback_rejects_accumulated_gradient_mutation():
    """The learner check must fail if zero_grad is accidentally omitted."""
    import copy

    node = copy.deepcopy(method_ast('train_tuned_lens'))

    class OmitZeroGrad(ast.NodeTransformer):
        removed = 0

        def visit_Expr(self, expression):
            if (isinstance(expression.value, ast.Call)
                    and isinstance(expression.value.func, ast.Attribute)
                    and expression.value.func.attr == 'zero_grad'):
                self.removed += 1
                return None
            return self.generic_visit(expression)

    mutation = OmitZeroGrad()
    node = ast.fix_missing_locations(mutation.visit(node))
    assert mutation.removed == 1
    namespace = {'t': t, 'tqdm': lambda iterable, **kwargs: iterable}
    source = ast.unparse(ast.Module(body=[node], type_ignores=[]))
    exec('from __future__ import annotations\n' + source, namespace)
    with pytest.raises(AssertionError, match='Clear gradients'):
        section_tests().test_train_tuned_lens(namespace['train_tuned_lens'])
