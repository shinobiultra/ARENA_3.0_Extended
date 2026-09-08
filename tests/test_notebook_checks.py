import ast

import pytest

from arena_ext.notebook_checks import unfinished_definitions


@pytest.mark.parametrize("body", ["pass", "...", '"Only a docstring"', "raise NotImplementedError", "raise NotImplementedError()", "raise builtins.NotImplementedError('finish me')"])
def test_detects_unfinished_body(body):
    assert unfinished_definitions(f"def learner(x):\n    {body}\n") == ["learner"]


def test_checks_methods_and_async_functions():
    tree = ast.parse("class Encoder:\n    def forward(self, x): ...\nasync def fetch(): pass\n")
    assert set(unfinished_definitions(tree)) == {"forward", "fetch"}


def test_ignores_prose_and_valid_error_handling():
    code = '''
def completed(x):
    """Replace the old NotImplementedError when solving this exercise."""
    if x < 0:
        raise ValueError("x must be nonnegative")
    return x + 1
'''
    assert unfinished_definitions(code) == []
