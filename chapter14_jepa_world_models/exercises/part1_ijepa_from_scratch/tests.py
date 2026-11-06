"""Lazy bridge to unchanged legacy 14.2 tests; every learner argument is forwarded.

The original test module contains historical path assertions and reference defaults.
It is loaded only when an explicitly requested test attribute is called.
"""
from importlib import import_module


def __getattr__(name):
    if not name.startswith("test_"):
        raise AttributeError(name)

    def run_legacy_test(*args, **kwargs):
        legacy = import_module("part2_ijepa_from_scratch.tests")
        return getattr(legacy, name)(*args, **kwargs)

    return run_legacy_test
