"""Lazy bridge to unchanged ROME tests. Import never loads a model or runs a test."""
from importlib import import_module

def __getattr__(name):
    if not name.startswith("test_") and not name.startswith("check_"):
        raise AttributeError(name)
    def run_legacy_test(*args, **kwargs):
        original = import_module("chapter9_alignment_interpretability.exercises.part5_rome_model_editing.tests")
        return getattr(original, name)(*args, **kwargs)
    return run_legacy_test
