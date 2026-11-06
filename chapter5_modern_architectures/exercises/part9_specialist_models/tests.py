"""Explicit lazy bridge to unchanged historical tests; no test or model runs at import."""
from importlib import import_module


def __getattr__(name):
    if not name.startswith("test_"):
        raise AttributeError(name)

    def run_legacy_test(*args, **kwargs):
        original=import_module("chapter5_modern_architectures.exercises.part6_multimodal_embedding_function_models.tests")
        return getattr(original,name)(*args,**kwargs)

    return run_legacy_test
