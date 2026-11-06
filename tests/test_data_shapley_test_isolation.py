"""Collection must not leak a teaching notebook's process-wide Torch setup."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "dtype,modules",
    [
        ("float32", ["test_data_shapley.py", "test_inrun_data_shapley.py"]),
        ("float64", ["test_inrun_data_shapley.py", "test_data_shapley.py"]),
    ],
)
def test_collection_and_test_execution_preserve_caller_torch_state(dtype, modules):
    # Fresh processes exercise both actual first-import paths. The probe checks
    # collection, every mathematical test's intended precision, and teardown.
    script = r'''
import sys
import pytest
import torch

caller_dtype = getattr(torch, sys.argv[1])
caller_threads = 2
torch.set_default_dtype(caller_dtype)
torch.set_num_threads(caller_threads)

def caller_state():
    assert torch.get_default_dtype() == caller_dtype
    assert torch.get_num_threads() == caller_threads

class Probe:
    def pytest_collection_finish(self, session):
        caller_state()

    def pytest_runtest_call(self, item):
        assert torch.get_default_dtype() == torch.float64
        assert torch.get_num_threads() == 1

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        yield
        caller_state()

result = pytest.main(
    ["-q", "-p", "no:cacheprovider", *sys.argv[2:]], plugins=[Probe()]
)
caller_state()
raise SystemExit(int(result))
'''
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, dtype, *(str(ROOT / "tests" / name) for name in modules)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
