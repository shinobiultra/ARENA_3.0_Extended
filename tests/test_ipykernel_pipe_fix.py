"""Kernels must not leak zmq sockets for output from threads Python did not start.

hf_xet's Rust threads write Hugging Face download progress into the notebook, and each update
gets a fresh Python thread state. On CPython >= 3.13, stock ipykernel keeps one zmq socket for
every such thread state, forever. A first-time multi-GB download (11.1 Mistral-7B, 13.1
SD-1.5) therefore reaches libzmq's 1023-socket limit, and the kernel hangs.
``arena_ipykernel_fix`` (infrastructure/extension/ipykernel_pipe_fix) closes those sockets.

The foreign threads here are ctypes ``pthread_create`` threads that write to stdout. The
mechanism is the same as hf_xet's, without the network or hf_xet. The kernel tests start the
venv's ``python3`` kernel through jupyter_client, as JupyterLab and ``build.py --execute`` do;
VS Code also runs ``<venv>/bin/python -m ipykernel_launcher``. So they also check that
``uv sync`` installed ``<venv>/etc/ipython/ipython_kernel_config.py`` and that kernels load it.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import os
import sys
import threading
import time

import pytest

zmq = pytest.importorskip("zmq")
iostream = pytest.importorskip("ipykernel.iostream")
psutil = pytest.importorskip("psutil")
jupyter_client_manager = pytest.importorskip("jupyter_client.manager")

if os.name != "posix":
    pytest.skip("foreign threads are made with pthread_create", allow_module_level=True)

# Not importorskip: an environment with ipykernel but without this package (from the lock) is
# exactly the broken setup these tests exist to catch.
import arena_ipykernel_fix

FOREIGN_THREADS = 300  # each one leaks one socket (= one fd) in a stock kernel
GC_WAIT_SECONDS = 25  # ipykernel sweeps event pipes every 10 s

_START_ROUTINE = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
_libc = ctypes.CDLL(ctypes.util.find_library("c"))


class ForeignThread:
    """Run ``fn`` on a pthread that Python did not start (it sees a ``_DummyThread``)."""

    def __init__(self, fn):
        def start_routine(_arg):
            fn()

        self._start_routine = _START_ROUTINE(start_routine)  # keep alive until joined
        self._tid = ctypes.c_void_p()
        assert _libc.pthread_create(ctypes.byref(self._tid), None, self._start_routine, None) == 0

    def join(self):
        assert _libc.pthread_join(self._tid, None) == 0


@pytest.fixture
def iopub_thread():
    context = zmq.Context()
    thread = iostream.IOPubThread(context.socket(zmq.PUB))
    thread.start()
    yield thread
    gc_task = thread._event_pipe_gc_task
    if gc_task is not None:  # finish cancelling it on its own loop, or asyncio warns at exit

        async def cancel_gc():
            gc_task.cancel()
            await asyncio.gather(gc_task, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(cancel_gc(), gc_task.get_loop()).result(10)
    thread.stop()
    thread.close()
    context.term()


def _schedule_noop(iopub_thread):
    """What an ``OutStream`` flush does: wake the IOPub thread via this thread's event pipe."""
    iopub_thread.schedule(lambda: None)


def test_sweep_closes_only_pipes_of_finished_foreign_threads(iopub_thread):
    pipes = iopub_thread._event_pipes

    for _ in range(3):
        ForeignThread(lambda: _schedule_noop(iopub_thread)).join()
    finished = [thread for thread in pipes if isinstance(thread, threading._DummyThread)]
    assert len(finished) == 3

    release, scheduled = threading.Event(), threading.Event()

    def schedule_then_wait():
        _schedule_noop(iopub_thread)
        scheduled.set()
        release.wait()

    running_foreign = ForeignThread(schedule_then_wait)
    assert scheduled.wait(10)
    scheduled.clear()
    running_python = threading.Thread(target=schedule_then_wait)
    running_python.start()
    assert scheduled.wait(10)
    _schedule_noop(iopub_thread)  # the main thread

    sockets = {thread: pipes[thread] for thread in finished}
    assert arena_ipykernel_fix.close_stale_dummy_thread_pipes(iopub_thread) == 3
    assert all(socket.closed for socket in sockets.values())
    assert not set(finished) & set(pipes)
    assert running_python in pipes and threading.main_thread() in pipes
    assert [t for t in pipes if isinstance(t, threading._DummyThread)], "live foreign thread kept"

    release.set()
    running_foreign.join()
    running_python.join()
    # The foreign thread's state is gone now. The finished Python thread is ipykernel's own job.
    assert arena_ipykernel_fix.close_stale_dummy_thread_pipes(iopub_thread) == 1
    assert running_python in pipes


def test_install_wraps_ipykernel_gc_once(iopub_thread, monkeypatch):
    monkeypatch.setattr(iostream.IOPubThread, "_event_pipe_gc", iostream.IOPubThread._event_pipe_gc)
    assert arena_ipykernel_fix.install() and arena_ipykernel_fix.install()
    assert arena_ipykernel_fix.is_installed()
    assert (
        iostream.IOPubThread._event_pipe_gc.__wrapped__.__qualname__ == "IOPubThread._event_pipe_gc"
    )

    ForeignThread(lambda: _schedule_noop(iopub_thread)).join()
    assert any(isinstance(thread, threading._DummyThread) for thread in iopub_thread._event_pipes)
    asyncio.run(iopub_thread._event_pipe_gc())  # one pass of the GC the IOPub thread runs
    assert not any(
        isinstance(thread, threading._DummyThread) for thread in iopub_thread._event_pipes
    )


FOREIGN_THREAD_CELL = f"""
import ctypes, ctypes.util, sys
_libc = ctypes.CDLL(ctypes.util.find_library("c"))

@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
def _foreign(_arg):
    sys.stdout.write(".")
    sys.stdout.flush()
    return None

for _ in range({FOREIGN_THREADS}):
    _tid = ctypes.c_void_p()
    assert _libc.pthread_create(ctypes.byref(_tid), None, _foreign, None) == 0
    assert _libc.pthread_join(_tid, None) == 0
"""


def _kernel_fd_growth(tmp_path, **env) -> tuple[bool, int]:
    """Start the venv's python3 kernel, write from foreign threads, return (fix active, fd growth)."""
    km, kc = jupyter_client_manager.start_new_kernel(
        kernel_name="python3", cwd=str(tmp_path), env={**os.environ, **env}
    )
    try:
        kernel = psutil.Process(km.provisioner.process.pid)

        def execute(code):
            reply = kc.execute_interactive(code, timeout=120, output_hook=lambda msg: None)
            assert reply["content"]["status"] == "ok", reply["content"]

        execute("import arena_ipykernel_fix, sys; sys.stdout.write('warm'); sys.stdout.flush()")
        reply = kc.execute_interactive(
            "",
            user_expressions={"fix": "arena_ipykernel_fix.is_installed()"},
            timeout=30,
            output_hook=lambda msg: None,
        )
        fix_active = reply["content"]["user_expressions"]["fix"]["data"]["text/plain"] == "True"
        baseline = kernel.num_fds()
        execute(FOREIGN_THREAD_CELL)
        growth = kernel.num_fds() - baseline
        deadline = time.monotonic() + GC_WAIT_SECONDS
        while fix_active and growth > 10 and time.monotonic() < deadline:
            time.sleep(1)
            growth = kernel.num_fds() - baseline
        if not fix_active:
            time.sleep(12)  # give stock ipykernel's own sweep a chance
            growth = kernel.num_fds() - baseline
        return fix_active, growth
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


def test_venv_kernel_does_not_leak_foreign_thread_pipes(tmp_path):
    fix_active, growth = _kernel_fd_growth(tmp_path)
    assert fix_active, (
        f"the python3 kernel of {sys.prefix} did not load etc/ipython/ipython_kernel_config.py; "
        "run `uv sync` so arena-ipykernel-fix installs it"
    )
    assert growth <= 10, f"{growth} fds still open after {FOREIGN_THREADS} foreign-thread writes"


def test_stock_kernel_still_leaks_foreign_thread_pipes(tmp_path):
    """Control: with the fix off, the same cell leaks one fd per foreign thread.

    If this fails, ipykernel fixed the leak upstream: remove arena-ipykernel-fix and these tests.
    """
    fix_active, growth = _kernel_fd_growth(tmp_path, **{arena_ipykernel_fix.DISABLE_ENV: "1"})
    assert not fix_active
    assert growth >= 0.9 * FOREIGN_THREADS, (
        f"stock ipykernel leaked only {growth} fds for {FOREIGN_THREADS} foreign-thread writes: "
        "upstream may have fixed it; drop infrastructure/extension/ipykernel_pipe_fix"
    )
