"""Stop Jupyter kernels from wedging during long Hugging Face downloads.

The bug
-------
ipykernel's ``IOPubThread`` gives every thread that writes output its own inproc zmq PUSH
socket (the "event pipe"). It keeps the socket in a ``threading.local`` and in the dict
``IOPubThread._event_pipes``, keyed by ``threading.current_thread()``. Every 10 s
``IOPubThread._event_pipe_gc`` closes the sockets whose thread reports ``is_alive() == False``.

Threads that Python did not start get a new ``PyThreadState`` every time they call into
Python. hf_xet's Rust download threads are an example: they report progress through
huggingface_hub's tqdm bars. On CPython >= 3.13 (gh-114423) each of those thread states gets
a fresh ``threading._DummyThread``. CPython drops it from ``threading.enumerate()`` when the
thread state ends, and the thread-local that held the socket goes away with it. A
``_DummyThread`` never reports ``is_alive() == False``, so ipykernel never closes its socket.

Each socket holds a file descriptor. A download leaks a few sockets per second (2.5 to 4 in
our measurements). After a few minutes libzmq refuses socket 1024 ("Too many open files"),
IOPub dies, and the notebook hangs. Mistral-7B (11.1) and SD-1.5 (13.1) hit this on a clean
cache.

The fix
-------
Before ipykernel's own sweep, close the pipes whose key is a ``_DummyThread`` that is no
longer registered (not in ``threading.enumerate()``). That thread state is gone, together
with the thread-local that held the socket, so nothing can use the socket again. Pipes of
live threads, whether started by Python or not, are left alone. On CPython 3.12, dummy
threads stay registered forever, so this is a no-op there; 3.12 doesn't leak.

How it is loaded
----------------
``uv sync`` installs this package and ``<venv>/etc/ipython/ipython_kernel_config.py``. Every
IPython kernel started from that venv reads that file, whether it was launched by VS Code,
JupyterLab, Jupyter Notebook or nbclient. In other environments, run
``%load_ext arena_ipykernel_fix`` in the notebook. Set ``ARENA_DISABLE_IPYKERNEL_PIPE_FIX=1``
in the kernel's environment to turn the kernel-config hook off.

Delete this package once ipykernel closes these pipes itself. ``tests/test_ipykernel_pipe_fix.py``
has a control that fails when the stock kernel stops leaking.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
from typing import Any

__all__ = [
    "close_stale_dummy_thread_pipes",
    "install",
    "install_from_kernel_config",
    "is_installed",
    "load_ipython_extension",
]

DISABLE_ENV = "ARENA_DISABLE_IPYKERNEL_PIPE_FIX"
_MARKER = "_arena_closes_stale_dummy_thread_pipes"


def close_stale_dummy_thread_pipes(iopub_thread: Any) -> int:
    """Close and forget the event pipes of foreign thread states that no longer exist.

    ``iopub_thread`` is an ``ipykernel.iostream.IOPubThread``. Returns the number of pipes
    closed.
    """
    pipes = getattr(iopub_thread, "_event_pipes", None)
    lock = getattr(iopub_thread, "_event_pipe_gc_lock", None)
    if not pipes or lock is None:
        return 0
    with lock:
        # Snapshot under the lock: ipykernel registers a thread (current_thread()) before it
        # inserts that thread's pipe under this same lock, so every pipe owned by a live thread
        # state has its thread in this snapshot.
        registered = set(threading.enumerate())
        stale = [
            thread
            for thread in pipes
            if isinstance(thread, threading._DummyThread) and thread not in registered
        ]
        for thread in stale:
            pipes.pop(thread).close()
    return len(stale)


def is_installed() -> bool:
    """True if ``IOPubThread._event_pipe_gc`` already closes stale dummy-thread pipes."""
    try:
        from ipykernel.iostream import IOPubThread
    except ImportError:
        return False
    return getattr(IOPubThread._event_pipe_gc, _MARKER, False)


def install() -> bool:
    """Make ipykernel's periodic event-pipe GC also close stale dummy-thread pipes.

    Patches the ``IOPubThread`` class, so it also covers an IOPub thread that is already
    running. The running GC loop looks the method up on every pass. Returns False and
    changes nothing if ipykernel is missing or its internals differ from 7.x.
    """
    try:
        from ipykernel.iostream import IOPubThread
    except ImportError:
        return False
    original = getattr(IOPubThread, "_event_pipe_gc", None)
    if original is None or not inspect.iscoroutinefunction(original):
        return False
    if getattr(original, _MARKER, False):
        return True

    @functools.wraps(original)
    async def _event_pipe_gc(self):
        close_stale_dummy_thread_pipes(self)
        await original(self)

    setattr(_event_pipe_gc, _MARKER, True)
    IOPubThread._event_pipe_gc = _event_pipe_gc  # ty: ignore[invalid-assignment]
    return True


def install_from_kernel_config() -> bool:
    """Entry point for ``<venv>/etc/ipython/ipython_kernel_config.py``. Honors the opt-out."""
    if os.environ.get(DISABLE_ENV) == "1":
        return False
    return install()


def load_ipython_extension(ipython: Any) -> None:
    """``%load_ext arena_ipykernel_fix``, for kernels that do not read the venv config."""
    install()
