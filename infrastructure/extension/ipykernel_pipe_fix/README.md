# arena-ipykernel-fix

A first-time multi-GB Hugging Face download inside a Jupyter kernel can hang the notebook.
On a clean cache this happens in 11.1 (Mistral-7B, 14.5 GB) and 13.1 (SD-1.5, 2.1 GB).

**Cause.** hf_xet reports download progress from Rust threads. On CPython ≥ 3.13, each
progress update from such a thread opens a zmq socket in ipykernel, and ipykernel never
closes it. After a few minutes the kernel reaches libzmq's limit of 1023 sockets ("Too many
open files"). IOPub dies and the cell never finishes.

**What this package does.** ipykernel already runs a sweep every 10 s that closes the
sockets of threads that have finished. This package extends that sweep to the sockets of
foreign threads whose Python thread state is gone. The module docstring in
`src/arena_ipykernel_fix/__init__.py` has the details.

`uv sync` installs the package and `.venv/etc/ipython/ipython_kernel_config.py`. Every
kernel started from `.venv` loads that file, whether VS Code, JupyterLab, Jupyter Notebook or
nbclient (`build.py --execute`) launched it. Nothing in the lessons changes.

- To check it is active, run this in a notebook:
  `import arena_ipykernel_fix; arena_ipykernel_fix.is_installed()`.
- For a kernel from another environment (Colab, conda, a system Jupyter), install the package
  there, `pip install ./infrastructure/extension/ipykernel_pipe_fix`, and restart the kernel.
  Or run `%load_ext arena_ipykernel_fix` in the notebook.
- To turn it off, set `ARENA_DISABLE_IPYKERNEL_PIPE_FIX=1` in the kernel's environment.
- `tests/test_ipykernel_pipe_fix.py` checks the fix in a real kernel. It also checks that
  stock ipykernel still leaks; once that check fails, upstream has fixed the bug and this
  package can go.
