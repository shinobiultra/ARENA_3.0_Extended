# Installed into <venv>/etc/ipython/ by the arena-ipykernel-fix package
# (ARENA_3.0_Extended: infrastructure/extension/ipykernel_pipe_fix).
#
# Every IPython kernel started from this venv reads this file, whether it was launched by
# VS Code, JupyterLab, Jupyter Notebook or nbclient. It makes ipykernel close the zmq event
# pipes of finished foreign threads, such as the hf_xet threads that report Hugging Face
# download progress. Without it, a long first-time download exhausts the kernel's sockets and
# the notebook hangs. See arena_ipykernel_fix.__doc__.
# Opt out with ARENA_DISABLE_IPYKERNEL_PIPE_FIX=1.
try:
    import arena_ipykernel_fix
except ImportError:
    pass
else:
    arena_ipykernel_fix.install_from_kernel_config()
