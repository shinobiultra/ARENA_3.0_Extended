# Install the study reader

Use a reviewed copy of this repository. The current working checkout contains
unpublished materials; its `main` Colab links can precede the local lessons. A
published v1.0 must identify its exact release snapshot. These instructions do
not publish it or turn historical model results into fresh execution.

## A separate environment for studying

From the repository root, with [uv](https://docs.astral.sh/uv/getting-started/installation/)
available:

```bash
uv venv --python 3.14 .venv-study
uv pip install --python .venv-study/bin/python -r mastery_hub/requirements-reader.txt
.venv-study/bin/python -m streamlit run mastery_hub/read.py --server.address=127.0.0.1 --server.port=18505 --server.headless=true --browser.gatherUsageStats=false
```

Open [the reader](http://127.0.0.1:18505/). This Linux profile installs the reader,
optimizer, Jupyter and basic NumPy/plotting tools. It requires no account, model
weights, CUDA installation, root access or cloud service. The dependency download
happens during installation; reading and the quadratic optimizer then work locally.
It leaves the existing `.venv`, root `uv.lock` and historical model environment
unchanged.

For the optional optimizer, use the same environment on a different port:

```bash
.venv-study/bin/python -m streamlit run mastery_hub/app.py --server.address=127.0.0.1 --server.port=18506 --server.headless=true --browser.gatherUsageStats=false
```

## Work in a notebook

Choose one lesson. When offered, **Study Practice** includes the reviewed teaching
corrections. Save the notebook beside the original exercise in the exact directory
shown by the reader. Its imports, data and assets depend on that location. Original
notebooks and previous blank copies remain separate downloads.

Expand **Open your notebook in Jupyter**. Its command uses the interpreter actually
running the reader. Register that named kernel, open Jupyter, and confirm it with
`import sys; print(sys.executable)` before running setup. Keep your personal filled
copy; generators refuse to overwrite saved work.

The small reader profile is not the full scientific environment. Many notebooks
also require PyTorch or specialized packages, weights, data, hardware or access
approval. Follow their stated prerequisites. In the revised 5.3/5.4/5.8/6.2
Practice routes, optional native bridge calls are disabled initially while the
CPU numerical exercises remain available. Enabling a bridge requires its actual
dependencies and resource allowance. Preserved references retain their original
setup and historical outputs.

## Existing full checkout

If this checkout already has its qualified full environment, retain the documented
`uv run --no-sync --offline` launch commands. They deliberately reuse that environment;
they do not install missing dependencies. The current `install.sh` is a separate
RunPod/VastAI-oriented **full CUDA extension installer** and can install system
packages. It is not needed for the reader profile above.

Original ARENA's dependency snapshot is `requirements-original.txt`; the legacy RL
stack is separately recorded in `requirements-legacy-rl.txt`. Native reproduction
uses the relevant recorded protocol and runtime, rather than treating any
same-version package installation as identical execution evidence.
