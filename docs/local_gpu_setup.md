# Local GPU Setup

How to set up this repo on your own machine, and what to expect from your hardware.

## Supported setup

- Linux x86_64. WSL2 on Windows counts as Linux.
- NVIDIA driver 580 or newer, if you want to use a GPU.
- Python managed by [uv](https://docs.astral.sh/uv/); you don't need to install Python yourself.
- macOS is not supported.

The reference machine is a laptop with a 24 GB RTX 5090 (Laptop GPU). Everything in the course
was run there with PyTorch 2.15 built for CUDA 13.4.

## Install

One lockfile covers the whole course, including the original chapters:

```bash
uv sync --locked
```

Lesson 5.4 also needs the CUDA-only Mamba kernels. These are an optional extra. They are built
from source (`mamba-ssm` and `causal-conv1d`, about ten minutes) and need the CUDA toolkit
(`nvcc`) on your `PATH`:

```bash
uv sync --locked --extra cuda-kernels
```

Skip the extra if you don't plan to do 5.4.

## Which route fits your hardware

| Your machine | What you can do |
| --- | --- |
| No GPU | 62 lessons run fully on CPU and 16 run slowly. The 8 lessons that need CUDA (5.4, 5.8, 11.1, 13.1, 13.2, 14.1, 14.2, 16.7) can't run, but you can read their saved outputs in each solutions notebook. |
| 8 GB GPU | Everything except 11.1. |
| 16 GB GPU or more | Everything. 11.1 peaks at 13.6 GiB. |

Slow CPU lessons are marked as such at the top of the lesson. If you are unsure, start the
lesson and read its first cell: it says what it needs.

## Check your environment

```python
from arena_ext import get_environment_report

report = get_environment_report()
print(report.as_dict())
print(report.warnings(required_vram_gb=16.0))
```

To estimate whether a model will fit before you load it:

```python
from arena_ext import estimate_inference_memory

budget = estimate_inference_memory(
    num_parameters=1_000_000_000, dtype="bfloat16", batch_size=1, context_length=2048,
    hidden_size=2048, num_layers=18, num_key_value_heads=8, head_dim=256, overhead_gb=1.5,
)
print(budget.as_dict(), budget.fits(16.0))
```

This is a rough, conservative estimate. Measured peak VRAM from PyTorch is the real answer.

## Large downloads

With stock ipykernel on Python 3.13 or newer, a first-time Hugging Face download of a few GB
can hang the kernel (11.1 and 13.1 hit this on a clean cache). The lock installs
[`arena-ipykernel-fix`](../infrastructure/extension/ipykernel_pipe_fix/README.md), which loads in
every kernel started from `.venv`. For kernels from other environments, see its README.

Big downloads are also easier to watch and resume from a terminal. For 11.1:

```bash
uv run hf download mistralai/Mistral-7B-v0.1 --revision 27d67f1b5f57dc0953326b2601d68371d40ea8da \
  --include "*.json" --include "*.safetensors" --include "tokenizer.model"
```

## Gated models

Some lessons use Google Gemma-family models, which need you to accept the terms on Hugging Face
first. After accepting and running `hf auth login`:

```bash
uv run python scripts/prepare_gated_gemma_family_artifacts.py          # check access
uv run python scripts/prepare_gated_gemma_family_artifacts.py --download --require-ready
```

DiffusionGemma uses the quantized NVFP4 checkpoint, which fits in 24 GB:

```bash
uv run python scripts/prepare_diffusiongemma_artifacts.py --artifact nvfp4 --download --require-local --max-workers 1
```

## For maintainers

The root `requirements.txt` is an export of `uv.lock`, for people who can't use uv. Each
upgrade batch regenerates it; never edit it by hand:

```bash
uv export --frozen --no-hashes --no-dev --emit-index-url --format requirements-txt -o requirements.txt
```

`--frozen` reads only `uv.lock`, so it needs no network. The export keeps the PyTorch CUDA wheel
index (`--extra-index-url`) and the local `./infrastructure/extension/ipykernel_pipe_fix`
package, and leaves out the `cuda-kernels` extra. `tests/test_requirements_export.py` fails
if the committed file drifts from a fresh export.

`requirements-original.txt` (upstream ARENA) and `requirements-ci-cpu.txt` (hosted CI) are
separate hand-maintained files, not exports.
