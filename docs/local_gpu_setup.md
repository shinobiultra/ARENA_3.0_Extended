# Local GPU Setup

This fork is designed to keep the original ARENA material intact while adding local-first frontier model labs. The default hardware target is a single 24GB CUDA GPU, with CPU smoke tests available for most infrastructure checks.

## Local tiers

Use these tiers when writing or running extension notebooks:

| Tier | Requirement | Use |
| --- | --- | --- |
| Green | CPU or <= 24GB CUDA GPU | Smoke tests, small toy models, GPT-2/Pythia-scale checks, activation-store tests |
| Yellow | 24GB CUDA GPU with quantization/offload | Gemma-sized inference, small SAE/transcoder runs, video feature extraction with reduced batch sizes |
| Red | More than a laptop GPU or unsafe training objective | Full-scale SAE training, large activation-oracle/NLA training, dangerous misalignment finetunes |

Required exercises should fit in the green or yellow tier. Red-tier work should only appear as optional research direction text.

## Environment checks

## Verified local stack

This checkout has been verified on 2026-11-07 with:

```text
Python: 3.14.10
PyTorch: 2.15.0+cu134
torchvision: 0.29.2+cu134
CUDA build: 13.4 (driver 615.78)
GPU: NVIDIA GeForce RTX 5090 Laptop GPU
VRAM: 23.46 GiB
```

One uv lock covers the whole course, original chapters included (#37). Install it with:

```bash
uv sync --locked
```

On a machine with the CUDA toolkit (`nvcc`), add the CUDA-only Mamba kernels that 5.4 needs:

```bash
uv sync --locked --extra cuda-kernels
```

That builds `causal-conv1d` and `mamba-ssm` from source for sm_75 through sm_121 (about ten
minutes). `pyproject.toml` takes `torch` and `torchvision` from the PyTorch CUDA 13.4 wheel index
`https://download.pytorch.org/whl/cu134`; `bitsandbytes` picks its closest CUDA 13 binary by
itself, so no `BNB_CUDA_VERSION` override is needed. The Chapter 2 JAX/Brax/EnvPool/Atari stack
(`jax[cuda13]`) shares the same CUDA 13 libraries as torch and is part of the lock.

### Large downloads inside notebooks

With stock ipykernel on Python ≥ 3.13, a first-time Hugging Face download of a few GB hangs
the kernel. 11.1 (Mistral-7B, 14.5 GB) and 13.1 (SD-1.5, 2.1 GB) hit this on a clean cache.
hf_xet reports download progress from Rust threads, and ipykernel never closes the zmq socket
it opens for each of those progress updates. After a few minutes the kernel runs out of
sockets.

The lock installs [`arena-ipykernel-fix`](../infrastructure/extension/ipykernel_pipe_fix/README.md)
to prevent this. Every kernel started from `.venv` loads it, whether VS Code, JupyterLab,
Jupyter Notebook or `build.py --execute` launched it. For kernels from other environments,
see its README.

Large downloads are still easier to watch and resume from a terminal. You can fetch a model
before opening its lesson; for 11.1:

```bash
uv run hf download mistralai/Mistral-7B-v0.1 --revision 27d67f1b5f57dc0953326b2601d68371d40ea8da \
  --include "*.json" --include "*.safetensors" --include "tokenizer.model"
```

## Required gated artifacts

Some roadmap targets are Google Gemma-family artifacts with manual Hugging Face
access requirements. Do not replace these with public stand-ins when making a
full completion claim. The scoped local reports may validate public or toy
preflights, but the strict roadmap gate requires the exact gated artifacts.

After accepting the model terms and logging in, prepare them with:

```bash
hf auth login
uv run python scripts/prepare_gated_gemma_family_artifacts.py --download --require-ready
```

To inspect the current access state without downloading weights:

```bash
uv run python scripts/prepare_gated_gemma_family_artifacts.py
```

Lessons that need Gemma 3 base activations, EmbeddingGemma, or the base
FunctionGemma checkpoint fail their fresh-kernel run if those gated artifacts
are unavailable. In the current validated environment they are authenticated
and ready.

DiffusionGemma is prepared separately because the required 24GB local path is
the quantized NVIDIA NVFP4 checkpoint, while the Google BF16 checkpoint is much
larger. Inspect or download the exact pinned NVFP4 artifact with:

```bash
uv run python scripts/prepare_diffusiongemma_artifacts.py --artifact nvfp4 --json
uv run python scripts/prepare_diffusiongemma_artifacts.py --artifact nvfp4 --download --require-local --max-workers 1
```

The BF16 artifact can be inspected with `--artifact bf16`, but it is not the
required local generation path for the 24GB roadmap target.

Before loading a model, run:

```python
from arena_ext import get_environment_report

report = get_environment_report()
print(report.as_dict())
print(report.warnings(required_vram_gb=24.0))
```

Every GPU notebook should print:

```text
Python version
PyTorch version
CUDA availability and CUDA version
GPU name and total memory
BF16 support
flash-attn / xformers availability when relevant
```

## Memory checks

Before loading a checkpoint, estimate memory:

```python
from arena_ext import estimate_inference_memory

budget = estimate_inference_memory(
    num_parameters=1_000_000_000,
    dtype="bfloat16",
    batch_size=1,
    context_length=2048,
    hidden_size=2048,
    num_layers=18,
    num_key_value_heads=8,
    head_dim=256,
    overhead_gb=1.5,
)

print(budget.as_dict())
assert budget.fits(24.0)
```

The estimate is intentionally conservative and simple. It is a preflight check, not a replacement for peak VRAM measurements from PyTorch.

## Notebook footer

Every extension notebook should end with a compact footer:

```text
checkpoint:
dtype:
device:
seed:
dataset/subset:
estimated memory:
peak VRAM:
runtime:
artifacts:
tests:
```

If a notebook cannot run the full GPU path, it should still run `run_smoke_test(cpu=True)` and clearly state which dependency or hardware constraint blocked the full run.
