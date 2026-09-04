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

This checkout has been verified on 2026-06-28 with:

```text
Python: 3.14.6
PyTorch: 2.12.1+cu132
torchvision: 0.27.1+cu132
CUDA build: 13.2
GPU: NVIDIA GeForce RTX 5090 Laptop GPU
VRAM: 23.46 GiB
```

Install the locked uv environment with:

```bash
export BNB_CUDA_VERSION=130
uv venv --python 3.14
uv sync --locked
```

`pyproject.toml` pins `torch==2.12.1+cu132` and
`torchvision==0.27.1+cu132` to the explicit PyTorch CUDA 13.2 wheel index
`https://download.pytorch.org/whl/cu132`; `uv.lock` records the full extension
dependency set.

`bitsandbytes==0.49.2` currently ships a CUDA 13.0 binary but not a CUDA 13.2
binary. On this machine the CUDA 13.0 binary runs against the CUDA 13.2 PyTorch
stack by setting:

```bash
export BNB_CUDA_VERSION=130
```

The current `requirements.txt` keeps older CUDA12/JAX/Brax/Atari dependencies
out of the default Python 3.14 environment. Use `requirements-legacy-rl.txt`
only for a separate legacy Chapter 2 RL environment.

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

The strict release gate is:

```bash
uv run python scripts/audit_extension_completion_strict.py
```

It intentionally fails if Gemma 3 base activations, EmbeddingGemma, or the base
FunctionGemma checkpoint are unavailable. In the current validated environment,
those required Gemma-family artifacts are authenticated and ready.

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
