# [5.5] Toy Discrete Diffusion Language Models and Local DiffusionGemma Proof Verification Assets

Generated support files for the roadmap verification contract.

- `5.5_Diffusion_Language_Models_exercises.ipynb` is the learner-facing
  notebook with local stubs for noising, denoising loss, remasking, sampling,
  diagnostics, and the tiny denoiser wrapper.
- `5.5_Diffusion_Language_Models_solutions.ipynb` validates the section-local
  reference implementation against the visible tests.
- `artifacts.lock.yml` pins the current smoke-test artifact contract, the
  CUDA-trained tiny conditional diffusion LM preflight, and the exact
  DiffusionGemma real-model readiness contract.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The real graded path is `run_gpu_test`: it trains a tiny Transformer denoiser on
a generated copy-pair grammar, checks held-out masked-token accuracy, verifies
confidence-remasking sampler reconstruction, records activation-trajectory
shape checks, and rejects a shuffled-label control. It also checks pinned
`google/diffusiongemma-26B-A4B-it` and
`nvidia/diffusiongemma-26B-A4B-it-NVFP4` metadata, exact revisions,
Transformers loader support, local shard readiness, and NVFP4 runtime
readiness. Config/tokenizer support is not accepted as released-checkpoint
generation evidence; the current report must surface
`diffusiongemma_generation_ready`, the isolated vLLM proof fields, and any
remaining blockers directly.

Prepare the 24GB-local quantized artifact with:

```bash
BNB_CUDA_VERSION=130 uv run python scripts/prepare_diffusiongemma_artifacts.py --artifact nvfp4 --json
BNB_CUDA_VERSION=130 uv run python scripts/prepare_diffusiongemma_artifacts.py --artifact nvfp4 --download --require-local --max-workers 1
```

The pinned NVFP4 checkpoint advertises `quant_method: modelopt`. The main uv
environment keeps torch `2.12.1+cu132` and CUDA `13.2` and intentionally does
not install vLLM, because public vLLM `0.24.0` resolves to torch `2.11.0+cu130`.
The current released-checkpoint proof is therefore captured in
`artifacts/diffusiongemma_vllm_probe.json` from an isolated vLLM environment:
it loads the pinned NVIDIA NVFP4 revision, runs on the RTX 5090 Laptop GPU, and
generates a non-empty answer. The Google BF16 direct-loading path remains
deferred for the 24GB tier.
