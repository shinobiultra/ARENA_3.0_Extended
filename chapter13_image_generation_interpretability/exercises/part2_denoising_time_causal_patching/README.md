# [13.2] Denoising-Time Causal Patching

This section is an ARENA-style model-organism lab for locating when an image
feature becomes causally effective during diffusion sampling.

The required learner path starts with exact forward-noising ground truth, then
trains a 12-channel convolutional denoiser end to end on a deterministic
16x16 world containing every combination of shape, color, row, and column.
Students implement the hook and intervention machinery themselves. The
signature result shows generated clean/corrupt/patched images, a complete
four-layer by eight-timestep causal heatmap, a loss curve, and preregistered
controls.

## Files

- `13.2_Denoising-Time_Causal_Patching_exercises.ipynb`: learner notebook with
  implementation stubs, immediate tests, expected outputs, help, solutions,
  interpretation, play cells, and anomaly hunting.
- `13.2_Denoising-Time_Causal_Patching_solutions.ipynb`: fully executed CPU
  solution path.
- `solutions.py`: reference implementations and CUDA-only verification entrypoints.
- `tests.py`: semantic sub-function, causal-control, notebook, and package tests.
- `artifacts.lock.yml`: evidence and acceptance contract.
- `expected_outputs/`: smoke and signature-result expectations.
- `verification_report.schema.json`: report schema.

## Signature claim

On the pinned seed, replacing the exact concept channels from a clean red-square
trajectory into a matched blue-circle trajectory at the final denoising step
recovers the clean target region. Equal-size random channels, a shifted spatial
mask, the opposite-end timestep, shuffled donor labels, the unpatched trajectory,
and an untrained model do not reproduce the effect. Direct target-pixel copying
is retained only as a calibrated upper bound.

The pinned Stable Diffusion 1.5 path implements explicit DDIM latent
trajectories and same-seed donor-latent interventions. The exact toy result
establishes the method against known ground truth, then the real CUDA sweep
tests whether the intervention survives contact with a released model.

On Torch `2.12.1+cu132`, CUDA 13.2, and the RTX 5090 Laptop GPU, the pinned
20-step SD1.5 run passes both preregistered counterfactual cases. Best CLIP
recovery is `0.654` for red-into-blue and `0.505` for blue-into-red; the
corresponding regional selectivities are `0.222` and `0.222`. The target patch
beats the strongest matched control by `0.194` and `0.213`, respectively, at
`3.235 GiB` peak VRAM. These are two selected safe geometric cases, so the
claim is scoped to latent-state color transfer rather than a general U-Net
circuit or image-quality result.

## Running

```bash
uv run pytest -q chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching/tests.py
```

The solution notebook's required path is CPU-executable. `run_gpu_test` and
`run_full_experiment` deliberately require CUDA, enforce a maximum of 24 GiB,
have no CPU fallback, and refresh the real-model signature figure.
