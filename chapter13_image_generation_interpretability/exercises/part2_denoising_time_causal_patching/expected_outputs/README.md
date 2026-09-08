# Expected outputs for [13.2]

## Exact sub-functions

- `q_sample(..., t=0, ...)` is bitwise equal to the clean input.
- A 16x16 square occupies exactly 25 pixels; the circle occupies 21 pixels.
- Timestep-zero embeddings contain a zero sine half and one cosine half.
- Activation replacement modifies only the requested channels and locations.
- A complete cache contains `4 layers x 8 timesteps = 32` entries.
- The unpatched matched-seed recipient has zero calibrated recovery.
- Direct target-pixel copying has exactly one recovery.

## Executed signature result

With seed zero and 350 CPU training steps, expected approximate values are:

```text
training loss:       0.61 -> 0.013
best layer/time:     concept, t=1
target recovery:     0.96
target selectivity:  0.92
random channels:     0.05
shifted location:   -0.25
wrong timestep:      0.00
shuffled labels:     0.00
untrained model:    -0.11
pixel upper bound:   1.00
```

Exact values can vary slightly across PyTorch builds. Acceptance uses margins,
not string equality. The visible output must still contain recognizable clean,
corrupt, and patched images plus the complete causal heatmap and control chart.

## CUDA state

The strict release entrypoint was executed on Torch `2.12.1+cu132`, CUDA 13.2,
and an RTX 5090 Laptop GPU. With 20 DDIM steps it produced:

```text
case                         best CLIP   selectivity   strongest-control margin
red_into_blue_square           0.654       0.222                 0.194
blue_into_red_circle           0.505       0.222                 0.213
peak VRAM: 3.235 GiB
```

The real result must still be treated as a two-case, selected-prompt latent
intervention. It does not establish an internal U-Net activation circuit,
multi-seed robustness, or broad image-generation quality.

## How this fixture was produced

The toy signature runs `run_toy_experiment(training_steps=350, seed=0)` and
recomputes the full layer/time intervention grid from live denoiser activations.
The independent CUDA path executes the pinned SD1.5 DDIM trajectory and all
nineteen non-initial intervention times for both declared color pairs.

## Trusted implementation

The noising equation, zero-time endpoint, analytically rendered shape masks,
and direct-pixel recovery anchors are exact sub-function oracles. Real SD1.5
scheduler steps and actual VAE decoding supply the released-model computation;
CLIP measures the declared regional color transfer. The selected best time and
two fixed examples do not establish broad or internal-activation causality.

## Random seed

Toy training uses seed 0. Deterministic trajectory unit tests use model seed 7
and noise seed 11; the same-size random-latent oracle uses seed 19. Real donor
and recipient trajectories use the same per-case noise specified in the
implementation. Keep their pairing when introducing additional seeds.

## Allowed tolerances

Exact masks, zero-time identities, and untouched coordinates use `rtol=0,
atol=0` semantics. Float32 `torch.testing.assert_close` uses `rtol=1.3e-6,
atol=1e-5` unless a test overrides it; the closed-form noising comparison has
`atol=1e-7`. Scientific recovery margins are separately declared in the lock.

## When to regenerate

Regenerate after changes to the noising schedule, denoiser, mask, hook location,
training, timestep selection, prompt, checkpoint, or metric. Execute the
learner images, heatmap, and controls before refreshing report hashes. Retain
wrong-time, wrong-region, and random-latent failures visibly.
