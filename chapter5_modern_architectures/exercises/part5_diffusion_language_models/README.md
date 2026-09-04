# [5.5] Discrete Diffusion LMs: Watching Tokens Commit

This section teaches discrete diffusion by making the learner build and inspect
the full toy path. The exercise notebook implements the masking schedule,
forward corruption, masked loss, confidence and uniform remasking, an oracle
sampler, stable commitment time, the exact copy-pair dataset, CUDA training, and
conditional trajectory capture.

The signature experiment trains two identical tiny Transformer denoisers:

- the main model receives the exact grammar `[a, b] -> [a, b, a, a, b, b]`;
- the control receives suffix labels permuted across training prefixes.

The executed solution notebook shows loss curves, held-out accuracy, entropy by
denoising step, stable commitment by token position, and side-by-side token
trajectories. The main model must reach at least 95% held-out exact match while
the independently trained shuffled-label model must remain below 25% suffix
accuracy. A `Try It Yourself` cell lets learners change the prefix, remasking
rule, and seed.

`verification_report.json` records the same CUDA experiment as supporting
evidence. The final notebook cell separately displays the pinned real
DiffusionGemma NVFP4 generation artifact. That artifact proves one local
released-checkpoint generation on the RTX 5090 Laptop GPU; it does not claim
denoising-step activation capture or diffusion patching.

The main `uv` environment remains on torch `2.12.1+cu132` and CUDA `13.2`.
Public vLLM `0.24.0` requires torch `2.11.0+cu130`, so the NVFP4 proof remains
in its isolated runtime rather than changing the course environment.
