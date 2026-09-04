# [5.3] Mamba from Scratch Verification Assets

Learner and verification files for the complete Mamba-1 from-scratch progression.

- `artifacts.lock.yml` pins the current toy scan/cache and official Mamba artifact contract.
- `5.3_Mamba_from_Scratch_exercises.ipynb` and its solution notebook progress from
  exact continuous-time discretization through causal depthwise convolution,
  input-dependent selective parameters, recurrent/parallel scans, a gated pre-norm
  residual block, a recurrent cache, a stacked LM, and strict Mamba-130M weight loading.
- `tests.py` provides immediate toy oracles, shape/causality/gradient checks, cache
  parity, and a real 243-tensor CPU hidden/logits/generation parity test.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The learner notebook loads the cached pinned `state-spaces/mamba-130m-hf` on CPU and
compares every hidden state, final logits, recurrent cache logits, and greedy tokens.
The separate graded GPU path checks the official fused-kernel runtime and measured
VRAM. Update `artifacts.lock.yml` whenever the checkpoint revision, architecture,
prompt set, dtype, or parity thresholds change.
