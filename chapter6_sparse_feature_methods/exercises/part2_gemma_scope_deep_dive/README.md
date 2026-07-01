# [6.2] Gemma Scope Deep Dive Verification Assets

Generated support files for the roadmap verification contract.

- `6.2_Gemma_Scope_Deep_Dive_exercises.ipynb` is the learner notebook with
  local stubs for metadata checks, feature scoring, validation, ablation,
  steering, and direct logit attribution.
- `6.2_Gemma_Scope_Deep_Dive_solutions.ipynb` executes the visible tests
  against `solutions.py` and checks the committed CUDA report highlights.
- `artifacts.lock.yml` pins the current notebook contract plus the real Gemma Scope
  2 1B-IT layer-13 residual JumpReLU SAE artifact preflight and authenticated
  Gemma 3 activation validation.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current GPU report verifies the pinned SAE config, safetensor shapes,
finiteness, CUDA load, JumpReLU encode/decode, and real Gemma 3 residual
activation scoring against random-feature and label-shuffle controls.
